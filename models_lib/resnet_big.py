"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
}


class LinearBatchNorm(nn.Module):
    """Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose"""
    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x


class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet50', head='mlp', feat_dim=128):
        super(SupConResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat


class SupCEResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='resnet50', num_classes=10, in_channel=3):
        super(SupCEResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun(in_channel=in_channel)
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        return self.fc(self.encoder(x))


class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)
    
    
class MLPClassifier(nn.Module):
    """MLP classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(MLPClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, features):
        return self.fc(features)

class Proto_Classifier(nn.Module):
    def __init__(self, feat_in, num_classes):
        super(Proto_Classifier, self).__init__()
        # Use pure PyTorch for orthogonal matrix generation to avoid numpy/CUDA conflicts
        P = self.generate_random_orthogonal_matrix(feat_in, num_classes)
        I = torch.eye(num_classes)
        one = torch.ones(num_classes, num_classes)
        scale = (num_classes / (num_classes - 1)) ** 0.5  # Pure Python sqrt
        M = scale * torch.matmul(P, I - ((1.0 / num_classes) * one))

        # Register as buffer so it moves to the correct device with the model
        self.register_buffer('proto', M.float())

    def generate_random_orthogonal_matrix(self, feat_in, num_classes):
        # Use PyTorch's QR decomposition instead of numpy to avoid potential segfaults
        a = torch.randn(feat_in, num_classes)
        Q, R = torch.linalg.qr(a)
        # Ensure orthonormality by normalizing columns
        P = Q[:, :num_classes].float()
        return P

    def load_proto(self, proto):
        # Use copy_ to properly update the registered buffer in-place
        self.proto.copy_(proto)

    def forward(self, label):
        # produce the prototypes w.r.t. the labels
        target = self.proto[:, label].T ## B, d  output: B, d
        return target
   
class ETFCEResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='resnet50', num_classes=10, in_channel=3):
        super(ETFCEResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun(in_channel=in_channel)
        self.fc = nn.Linear(dim_in, num_classes)
        
        self.linear_proto = nn.Linear(dim_in, num_classes)
        self.proto_classifier = Proto_Classifier(num_classes, num_classes)
        self.scaling_train = torch.nn.Parameter(torch.tensor(1.0))
        

    def forward(self, x):
        out = self.encoder(x)
        
        feature = self.linear_proto(out)
        feature_norm = torch.norm(feature, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        feature = torch.div(feature, feature_norm)
        
        logits = torch.matmul(feature, self.proto_classifier.proto)
        # logits = logits * self.scaling_train
        
        # logits = self.fc(out)
        
        return logits, feature
    
class LearnableProtoResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='resnet50', num_classes=10, in_channel=3):
        super(LearnableProtoResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun(in_channel=in_channel)
        
        self.learnable_proto = torch.nn.Parameter(torch.randn(num_classes, dim_in))

    def forward(self, x):
        feature = self.encoder(x)
        feature_norm = torch.nn.functional.normalize(feature, p=2, dim=1, eps=1e-12) # size (batchsize, feature_dim)

        logits = torch.matmul(feature_norm, self.learnable_proto.t()) # (batchsize, feature_dim) * (feature_dim, num_classes)

        # training and inference
        return logits, feature_norm
        
        # only used in visualization with gradcam
        # return logits

    def get_proto(self, weight=None):
        if weight is not None:
            assert len(weight.shape) == 1 and weight.shape[0] == self.learnable_proto.shape[0]
            exec_device = self.learnable_proto.device
            return weight.view(-1, 1).to(exec_device) * self.learnable_proto
        else:
            return self.learnable_proto

class DualHeadProtoResNet(nn.Module):
    """encoder + linear classification head + learnable prototypes for alignment.

    Decouples classification (unconstrained linear head) from federated
    alignment (cosine-similarity prototypes).  The linear head provides
    full capacity for local CE training while prototypes enable
    cross-client representation alignment via contrastive / ETF losses.
    """
    def __init__(self, name='resnet50', num_classes=10, in_channel=3):
        super(DualHeadProtoResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun(in_channel=in_channel)
        self.fc = nn.Linear(dim_in, num_classes)
        self.learnable_proto = torch.nn.Parameter(torch.randn(num_classes, dim_in))

    def forward(self, x):
        feature = self.encoder(x)
        feature_norm = torch.nn.functional.normalize(feature, p=2, dim=1, eps=1e-12)
        logits = self.fc(feature)
        return logits, feature_norm

    def get_proto(self, weight=None):
        if weight is not None:
            assert len(weight.shape) == 1 and weight.shape[0] == self.learnable_proto.shape[0]
            exec_device = self.learnable_proto.device
            return weight.view(-1, 1).to(exec_device) * self.learnable_proto
        else:
            return self.learnable_proto


class LearnableProtoResNetWithProjector(nn.Module):
    """
    Backbone + MLP Projector + Classifier/Proto.
    This resolves the d >= C-1 limitation by projecting features to a higher dimension
    before ETF alignment.
    """
    def __init__(self, name='resnet50', num_classes=10, projector_dim=2048):
        super(LearnableProtoResNetWithProjector, self).__init__()
        model_fun, dim_in = model_dict[name]
        backbone = model_fun()
        
        # MLP Projector: dim_in -> projector_dim -> projector_dim
        # Example for ResNet18: 512 -> 2048 -> 2048
        projector = nn.Sequential(
            nn.Linear(dim_in, projector_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projector_dim, projector_dim)
        )
        
        # Combine backbone and projector into the "encoder" property
        # so that external calls to model.encoder(x) get the projected features.
        self.encoder = nn.Sequential(backbone, projector)
        
        # The learnable prototype (and alignment target) now lives in the projected space
        self.learnable_proto = torch.nn.Parameter(torch.randn(num_classes, projector_dim))

    def forward(self, x):
        # Now self.encoder returns the projected features directly
        projected = self.encoder(x) # (B, 2048)
        
        # Normalize the projected feature for ETF alignment
        feature_norm = torch.nn.functional.normalize(projected, p=2, dim=1, eps=1e-12) # (B, 2048)
        
        # Logits are computed in the high-dimensional projected space
        logits = torch.matmul(feature_norm, self.learnable_proto.t()) # (B, 2048) * (2048, C) -> (B, C)

        return logits, feature_norm

    def get_proto(self, weight=None):
        if weight is not None:
            assert len(weight.shape) == 1 and weight.shape[0] == self.learnable_proto.shape[0]
            exec_device = self.learnable_proto.device
            return weight.view(-1, 1).to(exec_device) * self.learnable_proto
        else:
            return self.learnable_proto




        


class BottleneckProtoResNet(nn.Module):
    """
    Experimental model for ETF Bottleneck analysis.
    Forces features into a low dimension (feature_dim) and then optionally projects back up (projector_dim).
    """
    def __init__(self, name='resnet50', num_classes=10, feature_dim=512, projector_dim=None):
        super(BottleneckProtoResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        
        # Bottleneck layer to force dimension d
        # Note: dim_in is usually 512 for ResNet18
        self.bottleneck = nn.Linear(dim_in, feature_dim)
        
        # Optional Projector
        if projector_dim:
            self.projector = nn.Sequential(
                nn.Linear(feature_dim, projector_dim), # MLP layer 1 (Expand immediately)
                nn.ReLU(inplace=True),
                nn.Linear(projector_dim, projector_dim) # MLP layer 2
            )
            self.final_dim = projector_dim
        else:
            self.projector = None
            self.final_dim = feature_dim
            
        self.learnable_proto = torch.nn.Parameter(torch.randn(num_classes, self.final_dim))

    def forward(self, x):
        feat = self.encoder(x) # (B, 512)
        feat = self.bottleneck(feat) # (B, feature_dim)
        
        if self.projector:
            feat_final = self.projector(feat) # (B, projector_dim)
        else:
            feat_final = feat # (B, feature_dim)
            
        feature_norm = torch.nn.functional.normalize(feat_final, p=2, dim=1, eps=1e-12)
        logits = torch.matmul(feature_norm, self.learnable_proto.t())
        
        return logits, feature_norm

    def get_proto(self, weight=None):
        if weight is not None:
            assert len(weight.shape) == 1 and weight.shape[0] == self.learnable_proto.shape[0]
            exec_device = self.learnable_proto.device
            return weight.view(-1, 1).to(exec_device) * self.learnable_proto
        else:
            return self.learnable_proto
