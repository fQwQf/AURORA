import torch
import torch.nn.functional as F


class AlignmentLoss(torch.nn.Module):
    """Alignment component from Wang & Isola (ICML 2020).
    
    Alignment-only loss: pushes positive pairs (two augmented views) together.
    Unlike full Alignment+Uniformity, this is:
    - Bounded (always >= 0 for L2-normalized features)
    - Compatible with CE from epoch 0 (doesn't oppose class clustering)
    - Naturally gated by augmentation quality (bad augmentation → noisy but bounded gradient)
    
    Collapse prevention is handled by ETF prototype alignment (pulling prototypes
    toward maximally separated targets), NOT by uniformity on features.
    """
    def __init__(self, alpha=2):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x, y):
        return (x - y).norm(dim=1).pow(self.alpha).mean()


class AlignmentUniformityLoss(torch.nn.Module):
    """Alignment + Uniformity on the Hypersphere (Wang & Isola, ICML 2020)."""
    def __init__(self, alpha=2, t=2):
        super().__init__()
        self.alpha = alpha
        self.t = t
    
    def forward(self, x, y):
        align_loss = (x - y).norm(dim=1).pow(self.alpha).mean()
        uniform_loss = 0.5 * (self._uniform(x, self.t) + self._uniform(y, self.t))
        return align_loss + uniform_loss
    
    @staticmethod
    def _uniform(x, t=2):
        sq_pdist = torch.pdist(x, p=2).pow(2)
        return sq_pdist.mul(-t).exp().mean().log()


class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
class InfoNCELoss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, features):
        """
        Computes the InfoNCE loss.
        
        Args:
            features (torch.Tensor): The feature matrix of shape [2 * batch_size, feature_dim], 
                                     where features[:batch_size] are the representations of 
                                     the first set of augmented images, and features[batch_size:] 
                                     are the representations of the second set.
        
        Returns:
            torch.Tensor: The computed InfoNCE loss.
        """
        # Normalize features to have unit norm
        features = torch.nn.functional.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Get batch size
        batch_size = features.shape[0] // 2
        
        # Construct labels where each sample's positive pair is in the other view
        labels = torch.arange(batch_size, device=features.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)

        # Mask out self-similarities by setting the diagonal elements to -inf
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=features.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        # InfoNCE loss
        loss = torch.nn.functional.cross_entropy(similarity_matrix, labels)
        
        return loss  
    
class Contrastive_proto_feature_loss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(Contrastive_proto_feature_loss, self).__init__()
        self.temperature = temperature
    
    def forward(self, feature, proto, labels, active_indices=None):
        if active_indices is not None:
            proto = proto[active_indices]
            label_map = {c.item(): i for i, c in enumerate(active_indices)}
            mapped_labels = torch.tensor([label_map[l.item()] for l in labels], device=labels.device)
        else:
            mapped_labels = labels
        
        similarity_matrix = torch.matmul(feature, proto.T) / self.temperature
        loss = torch.nn.functional.cross_entropy(similarity_matrix, mapped_labels)
        
        return loss
    
class Contrastive_proto_loss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(Contrastive_proto_loss, self).__init__()
        self.temperature = temperature
    
    def forward(self, proto, active_indices=None):
        """
        Prototype self-contrastive loss. Forces prototypes to be distinct.
        
        Args:
            proto: [num_classes, feature_dim] all learnable prototypes
            active_indices: optional tensor of class indices actually present in training data.
                         If provided, only compute loss for these classes, This prevents
                         gradients from flowing to prototypes for unseen classes, which
                         is critical for non-IID settings (e.g., FEMNIST with 62 classes
                         where each client only sees a subset).
        """
        if active_indices is not None:
            # Only compute loss for classes actually present in training data
            proto_subset = proto[active_indices]
        else:
            proto_subset = proto
        
        proto_len = proto_subset.shape[0]
        if proto_len <= 1:
            return torch.tensor(0.0, device=proto.device)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(proto_subset, proto_subset.T) / self.temperature
        
        labels = torch.arange(proto_len, device=proto.device)
        
        loss = torch.nn.functional.cross_entropy(similarity_matrix, labels)
        
        return loss         