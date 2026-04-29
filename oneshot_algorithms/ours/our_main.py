from oneshot_algorithms.utils import prepare_checkpoint_dir, visualize_pic, compute_local_model_variance
from dataset_helper import get_supervised_transform, get_client_dataloader, NORMALIZE_DICT
from models_lib import get_train_models

from common_libs import *

import torch.nn.functional as F

from oneshot_algorithms.ours.our_local_training import ours_local_training
from oneshot_algorithms.ours.gpu_augmentation import get_gpu_augmentation
from oneshot_algorithms.ours.unsupervised_loss import SupConLoss, Contrastive_proto_feature_loss, Contrastive_proto_loss

import math

import torch.optim as optim
from oneshot_algorithms.fedavg import parameter_averaging
from oneshot_algorithms.utils import test_acc



def test_acc_our_model(model, test_loader, device):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            _, predicted = torch.max(logits, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct / total

def calculate_adaptive_lambda(client_dataloader, num_classes, lambda_min, lambda_max, device):
    """Calculates a client-specific lambda based on the entropy of its local data distribution."""
    counts = torch.zeros(num_classes, device=device)
    total_samples = 0
    for _, targets in client_dataloader:
        targets = targets.to(device)
        for i in range(num_classes):
            counts[i] += (targets == i).sum()
        total_samples += len(targets)

    if total_samples == 0:
        return (lambda_min + lambda_max) / 2 # A safe default

    probs = counts / total_samples
    probs = probs[probs > 0] # Remove zero probabilities for log calculation
    
    entropy = -torch.sum(probs * torch.log2(probs))
    
    # Normalize entropy to [0, 1]
    max_entropy = math.log2(num_classes)
    if max_entropy == 0: # Handle single-class case
        normalized_entropy = 0
    else:
        normalized_entropy = entropy.item() / max_entropy

    # Map normalized entropy to the [lambda_min, lambda_max] range
    adaptive_lambda = lambda_min + normalized_entropy * (lambda_max - lambda_min)
    
    logger.info(f"Data entropy: {entropy:.4f}, Normalized: {normalized_entropy:.4f} -> Adaptive Lambda: {adaptive_lambda:.4f}")
    
    return adaptive_lambda

def get_supcon_transform(dataset_name):
    if dataset_name == 'CIFAR10' or dataset_name == 'CIFAR100' or dataset_name == 'SVHN' or dataset_name == 'PathMNIST':
        return torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(size=32, scale=(0.2, 1.), antialias=False),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomApply([
                torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2),
            torchvision.transforms.Normalize(**NORMALIZE_DICT[dataset_name])
        ])                
    else:    
        return torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(size=64, scale=(0.2, 1.), antialias=False),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomApply([
            torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        torchvision.transforms.RandomGrayscale(p=0.2),
        torchvision.transforms.Normalize(**NORMALIZE_DICT[dataset_name])
        ])


def generate_etf_anchors(num_classes, feature_dim, device):
    """
    Generate Equiangular Tight Frame (ETF) anchors based on Neural Collapse theory.
    This creates a set of maximally separated and geometrically optimal prototype targets.
    """
    # Ensure feature dimension is at least number of classes, a common requirement for ETF construction
    if feature_dim < num_classes:
        # If dimension is insufficient, we can fallback to orthogonal basis, a good suboptimal choice
        logger.warning(f"Feature dim ({feature_dim}) is less than num_classes ({num_classes}). Falling back to orthogonal anchors.")
        H = torch.randn(feature_dim, num_classes)
        Q, _ = torch.qr(H)
        return Q.T.to(device)

    # 1. Construct the Gram matrix of ETF: M = I - (1/C) * J
    I = torch.eye(num_classes)
    J = torch.ones(num_classes, num_classes)
    M = I - (1 / num_classes) * J

    # 2. Find M_sqrt via Cholesky decomposition
    # M is positive semi-definite, may need to add a small epsilon for numerical stability
    try:
        L = torch.linalg.cholesky(M + 1e-6 * I)
    except torch.linalg.LinAlgError:
        # If Cholesky decomposition fails, use eigenvalue decomposition
        eigvals, eigvecs = torch.linalg.eigh(M)
        eigvals[eigvals < 0] = 0 # Eliminate negative eigenvalues caused by numerical errors
        L = eigvecs @ torch.diag(torch.sqrt(eigvals))

    # 3. Generate a "basis" of a random orthogonal matrix
    H_ortho = torch.randn(feature_dim, num_classes)
    Q, _ = torch.linalg.qr(H_ortho) # Columns of Q are orthogonal

    # 4. Multiply the basis with M_sqrt to generate the final ETF matrix
    # Column vectors of W constitute the ETF
    W = Q @ L.T
    
    # We need prototypes of shape (num_classes, feature_dim), so return transpose
    etf_anchors = W.T.to(device)
    
    # Final normalization to ensure all anchors are unit vectors
    etf_anchors = torch.nn.functional.normalize(etf_anchors, dim=1)
    
    return etf_anchors


def optimize_global_prototypes_on_server(
    local_protos_list, 
    trainable_global_prototypes, 
    etf_anchors,
    server_lr, 
    server_epochs, 
    gamma_etf_reg, 
    device
):
    """
    Implements Plan B: Aggregation-as-Optimization on the server.
    This function takes the client protos as a mini-dataset and trains
    the server's own global prototypes.
    """
    logger.info("--- Starting Server-Side Optimization (Plan B) ---")
    
    # Set up a dedicated optimizer for the global prototypes
    server_optimizer = optim.Adam(trainable_global_prototypes.parameters(), lr=server_lr)
    
    # The uploaded local prototypes are our "training data"
    # Shape: [num_clients, num_classes, feature_dim]
    local_protos_tensor = torch.stack(local_protos_list).to(device)
    num_clients, num_classes, _ = local_protos_tensor.shape

    # Loss functions
    contrastive_loss_fn = torch.nn.CrossEntropyLoss()
    etf_reg_loss_fn = torch.nn.MSELoss()

    # The server's internal training loop
    for epoch in range(server_epochs):
        total_server_loss = 0
        for i in range(num_clients): # Iterate through each client's perspective
            client_protos = local_protos_tensor[i] # [num_classes, feature_dim]
            
            server_optimizer.zero_grad()
            
            # The "model" is just the global prototype embedding layer
            current_global_prototypes = trainable_global_prototypes.weight

            # --- L_contrastive (inspired by FedTGP) ---
            # Calculate similarity: what the server thinks vs. what the client thinks
            # The "logits" are the similarities between the client's protos and the server's global protos
            similarity_matrix = torch.matmul(client_protos, current_global_prototypes.t())
            
            # The "labels" are just 0, 1, 2, ..., num_classes-1
            labels = torch.arange(num_classes).to(device)
            
            l_contrastive = contrastive_loss_fn(similarity_matrix, labels)

            # --- L_etf_reg (our innovation) ---
            # Gently pull the trainable protos towards the ideal ETF structure
            l_etf_reg = etf_reg_loss_fn(current_global_prototypes, etf_anchors)

            # --- Total Server Loss ---
            loss = l_contrastive + gamma_etf_reg * l_etf_reg
            
            loss.backward()
            server_optimizer.step()
            total_server_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Server Epoch {epoch+1}/{server_epochs}, Avg Loss: {total_server_loss / num_clients:.4f}")

    logger.info("--- Finished Server-Side Optimization ---")
    return trainable_global_prototypes # Return the updated parameter wrapper

def agg_protos(protos):
    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos

def collect_protos(model, trainloader, device):
    model.eval()
    protos = defaultdict(list)
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(trainloader):
            if type(data) == type([]):
                data[0] = data[0].to(device)
            else:
                data = data.to(device)
            target = target.to(device) 

            rep = model.encoder(data)

            for i, yy in enumerate(target):
                y_c = yy.item()
                protos[y_c].append(rep[i, :].detach().data)

    local_protos = agg_protos(protos)

    return local_protos  

def generate_sample_per_class(num_classes, local_data, data_num):
    sample_per_class = torch.tensor([0 for _ in range(num_classes)])

    for idx, (data, target) in enumerate(local_data):
        sample_per_class += torch.tensor([sum(target==i) for i in range(num_classes)])

    sample_per_class = sample_per_class / data_num

    return sample_per_class

def aggregate_local_protos(local_protos):

    local_protos = torch.stack(local_protos, dim=0)

    g_protos = torch.mean(local_protos, dim=0).detach()

    inter_client_proto_std = torch.std(local_protos, dim=0).mean().item()

    g_protos_std = torch.std(g_protos).item()

    logger.info(f'g_protos_std (global internal): {g_protos_std:.6f}')
    logger.info(f'inter_client_proto_std (cross-client): {inter_client_proto_std:.6f}')

    return g_protos



class WEnsembleFeatureNoise(torch.nn.Module):
    def __init__(self, model_list, weight_list=None):
        super(WEnsembleFeatureNoise, self).__init__()
        self.models = model_list
        if weight_list is None:
            self.weight_list = [1.0 / len(model_list) for _ in range(len(model_list))]
        else:
            self.weight_list = weight_list
            
    
    def forward(self, x):
        noise = torch.randn_like(x)
        
        dis_noise_list = []
        feature_total = []
        for model, weight in zip(self.models, self.weight_list):
            feature = model.encoder(x) # batchsize, feature_dim
            noise_feature = model.encoder(noise) 
            dis_sim = 1 - torch.nn.functional.cosine_similarity(feature, noise_feature, dim=1) # batchsize,
            dis_noise_list.append(dis_sim) # model_num, batchsize
            feature_total.append(feature) # model_num, batchsize, feature_dim
        
        dis_noise_list = torch.stack(dis_noise_list).mean(dim=0).unsqueeze(-1) # model_num, batchsize, 1
        
        feature_total = torch.stack(feature_total) # model_num, batchsize, feature_dim
        
        feature_total = dis_noise_list * feature_total # model_num, batchsize, feature_dim
        feature_total = feature_total.sum(dim=0) # batchsize, feature_dim
        
        return feature_total


class WEnsembleFeature(torch.nn.Module):
    def __init__(self, model_list, weight_list=None):
        super(WEnsembleFeature, self).__init__()
        self.models = model_list
        if weight_list is None:
            self.weight_list = [1.0 / len(model_list) for _ in range(len(model_list))]
        else:
            self.weight_list = weight_list
            
    def forward(self, x):
        feature_total = 0
        for model, weight in zip(self.models, self.weight_list):
            feature = weight * model.encoder(x)
            feature_total += feature
        return feature_total   
    

class TrueSimpleEnsembleServer(torch.nn.Module):

    def __init__(self, model_list, weight_list=None):
        super(TrueSimpleEnsembleServer, self).__init__()
        # Ensure models are in eval mode for inference
        self.models = [model.eval() for model in model_list]
        if weight_list is None:
            self.weight_list = [1.0 / len(model_list) for _ in range(len(model_list))]
        else:
            self.weight_list = weight_list

    def forward(self, x):
        all_probs = []
        with torch.no_grad():
            for model, weight in zip(self.models, self.weight_list):
                # Get logits from the full model (encoder + classifier/proto)
                logits, _ = model(x)
                # Convert to probabilities and apply weight
                probs = F.softmax(logits, dim=1) * weight
                all_probs.append(probs)
        
        # Sum the weighted probabilities
        final_probs = torch.sum(torch.stack(all_probs), dim=0)
        
        # Return the final probability distribution
        return final_probs

    
def eval_with_proto(model, test_loader, device, proto):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            feature = model(data)
            feature = torch.nn.functional.normalize(feature, dim=1)

            logits = torch.matmul(feature, proto.t())
            
            _, predicted = torch.max(logits.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    acc = correct / total
    return acc        


class WEnsembleDualHead(torch.nn.Module):
    def __init__(self, model_list, weight_list=None):
        super(WEnsembleDualHead, self).__init__()
        self.models = model_list
        if weight_list is None:
            self.weight_list = [1.0 / len(model_list) for _ in range(len(model_list))]
        else:
            self.weight_list = weight_list
        avg_fc_weight = sum(w * m.fc.weight.data for w, m in zip(self.weight_list, self.models))
        avg_fc_bias = sum(w * m.fc.bias.data for w, m in zip(self.weight_list, self.models))
        self.register_buffer('avg_fc_weight', avg_fc_weight)
        self.register_buffer('avg_fc_bias', avg_fc_bias)

    def forward(self, x):
        feature_total = 0
        for model, weight in zip(self.models, self.weight_list):
            feature_total += weight * model.encoder(x)
        return F.linear(feature_total, self.avg_fc_weight, self.avg_fc_bias)


def eval_with_linear_head(ensemble, test_loader, device):
    ensemble.eval()
    ensemble.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits = ensemble(data)
            _, predicted = torch.max(logits.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct / total

def eval_output_ensemble(model, test_loader, device):
    """
    Evaluates an ensemble model that directly outputs class probabilities.
    """
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Get final probabilities directly from the model
            final_probs = model(data)
            
            # Get predictions
            _, predicted = torch.max(final_probs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    acc = correct / total
    return acc


def OneshotOurs(trainset, test_loader, client_idx_map, config, device, server_strategy='simple_feature', lambda_val=0):
    logger.info('OneshotOurs')
    
    use_imagenet_pretrain = config.get('DBCD', {}).get('use_imagenet_pretrain', False)
    use_pretrain_arg = False
    if use_imagenet_pretrain:
        use_pretrain_arg = True
        logger.info("[OneshotOurs] Loading ImageNet pretrained weights for ResNet-18.")
    
    # get the global model
    global_model = get_train_models(
        model_name=config['server']['model_name'],
        num_classes=config['dataset']['num_classes'],
        mode='our',
        use_pretrain=use_pretrain_arg,
        in_channel=config['dataset'].get('channels', 3)
    )
    global_model.to(device)
    global_model.train()
    feature_dim = global_model.learnable_proto.shape[1]
    num_classes = config['dataset']['num_classes']
    fixed_anchors = generate_etf_anchors(num_classes, feature_dim, device)
    logger.info(f"Initialized ETF fixed anchors with shape: {fixed_anchors.shape}")

    method_results = defaultdict(list)
    save_path, local_model_dir = prepare_checkpoint_dir(config)
    if not os.path.exists(save_path + "/config.yaml"):
        save_yaml_config(save_path + "/config.yaml", config) 

    local_models = [copy.deepcopy(global_model) for _ in range(config['client']['num_clients'])]
    
    local_data_size = [len(client_idx_map[c]) for c in range(config['client']['num_clients'])]
    if config['server']['aggregated_by_datasize']:
        weights = [i/sum(local_data_size) for i in local_data_size]
    else:
        weights = [1/config['client']['num_clients'] for _ in range(config['client']['num_clients'])]        
    
    aug_transformer = get_gpu_augmentation(config['dataset']['data_name'], device)

    clients_sample_per_class = []

    total_rounds = config['server']['num_rounds']

    for cr in trange(config['server']['num_rounds']):
        logger.info(f"Round {cr} starts--------|")
        
        local_protos = []
        
        for c in range(config['client']['num_clients']):
            logger.info(f"Client {c} Starts Local Trainning--------|")
            client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])

            if cr == 0:
                clients_sample_per_class.append(generate_sample_per_class(config['dataset']['num_classes'], client_dataloader, len(client_idx_map[c])))

            if (lambda_val > 0):
                lambda_align_initial = lambda_val
                
            else:
                lambda_align_initial = config.get('lambda_align_initial', 5.0)

            logger.info(f"Using provided lambda_align_initial: {lambda_align_initial}")

            # Call the exact same local training function as V6, but pass high-quality ETF anchors
            local_model_c = ours_local_training(
                model=copy.deepcopy(local_models[c]),
                training_data=client_dataloader,
                test_dataloader=test_loader,
                start_epoch=cr * config['server']['local_epochs'],
                local_epochs=config['server']['local_epochs'],
                optim_name=config['server']['optimizer'],
                lr=config['server']['lr'],
                momentum=config['server']['momentum'],
                loss_name=config['server']['loss_name'],
                device=device,
                num_classes=config['dataset']['num_classes'],
                sample_per_class=clients_sample_per_class[c],
                aug_transformer=aug_transformer,
                client_model_dir=local_model_dir + f"/client_{c}",
                total_rounds=total_rounds,
                 save_freq=config['checkpoint']['save_freq'],
                 use_drcl=False,
                 fixed_anchors=None,
                 lambda_align=lambda_align_initial,
                 use_fafi=True
            )
            
            local_models[c] = local_model_c

            local_proto_c = local_model_c.get_proto().detach()
            local_protos.append(local_proto_c)

        logger.info(f"Round {cr} Finish--------|")
        model_var_m, model_var_s = compute_local_model_variance(local_models)
        logger.info(f"Model variance: mean: {model_var_m}, sum: {model_var_s}")

        global_proto = aggregate_local_protos(local_protos)
        
        if server_strategy == 'true_simple_output':
            method_name = 'OursV7+TrueSimpleOutputServer'
            ensemble_model = TrueSimpleEnsembleServer(model_list=local_models, weight_list=weights)
            logger.info("V7 Training | Using TRULY SIMPLE Output-level Server.")
            ens_acc = eval_output_ensemble(ensemble_model, test_loader, device)

        elif server_strategy == 'simple_feature':
            method_name = 'OursV7+SimpleFeatureServer'
            ensemble_model = WEnsembleFeature(model_list=local_models, weight_list=weights)
            global_proto = aggregate_local_protos(local_protos)
            logger.info("V7 Training | Using Simple Feature-level Server.")
            ens_acc = eval_with_proto(ensemble_model, test_loader, device, global_proto)

        elif server_strategy == 'advanced_iffi':
            method_name = 'OursV7+AdvancedIFFIServer'
            ensemble_model = WEnsembleFeatureNoise(model_list=local_models, weight_list=weights)
            global_proto = aggregate_local_protos(local_protos)
            logger.info("V7 Training | Using Advanced IFFI Server.")
            ens_acc = eval_with_proto(ensemble_model, test_loader, device, global_proto)
        else:
            raise ValueError(f"Unknown server_strategy: {server_strategy}")

        logger.info(f"The test accuracy of {method_name}: {ens_acc}")


        method_results[method_name].append(ens_acc)

        save_yaml_config(save_path + "/baselines_" + method_name +"_" + config['checkpoint']['result_file'], method_results)

def OneshotOursV8(trainset, test_loader, client_idx_map, config, device):
    logger.info('OneshotOursV8 - Final Version: Consensus-Start Progressive Alignment')
    
    global_model = get_train_models(
        model_name=config['server']['model_name'],
        num_classes=config['dataset']['num_classes'],
        mode='our'
    ,
        in_channel=config['dataset'].get('channels', 3))
    global_model.to(device)
    global_model.train()
    
    feature_dim = global_model.learnable_proto.shape[1]
    num_classes = config['dataset']['num_classes']
    fixed_anchors_etf = generate_etf_anchors(num_classes, feature_dim, device)
    logger.info(f"Initialized FINAL ETF anchors with shape: {fixed_anchors_etf.shape}")

    method_results, save_path, local_model_dir = defaultdict(list), *prepare_checkpoint_dir(config)
    if not os.path.exists(save_path + "/config.yaml"):
        save_yaml_config(save_path + "/config.yaml", config) 

    local_models = [copy.deepcopy(global_model) for _ in range(config['client']['num_clients'])]
    
    # Calculate a common "Consensus Start Point"
    initial_local_protos = [model.get_proto().detach().clone() for model in local_models]
    # Calculate the mean of all initial prototypes
    consensus_start_protos = torch.stack(initial_local_protos).mean(dim=0)
    logger.info("Calculated a shared CONSENSUS start point for all clients.")

    local_data_size = [len(client_idx_map[c]) for c in range(config['client']['num_clients'])]
    weights = [i/sum(local_data_size) for i in local_data_size] if config['server']['aggregated_by_datasize'] else [1/config['client']['num_clients']] * config['client']['num_clients']
    
    aug_transformer = get_gpu_augmentation(config['dataset']['data_name'], NORMALIZE_DICT[config['dataset']['data_name']], device)
    clients_sample_per_class = []

    total_rounds = config['server']['num_rounds']

    for cr in trange(config['server']['num_rounds']):
        logger.info(f"Round {cr} starts--------|")
        local_protos = []
        
        for c in range(config['client']['num_clients']):
            logger.info(f"Client {c} Starts Local Trainning--------|")
            client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])

            if cr == 0:
                clients_sample_per_class.append(generate_sample_per_class(config['dataset']['num_classes'], client_dataloader, len(client_idx_map[c])))

            local_model_c = ours_local_training(
                model=local_models[c],
                training_data=client_dataloader,
                test_dataloader=test_loader,
                start_epoch=cr * config['server']['local_epochs'],
                local_epochs=config['server']['local_epochs'],
                optim_name=config['server']['optimizer'],
                lr=config['server']['lr'],
                momentum=config['server']['momentum'],
                loss_name=config['server']['loss_name'],
                device=device,
                num_classes=config['dataset']['num_classes'],
                sample_per_class=clients_sample_per_class[c],
                aug_transformer=aug_transformer,
                client_model_dir=local_model_dir + f"/client_{c}",
                total_rounds=total_rounds,
                save_freq=config['checkpoint']['save_freq'],
                use_drcl=True,
                use_progressive_alignment=True,
                initial_protos=consensus_start_protos,
                fixed_anchors=fixed_anchors_etf,
                lambda_align=config.get('lambda_align_initial', 5.0)
            )
            
            local_models[c] = local_model_c

            local_protos.append(local_model_c.get_proto().detach())

        logger.info(f"Round {cr} Finish--------|")
        model_var_m, model_var_s = compute_local_model_variance(local_models)
        logger.info(f"Model variance: mean: {model_var_m}, sum: {model_var_s}")

        global_proto = aggregate_local_protos(local_protos)
        
        method_name = 'OneShotOursV8+Ensemble'
        ensemble_model = WEnsembleFeature(model_list=local_models, weight_list=weights)
        ens_proto_acc = eval_with_proto(copy.deepcopy(ensemble_model), test_loader, device, global_proto)
        logger.info(f"The test accuracy (with prototype) of {method_name}: {ens_proto_acc}")
        method_results[method_name].append(ens_proto_acc)

        save_yaml_config(save_path + "/baselines_" + method_name +"_" + config['checkpoint']['result_file'], method_results)

# OneshotOursV9 
def OneshotOursV9(trainset, test_loader, client_idx_map, config, device, server_strategy='simple_feature'):
    logger.info('OneshotOursV9 with a lot of things combined')

    v9_cfg = config.get('v9_config', {})
    use_adaptive_lambda = v9_cfg.get('use_adaptive_lambda', False)
    use_server_optimization = v9_cfg.get('use_server_optimization', False)
    logger.info(f"V9 Config: Adaptive Lambda (Plan A): {use_adaptive_lambda}, Server Optimization (Plan B): {use_server_optimization}")
    
    global_model = get_train_models(
        model_name=config['server']['model_name'],
        num_classes=config['dataset']['num_classes'],
        mode='our'
    ,
        in_channel=config['dataset'].get('channels', 3))
    global_model.to(device)
    global_model.train()

    # --- Core modification: Replace random anchors with ETF anchors ---
    feature_dim = global_model.learnable_proto.shape[1]
    num_classes = config['dataset']['num_classes']
    etf_anchors = generate_etf_anchors(num_classes, feature_dim, device)
    logger.info(f"Initialized ETF fixed anchors with shape: {etf_anchors.shape}")
    
    logger.info(v9_cfg)

    # Initialize Server-Side Learnable Components (for Plan B) ---
    if use_server_optimization:
        feature_dim = global_model.learnable_proto.shape[1]
        num_classes = config['dataset']['num_classes']
        # This is our trainable parameter on the server
        trainable_global_prototypes = torch.nn.Embedding(num_classes, feature_dim).to(device)
        # Initialize with the ideal ETF structure
        trainable_global_prototypes.weight.data.copy_(etf_anchors)
    else:
        # If Plan B is off, the "global prototype" is just the static ETF anchor
        trainable_global_prototypes = None 

    method_results = defaultdict(list)
    save_path, local_model_dir = prepare_checkpoint_dir(config)
    if not os.path.exists(save_path + "/config.yaml"):
        save_yaml_config(save_path + "/config.yaml", config) 

    local_models = [copy.deepcopy(global_model) for _ in range(config['client']['num_clients'])]
    
    local_data_size = [len(client_idx_map[c]) for c in range(config['client']['num_clients'])]
    if config['server']['aggregated_by_datasize']:
        weights = [i/sum(local_data_size) for i in local_data_size]
    else:
        weights = [1/config['client']['num_clients'] for _ in range(config['client']['num_clients'])]        
    
    aug_transformer = get_gpu_augmentation(config['dataset']['data_name'], NORMALIZE_DICT[config['dataset']['data_name']], device)

    clients_sample_per_class = []

    total_rounds = config['server']['num_rounds']

    for cr in trange(config['server']['num_rounds']):
        logger.info(f"Round {cr} starts--------|")

        # Determine the alignment target for this round
        if use_server_optimization:
            # The target is the current state of our *learned* global prototypes
            alignment_target = trainable_global_prototypes.weight.detach().clone()
        else:
            # The target is the static, ideal ETF anchor (as in V7)
            alignment_target = etf_anchors
        
        local_protos = []
        
        for c in range(config['client']['num_clients']):
            logger.info(f"Client {c} Starts Local Trainning--------|")
            client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])

            if cr == 0:
                clients_sample_per_class.append(generate_sample_per_class(config['dataset']['num_classes'], client_dataloader, len(client_idx_map[c])))

            if use_adaptive_lambda:
                current_lambda = calculate_adaptive_lambda(client_dataloader, config['dataset']['num_classes'], v9_cfg['lambda_min'], v9_cfg['lambda_max'], device)
            else:
                current_lambda = config.get('lambda_align_initial', 5.0) 

            # Call the exact same local training function as V6, but pass high-quality ETF anchors
            local_model_c = ours_local_training(
                model=copy.deepcopy(local_models[c]),
                training_data=client_dataloader,
                test_dataloader=test_loader,
                start_epoch=cr * config['server']['local_epochs'],
                local_epochs=config['server']['local_epochs'],
                optim_name=config['server']['optimizer'],
                lr=config['server']['lr'],
                momentum=config['server']['momentum'],
                loss_name=config['server']['loss_name'],
                device=device,
                num_classes=config['dataset']['num_classes'],
                sample_per_class=clients_sample_per_class[c],
                aug_transformer=aug_transformer,
                client_model_dir=local_model_dir + f"/client_{c}",
                total_rounds=total_rounds,
                save_freq=config['checkpoint']['save_freq'],
                use_drcl=True,
                fixed_anchors=alignment_target,
                lambda_align=current_lambda
            )
            
            local_models[c] = local_model_c

            local_protos.append(local_model_c.get_proto().detach())

        logger.info(f"Round {cr} Finish--------|")
        model_var_m, model_var_s = compute_local_model_variance(local_models)
        logger.info(f"Model variance: mean: {model_var_m}, sum: {model_var_s}")

        if use_server_optimization:
            trainable_global_prototypes = optimize_global_prototypes_on_server(
                local_protos,
                trainable_global_prototypes,
                etf_anchors, # Pass ideal ETF for regularization
                v9_cfg.get('server_lr', 0.001),
                v9_cfg.get('server_epochs', 20),
                v9_cfg.get('gamma_etf_reg', 0.1),
                device
            )
            # After learning, the final global proto for evaluation is the learned one
            global_proto = trainable_global_prototypes.weight.detach().clone()
        else:
            # If Plan B is off, aggregate protos the old way (simple average)
            global_proto = aggregate_local_protos(local_protos)

        
        if server_strategy == 'true_simple_output':
            method_name = 'OursV9+TrueSimpleOutputServer'
            ensemble_model = TrueSimpleEnsembleServer(model_list=local_models, weight_list=weights)
            logger.info("V9 Training | Using TRULY SIMPLE Output-level Server.")
            ens_acc = eval_output_ensemble(ensemble_model, test_loader, device)

        elif server_strategy == 'simple_feature':
            method_name = 'OursV9+SimpleFeatureServer'
            ensemble_model = WEnsembleFeature(model_list=local_models, weight_list=weights)
            global_proto = aggregate_local_protos(local_protos)
            logger.info("V9 Training | Using Simple Feature-level Server.")
            ens_acc = eval_with_proto(ensemble_model, test_loader, device, global_proto)

        elif server_strategy == 'advanced_iffi':
            method_name = 'OursV9+AdvancedIFFIServer'
            ensemble_model = WEnsembleFeatureNoise(model_list=local_models, weight_list=weights)
            global_proto = aggregate_local_protos(local_protos)
            logger.info("V9 Training | Using Advanced IFFI Server.")
            ens_acc = eval_with_proto(ensemble_model, test_loader, device, global_proto)
        else:
            raise ValueError(f"Unknown server_strategy: {server_strategy}")

        logger.info(f"The test accuracy of {method_name}: {ens_acc}")


        method_results[method_name].append(ens_acc)

        save_yaml_config(save_path + "/baselines_" + method_name +"_" + config['checkpoint']['result_file'], method_results)

def OneshotOursV10(trainset, test_loader, client_idx_map, config, device, **kwargs):
    logger.info('OneshotOursV10 with uncertainty weighting, Pre-heated with V9 Adaptive Lambda')
    
    # 1. --- Read Master Switch ---
    v10_cfg = config.get('v10_config', {})
    use_uncertainty_weighting = v10_cfg.get('use_uncertainty_weighting', False)
    if not use_uncertainty_weighting:
        logger.warning("Running V10 but 'use_uncertainty_weighting' is false. This will run like V7.")


    # 2. --- Standard Initialization ---
    # Check if we should use custom path
    use_pretrain_bool = config.get('DBCD', {}).get('use_pretrain', False)
    custom_pretrain_path = config.get('pretrain', {}).get('model_path', '')
    dataset_name_lower = config['dataset']['data_name'].lower()
    model_name = config['server']['model_name']
    
    # Construct expected path if not explicit: e.g. ./pretrain/tiny-imagenet_resnet18_centralized.pth
    expected_custom_path = os.path.join(custom_pretrain_path, f"{dataset_name_lower}_{model_name}_centralized.pth")
    
    use_pretrain_arg = False
    if use_pretrain_bool:
        if os.path.exists(expected_custom_path):
            use_pretrain_arg = expected_custom_path
            logger.info(f"[OneshotOursV10] Found custom centralized weights at {expected_custom_path}. Using them!")
        else:
            use_pretrain_arg = False # Changed from True
            logger.warning(f"[OneshotOursV10] Custom weights NOT found at {expected_custom_path}. Using Random Initialization (ImageNet Disabled).")

    global_model = get_train_models(
        model_name=config['server']['model_name'],
        num_classes=config['dataset']['num_classes'],
        mode='our',
        use_pretrain=use_pretrain_arg,
        in_channel=config['dataset'].get('channels', 3)
    )

    feature_dim = global_model.learnable_proto.shape[1]
    num_classes = config['dataset']['num_classes']
    etf_anchors = generate_etf_anchors(num_classes, feature_dim, device)
    
    method_results = defaultdict(list)
    save_path, local_model_dir = prepare_checkpoint_dir(config)
    if not os.path.exists(save_path + "/config.yaml"):
        save_yaml_config(save_path + "/config.yaml", config) 
    
    local_models = [copy.deepcopy(global_model) for _ in range(config['client']['num_clients'])]
    
    local_data_size = [len(client_idx_map[c]) for c in range(config['client']['num_clients'])]
    if config['server']['aggregated_by_datasize']:
        weights = [i/sum(local_data_size) for i in local_data_size]
    else:
        weights = [1/config['client']['num_clients'] for _ in range(config['client']['num_clients'])]        
    
    aug_transformer = get_supcon_transform(config['dataset']['data_name'])

    clients_sample_per_class = []

    # Pre-heating Initialization Logic ---
    logger.info("--- Pre-heating V10 models with V9's adaptive lambda strategy ---")
    for c in range(config['client']['num_clients']):
        # Get the client's data loader to calculate its data entropy
        client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])
        
        # Use V9's method to calculate the optimal starting lambda.
        # It's important that your config file has the v9_config section for this to work best.
        v9_cfg = config.get('v9_config', {})
        initial_lambda = calculate_adaptive_lambda(
            client_dataloader,
            config['dataset']['num_classes'],
            v9_cfg.get('lambda_min', 0.1),
            v9_cfg.get('lambda_max', 15.0), # Default max from the successful V9 run
            device
        )
        logger.info(f"Client {c}: Calculated initial lambda = {initial_lambda:.4f}")
        
        # Convert this initial lambda into the starting value for the learnable parameter.
        # We set initial Effective Lambda = adaptive_lambda by setting:
        #   log_sigma_sq_local = log(adaptive_lambda)
        #   log_sigma_sq_align = 0.0
        initial_log_lambda = torch.tensor(initial_lambda, device=device).log()
        
        # "Implant" the learnable parameters into each client model with these calculated values
        local_models[c].log_sigma_sq_local = torch.nn.Parameter(initial_log_lambda)
        local_models[c].log_sigma_sq_align = torch.nn.Parameter(torch.tensor(0.0, device=device))
        
    logger.info("--- V10 model pre-heating complete. Starting training. ---")

    sigma_lr_val  = v10_cfg.get('sigma_lr', 0)

    if sigma_lr_val <= 0:
        logger.warning("Sigma learning rate is not positive. Setting to default 0.005")
        sigma_lr_val = 0.005

    total_rounds = config['server']['num_rounds']
    
    for cr in trange(total_rounds):
        logger.info(f"Round {cr} starts--------|")
        local_protos = []
        
        for c in range(config['client']['num_clients']):
            logger.info(f"Client {c} Starts Local Trainning--------|")
            client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])

            if cr == 0:
                clients_sample_per_class.append(generate_sample_per_class(config['dataset']['num_classes'], client_dataloader, len(client_idx_map[c])))


            # 4. --- Call unified local training function, activate V10 mode ---
            # Note that we no longer need to pass any lambda value
            local_model_c = ours_local_training(
                model=copy.deepcopy(local_models[c]),
                training_data=client_dataloader,
                test_dataloader=test_loader,
                start_epoch=cr * config['server']['local_epochs'],
                local_epochs=config['server']['local_epochs'],
                optim_name=config['server']['optimizer'],
                lr=config['server']['lr'],
                momentum=config['server']['momentum'],
                loss_name=config['server']['loss_name'],
                device=device,
                num_classes=config['dataset']['num_classes'],
                sample_per_class=clients_sample_per_class[c],
                aug_transformer=aug_transformer,
                client_model_dir=local_model_dir + f"/client_{c}",
                total_rounds=total_rounds,
                use_drcl=True, # Still need to enable to calculate align_loss
                fixed_anchors=etf_anchors,
                use_uncertainty_weighting=True, # Activate V10
                sigma_lr=sigma_lr_val,
            )
            
            local_models[c] = local_model_c

            local_protos.append(local_model_c.get_proto().detach())
            
            # Print learned sigma values in log for analysis
            learned_sigma_local = torch.exp(local_model_c.log_sigma_sq_local).item()**0.5
            learned_sigma_align = torch.exp(local_model_c.log_sigma_sq_align).item()**0.5
            effective_lambda = (learned_sigma_local**2) / (learned_sigma_align**2)
            logger.info(f"Client {c} Learned Sigmas -> L_local: {learned_sigma_local:.4f}, L_align: {learned_sigma_align:.4f}. Effective Lambda: {effective_lambda:.4f}")

        logger.info(f"Round {cr} Finish--------|")
        model_var_m, model_var_s = compute_local_model_variance(local_models)
        logger.info(f"Model variance: mean: {model_var_m}, sum: {model_var_s}")

        method_name = 'OursV10+SimpleFeatureServer'
        ensemble_model = WEnsembleFeature(model_list=local_models, weight_list=weights)
        global_proto = aggregate_local_protos(local_protos)
        logger.info("V10 Training | Using Simple Feature-level Server.")
        ens_acc = eval_with_proto(ensemble_model, test_loader, device, global_proto)

        logger.info(f"The test accuracy of {method_name}: {ens_acc}")

        method_results[method_name].append(ens_acc)

        save_yaml_config(save_path + "/baselines_" + method_name +"_" + config['checkpoint']['result_file'], method_results)


def OneshotOursV11(trainset, test_loader, client_idx_map, config, device,annealing_strategy='none' , **kwargs):
    logger.info('OneshotOursV11: Consensus-Driven Dynamic Annealing with Uncertainty Weighting')
    
    # --- Standard initialization code ---
    # ... (Read config, init model, ETF anchors, data loaders etc.)
    v10_cfg = config.get('v10_config', {})
    
    use_pretrain_bool = config.get('DBCD', {}).get('use_pretrain', False)
    custom_pretrain_path = config.get('pretrain', {}).get('model_path', '')
    dataset_name_lower = config['dataset']['data_name'].lower()
    model_name = config['server']['model_name']
    expected_custom_path = os.path.join(custom_pretrain_path, f"{dataset_name_lower}_{model_name}_centralized.pth")
    
    use_pretrain_arg = False
    if use_pretrain_bool:
        if os.path.exists(expected_custom_path):
            use_pretrain_arg = expected_custom_path
            logger.info(f"[OneshotOursV11] Found custom centralized weights at {expected_custom_path}. Using them!")
        else:
            use_pretrain_arg = False # Changed from True
            logger.warning(f"[OneshotOursV11] Custom weights NOT found at {expected_custom_path}. Using Random Initialization (ImageNet Disabled).")

    global_model = get_train_models(
        model_name=config['server']['model_name'],
        num_classes=config['dataset']['num_classes'],
        mode='our',
        use_pretrain=use_pretrain_arg,
        in_channel=config['dataset'].get('channels', 3)
    )
    global_model.to(device)
    global_model.train()
    feature_dim = global_model.learnable_proto.shape[1]
    num_classes = config['dataset']['num_classes']
    etf_anchors = generate_etf_anchors(num_classes, feature_dim, device)
    method_results = defaultdict(list)
    save_path, local_model_dir = prepare_checkpoint_dir(config)
    if not os.path.exists(save_path + "/config.yaml"):
        save_yaml_config(save_path + "/config.yaml", config)
    local_models = [copy.deepcopy(global_model) for _ in range(config['client']['num_clients'])]
    local_data_size = [len(client_idx_map[c]) for c in range(config['client']['num_clients'])]
    if config['server']['aggregated_by_datasize']:
        weights = [i/sum(local_data_size) for i in local_data_size]
    else:
        weights = [1/config['client']['num_clients'] for _ in range(config['client']['num_clients'])]        
    aug_transformer = get_supcon_transform(config['dataset']['data_name'])
    clients_sample_per_class = []
    
    # --- Lambda warm-up ---
    logger.info("--- Pre-heating V11 models with V9's adaptive lambda strategy ---")
    for c in range(config['client']['num_clients']):
        # ... (Calculate initial_lambda and implant into sigma parameters)
        client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])
        v9_cfg = config.get('v9_config', {})
        initial_lambda = calculate_adaptive_lambda(
            client_dataloader,
            config['dataset']['num_classes'],
            v9_cfg.get('lambda_min', 0.1),
            v9_cfg.get('lambda_max', 15.0), # Default max from the successful V9 run
            device
        )
        logger.info(f"Client {c}: Calculated initial lambda = {initial_lambda:.4f}")
        initial_log_lambda = torch.tensor(initial_lambda, device=device).log()
        local_models[c].log_sigma_sq_local = torch.nn.Parameter(initial_log_lambda)
        local_models[c].log_sigma_sq_align = torch.nn.Parameter(torch.tensor(0.0, device=device))
    logger.info("--- V11 model pre-heating complete. ---")

    sigma_lr_val = v10_cfg.get('sigma_lr', 0.005)

    total_rounds = config['server']['num_rounds']

    # --- Lambda annealing ---
    # Initialize state variables for consensus-driven annealing
    initial_proto_std = None
    current_annealing_factor = 1.0  # Apply no annealing in the first round

    for cr in trange(total_rounds):
        logger.info(f"Round {cr} starts---| Annealing Factor for this round: {current_annealing_factor:.4f}")
        local_protos = []
        
        for c in range(config['client']['num_clients']):
            logger.info(f"Client {c} Starts Local Trainning--------|")
            client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])

            if cr == 0:
                clients_sample_per_class.append(generate_sample_per_class(config['dataset']['num_classes'], client_dataloader, len(client_idx_map[c])))

            local_model_c = ours_local_training(
                model=copy.deepcopy(local_models[c]),
                training_data=client_dataloader,
                test_dataloader=test_loader,
                start_epoch=cr * config['server']['local_epochs'],
                local_epochs=config['server']['local_epochs'],
                optim_name=config['server']['optimizer'],
                lr=config['server']['lr'],
                momentum=config['server']['momentum'],
                loss_name=config['server']['loss_name'],
                device=device,
                num_classes=config['dataset']['num_classes'],
                sample_per_class=clients_sample_per_class[c],
                aug_transformer=aug_transformer,
                client_model_dir=local_model_dir + f"/client_{c}",
                total_rounds=total_rounds,
                use_drcl=True, # Still need to enable to calculate align_loss
                fixed_anchors=etf_anchors,
                use_uncertainty_weighting=True, # Activate V10
                sigma_lr=sigma_lr_val,
                annealing_factor=current_annealing_factor # Pass annealing factor for current round
            )
            
            local_models[c] = local_model_c

            local_protos.append(local_model_c.get_proto().detach())

            # Print learned sigma values in log for analysis
            learned_sigma_local = torch.exp(local_model_c.log_sigma_sq_local).item()**0.5
            learned_sigma_align = torch.exp(local_model_c.log_sigma_sq_align).item()**0.5
            effective_lambda = (learned_sigma_local**2) / (learned_sigma_align**2)
            logger.info(f"Client {c} Learned Sigmas -> L_local: {learned_sigma_local:.4f}, L_align: {learned_sigma_align:.4f}. Effective Lambda: {effective_lambda:.4f}")

        # ================== [ V11 Core Innovation Logic START ] ==================
        # Calculate annealing factor for next round (cr + 1) after each round finishes
        next_round_progress = (cr + 1) / total_rounds

        if annealing_strategy == 'linear':
            next_annealing_factor = 1.0 - next_round_progress
        
        elif annealing_strategy == 'cosine':
            next_annealing_factor = 0.5 * (1.0 + math.cos(math.pi * next_round_progress))
        
        elif annealing_strategy == 'consensus':
            local_protos_tensor = torch.stack(local_protos)
            current_proto_std = torch.std(local_protos_tensor, dim=0).mean().item()
            if initial_proto_std is None:
                initial_proto_std = current_proto_std
            
            if initial_proto_std > 1e-6:
                next_annealing_factor = min(1.0, current_proto_std / initial_proto_std)
            else:
                next_annealing_factor = 1.0
        
        else: # 'none'
            next_annealing_factor = 1.0
            
        current_annealing_factor = max(0.0, next_annealing_factor)
        
        logger.info(f"Current annealing strategy: {annealing_strategy}. Next round's annealing factor set to: {current_annealing_factor:.4f}")
        # =================== [ V11 Core Innovation Logic END ] ===================

        logger.info(f"Round {cr} Finish--------|")
        
        # ... [Same evaluation and save logic as V10]
        method_name = 'OursV11+SimpleFeatureServer'
        ensemble_model = WEnsembleFeature(model_list=local_models, weight_list=weights)
        global_proto = aggregate_local_protos(local_protos)
        ens_acc = eval_with_proto(ensemble_model, test_loader, device, global_proto)
        logger.info(f"The test accuracy of {method_name}: {ens_acc}")
        method_results[method_name].append(ens_acc)
        save_yaml_config(save_path + "/baselines_" + method_name +"_" + config['checkpoint']['result_file'], method_results)

def OneshotOursV12(trainset, test_loader, client_idx_map, config, device, **kwargs):
    logger.info('OneshotOursV12: Uncertainty Weighting with INTERNAL Dynamic Task Attenuation')
    
    # --- Standard Initialization ---
    v10_cfg = config.get('v10_config', {})
    # --- 标准初始化 ---
    v10_cfg = config.get('v10_config', {})
    
    use_pretrain_bool = config.get('DBCD', {}).get('use_pretrain', False)
    custom_pretrain_path = config.get('pretrain', {}).get('model_path', '')
    dataset_name_lower = config['dataset']['data_name'].lower()
    model_name = config['server']['model_name']
    expected_custom_path = os.path.join(custom_pretrain_path, f"{dataset_name_lower}_{model_name}_centralized.pth")
    
    use_pretrain_arg = False
    if use_pretrain_bool:
        if os.path.exists(expected_custom_path):
            use_pretrain_arg = expected_custom_path
            logger.info(f"[OneshotOursV12] Found custom centralized weights at {expected_custom_path}. Using them!")
        else:
            use_pretrain_arg = False # Changed from True
            logger.warning(f"[OneshotOursV12] Custom weights NOT found at {expected_custom_path}. Using Random Initialization (ImageNet Disabled).")

    global_model = get_train_models(
        model_name=config['server']['model_name'],
        num_classes=config['dataset']['num_classes'],
        mode='our',
        use_pretrain=use_pretrain_arg,
        in_channel=config['dataset'].get('channels', 3)
    )
    feature_dim = global_model.learnable_proto.shape[1]
    num_classes = config['dataset']['num_classes']
    etf_anchors = generate_etf_anchors(num_classes, feature_dim, device)
    method_results = defaultdict(list)
    save_path, local_model_dir = prepare_checkpoint_dir(config)
    if not os.path.exists(save_path + "/config.yaml"):
        save_yaml_config(save_path + "/config.yaml", config)
    local_models = [copy.deepcopy(global_model) for _ in range(config['client']['num_clients'])]
    local_data_size = [len(client_idx_map[c]) for c in range(config['client']['num_clients'])]
    if config['server']['aggregated_by_datasize']:
        weights = [i/sum(local_data_size) for i in local_data_size]
    else:
        weights = [1/config['client']['num_clients'] for _ in range(config['client']['num_clients'])]        
    aug_transformer = get_supcon_transform(config['dataset']['data_name'])
    clients_sample_per_class = []
    
    # --- Same lambda warm-up logic as V11 ---
    logger.info("--- Pre-heating V12 models with V9's adaptive lambda strategy ---")
    for c in range(config['client']['num_clients']):
        client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])
        v9_cfg = config.get('v9_config', {})
        initial_lambda = calculate_adaptive_lambda(
            client_dataloader,
            config['dataset']['num_classes'],
            v9_cfg.get('lambda_min', 0.1),
            v9_cfg.get('lambda_max', 15.0),
            device
        )
        logger.info(f"Client {c}: Calculated initial lambda = {initial_lambda:.4f}")
        initial_log_lambda = torch.tensor(initial_lambda, device=device).log()
        local_models[c].log_sigma_sq_local = torch.nn.Parameter(initial_log_lambda)
        local_models[c].log_sigma_sq_align = torch.nn.Parameter(torch.tensor(0.0, device=device))
    logger.info("--- V12 model pre-heating complete. ---")

    sigma_lr_val = v10_cfg.get('sigma_lr', 0.005)
    total_rounds = config['server']['num_rounds']

    for cr in trange(total_rounds):
        logger.info(f"Round {cr} starts--------|")
        local_protos = []
        
        for c in range(config['client']['num_clients']):
            logger.info(f"Client {c} Starts Local Trainning--------|")
            client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])

            if cr == 0:
                clients_sample_per_class.append(generate_sample_per_class(config['dataset']['num_classes'], client_dataloader, len(client_idx_map[c])))

            # --- Core call change: Activate new V12 logic ---
            local_model_c = ours_local_training(
                model=copy.deepcopy(local_models[c]),
                training_data=client_dataloader,
                test_dataloader=test_loader,
                start_epoch=cr * config['server']['local_epochs'],
                local_epochs=config['server']['local_epochs'],
                optim_name=config['server']['optimizer'],
                lr=config['server']['lr'],
                momentum=config['server']['momentum'],
                loss_name=config['server']['loss_name'],
                device=device,
                num_classes=config['dataset']['num_classes'],
                sample_per_class=clients_sample_per_class[c],
                aug_transformer=aug_transformer,
                client_model_dir=local_model_dir + f"/client_{c}",
                total_rounds=total_rounds,
                use_drcl=True,
                fixed_anchors=etf_anchors,
                use_uncertainty_weighting=True,
                sigma_lr=sigma_lr_val,
                # New switch to activate V12 internal annealing logic
                use_dynamic_task_attenuation=True 
            )
            
            local_models[c] = local_model_c
            local_protos.append(local_model_c.get_proto().detach())

            # Log Analysis
            # 1. As before, get final sigma values from returned model
            log_sigma_sq_local_val = local_model_c.log_sigma_sq_local.item()
            log_sigma_sq_align_val = local_model_c.log_sigma_sq_align.item()

            # 2. Calculate "Raw Lambda" learned by sigma parameters
            #    This value will grow explosively in later stages
            raw_lambda = math.exp(log_sigma_sq_local_val - log_sigma_sq_align_val)

            # 3. Calculate "Meta-Annealing" factor s(p) at the end of current local training
            #    This requires knowing total training steps and current step
            total_training_epochs = config['server']['num_rounds'] * config['server']['local_epochs']
            
            # cr is current round index starting from 0
            end_epoch_of_this_round = (cr + 1) * config['server']['local_epochs']
            
            global_progress = end_epoch_of_this_round / total_training_epochs
            
            # Assume s(p) is linear decay s(p) = 1 - p
            # Use max(0.0, ...) to ensure it doesn't become negative
            s_p_value = max(0.0, 1.0 - global_progress)

            # 4. Calculate "Truly Effective Lambda" that actually acts on model weights W
            truly_effective_lambda = raw_lambda * s_p_value

            # 5. Print brand new, highly informative log
            log_message = (
                f"Client {c} Post-Training State -> "
                f"Raw λ: {raw_lambda:9.4f} "
                f"| s(p): {s_p_value:.3f} "
                f"| Truly Effective λ (for W): {truly_effective_lambda:9.4f}"
            )
            logger.info(log_message)


        logger.info(f"Round {cr} Finish--------|")
        
        # --- Evaluation and Save Logic ---
        method_name = 'OursV12+SimpleFeatureServer'
        ensemble_model = WEnsembleFeature(model_list=local_models, weight_list=weights)
        global_proto = aggregate_local_protos(local_protos)
        ens_acc = eval_with_proto(ensemble_model, test_loader, device, global_proto)
        logger.info(f"The test accuracy of {method_name}: {ens_acc}")
        method_results[method_name].append(ens_acc)
        save_yaml_config(save_path + "/baselines_" + method_name +"_" + config['checkpoint']['result_file'], method_results)


def OneshotOursV13(trainset, test_loader, client_idx_map, config, device, gamma_reg, lambda_max=50.0, **kwargs):
    logger.info('OneshotOursV13: V12 with stability_anchor')
    
    # --- Standard Initialization ---
    v10_cfg = config.get('v10_config', {})
    # --- 标准初始化 ---
    v10_cfg = config.get('v10_config', {})
    
    use_pretrain_bool = config.get('DBCD', {}).get('use_pretrain', False)
    custom_pretrain_path = config.get('pretrain', {}).get('model_path', '')
    dataset_name_lower = config['dataset']['data_name'].lower()
    model_name = config['server']['model_name']
    expected_custom_path = os.path.join(custom_pretrain_path, f"{dataset_name_lower}_{model_name}_centralized.pth")
    
    use_pretrain_arg = False
    if use_pretrain_bool:
        if os.path.exists(expected_custom_path):
            use_pretrain_arg = expected_custom_path
            logger.info(f"[OneshotOursV13] Found custom centralized weights at {expected_custom_path}. Using them!")
        else:
            use_pretrain_arg = False # Changed from True
            logger.warning(f"[OneshotOursV13] Custom weights NOT found at {expected_custom_path}. Using Random Initialization (ImageNet Disabled).")

    global_model = get_train_models(
        model_name=config['server']['model_name'],
        num_classes=config['dataset']['num_classes'],
        mode='our',
        use_pretrain=use_pretrain_arg,
        in_channel=config['dataset'].get('channels', 3)
    )
    feature_dim = global_model.learnable_proto.shape[1]
    num_classes = config['dataset']['num_classes']
    etf_anchors = generate_etf_anchors(num_classes, feature_dim, device)
    method_results = defaultdict(list)
    save_path, local_model_dir = prepare_checkpoint_dir(config)
    if not os.path.exists(save_path + "/config.yaml"):
        save_yaml_config(save_path + "/config.yaml", config)
    local_models = [copy.deepcopy(global_model) for _ in range(config['client']['num_clients'])]
    local_data_size = [len(client_idx_map[c]) for c in range(config['client']['num_clients'])]
    if config['server']['aggregated_by_datasize']:
        weights = [i/sum(local_data_size) for i in local_data_size]
    else:
        weights = [1/config['client']['num_clients'] for _ in range(config['client']['num_clients'])]        
    aug_transformer = get_supcon_transform(config['dataset']['data_name'])
    clients_sample_per_class = []
    
    # --- Same lambda warm-up logic as V11 ---
    logger.info("--- Pre-heating V13 models with V9's adaptive lambda strategy ---")
    for c in range(config['client']['num_clients']):
        client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])
        v9_cfg = config.get('v9_config', {})
        initial_lambda = calculate_adaptive_lambda(
            client_dataloader,
            config['dataset']['num_classes'],
            v9_cfg.get('lambda_min', 0.1),
            v9_cfg.get('lambda_max', 15.0),
            device
        )
        logger.info(f"Client {c}: Calculated initial lambda = {initial_lambda:.4f}")
        initial_log_lambda = torch.tensor(initial_lambda, device=device).log()
        local_models[c].log_sigma_sq_local = torch.nn.Parameter(initial_log_lambda)
        local_models[c].log_sigma_sq_align = torch.nn.Parameter(torch.tensor(0.0, device=device))
    logger.info("--- V13 model pre-heating complete. ---")

    logger.info(f"Gamma for stability anchor regularization: {gamma_reg}")

    sigma_lr_val = v10_cfg.get('sigma_lr', 0.005)
    total_rounds = config['server']['num_rounds']

    for cr in trange(total_rounds):
        logger.info(f"Round {cr} starts--------|")
        local_protos = []
        
        for c in range(config['client']['num_clients']):
            logger.info(f"Client {c} Starts Local Trainning--------|")
            client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])

            if cr == 0:
                clients_sample_per_class.append(generate_sample_per_class(config['dataset']['num_classes'], client_dataloader, len(client_idx_map[c])))

            # --- Core call change: Activate new V12 logic ---
            local_model_c = ours_local_training(
                model=copy.deepcopy(local_models[c]),
                training_data=client_dataloader,
                test_dataloader=test_loader,
                start_epoch=cr * config['server']['local_epochs'],
                local_epochs=config['server']['local_epochs'],
                optim_name=config['server']['optimizer'],
                lr=config['server']['lr'],
                momentum=config['server']['momentum'],
                loss_name=config['server']['loss_name'],
                device=device,
                num_classes=config['dataset']['num_classes'],
                sample_per_class=clients_sample_per_class[c],
                aug_transformer=aug_transformer,
                client_model_dir=local_model_dir + f"/client_{c}",
                total_rounds=total_rounds,
                use_drcl=True,
                fixed_anchors=etf_anchors,
                use_uncertainty_weighting=True,
                sigma_lr=sigma_lr_val,
                # New switch to activate V12 internal annealing logic
                use_dynamic_task_attenuation=True,
                gamma_reg = gamma_reg,
                lambda_max = lambda_max
            )
            
            local_models[c] = local_model_c
            local_protos.append(local_model_c.get_proto().detach())

            # Log Analysis
            # 1. As before, get final sigma values from returned model
            log_sigma_sq_local_val = local_model_c.log_sigma_sq_local.item()
            log_sigma_sq_align_val = local_model_c.log_sigma_sq_align.item()

            # 2. Calculate "Raw Lambda" learned by sigma parameters
            #    This value will grow explosively in later stages
            raw_lambda = math.exp(log_sigma_sq_local_val - log_sigma_sq_align_val)

            # 3. Calculate "Meta-Annealing" factor s(p) at the end of current local training
            #    This requires knowing total training steps and current step
            total_training_epochs = config['server']['num_rounds'] * config['server']['local_epochs']
            
            # cr is current round index starting from 0
            end_epoch_of_this_round = (cr + 1) * config['server']['local_epochs']
            
            global_progress = end_epoch_of_this_round / total_training_epochs
            
            # Assume s(p) is linear decay s(p) = 1 - p
            # Use max(0.0, ...) to ensure it doesn't become negative
            s_p_value = max(0.0, 1.0 - global_progress)

            # 4. Calculate "Truly Effective Lambda" that actually acts on model weights W
            truly_effective_lambda = raw_lambda * s_p_value

            # 5. Print brand new, highly informative log
            log_message = (
                f"Client {c} Post-Training State -> "
                f"Raw λ: {raw_lambda:9.4f} "
                f"| s(p): {s_p_value:.3f} "
                f"| Truly Effective λ (for W): {truly_effective_lambda:9.4f}"
            )
            logger.info(log_message)


        logger.info(f"Round {cr} Finish--------|")
        
        # --- Evaluation and Save Logic ---
        method_name = 'OursV13+SimpleFeatureServer'
        ensemble_model = WEnsembleFeature(model_list=local_models, weight_list=weights)
        global_proto = aggregate_local_protos(local_protos)
        ens_acc = eval_with_proto(ensemble_model, test_loader, device, global_proto)
        logger.info(f"The test accuracy of {method_name}: {ens_acc}")
        method_results[method_name].append(ens_acc)
        save_yaml_config(save_path + "/baselines_" + method_name +"_" + config['checkpoint']['result_file'], method_results)

def OneshotOursV14(trainset, test_loader, client_idx_map, config, device, gamma_reg, lambda_max=50.0, **kwargs):
    logger.info('OneshotOursV14: V13 with lambda_max_threshold ')
    
    # --- Standard Initialization ---
    v10_cfg = config.get('v10_config', {})
    
    use_pretrain_bool = config.get('DBCD', {}).get('use_pretrain', False)
    use_imagenet_pretrain = config.get('DBCD', {}).get('use_imagenet_pretrain', False)
    custom_pretrain_path = config.get('pretrain', {}).get('model_path', '')
    dataset_name_lower = config['dataset']['data_name'].lower()
    model_name = config['server']['model_name']
    expected_custom_path = os.path.join(custom_pretrain_path, f"{dataset_name_lower}_{model_name}_centralized.pth")
    
    use_pretrain_arg = False
    if use_pretrain_bool:
        if os.path.exists(expected_custom_path):
            use_pretrain_arg = expected_custom_path
            logger.info(f"[OneshotOursV14] Found custom centralized weights at {expected_custom_path}. Using them!")
        else:
            use_pretrain_arg = False
            logger.warning(f"[OneshotOursV14] Custom weights NOT found at {expected_custom_path}. Using Random Initialization.")
    elif use_imagenet_pretrain:
        use_pretrain_arg = True
        logger.info("[OneshotOursV14] Loading ImageNet pretrained weights for ResNet-18.")

    global_model = get_train_models(
        model_name=config['server']['model_name'],
        num_classes=config['dataset']['num_classes'],
        mode='our',
        use_pretrain=use_pretrain_arg,
        in_channel=config['dataset'].get('channels', 3)
    )
    feature_dim = global_model.learnable_proto.shape[1]
    num_classes = config['dataset']['num_classes']
    etf_anchors = generate_etf_anchors(num_classes, feature_dim, device)
    method_results = defaultdict(list)
    save_path, local_model_dir = prepare_checkpoint_dir(config)
    if not os.path.exists(save_path + "/config.yaml"):
        save_yaml_config(save_path + "/config.yaml", config)
    local_models = [copy.deepcopy(global_model) for _ in range(config['client']['num_clients'])]
    local_data_size = [len(client_idx_map[c]) for c in range(config['client']['num_clients'])]
    if config['server']['aggregated_by_datasize']:
        weights = [i/sum(local_data_size) for i in local_data_size]
    else:
        weights = [1/config['client']['num_clients'] for _ in range(config['client']['num_clients'])]        
    aug_transformer = get_supcon_transform(config['dataset']['data_name'])
    clients_sample_per_class = []
    
    # --- Same lambda warm-up logic as V11 ---
    logger.info("--- Pre-heating V14 models with V9's adaptive lambda strategy ---")
    for c in range(config['client']['num_clients']):
        client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])
        v9_cfg = config.get('v9_config', {})
        initial_lambda = calculate_adaptive_lambda(
            client_dataloader,
            config['dataset']['num_classes'],
            v9_cfg.get('lambda_min', 0.1),
            v9_cfg.get('lambda_max', 15.0),
            device
        )
        logger.info(f"Client {c}: Calculated initial lambda = {initial_lambda:.4f}")
        initial_log_lambda = torch.tensor(initial_lambda, device=device).log()
        local_models[c].log_sigma_sq_local = torch.nn.Parameter(initial_log_lambda)
        local_models[c].log_sigma_sq_align = torch.nn.Parameter(torch.tensor(0.0, device=device))
    logger.info("--- V14 model pre-heating complete. ---")

    logger.info(f"Gamma for stability anchor regularization: {gamma_reg}")

    sigma_lr_val = v10_cfg.get('sigma_lr', 0.005)
    total_rounds = config['server']['num_rounds']

    for cr in trange(total_rounds):
        logger.info(f"Round {cr} starts--------|")
        local_protos = []
        
        for c in range(config['client']['num_clients']):
            logger.info(f"Client {c} Starts Local Trainning--------|")
            client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])

            if cr == 0:
                clients_sample_per_class.append(generate_sample_per_class(config['dataset']['num_classes'], client_dataloader, len(client_idx_map[c])))

            # --- Core call change: Activate new V12 logic ---
            local_model_c = ours_local_training(
                model=copy.deepcopy(local_models[c]),
                training_data=client_dataloader,
                test_dataloader=test_loader,
                start_epoch=cr * config['server']['local_epochs'],
                local_epochs=config['server']['local_epochs'],
                optim_name=config['server']['optimizer'],
                lr=config['server']['lr'],
                momentum=config['server']['momentum'],
                loss_name=config['server']['loss_name'],
                device=device,
                num_classes=config['dataset']['num_classes'],
                sample_per_class=clients_sample_per_class[c],
                aug_transformer=aug_transformer,
                client_model_dir=local_model_dir + f"/client_{c}",
                total_rounds=total_rounds,
                use_drcl=True,
                fixed_anchors=etf_anchors,
                use_uncertainty_weighting=True,
                sigma_lr=sigma_lr_val,
                # New switch to activate V12 internal annealing logic
                use_dynamic_task_attenuation=True,
                gamma_reg = gamma_reg,
                lambda_max = lambda_max
            )
            
            local_models[c] = local_model_c
            local_protos.append(local_model_c.get_proto().detach())

            # Log Analysis
            # 1. As before, get final sigma values from returned model
            log_sigma_sq_local_val = local_model_c.log_sigma_sq_local.item()
            log_sigma_sq_align_val = local_model_c.log_sigma_sq_align.item()

            # 2. Calculate "Raw Lambda" learned by sigma parameters
            #    This value will grow explosively in later stages
            raw_lambda = math.exp(log_sigma_sq_local_val - log_sigma_sq_align_val)

            # 3. Calculate "Meta-Annealing" factor s(p) at the end of current local training
            #    This requires knowing total training steps and current step
            total_training_epochs = config['server']['num_rounds'] * config['server']['local_epochs']
            
            # cr is current round index starting from 0
            end_epoch_of_this_round = (cr + 1) * config['server']['local_epochs']
            
            global_progress = end_epoch_of_this_round / total_training_epochs
            
            # Assume s(p) is linear decay s(p) = 1 - p
            # Use max(0.0, ...) to ensure it doesn't become negative
            s_p_value = max(0.0, 1.0 - global_progress)

            # 4. Calculate "Truly Effective Lambda" that actually acts on model weights W
            truly_effective_lambda = raw_lambda * s_p_value

            # 5. Print brand new, highly informative log
            log_message = (
                f"Client {c} Post-Training State -> "
                f"Raw λ: {raw_lambda:9.4f} "
                f"| s(p): {s_p_value:.3f} "
                f"| Truly Effective λ (for W): {truly_effective_lambda:9.4f}"
            )
            logger.info(log_message)


        logger.info(f"Round {cr} Finish--------|")
        
        # --- Evaluation and Save Logic ---
        method_name = 'OursV14+SimpleFeatureServer'
        ensemble_model = WEnsembleFeature(model_list=local_models, weight_list=weights)
        global_proto = aggregate_local_protos(local_protos)
        ens_acc = eval_with_proto(ensemble_model, test_loader, device, global_proto)
        logger.info(f"The test accuracy of {method_name}: {ens_acc}")
        method_results[method_name].append(ens_acc)
        save_yaml_config(save_path + "/baselines_" + method_name +"_" + config['checkpoint']['result_file'], method_results)


def OneshotFAFIFedAvg(trainset, test_loader, client_idx_map, config, device, lambda_val=0):
    logger.info('OneshotFAFIFedAvg: OursV4 + FedAvg (Parameter Averaging)')
    
    global_model = get_train_models(
        model_name=config['server']['model_name'],
        num_classes=config['dataset']['num_classes'],
        mode='our'
    ,
        in_channel=config['dataset'].get('channels', 3))
    global_model.to(device)
    global_model.train()

    # V4 does not use ETF anchors
    # feature_dim = global_model.learnable_proto.shape[1]
    # num_classes = config['dataset']['num_classes']
    # fixed_anchors = generate_etf_anchors(num_classes, feature_dim, device)
    # logger.info(f"Initialized ETF fixed anchors with shape: {fixed_anchors.shape}")

    method_results = defaultdict(list)
    save_path, local_model_dir = prepare_checkpoint_dir(config)
    if not os.path.exists(save_path + "/config.yaml"):
        save_yaml_config(save_path + "/config.yaml", config) 

    local_models = [copy.deepcopy(global_model) for _ in range(config['client']['num_clients'])]
    
    local_data_size = [len(client_idx_map[c]) for c in range(config['client']['num_clients'])]
    if config['server']['aggregated_by_datasize']:
        weights = [i/sum(local_data_size) for i in local_data_size]
    else:
        weights = [1/config['client']['num_clients'] for _ in range(config['client']['num_clients'])]        
    
    aug_transformer = get_gpu_augmentation(config['dataset']['data_name'], device)

    clients_sample_per_class = []

    total_rounds = config['server']['num_rounds']

    for cr in trange(config['server']['num_rounds']):
        logger.info(f"Round {cr} starts--------|")
        
        for c in range(config['client']['num_clients']):
            logger.info(f"Client {c} Starts Local Trainning--------|")
            client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])

            if cr == 0:
                clients_sample_per_class.append(generate_sample_per_class(config['dataset']['num_classes'], client_dataloader, len(client_idx_map[c])))

            # V4 does not use lambda_align
            # if (lambda_val > 0):
            #     lambda_align_initial = lambda_val
            # else:
            #     lambda_align_initial = config.get('lambda_align_initial', 5.0)

            local_model_c = ours_local_training(
                model=copy.deepcopy(local_models[c]),
                training_data=client_dataloader,
                test_dataloader=test_loader,
                start_epoch=cr * config['server']['local_epochs'],
                local_epochs=config['server']['local_epochs'],
                optim_name=config['server']['optimizer'],
                lr=config['server']['lr'],
                momentum=config['server']['momentum'],
                loss_name=config['server']['loss_name'],
                device=device,
                num_classes=config['dataset']['num_classes'],
                sample_per_class=clients_sample_per_class[c],
                aug_transformer=aug_transformer,
                client_model_dir=local_model_dir + f"/client_{c}",
                total_rounds=total_rounds,
                save_freq=config['checkpoint']['save_freq'],
                # Removed V7 params: use_drcl, fixed_anchors, lambda_align
                use_fafi=True
            )
            
            local_models[c] = local_model_c

        logger.info(f"Round {cr} Finish--------|")
        model_var_m, model_var_s = compute_local_model_variance(local_models)
        logger.info(f"Model variance: mean: {model_var_m}, sum: {model_var_s}")

        # FedAvg Aggregation Logic
        method_name = 'FAFIFedAvg'
        aggregated_model = parameter_averaging(local_models, weights)
        acc = test_acc_our_model(aggregated_model, test_loader, device)
        logger.info(f"The test accuracy of {method_name}: {acc}")
        method_results[method_name].append(acc)
        
        # Broadcast the aggregated model to all clients
        local_models = [copy.deepcopy(aggregated_model) for _ in range(config['client']['num_clients'])]

        save_yaml_config(save_path + "/baselines_" + method_name +"_" + config['checkpoint']['result_file'], method_results)


def OneshotAURORAFedAvg(trainset, test_loader, client_idx_map, config, device, gamma_reg, lambda_max=50.0, **kwargs):
    logger.info('OneshotAURORAFedAvg: OursV14 + FedAvg (Parameter Averaging)')
    
    v10_cfg = config.get('v10_config', {})
    global_model = get_train_models(
        model_name=config['server']['model_name'],
        num_classes=config['dataset']['num_classes'],
        mode='our'
    ,
        in_channel=config['dataset'].get('channels', 3))
    feature_dim = global_model.learnable_proto.shape[1]
    num_classes = config['dataset']['num_classes']
    etf_anchors = generate_etf_anchors(num_classes, feature_dim, device)
    method_results = defaultdict(list)
    save_path, local_model_dir = prepare_checkpoint_dir(config)
    if not os.path.exists(save_path + "/config.yaml"):
        save_yaml_config(save_path + "/config.yaml", config)
    local_models = [copy.deepcopy(global_model) for _ in range(config['client']['num_clients'])]
    local_data_size = [len(client_idx_map[c]) for c in range(config['client']['num_clients'])]
    if config['server']['aggregated_by_datasize']:
        weights = [i/sum(local_data_size) for i in local_data_size]
    else:
        weights = [1/config['client']['num_clients'] for _ in range(config['client']['num_clients'])]        
    aug_transformer = get_supcon_transform(config['dataset']['data_name'])
    clients_sample_per_class = []
    
    logger.info("--- Pre-heating AURORA models with V9's adaptive lambda strategy ---")
    for c in range(config['client']['num_clients']):
        client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])
        v9_cfg = config.get('v9_config', {})
        initial_lambda = calculate_adaptive_lambda(
            client_dataloader,
            config['dataset']['num_classes'],
            v9_cfg.get('lambda_min', 0.1),
            v9_cfg.get('lambda_max', 15.0),
            device
        )
        logger.info(f"Client {c}: Calculated initial lambda = {initial_lambda:.4f}")
        initial_log_lambda = torch.tensor(initial_lambda, device=device).log()
        local_models[c].log_sigma_sq_local = torch.nn.Parameter(initial_log_lambda)
        local_models[c].log_sigma_sq_align = torch.nn.Parameter(torch.tensor(0.0, device=device))
    logger.info("--- AURORA model pre-heating complete. ---")

    logger.info(f"Gamma for stability anchor regularization: {gamma_reg}")
    sigma_lr_val = v10_cfg.get('sigma_lr', 0.005)
    total_rounds = config['server']['num_rounds']

    for cr in trange(total_rounds):
        logger.info(f"Round {cr} starts--------|")
        
        for c in range(config['client']['num_clients']):
            logger.info(f"Client {c} Starts Local Trainning--------|")
            client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])

            if cr == 0:
                clients_sample_per_class.append(generate_sample_per_class(config['dataset']['num_classes'], client_dataloader, len(client_idx_map[c])))

            local_model_c = ours_local_training(
                model=copy.deepcopy(local_models[c]),
                training_data=client_dataloader,
                test_dataloader=test_loader,
                start_epoch=cr * config['server']['local_epochs'],
                local_epochs=config['server']['local_epochs'],
                optim_name=config['server']['optimizer'],
                lr=config['server']['lr'],
                momentum=config['server']['momentum'],
                loss_name=config['server']['loss_name'],
                device=device,
                num_classes=config['dataset']['num_classes'],
                sample_per_class=clients_sample_per_class[c],
                aug_transformer=aug_transformer,
                client_model_dir=local_model_dir + f"/client_{c}",
                total_rounds=total_rounds,
                use_drcl=True,
                fixed_anchors=etf_anchors,
                use_uncertainty_weighting=True,
                sigma_lr=sigma_lr_val,
                use_dynamic_task_attenuation=True,
                gamma_reg = gamma_reg,
                lambda_max = lambda_max
            )
            
            local_models[c] = local_model_c

            log_sigma_sq_local_val = local_model_c.log_sigma_sq_local.item()
            log_sigma_sq_align_val = local_model_c.log_sigma_sq_align.item()
            raw_lambda = math.exp(log_sigma_sq_local_val - log_sigma_sq_align_val)
            
            end_epoch_of_this_round = (cr + 1) * config['server']['local_epochs']
            total_training_epochs = config['server']['num_rounds'] * config['server']['local_epochs']
            global_progress = end_epoch_of_this_round / total_training_epochs
            s_p_value = max(0.0, 1.0 - global_progress)
            truly_effective_lambda = raw_lambda * s_p_value

            log_message = (
                f"Client {c} Post-Training State -> "
                f"Raw λ: {raw_lambda:9.4f} "
                f"| s(p): {s_p_value:.3f} "
                f"| Truly Effective λ (for W): {truly_effective_lambda:9.4f}"
            )
            logger.info(log_message)

        logger.info(f"Round {cr} Finish--------|")
        
        # FedAvg Aggregation Logic
        method_name = 'AURORAFedAvg'
        aggregated_model = parameter_averaging(local_models, weights)
        acc = test_acc_our_model(aggregated_model, test_loader, device)
        logger.info(f"The test accuracy of {method_name}: {acc}")
        method_results[method_name].append(acc)
        
        # Broadcast
        local_models = [copy.deepcopy(aggregated_model) for _ in range(config['client']['num_clients'])]

        save_yaml_config(save_path + "/baselines_" + method_name +"_" + config['checkpoint']['result_file'], method_results)


def OneshotOursV15(trainset, test_loader, client_idx_map, config, device, gamma_reg, lambda_max=50.0, **kwargs):
    logger.info('OneshotOursV15: AURORA (V14) with Projector Head (Addressing d >= C-1)')
    
    v10_cfg = config.get('v10_config', {})
    
    # CHANGE 1: Request the model with 'our_projector' mode to get LearnableProtoResNetWithProjector
    use_pretrain_bool = config.get('DBCD', {}).get('use_pretrain', False)
    custom_pretrain_path = config.get('pretrain', {}).get('model_path', '')
    dataset_name_lower = config['dataset']['data_name'].lower()
    model_name = config['server']['model_name']
    expected_custom_path = os.path.join(custom_pretrain_path, f"{dataset_name_lower}_{model_name}_centralized.pth")
    
    use_pretrain_arg = False
    if use_pretrain_bool:
        if os.path.exists(expected_custom_path):
            use_pretrain_arg = expected_custom_path
            logger.info(f"[OneshotOursV15] Found custom centralized weights at {expected_custom_path}. Using them!")
        else:
            use_pretrain_arg = False # Changed from True
            logger.warning(f"[OneshotOursV15] Custom weights NOT found at {expected_custom_path}. Using Random Initialization (ImageNet Disabled).")

    in_channel = config['dataset'].get('channels', 3)
    global_model = get_train_models(
        model_name=config['server']['model_name'],
        num_classes=config['dataset']['num_classes'],
        mode='our',
        use_pretrain=use_pretrain_arg,
        in_channel=in_channel
    )
    
    feature_dim = global_model.learnable_proto.shape[1]
    num_classes = config['dataset']['num_classes']
    etf_anchors = generate_etf_anchors(num_classes, feature_dim, device)
    
    logger.info(f"Initialized V15 with Projector. Feature Dim for ETF Alignment: {feature_dim}")
    
    method_results = defaultdict(list)
    save_path, local_model_dir = prepare_checkpoint_dir(config)
    if not os.path.exists(save_path + "/config.yaml"):
        save_yaml_config(save_path + "/config.yaml", config)
    local_models = [copy.deepcopy(global_model) for _ in range(config['client']['num_clients'])]
    local_data_size = [len(client_idx_map[c]) for c in range(config['client']['num_clients'])]
    if config['server']['aggregated_by_datasize']:
        weights = [i/sum(local_data_size) for i in local_data_size]
    else:
        weights = [1/config['client']['num_clients'] for _ in range(config['client']['num_clients'])]        
    aug_transformer = get_supcon_transform(config['dataset']['data_name'])
    clients_sample_per_class = []
    
    logger.info("--- Pre-heating V15 models with V9's adaptive lambda strategy ---")
    for c in range(config['client']['num_clients']):
        client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])
        v9_cfg = config.get('v9_config', {})
        initial_lambda = calculate_adaptive_lambda(
            client_dataloader,
            config['dataset']['num_classes'],
            v9_cfg.get('lambda_min', 0.1),
            v9_cfg.get('lambda_max', 15.0),
            device
        )
        logger.info(f"Client {c}: Calculated initial lambda = {initial_lambda:.4f}")
        initial_log_lambda = torch.tensor(initial_lambda, device=device).log()
        local_models[c].log_sigma_sq_local = torch.nn.Parameter(initial_log_lambda)
        local_models[c].log_sigma_sq_align = torch.nn.Parameter(torch.tensor(0.0, device=device))
    logger.info("--- V15 model pre-heating complete. ---")

    logger.info(f"Gamma for stability anchor regularization: {gamma_reg}")
    sigma_lr_val = v10_cfg.get('sigma_lr', 0.005)
    total_rounds = config['server']['num_rounds']

    for cr in trange(total_rounds):
        logger.info(f"Round {cr} starts--------|")
        local_protos = []
        
        for c in range(config['client']['num_clients']):
            logger.info(f"Client {c} Starts Local Trainning--------|")
            client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])

            if cr == 0:
                clients_sample_per_class.append(generate_sample_per_class(config['dataset']['num_classes'], client_dataloader, len(client_idx_map[c])))

            local_model_c = ours_local_training(
                model=copy.deepcopy(local_models[c]),
                training_data=client_dataloader,
                test_dataloader=test_loader,
                start_epoch=cr * config['server']['local_epochs'],
                local_epochs=config['server']['local_epochs'],
                optim_name=config['server']['optimizer'],
                lr=config['server']['lr'],
                momentum=config['server']['momentum'],
                loss_name=config['server']['loss_name'],
                device=device,
                num_classes=config['dataset']['num_classes'],
                sample_per_class=clients_sample_per_class[c],
                aug_transformer=aug_transformer,
                client_model_dir=local_model_dir + f"/client_{c}",
                total_rounds=total_rounds,
                use_drcl=True,
                fixed_anchors=etf_anchors,
                use_uncertainty_weighting=True,
                sigma_lr=sigma_lr_val,
                use_dynamic_task_attenuation=True,
                gamma_reg = gamma_reg,
                lambda_max = lambda_max
            )
            
            local_models[c] = local_model_c
            local_protos.append(local_model_c.get_proto().detach())

            log_sigma_sq_local_val = local_model_c.log_sigma_sq_local.item()
            log_sigma_sq_align_val = local_model_c.log_sigma_sq_align.item()
            raw_lambda = math.exp(log_sigma_sq_local_val - log_sigma_sq_align_val)
            
            end_epoch_of_this_round = (cr + 1) * config['server']['local_epochs']
            total_training_epochs = config['server']['num_rounds'] * config['server']['local_epochs']
            global_progress = end_epoch_of_this_round / total_training_epochs
            s_p_value = max(0.0, 1.0 - global_progress)
            truly_effective_lambda = raw_lambda * s_p_value

            log_message = (
                f"Client {c} Post-Training State -> "
                f"Raw λ: {raw_lambda:9.4f} "
                f"| s(p): {s_p_value:.3f} "
                f"| Truly Effective λ (for W): {truly_effective_lambda:9.4f}"
            )
            logger.info(log_message)

        logger.info(f"Round {cr} Finish--------|")
        
        
        # --- IFFI / Ensemble Aggregation Logic (Matches AURORA/V14) ---
        method_name = 'OursV15+SimpleFeatureServer'
        ensemble_model = WEnsembleFeature(model_list=local_models, weight_list=weights)
        # Note: global_proto is aggregated from local prototypes
        global_proto = aggregate_local_protos(local_protos)
        ens_acc = eval_with_proto(ensemble_model, test_loader, device, global_proto)
        logger.info(f"The test accuracy of {method_name}: {ens_acc}")
        method_results[method_name].append(ens_acc)
        save_yaml_config(save_path + "/baselines_" + method_name +"_" + config['checkpoint']['result_file'], method_results)

        # FedAvg Aggregation Logic
        method_name = 'OursV15FedAvg'
        aggregated_model = parameter_averaging(local_models, weights)
        acc = test_acc_our_model(aggregated_model, test_loader, device)
        logger.info(f"The test accuracy of {method_name}: {acc}")
        method_results[method_name].append(acc)
        
        local_models = [copy.deepcopy(aggregated_model) for _ in range(config['client']['num_clients'])]

        save_yaml_config(save_path + "/baselines_" + method_name +"_" + config['checkpoint']['result_file'], method_results)


def OneshotOursV16(trainset, test_loader, client_idx_map, config, device, gamma_reg, lambda_max=50.0, **kwargs):
    logger.info('OneshotOursV16: AURORA with Reclassified Losses (Task vs Regularization)')
    
    v10_cfg = config.get('v10_config', {})
    
    use_pretrain_bool = config.get('DBCD', {}).get('use_pretrain', False)
    use_imagenet_pretrain = config.get('DBCD', {}).get('use_imagenet_pretrain', False)
    custom_pretrain_path = config.get('pretrain', {}).get('model_path', '')
    dataset_name_lower = config['dataset']['data_name'].lower()
    model_name = config['server']['model_name']
    expected_custom_path = os.path.join(custom_pretrain_path, f"{dataset_name_lower}_{model_name}_centralized.pth")
    
    use_pretrain_arg = False
    if use_pretrain_bool:
        if os.path.exists(expected_custom_path):
            use_pretrain_arg = expected_custom_path
            logger.info(f"[OneshotOursV16] Found custom centralized weights at {expected_custom_path}. Using them!")
        else:
            use_pretrain_arg = False
            logger.warning(f"[OneshotOursV16] Custom weights NOT found at {expected_custom_path}. Using Random Initialization.")
    elif use_imagenet_pretrain:
        use_pretrain_arg = True
        logger.info("[OneshotOursV16] Loading ImageNet pretrained weights for ResNet-18.")

    global_model = get_train_models(
        model_name=config['server']['model_name'],
        num_classes=config['dataset']['num_classes'],
        mode='our',
        use_pretrain=use_pretrain_arg,
        in_channel=config['dataset'].get('channels', 3)
    )
    feature_dim = global_model.learnable_proto.shape[1]
    num_classes = config['dataset']['num_classes']
    etf_anchors = generate_etf_anchors(num_classes, feature_dim, device)
    method_results = defaultdict(list)
    save_path, local_model_dir = prepare_checkpoint_dir(config)
    if not os.path.exists(save_path + "/config.yaml"):
        save_yaml_config(save_path + "/config.yaml", config)
    local_models = [copy.deepcopy(global_model) for _ in range(config['client']['num_clients'])]
    local_data_size = [len(client_idx_map[c]) for c in range(config['client']['num_clients'])]
    if config['server']['aggregated_by_datasize']:
        weights = [i/sum(local_data_size) for i in local_data_size]
    else:
        weights = [1/config['client']['num_clients'] for _ in range(config['client']['num_clients'])]        
    aug_transformer = get_supcon_transform(config['dataset']['data_name'])
    clients_sample_per_class = []
    
    logger.info("--- Pre-heating V16 models with loss-ratio calibration ---")
    
    for c in range(config['client']['num_clients']):
        client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])
        
        model_copy = copy.deepcopy(local_models[c]).to(device)
        model_copy.eval()
        with torch.no_grad():
            batch = next(iter(client_dataloader))
            data, target = batch[0].to(device), batch[1].to(device)
            bsz = data.shape[0]
            
            logits, feature_norm = model_copy(data)
            cls_val = torch.nn.functional.cross_entropy(logits, target).item()
            
            # Calibration uses a single forward pass (one view), so features match single-batch labels.
            # SupConLoss needs paired views — skip it in calibration; proto losses dominate reg_val anyway.
            unique_classes = torch.unique(target)
            pf_val = Contrastive_proto_feature_loss(temperature=1.0)(
                feature_norm, model_copy.learnable_proto, target, active_indices=unique_classes
            ).item()
            pc_val = Contrastive_proto_loss(temperature=1.0)(
                model_copy.learnable_proto, active_indices=unique_classes
            ).item()
            
            reg_val = pf_val + pc_val  # SupCon skipped (needs 2 views)
            
            if reg_val > 1e-6:
                initial_lambda = min(max(cls_val / reg_val, 0.01), 2.0)
            else:
                initial_lambda = 0.5
        
        del model_copy
        
        logger.info(f"Client {c}: cls={cls_val:.4f}, reg={reg_val:.4f}, initial_lambda={initial_lambda:.4f}")
        local_models[c] = local_models[c].to(device)
        initial_log_lambda = torch.tensor(initial_lambda, device=device).log()
        local_models[c].log_sigma_sq_local = torch.nn.Parameter(initial_log_lambda)
        local_models[c].log_sigma_sq_align = torch.nn.Parameter(torch.tensor(0.0, device=device))
    logger.info("--- V16 model pre-heating complete. ---")

    logger.info(f"Gamma for stability anchor regularization: {gamma_reg}")

    sigma_lr_val = v10_cfg.get('sigma_lr', 0.005)
    total_rounds = config['server']['num_rounds']

    for cr in trange(total_rounds):
        logger.info(f"Round {cr} starts--------|")
        local_protos = []
        
        for c in range(config['client']['num_clients']):
            logger.info(f"Client {c} Starts Local Trainning--------|")
            client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])

            if cr == 0:
                clients_sample_per_class.append(generate_sample_per_class(config['dataset']['num_classes'], client_dataloader, len(client_idx_map[c])))

            local_model_c = ours_local_training(
                model=copy.deepcopy(local_models[c]),
                training_data=client_dataloader,
                test_dataloader=test_loader,
                start_epoch=cr * config['server']['local_epochs'],
                local_epochs=config['server']['local_epochs'],
                optim_name=config['server']['optimizer'],
                lr=config['server']['lr'],
                momentum=config['server']['momentum'],
                loss_name=config['server']['loss_name'],
                device=device,
                num_classes=config['dataset']['num_classes'],
                sample_per_class=clients_sample_per_class[c],
                aug_transformer=aug_transformer,
                client_model_dir=local_model_dir + f"/client_{c}",
                total_rounds=total_rounds,
                use_drcl=True,
                fixed_anchors=etf_anchors,
                use_uncertainty_weighting=True,
                sigma_lr=sigma_lr_val,
                use_dynamic_task_attenuation=True,
                gamma_reg=gamma_reg,
                lambda_max=lambda_max,
                use_reclassified_losses=True
            )
            
            local_models[c] = local_model_c
            local_protos.append(local_model_c.get_proto().detach())

            log_sigma_sq_local_val = local_model_c.log_sigma_sq_local.item()
            log_sigma_sq_align_val = local_model_c.log_sigma_sq_align.item()
            raw_lambda = math.exp(log_sigma_sq_local_val - log_sigma_sq_align_val)
            
            end_epoch_of_this_round = (cr + 1) * config['server']['local_epochs']
            total_training_epochs = config['server']['num_rounds'] * config['server']['local_epochs']
            global_progress = end_epoch_of_this_round / total_training_epochs
            s_p_value = max(0.0, 1.0 - global_progress)
            truly_effective_lambda = raw_lambda * s_p_value

            logger.info(
                f"Client {c} Post-Training State -> "
                f"Raw λ: {raw_lambda:9.4f} "
                f"| s(p): {s_p_value:.3f} "
                f"| Truly Effective λ (for W): {truly_effective_lambda:9.4f}"
            )

        logger.info(f"Round {cr} Finish--------|")
        
        method_name = 'OursV16+SimpleFeatureServer'
        ensemble_model = WEnsembleFeature(model_list=local_models, weight_list=weights)
        global_proto = aggregate_local_protos(local_protos)
        ens_acc = eval_with_proto(ensemble_model, test_loader, device, global_proto)
        logger.info(f"The test accuracy of {method_name}: {ens_acc}")
        method_results[method_name].append(ens_acc)
        save_yaml_config(save_path + "/baselines_" + method_name +"_" + config['checkpoint']['result_file'], method_results)


def OneshotOursV17(trainset, test_loader, client_idx_map, config, device, gamma_reg, lambda_max=50.0, **kwargs):
    logger.info('OneshotOursV17: AURORA with Dual-Head (Linear Classification + Proto Alignment)')
    
    v10_cfg = config.get('v10_config', {})
    
    use_pretrain_bool = config.get('DBCD', {}).get('use_pretrain', False)
    use_imagenet_pretrain = config.get('DBCD', {}).get('use_imagenet_pretrain', False)
    custom_pretrain_path = config.get('pretrain', {}).get('model_path', '')
    dataset_name_lower = config['dataset']['data_name'].lower()
    model_name = config['server']['model_name']
    expected_custom_path = os.path.join(custom_pretrain_path, f"{dataset_name_lower}_{model_name}_centralized.pth")
    
    use_pretrain_arg = False
    if use_pretrain_bool:
        if os.path.exists(expected_custom_path):
            use_pretrain_arg = expected_custom_path
            logger.info(f"[OneshotOursV17] Found custom centralized weights at {expected_custom_path}. Using them!")
        else:
            use_pretrain_arg = False
            logger.warning(f"[OneshotOursV17] Custom weights NOT found at {expected_custom_path}. Using Random Initialization.")
    elif use_imagenet_pretrain:
        use_pretrain_arg = True
        logger.info("[OneshotOursV17] Loading ImageNet pretrained weights for ResNet-18.")

    global_model = get_train_models(
        model_name=config['server']['model_name'],
        num_classes=config['dataset']['num_classes'],
        mode='our_dual',
        use_pretrain=use_pretrain_arg,
        in_channel=config['dataset'].get('channels', 3)
    )
    feature_dim = global_model.learnable_proto.shape[1]
    num_classes = config['dataset']['num_classes']
    etf_anchors = generate_etf_anchors(num_classes, feature_dim, device)
    method_results = defaultdict(list)
    save_path, local_model_dir = prepare_checkpoint_dir(config)
    if not os.path.exists(save_path + "/config.yaml"):
        save_yaml_config(save_path + "/config.yaml", config)
    local_models = [copy.deepcopy(global_model) for _ in range(config['client']['num_clients'])]
    local_data_size = [len(client_idx_map[c]) for c in range(config['client']['num_clients'])]
    if config['server']['aggregated_by_datasize']:
        weights = [i/sum(local_data_size) for i in local_data_size]
    else:
        weights = [1/config['client']['num_clients'] for _ in range(config['client']['num_clients'])]        
    aug_transformer = get_gpu_augmentation(config['dataset']['data_name'], device)
    clients_sample_per_class = []
    
    logger.info("--- Pre-heating V17 models with loss-ratio calibration ---")
    
    for c in range(config['client']['num_clients']):
        client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])
        
        model_copy = copy.deepcopy(local_models[c]).to(device)
        model_copy.eval()
        with torch.no_grad():
            batch = next(iter(client_dataloader))
            data, target = batch[0].to(device), batch[1].to(device)
            
            logits, feature_norm = model_copy(data)
            cls_val = F.cross_entropy(logits, target).item()
            
            unique_classes = torch.unique(target)
            pf_val = Contrastive_proto_feature_loss(temperature=1.0)(
                feature_norm, model_copy.learnable_proto, target, active_indices=unique_classes
            ).item()
            pc_val = Contrastive_proto_loss(temperature=1.0)(
                model_copy.learnable_proto, active_indices=unique_classes
            ).item()
            
            reg_val = pf_val + pc_val
            
            if reg_val > 1e-6:
                initial_lambda = min(max(cls_val / reg_val, 0.01), 2.0)
            else:
                initial_lambda = 0.5
        
        del model_copy
        
        logger.info(f"Client {c}: cls={cls_val:.4f}, reg={reg_val:.4f}, initial_lambda={initial_lambda:.4f}")
        local_models[c] = local_models[c].to(device)
        initial_log_lambda = torch.tensor(initial_lambda, device=device).log()
        local_models[c].log_sigma_sq_local = torch.nn.Parameter(initial_log_lambda)
        local_models[c].log_sigma_sq_align = torch.nn.Parameter(torch.tensor(0.0, device=device))
    logger.info("--- V17 model pre-heating complete. ---")

    logger.info(f"Gamma for stability anchor regularization: {gamma_reg}")

    sigma_lr_val = v10_cfg.get('sigma_lr', 0.005)
    total_rounds = config['server']['num_rounds']

    for cr in trange(total_rounds):
        logger.info(f"Round {cr} starts--------|")
        local_protos = []
        
        for c in range(config['client']['num_clients']):
            logger.info(f"Client {c} Starts Local Trainning--------|")
            client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])

            if cr == 0:
                clients_sample_per_class.append(generate_sample_per_class(config['dataset']['num_classes'], client_dataloader, len(client_idx_map[c])))

            local_model_c = ours_local_training(
                model=copy.deepcopy(local_models[c]),
                training_data=client_dataloader,
                test_dataloader=test_loader,
                start_epoch=cr * config['server']['local_epochs'],
                local_epochs=config['server']['local_epochs'],
                optim_name=config['server']['optimizer'],
                lr=config['server']['lr'],
                momentum=config['server']['momentum'],
                loss_name=config['server']['loss_name'],
                device=device,
                num_classes=config['dataset']['num_classes'],
                sample_per_class=clients_sample_per_class[c],
                aug_transformer=aug_transformer,
                client_model_dir=local_model_dir + f"/client_{c}",
                total_rounds=total_rounds,
                use_drcl=True,
                fixed_anchors=etf_anchors,
                use_uncertainty_weighting=True,
                sigma_lr=sigma_lr_val,
                use_dynamic_task_attenuation=True,
                gamma_reg=gamma_reg,
                lambda_max=lambda_max,
                use_reclassified_losses=True
            )
            
            local_models[c] = local_model_c
            local_protos.append(local_model_c.get_proto().detach())

            log_sigma_sq_local_val = local_model_c.log_sigma_sq_local.item()
            log_sigma_sq_align_val = local_model_c.log_sigma_sq_align.item()
            raw_lambda = math.exp(log_sigma_sq_local_val - log_sigma_sq_align_val)
            
            end_epoch_of_this_round = (cr + 1) * config['server']['local_epochs']
            total_training_epochs = config['server']['num_rounds'] * config['server']['local_epochs']
            global_progress = end_epoch_of_this_round / total_training_epochs
            s_p_value = max(0.0, 1.0 - global_progress)
            truly_effective_lambda = raw_lambda * s_p_value

            logger.info(
                f"Client {c} Post-Training State -> "
                f"Raw λ: {raw_lambda:9.4f} "
                f"| s(p): {s_p_value:.3f} "
                f"| Truly Effective λ (for W): {truly_effective_lambda:9.4f}"
            )

        logger.info(f"Round {cr} Finish--------|")

        g_protos = torch.stack(local_protos)
        g_protos_std = g_protos.std(dim=0).norm().item()
        inter_proto_std = g_protos_std / g_protos.norm(dim=2).mean().item()
        logger.info(f"g_protos_std (global internal): {g_protos_std}")
        logger.info(f"inter_client_proto_std (cross-client): {inter_proto_std}")

        method_name = 'OursV17+ProtoInference'
        ensemble_model = WEnsembleFeature(model_list=local_models, weight_list=weights)
        global_proto = aggregate_local_protos(local_protos)
        ens_acc = eval_with_proto(ensemble_model, test_loader, device, global_proto)
        logger.info(f"The test accuracy of {method_name}: {ens_acc}")
        method_results[method_name].append(ens_acc)
        save_yaml_config(save_path + "/baselines_" + method_name +"_" + config['checkpoint']['result_file'], method_results)



def OneshotOursV18(trainset, test_loader, client_idx_map, config, device, gamma_reg, lambda_max=50.0, **kwargs):
    logger.info('OneshotOursV18: AURORA with Reclassified Losses + No Annealing')
    
    v10_cfg = config.get('v10_config', {})
    
    use_pretrain_bool = config.get('DBCD', {}).get('use_pretrain', False)
    use_imagenet_pretrain = config.get('DBCD', {}).get('use_imagenet_pretrain', False)
    custom_pretrain_path = config.get('pretrain', {}).get('model_path', '')
    dataset_name_lower = config['dataset']['data_name'].lower()
    model_name = config['server']['model_name']
    expected_custom_path = os.path.join(custom_pretrain_path, f"{dataset_name_lower}_{model_name}_centralized.pth")
    
    use_pretrain_arg = False
    if use_pretrain_bool:
        if os.path.exists(expected_custom_path):
            use_pretrain_arg = expected_custom_path
            logger.info(f"[OneshotOursV18] Found custom centralized weights at {expected_custom_path}. Using them!")
        else:
            use_pretrain_arg = False
            logger.warning(f"[OneshotOursV18] Custom weights NOT found at {expected_custom_path}. Using Random Initialization.")
    elif use_imagenet_pretrain:
        use_pretrain_arg = True
        logger.info("[OneshotOursV18] Loading ImageNet pretrained weights for ResNet-18.")

    global_model = get_train_models(
        model_name=config['server']['model_name'],
        num_classes=config['dataset']['num_classes'],
        mode='our',
        use_pretrain=use_pretrain_arg,
        in_channel=config['dataset'].get('channels', 3)
    )
    feature_dim = global_model.learnable_proto.shape[1]
    num_classes = config['dataset']['num_classes']
    etf_anchors = generate_etf_anchors(num_classes, feature_dim, device)
    method_results = defaultdict(list)
    save_path, local_model_dir = prepare_checkpoint_dir(config)
    if not os.path.exists(save_path + "/config.yaml"):
        save_yaml_config(save_path + "/config.yaml", config)
    local_models = [copy.deepcopy(global_model) for _ in range(config['client']['num_clients'])]
    local_data_size = [len(client_idx_map[c]) for c in range(config['client']['num_clients'])]
    if config['server']['aggregated_by_datasize']:
        weights = [i/sum(local_data_size) for i in local_data_size]
    else:
        weights = [1/config['client']['num_clients'] for _ in range(config['client']['num_clients'])]        
    aug_transformer = get_supcon_transform(config['dataset']['data_name'])
    clients_sample_per_class = []
    
    logger.info("--- Pre-heating V18 models with loss-ratio calibration ---")
    
    for c in range(config['client']['num_clients']):
        client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])
        
        model_copy = copy.deepcopy(local_models[c]).to(device)
        model_copy.eval()
        with torch.no_grad():
            batch = next(iter(client_dataloader))
            data, target = batch[0].to(device), batch[1].to(device)
            
            logits, feature_norm = model_copy(data)
            cls_val = torch.nn.functional.cross_entropy(logits, target).item()
            
            unique_classes = torch.unique(target)
            pf_val = Contrastive_proto_feature_loss(temperature=1.0)(
                feature_norm, model_copy.learnable_proto, target, active_indices=unique_classes
            ).item()
            pc_val = Contrastive_proto_loss(temperature=1.0)(
                model_copy.learnable_proto, active_indices=unique_classes
            ).item()
            
            reg_val = pf_val + pc_val
            
            if reg_val > 1e-6:
                initial_lambda = min(max(cls_val / reg_val, 0.01), 2.0)
            else:
                initial_lambda = 0.5
        
        del model_copy
        
        logger.info(f"Client {c}: cls={cls_val:.4f}, reg={reg_val:.4f}, initial_lambda={initial_lambda:.4f}")
        local_models[c] = local_models[c].to(device)
        initial_log_lambda = torch.tensor(initial_lambda, device=device).log()
        local_models[c].log_sigma_sq_local = torch.nn.Parameter(initial_log_lambda)
        local_models[c].log_sigma_sq_align = torch.nn.Parameter(torch.tensor(0.0, device=device))
    logger.info("--- V18 model pre-heating complete. ---")

    logger.info(f"Gamma for stability anchor regularization: {gamma_reg}")

    sigma_lr_val = v10_cfg.get('sigma_lr', 0.005)
    total_rounds = config['server']['num_rounds']
    local_epochs = config['server']['local_epochs']
    warmup_ep = local_epochs // 2  # First half of each round is warmup (pure CE, no augmentation)

    for cr in trange(total_rounds):
        logger.info(f"Round {cr} starts--------|")
        local_protos = []
        
        for c in range(config['client']['num_clients']):
            logger.info(f"Client {c} Starts Local Trainning--------|")
            client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])

            if cr == 0:
                clients_sample_per_class.append(generate_sample_per_class(config['dataset']['num_classes'], client_dataloader, len(client_idx_map[c])))

            start_ep = cr * local_epochs
            local_model_c = ours_local_training(
                model=copy.deepcopy(local_models[c]),
                training_data=client_dataloader,
                test_dataloader=test_loader,
                start_epoch=start_ep,
                local_epochs=local_epochs,
                optim_name=config['server']['optimizer'],
                lr=config['server']['lr'],
                momentum=config['server']['momentum'],
                loss_name=config['server']['loss_name'],
                device=device,
                num_classes=config['dataset']['num_classes'],
                sample_per_class=clients_sample_per_class[c],
                aug_transformer=aug_transformer,
                client_model_dir=local_model_dir + f"/client_{c}",
                total_rounds=total_rounds,
                use_drcl=True,
                fixed_anchors=etf_anchors,
                use_uncertainty_weighting=True,
                sigma_lr=sigma_lr_val,
                use_dynamic_task_attenuation=False,
                gamma_reg=gamma_reg,
                lambda_max=lambda_max,
                use_reclassified_losses=True,
                warmup_epochs=warmup_ep
            )
            
            local_models[c] = local_model_c
            local_protos.append(local_model_c.get_proto().detach())

            log_sigma_sq_local_val = local_model_c.log_sigma_sq_local.item()
            log_sigma_sq_align_val = local_model_c.log_sigma_sq_align.item()
            raw_lambda = math.exp(log_sigma_sq_local_val - log_sigma_sq_align_val)

            logger.info(
                f"Client {c} Post-Training State -> "
                f"Raw λ: {raw_lambda:9.4f}"
            )

        logger.info(f"Round {cr} Finish--------|")

        method_name = 'OursV18+WarmupNoAnneal'
        ensemble_model = WEnsembleFeature(model_list=local_models, weight_list=weights)
        global_proto = aggregate_local_protos(local_protos)
        ens_acc = eval_with_proto(ensemble_model, test_loader, device, global_proto)
        logger.info(f"The test accuracy of {method_name}: {ens_acc}")
        method_results[method_name].append(ens_acc)
        save_yaml_config(save_path + "/baselines_" + method_name +"_" + config['checkpoint']['result_file'], method_results)


def OneshotOursV19(trainset, test_loader, client_idx_map, config, device, gamma_reg, lambda_max=50.0, **kwargs):
    logger.info('OneshotOursV19: AURORA with Decaying Warmup')
    
    v10_cfg = config.get('v10_config', {})
    
    use_pretrain_bool = config.get('DBCD', {}).get('use_pretrain', False)
    use_imagenet_pretrain = config.get('DBCD', {}).get('use_imagenet_pretrain', False)
    custom_pretrain_path = config.get('pretrain', {}).get('model_path', '')
    dataset_name_lower = config['dataset']['data_name'].lower()
    model_name = config['server']['model_name']
    expected_custom_path = os.path.join(custom_pretrain_path, f"{dataset_name_lower}_{model_name}_centralized.pth")
    
    use_pretrain_arg = False
    if use_pretrain_bool:
        if os.path.exists(expected_custom_path):
            use_pretrain_arg = expected_custom_path
            logger.info(f"[OneshotOursV19] Found custom centralized weights at {expected_custom_path}. Using them!")
        else:
            use_pretrain_arg = False
            logger.warning(f"[OneshotOursV19] Custom weights NOT found at {expected_custom_path}. Using Random Initialization.")
    elif use_imagenet_pretrain:
        use_pretrain_arg = True
        logger.info("[OneshotOursV19] Loading ImageNet pretrained weights for ResNet-18.")

    global_model = get_train_models(
        model_name=config['server']['model_name'],
        num_classes=config['dataset']['num_classes'],
        mode='our',
        use_pretrain=use_pretrain_arg,
        in_channel=config['dataset'].get('channels', 3)
    )
    feature_dim = global_model.learnable_proto.shape[1]
    num_classes = config['dataset']['num_classes']
    etf_anchors = generate_etf_anchors(num_classes, feature_dim, device)
    method_results = defaultdict(list)
    save_path, local_model_dir = prepare_checkpoint_dir(config)
    if not os.path.exists(save_path + "/config.yaml"):
        save_yaml_config(save_path + "/config.yaml", config)
    local_models = [copy.deepcopy(global_model) for _ in range(config['client']['num_clients'])]
    local_data_size = [len(client_idx_map[c]) for c in range(config['client']['num_clients'])]
    if config['server']['aggregated_by_datasize']:
        weights = [i/sum(local_data_size) for i in local_data_size]
    else:
        weights = [1/config['client']['num_clients'] for _ in range(config['client']['num_clients'])]        
    aug_transformer = get_supcon_transform(config['dataset']['data_name'])
    clients_sample_per_class = []
    
    logger.info("--- Pre-heating V19 models with loss-ratio calibration ---")
    
    for c in range(config['client']['num_clients']):
        client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])
        
        model_copy = copy.deepcopy(local_models[c]).to(device)
        model_copy.eval()
        with torch.no_grad():
            batch = next(iter(client_dataloader))
            data, target = batch[0].to(device), batch[1].to(device)
            
            logits, feature_norm = model_copy(data)
            cls_val = torch.nn.functional.cross_entropy(logits, target).item()
            
            unique_classes = torch.unique(target)
            pf_val = Contrastive_proto_feature_loss(temperature=1.0)(
                feature_norm, model_copy.learnable_proto, target, active_indices=unique_classes
            ).item()
            pc_val = Contrastive_proto_loss(temperature=1.0)(
                model_copy.learnable_proto, active_indices=unique_classes
            ).item()
            
            reg_val = pf_val + pc_val
            
            if reg_val > 1e-6:
                initial_lambda = min(max(cls_val / reg_val, 0.01), 2.0)
            else:
                initial_lambda = 0.5
        
        del model_copy
        
        logger.info(f"Client {c}: cls={cls_val:.4f}, reg={reg_val:.4f}, initial_lambda={initial_lambda:.4f}")
        local_models[c] = local_models[c].to(device)
        initial_log_lambda = torch.tensor(initial_lambda, device=device).log()
        local_models[c].log_sigma_sq_local = torch.nn.Parameter(initial_log_lambda)
        local_models[c].log_sigma_sq_align = torch.nn.Parameter(torch.tensor(0.0, device=device))
    logger.info("--- V19 model pre-heating complete. ---")

    logger.info(f"Gamma for stability anchor regularization: {gamma_reg}")

    sigma_lr_val = v10_cfg.get('sigma_lr', 0.005)
    total_rounds = config['server']['num_rounds']
    local_epochs = config['server']['local_epochs']
    base_warmup = local_epochs // 2

    for cr in trange(total_rounds):
        # Decaying warmup: full warmup at round 0, linearly decay to 0 at last round
        warmup_ep = max(0, int(base_warmup * (1 - cr / max(1, total_rounds - 1))))
        logger.info(f"Round {cr} starts--------| warmup_epochs={warmup_ep}")
        local_protos = []
        
        for c in range(config['client']['num_clients']):
            logger.info(f"Client {c} Starts Local Trainning--------|")
            client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])

            if cr == 0:
                clients_sample_per_class.append(generate_sample_per_class(config['dataset']['num_classes'], client_dataloader, len(client_idx_map[c])))

            start_ep = cr * local_epochs
            local_model_c = ours_local_training(
                model=copy.deepcopy(local_models[c]),
                training_data=client_dataloader,
                test_dataloader=test_loader,
                start_epoch=start_ep,
                local_epochs=local_epochs,
                optim_name=config['server']['optimizer'],
                lr=config['server']['lr'],
                momentum=config['server']['momentum'],
                loss_name=config['server']['loss_name'],
                device=device,
                num_classes=config['dataset']['num_classes'],
                sample_per_class=clients_sample_per_class[c],
                aug_transformer=aug_transformer,
                client_model_dir=local_model_dir + f"/client_{c}",
                total_rounds=total_rounds,
                use_drcl=True,
                fixed_anchors=etf_anchors,
                use_uncertainty_weighting=True,
                sigma_lr=sigma_lr_val,
                use_dynamic_task_attenuation=False,
                gamma_reg=gamma_reg,
                lambda_max=lambda_max,
                use_reclassified_losses=True,
                warmup_epochs=warmup_ep,
                use_confidence_gating=False
            )
            
            local_models[c] = local_model_c
            local_protos.append(local_model_c.get_proto().detach())

            log_sigma_sq_local_val = local_model_c.log_sigma_sq_local.item()
            log_sigma_sq_align_val = local_model_c.log_sigma_sq_align.item()
            raw_lambda = math.exp(log_sigma_sq_local_val - log_sigma_sq_align_val)

            logger.info(
                f"Client {c} Post-Training State -> "
                f"Raw λ: {raw_lambda:9.4f}"
            )

        logger.info(f"Round {cr} Finish--------|")

        method_name = 'OursV19+DecayingWarmup'
        ensemble_model = WEnsembleFeature(model_list=local_models, weight_list=weights)
        global_proto = aggregate_local_protos(local_protos)
        ens_acc = eval_with_proto(ensemble_model, test_loader, device, global_proto)
        logger.info(f"The test accuracy of {method_name}: {ens_acc}")
        method_results[method_name].append(ens_acc)
        save_yaml_config(save_path + "/baselines_" + method_name +"_" + config['checkpoint']['result_file'], method_results)


def OneshotOursV22(trainset, test_loader, client_idx_map, config, device, gamma_reg, lambda_max=50.0, **kwargs):
    """
    AURORA V22: Raw-Data CE + Consistency²-Gated AU.
    
    Principle: CE always trains on clean (raw) data. AU trains on augmented views
    but is multiplicatively gated by squared prediction consistency between views.
    
    - FEMNIST (destructive aug): low consistency → AU suppressed → CE dominates
    - CIFAR-100 (helpful aug): high consistency → AU active → CE + AU jointly
    - AU's uniformity provides inter-class repulsion (unlike V21 alignment-only)
    - No ETF anchors, no warmup, no uncertainty weighting
    """
    logger.info('OneshotOursV22: Raw-Data CE + Consistency²-Gated AU')
    
    use_pretrain_bool = config.get('DBCD', {}).get('use_pretrain', False)
    use_imagenet_pretrain = config.get('DBCD', {}).get('use_imagenet_pretrain', False)
    custom_pretrain_path = config.get('pretrain', {}).get('model_path', '')
    dataset_name_lower = config['dataset']['data_name'].lower()
    model_name = config['server']['model_name']
    expected_custom_path = os.path.join(custom_pretrain_path, f"{dataset_name_lower}_{model_name}_centralized.pth")
    
    use_pretrain_arg = False
    if use_pretrain_bool:
        if os.path.exists(expected_custom_path):
            use_pretrain_arg = expected_custom_path
            logger.info(f"[OneshotOursV22] Custom weights at {expected_custom_path}")
        else:
            logger.warning(f"[OneshotOursV22] Custom weights NOT found. Random init.")
    elif use_imagenet_pretrain:
        use_pretrain_arg = True
        logger.info("[OneshotOursV22] ImageNet pretrained weights.")

    global_model = get_train_models(
        model_name=config['server']['model_name'],
        num_classes=config['dataset']['num_classes'],
        mode='our',
        use_pretrain=use_pretrain_arg,
        in_channel=config['dataset'].get('channels', 3)
    )
    method_results = defaultdict(list)
    save_path, local_model_dir = prepare_checkpoint_dir(config)
    if not os.path.exists(save_path + "/config.yaml"):
        save_yaml_config(save_path + "/config.yaml", config)
    local_models = [copy.deepcopy(global_model) for _ in range(config['client']['num_clients'])]
    local_data_size = [len(client_idx_map[c]) for c in range(config['client']['num_clients'])]
    if config['server']['aggregated_by_datasize']:
        weights = [i/sum(local_data_size) for i in local_data_size]
    else:
        weights = [1/config['client']['num_clients'] for _ in range(config['client']['num_clients'])]
    aug_transformer = get_supcon_transform(config['dataset']['data_name'])
    clients_sample_per_class = []
    
    total_rounds = config['server']['num_rounds']
    local_epochs = config['server']['local_epochs']
    gate_power = config.get('v22_config', {}).get('gate_power', 1.0)
    logger.info(f'V22 config: gate_power={gate_power}')

    for cr in trange(total_rounds):
        logger.info(f"Round {cr} starts--------| [V22: RawCE + consistency^{gate_power}-gated AU]")
        local_protos = []
        
        for c in range(config['client']['num_clients']):
            logger.info(f"Client {c} Starts Local Training--------|")
            client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])

            if cr == 0:
                clients_sample_per_class.append(generate_sample_per_class(config['dataset']['num_classes'], client_dataloader, len(client_idx_map[c])))

            start_ep = cr * local_epochs
            local_model_c = ours_local_training(
                model=copy.deepcopy(local_models[c]),
                training_data=client_dataloader,
                test_dataloader=test_loader,
                start_epoch=start_ep,
                local_epochs=local_epochs,
                optim_name=config['server']['optimizer'],
                lr=config['server']['lr'],
                momentum=config['server']['momentum'],
                loss_name=config['server']['loss_name'],
                device=device,
                num_classes=config['dataset']['num_classes'],
                sample_per_class=clients_sample_per_class[c],
                aug_transformer=aug_transformer,
                client_model_dir=local_model_dir + f"/client_{c}",
                total_rounds=total_rounds,
                lambda_align=1.0,
                warmup_epochs=0,
                use_raw_ce_au=True,
                gate_power=gate_power
            )
            
            local_models[c] = local_model_c
            local_protos.append(local_model_c.get_proto().detach())

        logger.info(f"Round {cr} Finish--------|")

        method_name = 'OursV22_RawCE_AU'
        ensemble_model = WEnsembleFeature(model_list=local_models, weight_list=weights)
        global_proto = aggregate_local_protos(local_protos)
        ens_acc = eval_with_proto(ensemble_model, test_loader, device, global_proto)
        logger.info(f"The test accuracy of {method_name}: {ens_acc}")
        method_results[method_name].append(ens_acc)
        save_yaml_config(save_path + "/baselines_" + method_name +"_" + config['checkpoint']['result_file'], method_results)


def OneshotOursV23(trainset, test_loader, client_idx_map, config, device, gamma_reg, lambda_max=50.0, **kwargs):
    """
    AURORA V23: Raw-Data CE + Consistency²-Gated SupCon.
    
    Key insight: AU's uniformity term is unbounded negative (log of small numbers),
    making it incompatible with additive loss formulation when the gate opens.
    SupCon (softmax cross-entropy) is always positive, so total loss >= CE always.
    
    The consistency gate handles augmentation quality:
    - FEMNIST (destructive aug): low consistency → SupCon suppressed → CE dominates
    - CIFAR-100 (helpful aug): moderate consistency → SupCon active → CE + SupCon jointly
    
    lambda needs to be higher (~30) because gate² reduces effective weight:
    - CIFAR-100: effective_λ = 30 * 0.04 = 1.2 (meaningful)
    - FEMNIST:   effective_λ = 30 * 0.002 = 0.06 (negligible)
    """
    logger.info('OneshotOursV23: Raw-Data CE + Consistency²-Gated SupCon')
    
    use_pretrain_bool = config.get('DBCD', {}).get('use_pretrain', False)
    use_imagenet_pretrain = config.get('DBCD', {}).get('use_imagenet_pretrain', False)
    custom_pretrain_path = config.get('pretrain', {}).get('model_path', '')
    dataset_name_lower = config['dataset']['data_name'].lower()
    model_name = config['server']['model_name']
    expected_custom_path = os.path.join(custom_pretrain_path, f"{dataset_name_lower}_{model_name}_centralized.pth")
    
    use_pretrain_arg = False
    if use_pretrain_bool:
        if os.path.exists(expected_custom_path):
            use_pretrain_arg = expected_custom_path
            logger.info(f"[OneshotOursV23] Custom weights at {expected_custom_path}")
        else:
            logger.warning(f"[OneshotOursV23] Custom weights NOT found. Random init.")
    elif use_imagenet_pretrain:
        use_pretrain_arg = True
        logger.info("[OneshotOursV23] ImageNet pretrained weights.")

    global_model = get_train_models(
        model_name=config['server']['model_name'],
        num_classes=config['dataset']['num_classes'],
        mode='our',
        use_pretrain=use_pretrain_arg,
        in_channel=config['dataset'].get('channels', 3)
    )
    method_results = defaultdict(list)
    save_path, local_model_dir = prepare_checkpoint_dir(config)
    if not os.path.exists(save_path + "/config.yaml"):
        save_yaml_config(save_path + "/config.yaml", config)
    local_models = [copy.deepcopy(global_model) for _ in range(config['client']['num_clients'])]
    local_data_size = [len(client_idx_map[c]) for c in range(config['client']['num_clients'])]
    if config['server']['aggregated_by_datasize']:
        weights = [i/sum(local_data_size) for i in local_data_size]
    else:
        weights = [1/config['client']['num_clients'] for _ in range(config['client']['num_clients'])]
    aug_transformer = get_supcon_transform(config['dataset']['data_name'])
    clients_sample_per_class = []
    
    total_rounds = config['server']['num_rounds']
    local_epochs = config['server']['local_epochs']
    v23_cfg = config.get('v23_config', {})
    gate_power = v23_cfg.get('gate_power', 2.0)
    lambda_supcon = v23_cfg.get('lambda', 30.0)
    logger.info(f'V23 config: gate_power={gate_power}, lambda_supcon={lambda_supcon}')

    for cr in trange(total_rounds):
        logger.info(f"Round {cr} starts--------| [V23: RawCE + consistency^{gate_power}-gated SupCon]")
        local_protos = []
        
        for c in range(config['client']['num_clients']):
            logger.info(f"Client {c} Starts Local Training--------|")
            client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])

            if cr == 0:
                clients_sample_per_class.append(generate_sample_per_class(config['dataset']['num_classes'], client_dataloader, len(client_idx_map[c])))

            start_ep = cr * local_epochs
            local_model_c = ours_local_training(
                model=copy.deepcopy(local_models[c]),
                training_data=client_dataloader,
                test_dataloader=test_loader,
                start_epoch=start_ep,
                local_epochs=local_epochs,
                optim_name=config['server']['optimizer'],
                lr=config['server']['lr'],
                momentum=config['server']['momentum'],
                loss_name=config['server']['loss_name'],
                device=device,
                num_classes=config['dataset']['num_classes'],
                sample_per_class=clients_sample_per_class[c],
                aug_transformer=aug_transformer,
                client_model_dir=local_model_dir + f"/client_{c}",
                total_rounds=total_rounds,
                lambda_align=lambda_supcon,
                warmup_epochs=0,
                use_raw_ce_supcon=True,
                gate_power=gate_power
            )
            
            local_models[c] = local_model_c
            local_protos.append(local_model_c.get_proto().detach())

        logger.info(f"Round {cr} Finish--------|")

        method_name = 'OursV23_RawCE_SupCon'
        ensemble_model = WEnsembleFeature(model_list=local_models, weight_list=weights)
        global_proto = aggregate_local_protos(local_protos)
        ens_acc = eval_with_proto(ensemble_model, test_loader, device, global_proto)
        logger.info(f"The test accuracy of {method_name}: {ens_acc}")
        method_results[method_name].append(ens_acc)
        save_yaml_config(save_path + "/baselines_" + method_name +"_" + config['checkpoint']['result_file'], method_results)


def OneshotAblationCEOnly(trainset, test_loader, client_idx_map, config, device, gamma_reg=1e-5, lambda_max=50.0, **kwargs):
    """
    Ablation: CE-only on raw data (no SupCon).
    
    Tests the classification baseline without any contrastive learning.
    """
    logger.info('OneshotAblationCEOnly: CE only on raw data (no SupCon)')
    
    use_pretrain_bool = config.get('DBCD', {}).get('use_pretrain', False)
    use_imagenet_pretrain = config.get('DBCD', {}).get('use_imagenet_pretrain', False)
    custom_pretrain_path = config.get('pretrain', {}).get('model_path', '')
    dataset_name_lower = config['dataset']['data_name'].lower()
    model_name = config['server']['model_name']
    expected_custom_path = os.path.join(custom_pretrain_path, f"{dataset_name_lower}_{model_name}_centralized.pth")
    
    use_pretrain_arg = False
    if use_pretrain_bool:
        if os.path.exists(expected_custom_path):
            use_pretrain_arg = expected_custom_path
            logger.info(f"[OneshotAblationCEOnly] Custom weights at {expected_custom_path}")
        else:
            logger.warning(f"[OneshotAblationCEOnly] Custom weights NOT found. Random init.")
    elif use_imagenet_pretrain:
        use_pretrain_arg = True
        logger.info("[OneshotAblationCEOnly] ImageNet pretrained weights.")

    global_model = get_train_models(
        model_name=config['server']['model_name'],
        num_classes=config['dataset']['num_classes'],
        mode='our',
        use_pretrain=use_pretrain_arg,
        in_channel=config['dataset'].get('channels', 3)
    )
    method_results = defaultdict(list)
    save_path, local_model_dir = prepare_checkpoint_dir(config)
    if not os.path.exists(save_path + "/config.yaml"):
        save_yaml_config(save_path + "/config.yaml", config)
    local_models = [copy.deepcopy(global_model) for _ in range(config['client']['num_clients'])]
    local_data_size = [len(client_idx_map[c]) for c in range(config['client']['num_clients'])]
    if config['server']['aggregated_by_datasize']:
        weights = [i/sum(local_data_size) for i in local_data_size]
    else:
        weights = [1/config['client']['num_clients'] for _ in range(config['client']['num_clients'])]
    aug_transformer = get_supcon_transform(config['dataset']['data_name'])
    clients_sample_per_class = []
    
    total_rounds = config['server']['num_rounds']
    local_epochs = config['server']['local_epochs']
    logger.info(f'AblationCEOnly config: no lambda needed')

    for cr in trange(total_rounds):
        logger.info(f"Round {cr} starts--------| [AblationCEOnly: CE-only on raw data]")
        local_protos = []
        
        for c in range(config['client']['num_clients']):
            logger.info(f"Client {c} Starts Local Training--------|")
            client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])

            if cr == 0:
                clients_sample_per_class.append(generate_sample_per_class(config['dataset']['num_classes'], client_dataloader, len(client_idx_map[c])))

            start_ep = cr * local_epochs
            local_model_c = ours_local_training(
                model=copy.deepcopy(local_models[c]),
                training_data=client_dataloader,
                test_dataloader=test_loader,
                start_epoch=start_ep,
                local_epochs=local_epochs,
                optim_name=config['server']['optimizer'],
                lr=config['server']['lr'],
                momentum=config['server']['momentum'],
                loss_name=config['server']['loss_name'],
                device=device,
                num_classes=config['dataset']['num_classes'],
                sample_per_class=clients_sample_per_class[c],
                aug_transformer=aug_transformer,
                client_model_dir=local_model_dir + f"/client_{c}",
                total_rounds=total_rounds,
                warmup_epochs=0,
                use_ce_only_raw=True
            )
            
            local_models[c] = local_model_c
            local_protos.append(local_model_c.get_proto().detach())

        logger.info(f"Round {cr} Finish--------|")

        method_name = 'Ablation_CE_Only_Raw'
        ensemble_model = WEnsembleFeature(model_list=local_models, weight_list=weights)
        global_proto = aggregate_local_protos(local_protos)
        ens_acc = eval_with_proto(ensemble_model, test_loader, device, global_proto)
        logger.info(f"The test accuracy of {method_name}: {ens_acc}")
        method_results[method_name].append(ens_acc)
        save_yaml_config(save_path + "/baselines_" + method_name +"_" + config['checkpoint']['result_file'], method_results)


def OneshotAblationSupConOnly(trainset, test_loader, client_idx_map, config, device, gamma_reg=1e-5, lambda_max=50.0, **kwargs):
    """
    Ablation: SupCon-only on augmented views (no CE).
    
    Tests contrastive learning without classification supervision.
    """
    logger.info('OneshotAblationSupConOnly: SupCon only on augmented views (no CE)')
    
    use_pretrain_bool = config.get('DBCD', {}).get('use_pretrain', False)
    use_imagenet_pretrain = config.get('DBCD', {}).get('use_imagenet_pretrain', False)
    custom_pretrain_path = config.get('pretrain', {}).get('model_path', '')
    dataset_name_lower = config['dataset']['data_name'].lower()
    model_name = config['server']['model_name']
    expected_custom_path = os.path.join(custom_pretrain_path, f"{dataset_name_lower}_{model_name}_centralized.pth")
    
    use_pretrain_arg = False
    if use_pretrain_bool:
        if os.path.exists(expected_custom_path):
            use_pretrain_arg = expected_custom_path
            logger.info(f"[OneshotAblationSupConOnly] Custom weights at {expected_custom_path}")
        else:
            logger.warning(f"[OneshotAblationSupConOnly] Custom weights NOT found. Random init.")
    elif use_imagenet_pretrain:
        use_pretrain_arg = True
        logger.info("[OneshotAblationSupConOnly] ImageNet pretrained weights.")

    global_model = get_train_models(
        model_name=config['server']['model_name'],
        num_classes=config['dataset']['num_classes'],
        mode='our',
        use_pretrain=use_pretrain_arg,
        in_channel=config['dataset'].get('channels', 3)
    )
    method_results = defaultdict(list)
    save_path, local_model_dir = prepare_checkpoint_dir(config)
    if not os.path.exists(save_path + "/config.yaml"):
        save_yaml_config(save_path + "/config.yaml", config)
    local_models = [copy.deepcopy(global_model) for _ in range(config['client']['num_clients'])]
    local_data_size = [len(client_idx_map[c]) for c in range(config['client']['num_clients'])]
    if config['server']['aggregated_by_datasize']:
        weights = [i/sum(local_data_size) for i in local_data_size]
    else:
        weights = [1/config['client']['num_clients'] for _ in range(config['client']['num_clients'])]
    aug_transformer = get_supcon_transform(config['dataset']['data_name'])
    clients_sample_per_class = []
    
    total_rounds = config['server']['num_rounds']
    local_epochs = config['server']['local_epochs']
    lambda_supcon = config.get('v24_config', {}).get('lambda', 1.0)
    logger.info(f'AblationSupConOnly config: lambda_supcon={lambda_supcon}')

    for cr in trange(total_rounds):
        logger.info(f"Round {cr} starts--------| [AblationSupConOnly: SupCon-only on augmented, lambda={lambda_supcon}]")
        local_protos = []
        
        for c in range(config['client']['num_clients']):
            logger.info(f"Client {c} Starts Local Training--------|")
            client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])

            if cr == 0:
                clients_sample_per_class.append(generate_sample_per_class(config['dataset']['num_classes'], client_dataloader, len(client_idx_map[c])))

            start_ep = cr * local_epochs
            local_model_c = ours_local_training(
                model=copy.deepcopy(local_models[c]),
                training_data=client_dataloader,
                test_dataloader=test_loader,
                start_epoch=start_ep,
                local_epochs=local_epochs,
                optim_name=config['server']['optimizer'],
                lr=config['server']['lr'],
                momentum=config['server']['momentum'],
                loss_name=config['server']['loss_name'],
                device=device,
                num_classes=config['dataset']['num_classes'],
                sample_per_class=clients_sample_per_class[c],
                aug_transformer=aug_transformer,
                client_model_dir=local_model_dir + f"/client_{c}",
                total_rounds=total_rounds,
                lambda_align=lambda_supcon,
                warmup_epochs=0,
                use_supcon_only_aug=True
            )
            
            local_models[c] = local_model_c
            local_protos.append(local_model_c.get_proto().detach())

        logger.info(f"Round {cr} Finish--------|")

        method_name = 'Ablation_SupCon_Only_Aug'
        ensemble_model = WEnsembleFeature(model_list=local_models, weight_list=weights)
        global_proto = aggregate_local_protos(local_protos)
        ens_acc = eval_with_proto(ensemble_model, test_loader, device, global_proto)
        logger.info(f"The test accuracy of {method_name}: {ens_acc}")
        method_results[method_name].append(ens_acc)
        save_yaml_config(save_path + "/baselines_" + method_name +"_" + config['checkpoint']['result_file'], method_results)


def OneshotAblationAugCEAugSupCon(trainset, test_loader, client_idx_map, config, device, gamma_reg=1e-5, lambda_max=50.0, **kwargs):
    """
    Ablation: CE on augmented + SupCon on augmented (collapse test).
    
    Tests the scenario where both CE and SupCon operate on augmented views.
    This is the "collapse" experiment - if augmentation is too destructive,
    both CE and SupCon suffer from degraded inputs.
    """
    logger.info('OneshotAblationAugCEAugSupCon: CE on augmented + SupCon on augmented (collapse test)')
    
    use_pretrain_bool = config.get('DBCD', {}).get('use_pretrain', False)
    use_imagenet_pretrain = config.get('DBCD', {}).get('use_imagenet_pretrain', False)
    custom_pretrain_path = config.get('pretrain', {}).get('model_path', '')
    dataset_name_lower = config['dataset']['data_name'].lower()
    model_name = config['server']['model_name']
    expected_custom_path = os.path.join(custom_pretrain_path, f"{dataset_name_lower}_{model_name}_centralized.pth")
    
    use_pretrain_arg = False
    if use_pretrain_bool:
        if os.path.exists(expected_custom_path):
            use_pretrain_arg = expected_custom_path
            logger.info(f"[OneshotAblationAugCEAugSupCon] Custom weights at {expected_custom_path}")
        else:
            logger.warning(f"[OneshotAblationAugCEAugSupCon] Custom weights NOT found. Random init.")
    elif use_imagenet_pretrain:
        use_pretrain_arg = True
        logger.info("[OneshotAblationAugCEAugSupCon] ImageNet pretrained weights.")

    global_model = get_train_models(
        model_name=config['server']['model_name'],
        num_classes=config['dataset']['num_classes'],
        mode='our',
        use_pretrain=use_pretrain_arg,
        in_channel=config['dataset'].get('channels', 3)
    )
    method_results = defaultdict(list)
    save_path, local_model_dir = prepare_checkpoint_dir(config)
    if not os.path.exists(save_path + "/config.yaml"):
        save_yaml_config(save_path + "/config.yaml", config)
    local_models = [copy.deepcopy(global_model) for _ in range(config['client']['num_clients'])]
    local_data_size = [len(client_idx_map[c]) for c in range(config['client']['num_clients'])]
    if config['server']['aggregated_by_datasize']:
        weights = [i/sum(local_data_size) for i in local_data_size]
    else:
        weights = [1/config['client']['num_clients'] for _ in range(config['client']['num_clients'])]
    aug_transformer = get_supcon_transform(config['dataset']['data_name'])
    clients_sample_per_class = []
    
    total_rounds = config['server']['num_rounds']
    local_epochs = config['server']['local_epochs']
    lambda_supcon = config.get('v24_config', {}).get('lambda', 1.0)
    logger.info(f'AblationAugCEAugSupCon config: lambda_supcon={lambda_supcon}')

    for cr in trange(total_rounds):
        logger.info(f"Round {cr} starts--------| [AblationAugCEAugSupCon: CE+SupCon on augmented, lambda={lambda_supcon}]")
        local_protos = []
        
        for c in range(config['client']['num_clients']):
            logger.info(f"Client {c} Starts Local Training--------|")
            client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])

            if cr == 0:
                clients_sample_per_class.append(generate_sample_per_class(config['dataset']['num_classes'], client_dataloader, len(client_idx_map[c])))

            start_ep = cr * local_epochs
            local_model_c = ours_local_training(
                model=copy.deepcopy(local_models[c]),
                training_data=client_dataloader,
                test_dataloader=test_loader,
                start_epoch=start_ep,
                local_epochs=local_epochs,
                optim_name=config['server']['optimizer'],
                lr=config['server']['lr'],
                momentum=config['server']['momentum'],
                loss_name=config['server']['loss_name'],
                device=device,
                num_classes=config['dataset']['num_classes'],
                sample_per_class=clients_sample_per_class[c],
                aug_transformer=aug_transformer,
                client_model_dir=local_model_dir + f"/client_{c}",
                total_rounds=total_rounds,
                lambda_align=lambda_supcon,
                warmup_epochs=0,
                use_aug_ce_aug_supcon=True
            )
            
            local_models[c] = local_model_c
            local_protos.append(local_model_c.get_proto().detach())

        logger.info(f"Round {cr} Finish--------|")

        method_name = 'Ablation_AugCE_AugSupCon'
        ensemble_model = WEnsembleFeature(model_list=local_models, weight_list=weights)
        global_proto = aggregate_local_protos(local_protos)
        ens_acc = eval_with_proto(ensemble_model, test_loader, device, global_proto)
        logger.info(f"The test accuracy of {method_name}: {ens_acc}")
        method_results[method_name].append(ens_acc)
        save_yaml_config(save_path + "/baselines_" + method_name +"_" + config['checkpoint']['result_file'], method_results)


def OneshotOursV24(trainset, test_loader, client_idx_map, config, device, gamma_reg, lambda_max=50.0, **kwargs):
    """
    AURORA V24: Raw-Data CE + Flat SupCon (No Gate).
    
    Key insight: The consistency gate was fundamentally broken — it measured
    prediction confidence, not augmentation quality. FEMNIST had HIGHER consistency
    than CIFAR-100 (0.19 vs 0.12), so the gate was backwards.
    
    V24 removes the gate entirely. CE on raw data protects classification.
    SupCon on augmented views provides contrastive signal at full strength.
    lambda=1.0 (same weight as V14 baseline, no gate reduction)
    """
    logger.info('OneshotOursV24: Raw-Data CE + Flat SupCon (No Gate)')
    
    use_pretrain_bool = config.get('DBCD', {}).get('use_pretrain', False)
    use_imagenet_pretrain = config.get('DBCD', {}).get('use_imagenet_pretrain', False)
    custom_pretrain_path = config.get('pretrain', {}).get('model_path', '')
    dataset_name_lower = config['dataset']['data_name'].lower()
    model_name = config['server']['model_name']
    expected_custom_path = os.path.join(custom_pretrain_path, f"{dataset_name_lower}_{model_name}_centralized.pth")
    
    use_pretrain_arg = False
    if use_pretrain_bool:
        if os.path.exists(expected_custom_path):
            use_pretrain_arg = expected_custom_path
            logger.info(f"[OneshotOursV24] Custom weights at {expected_custom_path}")
        else:
            logger.warning(f"[OneshotOursV24] Custom weights NOT found. Random init.")
    elif use_imagenet_pretrain:
        use_pretrain_arg = True
        logger.info("[OneshotOursV24] ImageNet pretrained weights.")

    global_model = get_train_models(
        model_name=config['server']['model_name'],
        num_classes=config['dataset']['num_classes'],
        mode='our',
        use_pretrain=use_pretrain_arg,
        in_channel=config['dataset'].get('channels', 3)
    )
    method_results = defaultdict(list)
    save_path, local_model_dir = prepare_checkpoint_dir(config)
    if not os.path.exists(save_path + "/config.yaml"):
        save_yaml_config(save_path + "/config.yaml", config)
    local_models = [copy.deepcopy(global_model) for _ in range(config['client']['num_clients'])]
    local_data_size = [len(client_idx_map[c]) for c in range(config['client']['num_clients'])]
    if config['server']['aggregated_by_datasize']:
        weights = [i/sum(local_data_size) for i in local_data_size]
    else:
        weights = [1/config['client']['num_clients'] for _ in range(config['client']['num_clients'])]
    aug_transformer = get_supcon_transform(config['dataset']['data_name'])
    clients_sample_per_class = []
    
    total_rounds = config['server']['num_rounds']
    local_epochs = config['server']['local_epochs']
    lambda_supcon = config.get('v24_config', {}).get('lambda', 1.0)
    logger.info(f'V24 config: lambda_supcon={lambda_supcon}')

    for cr in trange(total_rounds):
        logger.info(f"Round {cr} starts--------| [V24: RawCE + flat SupCon, lambda={lambda_supcon}]")
        local_protos = []
        
        for c in range(config['client']['num_clients']):
            logger.info(f"Client {c} Starts Local Training--------|")
            client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])

            if cr == 0:
                clients_sample_per_class.append(generate_sample_per_class(config['dataset']['num_classes'], client_dataloader, len(client_idx_map[c])))

            start_ep = cr * local_epochs
            local_model_c = ours_local_training(
                model=copy.deepcopy(local_models[c]),
                training_data=client_dataloader,
                test_dataloader=test_loader,
                start_epoch=start_ep,
                local_epochs=local_epochs,
                optim_name=config['server']['optimizer'],
                lr=config['server']['lr'],
                momentum=config['server']['momentum'],
                loss_name=config['server']['loss_name'],
                device=device,
                num_classes=config['dataset']['num_classes'],
                sample_per_class=clients_sample_per_class[c],
                aug_transformer=aug_transformer,
                client_model_dir=local_model_dir + f"/client_{c}",
                total_rounds=total_rounds,
                lambda_align=lambda_supcon,
                warmup_epochs=0,
                use_raw_ce_flat_supcon=True
            )
            
            local_models[c] = local_model_c
            local_protos.append(local_model_c.get_proto().detach())

        logger.info(f"Round {cr} Finish--------|")

        method_name = 'OursV24_RawCE_FlatSupCon'
        ensemble_model = WEnsembleFeature(model_list=local_models, weight_list=weights)
        global_proto = aggregate_local_protos(local_protos)
        ens_acc = eval_with_proto(ensemble_model, test_loader, device, global_proto)
        logger.info(f"The test accuracy of {method_name}: {ens_acc}")
        method_results[method_name].append(ens_acc)
        save_yaml_config(save_path + "/baselines_" + method_name +"_" + config['checkpoint']['result_file'], method_results)
    return  # V24 only — V21 removed to avoid running twice
    """
    AURORA V21: Consistency-Gated Alignment-Only (No Warmup, No Uniformity).
    
    Key insight: The warmup/no-warmup trade-off is a SYMPTOM of a deeper problem —
    training CE on destructively augmented data. The fix:
    
    1. CE on RAW data always → clean classification signal regardless of augmentation
    2. Alignment between augmented views, gated by prediction consistency → 
       contrastive learning activates only when augmentation preserves content
    3. No uniformity → ETF anchor alignment prevents collapse instead
    4. No warmup needed → consistency gating provides adaptive "soft warmup"
    
    For FEMNIST (destructive aug): consistency starts ~1.6% → alignment gated to ~0 → CE-only
    For CIFAR-100 (mild aug): consistency starts ~1% → alignment gated to ~0 → CE-only
    Both converge as model learns, alignment activates naturally.
    """
    logger.info('OneshotOursV21: Consistency-Gated Alignment-Only (No Warmup)')
    
    v10_cfg = config.get('v10_config', {})
    
    use_pretrain_bool = config.get('DBCD', {}).get('use_pretrain', False)
    use_imagenet_pretrain = config.get('DBCD', {}).get('use_imagenet_pretrain', False)
    custom_pretrain_path = config.get('pretrain', {}).get('model_path', '')
    dataset_name_lower = config['dataset']['data_name'].lower()
    model_name = config['server']['model_name']
    expected_custom_path = os.path.join(custom_pretrain_path, f"{dataset_name_lower}_{model_name}_centralized.pth")
    
    use_pretrain_arg = False
    if use_pretrain_bool:
        if os.path.exists(expected_custom_path):
            use_pretrain_arg = expected_custom_path
            logger.info(f"[OneshotOursV21] Custom weights at {expected_custom_path}")
        else:
            logger.warning(f"[OneshotOursV21] Custom weights NOT found. Random init.")
    elif use_imagenet_pretrain:
        use_pretrain_arg = True
        logger.info("[OneshotOursV21] ImageNet pretrained weights.")

    global_model = get_train_models(
        model_name=config['server']['model_name'],
        num_classes=config['dataset']['num_classes'],
        mode='our',
        use_pretrain=use_pretrain_arg,
        in_channel=config['dataset'].get('channels', 3)
    )
    feature_dim = global_model.learnable_proto.shape[1]
    num_classes = config['dataset']['num_classes']
    etf_anchors = generate_etf_anchors(num_classes, feature_dim, device)
    method_results = defaultdict(list)
    save_path, local_model_dir = prepare_checkpoint_dir(config)
    if not os.path.exists(save_path + "/config.yaml"):
        save_yaml_config(save_path + "/config.yaml", config)
    local_models = [copy.deepcopy(global_model) for _ in range(config['client']['num_clients'])]
    local_data_size = [len(client_idx_map[c]) for c in range(config['client']['num_clients'])]
    if config['server']['aggregated_by_datasize']:
        weights = [i/sum(local_data_size) for i in local_data_size]
    else:
        weights = [1/config['client']['num_clients'] for _ in range(config['client']['num_clients'])]        
    aug_transformer = get_supcon_transform(config['dataset']['data_name'])
    clients_sample_per_class = []
    
    logger.info("--- Pre-heating V21: calibration with raw-data CE ---")
    client_lambdas = []
    for c in range(config['client']['num_clients']):
        client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])
        model_copy = copy.deepcopy(local_models[c]).to(device)
        model_copy.eval()
        with torch.no_grad():
            batch = next(iter(client_dataloader))
            data, target = batch[0].to(device), batch[1].to(device)
            logits, _ = model_copy(data)
            cls_val = torch.nn.functional.cross_entropy(logits, target).item()
            
            aug1, aug2 = aug_transformer(data), aug_transformer(data)
            _, f1 = model_copy(aug1)
            _, f2 = model_copy(aug2)
            align_val = torch.mean((f1 - f2).norm(dim=1).pow(2)).item()
            
            logits1, _ = model_copy(aug1)
            logits2, _ = model_copy(aug2)
            cons_val = (logits1.argmax(dim=1) == logits2.argmax(dim=1)).float().mean().item()
            
            unique_classes = torch.unique(target)
            if len(unique_classes) > 0:
                proto_subset = model_copy.learnable_proto[unique_classes]
                anchor_subset = etf_anchors[unique_classes]
                etf_val = torch.nn.functional.mse_loss(proto_subset, anchor_subset).item()
            else:
                etf_val = 0.0
            
            reg_val = max(cons_val, 0.01) * align_val + etf_val
            initial_lambda = min(max(cls_val / max(reg_val, 1e-6), 0.01), 2.0) if reg_val > 1e-6 else 0.5
        
        client_lambdas.append(initial_lambda)
        del model_copy
        logger.info(f"Client {c}: cls={cls_val:.4f}, align={align_val:.4f}, cons={cons_val:.4f}, etf={etf_val:.4f}, lambda={initial_lambda:.4f}")
        local_models[c] = local_models[c].to(device)
    logger.info("--- V21 pre-heating complete. ---")

    logger.info(f"Gamma: {gamma_reg}")
    sigma_lr_val = v10_cfg.get('sigma_lr', 0.005)
    total_rounds = config['server']['num_rounds']
    local_epochs = config['server']['local_epochs']

    for cr in trange(total_rounds):
        logger.info(f"Round {cr} starts--------| [V21: no warmup, align-only, consistency-gated]")
        local_protos = []
        
        for c in range(config['client']['num_clients']):
            logger.info(f"Client {c} Starts Local Training--------|")
            client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])

            if cr == 0:
                clients_sample_per_class.append(generate_sample_per_class(config['dataset']['num_classes'], client_dataloader, len(client_idx_map[c])))

            start_ep = cr * local_epochs
            local_model_c = ours_local_training(
                model=copy.deepcopy(local_models[c]),
                training_data=client_dataloader,
                test_dataloader=test_loader,
                start_epoch=start_ep,
                local_epochs=local_epochs,
                optim_name=config['server']['optimizer'],
                lr=config['server']['lr'],
                momentum=config['server']['momentum'],
                loss_name=config['server']['loss_name'],
                device=device,
                num_classes=config['dataset']['num_classes'],
                sample_per_class=clients_sample_per_class[c],
                aug_transformer=aug_transformer,
                client_model_dir=local_model_dir + f"/client_{c}",
                total_rounds=total_rounds,
                use_drcl=True,
                fixed_anchors=etf_anchors,
                lambda_align=client_lambdas[c],
                use_uncertainty_weighting=False,
                use_dynamic_task_attenuation=False,
                gamma_reg=gamma_reg,
                lambda_max=lambda_max,
                use_reclassified_losses=True,
                warmup_epochs=0,
                use_confidence_gating=False,
                use_align_uniform=False,
                use_align_only=True
            )
            
            local_models[c] = local_model_c
            local_protos.append(local_model_c.get_proto().detach())

            logger.info(f"Client {c} Post-Training -> Fixed λ: {client_lambdas[c]:.4f}")

        logger.info(f"Round {cr} Finish--------|")

        method_name = 'OursV21_AlignOnly'
        ensemble_model = WEnsembleFeature(model_list=local_models, weight_list=weights)
        global_proto = aggregate_local_protos(local_protos)
        ens_acc = eval_with_proto(ensemble_model, test_loader, device, global_proto)
        logger.info(f"The test accuracy of {method_name}: {ens_acc}")
        method_results[method_name].append(ens_acc)
        save_yaml_config(save_path + "/baselines_" + method_name +"_" + config['checkpoint']['result_file'], method_results)


def OneshotOursV20(trainset, test_loader, client_idx_map, config, device, gamma_reg, lambda_max=50.0, **kwargs):
    """
    AURORA V20: Alignment + Uniformity (Wang & Isola, ICML 2020).
    
    Replaces SupCon + ProtoFeatCon + ProtoCon with augmentation-robust
    Alignment + Uniformity loss. Key insight: uniformity prevents collapse
    via geometric regularization on the hypersphere, NOT through negative
    pairs from strong augmentation. This makes it robust to augmentation
    strength — the same algorithm works for both FEMNIST (fine-grained,
    needs mild aug) and CIFAR-100 (natural images, strong aug OK).
    
    NO warmup needed — the whole point is that AU doesn't suffer from
    augmentation-induced feature destruction like SupCon does.
    """
    logger.info('OneshotOursV20: AURORA with Alignment + Uniformity (Augmentation-Robust)')
    
    v10_cfg = config.get('v10_config', {})
    
    use_pretrain_bool = config.get('DBCD', {}).get('use_pretrain', False)
    use_imagenet_pretrain = config.get('DBCD', {}).get('use_imagenet_pretrain', False)
    custom_pretrain_path = config.get('pretrain', {}).get('model_path', '')
    dataset_name_lower = config['dataset']['data_name'].lower()
    model_name = config['server']['model_name']
    expected_custom_path = os.path.join(custom_pretrain_path, f"{dataset_name_lower}_{model_name}_centralized.pth")
    
    use_pretrain_arg = False
    if use_pretrain_bool:
        if os.path.exists(expected_custom_path):
            use_pretrain_arg = expected_custom_path
            logger.info(f"[OneshotOursV20] Found custom centralized weights at {expected_custom_path}. Using them!")
        else:
            use_pretrain_arg = False
            logger.warning(f"[OneshotOursV20] Custom weights NOT found at {expected_custom_path}. Using Random Initialization.")
    elif use_imagenet_pretrain:
        use_pretrain_arg = True
        logger.info("[OneshotOursV20] Loading ImageNet pretrained weights for ResNet-18.")

    global_model = get_train_models(
        model_name=config['server']['model_name'],
        num_classes=config['dataset']['num_classes'],
        mode='our',
        use_pretrain=use_pretrain_arg,
        in_channel=config['dataset'].get('channels', 3)
    )
    feature_dim = global_model.learnable_proto.shape[1]
    num_classes = config['dataset']['num_classes']
    etf_anchors = generate_etf_anchors(num_classes, feature_dim, device)
    method_results = defaultdict(list)
    save_path, local_model_dir = prepare_checkpoint_dir(config)
    if not os.path.exists(save_path + "/config.yaml"):
        save_yaml_config(save_path + "/config.yaml", config)
    local_models = [copy.deepcopy(global_model) for _ in range(config['client']['num_clients'])]
    local_data_size = [len(client_idx_map[c]) for c in range(config['client']['num_clients'])]
    if config['server']['aggregated_by_datasize']:
        weights = [i/sum(local_data_size) for i in local_data_size]
    else:
        weights = [1/config['client']['num_clients'] for _ in range(config['client']['num_clients'])]        
    aug_transformer = get_supcon_transform(config['dataset']['data_name'])
    clients_sample_per_class = []
    
    logger.info("--- Pre-heating V20 models with loss-ratio calibration ---")
    
    from oneshot_algorithms.ours.unsupervised_loss import AlignmentUniformityLoss
    au_fn = AlignmentUniformityLoss(alpha=2, t=2)
    
    for c in range(config['client']['num_clients']):
        client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])
        
        model_copy = copy.deepcopy(local_models[c]).to(device)
        model_copy.eval()
        with torch.no_grad():
            batch = next(iter(client_dataloader))
            data, target = batch[0].to(device), batch[1].to(device)
            
            logits, feature_norm = model_copy(data)
            cls_val = torch.nn.functional.cross_entropy(logits, target).item()
            
            # Estimate AU loss on two augmented views
            aug1, aug2 = aug_transformer(data), aug_transformer(data)
            _, f1 = model_copy(aug1)
            _, f2 = model_copy(aug2)
            au_val = au_fn(f1, f2).item()
            
            # Estimate ETF alignment loss
            unique_classes = torch.unique(target)
            if len(unique_classes) > 0:
                proto_subset = model_copy.learnable_proto[unique_classes]
                anchor_subset = etf_anchors[unique_classes]
                align_val = torch.nn.functional.mse_loss(proto_subset, anchor_subset).item()
            else:
                align_val = 0.0
            
            reg_val = au_val + align_val
            
            if reg_val > 1e-6:
                initial_lambda = min(max(cls_val / reg_val, 0.01), 2.0)
            else:
                initial_lambda = 0.5
        
        del model_copy
        
        logger.info(f"Client {c}: cls={cls_val:.4f}, au={au_val:.4f}, align={align_val:.4f}, reg={reg_val:.4f}, initial_lambda={initial_lambda:.4f}")
        local_models[c] = local_models[c].to(device)
        initial_log_lambda = torch.tensor(initial_lambda, device=device).log()
        local_models[c].log_sigma_sq_local = torch.nn.Parameter(initial_log_lambda)
        local_models[c].log_sigma_sq_align = torch.nn.Parameter(torch.tensor(0.0, device=device))
    logger.info("--- V20 model pre-heating complete. ---")

    logger.info(f"Gamma for stability anchor regularization: {gamma_reg}")

    sigma_lr_val = v10_cfg.get('sigma_lr', 0.005)
    total_rounds = config['server']['num_rounds']
    local_epochs = config['server']['local_epochs']

    for cr in trange(total_rounds):
        warmup_ep = v10_cfg.get('warmup_epochs', 5)
        logger.info(f"Round {cr} starts--------| [V20: warmup={warmup_ep}, AU loss]")
        local_protos = []
        
        for c in range(config['client']['num_clients']):
            logger.info(f"Client {c} Starts Local Training--------|")
            client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])

            if cr == 0:
                clients_sample_per_class.append(generate_sample_per_class(config['dataset']['num_classes'], client_dataloader, len(client_idx_map[c])))

            start_ep = cr * local_epochs
            local_model_c = ours_local_training(
                model=copy.deepcopy(local_models[c]),
                training_data=client_dataloader,
                test_dataloader=test_loader,
                start_epoch=start_ep,
                local_epochs=local_epochs,
                optim_name=config['server']['optimizer'],
                lr=config['server']['lr'],
                momentum=config['server']['momentum'],
                loss_name=config['server']['loss_name'],
                device=device,
                num_classes=config['dataset']['num_classes'],
                sample_per_class=clients_sample_per_class[c],
                aug_transformer=aug_transformer,
                client_model_dir=local_model_dir + f"/client_{c}",
                total_rounds=total_rounds,
                use_drcl=True,
                fixed_anchors=etf_anchors,
                use_uncertainty_weighting=True,
                sigma_lr=sigma_lr_val,
                use_dynamic_task_attenuation=False,
                gamma_reg=gamma_reg,
                lambda_max=lambda_max,
                use_reclassified_losses=True,
                warmup_epochs=v10_cfg.get('warmup_epochs', 5),  # Warmup for initial features; AU then takes over
                use_confidence_gating=False,
                use_align_uniform=True      # V20: use AU instead of SupCon
            )
            
            local_models[c] = local_model_c
            local_protos.append(local_model_c.get_proto().detach())

            log_sigma_sq_local_val = local_model_c.log_sigma_sq_local.item()
            log_sigma_sq_align_val = local_model_c.log_sigma_sq_align.item()
            raw_lambda = math.exp(log_sigma_sq_local_val - log_sigma_sq_align_val)

            logger.info(
                f"Client {c} Post-Training State -> "
                f"Raw λ: {raw_lambda:9.4f}"
            )

        logger.info(f"Round {cr} Finish--------|")

        method_name = 'OursV20_AU'
        ensemble_model = WEnsembleFeature(model_list=local_models, weight_list=weights)
        global_proto = aggregate_local_protos(local_protos)
        ens_acc = eval_with_proto(ensemble_model, test_loader, device, global_proto)
        logger.info(f"The test accuracy of {method_name}: {ens_acc}")
        method_results[method_name].append(ens_acc)
        save_yaml_config(save_path + "/baselines_" + method_name +"_" + config['checkpoint']['result_file'], method_results)



    logger.info('Running Feature Collapse Ablation: Direct Alignment (No Detach)')
    
    global_model = get_train_models(
        model_name=config['server']['model_name'],
        num_classes=config['dataset']['num_classes'],
        mode='our'
    ,
        in_channel=config['dataset'].get('channels', 3))
    global_model.to(device)
    global_model.train()

    # --- Use ETF Anchors ---
    feature_dim = global_model.learnable_proto.shape[1]
    num_classes = config['dataset']['num_classes']
    fixed_anchors = generate_etf_anchors(num_classes, feature_dim, device)
    
    method_results = defaultdict(list)
    save_path, local_model_dir = prepare_checkpoint_dir(config)
    save_yaml_config(save_path + "/config.yaml", config) 

    local_models = [copy.deepcopy(global_model) for _ in range(config['client']['num_clients'])]
    
    local_data_size = [len(client_idx_map[c]) for c in range(config['client']['num_clients'])]
    if config['server']['aggregated_by_datasize']:
        weights = [i/sum(local_data_size) for i in local_data_size]
    else:
        weights = [1/config['client']['num_clients'] for _ in range(config['client']['num_clients'])]        
    
    aug_transformer = get_gpu_augmentation(config['dataset']['data_name'], device)

    clients_sample_per_class = []
    total_rounds = config['server']['num_rounds']

    for cr in trange(total_rounds):
        logger.info(f"Round {cr} starts--------|")
        local_protos = []
        for c in range(config['client']['num_clients']):
            client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])
            if cr == 0:
                clients_sample_per_class.append(generate_sample_per_class(config['dataset']['num_classes'], client_dataloader, len(client_idx_map[c])))

            if (lambda_val > 0):
                lambda_align_initial = lambda_val
            elif 'lambda_align' in config:
                lambda_align_initial = config['lambda_align']
            else:
                 lambda_align_initial = config.get('lambda_align', 1.0)

            local_model_c = ours_local_training(
                model=copy.deepcopy(local_models[c]),
                training_data=client_dataloader,
                test_dataloader=test_loader,
                start_epoch=cr * config['server']['local_epochs'],
                local_epochs=config['server']['local_epochs'],
                optim_name=config['server']['optimizer'],
                lr=config['server']['lr'],
                momentum=config['server']['momentum'],
                loss_name=config['server']['loss_name'],
                device=device,
                num_classes=config['dataset']['num_classes'],
                sample_per_class=clients_sample_per_class[c],
                aug_transformer=aug_transformer,
                client_model_dir=local_model_dir + f"/client_{c}",
                total_rounds=total_rounds,
                save_freq=config['checkpoint']['save_freq'],
                use_drcl=True,
                fixed_anchors=fixed_anchors,
                lambda_align=lambda_align_initial,
                force_feature_alignment=True # <--- KEY CHANGE: Force direct feature alignment
            )
            
            local_models[c] = local_model_c
            local_protos.append(local_model_c.get_proto().detach())

        # Evaluation (using standard feature ensemble)
        method_name = 'Ours_FeatureCollapse_Ablation'
        ensemble_model = WEnsembleFeature(model_list=local_models, weight_list=weights)
        global_proto = aggregate_local_protos(local_protos)
        ens_acc = eval_with_proto(ensemble_model, test_loader, device, global_proto)
        
        logger.info(f"The test accuracy of {method_name}: {ens_acc}")
        method_results[method_name].append(ens_acc)
        save_yaml_config(save_path + "/baselines_" + method_name +"_" + config['checkpoint']['result_file'], method_results)


def OneshotOursV24FedAvg(trainset, test_loader, client_idx_map, config, device, gamma_reg=1e-5, lambda_max=50.0, **kwargs):
    logger.info('OneshotOursV24FedAvg: V24 RawCE + Flat SupCon + FedAvg Parameter Averaging')
    
    use_pretrain_bool = config.get('DBCD', {}).get('use_pretrain', False)
    use_imagenet_pretrain = config.get('DBCD', {}).get('use_imagenet_pretrain', False)
    custom_pretrain_path = config.get('pretrain', {}).get('model_path', '')
    dataset_name_lower = config['dataset']['data_name'].lower()
    model_name = config['server']['model_name']
    expected_custom_path = os.path.join(custom_pretrain_path, f"{dataset_name_lower}_{model_name}_centralized.pth")
    
    use_pretrain_arg = False
    if use_pretrain_bool:
        if os.path.exists(expected_custom_path):
            use_pretrain_arg = expected_custom_path
            logger.info(f"[OneshotOursV24FedAvg] Custom weights at {expected_custom_path}")
        else:
            logger.warning(f"[OneshotOursV24FedAvg] Custom weights NOT found. Random init.")
    elif use_imagenet_pretrain:
        use_pretrain_arg = True
        logger.info("[OneshotOursV24FedAvg] ImageNet pretrained weights.")

    global_model = get_train_models(
        model_name=config['server']['model_name'],
        num_classes=config['dataset']['num_classes'],
        mode='our',
        use_pretrain=use_pretrain_arg,
        in_channel=config['dataset'].get('channels', 3)
    )
    method_results = defaultdict(list)
    save_path, local_model_dir = prepare_checkpoint_dir(config)
    if not os.path.exists(save_path + "/config.yaml"):
        save_yaml_config(save_path + "/config.yaml", config)
    local_models = [copy.deepcopy(global_model) for _ in range(config['client']['num_clients'])]
    local_data_size = [len(client_idx_map[c]) for c in range(config['client']['num_clients'])]
    if config['server']['aggregated_by_datasize']:
        weights = [i/sum(local_data_size) for i in local_data_size]
    else:
        weights = [1/config['client']['num_clients'] for _ in range(config['client']['num_clients'])]
    aug_transformer = get_supcon_transform(config['dataset']['data_name'])
    clients_sample_per_class = []
    
    total_rounds = config['server']['num_rounds']
    local_epochs = config['server']['local_epochs']
    lambda_supcon = config.get('v24_config', {}).get('lambda', 1.0)
    logger.info(f'V24FedAvg config: lambda_supcon={lambda_supcon}')

    for cr in trange(total_rounds):
        logger.info(f"Round {cr} starts--------| [V24FedAvg: RawCE + flat SupCon + FedAvg, lambda={lambda_supcon}]")
        
        for c in range(config['client']['num_clients']):
            logger.info(f"Client {c} Starts Local Training--------|")
            client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])

            if cr == 0:
                clients_sample_per_class.append(generate_sample_per_class(config['dataset']['num_classes'], client_dataloader, len(client_idx_map[c])))

            start_ep = cr * local_epochs
            local_model_c = ours_local_training(
                model=copy.deepcopy(local_models[c]),
                training_data=client_dataloader,
                test_dataloader=test_loader,
                start_epoch=start_ep,
                local_epochs=local_epochs,
                optim_name=config['server']['optimizer'],
                lr=config['server']['lr'],
                momentum=config['server']['momentum'],
                loss_name=config['server']['loss_name'],
                device=device,
                num_classes=config['dataset']['num_classes'],
                sample_per_class=clients_sample_per_class[c],
                aug_transformer=aug_transformer,
                client_model_dir=local_model_dir + f"/client_{c}",
                total_rounds=total_rounds,
                lambda_align=lambda_supcon,
                warmup_epochs=0,
                use_raw_ce_flat_supcon=True
            )
            
            local_models[c] = local_model_c

        logger.info(f"Round {cr} Finish--------|")
        
        method_name = 'OursV24FedAvg'
        aggregated_model = parameter_averaging(local_models, weights)
        acc = test_acc_our_model(aggregated_model, test_loader, device)
        logger.info(f"The test accuracy of {method_name}: {acc}")
        method_results[method_name].append(acc)
        
        local_models = [copy.deepcopy(aggregated_model) for _ in range(config['client']['num_clients'])]

        save_yaml_config(save_path + "/baselines_" + method_name +"_" + config['checkpoint']['result_file'], method_results)


def OneshotOursV24Projector(trainset, test_loader, client_idx_map, config, device, gamma_reg=1e-5, lambda_max=50.0, **kwargs):
    logger.info('OneshotOursV24Projector: V24 RawCE + Flat SupCon + Projector Head')
    
    use_pretrain_bool = config.get('DBCD', {}).get('use_pretrain', False)
    use_imagenet_pretrain = config.get('DBCD', {}).get('use_imagenet_pretrain', False)
    custom_pretrain_path = config.get('pretrain', {}).get('model_path', '')
    dataset_name_lower = config['dataset']['data_name'].lower()
    model_name = config['server']['model_name']
    expected_custom_path = os.path.join(custom_pretrain_path, f"{dataset_name_lower}_{model_name}_centralized.pth")
    
    use_pretrain_arg = False
    if use_pretrain_bool:
        if os.path.exists(expected_custom_path):
            use_pretrain_arg = expected_custom_path
            logger.info(f"[OneshotOursV24Projector] Custom weights at {expected_custom_path}")
        else:
            logger.warning(f"[OneshotOursV24Projector] Custom weights NOT found. Random init.")
    elif use_imagenet_pretrain:
        use_pretrain_arg = True
        logger.info("[OneshotOursV24Projector] ImageNet pretrained weights.")

    global_model = get_train_models(
        model_name=config['server']['model_name'],
        num_classes=config['dataset']['num_classes'],
        mode='our_projector',
        use_pretrain=use_pretrain_arg,
        in_channel=config['dataset'].get('channels', 3)
    )
    
    feature_dim = global_model.learnable_proto.shape[1]
    logger.info(f"Initialized V24Projector with Projector. Feature Dim: {feature_dim}")
    
    method_results = defaultdict(list)
    save_path, local_model_dir = prepare_checkpoint_dir(config)
    if not os.path.exists(save_path + "/config.yaml"):
        save_yaml_config(save_path + "/config.yaml", config)
    local_models = [copy.deepcopy(global_model) for _ in range(config['client']['num_clients'])]
    local_data_size = [len(client_idx_map[c]) for c in range(config['client']['num_clients'])]
    if config['server']['aggregated_by_datasize']:
        weights = [i/sum(local_data_size) for i in local_data_size]
    else:
        weights = [1/config['client']['num_clients'] for _ in range(config['client']['num_clients'])]
    aug_transformer = get_supcon_transform(config['dataset']['data_name'])
    clients_sample_per_class = []
    
    total_rounds = config['server']['num_rounds']
    local_epochs = config['server']['local_epochs']
    lambda_supcon = config.get('v24_config', {}).get('lambda', 1.0)
    logger.info(f'V24Projector config: lambda_supcon={lambda_supcon}')

    for cr in trange(total_rounds):
        logger.info(f"Round {cr} starts--------| [V24Projector: RawCE + flat SupCon + Projector, lambda={lambda_supcon}]")
        local_protos = []
        
        for c in range(config['client']['num_clients']):
            logger.info(f"Client {c} Starts Local Training--------|")
            client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])

            if cr == 0:
                clients_sample_per_class.append(generate_sample_per_class(config['dataset']['num_classes'], client_dataloader, len(client_idx_map[c])))

            start_ep = cr * local_epochs
            local_model_c = ours_local_training(
                model=copy.deepcopy(local_models[c]),
                training_data=client_dataloader,
                test_dataloader=test_loader,
                start_epoch=start_ep,
                local_epochs=local_epochs,
                optim_name=config['server']['optimizer'],
                lr=config['server']['lr'],
                momentum=config['server']['momentum'],
                loss_name=config['server']['loss_name'],
                device=device,
                num_classes=config['dataset']['num_classes'],
                sample_per_class=clients_sample_per_class[c],
                aug_transformer=aug_transformer,
                client_model_dir=local_model_dir + f"/client_{c}",
                total_rounds=total_rounds,
                lambda_align=lambda_supcon,
                warmup_epochs=0,
                use_raw_ce_flat_supcon=True
            )
            
            local_models[c] = local_model_c
            local_protos.append(local_model_c.get_proto().detach())

        logger.info(f"Round {cr} Finish--------|")

        method_name = 'OursV24Projector'
        ensemble_model = WEnsembleFeature(model_list=local_models, weight_list=weights)
        global_proto = aggregate_local_protos(local_protos)
        ens_acc = eval_with_proto(ensemble_model, test_loader, device, global_proto)
        logger.info(f"The test accuracy of {method_name}: {ens_acc}")
        method_results[method_name].append(ens_acc)
        save_yaml_config(save_path + "/baselines_" + method_name +"_" + config['checkpoint']['result_file'], method_results)
