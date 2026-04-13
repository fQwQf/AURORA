"""
Gradient Cosine Similarity Analysis for AURORA V24.

Computes per-round cosine similarity between CE and SupCon gradients
to validate the core claim: CE on raw data and SupCon on augmented views
produce orthogonal gradients.

Runs on CIFAR-10 (alpha=0.05, K=5) for 20 rounds.
"""

import sys
sys.path.insert(0, '/data1/tongjizhou/FAFI_ICML25')

import torch
import torch.nn.functional as F
import numpy as np
import yaml
import os
import copy
import pickle
from collections import defaultdict
from torch.utils.data import DataLoader

from dataset_helper import get_fl_dataset
from models_lib import get_train_models
from oneshot_algorithms.ours.our_main import get_supcon_transform
from oneshot_algorithms.ours.unsupervised_loss import SupConLoss
from oneshot_algorithms.utils import init_optimizer


def compute_grad_cosine_similarity(model, data_batch, target_batch, aug_transformer,
                                     cls_loss_fn, supcon_fn, device, lambda_val=1.0):
    data = data_batch.to(device)
    target = target_batch.to(device)
    
    param_list = list(model.parameters())
    
    # --- Compute CE gradient ---
    model.zero_grad()
    logits_raw, _ = model(data)
    ce_loss = cls_loss_fn(logits_raw, target)
    ce_loss.backward()
    
    ce_grads = []
    for p in param_list:
        if p.grad is not None:
            ce_grads.append(p.grad.clone().flatten().detach())
    ce_grad_vec = torch.cat(ce_grads)
    ce_grad_norm = ce_grad_vec.norm().item()
    
    # --- Compute SupCon gradient ---
    model.zero_grad()
    aug_data1 = aug_transformer(data)
    aug_data2 = aug_transformer(data)
    _, f1 = model(aug_data1)
    _, f2 = model(aug_data2)
    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
    supcon_loss = supcon_fn(features, target)
    (lambda_val * supcon_loss).backward()
    
    supcon_grads = []
    for p in param_list:
        if p.grad is not None:
            supcon_grads.append(p.grad.clone().flatten().detach())
    supcon_grad_vec = torch.cat(supcon_grads)
    supcon_grad_norm = supcon_grad_vec.norm().item()
    
    # Handle size mismatch (pad shorter with zeros)
    max_len = max(len(ce_grad_vec), len(supcon_grad_vec))
    if len(ce_grad_vec) < max_len:
        ce_grad_vec = torch.cat([ce_grad_vec, torch.zeros(max_len - len(ce_grad_vec), device=device)])
    if len(supcon_grad_vec) < max_len:
        supcon_grad_vec = torch.cat([supcon_grad_vec, torch.zeros(max_len - len(supcon_grad_vec), device=device)])
    
    if ce_grad_norm < 1e-8 or supcon_grad_norm < 1e-8:
        cosine_sim = 0.0
    else:
        cosine_sim = F.cosine_similarity(ce_grad_vec.unsqueeze(0), 
                                          supcon_grad_vec.unsqueeze(0)).item()
    
    return cosine_sim, ce_grad_norm, supcon_grad_norm


def main():
    device = torch.device('cuda:7')  # Use free GPU
    
    # Config
    config = {
        'dataset': {
            'data_name': 'CIFAR10',
            'root_path': '/data1/tongjizhou/FAFI_ICML25/data/',
            'train_batch_size': 256,
            'test_batch_size': 256,
            'channels': 3,
            'num_classes': 10,
            'image_size': 32,
        },
        'distribution': {
            'type': 'dirichlet',
            'label_num_per_client': 2,
            'alpha': 0.05,
        },
        'client': {'num_clients': 5},
        'server': {
            'num_rounds': 20,
            'frac_clients': 1.0,
            'lr': 0.05,
            'local_epochs': 1,
            'optimizer': 'sgd',
            'momentum': 0.9,
            'weight_decay': 0.0005,
            'loss_name': 'ce',
            'model_name': 'resnet18',
            'aggregated_by_datasize': True,
            'lr_decay_per_round': 0.998,
        },
        'seed': 1,
    }
    
    print("Loading dataset...")
    trainset, testset, client_idx_map = get_fl_dataset(
        config['dataset']['data_name'],
        config['dataset']['root_path'],
        config['client']['num_clients'],
        config['distribution']['type'],
        config['distribution']['label_num_per_client'],
        config['distribution']['alpha'],
        normalize_train=False,
        normalize_test=True
    )
    test_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=True)
    
    print("Initializing model...")
    global_model = get_train_models(
        model_name='resnet18',
        num_classes=10,
        mode='our',
        use_pretrain=False,
        in_channel=3
    )
    
    aug_transformer = get_supcon_transform('CIFAR10')
    cls_loss_fn = torch.nn.CrossEntropyLoss()
    supcon_fn = SupConLoss(temperature=0.07)
    
    num_rounds = config['server']['num_rounds']
    num_clients = config['client']['num_clients']
    
    results = {
        'round': [],
        'client': [],
        'batch': [],
        'cosine_sim': [],
        'ce_grad_norm': [],
        'supcon_grad_norm': [],
    }
    
    # Aggregate stats per round
    round_stats = {
        'round': [],
        'mean_cosine_sim': [],
        'std_cosine_sim': [],
        'mean_ce_norm': [],
        'mean_supcon_norm': [],
    }
    
    print(f"Starting gradient analysis for {num_rounds} rounds...")
    
    for cr in range(num_rounds):
        local_models = [copy.deepcopy(global_model) for _ in range(num_clients)]
        round_cosines = []
        round_ce_norms = []
        round_supcon_norms = []
        
        for c in range(num_clients):
            model_c = local_models[c].to(device)
            model_c.train()
            optimizer = torch.optim.SGD(model_c.parameters(), lr=0.05, momentum=0.9, weight_decay=0.0005)
            
            client_indices = client_idx_map[c]
            client_dataset = torch.utils.data.Subset(trainset, client_indices)
            client_loader = DataLoader(client_dataset, batch_size=256, shuffle=True)
            
            # Analyze gradients on first 3 batches, then train normally
            batch_cosines = []
            for batch_idx, (data, target) in enumerate(client_loader):
                if batch_idx < 3:  # Sample first 3 batches for analysis
                    cos_sim, ce_norm, supcon_norm = compute_grad_cosine_similarity(
                        model_c, data, target, aug_transformer, cls_loss_fn, supcon_fn, device, lambda_val=1.0
                    )
                    batch_cosines.append(cos_sim)
                    round_cosines.append(cos_sim)
                    round_ce_norms.append(ce_norm)
                    round_supcon_norms.append(supcon_norm)
                
                # Normal training step (same as V24)
                model_c.zero_grad()
                data, target = data.to(device), target.to(device)
                
                logits_raw, _ = model_c(data)
                ce_loss = cls_loss_fn(logits_raw, target)
                
                aug_data1, aug_data2 = aug_transformer(data), aug_transformer(data)
                _, f1 = model_c(aug_data1)
                _, f2 = model_c(aug_data2)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                supcon_loss = supcon_fn(features, target)
                
                loss = ce_loss + supcon_loss
                loss.backward()
                optimizer.step()
            
            local_models[c] = model_c.cpu()
            
            if c == 0:  # Log for first client
                mean_cos = np.mean(batch_cosines)
                print(f"  Round {cr}, Client {c}: mean cosine_sim={mean_cos:.4f}, "
                      f"ce_norm={round_ce_norms[-1]:.2f}, supcon_norm={round_supcon_norms[-1]:.2f}")
        
        # Prototype aggregation (simple average for this analysis)
        global_state = copy.deepcopy(local_models[0].state_dict())
        for key in global_state:
            if 'learnable_proto' in key:
                global_state[key] = sum([m.state_dict()[key] for m in local_models]) / num_clients
            else:
                # IFFI-style: take average
                global_state[key] = sum([m.state_dict()[key] for m in local_models]) / num_clients
        global_model.load_state_dict(global_state)
        
        # Record round stats
        round_stats['round'].append(cr)
        round_stats['mean_cosine_sim'].append(float(np.mean(round_cosines)))
        round_stats['std_cosine_sim'].append(float(np.std(round_cosines)))
        round_stats['mean_ce_norm'].append(float(np.mean(round_ce_norms)))
        round_stats['mean_supcon_norm'].append(float(np.mean(round_supcon_norms)))
        
        print(f"Round {cr}: mean_cosine={np.mean(round_cosines):.4f} ± {np.std(round_cosines):.4f}")
    
    # Save results
    save_path = '/data1/tongjizhou/FAFI_ICML25/analysis/gradient_cosine_results.yaml'
    with open(save_path, 'w') as f:
        yaml.dump(round_stats, f, default_flow_style=False)
    
    print(f"\nResults saved to {save_path}")
    print("\n=== Summary ===")
    for i, r in enumerate(round_stats['round']):
        print(f"Round {r:2d}: cosine_sim = {round_stats['mean_cosine_sim'][i]:+.4f} ± {round_stats['std_cosine_sim'][i]:.4f}")
    
    overall_mean = np.mean(round_stats['mean_cosine_sim'])
    overall_std = np.std(round_stats['mean_cosine_sim'])
    print(f"\nOverall: cosine_sim = {overall_mean:+.4f} ± {overall_std:.4f}")


if __name__ == '__main__':
    main()
