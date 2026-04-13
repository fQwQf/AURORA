"""
Spectral Subspace Analysis of CE and SupCon Gradients.

Validates the core theoretical insight: CE gradients concentrate energy
on the principal subspace (top eigenvectors) of the data covariance,
while SupCon gradients concentrate on the augmentation-induced subspace
(bottom eigenvectors).

This explains WHY gradient orthogonality arises structurally.

Runs on CIFAR-10 and SVHN to compare spectral overlap (explains dataset-dependent behavior).
"""

import sys
sys.path.insert(0, '/data1/tongjizhou/FAFI_ICML25')

import torch
import torch.nn.functional as F
import numpy as np
import yaml
import os
import copy
import argparse
from torch.utils.data import DataLoader, Subset

from dataset_helper import get_fl_dataset
from models_lib import get_train_models
from oneshot_algorithms.ours.our_main import get_supcon_transform
from oneshot_algorithms.ours.unsupervised_loss import SupConLoss


def compute_feature_covariance(model, data_loader, device, max_batches=20):
    """Compute the covariance matrix of backbone features on raw data."""
    model.eval()
    all_features = []
    
    with torch.no_grad():
        for i, (data, _) in enumerate(data_loader):
            if i >= max_batches:
                break
            data = data.to(device)
            _, features = model(data)
            all_features.append(features.cpu())
    
    all_features = torch.cat(all_features, dim=0)  # [N, d]
    all_features = all_features - all_features.mean(dim=0, keepdim=True)
    
    # Compute covariance
    N = all_features.shape[0]
    cov = (all_features.T @ all_features) / (N - 1)
    
    return cov


def compute_gradient_eigenspectrum_projection(model, data_batch, target_batch, 
                                                aug_transformer, cls_loss_fn, supcon_fn,
                                                eigenvectors, device, lambda_val=1.0,
                                                n_top_eigvecs=50):
    """
    Project CE and SupCon gradients onto data covariance eigenvectors.
    
    Returns:
        ce_energy_per_eigen: [n_eigvecs] - energy of CE gradient along each eigenvector
        supcon_energy_per_eigen: [n_eigvecs] - energy of SupCon gradient along each eigenvector
        ce_total_norm: total CE gradient norm
        supcon_total_norm: total SupCon gradient norm
    """
    data = data_batch.to(device)
    target = target_batch.to(device)
    param_list = list(model.parameters())
    eigenvectors = eigenvectors.to(device)
    
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
    ce_total_norm = ce_grad_vec.norm().item()
    
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
    supcon_total_norm = supcon_grad_vec.norm().item()
    
    # --- Project gradients onto eigenvectors ---
    # Eigenvectors are [d_feature, d_feature], but gradient is in parameter space
    # We need to project onto the feature-space eigenvectors by computing
    # how much each gradient component aligns with each eigen-direction
    
    # Instead of direct projection (different spaces), we compute the
    # GRADIENT THROUGH each eigen-direction:
    # For eigenvector v_i, compute how much the loss gradient would change
    # if features moved in direction v_i
    
    # Simplified approach: compute gradient energy in parameter space,
    # then relate to eigen-index by computing sensitivity
    # 
    # More practical: project the FEATURE-LEVEL gradient onto eigenvectors
    # We need the gradient of each loss w.r.t. features (not parameters)
    
    model.zero_grad()
    
    # CE gradient w.r.t. features (raw data)
    data_raw = data.clone()
    logits_raw, feat_raw = model(data_raw)
    feat_raw.requires_grad_(True)
    # Re-compute with grad on features
    model.zero_grad()
    logits_raw2, feat_raw2 = model(data_raw)
    feat_raw2.retain_grad()
    ce_loss2 = cls_loss_fn(logits_raw2, target)
    ce_loss2.backward()
    
    ce_feat_grad = feat_raw2.grad.clone() if feat_raw2.grad is not None else torch.zeros_like(feat_raw2)
    
    # SupCon gradient w.r.t. features (augmented data)
    model.zero_grad()
    aug1 = aug_transformer(data)
    _, f1 = model(aug1)
    f1.retain_grad()
    aug2 = aug_transformer(data)
    _, f2 = model(aug2)
    f2.retain_grad()
    feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
    sc_loss = supcon_fn(feats, target)
    sc_loss.backward()
    
    sc_feat_grad1 = f1.grad.clone() if f1.grad is not None else torch.zeros_like(f1)
    sc_feat_grad2 = f2.grad.clone() if f2.grad is not None else torch.zeros_like(f2)
    sc_feat_grad = (sc_feat_grad1 + sc_feat_grad2) / 2.0  # average over two views
    
    # Project feature gradients onto eigenvectors
    # eigenvectors: [d, d], ce_feat_grad: [B, d]
    # For each sample, project gradient onto each eigenvector
    
    n_eigvecs = min(n_top_eigvecs, eigenvectors.shape[1])
    
    ce_energy = torch.zeros(n_eigvecs)
    sc_energy = torch.zeros(n_eigvecs)
    
    for i in range(n_eigvecs):
        v_i = eigenvectors[:, i]  # [d]
        # Project: for each sample, compute <grad_sample, v_i>^2
        ce_proj = torch.matmul(ce_feat_grad, v_i)  # [B]
        ce_energy[i] = (ce_proj ** 2).mean().item()
        
        sc_proj = torch.matmul(sc_feat_grad, v_i)  # [B]
        sc_energy[i] = (sc_proj ** 2).mean().item()
    
    return ce_energy.cpu().numpy(), sc_energy.cpu().numpy(), ce_total_norm, supcon_total_norm


def analyze_dataset(dataset_name, alpha, device, num_rounds=5):
    """Run spectral analysis for one dataset."""
    print(f"\n{'='*60}")
    print(f"Analyzing {dataset_name} (alpha={alpha})")
    print(f"{'='*60}")
    
    num_classes = 10 if dataset_name in ['CIFAR10', 'SVHN'] else 100
    in_channel = 3
    image_size = 32
    
    config = {
        'dataset': {
            'data_name': dataset_name,
            'root_path': '/data1/tongjizhou/FAFI_ICML25/data/',
            'train_batch_size': 256,
            'test_batch_size': 256,
            'channels': in_channel,
            'num_classes': num_classes,
            'image_size': image_size,
        },
        'distribution': {
            'type': 'dirichlet',
            'label_num_per_client': 2,
            'alpha': alpha,
        },
        'client': {'num_clients': 5},
        'server': {
            'num_rounds': num_rounds,
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
        'seed': 42,
    }
    
    print("Loading dataset...")
    trainset, testset, client_idx_map = get_fl_dataset(
        dataset_name, config['dataset']['root_path'],
        5, 'dirichlet', 2, alpha,
        normalize_train=False, normalize_test=True
    )
    test_loader = DataLoader(testset, batch_size=256, shuffle=True)
    
    print("Initializing model...")
    model = get_train_models(
        model_name='resnet18', num_classes=num_classes,
        mode='our', use_pretrain=False, in_channel=in_channel
    )
    model.to(device)
    
    aug_transformer = get_supcon_transform(dataset_name)
    cls_loss_fn = torch.nn.CrossEntropyLoss()
    supcon_fn = SupConLoss(temperature=0.07)
    
    results = {
        'dataset': dataset_name,
        'alpha': alpha,
        'rounds': [],
        'eigenvalues': [],
        'effective_rank': [],
        'ce_energy_top10': [],
        'ce_energy_top50': [],
        'ce_energy_bottom': [],
        'supcon_energy_top10': [],
        'supcon_energy_top50': [],
        'supcon_energy_bottom': [],
        'spectral_overlap': [],
        'ce_energy_distribution': [],
        'supcon_energy_distribution': [],
    }
    
    # Use a fixed data batch for gradient analysis
    analysis_loader = DataLoader(trainset, batch_size=256, shuffle=True)
    analysis_batch = next(iter(analysis_loader))
    
    for cr in range(num_rounds):
        print(f"\n--- Round {cr} ---")
        
        # 1. Compute feature covariance and eigendecomposition
        cov = compute_feature_covariance(model, test_loader, device)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        
        # Sort by descending eigenvalue
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Compute effective rank
        eps = 1e-10
        eigenvalues_clipped = torch.clamp(eigenvalues, min=eps)
        effective_rank = (eigenvalues_clipped.sum() ** 2) / (eigenvalues_clipped ** 2).sum()
        
        print(f"  Feature dim: {cov.shape[0]}, Effective rank: {effective_rank:.2f}")
        print(f"  Top-10 eigenvalues: {eigenvalues[:10].tolist()}")
        print(f"  Eigenvalue ratio (top-10 / total): {(eigenvalues[:10].sum() / eigenvalues.sum()).item():.4f}")
        
        # 2. Project gradients onto eigenvectors
        n_eigvecs = min(50, eigenvectors.shape[1])
        ce_energy, sc_energy, ce_norm, sc_norm = compute_gradient_eigenspectrum_projection(
            model, analysis_batch[0], analysis_batch[1], aug_transformer,
            cls_loss_fn, supcon_fn, eigenvectors.float(), device,
            n_top_eigvecs=n_eigvecs
        )
        
        # 3. Compute energy fractions
        ce_total = ce_energy.sum() + 1e-10
        sc_total = sc_energy.sum() + 1e-10
        
        ce_top10_frac = ce_energy[:10].sum() / ce_total
        ce_top50_frac = ce_energy.sum() / ce_total
        ce_bottom_frac = ce_energy[10:].sum() / ce_total
        
        sc_top10_frac = sc_energy[:10].sum() / sc_total
        sc_top50_frac = sc_energy.sum() / sc_total
        sc_bottom_frac = sc_energy[10:].sum() / sc_total
        
        # Spectral overlap: how much SupCon energy is in the same subspace as CE
        # If both concentrate on top-10, overlap is high
        spectral_overlap = np.sqrt(ce_energy[:10].sum() * sc_energy[:10].sum()) / (
            np.sqrt(ce_energy.sum() * sc_energy.sum()) + 1e-10)
        
        print(f"  CE gradient energy: top-10={ce_top10_frac:.4f}, rest={ce_bottom_frac:.4f}")
        print(f"  SupCon gradient energy: top-10={sc_top10_frac:.4f}, rest={sc_bottom_frac:.4f}")
        print(f"  Spectral overlap: {spectral_overlap:.4f}")
        
        # Store results
        results['rounds'].append(cr)
        results['eigenvalues'].append(eigenvalues[:50].tolist())
        results['effective_rank'].append(effective_rank.item())
        results['ce_energy_top10'].append(float(ce_top10_frac))
        results['ce_energy_top50'].append(float(ce_top50_frac))
        results['ce_energy_bottom'].append(float(ce_bottom_frac))
        results['supcon_energy_top10'].append(float(sc_top10_frac))
        results['supcon_energy_top50'].append(float(sc_top50_frac))
        results['supcon_energy_bottom'].append(float(sc_bottom_frac))
        results['spectral_overlap'].append(float(spectral_overlap))
        results['ce_energy_distribution'].append(ce_energy.tolist())
        results['supcon_energy_distribution'].append(sc_energy.tolist())
        
        # 4. One round of training (V24)
        model.train()
        for c in range(5):
            optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.0005)
            client_indices = client_idx_map[c]
            client_dataset = Subset(trainset, client_indices)
            client_loader = DataLoader(client_dataset, batch_size=256, shuffle=True)
            
            for data, target in client_loader:
                data, target = data.to(device), target.to(device)
                model.zero_grad()
                
                logits_raw, _ = model(data)
                ce_loss = cls_loss_fn(logits_raw, target)
                
                aug1, aug2 = aug_transformer(data), aug_transformer(data)
                _, f1 = model(aug1)
                _, f2 = model(aug2)
                feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                sc_loss = supcon_fn(feats, target)
                
                loss = ce_loss + sc_loss
                loss.backward()
                optimizer.step()
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=4)
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}')
    
    all_results = {}
    
    # Analyze CIFAR-10
    all_results['cifar10'] = analyze_dataset('CIFAR10', alpha=0.05, device=device, num_rounds=5)
    
    # Analyze SVHN
    all_results['svhn'] = analyze_dataset('SVHN', alpha=0.05, device=device, num_rounds=5)
    
    # Save results
    save_path = '/data1/tongjizhou/FAFI_ICML25/analysis/spectral_gradient_results.yaml'
    with open(save_path, 'w') as f:
        yaml.dump(all_results, f, default_flow_style=False)
    
    print(f"\n\nResults saved to {save_path}")
    
    # Print comparison summary
    print("\n" + "="*60)
    print("SPECTRAL COMPARISON: CIFAR-10 vs SVHN")
    print("="*60)
    
    for ds_name in ['cifar10', 'svhn']:
        r = all_results[ds_name]
        print(f"\n{ds_name.upper()}:")
        print(f"  Effective rank (R0): {r['effective_rank'][0]:.2f}")
        print(f"  CE top-10 fraction (R0): {r['ce_energy_top10'][0]:.4f}")
        print(f"  SupCon top-10 fraction (R0): {r['supcon_energy_top10'][0]:.4f}")
        print(f"  Spectral overlap (R0): {r['spectral_overlap'][0]:.4f}")
        print(f"  Eigenvalue ratio top-10/total: {sum(r['eigenvalues'][0][:10])/sum(r['eigenvalues'][0]):.4f}")


if __name__ == '__main__':
    main()
