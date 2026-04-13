"""
Linear Probe for SupCon-Only Ablation.

The SupCon-only ablation gets 10% (random chance) because:
1. The classifier head (learnable_proto) is never trained with CE loss
2. The features may or may not have collapsed

This script:
1. Loads SupCon-only trained models (per-client)
2. Freezes the backbone
3. Retrains ONLY the prototypes with CE loss (linear probe)
4. Evaluates to determine if features are informative

If linear probe gets >>10%: features are good but prototypes weren't trained
If linear probe gets ~10%: features truly collapsed
"""

import sys
sys.path.insert(0, '/data1/tongjizhou/FAFI_ICML25')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy
import argparse
from collections import defaultdict

from dataset_helper import get_fl_dataset
from models_lib import get_train_models


def linear_probe(model, train_loader, test_loader, device, num_epochs=50, lr=0.1):
    """
    Freeze backbone, retrain prototypes with CE loss.
    Returns test accuracy.
    """
    model.eval()
    
    # Freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    # Only optimize learnable_proto
    optimizer = torch.optim.SGD([model.learnable_proto], lr=lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    cls_loss_fn = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        # Keep encoder frozen
        model.encoder.eval()
        
        total_loss = 0
        total_samples = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            logits, features = model(data)
            loss = cls_loss_fn(logits, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
        
        scheduler.step()
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                logits, _ = model(data)
                pred = logits.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}: loss={total_loss/total_samples:.4f}, acc={acc:.4f}, best={best_acc:.4f}")
    
    return best_acc


def feature_quality_analysis(model, train_loader, test_loader, device, num_classes=10):
    """
    Analyze feature quality: intra-class compactness, inter-class separation.
    """
    model.eval()
    
    # Collect features by class
    class_features = defaultdict(list)
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            _, features = model(data)
            for i in range(data.size(0)):
                class_features[target[i].item()].append(features[i].cpu())
    
    # Compute centroids and metrics
    centroids = {}
    for c in range(num_classes):
        if c in class_features and len(class_features[c]) > 0:
            feats = torch.stack(class_features[c])
            centroids[c] = feats.mean(dim=0)
    
    # Intra-class variance
    intra_vars = []
    for c in range(num_classes):
        if c in class_features and len(class_features[c]) > 0:
            feats = torch.stack(class_features[c])
            dist = ((feats - centroids[c]) ** 2).sum(dim=1).mean()
            intra_vars.append(dist.item())
    
    # Inter-class distances
    inter_dists = []
    class_list = sorted(centroids.keys())
    for i in range(len(class_list)):
        for j in range(i+1, len(class_list)):
            dist = ((centroids[class_list[i]] - centroids[class_list[j]]) ** 2).sum()
            inter_dists.append(dist.item())
    
    avg_intra = np.mean(intra_vars) if intra_vars else float('inf')
    avg_inter = np.mean(inter_dists) if inter_dists else 0.0
    
    # Feature collapse check: are all features the same?
    all_centroids = torch.stack([centroids[c] for c in class_list])
    centroid_var = all_centroids.var(dim=0).mean().item()
    
    # Cosine similarity between centroids
    cos_sims = []
    for i in range(len(class_list)):
        for j in range(i+1, len(class_list)):
            cos = F.cosine_similarity(centroids[class_list[i]].unsqueeze(0), 
                                       centroids[class_list[j]].unsqueeze(0)).item()
            cos_sims.append(cos)
    
    avg_cos_sim = np.mean(cos_sims) if cos_sims else 0.0
    
    return {
        'avg_intra_class_var': avg_intra,
        'avg_inter_class_dist': avg_inter,
        'centroid_variance': centroid_var,
        'avg_cosine_sim_between_centroids': avg_cos_sim,
        'num_classes_with_features': len(centroids),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=7)
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}')
    
    datasets = [
        {
            'name': 'CIFAR10',
            'num_classes': 10,
            'checkpoint_dir': '/data1/tongjizhou/FAFI_ICML25/checkpoints/neurips_ablation/supcon_only_cifar10/Ablation_SupCon_Only_CIFAR10/local_models',
            'alpha': 0.05,
        },
        {
            'name': 'SVHN', 
            'num_classes': 10,
            'checkpoint_dir': '/data1/tongjizhou/FAFI_ICML25/checkpoints/neurips_ablation/supcon_only_svhn/Ablation_SupCon_Only_SVHN/local_models',
            'alpha': 0.05,
        },
    ]
    
    for ds_config in datasets:
        ds_name = ds_config['name']
        num_classes = ds_config['num_classes']
        
        print(f"\n{'='*60}")
        print(f"Linear Probe: {ds_name} (SupCon-only)")
        print(f"{'='*60}")
        
        # Load dataset
        trainset, testset, client_idx_map = get_fl_dataset(
            ds_name, '/data1/tongjizhou/FAFI_ICML25/data/',
            5, 'dirichlet', 2, ds_config['alpha'],
            normalize_train=False, normalize_test=True
        )
        test_loader = DataLoader(testset, batch_size=256, shuffle=False)
        train_loader = DataLoader(trainset, batch_size=256, shuffle=True)
        
        # Load each client's model and do linear probe
        all_results = {}
        
        for client_id in range(5):
            ckpt_path = f"{ds_config['checkpoint_dir']}/client_{client_id}/epoch_19.pth"
            print(f"\n--- Client {client_id} ---")
            
            # Load model
            model = get_train_models(
                model_name='resnet18', num_classes=num_classes,
                mode='our', use_pretrain=False, in_channel=3
            )
            
            state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            model.load_state_dict(state_dict)
            model.to(device)
            
            # Feature quality analysis BEFORE linear probe
            print("Feature quality (before probe):")
            quality = feature_quality_analysis(model, train_loader, test_loader, device, num_classes)
            for k, v in quality.items():
                print(f"  {k}: {v:.6f}")
            
            # Linear probe
            print("Running linear probe...")
            probe_acc = linear_probe(model, train_loader, test_loader, device, 
                                      num_epochs=50, lr=0.1)
            print(f"Linear probe accuracy: {probe_acc:.4f}")
            
            all_results[f'client_{client_id}'] = {
                'probe_accuracy': probe_acc,
                'feature_quality': quality,
            }
        
        # Summary
        print(f"\n{'='*60}")
        print(f"SUMMARY: {ds_name}")
        print(f"{'='*60}")
        for client_id in range(5):
            r = all_results[f'client_{client_id}']
            q = r['feature_quality']
            print(f"  Client {client_id}: probe_acc={r['probe_accuracy']:.4f}, "
                  f"centroid_var={q['centroid_variance']:.6f}, "
                  f"avg_cos_sim={q['avg_cosine_sim_between_centroids']:.6f}")
        
        avg_probe = np.mean([all_results[f'client_{c}']['probe_accuracy'] for c in range(5)])
        print(f"\n  Average linear probe accuracy: {avg_probe:.4f}")


if __name__ == '__main__':
    main()
