from oneshot_algorithms.utils import init_optimizer, init_loss_fn, test_acc, save_best_local_model
from oneshot_algorithms.ours.unsupervised_loss import SupConLoss, Contrastive_proto_feature_loss, Contrastive_proto_loss, AlignmentUniformityLoss, AlignmentLoss

from common_libs import *
import torch.nn.functional as F



def ours_local_training(model, training_data, test_dataloader, start_epoch, local_epochs, optim_name, lr, momentum, loss_name, device, num_classes, sample_per_class, aug_transformer, client_model_dir, total_rounds, save_freq=1, use_drcl=False, fixed_anchors=None, lambda_align=1.0, use_progressive_alignment=False, initial_protos=None, use_uncertainty_weighting=False, sigma_lr=None, annealing_factor=1.0, use_dynamic_task_attenuation=False, gamma_reg=0, lambda_max=50.0, force_feature_alignment=False, use_reclassified_losses=False, warmup_epochs=0, use_confidence_gating=False, use_align_uniform=False, use_align_only=False, use_raw_ce_au=False, use_raw_ce_supcon=False, use_raw_ce_flat_supcon=False, use_aug_ce_flat_supcon=False, use_ce_only_raw=False, use_supcon_only_aug=False, use_aug_ce_aug_supcon=False, use_fafi=False, gate_power=2.0):
    
    model.train()
    model.to(device)

    if sigma_lr is None:
        sigma_lr = 0.05 * lr

    if use_uncertainty_weighting:
        sigma_params = [
            model.log_sigma_sq_local,
            model.log_sigma_sq_align
        ]
        sigma_param_ids = {id(p) for p in sigma_params}
        base_params = [p for p in model.parameters() if id(p) not in sigma_param_ids]
        param_groups = [
            {'params': base_params},
            {'params': sigma_params, 'lr': sigma_lr}
        ]
        if optim_name.lower() == 'sgd':
            optimizer = torch.optim.SGD(param_groups, lr=lr, momentum=momentum)
        else:
            logger.warning(f"Creating Adam optimizer for V10 with custom sigma_lr. Check if this is intended.")
            optimizer = torch.optim.Adam(param_groups, lr=lr)
        logger.info(f"V10 mode: Optimizer created with base_lr={lr} and sigma_lr={sigma_lr}")
    else:
        optimizer = init_optimizer(model, optim_name, lr, momentum)

    cls_loss_fn = torch.nn.CrossEntropyLoss()
    contrastive_loss_fn = SupConLoss(temperature=0.07)
    con_proto_feat_loss_fn = Contrastive_proto_feature_loss(temperature=1.0)
    con_proto_loss_fn = Contrastive_proto_loss(temperature=1.0)
    
    if use_align_uniform:
        align_uniform_fn = AlignmentUniformityLoss(alpha=2, t=2)
        logger.info("V20 mode: Using Alignment+Uniformity loss (no SupCon, no warmup needed)")
    
    if use_align_only:
        align_only_fn = AlignmentLoss(alpha=2)
        logger.info("V21 mode: Alignment-only (no uniformity), CE on raw data, consistency-gated")

    if use_raw_ce_au:
        raw_ce_au_fn = AlignmentUniformityLoss(alpha=2, t=2)
        logger.info("V22 mode: CE on raw data + consistency-gated AU on augmented views")

    if use_raw_ce_supcon:
        raw_ce_supcon_fn = SupConLoss(temperature=0.07)
        logger.info(f"V23 mode: CE on raw data + consistency^{gate_power}-gated SupCon on augmented views")

    if use_fafi:
        logger.info("FAFI mode: L_ssl (SupCon) + L_proto (Prototype contrastive), NO CE, NO ETF align")
    elif use_raw_ce_flat_supcon:
        raw_ce_flat_supcon_fn = SupConLoss(temperature=0.07)
        logger.info("V24 mode: CE on raw data + flat SupCon on augmented views (no gate)")

    if use_aug_ce_flat_supcon:
        aug_ce_flat_supcon_fn = SupConLoss(temperature=0.07)
        logger.info("Ablation mode: CE on augmented + flat SupCon on augmented views")
        logger.info("Ablation mode: CE on augmented data + flat SupCon on augmented views")

    if use_ce_only_raw:
        logger.info("Ablation mode: CE only on raw data")

    if use_supcon_only_aug:
        supcon_only_fn = SupConLoss(temperature=0.07)
        logger.info("Ablation mode: SupCon only on augmented views")

    if use_aug_ce_aug_supcon:
        aug_ce_supcon_fn = SupConLoss(temperature=0.07)
        logger.info("Ablation mode: CE on augmented + SupCon on augmented views (collapse test)")

    if use_drcl or use_progressive_alignment:
        alignment_loss_fn = torch.nn.MSELoss()

    initial_lambda = lambda_align
    total_training_steps = total_rounds * local_epochs
    warmup_end = start_epoch + warmup_epochs

    for e in range(start_epoch, start_epoch + local_epochs):
        total_loss = 0
        in_warmup = warmup_epochs > 0 and e < warmup_end
        epoch_consistencies = []

        for batch_idx, (data, target) in enumerate(training_data):
            data, target = data.to(device), target.to(device)

            if in_warmup:
                optimizer.zero_grad()
                logits, feature_norm = model(data)
                base_loss = cls_loss_fn(logits, target)
                loss = base_loss
            elif use_align_only:
                optimizer.zero_grad()
                
                logits_raw, _ = model(data)
                cls_loss = cls_loss_fn(logits_raw, target)
                
                aug_data1, aug_data2 = aug_transformer(data), aug_transformer(data)
                logits1, f1 = model(aug_data1)
                logits2, f2 = model(aug_data2)
                
                with torch.no_grad():
                    consistency = (logits1.argmax(dim=1) == logits2.argmax(dim=1)).float().mean().item()
                    epoch_consistencies.append(consistency)
                
                gated_alignment = consistency * align_only_fn(f1, f2)
                
                etf_align = torch.tensor(0.0, device=device)
                if use_drcl and fixed_anchors is not None:
                    unique_classes = torch.unique(target)
                    if len(unique_classes) > 0:
                        proto_subset = model.learnable_proto[unique_classes]
                        anchor_subset = fixed_anchors[unique_classes]
                        etf_align = alignment_loss_fn(proto_subset, anchor_subset)
                
                base_loss = cls_loss
                align_loss = gated_alignment + etf_align
            elif use_raw_ce_au:
                optimizer.zero_grad()

                logits_raw, _ = model(data)
                cls_loss = cls_loss_fn(logits_raw, target)

                aug_data1, aug_data2 = aug_transformer(data), aug_transformer(data)
                logits1, f1 = model(aug_data1)
                logits2, f2 = model(aug_data2)

                with torch.no_grad():
                    probs1 = F.softmax(logits1, dim=1)
                    probs2 = F.softmax(logits2, dim=1)
                    consistency = (probs1 * probs2).sum(dim=1).mean().item()
                    epoch_consistencies.append(consistency)

                gate = consistency ** gate_power
                au_loss = raw_ce_au_fn(f1, f2)

                loss = cls_loss + lambda_align * gate * au_loss
            elif use_raw_ce_supcon:
                optimizer.zero_grad()

                logits_raw, _ = model(data)
                cls_loss = cls_loss_fn(logits_raw, target)

                aug_data1, aug_data2 = aug_transformer(data), aug_transformer(data)
                logits1, f1 = model(aug_data1)
                logits2, f2 = model(aug_data2)

                with torch.no_grad():
                    probs1 = F.softmax(logits1, dim=1)
                    probs2 = F.softmax(logits2, dim=1)
                    consistency = (probs1 * probs2).sum(dim=1).mean().item()
                    epoch_consistencies.append(consistency)

                gate = consistency ** gate_power
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                supcon_loss = raw_ce_supcon_fn(features, target)

                loss = cls_loss + lambda_align * gate * supcon_loss
            elif use_ce_only_raw:
                # ── Ablation: CE only on raw data (no SupCon) ──
                optimizer.zero_grad()
                logits_raw, _ = model(data)
                loss = cls_loss_fn(logits_raw, target)
            elif use_supcon_only_aug:
                # ── Ablation: SupCon only on augmented views (no CE) ──
                optimizer.zero_grad()
                aug_data1, aug_data2 = aug_transformer(data), aug_transformer(data)
                _, f1 = model(aug_data1)
                _, f2 = model(aug_data2)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                loss = supcon_only_fn(features, target)
            elif use_aug_ce_aug_supcon:
                # ── Ablation: CE on augmented + SupCon on augmented (collapse test) ──
                optimizer.zero_grad()
                aug_data1, aug_data2 = aug_transformer(data), aug_transformer(data)
                logits1, f1 = model(aug_data1)
                logits2, f2 = model(aug_data2)
                cls_loss = (cls_loss_fn(logits1, target) + cls_loss_fn(logits2, target)) / 2.0
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                supcon_loss = aug_ce_supcon_fn(features, target)
                loss = cls_loss + lambda_align * supcon_loss
            elif use_raw_ce_flat_supcon:
                # ── V24: CE on raw + flat SupCon (no gate) ──
                optimizer.zero_grad()

                logits_raw, _ = model(data)
                cls_loss = cls_loss_fn(logits_raw, target)

                aug_data1, aug_data2 = aug_transformer(data), aug_transformer(data)
                _, f1 = model(aug_data1)
                _, f2 = model(aug_data2)

                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                supcon_loss = raw_ce_flat_supcon_fn(features, target)

                loss = cls_loss + lambda_align * supcon_loss
            elif use_aug_ce_flat_supcon:
                # ── Ablation: CE on augmented + flat SupCon on augmented ──
                optimizer.zero_grad()

                aug_data1, aug_data2 = aug_transformer(data), aug_transformer(data)

                logits1, f1 = model(aug_data1)
                logits2, f2 = model(aug_data2)

                # CE on both augmented views
                cls_loss = (cls_loss_fn(logits1, target) + cls_loss_fn(logits2, target)) / 2.0

                # SupCon on augmented view features
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                supcon_loss = aug_ce_flat_supcon_fn(features, target)

                loss = cls_loss + lambda_align * supcon_loss
            elif use_fafi:
                optimizer.zero_grad()
                aug_data1, aug_data2 = aug_transformer(data), aug_transformer(data)
                _, f1 = model(aug_data1)
                _, f2 = model(aug_data2)

                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                ssl_loss = contrastive_loss_fn(features, target)

                unique_classes = torch.unique(target)
                proto_loss = con_proto_feat_loss_fn(f1, model.learnable_proto, target, active_indices=unique_classes)

                loss = ssl_loss + proto_loss
            else:
                # Full AURORA pipeline with augmented data
                aug_data1, aug_data2 = aug_transformer(data), aug_transformer(data)
                aug_data = torch.cat([aug_data1, aug_data2], dim=0)
                bsz = target.shape[0]

                optimizer.zero_grad()
                logits, feature_norm = model(aug_data)

                if use_confidence_gating:
                    with torch.no_grad():
                        logits1, logits2 = torch.split(logits, [bsz, bsz], dim=0)
                        consistency = (logits1.argmax(dim=1) == logits2.argmax(dim=1)).float().mean().item()
                        epoch_consistencies.append(consistency)
                        gate = consistency ** 3
                else:
                    consistency = 1.0
                    gate = 1.0

                aug_labels = torch.cat([target, target], dim=0).to(device)
                cls_loss = cls_loss_fn(logits, aug_labels)

                f1, f2 = torch.split(feature_norm, [bsz, bsz], dim=0)
                
                if use_align_uniform:
                    align_uniform_loss = align_uniform_fn(f1, f2)
                    base_loss = cls_loss
                    reg_loss = align_uniform_loss
                else:
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    contrastive_loss = contrastive_loss_fn(features, target)

                    unique_classes = torch.unique(target)
                    pro_feat_con_loss = con_proto_feat_loss_fn(feature_norm, model.learnable_proto, aug_labels, active_indices=unique_classes)
                    pro_con_loss = con_proto_loss_fn(model.learnable_proto, active_indices=unique_classes)

                    gated_contrastive = gate * contrastive_loss
                    gated_pro_feat = gate * pro_feat_con_loss
                    gated_pro_con = gate * pro_con_loss

                    if use_reclassified_losses:
                        base_loss = cls_loss
                        reg_loss = gated_contrastive + gated_pro_con + gated_pro_feat
                    else:
                        base_loss = cls_loss + gated_contrastive + gated_pro_con + gated_pro_feat
                        reg_loss = 0

                align_loss = 0

                if use_progressive_alignment and initial_protos is not None and fixed_anchors is not None:
                    progress = (e - start_epoch) / local_epochs
                    target_anchor = (1 - progress) * initial_protos + progress * fixed_anchors
                    align_loss = alignment_loss_fn(model.learnable_proto, target_anchor)
                elif use_drcl and fixed_anchors is not None:
                    unique_classes = torch.unique(target)
                    if len(unique_classes) > 0:
                        if force_feature_alignment:
                            aug_targets = torch.cat([target, target], dim=0)
                            batch_anchors = fixed_anchors[aug_targets]
                            align_loss = alignment_loss_fn(feature_norm, batch_anchors)
                        else:
                            proto_subset = model.learnable_proto[unique_classes]
                            anchor_subset = fixed_anchors[unique_classes]
                            align_loss = alignment_loss_fn(proto_subset, anchor_subset)
                    else:
                        align_loss = 0

                if use_confidence_gating:
                    align_loss = gate * align_loss

                if use_align_uniform:
                    align_loss = 0.05 * reg_loss + align_loss
                elif use_reclassified_losses:
                    align_loss = reg_loss + align_loss

            if use_uncertainty_weighting and not in_warmup:
                sigma_sq_local = torch.exp(model.log_sigma_sq_local)
                sigma_sq_align = torch.exp(model.log_sigma_sq_align)

                if use_dynamic_task_attenuation:
                    current_step = e
                    progress = current_step / total_training_steps
                    schedule_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
                    schedule_factor = max(0.0, schedule_factor)
                    lambda_eff_for_reg = sigma_sq_local / sigma_sq_align
                    stability_reg = gamma_reg * torch.relu(lambda_eff_for_reg - lambda_max)
                    loss_sigma_main = (0.5 / sigma_sq_local) * base_loss.detach() + \
                                    (0.5 / sigma_sq_align) * align_loss.detach()
                    effective_lambda = (sigma_sq_local / sigma_sq_align).detach()
                    loss_for_weights = base_loss + effective_lambda * align_loss
                else:
                    schedule_factor = 1.0
                    stability_reg = 0
                    loss_sigma_main = (0.5 / sigma_sq_local) * base_loss.detach() + \
                                    (0.5 / sigma_sq_align) * align_loss.detach()
                    effective_lambda = (sigma_sq_local / sigma_sq_align).detach()
                    lambda_annealed = effective_lambda * annealing_factor
                    loss_for_weights = base_loss + lambda_annealed * align_loss

                loss_for_sigma_total = loss_sigma_main + \
                           0.5 * (torch.log(sigma_sq_local) + schedule_factor * torch.log(sigma_sq_align)) + \
                           stability_reg
                loss = loss_for_weights + loss_for_sigma_total

            elif use_drcl and not in_warmup and not use_align_only:
                global_progress = e / total_training_steps
                lambda_annealed = lambda_align * (1 - global_progress)
                loss = base_loss + lambda_annealed * align_loss

            elif use_align_only and not in_warmup:
                loss = base_loss + lambda_align * align_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        train_test_acc = test_acc(copy.deepcopy(model), test_dataloader, device, mode='etf')
        train_set_acc = test_acc(copy.deepcopy(model), training_data, device, mode='etf')

        phase_tag = " [WARMUP]" if in_warmup else ""
        consistency_info = ""
        if (use_confidence_gating or use_align_only or use_raw_ce_au or use_raw_ce_supcon) and epoch_consistencies:
            avg_cons = sum(epoch_consistencies) / len(epoch_consistencies)
            consistency_info = f" [cons={avg_cons:.4f}]"
        au_info = ""
        if use_align_uniform and not in_warmup:
            au_info = " [AU]"
        if use_align_only and not in_warmup:
            au_info = " [Align-Only]"
        if use_raw_ce_au and not in_warmup:
            au_info = " [RawCE+AU]"
        if use_raw_ce_supcon and not in_warmup:
            au_info = " [RawCE+SupCon]"
        if use_raw_ce_flat_supcon and not in_warmup:
            au_info = " [RawCE+FlatSupCon]"
        if use_aug_ce_flat_supcon and not in_warmup:
            au_info = " [AugCE+FlatSupCon]"
        if use_ce_only_raw and not in_warmup:
            au_info = " [CE-only-raw]"
        if use_supcon_only_aug and not in_warmup:
            au_info = " [SupCon-only-aug]"
        if use_aug_ce_aug_supcon and not in_warmup:
            au_info = " [AugCE+AugSupCon]"
        if use_fafi and not in_warmup:
            au_info = " [FAFI-Lssl+Lproto]"
        logger.info(f'Epoch {e}{phase_tag}{au_info}{consistency_info} loss: {total_loss}; train accuracy: {train_set_acc}; test accuracy: {train_test_acc}')

        if e % save_freq == 0:
            save_best_local_model(client_model_dir, model, f'epoch_{e}.pth', keep_only_last=True)

    return model
