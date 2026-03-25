import json
import math
import numpy as np
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path('analysis/figures')
OUT.mkdir(parents=True, exist_ok=True)

PLT_STYLE = dict(linewidth=2, markersize=6)
COLORS = {
    'AURORA':              '#2196F3',
    'AURORA w/o UncW':     '#FF9800',
    'FAFI (No Align)':     '#F44336',
    'FedETF+Ens':          '#4CAF50',
    'FedAvg':              '#795548',
    'FedETF':              '#8BC34A',
}

METHOD_MAP = {
    'OursV14+SimpleFeatureServer': 'AURORA',
    'OursV7+SimpleFeatureServer':  'AURORA w/o UncW',
    'OursV4+SimpleFeatureServer':  'FAFI (No Align)',
    'OneShotFedAvg':               'FedAvg',
    'OneshotFedETF+Ensemble':      'FedETF+Ens',
    'OneShotFedETF':               'FedETF',
}


def label(m):
    return METHOD_MAP.get(m, m)


def color(l):
    for k, v in COLORS.items():
        if k.lower() in l.lower():
            return v
    return '#607D8B'


def load(path='analysis/parsed_logs.json'):
    with open(path) as f:
        return json.load(f)


def fig1_ablation_bar(data):
    method_seeds = defaultdict(list)
    for fname, rec in data.items():
        if 'CIFAR100_a005' not in fname:
            continue
        for method, accs in rec['method_accs'].items():
            lbl = label(method)
            if accs:
                method_seeds[lbl].append(max(accs))

    order = ['FedAvg', 'FedETF+Ens', 'FAFI (No Align)', 'AURORA w/o UncW', 'AURORA']
    labels, means, stds = [], [], []
    for m in order:
        if m in method_seeds:
            vals = method_seeds[m]
            labels.append(m)
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=[color(l) for l in labels],
                  alpha=0.85, width=0.55, ecolor='black',
                  error_kw=dict(lw=1.5))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=15, ha='right')
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('CIFAR-100, alpha=0.05, K=5  (3 seeds, mean +/- std)', fontsize=11)
    ax.set_ylim(0, max(means) * 1.18)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + s + 0.005,
                f'{m:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / 'fig1_ablation_bar.pdf', dpi=150)
    fig.savefig(OUT / 'fig1_ablation_bar.png', dpi=150)
    plt.close(fig)
    print('fig1 saved:', list(zip(labels, [f"{m:.4f}+/-{s:.4f}" for m,s in zip(means,stds)])))
    return dict(zip(labels, zip(means, stds)))


def fig2_lambda_trajectory(data):
    target = [k for k in data if 'OursV14' in k and 'CIFAR100' in k]
    if not target:
        print('fig2: no V14 CIFAR100 logs')
        return

    rec    = data[target[0]]
    rounds = rec['rounds']

    all_raw = defaultdict(list)
    all_eff = defaultdict(list)
    all_sp  = []

    for rnd in rounds:
        r = rnd['round']
        sp_vals = []
        for cid_s, cobj in rnd['clients'].items():
            cid = int(cid_s)
            raw = cobj.get('raw_lambda')
            eff = cobj.get('eff_lambda')
            sp  = cobj.get('s_p')
            if raw is not None:
                all_raw[cid].append((r, raw))
            if eff is not None:
                all_eff[cid].append((r, eff))
            if sp is not None:
                sp_vals.append(sp)
        if sp_vals:
            all_sp.append((r, float(np.mean(sp_vals))))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    cmap = plt.get_cmap('tab10')

    ax0 = axes[0]
    for i, cid in enumerate(sorted(all_raw)):
        xs = [p[0] for p in all_raw[cid]]
        ys = [p[1] for p in all_raw[cid]]
        ax0.plot(xs, ys, marker='o', label=f'Client {cid}',
                 color=cmap(i), **PLT_STYLE)
    ax0.set_xlabel('Round', fontsize=11)
    ax0.set_ylabel('Raw lambda', fontsize=11)
    ax0.set_title('Raw lambda per Client (sigma_local^2 / sigma_align^2)', fontsize=10)
    ax0.legend(fontsize=8)
    ax0.grid(linestyle='--', alpha=0.4)
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)

    ax1 = axes[1]
    for i, cid in enumerate(sorted(all_eff)):
        xs = [p[0] for p in all_eff[cid]]
        ys = [p[1] for p in all_eff[cid]]
        ax1.plot(xs, ys, marker='s', label=f'Client {cid}',
                 color=cmap(i), **PLT_STYLE)
    if all_sp:
        ax1b = ax1.twinx()
        sx = [p[0] for p in all_sp]
        sy = [p[1] for p in all_sp]
        ax1b.plot(sx, sy, '--', color='gray', linewidth=1.5, label='s(p)')
        ax1b.set_ylabel('Attenuation s(p)', fontsize=9, color='gray')
        ax1b.tick_params(axis='y', labelcolor='gray')
        ax1b.set_ylim(0, 1.1)
        ax1b.legend(loc='upper right', fontsize=8)
    ax1.set_xlabel('Round', fontsize=11)
    ax1.set_ylabel('Effective lambda (Raw x s(p))', fontsize=11)
    ax1.set_title('Effective lambda: Meta-Annealing in Action', fontsize=10)
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(linestyle='--', alpha=0.4)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    fig.suptitle(f'AURORA lambda Trajectory -- {target[0]}', fontsize=9, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / 'fig2_lambda_trajectory.pdf', dpi=150, bbox_inches='tight')
    fig.savefig(OUT / 'fig2_lambda_trajectory.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('fig2 saved')


def fig3_proto_std(data):
    entries = []
    for fname, rec in data.items():
        if 'OursV14' not in fname:
            continue
        for rnd in rec['rounds']:
            g = rnd.get('g_proto_std')
            ic = rnd.get('inter_client_std')
            mv = rnd.get('model_var_mean')
            if any(v is not None for v in [g, ic, mv]):
                entries.append((rnd['round'], g, ic, mv, fname))

    if not entries:
        print('fig3: no prototype std data in logs')
        return

    entries.sort(key=lambda x: x[0])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax0, ax1 = axes

    for fname in set(e[4] for e in entries):
        sub = [(e[0], e[1], e[2], e[3]) for e in entries if e[4] == fname]
        sub.sort(key=lambda x: x[0])
        rs = [s[0] for s in sub]
        gs = [s[1] for s in sub if s[1] is not None]
        ic = [s[2] for s in sub if s[2] is not None]
        gr = [s[0] for s in sub if s[1] is not None]
        ir = [s[0] for s in sub if s[2] is not None]
        mv = [s[3] for s in sub if s[3] is not None]
        mr = [s[0] for s in sub if s[3] is not None]
        short = fname.replace('CIFAR100_a005_', '').replace('.log', '')
        if gs:
            ax0.plot(gr, gs, 'o-', label=f'{short} global', **PLT_STYLE)
        if ic:
            ax0.plot(ir, ic, 's--', label=f'{short} inter', **PLT_STYLE)
        if mv:
            ax1.plot(mr, mv, '^-', label=short, **PLT_STYLE)

    ax0.set_xlabel('Round', fontsize=11)
    ax0.set_ylabel('Prototype Std', fontsize=11)
    ax0.set_title('Prototype Geometry across Rounds', fontsize=11)
    ax0.legend(fontsize=8)
    ax0.grid(linestyle='--', alpha=0.4)
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)

    ax1.set_xlabel('Round', fontsize=11)
    ax1.set_ylabel('Model Param Variance (mean)', fontsize=11)
    ax1.set_title('Model Divergence across Rounds', fontsize=11)
    ax1.legend(fontsize=8)
    ax1.grid(linestyle='--', alpha=0.4)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / 'fig3_proto_std_model_var.pdf', dpi=150)
    fig.savefig(OUT / 'fig3_proto_std_model_var.png', dpi=150)
    plt.close(fig)
    print('fig3 saved')


def fig4_entropy_vs_lambda(data):
    entropies, lambdas = [], []
    for fname, rec in data.items():
        for ev in rec.get('entropy_events', []):
            entropies.append(ev['normalized'])
            lambdas.append(ev['adaptive_lambda'])

    if not entropies:
        print('fig4: no entropy events in logs')
        return

    entropies = np.array(entropies)
    lambdas   = np.array(lambdas)

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(entropies, lambdas, alpha=0.6, edgecolors='none', c=lambdas,
                    cmap='viridis', s=30)
    fig.colorbar(sc, ax=ax, label='Adaptive lambda')
    m, b = np.polyfit(entropies, lambdas, 1)
    xs = np.linspace(entropies.min(), entropies.max(), 200)
    ax.plot(xs, m * xs + b, 'r--', linewidth=1.5, label=f'Linear fit (slope={m:.3f})')
    ax.set_xlabel('Normalized Entropy', fontsize=11)
    ax.set_ylabel('Adaptive Lambda', fontsize=11)
    ax.set_title('Entropy vs Adaptive Lambda (all clients, all rounds)', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / 'fig4_entropy_vs_lambda.pdf', dpi=150)
    fig.savefig(OUT / 'fig4_entropy_vs_lambda.png', dpi=150)
    plt.close(fig)
    print(f'fig4 saved ({len(entropies)} points, slope={m:.4f})')


def fig5_loss_curves(data):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.get_cmap('tab10')
    idx  = 0
    for fname, rec in sorted(data.items()):
        rounds = rec['rounds']
        train_r, train_l = [], []
        test_r,  test_l  = [], []
        for rnd in rounds:
            r = rnd['round']
            tl = rnd.get('train_loss')
            vl = rnd.get('test_loss') or rnd.get('val_loss')
            if tl is not None:
                train_r.append(r)
                train_l.append(tl)
            if vl is not None:
                test_r.append(r)
                test_l.append(vl)
        short = fname.replace('.log', '')
        col   = cmap(idx % 10)
        if train_l:
            axes[0].plot(train_r, train_l, color=col, label=short, **PLT_STYLE)
        if test_l:
            axes[1].plot(test_r,  test_l,  color=col, label=short, **PLT_STYLE)
        idx += 1

    for ax, title, ylabel in zip(axes,
                                  ['Training Loss', 'Test / Val Loss'],
                                  ['Loss', 'Loss']):
        ax.set_xlabel('Round', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(linestyle='--', alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / 'fig5_loss_curves.pdf', dpi=150)
    fig.savefig(OUT / 'fig5_loss_curves.png', dpi=150)
    plt.close(fig)
    print('fig5 saved')


def fig6_accuracy_curves(data):
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap('tab10')
    idx  = 0
    for fname, rec in sorted(data.items()):
        rounds = rec['rounds']
        rs, accs = [], []
        for rnd in rounds:
            acc = rnd.get('test_acc') or rnd.get('accuracy')
            if acc is not None:
                rs.append(rnd['round'])
                accs.append(acc)
        if accs:
            short = fname.replace('.log', '')
            ax.plot(rs, accs, color=cmap(idx % 10), label=short, **PLT_STYLE)
            idx += 1

    ax.set_xlabel('Round', fontsize=11)
    ax.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax.set_title('Test Accuracy across Rounds', fontsize=11)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / 'fig6_accuracy_curves.pdf', dpi=150)
    fig.savefig(OUT / 'fig6_accuracy_curves.png', dpi=150)
    plt.close(fig)
    print('fig6 saved')


def fig7_communication_efficiency(data):
    labels, accs, rounds_to_target = [], [], []
    target_acc = 40.0
    for fname, rec in sorted(data.items()):
        best_acc = -1
        conv_round = None
        for rnd in rec['rounds']:
            acc = rnd.get('test_acc') or rnd.get('accuracy')
            if acc is not None:
                if acc > best_acc:
                    best_acc = acc
                if conv_round is None and acc >= target_acc:
                    conv_round = rnd['round']
        short = fname.replace('.log', '')
        labels.append(short)
        accs.append(best_acc if best_acc > 0 else float('nan'))
        rounds_to_target.append(conv_round if conv_round is not None else float('nan'))

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax0 = axes[0]
    bars = ax0.bar(x, accs, color='steelblue', edgecolor='white')
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
    ax0.set_ylabel('Best Test Accuracy (%)', fontsize=11)
    ax0.set_title('Peak Accuracy per Method', fontsize=11)
    for bar, v in zip(bars, accs):
        if not np.isnan(v):
            ax0.text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                     f'{v:.1f}', ha='center', va='bottom', fontsize=8)
    ax0.grid(axis='y', linestyle='--', alpha=0.4)
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)

    ax1 = axes[1]
    bars2 = ax1.bar(x, rounds_to_target, color='coral', edgecolor='white')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
    ax1.set_ylabel(f'Rounds to reach {target_acc}% accuracy', fontsize=10)
    ax1.set_title('Communication Efficiency', fontsize=11)
    for bar, v in zip(bars2, rounds_to_target):
        if not np.isnan(v):
            ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                     f'{int(v)}', ha='center', va='bottom', fontsize=8)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / 'fig7_comm_efficiency.pdf', dpi=150)
    fig.savefig(OUT / 'fig7_comm_efficiency.png', dpi=150)
    plt.close(fig)
    print('fig7 saved')


def fig8_heatmap_client_acc(data):
    target = [k for k in data if 'OursV14' in k and 'CIFAR100' in k]
    if not target:
        print('fig8: no V14 CIFAR100 data')
        return

    rec    = data[target[0]]
    rounds = rec['rounds']
    client_ids = set()
    for rnd in rounds:
        client_ids.update(int(c) for c in rnd['clients'])
    client_ids = sorted(client_ids)
    round_ids  = [rnd['round'] for rnd in rounds]

    matrix = np.full((len(client_ids), len(round_ids)), np.nan)
    cid_idx = {cid: i for i, cid in enumerate(client_ids)}
    for j, rnd in enumerate(rounds):
        for cid_s, cobj in rnd['clients'].items():
            cid = int(cid_s)
            acc = cobj.get('local_acc') or cobj.get('acc')
            if acc is not None:
                matrix[cid_idx[cid], j] = acc

    if np.all(np.isnan(matrix)):
        print('fig8: no per-client accuracy found')
        return

    fig, ax = plt.subplots(figsize=(max(8, len(round_ids) * 0.4),
                                    max(4, len(client_ids) * 0.35)))
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', interpolation='nearest')
    fig.colorbar(im, ax=ax, label='Local Accuracy (%)')
    ax.set_xlabel('Round', fontsize=11)
    ax.set_ylabel('Client ID', fontsize=11)
    ax.set_xticks(range(len(round_ids)))
    ax.set_xticklabels(round_ids, rotation=90, fontsize=6)
    ax.set_yticks(range(len(client_ids)))
    ax.set_yticklabels(client_ids, fontsize=7)
    ax.set_title(f'Per-Client Local Accuracy Heatmap -- {target[0]}', fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / 'fig8_client_acc_heatmap.pdf', dpi=150)
    fig.savefig(OUT / 'fig8_client_acc_heatmap.png', dpi=150)
    plt.close(fig)
    print('fig8 saved')


def main():
    print(f'Scanning log directory: {LOG_DIR}')
    data = load_logs(LOG_DIR)
    if not data:
        print('No log files found. Exiting.')
        return
    print(f'Loaded {len(data)} log file(s): {list(data.keys())}')
    OUT.mkdir(parents=True, exist_ok=True)

    fig1_ablation_bar(data)
    fig2_lambda_trajectory(data)
    fig3_proto_std(data)
    fig4_entropy_vs_lambda(data)
    fig5_loss_curves(data)
    fig6_accuracy_curves(data)
    fig7_communication_efficiency(data)
    fig8_heatmap_client_acc(data)

    print(f'\nAll figures saved to: {OUT.resolve()}')


if __name__ == '__main__':
    main()
