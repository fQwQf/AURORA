import json, sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis.parse_logs import parse_all

OUT = Path(__file__).parent / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

METHOD_MAP = {
    'OursV14+SimpleFeatureServer': 'AURORA',
    'OursV7+SimpleFeatureServer':  'AURORA w/o UncW',
    'OursV4+SimpleFeatureServer':  'FAFI (No ETF-Align)',
    'OneShotFedAvg':               'FedAvg',
    'OneshotFedETF+Ensemble':      'FedETF+Ensemble',
    'OneShotFedETF':               'FedETF',
}
COLOR_MAP = {
    'FedAvg':               '#795548',
    'FedETF+Ensemble':      '#4CAF50',
    'FAFI (No ETF-Align)':  '#F44336',
    'AURORA w/o UncW':      '#FF9800',
    'AURORA':               '#2196F3',
    'FedETF':               '#8BC34A',
}

def lbl(m):
    return METHOD_MAP.get(m, m)

def col(l):
    return COLOR_MAP.get(l, '#607D8B')


def fig1_ablation_bar(data):
    method_seeds = defaultdict(list)
    for fname, rec in data.items():
        if 'CIFAR100_a005' not in fname:
            continue
        for method, accs in rec['method_accs'].items():
            lb = lbl(method)
            if accs:
                method_seeds[lb].append(max(accs))

    order = ['FedAvg', 'FedETF+Ensemble', 'FAFI (No ETF-Align)', 'AURORA w/o UncW', 'AURORA']
    L, M, S = [], [], []
    for m in order:
        if m in method_seeds:
            vals = method_seeds[m]
            L.append(m); M.append(float(np.mean(vals))); S.append(float(np.std(vals)))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(L))
    bars = ax.bar(x, M, yerr=S, capsize=5, color=[col(l) for l in L],
                  alpha=0.85, width=0.55, ecolor='black', error_kw=dict(lw=1.5))
    ax.set_xticks(x)
    ax.set_xticklabels(L, fontsize=10, rotation=12, ha='right')
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Ablation Study — CIFAR-100, alpha=0.05, K=5  (3 seeds)', fontsize=11)
    ax.set_ylim(0, max(M) * 1.22)
    for bar, m, s in zip(bars, M, S):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.005,
                f'{m:.3f}+/-{s:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / 'fig1_ablation_bar.pdf', dpi=150)
    fig.savefig(OUT / 'fig1_ablation_bar.png', dpi=150)
    plt.close(fig)
    print('fig1 done:', list(zip(L, [f"{m:.4f}" for m in M])))


def fig2_lambda_trajectory(data):
    v14_logs = [k for k in data if 'OursV14' in k and 'CIFAR100' in k]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.get_cmap('tab10')

    round_raw_agg = defaultdict(list)
    round_eff_agg = defaultdict(list)
    round_sp_agg  = defaultdict(list)

    for seed_i, fname in enumerate(sorted(v14_logs)):
        rec = data[fname]
        all_raw = defaultdict(list)
        all_eff = defaultdict(list)
        for rnd in rec['rounds']:
            r = rnd['round']
            for cid_s, cobj in rnd['clients'].items():
                cid = int(cid_s)
                raw = cobj.get('raw_lambda')
                eff = cobj.get('eff_lambda')
                sp  = cobj.get('s_p')
                if raw is not None:
                    all_raw[cid].append((r, raw))
                    round_raw_agg[r].append(raw)
                if eff is not None:
                    all_eff[cid].append((r, eff))
                    round_eff_agg[r].append(eff)
                if sp is not None:
                    round_sp_agg[r].append(sp)

        seed_tag = fname.split('_seed')[1].replace('.log', '')
        for i, cid in enumerate(sorted(all_raw)):
            xs = [p[0] for p in all_raw[cid]]
            ys = [p[1] for p in all_raw[cid]]
            axes[0].plot(xs, ys, marker='o', linewidth=1.8, markersize=5,
                         color=cmap(i),
                         linestyle='-' if seed_i == 0 else ('--' if seed_i == 1 else ':'),
                         label=f'C{cid} seed={seed_tag}' if seed_i == 0 else '_')
            xs = [p[0] for p in all_eff[cid]]
            ys = [p[1] for p in all_eff[cid]]
            axes[1].plot(xs, ys, marker='s', linewidth=1.8, markersize=5,
                         color=cmap(i),
                         linestyle='-' if seed_i == 0 else ('--' if seed_i == 1 else ':'),
                         label=f'C{cid} seed={seed_tag}' if seed_i == 0 else '_')

    rs = sorted(round_sp_agg.keys())
    sp_means = [float(np.mean(round_sp_agg[r])) for r in rs]
    ax1b = axes[1].twinx()
    ax1b.plot(rs, sp_means, 'k--', linewidth=2, label='s(p) schedule', alpha=0.55)
    ax1b.set_ylabel('Attenuation s(p)', fontsize=9, color='gray')
    ax1b.tick_params(axis='y', labelcolor='gray')
    ax1b.set_ylim(0, 1.05)
    ax1b.legend(loc='upper right', fontsize=8)

    axes[0].set_xlabel('Round', fontsize=11)
    axes[0].set_ylabel('Raw lambda', fontsize=11)
    axes[0].set_title('(a) Raw lambda: sigma_local^2 / sigma_align^2', fontsize=10)
    axes[0].legend(fontsize=7, ncol=3)
    axes[0].grid(linestyle='--', alpha=0.4)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    axes[1].set_xlabel('Round', fontsize=11)
    axes[1].set_ylabel('Effective lambda', fontsize=11)
    axes[1].set_title('(b) Effective lambda = Raw lambda x s(p): Meta-Annealing in Action', fontsize=10)
    axes[1].legend(loc='upper left', fontsize=7, ncol=3)
    axes[1].grid(linestyle='--', alpha=0.4)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    fig.suptitle('AURORA Meta-Annealing: Lambda Trajectory (CIFAR-100, alpha=0.05, 3 seeds)', fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / 'fig2_lambda_trajectory.pdf', dpi=150, bbox_inches='tight')
    fig.savefig(OUT / 'fig2_lambda_trajectory.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('fig2 done')


def fig3_entropy_vs_lambda(data):
    entropies, lambdas, seed_labels = [], [], []
    for fname, rec in data.items():
        if 'OursV14' not in fname:
            continue
        seed = fname.split('_seed')[1].replace('.log', '')
        for ev in rec.get('entropy_events', []):
            entropies.append(ev['normalized'])
            lambdas.append(ev['adaptive_lambda'])
            seed_labels.append(seed)

    if not entropies:
        print('fig3 skipped: no entropy events in logs')
        return

    entropies = np.array(entropies)
    lambdas   = np.array(lambdas)
    fig, ax   = plt.subplots(figsize=(6.5, 5))

    seed_set = sorted(set(seed_labels))
    markers  = ['o', 's', '^']
    for i, s in enumerate(seed_set):
        mask = [x == s for x in seed_labels]
        ax.scatter(entropies[mask], lambdas[mask], s=55, alpha=0.75,
                   marker=markers[i % 3], label=f'seed {s}',
                   edgecolors='white', linewidth=0.5)

    m, b = np.polyfit(entropies, lambdas, 1)
    xs = np.linspace(entropies.min(), entropies.max(), 200)
    ax.plot(xs, m * xs + b, 'r--', linewidth=1.8, label=f'Linear fit (slope={m:.2f})')

    ax.set_xlabel('Normalized Data Entropy  H(p)/H_max  (lower = more heterogeneous)', fontsize=10)
    ax.set_ylabel('Adaptive Initial lambda', fontsize=12)
    ax.set_title('Lower Entropy -> Higher Initial Alignment Strength\n(AURORA Adaptive Init)', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / 'fig3_entropy_vs_lambda.pdf', dpi=150)
    fig.savefig(OUT / 'fig3_entropy_vs_lambda.png', dpi=150)
    plt.close(fig)
    print(f'fig3 done ({len(entropies)} points, slope={m:.4f}, range=[{lambdas.min():.2f},{lambdas.max():.2f}])')


def fig4_round_acc_curves(data):
    method_round_accs = defaultdict(lambda: defaultdict(list))
    for fname, rec in data.items():
        if 'CIFAR100_a005' not in fname:
            continue
        for rnd in rec['rounds']:
            r = rnd['round']
            for mname, acc in rnd['final_accs'].items():
                method_round_accs[lbl(mname)][r].append(acc)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.get_cmap('Set1')
    order = ['FedAvg', 'FedETF+Ensemble', 'FAFI (No ETF-Align)', 'AURORA w/o UncW', 'AURORA']
    plotted = 0
    for mname in order:
        if mname not in method_round_accs:
            continue
        rd = method_round_accs[mname]
        rs = sorted(rd.keys())
        means = [float(np.mean(rd[r])) for r in rs]
        stds  = [float(np.std(rd[r]))  for r in rs]
        c = col(mname)
        ax.plot(rs, means, 'o-', color=c, linewidth=2, markersize=5, label=mname)
        ax.fill_between(rs,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.12, color=c)
        plotted += 1

    ax.set_xlabel('Communication Round', fontsize=12)
    ax.set_ylabel('Test Accuracy (global ensemble)', fontsize=12)
    ax.set_title('Round-Level Accuracy Curves (CIFAR-100, alpha=0.05, 3 seeds mean+/-std)', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / 'fig4_round_acc_curves.pdf', dpi=150)
    fig.savefig(OUT / 'fig4_round_acc_curves.png', dpi=150)
    plt.close(fig)
    print(f'fig4 done ({plotted} methods plotted)')


def fig5_lambda_decay_aggregated(data):
    round_raw_agg = defaultdict(list)
    round_eff_agg = defaultdict(list)
    round_sp_agg  = defaultdict(list)
    for fname, rec in data.items():
        if 'OursV14' not in fname:
            continue
        for rnd in rec['rounds']:
            r = rnd['round']
            for cid_s, cobj in rnd['clients'].items():
                if cobj.get('raw_lambda') is not None:
                    round_raw_agg[r].append(cobj['raw_lambda'])
                if cobj.get('eff_lambda') is not None:
                    round_eff_agg[r].append(cobj['eff_lambda'])
                if cobj.get('s_p') is not None:
                    round_sp_agg[r].append(cobj['s_p'])

    if not round_raw_agg:
        print('fig5 skipped')
        return

    rs        = sorted(round_raw_agg.keys())
    raw_means = [float(np.mean(round_raw_agg[r])) for r in rs]
    raw_stds  = [float(np.std(round_raw_agg[r]))  for r in rs]
    eff_means = [float(np.mean(round_eff_agg[r])) for r in rs]
    eff_stds  = [float(np.std(round_eff_agg[r]))  for r in rs]
    sp_means  = [float(np.mean(round_sp_agg[r]))  for r in rs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rs, raw_means, 'o-', color='#FF5722', linewidth=2.5, markersize=6, label='Raw lambda')
    ax.fill_between(rs,
                    [m - s for m, s in zip(raw_means, raw_stds)],
                    [m + s for m, s in zip(raw_means, raw_stds)],
                    alpha=0.15, color='#FF5722')
    ax.plot(rs, eff_means, 's--', color='#2196F3', linewidth=2.5, markersize=6, label='Effective lambda')
    ax.fill_between(rs,
                    [m - s for m, s in zip(eff_means, eff_stds)],
                    [m + s for m, s in zip(eff_means, eff_stds)],
                    alpha=0.15, color='#2196F3')

    ax2 = ax.twinx()
    ax2.plot(rs, sp_means, 'g:', linewidth=2, alpha=0.7, label='s(p)')
    ax2.set_ylabel('Alignment Pressure s(p)', fontsize=11, color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylim(0, 1.05)

    ax.set_xlabel('Communication Round', fontsize=12)
    ax.set_ylabel('Lambda Value (mean +/- std across clients and seeds)', fontsize=10)
    ax.set_title('AURORA: Raw vs Effective Lambda Decay (3 seeds x 5 clients)', fontsize=10)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=9)
    ax.grid(linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / 'fig5_lambda_decay.pdf', dpi=150)
    fig.savefig(OUT / 'fig5_lambda_decay.png', dpi=150)
    plt.close(fig)
    print('fig5 done')


def fig6_per_epoch_acc(data):
    method_epoch = defaultdict(lambda: defaultdict(list))
    for fname, rec in data.items():
        if 'CIFAR100_a005' not in fname:
            continue
        method_lbl = None
        for mname in rec['method_accs']:
            method_lbl = lbl(mname)
            break
        if method_lbl is None:
            continue
        for rnd in rec['rounds']:
            for cid_s, cobj in rnd['clients'].items():
                for ep in cobj.get('epochs', []):
                    ep_idx = ep['epoch']
                    acc    = ep.get('train_acc', None)
                    if acc is not None:
                        method_epoch[method_lbl][ep_idx].append(acc)

    if not method_epoch:
        print('fig6 skipped (no per-epoch data)')
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    order = ['FedAvg', 'FedETF+Ensemble', 'FAFI (No ETF-Align)', 'AURORA w/o UncW', 'AURORA']
    plotted = 0
    for mname in order:
        if mname not in method_epoch:
            continue
        ed = method_epoch[mname]
        eps = sorted(ed.keys())
        means = [float(np.mean(ed[e])) for e in eps]
        stds  = [float(np.std(ed[e]))  for e in eps]
        c = col(mname)
        ax.plot(eps, means, '-', color=c, linewidth=2, label=mname)
        ax.fill_between(eps,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.12, color=c)
        plotted += 1

    ax.set_xlabel('Local Epoch Index', fontsize=12)
    ax.set_ylabel('Train Accuracy (mean across clients and seeds)', fontsize=10)
    ax.set_title('Per-Epoch Training Accuracy (CIFAR-100, alpha=0.05)', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / 'fig6_per_epoch_acc.pdf', dpi=150)
    fig.savefig(OUT / 'fig6_per_epoch_acc.png', dpi=150)
    plt.close(fig)
    print(f'fig6 done ({plotted} methods plotted)')


def main():
    data = load_all_json(ROOT)
    if not data:
        print('No JSON files found under', ROOT)
        return
    print(f'Loaded {len(data)} result files')
    OUT.mkdir(parents=True, exist_ok=True)
    fig1_main_bar(data)
    fig2_heterogeneity_sweep(data)
    fig3_entropy_vs_lambda(data)
    fig4_round_acc_curves(data)
    fig5_lambda_decay_aggregated(data)
    fig6_per_epoch_acc(data)
    print('All figures saved to', OUT)


if __name__ == '__main__':
    main()
