import sys, json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis.parse_logs import parse_all

OUT = Path(__file__).parent / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

LOG_DIR = Path(__file__).parent.parent / 'logs' / 'multi_seed'

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
}
ORDER = ['FedAvg', 'FedETF+Ensemble', 'FAFI (No ETF-Align)', 'AURORA w/o UncW', 'AURORA']

def lbl(m): return METHOD_MAP.get(m, m)
def col(l): return COLOR_MAP.get(l, '#607D8B')


def fig1_ablation_bar(data):
    method_seeds = defaultdict(list)
    for fname, rec in data.items():
        if 'CIFAR100_a005' not in fname:
            continue
        for method, accs in rec['method_accs'].items():
            lb = lbl(method)
            if accs:
                method_seeds[lb].append(max(accs))

    L, M, S = [], [], []
    for m in ORDER:
        if m in method_seeds:
            vals = method_seeds[m]
            L.append(m)
            M.append(float(np.mean(vals)))
            S.append(float(np.std(vals)))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(L))
    bars = ax.bar(x, M, yerr=S, capsize=5, color=[col(l) for l in L],
                  alpha=0.85, width=0.55, ecolor='black', error_kw=dict(lw=1.5))
    ax.set_xticks(x)
    ax.set_xticklabels(L, fontsize=10, rotation=12, ha='right')
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Ablation Study — CIFAR-100, alpha=0.05, K=5 (3 seeds)', fontsize=11)
    ax.set_ylim(0, max(M) * 1.22)
    for bar, m, s in zip(bars, M, S):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + 0.005,
                f'{m:.3f}+/-{s:.3f}',
                ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / 'fig1_ablation_bar.png', dpi=150)
    fig.savefig(OUT / 'fig1_ablation_bar.pdf', dpi=150)
    plt.close(fig)
    print('fig1 done:', list(zip(L, [f"{m:.4f}+/-{s:.4f}" for m, s in zip(M, S)])))
    return dict(zip(L, zip(M, S)))


def fig2_lambda_trajectory(data):
    per_client_raw = defaultdict(lambda: defaultdict(list))
    per_client_eff = defaultdict(lambda: defaultdict(list))
    round_sp = defaultdict(list)

    for fname, rec in data.items():
        if 'OursV14' not in fname or 'CIFAR100' not in fname:
            continue
        for rnd in rec['rounds']:
            r = rnd['round']
            for cid_s, cobj in rnd['clients'].items():
                cid = int(cid_s)
                if cobj.get('raw_lambda') is not None:
                    per_client_raw[cid][r].append(cobj['raw_lambda'])
                if cobj.get('eff_lambda') is not None:
                    per_client_eff[cid][r].append(cobj['eff_lambda'])
                if cobj.get('s_p') is not None:
                    round_sp[r].append(cobj['s_p'])

    if not per_client_raw:
        print('fig2 skipped: no lambda data')
        return

    rs = sorted(round_sp.keys())
    sp_means = [float(np.mean(round_sp[r])) for r in rs]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.get_cmap('tab10')

    ax0 = axes[0]
    for i, cid in enumerate(sorted(per_client_raw)):
        cr = sorted(per_client_raw[cid].keys())
        cm = [float(np.mean(per_client_raw[cid][r])) for r in cr]
        cs = [float(np.std(per_client_raw[cid][r])) for r in cr]
        ax0.plot(cr, cm, 'o-', color=cmap(i), linewidth=2, markersize=5, label=f'Client {cid}')
        ax0.fill_between(cr,
                         [m - s for m, s in zip(cm, cs)],
                         [m + s for m, s in zip(cm, cs)],
                         alpha=0.12, color=cmap(i))
    ax0.set_xlabel('Round', fontsize=11)
    ax0.set_ylabel('Raw lambda', fontsize=11)
    ax0.set_title('(a) Raw lambda per client (mean+/-std over 3 seeds)', fontsize=10)
    ax0.legend(fontsize=8)
    ax0.grid(linestyle='--', alpha=0.4)
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)

    ax1 = axes[1]
    for i, cid in enumerate(sorted(per_client_eff)):
        cr = sorted(per_client_eff[cid].keys())
        cm = [float(np.mean(per_client_eff[cid][r])) for r in cr]
        cs = [float(np.std(per_client_eff[cid][r])) for r in cr]
        ax1.plot(cr, cm, 's--', color=cmap(i), linewidth=2, markersize=5, label=f'Client {cid}')
        ax1.fill_between(cr,
                         [m - s for m, s in zip(cm, cs)],
                         [m + s for m, s in zip(cm, cs)],
                         alpha=0.12, color=cmap(i))
    ax1b = ax1.twinx()
    ax1b.plot(rs, sp_means, 'k--', linewidth=2, alpha=0.5, label='s(p)')
    ax1b.set_ylabel('s(p)', fontsize=9, color='gray')
    ax1b.tick_params(axis='y', labelcolor='gray')
    ax1b.set_ylim(0, 1.05)
    ax1b.legend(loc='upper right', fontsize=8)
    ax1.set_xlabel('Round', fontsize=11)
    ax1.set_ylabel('Effective lambda', fontsize=11)
    ax1.set_title('(b) Effective lambda = Raw x s(p): Meta-Annealing', fontsize=10)
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(linestyle='--', alpha=0.4)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    fig.suptitle('AURORA Lambda Trajectory — CIFAR-100 alpha=0.05, 3 seeds', fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / 'fig2_lambda_trajectory.png', dpi=150, bbox_inches='tight')
    fig.savefig(OUT / 'fig2_lambda_trajectory.pdf', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('fig2 done')


def fig3_entropy_vs_lambda(data):
    entropies, lambdas, slabels = [], [], []
    for fname, rec in data.items():
        if 'OursV14' not in fname:
            continue
        seed = fname.split('_seed')[1].replace('.log', '') if '_seed' in fname else 'unknown'
        for ev in rec.get('entropy_events', []):
            entropies.append(ev['normalized'])
            lambdas.append(ev['adaptive_lambda'])
            slabels.append(seed)

    if not entropies:
        print('fig3 skipped: no entropy events')
        return

    en = np.array(entropies)
    la = np.array(lambdas)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for i, s in enumerate(sorted(set(slabels))):
        mask = np.array([x == s for x in slabels])
        ax.scatter(en[mask], la[mask], s=55, alpha=0.75,
                   marker=['o', 's', '^'][i % 3], label=f'seed {s}',
                   edgecolors='white', linewidth=0.5)
    m, b = np.polyfit(en, la, 1)
    xs = np.linspace(en.min(), en.max(), 200)
    ax.plot(xs, m * xs + b, 'r--', linewidth=1.8, label=f'Linear fit (slope={m:.2f})')
    ax.set_xlabel('Normalized Entropy (lower = more heterogeneous)', fontsize=10)
    ax.set_ylabel('Adaptive Initial lambda', fontsize=12)
    ax.set_title('Lower Entropy -> Higher Alignment Strength\n(AURORA Adaptive Init Validation)', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / 'fig3_entropy_vs_lambda.png', dpi=150)
    fig.savefig(OUT / 'fig3_entropy_vs_lambda.pdf', dpi=150)
    plt.close(fig)
    print(f'fig3 done ({len(en)} pts, slope={m:.4f})')


def fig4_round_acc_curves(data):
    method_round = defaultdict(lambda: defaultdict(list))
    for fname, rec in data.items():
        if 'CIFAR100_a005' not in fname:
            continue
        for rnd in rec['rounds']:
            r = rnd['round']
            for mname, acc in rnd['final_accs'].items():
                method_round[lbl(mname)][r].append(acc)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    plotted = 0
    for mname in ORDER:
        if mname not in method_round:
            continue
        rd = method_round[mname]
        rr = sorted(rd.keys())
        ms = [float(np.mean(rd[r])) for r in rr]
        ss = [float(np.std(rd[r])) for r in rr]
        c = col(mname)
        ax.plot(rr, ms, 'o-', color=c, linewidth=2, markersize=5, label=mname)
        ax.fill_between(rr,
                        [m - s for m, s in zip(ms, ss)],
                        [m + s for m, s in zip(ms, ss)],
                        alpha=0.12, color=c)
        plotted += 1

    ax.set_xlabel('Communication Round', fontsize=12)
    ax.set_ylabel('Test Accuracy (global ensemble)', fontsize=12)
    ax.set_title('Round-Level Accuracy Curves — CIFAR-100 alpha=0.05 (3 seeds)', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / 'fig4_round_acc_curves.png', dpi=150)
    fig.savefig(OUT / 'fig4_round_acc_curves.pdf', dpi=150)
    plt.close(fig)
    print(f'fig4 done ({plotted} methods)')


def fig5_lambda_decay_aggregated(data):
    round_raw = defaultdict(list)
    round_eff = defaultdict(list)
    round_sp = defaultdict(list)
    for fname, rec in data.items():
        if 'OursV14' not in fname:
            continue
        for rnd in rec['rounds']:
            r = rnd['round']
            for cid_s, cobj in rnd['clients'].items():
                if cobj.get('raw_lambda') is not None:
                    round_raw[r].append(cobj['raw_lambda'])
                if cobj.get('eff_lambda') is not None:
                    round_eff[r].append(cobj['eff_lambda'])
                if cobj.get('s_p') is not None:
                    round_sp[r].append(cobj['s_p'])

    if not round_raw:
        print('fig5 skipped')
        return

    rs = sorted(round_raw.keys())
    raw_m = [float(np.mean(round_raw[r])) for r in rs]
    raw_s = [float(np.std(round_raw[r])) for r in rs]
    eff_m = [float(np.mean(round_eff[r])) for r in rs]
    eff_s = [float(np.std(round_eff[r])) for r in rs]
    sp_m  = [float(np.mean(round_sp[r]))  for r in rs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rs, raw_m, 'o-', color='#FF5722', linewidth=2.5, markersize=6, label='Raw lambda')
    ax.fill_between(rs,
                    [m - s for m, s in zip(raw_m, raw_s)],
                    [m + s for m, s in zip(raw_m, raw_s)],
                    alpha=0.15, color='#FF5722')
    ax.plot(rs, eff_m, 's--', color='#2196F3', linewidth=2.5, markersize=6, label='Effective lambda')
    ax.fill_between(rs,
                    [m - s for m, s in zip(eff_m, eff_s)],
                    [m + s for m, s in zip(eff_m, eff_s)],
                    alpha=0.15, color='#2196F3')
    ax2 = ax.twinx()
    ax2.plot(rs, sp_m, 'g:', linewidth=2, alpha=0.7, label='s(p)')
    ax2.set_ylabel('s(p)', fontsize=11, color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylim(0, 1.05)
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Lambda (mean+/-std, 3 seeds x 5 clients)', fontsize=10)
    ax.set_title('Raw vs Effective Lambda Decay — AURORA (CIFAR-100, alpha=0.05)', fontsize=10)
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, fontsize=9)
    ax.grid(linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / 'fig5_lambda_decay.png', dpi=150)
    fig.savefig(OUT / 'fig5_lambda_decay.pdf', dpi=150)
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
        if not method_lbl:
            continue
        for rnd in rec['rounds']:
            for cid_s, cobj in rnd['clients'].items():
                for ep in cobj.get('epochs', []):
                    if ep.get('train_acc') is not None:
                        method_epoch[method_lbl][ep['epoch']].append(ep['train_acc'])

    if not method_epoch:
        print('fig6 skipped (no per-epoch data)')
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for mname in ORDER:
        if mname not in method_epoch:
            continue
        ed = method_epoch[mname]
        eps = sorted(ed.keys())
        ms = [float(np.mean(ed[e])) for e in eps]
        ss = [float(np.std(ed[e])) for e in eps]
        c = col(mname)
        ax.plot(eps, ms, '-', color=c, linewidth=2, label=mname)
        ax.fill_between(eps,
                        [m - s for m, s in zip(ms, ss)],
                        [m + s for m, s in zip(ms, ss)],
                        alpha=0.12, color=c)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Train Accuracy', fontsize=12)
    ax.set_title('Per-Epoch Training Accuracy per Client (CIFAR-100, alpha=0.05)', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / 'fig6_per_epoch_acc.png', dpi=150)
    fig.savefig(OUT / 'fig6_per_epoch_acc.pdf', dpi=150)
    plt.close(fig)
    print('fig6 done')


def summary_table(data):
    method_seeds = defaultdict(list)
    for fname, rec in data.items():
        if 'CIFAR100_a005' not in fname:
            continue
        for method, accs in rec['method_accs'].items():
            lb = lbl(method)
            if accs:
                method_seeds[lb].append(max(accs))
    print('\n' + '=' * 60)
    print('SUMMARY TABLE — CIFAR-100, alpha=0.05, K=5 (3 seeds)')
    print('=' * 60)
    print(f'{"Method":<30} {"Mean":>8} {"Std":>8} {"Seeds":>6}')
    print('-' * 60)
    for mname in ORDER:
        if mname not in method_seeds:
            continue
        vals = method_seeds[mname]
        mean = float(np.mean(vals))
        std  = float(np.std(vals))
        print(f'{mname:<30} {mean:>8.4f} {std:>8.4f} {len(vals):>6}')
    print('=' * 60 + '\n')


if __name__ == '__main__':
    data = parse_all(str(LOG_DIR))
    if not data:
        print('No log files found at', LOG_DIR)
        sys.exit(1)
    print(f'Loaded {len(data)} log files from {LOG_DIR}')
    summary_table(data)
    fig1_ablation_bar(data)
    fig2_lambda_trajectory(data)
    fig3_entropy_vs_lambda(data)
    fig4_round_acc_curves(data)
    fig5_lambda_decay_aggregated(data)
    fig6_per_epoch_acc(data)
    print(f'\nAll figures saved to: {OUT.resolve()}')
