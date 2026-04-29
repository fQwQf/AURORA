#!/usr/bin/env python3
"""Generate convergence trajectory plots from AURORA experiment logs.

Parses per-round test accuracy from log files and produces a multi-panel
figure suitable for inclusion in the paper (PDF).

Usage:
    python scripts/plot_convergence.py
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ── Global plot style ──────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.4,
    "lines.markersize": 3.5,
    "markers.fillstyle": "none",
})

# ── Log paths ──────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
LOG_DIR = BASE / "logs"
FIG_DIR = BASE / "paper" / "figures"

LOG_FILES = {
    "FEMNIST": LOG_DIR / "neurips_femnist" / "AURORA_V24_EXT.log",
    "CIFAR-100": LOG_DIR / "neurips_cifar100" / "AURORA_V24_EXT.log",
    "CIFAR-10": LOG_DIR / "run_baselines" / "AURORA_CIFAR10_a005_ext100.log",
    "SVHN": LOG_DIR / "run_baselines" / "AURORA_SVHN_a005_ext100.log",
    "Tiny-IN": LOG_DIR / "tiny_rerun" / "aurora_tiny_100ep.log",
}

# Colors – colour-blind safe palette
COLORS = {
    "FEMNIST": "#E69F00",
    "CIFAR-100": "#56B4E9",
    "CIFAR-10": "#009E73",
    "SVHN": "#D55E00",
    "Tiny-IN": "#CC79A7",
}

# Dataset display order (left → right)
ORDER = ["FEMNIST", "CIFAR-100", "SVHN", "CIFAR-10", "Tiny-IN"]

# ── Parsing ────────────────────────────────────────────────────────────
_RE_ACC = re.compile(
    r"The test accuracy of OursV24_RawCE_FlatSupCon:\s+([0-9.]+)"
)


def parse_round_accuracies(path: Path) -> list[float]:
    """Extract *OursV24* per-round test accuracy from a log file.

    Only the first contiguous block of ``OursV24`` results is captured
    (i.e. before any baseline / ablation algorithm kicks in).
    """
    accs: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "OursV21" in line and accs:
            # switched to ablation – stop collecting
            break
        m = _RE_ACC.search(line)
        if m:
            accs.append(float(m.group(1)) * 100)  # → percentage
    return accs


# ── Plotting ───────────────────────────────────────────────────────────
def main() -> None:
    data: dict[str, list[float]] = {}
    for name in ORDER:
        p = LOG_FILES[name]
        if not p.exists():
            print(f"[WARN] log not found: {p}")
            continue
        accs = parse_round_accuracies(p)
        data[name] = accs
        print(f"  {name:10s}: {len(accs):>3d} rounds  "
              f"(R0={accs[0]:.2f}% → R{len(accs)-1}={accs[-1]:.2f}%)")

    # ── Figure layout: 1 row × 5 cols ─────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 1.55), sharey=False)

    for idx, name in enumerate(ORDER):
        ax = axes[idx]
        accs = data[name]
        rounds = np.arange(len(accs))

        ax.plot(rounds, accs,
                color=COLORS[name], marker="o", markevery=max(1, len(accs) // 8),
                markeredgewidth=0.8, zorder=3)

        # Dashed vertical line at R19 (standard 20-round budget)
        if len(accs) > 20:
            ax.axvline(x=19, color="grey", ls="--", lw=0.7, alpha=0.6)
            ax.text(20, ax.get_ylim()[0] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                    "R19", fontsize=6.5, color="grey", va="bottom")

        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Round")

        # y-axis: start from a bit below min, leave room at top
        y_min = max(0, min(accs) - 2)
        y_max = max(accs) + 2
        ax.set_ylim(y_min, y_max)

        # Light grid
        ax.yaxis.grid(True, lw=0.3, alpha=0.5)
        ax.set_axisbelow(True)

    # Only left-most subplot gets y-label
    axes[0].set_ylabel("Test Accuracy (%)")

    fig.tight_layout(w_pad=1.2)

    # ── Save ───────────────────────────────────────────────────────────
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = FIG_DIR / "convergence_curves.pdf"
    out_png = FIG_DIR / "convergence_curves.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"\nSaved → {out_pdf}")
    print(f"Saved → {out_png}")


if __name__ == "__main__":
    main()
