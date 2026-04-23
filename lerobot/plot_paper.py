#!/usr/bin/env python3
"""
Publication-quality plotting for HIL-SERL paper figures.

Outputs vector PDF with consistent fonts, muted gridlines, and minimal
chart-junk so figures embed cleanly in LaTeX papers.

Usage:
    # Single run figures (one set of curves)
    python plot_paper.py --run outputs/train/<run_dir>

    # V1 vs V2 comparison (overlay two runs)
    python plot_paper.py --v1 outputs/train/<v1_dir> --v2 outputs/train/<v2_dir>

Outputs:
    figures/fig_v1_curves.pdf       # 6-panel grid for single V1 run
    figures/fig_v2_curves.pdf       # 6-panel grid for single V2 run
    figures/fig_compare_v1v2.pdf    # 4-panel V1-vs-V2 overlay
    figures/fig_q_vs_entropy.pdf    # Q-value vs entropy term (V2 verification)
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path


# ── Color palette (colorblind-safe, paper-friendly) ──
COLOR_V1 = "#0D9488"          # Teal
COLOR_V2 = "#7C3AED"          # Purple
COLOR_NEUTRAL = "#475569"     # Slate gray
COLOR_ACCENT = "#EA580C"      # Orange
COLOR_GRID = "#E2E8F0"        # Pale gray


def setup_matplotlib():
    """Configure matplotlib for publication-quality output."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required. Install with: pip install matplotlib")
        sys.exit(1)

    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": COLOR_GRID,
        "grid.linestyle": "-",
        "grid.linewidth": 0.4,
        "grid.alpha": 0.6,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "legend.borderaxespad": 0.4,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,           # Embed TrueType, not Type 3 (LaTeX-safe)
        "ps.fonttype": 42,
    })
    return plt


def read_csv(filepath):
    if not os.path.exists(filepath):
        return []
    with open(filepath) as f:
        return list(csv.DictReader(f))


def safe_float(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def col(rows, key):
    """Extract numeric column from CSV rows, dropping invalid entries."""
    pairs = [(safe_float(r.get("optimization_step") or r.get("interaction_step")),
              safe_float(r.get(key))) for r in rows]
    pairs = [(s, v) for s, v in pairs if s is not None and v is not None]
    if not pairs:
        return [], []
    xs, ys = zip(*pairs)
    return list(xs), list(ys)


def moving_avg(values, window):
    if window < 2 or len(values) < window:
        return values
    return [sum(values[i:i + window]) / window for i in range(len(values) - window + 1)]


def load_run(run_dir):
    """Load all CSV + JSON data for a single training run."""
    log_dir = os.path.join(run_dir, "training_logs")
    return {
        "name": os.path.basename(run_dir),
        "training": read_csv(os.path.join(log_dir, "training_metrics.csv")),
        "episode": read_csv(os.path.join(log_dir, "episode_metrics.csv")),
        "eval": read_csv(os.path.join(log_dir, "eval_metrics.csv")),
        "summary": _load_json(os.path.join(log_dir, "training_summary.json")),
        "metadata": _load_json(os.path.join(log_dir, "experiment_metadata.json")),
    }


def _load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def plot_single_run(plt, run, color, out_path, title=""):
    """6-panel figure for a single training run."""
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.0))
    fig.suptitle(title or f"Training Curves — {run['name']}", fontsize=10, y=0.995)

    # Panel 1: Episodic reward (smoothed)
    ax = axes[0, 0]
    xs, ys = col(run["episode"], "episodic_reward")
    if xs:
        ax.plot(xs, ys, "o", markersize=1.2, alpha=0.25, color=color)
        if len(ys) >= 10:
            w = max(len(ys) // 30, 5)
            sm = moving_avg(ys, w)
            ax.plot(xs[:len(sm)], sm, "-", linewidth=1.4, color=color, label=f"MA({w})")
            ax.legend(loc="best")
    ax.set_xlabel("Interaction step")
    ax.set_ylabel("Episodic reward")
    ax.set_title("(a) Episodic reward")

    # Panel 2: Rolling success rate
    ax = axes[0, 1]
    xs, ys = col(run["episode"], "rolling_success_rate_50")
    if xs:
        ax.plot(xs, [v * 100 for v in ys], "-", linewidth=1.4, color=color, label="All episodes")
    xs2, ys2 = col(run["episode"], "rolling_policy_only_success_20")
    if xs2:
        ax.plot(xs2, [v * 100 for v in ys2], "--", linewidth=1.2,
                color=COLOR_ACCENT, label="Policy-only")
    ax.set_xlabel("Interaction step")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(-5, 105)
    ax.set_title("(b) Success rate")
    if xs or xs2:
        ax.legend(loc="best")

    # Panel 3: Intervention rate
    ax = axes[0, 2]
    xs, ys = col(run["episode"], "rolling_intervention_rate_50")
    if xs:
        ax.plot(xs, [v * 100 for v in ys], "-", linewidth=1.4, color=color)
    ax.set_xlabel("Interaction step")
    ax.set_ylabel("Intervention rate (%)")
    ax.set_ylim(-5, 105)
    ax.set_title("(c) Human intervention")

    # Panel 4: Critic loss
    ax = axes[1, 0]
    xs, ys = col(run["training"], "loss_critic")
    if xs:
        ax.plot(xs, ys, "-", linewidth=0.6, alpha=0.4, color=color)
        if len(ys) >= 20:
            w = max(len(ys) // 50, 5)
            sm = moving_avg(ys, w)
            ax.plot(xs[:len(sm)], sm, "-", linewidth=1.4, color=color)
        ax.set_yscale("log")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Critic loss")
    ax.set_title("(d) Critic loss")

    # Panel 5: Q-value statistics
    ax = axes[1, 1]
    xs, q_mean = col(run["training"], "q_mean")
    _, q_std = col(run["training"], "q_std")
    _, ent = col(run["training"], "entropy_term")
    if xs and q_mean:
        ax.plot(xs, q_mean, "-", linewidth=1.4, color=color, label="Q (mean)")
        if q_std and len(q_std) == len(q_mean):
            lo = [m - s for m, s in zip(q_mean, q_std)]
            hi = [m + s for m, s in zip(q_mean, q_std)]
            ax.fill_between(xs, lo, hi, alpha=0.2, color=color)
    if xs and ent:
        ax.plot(xs[:len(ent)], ent, "--", linewidth=1.2,
                color=COLOR_ACCENT, label="α·|log π|")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Value")
    ax.set_title("(e) Q vs entropy term")
    if (q_mean or ent):
        ax.legend(loc="best")

    # Panel 6: SAC temperature
    ax = axes[1, 2]
    xs, ys = col(run["training"], "temperature")
    if xs:
        ax.plot(xs, ys, "-", linewidth=1.4, color=color)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Temperature α")
    ax.set_title("(f) SAC temperature")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  saved: {out_path}")


def plot_compare(plt, v1, v2, out_path):
    """4-panel V1 vs V2 comparison overlay."""
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.5))
    fig.suptitle("V1 (sparse) vs V2 (dense) — convergence comparison", fontsize=10, y=0.995)

    # Panel a: Episodic reward (smoothed only, normalized)
    ax = axes[0, 0]
    for run, color, label in [(v1, COLOR_V1, "V1 sparse"), (v2, COLOR_V2, "V2 dense")]:
        xs, ys = col(run["episode"], "episodic_reward")
        if xs and len(ys) >= 10:
            w = max(len(ys) // 30, 5)
            sm = moving_avg(ys, w)
            ax.plot(xs[:len(sm)], sm, "-", linewidth=1.5, color=color, label=label)
    ax.set_xlabel("Interaction step")
    ax.set_ylabel("Episodic reward (smoothed)")
    ax.set_title("(a) Reward learning curve")
    ax.legend(loc="best")

    # Panel b: Rolling success rate
    ax = axes[0, 1]
    for run, color, label in [(v1, COLOR_V1, "V1 sparse"), (v2, COLOR_V2, "V2 dense")]:
        xs, ys = col(run["episode"], "rolling_success_rate_50")
        if xs:
            ax.plot(xs, [v * 100 for v in ys], "-", linewidth=1.5, color=color, label=label)
    ax.set_xlabel("Interaction step")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(-5, 105)
    ax.set_title("(b) Rolling success rate")
    ax.legend(loc="best")

    # Panel c: Intervention rate
    ax = axes[1, 0]
    for run, color, label in [(v1, COLOR_V1, "V1 sparse"), (v2, COLOR_V2, "V2 dense")]:
        xs, ys = col(run["episode"], "rolling_intervention_rate_50")
        if xs:
            ax.plot(xs, [v * 100 for v in ys], "-", linewidth=1.5, color=color, label=label)
    ax.set_xlabel("Interaction step")
    ax.set_ylabel("Intervention rate (%)")
    ax.set_ylim(-5, 105)
    ax.set_title("(c) Human intervention")
    ax.legend(loc="best")

    # Panel d: Q vs entropy term
    ax = axes[1, 1]
    for run, color, label in [(v1, COLOR_V1, "V1 Q"), (v2, COLOR_V2, "V2 Q")]:
        xs, ys = col(run["training"], "q_mean")
        if xs:
            ax.plot(xs, ys, "-", linewidth=1.5, color=color, label=label)
    # Plot one entropy reference (V2's, since both use same SAC)
    xs, ent = col(v2["training"] or v1["training"], "entropy_term")
    if xs and ent:
        ax.plot(xs[:len(ent)], ent, "--", linewidth=1.2,
                color=COLOR_ACCENT, label="α·|log π|")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Value")
    ax.set_title("(d) Q vs entropy term")
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  saved: {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", help="Single run directory (outputs/train/<dir>)")
    p.add_argument("--v1", help="V1 run directory for comparison")
    p.add_argument("--v2", help="V2 run directory for comparison")
    p.add_argument("--out", default="figures", help="Output directory (default: ./figures)")
    args = p.parse_args()

    if not args.run and not (args.v1 and args.v2):
        p.error("Provide either --run <dir> or both --v1 <dir> --v2 <dir>")

    plt = setup_matplotlib()
    os.makedirs(args.out, exist_ok=True)

    if args.run:
        run = load_run(args.run)
        out = os.path.join(args.out, f"fig_{run['name']}_curves.pdf")
        plot_single_run(plt, run, COLOR_V1, out, title=f"Training Curves — {run['name']}")

    if args.v1 and args.v2:
        v1 = load_run(args.v1)
        v2 = load_run(args.v2)
        # Single-run figures for each
        plot_single_run(plt, v1, COLOR_V1,
                        os.path.join(args.out, "fig_v1_curves.pdf"),
                        title="V1 sparse reward — training curves")
        plot_single_run(plt, v2, COLOR_V2,
                        os.path.join(args.out, "fig_v2_curves.pdf"),
                        title="V2 dense reward — training curves")
        # Overlay comparison
        plot_compare(plt, v1, v2, os.path.join(args.out, "fig_compare_v1v2.pdf"))


if __name__ == "__main__":
    main()
