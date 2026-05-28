#!/usr/bin/env python3
"""Generate closure plots for the VCC autoresearch loop.

This is a starting template the autoresearch agent should adapt to fit the
actual VCC run. It ports the aesthetic settings and the helpers from the
MoFNet PoC blog plots (white background, soft palette, wrap_caption helper,
path-effects label legibility) and provides the eight plot skeletons the
autoresearch.md prompt requires.

Run after closure:

    python scripts/generate_closure_plots.py \
        --results outputs/results.tsv \
        --baseline outputs/BASELINE_REGISTRY.md \
        --out outputs/closure_plots/

The agent should:

1. Inspect outputs/results.tsv to confirm the columns are present
   (experiment_num, status, family, cell_eval_pds, primary_metric,
   parent_experiment_ids, branch_type, subtree_status, leakage_guard).
2. Extract per-seed cell-eval PDS for Tier 2 / Tier 3 backbone nodes
   from outputs/<EXP_ID>/seed_*/metrics.json (or wherever the model
   wrote them).
3. Fill in the data-loading stubs marked `# TODO(agent)` below.
4. Run this script, verify the eight PNGs are readable, and embed the
   results in final_report.md.

Style: white background, single accent per plot, no em-dashes in captions,
85-character caption wrap, dark ink labels with white stroke.
"""

from __future__ import annotations

import argparse
import csv
import math
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
import numpy as np


# --- style ---------------------------------------------------------------

INK = "#1a1a1a"
MUTED = "#9a9a9a"
SOFT = "#d8d8d8"
ACCENT_TEAL = "#2A9D8F"
ACCENT_CORAL = "#E76F51"
ACCENT_AMBER = "#E9C46A"
ACCENT_NAVY = "#264653"
ACCENT_LILAC = "#9B7EBD"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.titlepad": 18,
    "axes.labelsize": 11,
    "axes.labelpad": 8,
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

LABEL_STROKE = [
    path_effects.Stroke(linewidth=2.4, foreground="white"),
    path_effects.Normal(),
]


def wrap_caption(text: str, width: int = 85) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def _load_results(tsv_path: Path) -> list[dict]:
    with tsv_path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


# --- Plot 1: PDS trajectory ----------------------------------------------


def plot_pds_trajectory(rows: list[dict], baseline_pds: float, out: Path) -> None:
    """Every experiment as a dot in chronological order. y = cell-eval PDS if
    present, else local validation PDS. Status colors the dot.

    # TODO(agent): adjust if the local PDS lives under a different column name
    # (the model writes it as `primary_metric` or a dedicated `local_pds` field).
    """
    xs = list(range(len(rows)))
    cell_eval_pds = []
    local_pds = []
    statuses = []
    for r in rows:
        try:
            ce = float(r.get("cell_eval_pds", "") or "nan")
        except ValueError:
            ce = float("nan")
        try:
            lp = float(r.get("primary_metric_value", "") or r.get("local_pds", "") or "nan")
        except ValueError:
            lp = float("nan")
        cell_eval_pds.append(ce)
        local_pds.append(lp)
        statuses.append(r.get("status", ""))

    fig, ax = plt.subplots(figsize=(11, 5.5))

    # Reference lines
    ax.axhline(0.80, color=ACCENT_NAVY, lw=1.2, ls="--", zorder=0)
    ax.text(0.5, 0.805, "Stop threshold 0.80", fontsize=9, color=ACCENT_NAVY, ha="left", va="bottom")
    if not math.isnan(baseline_pds):
        ax.axhline(baseline_pds, color=SOFT, lw=1.0, zorder=0)
        ax.text(0.5, baseline_pds + 0.003, f"Step 0 baseline {baseline_pds:.4f}",
                fontsize=9, color=MUTED, ha="left", va="bottom")

    def _scatter(idx, color, label, s=22, zorder=2, edgecolor="none"):
        ys = [cell_eval_pds[i] if not math.isnan(cell_eval_pds[i]) else local_pds[i] for i in idx]
        kept = [(i, y) for i, y in zip(idx, ys) if not math.isnan(y)]
        if not kept:
            return
        ix = [k[0] for k in kept]
        iy = [k[1] for k in kept]
        ax.scatter(ix, iy, c=color, s=s, label=label, zorder=zorder,
                   edgecolors=edgecolor, linewidths=0.5, alpha=0.9)

    discard_idx = [i for i, s in enumerate(statuses) if "DISCARD" in s]
    keep_idx = [i for i, s in enumerate(statuses) if "TIER1_KEEP" in s]
    tier2_idx = [i for i, s in enumerate(statuses) if "TIER2" in s]
    promoted_idx = [i for i, s in enumerate(statuses) if "TIER3_PROMOTED" in s]
    baseline_idx = [i for i, s in enumerate(statuses) if s == "STEP0_COMPLETE" or s == "COMPLETE"]

    _scatter(discard_idx, MUTED, "Tier 1 discard")
    _scatter(tier2_idx, ACCENT_AMBER, "Tier 2", s=34, zorder=3)
    _scatter(keep_idx, ACCENT_TEAL, "Tier 1 keep", s=40, zorder=4)
    _scatter(baseline_idx, ACCENT_NAVY, "Step 0 baseline", s=80, zorder=5, edgecolor="white")
    _scatter(promoted_idx, ACCENT_CORAL, "Tier 3 promoted", s=110, zorder=6, edgecolor="white")

    ax.set_xlabel("Experiment index (chronological)")
    ax.set_ylabel("PDS (cell-eval when available, local otherwise)")
    ax.set_title(f"{len(rows)} experiments toward PDS = 0.80")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", color=SOFT, lw=0.6, zorder=0)
    ax.legend(loc="lower right")

    fig.text(0.012, -0.02, wrap_caption(
        "Each dot is one experiment. Solid horizontal line is the stop "
        "threshold of 0.80. Light line is the Step 0 baseline."
    ), fontsize=9, color=MUTED, ha="left", va="top")
    fig.savefig(out / "01_pds_trajectory.png")
    plt.close(fig)


# --- Plot 2: status donut ------------------------------------------------


def plot_status_donut(rows: list[dict], out: Path) -> None:
    buckets = {
        "Tier 1 discard": 0, "Tier 1 keep": 0, "Tier 2 fail": 0,
        "Tier 2 pass": 0, "Tier 3 promoted": 0, "Step 0 baseline": 0,
    }
    for r in rows:
        s = r.get("status", "")
        if "TIER3_PROMOTED" in s:
            buckets["Tier 3 promoted"] += 1
        elif s in ("STEP0_COMPLETE", "COMPLETE"):
            buckets["Step 0 baseline"] += 1
        elif "TIER2_PASS" in s:
            buckets["Tier 2 pass"] += 1
        elif "TIER2" in s:
            buckets["Tier 2 fail"] += 1
        elif "TIER1_KEEP" in s:
            buckets["Tier 1 keep"] += 1
        else:
            buckets["Tier 1 discard"] += 1

    colors = [MUTED, ACCENT_TEAL, ACCENT_AMBER, "#5FA8A0", ACCENT_CORAL, ACCENT_NAVY]
    labels, sizes = list(buckets.keys()), list(buckets.values())
    total = sum(sizes)

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    wedges, _ = ax.pie(sizes, colors=colors, startangle=90, counterclock=False,
                       wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2))
    ax.text(0, 0.08, f"{total}", ha="center", va="center", fontsize=44,
            fontweight="bold", color=INK)
    ax.text(0, -0.10, "trials", ha="center", va="center", fontsize=12, color=MUTED)

    # Outside labels with collision avoidance for small slices.
    small_right, small_left = [], []
    for w, lbl, val in zip(wedges, labels, sizes):
        if val == 0:
            continue
        ang = (w.theta2 + w.theta1) / 2.0
        if val / total < 0.05:
            (small_right if math.cos(math.radians(ang)) > 0 else small_left).append((lbl, val, ang))
        else:
            x = 1.15 * math.cos(math.radians(ang))
            y = 1.15 * math.sin(math.radians(ang))
            ha = "left" if x > 0 else "right"
            ax.text(x, y, f"{lbl}\n{val} ({val/total*100:.0f}%)",
                    ha=ha, va="center", fontsize=11, color=INK, linespacing=1.4)
    base_y = 0.95
    for j, (lbl, val, ang) in enumerate(small_right + small_left):
        x_anchor = 1.05 * math.cos(math.radians(ang))
        y_anchor = 1.05 * math.sin(math.radians(ang))
        ax.annotate(f"{lbl} {val} ({val/total*100:.1f}%)",
                    xy=(x_anchor, y_anchor), xytext=(1.55, base_y - j * 0.13),
                    ha="left", va="center", fontsize=10, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))

    ax.set_xlim(-1.5, 2.5)
    ax.set_title("Outcome distribution across the loop", pad=24)
    fig.text(0.5, 0.02, wrap_caption(
        "Most autonomous search is failure. The shape of the donut tells "
        "you how aggressive the gates were and how much exploration was "
        "absorbed by discards."
    ), fontsize=9, color=MUTED, ha="center", va="bottom")
    fig.savefig(out / "02_status_donut.png")
    plt.close(fig)


# --- Plots 3-8: stubs the agent fills with the actual run's data ---------


def plot_family_bars(rows: list[dict], out: Path) -> None:
    """# TODO(agent): aggregate experiments per family from results.tsv,
    then render a horizontal bar chart sorted by count. Annotate which
    family produced the model of record."""
    fams: dict[str, int] = {}
    for r in rows:
        key = (r.get("family", "") or "unknown").strip()
        fams[key] = fams.get(key, 0) + 1
    items = sorted(fams.items(), key=lambda x: x[1])
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(10, 6))
    ypos = np.arange(len(labels))
    bars = ax.barh(ypos, vals, color=ACCENT_NAVY, height=0.62, alpha=0.92,
                   edgecolor="white", linewidth=1.0)
    for bar, v in zip(bars, vals):
        ax.text(v + 0.6, bar.get_y() + bar.get_height()/2, f"{v}",
                va="center", ha="left", fontsize=10, color=INK)
    ax.set_yticks(ypos); ax.set_yticklabels(labels)
    ax.set_xlabel("Experiments")
    ax.set_title("Where the experiments went, by family")
    ax.grid(True, axis="x", color=SOFT, lw=0.6, zorder=0)
    ax.set_xlim(0, (max(vals) if vals else 1) * 1.15)
    fig.text(0.012, -0.02, wrap_caption(
        "Counts per pre-specified family. The model of record's family is "
        "annotated; retired or cooled families are reported in family_allocation.md."
    ), fontsize=9, color=MUTED, ha="left", va="top")
    fig.savefig(out / "03_family_bars.png")
    plt.close(fig)


def plot_mcc_floor(rows: list[dict], baseline_std: float,
                   strongest_z: float | None, strongest_n: int | None,
                   out: Path) -> None:
    """# TODO(agent): pass in the strongest candidate's beneficial z and the
    candidate count it was observed at. baseline_std comes from
    BASELINE_REGISTRY.md."""
    N = np.arange(1, max(500, len(rows) + 50))
    z_floor = 2 + np.sqrt(np.log(N) / 2)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(N, z_floor, color=ACCENT_NAVY, lw=2.5,
            label=r"$z_{\rm floor}(N) = 2 + \sqrt{\log N / 2}$")
    ax.fill_between(N, 0, z_floor, color=ACCENT_NAVY, alpha=0.07)
    if strongest_z is not None and strongest_n is not None:
        ax.scatter([strongest_n], [strongest_z], color=ACCENT_CORAL, s=140,
                   zorder=5, edgecolor="white", linewidth=2,
                   label=f"strongest candidate z = {strongest_z:.2f}")
    ax.set_xlabel("Single-seed Tier 1 candidates evaluated against the same metric")
    ax.set_ylabel("Beneficial z-score over baseline")
    ax.set_title("Multiple-comparison floor")
    ax.set_xlim(0, len(N))
    ax.set_ylim(0, max(8, (strongest_z or 0) * 1.2 + 1))
    ax.grid(True, color=SOFT, lw=0.6, zorder=0)
    ax.legend(loc="upper right")
    fig.text(0.012, -0.02, wrap_caption(
        "Family-wise floor required for a candidate to be cited as "
        "evidence. A candidate below the floor is below the noise the "
        "search is allowed to claim, regardless of metric value."
    ), fontsize=9, color=MUTED, ha="left", va="top")
    fig.savefig(out / "04_mcc_floor.png")
    plt.close(fig)


def plot_lineage_backbone(out: Path) -> None:
    """# TODO(agent): trace the promoted lineage from results.tsv using
    parent_experiment_ids, then render the same kind of backbone plot the
    MoFNet PoC used (06_lineage.png). One node per backbone experiment,
    PDS on y, chronological index on x, annotation text describing what
    changed at each step, dark ink labels with white stroke."""
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.text(0.5, 0.5,
            "Lineage backbone plot — agent fills with the run's actual lineage.",
            ha="center", va="center", fontsize=12, color=MUTED,
            transform=ax.transAxes)
    ax.set_axis_off()
    fig.savefig(out / "05_lineage_backbone.png")
    plt.close(fig)


def plot_per_seed_variance(out: Path) -> None:
    """# TODO(agent): pull per-seed cell-eval PDS for every Tier 2 / Tier 3
    backbone node and plot one column per node with the 5 individual seed
    dots plus a mean horizontal tick. Match the MoFNet PoC's 07 plot."""
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.text(0.5, 0.5,
            "Per-seed variance plot — agent fills.",
            ha="center", va="center", fontsize=12, color=MUTED,
            transform=ax.transAxes)
    ax.set_axis_off()
    fig.savefig(out / "06_per_seed_variance.png")
    plt.close(fig)


def plot_local_vs_celleval_calibration(rows: list[dict], out: Path) -> None:
    """Scatter local validation PDS vs cell-eval PDS for every experiment that
    has both. Identity line for reference."""
    pts = []
    for r in rows:
        try:
            ce = float(r.get("cell_eval_pds", "") or "nan")
            lp = float(r.get("primary_metric_value", "") or r.get("local_pds", "") or "nan")
        except ValueError:
            continue
        if not (math.isnan(ce) or math.isnan(lp)):
            pts.append((lp, ce))
    fig, ax = plt.subplots(figsize=(8, 8))
    if pts:
        lp = [p[0] for p in pts]; ce = [p[1] for p in pts]
        ax.scatter(lp, ce, color=ACCENT_TEAL, s=60, alpha=0.85,
                   edgecolor="white", linewidth=1)
    lo, hi = 0.0, 1.0
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1.0, ls="--", zorder=0)
    ax.set_xlabel("Local validation PDS (in-loop approximation)")
    ax.set_ylabel("cell-eval PDS on locked_test")
    ax.set_title("Does the local approximation track the gating metric?")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.grid(True, color=SOFT, lw=0.6, zorder=0)
    fig.text(0.012, -0.02, wrap_caption(
        "If the local PDS hugs the diagonal, you can trust the in-loop "
        "early-stop signal. If it does not, the loop has been making "
        "selection decisions on a metric that does not match the gate."
    ), fontsize=9, color=MUTED, ha="left", va="top")
    fig.savefig(out / "07_local_vs_celleval_calibration.png")
    plt.close(fig)


def plot_three_acts(baseline_pds: float, strongest_tier1: float,
                    promoted_pds: float, out: Path) -> None:
    """Step 0 / strongest Tier 1 keep / promoted (or strongest Tier 3) / stop threshold."""
    labels = [
        "Step 0\nbaseline",
        "Strongest\nTier 1 keep",
        "Promoted\nor strongest\nTier 3",
        "Stop threshold\n0.80",
    ]
    values = [baseline_pds, strongest_tier1, promoted_pds, 0.80]
    colors = [ACCENT_NAVY, ACCENT_TEAL, ACCENT_CORAL, MUTED]

    fig, ax = plt.subplots(figsize=(11, 7))
    xpos = np.arange(len(labels))
    bars = ax.bar(xpos, values, color=colors, width=0.55, edgecolor="white",
                  linewidth=2, alpha=0.95, zorder=3)
    for bar, v, c in zip(bars, values, colors):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.4f}",
                ha="center", va="bottom", fontsize=14, fontweight="bold",
                color=c, zorder=4)
    ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("PDS")
    ax.set_title("Four numbers: where the loop started, what it found, where the gate is",
                 pad=24)
    ax.grid(True, axis="y", color=SOFT, lw=0.6, zorder=0)
    fig.text(0.012, -0.04, wrap_caption(
        "Step 0 is the baseline pseudobulk to single-cell warmstart. "
        "Strongest Tier 1 keep is the in-loop high-water mark. Promoted is "
        "the model of record if a Tier 3 fired, otherwise the strongest "
        "Tier 3 candidate. Stop threshold is the autoresearch-bio gate."
    ), fontsize=9, color=MUTED, ha="left", va="top")
    fig.savefig(out / "08_three_acts.png")
    plt.close(fig)


# --- main ----------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True, help="Path to results.tsv")
    p.add_argument("--baseline-pds", type=float, default=float("nan"),
                   help="Step 0 baseline PDS (from BASELINE_REGISTRY.md)")
    p.add_argument("--baseline-std", type=float, default=0.01,
                   help="Per-seed baseline std (from BASELINE_REGISTRY.md)")
    p.add_argument("--strongest-z", type=float, default=None,
                   help="Strongest candidate's beneficial z-score over baseline")
    p.add_argument("--strongest-n", type=int, default=None,
                   help="Candidate count at which the strongest z was observed")
    p.add_argument("--strongest-tier1", type=float, default=float("nan"),
                   help="Strongest Tier 1 keep PDS (cell-eval if available, else local)")
    p.add_argument("--promoted-pds", type=float, default=float("nan"),
                   help="Promoted (or strongest Tier 3) PDS")
    p.add_argument("--out", required=True, help="Output directory for plots")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = _load_results(Path(args.results))

    plot_pds_trajectory(rows, args.baseline_pds, out)
    plot_status_donut(rows, out)
    plot_family_bars(rows, out)
    plot_mcc_floor(rows, args.baseline_std, args.strongest_z, args.strongest_n, out)
    plot_lineage_backbone(out)
    plot_per_seed_variance(out)
    plot_local_vs_celleval_calibration(rows, out)
    plot_three_acts(args.baseline_pds, args.strongest_tier1, args.promoted_pds, out)

    print(f"Wrote 8 plots to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
