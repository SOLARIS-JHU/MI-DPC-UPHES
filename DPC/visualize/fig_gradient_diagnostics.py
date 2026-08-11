"""Figure 4 — Gradient diagnostics: norm vs epoch and late/early ratio.

Two-panel figure comparing Batch (Transformer baseline) vs Step dynamics
gradient norms across training, averaged over the retained seed set.

Usage
-----
    python3 -m DPC.visualize.fig_gradient_diagnostics [--output-dir DIR]
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from DPC.visualize.style import (
    apply_style,
    cleanup_axes,
    C_MIDPC,
    C_STEP,
    COL_WIDTH,
    FIGS_OUT,
    BENCH_DIR as _BENCH_DIR,
)
from DPC.visualize.data import filter_runs, load_ablation_runs_csv

# ── Constants ──────────────────────────────────────────────────────────────────
BENCH_DIR = Path(_BENCH_DIR)
RUNS_CSV = BENCH_DIR / "ABLATION_47SEED_RUNS.csv"
W = 5  # window size for running ratio


# ── Data helpers ───────────────────────────────────────────────────────────────

def _read_grad_norm(csv_path: str) -> list[float]:
    """Read grad_norm column from a history CSV, returning a list of floats."""
    values = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            v = row.get("grad_norm", "")
            if v and v.strip():
                values.append(float(v))
    return values


def _load_series(bench_dir: Path, rows) -> list[list[float]]:
    """Load grad_norm series for a study/variant slice."""
    series = []
    for row in rows:
        csv_path = bench_dir / row["run_dir"] / "history.csv"
        if csv_path.is_file():
            gn = _read_grad_norm(str(csv_path))
            if gn:
                series.append(gn)
    return series


def _aggregate(series: list[list[float]]) -> tuple[list[float], list[float], list[float]]:
    """Truncate to shortest length, return (epochs, mean, std) as lists."""
    if not series:
        raise ValueError("No series to aggregate")
    min_len = min(len(s) for s in series)
    truncated = [s[:min_len] for s in series]
    epochs = list(range(1, min_len + 1))
    means = []
    stds = []
    for i in range(min_len):
        col = [s[i] for s in truncated]
        n = len(col)
        mu = sum(col) / n
        var = sum((x - mu) ** 2 for x in col) / n
        means.append(mu)
        stds.append(math.sqrt(var))
    return epochs, means, stds


def _running_ratio(gn: list[float], w: int = W) -> list[float]:
    """Compute running late/early ratio for a single seed's grad_norm series.

    ratio[e] = mean(gn[max(0,e-w+1):e+1]) / mean(gn[0:w])
    """
    early_mean = sum(gn[:w]) / w if len(gn) >= w else sum(gn) / len(gn)
    ratios = []
    for e in range(len(gn)):
        start = max(0, e - w + 1)
        window = gn[start : e + 1]
        late_mean = sum(window) / len(window)
        ratios.append(late_mean / early_mean if early_mean != 0 else float("nan"))
    return ratios


# ── Main ───────────────────────────────────────────────────────────────────────

def main(output_dir: str = ".") -> None:
    apply_style()

    bench = Path(BENCH_DIR).resolve()
    runs = load_ablation_runs_csv(RUNS_CSV)

    # ── Load Batch (Transformer) ──
    batch_rows = filter_runs(runs, study="dynamics", variant="batch")
    batch_series = _load_series(bench, batch_rows)
    if not batch_series:
        raise RuntimeError(f"No Batch dynamics runs found under {bench}")

    # ── Load Step ──
    step_rows = filter_runs(runs, study="dynamics", variant="step")
    step_series = _load_series(bench, step_rows)
    if not step_series:
        raise RuntimeError(f"No Step dynamics runs found under {bench}")

    # ── Aggregate grad norm ──
    b_epochs, b_mean, b_std = _aggregate(batch_series)
    s_epochs, s_mean, s_std = _aggregate(step_series)

    # ── Plot ──
    fig, ax = plt.subplots(1, 1, figsize=(COL_WIDTH, 1.6))

    # ── Gradient norm vs epoch ──
    ax.plot(b_epochs, b_mean, color=C_MIDPC, label="Batch")
    ax.fill_between(
        b_epochs,
        [m - sd for m, sd in zip(b_mean, b_std)],
        [m + sd for m, sd in zip(b_mean, b_std)],
        color=C_MIDPC,
        alpha=0.15,
    )
    ax.plot(s_epochs, s_mean, color=C_STEP, label="Step")
    ax.fill_between(
        s_epochs,
        [m - sd for m, sd in zip(s_mean, s_std)],
        [m + sd for m, sd in zip(s_mean, s_std)],
        color=C_STEP,
        alpha=0.15,
    )

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient norm")
    ax.legend(frameon=False, fontsize=7)
    cleanup_axes(ax, grid=True)

    # ── Save ──
    fig.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "gradient_diagnostics.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}  ({out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate gradient diagnostics figure.")
    parser.add_argument(
        "--output-dir",
        default=Path(FIGS_OUT),
        help="Directory to write gradient_diagnostics.pdf",
    )
    args = parser.parse_args()
    main(output_dir=args.output_dir)
