"""Figure 2 — Training convergence curves.

1-by-3 panels:
  (a) Architecture comparison  — Transformer / Bi-LSTM / MLP / CNN
  (b) Temperature comparison   — Annealed / Fixed-low
  (c) Dynamics comparison      — Batch / Step

Usage
-----
    python3 -m DPC.visualize.fig_training_convergence [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from DPC.visualize.data import filter_runs, load_ablation_runs_csv
from DPC.visualize.style import (
    apply_style,
    cleanup_axes,
    C_MIDPC,
    C_BILSTM,
    C_MLP,
    C_CNN,
    C_FIXED,
    C_STEP,
    FULL_WIDTH,
    FIGS_OUT,
    BENCH_DIR,
)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
_BENCH_DIR = Path(BENCH_DIR)
_DEFAULT_OUT = Path(FIGS_OUT)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def _forward_fill(values):
    """Forward-fill a list that may contain None/NaN sentinels, then back-fill leading Nones."""
    last = None
    result = []
    for v in values:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            result.append(last)
        else:
            last = v
            result.append(v)
    first_valid = None
    for v in result:
        if v is not None:
            first_valid = v
            break
    if first_valid is not None:
        result = [first_valid if v is None else v for v in result]
    return result


def _parse_float(s):
    """Return float or None for blank / 'NaN' / 'nan' strings."""
    s = s.strip()
    if not s or s.lower() == "nan":
        return None
    return float(s)


def _load_seed(csv_path: Path):
    """Read history.csv and return forward-filled dev_expost list."""
    series = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            series.append(_parse_float(row["dev_expost"]))
    return _forward_fill(series)


def load_config(bench_dir: Path, rows):
    """
    Load dev_expost for all rows in a study/variant slice.

    Returns (mean_array, std_array, epochs_array) truncated to the
    shortest seed length. Rows that yield no data are skipped.
    """
    all_series = []
    for row in rows:
        csv_path = bench_dir / row["run_dir"] / "history.csv"
        if not csv_path.exists():
            continue
        series = _load_seed(csv_path)
        if series:
            all_series.append(series)

    if not all_series:
        raise FileNotFoundError("No history CSVs found for requested run slice")

    min_len = min(len(s) for s in all_series)
    mat = [[s[i] for s in all_series] for i in range(min_len)]

    means, stds, epochs = [], [], []
    for i, row in enumerate(mat):
        valid = [v for v in row if v is not None]
        if not valid:
            continue
        n = len(valid)
        mu = sum(valid) / n
        variance = sum((v - mu) ** 2 for v in valid) / n
        means.append(mu)
        stds.append(math.sqrt(variance))
        epochs.append(i + 1)

    return epochs, means, stds


# ---------------------------------------------------------------------------
# Panel definitions
# ---------------------------------------------------------------------------
def _define_panels():
    return [
        {
            "label": "(a) Architecture",
            "curves": [
                ("Transformer", C_MIDPC, "architecture", "transformer"),
                ("Bi-LSTM", C_BILSTM, "architecture", "bilstm"),
                ("MLP", C_MLP, "architecture", "mlp"),
                ("CNN", C_CNN, "architecture", "cnn"),
            ],
        },
        {
            "label": "(b) Temperature",
            "curves": [
                ("Annealed", C_MIDPC, "temperature", "annealed"),
                ("Fixed-low", C_FIXED, "temperature", "fixed_low"),
            ],
        },
        {
            "label": "(c) Dynamics",
            "curves": [
                ("Batch", C_MIDPC, "dynamics", "batch"),
                ("Step", C_STEP, "dynamics", "step"),
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def make_figure(bench_dir, output_dir):
    apply_style()

    bench_dir = Path(bench_dir)
    output_dir = Path(output_dir)
    runs = load_ablation_runs_csv(bench_dir / "ABLATION_47SEED_RUNS.csv")

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(FULL_WIDTH, 2.0),
        sharey=True,
    )

    for col, (ax, panel) in enumerate(zip(axes, _define_panels())):
        for label, color, study, variant in panel["curves"]:
            rows = filter_runs(runs, study=study, variant=variant)
            try:
                epochs, means, stds = load_config(bench_dir, rows)
            except FileNotFoundError as e:
                print(f"  WARNING: {e}")
                continue

            lo = [m - s for m, s in zip(means, stds)]
            hi = [m + s for m, s in zip(means, stds)]

            ax.plot(epochs, means, color=color, label=label, linewidth=1.5)
            ax.fill_between(epochs, lo, hi, color=color, alpha=0.15)

        ax.text(
            0.04,
            0.97,
            panel["label"],
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            va="top",
            ha="left",
        )

        ax.set_xlabel("Epoch")
        if col == 0:
            ax.set_ylabel("Dev ex-post profit\n(EUR/day)")

        ax.legend(frameon=False, fontsize=7, loc="lower right")
        cleanup_axes(ax, grid=True)

    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "training_convergence.pdf"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate training convergence figure.")
    parser.add_argument(
        "--output-dir",
        default=_DEFAULT_OUT,
        type=Path,
        help="Directory to write training_convergence.pdf",
    )
    parser.add_argument(
        "--bench-dir",
        default=_BENCH_DIR,
        type=Path,
        help="Path to benchmark_suite directory",
    )
    args = parser.parse_args()
    make_figure(args.bench_dir, args.output_dir)


if __name__ == "__main__":
    main()
