"""Figure 1 — Ex-post profit distribution KDE.

Single-panel KDE comparison of MI-DPC retained transformer seeds vs
MIQP-PW vs MIQP-GL on the 19-day benchmark.

Usage
-----
    python3 -m DPC.visualize.fig_profit_distribution [--output-dir DIR]
"""

import argparse
import csv
import json
import pathlib

import numpy as np
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from DPC.visualize.style import (
    apply_style,
    cleanup_axes,
    C_MIDPC,
    C_MIQP_PW,
    C_MIQP_GL,
    COL_WIDTH,
    FIGS_OUT, BENCH_DIR, MIQP_ROOT,
)
from DPC.visualize.data import filter_runs, load_ablation_runs_csv

# ── Paths ──────────────────────────────────────────────────────────────────────
_DEFAULT_OUT = FIGS_OUT

_RUNS_CSV = BENCH_DIR / "ABLATION_47SEED_RUNS.csv"

_MIQP_GL_CSV = MIQP_ROOT / "MIQP_linear" / "MILP_global_linear_benchmark.csv"
_MIQP_PW_CSV = MIQP_ROOT / "MIQP_piecewise" / "MIQP_piecewise_benchmark.csv"


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_dpc_profits_per_seed(run_rows: list[dict[str, str]]) -> list[np.ndarray]:
    """Return list of per-seed arrays of per-day ex-post profits (19 values each)."""
    seed_profits: list[np.ndarray] = []
    for row in run_rows:
        path = BENCH_DIR / row["run_dir"] / "eval_results.json"
        if not path.exists():
            raise FileNotFoundError(f"No eval_results.json found at {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        per_day = data.get("per_day", {})
        if not per_day:
            raise ValueError(f"No 'per_day' key in {path}")
        profits = np.array([v["expost_profit"] for v in per_day.values()], dtype=float)
        seed_profits.append(profits)
    return seed_profits


def _load_miqp_profits(csv_path: pathlib.Path) -> np.ndarray:
    """Load per-day ex-post profits from a MIQP benchmark CSV."""
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        # Find column containing "ex-post" case-insensitively
        profit_col = None
        for col in reader.fieldnames or []:
            if "ex-post" in col.lower():
                profit_col = col
                break
        if profit_col is None:
            raise KeyError(
                f"No column containing 'ex-post' found in {csv_path}. "
                f"Columns: {reader.fieldnames}"
            )
        profits = [float(row[profit_col]) for row in reader]
    return np.array(profits, dtype=float)


# ── Plotting ───────────────────────────────────────────────────────────────────

def _make_kde_grid(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    kde = gaussian_kde(values)
    return kde(x)


def make_figure(output_dir: pathlib.Path) -> pathlib.Path:
    """Build and save the profit distribution figure. Returns path to PDF."""
    # ── Load data ──
    rows = load_ablation_runs_csv(_RUNS_CSV)
    dpc_rows = filter_runs(rows, study="architecture", variant="transformer")
    dpc_seed_profits = _load_dpc_profits_per_seed(dpc_rows)
    miqp_pw_profits = _load_miqp_profits(_MIQP_PW_CSV)
    miqp_gl_profits = _load_miqp_profits(_MIQP_GL_CSV)

    # ── Build shared x grid ──
    all_vals = np.concatenate(dpc_seed_profits + [miqp_pw_profits, miqp_gl_profits])
    x_min = all_vals.min() - 0.05 * (all_vals.max() - all_vals.min())
    x_max = all_vals.max() + 0.05 * (all_vals.max() - all_vals.min())
    x = np.linspace(x_min, x_max, 500)

    # ── MI-DPC: per-seed KDEs → mean ± envelope ──
    dpc_kdes = np.stack([_make_kde_grid(p, x) for p in dpc_seed_profits], axis=0)
    dpc_mean_kde = dpc_kdes.mean(axis=0)
    dpc_min_kde  = dpc_kdes.min(axis=0)
    dpc_max_kde  = dpc_kdes.max(axis=0)

    # ── MIQP KDEs ──
    pw_kde = _make_kde_grid(miqp_pw_profits, x)
    gl_kde = _make_kde_grid(miqp_gl_profits, x)

    # ── Mean values for vertical lines ──
    dpc_mean_val = np.mean([p.mean() for p in dpc_seed_profits])
    pw_mean_val  = miqp_pw_profits.mean()
    gl_mean_val  = miqp_gl_profits.mean()

    # ── Plot ──
    apply_style()
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.2))

    # MI-DPC
    ax.plot(x, dpc_mean_kde, color=C_MIDPC, lw=1.5, label="MI-DPC (ours)")
    ax.fill_between(x, dpc_min_kde, dpc_max_kde, color=C_MIDPC, alpha=0.15)
    ax.axvline(dpc_mean_val, color=C_MIDPC, lw=1.0, ls="--")

    # MIQP-PW
    ax.plot(x, pw_kde, color=C_MIQP_PW, lw=1.5, label="MIQP-PW")
    ax.axvline(pw_mean_val, color=C_MIQP_PW, lw=1.0, ls="--")

    # MIQP-GL
    ax.plot(x, gl_kde, color=C_MIQP_GL, lw=1.5, label="MIQP-GL")
    ax.axvline(gl_mean_val, color=C_MIQP_GL, lw=1.0, ls="--")

    # Labels
    ax.set_xlabel("Ex-post profit (EUR/day)")
    ax.set_ylabel("Density")

    # Legend
    ax.legend(frameon=False)

    cleanup_axes(ax, grid=False)

    fig.tight_layout()

    # ── Save ──
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "profit_distribution.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate profit distribution KDE figure.")
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=_DEFAULT_OUT,
        help=f"Directory to write profit_distribution.pdf (default: {_DEFAULT_OUT})",
    )
    args = parser.parse_args()
    make_figure(args.output_dir)


if __name__ == "__main__":
    main()
