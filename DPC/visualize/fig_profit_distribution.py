"""Figure 1 — Ex-post profit distribution KDE.

Two density panels (MI-DPC against MIQP-GL, and against MIQP-PW) of per-day
ex-post profit on the 19-day benchmark, in the style of the DFL-UPHES paper:
Gaussian KDE at half of Scott's bandwidth, drawn over the observed range.
MI-DPC is averaged over the retained transformer seeds per day.

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
    FULL_WIDTH,
    themed,
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

BW_FACTOR = 0.5   # fraction of Scott's bandwidth, as in the DFL-UPHES density figure
FILL_ALPHA = 0.2


def density_curve(values: np.ndarray, bw_factor: float = BW_FACTOR, points: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian KDE at bw_factor × Scott's bandwidth, evaluated only over the observed range."""
    values = np.asarray(values, dtype=float)
    kde = gaussian_kde(values, bw_method="scott")
    kde.set_bandwidth(kde.factor * bw_factor)
    x = np.linspace(values.min(), values.max(), points)
    return x, kde(x)


def _draw_method(ax, values: np.ndarray, label: str, color: str) -> None:
    x, density = density_curve(values)
    ax.plot(x, density, color=color, linewidth=1.0, alpha=0.95,
            label=f"{label} (€{values.mean():.0f}±{values.std(ddof=1):.0f})")
    ax.fill_between(x, density, alpha=FILL_ALPHA, color=color)
    ax.axvline(values.mean(), color=color, linestyle="--", linewidth=1.0, alpha=0.7)


def build_figure(midpc: np.ndarray, miqp_gl: np.ndarray, miqp_pw: np.ndarray) -> plt.Figure:
    """Two side-by-side density panels: MI-DPC against each MIQP baseline.

    Every input holds one ex-post profit per benchmark day; the legend gives the
    mean and the standard deviation across days.
    """
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.0))
    panels = (
        ("(a) MI-DPC vs. MIQP-GL", "MIQP-GL", miqp_gl, C_MIQP_GL),
        ("(b) MI-DPC vs. MIQP-PW", "MIQP-PW", miqp_pw, C_MIQP_PW),
    )
    frame = themed("black", "fg")
    for ax, (title, baseline_label, baseline, color) in zip(axes, panels):
        _draw_method(ax, np.asarray(midpc, dtype=float), "MI-DPC", C_MIDPC)
        _draw_method(ax, np.asarray(baseline, dtype=float), baseline_label, color)
        ax.set_xlabel("Ex-post profit (EUR/day)")
        ax.set_ylabel("Density")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.grid(True, alpha=0.25, linestyle=":", linewidth=0.4, color=themed("0.5", "muted"))
        cleanup_axes(ax, grid=False)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.38)  # headroom so the legend clears the curves
        ax.legend(loc="upper center", ncol=2, frameon=True, framealpha=0.95, edgecolor=frame, fancybox=False,
                  fontsize=6.5, columnspacing=1.0, handlelength=1.4)
        ax.set_title(title, pad=5)
    fig.tight_layout(pad=0.3, w_pad=0.5)
    return fig


def make_figure(output_dir: pathlib.Path) -> pathlib.Path:
    """Build and save the profit distribution figure. Returns path to PDF."""
    rows = load_ablation_runs_csv(_RUNS_CSV)
    dpc_rows = filter_runs(rows, study="architecture", variant="transformer")
    # One value per benchmark day for every method: MI-DPC is averaged over its seeds.
    midpc = np.stack(_load_dpc_profits_per_seed(dpc_rows), axis=0).mean(axis=0)
    fig = build_figure(midpc, _load_miqp_profits(_MIQP_GL_CSV), _load_miqp_profits(_MIQP_PW_CSV))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "profit_distribution.pdf"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.01)
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
