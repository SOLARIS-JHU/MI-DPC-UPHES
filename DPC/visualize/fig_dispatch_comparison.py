"""Figure: MIQP-PW dispatch schedule for a representative day.

Usage
-----
    python3 -m DPC.visualize.fig_dispatch_comparison [--date YYYY/MM/DD] [--output-dir DIR]

Selects a representative day from the benchmark CSV unless ``--date`` is
provided. The retained-study heuristic prefers zero volume penalty and then
higher ex-post profit.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Style & shared paths
# ---------------------------------------------------------------------------
from DPC.visualize.style import (
    apply_style, cleanup_axes,
    C_MIQP_PW, C_TURBINE, C_IDLE, C_PUMP, COL_WIDTH,
    FIGS_OUT, MIQP_ROOT,
)
from DPC.visualize.data import pick_representative_day

BENCHMARK_CSV = MIQP_ROOT / "MIQP_piecewise" / "MIQP_piecewise_benchmark.csv"
RESULTS_CSV   = MIQP_ROOT / "MIQP_piecewise" / "MIQP_piecewise_results.csv"

# Reference volume (m³) — used as the "initial volume" horizontal reference
V_INIT = 370_000.0

# ---------------------------------------------------------------------------
# Data loading helpers  (no pandas)
# ---------------------------------------------------------------------------


def load_best_date(benchmark_csv: Path) -> str:
    """Return the representative date string from the benchmark CSV."""
    with benchmark_csv.open(newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if not rows:
        raise RuntimeError("Could not find any rows in benchmark CSV.")
    return pick_representative_day(rows)


def load_hourly(results_csv: Path) -> dict[str, dict[str, np.ndarray]]:
    """Load results CSV into {date: {column: np.ndarray(24)}}."""
    day_rows: dict[str, list[dict]] = defaultdict(list)

    with results_csv.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            day_rows[row["date"].strip()].append(row)

    result: dict[str, dict[str, np.ndarray]] = {}
    for date, rows in day_rows.items():
        # Sort by hour to be safe
        rows_sorted = sorted(rows, key=lambda r: int(r["hour"]))
        cols = [c for c in rows_sorted[0].keys() if c != "date"]
        result[date] = {
            c: np.array([float(r[c]) for r in rows_sorted])
            for c in cols
        }
    return result


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------

def build_figure(date: str, hourly: dict[str, np.ndarray]) -> plt.Figure:
    apply_style()

    hours   = hourly["hour"].astype(int)          # 0..23
    power   = hourly["power"]                      # MW (positive=turbine, negative=pump)
    volume  = hourly["volume"]                     # m³
    price   = hourly["price"]                      # EUR/MWh

    # Step-plot x values: hour centres for "mid" steps
    x = hours                                      # 0..23

    # ── Figure layout ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        4, 1,
        figsize=(COL_WIDTH, 4.5),
        gridspec_kw={"height_ratios": [1, 1.5, 1.5, 0.4], "hspace": 0.08},
        sharex=True,
    )
    ax_price, ax_power, ax_vol, ax_mode = axes

    # ── Row 1: Price ────────────────────────────────────────────────────────
    ax_price.step(x, price, where="mid", color="black", linewidth=1.0)
    ax_price.set_ylabel("Price\n(EUR/MWh)")
    cleanup_axes(ax_price)

    # ── Row 2: Power bars ───────────────────────────────────────────────────
    bar_colors = []
    for p in power:
        if p > 0.5:
            bar_colors.append(C_TURBINE)
        elif p < -0.5:
            bar_colors.append(C_PUMP)
        else:
            bar_colors.append(C_IDLE)

    ax_power.bar(x, power, color=bar_colors, width=0.8, zorder=2)
    ax_power.axhline(0, color="#888888", linewidth=0.6, zorder=1)
    ax_power.set_ylabel("Power\n(MW)")
    cleanup_axes(ax_power)

    # ── Row 3: Volume ───────────────────────────────────────────────────────
    ax_vol.step(x, volume, where="mid", color=C_MIQP_PW, linewidth=1.5)
    ax_vol.axhline(V_INIT, color="#888888", linestyle="--", linewidth=0.8,
                   label=f"$v_0$={V_INIT/1e3:.0f}k m$^3$")
    ax_vol.set_ylabel("Volume\n(m$^3$)")
    ax_vol.legend(loc="upper right", fontsize=7, frameon=False)
    cleanup_axes(ax_vol)

    # ── Row 4: Mode bars ────────────────────────────────────────────────────
    ax_mode.set_ylim(0, 1)
    ax_mode.set_yticks([])
    ax_mode.set_ylabel("Mode", labelpad=8)
    ax_mode.spines["top"].set_visible(False)
    ax_mode.spines["right"].set_visible(False)
    ax_mode.spines["left"].set_visible(False)
    ax_mode.yaxis.grid(False)

    for h, p in zip(hours, power):
        if p > 0.5:
            color = C_TURBINE
        elif p < -0.5:
            color = C_PUMP
        else:
            color = C_IDLE
        rect = mpatches.Rectangle((h - 0.4, 0), 0.8, 1.0,
                                   facecolor=color, edgecolor="none")
        ax_mode.add_patch(rect)

    # ── X axis ──────────────────────────────────────────────────────────────
    ax_mode.set_xlabel("Hour")
    ax_mode.set_xlim(-0.5, 23.5)
    ax_mode.set_xticks(range(0, 24, 4))

    # ── Title ───────────────────────────────────────────────────────────────
    fig.suptitle(f"MIQP-PW schedule — {date}", fontsize=9)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MIQP-PW dispatch figure.")
    parser.add_argument("--date", default=None,
                        help="Day to plot (YYYY/MM/DD). Defaults to the representative day heuristic.")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for output PDF. Defaults to figs/ next to script.")
    args = parser.parse_args()

    # Resolve output directory
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    else:
        out_dir = FIGS_OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    hourly_all = load_hourly(RESULTS_CSV)

    if args.date is not None:
        date = args.date
    else:
        date = load_best_date(BENCHMARK_CSV)

    if date not in hourly_all:
        sys.exit(f"ERROR: date {date!r} not found in results CSV. "
                 f"Available: {sorted(hourly_all)}")

    print(f"Plotting date: {date}")
    hourly = hourly_all[date]

    fig = build_figure(date, hourly)

    out_path = out_dir / "dispatch_comparison.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}  ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
