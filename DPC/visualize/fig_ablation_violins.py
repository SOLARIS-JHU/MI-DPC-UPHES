"""Figure 5 — Ablation violin plots (3 panels: Architecture, Temperature, Dynamics).

Usage:
    python3 -m DPC.visualize.fig_ablation_violins [--output-dir DIR] [--data-dir DIR]

Defaults:
    --output-dir  figs/ (at the repository root)
    --data-dir    DPC/outputs/benchmark_suite
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

from DPC.visualize.data import filter_runs, load_ablation_runs_csv
from DPC.visualize.style import (
    apply_style,
    cleanup_axes,
    C_MIQP_PW,
    C_MIQP_GL,
    C_MIDPC,
    C_BILSTM,
    C_MLP,
    C_CNN,
    C_FIXED,
    C_STEP,
    FULL_WIDTH,
    MIQP_PW_MEAN_EXPOST,
    MIQP_GL_MEAN_EXPOST,
    FIGS_OUT,
    BENCH_DIR,
)


def _load_metric(rows, key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) not in (None, "")]


def _format_stat(value: float) -> str:
    value = float(value)
    abs_value = abs(value)
    if abs_value >= 100:
        return f"{value:.0f}"
    if abs_value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def load_all(data_dir: str | Path) -> dict[str, dict[str, list[float]]]:
    """Return a dict mapping group key -> retained metrics."""
    data_dir = Path(data_dir)
    runs = load_ablation_runs_csv(data_dir / "ABLATION_47SEED_RUNS.csv")

    results = {}
    for variant, key in [
        ("transformer", "transformer"),
        ("bilstm", "bilstm"),
        ("mlp", "mlp"),
        ("cnn", "cnn"),
    ]:
        rows = filter_runs(runs, study="architecture", variant=variant)
        results[key] = {
            "expost": _load_metric(rows, "mean_expost_profit"),
            "train_s": _load_metric(rows, "train_wall_s"),
            "infer_ms": _load_metric(rows, "benchmark_policy_inference_ms_per_day"),
        }

    for study, variant, key in [
        ("temperature", "annealed", "annealed"),
        ("temperature", "fixed_low", "fixed"),
        ("dynamics", "batch", "batch"),
        ("dynamics", "step", "step"),
    ]:
        rows = filter_runs(runs, study=study, variant=variant)
        results[key] = {
            "expost": _load_metric(rows, "mean_expost_profit"),
            "train_s": _load_metric(rows, "train_wall_s"),
            "infer_ms": _load_metric(rows, "benchmark_policy_inference_ms_per_day"),
        }

    # These are the same baseline experiment shown in different ablation panels.
    baseline = results["transformer"]
    for key in ["annealed", "batch"]:
        results[key]["expost"] = list(baseline["expost"])
        results[key]["train_s"] = list(baseline["train_s"])
        results[key]["infer_ms"] = list(baseline["infer_ms"])
    return results


def draw_violin(
    ax,
    x_pos: float,
    metrics: dict[str, list[float]],
    color: str,
    text_offset: float,
):
    """Draw a single violin with min/mean/max guides and compact stats text."""
    values = metrics.get("expost", [])
    if not values:
        return
    arr = np.array(values)

    parts = ax.violinplot(
        arr,
        positions=[x_pos],
        widths=0.6,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for pc in parts["bodies"]:
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
        pc.set_alpha(0.3)
        pc.set_linewidth(0.8)

    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    train_mean = float(np.mean(metrics.get("train_s", []))) if metrics.get("train_s") else float("nan")
    infer_mean = float(np.mean(metrics.get("infer_ms", []))) if metrics.get("infer_ms") else float("nan")

    ax.vlines(x_pos, vmin, vmax, colors=color, linewidth=0.8, linestyles="-", zorder=4)
    ax.hlines(vmin, x_pos - 0.12, x_pos + 0.12, colors=color, linewidth=0.9, linestyles="-", zorder=4)
    ax.hlines(mean, x_pos - 0.16, x_pos + 0.16, colors=color, linewidth=1.4, linestyles="-", zorder=4)
    ax.hlines(vmax, x_pos - 0.12, x_pos + 0.12, colors=color, linewidth=0.9, linestyles="-", zorder=4)
    ax.text(
        x_pos,
        vmax + 0.6 * text_offset,
        f"€{_format_stat(mean)} ± {_format_stat(std)}",
        color=color,
        fontsize=5.5,
        ha="center",
        va="bottom",
        zorder=5,
    )
    if np.isfinite(train_mean) and np.isfinite(infer_mean):
        ax.text(
            x_pos,
            vmin - 0.9 * text_offset,
            f"trained {_format_stat(train_mean)} s\ninfer {_format_stat(infer_mean)} ms",
            color=color,
            fontsize=5.1,
            ha="center",
            va="top",
            zorder=5,
        )


def _add_benchmark_lines(
    ax: plt.Axes,
    *,
    label_x: float,
    y_min: float,
    y_max: float,
    show_labels: bool,
) -> None:
    label_offset = 0.02 * (y_max - y_min)
    for value, color, label in (
        (MIQP_PW_MEAN_EXPOST, C_MIQP_PW, "MIQP-PW"),
        (MIQP_GL_MEAN_EXPOST, C_MIQP_GL, "MIQP-GL"),
    ):
        ax.axhline(
            value,
            color=color,
            linestyle="-",
            linewidth=0.8,
            zorder=2,
        )
        if show_labels:
            ax.text(
                label_x,
                value - label_offset,
                label,
                fontsize=6,
                color=color,
                va="top",
                ha="right",
            )


def _build_figure(data: dict[str, dict[str, list[float]]]) -> plt.Figure:
    apply_style()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(FULL_WIDTH, 2.0),
        gridspec_kw={"width_ratios": [4, 2, 2]},
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.16, wspace=0.22)

    panels = [
        {
            "ax": axes[0],
            "label": "(a) Architecture",
            "violins": [
                ("Transformer", "transformer", C_MIDPC),
                ("Bi-LSTM", "bilstm", C_BILSTM),
                ("MLP", "mlp", C_MLP),
                ("CNN", "cnn", C_CNN),
            ],
        },
        {
            "ax": axes[1],
            "label": "(b) Temperature",
            "violins": [
                ("Annealed", "annealed", C_MIDPC),
                ("Fixed", "fixed", C_FIXED),
            ],
        },
        {
            "ax": axes[2],
            "label": "(c) Dynamics",
            "violins": [
                ("Parallel", "batch", C_MIDPC),
                ("Sequential", "step", C_STEP),
            ],
        },
    ]

    all_vals = []
    for p in panels:
        for _, key, _ in p["violins"]:
            all_vals.extend(data.get(key, {}).get("expost", []))
    value_span = max(all_vals) - min(all_vals)
    text_offset = max(0.025 * value_span, 12.0)
    tick_step = 200.0
    y_min = 1250.0
    y_max = tick_step * np.ceil((max(all_vals) + 2.0 * text_offset) / tick_step)

    for panel_idx, panel in enumerate(panels):
        ax = panel["ax"]
        cleanup_axes(ax, grid=True)

        x_positions = list(range(1, len(panel["violins"]) + 1))
        tick_labels = []

        for x_pos, (label, key, color) in zip(x_positions, panel["violins"]):
            metrics = data.get(key, {})
            draw_violin(ax, x_pos, metrics, color, text_offset)
            tick_labels.append(label)

        label_x = x_positions[-1] + 0.55
        _add_benchmark_lines(
            ax,
            label_x=label_x,
            y_min=y_min,
            y_max=y_max,
            show_labels=panel_idx == 0,
        )

        if panel_idx == 0:
            ax.set_ylabel("Ex-post profit (EUR/day)", fontsize=9)
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(tick_labels, fontsize=7)
        ax.set_xlim(0.4, x_positions[-1] + 0.6)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(np.arange(y_min, y_max + 1e-9, tick_step))
        ax.set_title(panel["label"], loc="left", fontsize=8, fontweight="bold", pad=8)

    return fig


def make_figure(data: dict[str, list[float]], output_path: str | Path):
    fig = _build_figure(data)

    output_path = Path(output_path)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate ablation violin plot figure.")
    parser.add_argument(
        "--output-dir",
        default=Path(FIGS_OUT),
        type=Path,
        help="Directory to write the PDF",
    )
    parser.add_argument(
        "--data-dir",
        default=Path(BENCH_DIR),
        type=Path,
        help="Path to benchmark_suite directory",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_all(args.data_dir)

    for key, metrics in data.items():
        values = metrics.get("expost", [])
        print(f"  {key}: {len(values)} seeds, mean={np.mean(values):.1f}" if values else f"  {key}: 0 seeds")

    output_path = args.output_dir / "ablation_violins.pdf"
    make_figure(data, output_path)


if __name__ == "__main__":
    main()
