"""Compact CDC-style training diagnostics figure for a single DPC run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from DPC.visualize.fig_schedule_evolution import (
    AXIS_FONT_SIZE,
    BASE_FIGURE_HEIGHT,
    FIGURE_HEIGHT_SCALE,
    FIGURE_WIDTH_SCALE,
    LEGEND_FONT_SIZE,
    LINEWIDTH_MAIN,
    TICK_FONT_SIZE,
    load_cache_metadata,
)
from DPC.visualize.data import filter_runs, load_ablation_runs_csv
from DPC.visualize.style import C_MIDPC, C_STEP, FIGS_OUT, FULL_WIDTH, GRID_KW, apply_style, cleanup_axes


OUTPUT_PDF = "training_diagnostics.pdf"
BATCH_RUNS_CSV = "ABLATION_47SEED_RUNS.csv"
GRAD_COLOR = C_STEP
BATCH_GRAD_COLOR = C_MIDPC
LOSS_COLOR = "#7B1FA2"
TAU_COLOR = "#FF7F0E"


def _read_history_series(history_path: Path) -> dict[str, np.ndarray]:
    if not history_path.exists():
        raise FileNotFoundError(f"Could not find history.csv at {history_path}")

    epochs: list[int] = []
    grad_norm: list[float] = []
    losses: list[float] = []
    taus: list[float] = []
    with history_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                epochs.append(int(row["epoch"]))
                losses.append(float(row["loss"]))
                taus.append(float(row["tau"]))
                grad_norm.append(float(row["grad_norm"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Malformed history.csv in {history_path!s}") from exc

    if not epochs:
        raise ValueError(f"No epoch rows found in {history_path!s}")

    return {
        "epochs": np.asarray(epochs, dtype=int),
        "grad_norm": np.asarray(grad_norm, dtype=float),
        "loss": np.asarray(losses, dtype=float),
        "tau": np.asarray(taus, dtype=float),
    }


def _resolve_partner_history(run_dir: Path) -> Path | None:
    eval_results_path = run_dir / "eval_results.json"
    if not eval_results_path.exists():
        return None

    with eval_results_path.open(encoding="utf-8") as fh:
        cfg = json.load(fh).get("config", {})

    seed = cfg.get("seed")
    architecture = cfg.get("architecture", "transformer")
    if seed is None:
        return None

    bench_dir = run_dir.parent
    step_pattern = f"abl_dyn_step_{architecture}*seed{seed}*"
    candidates = sorted(path for path in bench_dir.glob(step_pattern) if (path / "history.csv").exists())
    if not candidates:
        return None
    return candidates[0] / "history.csv"


def _aggregate_mean(series: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not series:
        raise ValueError("No series to aggregate")
    min_len = min(len(values) for values in series)
    truncated = [np.asarray(values[:min_len], dtype=float) for values in series]
    stacked = np.vstack(truncated)
    return np.arange(1, min_len + 1, dtype=int), stacked.mean(axis=0)


def _load_mean_grad_norm_from_runs(bench_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    runs_csv = bench_dir / BATCH_RUNS_CSV
    if not runs_csv.exists():
        raise FileNotFoundError(f"Could not find retained-run table at {runs_csv}")

    runs = load_ablation_runs_csv(runs_csv)
    batch_rows = filter_runs(runs, study="dynamics", variant="batch")
    step_rows = filter_runs(runs, study="dynamics", variant="step")
    if not batch_rows or not step_rows:
        raise ValueError(f"Retained-run table {runs_csv} does not include both dynamics slices.")

    def load_series(rows: list[dict[str, str]]) -> list[np.ndarray]:
        series: list[np.ndarray] = []
        for row in rows:
            history = _read_history_series(bench_dir / row["run_dir"] / "history.csv")
            series.append(history["grad_norm"])
        return series

    batch_epochs, batch_mean = _aggregate_mean(load_series(batch_rows))
    step_epochs, step_mean = _aggregate_mean(load_series(step_rows))
    if not np.array_equal(batch_epochs, step_epochs):
        raise ValueError("Batch and step mean gradient histories do not share the same epoch grid.")
    return batch_epochs, batch_mean, step_mean


def load_training_payload(cache_path: str | Path) -> dict[str, np.ndarray]:
    meta = load_cache_metadata(cache_path)
    run_dir = Path(str(meta["run_dir"]))
    primary = _read_history_series(run_dir / "history.csv")

    bench_dir = run_dir.parent
    try:
        grad_epochs, grad_norm_batch, grad_norm_step = _load_mean_grad_norm_from_runs(bench_dir)
    except (FileNotFoundError, ValueError):
        partner_history_path = _resolve_partner_history(run_dir)
        partner = _read_history_series(partner_history_path) if partner_history_path is not None else primary
        if not np.array_equal(primary["epochs"], partner["epochs"]):
            raise ValueError("Batch and step training histories do not share the same epoch grid.")
        grad_epochs = primary["epochs"]
        grad_norm_batch = primary["grad_norm"]
        grad_norm_step = partner["grad_norm"]

    common_len = min(len(primary["epochs"]), len(grad_epochs))

    return {
        "epochs": primary["epochs"][:common_len],
        "grad_norm_batch": grad_norm_batch[:common_len],
        "grad_norm_step": grad_norm_step[:common_len],
        "loss": primary["loss"][:common_len],
        "tau": primary["tau"][:common_len],
    }


def _epoch_xticks(epochs: np.ndarray) -> np.ndarray:
    epochs = np.asarray(epochs, dtype=int)
    if epochs.size == 0:
        return epochs
    start = int(epochs.min())
    stop = int(epochs.max())
    if stop <= start:
        return np.asarray([start], dtype=int)
    count = min(5, stop - start + 1)
    return np.unique(np.linspace(start, stop, num=count, dtype=int))


def build_figure(payload: dict[str, object]) -> plt.Figure:
    apply_style()

    epochs = np.asarray(payload["epochs"], dtype=float)
    grad_norm_batch = np.asarray(payload["grad_norm_batch"], dtype=float)
    grad_norm_step = np.asarray(payload["grad_norm_step"], dtype=float)
    losses = np.asarray(payload["loss"], dtype=float)
    tau = np.asarray(payload["tau"], dtype=float)

    fig, (ax_grad, ax_loss) = plt.subplots(
        2,
        1,
        figsize=(FULL_WIDTH * FIGURE_WIDTH_SCALE, BASE_FIGURE_HEIGHT * FIGURE_HEIGHT_SCALE),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.08},
    )
    ax_tau = ax_loss.twinx()

    ax_grad.plot(epochs, grad_norm_batch, color=BATCH_GRAD_COLOR, linewidth=LINEWIDTH_MAIN, label="Batch")
    ax_grad.plot(epochs, grad_norm_step, color=GRAD_COLOR, linewidth=LINEWIDTH_MAIN, label="Step")
    ax_grad.set_yscale("log")
    ax_grad.set_ylim(bottom=10e3)
    ax_grad.set_ylabel("Gradient norm", color=GRAD_COLOR, fontsize=AXIS_FONT_SIZE)
    ax_grad.tick_params(axis="y", colors=GRAD_COLOR, labelsize=TICK_FONT_SIZE)
    ax_grad.tick_params(axis="x", which="both", labelbottom=False)
    ax_grad.legend(loc="lower center", frameon=False, fontsize=LEGEND_FONT_SIZE, ncol=2)

    ax_loss.plot(epochs, losses, color=LOSS_COLOR, linewidth=LINEWIDTH_MAIN)
    ax_loss.set_ylabel("Loss", color=LOSS_COLOR, fontsize=AXIS_FONT_SIZE)
    ax_loss.yaxis.set_label_coords(0.02, 0.5)
    ax_loss.tick_params(axis="y", colors=LOSS_COLOR, labelsize=TICK_FONT_SIZE)
    ax_loss.set_xlabel("Epoch", fontsize=AXIS_FONT_SIZE)

    ax_tau.plot(epochs, tau, color=TAU_COLOR, linewidth=LINEWIDTH_MAIN)
    ax_tau.set_ylabel("Gumbel-Softmax\ntempreture τ", color=TAU_COLOR, fontsize=AXIS_FONT_SIZE, labelpad=1.5)
    ax_tau.tick_params(axis="y", colors=TAU_COLOR, labelsize=TICK_FONT_SIZE)
    ax_tau.spines["top"].set_visible(False)

    xticks = _epoch_xticks(epochs)
    for ax in (ax_grad, ax_loss):
        cleanup_axes(ax)
        ax.xaxis.grid(True, **GRID_KW)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE)
        if epochs.size > 1:
            ax.set_xlim(float(epochs.min()), float(epochs.max()))
        ax.set_xticks(xticks)

    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.14, top=0.98, hspace=0.08)
    return fig


def make_figure(
    output_dir: str | Path,
    *,
    cache_path: str | Path,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = load_training_payload(cache_path)
    fig = build_figure(payload)
    out_path = output_dir / OUTPUT_PDF
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Generate a compact DPC training diagnostics figure.")
    parser.add_argument(
        "--cache",
        type=Path,
        required=True,
        help=(
            "Path to an epoch_dispatch_trace.npz produced by DPC.visualize.epoch_replay "
            "from a training run with epoch checkpoints enabled (benchmark_tuner's "
            "--save-all-epochs flag)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(FIGS_OUT),
        help="Directory for the output PDF.",
    )
    args = parser.parse_args(argv)
    return make_figure(args.output_dir, cache_path=args.cache)


if __name__ == "__main__":
    main()
