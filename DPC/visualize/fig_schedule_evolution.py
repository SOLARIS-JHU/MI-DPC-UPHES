"""CDC-style DPC representative schedule figure without MIQP or epoch overlays."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

from DPC.config import load_system_params
from DPC.dynamics import UPHESDynamicsBatch, UPHESDynamicsStep
from DPC.evaluate import evaluate_day_oneshot
from DPC.experiments import benchmark_tuner
from DPC.experiments.benchmark_data import load_benchmark_price_days
from DPC.system import build_oneshot_system
from DPC.visualize.epoch_replay import discover_epoch_checkpoints
from DPC.visualize.style import (
    FIGS_OUT,
    FULL_WIDTH,
    GRID_KW,
    apply_style,
    cleanup_axes,
)


OUTPUT_PDF = "schedule_evolution.pdf"
POWER_COLOR = "#D62728"
PRICE_COLOR = "#1F77B4"
HEAD_COLOR = "#E8850C"
VOLUME_COLOR = "#2CA02C"
FEASIBLE_ALPHA = 0.14
BOUND_ALPHA = 0.45
LINEWIDTH_MAIN = 1.15
LINEWIDTH_BOUND = 0.65
LINEWIDTH_REF = 0.7
FIGURE_WIDTH_SCALE = 0.54
FIGURE_HEIGHT_SCALE = 0.82
AXIS_FONT_SIZE = 8
TICK_FONT_SIZE = 7
LEGEND_FONT_SIZE = 7
BASE_FIGURE_HEIGHT = 2.8
VOLUME_UNIT_SCALE = 1e5

_EXPERIMENT_CONFIG_FIELDS = {field.name for field in fields(benchmark_tuner.ExperimentConfig)}


def load_cache_metadata(cache_path: str | Path) -> dict[str, object]:
    cache_path = Path(cache_path)
    with np.load(cache_path, allow_pickle=True) as cache:
        meta = json.loads(str(cache["meta_json"].item()))
        meta.setdefault("date", str(np.asarray(cache["date"]).item()))
    return meta


def load_final_cached_trace(cache_path: str | Path) -> tuple[str, np.ndarray, np.ndarray]:
    cache_path = Path(cache_path)
    with np.load(cache_path, allow_pickle=True) as cache:
        date = str(np.asarray(cache["date"]).item())
        power = np.asarray(cache["p_exec"], dtype=float)
        head = np.asarray(cache["h"], dtype=float)
    if power.ndim == 2:
        power = power[-1]
    if head.ndim == 2:
        head = head[-1]
    return date, power, head


def _experiment_config_from_saved(saved_cfg: dict) -> benchmark_tuner.ExperimentConfig:
    filtered = {}
    for key in _EXPERIMENT_CONFIG_FIELDS:
        if key not in saved_cfg:
            continue
        value = saved_cfg[key]
        if key == "mlp_hidden_sizes" and value is not None:
            value = tuple(value)
        filtered[key] = value
    return benchmark_tuner.ExperimentConfig(**filtered)


def _resolve_inverse_pkl_path(
    explicit_inverse_pkl: str | None,
    saved_inverse_name: str | None,
) -> str | None:
    if explicit_inverse_pkl is not None:
        return explicit_inverse_pkl
    if not saved_inverse_name:
        return None

    saved_path = Path(saved_inverse_name)
    if not saved_path.is_absolute() and len(saved_path.parts) == 1:
        return str(Path("Data") / "UPCs" / saved_path)
    return saved_inverse_name


def _resolve_final_checkpoint(run_dir: Path, meta: dict[str, object]) -> Path:
    checkpoint_files = meta.get("checkpoint_files")
    if isinstance(checkpoint_files, list) and checkpoint_files:
        candidate = run_dir / str(checkpoint_files[-1])
        if candidate.exists():
            return candidate

    checkpoints = discover_epoch_checkpoints(run_dir)
    if checkpoints:
        return checkpoints[-1]

    candidate = run_dir / "policy_best.pt"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not locate a final checkpoint in {run_dir}")


def _build_problem_from_run(
    run_dir: Path,
    *,
    pkl_path: str,
    inverse_pkl: str | None,
    device: str,
) -> tuple[benchmark_tuner.ExperimentConfig, dict, torch.nn.Module]:
    eval_path = run_dir / "eval_results.json"
    with eval_path.open(encoding="utf-8") as fh:
        eval_results = json.load(fh)

    cfg = _experiment_config_from_saved(eval_results.get("config", {}))
    saved_cfg = eval_results.get("config", {})
    dynamics_name = str(saved_cfg.get("dynamics_name") or "batch").lower()

    torch_device = torch.device(device)
    system_params = load_system_params(
        pkl_path,
        device=torch_device,
        physics_mode=cfg.physics_mode,
        inverse_pkl_path=inverse_pkl,
    )

    if dynamics_name == "step":
        dynamics = UPHESDynamicsStep(system_params)
    else:
        dynamics = UPHESDynamicsBatch(system_params)

    net_cont, net_int = benchmark_tuner.build_oneshot_architecture(
        architecture=cfg.architecture,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        nhead=cfg.nhead,
        dim_ff=cfg.dim_ff,
        cnn_kernel_size=cfg.cnn_kernel_size,
        mlp_hidden_sizes=tuple(cfg.mlp_hidden_sizes),
    )
    net_cont = net_cont.to(torch_device)
    net_int = net_int.to(torch_device)
    ste_fn = benchmark_tuner.create_oneshot_ste(cfg.ste_method, tau=cfg.tau_start).to(torch_device)
    nodes = build_oneshot_system(dynamics, net_cont, net_int, ste_fn)
    loss, _ = benchmark_tuner.build_suite_loss(cfg, system_params)
    problem = benchmark_tuner.build_problem(nodes, loss).to(torch_device)
    return cfg, system_params, problem


def compute_feasible_power_bounds(system_params: dict, head: np.ndarray) -> dict[str, np.ndarray]:
    head_t = torch.as_tensor(np.asarray(head, dtype=float), dtype=torch.float32)
    return {
        "pos_min": system_params["pos_min"](head_t).detach().cpu().numpy(),
        "pos_max": system_params["pos_max"](head_t).detach().cpu().numpy(),
        "neg_min": system_params["neg_min"](head_t).detach().cpu().numpy(),
        "neg_max": system_params["neg_max"](head_t).detach().cpu().numpy(),
    }


def load_schedule_payload(cache_path: str | Path, device: str = "cpu") -> dict[str, object]:
    meta = load_cache_metadata(cache_path)
    run_dir = Path(str(meta["run_dir"]))
    date, power, head = load_final_cached_trace(cache_path)
    pkl_path = str(meta.get("pkl_path", "preprocess.pkl"))
    benchmark_csv = str(meta.get("benchmark_csv", "Data/price_data_2024.csv"))

    with (run_dir / "eval_results.json").open(encoding="utf-8") as fh:
        saved_results = json.load(fh)
    inverse_pkl = _resolve_inverse_pkl_path(
        meta.get("inverse_pkl") if isinstance(meta.get("inverse_pkl"), str) else None,
        saved_results.get("config", {}).get("inverse_pkl_name"),
    )

    torch_device = torch.device(device)
    system_params = load_system_params(
        pkl_path,
        device=torch_device,
        physics_mode="nonlinear",
        inverse_pkl_path=inverse_pkl,
    )

    benchmark_prices = load_benchmark_price_days(benchmark_csv)
    if date not in benchmark_prices:
        raise KeyError(f"Date {date!r} not found in benchmark CSV {benchmark_csv!r}")
    prices = np.asarray(benchmark_prices[date], dtype=float)
    h_to_v_low = system_params["h_to_v_low"]
    volume = h_to_v_low(torch.as_tensor(head, dtype=torch.float32)).detach().cpu().numpy()
    feasible_power_bounds = compute_feasible_power_bounds(system_params, head)

    return {
        "date": date,
        "hours": np.arange(prices.shape[0], dtype=int),
        "power": power,
        "price": prices,
        "head": head,
        "volume": volume,
        "min_volume": 0.0,
        "max_volume": float(system_params.get("max_vol_low", np.max(volume))),
        "target_volume": float(system_params["target_vol_low"]),
        "feasible_power_bounds": feasible_power_bounds,
    }


def build_figure(payload: dict[str, object]) -> plt.Figure:
    apply_style()

    hours = np.asarray(payload["hours"], dtype=float)
    power = np.asarray(payload["power"], dtype=float)
    price = np.asarray(payload["price"], dtype=float)
    head = np.asarray(payload["head"], dtype=float)
    volume = np.asarray(payload["volume"], dtype=float)
    min_volume = float(payload.get("min_volume", np.min(volume)))
    max_volume = float(payload.get("max_volume", np.max(volume)))
    target_volume = float(payload["target_volume"])
    feasible = payload["feasible_power_bounds"]

    pos_min = np.asarray(feasible["pos_min"], dtype=float)
    pos_max = np.asarray(feasible["pos_max"], dtype=float)
    neg_min = np.asarray(feasible["neg_min"], dtype=float)
    neg_max = np.asarray(feasible["neg_max"], dtype=float)

    fig, (ax_power, ax_head) = plt.subplots(
        2,
        1,
        figsize=(FULL_WIDTH, BASE_FIGURE_HEIGHT),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.08},
    )

    ax_price = ax_power.twinx()
    ax_volume = ax_head.twinx()
    ax_price.set_zorder(1)
    ax_power.set_zorder(2)
    ax_power.patch.set_alpha(0.0)

    ax_power.fill_between(hours, pos_min, pos_max, step="mid", color=POWER_COLOR, alpha=FEASIBLE_ALPHA, zorder=0)
    ax_power.fill_between(hours, neg_min, neg_max, step="mid", color=POWER_COLOR, alpha=FEASIBLE_ALPHA, zorder=0)
    for bound in (pos_min, pos_max, neg_min, neg_max):
        ax_power.step(hours, bound, where="mid", color=POWER_COLOR, linewidth=LINEWIDTH_BOUND, alpha=BOUND_ALPHA, zorder=1)
    ax_power.axhline(0.0, color=POWER_COLOR, linewidth=LINEWIDTH_REF, alpha=0.65, zorder=2)
    ax_price.step(hours, price, where="mid", color=PRICE_COLOR, linewidth=LINEWIDTH_MAIN, zorder=0)
    ax_power.step(hours, power, where="mid", color=POWER_COLOR, linewidth=LINEWIDTH_MAIN, zorder=10)

    ax_head.plot(hours, head, color=HEAD_COLOR, linewidth=LINEWIDTH_MAIN, zorder=3)
    ax_volume.plot(hours, volume, color=VOLUME_COLOR, linewidth=LINEWIDTH_MAIN, zorder=3)
    ax_volume.axhline(
        target_volume,
        color=VOLUME_COLOR,
        linestyle=(0, (4, 2)),
        linewidth=LINEWIDTH_REF,
        zorder=2,
        label="Target volume",
    )
    y_pad = max((max_volume - min_volume) * 0.05, 1.0)
    ax_volume.set_ylim(min_volume - y_pad, max_volume + y_pad)
    ax_volume.legend(loc="upper right", frameon=False, fontsize=LEGEND_FONT_SIZE)

    for ax in (ax_power, ax_head):
        cleanup_axes(ax)
        ax.xaxis.grid(True, **GRID_KW)
        ax.set_axisbelow(True)
        ax.set_xlim(0, 23)
        ax.set_xticks(range(0, 24, 4))

    ax_power.tick_params(axis="x", which="both", labelbottom=False)
    ax_power.set_ylabel("Power (MW)", color=POWER_COLOR, fontsize=AXIS_FONT_SIZE)
    ax_power.tick_params(axis="y", colors=POWER_COLOR, labelsize=TICK_FONT_SIZE)
    ax_price.set_ylabel("Price (€/MWh)", color=PRICE_COLOR, fontsize=AXIS_FONT_SIZE)
    ax_price.tick_params(axis="y", colors=PRICE_COLOR, labelsize=TICK_FONT_SIZE)
    ax_price.spines["top"].set_visible(False)

    ax_head.set_ylabel("Head (m)", color=HEAD_COLOR, fontsize=AXIS_FONT_SIZE)
    ax_head.tick_params(axis="y", colors=HEAD_COLOR, labelsize=TICK_FONT_SIZE)
    ax_head.set_xlabel("Hour", fontsize=AXIS_FONT_SIZE)
    ax_head.tick_params(axis="x", labelsize=TICK_FONT_SIZE)
    ax_volume.set_ylabel("Volume (m$^3$)", color=VOLUME_COLOR, fontsize=AXIS_FONT_SIZE)
    ax_volume.tick_params(axis="y", colors=VOLUME_COLOR, labelsize=TICK_FONT_SIZE)
    ax_volume.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax_volume.spines["top"].set_visible(False)

    fig.subplots_adjust(left=0.10, right=0.90, bottom=0.14, top=0.98, hspace=0.08)
    return fig


def make_figure(
    output_dir: str | Path,
    *,
    cache_path: str | Path,
    device: str = "cpu",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = load_schedule_payload(cache_path, device=device)
    fig = build_figure(payload)
    out_path = output_dir / OUTPUT_PDF
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Generate a DPC representative schedule evolution figure.")
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
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device used to rebuild and evaluate the final DPC policy.",
    )
    args = parser.parse_args(argv)
    return make_figure(args.output_dir, cache_path=args.cache, device=args.device)


if __name__ == "__main__":
    main()
