"""Utilities for replaying saved benchmark checkpoints on a single day."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from DPC.config import load_system_params
from DPC.dynamics import UPHESDynamicsBatch, UPHESDynamicsStep
from DPC.evaluate import evaluate_day_oneshot
from DPC.experiments import benchmark_tuner
from DPC.experiments.benchmark_data import load_benchmark_price_days
from DPC.system import build_oneshot_system


_EPOCH_CHECKPOINT_RE = re.compile(r"^policy_epoch(?P<epoch>\d+)\.pt$")
_EXPERIMENT_CONFIG_FIELDS = {field.name for field in fields(benchmark_tuner.ExperimentConfig)}


def discover_epoch_checkpoints(run_dir: Path) -> list[Path]:
    """Return policy_epochNNN.pt checkpoints sorted by epoch number."""
    run_dir = Path(run_dir)
    matches: list[tuple[int, Path]] = []
    for path in run_dir.glob("policy_epoch*.pt"):
        match = _EPOCH_CHECKPOINT_RE.match(path.name)
        if match is None:
            continue
        matches.append((int(match.group("epoch")), path))
    return [path for _, path in sorted(matches, key=lambda item: (item[0], item[1].name))]


def replay_epoch_dispatch(
    run_dir: Path,
    date: str,
    out_path: Path,
    *,
    device: str = "cpu",
    pkl_path: str = "preprocess.pkl",
    benchmark_csv: str = "Data/price_data_2024.csv",
    inverse_pkl: str | None = None,
) -> Path:
    """Replay every saved epoch checkpoint on one benchmark day and cache curves."""
    run_dir = Path(run_dir)
    out_path = Path(out_path)
    eval_path = run_dir / "eval_results.json"
    with eval_path.open(encoding="utf-8") as fh:
        eval_results = json.load(fh)

    cfg = _experiment_config_from_saved(eval_results.get("config", {}))
    saved_cfg = eval_results.get("config", {})
    dynamics_name = str(saved_cfg.get("dynamics_name") or "batch").lower()
    inverse_name = _resolve_inverse_pkl_path(inverse_pkl, saved_cfg.get("inverse_pkl_name"))

    torch_device = torch.device(device)
    system_params = load_system_params(
        pkl_path,
        device=torch_device,
        physics_mode=cfg.physics_mode,
        inverse_pkl_path=inverse_name,
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
    loss, _handles = benchmark_tuner.build_suite_loss(cfg, system_params)
    problem = benchmark_tuner.build_problem(nodes, loss).to(torch_device)

    benchmark_prices = load_benchmark_price_days(benchmark_csv)
    if date not in benchmark_prices:
        raise KeyError(f"Date {date!r} not found in benchmark CSV {benchmark_csv!r}")
    prices = np.asarray(benchmark_prices[date], dtype=float)

    checkpoints = discover_epoch_checkpoints(run_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No policy_epoch*.pt checkpoints found in {run_dir}")

    epochs: list[int] = []
    checkpoint_names: list[str] = []
    p_exec_curves: list[np.ndarray] = []
    head_curves: list[np.ndarray] = []

    for checkpoint in checkpoints:
        epoch = int(_EPOCH_CHECKPOINT_RE.match(checkpoint.name).group("epoch"))
        state = torch.load(checkpoint, map_location=torch_device)
        problem.load_state_dict(state)
        result = evaluate_day_oneshot(
            problem,
            prices,
            system_params["h_init"],
            system_params["v_init"],
            c_op=cfg.c_op,
            system_params=system_params,
            return_trace=True,
        )
        trace = result.get("trace") or {}
        p_exec = trace.get("p_exec")
        if p_exec is None:
            p_exec = result.get("p_sim")
        head = trace.get("h")
        if head is None:
            head = result.get("h_traj")
        p_exec_curves.append(np.asarray(p_exec, dtype=float))
        head_curves.append(np.asarray(head, dtype=float))
        epochs.append(epoch)
        checkpoint_names.append(checkpoint.name)

    metadata = {
        "run_dir": str(run_dir),
        "date": date,
        "device": device,
        "pkl_path": pkl_path,
        "benchmark_csv": benchmark_csv,
        "inverse_pkl": inverse_name,
        "dynamics_name": dynamics_name,
        "checkpoint_count": len(checkpoints),
        "checkpoint_files": checkpoint_names,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        date=np.array(date),
        epoch=np.asarray(epochs, dtype=np.int64),
        epoch_files=np.asarray(checkpoint_names, dtype=object),
        meta_json=np.array(json.dumps(metadata, sort_keys=True)),
        price=np.asarray(prices, dtype=np.float32),
        p_exec=np.asarray(p_exec_curves, dtype=np.float32),
        h=np.asarray(head_curves, dtype=np.float32),
    )
    return out_path


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


def main(argv: list[str] | None = None) -> Path:
    """CLI entrypoint for replaying checkpoints and writing a cache."""
    parser = argparse.ArgumentParser(description="Replay epoch checkpoints on a benchmark day.")
    parser.add_argument("--run-dir", required=True, type=Path, help="Directory containing policy_epoch*.pt files.")
    parser.add_argument("--date", required=True, help="Benchmark date in YYYY/MM/DD format.")
    parser.add_argument("--output", required=True, type=Path, help="Output NPZ path.")
    parser.add_argument("--device", default="cpu", help="Torch device to use.")
    parser.add_argument(
        "--pkl",
        default="preprocess.pkl",
        dest="pkl_path",
        help="Path to the system preprocessing pickle.",
    )
    parser.add_argument(
        "--benchmark-csv",
        default="Data/price_data_2024.csv",
        help="Benchmark price CSV used for replay.",
    )
    parser.add_argument(
        "--inverse-pkl",
        default=None,
        help="Override inverse preprocessing pickle path.",
    )
    args = parser.parse_args(argv)
    return replay_epoch_dispatch(
        args.run_dir,
        args.date,
        args.output,
        device=args.device,
        pkl_path=args.pkl_path,
        benchmark_csv=args.benchmark_csv,
        inverse_pkl=args.inverse_pkl,
    )


__all__ = ["discover_epoch_checkpoints", "replay_epoch_dispatch", "main"]


if __name__ == "__main__":
    main()
