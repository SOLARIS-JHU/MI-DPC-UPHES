"""Generate corrected rerun configs/commands from saved benchmark runs."""

from __future__ import annotations

import json
import os
from pathlib import Path


def _arg(flag: str, value) -> list[str]:
    if value is None:
        return []
    return [flag, str(value)]


def build_command_from_saved_config(
    cfg: dict,
    *,
    output_root: str,
    new_prefix: str,
    dynamics_name: str | None = None,
    inverse_pkl_path: str | None = None,
) -> list[str]:
    dynamics_name = dynamics_name or cfg.get("dynamics_name", "batch")
    cmd = [
        "python",
        "-m",
        "DPC.experiments.benchmark_tuner",
        "--run-prefix",
        new_prefix,
        "--output-root",
        output_root,
        "--architectures",
        cfg["architecture"],
        "--samplers",
        cfg["sampler"],
        "--batch-sizes",
        str(cfg["batch_size"]),
        "--optimizers",
        cfg["optimizer"],
        "--schedulers",
        cfg["scheduler"],
        "--tau-schedules",
        cfg["tau_schedule"],
        "--tau-ends",
        str(cfg["tau_end"]),
        "--lrs",
        str(cfg["lr"]),
        "--grad-clips",
        str(cfg["grad_clip"]),
        "--weight-decays",
        str(cfg["weight_decay"]),
        "--vol-balance-modes",
        cfg["vol_balance_mode"],
        "--vol-balance-weights",
        str(cfg["vol_balance_weight"]),
        "--vol-surplus-factors",
        str(cfg["vol_surplus_factor"]),
        "--head-penalties",
        str(cfg.get("head_penalty", 50.0)),
        "--vol-traj-penalties",
        str(cfg.get("vol_traj_penalty", 50.0)),
        "--h-lb-scales",
        str(cfg.get("h_lb_scale", 1.0)),
        "--h-ub-scales",
        str(cfg.get("h_ub_scale", 1.0)),
        "--vol-lb-scales",
        str(cfg.get("vol_lb_scale", 1.0)),
        "--vol-ub-scales",
        str(cfg.get("vol_ub_scale", 1.0)),
        "--seeds",
        str(cfg["seed"]),
        "--epochs",
        str(cfg.get("epochs", 25)),
        "--c-op",
        "0.4",
        "--si-penalty-weights",
        str(cfg.get("si_penalty_weight", 1.0)),
        "--target-vol-penalty-weights",
        str(cfg.get("target_vol_penalty_weight", 1.0)),
        "--hidden-size",
        str(cfg["hidden_size"]),
        "--num-layers",
        str(cfg["num_layers"]),
        "--dropout",
        str(cfg["dropout"]),
        "--nhead",
        str(cfg["nhead"]),
        "--dim-ff",
        str(cfg["dim_ff"]),
        "--physics",
        cfg.get("physics_mode", "nonlinear"),
        "--dynamics",
        dynamics_name,
    ]
    inverse_name = cfg.get("inverse_pkl_name")
    inverse_path = inverse_pkl_path
    if inverse_path is None and inverse_name:
        inverse_path = f"Data/UPCs/{inverse_name}"
    cmd.extend(_arg("--inverse-pkl", inverse_path))
    return cmd


def rerun_from_eval_config(source_run_dir, *, new_prefix="cop04_rerun", output_root="DPC/outputs/benchmark_suite"):
    """Emit a corrected rerun command/manifest from a saved run directory."""
    source_run_dir = Path(source_run_dir)
    manifest_dir = Path(output_root) / f"{new_prefix}__{source_run_dir.name or 'source'}"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    eval_path = source_run_dir / "eval_results.json"
    payload = {
        "source_run_dir": str(source_run_dir),
        "output_root": str(output_root),
        "new_prefix": new_prefix,
    }

    if eval_path.exists():
        with eval_path.open() as f:
            data = json.load(f)
        cfg = data.get("config", {})
        payload["source_config"] = cfg
        payload["command"] = build_command_from_saved_config(cfg, output_root=output_root, new_prefix=new_prefix)
    else:
        payload["missing_eval_results"] = True
        payload["command"] = []

    manifest_path = manifest_dir / "rerun_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(payload, f, indent=2)

    shell_path = manifest_dir / "rerun_command.sh"
    with shell_path.open("w") as f:
        f.write("#!/usr/bin/env bash\n")
        if payload["command"]:
            f.write(" ".join(payload["command"]) + "\n")
        else:
            f.write("# eval_results.json not found; fill in the command manually.\n")

    return manifest_path


__all__ = ["build_command_from_saved_config", "rerun_from_eval_config"]
