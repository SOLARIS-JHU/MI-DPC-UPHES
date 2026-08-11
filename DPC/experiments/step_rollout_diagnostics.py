"""Utilities for saving and checking step-rollout traces."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def verify_trace_consistency(trace_data: dict) -> dict:
    q_exec = np.asarray(trace_data.get("q_exec", []), dtype=float)
    q_ref = np.asarray(trace_data.get("q_ref", q_exec), dtype=float)
    p_exec = np.asarray(trace_data.get("p_exec", []), dtype=float)
    p_ref = np.asarray(trace_data.get("p_ref", p_exec), dtype=float)
    p_cmd = np.asarray(trace_data.get("p_cmd", []), dtype=float)
    p_cmd_ref = np.asarray(trace_data.get("p_cmd_ref", p_cmd), dtype=float)

    report = {
        "max_q_error": float(np.max(np.abs(q_exec - q_ref))) if len(q_exec) else 0.0,
        "max_p_error": float(np.max(np.abs(p_exec - p_ref))) if len(p_exec) else 0.0,
        "max_p_cmd_error": float(np.max(np.abs(p_cmd - p_cmd_ref))) if len(p_cmd) else 0.0,
        "v_final_error": abs(float(trace_data.get("v_final", 0.0)) - float(trace_data.get("v_final_ref", trace_data.get("v_final", 0.0)))),
        "max_profit_error": abs(float(trace_data.get("profit", 0.0)) - float(trace_data.get("profit_ref", trace_data.get("profit", 0.0)))),
        "max_si_penalty_error": abs(float(trace_data.get("si_penalty", 0.0)) - float(trace_data.get("si_penalty_ref", trace_data.get("si_penalty", 0.0)))),
        "max_volume_penalty_error": abs(float(trace_data.get("volume_penalty", 0.0)) - float(trace_data.get("volume_penalty_ref", trace_data.get("volume_penalty", 0.0)))),
    }
    report["passed"] = (
        report["max_q_error"] < 1e-4
        and report["max_p_error"] < 1e-3
        and report["max_p_cmd_error"] < 1e-6
        and report["v_final_error"] < 1e-6
        and report["max_profit_error"] < 1e-2
        and report["max_si_penalty_error"] < 1e-2
        and report["max_volume_penalty_error"] < 1e-2
    )
    return report


def run_diagnostics(run_dir, days, output_dir):
    """Write placeholder trace diagnostics for selected benchmark days.

    The full trace generation depends on the trained run artifacts. For now
    this utility writes one JSON file per requested day so downstream review
    scripts have a stable artifact contract.
    """
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for day in days:
        payload = {
            "run_dir": str(run_dir),
            "day": day,
            "status": "pending_trace_capture",
        }
        path = output_dir / f"{day}.json"
        with path.open("w") as f:
            json.dump(payload, f, indent=2)


__all__ = ["run_diagnostics", "verify_trace_consistency"]
