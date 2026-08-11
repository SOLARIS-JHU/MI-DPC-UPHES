"""Evaluation helpers for one-shot DPC policies."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from DPC.config import C_OP, load_system_params
from DPC.experiments.exact_stepwise_evaluator import score_exact_schedule


def evaluate_day_oneshot(
    problem,
    prices_24h,
    h_init,
    v_init,
    c_op=C_OP,
    *,
    system_params: dict | None = None,
    si_shortage_multiplier: float = -2.0,
    si_surplus_multiplier: float = -0.5,
    return_trace: bool = False,
) -> dict[str, Any]:
    """Evaluate a single 24-hour day using the shared exact hourly simulator."""
    prices_t = torch.tensor(prices_24h, dtype=torch.float32).reshape(1, len(prices_24h), 1)
    x0 = torch.tensor([[h_init, v_init]], dtype=torch.float32).unsqueeze(1)
    data = {"x": x0, "d": prices_t, "name": "test"}

    device = next(problem.parameters()).device
    data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}

    with torch.no_grad():
        output = problem(data)

    name = data["name"]
    aux = output[f"{name}_aux"][0].detach().cpu().numpy()
    u = output[f"{name}_u"][0].detach().cpu().numpy()

    if system_params is None:
        system_params = load_system_params("preprocess.pkl", device=device)
    else:
        system_params = dict(system_params)
    system_params["h_init"] = h_init
    system_params["v_init"] = v_init

    p_T = np.asarray(aux[:, 0], dtype=float)
    p_P = np.asarray(aux[:, 1], dtype=float)
    exact = score_exact_schedule(
        np.asarray(p_T + p_P, dtype=float),
        np.asarray(prices_24h, dtype=float),
        system_params,
        c_op=c_op,
        si_shortage_multiplier=si_shortage_multiplier,
        si_surplus_multiplier=si_surplus_multiplier,
        return_trace=return_trace,
        device=device,
    )

    result = {
        "h_traj": np.asarray(exact["h_traj"], dtype=float),
        "v_traj": np.asarray(exact["v_traj"], dtype=float),
        "p_T": p_T,
        "p_P": p_P,
        "p_net": np.asarray(exact["p_net"], dtype=float),
        "p_sim": np.asarray(exact["p_sim"], dtype=float),
        "mode": np.asarray(u[:, 2], dtype=float),
        "profit": float(exact["profit"]),
        "revenue": float(exact["revenue"]),
        "op_cost": float(exact["op_cost"]),
        "si_penalty": float(exact["si_penalty"]),
        "volume_penalty": float(exact["volume_penalty"]),
        "expost_profit": float(exact["expost_profit"]),
        "n_turbine": int(exact["n_turbine"]),
        "n_idle": int(exact["n_idle"]),
        "n_pump": int(exact["n_pump"]),
        "v_final": float(exact["v_final"]),
    }
    if return_trace:
        result["trace"] = exact["trace"]
    return result
