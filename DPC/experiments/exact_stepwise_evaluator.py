"""Shared exact hourly simulator for DPC evaluation.

This mirrors the MIQP ex-post simulation logic: execute commanded net power
hour by hour against the current head, clamp to feasible power bounds, and
cancel the hour if the implied reservoir update would leave the feasible
volume range.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from DPC.config import DT, ETA, G, RHO


def _as_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32)
    return torch.tensor(value, dtype=torch.float32, device=device)


def simulate_exact_hourly_schedule(
    p_cmd: np.ndarray,
    system_params: dict,
    *,
    device: torch.device | str | None = None,
) -> dict[str, np.ndarray | float]:
    """Run the shared exact hourly simulator on a commanded net-power schedule."""
    if device is None:
        device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    p_sched = _as_tensor(np.asarray(p_cmd, dtype=np.float32), device)
    horizon = int(p_sched.numel())

    pos_min = system_params["pos_min"]
    pos_max = system_params["pos_max"]
    neg_min = system_params["neg_min"]
    neg_max = system_params["neg_max"]
    upc_tur = system_params["UPC_poly_tur"]
    upc_pump = system_params["UPC_poly_pump"]
    v_low_to_h = system_params["v_low_to_h"]

    max_vol_low = _as_tensor(system_params["max_vol_low"], device)
    h_current = _as_tensor(system_params["h_init"], device)
    v_current = _as_tensor(system_params["v_init"], device)

    p_exec_hist = []
    q_exec_hist = []
    h_hist = []
    v_hist = []
    v_next_hist = []

    for t in range(horizon):
        p_current = p_sched[t]
        q_candidate = torch.zeros_like(p_current)
        p_clamped = torch.zeros_like(p_current)

        h_hist.append(h_current)
        v_hist.append(v_current)

        if p_current > 0.5:
            p_clamped = torch.clamp(p_current, min=pos_min(h_current), max=pos_max(h_current))
            q_candidate = upc_tur(p_clamped.unsqueeze(0), h_current.unsqueeze(0)).squeeze(0)
        elif p_current < -0.5:
            p_clamped = torch.clamp(p_current, min=neg_min(h_current), max=neg_max(h_current))
            q_candidate = upc_pump(p_clamped.unsqueeze(0), h_current.unsqueeze(0)).squeeze(0)

        candidate_v_next = v_current + q_candidate * DT
        out_of_bounds = (candidate_v_next > max_vol_low) | (candidate_v_next < 0.0)

        if bool(out_of_bounds.item()):
            p_final = torch.zeros_like(p_current)
            q_final = torch.zeros_like(q_candidate)
            v_next = v_current
            h_next = h_current
        else:
            p_final = p_clamped if p_current != 0 else torch.zeros_like(p_current)
            q_final = q_candidate
            v_next = candidate_v_next
            h_next = v_low_to_h(v_next)

        p_exec_hist.append(p_final)
        q_exec_hist.append(q_final)
        v_next_hist.append(v_next)

        v_current = v_next
        h_current = h_next

    p_exec = torch.stack(p_exec_hist) if p_exec_hist else torch.zeros(0, device=device)
    q_exec = torch.stack(q_exec_hist) if q_exec_hist else torch.zeros(0, device=device)
    h_path = torch.stack(h_hist) if h_hist else torch.zeros(0, device=device)
    v_path = torch.stack(v_hist) if v_hist else torch.zeros(0, device=device)
    v_next_path = torch.stack(v_next_hist) if v_next_hist else torch.zeros(0, device=device)

    return {
        "p_cmd": p_sched.detach().cpu().numpy(),
        "p_exec": p_exec.detach().cpu().numpy(),
        "q_exec": q_exec.detach().cpu().numpy(),
        "h": h_path.detach().cpu().numpy(),
        "v_t": v_path.detach().cpu().numpy(),
        "v_next": v_next_path.detach().cpu().numpy(),
        # MIQP benchmark scripts use the last pre-hour volume state for the
        # terminal penalty calculation, so expose that explicitly.
        "v_penalty_state": float(v_path[-1].item()) if len(v_path) else float(system_params["v_init"]),
        "v_terminal": float(v_current.item()),
    }


def score_exact_schedule(
    p_cmd: np.ndarray,
    prices: np.ndarray,
    system_params: dict,
    *,
    c_op: float,
    si_shortage_multiplier: float,
    si_surplus_multiplier: float,
    return_trace: bool = False,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Score a commanded schedule with the shared exact hourly simulator."""
    trace = simulate_exact_hourly_schedule(p_cmd, system_params, device=device)
    prices_np = np.asarray(prices, dtype=float)
    p_cmd_np = np.asarray(trace["p_cmd"], dtype=float)
    p_exec_np = np.asarray(trace["p_exec"], dtype=float)

    revenue = float(np.sum(prices_np * p_exec_np))
    op_cost = float(c_op * np.sum(p_exec_np ** 2))
    profit = revenue - op_cost

    si_price = np.where(
        p_exec_np < p_cmd_np,
        si_shortage_multiplier * prices_np,
        si_surplus_multiplier * prices_np,
    )
    imbalance = p_exec_np - p_cmd_np
    si_penalty = float(np.sum(imbalance * si_price))

    vol_surplus = max(0.0, float(trace["v_penalty_state"]) - float(system_params["target_vol_low"]))
    energy_loss = RHO * vol_surplus * G * float(system_params["target_head"]) * ETA / 3.6e9
    volume_penalty = float(energy_loss * np.median(prices_np))

    result = {
        "profit": profit,
        "revenue": revenue,
        "op_cost": op_cost,
        "si_penalty": si_penalty,
        "volume_penalty": volume_penalty,
        "expost_profit": float(profit - si_penalty - volume_penalty),
        "p_net": p_cmd_np,
        "p_sim": p_exec_np,
        "q_sim": np.asarray(trace["q_exec"], dtype=float),
        "h_traj": np.asarray(trace["h"], dtype=float),
        "v_traj": np.asarray(trace["v_next"], dtype=float),
        "v_final": float(trace["v_terminal"]),
        "n_turbine": int(np.sum(p_exec_np > 0.5)),
        "n_idle": int(np.sum(np.abs(p_exec_np) <= 0.5)),
        "n_pump": int(np.sum(p_exec_np < -0.5)),
    }
    if return_trace:
        result["trace"] = trace
    return result
