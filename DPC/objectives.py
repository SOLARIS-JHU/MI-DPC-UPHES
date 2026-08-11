"""Augmented loss for MVP: adds volume trajectory lower/upper bound penalties.

Loss construction for one-shot UPHES DPC, with two extra Loss terms that
penalise pre-clamp volume violations. The aux layout depends on the rollout mode:
legacy batch aux uses 8 channels, while step rollout aux uses 12 channels.
"""

import torch
from neuromancer.constraint import variable, Loss
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem

from DPC.config import C_OP, HEAD_PENALTY, RHO, G, ETA

VOL_TRAJ_PENALTY = 50.0   # same order as HEAD_PENALTY; tune as needed
VOL_END_WEIGHT   = 1.2    # EXPERIMENT 3: intermediate between 1.0 (baseline) and 1.5 (over-corrected)


def _power_terms_from_aux(aux):
    """Return commanded and actual/simulated power terms from aux.

    Legacy batch aux (8 channels) exposes commanded power in channels 0-1 and
    derives simulated power from the infeasibility mask in channels 4-7.
    Step rollout aux (12 channels) exposes commanded power in channels 0-1 and
    executed power in channels 2-3.
    """
    p_net = aux[..., 0:1] + aux[..., 1:2]
    if aux.shape[-1] >= 12:
        p_actual = aux[..., 2:3] + aux[..., 3:4]
        return p_net, p_actual

    infeas = (aux[..., 4:5] + aux[..., 5:6] + aux[..., 6:7] + aux[..., 7:8]) > 1e-9
    p_sim = p_net * (~infeas)
    return p_net, p_sim


def _volume_violation_terms_from_aux(aux):
    """Return pre-clamp volume violation channels for batch or step rollout."""
    if aux.shape[-1] >= 12:
        return aux[..., 8], aux[..., 9]
    return aux[..., 4], aux[..., 5]


def build_loss(head_min, head_max, max_vol_low, target_vol_low, target_head,
               c_op=C_OP, head_penalty_weight=HEAD_PENALTY):
    """Construct PenaltyLoss with profit objective, volume trajectory penalties,
    head bound constraints, and end-of-day volume penalty."""

    # --- Profit objective ---------------------------------------------------
    def profit_loss_fn(aux, d):
        # aux: (B, T, 8) or (B, T, 12); d: (B, 24, 1)
        p_net, p_actual = _power_terms_from_aux(aux)
        T = p_actual.shape[1]
        prices = d[:, :T, :]                        # (B, T, 1)
        revenue = torch.sum(p_actual * prices, dim=1)   # (B, 1)
        op_cost = c_op * torch.sum(p_actual ** 2, dim=1)
        return torch.mean(-revenue + op_cost)

    profit_obj = Loss(['aux', 'd'], profit_loss_fn, weight=1.0, name='profit_loss')

    # --- End-of-day volume penalty ------------------------------------------
    # Penalises surplus water in the lower reservoir above the target level.
    # Formula: energy_conv * surplus_m3 * median_price → EUR units.
    # VOL_END_WEIGHT amplifies the gradient so the penalty competes with profit.
    energy_conv = (RHO * G * ETA * target_head) / 3.6e9  # MWh/m³

    def volume_penalty_fn(x, d):
        # x: (B, 1+T, 2)  d: (B, 24, 1)
        # Bilateral: penalise deviation from target in EITHER direction (EUR units).
        # Prevents both over-turbining (surplus) and over-pumping (deficit).
        T = x.shape[1] - 1
        v_final = x[:, -1, 1]                                  # (B,)
        deviation = torch.abs(v_final - target_vol_low)         # (B,) always ≥ 0
        med_price = torch.median(d[:, :T, 0], dim=1)[0]        # (B,)
        return torch.mean(energy_conv * deviation * med_price)

    vol_obj = Loss(['x', 'd'], volume_penalty_fn,
                   weight=VOL_END_WEIGHT, name='volume_penalty')

    # --- Volume trajectory lower bound penalty (pre-clamp v_raw) ------------
    def vol_lb_fn(aux):
        return _volume_violation_terms_from_aux(aux)[0].mean()   # mean relu(-v_raw) over (B, T)

    vol_lb_obj = Loss(['aux'], vol_lb_fn, weight=VOL_TRAJ_PENALTY, name='vol_lb')

    # --- Volume trajectory upper bound penalty (pre-clamp v_raw) ------------
    def vol_ub_fn(aux):
        return _volume_violation_terms_from_aux(aux)[1].mean()   # mean relu(v_raw - max_vol) over (B, T)

    vol_ub_obj = Loss(['aux'], vol_ub_fn, weight=VOL_TRAJ_PENALTY, name='vol_ub')

    # --- Head constraints (symbolic) ----------------------------------------
    x = variable('x')
    h_traj = x[:, 1:, [0]]   # (B, T, 1) — head at every step

    con_lb = head_penalty_weight * (h_traj >= head_min)
    con_ub = head_penalty_weight * (h_traj <= head_max)
    con_lb.name = 'h_lb'
    con_ub.name = 'h_ub'

    return PenaltyLoss(
        objectives=[profit_obj, vol_obj, vol_lb_obj, vol_ub_obj],
        constraints=[con_lb, con_ub],
    )


def build_problem(nodes, loss):
    """Wrap nodes/system and loss into a Neuromancer Problem."""
    if isinstance(nodes, list):
        return Problem(nodes, loss)
    return Problem([nodes], loss)
