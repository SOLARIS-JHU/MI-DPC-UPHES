"""UPHES dynamics with straight-through clamp fix.

Replaces torch.clamp with _ste_clamp so gradients flow at volume/head
boundaries instead of being zeroed out.
"""

import torch
import torch.nn as nn

from DPC.config import DT


def _ste_clamp(x, lo, hi):
    """Clamp x to [lo, hi] in forward; gradient passes through as identity."""
    x_clamped = torch.clamp(x, lo, hi)
    return x + (x_clamped - x).detach()


def _ste_relu(x):
    """relu in forward; identity gradient in backward."""
    return x + (torch.relu(x) - x).detach()


class UPHESDynamicsStep(nn.Module):
    """Exact sequential UPHES rollout over a control horizon."""

    def __init__(self, system_params):
        super().__init__()
        self.pos_min = system_params['pos_min']
        self.pos_max = system_params['pos_max']
        self.neg_min = system_params['neg_min']
        self.neg_max = system_params['neg_max']
        self.UPC_poly_tur = system_params['UPC_poly_tur']
        self.UPC_poly_pump = system_params['UPC_poly_pump']
        self.UPC_inv_tur = system_params.get('UPC_inv_tur')
        self.UPC_inv_pump = system_params.get('UPC_inv_pump')
        if self.UPC_inv_tur is None or self.UPC_inv_pump is None:
            raise ValueError("UPHESDynamicsStep requires UPC_inv_tur and UPC_inv_pump")
        self.v_low_to_h = system_params['v_low_to_h']
        self.head_min = system_params['head_min']
        self.head_max = system_params['head_max']
        self.max_vol_low = system_params['max_vol_low']

    def _scale_power(self, p_T_ratio, p_P_ratio, h, m_pos, m_neg):
        """Scale power ratios to head-feasible commanded powers."""
        p_T_min = self.pos_min(h)
        p_T_max = self.pos_max(h)
        p_P_min = self.neg_min(h)
        p_P_max = self.neg_max(h)
        p_cmd_tur = (p_T_min + p_T_ratio * (p_T_max - p_T_min)) * m_pos
        p_cmd_pump = (p_P_min + p_P_ratio * (p_P_max - p_P_min)) * m_neg
        return p_cmd_tur, p_cmd_pump

    def _compute_flow(self, p_cmd_tur, p_cmd_pump, h, m_pos, m_neg):
        """Compute commanded flow from the forward UPC surfaces."""
        q_tur = self.UPC_poly_tur(p_cmd_tur, h)
        q_pump = self.UPC_poly_pump(p_cmd_pump, h)
        return q_tur * m_pos + q_pump * m_neg

    def _reconstruct_power(self, q_exec, h, m_pos, m_neg):
        """Reconstruct executed powers from executed flow via inverse UPCs."""
        p_exec_tur = self.UPC_inv_tur(q_exec, h) * m_pos
        p_exec_pump = self.UPC_inv_pump(q_exec, h) * m_neg
        return p_exec_tur, p_exec_pump

    def forward(self, x, u):
        """
        Args:
            x: (B, 1, 2) initial state [h_init, v_init]
            u: (B, T, 3) controls [p_T_ratio, p_P_ratio, mode]
        Returns:
            x_full: (B, 1+T, 2) full state trajectory
            aux:    (B, T, 12) rollout auxiliaries
        """
        if x.ndim != 3 or x.shape[1:] != (1, 2):
            raise ValueError(f"x must have shape (B, 1, 2), got {tuple(x.shape)}")
        if u.ndim != 3 or u.shape[-1] != 3:
            raise ValueError(f"u must have shape (B, T, 3), got {tuple(u.shape)}")
        if x.shape[0] != u.shape[0]:
            raise ValueError(
                f"x and u must have the same batch size, got x.shape[0]={x.shape[0]} "
                f"and u.shape[0]={u.shape[0]}"
            )

        h_t = x[:, 0, 0]
        v_t = x[:, 0, 1]
        B, T, _ = u.shape

        x_steps = []
        aux_steps = []
        for t in range(T):
            p_T_ratio = u[:, t, 0]
            p_P_ratio = u[:, t, 1]
            mode = u[:, t, 2]

            # Mode indicators (STE keeps gradient alive at idle)
            m_pos = _ste_relu(mode)        # turbine (mode = +1)
            m_neg = _ste_relu(-mode)       # pump    (mode = -1)

            p_cmd_tur, p_cmd_pump = self._scale_power(p_T_ratio, p_P_ratio, h_t, m_pos, m_neg)
            q_raw = self._compute_flow(p_cmd_tur, p_cmd_pump, h_t, m_pos, m_neg)

            v_raw_next = v_t + DT * q_raw
            v_next = _ste_clamp(v_raw_next, 0.0, self.max_vol_low)
            q_exec = (v_next - v_t) / DT

            h_raw_next = self.v_low_to_h(v_next)
            h_next = _ste_clamp(h_raw_next, self.head_min, self.head_max)

            p_exec_tur, p_exec_pump = self._reconstruct_power(q_exec, h_t, m_pos, m_neg)

            v_viol_lo = torch.relu(-v_raw_next)
            v_viol_hi = torch.relu(v_raw_next - self.max_vol_low)
            h_viol_lo = torch.relu(self.head_min - h_raw_next)
            h_viol_hi = torch.relu(h_raw_next - self.head_max)

            x_steps.append(torch.stack([h_next, v_next], dim=-1))
            aux_steps.append(torch.stack([
                p_cmd_tur,
                p_cmd_pump,
                p_exec_tur,
                p_exec_pump,
                q_raw,
                q_exec,
                m_pos,
                m_neg,
                v_viol_lo,
                v_viol_hi,
                h_viol_lo,
                h_viol_hi,
            ], dim=-1))

            h_t = h_next
            v_t = v_next

        x_steps = x.new_zeros((B, 0, 2)) if T == 0 else torch.stack(x_steps, dim=1)
        aux = u.new_zeros((B, 0, 12)) if T == 0 else torch.stack(aux_steps, dim=1)
        x_full = torch.cat([x, x_steps], dim=1)
        return x_full, aux


class UPHESDynamicsBatch(nn.Module):
    """Batch dynamics for one-shot policy — processes all T timesteps at once.

    Uses 2-pass iterative refinement: first estimates head from h_init,
    computes flow/volume, then refines power & flow with the actual head.

    Input:  x = (B, 1, 2)  initial state [h_init, v_init]
            u = (B, T, 3)  controls [p_T_ratio, p_P_ratio, mode]
    Output: x = (B, 1+T, 2)  full state trajectory (init + T steps)
            aux = (B, T, 6)   [p_T, p_P, m_pos, m_neg, v_viol_lo, v_viol_hi] per step
    """

    def __init__(self, system_params):
        super().__init__()
        self.pos_min = system_params['pos_min']
        self.pos_max = system_params['pos_max']
        self.neg_min = system_params['neg_min']
        self.neg_max = system_params['neg_max']
        self.UPC_poly_tur = system_params['UPC_poly_tur']
        self.UPC_poly_pump = system_params['UPC_poly_pump']
        self.v_low_to_h = system_params['v_low_to_h']
        self.head_min = system_params['head_min']
        self.head_max = system_params['head_max']
        self.max_vol_low = system_params['max_vol_low']

    def _scale_power(self, p_T_ratio, p_P_ratio, h, m_pos, m_neg):
        """Scale ratios to feasible power using head-dependent bounds."""
        B, T = h.shape
        h_flat = h.reshape(-1)
        p_T_min = self.pos_min(h_flat).reshape(B, T)
        p_T_max = self.pos_max(h_flat).reshape(B, T)
        p_P_min = self.neg_min(h_flat).reshape(B, T)
        p_P_max = self.neg_max(h_flat).reshape(B, T)
        p_T = (p_T_min + p_T_ratio * (p_T_max - p_T_min)) * m_pos
        p_P = (p_P_min + p_P_ratio * (p_P_max - p_P_min)) * m_neg
        return p_T, p_P

    def _compute_flow(self, p_T, p_P, h, m_pos, m_neg):
        """Compute flow via UPC polynomials for all timesteps."""
        B, T = p_T.shape
        q_tur = self.UPC_poly_tur(p_T.reshape(-1), h.reshape(-1)).reshape(B, T)
        q_pump = self.UPC_poly_pump(p_P.reshape(-1), h.reshape(-1)).reshape(B, T)
        return q_tur * m_pos + q_pump * m_neg

    def _volume_trajectory(self, v_init, q):
        """Compute volume trajectory using cumulative-sum matrix form (STE clamp).

        Returns:
            v_clamped: (B, T) STE-clamped volume (used for physics).
            v_raw:     (B, T) pre-clamp volume (used for violation penalties).
        """
        B, T = q.shape
        cumsum_mat = torch.tril(torch.ones(T, T, device=q.device))
        v_raw = v_init.unsqueeze(1) + DT * torch.matmul(q, cumsum_mat.T)
        v_clamped = _ste_clamp(v_raw, 0.0, self.max_vol_low)
        return v_clamped, v_raw

    def _volume_to_head(self, v):
        """Convert volume trajectory to head trajectory (STE clamp).

        Returns:
            h_clamped: (B, T) STE-clamped head (used for physics).
            h_raw:     (B, T) pre-clamp head (used for violation penalties).
        """
        B, T = v.shape
        h_raw = self.v_low_to_h(v.reshape(-1)).reshape(B, T)
        h_clamped = _ste_clamp(h_raw, self.head_min, self.head_max)
        return h_clamped, h_raw

    def forward(self, x, u):
        h_init = x[:, 0, 0]          # (B,)
        v_init = x[:, 0, 1]          # (B,)
        p_T_ratio = u[:, :, 0]       # (B, T)
        p_P_ratio = u[:, :, 1]       # (B, T)
        mode = u[:, :, 2]            # (B, T)
        B, T = mode.shape

        # Mode indicators (STE keeps gradient alive at idle)
        m_pos = _ste_relu(mode)
        m_neg = _ste_relu(-mode)

        # --- Pass 1: estimate with h_init ---
        h_est = h_init.unsqueeze(1).expand(B, T)
        p_T, p_P = self._scale_power(p_T_ratio, p_P_ratio, h_est, m_pos, m_neg)
        q = self._compute_flow(p_T, p_P, h_est, m_pos, m_neg)
        v_traj, _ = self._volume_trajectory(v_init, q)
        h_traj, _ = self._volume_to_head(v_traj)

        # --- Pass 2: refine with actual head ---
        p_T, p_P = self._scale_power(p_T_ratio, p_P_ratio, h_traj, m_pos, m_neg)
        q = self._compute_flow(p_T, p_P, h_traj, m_pos, m_neg)
        v_traj, v_raw = self._volume_trajectory(v_init, q)
        h_traj, h_raw = self._volume_to_head(v_traj)

        # Constraint violations (pre-clamp, so signal is non-zero when clamp fires)
        v_viol_lo = torch.relu(-v_raw)                        # (B, T) — vol below 0
        v_viol_hi = torch.relu(v_raw - self.max_vol_low)     # (B, T) — vol above max
        h_viol_lo = torch.relu(self.head_min - h_raw)        # (B, T) — head below min
        h_viol_hi = torch.relu(h_raw - self.head_max)        # (B, T) — head above max

        # Build output tensors matching SystemPreview shapes
        x_init = x[:, 0:1, :]                                            # (B, 1, 2)
        x_steps = torch.stack([h_traj, v_traj], dim=-1)                  # (B, T, 2)
        x_full = torch.cat([x_init, x_steps], dim=1)                     # (B, 1+T, 2)
        aux = torch.stack([p_T, p_P, m_pos, m_neg,
                           v_viol_lo, v_viol_hi,
                           h_viol_lo, h_viol_hi], dim=-1)                 # (B, T, 8)
        return x_full, aux
