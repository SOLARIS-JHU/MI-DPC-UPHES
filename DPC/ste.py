"""Straight-Through Estimator methods for discrete mode selection.

Each class is a callable with a mutable ``tau`` attribute that is updated
externally during training.  The ``__call__`` signature matches the Node
interface: ``(u_c, mode_logits) -> u`` where
    u_c:          (B, 2)  continuous controls [p_T_ratio, p_P_ratio]
    mode_logits:  (B, 3)  logits for {pump, idle, turbine}
    u:            (B, 3)  concatenated [p_T_ratio, p_P_ratio, mode_scalar]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Mode mapping: index 0 → -1 (pump), 1 → 0 (idle), 2 → +1 (turbine)
# ---------------------------------------------------------------------------
_MODE_MAP = torch.tensor([-1.0, 0.0, 1.0])


def _mode_mapping(device):
    return _MODE_MAP.to(device)


def _hard_mode_from_logits(mode_logits):
    """Deterministic one-hot mode selection for evaluation."""
    mm = _mode_mapping(mode_logits.device)
    indices = mode_logits.argmax(dim=-1)
    mode = mm[indices]
    return mode.unsqueeze(-1)


# ---------------------------------------------------------------------------
# Temperature schedule (shared by all methods that use tau)
# ---------------------------------------------------------------------------

def get_temperature(epoch, tau_start=10.0, tau_end=0.3, tau_decay_epochs=900):
    """Exponential decay from tau_start to tau_end over tau_decay_epochs."""
    if epoch < tau_decay_epochs:
        return tau_start * (tau_end / tau_start) ** (epoch / tau_decay_epochs)
    return tau_end


# ---------------------------------------------------------------------------
# 1. Gumbel-Softmax STE
# ---------------------------------------------------------------------------

class GumbelSTE(nn.Module):
    def __init__(self, tau=10.0):
        super().__init__()
        self.tau = tau

    def forward(self, u_c, mode_logits):
        if self.training:
            mm = _mode_mapping(mode_logits.device)
            # mode_logits: (B, 3)  →  unsqueeze not needed for gumbel_softmax
            mode_soft = F.gumbel_softmax(mode_logits, tau=self.tau, hard=True, dim=-1)
            mode = (mode_soft * mm).sum(dim=-1, keepdim=True)  # (B, 1)
        else:
            mode = _hard_mode_from_logits(mode_logits)
        return torch.cat([u_c, mode], dim=-1)  # (B, 3)


# ---------------------------------------------------------------------------
# 2. Sparsemax STE
# ---------------------------------------------------------------------------

def sparsemax(logits, dim=-1):
    """Sparsemax activation (Martins & Astudillo, 2016)."""
    logits = logits - logits.max(dim=dim, keepdim=True)[0]
    sorted_logits, _ = torch.sort(logits, dim=dim, descending=True)
    cumsum = torch.cumsum(sorted_logits, dim=dim)
    k_range = torch.arange(1, logits.shape[dim] + 1,
                           device=logits.device, dtype=logits.dtype)
    k_range = k_range.view(*([1] * (logits.dim() - 1)), -1)
    threshold = (cumsum - 1) / k_range
    support = (sorted_logits > threshold).float()
    k = support.sum(dim=dim, keepdim=True)
    tau_sp = (torch.sum(sorted_logits * support, dim=dim, keepdim=True) - 1) / k
    return torch.clamp(logits - tau_sp, min=0)


class SparsemaxSTE(nn.Module):
    def __init__(self, tau=10.0):
        super().__init__()
        self.tau = tau  # unused but kept for uniform interface

    def forward(self, u_c, mode_logits):
        mm = _mode_mapping(mode_logits.device)
        probs = sparsemax(mode_logits, dim=-1)
        # Hard ST: one-hot forward, sparsemax backward
        _, indices = probs.max(dim=-1, keepdim=True)
        one_hot = torch.zeros_like(probs).scatter_(-1, indices, 1.0)
        mode_soft = one_hot - probs.detach() + probs
        mode = (mode_soft * mm).sum(dim=-1, keepdim=True)
        return torch.cat([u_c, mode], dim=-1)


# ---------------------------------------------------------------------------
# 3. Soft Straight-Through (temperature-scaled softmax, no hard assignment)
# ---------------------------------------------------------------------------

class SoftSTE(nn.Module):
    def __init__(self, tau=10.0):
        super().__init__()
        self.tau = tau

    def forward(self, u_c, mode_logits):
        mm = _mode_mapping(mode_logits.device)
        probs = F.softmax(mode_logits / self.tau, dim=-1)
        mode = (probs * mm).sum(dim=-1, keepdim=True)
        return torch.cat([u_c, mode], dim=-1)


# ===========================================================================
# One-shot STE variants — handle (B, T, 3) mode logits + (B, T, 2) u_c
# Output: u (B, T, 3) = [p_T_ratio, p_P_ratio, mode]
# ===========================================================================

class OneShotGumbelSTE(nn.Module):
    def __init__(self, tau=10.0):
        super().__init__()
        self.tau = tau

    def forward(self, u_c, mode_logits):
        # u_c: (B,T,2), mode_logits: (B,T,3)
        if self.training:
            mm = _mode_mapping(mode_logits.device)
            B, T, _ = mode_logits.shape
            logits_flat = mode_logits.reshape(B * T, 3)
            soft_flat = F.gumbel_softmax(logits_flat, tau=self.tau, hard=True, dim=-1)
            mode_flat = (soft_flat * mm).sum(dim=-1, keepdim=True)   # (B*T, 1)
            mode = mode_flat.reshape(B, T, 1)
        else:
            mode = _hard_mode_from_logits(mode_logits)
        return torch.cat([u_c, mode], dim=-1)                    # (B, T, 3)


class OneShotSparsemaxSTE(nn.Module):
    def __init__(self, tau=10.0):
        super().__init__()
        self.tau = tau

    def forward(self, u_c, mode_logits):
        mm = _mode_mapping(mode_logits.device)
        probs = sparsemax(mode_logits, dim=-1)
        _, indices = probs.max(dim=-1, keepdim=True)
        one_hot = torch.zeros_like(probs).scatter_(-1, indices, 1.0)
        mode_soft = one_hot - probs.detach() + probs
        mode = (mode_soft * mm).sum(dim=-1, keepdim=True)
        return torch.cat([u_c, mode], dim=-1)


class OneShotSoftSTE(nn.Module):
    def __init__(self, tau=10.0):
        super().__init__()
        self.tau = tau

    def forward(self, u_c, mode_logits):
        mm = _mode_mapping(mode_logits.device)
        probs = F.softmax(mode_logits / self.tau, dim=-1)
        mode = (probs * mm).sum(dim=-1, keepdim=True)
        return torch.cat([u_c, mode], dim=-1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_STE_CLASSES = {
    'gumbel': GumbelSTE,
    'sparsemax': SparsemaxSTE,
    'soft': SoftSTE,
}

_ONESHOT_STE_CLASSES = {
    'gumbel': OneShotGumbelSTE,
    'sparsemax': OneShotSparsemaxSTE,
    'soft': OneShotSoftSTE,
}


def create_ste(method='gumbel', tau=10.0):
    """Return a per-timestep STE instance by name."""
    if method not in _STE_CLASSES:
        raise ValueError(f"Unknown STE method '{method}'. Choose from {list(_STE_CLASSES)}")
    return _STE_CLASSES[method](tau=tau)


def create_oneshot_ste(method='gumbel', tau=10.0):
    """Return a one-shot (batch) STE instance by name."""
    if method not in _ONESHOT_STE_CLASSES:
        raise ValueError(f"Unknown STE method '{method}'. Choose from {list(_ONESHOT_STE_CLASSES)}")
    return _ONESHOT_STE_CLASSES[method](tau=tau)
