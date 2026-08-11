# DPC package

Implementation of MI-DPC (Mixed-Integer Differentiable Predictive Control) for one-shot 24-hour UPHES dispatch. A neural policy maps problem parameters (initial head, initial volume, 24 day-ahead prices) to hourly mode decisions (pump, idle, turbine) and continuous power ratios, trained end-to-end through a differentiable simulator with a Gumbel-Softmax straight-through estimator.

## Modules

| Module | Purpose |
|---|---|
| `config.py` | Physical constants and hyperparameter defaults |
| `dynamics.py` | Differentiable UPHES simulators (`UPHESDynamicsBatch` for parallel batch rollout, `UPHESDynamicsStep` for sequential step rollout) with the STE clamp |
| `ste.py` | Gumbel-Softmax straight-through estimator variants |
| `system.py` | `build_oneshot_system()`: assembles policy, STE, and simulator into a NeuroMANCER system |
| `objectives.py` | Ex-post profit surrogate and feasibility penalty loss |
| `evaluate.py` | `evaluate_day_oneshot()`: exact ex-post evaluation of a trained policy |
| `experiments/` | Training harness (`benchmark_tuner.py`), data sampling, architectures, ablation tooling |
| `visualize/` | Paper figure scripts |
| `compare_benchmark_sweeps.py` | Aggregates sweep outputs into a benchmark report |

## The STE clamp

### The problem: dead gradients at physical boundaries

`dynamics.py` uses `torch.clamp` to enforce volume and head bounds:

```python
v_new = torch.clamp(v + DT * q, 0.0, max_vol_low)
h_new = torch.clamp(h_new, head_min, head_max)
```

`torch.clamp` has zero gradient everywhere the bound is active:

$$
\frac{\partial \,\text{clamp}(x,\,lo,\,hi)}{\partial x} =
\begin{cases}
1 & \text{if } lo < x < hi \\
0 & \text{if } x \leq lo \text{ or } x \geq hi
\end{cases}
$$

In one-shot training the cumulative-sum volume trajectory frequently saturates the lower or upper bound early in the horizon. Every timestep beyond the saturated boundary has zero gradient, so the policy receives no learning signal about the consequences of those decisions.

### The fix: straight-through estimator (STE) clamp

`dynamics.py` replaces `torch.clamp` with:

```python
def _ste_clamp(x, lo, hi):
    x_clamped = torch.clamp(x, lo, hi)
    return x + (x_clamped - x).detach()
```

The `.detach()` on the correction term means:

- **Forward pass**: identical to `torch.clamp`, physical bounds are still enforced exactly.
- **Backward pass**: gradient passes through as the identity ($\partial y / \partial x = 1$ everywhere), regardless of whether the bound is active.

Applied at four sites in `UPHESDynamicsStep` and `UPHESDynamicsBatch`:

| Site | Original | Fixed |
|------|----------|-------|
| Step: volume update | `torch.clamp(v + DT*q, 0, max_vol)` | `_ste_clamp(...)` |
| Step: head from volume | `torch.clamp(h_new, h_min, h_max)` | `_ste_clamp(...)` |
| Batch: `_volume_trajectory` | `torch.clamp(v, 0, max_vol)` | `_ste_clamp(...)` |
| Batch: `_volume_to_head` | `torch.clamp(h, h_min, h_max)` | `_ste_clamp(...)` |
