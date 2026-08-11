# MIQP Benchmark Results — Nonlinear Volume-Head Dynamics

Both variants run with `vh_mode = 'nonlinear'` (polynomial volume-head relationship from `preprocess.pkl`).
Test set: 19 days from 2024. Solver: Gurobi 13.0.0 (academic), 24-thread, Intel Core Ultra 9 275HX.
Time limit: 3,600 s per day (MIQP-PW only). MIP gap tolerance: 1% (MIQP-PW), default (MIQP-GL).

---

## Summary

| Metric | MIQP-GL | MIQP-PW |
|---|---|---|
| **Mean ex-post profit** | **1,997 EUR/day** | **2,530 EUR/day** |
| Std ex-post profit | 1,582 EUR/day | 1,510 EUR/day |
| Mean expected profit (optimizer) | 2,274 EUR/day | 2,490 EUR/day |
| Mean SI penalty | 55 EUR/day | −49 EUR/day† |
| Mean volume penalty | 206 EUR/day | 130 EUR/day |
| Mean operating cost | 347 EUR/day | 308 EUR/day |
| Days solved optimally | 19 / 19 | 15 / 19 |
| Days hitting 3,600 s limit | 0 | 4 (01/09, 01/30, 02/14, 11/21) |
| **Mean solve time** | **1.9 s** | **918.9 s** |
| Median solve time | 0.3 s | 125.8 s |
| Max solve time | 16.0 s (01/30) | 3,602.1 s (11/21) |
| **Total solve time** | **36.2 s** | **4.85 h** |
| Variables per day | 192 (72 binary, 120 cont.) | 6,240 (72 binary, 6,168 cont.) |
| Constraints per day | 98 | 314 |

†Negative mean SI penalty indicates the simulated dispatch slightly over-performs the plan (surplus imbalance).

---

## Model Descriptions

### MIQP-GL — Global Linearization
- Volume-head curve linearized end-to-end across `[h_min, h_max]` as a single affine map: `v_low = slope·h + intercept`
- UPC (Unit Power Curve) linearized globally: `q = coefs_tur @ [p, h] + intercept_tur`
- Bilinear terms `z_mode · h` remain quadratic in the model — Gurobi handles them natively
- **192 variables, 98 constraints** per 24-hour problem
- Very fast: all 19 days solved in ≤16 s; total 36 s

### MIQP-PW — Piecewise SOS2 Linearization
- Volume-head curve approximated by **10-segment** piecewise linear interpolation with SOS2 constraints
- UPC approximated on a **11×11 grid** (10 head segments × 10 power segments) per mode per hour, with SOS2 along power dimension
- Head selection is shared: pump/turbine lambda weights must match the volume-head lambda weights exactly
- **6,240 variables, 314 constraints** per 24-hour problem (≈32× more variables than MIQP-GL)
- 4 hard days hit the 3,600 s time limit with large remaining gaps (01/30: 97.97%, 02/14: 133.71%, 11/21: 18.92%, 01/09: 34.63%)

---

## Per-Day Results

| Date | MIQP-GL Ex-post | MIQP-GL Time | MIQP-PW Ex-post | MIQP-PW Time | MIQP-PW Gap | PW − GL |
|---|---:|---:|---:|---:|---:|---:|
| 2024/01/09 | −346 | 2.8 s | 369 | 3,600 s ★ | 34.63% | +715 |
| 2024/01/30 | −165 | 16.0 s | 177 | 3,601 s ★ | 97.97% | +342 |
| 2024/02/06 | 1,407 | 1.0 s | 1,742 | 546 s | 0.99% | +334 |
| 2024/02/14 | −338 | 6.8 s | 177 | 3,601 s ★ | 133.71% | +516 |
| 2024/03/23 | 1,601 | 0.2 s | 1,694 | 44 s | 0.97% | +93 |
| 2024/04/09 | 896 | 0.2 s | 1,243 | 49 s | 0.00% | +346 |
| 2024/04/24 | 1,122 | 0.7 s | 1,469 | 389 s | 0.82% | +348 |
| 2024/06/16 | 3,952 | 0.1 s | 3,975 | 5 s | 0.70% | +23 |
| 2024/07/04 | 3,960 | 0.1 s | 3,592 | 5 s | 0.92% | −369 |
| 2024/07/09 | 3,093 | 0.2 s | 3,440 | 38 s | 0.87% | +348 |
| 2024/07/19 | 1,619 | 0.3 s | 2,893 | 215 s | 0.36% | +1,274 |
| 2024/08/03 | 2,205 | 0.2 s | 3,001 | 87 s | 0.36% | +795 |
| 2024/08/05 | 4,196 | 0.1 s | 4,883 | 29 s | 0.69% | +687 |
| 2024/08/09 | 4,419 | 0.1 s | 4,687 | 19 s | 0.85% | +268 |
| 2024/08/15 | 3,572 | 0.2 s | 3,925 | 44 s | 0.81% | +354 |
| 2024/10/05 | 1,210 | 0.4 s | 1,852 | 126 s | 0.86% | +642 |
| 2024/10/21 | 2,769 | 1.6 s | 3,658 | 1,284 s | 1.00% | +889 |
| 2024/11/21 | 223 | 4.4 s | 1,436 | 3,602 s ★ | 18.92% | +1,213 |
| 2024/12/13 | 2,555 | 0.6 s | 3,855 | 176 s | 0.49% | +1,300 |
| **Mean** | **1,997** | **1.9 s** | **2,530** | **918.9 s** | — | **+533** |

★ = hit 3,600 s time limit (solution is feasible but not proven optimal).

---

## Key Observations

### Profit gap: MIQP-PW +533 EUR/day over MIQP-GL
MIQP-PW achieves 2,530 vs MIQP-GL 1,997 EUR/day mean ex-post (+26.7%). The gap is driven by:
- **Better volume management**: MIQP-PW mean vol penalty 130 vs MIQP-GL 206 EUR/day. The piecewise VH approximation gives Gurobi a more accurate volume trajectory model, allowing it to stay closer to target.
- **Better power feasibility**: MIQP-GL's global linearization of the UPC can yield power values that the nonlinear simulation recalculates differently, accumulating SI penalties. MIQP-PW's grid-based UPC is closer to the true polynomial.
- **Worst days for MIQP-GL** (01/09, 02/14, 01/30): negative ex-post due to large SI + vol penalties when the global linear model deviates most from nonlinear reality on low-price, low-activity days.

### MIQP-GL: one exception (07/04, −369 EUR vs PW)
On 2024/07/04 MIQP-GL outperforms MIQP-PW by 369 EUR. This is a fast day for both (0.1 s vs 5.2 s); the piecewise model's SOS2 structure introduces slight feasibility rounding that harms the simulation re-evaluation on that particular price/volume profile.

### Time-limit days (MIQP-PW)
The 4 days with poor gaps (01/30: 97.97%, 02/14: 133.71%) are low-price winter days with near-zero or very low expected profits. The SOS2 branching has poor LP relaxation bounds on these instances. The feasible solutions found are usable (positive ex-post) but likely far from the true optimum.

### Solve time tradeoff
MIQP-GL is **482× faster** on average (1.9 s vs 918.9 s; 36 s total vs 4.85 h total). For operational use in a 24-hour ahead scheduling cycle, MIQP-GL is deployable in real-time; MIQP-PW is a research benchmark only.

---

## Comparison with NM-DPC

The NM-DPC best config (`temporal_tuned2_vw2.0`) was evaluated using a different ex-post metric (no SI penalty, volume penalty = `energy_conv × |Δv| × median_price`). The MIQP scripts use a simulation layer that adds SI imbalance penalties, making direct comparison of absolute EUR values unreliable. The relative ordering and gap structure remain informative.

| Method | Mean Ex-post (EUR/day) | Solve time | Notes |
|---|---|---|---|
| MIQP-PW (nonlinear, this run) | 2,530 | 4.85 h total | 4 days suboptimal |
| MIQP-GL (nonlinear, this run) | 1,997 | 36 s total | All optimal |
| NM-DPC `temporal_tuned2_vw2.0` | 2,612* | ~5 min train | *Different eval metric |

*NM-DPC ex-post uses a different simulation (no SI penalty).
