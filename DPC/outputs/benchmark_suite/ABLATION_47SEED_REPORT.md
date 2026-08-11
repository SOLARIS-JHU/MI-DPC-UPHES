# DPC Ablation Report: 47-Seed Corrected Study

Date: 2026-03-30

## Scope

This report summarizes the corrected DPC ablation study after:

- fixing `c_op = 0.4`
- evaluating all DPC models with the shared exact hourly nonlinear evaluator
- removing the active `vw * vol_balance` term from the training loss

The kept ablation set uses seeds `0–46`. The unfinished `seed47` was discarded, so the final retained study contains `376` completed runs:

- architecture ablation: `4 x 47 = 188`
- temperature ablation: `2 x 47 = 94`
- dynamics ablation: `2 x 47 = 94`

The active training loss for this study is:

\[
\mathcal{L}
=
\text{profit\_loss}
+ \text{vol\_lb}
+ \text{vol\_ub}
+ \text{h\_lb}
+ \text{h\_ub}
\]

The run directory names still contain tokens such as `vw2` and `vwslate_ramp`, but for this study those are inert metadata only. The `vol_balance` objective was removed from the active `PenaltyLoss.objectives` before these runs were launched.

## Shared Setup

Common training configuration unless it is the ablated factor:

- architecture backbone reference: `transformer`, `hs128`, `nl3`, `ff256`
- sampler: `cluster_balanced`
- batch size: `32`
- optimizer: `adamw`
- LR scheduler: `warmup_cosine`
- learning rate: `3e-4`
- gradient clipping threshold `gc`: `1.0`
- dropout: `0.2`
- weight decay: `0.005`
- operating cost `c_op`: `0.4`
- benchmark evaluation: exact hourly nonlinear simulator on the fixed 19 benchmark days

Timing definitions:

- `train_wall_s`: end-to-end training wall time
- `policy inference ms/day`: policy rollout time only, excluding exact evaluation
- `exact eval ms/day`: exact hourly evaluator time only

Gradient-quality proxies are computed from saved training histories in [ablation_summary.py](/mnt/d/Repositories/L2O4PHES/.worktrees/uphes-step-rollout-compare/DPC/experiments/ablation_summary.py):

- `grad_norm_mean`
- `late_to_early_grad_ratio`
- `best_dev_epoch`
- `dev_expost_slope_tail`

Interpretation:

- larger `grad_norm_mean` suggests stronger gradient signal magnitude
- `late_to_early_grad_ratio < 1` means gradients decay over training
- later `best_dev_epoch` suggests slower convergence
- larger positive `dev_expost_slope_tail` means dev ex-post was still improving near the end

## Architecture Ablation

Annealed temperature and batch dynamics were used for all four architectures.

| Architecture | Seeds | Mean ex-post profit | Mean gross profit | Mean volume penalty | Mean SI penalty | Train time (s) | Policy inference (ms/day) | Exact eval (ms/day) | Best single seed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Transformer | 47 | 2489.21 ± 70.79 | 3260.41 ± 178.24 | 795.21 ± 172.59 | -24.00 ± 10.50 | 161.33 ± 40.07 | 4.88 ± 0.52 | 23.23 ± 10.67 | seed1: 2644.78 |
| Bi-LSTM | 47 | 2421.40 ± 133.42 | 2966.24 ± 187.76 | 573.73 ± 217.40 | -28.88 ± 19.07 | 138.37 ± 36.76 | 4.71 ± 0.91 | 23.13 ± 11.29 | seed34: 2630.23 |
| MLP | 47 | 2353.06 ± 143.18 | 2843.33 ± 180.79 | 517.33 ± 254.05 | -27.07 ± 5.02 | 112.95 ± 31.57 | 3.74 ± 1.08 | 23.03 ± 11.98 | seed20: 2657.23 |
| CNN | 47 | 2080.51 ± 186.79 | 2814.96 ± 169.98 | 760.07 ± 212.23 | -25.63 ± 16.30 | 118.90 ± 32.17 | 4.03 ± 0.64 | 22.83 ± 10.79 | seed34: 2228.68 |

### Architecture Findings

- Transformer remains the best architecture by 47-seed mean ex-post profit.
- Bi-LSTM remains the second-best family and stays relatively close to Transformer.
- MLP is materially weaker on mean ex-post, even though its best single seed slightly exceeds Transformer’s best seed.
- CNN is decisively the weakest architecture on ex-post performance.

Key gaps relative to Transformer:

- Transformer vs Bi-LSTM mean ex-post: `+67.81 EUR/day`
- Transformer vs MLP mean ex-post: `+136.14 EUR/day`
- Transformer vs CNN mean ex-post: `+408.69 EUR/day`

## Temperature Ablation

Transformer + batch dynamics were fixed. The only change was temperature handling:

- `annealed`: `tau_start = 10.0`, `tau_end = 0.08`, `two_stage`
- `fixed_low`: `tau_start = tau_end = 0.08`

| Temperature setting | Seeds | Mean ex-post profit | Mean gross profit | Mean volume penalty | Mean SI penalty | Train time (s) | Policy inference (ms/day) | Exact eval (ms/day) | Best single seed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Annealed | 47 | 2489.21 ± 70.79 | 3260.41 ± 178.24 | 795.21 ± 172.59 | -24.00 ± 10.50 | 169.03 ± 44.66 | 5.05 ± 0.92 | 27.21 ± 15.86 | seed1: 2644.78 |
| Fixed low | 47 | 2284.88 ± 162.90 | 3294.80 ± 296.60 | 1044.79 ± 353.37 | -34.87 ± 16.48 | 173.86 ± 53.80 | 5.10 ± 1.04 | 26.18 ± 14.17 | seed10: 2610.37 |

### Temperature Findings

- Temperature annealing is still clearly better on final ex-post profit.
- The fixed-low setting still earns slightly higher gross profit, but it pays a much larger end-volume penalty.
- Annealing remains more stable across seeds: `70.79` vs `162.90 EUR/day` standard deviation.

Effect size:

- annealed minus fixed-low mean ex-post: `+204.33 EUR/day`
- fixed-low minus annealed mean gross profit: `+34.39 EUR/day`
- fixed-low minus annealed mean volume penalty: `+249.58 EUR/day`

Interpretation:

- Holding temperature low from the start still hardens the discrete decisions too early.
- The controller stays more profit-seeking in gross terms, but exact evaluation shows weaker volume discipline.

## Dynamics Ablation

Transformer + annealed temperature were fixed. The only change was the training simulator:

- `batch`: batch rollout dynamics during training
- `step`: differentiable sequential step rollout during training

All final evaluation was still done with the same exact hourly nonlinear evaluator.

| Dynamics | Seeds | Mean ex-post profit | Mean gross profit | Mean volume penalty | Mean SI penalty | Mean idle hours | Train time (s) | Policy inference (ms/day) | Exact eval (ms/day) | Best single seed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Batch | 47 | 2489.21 ± 70.79 | 3260.41 ± 178.24 | 795.21 ± 172.59 | -24.00 ± 10.50 | 0.19 ± 0.27 | 175.75 ± 56.55 | 4.86 ± 0.53 | 28.15 ± 16.92 | seed1: 2644.78 |
| Step | 47 | 2142.29 ± 50.87 | 3376.49 ± 264.72 | 1230.59 ± 245.89 | 3.60 ± 8.58 | 11.95 ± 1.58 | 2133.28 ± 495.14 | 71.13 ± 32.91 | 21.15 ± 13.51 | seed46: 2232.45 |

### Gradient-Quality Comparison

| Dynamics | Grad norm mean | Late/Early grad ratio | Best dev epoch | Dev ex-post tail slope |
| --- | ---: | ---: | ---: | ---: |
| Batch | 310463.27 ± 83576.91 | 1.46 ± 0.53 | 18.72 ± 6.80 | 9.53 ± 19.40 |
| Step | 90361.60 ± 6149.67 | 0.44 ± 0.05 | 23.09 ± 3.02 | 9.15 ± 21.81 |

### Dynamics Findings

- Batch training dominates step training on final ex-post profit across all `47` paired seeds.
- The mean ex-post gap is `+346.91 EUR/day` in favor of batch.
- The smallest observed step-vs-batch gap is `-112.13 EUR/day`; the largest is `-510.03 EUR/day`.
- Step training still produces higher gross profit, but gives it back through:
  - much larger volume penalty
  - worse SI penalty
  - dramatically higher idle operation

Runtime impact:

- step is `12.14x` slower in training wall time
- step is `14.65x` slower in policy inference time
- exact evaluation cost stays broadly comparable because the evaluator is shared

The gradient proxies continue to support the “long sequential backpropagation chain hurts trainability” hypothesis:

- `grad_norm_mean` is about `3.44x` larger for batch than step
- `late_to_early_grad_ratio` is `1.46` for batch vs `0.44` for step
- `best_dev_epoch` is later for step
- `near_zero_grad_frac` remains `0.0` in both settings, so this is not total gradient collapse

Interpretation:

- the step simulator does not eliminate gradients, but it weakens and decays them enough to harm optimization quality materially
- this shows up jointly in final profit, convergence behavior, runtime, and operating behavior

## Overall Conclusions

1. Transformer remains the best-performing DPC architecture in the corrected study.
2. Temperature annealing remains important. Fixed low temperature still increases gross profit slightly, but degrades ex-post profit through larger end-volume penalties.
3. Batch-simulator training remains decisively better than step-simulator training under exact evaluation.
4. The step simulator evidence continues to support a degraded-gradient explanation rather than a zero-gradient explanation:
   - smaller gradient norms
   - stronger gradient decay through training
   - later convergence
   - much higher runtime cost

## Recommended Default After This Ablation

For the current DPC pipeline, the strongest default remains:

- `transformer`
- `warmup_cosine / two_stage`
- `batch` dynamics for training
- exact hourly nonlinear evaluation for final scoring
- corrected `c_op = 0.4`
- active loss without the separate `vol_balance` objective term

## Artifact Locations

- canonical report: [ABLATION_47SEED_REPORT.md](/mnt/d/Repositories/L2O4PHES/.worktrees/uphes-step-rollout-compare/DPC/outputs/benchmark_suite/ABLATION_47SEED_REPORT.md)
- compatibility note at old path: [ABLATION_7SEED_REPORT.md](/mnt/d/Repositories/L2O4PHES/.worktrees/uphes-step-rollout-compare/DPC/outputs/benchmark_suite/ABLATION_7SEED_REPORT.md)
- aggregated summary table: [ABLATION_47SEED_SUMMARY.csv](/mnt/d/Repositories/L2O4PHES/.worktrees/uphes-step-rollout-compare/DPC/outputs/benchmark_suite/ABLATION_47SEED_SUMMARY.csv)
- per-run metrics table: [ABLATION_47SEED_RUNS.csv](/mnt/d/Repositories/L2O4PHES/.worktrees/uphes-step-rollout-compare/DPC/outputs/benchmark_suite/ABLATION_47SEED_RUNS.csv)
- per-seed batch vs step table: [ABLATION_47SEED_DYNAMICS_PER_SEED.csv](/mnt/d/Repositories/L2O4PHES/.worktrees/uphes-step-rollout-compare/DPC/outputs/benchmark_suite/ABLATION_47SEED_DYNAMICS_PER_SEED.csv)
- active benchmark outputs: [benchmark_suite](/mnt/d/Repositories/L2O4PHES/.worktrees/uphes-step-rollout-compare/DPC/outputs/benchmark_suite)
- archived pre-ablation outputs: [deprecated_pre_ablation_2026-03-28](/mnt/d/Repositories/L2O4PHES/.worktrees/uphes-step-rollout-compare/DPC/outputs/benchmark_suite/deprecated_pre_ablation_2026-03-28)
