# Full Sol selective repair experiment

32 faults and 8 clean controls across 8 seed blocks. Each session has up to 5 turns, 48,000 input tokens, 24,000 output tokens, 2 pre-commit probes and 4 post-commit probes.

Adding weights changed fault repair success by +50.0 pp across 32/32 evaluable matched pairs (16 gained, 0 lost, 16 tied).
The 95% percentile interval from 10,000 seed-block bootstrap draws is +28.1 pp to +71.9 pp. It describes uncertainty across these generated cases.
Among the 32 fault pairs with valid patches in both arms, the repair difference was +50.0 pp. This subset conditions on submission validity.
This comparison counts invalid submissions and protocol violations as failures. It omits infrastructure failures and pending sessions from the point estimate.

| Arm | Recorded / planned | Faults repaired / scored | Clean preserved / scored | Valid patches | Output caps | Other invalid | Infra failures | Forecasts / planned | Estimated API cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Sol, behavior | 40/40 | 14/32 | 8/8 | 40 | 0 | 0 | 0 | 40/40 | $2.0224 |
| Sol, weights + behavior | 40/40 | 30/32 | 8/8 | 40 | 0 | 0 | 0 | 40/40 | $6.3445 |
| Scripted, behavior | 40/40 | 5/32 | 8/8 | 40 | 0 | 0 | 0 | 40/40 | — |

Total estimated API cost from recorded usage is $8.3669. 0 recorded sessions have unknown cost. The frozen reservation ceiling is $28.80.

| Fault mechanism | Planned pairs | Behavior repairs | With weights repairs | Scripted repairs | Matched repair change | Matched forecast MAE change |
|---|---:|---:|---:|---:|---:|---:|
| coupled gates | 8 | 3/8 | 8/8 | 0/8 | +62.5 pp (8 pairs) | -0.133 (8 pairs) |
| cross layer drift | 8 | 3/8 | 8/8 | 1/8 | +62.5 pp (8 pairs) | -0.128 (8 pairs) |
| readout redistribution | 8 | 1/8 | 6/8 | 1/8 | +62.5 pp (8 pairs) | -0.117 (8 pairs) |
| sensor routing | 8 | 7/8 | 8/8 | 3/8 | +12.5 pp (8 pairs) | -0.089 (8 pairs) |
| All faults | 32 | 14/32 | 30/32 | 5/32 | +50.0 pp (32 pairs) | -0.117 (32 pairs) |

Forecast changes compare committed estimates for the same case. A negative change means lower error with weights. Counts exclude sessions without a committed forecast.
