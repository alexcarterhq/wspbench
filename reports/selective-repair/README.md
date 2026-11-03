# Diagnosis and selective repair calibration

Recorded 24/24 planned API sessions. This development calibration uses a small number of cases. Paired differences describe these cases and carry substantial uncertainty.

Repair success uses the grader's target and preservation criteria. Invalid submissions and protocol violations count as repair failures. Missing or unscorable sessions are excluded from accuracy denominators and remain visible in coverage and outcome counts. Target success and retention loss average over graded patches. Forecast MAE includes every graded commitment, even when the patch fails validation.

| Actor | Access | Recorded / planned | Repair success | Target success | Retention loss | Forecast MAE | Probes used | Outcomes | Estimated API cost |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| gpt-6-luna | weights_only | 3/3 | 1/3 (33.3%) | 66.7% (n=3) | 0.0% (n=3) | 0.030 (n=3) | 0 | completed: 3 | $0.0049 |
| gpt-6-luna | behavior | 3/3 | 1/3 (33.3%) | 66.7% (n=3) | 0.0% (n=3) | 0.195 (n=3) | 18 | completed: 3 | $0.0049 |
| gpt-6-luna | weights_behavior | 3/3 | 3/3 (100.0%) | 100.0% (n=3) | 0.0% (n=3) | 0.032 (n=3) | 10 | completed: 3 | $0.0053 |
| gpt-6-luna | weights_activations | 3/3 | 3/3 (100.0%) | 100.0% (n=3) | 0.0% (n=3) | 0.049 (n=3) | 8 | completed: 3 | $0.0057 |
| gpt-6.1-sol | weights_only | 3/3 | 3/3 (100.0%) | 100.0% (n=3) | 0.0% (n=3) | 0.029 (n=3) | 0 | completed: 3 | $0.0668 |
| gpt-6.1-sol | behavior | 3/3 | 3/3 (100.0%) | 100.0% (n=3) | 0.0% (n=3) | 0.179 (n=3) | 18 | completed: 3 | $0.0702 |
| gpt-6.1-sol | weights_behavior | 3/3 | 3/3 (100.0%) | 100.0% (n=3) | 0.0% (n=3) | 0.016 (n=3) | 10 | completed: 3 | $0.0779 |
| gpt-6.1-sol | weights_activations | 3/3 | 3/3 (100.0%) | 100.0% (n=3) | 0.0% (n=3) | 0.017 (n=3) | 4 | completed: 3 | $0.0715 |
| scripted | behavior | 3/3 | 2/3 (66.7%) | 66.7% (n=3) | 0.0% (n=3) | 0.109 (n=3) | 14 | completed: 3 | — |

| Matched repair contrast | Actor | Scored / planned pairs | Change | Case transitions |
|---|---|---:|---:|---|
| Weights added to behavior access (primary) | gpt-6-luna | 3/3 | +66.7 pp | 2 gained / 0 lost |
| Activations added to weights and behavior | gpt-6-luna | 3/3 | +0.0 pp | 0 gained / 0 lost |
| Agent − scripted under behavior | gpt-6-luna | 3/3 | -33.3 pp | 0 gained / 1 lost |
| Weights added to behavior access (primary) | gpt-6.1-sol | 3/3 | +0.0 pp | 0 gained / 0 lost |
| Activations added to weights and behavior | gpt-6.1-sol | 3/3 | +0.0 pp | 0 gained / 0 lost |
| Agent − scripted under behavior | gpt-6.1-sol | 3/3 | +33.3 pp | 1 gained / 0 lost |

Total estimated API cost $0.3072 across 24 sessions with recorded usage cost. Unknown costs 0.
Weights-only scores are descriptive because that arm has no behavior observations.

| Actor | Case | Access | Submitted patch | Repair | Target success | Retention loss | Forecast MAE |
|---|---|---|---|---:|---:|---:|---:|
| gpt-6-luna | controller-1 | behavior | i2 | 1 | 100.0% | 0.0% | 0.205 |
| gpt-6-luna | controller-1 | weights_activations | i2 | 1 | 100.0% | 0.0% | 0.089 |
| gpt-6-luna | controller-1 | weights_behavior | i2 | 1 | 100.0% | 0.0% | 0.045 |
| gpt-6-luna | controller-1 | weights_only | none | 0 | 0.0% | 0.0% | 0.010 |
| gpt-6-luna | controller-2 | behavior | i2 | 0 | 0.0% | 0.0% | 0.253 |
| gpt-6-luna | controller-2 | weights_activations | i5 | 1 | 100.0% | 0.0% | 0.027 |
| gpt-6-luna | controller-2 | weights_behavior | i5 | 1 | 100.0% | 0.0% | 0.029 |
| gpt-6-luna | controller-2 | weights_only | i5 | 1 | 100.0% | 0.0% | 0.026 |
| gpt-6-luna | controller-3 (clean) | behavior | i4 | 0 | 100.0% | 0.0% | 0.126 |
| gpt-6-luna | controller-3 (clean) | weights_activations | none | 1 | 100.0% | 0.0% | 0.031 |
| gpt-6-luna | controller-3 (clean) | weights_behavior | none | 1 | 100.0% | 0.0% | 0.021 |
| gpt-6-luna | controller-3 (clean) | weights_only | i3 | 0 | 100.0% | 0.0% | 0.055 |
| gpt-6.1-sol | controller-1 | behavior | i2 | 1 | 100.0% | 0.0% | 0.197 |
| gpt-6.1-sol | controller-1 | weights_activations | i2 | 1 | 100.0% | 0.0% | 0.011 |
| gpt-6.1-sol | controller-1 | weights_behavior | i2 | 1 | 100.0% | 0.0% | 0.011 |
| gpt-6.1-sol | controller-1 | weights_only | i2 | 1 | 100.0% | 0.0% | 0.017 |
| gpt-6.1-sol | controller-2 | behavior | i5 | 1 | 100.0% | 0.0% | 0.240 |
| gpt-6.1-sol | controller-2 | weights_activations | i5 | 1 | 100.0% | 0.0% | 0.023 |
| gpt-6.1-sol | controller-2 | weights_behavior | i5 | 1 | 100.0% | 0.0% | 0.011 |
| gpt-6.1-sol | controller-2 | weights_only | i5 | 1 | 100.0% | 0.0% | 0.037 |
| gpt-6.1-sol | controller-3 (clean) | behavior | none | 1 | 100.0% | 0.0% | 0.100 |
| gpt-6.1-sol | controller-3 (clean) | weights_activations | none | 1 | 100.0% | 0.0% | 0.017 |
| gpt-6.1-sol | controller-3 (clean) | weights_behavior | none | 1 | 100.0% | 0.0% | 0.028 |
| gpt-6.1-sol | controller-3 (clean) | weights_only | none | 1 | 100.0% | 0.0% | 0.033 |
| scripted | controller-1 | behavior | i2 | 1 | 100.0% | 0.0% | 0.132 |
| scripted | controller-2 | behavior | none | 0 | 0.0% | 0.0% | 0.139 |
| scripted | controller-3 (clean) | behavior | none | 1 | 100.0% | 0.0% | 0.056 |

The scored calibration used structured function calls. Two earlier transport
passes used text actions and exposed ambiguity between commentary and final
answers. They cost $0.2023 and $0.2889. Total estimated cost across the
scored run and those two passes was $0.7984.

The audit replayed all 27 records, including the scripted baseline, checked
the tool budgets and forecast commitments, and reconciled all 24 scored
API settlements. Local validation covered 192 cases across 64 seeds.
Reference repairs passed every case. The scripted solver repaired 59 of 128
faults and correctly left all 64 clean controls alone.

Each API cell has one attempt. All four access arms reuse the same two
faults and one clean case, so the result is a calibration of a six-choice
repair task. Broader claims require more fault mechanisms and a larger
independent case bank.
