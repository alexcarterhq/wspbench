# Diagnosis and selective repair calibration

Recorded 10/10 planned API sessions. This development calibration uses a small number of cases. Paired differences describe these cases and carry substantial uncertainty.

Per-session limits are 5 model turns, 48,000 input tokens, 12,000 output tokens, 2 pre-commit probes, 4 post-commit probes.

Repair success uses the grader's target and preservation criteria. Invalid submissions and protocol violations count as repair failures. Missing or unscorable sessions are excluded from accuracy denominators and remain visible in coverage and outcome counts. Target success and retention loss average over graded patches. Forecast MAE includes every graded commitment, even when the patch fails validation.

| Actor | Access | Recorded / planned | Repair success | Target success | Retention loss | Forecast MAE | Probes used | Outcomes | Estimated API cost |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| gpt-6.1-sol | behavior | 5/5 | 2/5 (40.0%) | 62.3% (n=5) | 9.5% (n=5) | 0.150 (n=5) | 30 | completed: 5 | $0.2605 |
| gpt-6.1-sol | weights_behavior | 5/5 | 3/5 (60.0%) | 100.0% (n=3) | 0.0% (n=3) | 0.063 (n=3) | 22 | completed: 3, invalid_submission: 2 | $0.6503 |
| scripted | behavior | 5/5 | 1/5 (20.0%) | 45.4% (n=5) | 5.8% (n=5) | 0.138 (n=5) | 26 | completed: 5 | — |

| Matched contrast | Actor | Cases | Scored / planned repair pairs | Repair change | Repair transitions | Scored / planned forecast pairs | Forecast MAE change |
|---|---|---|---:|---:|---|---:|---:|
| Weights added to behavior access (primary) | gpt-6.1-sol | All | 5/5 | +20.0 pp | 1 gained / 0 lost | 3/5 | -0.078 |
| Weights added to behavior access (primary) | gpt-6.1-sol | Faults | 4/4 | +25.0 pp | 1 gained / 0 lost | 2/4 | -0.075 |
| Agent − scripted under behavior | gpt-6.1-sol | All | 5/5 | +20.0 pp | 1 gained / 0 lost | 5/5 | +0.011 |
| Agent − scripted under behavior | gpt-6.1-sol | Faults | 4/4 | +25.0 pp | 1 gained / 0 lost | 4/4 | -0.005 |

| gpt-6.1-sol access | Valid repairs / valid scored faults | Valid scored / planned faults | Ceiling |
|---|---:|---:|---|
| behavior | 1/4 | 4/4 | Below ceiling |
| weights_behavior | 2/2 | 2/4 | Incomplete |

Total estimated API cost $0.9108 across 10 sessions with recorded usage cost. Unknown costs 0.
Weights-only scores are descriptive because that arm has no behavior observations.

| Actor | Case | Access | Submitted patch | Repair | Target success | Retention loss | Forecast MAE |
|---|---|---|---|---:|---:|---:|---:|
| gpt-6.1-sol | hard-01 (coupled_gates) | behavior | i02 | 0 | 71.9% | 10.9% | 0.129 |
| gpt-6.1-sol | hard-01 (coupled_gates) | weights_behavior | — | 0 | — | — | — |
| gpt-6.1-sol | hard-02 (sensor_routing) | behavior | i10 | 0 | 1.0% | 15.6% | 0.198 |
| gpt-6.1-sol | hard-02 (sensor_routing) | weights_behavior | — | 0 | — | — | — |
| gpt-6.1-sol | hard-03 (readout_redistribution) | behavior | i06 | 0 | 38.5% | 21.1% | 0.142 |
| gpt-6.1-sol | hard-03 (readout_redistribution) | weights_behavior | i03 | 1 | 100.0% | 0.0% | 0.046 |
| gpt-6.1-sol | hard-04 (cross_layer_drift) | behavior | i11 | 1 | 100.0% | 0.0% | 0.116 |
| gpt-6.1-sol | hard-04 (cross_layer_drift) | weights_behavior | i11 | 1 | 100.0% | 0.0% | 0.062 |
| gpt-6.1-sol | hard-05 (clean) | behavior | none | 1 | 100.0% | 0.0% | 0.163 |
| gpt-6.1-sol | hard-05 (clean) | weights_behavior | none | 1 | 100.0% | 0.0% | 0.081 |
| scripted | hard-01 (coupled_gates) | behavior | i04 | 0 | 61.5% | 19.5% | 0.163 |
| scripted | hard-02 (sensor_routing) | behavior | none | 0 | 0.0% | 0.0% | 0.162 |
| scripted | hard-03 (readout_redistribution) | behavior | i02 | 0 | 65.6% | 9.4% | 0.126 |
| scripted | hard-04 (cross_layer_drift) | behavior | none | 0 | 0.0% | 0.0% | 0.154 |
| scripted | hard-05 (clean) | behavior | none | 1 | 100.0% | 0.0% | 0.088 |

The follow-up kept the checkpoints and practice and scoring banks from
the first run. It raised the output limit to 12,000 tokens in both arms
and replaced the target-derived global decoy with a fixed shift.

Sol submitted valid behavior-only patches for all four faults and fixed
one. With weights it fixed both faults for which it submitted a patch.
It exhausted the output allowance on the other two. Both arms left the
clean checkpoint unchanged. One attempt per cell and two unsubmitted
weight-assisted repairs limit the estimate of the weight-access effect.
