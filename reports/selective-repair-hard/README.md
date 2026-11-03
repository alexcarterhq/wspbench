# Diagnosis and selective repair calibration

Recorded 40/40 planned API sessions. This development calibration uses a small number of cases. Paired differences describe these cases and carry substantial uncertainty.

Per-session limits are 5 model turns, 48,000 input tokens, 5,000 output tokens, 2 pre-commit probes, 4 post-commit probes.

Repair success uses the grader's target and preservation criteria. Invalid submissions and protocol violations count as repair failures. Missing or unscorable sessions are excluded from accuracy denominators and remain visible in coverage and outcome counts. Target success and retention loss average over graded patches. Forecast MAE includes every graded commitment, even when the patch fails validation.

| Actor | Access | Recorded / planned | Repair success | Target success | Retention loss | Forecast MAE | Probes used | Outcomes | Estimated API cost |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| gpt-6-luna | weights_only | 5/5 | 0/5 (0.0%) | — | — | 0.134 (n=3) | 0 | invalid_submission: 5 | $0.0151 |
| gpt-6-luna | behavior | 5/5 | 0/5 (0.0%) | 81.6% (n=3) | 25.5% (n=3) | 0.169 (n=5) | 26 | completed: 3, invalid_submission: 2 | $0.0165 |
| gpt-6-luna | weights_behavior | 5/5 | 0/5 (0.0%) | 75.7% (n=3) | 16.4% (n=3) | 0.166 (n=4) | 24 | completed: 3, invalid_submission: 2 | $0.0147 |
| gpt-6-luna | weights_activations | 5/5 | 0/5 (0.0%) | 57.3% (n=2) | 10.5% (n=2) | 0.152 (n=5) | 30 | completed: 2, invalid_submission: 3 | $0.0188 |
| gpt-6.1-sol | weights_only | 5/5 | 0/5 (0.0%) | — | — | — | 0 | invalid_submission: 5 | $0.2816 |
| gpt-6.1-sol | behavior | 5/5 | 4/5 (80.0%) | 80.8% (n=5) | 2.5% (n=5) | 0.143 (n=5) | 30 | completed: 5 | $0.2591 |
| gpt-6.1-sol | weights_behavior | 5/5 | 0/5 (0.0%) | — | — | — | 10 | invalid_submission: 5 | $0.3148 |
| gpt-6.1-sol | weights_activations | 5/5 | 0/5 (0.0%) | — | — | — | 10 | invalid_submission: 5 | $0.3164 |
| scripted | behavior | 5/5 | 1/5 (20.0%) | 45.4% (n=5) | 5.8% (n=5) | 0.146 (n=5) | 26 | completed: 5 | — |

| Matched contrast | Actor | Cases | Scored / planned repair pairs | Repair change | Repair transitions | Scored / planned forecast pairs | Forecast MAE change |
|---|---|---|---:|---:|---|---:|---:|
| Weights added to behavior access (primary) | gpt-6-luna | All | 5/5 | +0.0 pp | 0 gained / 0 lost | 4/5 | +0.008 |
| Weights added to behavior access (primary) | gpt-6-luna | Faults | 4/4 | +0.0 pp | 0 gained / 0 lost | 3/4 | +0.002 |
| Activations added to weights and behavior | gpt-6-luna | All | 5/5 | +0.0 pp | 0 gained / 0 lost | 4/5 | -0.002 |
| Activations added to weights and behavior | gpt-6-luna | Faults | 4/4 | +0.0 pp | 0 gained / 0 lost | 3/4 | +0.013 |
| Agent − scripted under behavior | gpt-6-luna | All | 5/5 | -20.0 pp | 0 gained / 1 lost | 5/5 | +0.023 |
| Agent − scripted under behavior | gpt-6-luna | Faults | 4/4 | +0.0 pp | 0 gained / 0 lost | 4/4 | +0.022 |
| Weights added to behavior access (primary) | gpt-6.1-sol | All | 5/5 | -80.0 pp | 0 gained / 4 lost | 0/5 | — |
| Weights added to behavior access (primary) | gpt-6.1-sol | Faults | 4/4 | -75.0 pp | 0 gained / 3 lost | 0/4 | — |
| Activations added to weights and behavior | gpt-6.1-sol | All | 5/5 | +0.0 pp | 0 gained / 0 lost | 0/5 | — |
| Activations added to weights and behavior | gpt-6.1-sol | Faults | 4/4 | +0.0 pp | 0 gained / 0 lost | 0/4 | — |
| Agent − scripted under behavior | gpt-6.1-sol | All | 5/5 | +60.0 pp | 3 gained / 0 lost | 5/5 | -0.003 |
| Agent − scripted under behavior | gpt-6.1-sol | Faults | 4/4 | +75.0 pp | 3 gained / 0 lost | 4/4 | -0.006 |

| gpt-6.1-sol access | Valid repairs / valid scored faults | Valid scored / planned faults | Ceiling |
|---|---:|---:|---|
| weights_only | 0/0 | 0/4 | Incomplete |
| behavior | 3/4 | 4/4 | Below ceiling |
| weights_behavior | 0/0 | 0/4 | Incomplete |
| weights_activations | 0/0 | 0/4 | Incomplete |

Total estimated API cost $1.2370 across 40 sessions with recorded usage cost. Unknown costs 0.
Weights-only scores are descriptive because that arm has no behavior observations.

| Actor | Case | Access | Submitted patch | Repair | Target success | Retention loss | Forecast MAE |
|---|---|---|---|---:|---:|---:|---:|
| gpt-6-luna | hard-01 (coupled_gates) | behavior | i01 | 0 | 100.0% | 23.4% | 0.153 |
| gpt-6-luna | hard-01 (coupled_gates) | weights_activations | — | 0 | — | — | 0.168 |
| gpt-6-luna | hard-01 (coupled_gates) | weights_behavior | i04 | 0 | 61.5% | 19.5% | 0.158 |
| gpt-6-luna | hard-01 (coupled_gates) | weights_only | — | 0 | — | — | 0.150 |
| gpt-6-luna | hard-02 (sensor_routing) | behavior | i05 | 0 | 100.0% | 38.3% | 0.215 |
| gpt-6-luna | hard-02 (sensor_routing) | weights_activations | — | 0 | — | — | 0.103 |
| gpt-6-luna | hard-02 (sensor_routing) | weights_behavior | — | 0 | — | — | — |
| gpt-6-luna | hard-02 (sensor_routing) | weights_only | — | 0 | — | — | 0.150 |
| gpt-6-luna | hard-03 (readout_redistribution) | behavior | i08 | 0 | 44.8% | 14.8% | 0.195 |
| gpt-6-luna | hard-03 (readout_redistribution) | weights_activations | i02 | 0 | 65.6% | 9.4% | 0.222 |
| gpt-6-luna | hard-03 (readout_redistribution) | weights_behavior | i02 | 0 | 65.6% | 9.4% | 0.197 |
| gpt-6-luna | hard-03 (readout_redistribution) | weights_only | — | 0 | — | — | 0.103 |
| gpt-6-luna | hard-04 (cross_layer_drift) | behavior | — | 0 | — | — | 0.160 |
| gpt-6-luna | hard-04 (cross_layer_drift) | weights_activations | i09 | 0 | 49.0% | 11.7% | 0.163 |
| gpt-6-luna | hard-04 (cross_layer_drift) | weights_behavior | i04 | 0 | 100.0% | 20.3% | 0.160 |
| gpt-6-luna | hard-04 (cross_layer_drift) | weights_only | — | 0 | — | — | — |
| gpt-6-luna | hard-05 (clean) | behavior | — | 0 | — | — | 0.124 |
| gpt-6-luna | hard-05 (clean) | weights_activations | — | 0 | — | — | 0.104 |
| gpt-6-luna | hard-05 (clean) | weights_behavior | — | 0 | — | — | 0.150 |
| gpt-6-luna | hard-05 (clean) | weights_only | — | 0 | — | — | — |
| gpt-6.1-sol | hard-01 (coupled_gates) | behavior | i13 | 0 | 4.2% | 12.5% | 0.144 |
| gpt-6.1-sol | hard-01 (coupled_gates) | weights_activations | — | 0 | — | — | — |
| gpt-6.1-sol | hard-01 (coupled_gates) | weights_behavior | — | 0 | — | — | — |
| gpt-6.1-sol | hard-01 (coupled_gates) | weights_only | — | 0 | — | — | — |
| gpt-6.1-sol | hard-02 (sensor_routing) | behavior | i04 | 1 | 100.0% | 0.0% | 0.187 |
| gpt-6.1-sol | hard-02 (sensor_routing) | weights_activations | — | 0 | — | — | — |
| gpt-6.1-sol | hard-02 (sensor_routing) | weights_behavior | — | 0 | — | — | — |
| gpt-6.1-sol | hard-02 (sensor_routing) | weights_only | — | 0 | — | — | — |
| gpt-6.1-sol | hard-03 (readout_redistribution) | behavior | i03 | 1 | 100.0% | 0.0% | 0.154 |
| gpt-6.1-sol | hard-03 (readout_redistribution) | weights_activations | — | 0 | — | — | — |
| gpt-6.1-sol | hard-03 (readout_redistribution) | weights_behavior | — | 0 | — | — | — |
| gpt-6.1-sol | hard-03 (readout_redistribution) | weights_only | — | 0 | — | — | — |
| gpt-6.1-sol | hard-04 (cross_layer_drift) | behavior | i11 | 1 | 100.0% | 0.0% | 0.126 |
| gpt-6.1-sol | hard-04 (cross_layer_drift) | weights_activations | — | 0 | — | — | — |
| gpt-6.1-sol | hard-04 (cross_layer_drift) | weights_behavior | — | 0 | — | — | — |
| gpt-6.1-sol | hard-04 (cross_layer_drift) | weights_only | — | 0 | — | — | — |
| gpt-6.1-sol | hard-05 (clean) | behavior | none | 1 | 100.0% | 0.0% | 0.103 |
| gpt-6.1-sol | hard-05 (clean) | weights_activations | — | 0 | — | — | — |
| gpt-6.1-sol | hard-05 (clean) | weights_behavior | — | 0 | — | — | — |
| gpt-6.1-sol | hard-05 (clean) | weights_only | — | 0 | — | — | — |
| scripted | hard-01 (coupled_gates) | behavior | i04 | 0 | 61.5% | 19.5% | 0.172 |
| scripted | hard-02 (sensor_routing) | behavior | none | 0 | 0.0% | 0.0% | 0.166 |
| scripted | hard-03 (readout_redistribution) | behavior | i02 | 0 | 65.6% | 9.4% | 0.131 |
| scripted | hard-04 (cross_layer_drift) | behavior | none | 0 | 0.0% | 0.0% | 0.164 |
| scripted | hard-05 (clean) | behavior | none | 1 | 100.0% | 0.0% | 0.097 |

The suite contains four independently fitted checkpoints with faults in
coupled hidden gates, sensor routing, readout redistribution, and cross-layer
drift, plus one independently fitted clean checkpoint. Each case has 14
candidate edits and one passing choice. The hidden retention bank deliberately
includes states sensitive to the offered edits. It measures preservation on
that challenge bank.

The 5,000-token run returned 13 valid submissions. Of the 27 invalid
submissions, 26 consumed the full output allowance. Sol consumed the full
allowance in all 15 arms with weights. Its behavior-only arm returned valid
patches on all five cases and missed the coupled-gates fault.

The global threshold decoy in this run was calculated from the maximum
target logit in the hidden bank, which disclosed an aggregate cue equally
to all access arms. The separate Sol budget check uses a fixed decoy and
a 12,000-token allowance. It reuses the same checkpoints and banks.

See the [Sol budget check](budget-check.md) for the matched follow-up. All
50 API settlements across both runs were reconciled. Combined estimated
cost was $2.1478.
