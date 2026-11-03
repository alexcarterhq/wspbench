# Full selective repair model comparison

Each model faced 32 faults and 8 clean controls from 8 seed blocks, once with behavior probes and once with the same probes plus weights. Limits were 5 turns, 48,000 input tokens, 24,000 output tokens, 2 pre-commit probes, and 4 post-commit probes per session.

The comparison reuses the original Sol run. Astra, Luna, and Terra ran in a separate interleaved batch, with one attempt per case and condition. All four ran at high reasoning effort using the listed API aliases.

Recorded 320/320 planned API sessions. Estimated cost from recorded usage was $41.1430 (50 sessions with incomplete usage).

| API model | Reasoning effort | Access | Faults repaired | Clean preserved | Recorded / planned | Invalid (output cap) | Infrastructure failures | Estimated cost |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `gpt-6.1-sol` | high | Behavior probes | 14/32 | 8/8 | 40/40 | 0 (0) | 0 | $2.0224 |
| `gpt-6.1-sol` | high | Weights and behavior probes | 30/32 | 8/8 | 40/40 | 0 (0) | 0 | $6.3445 |
| `gpt-6-luna` | high | Behavior probes | 7/28 | 1/8 | 40/40 | 0 (0) | 4 | $0.1184 (4 sessions with incomplete usage) |
| `gpt-6-luna` | high | Weights and behavior probes | 19/30 | 0/8 | 40/40 | 1 (0) | 2 | $0.1768 (2 sessions with incomplete usage) |
| `gpt-6.1-terra` | high | Behavior probes | 13/30 | 8/8 | 40/40 | 0 (0) | 2 | $0.9746 (2 sessions with incomplete usage) |
| `gpt-6.1-terra` | high | Weights and behavior probes | 25/28 | 6/6 | 40/40 | 0 (0) | 6 | $2.1259 (6 sessions with incomplete usage) |
| `gpt-6-astra` | high | Behavior probes | 12/19 | 7/7 | 40/40 | 0 (0) | 14 | $7.5135 (14 sessions with incomplete usage) |
| `gpt-6-astra` | high | Weights and behavior probes | 14/14 | 4/4 | 40/40 | 0 (0) | 22 | $21.8668 (22 sessions with incomplete usage) |
| Scripted baseline | — | Behavior probes | 5/32 | 8/8 | 40/40 | — | — | — |

Invalid submissions and protocol violations count as failures. Pending sessions and infrastructure failures are excluded from score denominators. Costs include metered turns from interrupted sessions. Unknown costs refer to sessions with incomplete API usage records.

| API model | Matched fault pairs | Repairs gained / lost / tied | Change with weights | 95% interval | Forecast MAE change (pairs) | Model API cost |
|---|---:|---:|---:|---:|---:|---:|
| `gpt-6.1-sol` | 32/32 | 16 / 0 / 16 | +50.0 pp | +28.1 pp to +71.9 pp | -0.117 (32) | $8.3669 |
| `gpt-6-luna` | 27/32 | 14 / 2 / 11 | +44.4 pp | +25.0 pp to +66.7 pp | -0.035 (29) | $0.2952 (6 sessions with incomplete usage) |
| `gpt-6.1-terra` | 26/32 | 13 / 2 / 11 | +42.3 pp | +21.4 pp to +64.0 pp | -0.092 (29) | $3.1005 (8 sessions with incomplete usage) |
| `gpt-6-astra` | 9/32 | 2 / 0 / 7 | +22.2 pp | +0.0 pp to +50.0 pp | -0.106 (12) | $29.3804 (36 sessions with incomplete usage) |

Each interval resamples seed blocks in 10,000 bootstrap draws. Negative forecast MAE changes mean lower error with weights. Forecast differences use pairs with commitments in both arms.
Assigning all possible binary outcomes to unresolved sessions gives `gpt-6-luna` an all-pairs effect range of +25.0 pp to +43.8 pp.
Assigning all possible binary outcomes to unresolved sessions gives `gpt-6.1-terra` an all-pairs effect range of +31.2 pp to +50.0 pp.
Assigning all possible binary outcomes to unresolved sessions gives `gpt-6-astra` an all-pairs effect range of -34.4 pp to +62.5 pp.

| API model | Fault mechanism | Behavior probes | Weights and behavior probes |
|---|---|---:|---:|
| `gpt-6.1-sol` | coupled gates | 3/8 | 8/8 |
| `gpt-6.1-sol` | cross layer drift | 3/8 | 8/8 |
| `gpt-6.1-sol` | readout redistribution | 1/8 | 6/8 |
| `gpt-6.1-sol` | sensor routing | 7/8 | 8/8 |
| `gpt-6-luna` | coupled gates | 2/7 | 6/8 |
| `gpt-6-luna` | cross layer drift | 1/7 | 4/8 |
| `gpt-6-luna` | readout redistribution | 2/6 | 5/7 |
| `gpt-6-luna` | sensor routing | 2/8 | 4/7 |
| `gpt-6.1-terra` | coupled gates | 3/8 | 4/6 |
| `gpt-6.1-terra` | cross layer drift | 2/8 | 8/8 |
| `gpt-6.1-terra` | readout redistribution | 4/7 | 6/7 |
| `gpt-6.1-terra` | sensor routing | 4/7 | 7/7 |
| `gpt-6-astra` | coupled gates | 3/3 | 4/4 |
| `gpt-6-astra` | cross layer drift | 2/5 | 3/3 |
| `gpt-6-astra` | readout redistribution | 1/4 | 3/3 |
| `gpt-6-astra` | sensor routing | 6/7 | 4/4 |
