# Full selective repair model comparison

Each model faced 32 faults and 8 clean controls from 8 seed blocks, once with behavior probes and once with the same probes plus weights. Limits were 5 turns, 48,000 input tokens, 24,000 output tokens, 2 pre-commit probes, and 4 post-commit probes per session.

The comparison reuses the original Sol run. Astra, Luna, and Terra ran in a separate interleaved batch, initially with one attempt per case and condition. All four ran at high reasoning effort using the listed API aliases.

We retried only sessions interrupted by API errors, reducing concurrency from 16 to 4. 50/50 retries returned records and 42 resolved the original API interruption. The comparison uses the retry outcome for those sessions and keeps both attempts in the cost totals. Metered retry cost was $18.0192 (8 attempts with incomplete usage).

Recorded 320/320 planned API sessions. Estimated cost from recorded usage was $59.1622 (58 attempts with incomplete usage).

| API model | Reasoning effort | Access | Faults repaired | Clean preserved | Recorded / planned | Invalid (output cap) | Infrastructure failures | Estimated cost |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `gpt-6.1-sol` | high | Behavior probes | 14/32 | 8/8 | 40/40 | 0 (0) | 0 | $2.0224 |
| `gpt-6.1-sol` | high | Weights and behavior probes | 30/32 | 8/8 | 40/40 | 0 (0) | 0 | $6.3445 |
| `gpt-6-luna` | high | Behavior probes | 8/32 | 1/8 | 40/40 | 0 (0) | 0 | $0.1307 (4 attempts with incomplete usage) |
| `gpt-6-luna` | high | Weights and behavior probes | 20/32 | 0/8 | 40/40 | 2 (0) | 0 | $0.1844 (2 attempts with incomplete usage) |
| `gpt-6.1-terra` | high | Behavior probes | 14/32 | 8/8 | 40/40 | 0 (0) | 0 | $1.0357 (2 attempts with incomplete usage) |
| `gpt-6.1-terra` | high | Weights and behavior probes | 28/31 | 8/8 | 40/40 | 0 (0) | 1 | $2.4638 (7 attempts with incomplete usage) |
| `gpt-6-astra` | high | Behavior probes | 17/32 | 8/8 | 40/40 | 0 (0) | 0 | $11.4711 (14 attempts with incomplete usage) |
| `gpt-6-astra` | high | Weights and behavior probes | 27/27 | 6/6 | 40/40 | 0 (0) | 7 | $35.5095 (29 attempts with incomplete usage) |
| Scripted baseline | — | Behavior probes | 5/32 | 8/8 | 40/40 | — | — | — |

Invalid submissions and protocol violations count as failures. Pending sessions and infrastructure failures are excluded from score denominators. Costs include metered turns from interrupted attempts. Incomplete usage records leave some charges unknown.

| API model | Matched fault pairs | Repairs gained / lost / tied | Change with weights | 95% interval | Forecast MAE change (pairs) | Model API cost |
|---|---:|---:|---:|---:|---:|---:|
| `gpt-6.1-sol` | 32/32 | 16 / 0 / 16 | +50.0 pp | +28.1 pp to +71.9 pp | -0.117 (32) | $8.3669 |
| `gpt-6-luna` | 32/32 | 15 / 3 / 14 | +37.5 pp | +18.8 pp to +56.2 pp | -0.047 (32) | $0.3151 (6 attempts with incomplete usage) |
| `gpt-6.1-terra` | 31/32 | 16 / 2 / 13 | +45.2 pp | +23.3 pp to +67.7 pp | -0.092 (32) | $3.4995 (9 attempts with incomplete usage) |
| `gpt-6-astra` | 27/32 | 14 / 0 / 13 | +51.9 pp | +34.6 pp to +69.2 pp | -0.112 (27) | $46.9807 (43 attempts with incomplete usage) |

Each interval resamples seed blocks in 10,000 bootstrap draws. Negative forecast MAE changes mean lower error with weights. Forecast differences use pairs with commitments in both arms.
Assigning all possible binary outcomes to unresolved sessions gives `gpt-6.1-terra` an all-pairs effect range of +43.8 pp to +46.9 pp.
Assigning all possible binary outcomes to unresolved sessions gives `gpt-6-astra` an all-pairs effect range of +31.2 pp to +46.9 pp.

| API model | Fault mechanism | Behavior probes | Weights and behavior probes |
|---|---|---:|---:|
| `gpt-6.1-sol` | coupled gates | 3/8 | 8/8 |
| `gpt-6.1-sol` | cross layer drift | 3/8 | 8/8 |
| `gpt-6.1-sol` | readout redistribution | 1/8 | 6/8 |
| `gpt-6.1-sol` | sensor routing | 7/8 | 8/8 |
| `gpt-6-luna` | coupled gates | 2/8 | 6/8 |
| `gpt-6-luna` | cross layer drift | 2/8 | 4/8 |
| `gpt-6-luna` | readout redistribution | 2/8 | 5/8 |
| `gpt-6-luna` | sensor routing | 2/8 | 5/8 |
| `gpt-6.1-terra` | coupled gates | 3/8 | 5/7 |
| `gpt-6.1-terra` | cross layer drift | 2/8 | 8/8 |
| `gpt-6.1-terra` | readout redistribution | 4/8 | 7/8 |
| `gpt-6.1-terra` | sensor routing | 5/8 | 8/8 |
| `gpt-6-astra` | coupled gates | 5/8 | 5/5 |
| `gpt-6-astra` | cross layer drift | 3/8 | 8/8 |
| `gpt-6-astra` | readout redistribution | 2/8 | 7/7 |
| `gpt-6-astra` | sensor routing | 7/8 | 7/7 |
