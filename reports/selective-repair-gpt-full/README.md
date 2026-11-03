# Full selective repair model comparison

Each model faced 32 faults and 8 clean controls from 8 seed blocks, once with behavior probes and once with the same probes plus weights. Limits were 5 turns, 48,000 input tokens, 24,000 output tokens, 2 pre-commit probes, and 4 post-commit probes per session.

The comparison reuses the original Sol run. Astra, Luna, and Terra ran in a separate interleaved batch, initially with one attempt per case and condition. All four ran at high reasoning effort using the listed API aliases.

We retried only sessions interrupted by API errors, reducing concurrency from 16 to 4. The first round recorded 50/50 attempts and recovered 42 scoreable outcomes. We then retried the remaining timeouts at concurrency 1. This recorded 8/8 attempts and recovered 8 more scoreable outcomes. Scores use the latest recorded outcome for each session. Costs include metered usage from every attempt. Metered retry cost was $24.6071 (8 attempts with incomplete usage).

Recorded 320/320 planned API sessions. Estimated cost from recorded usage was $65.7501 (58 attempts with incomplete usage).

| API model | Reasoning effort | Access | Faults repaired | Clean preserved | Recorded / planned | Invalid (output cap) | Infrastructure failures | Estimated cost |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `gpt-6.1-sol` | high | Behavior probes | 14/32 | 8/8 | 40/40 | 0 (0) | 0 | $2.0224 |
| `gpt-6.1-sol` | high | Weights and behavior probes | 30/32 | 8/8 | 40/40 | 0 (0) | 0 | $6.3445 |
| `gpt-6-luna` | high | Behavior probes | 8/32 | 1/8 | 40/40 | 0 (0) | 0 | $0.1307 (4 attempts with incomplete usage) |
| `gpt-6-luna` | high | Weights and behavior probes | 20/32 | 0/8 | 40/40 | 2 (0) | 0 | $0.1844 (2 attempts with incomplete usage) |
| `gpt-6.1-terra` | high | Behavior probes | 14/32 | 8/8 | 40/40 | 0 (0) | 0 | $1.0357 (2 attempts with incomplete usage) |
| `gpt-6.1-terra` | high | Weights and behavior probes | 29/32 | 8/8 | 40/40 | 0 (0) | 0 | $2.5085 (7 attempts with incomplete usage) |
| `gpt-6-astra` | high | Behavior probes | 17/32 | 8/8 | 40/40 | 0 (0) | 0 | $11.4711 (14 attempts with incomplete usage) |
| `gpt-6-astra` | high | Weights and behavior probes | 32/32 | 8/8 | 40/40 | 0 (0) | 0 | $42.0527 (29 attempts with incomplete usage) |
| Scripted baseline | — | Behavior probes | 5/32 | 8/8 | 40/40 | — | — | — |

Invalid submissions and protocol violations count as failures. Pending sessions and infrastructure failures are excluded from score denominators. Costs include metered turns from interrupted attempts. Incomplete usage records leave some charges unknown.

| API model | Matched fault pairs | Repairs gained / lost / tied | Change with weights | 95% interval | Forecast MAE change (pairs) | Model API cost |
|---|---:|---:|---:|---:|---:|---:|
| `gpt-6.1-sol` | 32/32 | 16 / 0 / 16 | +50.0 pp | +28.1 pp to +71.9 pp | -0.117 (32) | $8.3669 |
| `gpt-6-luna` | 32/32 | 15 / 3 / 14 | +37.5 pp | +18.8 pp to +56.2 pp | -0.047 (32) | $0.3151 (6 attempts with incomplete usage) |
| `gpt-6.1-terra` | 32/32 | 17 / 2 / 13 | +46.9 pp | +28.1 pp to +68.8 pp | -0.093 (32) | $3.5442 (9 attempts with incomplete usage) |
| `gpt-6-astra` | 32/32 | 15 / 0 / 17 | +46.9 pp | +34.4 pp to +59.4 pp | -0.110 (32) | $53.5238 (43 attempts with incomplete usage) |

Each interval resamples seed blocks in 10,000 bootstrap draws. Negative forecast MAE changes mean lower error with weights. Forecast differences use pairs with commitments in both arms.

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
| `gpt-6.1-terra` | coupled gates | 3/8 | 6/8 |
| `gpt-6.1-terra` | cross layer drift | 2/8 | 8/8 |
| `gpt-6.1-terra` | readout redistribution | 4/8 | 7/8 |
| `gpt-6.1-terra` | sensor routing | 5/8 | 8/8 |
| `gpt-6-astra` | coupled gates | 5/8 | 8/8 |
| `gpt-6-astra` | cross layer drift | 3/8 | 8/8 |
| `gpt-6-astra` | readout redistribution | 2/8 | 8/8 |
| `gpt-6-astra` | sensor routing | 7/8 | 8/8 |
