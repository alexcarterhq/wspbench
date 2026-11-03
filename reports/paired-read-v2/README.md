# Nonlinear paired Read API pilot

November 3, 2026. Three synthetic 42-scalar ReLU checkpoint pairs, three
OpenAI API aliases at high reasoning effort, and five fresh repeat blocks.
A and B shared a four-turn maximum and a 4,096-token aggregate output cap.
B exposed the same tensor tool schema in every view, with exact handles for
the available arrays. Both conditions could request another reasoning turn.
The scheduled A/B cells were adjacent and alternated which condition went
first. Concurrent requests could finish out of order.

True-checkpoint accuracy was 100% in every model, condition, and repeat.
B cost less in all 15 model-by-repeat comparisons. Across all views,
estimated token cost was $0.7210 in A and $0.3415 in B, a 52.6% reduction.
On weight-present sessions, average reported reasoning tokens per session
fell from 1,459 to 186 for Luna, 1,123 to 23 for Terra, and 1,264 to 0 for Sol.
The tool path sent more input tokens and used fewer output tokens. These
results apply to the tested API aliases, settings, tasks, and rates. Accuracy
at the ceiling leaves the tool effect on harder tasks unresolved.

All 270 submissions were valid. The audit independently replayed scores from
the numeric checkpoints and saved answers, checked matched prompts and tool
and token limits, and reconciled all reservations against 539 usage-bearing
model turns. The existing 17 access tests passed.

## Paired Read development experiment

3 checkpoint pairs, 5 repeat blocks, 270/270 recorded sessions.
Each cell shows the median [minimum, maximum] across complete repeat blocks. Accuracy uses only target/donor disagreement inputs. Invalid submissions and protocol violations count as zero. Blocks with missing or unresolved sessions are excluded from every accuracy column in their row. Contrasts are computed within tasks and repeat blocks before aggregation.

| Model | Access | Complete blocks | True | Donor shown | Metadata | True − donor | True − metadata | Donor-following |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gpt-6-luna | A | 5/5 | 100.0% [100.0, 100.0] | 0.0% [0.0, 8.3] | 50.0% [33.3, 58.3] | +100.0 pp [+91.7, +100.0] | +50.0 pp [+41.7, +66.7] | 100.0% [91.7, 100.0] |
| gpt-6-luna | B | 5/5 | 100.0% [100.0, 100.0] | 0.0% [0.0, 0.0] | 50.0% [33.3, 66.7] | +100.0 pp [+100.0, +100.0] | +50.0 pp [+33.3, +66.7] | 100.0% [100.0, 100.0] |
| gpt-6.1-terra | A | 5/5 | 100.0% [100.0, 100.0] | 0.0% [0.0, 0.0] | 50.0% [50.0, 50.0] | +100.0 pp [+100.0, +100.0] | +50.0 pp [+50.0, +50.0] | 100.0% [100.0, 100.0] |
| gpt-6.1-terra | B | 5/5 | 100.0% [100.0, 100.0] | 0.0% [0.0, 0.0] | 50.0% [50.0, 50.0] | +100.0 pp [+100.0, +100.0] | +50.0 pp [+50.0, +50.0] | 100.0% [100.0, 100.0] |
| gpt-6.1-sol | A | 5/5 | 100.0% [100.0, 100.0] | 0.0% [0.0, 0.0] | 50.0% [50.0, 50.0] | +100.0 pp [+100.0, +100.0] | +50.0 pp [+50.0, +50.0] | 100.0% [100.0, 100.0] |
| gpt-6.1-sol | B | 5/5 | 100.0% [100.0, 100.0] | 0.0% [0.0, 0.0] | 50.0% [50.0, 50.0] | +100.0 pp [+100.0, +100.0] | +50.0 pp [+50.0, +50.0] | 100.0% [100.0, 100.0] |

The B − A contrast compares true-checkpoint accuracy within common complete blocks. Costs include all recorded sessions with known usage for each condition.

| Model | Common blocks | A true scores by block | B true scores by block | Paired B − A | A estimated cost | B estimated cost |
|---|---:|---|---|---:|---:|---:|
| gpt-6-luna | 5/5 | 1: 100.0%, 2: 100.0%, 3: 100.0%, 4: 100.0%, 5: 100.0% | 1: 100.0%, 2: 100.0%, 3: 100.0%, 4: 100.0%, 5: 100.0% | +0.0 pp [+0.0, +0.0] | $0.0328 | $0.0236 |
| gpt-6.1-terra | 5/5 | 1: 100.0%, 2: 100.0%, 3: 100.0%, 4: 100.0%, 5: 100.0% | 1: 100.0%, 2: 100.0%, 3: 100.0%, 4: 100.0%, 5: 100.0% | +0.0 pp [+0.0, +0.0] | $0.2164 | $0.1405 |
| gpt-6.1-sol | 5/5 | 1: 100.0%, 2: 100.0%, 3: 100.0%, 4: 100.0%, 5: 100.0% | 1: 100.0%, 2: 100.0%, 3: 100.0%, 4: 100.0%, 5: 100.0% | +0.0 pp [+0.0, +0.0] | $0.4718 | $0.1773 |

Repeat blocks reuse the same checkpoint pairs. Their ranges describe run-to-run variation on those pairs. Estimating uncertainty on new targets requires more independent pairs.

| Model | Access | View | Recorded / planned | Outcome counts | Analysis calls: total / sessions with calls | Estimated cost | Unknown costs |
|---|---|---|---:|---|---:|---:|---:|
| gpt-6-luna | A | full | 15/15 | completed: 15 | 0 / 0 | $0.0123 | 0 |
| gpt-6-luna | A | matched_wrong | 15/15 | completed: 15 | 0 / 0 | $0.0122 | 0 |
| gpt-6-luna | A | masked | 15/15 | completed: 15 | 0 / 0 | $0.0083 | 0 |
| gpt-6-luna | B | full | 15/15 | completed: 15 | 43 / 15 | $0.0073 | 0 |
| gpt-6-luna | B | matched_wrong | 15/15 | completed: 15 | 42 / 15 | $0.0074 | 0 |
| gpt-6-luna | B | masked | 15/15 | completed: 15 | 4 / 4 | $0.0089 | 0 |
| gpt-6.1-terra | A | full | 15/15 | completed: 15 | 0 / 0 | $0.0961 | 0 |
| gpt-6.1-terra | A | matched_wrong | 15/15 | completed: 15 | 0 / 0 | $0.0978 | 0 |
| gpt-6.1-terra | A | masked | 15/15 | completed: 15 | 0 / 0 | $0.0226 | 0 |
| gpt-6.1-terra | B | full | 15/15 | completed: 15 | 45 / 15 | $0.0585 | 0 |
| gpt-6.1-terra | B | matched_wrong | 15/15 | completed: 15 | 45 / 15 | $0.0583 | 0 |
| gpt-6.1-terra | B | masked | 15/15 | completed: 15 | 0 / 0 | $0.0237 | 0 |
| gpt-6.1-sol | A | full | 15/15 | completed: 15 | 0 / 0 | $0.2129 | 0 |
| gpt-6.1-sol | A | matched_wrong | 15/15 | completed: 15 | 0 / 0 | $0.2172 | 0 |
| gpt-6.1-sol | A | masked | 15/15 | completed: 15 | 0 / 0 | $0.0417 | 0 |
| gpt-6.1-sol | B | full | 15/15 | completed: 15 | 45 / 15 | $0.0665 | 0 |
| gpt-6.1-sol | B | matched_wrong | 15/15 | completed: 15 | 45 / 15 | $0.0665 | 0 |
| gpt-6.1-sol | B | masked | 15/15 | completed: 15 | 0 / 0 | $0.0444 | 0 |

Total estimated token cost from recorded usage: $1.0625. Unknown costs: 0.
Audit status: pending: 270.

[Official OpenAI API pricing](https://developers.openai.com/api/docs/pricing)
