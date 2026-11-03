# GPT development comparison

Same preselected development cases, A/full, one attempt per model. No combined capability score or leaderboard claim. Website work remains deferred.

Status: **complete**. Planned 18; launched 18; scored 18; infrastructure failures 0; missing 0; inflight 0.

| Model | Read exact | Write exact | Confirmed JSON objects / responses | Median scored-session seconds | Known API estimate | Unknown-cost sessions |
|---|---:|---:|---:|---:|---:|---:|
| openai/gpt-6-luna | 0/4 | 0/2 | 6/6 | 117.29 | $0.083433 | 0 |
| openai/gpt-6.1-terra | 0/4 | 1/2 | 6/6 | 147.94 | $0.746173 | 0 |
| openai/gpt-6.1-sol | 0/4 | 1/2 | 6/6 | 122.94 | $1.428556 | 0 |

Infrastructure failures are not failed capability answers. Known cost excludes unresolved charges; it is not a complete total when unknown-cost sessions exist. Latency includes the local session and API round trip.

## Per-family results

| Model | Family | Exact / scored | Functional accuracy | Scored parents |
|---|---|---:|---:|---:|
| openai-luna | compositional_synthesis | 0/2 | 71.9% | 2 |
| openai-luna | edit_effects | 0/2 | 45.6% | 2 |
| openai-luna | rule_recovery | 0/2 | 42.7% | 2 |
| openai-terra | compositional_synthesis | 1/2 | 97.3% | 2 |
| openai-terra | edit_effects | 0/2 | 53.4% | 2 |
| openai-terra | rule_recovery | 0/2 | 78.5% | 2 |
| openai-sol | compositional_synthesis | 1/2 | 95.8% | 2 |
| openai-sol | edit_effects | 0/2 | 86.7% | 2 |
| openai-sol | rule_recovery | 0/2 | 88.6% | 2 |

## Write control context

Static difficulty control on the identical selected targets. Recorded in B, it reads no weights or feedback; it is neither an A model score nor additional independent tasks.

The static zero-start/no-edit control has exact success **0/2**, target accuracy 70.1%, affected-case accuracy 0.0%, and retention 100.0%. No independent tasks are added.

Class imbalance makes raw Write accuracy insufficient evidence of successful construction.

| Model | Write target accuracy | Affected-case accuracy | Retention | Exact / scored |
|---|---:|---:|---:|---:|
| openai-luna | 71.9% | 7.6% | 99.6% | 0/2 |
| openai-terra | 97.3% | 96.1% | 97.7% | 1/2 |
| openai-sol | 95.8% | 93.4% | 96.7% | 1/2 |

## API tokens

Output includes API-reported reasoning; the reasoning column is a subset, not an additional billable output category.

| Model | Input total | Uncached | Cache write | Cache read | Output | Reasoning subset |
|---|---:|---:|---:|---:|---:|---:|
| openai-luna | 310938 | 18 | 310920 | 0 | 89132 | 87169 |
| openai-terra | 310938 | 18 | 310920 | 0 | 71501 | 67367 |
| openai-sol | 310938 | 18 | 310920 | 0 | 65122 | 59867 |

## Replay and accounting

These are saved-submission results. CPU replay verifies their scores and artifact consistency without generating new model responses. API costs are estimates from released usage and frozen prices, not invoices.


## Interpretation

- No combined Read+Write capability score is calculated.
- Invalid submissions and protocol violations receive zero exact credit. Infrastructure failures are excluded from capability denominators and counted separately.
- Read functional accuracy is R1 agreement or the mean of R3 before/after agreement, once per parent. Write uses target accuracy. Invalid submissions/protocol violations receive zero derived functional credit; original measured family metrics retain their own observation counts.
- API output tokens include reported reasoning tokens. Reasoning is recorded separately and never charged twice. Visible-answer throughput subtracts reported reasoning tokens.
- Latency is local session elapsed time, not isolated provider compute time. Infrastructure latency is separate.
- Models are aliases unless their configuration states an immutable snapshot. This tiny development sample cannot support a headline ranking.
- Write outputs are class-imbalanced. Raw target accuracy alone does not demonstrate successful construction; compare the static zero-start/no-edit control, affected-case accuracy, retention and exact success.
- Skill and family rows overlap; do not sum them together. Transport canaries are not capability trials.
- Offline replay verifies saved submissions, scores and artifact consistency. It does not rerun proprietary model inference or independently attest provider billing.
