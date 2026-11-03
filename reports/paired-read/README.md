# Paired Read API pilot

Run on November 3, 2026. Three development checkpoint pairs, one attempt per
cell, three OpenAI API model aliases, and high reasoning effort throughout.
Each checkpoint has 20 scalars. Each input bank contains eight cases, four of
which have different target and donor labels.

The table reports target accuracy on the four disagreement cases per task.
Donor-following uses the donor label on those same inputs when the donor
checkpoint is shown. Invalid submissions and protocol violations receive zero
credit. Costs are estimated from the provider-reported token usage, including
reasoning and applicable cache categories, at Standard rates.

| Model | Condition | Scored / planned | True | Donor shown | Metadata | True − donor | Donor-following | Cost |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gpt-6-luna | A | 9/9 | 100% | 0% | 42% | +100 pp | 100% | $0.0029 |
| gpt-6-luna | B | 8/9 | 67% | 0% | 50% | +67 pp | 100% | $0.0042 |
| gpt-6.1-terra | A | 9/9 | 100% | 0% | 58% | +100 pp | 100% | $0.0191 |
| gpt-6.1-terra | B | 9/9 | 100% | 0% | 50% | +100 pp | 100% | $0.0233 |
| gpt-6.1-sol | A | 9/9 | 100% | 0% | 42% | +100 pp | 100% | $0.0477 |
| gpt-6.1-sol | B | 9/9 | 100% | 0% | 58% | +100 pp | 100% | $0.0515 |

Recorded usage cost: $0.1487 across 54/54 recorded sessions.

## Interpretation

A gives the model one response and zero exposed tools. B allows up to five
turns and three bounded tensor-analysis calls. B's metadata-only arm provides
no analysis operands. The A/B comparison therefore measures the complete
access protocol, including the extra turns. C is outside the paired Read
experiment.

The task-level sample size is three, with one shared linear mechanism. These
results measure checkpoint-following on these tiny networks. Model rankings
and generalization to deeper networks require broader task coverage.

Luna's B true-checkpoint score includes one zero-credit session. The tool
rejected an extra prefix in both operands of a matrix-multiplication request.
The tool help used the generic format `checkpoint:tensor` alongside a concrete
example such as `target:w0`. The original attempt and its cost are preserved.

[OpenAI API pricing](https://developers.openai.com/api/docs/pricing)
