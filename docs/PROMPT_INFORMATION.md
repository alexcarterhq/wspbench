# Prompt information and attribution

The access condition and the information in the prompt are independent variables. A uses serialized weights without external tools; B adds bounded tensor analysis; C adds metered behavior and activation probes. None of these labels says whether the task prompt supplies a diagnosis.

Every v0.2 generator records `public.prompt_information` with a hint level (`neutral`, `symptom`, or `mechanism`), diagnosis scope, source, optimization status and history. Missing provenance is unknown. A prompt must not be called neutral merely because its diagnostic hints were written by an LLM. Actual canonical payload and delivered prompt hashes are recorded. Reports separate information strata, and condition comparisons require the same task, weights, question and hint content.

The default mechanism-informed controls are coupled repair, whose declared fault class supplies diagnostic prior information, and loop repair, whose explicit objective supplies the looping diagnosis. These test repair conditional on that information. The other core tasks receive their required task specification without a fault diagnosis. Internal family names, split membership, mechanism identifiers and target provenance are omitted from the solver payload; they remain available to the evaluator for reporting and clustering.

## What the new author clarification changes

The supplied account says that the initial prompt did not contain a looping diagnosis; an autonomous prompt expansion introduced the hint before the successful repair, and the researchers recognized the looping explanation afterward. This weakens the accusation that humans deliberately supplied the diagnosis. It does not remove the information from the prompt received by the successful solver.

The released [experiment README](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/README.md) identifies the included run as prompt version 3 and says earlier runs are omitted. It therefore does not establish the complete sequence of prompt changes or the causal importance of each hint. WSPBench should neither infer human intervention from the final prompt nor infer that the hints were irrelevant.

The resulting benchmark claims must distinguish:

1. **Failure discovery:** identify a failure mechanism from a task-only instruction under the declared access condition.
2. **Diagnosis-conditioned repair:** improve behavior after receiving a symptom or mechanism, with retention checks.
3. **Capability improvement:** improve correctness beyond rescuing incomplete/looping answers, on independently selected tasks and completed-case controls.

The existing loop task measures the second claim. It is a C-only extension and cannot establish the first or third. Completion rate, conditional accuracy on originally completed cases, legitimate long/repetitive sequences, clean controls and sham edits must remain separate measurements.

## Later experiments

Do not multiply the initial paid matrix by every hint level. First validate the neutral core on CPU. A later paired discovery experiment can compare task-only, symptom and mechanism prompts on the same held-out targets, with access fixed. It needs a blind diagnostic answer committed before repair, a predeclared mechanism answer grammar and ambiguity review. The current implementation does not claim to provide this discovery experiment.

If prompt optimization is added, freeze and publish its initial prompt, every revision, optimizer model/configuration, timestamps, data exposure, feedback, human interventions and resource usage. Use development targets only and charge its complete cost. Treat the optimized prompt/scaffold as part of the evaluated system; do not attribute its contribution to a single downstream model invocation.

The original activation-direction method has a relevant [directional ablation precedent](https://arxiv.org/abs/2406.11717). A strong fixed contrastive-activation pipeline is consequently a required comparator for instrumented loop repair. Beating no-op or random edits alone cannot establish a novel form of weight understanding.
