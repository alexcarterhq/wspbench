# WSPBench v0.2: three access conditions

Status: v0.2 candidate implementation with CPU validation. The existing v0.1 pool remains a sanity suite, with a runtime guard against headline promotion. Six generators, the A/B/C runner, trusted scoring and conventional baselines are implemented under `src/wspbench/v2`. This document also contains research targets beyond the implemented bounded grammar. See the README for exact scope and the CPU report for admission decisions. No paid v0.2 runs or website changes are part of this implementation.

Prompt information is a separate axis from access. See [the prompt-information policy](PROMPT_INFORMATION.md): an LLM-generated diagnostic hint is still information received by the solver, while its origin matters to attribution. Current coupled repair is fault-aware and loop repair is explicitly diagnosis-supplied. Neither is an autonomous failure-discovery task.

## 1. What the benchmark measures

Keep Read, Repair and Write as task skills. Make access condition an independent experimental axis, rather than describing every successful weight edit as native weight understanding.

| Condition | Solver access | Supported interpretation |
| --- | --- | --- |
| A — Unaided | Architecture, specification/question and serialized weights; no external tools | Unaided performance on weight-based questions and zero-feedback edits |
| B — Computational weight analysis | Exactly A's information plus CPU tensor algebra; no target executions, outputs or activations | Benefit from external computation over weights |
| C — Instrumented diagnosis and repair | Exactly B's starting information plus metered behavioral and activation probes | Practical diagnosis/repair with experimental feedback |

For matched comparisons, use the same target, question, goal, architecture, initial tensors and numerical precision. B reconstructs its arrays from exactly the values shown to A. C begins with those same values. A's prohibition concerns external tools: the benchmark cannot establish whether the model internally simulates a forward pass.

Use two proposed small-target capacity tiers, at most 512 and 4,096 scalar values, with one fixed architecture per family/tier. These are feasibility targets, not verified claims about the new tasks: admit a tier only after its complete-domain reference works within the frozen architecture. A family may use only the larger tier if the smaller one is not representable. Count embeddings, positional state, biases and every persistent constant. Store float32 values and serialize them with round-trip precision; reject instances whose correct answers change through the serialization/materialization path. All conditions see the same neutral tensor and channel names. Do not give B/C more precise arrays or a larger checkpoint in a comparison labeled as an access-condition effect.

Keep compiled/constructed and freshly trained targets as separate reported strata. Compiled models provide exact mechanism controls; trained models test whether methods transfer beyond compiler layout. No semantic channel names, compiler intermediate representations, clean checkpoints, training labels or generating source programs enter the solver view unless explicitly declared as the task specification.

The matched core contains five families below. Loop repair is a separately reported C-only extension, because that task's diagnostic evidence should not be conflated with weights-only interpretation. Larger B/C-only networks can be added later, outside A/B comparisons.

## 2. Concrete task families

### R1 — Recover a composed sequence rule (A/B/C)

- **Target:** a small attention or recurrent network over a three-symbol vocabulary and sequences of length one through five. Its function composes routing, comparison, counting and conditional selection. A concrete example is selecting the symbol preceding the last occurrence of the sequence's first token, with a declared default when no predecessor exists. Training and compiled strata use matched interfaces.
- **Given:** the weights, architecture, token mapping, domain and a public grammar of possible operations. The generating program is hidden. Architecture and boundary layers are matched across candidate behaviors so dimensions alone cannot reveal the answer.
- **Submit:** a constrained expression/program describing the behavior, optionally with confidence. Equivalent expressions are accepted.
- **Score:** execute the submitted expression and target in the trusted evaluator and measure functional agreement, not agreement with the author's wording or preferred circuit. Admit trained checkpoints only when they match a representable teacher exactly over the finite domain, so every admitted case has an attainable answer in the public grammar. Exhaustive checks certify admission; headline A/B/C comparisons use the same fixed hidden final input bank, disjoint from C's practice bank.
- **Strong baselines:** a separately trained balanced statistical/spectral reader; signed path products; QK/OV compositions and feature matching; architecture-aware symbolic decompilation and candidate-program enumeration without target execution.
- **Admission risk:** compiler layouts or first/last-layer geometry may solve the task. Hold out compositions and implementation patterns, not just training seeds. Families saturated by these baselines remain sanity tasks. C additionally needs a no-informative-weights, behavior-only program-identification baseline: the 363 possible nonempty strings in this domain may be easy to identify from 128 queries alone.

### R2 — Forecast selective causal interventions (A/B/C)

- **Target:** paired networks with the same public interface and external behavior but different internal implementations. Example: one output computes parity and another counts a symbol; shared channels differ between the two implementations.
- **Given:** weights, the two declared behaviors and a menu of matched-budget head/channel/projection interventions. Include sham and norm-matched control edits. Randomize menu order and neutral component IDs.
- **Submit:** which interventions will alter each behavior, the signed effect or a predeclared effect band, and confidence. Do not score a free-text explanation.
- **Score:** the trusted evaluator applies every scored intervention and measures changed-output fractions over the entire public 363-input domain. The implementation holds out intervention identities, rather than asking for exact fractions on a secret random input subset. That latter quantity would contain unknowable sampling noise even for a correct mechanism. Accept empirical equivalence classes when several edits have the same effect. Reject ambiguous cases before any solver runs.
- **Strong baselines:** output-layer-only analysis, weight norms, path products, linearized circuit approximations and known circuit-template matching. C additionally receives behavior-only and activation-assisted causal discovery baselines.
- **Feedback separation:** predeclare distinct practice and final intervention identities/compositions, not just different inputs for the same intervention. The forecast probe endpoint accepts only IDs from the practice intervention catalog; arbitrary edited checkpoints cannot be executed in forecast trials. Match equivalent numeric encodings of a patch to the same intervention identity. C's forecast-transfer score uses unqueried interventions and the deterministic public-domain estimand. Other families retain their fixed disjoint practice/final input banks. A forecast committed before any feedback is a separate prefeedback diagnostic, not C's postfeedback forecast score. Report C as instrumented forecasting, not weights-only prediction.

### R3 — Explain an edit's behavioral consequence (A/B/C)

- **Target:** matched before/after checkpoints with identical architecture and input/output layers. The change affects a conditional internal computation: for example, copying the previous token rather than the next token only when a mode/query condition holds.
- **Given:** both weight sets, the input domain and a predeclared answer grammar for affected subpopulations and behavioral changes. Do not reveal the edit's generating label or intention.
- **Submit:** a predicate describing affected inputs and the resulting functional change, plus confidence or a predeclared effect band.
- **Score:** compare predicted and actual changed-input sets and output changes on the fixed hidden final bank, with exhaustive-domain correctness as an admission/secondary check. Report false positives as well as recall. Equal-norm edits with different behavioral consequences and function-preserving changes serve as controls.
- **Strong baselines:** delta norms, changed-tensor/head identity, output-boundary features, matrix-path differences and architecture-aware symbolic analysis. A change detector that merely identifies a modified tensor does not receive semantic-change credit.

### P1 — Restore coupled computation (A/B/C)

- **Target:** a dense network implementing several coupled behaviors. Apply an information-preserving fault from a declared class, such as a channel/projection routing permutation, sign/gain corruption or a composition of these at an internal site. Do not deliberately reserve unused repair neurons or supply a clean checkpoint.
- **Given:** the corrupted weights, fault-class description, desired behavioral constraints and a small fixed set of normative examples derived from the specification. A/B receive no measured target outputs or activations. Normative examples are desired answers, not traces from executing the faulty target.
- **Submit:** a numeric patch, including the empty patch for clean controls. Include a precommitted causal prediction when localization is being assessed; explanations are supplementary.
- **Score:** recovery of the affected behavior and preservation across diverse unaffected inputs and outputs. Include clean targets and faults at different depths. Report edit size and norm; sparsity alone does not establish diagnosis.
- **Strong baselines:** no-op; bias/gain correction; candidate inverse-corruption analysis; head replacement; public-spec construction plus copy-and-splice; auxiliary-null-space updates; permitted analytic low-rank edits. All are legitimate competitors, not posthoc rule violations.
- **Admission requirement:** construct matched targets with identical public specification/architecture but different checkpoint-specific learned behavior that must survive repair through the coupled computation. Merely copying disconnected auxiliary outputs is insufficient. Declare a finite fault-hypothesis class and supply normative witnesses that distinguish its relevant behavioral-equivalence classes over the final domain. Certify this identifiability before admission, including any tolerance ambiguity. Test for universal specification-only and structural splice patches. Independently demonstrate a solution under each claimed condition's information/budget: a C-only reference does not certify A/B feasibility, and possession of the hidden clean model alone is insufficient. Reject underdetermined faults; do not increase the fault class after observing model scores.

### W1 — Synthesize composed sequence programs (A/B/C)

- **Target:** a fixed, zero-initialized attention architecture in a declared capacity tier. The specification composes routing, counting, comparison and conditional selection. A concrete example is reversing only tokens selected by a query-dependent predicate, then emitting a count-dependent suffix.
- **Given:** the complete desired specification, token layout, architecture and capacity limit. Hold out new compositions, lengths within the declared domain, and encoding variations. Routine textbook cases stay in the sanity pool.
- **Submit:** tensor values/patches only; architecture and runtime code stay frozen.
- **Score:** exhaustive small-domain correctness, boundary/adversarial cases, and fidelity at predeclared capacity tiers. Count every persistent scalar/byte, including buffers if future formats introduce them. This score measures specification-to-weights synthesis, not interpretation of an existing learned network.
- **Strong baselines:** independently implemented public-spec constructors; circuit-library retrieval/composition; bounded lookup tables fitting the same capacity; small coding/reasoning models under the same condition.
- **Reference distinction:** a Tracr/compiler solution given the generating program proves representability. If the solver receives only a natural-language specification, a compiler given a hidden gold program is an oracle reference, not an information-matched competing baseline.

### P2 — Repair loop attractors (C-only extension)

- **Target:** small recurrent/attention models that perform finite sequence algorithms but sometimes enter repetition or halting failures. Use several mechanisms, and include clean/nonlooping targets and valid long or repetitive computations.
- **Given:** the weights, task goal and metered generation/teacher-forced/activation tools. No clean checkpoint or hidden final cases.
- **Submit:** a numeric weight patch. Keep diagnostic data and final evaluation separate.
- **Score:** correct completion, accuracy on originally completed cases, preservation, truncation, repetition and performance on legitimate long computations. Forcing early termination or merely shortening outputs must not earn recovery credit.
- **Strong baselines:** a fixed contrastive-activation mean/PCA direction pipeline with writer localization and rank-one suppression; bounded layer/strength selection; head ablations; norm-matched random directions; behavior-only candidate selection. Report decoding-only remedies as a separate practical comparator, not as eligible weight edits.
- **Interpretation:** success supports instrumented repair. It does not independently establish new reasoning ability or native weights-only interpretation.

## 3. Strong controls and fair access

### Baseline roster

1. **Metadata/no informative weights:** balanced constant/majority answers and specification-only patches.
2. **Weight statistics:** moments, signs, quantiles, norms, singular spectra and structural sparsity feeding logistic regression/random forests or nearest-neighbor methods. Fit on a separate population covering every label/mechanism with at least 32 independently generated development examples per label where labels exist. Disclose this task-specific training and its cost; these are conventional comparators, not off-the-shelf LLM rows.
3. **Architecture-aware algebra:** signed path products, QK/OV analysis, feature matching, linearized approximations, circuit templates and symbolic decompilers.
4. **Generic repair:** no-op, norm/bias/gain correction, head replacement, auxiliary-preserving splice, null-space edits and permitted analytic low-rank methods.
5. **Instrumented interpretation:** contrastive activation means/PCA, intervention-based component localization and a bounded layer/strength schedule. ACDC-style methods are relevant C comparators, not B methods because they consume executions.
6. **Synthesis competitors:** independent public-spec constructors, circuit libraries and capacity-matched lookup methods. Separate oracle compiler references from information-matched competitors.
7. **Agent controls:** a smaller reasoning/coding model and a flagship using the same scaffold, plus a C agent with behavioral observations but no internal activations.

Code-based algorithms evaluated on A's questions are informative difficulty controls, but cannot appear as eligible no-tool A model rows. Every comparator has an access/tuning card stating its data, computation, observations, prior training, capacity and whether it used any privileged reference information.

### Weight dependence without a copying advantage

All solvers submit numeric patches applied by a trusted materializer to the same true starting checkpoint. Untouched values are retained automatically, including unrelated circuitry. This applies to A, B, C and no-weight controls.

On development challenge cases compare full relevant weights, boundary-only weights, masked relevant weights and matched-but-wrong relevant weights. Masked or wrong views receive neutral typed redactions/matched same-shaped tensors, without changing the checkpoint to which patches are applied. All conditions preserve untouched values identically. Keep the same public constraints and observation budget. This prevents a full-weight advantage caused solely by copying an unknown auxiliary function.

Weight access is expected to help task-relevant prediction/repair after comparison with the strongest structural baseline. A drop under mismatched weights alone is insufficient: also test content invariance under verified function-preserving permutations/rescalings. Such transformations do not defeat invariant path-product baselines and are not a substitute for the baseline roster.

### Computation and training policy

A has no external tool calls. B/C share an audited tensor-algebra tool surface: inspect blocks, reduce statistics, multiply matrices, decompose matrices and construct numeric patches. B exposes no target-execution/activation endpoint. Explicitly simulating the target on real or synthetic inputs is prohibited and audited; absence of an execution endpoint does not prove that arbitrary calculation cannot emulate a forward pass. Prefer a bounded operation API over unrestricted Python for the matched core. C adds controlled target execution and activation access.

Before any solver runs, freeze disjoint practice and hidden final input banks for every C family. A/B/C headline scores and paired differences use that same hidden final bank. Exhaustive-domain checks remain admission checks or explicitly secondary results; never credit C for replaying queried answers in the headline score. The final bank is not defined retrospectively as whichever cases an agent happened not to query. R2 also requires the intervention-level separation above.

Allow estimation of diagnostic statistics/directions from metered observations. Prohibit gradient/loss fitting of target parameters, adding learned predictors to the inference runtime, and solver-specific fine-tuning in the off-the-shelf agent track. Do not describe activation means or PCA as violating a blanket prohibition on every form of estimation.

Allow the same bounded candidate-intervention selection to scripts and LLMs. For example, a fixed layer/strength sweep is charged identically whether scripted or performed across model turns. Generic parameter optimization can be reported separately with its own declared eligibility; do not create an LLM advantage by prohibiting only a baseline's control flow.

## 4. Budgets, scoring and admission

### Shared limits

Retain the existing maximum 32,000 output tokens, 160,000 cumulative input tokens, 120 CPU seconds, 2 GB memory and fifteen-minute active-time limit for the matched core. A returns one response; B/C use at most sixteen model turns and twelve analysis calls. Report the interaction difference: B minus A measures the declared tool-access package, not a pure internal reasoning mechanism. Record all token/cache categories and estimated cost.

C gets at most 128 logical example-evaluations across all original, edited and teacher-forced requests, no more than eight distinct candidate weight patches, and no more than 256,000 returned activation scalars. Original and edited evaluations count separately; repeats consume allowance again. Log sequence positions, generated tokens and activation bytes as well. Save submissions before feedback and at 32 and 128 evaluations without returning hidden scores. If a request would cross a snapshot boundary, split it. Loop rollouts count both episodes and actual steps/tokens, with a maximum of sixteen episodes and the task's declared horizon. Cross-family feedback units are reported separately, not presented as equal compute.

If a solver stops early, carry its final committed submission forward at later budget points, while recording the actual observations/cost consumed. Do not fabricate unused observations or further model turns. R2 prefeedback and transfer-forecast scores remain separately labeled.

### Outcomes

- Prediction: functional agreement, affected-subpopulation precision/recall, effect-prediction error, and a proper confidence score where elicited. Grade behavior and causal consequences, not explanatory prose.
- Repair: target recovery/fidelity, preservation, false-positive changes on clean controls, edit fraction/norm, and observation/cost curves. Calibrate feasible task-specific acceptance tolerances before model evaluation.
- Write: functional correctness at each declared capacity tier and specification complexity, with size/cost separate.
- Report A, B and C by task family/skill. No global native-understanding score or blended three-condition ranking. Compute paired B−A and C−B only on matched core targets/questions and common grading; C-only loop results never enter those differences.
- Cluster uncertainty by held-out program/mechanism, with checkpoints, seeds, attempts and symmetry variants nested. Repeated parameterizations are not new independent tasks. Disclose small sample sizes and degenerate bootstrap intervals rather than claiming certainty from all-pass development data.

### Admission gates

Before a family enters the challenge pool:

1. Verify unambiguous behavioral/causal ground truth, serialization stability and feasible references under the actual permitted information.
2. Counterbalance labels, architectures, metadata and sampling schedules; hold out program structures/implementation patterns as well as checkpoints.
3. Run the full relevant baseline roster, including the observed path-product and splice exploits. Saturated analytical cases are retained as sanity checks. Analytical solutions themselves are legitimate, and baselines are never banned after succeeding.
4. Run weight-access controls without the copying confound, plus verified reparameterizations and clean/nonlooping controls. Diagnose which information explains success.
5. Use development-only small-model/flagship calibration to detect ceiling/floor effects and forecast cost. Do not select tasks for a desired vendor ordering. Freeze tasks, tolerances, baselines, conditions, budgets and manifests before headline model feedback.

### Staged pilot

- Keep twelve development parents: two per proposed family. Start admission calibration with one representative per family: five matched parents × three conditions × two model tiers × two attempts = 60 sessions, plus one C-only loop parent × two model tiers × two attempts = four sessions. Total: 64 development sessions, after CPU-only validation.
- If admitted, a small v0.2 pilot can use twenty held-out matched parents (four different mechanisms/programs per core family) × three conditions × four vendors × three attempts = 720 sessions. Four additional C-only loop parents × four vendors × three attempts = 48 sessions. Total: 768, reported as a pilot with limited per-family precision.
- These are design counts, not authorized launches or a cost guarantee. Forecast the complete matrix from observed calibration charges with headroom, enforce the existing project-wide $3,000 ceiling and phase allocations, and stop before headline runs if it does not fit. Approximately $2,999.22 remains from the original envelope. No paid calibration is initiated by this proposal.

## 5. Implementation contracts and migration

Add independent fields to versioned task/run manifests: `suite_tier` (`sanity`/`challenge`), `condition` (`A`/`B`/`C`), `task_family`, `skill`, `parent_problem_id`, `mechanism_family_id`, `target_stratum` (`constructed`/`trained`), `capacity_tier`, `weight_view`, `observation_policy`, and a validated public serialization digest. `condition` and `weight_view` belong to the trial; task identity remains shared across matched trials.

Trial and budget-reservation identities must include protocol digest, parent, model configuration, condition, weight view and attempt. Exact retries reuse an existing reservation/result; a genuinely new declared protocol receives a distinct identity while sharing the same project-wide spending ledger. This prevents A/B/C collisions and prevents output-directory changes from resetting the budget.

Add declarative behavioral/causal answer schemas, numeric-patch materialization and diagnostic-evaluation counters. Frozen matrices must enumerate conditions, views and required attempts; reporting verifies those dimensions and keeps sanity results separate. New generators and scorers use new family IDs, rather than silently changing the semantics of v0.1 names.

Retain v0.1 targets, seeds, hashes, developer exports, reference checks, baseline results and the six audited Astra sessions. Existing `official` paths are historical split names, not current headline eligibility. The v0.1 generators are sanity-only and cannot be frozen as a new headline pool. Do not edit historical artifacts to relabel their execution conditions retrospectively.

Build v0.2 in this order: access/materialization contracts and fixtures; exact constructed tasks/scorers; trained counterparts; conventional baselines/red-team tests; 64-session admission calibration; frozen four-provider pilot. Website work remains deferred.

## Methodological references

- [Original WSP experiment and Qwen instrumentation](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/README.md).
- [Tracr: exact programs compiled into transformer weights](https://github.com/google-deepmind/tracr). Its named/disjoint subspaces are useful ground truth but can also provide decompilation shortcuts.
- [Automatic Circuit Discovery](https://github.com/ArthurConmy/Automatic-Circuit-Discovery). Use the method as a baseline reference; its repository itself cautions about implementation/dependency sharp edges.
- [Autoresearch report and evaluation caveats](https://csinva.io/weight-space-programming-autoresearch/analyze/toy_examples/report.html).
- The retained v0.1 pool is a sanity check. Current conventional controls are implemented in the [baseline module](../src/wspbench/v2/baselines.py); the public pilot bundles matching CPU evidence separately from model results.
