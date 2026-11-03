# Composed-circuit revision of WSPBench v0.2

This revision replaces the saturated scalar-catalog Read and Write candidates. The original v0.1 suite and earlier v0.2 artifacts remain available as historical sanity checks. Website work remains deferred.

## Tasks and controls

The new architecture is an actual four-stage ReLU network with ten binary inputs, 34 channels per stage and 4,049 persistent float32 scalars. It computes entirely from submitted tensors. Every task has a 1,024-input finite domain; practice and final inputs are fixed and disjoint. Constructed microcircuits are explicitly identified as constructed; this is not an end-to-end trained-transformer evaluation.

- **Read rule recovery:** fresh compositions of Boolean operators, muxes and majority gates replace the twelve-answer catalog. Answers can be expressions or complete output tables. A verified decoder derives the answer solely from the public numerical weights. Its B/C ceiling is retained and reported.
- **Read edit effects:** local internal operator edits preserve input/output boundary tensors. The score checks both functions, consequences, retention of unaffected inputs and false positives. The supervised baseline predicts all 1,024 binary output coordinates, so translated after-programs are representable rather than silently outside its label catalog.
- **Write compression:** the full desired truth table is public; the compact generating expression is private. Solvers must compress the behavior into the fixed network. Independent BDD/Shannon, algebraic-normal-form and prime-cover constructors obey the same architecture and parameter capacity. Capacity refusal is an attempted zero-credit solution, not missing coverage.

The reference circuit for Write proves representability against every public case. It is a privileged construction witness, not a competing public constructor. This is sufficient to define a complete, attainable synthesis problem for a small development calibration; it does not demonstrate that a generic algorithm or an LLM can find the compact solution within budget. Calibration measures that unknown difficulty. Headline admission remains blocked.

Retained scalar causal forecasts now have a static piecewise-affine/product-measure comparator. It consumes public weights and intervention definitions without calling the target runtime or enumerating concrete target inputs. A predeclared `.67` output scaling replaces a floating-point tie-sensitive `.65` control. The scalar tasks remain diagnostic when this method saturates them.

Coupled linear repair has an explicit finite public-inequality certificate from its disclosed weights, fault hypotheses and normative desired answers. It does not use the hidden clean checkpoint to choose a patch. This is an offline information-sufficiency derivation: numerically executing its input-dependent equations during a B trial would be forward emulation. It is not advertised as an eligible B software solver. The retained repair family remains diagnostic.

The old loop model has a complete public copying specification. A new specification-only reconstruction baseline closes the overlooked possibility of simply rebuilding that controller. The old loop family remains diagnostic rather than being promoted on the strength of activation-steering results.

## Baseline population and access

The default suite has 28 development and 56 held-out candidate/control checkpoints, plus 256 independent structured-reader training targets. The latter contain fresh original and edited functions, with 32 compositions per root-operator stratum for each Read family. Training never consumes evaluation-target labels. Parent problems, full behavior equivalence and mechanism clusters are kept distinct.

Behavior-only controls use metered practice feedback, decision trees and degree-one-through-three Boolean polynomial hypotheses, with practice-only selection. Exact gate derivation remains a legitimate code-based ceiling. Conventional code uses direct CPU algorithms rather than the model's twelve-call algebra interface; its computation surface is disclosed. These controls are never presented as eligible A model results.

B/C can analyze already-public truth-table chunks through bounded algebra handles. Those handles disclose no additional information beyond A's prompt and consume no target observations. C activation handles continue to charge the original practice observations and activation scalar quotas.

## Feasibility and admission

CPU checks verify complete-domain references, float32 materialization, response-size bounds, malformed artifacts, numerical intervention alias separation, condition-specific derivations and complete-semantics capacity witnesses. A derivation establishes information sufficiency and bounded answer representation, not that an unaided model will perform the reasoning successfully.

The previous gate incorrectly allowed B/C code ceilings to discard A and could authorize a whole matrix after a single passing cell. Admission now enumerates A/B/C from the manifests. A requires the same public-information baseline coverage but does not inherit a code-based B ceiling. B/C retain their same-condition ceiling criteria.

Each required baseline must cover every exact target and current weight digest in each applicable split, condition and stratum. A private coverage manifest is bound by a public digest. Coverage from another condition, stratum or target cannot substitute. Feasibility records are bound to the precise serialized checkpoint.

Paid development execution requires an independently reviewed allowlist of exact development target/condition/view tuples, current suite/harness/baseline/proof hashes, an explicit small session limit, and the existing project-wide budget ledger. The first GPT pilot may narrow that allowlist further. Hidden evaluation instances and diagnostic controls remain excluded. Primary scoring stays blocked regardless of development scores.
