**Contribute a reproducible check, task, baseline, adapter or review.** WSPBench is a research preview. The protocol implements A: no external tools, B: bounded weight computation, and C: instrumented feedback. Published model results currently cover A only. Read the [limitations](docs/LIMITATIONS.md) before interpreting those results.

From the repository root, install the locked environment and regrade the released pilot:

```sh
uv sync --frozen
uv run --frozen wspbench-replay
```

This is CPU grading of saved answers. It creates no model requests and needs no API keys. For the full test suite, include the generation and provider test dependencies:

```sh
uv sync --frozen --extra dev --extra generation --extra eval
uv run --no-sync python -m pytest -q
uv run --no-sync wspbench-replay
```

The generation tests train small local targets on CPU. Provider tests use mocks or fake transports. CI never runs paid inference. Setup may download dependencies; replay after installation reads local public artifacts. See the [replay guide](docs/PUBLIC_REPLAY.md) for inputs, outputs and verification scope.

Small corrections can arrive as pull requests. For a new task family or substantial baseline, open a proposal first so the measurement and acceptance checks can be reviewed before implementation. Useful starting contributions include:

- Regrade the public pilot on another platform and report the source/data versions, commands and sanitized summary.
- Review a task's information sufficiency, scoring, no-op control, preservation checks or split independence.
- Add an independent construction baseline that obeys the declared architecture and capacity, with transparent computation and failure reporting.
- Validate another provider adapter's request bounds, retry behavior, token categories and tool semantics using fake transports before any paid calibration.
- Improve documentation and the public development workflow.

Task proposals should state the research question, solver information, access condition, resource budget and independent task unit. Include a feasible reference and strong baseline coverage, not just a working generator. Describe boundary cases, malformed submissions, clean/no-op controls and behavior that a repair must preserve. Label constructed circuits, partially fitted models and end-to-end trained networks accurately. Keep related corruptions and equivalent parameterizations in one split and cluster.

Baseline proposals must disclose their access and computational surface. A conventional CPU algorithm is a difficulty control; its result does not become an eligible no-tool model score. Record capacity refusals and failed attempts rather than omitting them. Do not use private held-out answers or training checkpoints to select a baseline.

Scoring or runtime changes need behavior-focused tests and a new evaluation version where they change outcomes. Preserve the evaluated source snapshot and artifact hashes for existing results. Do not rewrite a published run to make it match a new harness. Harness `0.2.0` and dataset revision `0.2.1` are separate identifiers in this preview.

Public development fixtures are released for inspection and regrading. Active held-out data, original private manifests, generation seeds, training metadata and private feasibility artifacts stay outside pull requests. Seeds can share a base across task splits, so retiring a development fixture does not authorize releasing its original generation metadata. Use newly generated data for new tests.

New model inference is separate from offline regrading. Agree on the public task scope, provider configuration, attempts and cost cap before a coordinated run. Validate adapter boundedness and usage accounting first. Never share API keys. Submit all attempts with sanitized prompts, final answers or numeric patches, usage, timing, failures and access-event summaries. Community submissions remain labeled as community results until provenance, matrix completeness, scoring and traces pass independent review. Passing a trace review attests protocol compliance, not answer correctness or a model's internal cognition.

Contributors are acknowledged for task authorship, reviews, replications and maintenance. Funding and in-kind support are acknowledged separately from results; sponsors do not control scoring or gain advance access to active hidden tasks. Research authorship is based on substantive contribution and agreed with contributors, rather than promised for stars, sponsorship or a single issue.

Original source contributions use [Apache-2.0](LICENSE). Original benchmark data and annotations use [CC BY 4.0](LICENSE-DATA). Identify third-party content and retain its notices. Cite WSPBench using [CITATION.cff](CITATION.cff), including the actual source commit and dataset revision in reports. Website work remains deferred.
