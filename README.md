# WSPBench — research preview

WSPBench evaluates LLMs reading, editing and constructing neural-network weights under explicit tool budgets. It separates three access conditions and grades numeric artifacts inside fixed, benchmark-owned architectures.

| Condition | Solver access |
|---|---|
| **A — unaided** | Weights and task specification in the prompt; no external tools or target execution. |
| **B — weight analysis** | The same information plus bounded tensor algebra; no target execution. |
| **C — instrumented** | Weight analysis plus metered target behavior, activations and practice feedback. |

All three conditions are implemented. **The measured model pilot covers A only:** eighteen development sessions, six identical problems per model, one attempt, high reasoning effort. These are constructed small ReLU circuits, with retained training/repair families as diagnostics. They are not language-model capability-improvement experiments.

## Regrade the public pilot on CPU

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run these commands from the repository root with Python 3.13:

```sh
uv sync --frozen
uv run --frozen wspbench-replay
```

The replay checks hashes and provenance, validates **28 public development reference solutions**, regrades **18 saved model submissions** with the original grader, recomputes usage-based cost estimates, and rebuilds Markdown, JSON and CSV under `replay-output/`. It uses no API keys, network inference, Docker, GPU or private workspace files. Regrading saved answers does not rerun proprietary model inference.

The equivalent command is `uv run --frozen wspbench replay`. See [replay and artifact documentation](docs/PUBLIC_REPLAY.md) for paths, verification and reference-only validation. Public development scoring fixtures and answers are intentionally exposed; active held-out instances and their seeds are not included.

## Pilot observations

| Model | Read agreement | Read exact | Write accuracy | Write exact | Median session | API estimate |
|---|---:|---:|---:|---:|---:|---:|
| GPT-6 Luna | 44.1% | 0/4 | 71.9% | 0/2 | 117s | $0.083 |
| GPT-6.1 Terra | 66.0% | 0/4 | 97.3% | 1/2 | 148s | $0.746 |
| GPT-6.1 Sol | 87.7% | 0/4 | 95.8% | 1/2 | 123s | $1.429 |

Total token-price estimate: **$2.2581618**. All eighteen sessions returned valid submissions and known usage; none was excluded for infrastructure failure. Every session received a protocol trace review. Costs are estimates from reported token categories and frozen prices, not invoice verification. Model IDs are provider aliases, verified November 3, 2026, rather than immutable snapshots.

The no-edit Write control already scores **70.1%** accuracy and corrects none of the initially wrong examples. Luna's 71.9% therefore differs substantially from a successful construction. Terra and Sol each submitted one network that matched the complete **1,024-entry truth table**. Continuous accuracy and exact success answer different questions; read them alongside affected-case accuracy and retention.

Read agreement averages rule recovery and before/after function agreement. Exact Read success requires correct final-bank behavior, including both functions for edit tasks. This six-problem development study cannot establish a headline ranking. Original prompts, numeric submissions, scoring fixtures, usage, access events and provenance are in the [public pilot bundle](results/pilot-v0.2.1/manifest.json). View the generated [report](reports/pilot-v0.2.1/gpt-comparison-v0.2.1.md), [CSV](reports/pilot-v0.2.1/gpt-comparison-v0.2.1.csv) and [JSON](reports/pilot-v0.2.1/gpt-comparison-v0.2.1.json).

## Design and scope

The three challenge families are open Boolean-rule recovery, before/after edit prediction, and compression of a public truth table into a zero-start network. The microcircuit architecture has ten Boolean inputs, four ReLU stages and **4,049 persistent float32 scalars**. References are checked over the complete domain; published scores preserve the original fixed final-bank criteria.

Conventional decoders, statistical readers, no-edit controls and capacity-limited constructors provide difficulty controls. Code-based B/C comparators do not become eligible no-tool A model scores. Reference witnesses establish representability; they do not establish generic search success. Read, Repair and Write remain separate rather than being combined into one capability score.

The [current task revision](docs/REDESIGN_V0_2_1.md), [access design](docs/DESIGN_V0_2.md), [prompt-information policy](docs/PROMPT_INFORMATION.md) and [limitations](docs/LIMITATIONS.md) document these distinctions. v0.1 remains available as sanity-check code. Primary held-out scoring requires separate scientific admission, budget checks and a frozen protocol. A/B/C implementation and CPU checks alone do not establish headline admission.

Harness/package version **0.2.0** and dataset revision **v0.2.1** identify different artifacts. The public release preserves the evaluated harness digest; new export/replay utilities sit outside that evaluated module set. The dependency lock uses CPU-only PyTorch wheels on Linux/Windows for optional generation and tests. PyTorch is not needed for saved-answer replay.

## Contribute and reproduce

Start with [CONTRIBUTING.md](CONTRIBUTING.md). Useful contributions include an independent CPU regrading, a stronger construction baseline, a reviewed task proposal, or a bounded additional provider adapter and fresh model replication. Maintainer-reviewed and community results must be labeled separately; new runs need their own versioned configurations, budgets and traces.

```sh
uv sync --frozen --all-extras
uv run --frozen --all-extras python -m pytest -q
```

CI runs the public replay and tests on CPU without model credentials. Tests use generated fixtures and mock model responses. Real generation is separate from replay and is never part of CI. See the [changelog](CHANGELOG.md) and [errata](ERRATA.md) for revisions and historical result exclusions. Website publication remains deferred.

## License

Original code is [Apache-2.0](LICENSE); original benchmark data and development examples are [CC BY 4.0](LICENSE-DATA). Retain attribution and third-party notices in [NOTICE](NOTICE). Citation metadata is in [CITATION.cff](CITATION.cff).
