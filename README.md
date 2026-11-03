# WSPBench

WSPBench tests whether access to neural network weights helps an agent diagnose and repair a model. The agent forecasts the effects of candidate edits before it tests a patch. The scorer checks whether the patch fixes the target failure and preserves previously correct behavior on hidden examples.

[Run the benchmark](experiments/README.md#diagnosis-and-selective-repair) · [Full Sol results](reports/selective-repair-sol-full/README.md) · [Calibration results](reports/selective-repair-hard/README.md) · [Website](https://wspbench.com)

## Changelog

### Current benchmark: diagnosis and selective repair

The main comparison gives one arm behavior probes and another arm the same probes plus weights. A scripted solver uses the same probe budget. In the 80-session experiment, GPT-6.1 Sol repaired 14 of 32 faults with behavior probes and 30 of 32 with the same probes plus weights. The scripted solver repaired 5 of 32. All three preserved all eight clean controls.

The earlier pilot used one case per fault mechanism and one clean control. In a follow-up with equal 12,000-token limits, Sol repaired one of four faults with behavior alone and two of four with weights and behavior. Two weight-assisted sessions exhausted the token allowance. [Follow-up results](reports/selective-repair-hard/budget-check.md)

### Previous benchmark: Read and Write

The v0.2.1 preview tested whether models could infer rules from small circuits or construct weights for a supplied task. Later paired Read pilots swapped checkpoints to test whether answers followed the weights. All three GPT models reached 100% with the true checkpoint in the nonlinear pilot, so we kept those tasks as sanity checks. [Paired Read report](reports/paired-read-v2/README.md)

Matthew Honnibal's critique distinguished reading weights directly from repairing a model with code and practice feedback. That distinction helped shape the new benchmark. [Read Honnibal's post](https://honnibal.dev/blog/astra-read-write-misleading)

Download [v0.2.2](https://github.com/alexcarterhq/wspbench/releases/tag/v0.2.2) or read [the release notes](CHANGELOG.md).

## Contribute and reproduce

Start with [CONTRIBUTING.md](CONTRIBUTING.md). Useful contributions include an independent CPU regrading, a stronger construction baseline, a reviewed task proposal, or a bounded additional provider adapter and fresh model replication. Maintainer-reviewed and community results must be labeled separately; new runs need their own versioned configurations, budgets and traces.

```sh
uv sync --frozen --all-extras
uv run --frozen --all-extras python -m pytest -q
```

CI validates the selective repair cases and runs the CPU and mock-provider tests without model credentials. See the [changelog](CHANGELOG.md) and [errata](ERRATA.md) for revisions and historical result exclusions.

## License

Original code is [Apache-2.0](LICENSE); original benchmark data and development examples are [CC BY 4.0](LICENSE-DATA). Retain attribution and third-party notices in [NOTICE](NOTICE). Citation metadata is in [CITATION.cff](CITATION.cff).
