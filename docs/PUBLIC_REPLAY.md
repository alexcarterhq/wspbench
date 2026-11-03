**Regrade the released pilot on CPU without model inference.** From the repository root:

```sh
uv sync --frozen
uv run --frozen wspbench-replay
```

The same command is available as `uv run --frozen wspbench replay`. The standalone entry point is also equivalent to `python -m wspbench.public.replay` in the installed environment. No API keys, private run directories, Docker daemon or GPU are needed. Installing dependencies may use the network; the replay itself uses local public files.

Default input paths are `data/development-v0.2.1` and `results/pilot-v0.2.1`. To select paths explicitly:

```sh
uv run --frozen wspbench-replay \
  --data data/development-v0.2.1 \
  --results results/pilot-v0.2.1 \
  --output replay-output
```

The replay verifies the public manifest and artifact hashes, validates the 28 public development reference submissions, regrades all eighteen saved model submissions, checks the prescribed task/model matrix and recorded access/configuration/usage consistency, and rebuilds the aggregate report. Output goes under `replay-output`, including `replay-summary.json`, `reference-results.json` and `gpt-comparison-v0.2.1.{json,md,csv}`. A missing, corrupt or inconsistent fixture or run should cause a nonzero exit rather than a partially reconstructed success report.

To check only the 28 public references, without reading the saved model result bundle:

```sh
uv run --frozen wspbench-replay --references-only --output replay-output/references
```

The expected pilot contains six problems per model: two rule-recovery, two edit-effects and two construction tasks. Conditions are A/full with one attempt per task/model. All three models score 0/4 exact on Read. Write exact counts are Luna 0/2, Terra 1/2 and Sol 1/2. The two successful constructions match every entry of the complete 1,024-input domain. The published cohort estimate is $2.2581618; offline replay creates no new model charges.

Public development fixtures include the grading inputs and masks needed to verify those examples. They are released for development and inspection, and are retired as hidden assessment material. Original seed-bearing manifests, private feasibility/training metadata, active held-out fixtures and baseline-training checkpoints are excluded. Retiring a development fixture does not retire its shared generation seed. Public artifact manifests preserve original and export hashes without publishing those seeds.

Offline regrading verifies saved outputs and numerical scoring. It does not regenerate proprietary model inference, reproduce the original latency, independently observe historical API traffic, or establish what computations occurred inside a model. Original audit decisions are recorded provenance; newly replaying an answer is not a new model attempt or an independent task. See the [limitations](LIMITATIONS.md).

To run the full CPU and mock-provider test suite:

```sh
uv sync --frozen --extra dev --extra generation --extra eval
uv run --no-sync python -m pytest -q
```

For a reproduction report, include the source commit, harness and dataset versions, bundle digest, platform, Python version, exact commands and sanitized replay summary. Keep offline regrading separate from any fresh API experiment. The [contributor guide](../CONTRIBUTING.md) describes community result review.
