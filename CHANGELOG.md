# Changelog

## Diagnosis and selective repair

WSPBench now measures whether weight access helps an agent diagnose a failure and choose a patch that preserves correct behavior. Behavior probes, behavior plus weights, and behavior plus weights and activations are separate conditions. The scripted baseline uses the same probe budget. We kept the paired Read pilots as sanity checks after three GPT models reached 100% with true checkpoints.

The harder calibration adds coupled gates, sensor routing, readout redistribution, cross-layer drift, and a clean control. In the matched Sol follow-up, weights helped fix one additional fault. Two other weight-assisted sessions exhausted the output allowance. [Calibration report](reports/selective-repair-hard/README.md) · [Sol follow-up](reports/selective-repair-hard/budget-check.md)

Matthew Honnibal's critique drew a distinction between reading weights and repairing a model with code and feedback. The current benchmark tests that distinction with matched access conditions. [Read Honnibal's post](https://honnibal.dev/blog/astra-read-write-misleading)

## Archived v0.2.1 research preview

**Research preview: `pilot-v0.2.1`, November 3, 2026.** The harness/package version is `0.2.0`; the public development dataset revision is `0.2.1`. These identifiers are intentionally separate. The evaluated runtime snapshot is preserved in the public provenance manifests.

- Added a keyless CPU replay path for all 28 public development references and eighteen saved model submissions.
- Added development-only grading fixtures and a sanitized pilot bundle with prompts, final submissions, numeric patches, configurations, usage, timing, access events and provenance hashes.
- Added report reconstruction from the public bundle. Model inference is not repeated by replay.
- Added matching current CPU evidence summaries and explicit original/export provenance.
- Added contributor guidance, citation metadata, a data-license notice and CPU-only CI.

The pilot has six fixed problems per model and one attempt per problem, condition A/full only. GPT-6 Luna, GPT-6.1 Terra and GPT-6.1 Sol each score 0/4 exact on Read. Write exact success is 0/2, 1/2 and 1/2, respectively. Both successful Write submissions match the full 1,024-entry truth table. All eighteen sessions passed protocol review; no inference was added to prepare this release.

The cohort's token-price estimate is $2.2581618. The transport canary and historical experiments are excluded. The zero-start/no-edit control already scores 70.1% average Write accuracy, so raw accuracy alone is insufficient evidence of construction success. See [limitations](docs/LIMITATIONS.md) and [errata](ERRATA.md).

Earlier development replaced saturated catalog tasks with composed Boolean microcircuits, added structured-reader and construction controls, balanced real/sham edits, checked split equivalence, and strengthened condition-specific feasibility and admission. The v0.1 pool remains available in source as sanity checks. Earlier design drafts, sanity runs and CPU artifacts do not constitute results on this preview's measured harness.

The research preview covered development scoring and replay.
