# Operator experiments

`report_gpt_comparison.py` is retained as a historical aggregation implementation and test fixture. Its operator entry point expects a private evaluation workspace. The public research preview uses `wspbench-replay` and the released `data/` and `results/` bundles instead; see the repository README.

## Diagnosis and selective repair

Install the API dependencies with `uv sync --frozen --extra eval`. Run the commands below in that environment.


```sh
python experiments/run_selective_repair.py --seed 61073 --output .wspbench-runs/selective-repair-03
python experiments/run_selective_repair.py --seed 61073 --output .wspbench-runs/selective-repair-03 --execute
python experiments/report_selective_repair.py .wspbench-runs/selective-repair-03
```

The first command freezes synthetic development cases and checks reference
repairs before any paid requests. It also runs the scripted behavior baseline.
The second command uses the local OpenAI API credential and reserves costs in
the shared diagnostics ledger. The third reads saved records and reports matched
differences, invalid submissions, and estimated API cost.

Each solver commits predictions for a shared menu of interventions before it
can test any edit. The API runner exposes one structured function call per turn
and validates every argument locally. Behavior arms can use up to two unchanged-controller
practice examples before committing and up to four practice examples after
committing. Each case uses the same practice bank, edit menu, forecast schema,
and turn and token limits across arms. Activation access adds internal readings
to the metered behavior observations. The scripted solver receives the public
task and the same metered behavior interface.

The primary comparison is weights plus behavior versus behavior alone.
The hidden grader scores the chosen repair against the target and preservation
criteria and checks forecast accuracy against held-out intervention outcomes.
The small development run calibrates the benchmark and reporting protocol.

The calibration includes one hidden-bias fault, one readout-gain fault, and
one clean controller. They use the same architecture and a fitted readout.
A shared scenario makes either fault produce the reported unsafe action.
Candidate interventions include localized corrections, partial edits, a
global decision-threshold shift, and no edit. The submitted repair selects
one candidate.

A repair passes at 95% target accuracy and at most two percentage points
of retention loss. Clean controllers require no edit. The grader computes
forecast mean absolute error across the target and preservation predictions
for every candidate. Scoring uses separate hidden input banks.

### Harder selective repair calibration

```sh
python experiments/run_selective_repair.py --suite hard --seed 82417 --output .wspbench-runs/selective-repair-hard-01
python experiments/run_selective_repair.py --suite hard --seed 82417 --output .wspbench-runs/selective-repair-hard-01 --execute
python experiments/report_selective_repair.py .wspbench-runs/selective-repair-hard-01
```

The harder suite uses four fault mechanisms and a clean control. It retains
the same matched access arms, probe and token limits, and preservation
criteria. Intervention menus include compound edits. Each fault requires a
passing reference patch and a target-fixing decoy that breaks retention.
The private manifest binds the public task, weights, and hidden scoring
inputs. The report gives fault-only results alongside the clean control.

A separate Sol check runs the two primary access arms with a matched
12,000-token output allowance. It replaces the global decoy with a fixed
shift independent of the hidden target logits. This check changes both the
budget and decoy construction.

```sh
python experiments/run_selective_repair.py --suite hard-budget-check --seed 82417 --output .wspbench-runs/selective-repair-hard-check-01
python experiments/run_selective_repair.py --suite hard-budget-check --seed 82417 --output .wspbench-runs/selective-repair-hard-check-01 --execute
python experiments/report_selective_repair.py .wspbench-runs/selective-repair-hard-check-01
```

### Full Sol experiment

The full suite runs the main matched comparison on eight new seed blocks.
Each block contains four faults and one clean control. GPT-6.1 Sol runs
each case twice, with the same probe budget in both sessions. One session
also receives the weights. The scripted baseline runs once per case.
The suite excludes the calibration seed and uses a fixed decoy. The
commands below generate fresh cases from an example seed. The published
report shares aggregate results; the original cases and traces are private.

```sh
uv run python experiments/run_selective_repair.py --suite sol-full --seed 123456 --output .wspbench-runs/selective-repair-sol-full-01
uv run python experiments/run_selective_repair.py --suite sol-full --seed 123456 --output .wspbench-runs/selective-repair-sol-full-01 --execute
uv run python experiments/report_sol_full.py .wspbench-runs/selective-repair-sol-full-01
```

The first command freezes the cases and runs the scripted baseline. The
second runs the 80 API sessions with a 24,000-token output limit in each
arm. At the recorded rates, the maximum reservation is $28.80. The report
estimates the paired repair difference on fault cases and resamples seed
blocks for the uncertainty interval. It lists clean controls separately.
