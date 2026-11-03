"""Explicit v0.2 commands; CPU validation never loads API credentials."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

DATA = Path(".wspbench-data/challenge-v0.2.1")
RUNS = Path(".wspbench-runs/v0.2.1")


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="wspbench v2", description="Small-network challenge candidates; CPU evidence before paid evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("generate", "Generate private challenge candidates and independent baseline training population"),
        ("validate", "Check mathematical references, serialization and disjoint feedback banks on CPU"),
        ("baselines", "Run conventional, information-matched comparators on CPU"),
        ("smoke", "Exercise access conditions with explicitly privileged reference actors on CPU"),
        ("admission", "Assess CPU evidence; never automatically admit headline tasks"),
        ("report", "Aggregate local records without publishing task answers"),
        ("preflight", "Read readiness without contacting a provider"),
        ("run", "Run gated paid calibration or a frozen scored matrix"),
        ("freeze", "Freeze an admitted suite after model calibration"),
    ):
        command = sub.add_parser(name, help=help_text)
        if name != "report":
            command.add_argument("--data", type=Path, default=DATA)
        if name in {"validate", "smoke", "admission", "preflight", "run", "freeze"}:
            command.add_argument("--output", type=Path, default=RUNS)
        if name in {"preflight", "run", "freeze"}:
            command.add_argument("--config", type=Path, default=Path("configs/models.json"))
        if name in {"preflight", "run"}:
            command.add_argument("--model", action="append")
            command.add_argument("--phase", choices=("calibration", "primary"), default="calibration")
        if name in {"smoke", "run", "freeze"}:
            command.add_argument("--condition", action="append", choices=("A", "B", "C"))
            command.add_argument("--view", action="append", choices=("full", "boundary_only", "masked", "matched_wrong"))
    commands = sub.choices
    commands["generate"].add_argument("--seed", type=int)
    commands["generate"].add_argument("--dev-per-family", type=int, default=4)
    commands["generate"].add_argument("--heldout-per-family", type=int, default=8)
    commands["generate"].add_argument("--baseline-per-label", type=int, default=32)
    commands["generate"].add_argument("--legacy",action="store_true",help="Generate the retained scalar-catalog sanity implementation")
    commands["baselines"].add_argument("--output", type=Path, default=RUNS / "baselines.json")
    commands["smoke"].add_argument("--split", action="append", choices=("dev", "heldout"))
    commands["smoke"].add_argument("--actor", choices=("reference", "baseline"), default="reference")
    commands["smoke"].add_argument("--baseline", action="append")
    commands["smoke"].add_argument("--inspect", action="store_true", help="Exercise actual Inspect tools with a local mock model; synthetic usage, no provider")
    commands["admission"].add_argument("--baselines", type=Path, default=RUNS / "baselines.json")
    commands["report"].add_argument("--records", type=Path, required=True)
    commands["report"].add_argument("--output", type=Path, default=Path("artifacts/v0.2-results.json"))
    commands["run"].add_argument("--attempts", type=int, default=3)
    commands["run"].add_argument("--task",action="append",help="Only these reviewed allowlisted development targets")
    commands["run"].add_argument("--max-sessions",type=int,default=6)
    review = sub.add_parser("review", help="Record an independent audit of one model attempt")
    review.add_argument("run_dir", type=Path)
    review.add_argument("--reviewer", required=True)
    review.add_argument("--decision", choices=("approved", "rejected"), required=True)
    review.add_argument("--note", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            from .tasks import generate_suite,generate_composed_suite
            generator=generate_suite if args.legacy else generate_composed_suite
            result = generator(args.data, seed=args.seed, dev_per_family=args.dev_per_family,
                                    heldout_per_family=args.heldout_per_family, baseline_per_label=args.baseline_per_label)
        elif args.command == "validate":
            from .tasks import validate_suite
            from .runner import harness_digest, suite_digest
            result = validate_suite(args.data)
            result.update(suite_digest=suite_digest(args.data), harness_digest=harness_digest(), no_paid_calls=True)
            _write(args.output / "cpu-validation.json", result)
        elif args.command == "baselines":
            from .baselines import evaluate_baselines
            result = evaluate_baselines(args.data, args.output)
        elif args.command == "smoke":
            if args.inspect:
                from .runner import run_smoke
                result = run_smoke(args.data, args.output / "inspect-smoke")
            else:
                from .runner import run_cpu_matrix
                result = run_cpu_matrix(args.data, args.output / ("cpu-" + args.actor), actor=args.actor,
                    splits=tuple(args.split or ("dev",)), views=tuple(args.view or ("full",)),
                    conditions=tuple(args.condition or ("A", "B", "C")), baseline_names=args.baseline)
        elif args.command == "admission":
            from .admission import assess_admission
            from .runner import harness_digest, suite_digest
            validation = json.loads((args.output / "cpu-validation.json").read_text())
            if validation.get("suite_digest") != suite_digest(args.data) or validation.get("harness_digest") != harness_digest():
                raise ValueError("CPU validation refers to a different suite or harness")
            baseline_evidence = json.loads(args.baselines.read_text())
            if baseline_evidence.get("suite_digest") != suite_digest(args.data) or baseline_evidence.get("harness_digest") != harness_digest():
                raise ValueError("Baseline evidence refers to a different suite or harness; rerun CPU baselines")
            result = assess_admission(args.data, baseline_evidence, validation)
            result.update(suite_digest=suite_digest(args.data), harness_digest=harness_digest(),
                          reviewer=None, evidence=[str(args.baselines), str(args.output / "cpu-validation.json")])
            _write(args.output / "admission.json", result)
        elif args.command == "report":
            from .reporting import build_report
            result = build_report(json.loads(args.records.read_text()))
            _write(args.output, result)
        elif args.command == "preflight":
            from .runner import preflight
            result = preflight(args.data, args.output, args.config, model_ids=args.model, phase=args.phase)
        elif args.command == "review":
            from .audit import review_run
            result = review_run(args.run_dir, args.reviewer, args.decision, args.note)
        elif args.command == "freeze":
            from .runner import freeze_protocol
            result = freeze_protocol(args.data, args.output, args.config,
                conditions=tuple(args.condition or ("A", "B", "C")), views=tuple(args.view or ("full",)))
        else:
            from dotenv import load_dotenv
            from .runner import run_evaluation
            load_dotenv(Path.cwd() / ".env", override=False)
            result = run_evaluation(args.data, args.output, args.config, phase=args.phase,
                model_ids=args.model, attempts=args.attempts, conditions=tuple(args.condition or ("A", "B", "C")),
                views=tuple(args.view or ("full",)),task_ids=args.task,max_sessions=args.max_sessions)
        # All command outputs are local; detailed validation stays under the private run directory.
        printable = {k: v for k, v in result.items() if k not in {"records", "report", "tables", "tasks", "results", "seed", "unattested_reference_conditions", "source_hashes", "baseline_roster","feasibility_records","calibration_allowlist"}} if isinstance(result, dict) else {"records": len(result)}
        if isinstance(result, dict) and isinstance(result.get("tasks"), int):
            printable["task_count"] = result["tasks"]
        if isinstance(result, dict) and "unattested_reference_conditions" in result:
            printable["unattested_reference_count"] = len(result["unattested_reference_conditions"])
        print(json.dumps(printable, indent=2, allow_nan=False, default=str))
        if isinstance(result, dict) and (result.get("status") == "failed" or result.get("ready") is False):
            raise SystemExit(1)
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as error:
        print(f"wspbench v2: {error}", file=sys.stderr)
        raise SystemExit(2) from None
