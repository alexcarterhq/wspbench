from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DATA = Path(".wspbench-data/pilot-v0.1-preservation")


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, allow_nan=False, default=str))


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def main(argv: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "replay":
        from wspbench.public.replay import main as replay_main
        replay_main(arguments[1:])
        return
    if arguments and arguments[0] == "v2":
        from wspbench.v2.cli import main as v2_main
        v2_main(arguments[1:])
        return
    from dotenv import load_dotenv
    load_dotenv(Path.cwd() / ".env", override=False)
    parser = argparse.ArgumentParser(prog="wspbench", description="Independent weight-space programming pilot")
    parser.add_argument("--version", action="version", version="WSPBench 0.2.0 (v0.1 sanity suite retained)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("replay", help="Regrade the released development submissions on CPU without API calls")
    sub.add_parser("v2", help="Challenge candidates: access conditions, generators and CPU validation")
    generate = sub.add_parser("generate", help="Train fresh targets and create local private evaluation data")
    generate.add_argument("--data", type=Path, default=DEFAULT_DATA)
    generate.add_argument("--seed", type=int, help="Explicit reproducibility seed; keep official seeds private")
    validate = sub.add_parser("validate", help="Validate all references and score no-op/random controls")
    validate.add_argument("--data", type=Path, default=DEFAULT_DATA)
    validate.add_argument("--output", type=Path, default=Path(".wspbench-state/validation.json"))
    export = sub.add_parser("export-dev", help="Export only the public development set and solutions")
    export.add_argument("--data", type=Path, default=DEFAULT_DATA)
    export.add_argument("--output", type=Path, default=Path("artifacts/development"))
    baselines = sub.add_parser("baselines", help="Evaluate metadata, statistical and no-weight controls")
    baselines.add_argument("--data", type=Path, default=DEFAULT_DATA)
    baselines.add_argument("--output", type=Path, default=Path("artifacts/baselines.json"))
    preflight = sub.add_parser("preflight", help="Check provider configuration, keys, dependencies, and Docker")
    preflight.add_argument("--config", type=Path, default=Path("configs/models.json"))
    preflight.add_argument("--model", action="append", help="Check only this configured model ID")
    run = sub.add_parser("run", help="Run calibration or a frozen, budget-approved evaluation matrix")
    run.add_argument("--data", type=Path, default=DEFAULT_DATA)
    run.add_argument("--output", type=Path, default=Path(".wspbench-runs"))
    run.add_argument("--config", type=Path, default=Path("configs/models.json"))
    run.add_argument("--phase", choices=("calibration", "primary", "diagnostics"), default="calibration")
    run.add_argument("--attempts", type=int, default=3)
    run.add_argument("--model", action="append", help="Calibration only: select a configured model ID")
    freeze = sub.add_parser("freeze", help="Freeze validated suite, model configuration and calibration forecast")
    freeze.add_argument("--data", type=Path, default=DEFAULT_DATA)
    freeze.add_argument("--output", type=Path, default=Path(".wspbench-runs"))
    freeze.add_argument("--config", type=Path, default=Path("configs/models.json"))
    freeze.add_argument("--calibration", type=Path, default=Path(".wspbench-runs"))
    review = sub.add_parser("review", help="Record an independent protocol review for one run")
    review.add_argument("run_dir", type=Path)
    review.add_argument("--reviewer", required=True)
    review.add_argument("--decision", required=True, choices=("approved", "rejected"))
    review.add_argument("--note", required=True)
    report = sub.add_parser("report", help="Export only complete, audited official model results")
    report.add_argument("--runs", type=Path, default=Path(".wspbench-runs"))
    report.add_argument("--output", type=Path, default=Path("artifacts/results.json"))
    report.add_argument("--validation", type=Path, default=Path(".wspbench-state/validation.json"))
    calibration = sub.add_parser("calibration-report", help="Summarize reviewed development runs without creating official rankings")
    calibration.add_argument("--runs", type=Path, default=Path(".wspbench-runs"))
    calibration.add_argument("--output", type=Path, default=Path("artifacts/calibration.json"))
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            from wspbench.tasks import generate_suite
            _print(generate_suite(args.data, seed=args.seed))
        elif args.command == "validate":
            from wspbench.tasks import validate_suite, load_tasks
            from wspbench.scoring import baseline_submission, score
            result = validate_suite(args.data)
            tasks = load_tasks(args.data, "all")
            controls = {}
            for name in ("noop", "random"):
                controls[name] = {
                    skill: sum(score(t, baseline_submission(t, name)).success for t in tasks if t.skill == skill)
                    for skill in ("read", "repair", "write")
                }
            result["controls"] = controls
            result["n_tasks"] = result["task_count"]
            result["n_passed"] = result["reference_successes"]
            result["generated_at"] = datetime.now(timezone.utc).isoformat()
            _write(args.output, result)
            _print(result)
            if result.get("status") not in {"passed", "valid", "ok"}:
                raise SystemExit(1)
        elif args.command == "export-dev":
            from wspbench.export import export_development
            _print(export_development(args.data, args.output))
        elif args.command == "baselines":
            from wspbench.baselines import evaluate_baselines
            result = evaluate_baselines(args.data)
            _write(args.output, result)
            _print({"output": str(args.output), "notice": "Aggregate controls; no provider benchmark results"})
        elif args.command == "preflight":
            from wspbench.runner import preflight as check
            result = check(args.config, model_ids=args.model)
            _print(result)
            if not result.get("ready", result.get("ok", False)):
                raise SystemExit(1)
        elif args.command == "run":
            from wspbench.runner import run_evaluation
            result = run_evaluation(args.data, args.output, args.config, phase=args.phase, attempts=args.attempts, model_ids=args.model)
            _print({"records": len(result), "output": str(args.output)})
        elif args.command == "freeze":
            from wspbench.runner import freeze_protocol
            _print(freeze_protocol(args.data, args.output, args.config, args.calibration))
        elif args.command == "review":
            from wspbench.runner import review_run
            _print(review_run(args.run_dir, args.reviewer, args.decision, args.note))
        elif args.command == "report":
            from wspbench.report import write_report
            result = write_report(args.runs, args.output, validation=args.validation)
            _print({"status": result["status"], "official_models": len(result["models"]), "output": str(args.output)})
        elif args.command == "calibration-report":
            from wspbench.report import calibration_report
            result = calibration_report(args.runs, args.output)
            _print(result)
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as error:
        print(f"wspbench: {error}", file=sys.stderr)
        raise SystemExit(2) from None
