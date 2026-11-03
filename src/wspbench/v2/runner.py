"""CPU evidence first; paid runs require explicit scientific and budget gates."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile

from ..budget import (BudgetExceeded, BudgetLedger, PRICE_FIELDS, maximum_session_cost,
                      session_cost)
from ..provider_options import provider_client_options, provider_generate_options
from . import VERSION
from .access import AccessSession
from .contracts import Condition, WeightView
from .serialization import canonical_json, load_weights, serialization_digest, weights_digest


def _write(path: Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _inspect_paths(output_root):
    output_root = Path(output_root).resolve()
    for key, suffix in (("INSPECT_TRACE_FILE", "inspect-trace.log"), ("INSPECT_CACHE_DIR", "inspect-cache"),
                        ("XDG_DATA_HOME", "inspect-app-data"), ("XDG_CACHE_HOME", "inspect-app-cache")):
        os.environ.setdefault(key, str(output_root / suffix))


@contextmanager
def _mock_runtime():
    # Inspect discovers dotenv from cwd. A private empty file prevents it from
    # discovering any user credential file during this provider-free exercise.
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="wspbench-mock-", dir="/tmp") as scratch:
        Path(scratch, ".env").write_text("")
        try:
            os.chdir(scratch)
            yield
        finally:
            os.chdir(previous)


def _hash(value):
    return serialization_digest(value)


def _project_ledger() -> Path:
    project = Path(__file__).resolve().parents[3]
    if not (project / "pyproject.toml").is_file():
        raise ValueError("Paid execution must use the WSPBench checkout's shared project ledger")
    return project / ".wspbench-state" / "budget.sqlite3"


def suite_digest(root: Path) -> str:
    from .tasks import load_tasks
    root = Path(root).resolve()
    artifacts = {}
    for task in load_tasks(root, "all"):
        directory = task.directory.resolve()
        if not directory.is_relative_to(root):
            raise ValueError("Task directory leaves the suite")
        for file in sorted(directory.iterdir()):
            if file.is_file() and not file.name.startswith("._"):
                if not file.resolve().is_relative_to(root):
                    raise ValueError("Task artifact leaves the suite")
                artifacts[str(file.relative_to(root))] = hashlib.sha256(file.read_bytes()).hexdigest()
    for name in ("suite.json", "private-suite.json"):
        path = root / name
        if path.exists():
            artifacts[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return _hash(artifacts)


def harness_digest() -> str:
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(Path(__file__).parent.glob("*.py")) if not p.name.startswith("._")}
    files["shared_budget.py"] = hashlib.sha256((Path(__file__).parent.parent / "budget.py").read_bytes()).hexdigest()
    files["shared_provider_options.py"] = hashlib.sha256((Path(__file__).parent.parent / "provider_options.py").read_bytes()).hexdigest()
    return _hash(files)


def _wrong_view(task, tasks):
    choices = sorted((other for other in tasks if other.parent_problem_id != task.parent_problem_id
                      and other.family == task.family and other.architecture == task.architecture
                      and other.target_stratum == task.target_stratum
                      and other.capacity_tier == task.capacity_tier), key=lambda t: t.id)
    if not choices:
        raise ValueError(f"No independent architecture/stratum/capacity-matched wrong checkpoint for {task.id}")
    other = choices[0]
    weights = load_weights(other.weights_path)
    if task.family == "edit_effects":
        return {"target": weights, "before": load_weights(other.directory / "before.npz"), "after": weights}
    return weights


def _base_record(task, actor_id, actor_type, condition, view, protocol_digest, attempt=1):
    return {"schema_version": "2", "benchmark_version": VERSION, "suite_tier": "challenge_candidate",
            "headline_eligible": False, "task_id": task.id, "parent_problem_id": task.parent_problem_id,
            "mechanism_family_id": task.mechanism_family_id, "family": task.family, "skill": task.skill,
            "split": task.split, "target_stratum": task.target_stratum, "capacity_tier": task.capacity_tier,
            "actor_id": actor_id, "actor_name": actor_id, "actor_type": actor_type,
            "condition": condition, "weight_view": view, "attempt": attempt,
            "protocol_digest": protocol_digest, "target_digest": weights_digest(load_weights(task.weights_path)),
            "question_digest": _hash(task.to_public_dict()), "cost_usd": 0.,
            "prompt_information": task.public.get("prompt_information", {}),
            "hint_level": task.public.get("prompt_information", {}).get("hint_level", "unspecified")}


def _finalize_cpu(session, record, run_dir, submission, diagnostics):
    from .scoring import score
    session.submit(submission)
    finished = session.finish()
    result = score(session.task, finished["submission"])
    record.update({k: v for k, v in finished.items() if k != "trace"})
    record.update(success=bool(result.success), metrics=result.metrics, status="completed",
                  score_status=result.status, score_reason=result.reason, diagnostics=diagnostics,
                  usage=finished["counters"], observation_usage=finished["counters"])
    if result.status != "scored":
        record["status"] = result.status
    if record["audit"]["status"] == "rejected":
        record.update(success=False, status="protocol_violation")
    if session.condition is Condition.INSTRUMENTED:
        record["budget_curve"] = [{"budget": int(level), "actual_observations":
            finished["snapshot_actual_observations"][level], **score(session.task, answer).to_dict()}
            for level, answer in finished["snapshots"].items()]
    _write(run_dir / "trace.json", finished["trace"])
    _write(run_dir / "submission.json", finished["submission"])
    return record


def run_cpu_matrix(data_root: Path, output_root: Path, actor: str = "reference", *,
                   splits=("dev",), views=("full",), conditions=("A", "B", "C"),
                   baseline_names=None) -> dict:
    """Execute real reference/baseline sessions without provider calls or credentials."""
    from .tasks import load_tasks
    from .scoring import reference_submission
    from .reporting import build_report
    if actor not in {"reference", "baseline"}:
        raise ValueError("CPU actor must be reference or baseline")
    data_root, output_root = Path(data_root), Path(output_root)
    all_tasks = load_tasks(data_root, "all")
    chosen = [t for t in all_tasks if t.split in splits]
    if not chosen:
        raise ValueError("No tasks in selected CPU splits")
    views = tuple(WeightView(x).value for x in views)
    conditions = tuple(Condition(x).value for x in conditions)
    protocol = {"benchmark_version": VERSION, "suite_digest": suite_digest(data_root),
        "harness_digest": harness_digest(), "splits": list(splits), "views": list(views),
        "conditions": list(conditions), "actor": actor, "baseline_names": baseline_names,
        "observation_budget": {"core": [0, 32, 128], "loop": [0, 4, 16]},
        "no_paid_calls": True}
    protocol_digest = _hash(protocol)
    _write(output_root / "manifest.json", {**protocol, "protocol_digest": protocol_digest})
    records = []
    if actor == "baseline":
        from .baselines import BASELINES, access_card, fit_statistics_reader, run_baseline
        names = list(baseline_names or BASELINES)
        reader_error = None
        try:
            reader = fit_statistics_reader([t for t in all_tasks if t.split == "baseline_train"])
        except ValueError as exc:
            reader, reader_error = None, str(exc)
    else:
        names = ["oracle_reference"]
    for task in chosen:
        for condition in conditions:
            if condition not in task.conditions:
                continue
            for view in views:
                wrong = _wrong_view(task, all_tasks) if view == "matched_wrong" else None
                for name in names:
                    actor_type = "reference" if actor == "reference" else "baseline"
                    record = _base_record(task, name, actor_type, condition, view, protocol_digest)
                    record["actor_kind"] = "oracle_reference" if actor == "reference" else "conventional_comparator"
                    record["access_card"] = {"privileged": actor == "reference", "condition": condition,
                        "tools": [] if condition == "A" else ["bounded_algebra"] + (["metered_practice"] if condition == "C" else []),
                        "note": "Reference uses private solution only to validate the harness; it is not a solver score." if actor == "reference" else "Conventional algorithm; no internal LLM cognition claim."}
                    record["id"] = _hash({"protocol": protocol_digest, "actor": name,
                        "parent": task.parent_problem_id, "task": task.id, "target": record["target_digest"],
                        "condition": condition, "view": view, "attempt": 1})
                    run_dir = output_root / "sessions" / record["id"]
                    session = AccessSession(task, condition, view, wrong)
                    if actor == "reference":
                        submission, diagnostics = reference_submission(task), {"privileged_reference": True}
                    else:
                        record["access_card"].update(access_card(name, training=reader.training if name == "statistical_reader" and reader else None))
                        outcome = run_baseline(session, name, statistics_reader=reader)
                        if name == "statistical_reader" and reader_error:
                            outcome.setdefault("diagnostics", {})["reader_unavailable"] = reader_error
                        if outcome.get("status") == "not_applicable":
                            record.update(status="not_applicable", success=False,
                                          diagnostics=outcome.get("diagnostics", {}))
                            records.append(record)
                            _write(run_dir / "result.json", record)
                            continue
                        submission = outcome["submission"]
                        diagnostics = outcome.get("diagnostics", {})
                        record["access_card"].update(outcome.get("access_card", {}))
                    _finalize_cpu(session, record, run_dir, submission, diagnostics)
                    record["artifact_hashes"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in run_dir.iterdir() if p.is_file() and p.name != "result.json"}
                    _write(run_dir / "result.json", record)
                    records.append(record)
    report = build_report(records)
    summary = {"status": "passed" if actor != "reference" or all(r.get("success") for r in records) else "failed",
        "actor": actor, "protocol_digest": protocol_digest, "suite_digest": protocol["suite_digest"],
        "sessions": len(records), "successes": sum(bool(r.get("success")) for r in records),
        "no_paid_calls": True, "report": report, "tables": report["tables"]}
    _write(output_root / "records.json", records)
    _write(output_root / "report.json", report)
    _write(output_root / "summary.json", summary)
    return summary


def _load_models(config_path, model_ids=None):
    models = json.loads(Path(config_path).read_text())["models"]
    if not models or len({m["id"] for m in models}) != len(models):
        raise ValueError("Duplicate model configuration IDs")
    if model_ids and set(model_ids) - {m["id"] for m in models}:
        raise ValueError("Unknown model ID")
    return [m for m in models if not model_ids or m["id"] in model_ids]


def _calibration_cells(data_root, output_root, admission=None, validation=None):
    """Resolve reviewed permissions to exact current development checkpoints."""
    from .tasks import load_tasks
    output_root = Path(output_root)
    admission = admission or json.loads((output_root / "admission.json").read_text())
    validation = validation or json.loads((output_root / "cpu-validation.json").read_text())
    allowed = admission.get("calibration_allowlist")
    if not admission.get("calibration_permitted") or not isinstance(allowed, list) or not allowed:
        raise ValueError("Paid calibration needs a nonempty exact development-cell allowlist")
    baseline_path = Path(admission.get("baseline_evidence_path", output_root / "baselines.json"))
    baseline = json.loads(baseline_path.read_text())
    if baseline.get("suite_digest") != suite_digest(data_root) or baseline.get("harness_digest") != harness_digest():
        raise ValueError("Reviewed baseline evidence names another suite or harness")
    evidence_digest = _hash(baseline)
    if evidence_digest != admission.get("baseline_evidence_digest"):
        raise ValueError("Reviewed baseline evidence changed or is missing")
    if admission.get("feasibility_digest") != _hash(validation.get("feasibility_records", [])):
        raise ValueError("Reviewed feasibility evidence changed")
    from .admission import assess_admission
    computed = assess_admission(Path(data_root), baseline, validation)
    checked_permissions = {_hash(cell) for cell in computed["calibration_allowlist"]}
    if not computed["calibration_permitted"] or any(_hash(cell) not in checked_permissions for cell in allowed):
        raise ValueError("Reviewed permissions exceed the current scientific checks")
    tasks = {t.id: t for t in load_tasks(data_root, "all") if t.split == "dev"}
    proofs = {(p.get("task_id"), p.get("condition")): p for p in validation.get("feasibility_records", [])}
    seen = set()
    for cell in allowed:
        if not isinstance(cell, dict) or set(cell) != {"task_id", "target_digest", "condition", "weight_view",
                                                     "feasibility_digest", "baseline_evidence_digest"}:
            raise ValueError("Malformed calibration-cell permission")
        task = tasks.get(cell["task_id"])
        key = (cell["task_id"], cell["condition"], cell["weight_view"])
        if key in seen:
            raise ValueError("Duplicate calibration-cell permission")
        seen.add(key)
        if not task or cell["condition"] not in task.conditions or cell["weight_view"] != "full":
            raise ValueError("Calibration permission is not a declared full-view development cell")
        if cell["target_digest"] != weights_digest(load_weights(task.weights_path)):
            raise ValueError("Calibration permission names a different target checkpoint")
        proof = proofs.get((task.id, cell["condition"]))
        if not proof or cell["feasibility_digest"] != _hash(proof):
            raise ValueError("Calibration permission lacks its exact checked feasibility record")
        if cell["baseline_evidence_digest"] != evidence_digest:
            raise ValueError("Calibration permission lacks its exact baseline evidence")
    return allowed


def _select_calibration_cells(tasks, allowed, conditions, views, task_ids=None):
    ids = set(task_ids) if task_ids is not None else None
    available = {t.id for t in tasks if t.split == "dev"}
    if ids is not None and (not ids or ids - available):
        raise ValueError("Calibration task filter must name existing development targets")
    selected = [c for c in allowed if c["condition"] in conditions and c["weight_view"] in views
                and (ids is None or c["task_id"] in ids)]
    if not selected or (ids is not None and ids != {c["task_id"] for c in selected}):
        raise ValueError("Requested tasks/access conditions have no reviewed calibration permission")
    return sorted(selected, key=lambda c: (c["task_id"], c["condition"], c["weight_view"]))


def preflight(data_root: Path, output_root: Path, config_path: Path = Path("configs/models.json"),
               *, model_ids=None, admission_path: Path | None = None, phase="calibration") -> dict:
    """Read readiness only. Does not read dotenv or make network requests."""
    issues = []
    dependencies = {name: bool(importlib.util.find_spec(name)) for name in ("numpy", "inspect_ai")}
    issues.extend(f"Missing dependency {name}" for name, installed in dependencies.items() if not installed)
    states = []
    try:
        models = _load_models(config_path, model_ids)
        for model in models:
            priced = all(type(model.get("pricing", {}).get(k)) in (int, float) and
                         math.isfinite(model["pricing"][k]) and model["pricing"][k] >= 0 for k in PRICE_FIELDS)
            ready = bool(model.get("snapshot") and model.get("verified_at") and priced and
                         model.get("pricing", {}).get("verified_at") and model.get("pricing", {}).get("source"))
            credential = bool(os.environ.get(model["api_key_env"]))
            if not ready:
                issues.append(f"Verify immutable/upstream version and all token prices for {model['id']}")
            if not credential:
                issues.append(f"Credential unavailable for {model['id']}")
            states.append({"id": model["id"], "configuration_verified": ready,
                           "credential_available": credential})
    except (ValueError, OSError, KeyError) as exc:
        issues.append(str(exc))
    output_root = Path(output_root)
    validation_path = output_root / "cpu-validation.json"
    validation = {}
    try:
        validation = json.loads(validation_path.read_text())
        if validation.get("status") != "passed" or validation.get("suite_digest") != suite_digest(data_root) or validation.get("harness_digest") != harness_digest():
            issues.append("CPU validation is absent, failed, or refers to a different suite")
    except (OSError, ValueError, KeyError):
        issues.append("Run CPU validation and persist its current suite digest before paid work")
    admission_path = Path(admission_path or output_root / "admission.json")
    try:
        admission = json.loads(admission_path.read_text())
        permitted = admission.get("calibration_permitted") if phase == "calibration" else (
            admission.get("paid_runs_permitted") and admission.get("headline_admitted"))
        if not permitted:
            issues.append(f"Scientific admission has not permitted paid {phase}")
        if admission.get("suite_digest") != suite_digest(data_root):
            issues.append("Admission must name the exact validated suite digest")
        if admission.get("harness_digest") != harness_digest():
            issues.append("Admission must name the exact validated harness digest")
        if not admission.get("reviewer") or not admission.get("evidence"):
            issues.append("Admission needs an independent reviewer and evidence")
        if phase == "calibration" and permitted:
            try:
                _calibration_cells(data_root, output_root, admission, validation)
            except (OSError, ValueError, KeyError) as exc:
                issues.append(str(exc))
    except (OSError, ValueError, KeyError):
        issues.append("Missing reviewed scientific-admission artifact")
    return {"ready": not issues, "issues": issues, "dependencies": dependencies, "models": states,
            "execution_surface": "bounded JSON tensor tools; no general Python or arbitrary filesystem"}


def _matrix(models, tasks, attempts, conditions, views):
    return [{"model_id": m["id"], "model_config_digest": _hash(m), "task_id": t.id,
             "target_digest": weights_digest(load_weights(t.weights_path)),
             "parent_problem_id": t.parent_problem_id, "condition": c, "weight_view": v, "attempt": a}
            for m in models for t in tasks for c in conditions if c in t.conditions
            for v in views for a in range(1, attempts + 1)]


def freeze_protocol(data_root: Path, output_root: Path, config_path: Path,
                    *, conditions=("A", "B", "C"), views=("full",), attempts=3) -> dict:
    from .tasks import load_tasks
    data_root, output_root = Path(data_root), Path(output_root)
    readiness = preflight(data_root, output_root, config_path, phase="primary")
    if not readiness["ready"]:
        raise ValueError("Freeze blocked: " + "; ".join(readiness["issues"]))
    if attempts != 3:
        raise ValueError("Scored matrix requires three attempts, averaged without best-of selection")
    conditions = tuple(Condition(c).value for c in conditions)
    views = tuple(WeightView(v).value for v in views)
    models = _load_models(config_path)
    tasks = load_tasks(data_root, "heldout")
    calibration_paths = list((output_root / "calibration").rglob("result.json"))
    calibration = [json.loads(p.read_text()) for p in calibration_paths]
    from .audit import trace_digest
    for path, record in zip(calibration_paths, calibration):
        trace = json.loads((path.parent / "trace.json").read_text())
        if record.get("audit", {}).get("trace_sha256") != trace_digest(trace):
            raise ValueError("A calibration trace changed after audit")
        for name, expected_digest in record.get("artifact_hashes", {}).items():
            artifact = path.parent / name
            if artifact.parent != path.parent or hashlib.sha256(artifact.read_bytes()).hexdigest() != expected_digest:
                raise ValueError("A calibration artifact changed after execution")
    dev = load_tasks(data_root, "dev")
    expected_cal = _matrix(models, dev, 1, conditions, views)
    keyed = {(r["model_id"], r["task_id"], r["condition"], r["weight_view"], r["attempt"]): r for r in calibration}
    if len(keyed) != len(calibration):
        raise ValueError("Duplicate calibration attempt")
    costs = {m["id"]: [] for m in models}
    for item in expected_cal:
        key = tuple(item[k] for k in ("model_id", "task_id", "condition", "weight_view", "attempt"))
        r = keyed.get(key)
        if not r or r.get("actor_type") != "model" or r.get("audit", {}).get("status") not in {"approved", "rejected"}:
            raise ValueError("Complete and audit every matched development-calibration session before freeze")
        if r.get("suite_digest") != suite_digest(data_root) or r.get("harness_digest") != harness_digest() or r.get("model_config_digest") != item["model_config_digest"]:
            raise ValueError("Calibration suite, harness or model configuration differs")
        if r.get("usage_unknown") or r.get("cost_usd") is None or r.get("status") == "infrastructure_failure":
            raise ValueError("Reconcile all calibration outcomes and costs")
        costs[item["model_id"]].append(r["cost_usd"])
    expected = _matrix(models, tasks, attempts, conditions, views)
    forecast = sum(max(costs[m["id"]]) * sum(x["model_id"] == m["id"] for x in expected) * 1.3 for m in models)
    if forecast > 1800:
        raise BudgetExceeded("Complete v0.2 matrix exceeds the $1,800 primary allocation; revise before scoring")
    budget = BudgetLedger(_project_ledger()).snapshot()
    if budget["unsettled_reservations"] or forecast + budget["phases"]["primary"]["committed_usd"] > 1800 or forecast + budget["committed_usd"] > 3000:
        raise BudgetExceeded("Complete v0.2 matrix exceeds remaining shared budget or has unreconciled billing")
    frozen = {"schema_version": "2", "benchmark_version": VERSION, "suite_digest": suite_digest(data_root),
        "harness_digest": harness_digest(), "models": models, "config_digest": _hash(models),
        "conditions": list(conditions), "views": list(views), "attempts": attempts,
        "expected_matrix": expected, "expected_sessions": len(expected),
        "limits_by_task": {t.id: t.to_public_dict()["limits"] for t in tasks},
        "primary_projection_usd": forecast, "budget_ceiling_usd": 3000,
        "calibration_hashes": [_hash(r) for r in calibration],
        "admission_digest": _hash(json.loads((output_root / "admission.json").read_text())),
        "cpu_validation_digest": _hash(json.loads((output_root / "cpu-validation.json").read_text())),
        "frozen_at": datetime.now(timezone.utc).isoformat()}
    frozen["protocol_digest"] = _hash(frozen)
    _write(output_root / "freeze.json", frozen)
    return frozen


async def _run_paid(data_root, output_root, config_path, phase, attempts, model_ids, conditions, views, ledger_path,
                    task_ids=None, max_sessions=6):
    # All gates run before Inspect resolves a provider or a reservation is made.
    readiness = preflight(data_root, output_root, config_path, model_ids=model_ids, phase=phase)
    if not readiness["ready"]:
        raise ValueError("Paid execution blocked: " + "; ".join(readiness["issues"]))
    if phase not in {"calibration", "primary"}:
        raise ValueError("Paid v0.2 phase must be calibration or primary")
    from .tasks import load_tasks
    all_tasks = load_tasks(data_root, "all")
    all_models = _load_models(config_path)
    models = [m for m in all_models if not model_ids or m["id"] in model_ids]
    if phase == "primary":
        if model_ids or task_ids is not None or attempts != 3:
            raise ValueError("Primary uses the complete frozen model matrix and three attempts")
        frozen = json.loads((output_root / "freeze.json").read_text())
        unsigned = {k: v for k, v in frozen.items() if k != "protocol_digest"}
        if frozen.get("protocol_digest") != _hash(unsigned):
            raise ValueError("Frozen protocol integrity check failed")
        if (frozen["suite_digest"] != suite_digest(data_root) or frozen["harness_digest"] != harness_digest()
                or frozen["config_digest"] != _hash(models) or frozen["conditions"] != list(conditions)
                or frozen["views"] != list(views)):
            raise ValueError("Frozen protocol differs from requested run")
        if frozen["admission_digest"] != _hash(json.loads((output_root / "admission.json").read_text())) or frozen["cpu_validation_digest"] != _hash(json.loads((output_root / "cpu-validation.json").read_text())):
            raise ValueError("Scientific admission or CPU validation changed after freeze")
        tasks = [t for t in all_tasks if t.split == "heldout"]
        protocol_digest = frozen["protocol_digest"]
    else:
        attempts = 1
        tasks = [t for t in all_tasks if t.split == "dev"]
        admission = json.loads((output_root / "admission.json").read_text())
        selected_cells = _select_calibration_cells(all_tasks, _calibration_cells(data_root, output_root, admission),
                                                   conditions, views, task_ids)
        allowed_keys = {(c["task_id"], c["condition"], c["weight_view"]) for c in selected_cells}
        tasks = [t for t in tasks if t.id in {c["task_id"] for c in selected_cells}]
        tasks.sort(key=lambda task: ({"rule_recovery":0,"edit_effects":1,"compositional_synthesis":2}.get(task.family,99),task.id))
        if type(max_sessions) is not int or not 1 <= max_sessions <= 24:
            raise ValueError("Small development calibration requires a session limit from 1 through 24")
        if len(selected_cells) * len(models) > max_sessions:
            raise ValueError("Requested calibration exceeds its explicit session limit; select fewer admitted tasks")
        from .continuation import authorization_digest_if_present
        continuation_digest=authorization_digest_if_present(output_root)
        protocol_digest = _hash({"suite": suite_digest(data_root), "harness": harness_digest(),
                                 "models": all_models, "conditions": conditions, "views": views, "phase": phase,
                                 "admission_digest": _hash(admission), "cells": selected_cells,
                                 "continuation_authorization_digest":continuation_digest})
    manifest_models = all_models if phase == "calibration" else models
    manifest = {"schema_version": "2", "benchmark_version": VERSION, "protocol_digest": protocol_digest,
        "suite_digest": suite_digest(data_root), "harness_digest": harness_digest(),
        "conditions": list(conditions), "views": list(views), "models": manifest_models, "attempts": attempts,
        "execution_order": "task_then_model",
        "expected_matrix": [item for item in _matrix(manifest_models, tasks, attempts, conditions, views)
                            if phase == "primary" or (item["task_id"], item["condition"], item["weight_view"]) in allowed_keys], "phase": phase}
    manifest["expected_sessions"] = len(manifest["expected_matrix"])
    destination = output_root / phase
    old_path = destination / "manifest.json"
    if old_path.exists() and json.loads(old_path.read_text()) != manifest:
        raise ValueError("Existing run uses a different matrix; preserve it and use a fresh output root")
    _write(old_path, manifest)
    project_ledger = _project_ledger()
    if ledger_path is not None and Path(ledger_path).resolve() != project_ledger.resolve():
        raise ValueError("v0.2 must share the project-wide v0.1 budget ledger")
    ledger = BudgetLedger(project_ledger)
    from .continuation import validate_continuation, reserve_with_continuation
    continuation=validate_continuation(data_root,output_root,config_path,ledger=ledger,
        model_configs=models,cells=selected_cells if phase=="calibration" else [],phase=phase)
    _inspect_paths(output_root)
    from inspect_ai import eval_async
    from inspect_ai.model import GenerateConfig, get_model
    from .inspect_task import build_inspect_task
    records = []
    provider_models = {}
    for t in tasks:
        for m in models:
            if m["id"] not in provider_models:
                provider_models[m["id"]] = get_model(m["model"], base_url=m.get("base_url"), api_key=os.environ[m["api_key_env"]],
                    config=GenerateConfig(**provider_generate_options(m)), memoize=False, **provider_client_options(m))
            model = provider_models[m["id"]]
            for c in conditions:
                if c not in t.conditions:
                    continue
                for v in views:
                    if phase == "calibration" and (t.id, c, v) not in allowed_keys:
                        continue
                    wrong = _wrong_view(t, all_tasks) if v == "matched_wrong" else None
                    for attempt in range(1, attempts + 1):
                        run_id = _hash({"protocol": protocol_digest, "model": _hash(m),
                            "parent": t.parent_problem_id, "task": t.id,
                            "target": weights_digest(load_weights(t.weights_path)),
                            "condition": c, "view": v, "attempt": attempt})
                        run_dir = destination / run_id
                        result_path = run_dir / "result.json"
                        if result_path.exists():
                            existing = json.loads(result_path.read_text())
                            if existing.get("protocol_digest") != protocol_digest:
                                raise ValueError("Existing attempt belongs to a different protocol")
                            records.append(existing)
                            continue
                        maximum = maximum_session_cost(m, t.limits)
                        reservation = reserve_with_continuation(ledger,continuation,phase,maximum,run_id)
                        record = _base_record(t, m["id"], "model", c, v, protocol_digest, attempt)
                        record.update(id=run_id, model_id=m["id"], provider=m["provider"], model_config=m,
                            model_config_digest=_hash(m), suite_digest=manifest["suite_digest"],
                            harness_digest=manifest["harness_digest"], phase=phase, cost_usd=None,
                            maximum_reserved_usd=maximum, reservation_id=reservation, status="running")
                        try:
                            evaluation = build_inspect_task(t, run_dir, m, record, condition=c,
                                                            weight_view=v, wrong_weights=wrong)
                            logs = await eval_async(evaluation, model=model, log_dir=str(run_dir / "inspect"),
                                log_model_api=False, log_shared=False, retry_on_error=0, log_realtime=False,
                                ctl_server=False, max_samples=1, max_tasks=1)
                            if not logs or logs[0].status != "success" or record.get("usage_unknown"):
                                raise RuntimeError("Inspect failure or unknown billing; reservation retained")
                            record["cost_usd"] = session_cost(m, record["usage"])
                            record["cost_method"] = "Category-specific token estimate; reasoning included in output once"
                            ledger.settle(reservation, record["cost_usd"])
                        except Exception as exc:
                            provider_error=record.get("error")
                            message = f"{type(exc).__name__}: {exc}"
                            for item in models:
                                secret = os.environ.get(item["api_key_env"])
                                if secret:
                                    message = message.replace(secret, "[REDACTED]")
                                    if provider_error:provider_error=provider_error.replace(secret,"[REDACTED]")
                            if provider_error:record["provider_error"]=provider_error
                            record.update(status="infrastructure_failure", error=message[:1000])
                            _write(result_path, record)
                            raise RuntimeError("Paid execution stopped; reconcile retained reservation before continuing") from None
                        record["artifact_hashes"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in run_dir.iterdir() if p.is_file() and p.name != "result.json"}
                        _write(result_path, record)
                        records.append(record)
    return records


def run_evaluation(data_root: Path, output_root: Path, config_path=Path("configs/models.json"), *,
                    phase="calibration", attempts=3, model_ids=None, conditions=("A", "B", "C"),
                    views=("full",), ledger_path=None, task_ids=None, max_sessions=6):
    conditions = tuple(Condition(c).value for c in conditions)
    views = tuple(WeightView(v).value for v in views)
    if not conditions or len(set(conditions)) != len(conditions) or not views or len(set(views)) != len(views):
        raise ValueError("Paid evaluation conditions and weight views must be nonempty and unique")
    return asyncio.run(_run_paid(Path(data_root), Path(output_root), Path(config_path), phase,
                                attempts, model_ids, conditions, views, ledger_path, task_ids, max_sessions))


async def _smoke(data_root, output_root):
    """Exercise actual Inspect tools with canned mock outputs, never a provider."""
    from inspect_ai import eval_async
    from inspect_ai.model import ChatMessageAssistant, ModelOutput, ModelUsage
    from inspect_ai.tool import ToolCall
    from .inspect_task import build_inspect_task
    from .scoring import reference_submission
    from .tasks import load_tasks
    tasks = {}
    for task in load_tasks(data_root, "dev"):
        tasks.setdefault(task.family, task)
    if not tasks:
        raise ValueError("Smoke requires development tasks")
    output_root.mkdir(parents=True, exist_ok=True)
    _inspect_paths(output_root)
    records = []
    for task in tasks.values():
        for condition in task.conditions:
            answer = reference_submission(task)
            outputs = []
            def output(content="", function=None, arguments=None):
                calls = [ToolCall(id=str(len(outputs) + 1), function=function, arguments=arguments)] if function else None
                outputs.append(ModelOutput(model="mockllm", choices=[{"message": ChatMessageAssistant(content=content, tool_calls=calls)}],
                    usage=ModelUsage(input_tokens=3, input_tokens_cache_write=7, input_tokens_cache_read=2,
                                     output_tokens=5, reasoning_tokens=1, total_tokens=17)))
            if condition != "A":
                tensor = next(iter(task.architecture["shapes"]))
                output(function="analysis", arguments={"operation": "summary", "operands": [f"target:{tensor}"]})
                if condition == "C":
                    arguments = {"indices": [0], "observation": "activations"}
                    if task.family == "causal_forecast":
                        edit = task.public["practice_interventions"][0]
                        arguments["intervention_id"] = edit["id"] if isinstance(edit, dict) else edit
                    output(function="practice", arguments=arguments)
                    output(function="analysis", arguments={"operation": "mean", "operands": ["derived:0"],
                                                             "parameters_json": canonical_json({"axis": 0})})
                output(function="submit", arguments={"submission_json": canonical_json(answer)})
            else:
                output(content=canonical_json(answer))
            run_dir = output_root / task.family / condition
            record = _base_record(task, "mockllm_canned", "protocol_smoke", condition, "full", "mock-plumbing-only")
            record.update(actor_kind="mock_protocol_fixture", mock=True, usage_is_synthetic=True,
                          benchmark_result=False, cost_usd=0.)
            evaluation = build_inspect_task(task, run_dir, {}, record, condition=condition)
            logs = await eval_async(evaluation, model="mockllm/model", model_args={"custom_outputs": outputs},
                log_dir=str(run_dir / "inspect"), log_model_api=False, log_shared=False,
                log_realtime=False, ctl_server=False, retry_on_error=0, max_samples=1, max_tasks=1)
            record["inspect_completed"] = bool(logs and logs[0].status == "success")
            _write(run_dir / "result.json", record)
            records.append(record)
    summary = {"status": "passed" if all(r["inspect_completed"] and r.get("status") == "completed" for r in records) else "failed",
        "sessions": len(records), "mock": True, "benchmark_result": False, "cost_usd": 0.,
        "notice": "Canned reference answers and synthetic token usage test Inspect integration only; no model capability or paid inference was measured."}
    _write(output_root / "summary.json", summary)
    return summary


def run_smoke(data_root: Path, output_root: Path) -> dict:
    data_root, output_root = Path(data_root).resolve(), Path(output_root).resolve()
    with _mock_runtime():
        return asyncio.run(_smoke(data_root, output_root))
