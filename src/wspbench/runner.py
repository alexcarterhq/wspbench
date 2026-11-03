"""Paid evaluation orchestration, provenance, calibration and freeze gates."""
from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil

from .audit import review_run
from .budget import BudgetLedger, BudgetExceeded, maximum_session_cost, session_cost, PRICE_FIELDS
from .sandbox import docker_status
from .provider_options import provider_client_options,provider_generate_options


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_model_config(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text())
    models = data["models"]
    if len(models) != 4 or len({m["id"] for m in models}) != 4:
        raise ValueError("The pilot requires four distinct model configurations")
    if {m["provider"] for m in models} != {"openai", "anthropic", "deepseek", "qwen"}:
        raise ValueError("The pilot requires one model from each selected provider")
    return models


def suite_digest(root: Path) -> str:
    root = Path(root).resolve()
    h = hashlib.sha256()
    suite_path = root / "suite.json"
    suite = json.loads(suite_path.read_text())
    files = {suite_path, root / "private-suite.json"}
    for entry in suite["tasks"]:
        manifest = (root / entry["manifest"]).resolve()
        if not manifest.is_relative_to(root):
            raise ValueError("Task manifest leaves the suite directory")
        files.update(manifest.parent / name for name in (
            "manifest.json", "target.npz", "reference.npz", "private.json", "private.npz", "practice.npz"))
    files = sorted(files)
    for path in files:
        if not path.resolve().is_relative_to(root):
            raise ValueError("Task artifact leaves the suite directory")
        h.update(str(path.relative_to(root)).encode())
        h.update(b"\0")
        h.update(hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()


def harness_digest() -> str:
    names = ["runner.py", "inspect_task.py", "budget.py", "audit.py", "sandbox.py", "domain.py", "networks.py", "scoring.py"]
    return _hash({name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in names})


def _permuted_weights(architecture: dict, weights: dict, permutations: list):
    import numpy as np
    if architecture["kind"] == "rnn":
        p = permutations[0]
        return {"rnn.weight_ih": weights["rnn.weight_ih"][p].copy(),
                "rnn.weight_hh": weights["rnn.weight_hh"][np.ix_(p, p)].copy(),
                "rnn.bias": weights["rnn.bias"][p].copy(),
                "head.weight": weights["head.weight"][:, p].copy(), "head.bias": weights["head.bias"].copy()}
    if architecture["kind"] == "mlp":
        result = {}
        order = [np.arange(architecture["sizes"][0]), *permutations, np.arange(architecture["sizes"][-1])]
        for i in range(len(order) - 1):
            result[f"w{i}"] = weights[f"w{i}"][np.ix_(order[i + 1], order[i])].copy()
            result[f"b{i}"] = weights[f"b{i}"][order[i + 1]].copy()
        return result
    raise ValueError("Unsupported architecture for symmetry diagnostics")


def generate_diagnostics(data_root: Path, output_root: Path, frozen: dict) -> list:
    """Create true paired neuron permutations; zero-initialized Write is excluded."""
    import numpy as np
    from .tasks import load_tasks
    from .networks import forward, load_weights, save_weights
    from .scoring import reference_submission, score
    groups = {}
    for task in sorted(load_tasks(data_root, "official"), key=lambda t: t.id):
        if task.skill in {"read", "repair"}:
            groups.setdefault(task.family, []).append(task)
    if len(groups) != 4 or any(len(v) < 3 for v in groups.values()):
        raise ValueError("Diagnostics require at least three independent parents per Read/Repair family")
    tasks = []
    for family in sorted(groups):
        for original in groups[family][:3]:
            seed = int(_hash({"suite": frozen["suite_digest"], "parent": original.id})[:16], 16)
            rng = np.random.default_rng(seed)
            arch = original.architecture
            sizes = [arch["hidden"]] if arch["kind"] == "rnn" else arch["sizes"][1:-1]
            permutations = [rng.permutation(size) for size in sizes]
            directory = output_root / "diagnostic-tasks" / (original.id + "-permuted")
            task = replace(original, id=original.id + "-permuted", directory=directory,
                           public={**original.public, "condition": "equivalent_parameterization"})
            directory.mkdir(parents=True, exist_ok=True)
            for name in ("private.json", "private.npz", "practice.npz"):
                source = original.directory / name
                if source.exists():
                    shutil.copyfile(source, directory / name)
            for name in ("target.npz", "reference.npz"):
                source = original.directory / name
                if source.exists():
                    before = load_weights(source)
                    after = _permuted_weights(arch, before, permutations)
                    width = arch["length"] if arch["kind"] == "rnn" else arch["sizes"][0]
                    x = rng.uniform(-1, 1, (64, width))
                    if not np.allclose(forward(arch, before, x), forward(arch, after, x), rtol=1e-5, atol=1e-6):
                        raise ValueError("Neuron permutation failed functional-equivalence check")
                    save_weights(directory / name, after)
            _write(directory / "manifest.json", task.to_public_dict())
            if not score(task, reference_submission(task)).success:
                raise ValueError("Permuted reference no longer satisfies task")
            tasks.append(task)
    return tasks


def preflight(config_path: Path = Path("configs/models.json"), model_ids: list[str] | None = None) -> dict:
    issues = []
    dependencies = {}
    for name in ("numpy", "torch", "inspect_ai"):
        dependencies[name] = bool(importlib.util.find_spec(name))
        if not dependencies[name]:
            issues.append(f"Install {name} with pip install -e '.[eval]'")
    try:
        models = load_model_config(config_path)
        if model_ids:
            if set(model_ids) - {m["id"] for m in models}:
                raise ValueError("Unknown model filter")
            models = [m for m in models if m["id"] in model_ids]
    except (OSError, ValueError, KeyError) as exc:
        return {"ready": False, "issues": [str(exc)], "dependencies": dependencies, "models": []}
    states = []
    for model in models:
        key = bool(os.environ.get(model["api_key_env"]))
        verified = bool(model.get("verified_at") and model.get("snapshot"))
        prices = model.get("pricing", {})
        prices_verified = bool(prices.get("verified_at") and prices.get("source") and
                               all(type(prices.get(k)) in (int, float) for k in PRICE_FIELDS))
        states.append({"id": model["id"], "provider": model["provider"], "credential_available": key,
                       "version_verified": verified, "prices_verified": prices_verified})
        if not key:
            issues.append(f"Set {model['api_key_env']} securely; do not paste keys into chat")
        if not verified:
            issues.append(f"Resolve and record upstream version/snapshot and verified_at for {model['id']}")
        if not prices_verified:
            issues.append(f"Confirm uncached/peak token prices and verified_at for {model['id']}")
    docker = docker_status()
    if not docker.get("available"):
        issues.append(docker["reason"])
    elif not docker.get("image_ready"):
        issues.append("Build the worker: docker build -t wspbench-worker:0.1 docker")
    return {"ready": not issues, "issues": issues, "dependencies": dependencies, "models": states, "docker": docker}


def _calibration_tasks(tasks):
    selected = {}
    for task in sorted(tasks, key=lambda t: t.id):
        selected.setdefault(task.family, task)
    if len(selected) != 6:
        raise ValueError("Calibration requires all six task families")
    return list(selected.values())


def _expected(models, tasks, attempts):
    return [{"model_id": m["id"], "task_id": t.id, "attempt": a}
            for m in models for t in tasks for a in range(1, attempts + 1)]


def _require_challenge_suite(data_root: Path) -> None:
    from .domain import SANITY_FAMILIES
    from .tasks import load_tasks
    if any(task.family in SANITY_FAMILIES for task in load_tasks(data_root, "official")):
        raise ValueError("The v0.1 generators are sanity checks only and cannot enter a headline evaluation. "
                         "See docs/DESIGN_V0_2.md for challenge admission and the three access conditions.")


def freeze_protocol(data_root: Path, output_root: Path, config_path: Path,
                    calibration_root: Path | None = None) -> dict:
    from .tasks import load_tasks, validate_suite
    data_root, output_root = Path(data_root), Path(output_root)
    _require_challenge_suite(data_root)
    validation = validate_suite(data_root)
    if (validation.get("status") != "passed" or validation.get("task_count") != 60
            or validation.get("reference_successes") != 60):
        raise ValueError("Suite validation failed")
    models = load_model_config(config_path)
    expected_cal = _expected(models, _calibration_tasks(load_tasks(data_root, "dev")), 1)
    calibration_root = Path(calibration_root or output_root / "calibration")
    if (calibration_root / "calibration").is_dir():
        calibration_root = calibration_root / "calibration"
    records = [json.loads(p.read_text()) for p in calibration_root.rglob("result.json")]
    readiness = preflight(config_path)
    if not readiness["ready"]:
        raise ValueError("Resolve preflight issues before freezing: " + "; ".join(readiness["issues"]))
    keyed = {(r["model_id"], r["task_id"], r["attempt"]): r for r in records}
    if len(keyed) != len(records):
        raise ValueError("Duplicate calibration attempts")
    costs_by_model = {m["id"]: [] for m in models}
    for item in expected_cal:
        record = keyed.get((item["model_id"], item["task_id"], item["attempt"]))
        if not record or record.get("status") not in {"completed", "invalid_submission", "protocol_violation", "timeout"}:
            raise ValueError("Complete all 24 calibration sessions before freezing")
        if record.get("audit", {}).get("status") not in {"approved", "rejected"}:
            raise ValueError("Review every calibration trace before freezing")
        current_model = next(m for m in models if m["id"] == record["model_id"])
        if record.get("suite_digest") != suite_digest(data_root) or record.get("model_config_digest") != _hash(current_model):
            raise ValueError("Calibration suite/config differs from the proposed frozen evaluation")
        if record.get("harness_digest") != harness_digest() or record.get("worker_image_digest") != readiness["docker"]["image_digest"]:
            raise ValueError("Calibration harness/worker differs from the proposed frozen evaluation")
        if record.get("cost_usd") is None or record.get("usage_unknown"):
            raise ValueError("Reconcile unknown calibration billing before freezing")
        costs_by_model[record["model_id"]].append(record["cost_usd"])
    # Use the most expensive observed family, with 30% headroom, rather than a favorable mean.
    primary_projection = sum(max(v) * 144 * 1.30 for v in costs_by_model.values())
    diagnostic_projection = sum(max(v) * 36 * 1.30 for v in costs_by_model.values())
    if primary_projection > 1800:
        raise BudgetExceeded("Calibration projects primary evaluation above its $1,800 allocation")
    official = load_tasks(data_root, "official")
    if len(official) != 48:
        raise ValueError("The official pilot must contain exactly 48 independent tasks")
    frozen = {"schema_version": "1.0", "benchmark_version": "0.1", "frozen_at": datetime.now(timezone.utc).isoformat(),
              "suite_digest": suite_digest(data_root), "config_digest": _hash(models), "models": models,
              "worker_image_digest": readiness["docker"]["image_digest"], "attempts": 3,
              "harness_digest": harness_digest(),
              "expected_matrix": _expected(models, official, 3), "expected_sessions": 576,
              "primary_projection_usd": primary_projection, "diagnostics_projection_usd": diagnostic_projection,
              "diagnostics_enabled": diagnostic_projection <= 200,
              "diagnostic_parent_selection": "first three sorted IDs in each Read/Repair family; excludes all-zero Write targets",
              "budget_ceiling_usd": 3000, "limits": asdict(official[0].limits),
              "calibration_records": [_hash(r) for r in sorted(records, key=lambda x: x["id"])]}
    _write(output_root / "freeze.json", frozen)
    return frozen


def _check_frozen(data_root, output_root, models, docker):
    _require_challenge_suite(Path(data_root))
    path = Path(output_root) / "freeze.json"
    if not path.exists():
        raise ValueError("Run and audit development calibration, then freeze before official scoring")
    frozen = json.loads(path.read_text())
    if frozen["suite_digest"] != suite_digest(data_root) or frozen["config_digest"] != _hash(models):
        raise ValueError("Suite/model configuration changed after freeze")
    if frozen["worker_image_digest"] != docker["image_digest"]:
        raise ValueError("Worker image changed after freeze")
    if frozen.get("harness_digest") != harness_digest():
        raise ValueError("Harness source changed after freeze")
    return frozen


def _sanitize_error(error: Exception, models: list[dict]) -> str:
    text = f"{type(error).__name__}: {error}"
    for model in models:
        secret = os.environ.get(model["api_key_env"])
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text[:2000]


async def _run(data_root: Path, output_root: Path, config_path: Path, phase: str,
               attempts: int, ledger_path: Path | None, model_ids: list[str] | None):
    output_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("INSPECT_TRACE_FILE", str((output_root / "inspect-trace.log").resolve()))
    os.environ.setdefault("INSPECT_CACHE_DIR", str((output_root / "inspect-cache").resolve()))
    from inspect_ai import eval_async
    from inspect_ai.model import GenerateConfig, get_model
    from .tasks import load_tasks
    from .inspect_task import build_inspect_task

    if model_ids and phase != "calibration":
        raise ValueError("Model filters are allowed only for development calibration")
    readiness = preflight(config_path, model_ids=model_ids)
    if not readiness["ready"]:
        raise ValueError("Paid evaluation preflight failed:\n" + "\n".join(readiness["issues"]))
    all_models = load_model_config(config_path)
    models = [m for m in all_models if not model_ids or m["id"] in model_ids]
    if phase == "calibration":
        tasks = _calibration_tasks(load_tasks(data_root, "dev"))
        attempts = 1
    elif phase == "primary":
        frozen = _check_frozen(data_root, output_root, models, readiness["docker"])
        if attempts != 3:
            raise ValueError("Frozen primary protocol requires three attempts")
        tasks = load_tasks(data_root, "official")
    elif phase == "diagnostics":
        frozen = _check_frozen(data_root, output_root, models, readiness["docker"])
        if not frozen["diagnostics_enabled"]:
            raise BudgetExceeded("Symmetry diagnostics were not affordable at the frozen calibration forecast")
        if attempts != 3:
            raise ValueError("Frozen diagnostics require three attempts")
        tasks = generate_diagnostics(data_root, output_root, frozen)
    else:
        raise ValueError("phase must be calibration, primary, or diagnostics")
    project_ledger = Path(config_path).resolve().parent.parent / ".wspbench-state" / "budget.sqlite3"
    ledger = BudgetLedger(ledger_path or project_ledger)
    if ledger.snapshot()["unsettled_reservations"]:
        raise BudgetExceeded("Reconcile outstanding reservations before starting further paid sessions")
    output = output_root / phase
    manifest_models = all_models if phase == "calibration" else models
    manifest = {"schema_version": "1.0", "phase": phase, "suite_digest": suite_digest(data_root),
                "config_digest": _hash(manifest_models), "expected_matrix": _expected(manifest_models, tasks, attempts),
                "expected_sessions": len(manifest_models) * len(tasks) * attempts, "attempts": attempts,
                "models": manifest_models, "worker_image_digest": readiness["docker"]["image_digest"]}
    manifest["harness_digest"] = harness_digest()
    if (output / "manifest.json").exists():
        old = json.loads((output / "manifest.json").read_text())
        comparable_keys = ["suite_digest", "worker_image_digest", "expected_matrix", "harness_digest"]
        if any(old.get(k) != manifest.get(k) for k in comparable_keys) or (phase != "calibration" and old != manifest):
            raise ValueError("Existing run manifest uses a different suite, worker, or harness. Preserve it as engineering history and use a new output directory for fresh calibration; the project-wide budget ledger remains shared.")
    _write(output / "manifest.json", manifest)
    records = []
    for model_config in models:
        model = get_model(model_config["model"], base_url=model_config.get("base_url"),
                          api_key=os.environ[model_config["api_key_env"]],
                          config=GenerateConfig(**provider_generate_options(model_config)), memoize=False,
                          **provider_client_options(model_config))
        for task in tasks:
            for attempt in range(1, attempts + 1):
                run_dir = output / model_config["id"] / task.id / f"attempt-{attempt}"
                result_path = run_dir / "result.json"
                if result_path.exists():
                    existing = json.loads(result_path.read_text())
                    if existing.get("suite_digest") != manifest["suite_digest"] or existing.get("model_config_digest") != _hash(model_config):
                        raise ValueError("Refusing to reuse an attempt from a different protocol")
                    records.append(existing)
                    continue
                run_id = f"{phase}:{model_config['id']}:{task.id}:{attempt}"
                reserved = maximum_session_cost(model_config, task.limits)
                reservation = ledger.reserve(phase, reserved, run_id)
                record = {"id": run_id, "task_id": task.id, "parent_id": task.parent_id,
                          "skill": task.skill, "family": task.family, "split": task.split,
                          "phase": phase, "attempt": attempt, "model_id": model_config["id"],
                          "provider": model_config["provider"], "model_config": model_config,
                          "suite_digest": manifest["suite_digest"], "config_digest": manifest["config_digest"],
                          "model_config_digest": _hash(model_config),
                          "harness_digest": harness_digest(),
                          "benchmark_version": "0.1", "harness_version": "0.1.0",
                          "worker_image_digest": readiness["docker"]["image_digest"],
                          "started_at": datetime.now(timezone.utc).isoformat(), "status": "running",
                          "cost_usd": None, "maximum_reserved_usd": reserved, "success": False}
                try:
                    inspect_task = build_inspect_task(task, run_dir, model_config, record)
                    logs = await eval_async(inspect_task, model=model, log_dir=str(run_dir / "inspect"),
                                            log_model_api=False, log_shared=False, retry_on_error=0,
                                            log_realtime=False, ctl_server=False, max_samples=1, max_tasks=1)
                    record["inspect_log"] = str(logs[0].location) if logs else None
                    if not logs or logs[0].status != "success":
                        record["usage_unknown"] = True
                        raise RuntimeError("Inspect did not complete successfully; billing requires reconciliation")
                    if record.get("status") not in {"completed", "invalid_submission", "protocol_violation", "timeout"}:
                        raise RuntimeError("Session did not reach a valid terminal outcome")
                    if record.get("usage_unknown"):
                        raise RuntimeError("Unknown provider billing; reservation retained pending reconciliation")
                    record["cost_usd"] = session_cost(model_config, record["usage"])
                    record["cost_method"] = "verified category-specific token-price estimate; reasoning included in output once"
                    ledger.settle(reservation, record["cost_usd"])
                except Exception as exc:
                    record["status"] = "infrastructure_failure"
                    record["error"] = _sanitize_error(exc, models)
                    record.setdefault("audit", {"status": "pending", "findings": []})
                    _write(result_path, record)
                    raise RuntimeError(f"Stopped after {run_id}: {record['error']}. Reservation remains in budget ledger.") from exc
                record["finished_at"] = datetime.now(timezone.utc).isoformat()
                artifacts = [p for p in run_dir.iterdir() if p.is_file() and p.name != "result.json"]
                record["artifact_hashes"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts}
                _write(result_path, record)
                records.append(record)
                _write(output / "progress.json", {"completed_sessions": len(records), "expected_sessions": manifest["expected_sessions"], "budget": ledger.snapshot()})
    return records


def run_evaluation(data_root: Path, output_root: Path, config_path: Path = Path("configs/models.json"),
                   phase: str = "calibration", attempts: int = 3, ledger_path: Path | None = None,
                   model_ids: list[str] | None = None) -> list[dict]:
    return asyncio.run(_run(Path(data_root), Path(output_root), Path(config_path), phase, attempts, ledger_path, model_ids))
