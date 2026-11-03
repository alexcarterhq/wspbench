"""Public reporting never includes private task payloads or unaudited model scores."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib

import numpy as np

from wspbench.domain import VERSION as __version__
from wspbench.domain import SANITY_FAMILIES

SKILLS = ("read", "repair", "write")
INFRASTRUCTURE = {"infrastructure_error", "infrastructure_failure", "provider_error"}


def clustered_interval(values: dict[str, list[float]], *, seed: int = 10, draws: int = 10000) -> tuple[float, float, float]:
    """Resample independent parent tasks, keeping attempts within their parent."""
    means = np.array([np.mean(v) for v in values.values()], dtype=float)
    if not len(means):
        raise ValueError("Cannot summarize an empty task group")
    rng = np.random.default_rng(seed)
    samples = rng.choice(means, size=(draws, len(means)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(means.mean()), float(low), float(high)


def _audit(record: dict) -> str:
    audit = record.get("audit", {})
    return audit.get("status", "pending") if isinstance(audit, dict) else "pending"


def _summarize(records: list[dict], *, include_metrics: bool = False) -> dict:
    parents: dict[str, list[float]] = defaultdict(list)
    failures: dict[str, int] = defaultdict(int)
    for record in records:
        parents[record["parent_id"]].append(float(bool(record.get("success")) and _audit(record) == "approved"))
        if not record.get("success"):
            failures[record.get("status", "failed")] += 1
    score, low, high = clustered_interval(parents)
    result = {
        "score": score, "ci_low": low, "ci_high": high,
        "n_tasks": len(parents), "n_attempts": len(records),
        "cost_usd": round(sum(float(r.get("cost_usd") or 0) for r in records), 6),
        "failures": dict(failures),
    }
    if include_metrics:
        metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for record in records:
            if _audit(record) != "approved":
                continue
            for name, value in record.get("metrics", {}).items():
                if isinstance(value, (int, float)) and np.isfinite(value):
                    metrics[name][record["parent_id"]].append(float(value))
        result["metrics"] = {name: {
            "mean": float(np.mean([np.mean(v) for v in parents.values()])),
            "n_tasks": len(parents), "n_observations": sum(map(len, parents.values())),
        } for name, parents in metrics.items()}
    return result


def load_records(root: Path) -> list[dict]:
    from wspbench.audit import trace_digest
    records: list[dict] = []
    ids: set[str] = set()
    for path in sorted(root.rglob("result.json")):
        record = json.loads(path.read_text())
        if record.get("id") in ids:
            raise ValueError(f"Duplicate run ID in {path}")
        ids.add(record.get("id"))
        trace_path = path.parent / "trace.json"
        try:
            valid = record.get("audit", {}).get("trace_sha256") == trace_digest(json.loads(trace_path.read_text()))
            for name, expected in record.get("artifact_hashes", {}).items():
                artifact = path.parent / name
                if Path(name).name != name or not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest() != expected:
                    valid = False
            record["_integrity_valid"] = valid
        except (OSError, ValueError):
            record["_integrity_valid"] = False
        records.append(record)
    return records


def build_report(records: list[dict], *, expected_tasks: int = 48, attempts: int = 3, frozen: dict | None = None) -> dict:
    """Only complete, reviewed primary matrices are eligible for official display."""
    report = {
        "schema_version": "1", "benchmark_version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "evaluation_pending", "models": [], "pending": [],
        "method": "Mean over attempts; 95% parent-task bootstrap confidence interval",
    }
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record.get("split") == "official" and record.get("phase") == "primary":
            groups[record["model_id"]].append(record)
    for model_id, group in sorted(groups.items()):
        reason = []
        if any(r.get("family") in SANITY_FAMILIES for r in group):
            reason.append("sanity suite excluded from headline scores")
        if frozen is None:
            reason.append("missing frozen protocol")
        else:
            expected = {(r["task_id"], r["attempt"]) for r in frozen["expected_matrix"] if r["model_id"] == model_id}
            actual = {(r["task_id"], r.get("attempt")) for r in group}
            if not expected or actual != expected or len(actual) != len(group):
                reason.append("evaluation differs from frozen matrix")
            configurations = {m["id"]: m for m in frozen["models"]}
            selected = configurations.get(model_id)
            model_digest = hashlib.sha256(json.dumps(selected, sort_keys=True, allow_nan=False).encode()).hexdigest()
            if any(r.get("suite_digest") != frozen["suite_digest"] or
                   r.get("config_digest") != frozen["config_digest"] or
                   r.get("model_config_digest") != model_digest or
                   r.get("worker_image_digest") != frozen["worker_image_digest"] for r in group):
                reason.append("mixed or changed protocol provenance")
        if any(not r.get("_integrity_valid", False) for r in group):
            reason.append("artifact or trace integrity unverified")
        keyed: dict[str, list[dict]] = defaultdict(list)
        for record in group:
            keyed[record["task_id"]].append(record)
        if len(keyed) != expected_tasks or any(len(v) != attempts for v in keyed.values()):
            reason.append("incomplete evaluation matrix")
        if any(len({r.get('attempt') for r in v}) != attempts for v in keyed.values()):
            reason.append("missing or duplicate attempt indices")
        if any(r.get("status") in INFRASTRUCTURE for r in group):
            reason.append("unresolved infrastructure failures")
        if any(r.get("cost_usd") is None or r.get("usage_unknown") for r in group):
            reason.append("unresolved billing")
        if any(_audit(r) not in {"approved", "rejected"} for r in group):
            reason.append("trace review pending")
        skill_counts = {s: len({r['parent_id'] for r in group if r['skill'] == s}) for s in SKILLS}
        if any(n != expected_tasks // 3 for n in skill_counts.values()):
            reason.append("unbalanced skill coverage")
        if reason:
            report["pending"].append({"model_id": model_id, "n_attempts": len(group), "reasons": sorted(set(reason))})
            continue
        skills = {}
        for skill in SKILLS:
            selected = [r for r in group if r["skill"] == skill]
            summary = _summarize(selected)
            summary["families"] = {
                family: _summarize([r for r in selected if r["family"] == family], include_metrics=True)
                for family in sorted({r["family"] for r in selected})
            }
            skills[skill] = summary
        first = group[0]
        report["models"].append({
            "model_id": model_id, "provider": first.get("provider"),
            "configuration": first.get("model_config", {}), "audit_status": "verified",
            "skills": skills,
        })
    if report["models"]:
        report["status"] = "results_available"
    return report


def write_report(runs: Path, destination: Path, *, validation: Path | None = None) -> dict:
    freeze_path = runs / "freeze.json"
    frozen = json.loads(freeze_path.read_text()) if freeze_path.exists() else None
    report = build_report(load_records(runs), frozen=frozen)
    if validation is not None and validation.exists():
        data = json.loads(validation.read_text())
        # Explicit allowlist: never serialize task validation internals/answers.
        report["local_validation"] = {
            key: data[key] for key in ("status", "n_tasks", "n_passed", "generated_at", "report_url") if key in data
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def calibration_report(runs: Path, destination: Path) -> dict:
    """Development-only evidence; this function never creates official rankings."""
    records = [r for r in load_records(runs) if r.get("phase") == "calibration" and r.get("split") == "dev"]
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record["model_id"]].append(record)
    models = []
    for model_id, group in sorted(groups.items()):
        reviewed = all(_audit(r) in {"approved", "rejected"} and r.get("_integrity_valid") for r in group)
        models.append({
            "model_id": model_id, "phase": "development_calibration", "n_sessions": len(group),
            "model": group[0].get("model_config", {}).get("model"),
            "freeze_eligible": all(r.get("freeze_eligible", True) for r in group),
            "execution_harness_digests": sorted({r.get("harness_digest", "unrecorded") for r in group}),
            "accounting_corrections": sum(len(r.get("billing_corrections", [])) for r in group),
            "n_reviewed": sum(_audit(r) in {"approved", "rejected"} and bool(r.get("_integrity_valid")) for r in group),
            "estimated_cost_usd": round(sum(float(r.get("cost_usd") or 0) for r in group), 6),
            "unknown_billing_sessions": sum(bool(r.get("usage_unknown")) or r.get("cost_usd") is None for r in group),
            "skills": {skill: {
                **_summarize([r for r in group if r["skill"] == skill]),
                "families": {family: _summarize([r for r in group if r["family"] == family], include_metrics=True)
                             for family in sorted({r["family"] for r in group if r["skill"] == skill})},
            } for skill in SKILLS if reviewed and any(r["skill"] == skill for r in group)},
            "sessions": [{"task_id": r["task_id"], "family": r["family"], "status": r["status"],
                          "success": bool(r.get("success")) if _audit(r) in {"approved", "rejected"} else None,
                          "audit_status": _audit(r), "cost_usd": r.get("cost_usd")} for r in group],
        })
    report = {"schema_version": "1", "benchmark_version": __version__, "phase": "development_calibration",
              "notice": "Development examples only; not an official benchmark score or model ranking.",
              "models": models, "generated_at": datetime.now(timezone.utc).isoformat()}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
