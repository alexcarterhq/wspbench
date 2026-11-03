"""Audit explicit access and counters; do not infer a model's internal cognition."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .serialization import canonical_json


def trace_digest(trace: list[dict]) -> str:
    return hashlib.sha256(canonical_json(trace).encode()).hexdigest()


def audit_trace(trace, condition, limits, counters) -> dict:
    findings = []
    for index, event in enumerate(trace):
        tool = event.get("tool")
        if tool == "denied":
            findings.append({"event": index, "finding": "rejected_access_or_budget_request",
                             "reason": event.get("reason")})
        if condition == "A" and tool in {"analysis", "probe"}:
            findings.append({"event": index, "finding": "unaided_tool_access"})
        if condition == "B" and tool == "probe":
            findings.append({"event": index, "finding": "weight_only_target_execution"})
        if tool not in {"analysis", "probe", "commit", "submit", "model_turn", "denied"}:
            findings.append({"event": index, "finding": "unknown_tool"})
    checks = {"analysis_calls": limits.analysis_calls, "example_evaluations": limits.example_evaluations,
              "activation_scalars": limits.activation_scalars, "candidate_patches": limits.candidate_patches,
              "episodes": limits.episodes}
    for key, cap in checks.items():
        if counters.get(key, 0) > cap:
            findings.append({"finding": f"budget_exceeded:{key}"})
    return {"status": "rejected" if findings else "pending", "findings": findings,
            "trace_sha256": trace_digest(trace),
            "claim": "Access/counter checks do not prove absence of mental simulation or semantic forward emulation."}


def review_run(run_dir: Path, reviewer: str, decision: str, note: str) -> dict:
    if decision not in {"approved", "rejected"} or not reviewer.strip() or not note.strip():
        raise ValueError("Review requires an identity, decision and substantive evidence note")
    path = Path(run_dir) / "result.json"
    record = json.loads(path.read_text())
    trace = json.loads((Path(run_dir) / "trace.json").read_text())
    if record["audit"]["trace_sha256"] != trace_digest(trace):
        raise ValueError("Trace changed since execution")
    if record["audit"]["status"] == "rejected" and decision == "approved":
        raise ValueError("An access violation cannot be silently overridden")
    record["audit"].update(status=decision, reviewer=reviewer, note=note)
    if decision == "rejected":
        record.update(success=False, status="protocol_violation")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    return record
