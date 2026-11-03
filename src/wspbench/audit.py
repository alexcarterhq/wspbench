"""Conservative trace screening; external review is still required."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

PROHIBITED = {
    "backward", "grad", "autograd", "optimizer", "optim", "fit", "train",
    "minimize", "least_squares", "lstsq", "curve_fit", "differential_evolution",
    "subprocess", "socket", "requests", "urllib", "os", "sys", "ctypes",
    "pickle", "eval", "exec", "compile", "__import__", "open", "globals",
    "locals", "getattr", "setattr", "delattr", "breakpoint", "input",
}


def screen_code(code: str) -> list[str]:
    """Flag known violations; an empty result does not prove WSP compliance."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ["invalid_python"]
    findings = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id.lower() in PROHIBITED:
            findings.add(f"prohibited_name:{node.id}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr.lower() in PROHIBITED:
                findings.add(f"prohibited_attribute:{node.attr}")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [a.name.split(".")[0] for a in node.names] if isinstance(node, ast.Import) else [(node.module or "").split(".")[0]]
            for name in modules:
                if name not in {"numpy", "math"}:
                    findings.add(f"prohibited_import:{name}")
        if isinstance(node, (ast.For, ast.While, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
            # Analytic loops are permitted, but must be reviewed for hidden fitting/search.
            findings.add("review:iteration")
    return sorted(findings)


def trace_digest(trace: list[dict]) -> str:
    return hashlib.sha256(json.dumps(trace, sort_keys=True, allow_nan=False).encode()).hexdigest()


def audit_trace(trace: list[dict]) -> dict:
    findings = []
    for index, event in enumerate(trace):
        if event.get("tool") == "python":
            findings.extend({"event": index, "finding": f} for f in screen_code(event.get("code", "")))
    blocked = any(not f["finding"].startswith("review:") for f in findings)
    return {"status": "rejected" if blocked else "pending", "findings": findings,
            "trace_sha256": trace_digest(trace), "claim": "Automated screening is not proof of no training or no forward execution."}


def review_run(run_dir: Path, reviewer: str, decision: str, note: str) -> dict:
    if decision not in {"approved", "rejected"} or not reviewer.strip() or not note.strip():
        raise ValueError("Review requires reviewer identity, approved/rejected decision, and a substantive note")
    path = Path(run_dir) / "result.json"
    record = json.loads(path.read_text())
    trace = json.loads((Path(run_dir) / "trace.json").read_text())
    if record["audit"]["trace_sha256"] != trace_digest(trace):
        raise ValueError("Trace changed after scoring; review refused")
    if record["audit"]["status"] == "rejected" and decision == "approved":
        raise ValueError("Automated violation cannot be overridden silently; correct the protocol or rerun")
    record["audit"].update(status=decision, reviewer=reviewer, note=note)
    if decision == "rejected":
        record.update(success=False, status="protocol_violation")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)
    return record
