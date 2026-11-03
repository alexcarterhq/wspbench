"""Allowlisted, development-only exports from the private operator workspace.

This operator utility does not make provider calls. It never copies a private
manifest: public scoring fixtures are reconstructed from named numeric arrays.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np

from ..budget import normalize_usage, session_cost
from ..v2 import VERSION
from ..v2.access import AccessSession
from ..v2.audit import trace_digest
from ..v2.inspect_task import CONDITIONS, SYSTEM, parse_submission
from ..v2.runner import harness_digest, suite_digest
from ..v2.scoring import score
from ..v2.serialization import canonical_json, load_weights, serialization_digest, validate_weights, weights_digest
from ..v2.tasks import load_tasks

DEVELOPMENT_SCHEMA = "wspbench-public-development-1"
PILOT_SCHEMA = "wspbench-public-pilot-1"
DATA_DIRECTORY = Path("data/development-v0.2.1")
PILOT_DIRECTORY = Path("results/pilot-v0.2.1")
_ARRAYS = {"inputs", "expected", "practice", "final"}
_FORBIDDEN = {"seed", "generation_seed", "feature_seed", "api_key", "private_path",
              "authorization", "response_id", "request_id", "signature", "encrypted_content",
              "reasoning", "incident_path", "reservation_id", "quarantined_holds"}
_RECORD_FIELDS = {
    "schema_version", "benchmark_version", "suite_tier", "headline_eligible", "task_id",
    "parent_problem_id", "mechanism_family_id", "family", "skill", "split", "target_stratum",
    "capacity_tier", "actor_id", "actor_name", "actor_type", "condition", "weight_view", "attempt",
    "protocol_digest", "target_digest", "question_digest", "cost_usd", "prompt_information", "hint_level",
    "id", "model_id", "provider", "model_config", "model_config_digest", "suite_digest", "harness_digest",
    "phase", "status", "prompt_digest", "usage", "serialization_digest", "turns", "canonical_prompt_digest",
    "submission", "submitted", "counters", "snapshots", "snapshot_actual_observations", "snapshot_unit",
    "actual_observations", "cpu_seconds", "wall_seconds", "observation_usage", "success", "metrics",
    "score_status", "score_reason", "cost_method",
}
_MODEL_FIELDS = {"id", "provider", "model", "api_key_env", "snapshot", "snapshot_kind", "verified_at",
                 "generate_config", "model_args", "pricing"}
_TOKEN_FIELDS = {"input_tokens", "input_tokens_cache_read", "input_tokens_cache_write", "output_tokens",
                 "reasoning_tokens", "total_tokens"}


def _read(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_public(value)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _assert_public(value):
    """Reject structural private fields, rather than redacting unknown values."""
    if isinstance(value, dict):
        if any(key in _FORBIDDEN or key.endswith("_seed") for key in value):
            raise ValueError("Private or provider-internal field in proposed public export")
        for child in value.values():
            _assert_public(child)
    elif isinstance(value, list):
        for child in value:
            _assert_public(child)
    elif isinstance(value, str):
        if value.startswith(("/Users/", "/Volumes/", "/private/", "file://")):
            raise ValueError("Local path in proposed public export")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite value in public export")


def _public_model(model):
    if not isinstance(model, dict) or set(model) - _MODEL_FIELDS:
        raise ValueError("Unexpected model configuration fields")
    # Keep the environment-variable NAME so the original model-config digest
    # remains checkable; no credential value is read by this module.
    _assert_public(model)
    return model


def _public_trace(trace, submission):
    if [event.get("tool") for event in trace] != ["model_turn", "commit", "submit"]:
        raise ValueError("Only the reviewed single-turn, no-tool pilot can be exported")
    turn, commit, submit = trace
    if turn.get("turn") != 1 or commit.get("submission") != submission:
        raise ValueError("Trace turn or saved submission mismatch")
    if commit.get("observation_count") != 0 or submit.get("observation_count") != 0:
        raise ValueError("Condition-A trace contains observations")
    output = turn["output"]
    if parse_submission(output["completion"]) != submission:
        raise ValueError("Provider completion differs from saved submission")
    safe_output = {key: output[key] for key in ("model", "completion", "input_context_tokens", "time") if key in output}
    safe_output["usage"] = {key: output.get("usage", {})[key] for key in _TOKEN_FIELDS if key in output.get("usage", {})}
    safe_turn = {"tool": "model_turn", "turn": 1,
                 "usage": {key: turn["usage"][key] for key in _TOKEN_FIELDS if key in turn["usage"]},
                 "output": safe_output}
    result = [safe_turn, {"tool": "commit", "submission": submission, "observation_count": 0},
              {"tool": "submit", "observation_count": 0}]
    _assert_public(result)
    return result


def _file_catalog(root):
    return {str(path.relative_to(root)): _sha(path) for path in sorted(Path(root).rglob("*"))
            if path.is_file() and path.name != "manifest.json" and not path.name.startswith("._")}


def _export_task(task, destination):
    if task.split != "dev":
        raise ValueError("Only public development fixtures may be exported")
    destination = Path(destination)
    destination.mkdir(parents=True)
    public = task.to_public_dict()
    _assert_public(public)
    _write(destination / "task.json", public)
    artifacts = {"task.json": {"sha256": _sha(destination / "task.json"), "source_sha256": _sha(task.directory / "task.json")}}
    for name in ("target.npz", "before.npz"):
        source = task.directory / name
        if name == "before.npz" and task.family != "edit_effects":
            continue
        weights = load_weights(source)
        validate_weights(task.architecture, weights)
        shutil.copyfile(source, destination / name)
        artifacts[name] = {"sha256": _sha(destination / name), "source_sha256": _sha(source)}
    source = task.directory / "evaluation.npz"
    with np.load(source, allow_pickle=False) as bank:
        if set(bank.files) != _ARRAYS:
            raise ValueError("Scoring bank contains unrecognized arrays")
        arrays = {key: bank[key].copy() for key in sorted(_ARRAYS)}
    if any(value.dtype.hasobject or value.dtype.kind not in "biuf" or not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Scoring fixtures must contain finite numeric arrays")
    np.savez_compressed(destination / "evaluation.npz", **arrays)
    artifacts["evaluation.npz"] = {"sha256": _sha(destination / "evaluation.npz"), "source_sha256": _sha(source)}
    reference = {"reference_submission": _read(task.private_path)["reference_submission"]}
    _write(destination / "reference.json", reference)
    # Do not publish a checksum oracle for the original seed-bearing private
    # object. Bind the explicitly released reference projection instead.
    artifacts["reference.json"] = {"sha256": _sha(destination / "reference.json"),
                                   "source_sha256": serialization_digest(reference),
                                   "source_hash_method": "canonical_allowlisted_reference_projection"}
    if not score(task, reference["reference_submission"]).success:
        raise ValueError("Public development reference no longer passes")
    return {"task_id": task.id, "directory": f"dev/{task.id}", "family": task.family, "split": "dev",
            "artifacts": artifacts}


def _sanitize_record(record, trace, source_path, safe_hashes, original_hashes):
    public = {key: record[key] for key in _RECORD_FIELDS if key in record}
    _public_model(public["model_config"])
    public["audit"] = {
        "status": "approved", "reviewer": record["audit"]["reviewer"], "findings": [],
        "trace_sha256": trace_digest(trace), "original_trace_sha256": record["audit"]["trace_sha256"],
        "claim": record["audit"].get("claim"),
        "note": "Original recorded trace independently reviewed before export. Public trace omits provider identifiers and encrypted reasoning; scoring, usage and access events are retained. Offline replay checks these public artifacts, not provider-side execution.",
    }
    public["artifact_hashes"] = safe_hashes
    public["source_artifact_hashes"] = original_hashes
    public["source_result_sha256"] = _sha(source_path)
    _assert_public(public)
    return public


def _export_runs(data_root, run_root, destination, tasks):
    source = _read(run_root / "calibration/manifest.json")
    matrix = source["expected_matrix"]
    if (source.get("phase") != "calibration" or source.get("conditions") != ["A"] or source.get("views") != ["full"]
            or source.get("attempts") != 1 or source.get("expected_sessions") != 18 or len(matrix) != 18
            or len(source.get("models", [])) != 3):
        raise ValueError("Not the complete reviewed eighteen-session development pilot")
    for model in source["models"]:
        _public_model(model)
    ids = {item["task_id"] for item in matrix}
    if len(ids) != 6 or not ids <= set(tasks):
        raise ValueError("Pilot includes a non-public or unrecognized development target")
    for item in matrix:
        if item["condition"] != "A" or item["weight_view"] != "full" or item["attempt"] != 1:
            raise ValueError("Unexpected pilot access or attempt")
    expected = {(item["model_id"], item["task_id"], item["condition"], item["weight_view"], item["attempt"]): item for item in matrix}
    if len(expected) != 18:
        raise ValueError("Duplicate pilot matrix entry")
    result_paths = sorted((run_root / "calibration").glob("*/result.json"))
    if len(result_paths) != 18:
        raise ValueError("Pilot result count differs from its declared matrix")
    records, observed = [], set()
    for path in result_paths:
        record = _read(path)
        key = tuple(record.get(name) for name in ("model_id", "task_id", "condition", "weight_view", "attempt"))
        if key not in expected or key in observed:
            raise ValueError("Unexpected or duplicate pilot result")
        observed.add(key)
        task = tasks[record["task_id"]]
        if record.get("split") != "dev" or record.get("status") != "completed" or record.get("usage_unknown"):
            raise ValueError("Only finalized, known-usage development runs can be exported")
        if record.get("audit", {}).get("status") != "approved" or record["audit"].get("findings"):
            raise ValueError("Pilot trace lacks an independent clean audit")
        for name in ("protocol_digest", "suite_digest", "harness_digest"):
            if record.get(name) != source[name]:
                raise ValueError("Pilot record provenance mismatch")
        if record.get("model_config_digest") != serialization_digest(record["model_config"]) or record["model_config_digest"] != expected[key]["model_config_digest"]:
            raise ValueError("Model configuration hash mismatch")
        if record["target_digest"] != weights_digest(load_weights(task.weights_path)) or record["target_digest"] != expected[key]["target_digest"]:
            raise ValueError("Pilot target checkpoint mismatch")
        originals = {}
        for name in ("prompt.json", "trace.json", "submission.json"):
            originals[name] = _sha(path.parent / name)
            if record.get("artifact_hashes", {}).get(name) != originals[name]:
                raise ValueError("Original session artifact changed after execution")
        prompt, raw_trace, submission = (_read(path.parent / name) for name in ("prompt.json", "trace.json", "submission.json"))
        if trace_digest(raw_trace) != record["audit"]["trace_sha256"]:
            raise ValueError("Original trace changed after independent audit")
        payload = AccessSession(task, "A", "full").public_payload()
        if prompt != {"system": SYSTEM, "access": CONDITIONS["A"], "payload": payload}:
            raise ValueError("Saved prompt differs from the trusted development payload")
        if (record["prompt_digest"] != serialization_digest(prompt) or record["canonical_prompt_digest"] != serialization_digest(payload)
                or record["question_digest"] != serialization_digest(payload["task"]) or record["serialization_digest"] != payload["serialization_digest"]):
            raise ValueError("Pilot prompt/checkpoint serialization hash mismatch")
        if submission != record["submission"]:
            raise ValueError("Saved submission differs from the final result")
        graded = score(task, submission)
        if graded.success != record["success"] or graded.metrics != record["metrics"] or graded.status != record["score_status"]:
            raise ValueError("Trusted scoring no longer reproduces a pilot result")
        if not math.isclose(session_cost(record["model_config"], record["usage"]), record["cost_usd"], abs_tol=1e-12):
            raise ValueError("Pilot usage does not reproduce its estimated cost")
        trace = _public_trace(raw_trace, submission)
        if normalize_usage(trace[0]["usage"]) != normalize_usage(record["usage"]):
            raise ValueError("Event and result token usage differ")
        session_dir = destination / "sessions" / record["id"]
        for name, value in (("prompt.json", prompt), ("trace.json", trace), ("submission.json", submission)):
            _write(session_dir / name, value)
        hashes = {name: _sha(session_dir / name) for name in originals}
        public = _sanitize_record(record, trace, path, hashes, originals)
        _write(session_dir / "result.json", public)
        records.append(public)
    _write(destination / "records.json", records)
    raw_controls = _read(data_root / "baseline-records.private.json")
    write_ids = {task_id for task_id in ids if tasks[task_id].skill == "write"}
    controls = []
    for record in raw_controls:
        if record.get("task_id") not in write_ids or record.get("actor_id") != "metadata_prior" or record.get("condition") != "B" or record.get("weight_view") != "full":
            continue
        task = tasks[record["task_id"]]
        if record.get("split") != "dev" or record.get("target_digest") != weights_digest(load_weights(task.weights_path)) or record.get("question_digest") != serialization_digest({"instruction": task.instruction, "public": task.public}):
            raise ValueError("No-edit baseline is not bound to the selected public target")
        graded = score(task, {"patch": []})
        if graded.metrics != record["metrics"] or graded.success != record["success"]:
            raise ValueError("No-edit baseline score changed")
        allowed = {"actor_id", "actor_name", "actor_type", "condition", "weight_view", "task_id", "parent_problem_id", "mechanism_family_id", "family", "skill", "split", "target_stratum", "capacity_tier", "target_digest", "question_digest", "success", "metrics", "status"}
        controls.append({**{key: record[key] for key in allowed if key in record}, "actor_kind": "static_zero_start_control", "same_access_model_score": False, "submission": {"patch": []}})
    if len(controls) != len(write_ids) or len({record["task_id"] for record in controls}) != len(write_ids):
        raise ValueError("Missing or duplicate public Write controls")
    _write(destination / "write-controls.json", controls)
    manifest = {**source, "public_schema": PILOT_SCHEMA, "dataset_revision": "0.2.1", "public_development_path": "../../data/development-v0.2.1",
                "retired_public_development_task_ids": sorted(ids), "source_manifest_sha256": _sha(run_root / "calibration/manifest.json"),
                "sanitization": "Allowlisted development artifacts only; no provider IDs, account data, local paths, encrypted reasoning, seeds or operational budget records.",
                "replay_scope": "CPU regrading of saved submissions; no new model inference", "files": _file_catalog(destination)}
    _write(destination / "manifest.json", manifest)
    return records, manifest


def _cpu_evidence(run_root, destination, source_suite, source_harness):
    validation = _read(run_root / "cpu-validation.json")
    baseline = _read(run_root / "baselines.json")
    reference = _read(run_root / "cpu-reference/summary.json")
    reference_manifest = _read(run_root / "cpu-reference/manifest.json")
    smoke = _read(run_root / "inspect-smoke/summary.json")
    for value in (validation, baseline, reference_manifest):
        if value.get("suite_digest") != source_suite or value.get("harness_digest") != source_harness:
            raise ValueError("CPU evidence uses a different source suite or harness")
    if validation.get("status") != "passed" or reference.get("status") != "passed" or smoke.get("status") != "passed":
        raise ValueError("CPU validation/reference/mock evidence is not passing")
    table_fields = {"actor", "actor_type", "condition", "weight_view", "family", "target_stratum", "capacity_tier", "split", "score", "ci_low", "ci_high", "n_mechanisms", "n_parents", "n_attempts", "degenerate_bootstrap", "bounded_mean_95_interval", "interval_note", "metrics", "cpu_seconds", "estimated_cost_usd", "infrastructure_failures", "audit_status_counts", "example_evaluations", "activation_scalars", "episodes", "generated_tokens"}
    evidence = {"schema": "wspbench-public-cpu-evidence-1", "suite_digest": source_suite, "harness_digest": source_harness,
        "notice": "Sanitized aggregate evidence from the private qualification suite. Public replay separately checks the released development fixtures; aggregate private-suite counts do not expose held-out tasks or constitute provider scores.",
        "validation": {key: validation[key] for key in ("status", "valid", "tasks", "families", "strata", "checks", "cpu_only", "no_paid_calls")},
        "references": {key: reference[key] for key in ("status", "actor", "sessions", "successes", "no_paid_calls")},
        "inspect_mock": {key: smoke[key] for key in ("status", "sessions", "mock", "benchmark_result", "cost_usd", "notice")},
        "baselines": {"n_records": baseline["n_records"], "paid_calls": baseline["paid_calls"], "evaluation_scope": baseline["evaluation_scope"], "protocol_digest": baseline["protocol_digest"], "coverage_manifest_digest": baseline["coverage_manifest_digest"], "tables": [{key: row[key] for key in table_fields if key in row} for row in baseline["tables"]]},
        "source_artifact_sha256": {name: _sha(run_root / name) for name in ("cpu-validation.json", "baselines.json", "cpu-reference/summary.json", "cpu-reference/manifest.json", "inspect-smoke/summary.json")}}
    _write(destination / "cpu-evidence.json", evidence)


def export_release(data_root, run_root, output_root):
    """Export the reviewed pilot and all public development fixtures, fail closed."""
    data_root, run_root, output_root = map(Path, (data_root, run_root, output_root))
    source = _read(run_root / "calibration/manifest.json")
    if harness_digest() != source.get("harness_digest") or suite_digest(data_root) != source.get("suite_digest"):
        raise ValueError("Preserve the evaluated runtime/data snapshot before exporting")
    tasks = load_tasks(data_root, "dev")
    if len(tasks) != 28:
        raise ValueError("This release must contain exactly the reviewed 28 public development fixtures")
    for relative in (DATA_DIRECTORY, PILOT_DIRECTORY):
        if (output_root / relative).exists():
            raise ValueError("Refuse to overwrite an existing public release")
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wspbench-public-export-") as temporary:
        staging = Path(temporary)
        data_output, pilot_output = staging / DATA_DIRECTORY, staging / PILOT_DIRECTORY
        entries = [_export_task(task, data_output / "dev" / task.id) for task in tasks]
        _write(data_output / "manifest.json", {"schema_version": DEVELOPMENT_SCHEMA, "dataset_revision": "0.2.1", "harness_version": VERSION,
            "source_suite_digest": source["suite_digest"], "source_harness_digest": source["harness_digest"],
            "tasks": entries, "n_tasks": len(entries), "families": dict(Counter(task.family for task in tasks)),
            "split": "public_development", "exposure_date": "2026-11-03", "license": "CC-BY-4.0", "attribution": "WSPBench contributors",
            "retirement": "All released fixtures and scoring banks are public development material, excluded from future held-out evaluation. Shared suite and generation seeds remain private.", "files": _file_catalog(data_output)})
        records, manifest = _export_runs(data_root, run_root, pilot_output, {task.id: task for task in tasks})
        _cpu_evidence(run_root, pilot_output, source["suite_digest"], source["harness_digest"])
        manifest["files"] = _file_catalog(pilot_output)
        _write(pilot_output / "manifest.json", manifest)
        for relative in (DATA_DIRECTORY, PILOT_DIRECTORY):
            destination = output_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(staging / relative, destination)
    return {"development_tasks": len(tasks), "sessions": len(records), "known_cost_estimate_usd": sum(row["cost_usd"] for row in records),
            "data_path": str(DATA_DIRECTORY), "pilot_path": str(PILOT_DIRECTORY), "provider_calls": 0, "heldout_exported": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(".wspbench-data/challenge-v0.2.1"))
    parser.add_argument("--runs", type=Path, default=Path(".wspbench-runs/gpt-comparison-v0.2.1"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(export_release(args.data, args.runs, args.output), indent=2))


if __name__ == "__main__":
    main()
