"""Regrade released development answers without credentials or model inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

import numpy as np

from ..budget import normalize_usage, session_cost
from ..v2.access import AccessSession
from ..v2.contracts import Task
from ..v2.inspect_task import CONDITIONS, SYSTEM, parse_submission
from ..v2.runner import harness_digest
from ..v2.scoring import reference_submission, score
from ..v2.serialization import load_weights, serialization_digest, validate_weights, weights_digest
from .report import TOKEN_FIELDS, _csv, _markdown, build_comparison

DEFAULT_DATA = Path("data/development-v0.2.1")
DEFAULT_RESULTS = Path("results/pilot-v0.2.1")
COUNTERS = ("analysis_calls", "example_evaluations", "activation_scalars", "episodes", "steps",
            "generated_tokens", "candidate_patches")


class PublicTask(Task):
    """Use the released, minimal answer file instead of a private task manifest."""

    @property
    def private_path(self):
        return self.directory / "reference.json"


def _read(path):
    value = json.loads(Path(path).read_text())
    # Reject NaN/Infinity as well as other non-JSON state before using metadata.
    serialization_digest(value)
    return value


def _file(root, relative):
    if not isinstance(relative, str):
        raise ValueError("Artifact path must be a relative string")
    parts = PurePosixPath(relative)
    if parts.is_absolute() or not parts.parts or any(p in {".", ".."} for p in parts.parts) or "\\" in relative:
        raise ValueError("Artifact path leaves the public bundle")
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Artifact path leaves the public bundle")
    return path


def verify_manifest(root):
    """Verify the complete exported file catalog before grading anything."""
    root = Path(root)
    manifest = _read(root / "manifest.json")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Public manifest has no file catalog")
    actual = {str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*")
              if p.is_file() and p != root / "manifest.json" and not p.name.startswith("._")}
    if actual != set(files):
        raise ValueError("Public bundle files differ from the manifest catalog")
    for name, expected in files.items():
        path = _file(root, name)
        if not isinstance(expected, str) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Artifact hash mismatch: {name}")
    return manifest


def load_public_tasks(data_root):
    data_root = Path(data_root)
    manifest = verify_manifest(data_root)
    if manifest.get("schema_version") != "wspbench-public-development-1":
        raise ValueError("Unsupported public development schema")
    tasks = {}
    for row in manifest.get("tasks", []):
        if row.get("split") != "dev" or row.get("task_id") in tasks:
            raise ValueError("Only unique public development fixtures may be replayed")
        directory = _file(data_root, row["directory"])
        task = PublicTask.from_dict(_read(directory / "task.json"), directory)
        if task.id != row["task_id"] or task.split != "dev" or task.family != row["family"]:
            raise ValueError("Public fixture metadata does not match its catalog")
        allowed = {"task.json", "target.npz", "evaluation.npz", "reference.json"}
        if task.family == "edit_effects":
            allowed.add("before.npz")
        if set(row.get("artifacts", {})) != allowed:
            raise ValueError("Unexpected or missing public scoring artifact")
        for name, artifact in row["artifacts"].items():
            relative = str((directory / name).relative_to(data_root)).replace("\\", "/")
            if artifact.get("sha256") != manifest["files"].get(relative):
                raise ValueError("Fixture hash does not match the export catalog")
        reference = _read(task.private_path)
        if set(reference) != {"reference_submission"}:
            raise ValueError("Public references must contain only the released answer")
        validate_weights(task.architecture, load_weights(task.weights_path))
        with np.load(directory / "evaluation.npz", allow_pickle=False) as bank:
            if set(bank.files) != {"inputs", "expected", "practice", "final"}:
                raise ValueError("Evaluation fixture contains non-allowlisted arrays")
            x, expected, practice, final = (bank[k] for k in ("inputs", "expected", "practice", "final"))
            if any(a.dtype.kind not in "fiu" or not np.isfinite(a).all() for a in (x, expected, practice, final)):
                raise ValueError("Evaluation fixture must be finite numeric data")
            if len(x) != len(expected) or final.dtype.kind not in "iu" or practice.dtype.kind not in "iu":
                raise ValueError("Malformed evaluation fixture")
            indices = np.concatenate((practice, final))
            if (practice.ndim != 1 or final.ndim != 1 or len(final) == 0 or len(indices) != len(x)
                    or set(indices.tolist()) != set(range(len(x)))):
                raise ValueError("Practice and final banks must partition the public domain")
        tasks[task.id] = task
    if not tasks:
        raise ValueError("No released development fixtures")
    if manifest.get("n_tasks", len(tasks)) != len(tasks):
        raise ValueError("Public fixture count differs from its release manifest")
    return manifest, tasks


def _key(row):
    return tuple(row[k] for k in ("model_id", "task_id", "condition", "weight_view", "attempt"))


def _same(left, right, label):
    if serialization_digest(left) != serialization_digest(right):
        raise ValueError(label)


def _score_matches(result, record):
    if (bool(record.get("success")) != result.success or record.get("score_status") != result.status
            or record.get("score_reason") != result.reason or set(record.get("metrics", {})) != set(result.metrics)):
        raise ValueError("Regraded answer differs from the recorded score")
    for key, value in result.metrics.items():
        actual = record["metrics"][key]
        if type(actual) not in (float, int) or not math.isclose(actual, value, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Regraded metric differs: {key}")


def _validate_session(root, record, model, task):
    if (record.get("condition") != "A" or record.get("weight_view") != "full"
            or record.get("attempt") != 1 or record.get("status") != "completed"
            or record.get("actor_type") != "model" or record.get("usage_unknown")
            or record.get("audit", {}).get("status") != "approved" or record.get("turns") != 1):
        raise ValueError("The public pilot requires one approved, completed A/full model response")
    for name in ("task_id", "parent_problem_id", "mechanism_family_id", "family", "skill", "split"):
        expected = task.id if name == "task_id" else getattr(task, name)
        if record.get(name) != expected:
            raise ValueError(f"Recorded task metadata differs from the public fixture: {name}")
    directory = _file(root, f"sessions/{record['id']}")
    _same(_read(directory / "result.json"), record, "Session result differs from records.json")
    submission = _read(directory / "submission.json")
    _same(submission, record["submission"], "Saved submission differs from the result")
    prompt = _read(directory / "prompt.json")
    payload = AccessSession(task, "A").public_payload()
    _same(prompt, {"system": SYSTEM, "access": CONDITIONS["A"], "payload": payload},
          "Saved prompt differs from the released task and frozen protocol")
    required_digests = {"target_digest": weights_digest(load_weights(task.weights_path)),
                        "question_digest": serialization_digest(payload["task"]),
                        "serialization_digest": payload["serialization_digest"],
                        "canonical_prompt_digest": serialization_digest(payload),
                        "prompt_digest": serialization_digest(prompt),
                        "model_config_digest": serialization_digest(model)}
    for name, wanted in required_digests.items():
        if record.get(name) != wanted:
            raise ValueError(f"Session provenance differs: {name}")
    _same(record["model_config"], model, "Session model configuration differs from the manifest")
    trace = _read(directory / "trace.json")
    if (not isinstance(trace, list) or [e.get("tool") for e in trace] != ["model_turn", "commit", "submit"]
            or trace[0].get("turn") != 1 or any(e.get("observation_count", 0) != 0 for e in trace)):
        raise ValueError("Released A trace contains unexpected tools, turns or observations")
    if record["audit"].get("trace_sha256") != serialization_digest(trace):
        raise ValueError("Trace differs from its public audit digest")
    _same(trace[1].get("submission"), submission, "Committed answer differs from the submission")
    completion = trace[0].get("output", {}).get("completion")
    if not isinstance(completion, str):
        raise ValueError("Released trace is missing its final response text")
    _same(parse_submission(completion), submission, "Model response differs from the submitted answer")
    usage = normalize_usage(record["usage"])
    event_usage = normalize_usage(trace[0]["usage"])
    if any(usage[field] != event_usage[field] for field in TOKEN_FIELDS):
        raise ValueError("Trace usage differs from recorded token usage")
    for key in COUNTERS:
        if record.get("observation_usage", {}).get(key) != 0 or record["usage"].get(key) != 0:
            raise ValueError("Condition A contains metered computation or observation usage")
    if usage["input_tokens_total"] > task.limits.input_tokens or usage["output_tokens"] > task.limits.output_tokens:
        raise ValueError("Recorded token usage exceeds the frozen budget")
    for field, cap in (("wall_seconds", task.limits.wall_seconds), ("cpu_seconds", task.limits.cpu_seconds)):
        value = record.get(field)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= cap:
            raise ValueError(f"Invalid or over-budget resource measurement: {field}")
    estimate = session_cost(model, usage)
    if not math.isclose(estimate, record["cost_usd"], rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Token-price estimate differs from the released usage and prices")
    result = score(task, submission)
    _score_matches(result, record)
    return {**record, "_completion": completion}, result


def replay(data_root=DEFAULT_DATA, results_root=DEFAULT_RESULTS, output_root=Path("replay-output"), *, references_only=False):
    """Verify hashes, regrade references/answers and rebuild aggregates on CPU.

    Hash checks establish consistency with the published manifest. Saved traces
    and original hashes are evidence supplied by the publisher, not independent
    proof that a proprietary model generated a response or incurred a charge.
    """
    data_manifest, tasks = load_public_tasks(data_root)
    current = harness_digest()
    if data_manifest.get("source_harness_digest") != current:
        raise ValueError("Installed scoring harness differs from the evaluated source snapshot")
    reference_results = []
    for task in tasks.values():
        result = score(task, reference_submission(task))
        if not result.success or result.status != "scored":
            raise ValueError(f"Released reference solution fails: {task.id}")
        reference_results.append({"task_id": task.id, "family": task.family, **result.to_dict()})
    summary = {"status": "passed", "mode": "public_reference_validation" if references_only else "saved_submission_regrading", "provider_requests": 0,
               "references_checked": len(tasks), "source_harness_digest": current,
               "source_suite_digest": data_manifest.get("source_suite_digest"),
               "saved_model_submissions_checked": 0, "website": "deferred",
               "limitations": "Offline replay regrades saved submissions; it does not rerun model inference or independently attest provider billing."}
    output_root = Path(output_root)
    if not references_only:
        results_root = Path(results_root)
        manifest = verify_manifest(results_root)
        if manifest.get("public_schema", manifest.get("schema_version")) != "wspbench-public-pilot-1":
            raise ValueError("Unsupported public pilot schema")
        if manifest.get("source_harness_digest", manifest.get("harness_digest")) != current:
            raise ValueError("Model bundle uses a different evaluated harness")
        if manifest.get("source_suite_digest", manifest.get("suite_digest")) != data_manifest.get("source_suite_digest"):
            raise ValueError("Model bundle and development fixtures name different source suites")
        records = _read(results_root / "records.json")
        expected = manifest.get("expected_matrix", [])
        if (not isinstance(records, list) or len(records) != len(expected) or not expected
                or manifest.get("expected_sessions") != len(expected)):
            raise ValueError("Missing or extra saved model submissions")
        planned = {_key(row): row for row in expected}
        if len(planned) != len(expected) or len({_key(row) for row in records}) != len(records):
            raise ValueError("Duplicate prespecified or observed model attempt")
        if set(planned) != {_key(row) for row in records}:
            raise ValueError("Saved submissions differ from the prespecified matrix")
        models = {m["id"]: m for m in manifest["models"]}
        checked = []
        for record in records:
            task = tasks.get(record["task_id"])
            if task is None:
                raise ValueError("Model result names an unreleased development task")
            item = planned[_key(record)]
            for key in ("target_digest", "model_config_digest", "parent_problem_id"):
                if record.get(key) != item.get(key):
                    raise ValueError(f"Model result differs from its matrix cell: {key}")
            for key in ("protocol_digest", "suite_digest", "harness_digest"):
                if record.get(key) != manifest.get(key):
                    raise ValueError(f"Model result differs from the original protocol: {key}")
            normalized, _ = _validate_session(results_root, record, models[record["model_id"]], task)
            checked.append(normalized)
        metadata = {task.id: {"family": task.family, "skill": task.skill,
                    "parent_problem_id": task.parent_problem_id, "mechanism_family_id": task.mechanism_family_id,
                    "baseline_question_digest": serialization_digest({"instruction":task.instruction,"public":task.public})}
                    for task in tasks.values()}
        controls = _read(results_root / "write-controls.json")
        for control in controls:
            task = tasks.get(control["task_id"])
            if task is None or task.skill != "write":
                raise ValueError("Control names an unreleased Write task")
            result = score(task, {"patch": []})
            if result.success != control["success"]:
                raise ValueError("Regraded no-edit control differs")
            _same(result.metrics, control["metrics"], "Regraded no-edit control metrics differ")
        report = build_comparison(manifest, checked, task_metadata=metadata, write_controls=controls)
        if report["status"] != "complete":
            raise ValueError("Public replay did not produce a complete matched and reviewed report")
        report["replay"] = {"provider_requests": 0, "all_saved_scores_match": True,
                            "public_development_only": True, "mode": summary["mode"]}
        output_root.mkdir(parents=True, exist_ok=True)
        for extension, content in (("json", json.dumps(report, indent=2, allow_nan=False)+"\n"),
                                   ("md", _markdown(report)), ("csv", _csv(report))):
            (output_root / f"gpt-comparison-v0.2.1.{extension}").write_text(content)
        summary.update(saved_model_submissions_checked=len(checked),
                       known_api_estimate_usd=sum(row["cost_usd"] for row in checked),
                       all_saved_scores_match=True, all_paired_inputs_match=report["paired_inputs_match"])
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "reference-results.json").write_text(json.dumps(reference_results, indent=2, allow_nan=False)+"\n")
    (output_root / "replay-summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=Path("replay-output"))
    parser.add_argument("--references-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        summary = replay(args.data, args.results, args.output, references_only=args.references_only)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(1, f"Public replay failed: {exc}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
