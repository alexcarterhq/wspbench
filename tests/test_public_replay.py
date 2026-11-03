"""Public replay is provider-free and rejects changed evidence or answers."""
import hashlib
import json
from pathlib import Path
import socket

import numpy as np
import pytest

from wspbench.budget import session_cost
from wspbench.public.replay import PublicTask, replay
from wspbench.v2.access import AccessSession
from wspbench.v2.challenges import make_challenge
from wspbench.v2.inspect_task import CONDITIONS, SYSTEM
from wspbench.v2.runner import harness_digest
from wspbench.v2.scoring import score
from wspbench.v2.serialization import save_weights, serialization_digest, weights_digest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def catalog(root, manifest):
    manifest["files"] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(root.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    write(root / "manifest.json", manifest)


@pytest.fixture
def bundle(tmp_path):
    data, results = tmp_path / "data", tmp_path / "results"
    task_dir = data / "dev" / "public-fixture"
    task_dir.mkdir(parents=True)
    challenge = make_challenge(17, "compositional_synthesis")
    task = PublicTask("public-fixture", "public-parent", "public-mechanism", "dev", "compositional_synthesis",
                      "write", "constructed_microcircuit", "microcircuit_4049", challenge["architecture"],
                      challenge["instruction"], challenge["public"], task_dir)
    write(task_dir / "task.json", task.to_public_dict())
    save_weights(task.weights_path, challenge["weights"])
    reference = challenge["private"]["reference_submission"]
    write(task.private_path, {"reference_submission": reference})
    np.savez_compressed(task_dir / "evaluation.npz", inputs=challenge["inputs"], expected=challenge["expected"],
                        practice=challenge["practice"], final=challenge["final"])
    artifacts = {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                         "source_sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in task_dir.iterdir()}
    current = harness_digest()
    fixture_manifest = {"schema_version": "wspbench-public-development-1", "source_harness_digest": current,
                        "source_suite_digest": "source-suite", "tasks": [{"task_id": task.id,
                        "family": task.family, "split": "dev", "directory": "dev/public-fixture", "artifacts": artifacts}]}
    catalog(data, fixture_manifest)
    model = {"id": "fixture-model", "provider": "openai", "model": "openai/fixture",
             "generate_config": {"reasoning_effort": "high", "reasoning_summary": "none"},
             "model_args": {"service_tier": "default", "streaming": True, "responses_api": True},
             "pricing": {"verified_at": "test-fixture", "source": "test-fixture-not-current-provider-pricing",
                         "input_per_million": 1., "cache_write_per_million": 1.25,
                         "cache_read_per_million": .1, "output_per_million": 5.}}
    payload = AccessSession(task, "A").public_payload()
    prompt = {"system": SYSTEM, "access": CONDITIONS["A"], "payload": payload}
    usage = {"input_tokens": 10, "input_tokens_cache_write": 0, "input_tokens_cache_read": 0,
             "input_tokens_total": 10, "output_tokens": 2, "reasoning_tokens": 0, "total_tokens": 12}
    counters = {key: 0 for key in ("analysis_calls", "example_evaluations", "activation_scalars", "episodes",
                                 "steps", "generated_tokens", "candidate_patches")}
    usage.update(counters)
    result = score(task, reference)
    trace = [{"tool": "model_turn", "turn": 1, "usage": {key: value for key, value in usage.items() if key not in counters},
              "output": {"model": model["model"], "completion": json.dumps(reference)}},
             {"tool": "commit", "submission": reference, "observation_count": 0},
             {"tool": "submit", "observation_count": 0}]
    record = {"id": "public-run", "model_id": model["id"], "model_config": model,
              "model_config_digest": serialization_digest(model), "task_id": task.id,
              "parent_problem_id": task.parent_problem_id, "mechanism_family_id": task.mechanism_family_id,
              "condition": "A", "weight_view": "full", "attempt": 1, "actor_type": "model", "status": "completed",
              "split": "dev", "family": task.family, "skill": task.skill, "protocol_digest": "public-protocol",
              "suite_digest": "source-suite", "harness_digest": current, "turns": 1, "submitted": True,
              "target_digest": weights_digest(challenge["weights"]), "question_digest": serialization_digest(payload["task"]),
              "serialization_digest": payload["serialization_digest"], "canonical_prompt_digest": serialization_digest(payload),
              "prompt_digest": serialization_digest(prompt), "usage": usage, "observation_usage": counters,
              "cpu_seconds": .1, "wall_seconds": 1., "cost_usd": session_cost(model, usage),
              "submission": reference, "success": result.success, "metrics": result.metrics,
              "score_status": result.status, "score_reason": result.reason,
              "audit": {"status": "approved", "reviewer": "fixture", "trace_sha256": serialization_digest(trace)}}
    directory = results / "sessions" / record["id"]
    for name, value in (("prompt.json", prompt), ("submission.json", reference), ("trace.json", trace), ("result.json", record)):
        write(directory / name, value)
    write(results / "records.json", [record])
    control_score = score(task, {"patch": []})
    controls = [{"task_id": task.id, "actor_id": "metadata_prior", "actor_type": "baseline", "condition": "B",
                 "weight_view": "full", "target_digest": record["target_digest"],
                 "question_digest": serialization_digest({"instruction":task.instruction,"public":task.public}),
                 "family": task.family, "metrics": control_score.metrics, "success": control_score.success}]
    write(results / "write-controls.json", controls)
    expected = {key: record[key] for key in ("model_id", "model_config_digest", "task_id", "target_digest",
                                           "parent_problem_id", "condition", "weight_view", "attempt")}
    manifest = {"schema_version": "wspbench-public-pilot-1", "phase": "calibration", "benchmark_version": "0.2.0",
                "models": [model], "expected_matrix": [expected], "expected_sessions": 1,
                "source_harness_digest": current, "harness_digest": current, "source_suite_digest": "source-suite",
                "suite_digest": "source-suite", "protocol_digest": "public-protocol"}
    catalog(results, manifest)
    return data, results, tmp_path / "replayed"


def test_regrades_without_credentials_private_workspaces_or_provider_resolution(bundle, monkeypatch, tmp_path):
    import inspect_ai.model
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(inspect_ai.model, "get_model", lambda *a, **k: pytest.fail("Provider resolution during replay"))
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("Network access during replay"))
    working = tmp_path / "unrelated"; working.mkdir(); monkeypatch.chdir(working)
    summary = replay(*bundle)
    assert summary["references_checked"] == summary["saved_model_submissions_checked"] == 1
    assert summary["provider_requests"] == 0
    assert summary["all_saved_scores_match"]
    report = json.loads((bundle[2] / "gpt-comparison-v0.2.1.json").read_text())
    assert report["models"][0]["skills"][0]["exact_successes"] == 1
    assert report["write_difficulty_control"]["same_access_model_score"] is False
    assert "legacy_exposure" not in report
    assert "Legacy unresolved" not in (bundle[2] / "gpt-comparison-v0.2.1.md").read_text()


def test_rejects_corrupt_scoring_fixture(bundle):
    fixture = bundle[0] / "dev/public-fixture/evaluation.npz"
    fixture.write_bytes(fixture.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        replay(*bundle)


def test_rejects_missing_expected_attempt(bundle):
    manifest = json.loads((bundle[1] / "manifest.json").read_text())
    manifest["expected_matrix"] = []
    catalog(bundle[1], manifest)
    with pytest.raises(ValueError, match="Missing or extra"):
        replay(*bundle)


def test_semantic_regrading_catches_changed_answer_even_after_rehashing(bundle):
    data, results, output = bundle
    records = json.loads((results / "records.json").read_text())
    record = records[0]
    directory = results / "sessions" / record["id"]
    altered = {"patch": []}
    record["submission"] = altered
    trace = json.loads((directory / "trace.json").read_text())
    trace[0]["output"]["completion"] = json.dumps(altered)
    trace[1]["submission"] = altered
    record["audit"]["trace_sha256"] = serialization_digest(trace)
    for name, value in (("submission.json", altered), ("trace.json", trace), ("result.json", record)):
        write(directory / name, value)
    write(results / "records.json", records)
    catalog(results, json.loads((results / "manifest.json").read_text()))
    with pytest.raises(ValueError, match="Regraded answer differs"):
        replay(data, results, output)


def test_rejects_unexpected_tool_even_with_rehashed_catalog(bundle):
    directory = bundle[1] / "sessions/public-run"
    trace = json.loads((directory / "trace.json").read_text())
    trace.insert(1, {"tool": "probe", "indices": [0]})
    write(directory / "trace.json", trace)
    manifest = json.loads((bundle[1] / "manifest.json").read_text()); catalog(bundle[1], manifest)
    with pytest.raises(ValueError, match="unexpected tools"):
        replay(*bundle)


def test_rejects_non_development_fixture(bundle):
    manifest = json.loads((bundle[0] / "manifest.json").read_text())
    manifest["tasks"][0]["split"] = "heldout"
    write(bundle[0] / "manifest.json", manifest)
    with pytest.raises(ValueError, match="public development"):
        replay(*bundle)


def test_reference_only_replay_does_not_require_results(bundle):
    summary = replay(bundle[0], Path("does-not-exist"), bundle[2], references_only=True)
    assert summary["references_checked"] == 1
    assert summary["saved_model_submissions_checked"] == 0


def test_rejects_manifest_path_traversal(bundle):
    manifest = json.loads((bundle[0] / "manifest.json").read_text())
    manifest["tasks"][0]["directory"] = "../results"
    write(bundle[0] / "manifest.json", manifest)
    with pytest.raises(ValueError, match="leaves the public bundle"):
        replay(*bundle)


def test_cluster_metadata_cannot_be_reassigned_separately_from_fixture(bundle):
    records = json.loads((bundle[1] / "records.json").read_text())
    records[0]["parent_problem_id"] = "invented-independent-parent"
    write(bundle[1] / "records.json", records)
    write(bundle[1] / "sessions/public-run/result.json", records[0])
    manifest = json.loads((bundle[1] / "manifest.json").read_text())
    manifest["expected_matrix"][0]["parent_problem_id"] = records[0]["parent_problem_id"]
    catalog(bundle[1], manifest)
    with pytest.raises(ValueError, match="Recorded task metadata differs"):
        replay(*bundle)
