"""Release-boundary checks use generated fixtures, never the private workspace."""
import json
from pathlib import Path

import numpy as np
import pytest

from wspbench.public.export import (_assert_public, _export_task, _public_model,
                                   _public_trace, export_release)
from wspbench.v2.challenges import make_challenge
from wspbench.v2.contracts import Task
from wspbench.v2.serialization import load_weights, save_weights


def make_task(root, *, split="dev"):
    bundle = make_challenge(1729, "rule_recovery")
    root.mkdir()
    task = Task("public-example", "public-parent", bundle["mechanism_family_id"], split,
                "rule_recovery", "read", "constructed_microcircuit", "microcircuit_4049",
                bundle["architecture"], bundle["instruction"], bundle["public"], root)
    (root / "task.json").write_text(json.dumps(task.to_public_dict()))
    save_weights(root / "target.npz", bundle["weights"])
    private = {**bundle["private"], "generation_seed": 98765432123456789,
               "training": {"feature_seed": 98765432123456790}, "unrelated_operator_note": "must not export"}
    (root / "private.json").write_text(json.dumps(private))
    np.savez_compressed(root / "evaluation.npz", inputs=bundle["inputs"], expected=bundle["expected"],
                        practice=bundle["practice"], final=bundle["final"])
    return task, private


def test_dev_fixture_exports_only_replay_artifacts_and_no_shared_seed(tmp_path):
    task, private = make_task(tmp_path / "source")
    destination = tmp_path / "export"
    entry = _export_task(task, destination)
    assert set(path.name for path in destination.iterdir()) == {
        "task.json", "target.npz", "evaluation.npz", "reference.json"}
    reference = json.loads((destination / "reference.json").read_text())
    assert reference == {"reference_submission": private["reference_submission"]}
    assert entry["directory"] == "dev/public-example"
    for name, metadata in entry["artifacts"].items():
        assert len(metadata["sha256"]) == len(metadata["source_sha256"]) == 64
    for path in destination.glob("*.json"):
        assert str(private["generation_seed"]) not in path.read_text()
        assert "generation_seed" not in path.read_text()
        assert "unrelated_operator_note" not in path.read_text()
    assert all(np.array_equal(value, load_weights(destination / "target.npz")[key])
               for key, value in load_weights(task.weights_path).items())
    with np.load(destination / "evaluation.npz", allow_pickle=False) as bank:
        assert set(bank.files) == {"inputs", "expected", "practice", "final"}


def test_heldout_fixture_export_is_rejected_before_writing(tmp_path):
    task, _ = make_task(tmp_path / "source", split="heldout")
    with pytest.raises(ValueError, match="development"):
        _export_task(task, tmp_path / "export")
    assert not (tmp_path / "export").exists()


def test_unrecognized_scoring_array_is_not_silently_copied(tmp_path):
    task, _ = make_task(tmp_path / "source")
    with np.load(task.directory / "evaluation.npz", allow_pickle=False) as bank:
        arrays = {key: bank[key] for key in bank.files}
    np.savez_compressed(task.directory / "evaluation.npz", **arrays, generation_seed=np.array([12345]))
    with pytest.raises(ValueError, match="unrecognized arrays"):
        _export_task(task, tmp_path / "export")


def sample_trace():
    answer = {"program": {"op": "input", "index": 0}}
    usage = {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20, "reasoning_tokens": 2}
    trace = [{"tool": "model_turn", "turn": 1, "usage": usage,
              "output": {"model": "example-model", "completion": json.dumps(answer), "usage": usage,
                         "response_id": "provider-sensitive-id", "metadata": {"request_id": "request-sensitive-id"},
                         "choices": [{"message": {"content": [{"type": "reasoning", "reasoning": "encrypted-opaque-data"}]}}]}},
             {"tool": "commit", "submission": answer, "observation_count": 0},
             {"tool": "submit", "observation_count": 0}]
    return trace, answer


def test_trace_retains_answer_usage_access_but_removes_provider_internals():
    original, answer = sample_trace()
    public = _public_trace(original, answer)
    text = json.dumps(public)
    assert "provider-sensitive-id" not in text
    assert "request-sensitive-id" not in text
    assert "encrypted-opaque-data" not in text
    assert public[0]["output"]["completion"] == original[0]["output"]["completion"]
    assert public[0]["usage"] == original[0]["usage"]
    assert [event["tool"] for event in public] == ["model_turn", "commit", "submit"]


@pytest.mark.parametrize("mutation", ["tool", "observations", "answer"])
def test_trace_export_rejects_unreviewed_access_or_submission_changes(mutation):
    trace, answer = sample_trace()
    if mutation == "tool":
        trace.insert(1, {"tool": "analysis"})
    elif mutation == "observations":
        trace[1]["observation_count"] = 1
    else:
        trace[0]["output"]["completion"] = '{"program":{"op":"input","index":1}}'
    with pytest.raises(ValueError):
        _public_trace(trace, answer)


def test_unknown_credential_configuration_and_structural_seed_are_rejected():
    with pytest.raises(ValueError, match="configuration"):
        _public_model({"id": "example", "api_key": "never export credentials"})
    with pytest.raises(ValueError, match="Private"):
        _assert_public({"details": {"generation_seed": 1}})
    # An environment variable's name is public configuration, not its value.
    _public_model({"id": "example", "api_key_env": "OPENAI_API_KEY"})


def test_full_export_refuses_changed_evaluated_runtime_before_any_data_export(tmp_path):
    runs = tmp_path / "runs/calibration"
    runs.mkdir(parents=True)
    (runs / "manifest.json").write_text(json.dumps({"harness_digest": "changed", "suite_digest": "changed"}))
    with pytest.raises(ValueError, match="evaluated runtime"):
        export_release(tmp_path / "no-private-data", runs.parent, tmp_path / "release")
    assert not (tmp_path / "release").exists()
