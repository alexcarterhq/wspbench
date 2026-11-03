from __future__ import annotations
from pathlib import Path
import io
import json
import zipfile
import numpy as np
import pytest
from wspbench.tasks import generate_suite, load_tasks, validate_suite
from wspbench.export import export_development
from wspbench.scoring import score, reference_submission, baseline_submission, evaluate_practice
from wspbench.networks import load_weights, save_weights, forward


@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    root = tmp_path_factory.mktemp("scientific-suite")
    generate_suite(root, official_per_family=1, dev_per_family=2, seed=81723)
    return root


def test_all_six_families_have_feasible_references(suite):
    result = validate_suite(suite)
    assert result["status"] == "passed"
    assert result["reference_successes"] == result["task_count"] == 18
    tasks = load_tasks(suite, "official")
    assert len({t.parent_id for t in load_tasks(suite, "all")}) == 18
    assert {t.skill for t in tasks} == {"read", "repair", "write"}
    for task in tasks:
        assert sum(np.prod(s) for s in task.architecture["shapes"].values()) < 25_000


def test_read_ground_truth_is_empirically_verified_and_not_public(suite):
    for task in load_tasks(suite, "all"):
        public = task.to_public_dict()
        assert "generation_seed" not in json.dumps(public)
        assert "answer" not in public
        if task.skill != "read":
            continue
        private = json.loads(task.private_path.read_text())
        assert "admission" in private
        assert score(task, {"answer": reference_submission(task)}).success
        wrong = next(v for v in task.public["options"] if v != reference_submission(task))
        assert not score(task, wrong).success
        with pytest.raises(ValueError, match="prohibit"):
            evaluate_practice(task, load_weights(task.weights_path), [0])


def test_zero_and_random_programs_do_not_get_reference_credit(suite):
    for task in load_tasks(suite, "official"):
        if task.skill == "write":
            assert not score(task, baseline_submission(task, "noop")).success
            assert not score(task, baseline_submission(task, "random")).success
        if task.family in {"repair_fault", "repair_shortcut"}:
            broken = score(task, baseline_submission(task, "noop"))
            assert not broken.success
            assert broken.metrics["target_accuracy"] < .95
            assert score(task, reference_submission(task)).metrics["retention_loss"] <= .02


def test_practice_evaluator_has_only_bounded_public_cases(suite):
    for task in load_tasks(suite, "official"):
        if task.skill == "read":
            continue
        result = evaluate_practice(task, reference_submission(task), [0, 1])
        assert len(result["cases"]) == 2
        assert "answer" not in result
        with pytest.raises(ValueError):
            evaluate_practice(task, reference_submission(task), [0, 0])
        with pytest.raises(ValueError):
            evaluate_practice(task, reference_submission(task), [128])


def test_submission_shape_nan_and_pickle_are_rejected(suite, tmp_path):
    task = next(t for t in load_tasks(suite, "official") if t.family == "write_piecewise")
    reference = reference_submission(task)
    wrong = dict(reference)
    wrong["w0"] = np.zeros((1, 1))
    assert score(task, wrong).status == "invalid_submission"
    wrong = {k: v.copy() for k, v in reference.items()}
    wrong["b0"][0] = np.nan
    assert score(task, wrong).status == "invalid_submission"
    path = tmp_path / "object.npz"
    np.savez(path, unsafe=np.array([{}], dtype=object))
    with pytest.raises(ValueError):
        load_weights(path)
    assert score(task, path).status == "invalid_submission"
    path.write_text("not a zip")
    assert score(task, path).status == "invalid_submission"


def test_array_header_cannot_allocate_a_giant_tensor(tmp_path):
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {"descr": "<f4", "fortran_order": False, "shape": (1_000_000_000,)})
    path = tmp_path / "oversized.npz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("w0.npy", header.getvalue())
    with pytest.raises(ValueError, match="oversized"):
        load_weights(path)


def test_narrow_spike_between_dense_grid_points_is_detected(suite):
    task = next(t for t in load_tasks(suite, "official") if t.family == "write_piecewise")
    weights = reference_submission(task)
    center, epsilon, amplitude = .123456789, 1e-7, .01
    weights["w0"][5:, 0] = 1
    weights["b0"][5:] = -(np.array([center - epsilon, center, center + epsilon]))
    weights["w1"][0, 5:] = amplitude / epsilon * np.array([1, -2, 1])
    result = score(task, weights)
    assert not result.success
    assert result.metrics["max_absolute_error"] > .005


def test_development_export_never_copies_official_or_private_files(suite, tmp_path):
    destination = tmp_path / "export"
    exported = export_development(suite, destination)
    assert exported["n_tasks"] == 12
    index = json.loads((destination / "index.json").read_text())
    assert len(index["tasks"]) == 12
    assert all(json.loads((destination / t["id"] / "manifest.json").read_text())["split"] == "dev" for t in index["tasks"])
    assert not list(destination.rglob("private*"))
    official_ids = {t.id for t in load_tasks(suite, "official")}
    assert not any((destination / task_id).exists() for task_id in official_ids)
    with pytest.raises(ValueError, match="already exists"):
        generate_suite(suite)


def test_public_spec_rebuild_cannot_replace_learned_repair_behavior(suite):
    for task in load_tasks(suite, "official"):
        if task.skill != "repair":
            continue
        rebuilt = score(task, baseline_submission(task, "no_weight"))
        assert rebuilt.metrics["target_accuracy"] >= .95
        assert rebuilt.metrics["auxiliary_retention_accuracy"] < .98
        assert not rebuilt.success
        repaired = score(task, reference_submission(task))
        assert repaired.success
        assert repaired.metrics["auxiliary_max_absolute_error"] == 0


def test_validation_detects_modified_frozen_artifacts(suite, tmp_path):
    import shutil
    copied = tmp_path / "copied"
    shutil.copytree(suite, copied)
    task = load_tasks(copied, "official")[0]
    with (task.directory / "private.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="fingerprint"):
        validate_suite(copied)
