import json
import numpy as np
import pytest

import wspbench.baselines as baselines
from wspbench.domain import Task
from wspbench.networks import mlp_architecture, save_weights
from wspbench.scoring import score


@pytest.fixture
def read_pool(tmp_path):
    tasks = []
    entries = []
    for split, identity, scale, answer in [("dev", "development-a", 1., "alpha"),
                                            ("dev", "development-b", 3., "beta"),
                                            ("official", "heldout-secret-a", 1.1, "alpha"),
                                            ("official", "heldout-secret-b", 2.9, "beta")]:
        directory = tmp_path / split / identity
        directory.mkdir(parents=True)
        architecture = mlp_architecture([2, 2, 1])
        weights = {key: np.full(shape, scale, dtype=np.float32) for key, shape in architecture["shapes"].items()}
        save_weights(directory / "target.npz", weights)
        (directory / "private.json").write_text(json.dumps({"answer": answer}))
        task = Task(identity, identity, split, "read", "read_visual", architecture, "Identify behavior.",
                    {"options": ["alpha", "beta", "gamma", "delta"]}, directory)
        (directory / "manifest.json").write_text(json.dumps(task.to_public_dict()))
        tasks.append(task)
        entries.append({"id": identity, "parent_id": identity, "split": split,
                        "skill": "read", "family": "read_visual", "manifest": f"{split}/{identity}/manifest.json"})
    (tmp_path / "suite.json").write_text(json.dumps({"fingerprint": "fixture-fingerprint", "tasks": entries}))
    return tmp_path, tasks


def test_statistics_predictions_do_not_depend_on_heldout_answers(read_pool, monkeypatch):
    root, tasks = read_pool
    actual_reference = baselines.reference_submission
    observed = []
    def public_development_label(task):
        assert task.split == "dev", "Fitting or predicting accessed a held-out label"
        observed.append(task.id)
        return actual_reference(task)
    monkeypatch.setattr(baselines, "reference_submission", public_development_label)
    baseline = baselines.WeightStatisticsBaseline(tasks[:2])
    heldout = tasks[2]
    before = baseline.predict(heldout)
    assert before == "alpha"
    assert score(heldout, before).success
    heldout.private_path.write_text(json.dumps({"answer": "beta"}))
    after = baseline.predict(heldout)
    assert after == before
    assert not score(heldout, after).success
    assert set(observed) == {"development-a", "development-b"}
    with pytest.raises(ValueError, match="only public development"):
        baselines.WeightStatisticsBaseline(tasks)


def test_metadata_baseline_ignores_weights_and_ground_truth(read_pool, monkeypatch):
    _, tasks = read_pool
    task = tasks[2]
    def forbidden(*args, **kwargs):
        raise AssertionError("Metadata-only prediction accessed tensor or label data")
    monkeypatch.setattr(baselines, "load_weights", forbidden)
    monkeypatch.setattr(baselines, "reference_submission", forbidden)
    assert baselines.metadata_only_prediction(task) == "alpha"
    task.private_path.write_text(json.dumps({"answer": "delta"}))
    assert baselines.metadata_only_prediction(task) == "alpha"


def test_public_baseline_report_contains_only_aggregates(read_pool):
    root, tasks = read_pool
    report = baselines.evaluate_baselines(root)
    encoded = json.dumps(report)
    assert not any(task.id in encoded for task in tasks if task.split == "official")
    assert '"prediction"' not in encoded
    assert '"answer"' not in encoded
    assert report["results"]["official"]["read"]["task_count"] == 2
    assert report["results"]["official"]["read"]["baselines"]["weight_statistics"]["success_rate"] == 1
    assert report["weight_statistics"]["training"]["read_visual"]["development_examples"] == 2
    private = json.loads((root / "baseline-details.private.json").read_text())
    assert any(row["task_id"] == tasks[2].id and "prediction" in row for row in private["records"])
    saved = json.loads((root / "baseline-summary.json").read_text())
    assert saved == report
