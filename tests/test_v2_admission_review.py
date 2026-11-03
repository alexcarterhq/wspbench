import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from wspbench.v2 import admission, runner
from wspbench.v2.contracts import Task
from wspbench.v2.serialization import save_weights, serialization_digest, weights_digest, load_weights


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    tasks = []
    proofs = []
    for i in range(8):
        directory = tmp_path / f"checkpoint-{i}"
        directory.mkdir()
        task = Task(f"task-{i}", f"parent-{i}", f"mechanism-{i}", "dev" if i < 4 else "heldout",
                    "rule_recovery", "read", "constructed", "small_512",
                    {"kind": "fixture", "shapes": {"w": [1, 1]}}, "Recover the rule.",
                    {"generator_revision": "composed-v2.1", "evaluation_role": "challenge_candidate"}, directory)
        save_weights(task.weights_path, {"w": np.asarray([[i]], dtype=np.float32)})
        tasks.append(task)
        for condition in "ABC":
            proofs.append({"task_id": task.id, "condition": condition, "status": "passed",
                "public_information_only": True, "public_derivation_checked": True,
                "target_digest": weights_digest(load_weights(task.weights_path)),
                "uses_target_execution": condition == "C", "method": "checked_public_equations",
                "evidence_digest": serialization_digest({"task": task.id, "condition": condition}),
                "analysis_calls": 0, "cpu_seconds": .001})
    rows = []
    for split in ("dev", "heldout"):
        for actor in admission.REQUIRED_BASELINES["rule_recovery"]:
            for condition in ("C",) if actor == "behavior_only" else ("B", "C"):
                rows.append({"actor": actor, "actor_type": "baseline", "family": "rule_recovery",
                    "target_stratum": "constructed", "capacity_tier": "small_512", "split": split,
                    "condition": condition, "weight_view": "full", "score": .5,
                    "n_parents": 4, "n_mechanisms": 4, "n_attempts": 4,
                    "access_card": {"privileged": False, "qualifying_families": ["rule_recovery"]},
                    "infrastructure_failures": 0, "audit_status_counts": {"pending": 4}})
    baseline = {"tables": rows, "suite_digest": "fixture-suite", "harness_digest": "fixture-harness"}
    coverage = [{"task_id": task.id, "target_digest": weights_digest(load_weights(task.weights_path)),
                 "actor": row["actor"], "condition": row["condition"], "weight_view": "full",
                 "status": "scored", "audit_status": "pending", "qualifying_families": ["rule_recovery"]}
                for task in tasks for row in rows if task.split == row["split"]]
    (tmp_path / "baseline-coverage.private.json").write_text(json.dumps(coverage))
    baseline["coverage_manifest_digest"] = serialization_digest(coverage)
    validation = {"status": "passed", "valid": True, "feasibility_records": proofs}
    monkeypatch.setattr(admission, "_scope", lambda root: tasks)
    from wspbench.v2 import tasks as task_module
    monkeypatch.setattr(task_module, "load_tasks", lambda root, split="all": tasks)
    monkeypatch.setattr(runner, "suite_digest", lambda root: "fixture-suite")
    monkeypatch.setattr(runner, "harness_digest", lambda: "fixture-harness")
    return tmp_path, tasks, baseline, validation


def test_b_ceiling_never_becomes_an_unaided_a_ceiling(evidence):
    root, tasks, baseline, validation = evidence
    for row in baseline["tables"]:
        if row["condition"] == "B" and row["actor"] == "circuit_algebra":
            row["score"] = 1.
    result = admission.assess_admission(root, baseline, validation)
    allowed = {(r["task_id"], r["condition"]) for r in result["calibration_allowlist"]}
    assert (tasks[0].id, "A") in allowed
    assert (tasks[0].id, "B") not in allowed
    assert (tasks[0].id, "C") in allowed
    assert not result["headline_admitted"] and not result["paid_runs_permitted"]


def test_baseline_coverage_is_condition_and_split_specific(evidence):
    root, tasks, baseline, validation = evidence
    baseline["tables"] = [r for r in baseline["tables"] if not
                          (r["actor"] == "statistical_reader" and r["condition"] == "B" and r["split"] == "heldout")]
    result = admission.assess_admission(root, baseline, validation)
    allowed = {r["condition"] for r in result["calibration_allowlist"]}
    assert allowed == {"C"}  # C cannot donate its coverage to A's B difficulty controls.


def test_no_c_execution_certificate_is_reused_as_a_or_b(evidence):
    root, tasks, baseline, validation = evidence
    for proof in validation["feasibility_records"]:
        proof["uses_target_execution"] = True
    result = admission.assess_admission(root, baseline, validation)
    assert {r["condition"] for r in result["calibration_allowlist"]} == {"C"}


def test_every_target_needs_an_individually_checked_public_derivation(evidence):
    root, tasks, baseline, validation = evidence
    validation["feasibility_records"] = [p for p in validation["feasibility_records"] if not
                                         (p["task_id"] == tasks[4].id and p["condition"] == "A")]
    result = admission.assess_admission(root, baseline, validation)
    assert "A" not in {r["condition"] for r in result["calibration_allowlist"]}
    for task in tasks:
        task.public["evaluation_role"] = "diagnostic_sanity"
    assert not admission.assess_admission(root, baseline, validation)["calibration_permitted"]


def test_allowlist_is_bound_to_exact_evidence_and_only_development_targets(evidence):
    root, tasks, baseline, validation = evidence
    result = admission.assess_admission(root, baseline, validation)
    assert result["calibration_permitted"]
    assert {r["task_id"] for r in result["calibration_allowlist"]} == {t.id for t in tasks[:4]}
    (root / "baselines.json").write_text(json.dumps(baseline))
    checked = runner._calibration_cells(root, root, result, validation)
    assert len(checked) == 12
    result["calibration_allowlist"][0]["target_digest"] = "0" * 64
    with pytest.raises(ValueError, match="exceed|different target"):
        runner._calibration_cells(root, root, result, validation)


def test_one_good_cell_does_not_authorize_other_conditions_or_tasks(evidence):
    root, tasks, baseline, validation = evidence
    result = admission.assess_admission(root, baseline, validation)
    only_a = [c for c in result["calibration_allowlist"] if c["condition"] == "A"]
    chosen = runner._select_calibration_cells(tasks, only_a, ("A", "B", "C"), ("full",), [tasks[0].id])
    assert len(chosen) == 1 and chosen[0]["condition"] == "A"
    with pytest.raises(ValueError, match="permission"):
        runner._select_calibration_cells(tasks, only_a, ("B",), ("full",), [tasks[0].id])
    with pytest.raises(ValueError, match="existing development"):
        runner._select_calibration_cells(tasks, only_a, ("A",), ("full",), [tasks[4].id])


def test_aggregate_counts_cannot_replace_exact_baseline_target_coverage(evidence):
    root, tasks, baseline, validation = evidence
    path = root / "baseline-coverage.private.json"
    coverage = json.loads(path.read_text())
    coverage = [entry for entry in coverage if not (entry["task_id"] == tasks[0].id
                 and entry["actor"] == "statistical_reader" and entry["condition"] == "B")]
    path.write_text(json.dumps(coverage))
    baseline["coverage_manifest_digest"] = serialization_digest(coverage)
    result = admission.assess_admission(root, baseline, validation)
    assert {entry["condition"] for entry in result["calibration_allowlist"]} == {"C"}
    # Changing a manifest after its hash was reviewed is also fail closed.
    path.write_text("[]")
    assert not admission.assess_admission(root, baseline, validation)["calibration_permitted"]


def test_a_capacity_witness_requires_complete_public_behavior(evidence):
    root, tasks, baseline, validation = evidence
    for proof in validation["feasibility_records"]:
        proof["method"] = "public_semantics_capacity_witness"
        proof["public_derivation_checked"] = False
        proof["constructive_witness_verified"] = True
        proof["complete_public_semantics"] = False
    assert not admission.assess_admission(root, baseline, validation)["calibration_permitted"]
    for proof in validation["feasibility_records"]:
        proof["complete_public_semantics"] = True
    assert admission.assess_admission(root, baseline, validation)["calibration_permitted"]


def test_paid_calibration_session_limit_is_checked_before_provider_or_ledger(evidence, monkeypatch):
    root, tasks, baseline, validation = evidence
    result = admission.assess_admission(root, baseline, validation)
    (root / "admission.json").write_text(json.dumps(result))
    (root / "cpu-validation.json").write_text(json.dumps(validation))
    (root / "baselines.json").write_text(json.dumps(baseline))
    config = root / "models.json"
    config.write_text(json.dumps({"models": [{"id": "gpt", "model": "mockllm/model", "api_key_env": "TEST_KEY"}]}))
    monkeypatch.setattr(runner, "preflight", lambda *a, **k: {"ready": True})
    def forbidden(*args, **kwargs):
        raise AssertionError("paid infrastructure reached before session-limit validation")
    monkeypatch.setattr(runner, "BudgetLedger", forbidden)
    from inspect_ai import model
    monkeypatch.setattr(model, "get_model", forbidden)
    with pytest.raises(ValueError, match="session limit"):
        asyncio.run(runner._run_paid(root, root, config, "calibration", 1, ["gpt"],
                                    ("A", "B", "C"), ("full",), None, [tasks[0].id], 1))
