from pathlib import Path
import json
import numpy as np
import pytest

from wspbench.v2.tasks import generate_composed_suite,load_tasks,validate_suite
from wspbench.v2.access import AccessSession,AccessViolation
from wspbench.v2.certificates import linear_fault_derivation
from wspbench.v2.scoring import score
from wspbench.v2.serialization import load_weights


@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    root=tmp_path_factory.mktemp("composed")/"suite"
    generate_composed_suite(root,seed=281,dev_per_family=1,heldout_per_family=1,baseline_per_label=0)
    return root


def test_revision_feasibility_is_checked_from_correct_information(suite):
    original={p:p.read_bytes() for p in suite.glob("*/*/feasibility.json")}
    validation=validate_suite(suite)
    assert validation["status"]=="passed"
    assert original and all(p.read_bytes()==data for p,data in original.items())
    tasks={t.id:t for t in load_tasks(suite,"all")}
    assert len(validation["feasibility_records"])>20
    for proof in validation["feasibility_records"]:
        assert proof["target_digest"] and proof["evidence_digest"]
        assert proof["model_output_tokens_bound"]<=32000
        if proof["method"]=="public_semantics_capacity_witness":
            assert proof["witness_privileged"]
            assert proof["complete_public_semantics"]
            assert "reference_program" not in tasks[proof["task_id"]].public


def test_boolean_numeric_specification_algebra_is_same_information_in_all_conditions(suite):
    task=next(t for t in load_tasks(suite,"dev") if t.family=="compositional_synthesis")
    sessions=[AccessSession(task,c) for c in "ABC"]
    assert sessions[0].public_payload()==sessions[1].public_payload()==sessions[2].public_payload()
    name="spec:outputs:0"
    assert name in sessions[0].public_payload()["public_specification_handles"]
    assert sessions[1].analyze("mean",[name])["values"]==np.mean(task.public["specification"]["outputs"][:128])
    with pytest.raises(AccessViolation):sessions[0].analyze("mean",[name])
    with pytest.raises(AccessViolation):sessions[1].probe([0])


def test_public_linear_repair_derivation_needs_neither_clean_weights_nor_hidden_outputs(suite,monkeypatch):
    task=next(t for t in load_tasks(suite,"dev") if t.family=="coupled_repair")
    weights=load_weights(task.weights_path)
    submission,certificate=linear_fault_derivation(task.public,task.architecture,weights)
    assert certificate["trial_solver_claim"] is False
    assert score(task,submission).success
    assert "normative" in certificate["source"]


def test_old_copy_controls_are_explicitly_outside_paid_challenges(suite):
    for task in load_tasks(suite,"all"):
        if task.family in {"causal_forecast","coupled_repair","loop_repair"}:
            assert task.public["evaluation_role"]=="diagnostic_sanity"
