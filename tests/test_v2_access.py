from dataclasses import replace
import sys
import types

import numpy as np
import pytest

from wspbench.v2.access import AccessSession, AccessViolation
from wspbench.v2.contracts import Limits, Task
from wspbench.v2.serialization import save_weights


@pytest.fixture
def target(tmp_path):
    arch = {"kind": "fixture", "shapes": {"body": [2, 2], "head": [1, 2]},
            "boundary_tensors": ["head"]}
    task = Task("fixture", "parent", "mechanism", "dev", "coupled_repair", "repair",
                "constructed", "small", arch, "Repair the supplied checkpoint.",
                {"practice_size": 256, "activation_scalars_per_example": 4}, tmp_path)
    save_weights(task.weights_path, {"body": np.array([[1., 2.], [3., 4.]], dtype=np.float32),
                                     "head": np.array([[5., 6.]], dtype=np.float32)})
    return task


@pytest.fixture
def observed(monkeypatch):
    calls = []
    def probe(task, weights, indices, observation="behavior", **kwargs):
        calls.append({k: v.copy() for k, v in weights.items()})
        return {"outputs": [float(weights["body"][0, 0])] * len(indices), "cases": indices,
                **({"activations": {"hidden": [[1., 2., 3., 4.]] * len(indices)}} if observation == "activations" else {}),
                "accounting": {"example_evaluations": 0 if task.family == "loop_repair" else len(indices) * (2 if task.family == "causal_forecast" else 1),
                               "activation_scalars": 4 * len(indices) if observation == "activations" else 0,
                               "episodes": len(indices) if task.family == "loop_repair" else 0,
                               "steps": 64 * len(indices) if task.family == "loop_repair" else 0}}
    module = types.ModuleType("wspbench.v2.scoring")
    module.probe = probe
    monkeypatch.setitem(sys.modules, "wspbench.v2.scoring", module)
    return calls


def test_three_conditions_receive_identical_float32_payload(target):
    payloads = [AccessSession(target, c).public_payload() for c in "ABC"]
    assert payloads[0] == payloads[1] == payloads[2]
    assert "private" not in str(payloads)
    assert not {"family", "mechanism_family_id", "parent_problem_id", "split", "target_stratum"} & set(payloads[0]["task"])


def test_unaided_has_no_tools_and_analysis_cannot_execute(target, observed):
    a = AccessSession(target, "A")
    with pytest.raises(AccessViolation):
        a.analyze("read", ["target:body"])
    assert a.finish()["audit"]["status"] == "rejected"
    b = AccessSession(target, "B")
    assert b.analyze("norm", ["target:body"])["values"] > 0
    with pytest.raises(AccessViolation):
        b.probe([0])
    assert not observed


def test_analysis_is_bounded_weight_algebra(target):
    session = AccessSession(target, "B")
    value = session.analyze("matmul", ["target:body", "target:body"])
    assert value["values"] == [[7., 10.], [15., 22.]]
    for op in ("python", "forward", "read_file", "activation"):
        with pytest.raises(AccessViolation):
            session.analyze(op, ["target:body"])
    with pytest.raises(AccessViolation):
        session.analyze("read", ["../../private.json"])


def test_metered_activations_can_be_analyzed_without_extra_target_calls(target, observed):
    session = AccessSession(target, "C")
    feedback = session.probe([0, 1], observation="activations")
    handle = feedback["activation_handles"]["hidden"]["handle"]
    mean = session.analyze("mean", [handle], axis=0)
    assert mean["values"] == [1., 2., 3., 4.]
    assert len(observed) == 1
    assert session.counters["activation_scalars"] == 8
    assert session.counters["example_evaluations"] == 2


@pytest.mark.parametrize("view", ["masked", "boundary_only", "matched_wrong"])
def test_patch_preserves_unseen_true_weights(target, observed, view):
    wrong = {"body": np.ones((2, 2), np.float32) * 9, "head": np.ones((1, 2), np.float32) * 9}
    session = AccessSession(target, "C", view, wrong_weights=wrong)
    patch = [{"op": "set", "tensor": "body", "index": [0, 0], "value": -1}]
    session.probe([0], patch=patch)
    assert observed[-1]["body"].tolist() == [[-1., 2.], [3., 4.]]
    assert observed[-1]["head"].tolist() == [[5., 6.]]


def test_practice_counts_repeated_queries_and_carry_forward(target, observed):
    session = AccessSession(target, "C")
    session.probe(list(range(32)))
    session.probe([0])
    session.submit({"patch": []})
    record = session.finish()
    assert len(observed) == 2
    assert record["actual_observations"] == 33
    assert record["snapshot_actual_observations"] == {"0": 0, "32": 32, "128": 33}
    assert record["counters"]["example_evaluations"] == 33
    assert record["snapshots"]["128"] == {"patch": []}


def test_budget_checks_precede_execution(target, observed):
    session = AccessSession(target, "C")
    with pytest.raises(AccessViolation, match="boundary"):
        session.probe(list(range(33)))
    with pytest.raises(AccessViolation, match="duplicated"):
        session.probe([0, 0])
    assert not observed
    assert session.counters["example_evaluations"] == 0


def test_distinct_materialized_candidates_limit(target, observed):
    session = AccessSession(replace(target, limits=replace(Limits(), candidate_patches=1)), "C")
    patch = [{"op": "set", "tensor": "body", "index": [0, 0], "value": 7}]
    session.probe([0], patch=patch)
    session.probe([0], patch=patch)
    with pytest.raises(AccessViolation, match="Candidate"):
        session.probe([0], patch=[{**patch[0], "value": 8}])
    assert len(observed) == 2
    assert session.counters["candidate_patches"] == 1


def test_forecast_queries_cannot_probe_final_interventions_or_patch(target, observed):
    task = replace(target, family="causal_forecast", skill="read",
                   public={**target.public, "practice_interventions": ["practice-0"],
                           "forecast_interventions": [{"id": "hidden-0"}]})
    session = AccessSession(task, "C")
    with pytest.raises(AccessViolation, match="practice intervention"):
        session.probe([0], intervention_id="hidden-0")
    with pytest.raises(AccessViolation, match="arbitrary"):
        session.probe([0], intervention_id="practice-0", patch=[{"op": "zero", "tensor": "head"}])
    assert not observed
    session.probe([0], intervention_id="practice-0")
    assert len(observed) == 1
    assert session.counters["example_evaluations"] == 2


def test_before_after_read_checkpoints_never_accept_numeric_patch(target, observed):
    from wspbench.v2.serialization import load_weights
    save_weights(target.directory / "before.npz", load_weights(target.weights_path))
    task = replace(target, family="edit_effects", skill="read")
    session = AccessSession(task, "C")
    assert set(session.public_payload()["checkpoints"]) == {"target", "before", "after"}
    with pytest.raises(AccessViolation, match="public checkpoint identities"):
        session.probe([0], checkpoint="before", patch=[{"op": "zero", "tensor": "head"}])
    assert not observed


def test_loop_condition_and_episode_budgets(target, observed):
    task = replace(target, family="loop_repair", conditions=("C",))
    with pytest.raises(AccessViolation):
        AccessSession(task, "A")
    session = AccessSession(task, "C")
    session.probe([0, 1, 2, 3])
    session.probe(list(range(12)))
    session.submit({"patch": []})
    record = session.finish()
    assert record["counters"]["episodes"] == 16
    assert record["snapshot_unit"] == "episodes"
    assert set(record["snapshots"]) == {"0", "4", "16"}


def test_paid_gate_stops_before_provider_resolution_or_ledger(target, tmp_path, monkeypatch):
    import json
    import wspbench.v2.runner as runner
    import inspect_ai.model
    model = {"id": "fixture", "model": "never/call", "provider": "test", "api_key_env": "WSPBENCH_TEST_KEY"}
    config = tmp_path / "models.json"
    config.write_text(json.dumps({"models": [model]}))
    monkeypatch.delenv("WSPBENCH_TEST_KEY", raising=False)
    module = types.ModuleType("wspbench.v2.tasks")
    module.load_tasks = lambda root, split: [target]
    monkeypatch.setitem(sys.modules, "wspbench.v2.tasks", module)
    def forbidden(*args, **kwargs):
        pytest.fail("Blocked paid preflight resolved a provider or initialized a ledger")
    monkeypatch.setattr(inspect_ai.model, "get_model", forbidden)
    monkeypatch.setattr(runner, "BudgetLedger", forbidden)
    with pytest.raises(ValueError, match="Paid execution blocked"):
        runner.run_evaluation(tmp_path, tmp_path / "runs", config)
    assert not (tmp_path / ".wspbench-state").exists()


def test_two_targets_of_one_semantic_parent_have_distinct_session_files(target, tmp_path, monkeypatch):
    from wspbench.v2.contracts import Score
    from wspbench.v2.runner import run_cpu_matrix
    from wspbench.v2.serialization import load_weights
    root = tmp_path / "suite"
    targets = []
    for index in range(2):
        item = replace(target, id=f"target-{index}", directory=root / f"target-{index}")
        weights = load_weights(target.weights_path)
        weights["body"][0, 0] = index + 1
        save_weights(item.weights_path, weights)
        targets.append(item)
    module = types.ModuleType("wspbench.v2.tasks")
    module.load_tasks = lambda root, split: targets
    monkeypatch.setitem(sys.modules, "wspbench.v2.tasks", module)
    module = types.ModuleType("wspbench.v2.scoring")
    module.reference_submission = lambda task: {"patch": []}
    module.score = lambda task, answer: Score(True)
    monkeypatch.setitem(sys.modules, "wspbench.v2.scoring", module)
    result = run_cpu_matrix(root, tmp_path / "runs", conditions=("A",))
    assert result["sessions"] == 2
    assert len(list((tmp_path / "runs" / "sessions").glob("*/result.json"))) == 2


def test_denied_trace_cannot_be_approved(target, tmp_path):
    import json
    from wspbench.v2.audit import review_run
    session = AccessSession(target, "B")
    with pytest.raises(AccessViolation):
        session.probe([0])
    record = session.finish()
    (tmp_path / "trace.json").write_text(json.dumps(record.pop("trace")))
    (tmp_path / "result.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="cannot be silently overridden"):
        review_run(tmp_path, "reviewer", "approved", "This should fail")


def test_inspect_unaided_mock_is_one_answer_without_provider(target, tmp_path, monkeypatch):
    import json
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import ChatMessageAssistant, ModelOutput, ModelUsage
    from wspbench.v2.contracts import Score
    from wspbench.v2.inspect_task import build_inspect_task
    module = types.ModuleType("wspbench.v2.scoring")
    module.score = lambda task, answer: Score(answer == {"patch": []}, {"fixture": 1.})
    monkeypatch.setitem(sys.modules, "wspbench.v2.scoring", module)
    monkeypatch.setenv("INSPECT_TRACE_FILE", str(tmp_path / "inspect-trace.log"))
    monkeypatch.setenv("INSPECT_CACHE_DIR", str(tmp_path / "inspect-cache"))
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("")
    record = {"actor_type": "protocol_smoke", "benchmark_result": False}
    evaluation = build_inspect_task(target, tmp_path / "run", {}, record, condition="A")
    output = ModelOutput(model="mockllm", usage=ModelUsage(input_tokens=3, input_tokens_cache_write=7,
        input_tokens_cache_read=2, output_tokens=5, reasoning_tokens=1, total_tokens=17),
        choices=[{"message": ChatMessageAssistant(content=json.dumps({"patch": []}))}])
    logs = inspect_eval(evaluation, model="mockllm/model", model_args={"custom_outputs": [output]},
                        log_dir=str(tmp_path / "inspect"), log_realtime=False, ctl_server=False)
    assert logs[0].status == "success"
    assert record["status"] == "completed"
    assert record["turns"] == 1 and record["success"]
    assert record["counters"]["analysis_calls"] == record["counters"]["example_evaluations"] == 0
    assert record["usage"]["input_tokens_total"] == 12
    assert record["usage"]["output_tokens"] == 5
    assert not record["benchmark_result"]
