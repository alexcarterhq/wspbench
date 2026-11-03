from __future__ import annotations
import json
import numpy as np
import pytest
from wspbench.v2 import networks,programs
from wspbench.v2.tasks import generate_suite,load_tasks,validate_suite
from wspbench.v2.scoring import score,probe,reference_submission
from wspbench.v2.patches import materialize
from wspbench.v2.serialization import load_weights,encode_weights,decode_weights,weights_digest


@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    root=tmp_path_factory.mktemp("v2-core")/"suite"
    generate_suite(root,dev_per_family=2,heldout_per_family=4,baseline_per_label=2)
    return root


def test_real_neural_compiler_exhaustive_and_serialized():
    x=programs.domain_inputs()
    assert x.shape==(363,5)
    arch=networks.attention_architecture()
    assert sum(np.prod(s) for s in arch["shapes"].values())<=4096
    for p in programs.candidates():
        w=networks.compile_program(p)
        restored=decode_weights(encode_weights(w))
        expected=programs.evaluate_program(p,x)
        assert np.array_equal(networks.predictions(arch,restored,x)[:,0],expected)
        changed=networks.reparameterize(w,np.random.default_rng(7))
        assert np.array_equal(networks.predictions(arch,changed,x)[:,0],expected)
        # Removing every numeric tensor destroys the implementation; no source-program runtime.
        zero=networks.zeros(arch)
        assert not np.array_equal(networks.predictions(arch,zero,x)[:,0],expected)


def test_suite_references_and_private_truth_stay_out_of_manifest(suite):
    validation=validate_suite(suite)
    assert validation["status"]=="passed",validation
    tasks=load_tasks(suite,"all")
    assert set(t.family for t in tasks)==set(("rule_recovery","causal_forecast","edit_effects","coupled_repair","compositional_synthesis","loop_repair"))
    for task in tasks:
        assert score(task,reference_submission(task)).success
        assert score(task,{"malformed":True}).status=="invalid"
        public=json.dumps(task.to_public_dict())
        assert 'reference_submission' not in public
        assert 'generation_seed' not in public
        assert 'clean.npz' not in public
        assert 'final_indices' not in public
        if task.family=="loop_repair":
            assert task.conditions==("C",)
            assert task.public["prompt_information"]["hint_level"]=="mechanism"


def test_trained_features_are_independent_fits_not_just_symmetries(suite):
    tasks=load_tasks(suite,"baseline_train")
    by_label={}
    for task in tasks:
        private=json.loads(task.private_path.read_text());training=private["training"]
        assert training["population_independence"]=="fresh_continuous_features_and_fit"
        assert task.target_stratum=="trained"
        by_label.setdefault(private["label"],[]).append(task)
    assert len(by_label)==len(programs.candidates())
    for matches in by_label.values():
        a,b=matches
        wa,wb=load_weights(a.weights_path),load_weights(b.weights_path)
        # Hidden knot multisets are invariant to permutations/scalings; these must differ.
        ka=np.sort(-wa["b0"]/wa["w0"][:,0]);kb=np.sort(-wb["b0"]/wb["w0"][:,0])
        assert not np.allclose(ka,kb)
        assert a.parent_problem_id!=b.parent_problem_id


def test_practice_and_final_banks_disjoint_and_immutable(suite):
    for task in load_tasks(suite,"dev"):
        with np.load(task.directory/"evaluation.npz") as bank:
            assert not set(bank["practice"])&set(bank["final"])
            assert sorted(np.r_[bank["practice"],bank["final"]].tolist())==list(range(363))
        with pytest.raises(ValueError):probe(task,load_weights(task.weights_path),[64])


def test_forecast_rejects_heldout_interventions_and_arbitrary_patch(suite):
    task=next(t for t in load_tasks(suite,"dev") if t.family=="causal_forecast")
    w=load_weights(task.weights_path)
    result=probe(task,w,[0,1],intervention_id="p1")
    assert result["accounting"]["example_evaluations"]==4
    assert len(result["baseline_outputs"])==2
    with pytest.raises(ValueError):probe(task,w,[0],intervention_id="f1")
    changed=materialize(task.architecture,w,{"patch":[{"op":"zero","tensor":"v"}]})
    with pytest.raises(ValueError):probe(task,changed,[0],intervention_id="p0")
    assert task.public["answer_schema"]["estimand"]=="exhaustive_domain_intervention_heldout"
    practice={weights_digest(materialize(task.architecture,w,{"patch":p["patch"]})) for p in task.public["practice_interventions"]}
    final={weights_digest(materialize(task.architecture,w,{"patch":p["patch"]})) for p in task.public["forecast_interventions"]}
    assert not practice&final


def test_coupled_fault_witnesses_identify_behavior_classes(suite):
    for task in load_tasks(suite,"heldout"):
        if task.family!="coupled_repair":continue
        w=load_weights(task.weights_path);clean=load_weights(task.directory/"clean.npz")
        with np.load(task.directory/"evaluation.npz") as bank:x=bank["inputs"]
        desired=networks.predictions(task.architecture,clean,x)
        compatible=[]
        for h in task.public["fault_hypotheses"]:
            edited=materialize(task.architecture,w,{"patch":h["patch"]})
            pred=networks.predictions(task.architecture,edited,x)
            if all(np.array_equal(networks.predictions(task.architecture,edited,[e["input"]])[0],e["expected"]) for e in task.public["normative_examples"]):
                compatible.append(pred)
        assert compatible
        assert all(np.array_equal(pred,desired) for pred in compatible)
        assert len(task.public["fault_hypotheses"])<=8
        for example in task.public["normative_examples"]:
            result=probe(task,w,[example["practice_index"]])
            assert result["inputs"]==[example["input"]]
            assert result["expected"]==[example["expected"]]


def test_edit_effect_boundary_layers_identical_and_semantic_changes(suite):
    changed=0
    for task in load_tasks(suite,"heldout"):
        if task.family!="edit_effects":continue
        before=load_weights(task.directory/"before.npz");after=load_weights(task.weights_path)
        for name in task.architecture["boundary_tensors"]:assert np.array_equal(before[name],after[name])
        reference=reference_submission(task)
        assert score(task,reference).metrics["exhaustive_agreement"]==1
        wrong={"before_program":reference["before_program"],"after_program":reference["before_program"]}
        if not score(task,wrong).success:changed+=1
    assert changed


def test_loop_controls_require_correct_tokens_not_just_termination(suite):
    task=next(t for t in load_tasks(suite,"dev") if t.family=="loop_repair" and not json.loads(t.private_path.read_text())["clean_control"])
    early_eos={"patch":[{"op":"zero","tensor":"out"},{"op":"set","tensor":"ob","index":[3],"value":10}]}
    result=score(task,early_eos)
    assert not result.success
    assert result.metrics["completion_rate"]==1
    assert result.metrics["truncation_rate"]==1
    assert score(task,reference_submission(task)).metrics["valid_repetition_accuracy"]==1
    observed=probe(task,load_weights(task.weights_path),[0,1],observation="activations")
    assert observed["accounting"]["episodes"]==2
    assert observed["accounting"]["example_evaluations"]==0
    assert observed["accounting"]["activation_scalars"]==observed["steps"]*24


def test_validator_rejects_equivalent_interventions_in_different_numeric_encodings(tmp_path):
    root=tmp_path/"numeric-alias"
    generate_suite(root,dev_per_family=1,heldout_per_family=0,baseline_per_label=0)
    task=next(t for t in load_tasks(root,"dev") if t.family=="causal_forecast")
    weights=load_weights(task.weights_path)
    changed=materialize(task.architecture,weights,{"patch":task.public["practice_interventions"][0]["patch"]})
    path=task.directory/"task.json"
    manifest=json.loads(path.read_text())
    # A replace encoding aliases the practice scale operation exactly.
    manifest["public"]["forecast_interventions"][0]["patch"]=[{"op":"replace","tensor":"q","values":changed["q"].tolist()}]
    path.write_text(json.dumps(manifest))
    result=validate_suite(root)
    assert result["status"]=="failed"
    assert any("Equivalent numeric interventions" in failure["reason"] for failure in result["failures"])
