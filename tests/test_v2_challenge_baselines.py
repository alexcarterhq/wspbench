from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pytest

from wspbench.v2.challenge_baselines import (StructuredReader, _anf_program, _bdd_program,
    _cover_program, _domain, _learn_table, _synthesize, construct_relu)
from wspbench.v2.challenges import architecture, evaluate_program, predictions


def _input(index):
    return {"op": "input", "index": index}


@pytest.mark.parametrize("op", ["and", "or", "xor", "nand", "nor", "not", "mux", "majority"])
def test_independent_numeric_constructor_handles_every_public_gate(op, monkeypatch):
    node = {"op": op}
    if op == "not": node["arg"] = _input(0)
    elif op == "mux": node.update(condition=_input(0), then=_input(1), **{"else": _input(2)})
    elif op == "majority": node["args"] = [_input(i) for i in range(3)]
    else: node.update(left=_input(0), right=_input(1))
    # A generating-compiler trap ensures the competitor's construction is independent.
    import wspbench.v2.challenges as challenges
    monkeypatch.setattr(challenges, "compile_program", lambda *a, **k: (_ for _ in ()).throw(AssertionError("private compiler")))
    weights = construct_relu(node, architecture())
    assert sum(value.size for value in weights.values()) == 4049
    assert np.array_equal(predictions(architecture(), weights, _domain())[:, 0], evaluate_program(node, _domain()))


@pytest.mark.parametrize("method", ["bdd_constructor", "anf_constructor", "dnf_constructor"])
def test_table_constructor_gets_no_gold_program_and_builds_exact_xor(method):
    table = np.bitwise_xor(_domain()[:, 2], _domain()[:, 6])
    public = {"specification": {"format": "truth_table", "inputs": _domain().tolist(), "outputs": table.tolist()}}
    result = _synthesize(public, architecture(), method)
    assert not result["diagnostics"].get("capacity_refusal")
    from wspbench.v2.patches import materialize
    initial = {name: np.zeros(shape, np.float32) for name, shape in architecture()["shapes"].items()}
    weights = materialize(architecture(), initial, result["submission"])
    assert np.array_equal(predictions(architecture(), weights, _domain())[:, 0], table)
    assert not result["diagnostics"]["private_program_used"]


def test_constructor_refuses_depth_and_width_before_returning_artifact():
    node = _input(0)
    for _ in range(5): node = {"op": "not", "arg": node}
    with pytest.raises(ValueError, match="four-stage"):
        construct_relu(node, architecture())


def test_behavior_learner_recovers_parity_from_practice_and_never_needs_weights():
    x = _domain()[np.random.default_rng(5).permutation(1024)[:128]]
    y = x[:, 1] ^ x[:, 5] ^ x[:, 9]
    table, diagnostic = _learn_table(x.tolist(), y.tolist())
    assert np.array_equal(table, _domain()[:, 1] ^ _domain()[:, 5] ^ _domain()[:, 9])
    assert diagnostic["practice_examples"] == 128


def test_structured_reader_has_unrestricted_binary_table_outputs_and_training_guard():
    examples = [{"split": "baseline_train", "independent_network_id": f"network-{i}",
                 "mechanism_family_id": f"pattern-{i % 8}",
                 "weights": {"w": np.full((2, 2), i % 2 + i / 10000)},
                 "outputs": [i % 2] * 1024} for i in range(128)]
    reader = StructuredReader().fit(examples)
    assert reader.predict({"w": np.ones((2, 2))}) == [1] * 1024
    assert reader.training["heldout_labels_used"] is False
    assert reader.training["output_coordinates"] == 1024
    leaked = deepcopy(examples); leaked[0]["split"] = "heldout"
    with pytest.raises(ValueError, match="baseline_train"):
        StructuredReader().fit(leaked)
