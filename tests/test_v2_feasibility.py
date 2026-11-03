from copy import deepcopy

import numpy as np
import pytest

from wspbench.v2.feasibility import symbolic_effects, translation_closed_catalog
from wspbench.v2.networks import attention_architecture, compile_program, fit_readout, predictions, reparameterize
from wspbench.v2.patches import materialize
from wspbench.v2.programs import candidates, domain_inputs, evaluate_program


def _edits():
    return [
        {"id": "sham", "patch": [{"op": "scale", "tensor": "q", "factor": 1}]},
        {"id": "internal", "patch": [{"op": "scale", "tensor": "o", "factor": .67}]},
        {"id": "second", "patch": [{"op": "zero", "tensor": "w1", "axis": 0, "indices": [3, 4, 5]}]},
        {"id": "compose", "patch": [{"op": "scale", "tensor": "o", "factor": .67},
                                      {"op": "zero", "tensor": "w1", "axis": 0, "indices": [0, 1, 2]}]},
    ]


@pytest.mark.parametrize("trained", [False, True])
def test_symbolic_measure_forecasts_match_frozen_runtime_without_executing_it(trained, monkeypatch):
    architecture = attention_architecture(6)
    inputs = domain_inputs()
    for index, program in enumerate(candidates()):
        other = deepcopy(program)
        other["readout"] = {"op": "mod", "modulus": 3, "offset": index % 3}
        pair = [program, other]
        weights = compile_program(pair, architecture)
        if trained:
            labels = np.column_stack([evaluate_program(p, inputs) for p in pair])
            weights = fit_readout(architecture, weights, labels, inputs, np.random.default_rng(index), pair)
        weights = reparameterize(weights, np.random.default_rng(200 + index))
        baseline = predictions(architecture, weights, inputs)
        expected = {edit["id"]: np.mean(predictions(architecture, materialize(architecture, weights, {"patch": edit["patch"]}), inputs) != baseline, axis=0)
                    for edit in _edits()}
        # Runtime access is unavailable to the comparator, even during validation.
        with monkeypatch.context() as context:
            import wspbench.v2.networks as networks
            context.setattr(networks, "forward", lambda *a, **k: (_ for _ in ()).throw(AssertionError("target execution")))
            result = symbolic_effects(weights, architecture, _edits())
        for edit in _edits():
            assert np.max(np.abs(np.asarray(result["effects"][edit["id"]]) - expected[edit["id"]])) <= .01
        assert result["certificate"]["target_executions"] == 0
        assert all(value["mass"] == 363 for value in result["certificate"]["interventions"].values())


def test_static_reference_refuses_uncertified_token_attention():
    architecture = attention_architecture()
    weights = compile_program(candidates()[0], architecture)
    weights["k"][0, 0] = 1
    with pytest.raises(ValueError, match="Token-dependent"):
        symbolic_effects(weights, architecture, _edits()[:1])


def test_r3_label_catalog_contains_both_sides_of_every_declared_translation():
    from wspbench.v2.tasks import _shift
    original = candidates()
    covered = translation_closed_catalog(original)
    assert len(covered) == 24
    assert all(p in covered for p in original)
    assert all({**p, "readout": _shift(p["readout"])} in covered for p in original)
