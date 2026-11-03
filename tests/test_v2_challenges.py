import numpy as np
import pytest

from wspbench.v2 import challenges as c


def test_fixed_architecture_and_procedural_functions_are_exact():
    arch=c.architecture()
    assert sum(np.prod(s) for s in arch["shapes"].values()) == 4049
    x=c.domain_inputs()
    fingerprints=set()
    for seed in range(8):
        program=c.sample_program(seed)
        weights=c.compile_program(program)
        expected=c.evaluate_program(program,x)
        assert np.array_equal(c.predictions(arch,weights,x)[:,0],expected)
        fingerprints.add(expected.tobytes())
    assert len(fingerprints)==8


def test_weights_only_derivation_never_uses_runtime(monkeypatch):
    x=c.domain_inputs()
    for seed in range(8):
        program=c.sample_program(seed)
        weights=c.compile_program(program)
        with monkeypatch.context() as patch:
            patch.setattr(c,"forward",lambda *a,**k: (_ for _ in ()).throw(AssertionError("target executed")))
            result=c.derive_program(weights)
        assert np.array_equal(c.evaluate_program(result["program"],x),c.evaluate_program(program,x))
        assert result["certificate"]["target_executions"]==0


def test_internal_edits_preserve_boundary_and_change_semantics():
    from collections import defaultdict,Counter
    balance=defaultdict(Counter)
    for index in range(8):
        task=c.make_challenge(100+index,"edit_effects",index=index)
        assert all(np.array_equal(task["before"][k],task["weights"][k]) for k in task["architecture"]["boundary_tensors"])
        before=c.predictions(task["architecture"],task["before"],task["inputs"])
        balance[task["private"]["program"]["op"]][bool(np.array_equal(before,task["expected"]))]+=1
    assert len(balance)==4
    assert all(counts[True]==counts[False]==1 for counts in balance.values())


def test_synthesis_public_table_complete_gold_program_private():
    task=c.make_challenge(9,"compositional_synthesis",index=1)
    table=task["public"]["specification"]
    assert len(table["inputs"])==len(table["outputs"])==1024
    assert "program" not in task["public"] and "program_candidates" not in task["public"]
    assert all(not np.any(v) for v in task["weights"].values())
    assert np.array_equal(c.predictions(task["architecture"],task["clean"],table["inputs"])[:,0],table["outputs"])


def test_activation_bound_covers_every_returned_scalar():
    task=c.make_challenge(47,"rule_recovery",index=1)
    _,states=c.forward(task["architecture"],task["weights"],task["inputs"][:7],True)
    assert sum(v.size for v in states.values()) == 7*task["public"]["activation_scalars_per_example"]
    assert task["public"]["activation_scalars_per_example"] == 136


def test_derivation_rejects_unsupported_coefficients_and_readouts():
    weights=c.compile_program(c.sample_program(18))
    weights["w2"][10,0] += np.float32(1e-5)
    with pytest.raises(ValueError):c.decode_program(weights)
    weights=c.compile_program(c.sample_program(18))
    weights["out"][2,0] = 1
    with pytest.raises(ValueError):c.decode_program(weights)


@pytest.mark.parametrize("bad", [{"op":"input","index":True},{"op":"eval","source":"1"},{"op":"majority","args":[]}])
def test_boolean_grammar_rejects_malformed_answers(bad):
    with pytest.raises(ValueError):c.evaluate_program(bad,c.domain_inputs())
