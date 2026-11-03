import json

import pytest

from wspbench.v2.admission import assess_admission
from wspbench.v2.reporting import build_report, paired_difference


def _record(parent, mechanism, condition, success, **changes):
    result = {
        "actor_id": "model-example", "actor_type": "model", "condition": condition,
        "weight_view": "full", "task_id": "private-" + parent, "parent_problem_id": parent,
        "mechanism_family_id": mechanism, "family": "rule_recovery", "skill": "read",
        "target_stratum": "constructed", "capacity_tier": "small", "split": "heldout",
        "success": success, "status": "scored", "attempt": 1, "metrics": {},
        "target_digest": "same-target-" + parent, "question_digest": "same-question-" + parent,
        "protocol_digest": "same-matched-protocol", "usage": {},
    }
    result.update(changes)
    return result


def test_report_clusters_independent_mechanisms_and_never_exposes_parent_ids():
    records = [_record("hidden-parent-a", "mechanism-one", "B", True),
               _record("hidden-parent-b", "mechanism-one", "B", True),
               _record("hidden-parent-c", "mechanism-two", "B", False)]
    report = build_report(records)
    row = report["tables"][0]
    assert row["score"] == .5
    assert row["n_parents"] == 3 and row["n_mechanisms"] == 2
    assert not report["headline_eligible"]
    encoded = json.dumps(report)
    assert "hidden-parent" not in encoded
    assert "private-hidden" not in encoded


def test_paired_access_effect_requires_same_question_target_and_attempts():
    records = [_record("p", "m", "A", False), _record("p", "m", "B", True)]
    assert paired_difference(records, "A", "B")["difference"] == 1
    records[1]["target_digest"] = "other-network"
    assert paired_difference(records, "A", "B")["difference"] is None
    records[1]["target_digest"] = records[0]["target_digest"]
    records[1]["attempt"] = 2
    assert paired_difference(records, "A", "B")["difference"] is None


def test_oracle_and_c_only_extension_cannot_create_access_effects():
    records = [_record("p", "m", "A", False, actor_type="reference"),
               _record("p", "m", "B", True, actor_type="reference"),
               _record("q", "m2", "B", False, family="loop_repair"),
               _record("q", "m2", "C", True, family="loop_repair")]
    assert paired_difference(records, "A", "B")["n_parents"] == 0
    assert paired_difference(records, "B", "C")["n_parents"] == 0


def test_all_pass_bootstrap_reports_degeneracy_without_claiming_certainty():
    row = build_report([_record("a", "m1", "B", True), _record("b", "m2", "B", True)])["tables"][0]
    assert row["degenerate_bootstrap"]
    assert row["bounded_mean_95_interval"][0] < 1
    assert row["ci_low"] == row["ci_high"] == 1


def test_cpu_reference_passes_do_not_admit_a_headline_benchmark(tmp_path):
    rows = [{"actor": "oracle", "actor_type": "reference", "condition": "B", "family": "rule_recovery",
             "score": 1., "n_parents": 4, "n_mechanisms": 4, "split": "dev", "access_card": {"privileged": True}}]
    result = assess_admission(tmp_path, {"tables": rows}, {"status": "passed"})
    assert not result["headline_admitted"] and not result["paid_runs_permitted"]
    assert next(f for f in result["families"] if f["family"] == "rule_recovery")["status"] != "sanity_only"
    rows[0].update(actor="symbolic-reader", actor_type="baseline", access_card={"privileged": False})
    result = assess_admission(tmp_path, {"tables": rows}, {"status": "passed"})
    assert next(f for f in result["families"] if f["family"] == "rule_recovery")["status"] == "sanity_only"


def test_failed_reference_validation_blocks_candidates(tmp_path):
    result = assess_admission(tmp_path, {"tables": []}, {"status": "failed"})
    assert not result["cpu_validation_passed"]
    assert all(f["status"] == "blocked" for f in result["families"])


def test_statistics_reader_requires_separate_balanced_population():
    import numpy as np
    import pytest
    from wspbench.v2.baselines import StatisticalReader
    population = [{"split": "baseline_train", "parent_problem_id": f"{label}-{i}",
                   "label": {"class": label}, "weights": {"w": np.full((2, 3), label + .001*i)}}
                  for label in (0, 1) for i in range(32)]
    reader = StatisticalReader().fit(population, required_labels=[{"class": 0}, {"class": 1}])
    assert reader.predict({"w": np.ones((2, 3))}) == {"class": 1}
    with pytest.raises(ValueError, match="exclusively"):
        StatisticalReader().fit([{**population[0], "split": "heldout"}], required_labels=[{"class": 0}])
    with pytest.raises(ValueError, match="catalog"):
        StatisticalReader().fit(population[:32], required_labels=[{"class": 0}, {"class": 1}])
    with pytest.raises(ValueError, match="32"):
        StatisticalReader().fit(population[1:32]+population[33:], required_labels=[{"class": 0}, {"class": 1}])


def test_circuit_algebra_invariant_and_independent_lookup_exact():
    import numpy as np
    from wspbench.v2.baselines import _attention_signature, _lookup_circuit, _recover
    from wspbench.v2.networks import attention_architecture, compile_program, predictions, reparameterize
    from wspbench.v2.programs import candidates, domain_inputs, evaluate_program
    architecture = attention_architecture()
    program = candidates()[7]
    original = compile_program(program)
    transformed = reparameterize(original, np.random.default_rng(9))
    assert np.allclose(_attention_signature(original), _attention_signature(transformed), atol=1e-5)
    recovered, distance = _recover(transformed, {"program_candidates": candidates()}, architecture)
    assert recovered == program
    assert distance < 1e-5
    independent = _lookup_circuit(program, architecture)
    assert not np.array_equal(independent["e"], original["e"])
    assert np.array_equal(predictions(architecture, independent, domain_inputs()).reshape(-1),
                          evaluate_program(program, domain_inputs()))


def test_hint_information_must_match_for_access_contrast():
    records = [_record("p", "m", "A", False), _record("p", "m", "B", True)]
    records[0]["prompt_information"] = {"hint_level": "neutral"}
    records[1]["prompt_information"] = {"hint_level": "mechanism"}
    assert paired_difference(records, "A", "B")["n_parents"] == 0
    assert len(build_report(records)["tables"]) == 2


def test_c_repair_scripts_share_metered_probes_and_never_query_final_score(tmp_path, monkeypatch):
    from wspbench.v2 import scoring
    from wspbench.v2.access import AccessSession
    from wspbench.v2.baselines import run_baseline
    from wspbench.v2.tasks import generate_suite, load_tasks
    generate_suite(tmp_path, dev_per_family=2, heldout_per_family=0, baseline_per_label=0)
    tasks = load_tasks(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("hidden final scorer consulted during baseline construction")
    monkeypatch.setattr(scoring, "score", forbidden)
    for family, baseline in [("coupled_repair", "generic_repair"), ("loop_repair", "contrastive_mean")]:
        for task in [t for t in tasks if t.family == family]:
            session = AccessSession(task, "C")
            result = run_baseline(session, baseline)
            assert result["status"] == "completed"
            session.submit(result["submission"])
            trace = session.finish()
            counters = trace["counters"]
            assert counters["candidate_patches"] <= task.limits.candidate_patches
            assert counters["example_evaluations"] <= task.limits.example_evaluations
            assert counters["episodes"] <= task.limits.episodes
            assert counters["activation_scalars"] <= task.limits.activation_scalars
            assert any(event["tool"] == "probe" for event in trace["trace"])


def test_independent_fit_ids_do_not_inflate_semantic_training_clusters():
    import numpy as np
    from wspbench.v2.baselines import StatisticalReader
    rows = [{"split": "baseline_train", "parent_problem_id": "one-semantic-rule",
             "independent_network_id": f"fresh-fit-{i}", "label": {"class": 0},
             "weights": {"w": np.array([[i/100.]])}} for i in range(32)]
    reader = StatisticalReader().fit(rows, required_labels=[{"class": 0}])
    assert reader.training["independently_fitted_networks"] == 32
    assert reader.training["semantic_problem_clusters"] == 1


def test_no_information_methods_never_read_disclosed_weights(tmp_path):
    from wspbench.v2.access import AccessSession
    from wspbench.v2.baselines import BASELINES, run_baseline
    from wspbench.v2.tasks import generate_suite, load_tasks
    class NoInformativeWeights(AccessSession):
        @property
        def view_weights(self):
            raise AssertionError("no-information comparator read target tensors")
        def checkpoint_weights(self, checkpoint="target"):
            raise AssertionError("no-information comparator read checkpoint tensors")
    generate_suite(tmp_path, dev_per_family=2, heldout_per_family=0, baseline_per_label=0)
    for task in load_tasks(tmp_path):
        for name, card in BASELINES.items():
            if card["weights"] or task.family not in card["families"]:
                continue
            session = NoInformativeWeights(task, "C")
            outcome = run_baseline(session, name)
            assert outcome["status"] in {"completed", "not_applicable"}


def test_r3_statistical_transfer_does_not_complete_qualifying_reader_coverage(tmp_path):
    from wspbench.v2.baselines import access_card
    rows = [{"actor": name, "actor_type": "baseline", "condition": "B", "family": "edit_effects",
             "score": .5, "n_parents": 4, "n_mechanisms": 4, "split": "dev", "access_card": access_card(name)}
            for name in ("metadata_prior", "circuit_algebra", "behavior_only", "statistical_reader")]
    result = assess_admission(tmp_path, {"tables": rows}, {"status": "passed"})
    r3 = next(f for f in result["families"] if f["family"] == "edit_effects")
    assert "statistical_reader" in r3["missing_baselines"]
    assert not r3["baseline_coverage_complete"]
