from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from experiments.report_gpt_comparison import build_comparison,_markdown,_csv,_public_canary
from wspbench.v2.serialization import serialization_digest


@pytest.fixture
def cohort():
    models=[{"id":f"model-{letter}","provider":"openai","model":f"openai/gpt-fixture-{letter}",
             "snapshot":"alias","snapshot_kind":"provider_alias","generate_config":{"reasoning_effort":"high"},
             "api_key":"SECRET_API_KEY_NOT_FOR_EXPORT"} for letter in "abc"]
    families=[("rule_recovery","read"),("rule_recovery","read"),("edit_effects","read"),
              ("edit_effects","read"),("compositional_synthesis","write"),("compositional_synthesis","write")]
    metadata={f"PRIVATE_TASK_{i}":{"family":family,"skill":skill,"parent_problem_id":f"PRIVATE_PARENT_{i}",
                                 "mechanism_family_id":f"PRIVATE_MECHANISM_{i}"} for i,(family,skill) in enumerate(families)}
    expected=[];records=[]
    for model in models:
        for i,(task_id,meta) in enumerate(metadata.items()):
            item={"model_id":model["id"],"model_config_digest":serialization_digest(model),"task_id":task_id,
                  "parent_problem_id":meta["parent_problem_id"],"target_digest":f"{i:064x}",
                  "condition":"A","weight_view":"full","attempt":1}
            expected.append(item)
            metric={"functional_agreement":1.} if meta["family"]=="rule_recovery" else {"before_agreement":1.,"after_agreement":1.} if meta["family"]=="edit_effects" else {"target_accuracy":1.}
            records.append({**item,**meta,"actor_type":"model","status":"completed","success":True,
                "protocol_digest":"protocol","suite_digest":"suite","harness_digest":"harness",
                "question_digest":f"{i+20:064x}","serialization_digest":f"{i+40:064x}",
                "audit":{"status":"approved"},"turns":1,"submitted":True,"usage_unknown":False,
                "usage":{"input_tokens":10,"input_tokens_cache_write":2,"input_tokens_cache_read":3,
                         "input_tokens_total":15,"output_tokens":5,"reasoning_tokens":2,"total_tokens":20},
                "cost_usd":.02,"maximum_reserved_usd":.3,"wall_seconds":2.,"cpu_seconds":.01,
                "metrics":metric,"submission":{"private_answer":"SECRET_SOLUTION_NOT_FOR_EXPORT"},
                "_completion":"{\"private_answer\":\"SECRET_SOLUTION_NOT_FOR_EXPORT\"}"})
    manifest={"phase":"calibration","benchmark_version":"0.2.0","models":models,"expected_matrix":expected,
              "protocol_digest":"protocol","suite_digest":"suite","harness_digest":"harness"}
    return manifest,records,metadata


def build(cohort):
    manifest,records,metadata=cohort
    return build_comparison(manifest,records,task_metadata=metadata)


def skill(report,model="model-a",name="read"):
    entry=next(row for row in report["models"] if row["configuration"]["id"]==model)
    return next(row for row in entry["skills"] if row["skill"]==name)


def test_paired_development_report_separates_read_and_write_without_a_grand_score(cohort):
    report=build(cohort)
    assert report["status"]=="complete" and report["matrix_complete"]
    assert report["planned_sessions"]==report["completed_scored_sessions"]==18
    assert skill(report)["requested_sessions"]==4
    assert skill(report,name="write")["requested_sessions"]==2
    for model in report["models"]:
        assert "exact_success" not in model["traffic_and_resources"]
        assert "functional_accuracy" not in model["traffic_and_resources"]
        assert all(row["requested_sessions"]==2 for row in model["families"])
    assert not report["headline_eligible"] and report["website"]=="deferred"


def test_infrastructure_is_not_a_zero_and_unknown_cost_is_not_free(cohort):
    _,records,_=cohort
    bad=records[0]
    bad.update(status="infrastructure_failure",success=False,usage_unknown=True,cost_usd=None,
               turns=None,submitted=False,_completion=None,usage={},metrics={})
    report=build(cohort);read=skill(report)
    assert read["scored_sessions"]==3 and read["infrastructure_failures"]==1
    assert read["exact_success"]["mean"]==1
    assert read["resources"]["unknown_cost_sessions"]==1
    assert read["resources"]["finalized_cost_estimate_usd"] is None
    assert read["resources"]["known_cost_estimate_usd"]==pytest.approx(.06)
    assert read["resources"]["unknown_reserved_upper_bound_usd"]==.3
    pair=next(p for p in report["paired_model_differences"] if p["left_model"]=="model-a" and p["right_model"]=="model-b" and p["scope_type"]=="family" and p["scope"]=="rule_recovery")
    assert pair["paired_cases"]==1 and pair["incomplete_or_infrastructure_pairs"]==1


def test_invalid_submissions_and_rejected_protocol_attempts_get_zero_credit(cohort):
    _,records,_=cohort
    records[0].update(status="invalid_submission",success=True,submitted=False,_completion="not JSON",metrics={})
    records[1]["audit"]["status"]="rejected"
    report=build(cohort);read=skill(report)
    assert read["scored_sessions"]==4 and read["exact_successes"]==2
    assert read["exact_success"]["mean"]==.5
    assert read["functional_accuracy"]["mean"]==.5
    assert read["invalid_submissions"]==1 and read["protocol_violations"]==1
    assert read["valid_json_objects"]==3 and read["invalid_json_objects"]==1


def test_all_infrastructure_failure_has_no_capability_rate(cohort):
    _,records,_=cohort
    for r in records[:6]:r.update(status="infrastructure_failure",usage_unknown=True,cost_usd=None,turns=None,submitted=False,_completion=None)
    report=build(cohort);read=skill(report)
    assert read["scored_sessions"]==0 and read["exact_success"]["mean"] is None
    assert read["functional_accuracy"]["mean"] is None
    assert "unavailable" in _markdown(report)


def test_failure_categories_are_counted_without_exporting_raw_provider_errors(cohort):
    _,records,_=cohort
    records[0].update(status="infrastructure_failure",usage_unknown=True,cost_usd=None,
                      error="RuntimeError: APIConnectionError SECRET_PROVIDER_ERROR")
    records[1].update(status="timeout",usage_unknown=True,cost_usd=None)
    records[2].update(status="invalid_submission",_completion="not JSON",metrics={})
    report=build(cohort)
    outcomes=report["models"][0]["traffic_and_resources"]["outcome_counts"]
    assert outcomes=={"infrastructure_connection":1,"infrastructure_timeout":1,"invalid_json":1,"correct":3}
    assert "SECRET_PROVIDER_ERROR" not in json.dumps(report)
    records[0]["status"]="silently_unrecognized"
    with pytest.raises(ValueError,match="unrecognized session status"):build(cohort)


def test_missing_and_running_cases_are_visible_without_zero_credit(cohort):
    _,records,_=cohort
    records.pop()
    records[0].update(status="running",usage_unknown=True,cost_usd=None,turns=None,submitted=False,_completion=None)
    report=build(cohort)
    assert report["missing_sessions"]==1 and report["inflight_sessions"]==1
    assert report["completed_scored_sessions"]==16
    assert report["infrastructure_failures"]==0 and not report["matrix_complete"]


def test_comparison_pairs_require_same_question_and_checkpoints(cohort):
    _,records,_=cohort
    records[6]["question_digest"]="different prompt"
    report=build(cohort)
    pair=next(p for p in report["paired_model_differences"] if p["left_model"]=="model-a" and p["right_model"]=="model-b" and p["scope_type"]=="family" and p["scope"]=="rule_recovery")
    assert pair["paired_cases"]==1 and pair["excluded_input_mismatch_pairs"]==1
    assert not report["paired_inputs_match"] and report["status"]=="partial"


def test_wrong_target_or_duplicate_record_cannot_become_a_comparison(cohort):
    original=deepcopy(cohort)
    cohort[1][0]["target_digest"]="wrong network"
    with pytest.raises(ValueError,match="target"):build(cohort)
    original[1].append(deepcopy(original[1][0]))
    with pytest.raises(ValueError,match="duplicate"):build(original)


def test_private_ids_seeds_and_solutions_never_enter_public_outputs(cohort):
    cohort[1][0]["generation_seed"]=987654321
    report=build(cohort)
    exported=json.dumps(report)+_markdown(report)+_csv(report)
    for marker in ("PRIVATE_TASK_","PRIVATE_PARENT_","PRIVATE_MECHANISM_","SECRET_API_KEY_NOT_FOR_EXPORT","SECRET_SOLUTION_NOT_FOR_EXPORT","generation_seed","987654321"):
        assert marker not in exported


def test_token_categories_and_reasoning_are_not_double_counted(cohort):
    report=build(cohort)
    resources=report["models"][0]["traffic_and_resources"]["resources"]
    tokens=resources["tokens"]
    assert tokens["input_tokens_total"]==90 and tokens["output_tokens"]==30
    assert tokens["reasoning_tokens"]==12 and tokens["total_tokens"]==120
    assert resources["api_reported_output_tokens_per_session_second"]["median"]==2.5
    assert resources["visible_answer_tokens_per_session_second"]["median"]==1.5


def test_json_syntax_and_submission_contract_are_separate(cohort):
    row=cohort[1][4]
    row.update(status="invalid",success=False,_completion='{"patch":[{"op":"unknown"}]}',metrics={})
    report=build(cohort);write=skill(report,name="write")
    assert write["valid_json_objects"]==2 and write["invalid_submissions"]==1
    assert write["exact_successes"]==1


def test_canaries_are_whitelisted_and_excluded_from_capability(cohort):
    raw={"benchmark_result":False,"model_id":"model-a","status":"completed","generation_requests":1,
         "sdk_retries":0,"wall_seconds":2.9,"cost_usd":.000004,"maximum_reserved_usd":.01,
         "usage":{},"request":{"private":"SECRET_CANARY"},"text":"SECRET_CANARY","response_id":"SECRET_CANARY"}
    canary=_public_canary(raw)
    manifest,records,metadata=cohort
    report=build_comparison(manifest,records,task_metadata=metadata,canaries=[canary],
                            legacy_exposure={"billing_reconciled":False,"reserved_upper_bound_usd":472.8})
    assert report["completed_scored_sessions"]==18
    assert report["legacy_exposure"]["reserved_upper_bound_usd"]==472.8
    assert "SECRET_CANARY" not in json.dumps(report)


def test_nonidentical_predeclared_model_cases_and_mock_outputs_are_rejected(cohort):
    original=deepcopy(cohort)
    cohort[0]["expected_matrix"].pop()
    with pytest.raises(ValueError,match="same predeclared"):build(cohort)
    original[1][0]["mock"]=True
    with pytest.raises(ValueError,match="Mock"):build(original)


def test_write_no_edit_control_exposes_class_imbalance_without_adding_model_trials(cohort):
    manifest,records,metadata=cohort
    controls=[]
    for index,row in enumerate(records[4:6]):
        metadata[row["task_id"]]["baseline_question_digest"]=row["question_digest"]
        controls.append({**row,"actor_id":"metadata_prior","actor_type":"baseline","condition":"B",
                         "success":False,"metrics":{"target_accuracy":.7+.1*index,"affected_accuracy":0.,
                         "retention_accuracy":1.,"retention_loss":0.,"edit_fraction":0.,"edit_norm":0.}})
    report=build_comparison(manifest,records,task_metadata=metadata,write_controls=controls)
    control=report["write_difficulty_control"]
    assert control["selected_targets"]==2 and control["exact_successes"]==0
    assert control["target_accuracy"]["mean"]==pytest.approx(.75)
    assert control["affected_accuracy"]["mean"]==0
    assert control["retention_accuracy"]["mean"]==1
    assert not control["same_access_model_score"] and control["independent_tasks_added"]==0
    assert report["planned_sessions"]==report["completed_scored_sessions"]==18
    exported=json.dumps(report)+_markdown(report)
    assert "Class imbalance" in exported and "PRIVATE_TASK_" not in exported
    controls[0]["target_digest"]="another target"
    with pytest.raises(ValueError,match="does not match"):
        build_comparison(manifest,records,task_metadata=metadata,write_controls=controls)
    controls[0]["target_digest"]=records[4]["target_digest"]
    controls[0]["question_digest"]="another zero-start function specification"
    with pytest.raises(ValueError,match="does not match"):
        build_comparison(manifest,records,task_metadata=metadata,write_controls=controls)


def test_writer_reads_saved_traces_and_keeps_an_unfinished_reserved_attempt_inflight(cohort,tmp_path,monkeypatch):
    from experiments import report_gpt_comparison as reporter
    import wspbench.v2.tasks as tasks_module

    manifest,records,metadata=cohort
    runs=tmp_path/"runs";calibration=runs/"calibration";calibration.mkdir(parents=True)
    (calibration/"manifest.json").write_text(json.dumps(manifest))
    tasks=[SimpleNamespace(id=task_id,limits=None,**meta) for task_id,meta in metadata.items()]
    monkeypatch.setattr(tasks_module,"load_tasks",lambda *_:tasks)
    monkeypatch.setattr(reporter,"maximum_session_cost",lambda *_:.3)
    for index,record in enumerate(records[:-1]):
        directory=calibration/str(index);directory.mkdir()
        raw={key:value for key,value in record.items() if key!="_completion"}
        (directory/"result.json").write_text(json.dumps(raw))
        trace=[{"tool":"model_turn","output":{"choices":[{"message":{"content":[
            {"type":"text","text":record["_completion"]}]}}]}}]
        (directory/"trace.json").write_text(json.dumps(trace))
    item=manifest["expected_matrix"][-1]
    run_id=serialization_digest({"protocol":manifest["protocol_digest"],"model":item["model_config_digest"],
        "parent":item["parent_problem_id"],"task":item["task_id"],"target":item["target_digest"],
        "condition":item["condition"],"view":item["weight_view"],"attempt":item["attempt"]})
    pending=calibration/run_id;pending.mkdir();(pending/"prompt.json").write_text('{"private":"SECRET_PROMPT"}')
    (runs/"continuation-authorization.private.json").write_text(json.dumps({
        "quarantined_holds":[{"reservation_id":"SECRET_RESERVATION","reserved_usd":472.8}],
        "authorization_digest":"digest"}))
    destination=tmp_path/"public"
    report=reporter.write_comparison(runs,destination,data_root=tmp_path/"data")
    assert report["launched_sessions"]==18 and report["completed_scored_sessions"]==17
    assert report["inflight_sessions"]==1 and report["infrastructure_failures"]==0
    assert report["unknown_cost_sessions"]==1 and report["status"]=="partial"
    assert report["legacy_exposure"]["reserved_upper_bound_usd"]==472.8
    assert len(report["reporter_digest"])==64
    exported="".join(path.read_text() for path in destination.iterdir())
    for marker in ("PRIVATE_TASK_","SECRET_PROMPT","SECRET_RESERVATION","SECRET_SOLUTION_NOT_FOR_EXPORT"):
        assert marker not in exported
    assert {path.suffix for path in destination.iterdir()}=={".json",".md",".csv"}
