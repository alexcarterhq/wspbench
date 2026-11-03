from copy import deepcopy
import json
import sqlite3
from types import SimpleNamespace

import pytest

from wspbench.budget import BudgetExceeded,BudgetLedger
from wspbench.v2 import continuation as c
from wspbench.v2.contracts import Limits
from wspbench.v2.serialization import serialization_digest


@pytest.fixture
def scope(tmp_path,monkeypatch):
    ledger=BudgetLedger(tmp_path/"budget.sqlite3")
    holds=[]
    for ident,amount in (("old-main",3.6),("old-retry-1",3.6),("old-retry-2",3.6),("audited-probe-exposure",4.2)):
        ledger.reserve("calibration",amount,ident)
        holds.append({"reservation_id":ident,"phase":"calibration","reserved_usd":amount})
    ledger.reserve("diagnostics",.05,"known-canary");ledger.settle("known-canary",.05)
    incident=tmp_path/"incident.private.json"
    incident.write_text(json.dumps({"usage_received":False,"reservation_holds":holds,"total_unknown_charge_hold_usd":15.0}))
    models=[{"id":f"model-{i}","provider":"openai","model":f"openai/model-{i}",
             "snapshot":f"snapshot-{i}","pricing":{"input_per_million":1.,"cache_write_per_million":1.,
               "cache_read_per_million":.1,"output_per_million":1.,"verified_at":"fixture","source":"fixture"}} for i in range(3)]
    config=tmp_path/"models.json";config.write_text(json.dumps({"models":models}))
    cells=[{"task_id":f"task-{i}","condition":"A","weight_view":"full","target_digest":str(i)*64} for i in range(6)]
    data=tmp_path/"data";data.mkdir()
    out=tmp_path/"new-comparison"
    monkeypatch.setattr(c,"_scope_digests",lambda root:("suite","harness"))
    from wspbench.v2 import tasks
    monkeypatch.setattr(tasks,"load_tasks",lambda *a,**k:[SimpleNamespace(id=f"task-{i}",limits=Limits()) for i in range(6)])
    return SimpleNamespace(ledger=ledger,holds=holds,incident=incident,models=models,config=config,cells=cells,data=data,out=out)


def authorize(s,**options):
    return c.write_continuation_authorization(s.data,s.out,s.config,ledger=s.ledger,cells=s.cells,
        model_configs=s.models,incident_path=s.incident,reviewer="independent-fixture-reviewer",
        user_instruction=c.USER_INSTRUCTION,evidence=["reviewed-incident-and-renewed-user-request"],
        prior_diagnostic_spend_usd=.05,**options)


def validate(s,**changes):
    values={"ledger":s.ledger,"model_configs":s.models,"cells":s.cells,"phase":"calibration"}
    values.update(changes)
    return c.validate_continuation(s.data,s.out,s.config,**values)


def test_renewed_comparison_retains_every_old_unknown_charge(scope):
    s=scope;auth=authorize(s)
    assert validate(s)==auth
    assert auth["billing_reconciled"] is False
    before=s.ledger.snapshot()
    c.reserve_with_continuation(s.ledger,auth,"calibration",.3,"new-request")
    s.ledger.settle("new-request",.2)
    after=s.ledger.snapshot()
    assert set(after["unsettled_reservations"])=={h["reservation_id"] for h in s.holds}
    assert after["committed_usd"]==pytest.approx(before["committed_usd"]+.2)
    assert after["spent_usd"]==pytest.approx(.25)
    assert c._rows(s.ledger)["old-main"]["actual"] is None


def test_without_explicit_authorization_unsettled_charges_still_stop(scope):
    with pytest.raises(BudgetExceeded):validate(scope)
    with pytest.raises(BudgetExceeded):c.reserve_with_continuation(scope.ledger,None,"calibration",.1,"not-permitted")


def test_new_unknown_request_cannot_join_legacy_quarantine(scope):
    s=scope;auth=authorize(s)
    c.reserve_with_continuation(s.ledger,auth,"calibration",.1,"new-unknown")
    with pytest.raises(BudgetExceeded,match="outside|new|missing"):
        c.reserve_with_continuation(s.ledger,auth,"calibration",.1,"must-not-dispatch")
    with pytest.raises(BudgetExceeded):validate(s)
    assert "must-not-dispatch" not in c._rows(s.ledger)


@pytest.mark.parametrize("operation",["delete","amount","phase","settle"])
def test_changed_or_removed_legacy_holds_stop(scope,operation):
    s=scope;authorize(s)
    with sqlite3.connect(s.ledger.path) as db:
        if operation=="delete":db.execute("DELETE FROM reservations WHERE id='old-main'")
        elif operation=="amount":db.execute("UPDATE reservations SET reserved=1 WHERE id='old-main'")
        elif operation=="phase":db.execute("UPDATE reservations SET phase='primary' WHERE id='old-main'")
        else:db.execute("UPDATE reservations SET actual=0 WHERE id='old-main'")
    with pytest.raises(BudgetExceeded):validate(s)


def test_forged_extra_hold_cannot_be_acknowledged(scope):
    s=scope;auth=authorize(s)
    fake={"reservation_id":"forged","phase":"calibration","reserved_usd":.1}
    auth["quarantined_holds"].append(fake)
    auth["authorization_digest"]=serialization_digest(c._unsigned(auth))
    (s.out/c.AUTHORIZATION_FILENAME).write_text(json.dumps(auth))
    with pytest.raises(ValueError,match="unreviewed"):validate(s)


def test_incident_and_configuration_hashes_are_binding(scope):
    s=scope;authorize(s)
    incident=json.loads(s.incident.read_text());incident["note"]="changed after review"
    s.incident.write_text(json.dumps(incident))
    with pytest.raises(ValueError,match="Incident"):validate(s)


def test_wrong_model_cell_suite_or_primary_never_uses_exception(scope,monkeypatch):
    s=scope;authorize(s)
    wrong=deepcopy(s.models);wrong[0]["snapshot"]="unreviewed"
    with pytest.raises(ValueError,match="model"):validate(s,model_configs=wrong)
    wrong_cells=deepcopy(s.cells);wrong_cells[0]["condition"]="C"
    with pytest.raises(ValueError,match="cell"):validate(s,cells=wrong_cells)
    with pytest.raises(BudgetExceeded):validate(s,phase="primary")
    monkeypatch.setattr(c,"_scope_digests",lambda root:("different-suite","harness"))
    with pytest.raises(ValueError,match="suite"):validate(s)


def test_cost_and_session_envelopes_are_checked_before_provider_creation(scope):
    s=scope
    with pytest.raises(BudgetExceeded,match="cost cap"):authorize(s,maximum_new_usd=.1)
    assert not (s.out/c.AUTHORIZATION_FILENAME).exists()
    with pytest.raises(ValueError,match="session"):authorize(s,maximum_new_sessions=17)


def test_each_reservation_is_within_remaining_new_cap_and_old_budget_stays_reserved(scope):
    s=scope;auth=authorize(s,maximum_new_usd=4)
    c.reserve_with_continuation(s.ledger,auth,"calibration",2.,"known-new")
    s.ledger.settle("known-new",2.)
    with pytest.raises(BudgetExceeded,match="cost cap"):
        c.reserve_with_continuation(s.ledger,auth,"calibration",2.,"too-expensive")
    assert "too-expensive" not in c._rows(s.ledger)
    assert s.ledger.snapshot()["committed_usd"]==pytest.approx(17.05)


def test_authorization_is_immutable_and_not_portable_to_another_root(scope):
    s=scope;auth=authorize(s)
    with pytest.raises(ValueError,match="existing"):authorize(s)
    changed=dict(auth);changed["maximum_new_usd"]=100
    (s.out/c.AUTHORIZATION_FILENAME).write_text(json.dumps(changed))
    with pytest.raises(ValueError,match="changed"):validate(s)
    with pytest.raises(ValueError,match="changed"):
        c.reserve_with_continuation(s.ledger,auth,"calibration",.1,"tampered")


def test_shared_project_budget_still_counts_the_quarantined_holds(scope,tmp_path):
    s=scope
    small=BudgetLedger(tmp_path/"small-budget.sqlite3",total=16)
    for hold in s.holds:small.reserve(hold["phase"],hold["reserved_usd"],hold["reservation_id"])
    s.ledger=small
    with pytest.raises(BudgetExceeded,match="shared project|calibration budget"):
        authorize(s)
    assert small.snapshot()["committed_usd"]==15


def test_new_session_count_is_bounded_even_when_every_charge_is_tiny(scope):
    s=scope;auth=authorize(s)
    for index in range(18):
        key=f"new-{index}"
        c.reserve_with_continuation(s.ledger,auth,"calibration",.1,key)
        s.ledger.settle(key,.001)
    with pytest.raises(BudgetExceeded,match="session count"):
        c.reserve_with_continuation(s.ledger,auth,"calibration",.1,"nineteenth")
    assert "nineteenth" not in c._rows(s.ledger)
