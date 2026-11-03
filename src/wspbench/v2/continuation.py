"""Narrow operator authorization for new work while old charges stay reserved.

This does not reconcile billing, release money, or grant primary-score admission.
The ordinary path continues to reject every unsettled reservation.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import fcntl
import json
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

from ..budget import BudgetExceeded, _units, maximum_session_cost
from .serialization import serialization_digest

AUTHORIZATION_FILENAME = "continuation-authorization.private.json"
USER_INSTRUCTION = "try different GPT models and gather stats on them"
SCHEMA = "bounded-gpt-continuation-v1"


def _read(path):
    return json.loads(Path(path).read_text())


def _rows(ledger):
    """Read existing accounting; never change or settle a legacy reservation."""
    with sqlite3.connect(f"{ledger.path.resolve().as_uri()}?mode=ro", uri=True) as db:
        return {row[0]: {"phase": row[1], "reserved": row[2], "actual": row[3]}
                for row in db.execute("SELECT id,phase,reserved,actual FROM reservations")}


def _scope_digests(data_root):
    from .runner import suite_digest, harness_digest
    return suite_digest(data_root), harness_digest()


def _incident_holds(path, ledger):
    path = Path(path).resolve()
    incident = _read(path)
    holds = incident.get("reservation_holds")
    if not isinstance(holds, list) or not holds:
        raise ValueError("Incident must explicitly list every audited reservation hold")
    if incident.get("usage_received") is not False:
        raise ValueError("Continuation is only for an explicitly unreconciled incident")
    ids = []
    for hold in holds:
        if not isinstance(hold, dict) or set(hold) != {"reservation_id", "phase", "reserved_usd"}:
            raise ValueError("Malformed audited incident hold")
        if not isinstance(hold["reservation_id"], str) or not hold["reservation_id"] or hold["phase"] != "calibration":
            raise ValueError("Only exact calibration incident holds can be quarantined")
        if _units(hold["reserved_usd"]) <= 0:
            raise ValueError("Incident holds must remain positive reservations")
        ids.append(hold["reservation_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate incident reservation hold")
    if "total_unknown_charge_hold_usd" in incident and _units(incident["total_unknown_charge_hold_usd"]) != sum(_units(hold["reserved_usd"]) for hold in holds):
        raise ValueError("Incident total does not reconcile with its explicit reservation holds")
    rows = _rows(ledger)
    _check_holds(rows, holds)
    return incident, sorted(holds, key=lambda hold: hold["reservation_id"])


def _check_holds(rows, holds):
    allowed = {hold["reservation_id"] for hold in holds}
    pending = {key for key, row in rows.items() if row["actual"] is None}
    if pending != allowed:
        raise BudgetExceeded("A new, missing, or reconciled reservation is outside the acknowledged incident holds")
    for hold in holds:
        row = rows.get(hold["reservation_id"])
        if row is None or row["actual"] is not None or row["phase"] != hold["phase"] or row["reserved"] != _units(hold["reserved_usd"]):
            raise BudgetExceeded("An acknowledged hold changed amount, phase, or settlement state")


def _unsigned(authorization):
    return {key: value for key, value in authorization.items() if key != "authorization_digest"}


def authorization_digest_if_present(output_root):
    path = Path(output_root) / AUTHORIZATION_FILENAME
    if not path.exists():
        return None
    authorization = _read(path)
    digest = serialization_digest(_unsigned(authorization))
    if authorization.get("authorization_digest") != digest:
        raise ValueError("Continuation authorization changed after review")
    return digest


def _check_context(authorization, data_root, output_root, config_path, model_configs, cells, ledger):
    if authorization.get("schema") != SCHEMA or authorization.get("phase") != "calibration":
        raise ValueError("Continuation never authorizes primary scoring")
    if (authorization.get("user_instruction") != USER_INSTRUCTION
            or not isinstance(authorization.get("reviewer"),str) or not authorization["reviewer"].strip()
            or not isinstance(authorization.get("evidence"),list) or not authorization["evidence"]):
        raise ValueError("Continuation needs explicit renewed instruction and operator review")
    if authorization.get("billing_reconciled") is not False:
        raise ValueError("Continuation must not claim unresolved billing was reconciled")
    if authorization.get("output_root") != str(Path(output_root).resolve()) or authorization.get("ledger_path") != str(ledger.path.resolve()):
        raise ValueError("Continuation is bound to one output root and the existing project ledger")
    if authorization.get("data_root") != str(Path(data_root).resolve()):
        raise ValueError("Continuation data root changed")
    if authorization.get("config_path") != str(Path(config_path).resolve()) or authorization.get("configuration_digest") != serialization_digest(_read(config_path)):
        raise ValueError("Continuation model configuration changed")
    suite, harness = _scope_digests(data_root)
    if authorization.get("suite_digest") != suite or authorization.get("harness_digest") != harness:
        raise ValueError("Continuation suite or harness changed")
    allowed_models = set(authorization.get("model_config_digests", []))
    if not model_configs or any(serialization_digest(model) not in allowed_models for model in model_configs):
        raise ValueError("A model is outside the reviewed continuation scope")
    allowed_cells = set(authorization.get("cell_digests", []))
    if not cells or any(serialization_digest(cell) not in allowed_cells or cell.get("condition") != "A" or cell.get("weight_view") != "full" for cell in cells):
        raise ValueError("A task/access cell is outside the reviewed continuation scope")
    sessions = len(model_configs) * len(cells)
    cap = authorization.get("maximum_new_sessions")
    if type(cap) is not int or not 1 <= cap <= 24 or sessions > cap:
        raise ValueError("Requested comparison exceeds its authorized session count")
    incident_path = Path(authorization["incident_path"])
    incident = _read(incident_path)
    if serialization_digest(incident) != authorization.get("incident_digest") or incident.get("reservation_holds") != authorization.get("incident_holds_source"):
        raise ValueError("Incident evidence changed after continuation review")
    _, holds = _incident_holds(incident_path, ledger)
    if holds != authorization.get("quarantined_holds"):
        raise ValueError("Continuation added an unreviewed incident hold")
    prior = _units(authorization.get("prior_diagnostic_spend_usd", 0))
    maximum = _units(authorization["maximum_new_usd"])
    if prior >= maximum:
        raise BudgetExceeded("Prior diagnostic spending exhausted the new comparison cap")
    from .tasks import load_tasks
    tasks = {task.id: task for task in load_tasks(data_root, "dev")}
    if any(cell["task_id"] not in tasks for cell in cells):
        raise ValueError("Continuation permits development targets only")
    projection = sum(maximum_session_cost(model, tasks[cell["task_id"]].limits)
                     for model in model_configs for cell in cells)
    if _units(projection) + prior > maximum:
        raise BudgetExceeded("Complete selected comparison exceeds the authorized new cost cap")
    budget=ledger.snapshot()
    if (budget["committed_usd"]+projection>budget["ceiling_usd"]
            or budget["phases"]["calibration"]["committed_usd"]+projection>budget["phases"]["calibration"]["ceiling_usd"]):
        raise BudgetExceeded("Complete selected comparison exceeds the remaining shared project or calibration budget")


def write_continuation_authorization(data_root, output_root, config_path, *, ledger, cells,
                                     model_configs, incident_path, maximum_new_usd=10,
                                     maximum_new_sessions=18, reviewer, user_instruction,
                                     evidence, prior_diagnostic_spend_usd=0):
    """Write one fresh, exact continuation after operator review; no API calls."""
    output_root = Path(output_root).resolve()
    path = output_root / AUTHORIZATION_FILENAME
    if path.exists():
        raise ValueError("Preserve an existing continuation authorization; use a fresh output root")
    incident, holds = _incident_holds(incident_path, ledger)
    suite, harness = _scope_digests(data_root)
    rows = _rows(ledger)
    authorization = {
        "schema": SCHEMA, "phase": "calibration", "issued_at": datetime.now(ZoneInfo("Europe/Moscow")).isoformat(),
        "user_instruction": user_instruction, "reviewer": reviewer, "evidence": list(evidence),
        "authorization_basis": "Renewed user request permits bounded new GPT comparison while all acknowledged legacy charges remain fully reserved.",
        "billing_reconciled": False, "output_root": str(output_root), "ledger_path": str(ledger.path.resolve()),
        "authorization_path": str(path), "data_root": str(Path(data_root).resolve()),
        "config_path": str(Path(config_path).resolve()), "configuration_digest": serialization_digest(_read(config_path)),
        "suite_digest": suite, "harness_digest": harness,
        "model_config_digests": sorted(serialization_digest(model) for model in model_configs),
        "cell_digests": sorted(serialization_digest(cell) for cell in cells),
        "maximum_new_usd": maximum_new_usd, "maximum_new_sessions": maximum_new_sessions,
        "prior_diagnostic_spend_usd": prior_diagnostic_spend_usd,
        "baseline_reservation_ids": sorted(rows), "incident_path": str(Path(incident_path).resolve()),
        "incident_digest": serialization_digest(incident), "incident_holds_source": incident["reservation_holds"],
        "quarantined_holds": holds,
    }
    _check_context(authorization, data_root, output_root, config_path, model_configs, cells, ledger)
    authorization["authorization_digest"] = serialization_digest(authorization)
    output_root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(authorization, indent=2, allow_nan=False) + "\n")
    return authorization


def validate_continuation(data_root, output_root, config_path, *, ledger, model_configs, cells, phase):
    """Return a checked calibration exception, or retain the ordinary stop rule."""
    pending = ledger.snapshot()["unsettled_reservations"]
    path = Path(output_root) / AUTHORIZATION_FILENAME
    if phase != "calibration" or not path.exists():
        if pending:
            raise BudgetExceeded("Reconcile outstanding reservations or obtain exact renewed development authorization")
        return None
    authorization_digest_if_present(output_root)
    authorization = _read(path)
    _check_context(authorization, data_root, output_root, config_path, model_configs, cells, ledger)
    return authorization


@contextmanager
def _budget_lock(authorization):
    path = Path(authorization["output_root"]) / "continuation-budget.lock"
    with path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def reserve_with_continuation(ledger, authorization, phase, maximum_usd, reservation_id):
    """Check acknowledged holds and the new cap immediately before reserving."""
    if authorization is None:
        if ledger.snapshot()["unsettled_reservations"]:
            raise BudgetExceeded("New unsettled billing blocks further dispatch")
        return ledger.reserve(phase, maximum_usd, reservation_id)
    if phase != "calibration":
        raise BudgetExceeded("Continuation is never valid for primary scoring")
    with _budget_lock(authorization):
        current = _read(authorization["authorization_path"])
        if current != authorization or current.get("authorization_digest") != serialization_digest(_unsigned(current)):
            raise ValueError("Continuation authorization changed during the run")
        if serialization_digest(_read(current["incident_path"])) != current["incident_digest"]:
            raise ValueError("Incident evidence changed during the run")
        rows = _rows(ledger)
        _check_holds(rows, current["quarantined_holds"])
        baseline = set(current["baseline_reservation_ids"])
        new_rows = [row for key, row in rows.items() if key not in baseline]
        if len(new_rows) >= current["maximum_new_sessions"]:
            raise BudgetExceeded("Authorized new session count exhausted")
        charged = sum(row["actual"] if row["actual"] is not None else row["reserved"] for row in new_rows)
        charged += _units(current["prior_diagnostic_spend_usd"])
        if charged + _units(maximum_usd) > _units(current["maximum_new_usd"]):
            raise BudgetExceeded("Authorized new comparison cost cap exhausted")
        return ledger.reserve(phase, maximum_usd, reservation_id)
