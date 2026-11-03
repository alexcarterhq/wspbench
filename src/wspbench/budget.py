"""Transactional reservations prevent parallel trials from overspending."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
import sqlite3
import uuid
import math

TOKEN_FIELDS = ("input_tokens", "input_tokens_cache_write", "input_tokens_cache_read",
                "output_tokens", "reasoning_tokens", "total_tokens")
PRICE_FIELDS = ("input_per_million", "cache_write_per_million", "cache_read_per_million", "output_per_million")


class BudgetExceeded(RuntimeError):
    pass


def _units(value: float) -> int:
    if not math.isfinite(value) or value < 0:
        raise ValueError("Costs must be nonnegative")
    return int((Decimal(str(value)) * 1_000_000).to_integral_value(rounding=ROUND_CEILING))


class BudgetLedger:
    """Crash-safe ledger. Unsettled reservations remain charged until reconciled."""

    def __init__(self, path: Path, total: float = 3000,
                 phases: dict[str, float] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.total = _units(total)
        self.phases = {k: _units(v) for k, v in (phases or {
            "calibration": 500, "primary": 1800, "diagnostics": 200,
        }).items()}
        with self._connection() as db:
            db.execute("CREATE TABLE IF NOT EXISTS reservations (id TEXT PRIMARY KEY, phase TEXT NOT NULL, reserved INTEGER NOT NULL, actual INTEGER, created TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS policy (key TEXT PRIMARY KEY, value INTEGER NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS corrections (id INTEGER PRIMARY KEY, reservation_id TEXT NOT NULL, old_actual INTEGER NOT NULL, new_actual INTEGER NOT NULL, reason TEXT NOT NULL, evidence_sha256 TEXT NOT NULL, reviewer TEXT NOT NULL, created TEXT NOT NULL, UNIQUE(reservation_id,evidence_sha256))")
            desired = {"total": self.total, **{f"phase:{k}": v for k, v in self.phases.items()}}
            existing = dict(db.execute("SELECT key,value FROM policy"))
            if existing and existing != desired:
                raise ValueError("Ledger budget policy differs from its creation policy")
            db.executemany("INSERT OR IGNORE INTO policy VALUES (?, ?)", desired.items())

    @contextmanager
    def _connection(self):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        try:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def reserve(self, phase: str, maximum_usd: float, reservation_id: str | None = None) -> str:
        if phase not in self.phases:
            raise ValueError(f"Unknown budget phase: {phase}")
        amount = _units(maximum_usd)
        reservation_id = reservation_id or uuid.uuid4().hex
        with self._connection() as db:
            if db.execute("SELECT 1 FROM reservations WHERE id=?", (reservation_id,)).fetchone():
                raise ValueError("Reservation already exists; reconcile it before retrying")
            total = db.execute("SELECT COALESCE(SUM(COALESCE(actual,reserved)),0) FROM reservations").fetchone()[0]
            used = db.execute("SELECT COALESCE(SUM(COALESCE(actual,reserved)),0) FROM reservations WHERE phase=?", (phase,)).fetchone()[0]
            if total + amount > self.total or used + amount > self.phases[phase]:
                raise BudgetExceeded(f"Reservation would exceed {phase} or total budget")
            db.execute("INSERT INTO reservations VALUES (?,?,?,?,?)", (reservation_id, phase, amount, None, datetime.now(timezone.utc).isoformat()))
        return reservation_id

    def settle(self, reservation_id: str, actual_usd: float) -> None:
        actual = _units(actual_usd)
        with self._connection() as db:
            row = db.execute("SELECT reserved,actual FROM reservations WHERE id=?", (reservation_id,)).fetchone()
            if row is None:
                raise ValueError("Unknown reservation")
            if row[1] is not None and row[1] != actual:
                raise ValueError("A settled reservation cannot be changed")
            db.execute("UPDATE reservations SET actual=? WHERE id=?", (actual, reservation_id))
        if actual > row[0]:
            raise BudgetExceeded("Actual billing exceeded the reservation; cost recorded, further runs must stop")

    def snapshot(self) -> dict:
        with self._connection() as db:
            rows = list(db.execute("SELECT id,phase,reserved,actual FROM reservations"))
            corrections = list(db.execute("SELECT reservation_id,old_actual,new_actual,reason,evidence_sha256,reviewer,created FROM corrections ORDER BY id"))
        return {
            "ceiling_usd": self.total / 1e6,
            "committed_usd": sum(r[3] if r[3] is not None else r[2] for r in rows) / 1e6,
            "spent_usd": sum(r[3] or 0 for r in rows) / 1e6,
            "unsettled_reservations": [r[0] for r in rows if r[3] is None],
            "phases": {p: {"ceiling_usd": cap / 1e6, "committed_usd": sum((r[3] if r[3] is not None else r[2]) for r in rows if r[1] == p) / 1e6} for p, cap in self.phases.items()},
            "corrections": [{"reservation_id": c[0], "old_actual_usd": c[1] / 1e6,
                             "new_actual_usd": c[2] / 1e6, "reason": c[3],
                             "evidence_sha256": c[4], "reviewer": c[5], "created_at": c[6]} for c in corrections],
        }

    def correct_cost(self, reservation_id: str, actual_usd: float, *, reason: str,
                     evidence_sha256: str, reviewer: str) -> dict:
        """Correct a settled cost with an immutable audit trail; never release unknown charges."""
        if not reason.strip() or not reviewer.strip() or len(evidence_sha256) != 64 or any(c not in "0123456789abcdef" for c in evidence_sha256):
            raise ValueError("Cost correction requires reviewer, reason and a SHA-256 evidence digest")
        actual = _units(actual_usd)
        with self._connection() as db:
            row = db.execute("SELECT reserved,actual FROM reservations WHERE id=?", (reservation_id,)).fetchone()
            if row is None or row[1] is None:
                raise ValueError("Only known, settled charges may be corrected; unknown billing remains reserved")
            prior = db.execute("SELECT old_actual,new_actual FROM corrections WHERE reservation_id=? AND evidence_sha256=?", (reservation_id, evidence_sha256)).fetchone()
            if prior:
                if prior[1] != actual:
                    raise ValueError("The same evidence cannot support two different corrected charges")
                return {"old_actual_usd": prior[0] / 1e6, "new_actual_usd": prior[1] / 1e6, "already_applied": True}
            db.execute("INSERT INTO corrections (reservation_id,old_actual,new_actual,reason,evidence_sha256,reviewer,created) VALUES (?,?,?,?,?,?,?)",
                       (reservation_id, row[1], actual, reason, evidence_sha256, reviewer, datetime.now(timezone.utc).isoformat()))
            db.execute("UPDATE reservations SET actual=? WHERE id=?", (actual, reservation_id))
        if actual > row[0]:
            raise BudgetExceeded("Corrected charge exceeds its original reservation; correction recorded and further execution must stop")
        return {"old_actual_usd": row[1] / 1e6, "new_actual_usd": actual / 1e6, "already_applied": False}


def empty_usage() -> dict[str, int]:
    return {**{key: 0 for key in TOKEN_FIELDS}, "input_tokens_total": 0}


def normalize_usage(value) -> dict[str, int]:
    """Inspect separates uncached, cache-write and cache-read input categories.

    Output already includes reasoning. Reasoning is recorded but never charged twice.
    """
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    counts = {}
    for key in TOKEN_FIELDS:
        count = value.get(key)
        count = 0 if count is None else count
        if type(count) is not int or count < 0:
            raise ValueError(f"Invalid provider token count: {key}")
        counts[key] = count
    counts["input_tokens_total"] = sum(counts[k] for k in (
        "input_tokens", "input_tokens_cache_write", "input_tokens_cache_read"))
    expected = counts["input_tokens_total"] + counts["output_tokens"]
    if counts["total_tokens"] != expected:
        raise ValueError("Provider token categories do not reconcile with total_tokens")
    if counts["reasoning_tokens"] > counts["output_tokens"]:
        raise ValueError("Reasoning tokens exceed output tokens; billing cannot be reconciled")
    return counts


def add_usage(aggregate: dict, value) -> dict[str, int]:
    incoming = normalize_usage(value)
    return {key: aggregate.get(key, 0) + incoming[key] for key in (*TOKEN_FIELDS, "input_tokens_total")}


def _prices(model: dict) -> dict:
    pricing = model.get("pricing", {})
    if not pricing.get("verified_at") or not pricing.get("source"):
        raise ValueError(f"Verify current prices for {model['id']} before paid execution")
    if any(type(pricing.get(k)) not in (int, float) or not math.isfinite(pricing[k]) or pricing[k] < 0 for k in PRICE_FIELDS):
        raise ValueError("Explicit verified uncached, cache-write, cache-read and output prices are required")
    return pricing


def session_cost(model: dict, usage: dict) -> float:
    pricing = _prices(model)
    usage = normalize_usage(usage)
    return (usage["input_tokens"] * pricing["input_per_million"]
            + usage["input_tokens_cache_write"] * pricing["cache_write_per_million"]
            + usage["input_tokens_cache_read"] * pricing["cache_read_per_million"]
            + usage["output_tokens"] * pricing["output_per_million"]) / 1e6


def maximum_session_cost(model: dict, limits) -> float:
    pricing = _prices(model)
    worst_input = max(pricing[k] for k in PRICE_FIELDS[:3])
    return (limits.input_tokens * worst_input + limits.output_tokens * pricing["output_per_million"]) / 1e6
