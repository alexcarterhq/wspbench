"""Aggregate-only three-condition reports and paired access-package contrasts."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import math
from pathlib import Path
import json

import numpy as np

from . import VERSION
from .serialization import canonical_json


def _actor(record: dict) -> str:
    return str(record.get("actor_id") or record.get("model_id") or record.get("actor_name") or "unknown")


def _information(record: dict) -> tuple[str, str, str]:
    information = record.get("prompt_information", {})
    return (str(record.get("hint_level", information.get("hint_level", "unknown"))),
            str(record.get("diagnosis_scope", information.get("diagnosis_scope", "unknown"))),
            canonical_json({k: information.get(k, "unspecified") for k in
                            ("provenance", "source", "optimization", "history")}))


def _estimate(records: list[dict], *, draws: int = 4000, seed: int = 20261103) -> dict:
    """Average attempts within parent, parents within independent mechanisms."""
    parents: dict[str, list[float]] = defaultdict(list)
    mechanism: dict[str, str] = {}
    for record in records:
        parent = str(record["parent_problem_id"])
        parents[parent].append(float(bool(record.get("success"))))
        current = str(record["mechanism_family_id"])
        if parent in mechanism and mechanism[parent] != current:
            raise ValueError("A parent cannot belong to multiple independent mechanisms")
        mechanism[parent] = current
    clusters: dict[str, list[float]] = defaultdict(list)
    for parent, values in parents.items():
        clusters[mechanism[parent]].append(float(np.mean(values)))
    values = np.asarray([np.mean(x) for x in clusters.values()], dtype=float)
    if not len(values):
        return {"score": None, "ci_low": None, "ci_high": None, "n_mechanisms": 0,
                "n_parents": 0, "n_attempts": 0}
    boot = np.random.default_rng(seed).choice(values, (draws, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(boot, [.025, .975])
    mean = float(values.mean())
    radius = math.sqrt(math.log(2 / .05) / (2 * len(values)))
    return {"score": mean, "ci_low": float(low), "ci_high": float(high),
            "n_mechanisms": len(values), "n_parents": len(parents), "n_attempts": len(records),
            "degenerate_bootstrap": bool(np.isclose(low, high)),
            "bounded_mean_95_interval": [max(0., mean - radius), min(1., mean + radius)],
            "interval_note": "Mechanism bootstrap describes this sampled population; all-pass/all-fail intervals can degenerate. The bounded-mean interval is a conservative independent-mechanism bound."}


def _metrics(records: list[dict]) -> dict:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    mechanisms = {str(r["parent_problem_id"]): str(r["mechanism_family_id"]) for r in records}
    for record in records:
        for name, value in record.get("metrics", {}).items():
            if isinstance(value, (int, float)) and math.isfinite(value):
                values[name][str(record["parent_problem_id"])].append(float(value))
    result = {}
    for name, parents in sorted(values.items()):
        clusters = defaultdict(list)
        for parent, observations in parents.items():
            clusters[mechanisms[parent]].append(float(np.mean(observations)))
        means = np.asarray([np.mean(v) for v in clusters.values()])
        bootstrap = np.random.default_rng(47).choice(means, (4000, len(means)), replace=True).mean(axis=1)
        low, high = np.quantile(bootstrap, [.025, .975])
        result[name] = {"mean": float(means.mean()), "ci_low": float(low), "ci_high": float(high),
                        "n_mechanisms": len(means), "n_parents": len(parents),
                        "n_observations": sum(map(len, parents.values())),
                        "degenerate_bootstrap": bool(np.isclose(low, high))}
    return result


def paired_difference(records: list[dict], left: str, right: str, *, draws: int = 4000) -> dict:
    """Compare right-left only for identical actor, question and checkpoint parents."""
    if (left, right) not in {("A", "B"), ("B", "C")}:
        raise ValueError("Supported paired access comparisons are B-A and C-B")
    matched: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        if record.get("family") == "loop_repair" or record.get("condition") not in {left, right}:
            continue
        if record.get("status") in {"not_applicable", "infrastructure_failure", "infrastructure_error"}:
            continue
        if record.get("actor_type") not in {"model", "agent"}:
            continue
        key = (_actor(record), record.get("protocol_digest"), record.get("family"),
               record.get("target_stratum"), record.get("capacity_tier"),
               record.get("weight_view", "full"), _information(record), record["parent_problem_id"])
        matched[key][record["condition"]].append(record)
    differences: dict[str, list[float]] = defaultdict(list)
    n_parents = 0
    for _, conditions in matched.items():
        if left not in conditions or right not in conditions:
            continue
        a, b = conditions[left], conditions[right]
        if {r.get("attempt", 1) for r in a} != {r.get("attempt", 1) for r in b}:
            continue
        if len(a) != len(b) or len({r.get("attempt", 1) for r in a}) != len(a):
            continue
        if len({r.get("target_digest") for r in a + b}) != 1 or len({r.get("question_digest") for r in a + b}) != 1:
            continue
        if not all(r.get("target_digest") and r.get("question_digest") for r in a + b):
            continue
        if len({r["mechanism_family_id"] for r in a + b}) != 1:
            continue
        n_parents += 1
        delta = np.mean([bool(r.get("success")) for r in b]) - np.mean([bool(r.get("success")) for r in a])
        differences[str(a[0]["mechanism_family_id"])].append(float(delta))
    values = np.asarray([np.mean(v) for v in differences.values()], dtype=float)
    if not len(values):
        return {"contrast": right + "-" + left, "difference": None, "n_parents": 0, "n_mechanisms": 0,
                "note": "No complete information-matched agent pairs; C-only loop and oracle/baseline actors excluded."}
    samples = np.random.default_rng(89).choice(values, (draws, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [.025, .975])
    return {"contrast": right + "-" + left, "difference": float(values.mean()), "ci_low": float(low),
            "ci_high": float(high), "n_parents": n_parents, "n_mechanisms": len(values),
            "note": "Difference between declared access packages, not an identified internal cognitive mechanism."}


def build_report(records: list[dict]) -> dict:
    """Never merge condition scores, unlike metrics, strata, or oracle/model rows."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    skipped: dict[str, int] = defaultdict(int)
    for record in records:
        if record.get("status") == "not_applicable":
            skipped[record.get("actor_name", "unknown")] += 1
            continue
        key = (_actor(record), record.get("actor_type", "baseline"), record["condition"],
               record.get("weight_view", "full"), record["family"], record.get("target_stratum", "constructed"),
               record.get("capacity_tier", "small"), record.get("split", "unknown"), _information(record))
        groups[key].append(record)
    tables = []
    for key, group in sorted(groups.items()):
        actor, actor_type, condition, view, family, stratum, tier, split, information = key
        infrastructure = [r for r in group if r.get("status") in {"infrastructure_failure", "infrastructure_error"}]
        scored = [r for r in group if r not in infrastructure]
        summary = _estimate(scored)
        curve_records = defaultdict(list)
        for record in scored:
            for point in record.get("budget_curve", []):
                curve_records[(point["budget"], point.get("unit", record.get("snapshot_unit", "episodes" if family == "loop_repair" else "example_evaluations")))].append({**record, **point})
        curves = [{"budget": budget, "unit": unit, **_estimate(points), "metrics": _metrics(points),
                   "mean_actual_observations": float(np.mean([p.get("actual_observations", 0) for p in points]))}
                  for (budget, unit), points in sorted(curve_records.items())]
        tables.append({"actor": actor, "actor_type": actor_type, "condition": condition,
                       "weight_view": view, "family": family, "target_stratum": stratum,
                       "capacity_tier": tier, "split": split, **summary,
                       "hint_level": information[0], "diagnosis_scope": information[1],
                       "hint_provenance": information[2],
                       "infrastructure_failures": len(infrastructure), "metrics": _metrics(scored),
                       "estimated_cost_usd": sum(float(r.get("cost_usd") or 0) for r in group),
                       "example_evaluations": sum(int(r.get("usage", {}).get("example_evaluations", 0)) for r in group),
                       "activation_scalars": sum(int(r.get("usage", {}).get("activation_scalars", 0)) for r in group),
                       "episodes": sum(int(r.get("usage", {}).get("episodes", 0)) for r in group),
                       "generated_tokens": sum(int(r.get("usage", {}).get("generated_tokens", 0)) for r in group),
                       "cpu_seconds": sum(float(r.get("cpu_seconds") or 0) for r in group),
                       "audit_status_counts": {status: sum(r.get("audit", {}).get("status", "unrecorded") == status for r in group)
                                                for status in sorted({r.get("audit", {}).get("status", "unrecorded") for r in group})},
                       "budget_curve": curves,
                       "access_card": group[0].get("access_card", {})})
    paired = []
    eligible_actors = sorted({_actor(r) for r in records if r.get("actor_type") in {"model", "agent"}})
    for actor in eligible_actors:
        chosen = [r for r in records if _actor(r) == actor]
        strata = sorted({(r["family"], r.get("target_stratum", "constructed"),
                          r.get("capacity_tier", "small"), r.get("weight_view", "full"), _information(r))
                         for r in chosen if r["family"] != "loop_repair"})
        for family, stratum, tier, view, information in strata:
            selected = [r for r in chosen if (r["family"], r.get("target_stratum", "constructed"),
                        r.get("capacity_tier", "small"), r.get("weight_view", "full"), _information(r)) ==
                        (family, stratum, tier, view, information)]
            for left, right in (("A", "B"), ("B", "C")):
                paired.append({"actor": actor, "family": family, "target_stratum": stratum,
                               "capacity_tier": tier, "weight_view": view,
                               "hint_level": information[0], "diagnosis_scope": information[1],
                               "hint_provenance": information[2],
                               **paired_difference(selected, left, right)})
    return {"schema_version": "2", "benchmark_version": VERSION, "report_type": "diagnostic",
            "headline_eligible": False, "generated_at": datetime.now(timezone.utc).isoformat(),
            "notice": "CPU/baseline or development evidence; challenge admission and a validated frozen model matrix are required before headline publication.",
            "tables": tables, "paired_access_differences": paired, "not_applicable_counts": dict(skipped),
            "n_records": len(records)}


def write_report(records: list[dict], destination: Path) -> dict:
    report = build_report(records)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
