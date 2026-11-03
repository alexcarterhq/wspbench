"""Condition-specific scientific gates; CPU references never imply model ability."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

from . import VERSION
from .contracts import FAMILIES
from .serialization import serialization_digest, weights_digest, load_weights

REQUIRED_BASELINES = {
    "rule_recovery": {"metadata_prior", "statistical_reader", "circuit_algebra", "behavior_only"},
    "causal_forecast": {"metadata_prior", "circuit_algebra", "behavior_only", "symbolic_effects"},
    "edit_effects": {"metadata_prior", "statistical_reader", "circuit_algebra", "behavior_only"},
    "coupled_repair": {"no_op", "generic_repair", "behavior_only"},
    "compositional_synthesis": {"no_op", "public_spec_constructor", "constrained_lookup", "anf_constructor"},
    "loop_repair": {"no_op", "contrastive_mean", "contrastive_pca", "random_direction", "behavior_only", "loop_spec_constructor"},
}


def _scope(root):
    from .tasks import load_tasks
    try:
        return [t for t in load_tasks(root, "all") if t.split != "baseline_train"]
    except ValueError:
        return []


def _proof(task, condition, validation):
    """Consume verifier attestations, never a solver's own declaration of success."""
    records = [r for r in validation.get("feasibility_records", [])
               if r.get("task_id") == task.id and r.get("condition") == condition]
    if len(records) != 1:
        return None
    item = records[0]
    digest = item.get("evidence_digest", "")
    common = (item.get("status") == "passed" and item.get("public_information_only") is True
              and len(digest) == 64 and all(x in "0123456789abcdef" for x in digest)
              and item.get("method") and item.get("target_digest") == weights_digest(load_weights(task.weights_path)))
    if not common:
        return None
    if condition in {"A", "B"} and item.get("uses_target_execution") is not False:
        return None
    if item.get("method") == "public_semantics_capacity_witness":
        if not (item.get("complete_public_semantics") is True and item.get("constructive_witness_verified") is True):
            return None
    elif item.get("public_derivation_checked") is not True:
        return None
    calls = item.get("analysis_calls")
    if type(calls) is not int or calls < 0 or calls > (0 if condition == "A" else task.limits.analysis_calls):
        return None
    cpu = item.get("cpu_seconds")
    if type(cpu) not in (float, int) or not 0 <= cpu <= task.limits.cpu_seconds:
        return None
    return item


def _row_qualifies(row, family):
    return (row.get("actor_type") == "baseline" and not row.get("access_card", {}).get("privileged")
            and family in row.get("access_card", {}).get("qualifying_families", [family]))


def _coverage_manifest(root, baseline_summary):
    try:
        entries = json.loads((Path(root) / "baseline-coverage.private.json").read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(entries, list) or serialization_digest(entries) != baseline_summary.get("coverage_manifest_digest"):
        return None
    result = {}
    for entry in entries:
        if not isinstance(entry, dict):
            return None
        key = tuple(entry.get(k) for k in ("task_id", "actor", "condition", "weight_view"))
        if any(not isinstance(k, str) for k in key) or key in result:
            return None
        result[key] = entry
    return result


def _roster(family, condition):
    from .baselines import BASELINES
    required = REQUIRED_BASELINES[family]
    # Conventional B/C algorithms are difficulty controls for A, never A model rows.
    source_condition = "B" if condition == "A" else condition
    return {name for name in required if source_condition in BASELINES.get(name, {}).get("conditions", ("B", "C"))}


def assess_admission(root: Path, baseline_summary: dict, validation: dict | None = None) -> dict:
    """Admit exact development cells only; keep primary scoring blocked."""
    root = Path(root)
    if validation is None:
        from .tasks import validate_suite
        validation = validate_suite(root)
    cpu_valid = validation.get("status") == "passed" and validation.get("valid", True) is True
    tasks = _scope(root)
    tables = baseline_summary.get("tables", baseline_summary.get("report", {}).get("tables", []))
    baseline_digest = serialization_digest(baseline_summary)
    coverage_manifest = _coverage_manifest(root, baseline_summary)
    grouped = defaultdict(list)
    for task in tasks:
        for condition in task.conditions:
            grouped[(task.family, condition, task.target_stratum, task.capacity_tier)].append(task)
    # Legacy aggregate-only input can still expose a ceiling, but can never authorize a task.
    if not tasks:
        for row in tables:
            if row.get("family") in FAMILIES:
                grouped[(row["family"], row.get("condition", "B"), row.get("target_stratum", "unspecified"),
                         row.get("capacity_tier", "unspecified"))]
    cells_by_family = defaultdict(list)
    allowlist = []
    for (family, condition, stratum, tier), members in sorted(grouped.items()):
        matching = [r for r in tables if r.get("family") == family
                    and r.get("target_stratum", "unspecified") == stratum
                    and r.get("capacity_tier", "unspecified") == tier
                    and r.get("weight_view", "full") == "full" and r.get("split") != "baseline_train"
                    and r.get("condition") == ("B" if condition == "A" else condition)]
        expected = defaultdict(set)
        for task in members:
            expected[task.split].add(task.parent_problem_id)
        covered = set()
        coverage = []
        for actor in sorted(_roster(family, condition)):
            rows = [r for r in matching if r.get("actor") == actor and _row_qualifies(r, family)]
            # Every populated split must have a complete, error-free run of this method.
            split_ok = all(any(r.get("split") == split and r.get("n_parents", 0) >= len(parents)
                               and not r.get("infrastructure_failures", 0)
                               and not r.get("audit_status_counts", {}).get("rejected", 0)
                               for r in rows) for split, parents in expected.items())
            exact_ok = coverage_manifest is not None
            for task in members:
                source_condition = "B" if condition == "A" else condition
                entry = (coverage_manifest or {}).get((task.id, actor, source_condition, "full"))
                exact_ok = exact_ok and bool(entry and entry.get("target_digest") == weights_digest(load_weights(task.weights_path))
                                            and entry.get("status") in {"scored", "completed"}
                                            and entry.get("audit_status") != "rejected"
                                            and family in entry.get("qualifying_families", []))
            if rows and split_ok and exact_ok:
                covered.add(actor)
            coverage.append({"actor": actor, "complete": bool(rows and split_ok and exact_ok),
                             "exact_target_manifest_verified": bool(exact_ok)})
        missing = sorted(_roster(family, condition) - covered)
        # Only eligible same-condition comparators can saturate an access condition.
        # Codes in B/C do not establish the unaided A ceiling.
        ceilings = [] if condition == "A" else [
            {"actor": r.get("actor"), "score": r.get("score"), "split": r.get("split"),
             "n_mechanisms": r.get("n_mechanisms", 0), "n_parents": r.get("n_parents", 0)}
            for r in matching if _row_qualifies(r, family) and r.get("score") is not None
            and r["score"] >= .95 and r.get("n_mechanisms", 0) >= 4]
        proofs = {task.id: _proof(task, condition, validation) for task in members}
        all_proven = bool(members) and all(proofs.values())
        reasons = []
        if not cpu_valid:
            reasons.append("CPU validation failed or is absent")
        if missing:
            reasons.append("Incomplete qualifying baseline coverage in this exact access/stratum/tier cell")
        if not all_proven:
            reasons.append("Missing verified public-information feasibility for one or more target checkpoints")
        in_scope = bool(members) and all(t.public.get("evaluation_role", "candidate") != "diagnostic_sanity" for t in members)
        if not in_scope:
            reasons.append("Retained diagnostic/sanity targets are outside paid challenge calibration")
        if ceilings:
            reasons.append("Same-condition conventional baseline saturates at least four independent mechanisms")
        # Small-sample perfect scores remain a warning, not four independent mechanisms.
        small_ceilings = [] if condition == "A" else [r.get("actor") for r in matching
            if _row_qualifies(r, family) and r.get("score") is not None and r["score"] >= .95
            and r.get("n_mechanisms", 0) < 4]
        if small_ceilings and not ceilings:
            reasons.append("Observed classical ceiling on a small sample remains unresolved")
        eligible = not reasons
        status = "sanity_only" if ceilings else "calibration_candidate" if eligible else "blocked"
        cell = {"condition": condition, "target_stratum": stratum, "capacity_tier": tier,
                "weight_view": "full", "status": status, "reasons": reasons,
                "calibration_scope": in_scope,
                "baseline_coverage_complete": not missing, "missing_baselines": missing,
                "baseline_coverage": coverage, "information_feasibility_attested": all_proven,
                "n_targets": len(members), "n_mechanisms": len({t.mechanism_family_id for t in members}),
                "saturating_baselines": ceilings, "small_sample_ceiling_warnings": sorted(set(small_ceilings)),
                "note": "A has code-based difficulty controls, not eligible code-based A model scores." if condition == "A" else
                        "Conventional controls share observation quotas; their computation surface remains disclosed."}
        cells_by_family[family].append(cell)
        if eligible:
            for task in members:
                if task.split == "dev":
                    allowlist.append({"task_id": task.id, "target_digest": weights_digest(load_weights(task.weights_path)),
                                      "condition": condition, "weight_view": "full",
                                      "feasibility_digest": serialization_digest(proofs[task.id]),
                                      "baseline_evidence_digest": baseline_digest})
    families = []
    for family in FAMILIES:
        cells = cells_by_family[family]
        eligible = any(c["status"] == "calibration_candidate" for c in cells)
        families.append({"family": family,
            "status": "calibration_candidate" if eligible else "sanity_only" if cells and all(c["status"] == "sanity_only" for c in cells) else "blocked",
            "baseline_coverage_complete": bool(cells) and all(c["baseline_coverage_complete"] for c in cells),
            "missing_baselines": sorted({name for c in cells for name in c["missing_baselines"]}),
            "information_feasibility_attested": bool(cells) and all(c["information_feasibility_attested"] for c in cells),
            "unattested_reference_conditions": sorted({c["condition"] for c in cells if not c["information_feasibility_attested"]}),
            "access_stratum_cells": cells})
    return {"schema_version": "2", "benchmark_version": VERSION, "cpu_validation_passed": cpu_valid,
            "headline_admitted": False, "paid_runs_permitted": False,
            "calibration_permitted": bool(allowlist), "calibration_allowlist": allowlist,
            "baseline_evidence_digest": baseline_digest,
            "feasibility_digest": serialization_digest(validation.get("feasibility_records", [])),
            "status": "cpu_validated_candidates" if cpu_valid else "blocked", "families": families,
            "next_gate": "Review and calibrate only exact allowlisted development cells. Primary scoring remains blocked; no automatic promotion from CPU proofs or model calibration."}


def write_admission(root: Path, baseline_summary: dict, destination: Path, validation: dict | None = None) -> dict:
    result = assess_admission(root, baseline_summary, validation)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
