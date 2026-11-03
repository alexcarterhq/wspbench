"""Pure aggregation and rendering for the released development pilot.

Derived from the audited pilot reporter; no provider or private-workspace access.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import median

from wspbench.budget import normalize_usage
from wspbench.v2.serialization import canonical_json, serialization_digest

INFRA = {"infrastructure_failure", "infrastructure_error", "timeout", "error"}
INVALID = {"invalid", "invalid_submission"}
SCORED = {"completed", "scored", "invalid", "invalid_submission", "protocol_violation"}
TOKEN_FIELDS = ("input_tokens", "input_tokens_cache_write", "input_tokens_cache_read",
                "input_tokens_total", "output_tokens", "reasoning_tokens", "total_tokens")
METRICS = {"functional_agreement", "exhaustive_agreement", "before_agreement", "after_agreement",
           "affected_precision", "affected_recall", "false_positive_rate", "consequence_agreement",
           "target_accuracy", "retention_accuracy", "retention_loss", "affected_accuracy",
           "false_positive_change_rate", "edit_fraction", "edit_norm", "parameters"}


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _key(record):
    return (record["model_id"], record["task_id"], record.get("condition", "A"),
            record.get("weight_view", "full"), record.get("attempt", 1))


def _infra(record):
    return record.get("status") in INFRA or record.get("usage_unknown") is True


def _scorable(record):
    return not _infra(record) and record.get("status") in SCORED


def _credit(record):
    if record.get("status") in INVALID | {"protocol_violation"} or record.get("audit", {}).get("status") == "rejected":
        return 0.
    return float(bool(record.get("success")))


def _responded(record):
    return (type(record.get("turns")) is int and record["turns"] > 0
            or record.get("_completion") is not None or record.get("provider_response_received") is True)


def _valid_json(record):
    text = record.get("_completion")
    if text is None:
        return True if record.get("submitted") is True else None
    text = text.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    try:
        value = json.loads(text)
        if not isinstance(value, dict):
            return False
        canonical_json(value)
        return True
    except (ValueError, TypeError):
        return False


def _outcome(record):
    """Export fixed categories, never raw provider errors or response contents."""
    if record.get("status") == "running":
        return "inflight"
    if _infra(record):
        message = " ".join(str(record.get(key, "")) for key in ("error", "provider_error")).lower()
        categories = (("timeout", ("timeout", "timed out")),
                      ("connection", ("connection", "connecterror", "network")),
                      ("rate_limit", ("ratelimit", "rate limit", "429")),
                      ("authentication", ("authentication", "unauthorized", "401")),
                      ("quota", ("insufficient_quota", "quota")))
        for category, markers in categories:
            if any(marker in message for marker in markers) or record.get("status") == category:
                return "infrastructure_" + category
        return "infrastructure_usage_unavailable" if record.get("usage_unknown") else "infrastructure_other"
    if record.get("status") == "protocol_violation" or record.get("audit", {}).get("status") == "rejected":
        return "protocol_violation"
    if record.get("status") in INVALID or record.get("score_status") == "invalid":
        return "invalid_json" if _valid_json(record) is False else "invalid_submission"
    return "correct" if _credit(record) else "incorrect"


def _sample(values):
    values = [float(value) for value in values if _number(value) and value >= 0]
    return {"n": len(values), "median": median(values) if values else None,
            "minimum": min(values) if values else None, "maximum": max(values) if values else None}


def _cluster_mean(records, value):
    parents = defaultdict(list)
    mechanisms = {}
    for record in records:
        observation = value(record)
        if observation is None:
            continue
        if not _number(observation):
            raise ValueError("A reported metric is not finite numeric data")
        parent = record["parent_problem_id"]
        mechanism = record["mechanism_family_id"]
        if parent in mechanisms and mechanisms[parent] != mechanism:
            raise ValueError("One parent was assigned multiple mechanism clusters")
        parents[parent].append(float(observation)); mechanisms[parent] = mechanism
    clusters = defaultdict(list)
    for parent, observations in parents.items():
        clusters[mechanisms[parent]].append(sum(observations) / len(observations))
    means = [sum(observations) / len(observations) for observations in clusters.values()]
    return {"mean": sum(means) / len(means) if means else None,
            "n_sessions": sum(map(len, parents.values())), "n_parents": len(parents), "n_mechanisms": len(means)}


def _functional(record):
    if record.get("status") in INVALID | {"protocol_violation"} or record.get("audit", {}).get("status") == "rejected":
        return 0.
    metrics = record.get("metrics", {})
    if record["family"] == "rule_recovery":
        return metrics.get("functional_agreement")
    if record["family"] == "edit_effects":
        before, after = metrics.get("before_agreement"), metrics.get("after_agreement")
        return (before + after) / 2 if _number(before) and _number(after) else None
    return metrics.get("target_accuracy")


def _resources(records):
    scored = [r for r in records if _scorable(r)]
    infra = [r for r in records if _infra(r)]
    known_usage = []
    for record in records:
        if isinstance(record.get("usage"), dict) and record.get("usage_unknown") is not True:
            known_usage.append((record, normalize_usage(record["usage"])))
    token_totals = {field: sum(usage[field] for _, usage in known_usage) for field in TOKEN_FIELDS}
    known_cost = [float(r["cost_usd"]) for r in records
                  if _number(r.get("cost_usd")) and r["cost_usd"] >= 0 and r.get("usage_unknown") is not True]
    unknown = [r for r in records if not _number(r.get("cost_usd")) or r.get("usage_unknown") is True]
    unknown_bounds = [r.get("maximum_reserved_usd") for r in unknown]
    all_bounds_known = all(_number(value) and value >= 0 for value in unknown_bounds)
    output_rates, visible_rates = [], []
    for record, usage in known_usage:
        seconds = record.get("wall_seconds")
        if _scorable(record) and _number(seconds) and seconds > 0:
            output_rates.append(usage["output_tokens"] / seconds)
            visible_rates.append((usage["output_tokens"] - usage["reasoning_tokens"]) / seconds)
    return {
        "tokens": token_totals, "usage_known_sessions": len(known_usage),
        "usage_unknown_sessions": len(records) - len(known_usage),
        "known_cost_estimate_usd": sum(known_cost), "known_cost_sessions": len(known_cost),
        "unknown_cost_sessions": len(unknown),
        "unknown_reserved_upper_bound_usd": sum(unknown_bounds) if all_bounds_known else None,
        "unknown_reservation_bounds_complete": all_bounds_known,
        "finalized_cost_estimate_usd": sum(known_cost) if not unknown else None,
        "session_wall_seconds": _sample(r.get("wall_seconds") for r in scored),
        "infrastructure_wall_seconds": _sample(r.get("wall_seconds") for r in infra),
        "local_session_cpu_seconds": _sample(r.get("cpu_seconds") for r in scored),
        "api_reported_output_tokens_per_session_second": _sample(output_rates),
        "visible_answer_tokens_per_session_second": _sample(visible_rates),
    }


def _summary(expected, records, *, capability=True):
    scored = [r for r in records if _scorable(r)]
    responded = [r for r in records if _responded(r)]
    valid_json = [_valid_json(r) for r in responded]
    finalized = [r for r in records if r.get("status") != "running"]
    result = {
        "requested_sessions": len(expected), "launched_sessions": len(records),
        "finalized_sessions": len(finalized), "responded_sessions": len(responded),
        "scored_sessions": len(scored), "infrastructure_failures": sum(_infra(r) for r in records if r.get("status") != "running"),
        "invalid_submissions": sum(r.get("status") in INVALID or r.get("score_status") == "invalid" for r in scored),
        "protocol_violations": sum(r.get("status") == "protocol_violation" or r.get("audit", {}).get("status") == "rejected" for r in scored),
        "unlaunched_sessions": len(expected) - len(records), "inflight_sessions": sum(r.get("status") == "running" for r in records),
        "valid_json_objects": sum(value is True for value in valid_json),
        "invalid_json_objects": sum(value is False for value in valid_json),
        "json_observation_unknown_sessions": sum(value is None for value in valid_json),
        "expected_independent_parents": len({r["parent_problem_id"] for r in expected}),
        "scored_independent_parents": len({r["parent_problem_id"] for r in scored}),
        "scored_mechanism_clusters": len({r["mechanism_family_id"] for r in scored}),
        "audit_statuses": dict(Counter(r.get("audit", {}).get("status", "unrecorded") for r in scored)),
        "outcome_counts": dict(Counter(_outcome(r) for r in records)),
        "resources": _resources(records),
    }
    if capability:
        exact = _cluster_mean(scored, _credit)
        result.update(exact_successes=sum(int(_credit(r)) for r in scored), exact_success=exact,
                      functional_accuracy=_cluster_mean(scored, _functional),
                      metrics={name: _cluster_mean(scored, lambda r, key=name: r.get("metrics", {}).get(key))
                               for name in sorted(METRICS) if any(name in r.get("metrics", {}) for r in scored)})
    return result


def _paired(records, left, right, expected):
    by_case = defaultdict(dict)
    for record in expected:
        if record["model_id"] in {left,right}:
            by_case[(record["task_id"],record["condition"],record["weight_view"],record["attempt"])]
    for record in records:
        if record["model_id"] in {left, right} and _scorable(record):
            key = (record["task_id"], record["condition"], record["weight_view"], record["attempt"])
            by_case[key][record["model_id"]] = record
    pairs, mismatches, incomplete = [], 0, 0
    for case in by_case.values():
        if left not in case or right not in case:
            incomplete += 1; continue
        a, b = case[left], case[right]
        required = ("target_digest", "question_digest", "serialization_digest", "protocol_digest", "parent_problem_id", "mechanism_family_id")
        if any(not a.get(key) or a.get(key) != b.get(key) for key in required):
            mismatches += 1; continue
        row = {"parent_problem_id": a["parent_problem_id"], "mechanism_family_id": a["mechanism_family_id"],
               "exact_delta": _credit(b) - _credit(a)}
        av, bv = _functional(a), _functional(b)
        if av is not None and bv is not None:
            row["functional_delta"] = bv - av
        pairs.append(row)
    return {"left_model": left, "right_model": right, "direction": "right_minus_left",
            "paired_cases": len(pairs), "incomplete_or_infrastructure_pairs": incomplete,
            "excluded_input_mismatch_pairs": mismatches,
            "exact_success_difference": _cluster_mean(pairs, lambda r: r["exact_delta"]),
            "functional_accuracy_difference": _cluster_mean(pairs, lambda r: r.get("functional_delta")),
            "uncertainty_note": "Tiny preselected development sample; no headline confidence interval or ranking claim."}


def _public_model(model):
    return {**{key: model.get(key) for key in ("id", "provider", "model", "snapshot", "snapshot_kind", "verified_at")},
            "reasoning_effort": model.get("generate_config", {}).get("reasoning_effort"),
            "reasoning_summary": model.get("generate_config", {}).get("reasoning_summary"),
            "responses_api": model.get("model_args", {}).get("responses_api"),
            "service_tier": model.get("model_args", {}).get("service_tier"),
            "pricing": {key:model.get("pricing",{}).get(key) for key in
                        ("input_per_million","cache_write_per_million","cache_read_per_million","output_per_million","source","verified_at","cache_ttl")},
            "configuration_digest": serialization_digest(model)}


def _write_control(records, expected):
    if not records:
        return None
    tasks = {row["task_id"]: row for row in expected if row["skill"] == "write"}
    seen, rows = set(), []
    for record in records:
        task_id = record["task_id"]
        if task_id not in tasks or task_id in seen:
            raise ValueError("Write controls must match each selected target exactly once")
        seen.add(task_id); task = tasks[task_id]
        if (record.get("actor_id") != "metadata_prior" or record.get("actor_type") != "baseline"
                or record.get("condition") != "B" or record.get("weight_view") != "full"
                or record.get("target_digest") != task["target_digest"]
                or (task.get("baseline_question_digest") and record.get("question_digest") != task["baseline_question_digest"])
                or record.get("family") != task["family"]):
            raise ValueError("Write control actor, access, family or target does not match")
        metrics = record.get("metrics", {})
        if metrics.get("edit_fraction") != 0 or metrics.get("edit_norm") != 0:
            raise ValueError("The static no-edit control must preserve every zero-start tensor")
        names = ("target_accuracy", "affected_accuracy", "retention_accuracy", "retention_loss")
        if any(not _number(metrics.get(name)) or not 0 <= metrics[name] <= 1 for name in names):
            raise ValueError("Write control metrics must be finite probabilities")
        rows.append({**task, "metrics": metrics, "success": bool(record.get("success")), "status": "completed"})
    if seen != set(tasks):
        raise ValueError("Write controls omit a preselected target")
    return {"name": "static_zero_start_no_edit", "source_actor": "metadata_prior", "recorded_condition": "B",
            "weight_view": "full", "same_access_model_score": False, "independent_tasks_added": 0,
            "selected_targets": len(rows), "exact_successes": sum(row["success"] for row in rows),
            "exact_success": _cluster_mean(rows, _credit),
            **{name: _cluster_mean(rows, lambda row, key=name: row["metrics"][key]) for name in names},
            "evidence_digest": serialization_digest(records),
            "interpretation": "Static difficulty control on the identical selected targets. Recorded in B, it reads no weights or feedback; it is neither an A model score nor additional independent tasks."}


def build_comparison(manifest, records, *, task_metadata, canaries=(), write_controls=()):
    """Pure aggregation. No private IDs, raw answers, prompts or arbitrary metadata are exported."""
    if manifest.get("phase") != "calibration":
        raise ValueError("This report accepts development calibration only")
    models = manifest.get("models", [])
    if not models or len({m["id"] for m in models}) != len(models):
        raise ValueError("Comparison manifest has missing or duplicate models")
    expected = {}
    for item in manifest.get("expected_matrix", []):
        if item.get("condition") != "A" or item.get("weight_view") != "full" or item.get("attempt") != 1:
            raise ValueError("The paired pilot requires one A/full attempt per model and target")
        metadata = task_metadata.get(item["task_id"])
        if not metadata:
            raise ValueError("Missing task-family metadata for an expected comparison cell")
        row = {**item, **{key: metadata[key] for key in ("family", "skill", "mechanism_family_id")}}
        if metadata.get("baseline_question_digest"):
            row["baseline_question_digest"] = metadata["baseline_question_digest"]
        row.setdefault("parent_problem_id", metadata["parent_problem_id"])
        key = _key(row)
        if key in expected:
            raise ValueError("Duplicate expected model attempt")
        expected[key] = row
    if not expected:
        raise ValueError("Empty comparison matrix")
    expected_models={r["model_id"] for r in expected.values()}
    if expected_models!={model["id"] for model in models}:
        raise ValueError("Expected matrix and configured models differ")
    case_sets=[{(r["task_id"],r["condition"],r["weight_view"],r["attempt"]) for r in expected.values() if r["model_id"]==model["id"]} for model in models]
    if any(cases!=case_sets[0] for cases in case_sets[1:]):
        raise ValueError("Every compared model must have the same predeclared cases")
    observed = {}
    for record in records:
        if record.get("mock") or record.get("usage_is_synthetic") or record.get("actor_type") != "model":
            raise ValueError("Mock or non-model records cannot become GPT statistics")
        if record.get("status") not in SCORED | INFRA | {"running"}:
            raise ValueError("An unrecognized session status cannot become comparison statistics")
        key = _key(record)
        if key not in expected or key in observed:
            raise ValueError("Unexpected or duplicate model attempt; never select the best record")
        planned = expected[key]
        for field in ("target_digest", "model_config_digest"):
            if planned.get(field) and planned[field] != record.get(field):
                raise ValueError("A result does not match the planned target/model configuration")
        for field in ("protocol_digest", "suite_digest", "harness_digest"):
            if manifest.get(field) and manifest[field] != record.get(field):
                raise ValueError("A result belongs to another protocol, suite or harness")
        for field in ("family", "skill", "parent_problem_id", "mechanism_family_id"):
            if planned.get(field) != record.get(field):
                raise ValueError("A result's task metadata changed")
        observed[key] = record
    model_reports = []
    for model in models:
        planned = [r for r in expected.values() if r["model_id"] == model["id"]]
        rows = [r for r in observed.values() if r["model_id"] == model["id"]]
        scopes = {}
        for scope, field in (("skills", "skill"), ("families", "family")):
            scopes[scope] = [{field: name, **_summary([r for r in planned if r[field] == name],
                                                    [r for r in rows if r[field] == name])}
                             for name in sorted({r[field] for r in planned})]
        model_reports.append({"configuration": _public_model(model),
                              "traffic_and_resources": _summary(planned, rows, capability=False), **scopes})
    pairs = []
    for left, right in combinations([m["id"] for m in models], 2):
        for kind, field in (("skill", "skill"), ("family", "family")):
            for name in sorted({r[field] for r in expected.values()}):
                rows = [r for r in observed.values() if r[field] == name]
                planned=[r for r in expected.values() if r[field]==name]
                pairs.append({"scope_type": kind, "scope": name, **_paired(rows, left, right, planned)})
    pending = sum(r.get("status") == "running" for r in observed.values())
    infra = sum(_infra(r) for r in observed.values() if r.get("status") != "running")
    scored = [r for r in observed.values() if _scorable(r)]
    complete = len(observed) == len(expected) and not pending
    review_complete = bool(scored) and all(r.get("audit", {}).get("status") in {"approved", "rejected"} for r in scored)
    unknown_cost = sum(not _number(r.get("cost_usd")) or r.get("usage_unknown") is True for r in observed.values())
    inputs_match=not any(pair["excluded_input_mismatch_pairs"] for pair in pairs)
    return {"benchmark_version": manifest.get("benchmark_version"), "report_type": "paired_GPT_development_comparison",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "headline_eligible": False, "website": "deferred", "condition": "A", "weight_view": "full",
            "suite_digest": manifest.get("suite_digest"), "harness_digest": manifest.get("harness_digest"),
            "protocol_digest": manifest.get("protocol_digest"), "manifest_digest": serialization_digest(manifest),
            "status": "complete" if complete and not infra and not unknown_cost and review_complete and inputs_match else "partial",
            "matrix_complete": complete, "review_complete": review_complete,
            "paired_inputs_match":inputs_match,
            "planned_sessions": len(expected), "launched_sessions": len(observed),
            "completed_scored_sessions": len(scored), "infrastructure_failures": infra,
            "inflight_sessions": pending, "missing_sessions": len(expected)-len(observed),
            "unknown_cost_sessions": unknown_cost, "models": model_reports,
            "paired_model_differences": pairs, "transport_canaries": [_public_canary(row) for row in canaries],
            "write_difficulty_control": _write_control(write_controls, list(expected.values())),
            "notes": ["No combined Read+Write capability score is calculated.",
                "Invalid submissions and protocol violations receive zero exact credit. Infrastructure failures are excluded from capability denominators and counted separately.",
                "Read functional accuracy is R1 agreement or the mean of R3 before/after agreement, once per parent. Write uses target accuracy. Invalid submissions/protocol violations receive zero derived functional credit; original measured family metrics retain their own observation counts.",
                "API output tokens include reported reasoning tokens. Reasoning is recorded separately and never charged twice. Visible-answer throughput subtracts reported reasoning tokens.",
                "Latency is local session elapsed time, not isolated provider compute time. Infrastructure latency is separate.",
                "Models are aliases unless their configuration states an immutable snapshot. This tiny development sample cannot support a headline ranking.",
                "Write outputs are class-imbalanced. Raw target accuracy alone does not demonstrate successful construction; compare the static zero-start/no-edit control, affected-case accuracy, retention and exact success.",
                "Skill and family rows overlap; do not sum them together. Transport canaries are not capability trials.",
                "Offline replay verifies saved submissions, scores and artifact consistency. It does not rerun proprietary model inference or independently attest provider billing."]}


def _trace_completion(path):
    if not path.exists():
        return None
    events = json.loads(path.read_text())
    turns = [e for e in events if e.get("tool") == "model_turn"]
    if not turns:
        return None
    output = turns[-1].get("output", {})
    if isinstance(output.get("completion"), str):
        return output["completion"]
    choices = output.get("choices", [])
    message = choices[0].get("message", {}) if choices else output.get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(item.get("text", "") for item in content if item.get("type") == "text")
    return None


def _public_canary(row):
    if row.get("benchmark_result") is not False:
        raise ValueError("A transport canary must explicitly disclaim benchmark-result status")
    return {"model_id": row.get("model_id"), "status": row.get("status"), "benchmark_result": False,
            "generation_requests": row.get("generation_requests"), "sdk_retries": row.get("sdk_retries"),
            "wall_seconds": row.get("wall_seconds"), "known_cost_estimate_usd": row.get("cost_usd",row.get("known_cost_estimate_usd")),
            "maximum_reserved_usd": row.get("maximum_reserved_usd"),
            "usage": {k: row.get("usage", {}).get(k) for k in TOKEN_FIELDS}}


def _format(value, *, percent=False, money=False):
    if value is None:
        return "unavailable"
    return f"${value:.6f}" if money else f"{value:.1%}" if percent else f"{value:.2f}"


def _markdown(report):
    lines = ["# GPT development comparison", "",
        "Same preselected development cases, A/full, one attempt per model. No combined capability score or leaderboard claim. Website work remains deferred.", "",
        f"Status: **{report['status']}**. Planned {report['planned_sessions']}; launched {report['launched_sessions']}; scored {report['completed_scored_sessions']}; infrastructure failures {report['infrastructure_failures']}; missing {report['missing_sessions']}; inflight {report['inflight_sessions']}.", "",
        "| Model | Read exact | Write exact | Confirmed JSON objects / responses | Median scored-session seconds | Known API estimate | Unknown-cost sessions |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for model in report["models"]:
        traffic = model["traffic_and_resources"]; resources = traffic["resources"]
        skills = {row["skill"]: row for row in model["skills"]}
        exact = lambda name: f"{skills[name]['exact_successes']}/{skills[name]['scored_sessions']}" if name in skills and skills[name]["scored_sessions"] else "unavailable"
        lines.append(f"| {model['configuration']['model']} | {exact('read')} | {exact('write')} | {traffic['valid_json_objects']}/{traffic['responded_sessions']} | {_format(resources['session_wall_seconds']['median'])} | {_format(resources['known_cost_estimate_usd'],money=True)} | {resources['unknown_cost_sessions']} |")
    lines += ["", "Infrastructure failures are not failed capability answers. Known cost excludes unresolved charges; it is not a complete total when unknown-cost sessions exist. Latency includes the local session and API round trip.", "",
        "## Per-family results", "", "| Model | Family | Exact / scored | Functional accuracy | Scored parents |", "|---|---|---:|---:|---:|"]
    for model in report["models"]:
        for family in model["families"]:
            lines.append(f"| {model['configuration']['id']} | {family['family']} | {family['exact_successes']}/{family['scored_sessions']} | {_format(family['functional_accuracy']['mean'],percent=True)} | {family['scored_independent_parents']} |")
    control = report.get("write_difficulty_control")
    if control:
        lines += ["", "## Write control context", "", control["interpretation"], "",
            f"The static zero-start/no-edit control has exact success **{control['exact_successes']}/{control['selected_targets']}**, target accuracy {_format(control['target_accuracy']['mean'],percent=True)}, affected-case accuracy {_format(control['affected_accuracy']['mean'],percent=True)}, and retention {_format(control['retention_accuracy']['mean'],percent=True)}. No independent tasks are added.", "",
            "Class imbalance makes raw Write accuracy insufficient evidence of successful construction.", "",
            "| Model | Write target accuracy | Affected-case accuracy | Retention | Exact / scored |", "|---|---:|---:|---:|---:|"]
        for model in report["models"]:
            write = next((row for row in model["skills"] if row["skill"] == "write"), None)
            if write:
                metric = lambda name: _format(write["metrics"].get(name, {}).get("mean"),percent=True)
                lines.append(f"| {model['configuration']['id']} | {metric('target_accuracy')} | {metric('affected_accuracy')} | {metric('retention_accuracy')} | {write['exact_successes']}/{write['scored_sessions']} |")
    lines += ["", "## API tokens", "", "Output includes API-reported reasoning; the reasoning column is a subset, not an additional billable output category.", "",
        "| Model | Input total | Uncached | Cache write | Cache read | Output | Reasoning subset |", "|---|---:|---:|---:|---:|---:|---:|"]
    for model in report["models"]:
        tokens = model["traffic_and_resources"]["resources"]["tokens"]
        lines.append(f"| {model['configuration']['id']} | {tokens['input_tokens_total']} | {tokens['input_tokens']} | {tokens['input_tokens_cache_write']} | {tokens['input_tokens_cache_read']} | {tokens['output_tokens']} | {tokens['reasoning_tokens']} |")
    lines += ["", "## Replay and accounting", "",
        "These are saved-submission results. CPU replay verifies their scores and artifact consistency without generating new model responses. API costs are estimates from released usage and frozen prices, not invoices.", ""]
    for canary in report["transport_canaries"]:
        lines.append(f"- {canary['model_id']} transport canary: {canary['status']}; {_format(canary['wall_seconds'])} seconds; {_format(canary['known_cost_estimate_usd'],money=True)} known estimate. Excluded from capability scores.")
    lines += ["", "## Interpretation", ""]
    lines.extend("- " + item for item in report["notes"])
    return "\n".join(lines) + "\n"


def _csv(report):
    stream = io.StringIO()
    fields = ["model_id", "scope_type", "scope", "requested", "launched", "responded", "scored", "successes",
              "exact_success_rate", "functional_accuracy", "infrastructure_failures", "invalid_submissions", "protocol_violations",
              "target_accuracy", "affected_accuracy", "retention_accuracy", "retention_loss",
              "valid_json_objects", "wall_median_seconds", "wall_min_seconds", "wall_max_seconds", "local_cpu_median_seconds",
              *TOKEN_FIELDS, "known_cost_estimate_usd", "unknown_cost_sessions", "unknown_reserved_upper_bound_usd"]
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n"); writer.writeheader()
    for model in report["models"]:
        for scope_type, key in (("skill", "skills"), ("family", "families")):
            for scope in model[key]:
                resource = scope["resources"]; wall = resource["session_wall_seconds"]
                writer.writerow({"model_id": model["configuration"]["id"], "scope_type": scope_type, "scope": scope[scope_type],
                    "requested": scope["requested_sessions"], "launched": scope["launched_sessions"],
                    "responded": scope["responded_sessions"], "scored": scope["scored_sessions"], "successes": scope["exact_successes"],
                    "exact_success_rate": scope["exact_success"]["mean"], "functional_accuracy": scope["functional_accuracy"]["mean"],
                    **{name: scope["metrics"].get(name, {}).get("mean") for name in
                       ("target_accuracy", "affected_accuracy", "retention_accuracy", "retention_loss")},
                    **{key: scope[key] for key in ("infrastructure_failures", "invalid_submissions", "protocol_violations", "valid_json_objects")},
                    "wall_median_seconds": wall["median"], "wall_min_seconds": wall["minimum"], "wall_max_seconds": wall["maximum"],
                    "local_cpu_median_seconds": resource["local_session_cpu_seconds"]["median"],
                    **resource["tokens"], **{key: resource[key] for key in ("known_cost_estimate_usd", "unknown_cost_sessions", "unknown_reserved_upper_bound_usd")}})
    return stream.getvalue()

