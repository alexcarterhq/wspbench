"""Compare full selective repair runs on the same cases and protocol."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean

from report_selective_repair import SCORED, ZERO_CREDIT, cell_key, forecast_error, load_results, success
from report_sol_full import bootstrap_interval, exhausted_output
from wspbench.budget import session_cost
from wspbench.v2.serialization import serialization_digest


ACCESS = ("behavior", "weights_behavior")
SCRIPTED = "scripted"
PROTOCOL_FIELDS = ("cases", "system", "arms", "limits", "primary_metric")
RETRYABLE_ERRORS = {"APIError", "APITimeoutError"}


def load_comparison(roots, retry_root=None):
    reference = None
    models = {}
    results = {}
    retry_candidates = {}
    retries = {}
    retry_manifest = None
    retry_plan = None
    if retry_root is not None:
        retry_manifest, _, retries = load_results(retry_root)
        retry_plan = json.loads((retry_root / "retry-plan.json").read_text())
    for root in roots:
        manifest, planned, rows = load_results(root)
        if reference is None:
            reference = manifest
        else:
            for field in PROTOCOL_FIELDS:
                if manifest[field] != reference[field]:
                    raise ValueError(f"{root.name} differs on {field}")
            for name in ("repair_cases.py", "repair_hard_cases.py"):
                if manifest["source_digests"][name] != reference["source_digests"][name]:
                    raise ValueError(f"{root.name} differs on {name}")
        roster = {model["snapshot"]: model for model in manifest["models"]}
        if not roster or len(roster) != len(manifest["models"]):
            raise ValueError(f"Invalid model roster in {root.name}")
        cases = manifest["cases"]
        ids = {case["id"] for case in cases}
        if len(ids) != len(cases) or any(type(case.get("block")) is not int for case in cases):
            raise ValueError(f"Invalid case metadata in {root.name}")
        expected = {(model, case_id, access) for model in roster for case_id in ids for access in ACCESS}
        expected.update((SCRIPTED, case_id, "behavior") for case_id in ids)
        if planned != expected:
            raise ValueError(f"{root.name} does not contain the full matched design")
        for model, config in roster.items():
            if model in models:
                raise ValueError(f"Multiple run directories for {model}")
            models[model] = config
        for key, row in rows.items():
            if key in results and results[key] != row:
                raise ValueError(f"Scripted baseline changed on {key[1]}")
            results[key] = row

        if retry_root is not None and manifest == retry_manifest:
            retry_candidates = {
                key: row for key, row in rows.items()
                if key[0] != SCRIPTED and row["status"] == "infrastructure_failure"
                and row.get("error") in RETRYABLE_ERRORS
            }

    if retry_root is not None:
        if not retry_candidates:
            raise ValueError("Retry manifest must match an original run with API failures")
        if set(retries) - retry_candidates.keys():
            raise ValueError("Retry records must belong to original API failures")
        manifest_digest = serialization_digest(retry_manifest)
        planned_retries = [cell_key(cell) for cell in retry_plan["cells"]]
        if (retry_plan["original_manifest_digest"] != manifest_digest
                or len(planned_retries) != len(retry_candidates)
                or set(planned_retries) != retry_candidates.keys()):
            raise ValueError("Retry plan must match the exact set of original API failures")
        concurrency = retry_plan["concurrency"]
        if type(concurrency) is not int or concurrency < 1:
            raise ValueError("Retry plan must record positive concurrency")
        for key, row in retries.items():
            original = retry_candidates[key]
            if row.get("prompt_digest") != original["prompt_digest"]:
                raise ValueError(f"Retry prompt differs for {key!r}")
            if row.get("retry_of") != original["reservation_id"]:
                raise ValueError(f"Retry lineage differs for {key!r}")
            cell = dict(zip(("model", "case", "arm"), key, strict=True))
            digest = serialization_digest([manifest_digest, cell])
            reservation = serialization_digest(["api-error-retry-1", digest])
            if row.get("reservation_id") != reservation:
                raise ValueError(f"Unexpected retry reservation for {key!r}")
            path = retry_root / "runs" / digest / "result.json"
            if not path.is_file():
                raise ValueError(f"Retry record path differs from the original cell hash for {key!r}")

    prompt_digests = defaultdict(set)
    for (model, case_id, access), row in results.items():
        if model != SCRIPTED:
            prompt_digests[case_id, access].add(row["prompt_digest"])
    if any(len(digests) > 1 for digests in prompt_digests.values()):
        raise ValueError("Initial model prompts differ within an access condition")
    return reference, models, results, retries, retry_plan


def repair_count(rows):
    scores = [success(row) for row in rows]
    scored = [score for score in scores if score is not None]
    return {"passed": int(sum(scored)), "scored": len(scored), "planned": len(scores)}


def metered_cost(records, config):
    costs = []
    unknown = 0
    for row in records:
        value = row.get("cost_usd")
        if value is None:
            unknown += 1
            if not row.get("responses"):
                continue
            value = session_cost(config, row["usage"])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid API cost for {config['snapshot']}")
        costs.append(value)
    return sum(costs), unknown


def summarize_arm(results, cases, model, access, output_limit, config=None,
                  *, originals=None, retries=None):
    keys = [(model, case["id"], access) for case in cases]
    rows = [results.get(key) for key in keys]
    recorded = [row for row in rows if row is not None]
    initial_cost, initial_unknown = (0.0, 0)
    retry_cost, retry_unknown = (0.0, 0)
    if config is not None:
        initial_cost, initial_unknown = metered_cost((originals[key] for key in keys if key in originals), config)
        retry_cost, retry_unknown = metered_cost((retries[key] for key in keys if key in retries), config)
    return {
        "faults": repair_count(row for row, case in zip(rows, cases, strict=True) if not case["clean"]),
        "clean": repair_count(row for row, case in zip(rows, cases, strict=True) if case["clean"]),
        "recorded": len(recorded),
        "valid": sum(row["status"] in SCORED for row in recorded),
        "invalid": sum(row["status"] in ZERO_CREDIT for row in recorded),
        "infrastructure_failures": sum(row["status"] not in SCORED | ZERO_CREDIT for row in recorded),
        "output_caps": sum(exhausted_output(row, output_limit) for row in recorded),
        "cost_usd": initial_cost + retry_cost if model != SCRIPTED else None,
        "unknown_costs": initial_unknown + retry_unknown,
        "retry_cost_usd": retry_cost,
        "retry_unknown_costs": retry_unknown,
    }


def paired_effect(results, cases, model):
    deltas = defaultdict(list)
    forecasts = []
    low = high = 0.0
    for case in cases:
        behavior, weights = (results.get((model, case["id"], arm)) for arm in ACCESS)
        left, right = success(behavior), success(weights)
        low += (0.0 if right is None else right) - (1.0 if left is None else left)
        high += (1.0 if right is None else right) - (0.0 if left is None else left)
        if left is not None and right is not None:
            deltas[case["block"]].append(right - left)
        left, right = forecast_error(behavior), forecast_error(weights)
        if left is not None and right is not None:
            forecasts.append(right - left)
    changes = [delta for block in deltas.values() for delta in block]
    return {
        "planned_pairs": len(cases),
        "scored_pairs": len(changes),
        "repair_change": mean(changes) if changes else None,
        "interval": bootstrap_interval(deltas),
        "all_pairs_bounds": (low / len(cases), high / len(cases)),
        "gained": changes.count(1.0),
        "lost": changes.count(-1.0),
        "tied": changes.count(0.0),
        "forecast_mae_change": mean(forecasts) if forecasts else None,
        "forecast_pairs": len(forecasts),
    }


def summarize(roots, retry_root=None):
    manifest, models, originals, retries, retry_plan = load_comparison(roots, retry_root)
    results = originals | retries
    cases = manifest["cases"]
    faults = [case for case in cases if not case["clean"]]
    limits = manifest["limits"]
    output_limit = limits["output_tokens"]
    mechanisms = {kind: [case for case in faults if case["mechanism"] == kind]
                  for kind in sorted({case["mechanism"] for case in faults})}
    baseline = summarize_arm(results, cases, SCRIPTED, "behavior", output_limit)
    models_summary = []
    for model, config in models.items():
        arms = {
            arm: summarize_arm(results, cases, model, arm, output_limit, config,
                               originals=originals, retries=retries)
            for arm in ACCESS
        }
        served = sorted({response["model"]
                         for batch in (originals, retries)
                         for key, row in batch.items() if key[0] == model
                         for response in row.get("responses", [])})
        models_summary.append({
            "model": model,
            "reasoning_effort": config["generate_config"]["reasoning_effort"],
            "served_models": served,
            "arms": arms,
            "paired": paired_effect(results, faults, model),
            "mechanisms": {
                kind: {arm: repair_count(results.get((model, case["id"], arm)) for case in group)
                       for arm in ACCESS}
                for kind, group in mechanisms.items()
            },
        })
    arms = [arm for model in models_summary for arm in model["arms"].values()]
    return {
        "faults": len(faults),
        "clean": len(cases) - len(faults),
        "blocks": len({case["block"] for case in cases}),
        "limits": limits,
        "recorded": sum(arm["recorded"] for arm in arms),
        "planned": len(models) * len(cases) * len(ACCESS),
        "cost_usd": sum(arm["cost_usd"] for arm in arms),
        "unknown_costs": sum(arm["unknown_costs"] for arm in arms),
        "retry": {
            "planned": len(retry_plan["cells"]),
            "recorded": len(retries),
            "resolved": sum(row["status"] in SCORED | ZERO_CREDIT for row in retries.values()),
            "cost_usd": sum(arm["retry_cost_usd"] for arm in arms),
            "unknown_costs": sum(arm["retry_unknown_costs"] for arm in arms),
            "concurrency": retry_plan["concurrency"],
        } if retry_root is not None else None,
        "models": models_summary,
        "baseline": baseline,
    }


def count(metric):
    return f"{metric['passed']}/{metric['scored']}" if metric["scored"] else "—"


def pp(value):
    return f"{100 * value:+.1f} pp" if value is not None else "—"


def interval(values):
    return " to ".join(map(pp, values)) if values else "—"


def dollars(value, unknown=0):
    return f"${value:.4f}" + (f" ({unknown} attempts with incomplete usage)" if unknown else "")


def render(summary):
    limits = summary["limits"]
    retry = summary["retry"]
    retry_note = ""
    if retry is not None:
        retry_note = (
            f"\n\nWe retried only sessions interrupted by API errors, reducing concurrency "
            f"from 16 to {retry['concurrency']}. "
            f"{retry['recorded']}/{retry['planned']} retries returned records and "
            f"{retry['resolved']} resolved the original API interruption. The comparison uses "
            "the retry outcome for those sessions and keeps both attempts in the cost totals. "
            f"Metered retry cost was {dollars(retry['cost_usd'], retry['unknown_costs'])}."
        )
    lines = [
        "# Full selective repair model comparison",
        "",
        f"Each model faced {summary['faults']} faults and {summary['clean']} clean controls from "
        f"{summary['blocks']} seed blocks, once with behavior probes and once with the same probes "
        f"plus weights. Limits were {limits['turns']} turns, {limits['input_tokens']:,} input tokens, "
        f"{limits['output_tokens']:,} output tokens, {limits['pre_probes']} pre-commit probes, "
        f"and {limits['post_probes']} post-commit probes per session.",
        "",
        "The comparison reuses the original Sol run. Astra, Luna, and Terra ran in a separate "
        "interleaved batch, initially with one attempt per case and condition. All four ran at high reasoning "
        f"effort using the listed API aliases.{retry_note}",
        "",
        f"Recorded {summary['recorded']}/{summary['planned']} planned API sessions. "
        f"Estimated cost from recorded usage was {dollars(summary['cost_usd'], summary['unknown_costs'])}.",
        "",
        "| API model | Reasoning effort | Access | Faults repaired | Clean preserved | Recorded / planned | Invalid (output cap) | Infrastructure failures | Estimated cost |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in summary["models"]:
        for access, label in zip(ACCESS, ("Behavior probes", "Weights and behavior probes"), strict=True):
            arm = model["arms"][access]
            planned = arm["faults"]["planned"] + arm["clean"]["planned"]
            lines.append(
                f"| `{model['model']}` | {model['reasoning_effort']} | {label} | "
                f"{count(arm['faults'])} | {count(arm['clean'])} | {arm['recorded']}/{planned} | "
                f"{arm['invalid']} ({arm['output_caps']}) | {arm['infrastructure_failures']} | "
                f"{dollars(arm['cost_usd'], arm['unknown_costs'])} |"
            )
    baseline = summary["baseline"]
    baseline_planned = baseline["faults"]["planned"] + baseline["clean"]["planned"]
    lines.extend([
        f"| Scripted baseline | — | Behavior probes | {count(baseline['faults'])} | "
        f"{count(baseline['clean'])} | {baseline['recorded']}/{baseline_planned} | — | — | — |",
        "",
        "Invalid submissions and protocol violations count as failures. Pending sessions and infrastructure "
        "failures are excluded from score denominators. Costs include metered turns from interrupted "
        "attempts. Incomplete usage records leave some charges unknown.",
        "",
        "| API model | Matched fault pairs | Repairs gained / lost / tied | Change with weights | 95% interval | Forecast MAE change (pairs) | Model API cost |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for model in summary["models"]:
        effect = model["paired"]
        arms = list(model["arms"].values())
        cost = dollars(sum(arm["cost_usd"] for arm in arms), sum(arm["unknown_costs"] for arm in arms))
        forecast = effect["forecast_mae_change"]
        forecast_text = f"{forecast:+.3f} ({effect['forecast_pairs']})" if forecast is not None else "—"
        lines.append(
            f"| `{model['model']}` | {effect['scored_pairs']}/{effect['planned_pairs']} | "
            f"{effect['gained']} / {effect['lost']} / {effect['tied']} | {pp(effect['repair_change'])} | "
            f"{interval(effect['interval'])} | {forecast_text} | {cost} |"
        )
    lines.extend([
        "",
        "Each interval resamples seed blocks in 10,000 bootstrap draws. Negative forecast MAE changes "
        "mean lower error with weights. Forecast differences use pairs with commitments in both arms.",
    ])
    for model in summary["models"]:
        effect = model["paired"]
        if effect["scored_pairs"] < effect["planned_pairs"]:
            lines.append(f"Assigning all possible binary outcomes to unresolved sessions gives "
                         f"`{model['model']}` an all-pairs effect range of "
                         f"{interval(effect['all_pairs_bounds'])}.")
        served = model["served_models"]
        if served and served != [model["model"]]:
            lines.append(f"The API reported {', '.join(f'`{alias}`' for alias in served)} "
                         f"when called with `{model['model']}`.")
    lines.extend([
        "",
        "| API model | Fault mechanism | Behavior probes | Weights and behavior probes |",
        "|---|---|---:|---:|",
    ])
    for model in summary["models"]:
        for kind, arms in model["mechanisms"].items():
            lines.append(f"| `{model['model']}` | {kind.replace('_', ' ')} | "
                         f"{count(arms[ACCESS[0]])} | {count(arms[ACCESS[1]])} |")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path, help="Private full experiment directories")
    parser.add_argument("--retry", type=Path, help="Private directory with retries of interrupted API sessions")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable aggregate metrics")
    args = parser.parse_args()
    summary = summarize(args.roots, args.retry)
    print(json.dumps(summary, indent=2) if args.json else render(summary))
