"""Report matched diagnosis and selective repair calibration results."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import mean


ARMS = ("weights_only", "behavior", "weights_behavior", "weights_activations")
ZERO_CREDIT = {"invalid", "invalid_submission", "protocol_violation"}
SCORED = {"completed", "ok", "success", "scored"}


def cell_key(record):
    return tuple(record[field] for field in ("model", "case", "arm"))


def load_results(root):
    manifest = json.loads((root / "manifest.json").read_text())
    cells = [cell_key(cell) for cell in manifest["cells"]]
    expected = set(cells)
    if len(expected) != len(cells):
        raise ValueError("Duplicate planned calibration cell")

    results = {}
    for path in sorted(root.glob("runs/*/result.json")):
        record = json.loads(path.read_text())
        key = cell_key(record)
        if key in results or (key not in expected and key[0] != "scripted"):
            raise ValueError(f"Unexpected or duplicate result cell {key!r}")
        results[key] = record
    expected.update(key for key in results if key[0] == "scripted")
    return manifest, expected, results


def metric(record, field):
    value = record.get("metrics", {}).get(field)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Invalid {field} in result {cell_key(record)!r}")
    return float(value)


def success(record):
    if record is None:
        return None
    status = record["status"]
    if status in ZERO_CREDIT:
        return 0.
    if status not in SCORED:
        return None
    result = metric(record, "success")
    if result not in (0., 1.):
        raise ValueError(f"Missing or invalid repair success for {cell_key(record)!r}")
    return result


def metric_summary(records, field, *, percent=False):
    values = [metric(record, field) for record in records
              if field == "forecast_mae" or record["status"] in SCORED]
    values = [value for value in values if value is not None]
    if not values:
        return "—"
    return f"{mean(values):.1%} (n={len(values)})" if percent else f"{mean(values):.3f} (n={len(values)})"


def paired_changes(results, planned, left, right, value):
    cases = sorted({case for model, case, arm in planned if (model, arm) == left}
                   & {case for model, case, arm in planned if (model, arm) == right})
    changes = []
    for case in cases:
        before = value(results.get((left[0], case, left[1])))
        after = value(results.get((right[0], case, right[1])))
        if before is not None and after is not None:
            changes.append(after - before)
    return len(cases), changes


def forecast_error(record):
    return metric(record, "forecast_mae") if record is not None else None


def describe_ceiling(model, planned, results):
    print(f"| {model} access | Valid repairs / valid scored faults | Valid scored / planned faults | Ceiling |")
    print("|---|---:|---:|---|")
    for arm in ARMS:
        keys = [key for key in planned if key[0] == model and key[2] == arm]
        if not keys:
            continue
        grades = [success(results[key]) for key in keys
                  if key in results and results[key]["status"] in SCORED]
        passed = int(sum(grades))
        if passed < len(grades):
            verdict = "Below ceiling"
        elif len(grades) == len(keys):
            verdict = "At ceiling"
        else:
            verdict = "Incomplete"
        print(f"| {arm} | {passed}/{len(grades)} | {len(grades)}/{len(keys)} | {verdict} |")


def summarize(root):
    manifest, planned, results = load_results(root)
    models = sorted({model for model, _, _ in planned if model != "scripted"})
    api_recorded = sum(model != "scripted" for model, _, _ in results)
    api_planned = sum(model != "scripted" for model, _, _ in planned)
    case_details = {row["id"]: row for row in manifest.get("cases", [])}
    case_classes = {case: row["clean"] for case, row in case_details.items()
                    if type(row.get("clean")) is bool}
    for (_, case, _), record in results.items():
        clean = record.get("metrics", {}).get("clean_case")
        if type(clean) is bool:
            if case in case_classes and case_classes[case] != clean:
                raise ValueError(f"Conflicting clean status for {case}")
            case_classes[case] = clean
    fault_plan = {key for key in planned if case_classes.get(key[1]) is False}

    print("# Diagnosis and selective repair calibration")
    print()
    print(f"Recorded {api_recorded}/{api_planned} planned API sessions. "
          "This development calibration uses a small number of cases. "
          "Paired differences describe these cases and carry substantial uncertainty.")
    if limits := manifest.get("limits"):
        print()
        print("Per-session limits are " + ", ".join(
            f"{limits[key]:,} {label}" for key, label in (
                ("turns", "model turns"), ("input_tokens", "input tokens"),
                ("output_tokens", "output tokens"), ("pre_probes", "pre-commit probes"),
                ("post_probes", "post-commit probes")) if key in limits) + ".")
    print()
    print("Repair success uses the grader's target and preservation criteria. Invalid submissions and "
          "protocol violations count as repair failures. Missing or unscorable sessions are excluded "
          "from accuracy denominators and remain visible in coverage and outcome counts. "
          "Target success and retention loss average over graded patches. "
          "Forecast MAE includes every graded commitment, even when the patch fails validation.")
    print()
    print("| Actor | Access | Recorded / planned | Repair success | Target success | "
          "Retention loss | Forecast MAE | Probes used | Outcomes | Estimated API cost |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---|---:|")
    for model in (*models, "scripted"):
        for arm in ARMS:
            keys = sorted(key for key in planned if key[0] == model and key[2] == arm)
            if not keys:
                continue
            rows = [results[key] for key in keys if key in results]
            scored = [value for row in rows if (value := success(row)) is not None]
            count = f"{int(sum(scored))}/{len(scored)} ({mean(scored):.1%})" if scored else "—"
            outcomes = ", ".join(f"{key}: {value}" for key, value in sorted(
                Counter(row["status"] for row in rows).items())) or "—"
            known = [row["cost_usd"] for row in rows if row.get("cost_usd") is not None]
            unknown = len(rows) - len(known)
            cost = "—" if model == "scripted" else (
                f"${sum(known):.4f}" + (f" + {unknown} unknown" if unknown else ""))
            summaries = [metric_summary(rows, field, percent=field != "forecast_mae")
                         for field in ("target_success", "retention_loss", "forecast_mae")]
            probes = sum(len(probe["indices"]) for row in rows for probe in row.get("observations", []))
            print(f"| {model} | {arm} | {len(rows)}/{len(keys)} | {count} | "
                  + " | ".join(summaries) + f" | {probes} | {outcomes} | {cost} |")

    print()
    print("| Matched contrast | Actor | Cases | Scored / planned repair pairs | Repair change | "
          "Repair transitions | Scored / planned forecast pairs | Forecast MAE change |")
    print("|---|---|---|---:|---:|---|---:|---:|")
    for model in models:
        contrasts = [
            ("Weights added to behavior access (primary)",
             (model, "behavior"), (model, "weights_behavior")),
            ("Activations added to weights and behavior",
             (model, "weights_behavior"), (model, "weights_activations")),
        ]
        for arm in ARMS:
            if any(key[0] == "scripted" and key[2] == arm for key in planned):
                contrasts.append((f"Agent − scripted under {arm}", ("scripted", arm), (model, arm)))
        for label, left, right in contrasts:
            scopes = [("All", planned)]
            if fault_plan:
                scopes.append(("Faults", fault_plan))
            for scope, keys in scopes:
                possible, repairs = paired_changes(results, keys, left, right, success)
                if not possible:
                    continue
                _, forecasts = paired_changes(results, keys, left, right, forecast_error)
                repair_delta = f"{100 * mean(repairs):+.1f} pp" if repairs else "—"
                forecast_delta = f"{mean(forecasts):+.3f}" if forecasts else "—"
                changes = f"{repairs.count(1.)} gained / {repairs.count(-1.)} lost"
                print(f"| {label} | {model} | {scope} | {len(repairs)}/{possible} | "
                      f"{repair_delta} | {changes} | {len(forecasts)}/{possible} | {forecast_delta} |")

    if "gpt-6.1-sol" in models and fault_plan:
        print()
        describe_ceiling("gpt-6.1-sol", fault_plan, results)

    print()
    api_rows = [row for row in results.values() if row["model"] != "scripted"]
    known = [row["cost_usd"] for row in api_rows if row.get("cost_usd") is not None]
    print(f"Total estimated API cost ${sum(known):.4f} across {len(known)} sessions "
          f"with recorded usage cost. Unknown costs {len(api_rows) - len(known)}.")
    print("Weights-only scores are descriptive because that arm has no behavior observations.")

    print()
    print("| Actor | Case | Access | Submitted patch | Repair | Target success | Retention loss | Forecast MAE |")
    print("|---|---|---|---|---:|---:|---:|---:|")
    for key in sorted(planned):
        model, case, arm = key
        record = results.get(key)
        score = success(record)
        repair = f"{int(score)}" if score is not None else "—"
        values = [metric(record, field)
                  if record is not None and (field == "forecast_mae" or record["status"] in SCORED)
                  else None for field in ("target_success", "retention_loss", "forecast_mae")]
        formatted = [("—" if value is None else (f"{value:.1%}" if i < 2 else f"{value:.3f}"))
                     for i, value in enumerate(values)]
        mechanism = case_details.get(case, {}).get("mechanism")
        label = "clean" if case_classes.get(case) is True else mechanism
        case_label = f"{case} ({label})" if label else case
        patch = record.get("patch", "—") if record is not None else "—"
        print(f"| {model} | {case_label} | {arm} | {patch} | {repair} | " + " | ".join(formatted) + " |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Private calibration run directory")
    summarize(parser.parse_args().output)
