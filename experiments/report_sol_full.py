"""Summarize the matched Sol experiment from local records without API calls."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import random
from statistics import mean

from report_selective_repair import SCORED, ZERO_CREDIT, forecast_error, load_results, success


MODEL = "gpt-6.1-sol"
BEHAVIOR = (MODEL, "behavior")
WEIGHTS = (MODEL, "weights_behavior")
SCRIPTED = ("scripted", "behavior")
ARMS = ((BEHAVIOR, "Sol, behavior"), (WEIGHTS, "Sol, weights + behavior"),
        (SCRIPTED, "Scripted, behavior"))


def record(results, arm, case):
    model, access = arm
    return results.get((model, case["id"], access))


def score_count(scores):
    observed = [score for score in scores if score is not None]
    return f"{int(sum(observed))}/{len(observed)}" if observed else "—"


def pp(value):
    return f"{100 * value:+.1f} pp"


def dollars(value, digits=4):
    return "$" + f"{value:.{digits}f}"


def bootstrap_interval(deltas, *, samples=10_000, seed=82417):
    """Percentile interval resampling seed blocks and retaining case pairs."""
    blocks = list(deltas.values())
    if len(blocks) < 2:
        return None
    rng = random.Random(seed)
    replicates = sorted(
        mean(delta for block in rng.choices(blocks, k=len(blocks)) for delta in block)
        for _ in range(samples)
    )
    return replicates[int(0.025 * samples)], replicates[int(0.975 * samples) - 1]


def paired_outcomes(results, cases):
    deltas = defaultdict(list)
    low = high = 0.0
    for case in cases:
        behavior = success(record(results, BEHAVIOR, case))
        weights = success(record(results, WEIGHTS, case))
        # Each missing repair is binary, so vary it over both extremes.
        low += (0.0 if weights is None else weights) - (1.0 if behavior is None else behavior)
        high += (1.0 if weights is None else weights) - (0.0 if behavior is None else behavior)
        if behavior is not None and weights is not None:
            deltas[case["block"]].append(weights - behavior)
    return deltas, (low / len(cases), high / len(cases))


def exhausted_output(row, limit):
    if row["status"] not in ZERO_CREDIT:
        return False
    used = row.get("usage", {}).get("output_tokens", 0)
    return used >= limit or (
        row.get("error") == "Token budget exhausted before submission" and limit - used < 16
    )


def summarize(root):
    manifest, planned, results = load_results(root)
    cases = manifest["cases"]
    ids = {case["id"] for case in cases}
    faults = [case for case in cases if not case["clean"]]
    clean = [case for case in cases if case["clean"]]
    expected = {(model, case["id"], access) for (model, access), _ in ARMS for case in cases}
    if len(ids) != len(cases) or planned != expected or not faults or not clean:
        raise ValueError("Expected a matched Sol experiment with one baseline per case")
    if any(type(case.get("block")) is not int for case in cases):
        raise ValueError("Every case needs an integer seed block")

    limits = manifest["limits"]
    output_limit = limits["output_tokens"]
    print("# Full Sol selective repair experiment")
    print()
    print(f"{len(faults)} faults and {len(clean)} clean controls across "
          f"{len({case['block'] for case in cases})} seed blocks. "
          f"Each session has up to {limits['turns']} turns, {limits['input_tokens']:,} input tokens, "
          f"{output_limit:,} output tokens, {limits['pre_probes']} pre-commit probes and "
          f"{limits['post_probes']} post-commit probes.")
    print()
    deltas, bounds = paired_outcomes(results, faults)
    changes = [delta for block in deltas.values() for delta in block]
    if changes:
        print(f"Adding weights changed fault repair success by {pp(mean(changes))} across "
              f"{len(changes)}/{len(faults)} evaluable matched pairs "
              f"({changes.count(1.)} gained, {changes.count(-1.)} lost, {changes.count(0.)} tied).")
    else:
        print("The fault repair contrast has no evaluable matched pairs yet.")
    if interval := bootstrap_interval(deltas):
        print(f"The 95% percentile interval from 10,000 seed-block bootstrap draws is "
              f"{pp(interval[0])} to {pp(interval[1])}. It describes uncertainty across these generated cases.")
    if len(changes) < len(faults):
        print(f"Assigning unresolved sessions all possible binary outcomes gives an all-planned-pair "
              f"range of {pp(bounds[0])} to {pp(bounds[1])}.")
    valid_pairs = [
        success(weights) - success(behavior)
        for case in faults
        if (behavior := record(results, BEHAVIOR, case)) is not None
        and (weights := record(results, WEIGHTS, case)) is not None
        and behavior["status"] in SCORED and weights["status"] in SCORED
    ]
    if valid_pairs:
        print(f"Among the {len(valid_pairs)} fault pairs with valid patches in both arms, "
              f"the repair difference was {pp(mean(valid_pairs))}. This subset conditions on submission validity.")
    print("This comparison counts invalid submissions and protocol violations as failures. "
          "It omits infrastructure failures and pending sessions from the point estimate.")
    print()
    print("| Arm | Recorded / planned | Faults repaired / scored | Clean preserved / scored | "
          "Valid patches | Output caps | Other invalid | Infra failures | Forecasts / planned | Estimated API cost |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    total_cost = 0.0
    unknown_costs = 0
    for arm, label in ARMS:
        rows = [row for case in cases if (row := record(results, arm, case)) is not None]
        counts = Counter(row["status"] for row in rows)
        caps = sum(exhausted_output(row, output_limit) for row in rows)
        invalid = sum(counts[status] for status in ZERO_CREDIT)
        valid = sum(counts[status] for status in SCORED)
        infra = len(rows) - valid - invalid
        forecasts = sum(forecast_error(row) is not None for row in rows)
        known = [row["cost_usd"] for row in rows if row.get("cost_usd") is not None]
        unknown = len(rows) - len(known)
        cost = dollars(sum(known)) + (f" + {unknown} unknown" if unknown else "")
        if arm == SCRIPTED:
            cost = "—"
        else:
            total_cost += sum(known)
            unknown_costs += unknown
        faults_scored = score_count(success(record(results, arm, case)) for case in faults)
        clean_scored = score_count(success(record(results, arm, case)) for case in clean)
        print(f"| {label} | {len(rows)}/{len(cases)} | {faults_scored} | {clean_scored} | "
              f"{valid} | {caps} | {invalid - caps} | {infra} | {forecasts}/{len(cases)} | {cost} |")
    print()
    print(f"Total estimated API cost from recorded usage is {dollars(total_cost)}. "
          f"{unknown_costs} recorded sessions have unknown cost. "
          f"The frozen reservation ceiling is {dollars(manifest['maximum_cost_usd'], 2)}.")
    print()
    print("| Fault mechanism | Planned pairs | Behavior repairs | With weights repairs | "
          "Scripted repairs | Matched repair change | Matched forecast MAE change |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for mechanism in sorted({case["mechanism"] for case in faults}) + ["All faults"]:
        group = faults if mechanism == "All faults" else [case for case in faults if case["mechanism"] == mechanism]
        by_arm = {arm: score_count(success(record(results, arm, case)) for case in group) for arm, _ in ARMS}
        paired, _ = paired_outcomes(results, group)
        scores = [delta for block in paired.values() for delta in block]
        forecast_pairs = [
            right - left for case in group
            if (left := forecast_error(record(results, BEHAVIOR, case))) is not None
            and (right := forecast_error(record(results, WEIGHTS, case))) is not None
        ]
        effect = f"{pp(mean(scores))} ({len(scores)} pairs)" if scores else "—"
        forecast_effect = f"{mean(forecast_pairs):+.3f} ({len(forecast_pairs)} pairs)" if forecast_pairs else "—"
        name = mechanism.replace("_", " ")
        print(f"| {name} | {len(group)} | {by_arm[BEHAVIOR]} | {by_arm[WEIGHTS]} | "
              f"{by_arm[SCRIPTED]} | {effect} | {forecast_effect} |")
    print()
    print("Forecast changes compare committed estimates for the same case. A negative change means "
          "lower error with weights. Counts exclude sessions without a committed forecast.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Private full experiment run directory")
    summarize(parser.parse_args().output)
