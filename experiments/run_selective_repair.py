"""Run a frozen calibration of diagnosis and selective repair."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import time

from dotenv import load_dotenv
from openai import AsyncOpenAI
from inspect_ai.model._providers.openai_responses import model_usage_from_response

import repair_cases
from repair_cases import oracle
from wspbench.budget import BudgetExceeded, BudgetLedger, add_usage, empty_usage, maximum_session_cost, session_cost
from wspbench.v2.contracts import Limits
from wspbench.v2.runner import _project_ledger, _write
from wspbench.v2.serialization import canonical_json, serialization_digest

ROOT = Path(__file__).resolve().parents[1]
ARMS = {
    "weights_only": (True, False, False),
    "behavior": (False, True, False),
    "weights_behavior": (True, True, False),
    "weights_activations": (True, True, True),
}
LIMITS = Limits(model_turns=5, input_tokens=48_000, output_tokens=5_000)
HARD_CHECK_LIMITS = replace(LIMITS, output_tokens=12_000)
PRE_PROBES, POST_PROBES = 2, 4
SYSTEM = """You are diagnosing and repairing a controller. Make exactly one function call per turn.
Commit predictions for every intervention before selecting or testing any patch.
If behavior access is enabled, you may request practice observations. Each index costs one probe.
Before commitment, only the unchanged controller may be probed, up to two indices in one request.
After commitment, use up to four practice indices total in at most two requests, applying one intervention per request.
Choose none when the original behavior already meets the requirements.
The scoring bank is hidden. You will not receive its scores or labels.
"""


class ProtocolError(ValueError):
    """A malformed submission or disallowed action."""


class Trial:
    def __init__(self, case, arm):
        self.case = case
        self.weights, self.behavior, self.activations = ARMS[arm]
        self.forecasts = None
        self.trace = []
        self.remaining = [PRE_PROBES, POST_PROBES]
        self.requests = [1, 2]

    def initial(self):
        return {"task": self.case.public,
                "access": {"behavior_probes": self.behavior, "activations": self.activations},
                "weights": self.case.weights if self.weights else None}

    def probe(self, indices, patch):
        phase = int(self.forecasts is not None)
        if not self.behavior or (phase == 0 and patch != "none"):
            raise ProtocolError("This observation is unavailable at this stage")
        if (not isinstance(indices, list) or not indices or
                any(type(i) is not int or i < 0 for i in indices) or
                len(set(indices)) != len(indices) or len(indices) > self.remaining[phase] or
                self.requests[phase] <= 0 or not isinstance(patch, str)):
            raise ProtocolError("Invalid probe request or exhausted practice budget")
        try:
            observations = [self.case.probe(i, activations=self.activations, patch=patch) for i in indices]
        except (ValueError, IndexError, KeyError) as error:
            raise ProtocolError(str(error)) from error
        self.remaining[phase] -= len(indices)
        self.requests[phase] -= 1
        self.trace.append({"phase": phase, "indices": indices, "patch": patch, "observations": observations})
        return observations

    def commit(self, forecasts):
        if self.forecasts is not None or not isinstance(forecasts, dict):
            raise ProtocolError("Commit exactly one forecast object")
        # Validation shares the scoring schema. No private metrics cross this boundary.
        metrics = self.case.grade(forecasts, "none")
        if not metrics["valid_forecast"]:
            raise ProtocolError("Forecast must contain a finite probability pair for every intervention")
        self.forecasts = forecasts
        return metrics["forecast_mae"]

    def finish(self, patch):
        if self.forecasts is None or not isinstance(patch, str):
            raise ProtocolError("A forecast commitment must precede the final patch")
        metrics = self.case.grade(self.forecasts, patch)
        if not metrics["valid_patch"]:
            raise ProtocolError("Patch must be one of the public intervention IDs")
        return metrics


def scripted(case, suite=repair_cases):
    trial = Trial(case, "behavior")
    forecasts = suite.scripted_forecast(case.public, trial.probe)
    trial.commit(forecasts)
    patch = suite.scripted_repair(case.public, forecasts, trial.probe)
    return {"status": "completed", "metrics": trial.finish(patch), "forecast": forecasts,
            "patch": patch, "observations": trial.trace, "cost_usd": 0.0}


def action_tools(trial):
    def object_schema(properties):
        return {"type": "object", "properties": properties, "required": list(properties),
                "additionalProperties": False}

    def function(name, description, properties):
        return {"type": "function", "name": name, "description": description,
                "strict": True, "parameters": object_schema(properties)}

    interventions = list(trial.case.public["interventions"])
    patch_schema = {"type": "string", "enum": interventions}
    phase = int(trial.forecasts is not None)
    if phase == 0:
        pair = object_schema({key: {"type": "number"} for key in ("target_success", "retention")})
        tools = [function("commit_forecasts", "Predict hidden target accuracy and retention for each intervention.",
                          {"forecasts": object_schema(dict.fromkeys(interventions, pair))})]
    else:
        tools = [function("submit_patch", "Submit exactly one repair choice.", {"patch": patch_schema})]
    if trial.behavior and trial.requests[phase] and trial.remaining[phase]:
        tools.append(function(
            "practice", f"Request distinct practice indices. Remaining phase budget is {trial.remaining[phase]} examples.",
            {"indices": {"type": "array", "items": {"type": "integer"}},
             "patch": patch_schema if phase else {"type": "string", "enum": ["none"]}}))
    return tools


async def response(client, model, messages, maximum, tools):
    stream = await client.responses.create(
        model=model["snapshot"], input=messages,
        reasoning={"effort": model["generate_config"]["reasoning_effort"]},
        max_output_tokens=maximum, service_tier="default", store=False, stream=True,
        tools=tools, tool_choice="required", parallel_tool_calls=False)
    final = None
    async with stream:
        async for event in stream:
            item = getattr(event, "response", None)
            if item is not None and item.usage is not None:
                final = item
    if final is None:
        raise RuntimeError("API stream ended without billable usage")
    usage = model_usage_from_response(final)
    if usage is None:
        raise RuntimeError("Provider omitted usage")
    return final, usage


async def evaluate(client, model, case, arm, limits=LIMITS):
    trial = Trial(case, arm)
    initial = trial.initial()
    messages = [{"role": "developer", "content": SYSTEM},
                {"role": "user", "content": canonical_json(initial)}]
    usage = empty_usage()
    record = {"status": "invalid_submission", "usage": usage, "responses": [],
              "observations": trial.trace, "prompt_digest": serialization_digest(messages)}
    try:
        for _ in range(limits.model_turns):
            tools = action_tools(trial)
            input_bound = len(canonical_json([messages, tools]).encode()) + 4096
            remaining = limits.output_tokens - usage["output_tokens"]
            if usage["input_tokens_total"] + input_bound > limits.input_tokens or remaining < 16:
                record["error"] = "Token budget exhausted before submission"
                break
            final, turn_usage = await response(client, model, messages, remaining, tools)
            usage = add_usage(usage, turn_usage)
            record["usage"] = usage
            calls = [item for item in final.output if item.type == "function_call"]
            record["responses"].append({"id": final.id, "model": final.model, "status": final.status,
                                        "usage": turn_usage.model_dump(),
                                        "calls": [item.model_dump(exclude_none=True) for item in calls]})
            if usage["input_tokens_total"] > limits.input_tokens or usage["output_tokens"] > limits.output_tokens:
                raise ProtocolError("Token allowance exceeded")
            if len(calls) != 1 or calls[0].name not in {tool["name"] for tool in tools}:
                record["error"] = "Expected exactly one available function call"
                break
            call = calls[0]
            try:
                action = json.loads(call.arguments)
                canonical_json(action)
            except (ValueError, TypeError):
                record["error"] = "Function arguments were not a complete JSON object"
                break
            if not isinstance(action, dict):
                raise ProtocolError("Function arguments must be an object")
            if call.name == "commit_forecasts" and action.keys() == {"forecasts"}:
                record["metrics"] = {"forecast_mae": trial.commit(action["forecasts"])}
                feedback = {"forecast_committed": True}
            elif call.name == "practice" and action.keys() == {"indices", "patch"}:
                feedback = {"observations": trial.probe(action["indices"], action["patch"])}
            elif call.name == "submit_patch" and action.keys() == {"patch"}:
                metrics = trial.finish(action["patch"])
                record.update(status="completed", metrics=metrics, patch=action["patch"])
                break
            else:
                raise ProtocolError("Unknown function argument shape")
            messages.extend(item.model_dump(exclude_none=True) for item in final.output
                            if item.type in ("message", "function_call"))
            messages.append({"type": "function_call_output", "call_id": call.call_id,
                             "output": canonical_json(feedback)})
    except ProtocolError as error:
        record.update(status="protocol_violation", error=str(error))
    except Exception as error:
        record.update(status="infrastructure_failure", error=type(error).__name__, usage_unknown=True)
    record["forecast"] = trial.forecasts
    record["cost_usd"] = None if record.get("usage_unknown") else session_cost(model, usage)
    return record


def freeze(output, seed, suite_name="original"):
    if suite_name in ("hard", "hard-budget-check"):
        import repair_hard_cases as suite
    elif suite_name == "original":
        suite = repair_cases
    else:
        raise ValueError(f"Unknown suite {suite_name!r}")
    models = [m for m in json.loads((ROOT / "configs/gpt-comparison.json").read_text())["models"]
              if m["id"] in ("openai-luna", "openai-sol")]
    check = suite_name == "hard-budget-check"
    limits = HARD_CHECK_LIMITS if check else LIMITS
    arms = ("behavior", "weights_behavior") if check else tuple(ARMS)
    if check:
        models = [model for model in models if model["id"] == "openai-sol"]
    cases = suite.make_cases(seed, fixed_decoy=True) if check else suite.make_cases(seed)
    if len(models) != (1 if check else 2) or len(cases) < 3 or len({c.id for c in cases}) != len(cases):
        raise ValueError("Unexpected calibration roster or case IDs")
    paid = [(m, c, a) for m in models for c in cases for a in arms]
    random.Random(seed).shuffle(paid)
    cap = sum(maximum_session_cost(m, limits) for m, _, _ in paid)
    if cap > 5:
        raise ValueError("Calibration exceeds the $5 reservation ceiling")
    baseline = {c.id: scripted(c, suite) for c in cases}
    fixtures = []
    for case in cases:
        forecasts, patch = oracle(case)
        if not case.grade(forecasts, patch)["success"]:
            raise ValueError(f"Reference repair infeasible: {case.id}")
        accepted = [candidate for candidate in forecasts if case.grade(forecasts, candidate)["success"]]
        fixtures.append({
            "id": case.id, "mechanism": case.kind, "clean": case.kind == "clean",
            "candidate_count": len(forecasts), "passing_candidate_count": len(accepted),
            "input_digest": serialization_digest([case.public, case.weights]),
            "scorer_digest": serialization_digest([
                case._target.tolist(), case._retention_inputs.tolist(),
                case._retention_labels.tolist(), forecasts,
            ]),
        })
    sources = ["run_selective_repair.py", "repair_cases.py"]
    if suite_name != "original":
        sources.append("repair_hard_cases.py")
    manifest = {"version": "selective-repair-calibration-4", "suite": suite_name,
                "seed": seed, "maximum_cost_usd": cap,
                "arms": {a: dict(zip(("weights", "behavior", "activations"), flags, strict=True))
                         for a, flags in ARMS.items() if a in arms},
                "models": models, "system": SYSTEM,
                "limits": {"turns": limits.model_turns, "input_tokens": limits.input_tokens,
                           "output_tokens": limits.output_tokens, "pre_probes": PRE_PROBES,
                           "post_probes": POST_PROBES},
                "source_digests": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                   for name in sources},
                "cases": fixtures,
                "cells": [{"model": "scripted", "case": c.id, "arm": "behavior"} for c in cases] +
                         [{"model": m["snapshot"], "case": c.id, "arm": a} for m, c, a in paid]}
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Calibration differs from frozen manifest")
    _write(path, manifest)
    return manifest, paid, baseline, limits


async def run(output, seed, execute, suite_name="original"):
    manifest, paid, baseline, limits = freeze(output, seed, suite_name)
    digest = serialization_digest(manifest)

    def result_path(cell):
        return output / "runs" / serialization_digest([digest, cell]) / "result.json"

    for case_id, result in baseline.items():
        cell = {"model": "scripted", "case": case_id, "arm": "behavior"}
        _write(result_path(cell), {**cell, **result})
    print(f"{len(paid)} API sessions. Worst-case reservation ${manifest['maximum_cost_usd']:.2f}.", flush=True)
    if not execute:
        return
    load_dotenv(ROOT / ".env")
    ledger = BudgetLedger(_project_ledger())
    queue = asyncio.Semaphore(12 if suite_name == "hard" else 8)
    async with AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=0, timeout=90) as client:
        async def dispatch(model, case, arm):
            cell = {"model": model["snapshot"], "case": case.id, "arm": arm}
            path = result_path(cell)
            async with queue:
                run_id = path.parent.name
                if path.exists():
                    saved = json.loads(path.read_text())
                    if saved.get("reservation_id") != run_id:
                        raise ValueError("Saved result does not belong to this reservation")
                    if saved["cost_usd"] is not None:
                        ledger.settle(run_id, saved["cost_usd"])
                    return
                with ledger._connection() as connection:
                    overrun = connection.execute(
                        "SELECT 1 FROM reservations WHERE actual > reserved LIMIT 1"
                    ).fetchone()
                if overrun:
                    raise BudgetExceeded("A recorded charge exceeded its reservation")
                ledger.reserve("diagnostics", maximum_session_cost(model, limits), run_id)
                started = time.monotonic()
                record = {**cell, **await evaluate(client, model, case, arm, limits),
                          "wall_seconds": time.monotonic() - started, "reservation_id": run_id}
                _write(path, record)
                if record["cost_usd"] is not None:
                    ledger.settle(run_id, record["cost_usd"])
                print(model["snapshot"], case.id, arm, record["status"], record["cost_usd"], flush=True)

        await asyncio.gather(*(dispatch(m, c, a) for m, c, a in paid))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--suite", choices=("original", "hard", "hard-budget-check"), default="original")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args.output.resolve(), args.seed, args.execute, args.suite))
