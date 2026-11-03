"""Retry only API errors from a frozen GPT selective repair run."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time

from dotenv import load_dotenv
from openai import AsyncOpenAI

from report_selective_repair import cell_key, load_results
from run_selective_repair import ROOT, evaluate, freeze
from wspbench.budget import BudgetExceeded, BudgetLedger, maximum_session_cost
from wspbench.v2.runner import _project_ledger, _write
from wspbench.v2.serialization import serialization_digest

CONCURRENCY = 4


def bind(path, data):
    if path.exists():
        if json.loads(path.read_text()) != data:
            raise ValueError(f"Frozen file changed: {path.name}")
    else:
        _write(path, data)


async def retry(original, output, execute):
    if original == output:
        raise ValueError("Retry results require a separate directory")
    manifest, expected, original_rows = load_results(original)
    if manifest["suite"] != "gpt-full" or set(original_rows) != expected:
        raise ValueError("Expected a complete original GPT comparison")
    with TemporaryDirectory(prefix="wspbench-retry-") as scratch:
        reconstructed, paid, _, limits = freeze(Path(scratch), manifest["seed"], "gpt-full")
    # The public runner uses a different ledger guard. Check generated tasks, pricing,
    # prompt, schedule and generator code against the original frozen manifest.
    for field, value in reconstructed.items():
        if field == "source_digests":
            for name in ("repair_cases.py", "repair_hard_cases.py"):
                if value[name] != manifest[field][name]:
                    raise ValueError(f"Generator changed: {name}")
        elif manifest[field] != value:
            raise ValueError(f"Original manifest differs in {field}")

    targets = []
    for model, case, arm in paid:
        cell = {"model": model["snapshot"], "case": case.id, "arm": arm}
        prior = original_rows[cell_key(cell)]
        if prior["status"] == "infrastructure_failure" and prior["error"] in (
            "APIError", "APITimeoutError",
        ):
            targets.append((model, case, arm, cell, prior))

    if not targets:
        raise ValueError("No retryable API errors in the original run")
    ceiling = sum(maximum_session_cost(model, limits) for model, *_ in targets)
    digest = serialization_digest(manifest)
    plan = {
        "concurrency": CONCURRENCY,
        "original_manifest_digest": digest,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "maximum_cost_usd": ceiling,
        "cells": [cell for _, _, _, cell, _ in targets],
    }
    bind(output / "manifest.json", manifest)
    bind(output / "retry-plan.json", plan)
    print(f"{len(targets)} retries at concurrency {CONCURRENCY}. "
          f"Worst-case additional reservation ${ceiling:.2f}.", flush=True)
    if not execute:
        return

    load_dotenv(ROOT / ".env")
    ledger = BudgetLedger(_project_ledger())
    queue = asyncio.Semaphore(CONCURRENCY)
    async with AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=0, timeout=90) as client:
        async def dispatch(model, case, arm, cell, prior):
            cell_hash = serialization_digest([digest, cell])
            path = output / "runs" / cell_hash / "result.json"
            run_id = serialization_digest(["api-error-retry-1", cell_hash])
            if prior["reservation_id"] != cell_hash:
                raise ValueError("Original record does not belong to its cell")
            async with queue:
                if path.exists():
                    saved = json.loads(path.read_text())
                    if saved["reservation_id"] != run_id or cell_key(saved) != cell_key(cell):
                        raise ValueError("Saved retry does not match its reservation")
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
                result = await evaluate(client, model, case, arm, limits)
                record = {**cell, **result, "wall_seconds": time.monotonic() - started,
                          "reservation_id": run_id, "retry_of": cell_hash}
                _write(path, record)
                if record["cost_usd"] is not None:
                    ledger.settle(run_id, record["cost_usd"])
                print(model["snapshot"], case.id, arm, record["status"],
                      record.get("error", ""), record["cost_usd"], flush=True)

        await asyncio.gather(*(dispatch(*target) for target in targets))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    asyncio.run(retry(args.original.resolve(), args.output.resolve(), args.execute))
