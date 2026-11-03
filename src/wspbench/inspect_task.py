"""Inspect task and shared solver. Hidden scoring remains outside the workspace."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from .audit import audit_trace, screen_code
from .budget import empty_usage, add_usage
from .provider_options import provider_generate_options
from .sandbox import DockerWorkspace, WorkspaceFailure, IMAGE

SYSTEM = """You are solving WSPBench, a small-network weight-space programming task.
Use only the supplied tools and fixed architecture. Inspect and edit the dictionary `weights`
in the persistent Python workspace; NumPy (`np`) and math are available. No network, filesystem,
external tools, gradients, fitting, surrogate training, finite differences, or scripted parameter
search are permitted. Analytic tensor construction and deliberate hypothesis/edit cycles are
permitted. Loops for analytic construction are allowed but audited. Do not implement or execute
the target network in Python, including manually composing its layers. Only the practice tool may
execute it, and only on Repair/Write tasks. Read must be answered from weights alone.
Call submit(answer=...) for Read, or submit() to submit the current numeric weights for Repair/Write.
Your explanations are recorded but are not the primary score. Keep submitted arrays finite and
their shapes/keys unchanged. The task's hidden tests and reference weights are never available.
"""


def build_inspect_task(task, run_dir: Path, model_config: dict, record: dict):
    from inspect_ai import Task
    from inspect_ai.dataset import Sample
    from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig
    from inspect_ai.scorer import Score as InspectScore, scorer, mean
    from inspect_ai.solver import solver
    from inspect_ai.tool import tool, ToolError
    from .networks import load_weights, save_weights, validate_weights
    from .scoring import evaluate_practice, score

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    trace: list[dict] = []
    record["usage"] = empty_usage()
    record["limits"] = vars(task.limits)
    workspace = DockerWorkspace(task.limits, image=record.get("worker_image_digest", IMAGE))
    used_examples = 0
    used_episodes = 0
    submitted = None
    protocol_violation = False

    def event(value):
        trace.append(value)
        with (run_dir / "events.jsonl").open("a") as f:
            f.write(json.dumps(value, allow_nan=False) + "\n")

    @tool(parallel=False)
    def python():
        async def execute(code: str) -> str:
            """Analyze or analytically edit numeric weights in a persistent isolated Python workspace.

            Args:
                code: Python source using weights, np, and math. Target execution and fitting are forbidden.
            """
            nonlocal protocol_violation
            findings = screen_code(code)
            event({"tool": "python", "code": code, "screening": findings})
            if any(not x.startswith("review:") for x in findings):
                protocol_violation = True
                raise ToolError("Prohibited operation; this attempt receives zero protocol credit")
            try:
                response = await workspace.execute(code)
            except WorkspaceFailure as exc:
                record["workspace_failed"] = True
                response = {"error": str(exc)}
            except Exception as exc:
                response = {"error": str(exc)}
            event({"tool": "python_result", **response})
            return json.dumps(response)
        return execute

    @tool(parallel=False)
    def practice():
        async def execute(indices: list[int]) -> str:
            """Evaluate current weights on public practice cases; never on hidden scoring cases.

            Args:
                indices: Practice case indices. Every requested case consumes allowance, including repeats.
            """
            nonlocal used_examples, used_episodes
            if task.skill == "read":
                raise ToolError("Read tasks prohibit target execution")
            if not indices or len(indices) > 16 or any(type(i) is not int or i < 0 for i in indices):
                raise ToolError("Request between 1 and 16 nonnegative integer indices")
            if task.family == "write_controller":
                if used_episodes + len(indices) > task.limits.practice_episodes:
                    raise ToolError("Practice episode allowance exhausted")
                used_episodes += len(indices)
            else:
                if used_examples + len(indices) > task.limits.practice_examples:
                    raise ToolError("Practice example allowance exhausted")
                used_examples += len(indices)
            try:
                weights = await workspace.weights()
                validate_weights(task.architecture, weights)
                result = evaluate_practice(task, weights, indices)
            except Exception as exc:
                result = {"error": str(exc)}
            event({"tool": "practice", "indices": indices, "examples_used": used_examples,
                   "episodes_used": used_episodes, "observation": result})
            return json.dumps(result, allow_nan=False)
        return execute

    @tool(parallel=False)
    def submit():
        async def execute(answer: str = "") -> str:
            """Finish the task: submit a behavioral label for Read, or current weights for Repair/Write.

            Args:
                answer: Exact option label for Read; leave empty for Repair/Write.
            """
            nonlocal submitted
            if submitted is not None:
                raise ToolError("Submission already finalized")
            submitted = answer if task.skill == "read" else await workspace.weights()
            if task.skill != "read":
                validate_weights(task.architecture, submitted)
                save_weights(run_dir / "submission.npz", submitted)
            event({"tool": "submit", "answer": answer if task.skill == "read" else None})
            return "Submission accepted for hidden evaluation; no hidden feedback is returned."
        return execute

    @solver
    def shared_solver():
        async def solve(state, generate):
            nonlocal protocol_violation
            started = time.monotonic()
            state.messages = [ChatMessageSystem(content=SYSTEM), ChatMessageUser(content=json.dumps(task.to_public_dict(), allow_nan=False))]
            state.tools = [python(), submit()] + ([] if task.skill == "read" else [practice()])
            try:
                await workspace.start(load_weights(task.weights_path))
                for turn in range(task.limits.max_turns):
                    if submitted is not None or protocol_violation or record.get("workspace_failed"):
                        break
                    # UTF-8 byte count is a conservative bound for textual input tokenization,
                    # plus room for provider/tool framing. No token-count API calls are billed here.
                    request_bound = sum(len(m.model_dump_json().encode()) for m in state.messages) + 8192
                    if record["usage"]["input_tokens_total"] + request_bound > task.limits.input_tokens:
                        record["stop_reason"] = "input_token_budget"
                        break
                    remaining = task.limits.output_tokens - record["usage"]["output_tokens"]
                    if remaining <= 0:
                        record["stop_reason"] = "output_token_budget"
                        break
                    state = await asyncio.wait_for(generate(state, tool_calls="single", max_tokens=remaining,
                                   parallel_tool_calls=False, internal_tools=False,**provider_generate_options(model_config)),
                                   max(1, task.limits.wall_seconds - (time.monotonic() - started)))
                    usage = state.output.usage
                    if usage is None:
                        record["usage_unknown"] = True
                        raise RuntimeError("Provider omitted usage accounting; paid execution stopped")
                    event({"tool": "model_turn", "turn": turn + 1, "usage": usage.model_dump(),
                           "output": state.output.model_dump(mode="json")})
                    record["usage"] = add_usage(record["usage"], usage)
                    record["turns"] = turn + 1
                    if not state.output.message.tool_calls:
                        break
                record["status"] = ("infrastructure_failure" if record.get("workspace_failed") else
                                    "protocol_violation" if protocol_violation else
                                    "completed" if submitted is not None else "invalid_submission")
            except asyncio.TimeoutError:
                record["status"] = "timeout"
                record["usage_unknown"] = True
            except asyncio.CancelledError:
                record["status"] = "timeout"
                record["usage_unknown"] = True
                record["error"] = "Inspect cancelled an in-flight session; billing requires reconciliation"
                raise
            except Exception as exc:
                record["status"] = "infrastructure_failure"
                record["error"] = f"{type(exc).__name__}: {str(exc)[:2000]}"
                record["usage_unknown"] = True
            finally:
                record["wall_seconds"] = time.monotonic() - started
                record["cpu_seconds"] = workspace.cpu_seconds
                record["python_calls"] = workspace.calls
                record["feedback"] = {"examples": used_examples, "episodes": used_episodes,
                                      "max_episode_steps": task.limits.episode_steps}
                await asyncio.shield(workspace.close())
                record["audit"] = audit_trace(trace)
                if protocol_violation:
                    record["audit"]["status"] = "rejected"
                (run_dir / "trace.json").write_text(json.dumps(trace, indent=2, allow_nan=False) + "\n")
                state.metadata["wspbench_record"] = record
            state.completed = True
            return state
        return solve

    @scorer(metrics=[mean()])
    def hidden_score():
        async def evaluate(state, target):
            if record.get("status") == "completed" and submitted is not None:
                result = score(task, submitted)
                record["success"] = bool(result.success)
                record["metrics"] = result.metrics
                if result.status != "scored":
                    record["score_status"] = result.status
                    record["status"] = result.status
                if result.reason:
                    record["score_reason"] = result.reason
            else:
                record["success"] = False
                record["metrics"] = {}
            return InspectScore(value=int(record["success"]), metadata={"audit": record["audit"]["status"], "metrics": record["metrics"]})
        return evaluate

    config = GenerateConfig(**{**model_config.get("generate_config", {}), **provider_generate_options(model_config),
                                "max_tokens": task.limits.output_tokens, "parallel_tool_calls": False,
                                })
    return Task(name=f"wspbench_{task.skill}", version="0.1", dataset=[Sample(id=task.id, input=task.instruction)],
                solver=shared_solver(), scorer=hidden_score(), config=config,
                turn_limit=task.limits.max_turns, time_limit=task.limits.wall_seconds,
                metadata={"task_id": task.id, "parent_id": task.parent_id, "family": task.family})
