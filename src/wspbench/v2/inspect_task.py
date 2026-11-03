"""Inspect adapter with no general code, filesystem or target-execution tool in A/B."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import time

from ..budget import add_usage, empty_usage
from ..provider_options import provider_generate_options
from .access import AccessSession, AccessViolation
from .serialization import canonical_json


SYSTEM = """You are solving a WSPBench v0.2 small-network challenge.
The task question, numeric checkpoint precision, and initial view are identical across
access conditions. Respond with the JSON submission specified by the task. Only the
declared tools may perform external computation. Never request filesystem, network,
general Python, target gradients, fitting, or unmetered target execution. Numeric
patches apply to the true checkpoint; undisclosed or matched-wrong values are not
implicitly copied into it. Hidden final scoring and answer keys are unavailable.
An answer's explanation is not the primary score. Every scored trace needs review.
"""
CONDITIONS = {
    "A": "Condition A: one response, with no external computation tools. Return only your JSON submission.",
    "B": "Condition B: bounded weight algebra only. No target behavior or activations are available. Use submit to finish, or return a JSON submission.",
    "C": "Condition C: bounded weight algebra plus metered practice observations. Every original/edited case counts separately. Commit your current answer at zero observations and after feedback; requests must stop at 32 and 128 example-evaluations (loop tasks: 4 and 16 episodes). Early stopping carries your last commitment forward with its actual cost. Only practice indices and intervention IDs may be queried. Use submit to finish.",
}


def parse_submission(text: str) -> dict:
    text = text.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Final answer must be one JSON object")
    canonical_json(value)
    return value


def build_inspect_task(task, run_dir: Path, model_config: dict, record: dict,
                       *, condition: str, weight_view: str = "full", wrong_weights=None):
    from inspect_ai import Task as InspectTask
    from inspect_ai.dataset import Sample
    from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig
    from inspect_ai.scorer import Score as InspectScore, mean, scorer
    from inspect_ai.solver import solver
    from inspect_ai.tool import ToolError, tool
    from .scoring import score

    session = AccessSession(task, condition, weight_view, wrong_weights)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = session.public_payload()
    prompt = {"system": SYSTEM, "access": CONDITIONS[condition], "payload": payload}
    (run_dir / "prompt.json").write_text(canonical_json(prompt) + "\n")
    record["prompt_digest"] = __import__("hashlib").sha256(canonical_json(prompt).encode()).hexdigest()
    record["usage"] = empty_usage()
    record["question_digest"] = __import__("hashlib").sha256(canonical_json(payload["task"]).encode()).hexdigest()
    record["serialization_digest"] = payload["serialization_digest"]
    record["condition"] = condition
    record["weight_view"] = weight_view

    def invoke(method, **kwargs):
        try:
            return canonical_json(method(**kwargs))
        except (ValueError, TypeError, FloatingPointError, KeyError) as exc:
            # AccessSession records denied ACL requests itself. Malformed numeric
            # artifacts and tool JSON are also explicit zero-credit failures.
            if not isinstance(exc, AccessViolation):
                session._event("denied", operation=method.__name__, reason=str(exc))
            raise ToolError(str(exc)) from exc

    def decoded(text, default):
        try:
            return json.loads(text) if text else default
        except (TypeError, ValueError) as exc:
            session._event("denied", operation="tool_json", reason="Malformed JSON argument")
            raise ToolError("Malformed JSON argument") from exc

    @tool(parallel=False)
    def analysis():
        async def execute(operation: str, operands: list[str], parameters_json: str = "{}") -> str:
            """Apply bounded pure algebra to disclosed weights or derived handles.

            Args:
                operation: read, summary, transpose, matmul, add, subtract, scale, center, svd, eigh, norm, mean, std, sum, abs, quantile, or slice.
                operands: One or two disclosed weight/derived handles, initially checkpoint:tensor (for example target:w0). In C, metered activation responses also provide handles. No literal target inputs.
                parameters_json: JSON parameters: factor for scale; axis for mean/std/sum; q for quantile; slices=[[start,stop],...] for slice; transpose_left/transpose_right for matmul. center subtracts column means.
            """
            parameters = decoded(parameters_json, {})
            if not isinstance(parameters, dict):
                raise ToolError("Analysis parameters must be a JSON object")
            return invoke(session.analyze, operation=operation, operands=operands, **parameters)
        return execute

    @tool(parallel=False)
    def practice():
        async def execute(indices: list[int], patch_json: str = "", observation: str = "behavior",
                          checkpoint: str = "target", intervention_id: str = "") -> str:
            """Observe only frozen practice cases; every requested logical evaluation consumes budget.

            Args:
                indices: Distinct practice-bank indices; split requests at the next observation boundary.
                patch_json: Numeric patch JSON for repair/write; empty uses the untouched true checkpoint. Forecast/read tasks forbid arbitrary patches.
                observation: behavior or activations. Returned activation scalars are separately capped.
                checkpoint: Disclosed checkpoint identity: target, before, or after.
                intervention_id: For causal forecasts, a practice catalog ID. Final forecast IDs cannot be queried.
            """
            return invoke(session.probe, indices=indices, patch=decoded(patch_json, None),
                          observation=observation, checkpoint=checkpoint,
                          intervention_id=intervention_id or None)
        return execute

    @tool(parallel=False)
    def commit():
        async def execute(submission_json: str) -> str:
            """Commit your current solution before requesting more observations.

            Args:
                submission_json: Complete task-specific JSON answer or numeric patch object.
            """
            return invoke(session.commit, submission=decoded(submission_json, {}))
        return execute

    @tool(parallel=False)
    def submit():
        async def execute(submission_json: str) -> str:
            """Finalize one JSON submission without receiving hidden scoring feedback.

            Args:
                submission_json: Complete task-specific JSON answer or numeric patch object.
            """
            return invoke(session.submit, submission=decoded(submission_json, {}))
        return execute

    @solver
    def shared_solver():
        async def solve(state, generate):
            started = time.monotonic()
            state.messages = [ChatMessageSystem(content=SYSTEM + "\n" + CONDITIONS[condition]),
                              ChatMessageUser(content=canonical_json(payload))]
            state.tools = ([] if condition == "A" else [analysis(), commit(), submit()] +
                           ([practice()] if condition == "C" else []))
            try:
                for turn in range(1 if condition == "A" else task.limits.model_turns):
                    if session._submitted or any(e["tool"] == "denied" for e in session.trace):
                        break
                    session._live()
                    # UTF-8 byte count plus framing is a conservative input-token
                    # bound; no provider token-count API call is needed.
                    request_bound = sum(len(m.model_dump_json().encode()) for m in state.messages) + 8192
                    if record["usage"]["input_tokens_total"] + request_bound > task.limits.input_tokens:
                        record["stop_reason"] = "input_token_budget"
                        break
                    remaining = task.limits.output_tokens - record["usage"]["output_tokens"]
                    if remaining <= 0:
                        record["stop_reason"] = "output_token_budget"
                        break
                    state = await asyncio.wait_for(generate(state, tool_calls="none" if condition == "A" else "single",
                        max_tokens=remaining, parallel_tool_calls=False, internal_tools=False,
                        **provider_generate_options(model_config)),
                        max(1., task.limits.wall_seconds - (time.monotonic() - started)))
                    if state.output.usage is None:
                        record["usage_unknown"] = True
                        raise RuntimeError("Provider omitted reconciliable token usage")
                    record["usage"] = add_usage(record["usage"], state.output.usage)
                    record["turns"] = turn + 1
                    session._event("model_turn", turn=turn + 1, usage=state.output.usage.model_dump(),
                                   output=state.output.model_dump(mode="json"))
                    if record["usage"]["input_tokens_total"] > task.limits.input_tokens or record["usage"]["output_tokens"] > task.limits.output_tokens:
                        session._event("denied", operation="token_budget", reason="Provider usage exceeded a session token cap")
                        break
                    if condition == "A" and state.output.message.tool_calls:
                        session._event("denied", operation="tool_call", reason="Condition A cannot call tools")
                        break
                    if not state.output.message.tool_calls:
                        try:
                            session.submit(parse_submission(state.output.completion))
                        except (TypeError, ValueError, KeyError) as exc:
                            record["submission_error"] = str(exc)[:1000]
                        break
                record["status"] = "completed" if session._submitted else "invalid_submission"
            except asyncio.TimeoutError:
                record["status"] = "timeout"
                record["usage_unknown"] = True
            except AccessViolation as exc:
                session._event("denied", operation="session_budget", reason=str(exc))
                record["status"] = "protocol_violation"
            except asyncio.CancelledError:
                record.update(status="timeout", usage_unknown=True)
                raise
            except Exception as exc:
                record.update(status="infrastructure_failure", usage_unknown=True,
                              error=f"{type(exc).__name__}: {str(exc)[:1000]}")
            finally:
                finished = session.finish()
                record.update({k: v for k, v in finished.items() if k != "trace"})
                if record["audit"]["status"] == "rejected":
                    record["status"] = "protocol_violation"
                record["observation_usage"] = finished["counters"]
                record["usage"].update(finished["counters"])
                (run_dir / "trace.json").write_text(json.dumps(finished["trace"], indent=2, allow_nan=False) + "\n")
                (run_dir / "submission.json").write_text(canonical_json(finished["submission"]) + "\n")
                state.metadata["wspbench_record"] = record
            state.completed = True
            return state
        return solve

    @scorer(metrics=[mean()])
    def hidden_score():
        async def evaluate(state, target):
            record["success"] = False
            record["metrics"] = {}
            if record.get("status") == "completed":
                result = score(task, record["submission"])
                record.update(success=bool(result.success), metrics=result.metrics,
                              score_status=result.status, score_reason=result.reason)
                if result.status != "scored":
                    record["status"] = result.status
            if condition == "C":
                record["budget_curve"] = [{"budget": int(level), "actual_observations":
                    record["snapshot_actual_observations"][level], **score(task, answer).to_dict()}
                    for level, answer in record["snapshots"].items()]
            return InspectScore(value=int(record["success"]), metadata={"audit": record["audit"]["status"]})
        return evaluate

    config = GenerateConfig(**{**model_config.get("generate_config", {}),**provider_generate_options(model_config),
        "max_tokens": task.limits.output_tokens, "parallel_tool_calls": False})
    return InspectTask(name=f"wspbench_v2_{task.family}_{condition}", version="0.2.0",
        dataset=[Sample(id=task.id, input=task.instruction)], solver=shared_solver(), scorer=hidden_score(),
        config=config, turn_limit=task.limits.model_turns, time_limit=task.limits.wall_seconds,
        metadata={"task_id": task.id, "parent_problem_id": task.parent_problem_id,
                  "condition": condition, "weight_view": weight_view, "suite_tier": "challenge_candidate"})
