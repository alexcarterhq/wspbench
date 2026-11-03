"""Bounded access regimes. The solver never receives the trusted scorer or paths."""
from __future__ import annotations

from copy import deepcopy
from functools import wraps
import time

import numpy as np

from .contracts import Condition, Task, WeightView
from .patches import materialize
from .serialization import (canonical_json, decode_weights, encode_weights, load_weights,
                            serialization_digest, validate_weights, weights_digest)


class AccessViolation(ValueError):
    pass


def _record_denial(method):
    @wraps(method)
    def checked(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except AccessViolation as exc:
            self._event("denied", operation=method.__name__, reason=str(exc))
            raise
    return checked


class AccessSession:
    """One trial's observable tool surface and metered commitments.

    B has no arbitrary code or execution endpoint. Auditing still cannot prove
    that a sequence of allowed arithmetic operations never emulates a network.
    """

    def __init__(self, task: Task, condition: str, weight_view: str = "full",
                 wrong_weights: dict | None = None):
        self.task = task
        self.condition = Condition(condition)
        self.weight_view = WeightView(weight_view)
        if self.condition.value not in task.conditions:
            raise AccessViolation("Condition is unavailable for this task")
        self.public_task = deepcopy(task.to_public_dict())
        self._starting = load_weights(task.weights_path)
        validate_weights(task.architecture, self._starting)
        checkpoints = {"target": self._starting}
        if task.family == "edit_effects":
            before = task.directory / "before.npz"
            if not before.exists():
                raise ValueError("Edit-effects task is missing its public before checkpoint")
            checkpoints["before"] = load_weights(before)
            checkpoints["after"] = self._starting
        self._payloads = {}
        self._views = {}
        self._redacted = {}
        boundary = set(task.architecture.get("boundary_tensors", []))
        for checkpoint, original in checkpoints.items():
            if self.weight_view is WeightView.WRONG:
                if wrong_weights is None:
                    raise ValueError("Matched-wrong view requires a separately matched checkpoint")
                if all(isinstance(v, np.ndarray) for v in wrong_weights.values()):
                    if len(checkpoints) > 1:
                        raise ValueError("Paired task requires wrong weights for every public checkpoint")
                    shown = wrong_weights
                else:
                    if checkpoint not in wrong_weights:
                        raise ValueError("Missing matched-wrong public checkpoint")
                    shown = wrong_weights[checkpoint]
                validate_weights(task.architecture, shown)
                shown = {k: v.copy() for k, v in shown.items()}
                redacted = []
            else:
                redacted = [name for name in original if self.weight_view is WeightView.MASKED or
                            (self.weight_view is WeightView.BOUNDARY and name not in boundary)]
                shown = {name: np.zeros_like(value) if name in redacted else value.copy()
                         for name, value in original.items()}
            payload = encode_weights(shown)
            self._payloads[checkpoint] = payload
            self._views[checkpoint] = decode_weights(payload)
            self._redacted[checkpoint] = redacted
        self._handles = {f"{checkpoint}:{name}": np.asarray(array, dtype=np.float64)
                         for checkpoint, weights in self._views.items() for name, array in weights.items()}
        specification=task.public.get("specification",{})
        if specification.get("format")=="truth_table":
            # Already-public numerical specifications are legitimate algebra
            # operands. Chunking keeps the same bounded array surface as weights.
            for name in ("inputs","outputs"):
                values=np.asarray(specification[name],dtype=np.float64)
                for start in range(0,len(values),128):
                    self._handles[f"spec:{name}:{start//128}"]=values[start:start+128].copy()
        self._next_handle = 0
        self.trace = []
        self.counters = {"analysis_calls": 0, "example_evaluations": 0,
                         "activation_scalars": 0, "episodes": 0, "steps": 0,
                         "generated_tokens": 0, "candidate_patches": 0}
        self._candidates = set()
        self._finished = False
        self._submitted = False
        self._cpu_start = time.process_time()
        self._wall_start = time.monotonic()
        self.current_submission = {} if task.skill == "read" else {"patch": []}
        self.snapshots = {0: deepcopy(self.current_submission)}

    @property
    def view_weights(self) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in self._views["target"].items()}

    def checkpoint_weights(self, checkpoint: str = "target") -> dict[str, np.ndarray]:
        if checkpoint not in self._views:
            raise AccessViolation("Checkpoint is not disclosed in this task")
        return {name: value.copy() for name, value in self._views[checkpoint].items()}

    @property
    def observation_count(self) -> int:
        return self.counters["episodes"] if self.task.family == "loop_repair" else self.counters["example_evaluations"]

    @property
    def logical_evaluations_per_index(self) -> int:
        return 2 if self.task.family == "causal_forecast" else 1

    @property
    def budget_levels(self) -> tuple[int, ...]:
        return (0, 4, 16) if self.task.family == "loop_repair" else (0, 32, 128)

    @property
    def remaining_observations(self) -> int:
        cap = self.task.limits.episodes if self.task.family == "loop_repair" else self.task.limits.example_evaluations
        return max(0, cap - self.observation_count)

    @property
    def next_boundary(self) -> int:
        return next((level for level in self.budget_levels if level > self.observation_count), self.budget_levels[-1])

    def public_payload(self) -> dict:
        checkpoints = deepcopy(self._payloads)
        # Generation/evaluation strata must not become diagnostic hints. The
        # trusted baseline runner retains public_task; the model sees this view.
        solver_task = {key: deepcopy(self.public_task[key]) for key in (
            "schema_version", "benchmark_version", "skill", "architecture", "instruction", "public", "limits")}
        return {"task": solver_task, "weights": checkpoints["target"],
                "checkpoints": checkpoints, "weight_view": self.weight_view.value,
                "public_specification_handles": [key for key in self._handles if key.startswith("spec:")],
                "redacted_tensors": deepcopy(self._redacted),
                "serialization_digest": serialization_digest(checkpoints)}

    def _live(self) -> None:
        if self._finished or self._submitted:
            raise AccessViolation("Trial is already finalized")
        if time.process_time() - self._cpu_start > self.task.limits.cpu_seconds:
            raise AccessViolation("Trial CPU allowance exhausted")
        if time.monotonic() - self._wall_start > self.task.limits.wall_seconds:
            raise AccessViolation("Trial active-time allowance exhausted")

    def _event(self, tool: str, **value) -> None:
        self.trace.append({"tool": tool, **deepcopy(value)})

    def _save(self, value: np.ndarray) -> dict:
        value = np.asarray(value, dtype=np.float64)
        if value.ndim > 2 or value.size > 4096 or any(d > 128 for d in value.shape):
            raise AccessViolation("Derived algebra array exceeds the bounded tool surface")
        if not np.isfinite(value).all():
            raise AccessViolation("Analysis produced nonfinite values")
        name = f"derived:{self._next_handle}"
        self._next_handle += 1
        self._handles[name] = value.copy()
        return {"handle": name, "shape": list(value.shape), "values": value.tolist()}

    @_record_denial
    def analyze(self, operation: str, operands: list[str] | None = None, **parameters) -> dict:
        self._live()
        if self.condition is Condition.UNAIDED:
            raise AccessViolation("Condition A has no external computation tools")
        if self.counters["analysis_calls"] >= self.task.limits.analysis_calls:
            raise AccessViolation("Analysis-call allowance exhausted")
        operations = {"read", "summary", "transpose", "matmul", "add", "subtract", "scale", "center",
                      "svd", "eigh", "norm", "mean", "std", "sum", "abs", "quantile", "slice"}
        if operation not in operations:
            raise AccessViolation("Unknown analysis operation; execution is not an analysis tool")
        operands = [] if operands is None else operands
        if not isinstance(operands, list) or not 1 <= len(operands) <= 2 or any(not isinstance(x, str) or x not in self._handles for x in operands):
            raise AccessViolation("Analysis operands must be disclosed weight/derived handles")
        self.counters["analysis_calls"] += 1
        arrays = [self._handles[name] for name in operands]
        allowed_parameters = {"factor", "q", "slices", "transpose_left", "transpose_right", "axis"}
        if set(parameters) - allowed_parameters:
            raise AccessViolation("Unknown analysis parameter")
        axis = parameters.get("axis")
        if axis is not None and (type(axis) is not int or not 0 <= axis < arrays[0].ndim):
            raise AccessViolation("Reduction axis is outside the disclosed array")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            if operation == "summary":
                a = arrays[0]
                result = {"shape": list(a.shape), "mean": float(a.mean()), "std": float(a.std()),
                          "norm": float(np.linalg.norm(a)), "positive_fraction": float(np.mean(a > 0))}
            elif operation == "svd":
                if arrays[0].ndim != 2:
                    raise AccessViolation("SVD requires a matrix")
                result = dict(zip(("u", "s", "vt"), (self._save(x) for x in np.linalg.svd(arrays[0], full_matrices=False))))
            elif operation == "eigh":
                a = arrays[0]
                if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.allclose(a, a.T):
                    raise AccessViolation("Eigendecomposition requires a symmetric matrix")
                result = dict(zip(("values", "vectors"), (self._save(x) for x in np.linalg.eigh(a))))
            else:
                a = arrays[0]
                if operation in {"matmul", "add", "subtract"}:
                    if len(arrays) != 2:
                        raise AccessViolation("This operation requires two operands")
                    b = arrays[1]
                    if operation == "matmul":
                        a = a.T if parameters.get("transpose_left") else a
                        b = b.T if parameters.get("transpose_right") else b
                        if a.ndim > 2 or b.ndim > 2:
                            raise AccessViolation("Only vector/matrix products are supported")
                        if a.ndim == b.ndim == 2 and a.shape[0] * b.shape[1] > 4096:
                            raise AccessViolation("Product exceeds analysis capacity")
                        value = a @ b
                    else:
                        if a.shape != b.shape:
                            raise AccessViolation("Elementwise operands must have the same shape")
                        value = a + b if operation == "add" else a - b
                elif operation == "read": value = a
                elif operation == "transpose": value = a.T
                elif operation == "scale":
                    factor = parameters.get("factor")
                    if type(factor) not in (int, float) or not np.isfinite(factor) or abs(factor) > 1e6:
                        raise AccessViolation("Scale factor must be a bounded finite scalar")
                    value = a * factor
                elif operation == "norm": value = np.asarray(np.linalg.norm(a))
                elif operation == "mean": value = np.asarray(a.mean(axis=axis))
                elif operation == "std": value = np.asarray(a.std(axis=axis))
                elif operation == "sum": value = np.asarray(a.sum(axis=axis))
                elif operation == "center": value = a - a.mean(axis=0, keepdims=True)
                elif operation == "abs": value = np.abs(a)
                elif operation == "quantile":
                    q = parameters.get("q", .5)
                    if type(q) not in (int, float) or not 0 <= q <= 1:
                        raise AccessViolation("Quantile must lie in [0,1]")
                    value = np.asarray(np.quantile(a, q))
                elif operation == "slice":
                    selections = parameters.get("slices")
                    if not isinstance(selections, list) or len(selections) != a.ndim:
                        raise AccessViolation("A bounded [start,stop] slice is required per axis")
                    indices = []
                    for pair, size in zip(selections, a.shape):
                        if not isinstance(pair, list) or len(pair) != 2 or any(type(i) is not int for i in pair) or not 0 <= pair[0] < pair[1] <= size:
                            raise AccessViolation("Slice outside disclosed array")
                        indices.append(slice(*pair))
                    value = a[tuple(indices)]
                result = self._save(value)
        self._event("analysis", operation=operation, operands=operands, parameters=parameters, result=result)
        self._live()
        return result

    @_record_denial
    def commit(self, submission: dict) -> dict:
        self._live()
        if not isinstance(submission, dict):
            raise AccessViolation("Submission must be a JSON object")
        canonical_json(submission)
        if self.task.skill != "read":
            materialize(self.task.architecture, self._starting, submission)
        self.current_submission = deepcopy(submission)
        if self.observation_count in self.budget_levels:
            self.snapshots[self.observation_count] = deepcopy(submission)
        self._event("commit", submission=submission, observation_count=self.observation_count)
        return {"committed": True, "observation_count": self.observation_count}

    @_record_denial
    def probe(self, indices: list[int], patch=None, observation: str = "behavior",
              checkpoint: str = "target", intervention_id: str | None = None) -> dict:
        self._live()
        if self.condition is not Condition.INSTRUMENTED:
            raise AccessViolation("Only condition C may execute the target or inspect activations")
        if observation not in {"behavior", "activations"}:
            raise AccessViolation("Unknown observation type")
        if checkpoint not in self._views:
            raise AccessViolation("Checkpoint is not disclosed for this task")
        if not isinstance(indices, list) or not indices or len(indices) > 128 or any(type(i) is not int or i < 0 for i in indices):
            raise AccessViolation("Probe requires a bounded list of practice indices")
        if len(indices) != len(set(indices)) or max(indices) >= self.task.public["practice_size"]:
            raise AccessViolation("Probe index is duplicated or outside the frozen practice bank")
        requested_cost = len(indices) * self.logical_evaluations_per_index
        if requested_cost > self.remaining_observations or self.observation_count + requested_cost > self.next_boundary:
            raise AccessViolation(f"Split this request at observation boundary {self.next_boundary}")
        if self.task.family == "causal_forecast":
            if patch not in (None, [], {"patch": []}):
                raise AccessViolation("Forecast probes cannot execute arbitrary edited checkpoints")
            allowed = self.task.public.get("practice_interventions", [])
            allowed_ids = [x["id"] if isinstance(x, dict) else x for x in allowed]
            if intervention_id not in allowed_ids:
                raise AccessViolation("Only practice intervention identities may be queried")
        elif intervention_id is not None:
            raise AccessViolation("This task has no intervention catalog")
        if self.task.skill != "read":
            submitted_patch = {"patch": []} if patch is None else ({"patch": patch} if isinstance(patch, list) else patch)
            candidate = materialize(self.task.architecture, self._starting, submitted_patch)
            digest = weights_digest(candidate)
            if digest != weights_digest(self._starting) and digest not in self._candidates:
                if len(self._candidates) >= self.task.limits.candidate_patches:
                    raise AccessViolation("Candidate-patch allowance exhausted")
                self._candidates.add(digest)
            self.commit(submitted_patch)
        else:
            if patch not in (None, [], {"patch": []}):
                raise AccessViolation("Prediction probes use only public checkpoint identities")
            candidate = self._starting
        from .scoring import probe as trusted_probe
        # The core computes these bounds without running the target; fail closed
        # if a future architecture cannot provide a safe activation upper bound.
        activation_per_example = int(self.task.public.get("activation_scalars_per_example", 4096))
        if observation == "activations" and self.counters["activation_scalars"] + len(indices) * activation_per_example > self.task.limits.activation_scalars:
            raise AccessViolation("Activation-scalar allowance would be exceeded")
        if self.task.family == "loop_repair" and self.counters["episodes"] + len(indices) > self.task.limits.episodes:
            raise AccessViolation("Episode allowance exhausted")
        result = trusted_probe(self.task, candidate, indices, observation=observation,
                               checkpoint=checkpoint, intervention_id=intervention_id)
        accounting = result.get("accounting")
        if not isinstance(accounting, dict):
            raise RuntimeError("Trusted probe omitted accounting")
        expected_examples = 0 if self.task.family == "loop_repair" else requested_cost
        if accounting.get("example_evaluations") != expected_examples:
            raise RuntimeError("Trusted probe example accounting differs from request")
        expected_episodes = len(indices) if self.task.family == "loop_repair" else 0
        if accounting.get("episodes", 0) != expected_episodes or accounting.get("steps", 0) > expected_episodes * self.task.limits.episode_steps:
            raise RuntimeError("Trusted episode accounting differs from bounded request")
        for key in ("example_evaluations", "activation_scalars", "episodes", "steps", "generated_tokens"):
            value = accounting.get(key, 0)
            if type(value) is not int or value < 0:
                raise RuntimeError("Invalid trusted-probe accounting")
            self.counters[key] += value
        if self.counters["activation_scalars"] > self.task.limits.activation_scalars:
            raise RuntimeError("Trusted activation bound was incorrect")
        self.counters["candidate_patches"] = len(self._candidates)
        if observation == "activations":
            handles = {}
            for name, values in result.get("activations", {}).items():
                if self.task.family == "loop_repair":
                    # Each episode is a separate bounded time-by-feature matrix;
                    # ragged episode lengths never require padding or fake states.
                    handles[name] = []
                    for episode in values:
                        saved = self._save(np.asarray(episode))
                        handles[name].append({k: saved[k] for k in ("handle", "shape")})
                else:
                    saved = self._save(np.asarray(values))
                    handles[name] = {k: saved[k] for k in ("handle", "shape")}
            result["activation_handles"] = handles
        if self.observation_count in self.budget_levels:
            self.snapshots[self.observation_count] = deepcopy(self.current_submission)
        self._event("probe", indices=indices, observation=observation, checkpoint=checkpoint,
                    intervention_id=intervention_id, accounting=accounting, result=result)
        self._live()
        return deepcopy(result)

    def submit(self, submission: dict) -> dict:
        self.commit(submission)
        self._submitted = True
        self._event("submit", observation_count=self.observation_count)
        return {"submitted": True, "hidden_feedback": "not returned"}

    def finish(self) -> dict:
        from .audit import audit_trace
        for level in self.budget_levels:
            if level not in self.snapshots:
                self.snapshots[level] = deepcopy(self.current_submission)
        self._finished = True
        return {"condition": self.condition.value, "weight_view": self.weight_view.value,
                "serialization_digest": self.public_payload()["serialization_digest"],
                "question_digest": serialization_digest(self.public_payload()["task"]),
                "canonical_prompt_digest": serialization_digest(self.public_payload()),
                "submission": deepcopy(self.current_submission), "submitted": self._submitted,
                "counters": deepcopy(self.counters),
                "snapshots": {str(k): v for k, v in self.snapshots.items()},
                "snapshot_actual_observations": {str(k): min(k, self.observation_count) for k in self.snapshots},
                "snapshot_unit": "episodes" if self.task.family == "loop_repair" else "example_evaluations",
                "actual_observations": self.observation_count,
                "cpu_seconds": time.process_time() - self._cpu_start,
                "wall_seconds": time.monotonic() - self._wall_start,
                "trace": deepcopy(self.trace),
                "audit": audit_trace(self.trace, self.condition.value, self.task.limits, self.counters)}
