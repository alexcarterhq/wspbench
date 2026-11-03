"""Trusted, objective evaluation. This module is never mounted into solver sandboxes."""
from __future__ import annotations
from pathlib import Path
from collections.abc import Mapping
import hashlib
import json
import zipfile
import numpy as np
from .domain import Task, Score
from .networks import forward, load_weights, validate_weights


def _private(task: Task) -> dict:
    return json.loads(task.private_path.read_text())


def reference_submission(task: Task):
    if task.skill == "read":
        return _private(task)["answer"]
    return load_weights(task.directory / "reference.npz")


def baseline_submission(task: Task, name: str = "noop"):
    if name not in {"noop", "random", "no_weight"}:
        raise ValueError("Baseline must be noop, random, or no_weight")
    seed = int.from_bytes(hashlib.sha256(task.id.encode()).digest()[:4], "little")
    rng = np.random.default_rng(seed)
    if task.skill == "read":
        return "unknown" if name == "noop" else str(rng.choice(task.public["options"]))
    weights = load_weights(task.weights_path)
    if name == "no_weight":
        if task.skill == "repair":
            from .tasks import _visual_reference, VISUAL_LABELS
            return _visual_reference(task.architecture, VISUAL_LABELS.index(task.public["target_motif"]))
        if task.skill == "write":
            # Public specifications determine these construction tasks completely.
            # Reference artifacts contain exactly the constructive solution.
            return reference_submission(task)
    if name == "random":
        return {k: rng.normal(0, .3, v.shape).astype(np.float32) for k, v in weights.items()}
    return weights


def _weights(task: Task, submission):
    if isinstance(submission, (str, Path)):
        submission = load_weights(submission)
    if not isinstance(submission, Mapping):
        raise ValueError("Submission must be a tensor mapping or .npz file")
    weights = {k: np.asarray(v) for k, v in submission.items()}
    validate_weights(task.architecture, weights)
    return weights


def controller_rollouts(task: Task, weights: dict, initial: np.ndarray) -> dict[str, np.ndarray]:
    d = task.public["dynamics"]
    x, v, goal = np.asarray(initial, dtype=np.float64).T.copy()
    for _ in range(d["steps"]):
        state = np.column_stack([x, v, goal])
        force = np.clip(forward(task.architecture, weights, state)[:, 0], -1, 1)
        v += d["dt"] * (force / d["mass"] - d["drag"] * v)
        x += d["dt"] * v
        if not np.isfinite(x).all() or not np.isfinite(v).all():
            raise ValueError("Controller produced nonfinite state")
    position_error = np.abs(x - goal)
    velocity_error = np.abs(v)
    success = (position_error <= d["position_tolerance"]) & (velocity_error <= d["velocity_tolerance"])
    return {"success": success, "position_error": position_error, "velocity_error": velocity_error,
            "final_position": x, "final_velocity": v}


def score(task: Task, submission) -> Score:
    try:
        if task.skill == "read":
            if isinstance(submission, Mapping):
                submission = submission.get("answer")
            if not isinstance(submission, str) or submission not in task.public["options"]:
                return Score(False, {"accuracy": 0.0}, "invalid_submission", "Answer is not a listed option")
            correct = submission == _private(task)["answer"]
            return Score(correct, {"accuracy": float(correct)})
        weights = _weights(task, submission)
        original = load_weights(task.weights_path)
        changed = sum(np.count_nonzero(np.asarray(weights[k], dtype=np.float32) != original[k]) for k in original)
        size = sum(v.size for v in original.values())
        difference = np.sqrt(sum(float(np.sum((np.asarray(weights[k], dtype=np.float64) - original[k]) ** 2)) for k in original))
        metrics = {"changed_parameter_fraction": float(changed / size), "edit_l2_norm": float(difference)}
        with np.load(task.directory / "private.npz", allow_pickle=False) as hidden:
            if task.family == "write_controller":
                outcomes = controller_rollouts(task, weights, hidden["initial"])
                rate = float(np.mean(outcomes["success"]))
                metrics.update({"episode_success_rate": rate, "mean_position_error": float(np.mean(outcomes["position_error"])),
                                "mean_velocity_error": float(np.mean(outcomes["velocity_error"]))})
                return Score(rate >= .95, metrics)
            outputs = forward(task.architecture, weights, hidden["x"])
            if not np.isfinite(outputs).all():
                return Score(False, metrics, "invalid_submission", "Nonfinite network outputs")
            if task.family == "write_piecewise":
                errors = np.abs(outputs - hidden["y"])
                # For this one-hidden-layer ReLU network the error is piecewise linear.
                # Its maximum lies at a reference knot or a submitted activation boundary.
                w = np.asarray(weights["w0"], dtype=np.float64)[:, 0]
                b = np.asarray(weights["b0"], dtype=np.float64)
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    boundaries = np.divide(-b, w, out=np.full_like(w, np.nan), where=w != 0)
                boundaries = boundaries[np.isfinite(boundaries) & (boundaries >= 0) & (boundaries <= 1)]
                exact_x = np.unique(np.r_[0., 1., task.public["knots"], boundaries]).reshape(-1, 1)
                exact_y = np.interp(exact_x[:, 0], task.public["knots"], task.public["values"]).reshape(-1, 1)
                exact_errors = np.abs(forward(task.architecture, weights, exact_x) - exact_y)
                max_error = float(max(np.max(errors), np.max(exact_errors)))
                metrics.update({"max_absolute_error": max_error, "mean_absolute_error": float(np.mean(errors))})
                return Score(max_error <= .001, metrics)
            target_accuracy = float(np.mean((outputs[:, :1] >= 0) == hidden["y"]))
            # Preservation measures previously correct examples, not the broken network's errors.
            before_outputs = forward(task.architecture, original, hidden["preserve_x"])
            after_outputs = forward(task.architecture, weights, hidden["preserve_x"])
            before = before_outputs[:, :1] >= 0
            correct_before = before[:, 0] == hidden["preserve_y"][:, 0]
            after = after_outputs[:, :1] >= 0
            if not correct_before.any():
                return Score(False, metrics, "invalid_task", "No previously correct preservation cases")
            retention = float(np.mean(after[:, 0][correct_before] == hidden["preserve_y"][:, 0][correct_before]))
            target_loss = 1 - retention
            auxiliary_error = np.abs(after_outputs[:, 1] - before_outputs[:, 1])
            auxiliary_retention = float(np.mean(auxiliary_error <= .001))
            loss = max(target_loss, 1 - auxiliary_retention)
            metrics.update({"target_accuracy": target_accuracy, "retention_accuracy": min(retention, auxiliary_retention),
                            "motif_retention_accuracy": retention, "auxiliary_retention_accuracy": auxiliary_retention,
                            "auxiliary_max_absolute_error": float(np.max(auxiliary_error)),
                            "retention_loss": loss, "preservation_case_count": int(correct_before.sum())})
            return Score(target_accuracy >= .95 and loss <= .02, metrics)
    except (ValueError, TypeError, KeyError, OSError, OverflowError, zipfile.BadZipFile, EOFError) as exc:
        return Score(False, {}, "invalid_submission", str(exc))


def evaluate_practice(task: Task, submission, indices: list[int]) -> dict:
    if task.skill == "read":
        raise ValueError("Read tasks prohibit target execution and behavioral probes")
    if not isinstance(indices, list) or not indices or any(type(i) is not int for i in indices):
        raise ValueError("Practice indices must be a nonempty list of integers")
    if len(set(indices)) != len(indices):
        raise ValueError("A practice batch cannot repeat indices")
    weights = _weights(task, submission)
    maximum = task.limits.practice_episodes if task.family == "write_controller" else task.limits.practice_examples
    if min(indices) < 0 or max(indices) >= maximum or len(indices) > maximum:
        raise ValueError("Practice index outside the permitted pool")
    with np.load(task.practice_path, allow_pickle=False) as practice:
        if task.family == "write_controller":
            initial = practice["initial"][indices]
            outcomes = controller_rollouts(task, weights, initial)
            return {"kind": "episodes", "episodes": len(indices), "environment_steps": len(indices) * task.limits.episode_steps,
                    "success_rate": float(np.mean(outcomes["success"])),
                    "cases": [{"index": i, "initial": initial[j].tolist(), "success": bool(outcomes["success"][j]),
                               "final_position": float(outcomes["final_position"][j]), "final_velocity": float(outcomes["final_velocity"][j])}
                              for j, i in enumerate(indices)]}
        x, y = practice["x"][indices], practice["y"][indices]
        output = forward(task.architecture, weights, x)
        result = {"kind": "examples", "examples": len(indices),
                  "cases": [{"index": i, "input": x[j].tolist(), "expected": y[j].tolist(), "output": output[j].tolist()} for j, i in enumerate(indices)]}
        if task.skill == "repair":
            result["accuracy"] = float(np.mean((output[:, :1] >= 0) == y))
            original_output = forward(task.architecture, load_weights(task.weights_path), x)
            result["auxiliary_retention_accuracy"] = float(np.mean(np.abs(output[:, 1] - original_output[:, 1]) <= .001))
            for j, case in enumerate(result["cases"]):
                case["original_output"] = original_output[j].tolist()
        else:
            result["max_absolute_error"] = float(np.max(np.abs(output - y)))
        return result
