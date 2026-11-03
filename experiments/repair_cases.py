"""Synthetic controller checkpoints and trusted scoring for selective repair."""

from dataclasses import dataclass

import numpy as np


NONE = "none"
TARGET_KEY = "target_success"
RETENTION_KEY = "retention"
TARGET_MIN = 0.95
RETENTION_MAX_LOSS = 0.02
PRACTICE_SIZE = 4
HIDDEN_SIZE = 24


def _hidden(weights, inputs):
    return np.maximum(inputs @ weights["w0"].T + weights["b0"], 0.0)


def _outputs(weights, inputs):
    hidden = _hidden(weights, inputs)
    logits = hidden @ weights["w1"] + weights["b1"]
    return logits > 0, hidden, logits


def _patch(weights, action):
    revised = {name: value.copy() for name, value in weights.items()}
    if action["operation"] != "none":
        tensor = revised[action["tensor"]]
        index = action["index"]
        if action["operation"] == "set":
            tensor[index] = action["value"]
        elif action["operation"] == "scale":
            tensor[index] *= action["value"]
        elif action["operation"] == "add":
            tensor[index] += action["value"]
        else:
            raise ValueError("Unsupported intervention")
    return revised


@dataclass
class Case:
    id: str
    kind: str
    public: dict
    weights: dict
    _practice_inputs: np.ndarray
    _practice_labels: np.ndarray
    _target: np.ndarray
    _retention_inputs: np.ndarray
    _retention_labels: np.ndarray

    def _numeric_weights(self):
        return {name: np.asarray(value, dtype=float) for name, value in self.weights.items()}

    def _intervention(self, patch):
        return _patch(self._numeric_weights(), self.public["interventions"][patch])

    def probe(self, index, activations=False, patch=NONE):
        if type(index) is not int or not 0 <= index < len(self._practice_inputs):
            raise ValueError("Unknown practice index")
        if not isinstance(patch, str) or patch not in self.public["interventions"]:
            raise ValueError("Unknown intervention ID")
        prediction, hidden, logits = _outputs(self._intervention(patch), self._practice_inputs[index:index + 1])
        result = {"index": index, "patch": patch, "action": int(prediction[0]),
                  "expected_action": int(self._practice_labels[index]),
                  "logit": round(float(logits[0]), 5)}
        if activations:
            result["hidden"] = hidden[0].round(5).tolist()
        return result

    def _scores(self, patch):
        weights = self._intervention(patch)
        target, _, _ = _outputs(weights, self._target)
        retention, _, _ = _outputs(weights, self._retention_inputs)
        return {TARGET_KEY: float(np.mean(~target)),
                RETENTION_KEY: float(np.mean(retention == self._retention_labels))}

    def grade(self, forecasts, patch):
        valid_patch = isinstance(patch, str) and patch in self.public["interventions"]
        reference = {name: self._scores(name) for name in self.public["interventions"]}
        valid_forecast = isinstance(forecasts, dict) and set(forecasts) == set(reference)
        if valid_forecast:
            for row in forecasts.values():
                if not isinstance(row, dict) or set(row) != {TARGET_KEY, RETENTION_KEY} or any(
                    type(value) not in (int, float) or not 0 <= value <= 1 for value in row.values()
                ):
                    valid_forecast = False
                    break
        forecast_mae = float(np.mean([
            abs(forecasts[name][key] - observed) for name, row in reference.items()
            for key, observed in row.items()
        ])) if valid_forecast else 1.0
        observed = reference[patch] if valid_patch else {TARGET_KEY: 0.0, RETENTION_KEY: 0.0}
        retention_loss = max(0.0, reference[NONE][RETENTION_KEY] - observed[RETENTION_KEY])
        success = (valid_patch and observed[TARGET_KEY] >= TARGET_MIN
                   and retention_loss <= RETENTION_MAX_LOSS and (self.kind != "clean" or patch == NONE))
        return {**observed, "retention_loss": retention_loss, "success": bool(success),
                "forecast_mae": forecast_mae, "forecast_score": 1.0 - forecast_mae,
                "valid_forecast": valid_forecast, "valid_patch": valid_patch, "clean_case": self.kind == "clean"}


def _inputs(rng, size, failure=False):
    if failure:
        bounds = ((0.19, 0.48), (-0.55, 0.55), (0.54, 0.68), (0.71, 0.79))
    else:
        bounds = ((0.0, 1.0), (-1.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    low, high = np.asarray(bounds).T
    return rng.uniform(low, high, size=(size, len(low))).round(4)


def _train(rng):
    # Fixed features provide interpretable interventions. Fit the action
    # readout to fresh noisy expert demonstrations before injecting faults.
    clean = {"w0": np.array([[1, 0, 0, 0], [-1, 0, 0, 0], [0, 1, 0, 0],
                              [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float),
             "b0": np.array([-0.5, 0.5, 0, 0, -0.7, -0.6], dtype=float)}
    demonstrations = _inputs(rng, 1024)
    features = _hidden(clean, demonstrations)
    expert = np.array([1, -1, 0.18, -0.18, 0.75, 0.50]) * rng.uniform(0.97, 1.03, 6)
    target = features @ expert + rng.normal(0, 0.005, len(features))
    # Fitted coefficients vary across independent checkpoints.
    design = np.column_stack([features, np.ones(len(features))])
    fitted = np.linalg.lstsq(design, target, rcond=None)[0].round(4)
    clean.update(w1=fitted[:-1], b1=np.asarray(fitted[-1]))
    return clean


def _make_case(rng, case_id, kind):
    clean = _train(rng)
    fault_a = _patch(clean, {"operation": "set", "tensor": "b0", "index": 4, "value": -0.15})
    fault_b = _patch(clean, {"operation": "scale", "tensor": "w1", "index": 5, "value": 4})
    actual = {"bias_fault": fault_a, "readout_fault": fault_b, "clean": clean}[kind]

    candidates = _inputs(rng, 4096, failure=True)
    clean_target, _, clean_logits = _outputs(clean, candidates)
    a, _, a_logits = _outputs(fault_a, candidates)
    b, _, b_logits = _outputs(fault_b, candidates)
    selected = candidates[~clean_target & a & b & (clean_logits < -0.08)
                          & (a_logits > 0.035) & (b_logits > 0.035)]
    if len(selected) < PRACTICE_SIZE + HIDDEN_SIZE:
        raise ValueError("Insufficient matched failure inputs")
    target_practice = selected[:PRACTICE_SIZE]
    target = selected[PRACTICE_SIZE:PRACTICE_SIZE + HIDDEN_SIZE]

    ordinary = _inputs(rng, 4096)
    nominal, _, nominal_logits = _outputs(clean, ordinary)
    original, _, original_logits = _outputs(actual, ordinary)
    # Include low-margin healthy GO examples to detect blanket brake patches.
    stable = (nominal == original) & (np.abs(nominal_logits) > 0.08) & (np.abs(original_logits) > 0.08)
    go = ordinary[stable & nominal & (original_logits < 0.47)]
    stop = ordinary[stable & ~nominal]
    half = HIDDEN_SIZE // 2
    if min(len(go), len(stop)) < half + PRACTICE_SIZE:
        raise ValueError("Insufficient preservation inputs")
    preservation_practice = np.concatenate([go[:2], stop[:2]])
    retention = np.concatenate([go[2:half + 2], stop[2:half + 2]])
    retention_labels, _, _ = _outputs(clean, retention)
    practice = np.concatenate([target_practice, preservation_practice])
    practice_labels, _, _ = _outputs(clean, practice)

    # Permute hidden units and candidate IDs independently to avoid positional
    # cues while keeping the actionable operations public to every arm.
    permutation = rng.permutation(6)
    for name in ("w0", "b0", "w1"):
        actual[name] = actual[name][permutation]
    actions = [
        {"operation": "set", "tensor": "b0", "index": int(np.flatnonzero(permutation == 4)[0]), "value": -0.7},
        {"operation": "set", "tensor": "b0", "index": int(np.flatnonzero(permutation == 4)[0]), "value": -0.42},
        {"operation": "scale", "tensor": "w1", "index": int(np.flatnonzero(permutation == 5)[0]), "value": 0.25},
        {"operation": "scale", "tensor": "w1", "index": int(np.flatnonzero(permutation == 5)[0]), "value": 0.60},
        {"operation": "add", "tensor": "b1", "index": (), "value": -0.65},
    ]
    rng.shuffle(actions)
    interventions = {NONE: {"operation": "none"}, **{f"i{i + 1}": action for i, action in enumerate(actions)}}
    public = {
        "architecture": {"inputs": ["forward clearance", "heading alignment", "particulate sensor", "reflected light"],
                         "equations": ["h = relu(w0 @ x + b0)", "logit = w1 @ h + b1",
                                       "output is 1 (GO) when logit > 0, otherwise 0 (BRAKE)"]},
        "report": "Investigate a report of unsafe GO decisions in the target scenario. Reports can be false alarms. "
                  "Restore BRAKE on the target slice while preserving previously correct actions on ordinary states.",
        "practice": [{"index": i, "slice": "target" if i < PRACTICE_SIZE else "retention",
                      "inputs": row.tolist(), "expected_action": int(practice_labels[i])}
                     for i, row in enumerate(practice)],
        "interventions": interventions,
        "forecast": {"definition": "For every intervention, predict the fractions correct on unseen target and "
                                   "retention examples after applying that intervention alone.",
                     "keys": [TARGET_KEY, RETENTION_KEY], "range": [0, 1]},
        "repair": {"definition": "Select one listed intervention ID, including none."},
    }
    return Case(case_id, kind, public, {name: value.tolist() for name, value in actual.items()},
                practice, practice_labels, target, retention, retention_labels)


def make_cases(seed):
    rng = np.random.default_rng(seed)
    cases = [_make_case(rng, f"controller-{index}", kind)
             for index, kind in enumerate(("bias_fault", "readout_fault", "clean"), start=1)]
    for case in cases:
        forecast, patch = oracle(case)
        if not case.grade(forecast, patch)["success"]:
            raise ValueError(f"No selective repair for {case.id}")
        if case.kind != "clean" and case.grade(forecast, NONE)["success"]:
            raise ValueError(f"Untreated fault passes {case.id}")
    return cases


def _practice_pair(public):
    return [next(row["index"] for row in public["practice"] if row["slice"] == name)
            for name in ("target", "retention")]


def scripted_forecast(public, probe_callable):
    target, retention = probe_callable(_practice_pair(public), NONE)
    base_target = float(target["action"] == target["expected_action"])
    base_retention = float(retention["action"] == retention["expected_action"])
    forecast = {NONE: {TARGET_KEY: base_target, RETENTION_KEY: base_retention}}
    for name, action in public["interventions"].items():
        if name != NONE:
            broad = action["tensor"] == "b1"
            forecast[name] = {TARGET_KEY: 1.0 if broad else 0.5 + base_target / 2,
                              RETENTION_KEY: base_retention / 2 if broad else base_retention}
    return forecast


def scripted_repair(public, forecasts, probe_callable):
    if forecasts[NONE][TARGET_KEY] == 1.0:
        return NONE
    # Try a localized edit on each component before shifting the policy's
    # decision threshold. The fixed budget permits one test per component.
    indices = _practice_pair(public)
    choices = []
    tested = set()
    for name, action in public["interventions"].items():
        if name == NONE or action["tensor"] == "b1":
            continue
        component = (action["tensor"], action["index"])
        if component in tested:
            continue
        tested.add(component)
        target, retention = probe_callable(indices, name)
        if target["action"] == target["expected_action"]:
            choices.append((retention["action"] == retention["expected_action"], name))
    return max(choices)[1] if choices else NONE


def oracle(case):
    forecasts = {name: case._scores(name) for name in case.public["interventions"]}
    if case.kind == "clean":
        return forecasts, NONE
    patch = max(forecasts, key=lambda name: (
        forecasts[name][TARGET_KEY] >= TARGET_MIN and
        1.0 - forecasts[name][RETENTION_KEY] <= RETENTION_MAX_LOSS,
        forecasts[name][TARGET_KEY], forecasts[name][RETENTION_KEY]))
    return forecasts, patch
