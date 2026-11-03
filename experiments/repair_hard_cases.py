"""Selective repair cases with coupled faults and retention-sensitive tests."""

from dataclasses import dataclass

import numpy as np

from repair_cases import (
    NONE, RETENTION_KEY, RETENTION_MAX_LOSS, TARGET_KEY, TARGET_MIN,
    Case, _hidden, _outputs, _patch, oracle,
)

FAULTS = ("coupled_gates", "sensor_routing", "readout_redistribution", "cross_layer_drift")
INPUTS = 6
WIDTH = 14
HIDDEN_TARGETS = 96
HIDDEN_RETENTION = 128
PRACTICE_PER_SLICE = 6
LOCAL_CANDIDATES = 12


def _sequence(edits):
    return {"operation": "sequence", "edits": edits}


def _edit(operation, tensor, index, value):
    return {"operation": operation, "tensor": tensor, "index": tuple(map(int, index)), "value": value}


def _apply(weights, action):
    if action["operation"] != "sequence":
        return _patch(weights, action)
    for edit in action["edits"]:
        weights = _patch(weights, edit)
    return weights


@dataclass
class HardCase(Case):
    def _intervention(self, patch):
        return _apply(self._numeric_weights(), self.public["interventions"][patch])


def _samples(rng, count):
    return rng.uniform(-1, 1, (count, INPUTS)).round(4)


def _fit(rng):
    rows = rng.normal(0, 0.55, (WIDTH, INPUTS))
    weights = {"w0": rows.round(4), "b0": rng.uniform(-0.45, 0.35, WIDTH).round(4)}
    examples = _samples(rng, 1200)
    basis = _hidden(weights, examples)
    target = basis @ rng.uniform(-0.9, 0.9, WIDTH)
    target -= np.median(target)
    target += rng.normal(0, 0.015, len(examples))
    fitted = np.linalg.lstsq(np.column_stack([basis, np.ones(len(examples))]), target, rcond=None)[0]
    return weights | {"w1": fitted[:-1].round(5), "b1": np.asarray(fitted[-1].round(5))}


def _candidates(rng, family):
    candidates = []
    signatures = set()
    while len(candidates) < LOCAL_CANDIDATES:
        u, v = rng.choice(WIDTH, 2, replace=False)
        s, t = rng.choice(INPUTS, 2, replace=False)
        sign = float(rng.choice([-1, 1]))
        if family == "coupled_gates":
            edits = [_edit("add", "b0", (u,), sign * .45), _edit("add", "b0", (v,), -sign * .45)]
        elif family == "sensor_routing":
            edits = [_edit("add", "w0", (u, s), sign * .65), _edit("add", "w0", (u, t), -sign * .65),
                     _edit("add", "w0", (v, s), -sign * .65), _edit("add", "w0", (v, t), sign * .65)]
        elif family == "readout_redistribution":
            edits = [_edit("add", "w1", (u,), sign * .42), _edit("add", "w1", (v,), -sign * .42)]
        elif family == "cross_layer_drift":
            edits = [_edit("add", "b0", (u,), sign * .38), _edit("add", "w1", (v,), -sign * .43)]
        else:
            raise ValueError(f"Unknown hard repair mechanism {family}")
        signature = tuple(sorted((e["tensor"], e["index"], e["value"]) for e in edits))
        if signature not in signatures:
            signatures.add(signature)
            candidates.append(_sequence(edits))
    return candidates


def _inverse(action):
    return _sequence([edit | {"value": -edit["value"]} for edit in reversed(action["edits"])])


def _retention_bank(rng, ordinary, clean, original, actions):
    truth, _, _ = _outputs(clean, ordinary)
    output, _, margins = _outputs(original, ordinary)
    stable = (truth == output) & (np.abs(margins) > .015)
    go, brake = (np.flatnonzero(stable & label) for label in (truth, ~truth))
    half = HIDDEN_RETENTION // 2
    if min(len(go), len(brake)) < half + PRACTICE_PER_SLICE:
        return None
    p = PRACTICE_PER_SLICE // 2
    practice = np.concatenate([go[:p], brake[:p]])
    allowed = stable.copy()
    allowed[practice] = False
    # Include ordinary observations whose prior correct action would be
    # broken by proposed edits. Fill the rest from the same input pool.
    chosen = {True: set(), False: set()}
    for action in actions:
        changed, _, _ = _outputs(_apply(original, action), ordinary)
        for label in (True, False):
            hits = np.flatnonzero(allowed & (truth == label) & (truth != changed))
            for index in hits[:5]:
                if len(chosen[label]) < half:
                    chosen[label].add(int(index))
    for label, pool in ((True, go[p:]), (False, brake[p:])):
        for index in pool:
            if len(chosen[label]) == half:
                break
            chosen[label].add(int(index))
    hidden = np.array(sorted(chosen[True]) + sorted(chosen[False]))
    rng.shuffle(hidden)
    return ordinary[practice], truth[practice], ordinary[hidden], truth[hidden]


def _make_case(rng, case_id, family, clean_case=False, fixed_decoy=False):
    for _ in range(80):
        clean = _fit(rng)
        actions = _candidates(rng, family)
        true_action = actions[int(rng.integers(len(actions)))]
        corrupted = _apply(clean, _inverse(true_action))
        samples = _samples(rng, 12_000)
        _, _, clean_logits = _outputs(clean, samples)
        _, _, corrupt_logits = _outputs(corrupted, samples)
        failures = samples[(clean_logits < -.015) & (corrupt_logits > .02)]
        if len(failures) < PRACTICE_PER_SLICE + HIDDEN_TARGETS:
            continue
        practice_target = failures[:PRACTICE_PER_SLICE]
        target = failures[PRACTICE_PER_SLICE:PRACTICE_PER_SLICE + HIDDEN_TARGETS]
        _, _, target_logits = _outputs(corrupted, target)
        shift = -2.0 if fixed_decoy else -float(np.max(target_logits).round(4)) - .025
        broad = _edit("add", "b1", (), shift)
        actions.append(broad)
        rng.shuffle(actions)
        actual = clean if clean_case else corrupted
        bank = _retention_bank(rng, _samples(rng, 4096), clean, actual, actions)
        if bank is None:
            continue
        practice_retention, practice_labels, retention, retention_labels = bank
        practice = np.concatenate([practice_target, practice_retention])
        labels = np.concatenate([np.zeros(PRACTICE_PER_SLICE, dtype=bool), practice_labels])
        if (set(map(tuple, practice)) & (set(map(tuple, target)) | set(map(tuple, retention)))
                or set(map(tuple, target)) & set(map(tuple, retention))):
            continue
        interventions = {NONE: {"operation": "none"},
                         **{f"i{i+1:02d}": action for i, action in enumerate(actions)}}
        public = {
            "architecture": {"inputs": ["clearance", "alignment", "particulate signal",
                                        "reflected light", "surface gradient", "lateral movement"],
                             "equations": ["h = relu(w0 @ x + b0)", "logit = w1 @ h + b1",
                                           "action is 1 (GO) when logit > 0, otherwise 0 (BRAKE)"]},
            "report": "Investigate unsafe GO actions in the target slice. A report may be a false alarm. "
                      "All target examples require BRAKE. Preserve previously correct actions on ordinary states. "
                      "The retention bank balances ordinary GO and BRAKE examples and includes states near "
                      "decision boundaries that expose unintended changes from proposed edits.",
            "practice": [{"index": i, "slice": "target" if i < PRACTICE_PER_SLICE else "retention",
                          "inputs": row.tolist(), "expected_action": int(labels[i])}
                         for i, row in enumerate(practice)],
            "interventions": interventions,
            "forecast": {"definition": "For each named intervention predict the fraction of correct actions "
                                        "on unseen target and retention examples.",
                         "keys": [TARGET_KEY, RETENTION_KEY], "range": [0, 1]},
            "repair": {"definition": "Select one intervention ID. An intervention marked sequence applies every "
                                     "listed edit in order. Tensor indices are zero-based.",
                       "target_min": TARGET_MIN, "retention_loss_max": RETENTION_MAX_LOSS},
        }
        case = HardCase(case_id, "clean" if clean_case else family, public,
                        {key: value.tolist() for key, value in actual.items()}, practice, labels,
                        target, retention, retention_labels)
        forecasts, patch = oracle(case)
        winners = [name for name in interventions if case.grade(forecasts, name)["success"]]
        retention_trap = any(row[TARGET_KEY] >= TARGET_MIN and
                             forecasts[NONE][RETENTION_KEY] - row[RETENTION_KEY] > RETENTION_MAX_LOSS
                             for row in forecasts.values())
        if winners == [patch] and (clean_case or (patch != NONE and retention_trap)):
            return case
    raise ValueError(f"Could not construct an admissible hard case for {family}")


def make_cases(seed, *, fixed_decoy=False):
    """Create four independently fitted faults and an independent clean control."""
    generators = [np.random.default_rng(key) for key in np.random.SeedSequence(seed).spawn(5)]
    cases = [_make_case(rng, f"hard-{i:02d}", family, fixed_decoy=fixed_decoy)
             for i, (rng, family) in enumerate(zip(generators, FAULTS), 1)]
    clean_rng = generators[-1]
    clean_family = FAULTS[int(clean_rng.integers(len(FAULTS)))]
    cases.append(_make_case(clean_rng, "hard-05", clean_family, clean_case=True, fixed_decoy=fixed_decoy))
    return cases


def _practice_pair(public):
    rows = public["practice"]
    return [next(row["index"] for row in rows if row["slice"] == "target"),
            next(row["index"] for row in rows if row["slice"] == "retention" and row["expected_action"] == 1)]


def scripted_forecast(public, probe_callable):
    target, retention = probe_callable(_practice_pair(public), NONE)
    target_rate = float(target["action"] == target["expected_action"])
    retention_rate = float(retention["action"] == retention["expected_action"])
    forecasts = {NONE: {TARGET_KEY: target_rate, RETENTION_KEY: retention_rate}}
    for name, action in public["interventions"].items():
        if name != NONE:
            broad = action["operation"] != "sequence"
            forecasts[name] = {
                TARGET_KEY: 1.0 if broad else .25 + .65 * target_rate,
                RETENTION_KEY: .5 if broad else .95 * retention_rate,
            }
    return forecasts


def scripted_repair(public, forecasts, probe_callable):
    if forecasts[NONE][TARGET_KEY] == 1.0:
        return NONE
    indices = _practice_pair(public)
    candidates = [name for name, action in public["interventions"].items()
                  if action["operation"] == "sequence"]
    best = None
    used_components = set()
    for name in candidates:
        action = public["interventions"][name]
        components = {(edit["tensor"], edit["index"][0]) for edit in action["edits"]}
        if used_components and components & used_components:
            continue
        target, retention = probe_callable(indices, name)
        if target["action"] == target["expected_action"] and retention["action"] == retention["expected_action"]:
            result = (abs(target["logit"]), name)
            best = result if best is None else max(best, result)
        if used_components:
            break
        used_components = components
    return best[1] if best else NONE
