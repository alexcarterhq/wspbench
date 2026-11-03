"""Conventional comparators with explicit information and observation accounting.

These scripts are B/C difficulty controls, never eligible no-tool A model rows.
The only supervised fitting uses the separate ``baseline_train`` population.
No construction or selection routine imports the hidden evaluator.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import time

import numpy as np

from .access import AccessSession
from .contracts import Condition, FAMILIES
from .patches import diff_patch, materialize
from .reporting import build_report
from .serialization import canonical_json, load_weights, serialization_digest, weights_digest


BASELINES = {
    "metadata_prior": {"conditions": ["B", "C"], "families": list(FAMILIES),
        "method": "Balanced public-catalog prior; empty patch for repair/write", "weights": False},
    "statistical_reader": {"conditions": ["B", "C"], "families": ["rule_recovery", "edit_effects"],
        "method": "Balanced supervised shrinkage nearest-centroid reader of moments and singular spectra", "weights": True},
    "circuit_algebra": {"conditions": ["B", "C"], "families": ["rule_recovery", "edit_effects", "causal_forecast"],
        "method": "QK/OV signed-path and ReLU-hinge invariant matching against public program templates", "weights": True},
    "no_op": {"conditions": ["B", "C"], "families": ["coupled_repair", "compositional_synthesis", "loop_repair"],
        "method": "Preserve every starting value", "weights": False},
    "generic_repair": {"conditions": ["B", "C"], "families": ["coupled_repair"],
        "method": "Declared inverse faults plus bias/gain/head and auxiliary-null-space candidates; fixed witness selection", "weights": True},
    "specification_splice": {"conditions": ["B", "C"], "families": ["coupled_repair"],
        "method": "Public-spec construction and auxiliary-preserving head splice when the public specification is complete", "weights": False},
    "public_spec_constructor": {"conditions": ["B", "C"], "families": ["compositional_synthesis"],
        "method": "Compile the solver-visible complete specification into the fixed architecture", "weights": False},
    "constrained_lookup": {"conditions": ["B", "C"], "families": ["compositional_synthesis"],
        "method": "Capacity-bounded table/circuit construction with explicit representability refusal", "weights": False},
    "behavior_only": {"conditions": ["C"], "families": ["rule_recovery", "edit_effects", "causal_forecast", "coupled_repair", "loop_repair"],
        "method": "Fixed practice-only candidate identification without informative starting weights or activations", "weights": False},
    "contrastive_mean": {"conditions": ["C"], "families": ["loop_repair"],
        "method": "Metered contrastive hidden-state means, writer localization and bounded rank-one suppression", "weights": True},
    "contrastive_pca": {"conditions": ["C"], "families": ["loop_repair"],
        "method": "Metered failure-state PCA, writer localization and bounded rank-one suppression", "weights": True},
    "random_direction": {"conditions": ["C"], "families": ["loop_repair"],
        "method": "Seeded unit random direction with the same writer/strength schedule and observation budget", "weights": True},
    "symbolic_effects": {"conditions": ["B", "C"], "families": ["causal_forecast"],
        "method": "Static piecewise-affine circuit integration over the complete public input measure", "weights": True},
    "anf_constructor": {"conditions": ["B", "C"], "families": ["compositional_synthesis"],
        "method": "Independent algebraic-normal-form synthesis under the frozen ReLU capacity", "weights": False},
    "loop_spec_constructor": {"conditions": ["C"], "families": ["loop_repair"],
        "method": "Public copy specification rebuilt analytically without informative weights", "weights": False},
}


def access_card(name: str, *, training: dict | None = None) -> dict:
    card = BASELINES[name]
    return {"name": name, "actor_type": "baseline", "eligible_unaided_model": False,
            "privileged": False, "uses_starting_weights": card["weights"],
            "method": card["method"], "tuning": "Fixed implementation; no hidden-score selection",
            "computation": "Direct in-process NumPy conventional difficulty control; CPU/observation caps match, but not the model's 12-call algebra scaffold",
            "observations": "AccessSession practice endpoint only" if "C" in card["conditions"] else "none",
            "target_parameter_optimization": False, "inference_runtime": "benchmark-owned architecture",
            "capacity": "Task's fixed architecture and scalar cap", "prior_training": training or "none",
            "qualifying_families": ["rule_recovery"] if name == "statistical_reader" else card["families"],
            "label_coverage": "R1 public catalog only; R3 translated after-programs are an out-of-distribution transfer diagnostic" if name == "statistical_reader" else "not supervised"}


def weight_features(weights: dict[str, np.ndarray]) -> np.ndarray:
    """Permutation-resistant scalar statistics and matrix singular spectra."""
    features = []
    for name in sorted(weights):
        a = np.asarray(weights[name], dtype=np.float64)
        flat = a.ravel()
        quantiles = np.quantile(flat, [0, .1, .25, .5, .75, .9, 1])
        scale = float(np.sqrt(np.mean(flat * flat)))
        moments = [flat.mean(), flat.std(), scale, np.mean(flat > 0), np.mean(flat == 0),
                   np.mean((flat / max(scale, 1e-12)) ** 3), np.mean((flat / max(scale, 1e-12)) ** 4)]
        spectrum = np.linalg.svd(a, compute_uv=False) if a.ndim == 2 else np.sort(np.abs(flat))[::-1]
        singular = np.zeros(16)
        singular[:min(16, len(spectrum))] = spectrum[:16]
        features.extend([*moments, *quantiles, *singular])
    return np.asarray(features)


class StatisticalReader:
    """A task-specific conventional reader fitted on independently generated labels."""
    def __init__(self):
        self.fitted = False

    def fit(self, examples: list[dict], *, required_labels: list[dict], min_per_label: int = 32):
        if not examples or any(x.get("split") != "baseline_train" for x in examples):
            raise ValueError("Reader fitting requires the separate baseline_train split exclusively")
        labels = [canonical_json(x["label"]) for x in examples]
        required = {canonical_json(x) for x in required_labels}
        if set(labels) != required:
            raise ValueError("Training population must cover exactly the declared label catalog")
        independent = [x.get("independent_network_id", x["parent_problem_id"]) for x in examples]
        if len(set(independent)) != len(examples):
            raise ValueError("Repeated parameterizations do not count as independent training examples")
        counts = Counter(labels)
        if min(counts.values()) < min_per_label or len(set(counts.values())) != 1:
            raise ValueError("Reader training requires a balanced population of at least 32 examples per label")
        x = np.stack([weight_features(e["weights"]) for e in examples])
        self.mean = x.mean(axis=0)
        self.scale = np.maximum(x.std(axis=0), 1e-7)
        z = (x - self.mean) / self.scale
        self.labels = sorted(required)
        self.centroids = np.stack([z[np.asarray(labels) == label].mean(axis=0) for label in self.labels])
        # Diagonal pooled within-class covariance is robust at this small N and
        # uses all labels without fitting thousands of underdetermined slopes.
        residual = np.stack([row - self.centroids[self.labels.index(label)] for row, label in zip(z, labels)])
        self.variance = .5 + .5 * residual.var(axis=0)
        self.training = {"split": "baseline_train", "examples": len(examples),
                         "independently_fitted_networks": len(set(independent)),
                         "semantic_problem_clusters": len({x.get("mechanism_family_id", x["parent_problem_id"]) for x in examples}), "labels": len(required),
                         "examples_per_label": min(counts.values()), "features": len(self.mean),
                         "estimator": "shrinkage diagonal discriminant / nearest centroid"}
        self.fitted = True
        return self

    def predict(self, weights: dict[str, np.ndarray]) -> dict:
        if not self.fitted:
            raise ValueError("Statistical reader has not been fitted")
        z = (weight_features(weights) - self.mean) / self.scale
        distance = (((self.centroids - z) ** 2) / self.variance).mean(axis=1)
        return json.loads(self.labels[int(np.argmin(distance))])


def _attention_signature(weights: dict[str, np.ndarray]) -> np.ndarray:
    """Contract circuits only; do not feed real/synthetic sequences through target."""
    required = {"e", "p", "q", "k", "v", "o", "w0", "b0", "w1", "b1"}
    if not required <= set(weights):
        raise ValueError("Circuit signature requires the declared attention_pool architecture")
    w = {k: np.asarray(v, dtype=np.float64) for k, v in weights.items()}
    # Attention table indexes independent token and position basis vectors;
    # it is an algebraic parameter contraction, not a sequence execution.
    cls = w["e"][4] + w["p"][5]
    qk = (cls @ w["q"].T) @ w["k"]
    route = np.concatenate((w["e"] @ qk, w["p"] @ qk))
    paths = w["w0"] @ w["o"] @ w["v"]
    incoming = np.concatenate((paths @ w["e"].T, paths @ w["p"].T, w["b0"][:, None]), axis=1)
    # Positive ReLU-rescaling and hidden-unit permutation invariance.
    norms = np.maximum(np.linalg.norm(incoming, axis=1), 1e-12)
    hinges = np.concatenate((incoming / norms[:, None], (w["w1"] * norms).T), axis=1)
    order = sorted(range(len(hinges)), key=lambda i: tuple(np.round(hinges[i], 6)))
    return np.concatenate((route, hinges[order].ravel(), w["b1"]))


def _compile(program, architecture):
    from .networks import compile_program
    result = compile_program(program, architecture=architecture)
    # Compiler returns weights; tuple compatibility keeps the baseline decoupled
    # from a possible architecture-plus-weights generation helper.
    if isinstance(result, tuple):
        result = next(x for x in result if isinstance(x, dict) and "e" in x)
    return result


def _catalog(public: dict) -> list[dict]:
    result = list(public.get("program_candidates", []))
    # R3's public answer grammar allows a unit translation of any scalar
    # readout. Enumerating that declared operation does not use an edit label.
    if public.get("answer_schema", {}).get("type") == "before_after_programs":
        def shift(node):
            node = deepcopy(node)
            if node["op"] == "mod": node["offset"] = (node["offset"]+1) % 3
            elif node["op"] == "threshold": node["cuts"] = [v-1 for v in node["cuts"]]
            else:
                node["threshold"] -= 1
                node["then"], node["else"] = shift(node["then"]), shift(node["else"])
            return node
        result += [{**deepcopy(p), "readout": shift(p["readout"])} for p in result]
    return result


def _recover(weights: dict, public: dict, architecture: dict) -> tuple[dict, float]:
    candidates = _catalog(public)
    if not candidates:
        raise ValueError("No public program catalog for algebraic template matching")
    signature = _attention_signature(weights)
    distances, answers = [], []
    for program in candidates:
        template = _attention_signature(_compile(program, architecture))
        distances.append(float(np.linalg.norm(signature - template) / max(np.linalg.norm(template), 1e-12)))
        answers.append(program)
        if public.get("answer_schema", {}).get("type") == "before_after_programs":
            # Enumerate the publicly allowed internal translation implementation,
            # as distinct from a translated output-layer compiler layout.
            shifted = _compile(program, architecture)
            shifted["b0"] += shifted["w0"][:, 0]
            internal = _attention_signature(shifted)
            distances.append(float(np.linalg.norm(signature-internal) / max(np.linalg.norm(internal), 1e-12)))
            answer = deepcopy(program)
            def translate(node):
                node = deepcopy(node)
                if node["op"] == "mod": node["offset"] = (node["offset"]+1)%3
                elif node["op"] == "threshold": node["cuts"] = [v-1 for v in node["cuts"]]
                else:
                    node["threshold"] -= 1
                    node["then"], node["else"] = translate(node["then"]), translate(node["else"])
                return node
            answer["readout"] = translate(answer["readout"])
            answers.append(answer)
    index = int(np.argmin(distances))
    return deepcopy(answers[index]), distances[index]


def _na(reason: str) -> dict:
    return {"status": "not_applicable", "submission": {}, "diagnostics": {"reason": reason}}


def _done(submission: dict, **diagnostics) -> dict:
    return {"status": "completed", "submission": submission, "diagnostics": diagnostics}


def _probe(session, indices, **kwargs) -> list[dict]:
    """Split at the exact shared snapshots; charge repeats and candidates."""
    results = []
    offset = 0
    cost = getattr(session, "logical_evaluations_per_index", 1)
    while offset < len(indices):
        size = min(len(indices) - offset, session.remaining_observations // cost,
                   (session.next_boundary - session.observation_count) // cost)
        if size <= 0:
            break
        results.append(session.probe(indices[offset:offset + size], **kwargs))
        offset += size
    return results


def _behavior_rows(results: list[dict]) -> tuple[list, list]:
    inputs, outputs = [], []
    for result in results:
        inputs.extend(result.get("inputs", []))
        outputs.extend(result.get("outputs", result.get("predictions", [])))
    return inputs, outputs


def _behavior_recover(session, checkpoint="target", allowance=64) -> tuple[dict, dict]:
    from .programs import evaluate_program
    public = session.public_task["public"]
    indices = list(range(min(allowance, public["practice_size"], session.remaining_observations)))
    inputs, outputs = _behavior_rows(_probe(session, indices, checkpoint=checkpoint))
    if not inputs or not outputs:
        raise ValueError("Practice endpoint did not return labeled behavior")
    candidates = _catalog(public)
    empirical = [np.mean(np.asarray(evaluate_program(p, np.asarray(inputs))) != np.asarray(outputs).reshape(-1))
                 for p in candidates]
    index = int(np.argmin(empirical))
    return deepcopy(candidates[index]), {"practice_error": float(empirical[index]), "queries": len(inputs),
                                         "surviving_hypotheses": int(np.sum(np.asarray(empirical) == empirical[index]))}


def run_baseline(session: AccessSession, name: str, *, statistics_reader: StatisticalReader | None = None) -> dict:
    """Choose a submission using only the session view and metered C endpoint."""
    if name not in BASELINES:
        raise ValueError("Unknown baseline")
    definition = BASELINES[name]
    task = session.public_task
    family, public, architecture = task["family"], task["public"], task["architecture"]
    if session.condition.value not in definition["conditions"] or family not in definition["families"]:
        return _na("Method is outside this declared family/access condition")
    if architecture.get("kind") == "boolean_relu":
        from . import challenge_baselines, challenges
        if name in {"no_op", "metadata_prior"}:
            answer = {"patch": []} if task["skill"] != "read" else {"before_outputs": [0]*1024, "after_outputs": [0]*1024} if family == "edit_effects" else {"outputs": [0]*1024}
            return _done(answer, informative_weights=False)
        if name == "circuit_algebra":
            try:
                if family == "edit_effects":
                    answer={"before_program":challenges.decode_program(session.checkpoint_weights("before"))["program"],
                            "after_program":challenges.decode_program(session.checkpoint_weights("after"))["program"]}
                else:answer={"program":challenges.decode_program(session.view_weights)["program"]}
                return _done(answer, target_executions=0, method="exact public numeric microcircuit derivation")
            except ValueError as error:
                answer={"before_outputs":[0]*1024,"after_outputs":[0]*1024} if family=="edit_effects" else {"outputs":[0]*1024}
                return _done(answer, certified_decoder_refusal=str(error), target_executions=0)
        aliases={"statistical_reader":"structured_reader","behavior_only":"boolean_behavior",
                 "public_spec_constructor":"bdd_constructor","constrained_lookup":"dnf_constructor"}
        return challenge_baselines.run(session,aliases.get(name,name),statistics_reader)
    if name == "symbolic_effects":
        from .feasibility import symbolic_effects
        try:
            result=symbolic_effects(session.view_weights, architecture, public["forecast_interventions"])
            return _done({"effects":result["effects"]}, **result["certificate"])
        except ValueError as error:
            return _done({"effects":{e["id"]:[0.]*architecture["heads"] for e in public["forecast_interventions"]}},
                         certified_refusal=str(error),target_executions=0)
    if name == "loop_spec_constructor":
        from .networks import compile_loop
        weights=compile_loop(faulty=False)
        return _done({"patch":[{"op":"replace","tensor":k,"values":v.tolist()} for k,v in weights.items()]},
                     informative_weights=False, source="complete public copy specification")
    if name == "anf_constructor":
        return _na("ANF synthesis is a Boolean truth-table control; the scalar-catalog implementation is retained as sanity only")
    if name in {"no_op", "metadata_prior"}:
        candidates = _catalog(public)
        if task["skill"] != "read":
            return _done({"patch": []}, informative_weights=False)
        if not candidates:
            return _na("No balanced public hypothesis catalog for this prediction family")
        if family == "rule_recovery":
            return _done({"program": candidates[0]}, informative_weights=False, prior="fixed balanced catalog label")
        if family == "edit_effects":
            return _done({"before_program": candidates[0], "after_program": candidates[0]}, informative_weights=False)
        if family == "causal_forecast":
            return _done({"effects": {edit["id"]: [.5, .5] for edit in public.get("forecast_interventions", [])}},
                         informative_weights=False, prior="fixed midpoint effect-rate prior")
        return _na("No public hypothesis catalog")
    if name == "statistical_reader":
        if statistics_reader is None or not statistics_reader.fitted:
            return _na("Balanced >=32-per-label baseline_train reader has not been fitted")
        if family == "rule_recovery":
            return _done({"program": statistics_reader.predict(session.view_weights)}, training=statistics_reader.training)
        return _done({"before_program": statistics_reader.predict(session.checkpoint_weights("before")),
                      "after_program": statistics_reader.predict(session.checkpoint_weights("after"))},
                     training=statistics_reader.training)
    if name == "circuit_algebra":
        if architecture.get("kind") != "attention_pool":
            return _na("QK/OV algebra is not applicable to this architecture")
        if family == "rule_recovery":
            program, distance = _recover(session.view_weights, public, architecture)
            return _done({"program": program}, relative_signature_distance=distance, target_executions=0)
        if family == "edit_effects":
            before, da = _recover(session.checkpoint_weights("before"), public, architecture)
            after, db = _recover(session.checkpoint_weights("after"), public, architecture)
            return _done({"before_program": before, "after_program": after}, relative_signature_distances=[da, db], target_executions=0)
        return _causal_algebra(session)
    if name in {"public_spec_constructor", "constrained_lookup"}:
        return _synthesis(session, lookup=name == "constrained_lookup")
    if name == "generic_repair":
        return _generic_repair(session, informative_weights=True)
    if name == "specification_splice":
        # P1 deliberately has checkpoint-specific coupled behavior. A gold
        # clean-head copy would be a privileged oracle, not this comparator.
        if "specification" not in public:
            return _na("Normative witnesses do not supply a complete checkpoint-specific construction; a clean-head splice would require privileged weights")
        return _na("This dense shared architecture has no independently implemented public-spec compiler")
    if name == "behavior_only":
        if family == "rule_recovery":
            session.commit({"program": _catalog(public)[0]})
            program, diagnostics = _behavior_recover(session, allowance=128)
            return _done({"program": program}, informative_weights=False, **diagnostics)
        if family == "edit_effects":
            session.commit({"before_program": _catalog(public)[0], "after_program": _catalog(public)[0]})
            before, da = _behavior_recover(session, "before", 64)
            session.commit({"before_program": before, "after_program": _catalog(public)[0]})
            after, db = _behavior_recover(session, "after", 64)
            return _done({"before_program": before, "after_program": after}, informative_weights=False,
                         before=da, after=db)
        if family == "coupled_repair":
            return _generic_repair(session, informative_weights=False)
        if family == "causal_forecast":
            return _causal_behavior(session)
        return _loop_repair(session, method="behavior_only")
    return _loop_repair(session, method=name)


def _synthesis(session, *, lookup=False):
    public, architecture = session.public_task["public"], session.public_task["architecture"]
    specification = public.get("specification")
    if specification is None:
        return _na("A solver-visible formal specification is required; hidden generating programs are not usable")
    weights = _lookup_circuit(specification, architecture) if lookup else _compile(specification, architecture)
    scalar_count = sum(a.size for a in weights.values())
    if scalar_count > 4096:
        return _na("Capacity-matched construction exceeds the frozen scalar cap")
    return _done({"patch": [{"op": "replace", "tensor": name, "values": value.tolist()}
                              for name, value in sorted(weights.items())]},
                 specification_source="public", persistent_scalars=scalar_count,
                 method="independent piecewise-affine finite-table construction" if lookup else "public specification compiler",
                 lookup_status="compressed bounded circuit; no external table" if lookup else None)


def _lookup_circuit(specification, architecture):
    """Independent LUT compression, using only the public mathematical function.

    Every persistent coefficient lives in the fixed tensors. This uses affine
    interpolation of a 21-entry output table rather than the reference's hat
    circuit; its embeddings and readout also use a different coordinate layout.
    """
    from .programs import readout_values
    if architecture.get("kind") != "attention_pool" or architecture.get("heads") != 1:
        raise ValueError("Independent bounded table constructor supports one attention head")
    if specification.get("op") != "weighted":
        raise ValueError("Unsupported public specification")
    w = {name: np.zeros(shape, dtype=np.float32) for name, shape in architecture["shapes"].items()}
    w["e"][:3, 0] = specification["codes"]
    w["e"][:, 1] = 1
    w["p"][:5, 2] = np.log(specification["positions"])
    w["q"][0, 1] = np.sqrt(12)
    w["k"][0, 2] = 1
    w["v"][0, 0] = 1
    w["o"][0, 0] = sum(specification["positions"])
    table = np.eye(3, dtype=np.float32)[readout_values(specification, np.arange(21))]
    slopes = np.diff(table, axis=0)
    w["b1"][:] = table[0]
    w["w0"][0, 0] = 1
    w["w1"][:, 0] = slopes[0]
    for knot in range(1, 20):
        w["w0"][knot, 0] = 1
        w["b0"][knot] = -knot
        w["w1"][:, knot] = slopes[knot] - slopes[knot-1]
    return w


def _regularity(weights):
    terms = []
    for value in weights.values():
        a = np.asarray(value, dtype=float)
        if a.ndim != 2:
            continue
        row = np.sqrt(np.mean(a * a, axis=1)) + 1e-8
        terms.append(float(np.std(np.log(row))))
        terms.append(float(abs(a.mean()) / (a.std() + 1e-8)))
    return float(np.mean(terms)) if terms else 0.


def _repair_candidates(session, *, informative_weights=True):
    public, arch = session.public_task["public"], session.public_task["architecture"]
    candidates = [("identity", {"patch": []})]
    for hypothesis in public.get("fault_hypotheses", []):
        patch = hypothesis.get("patch", [])
        candidates.append((str(hypothesis["id"]), patch if isinstance(patch, dict) else {"patch": patch}))
    if informative_weights:
        weights = session.view_weights
        # Generic strength normalization, bias reset, output-head reset and an
        # auxiliary-null-space correction are explicit competitors to fault priors.
        matrices = [(k, v) for k, v in weights.items() if v.ndim == 2]
        if matrices:
            name, value = max(matrices, key=lambda kv: _regularity({kv[0]: kv[1]}))
            norms = np.sqrt(np.mean(value.astype(float) ** 2, axis=1))
            row = int(np.argmax(norms))
            factor = float(np.median(norms) / max(norms[row], 1e-8))
            candidates.append(("row_gain_normalization", {"patch": [{"op": "scale", "tensor": name,
                                "axis": 0, "indices": [row], "factor": factor}]}))
        biases = [k for k, v in weights.items() if v.ndim == 1]
        if biases:
            candidates.append(("bias_reset", {"patch": [{"op": "zero", "tensor": biases[-1]}]}))
        if "w1" in weights and weights["w1"].shape[0] >= 6:
            head = weights["w1"][:3].astype(float)
            auxiliary = weights["w1"][3:].astype(float)
            projector = np.eye(head.shape[1]) - np.linalg.pinv(auxiliary) @ auxiliary
            centered = head - head.mean(axis=0, keepdims=True)
            update = (centered - head) @ projector
            candidates.append(("auxiliary_nullspace_head_centering", {"patch": [{"op": "add", "tensor": "w1",
                                "values": np.concatenate((update, np.zeros_like(auxiliary))).tolist()}]}))
            candidates.append(("target_head_reset", {"patch": [{"op": "zero", "tensor": "w1", "axis": 0,
                                "indices": [0, 1, 2]}]}))
    seen, unique = set(), []
    for name, submission in candidates:
        key = canonical_json(submission)
        if key not in seen:
            seen.add(key)
            unique.append((name, submission))
    return unique


def _practice_quality(result, *, normative=None):
    # ``expected`` is public normative feedback, never a hidden evaluation key.
    outputs = result.get("outputs", result.get("predictions", []))
    expected = result.get("expected", result.get("desired_outputs", []))
    if not expected and normative is not None:
        expected = [normative[tuple(row)] for row in result.get("inputs", [])]
    if len(expected) and len(outputs):
        actual, desired = np.asarray(outputs), np.asarray(expected)
        return float(np.mean(actual == desired))
    if "episodes" in result:
        episodes = result["episodes"]
        return float(np.mean([bool(e.get("correct", False)) and bool(e.get("completed", False)) for e in episodes]))
    if "correct" in result:
        correct = np.asarray(result["correct"], dtype=float)
        complete = np.asarray(result.get("completed", np.ones_like(correct)), dtype=float)
        return float(np.mean(correct * complete))
    raise ValueError("Practice response lacks normative scoring observations")


def _generic_repair(session, *, informative_weights):
    candidates = _repair_candidates(session, informative_weights=informative_weights)
    # B cannot execute witnesses. It uses a disclosed initialization regularity
    # prior; this is not a claim that norm regularity identifies every fault.
    if session.condition is Condition.ANALYSIS:
        original = session.view_weights
        chosen = min(candidates, key=lambda item: _regularity(materialize(session.public_task["architecture"], original, item[1])))
        return _done(chosen[1], selected=chosen[0], criterion="weight regularity only; no behavioral witness execution",
                     candidates_considered=len(candidates), informative_weights=informative_weights)
    cap = min(len(candidates), session.task.limits.candidate_patches + 1)
    candidates = candidates[:cap]
    normative_examples = session.public_task["public"].get("normative_examples", [])
    normative = {tuple(e["input"]): e["expected"] for e in normative_examples}
    if not normative_examples or any("practice_index" not in e for e in normative_examples):
        return _na("Finite-fault C selection requires solver-visible normative witness practice indices")
    indices = [e["practice_index"] for e in normative_examples]
    if len(indices) * len(candidates) > session.remaining_observations:
        return _na("Complete predeclared finite-fault witness schedule exceeds the common observation cap")
    chosen, best = candidates[0], -np.inf
    scores = []
    session.commit(chosen[1])
    for name, submission in candidates:
        results = _probe(session, indices, patch=submission)
        if not results:
            break
        quality = float(np.mean([_practice_quality(r, normative=normative) for r in results]))
        scores.append({"candidate": name, "practice_quality": quality})
        if quality > best:
            chosen, best = (name, submission), quality
        session.commit(chosen[1])
    return _done(chosen[1], selected=chosen[0], selection="practice-only fixed schedule", candidates=scores,
                 informative_weights=informative_weights)


def _causal_algebra(session):
    # Quantitative effects of nonlinear interventions cannot in general be
    # inferred from a delta norm. Report the actual linearized circuit baseline
    # explicitly, rather than inventing a symbolic answer for unsupported edits.
    public, weights, arch = session.public_task["public"], session.view_weights, session.public_task["architecture"]
    if not {"w0", "w1", "o", "v", "e"} <= set(weights):
        return _na("Linearized signed-path predictor requires attention_pool")
    paths = weights["w1"] @ weights["w0"] @ weights["o"] @ weights["v"] @ weights["e"].T
    effects = {}
    for edit in public.get("forecast_interventions", []):
        changed = materialize(arch, weights, {"patch": edit["patch"]})
        after = changed["w1"] @ changed["w0"] @ changed["o"] @ changed["v"] @ changed["e"].T
        rates = []
        for head in range(paths.shape[0] // 3):
            a, b = paths[3*head:3*head+3], after[3*head:3*head+3]
            relative = np.linalg.norm(b-a) / max(np.linalg.norm(a), 1e-8)
            rates.append(float(min(1., relative)))
        effects[edit["id"]] = rates
    if not effects:
        return _na("No public final intervention catalog")
    return _done({"effects": effects}, approximation="signed linearized paths; omits ReLU/softmax gating", target_executions=0)


def _causal_behavior(session):
    public = session.public_task["public"]
    practice = public.get("practice_interventions", [])
    if not practice:
        return _na("No disjoint public practice interventions")
    per = max(1, min(public["practice_size"], 128 // (2 * len(practice))))
    observations = []
    for edit in practice:
        results = _probe(session, list(range(per)), intervention_id=edit["id"])
        for result in results:
            if "change_rates" in result:
                observations.append(np.asarray(result["change_rates"], dtype=float))
            elif "baseline_outputs" in result and "outputs" in result:
                observations.append(np.mean(np.asarray(result["outputs"]) != np.asarray(result["baseline_outputs"]), axis=0))
    if not observations:
        return _na("Practice interface does not disclose behavioral intervention effects")
    prior = np.mean(observations, axis=0).tolist()
    return _done({"effects": {edit["id"]: prior for edit in public.get("forecast_interventions", [])}},
                 informative_weights=False, estimator="mean practice-intervention transfer prior")


def _episode_items(results):
    episodes = []
    for result in results:
        if "episodes" in result:
            episodes.extend(result["episodes"])
        elif "generated_tokens" in result:
            generated = result["generated_tokens"]
            if generated and isinstance(generated[0], list):
                activations = result.get("activations", {}).get("hidden", [])
                for i, tokens in enumerate(generated):
                    episodes.append({"generated_tokens": tokens,
                        "correct": result.get("correct", [False]*len(generated))[i],
                        "completed": result.get("completed", [False]*len(generated))[i],
                        "activations": {"hidden": activations[i]} if len(activations) > i else {}})
            else:
                episodes.append(result)
    return episodes


def _loop_repair(session, *, method):
    informative = method != "behavior_only"
    initial = _probe(session, list(range(min(4, session.public_task["public"]["practice_size"]))),
                     observation="activations" if informative else "behavior")
    episodes = _episode_items(initial)
    if not episodes:
        return _na("No recurrent practice episodes returned")
    failures = [e for e in episodes if not e.get("correct") or not e.get("completed")]
    if not failures:
        return _done({"patch": []}, clean_control=True, informative_weights=informative)
    weights = session.view_weights if informative else {}
    arch = session.public_task["architecture"]
    choices = []
    if informative:
        good, bad = [], []
        for episode in episodes:
            hidden = np.asarray(episode.get("activations", {}).get("hidden", []), dtype=float)
            if hidden.ndim == 2 and len(hidden):
                (good if episode.get("correct") and episode.get("completed") else bad).append(hidden)
        if not bad:
            return _na("Activation response lacks failure hidden states")
        negative = np.concatenate(bad)
        positive = np.concatenate(good) if good else np.zeros((1, negative.shape[1]))
        direction = negative.mean(axis=0) - positive.mean(axis=0)
        contrast_direction = direction / max(np.linalg.norm(direction), 1e-12)
        if method == "contrastive_pca":
            centered = negative - positive.mean(axis=0)
            direction = np.linalg.svd(centered, full_matrices=False)[2][0]
        elif method == "random_direction":
            direction = np.random.default_rng(207).normal(size=negative.shape[1])
        norm = float(np.linalg.norm(direction))
        if norm < 1e-10:
            return _done({"patch": []}, no_contrast=True, informative_weights=True)
        direction /= norm
        writers = []
        for name, matrix in weights.items():
            if matrix.ndim == 2 and matrix.shape[0] == len(direction):
                projection = direction @ matrix.astype(float)
                localization = contrast_direction @ matrix.astype(float) if method == "random_direction" else projection
                if method == "random_direction":
                    projection *= np.linalg.norm(localization) / max(np.linalg.norm(projection), 1e-12)
                writers.append((float(np.linalg.norm(localization) / max(np.linalg.norm(matrix), 1e-8)), name, projection))
        for _, name, projection in sorted(writers, reverse=True)[:2]:
            for strength in (.25, .5, 1.):
                choices.append((f"{name}:{strength}", {"patch": [{"op": "rank_one", "tensor": name,
                               "left": direction.tolist(), "right": projection.tolist(), "alpha": -strength}]}))
    else:
        # Architecture-only unit suppression: no values from the disclosed
        # checkpoint enter construction or ordering.
        for name, shape in sorted(arch["shapes"].items()):
            if len(shape) == 2 and shape[0] == shape[1]:
                for unit in range(min(6, shape[0])):
                    choices.append((f"{name}:unit{unit}", {"patch": [{"op": "zero", "tensor": name,
                                   "axis": 0, "indices": [unit]}]}))
                break
    diagnostic_indices = [i for i, e in enumerate(episodes) if not e.get("correct") or not e.get("completed")][:1]
    diagnostic_indices += [i for i in range(len(episodes)) if i not in diagnostic_indices][:2-len(diagnostic_indices)]
    chosen, best = ("identity", {"patch": []}), float(np.mean([episodes[i].get("correct", False) and episodes[i].get("completed", False) for i in diagnostic_indices]))
    scores = []
    for name, submission in choices[:6]:
        if session.remaining_observations < 2:
            break
        results = _probe(session, diagnostic_indices, patch=submission)
        current = _episode_items(results)
        quality = float(np.mean([e.get("correct", False) and e.get("completed", False) for e in current]))
        scores.append({"candidate": name, "practice_quality": quality})
        if quality > best:
            chosen, best = (name, submission), quality
        session.commit(chosen[1])
    return _done(chosen[1], selected=chosen[0], direction_estimator=method,
                 writer_selection="projected writer energy / Frobenius norm; at most two writers",
                 random_norm_matching=method == "random_direction",
                 strengths=[.25, .5, 1.], practice_candidates=scores, informative_weights=informative)


def fit_statistics_reader(tasks) -> StatisticalReader | None:
    """Read labels only from the designated supervised training population."""
    if any(task.split=="baseline_train" and task.architecture.get("kind")=="boolean_relu" for task in tasks):
        from .challenge_baselines import fit_reader
        return fit_reader(tasks)
    population = [task for task in tasks if task.split == "baseline_train" and task.family == "rule_recovery"]
    if not population:
        return None
    examples = []
    for task in population:
        private = json.loads(task.private_path.read_text())
        label = private["label"]
        if isinstance(label, str):
            label = json.loads(label)
        training = private.get("training") or {}
        example = {"split": task.split, "parent_problem_id": task.parent_problem_id,
                   "mechanism_family_id": task.mechanism_family_id,
                   "label": label, "weights": load_weights(task.weights_path)}
        if training.get("population_independence") == "fresh_continuous_features_and_fit":
            if training.get("feature_digest") != weights_digest(example["weights"]):
                raise ValueError("Independent-fit certificate does not match the training checkpoint")
            example["independent_network_id"] = training["independent_network_id"]
        examples.append(example)
    return StatisticalReader().fit(examples, required_labels=population[0].public["program_candidates"])


def _matched_wrong(task, tasks):
    candidates = [other for other in tasks if other.id != task.id and other.family == task.family
                  and other.target_stratum == task.target_stratum
                  and other.architecture == task.architecture
                  and other.parent_problem_id != task.parent_problem_id]
    if not candidates:
        return None
    other = sorted(candidates, key=lambda x: x.id)[0]
    if task.family == "edit_effects":
        return {"target": load_weights(other.weights_path), "after": load_weights(other.weights_path),
                "before": load_weights(other.directory / "before.npz")}
    return load_weights(other.weights_path)


def evaluate_baselines(root: Path, output: Path | None = None) -> dict:
    """Run fixed CPU comparators; publish aggregates and retain raw traces privately.

    Candidate qualification uses development and not-yet-frozen held-out candidate
    sets. These measurements are not official model scores or final admission.
    """
    from .tasks import load_tasks
    from .scoring import score as trusted_score
    from .runner import suite_digest as canonical_suite_digest, harness_digest
    root = Path(root)
    tasks = load_tasks(root, split="all")
    training_started = time.process_time()
    training_failure = None
    try:
        reader = fit_statistics_reader(tasks)
    except ValueError as exc:
        reader = None
        training_failure = str(exc)
    training_seconds = time.process_time() - training_started
    evaluated = [t for t in tasks if t.split != "baseline_train"]
    records = []
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(Path(__file__).parent.glob("*.py")) if not p.name.startswith("._")}
    suite_digest = canonical_suite_digest(root)
    harness_hash = harness_digest()
    protocol = serialization_digest({"baseline_version": "0.2.0-cpu1", "roster": BASELINES,
                                     "harness_digest": harness_hash, "suite_digest": suite_digest})
    for task in evaluated:
        wrong = _matched_wrong(task, evaluated)
        views = ["full", "boundary_only", "masked"] + (["matched_wrong"] if wrong is not None else [])
        for name, definition in BASELINES.items():
            if task.family not in definition["families"]:
                continue
            for condition in definition["conditions"]:
                if condition not in task.conditions:
                    continue
                for view in views:
                    # No-weight methods have exactly the same construction and
                    # starting true checkpoint across views; repeat view controls
                    # explicitly to preserve condition/view-separated reports.
                    session = AccessSession(task, condition, view, wrong_weights=wrong if view == "matched_wrong" else None)
                    start = time.process_time()
                    result = run_baseline(session, name, statistics_reader=reader)
                    if result["status"] == "not_applicable":
                        score = {"success": False, "metrics": {}, "status": "not_applicable"}
                        finished = session.finish()
                    else:
                        session.submit(result["submission"])
                        finished = session.finish()
                        # One hidden evaluation after the fixed practice-based
                        # choice. Its feedback never flows into construction.
                        scored = trusted_score(task, result["submission"])
                        score = scored.to_dict() if hasattr(scored, "to_dict") else scored
                        if finished["audit"]["status"] == "rejected":
                            score.update(success=False, status="protocol_violation")
                    card = access_card(name, training=reader.training if name == "statistical_reader" and reader else None)
                    if task.architecture.get("kind")=="boolean_relu":
                        card["qualifying_families"]=BASELINES[name]["families"]
                        card["method"]={"statistical_reader":"Ridge regression from numeric weight statistics and signed paths to all1024 binary output coordinates",
                            "circuit_algebra":"Exact static affine-gate decoding from the public numeric microcircuit layout",
                            "behavior_only":"Metered practice-only CART and GF2 degree-one-through-three hypothesis selection",
                            "public_spec_constructor":"Independent BDD/Shannon logic compression and fixed-capacity tensor construction",
                            "constrained_lookup":"Independent prime-cover DNF and complement compression with fixed-capacity construction",
                            "anf_constructor":"Independent algebraic-normal-form logic compression with fixed-capacity construction"}.get(name,card["method"])
                        if name=="statistical_reader":
                            card["label_coverage"]="All1024 binary output coordinates; original and edited training behaviors; no finite label-catalog restriction"
                    information = task.public.get("prompt_information", {})
                    records.append({"actor_id": name, "actor_name": name, "actor_type": "baseline",
                        "condition": condition, "weight_view": view, "task_id": task.id,
                        "parent_problem_id": task.parent_problem_id, "mechanism_family_id": task.mechanism_family_id,
                        "family": task.family, "skill": task.skill, "target_stratum": task.target_stratum,
                        "capacity_tier": task.capacity_tier, "split": task.split, "attempt": 1,
                        "protocol_digest": protocol, "target_digest": weights_digest(load_weights(task.weights_path)),
                        "question_digest": serialization_digest({"instruction": task.instruction, "public": task.public}),
                        "prompt_information": information, "access_card": card, **score,
                        "diagnostics": result["diagnostics"], "usage": finished["counters"],
                        "cpu_seconds": time.process_time()-start, "cost_usd": 0., "trace": finished["trace"],
                        "budget_curve": [{"budget": int(level), "actual_observations": finished["snapshot_actual_observations"][level],
                                          "unit": finished["snapshot_unit"], **trusted_score(task, answer).to_dict()}
                                         for level, answer in finished["snapshots"].items()] if condition == "C" and result["status"] != "not_applicable" else [],
                        "audit": finished["audit"], "snapshots": finished["snapshots"]})
    report = build_report(records)
    coverage=[{"task_id":r["task_id"],"target_digest":r["target_digest"],"actor":r["actor_id"],
               "condition":r["condition"],"weight_view":r["weight_view"],"status":r["status"],
               "audit_status":r["audit"]["status"],"qualifying_families":r["access_card"]["qualifying_families"]} for r in records]
    (root/"baseline-coverage.private.json").write_text(json.dumps(coverage,indent=2,allow_nan=False)+"\n")
    report.update({"evaluation_scope": "CPU candidate qualification; not a frozen headline pilot",
                   "coverage_manifest_digest":serialization_digest(coverage),
                   "protocol_digest": protocol, "suite_digest": suite_digest,
                   "harness_digest": harness_hash, "source_hashes": source_hashes,
                   "baseline_roster": {name: access_card(name, training=reader.training if name == "statistical_reader" and reader else None)
                                       for name in BASELINES},
                   "statistical_training_cpu_seconds": training_seconds,
                   "statistical_training": reader.training if reader else None,
                   "statistical_training_failure": training_failure,
                   "paid_calls": 0})
    if reader and reader.training.get("output_coordinates")==1024:
        card=report["baseline_roster"]["statistical_reader"]
        card.update(method="Ridge truth-table regression on numeric statistics and signed paths",
                    qualifying_families=["rule_recovery","edit_effects"],
                    label_coverage="Complete1024-coordinate binary outputs; original and edited training functions; no restricted label catalog")
    private_path = root / "baseline-records.private.json"
    private_path.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
