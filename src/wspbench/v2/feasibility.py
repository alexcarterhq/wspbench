"""Static, public-information feasibility witnesses for the scalar pilot targets.

This is a conventional symbolic comparator, not an unaided model certificate.
It compiles weights into affine regions and integrates a product measure. It
does not call the network runtime, read a task directory or query an evaluator.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy

import numpy as np

from .patches import materialize
from .serialization import canonical_json


def translation_closed_catalog(base_catalog: list[dict]) -> list[dict]:
    """Close the declared pilot grammar under its public unit edit operation."""
    def shifted(node):
        node = deepcopy(node)
        if node["op"] == "mod":
            node["offset"] = (node["offset"] + 1) % 3
        elif node["op"] == "threshold":
            node["cuts"] = [v - 1 for v in node["cuts"]]
        elif node["op"] == "conditional":
            node["threshold"] -= 1
            node["then"], node["else"] = shifted(node["then"]), shifted(node["else"])
        else:
            raise ValueError("Unsupported declared edit grammar")
        return node
    result = {canonical_json(p): deepcopy(p) for p in base_catalog}
    for program in base_catalog:
        translated = deepcopy(program)
        translated["readout"] = shifted(translated["readout"])
        result[canonical_json(translated)] = translated
    return [result[key] for key in sorted(result)]


def _scalar_circuit(weights: dict, architecture: dict) -> dict:
    if architecture.get("kind") != "attention_pool":
        raise ValueError("Symbolic scalar reference supports attention_pool only")
    w = {key: np.asarray(value, dtype=np.float64) for key, value in weights.items()}
    # Prove attention is independent of the symbol; only positions determine its measure.
    query = (w["e"][4] + w["p"][5]) @ w["q"].T
    token_scores = w["e"][:4] @ w["k"].T @ query / np.sqrt(12)
    if not np.allclose(token_scores, token_scores[0], rtol=0, atol=1e-8):
        raise ValueError("Token-dependent attention is outside the certified scalar subclass")
    positions = w["p"][:5] @ w["k"].T @ query / np.sqrt(12)
    attention = np.exp(positions - positions.max())
    attention /= attention.sum()
    tokens = w["e"][:4] @ w["v"].T @ w["o"].T
    positional = w["p"][:5] @ w["v"].T @ w["o"].T
    if not np.allclose(tokens[:, 1:], 0, rtol=0, atol=1e-8) or not np.allclose(positional[:, 1:], 0, rtol=0, atol=1e-8):
        raise ValueError("Pooled representation is not a certified one-dimensional circuit")
    if not np.allclose(w["w0"][:, 1:], 0, rtol=0, atol=1e-8):
        raise ValueError("Readout depends on undecompiled channels")
    return {"token_terms": attention[:, None] * tokens[None, :, 0],
            "offset": float(attention @ positional[:, 0]),
            "a": w["w0"][:, 0], "b": w["b0"],
            "out": w["w1"], "bias": w["b1"], "heads": architecture["heads"]}


def _regions(circuit: dict) -> list[dict]:
    """Compile every ReLU branch and every class crossing into affine cells."""
    a, b = circuit["a"], circuit["b"]
    knots = sorted(set(float(-offset / slope) for slope, offset in zip(a, b) if slope))
    bounds = [-np.inf, *knots, np.inf]
    cells = []
    for left, right in zip(bounds[:-1], bounds[1:]):
        witness = right - max(1., abs(right)) if np.isneginf(left) else left + max(1., abs(left)) if np.isposinf(right) else (left + right) / 2
        active = a * witness + b > 0
        slopes = circuit["out"] @ (a * active)
        intercepts = circuit["out"] @ (b * active) + circuit["bias"]
        crossings = []
        for head in range(circuit["heads"]):
            for i in range(3):
                for j in range(i + 1, 3):
                    first, second = 3 * head + i, 3 * head + j
                    delta = slopes[first] - slopes[second]
                    if delta:
                        crossing = (intercepts[second] - intercepts[first]) / delta
                        if left < crossing < right:
                            crossings.append(float(crossing))
        cuts = [left, *sorted(set(crossings)), right]
        cells.extend({"left": start, "right": end, "slopes": slopes, "intercepts": intercepts}
                     for start, end in zip(cuts[:-1], cuts[1:]))
    return cells


def _region_class(cells: list[dict], value: float, heads: int) -> tuple[int, ...]:
    # Adjacent affine cells agree at a true ReLU boundary. At a class tie the
    # declared argmax convention chooses the first class, exactly as the runtime.
    cell = next(cell for cell in cells if cell["left"] <= value <= cell["right"])
    logits = cell["slopes"] * value + cell["intercepts"]
    return tuple(logits.reshape(heads, 3).argmax(axis=1).tolist())


def _joint_measure(first: dict, second: dict) -> dict[tuple[float, float], int]:
    """Multiply symbol-choice polynomials; retain masses, never concrete inputs."""
    result = defaultdict(int)
    for length in range(1, 6):
        padding_first = float(first["token_terms"][length:, 3].sum())
        padding_second = float(second["token_terms"][length:, 3].sum())
        polynomial = {(first["offset"] + padding_first, second["offset"] + padding_second): 1}
        for position in range(length):
            product = defaultdict(int)
            for (x, y), multiplicity in polynomial.items():
                for symbol in range(3):
                    key = (x + float(first["token_terms"][position, symbol]),
                           y + float(second["token_terms"][position, symbol]))
                    product[key] += multiplicity
            polynomial = product
        for atom, multiplicity in polynomial.items():
            result[atom] += multiplicity
    if sum(result.values()) != 363:
        raise RuntimeError("Symbol-choice measure lost mass")
    return dict(result)


def symbolic_effects(weights: dict, architecture: dict, interventions: list[dict]) -> dict:
    """Derive exact public-domain effect fractions from symbolic circuit regions.

    All computation consumes only solver-visible weights, architecture and
    intervention definitions. Floating-point ties are checked by the ordinary
    validator against the trusted runtime; this function never imports it.
    """
    original = _scalar_circuit(weights, architecture)
    before_regions = _regions(original)
    effects, certificates = {}, {}
    for intervention in interventions:
        edited = materialize(architecture, weights, {"patch": intervention["patch"]})
        after = _scalar_circuit(edited, architecture)
        after_regions = _regions(after)
        mass = _joint_measure(original, after)
        changed = np.zeros(architecture["heads"], dtype=np.int64)
        for (before_value, after_value), multiplicity in mass.items():
            a = _region_class(before_regions, before_value, architecture["heads"])
            b = _region_class(after_regions, after_value, architecture["heads"])
            changed += multiplicity * (np.asarray(a) != np.asarray(b))
        effects[intervention["id"]] = (changed / 363).tolist()
        certificates[intervention["id"]] = {"mass": 363, "joint_atoms": len(mass),
            "before_regions": len(before_regions), "after_regions": len(after_regions)}
    return {"effects": effects, "certificate": {"method": "static_piecewise_affine_product_measure",
        "target_executions": 0, "concrete_inputs_enumerated": 0, "condition": "B",
        "compute_surface": "conventional NumPy symbolic algorithm, not an unaided or twelve-call scaffold certificate",
        "interventions": certificates}}
