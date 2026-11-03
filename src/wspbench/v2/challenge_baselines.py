"""Public-information controls for the open Boolean-circuit challenge revision.

The supervised reader fits only baseline_train labels. Behavior controls use the
metered practice endpoint. Synthesis uses independent truth-table algorithms and
an independently written ReLU constructor, never the generating program.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from itertools import combinations, product
import json

import numpy as np

from .baselines import weight_features
from .serialization import canonical_json, load_weights


def _domain():
    return np.asarray(list(product((0, 1), repeat=10)), dtype=np.int64)


def _features(weights):
    result = list(weight_features(weights))
    required = {"w0", "w1", "w2", "w3", "out"}
    if required <= set(weights):
        path = np.asarray(weights["out"], float)
        for name in ("w3", "w2", "w1", "w0"):
            path = path @ np.asarray(weights[name], float)
        result.extend(path.ravel())
    return np.asarray(result, dtype=float)


class StructuredReader:
    """Regularized multi-output reader; every truth-table coordinate is binary."""
    def fit(self, examples):
        if len(examples) < 128 or any(e["split"] != "baseline_train" for e in examples):
            raise ValueError("Structured reader requires at least 128 separate baseline_train targets")
        if len({e["independent_network_id"] for e in examples}) != len(examples):
            raise ValueError("Reader population contains repeated networks")
        x = np.stack([_features(e["weights"]) for e in examples])
        y = np.asarray([e["outputs"] for e in examples], dtype=float)
        if y.shape != (len(examples), 1024) or not np.isin(y, (0, 1)).all():
            raise ValueError("Reader labels must cover the complete public binary output representation")
        self.mean, self.scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-6)
        x = (x - self.mean) / self.scale
        # The selection split is fixed before any held-out benchmark labels are read.
        order = np.random.default_rng(905).permutation(len(x))
        cut = max(1, len(order) // 5)
        validation, training = order[:cut], order[cut:]
        target_mean = y[training].mean(axis=0)
        gram = x[training] @ x[training].T
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        projected = eigenvectors.T @ (y[training] - target_mean)
        candidates = []
        for strength in (.1, 1., 10., 100.):
            dual = eigenvectors @ (projected / (np.maximum(eigenvalues, 0)[:, None] + strength))
            coefficients = x[training].T @ dual
            predicted = (x[validation] @ coefficients + target_mean) >= .5
            candidates.append((float(np.mean(predicted != y[validation])), strength))
        self.strength = min(candidates)[1]
        self.intercept = y.mean(axis=0)
        values, vectors = np.linalg.eigh(x @ x.T)
        dual = vectors @ ((vectors.T @ (y - self.intercept)) / (np.maximum(values, 0)[:, None] + self.strength))
        self.coefficients = x.T @ dual
        self.training = {"split": "baseline_train", "networks": len(examples),
            "independent_networks": len(examples),
            "mechanism_clusters": len({e.get("mechanism_family_id", e["independent_network_id"]) for e in examples}),
            "output_coordinates": 1024, "output_classes": [0, 1],
            "feature_count": x.shape[1], "estimator": "ridge truth-table regression on statistics and signed paths",
            "regularization": self.strength, "selection_data": "fixed 20% baseline_train split only",
            "heldout_labels_used": False}
        return self

    def predict(self, weights):
        x = (_features(weights) - self.mean) / self.scale
        return ((x @ self.coefficients + self.intercept) >= .5).astype(int).tolist()


def fit_reader(tasks):
    examples = []
    domain = _domain()
    for task in tasks:
        if task.split != "baseline_train" or task.architecture.get("kind") != "boolean_relu":
            continue
        with np.load(task.directory / "evaluation.npz", allow_pickle=False) as data:
            inputs, outputs = data["inputs"], np.asarray(data["expected"]).reshape(-1)
            if not np.array_equal(inputs, domain):
                raise ValueError("Structured reader labels use a different input ordering")
        examples.append({"split": task.split, "independent_network_id": task.id,
                         "mechanism_family_id": task.mechanism_family_id,
                         "weights": load_weights(task.weights_path), "outputs": outputs.tolist()})
    return StructuredReader().fit(examples)


def _gf2_fit(x, y, degree):
    terms = [()] + [tuple(c) for size in range(1, degree + 1) for c in combinations(range(x.shape[1]), size)]
    design = np.stack([np.prod(x[:, term], axis=1) if term else np.ones(len(x), dtype=int) for term in terms], axis=1).astype(np.uint8)
    augmented = np.column_stack([design, np.asarray(y, np.uint8)])
    pivots, row = [], 0
    for column in range(design.shape[1]):
        found = np.flatnonzero(augmented[row:, column])
        if not len(found):
            continue
        pivot = row + found[0]
        augmented[[row, pivot]] = augmented[[pivot, row]]
        others = np.flatnonzero(augmented[:, column])
        others = others[others != row]
        augmented[others] ^= augmented[row]
        pivots.append(column)
        row += 1
        if row == len(augmented):
            break
    if np.any((augmented[:, :-1].sum(axis=1) == 0) & augmented[:, -1]):
        return None
    coefficients = np.zeros(design.shape[1], np.uint8)
    for i, column in enumerate(pivots):
        coefficients[column] = augmented[i, -1]
    chosen = [term for term, active in zip(terms, coefficients) if active]
    def predict(points):
        result = np.zeros(len(points), dtype=np.uint8)
        for term in chosen:
            result ^= np.prod(points[:, term], axis=1).astype(np.uint8) if term else 1
        return result.astype(int)
    return predict


def _tree_fit(x, y, depth=10):
    majority = int(np.mean(y) > .5)
    if not len(y) or depth == 0 or np.all(y == y[0]):
        return ("leaf", majority)
    best = None
    impurity = float(np.mean(y) * (1 - np.mean(y)))
    for column in range(x.shape[1]):
        left = x[:, column] == 0
        if not left.any() or left.all():
            continue
        risk = sum(float(mask.mean()) * float(np.mean(y[mask]) * (1 - np.mean(y[mask]))) for mask in (left, ~left))
        key = (risk, column)
        if best is None or key < best[0]:
            best = (key, left)
    if best is None:
        return ("leaf", majority)
    # Zero immediate gain is still useful for parity functions; retain the depth bound.
    (_, column), mask = best
    return ("branch", column, _tree_fit(x[mask], y[mask], depth - 1), _tree_fit(x[~mask], y[~mask], depth - 1))


def _tree_predict(tree, points):
    if tree[0] == "leaf":
        return np.full(len(points), tree[1], dtype=int)
    result = np.empty(len(points), dtype=int)
    mask = points[:, tree[1]] == 0
    result[mask] = _tree_predict(tree[2], points[mask])
    result[~mask] = _tree_predict(tree[3], points[~mask])
    return result


def _learn_table(inputs, outputs):
    x, y = np.asarray(inputs, int), np.asarray(outputs, int).reshape(-1)
    if not len(x):
        return [0] * 1024, {"method": "constant", "practice_examples": 0}
    order = np.random.default_rng(128).permutation(len(x))
    size = max(1, len(x) // 4)
    validation, training = order[:size], order[size:]
    if not len(training):
        training = validation
    candidates = []
    for depth in (3, 6, 10):
        tree = _tree_fit(x[training], y[training], depth)
        candidates.append((float(np.mean(_tree_predict(tree, x[validation]) != y[validation])), "tree", depth))
    for degree in (1, 2, 3):
        model = _gf2_fit(x[training], y[training], degree)
        if model is not None:
            candidates.append((float(np.mean(model(x[validation]) != y[validation])), "anf", degree))
    _, kind, parameter = min(candidates)
    model = _gf2_fit(x, y, parameter) if kind == "anf" else None
    if model is None:
        tree = _tree_fit(x, y, parameter if kind == "tree" else 10)
        predicted = _tree_predict(tree, _domain())
    else:
        predicted = model(_domain())
    return predicted.tolist(), {"method": kind, "parameter": parameter, "practice_examples": len(x),
                               "selection": "fixed internal split of metered practice examples"}


def _constant(value):
    return {"op": "constant", "value": int(value)}


def _binary(op, left, right):
    return {"op": op, "left": left, "right": right}


def _balanced(op, nodes):
    if not nodes:
        return _constant(op == "and")
    if len(nodes) == 1:
        return nodes[0]
    mid = len(nodes) // 2
    return _binary(op, _balanced(op, nodes[:mid]), _balanced(op, nodes[mid:]))


def _anf_program(table):
    # Public table order is lexicographic input bits; use the matching binary index.
    coefficients = np.asarray(table, np.uint8).copy()
    for bit in range(10):
        for mask in range(1024):
            if mask & (1 << bit):
                coefficients[mask] ^= coefficients[mask ^ (1 << bit)]
    terms = []
    for mask in np.flatnonzero(coefficients):
        leaves = [{"op": "input", "index": 9 - bit} for bit in range(10) if mask & (1 << bit)]
        terms.append(_balanced("and", leaves))
    return _balanced("xor", terms) if terms else _constant(0)


def _bdd_program(table, order):
    domain = _domain()
    @lru_cache(None)
    def solve(indices, stage):
        values = np.asarray(table, int)[list(indices)]
        if np.all(values == values[0]):
            return _constant(values[0])
        column = order[stage]
        lo_indices = tuple(i for i in indices if domain[i, column] == 0)
        hi_indices = tuple(i for i in indices if domain[i, column] == 1)
        low, high = solve(lo_indices, stage + 1), solve(hi_indices, stage + 1)
        leaf = {"op": "input", "index": int(column)}
        if low == high:
            return low
        if low == _constant(0):
            return _binary("and", leaf, high) if high != _constant(1) else leaf
        if high == _constant(1):
            return _binary("or", leaf, low)
        if low == _constant(1) and high == _constant(0):
            return {"op": "not", "arg": leaf}
        if high == _constant(0):
            return _binary("and", {"op": "not", "arg": leaf}, low)
        if low == _constant(1):
            return _binary("or", {"op": "not", "arg": leaf}, high)
        # Recognize complementary cofactors as a parity factor instead of a mux.
        if np.array_equal(np.asarray(table)[list(lo_indices)], 1 - np.asarray(table)[list(hi_indices)]):
            return _binary("xor", leaf, low)
        return {"op": "mux", "condition": leaf, "then": high, "else": low}
    return solve(tuple(range(1024)), 0)


def _cover_program(table, complement=False):
    desired = np.flatnonzero(np.asarray(table, int) == (0 if complement else 1))
    if not len(desired):
        return _constant(1 if complement else 0)
    cubes = {(1023, int(value)) for value in desired}
    primes = set()
    for _ in range(11):
        combined, following = set(), set()
        for mask, value in cubes:
            for bit in range(10):
                flag = 1 << bit
                if mask & flag and value & flag and (mask, value ^ flag) in cubes:
                    combined.update(((mask, value), (mask, value ^ flag)))
                    reduced = mask ^ flag
                    following.add((reduced, value & reduced))
        primes |= cubes - combined
        if not following:
            break
        cubes = following
        if len(cubes) > 20000:
            raise ValueError("Boolean cover construction exceeded its fixed candidate bound")
    pending = set(int(v) for v in desired)
    chosen = []
    while pending:
        cube = min(primes, key=lambda item: (-sum((v & item[0]) == item[1] for v in pending), item[0].bit_count(), item))
        covered = {v for v in pending if v & cube[0] == cube[1]}
        if not covered:
            raise ValueError("Prime cover failed to cover the public table")
        pending -= covered
        chosen.append(cube)
    clauses = []
    for mask, value in chosen:
        literals = []
        for bit in range(10):
            if mask & (1 << bit):
                leaf = {"op": "input", "index": 9 - bit}
                literals.append(leaf if value & (1 << bit) else {"op": "not", "arg": leaf})
        clauses.append(_balanced("and", literals))
    result = _balanced("or", clauses)
    return {"op": "not", "arg": result} if complement else result


def construct_relu(program, architecture):
    """Independent logical constructor, including explicit fixed-capacity refusal."""
    if architecture.get("kind") != "boolean_relu":
        raise ValueError("Boolean constructor requires the frozen Boolean architecture")
    shapes = architecture["shapes"]
    weights = {name: np.zeros(shape, np.float32) for name, shape in shapes.items()}
    width = 34
    for layer in range(4):
        weights[f"w{layer}"][:10, :10] = np.eye(10)
    assigned = [dict() for _ in range(4)]

    def reference(node, stage):
        size = 10 if stage == 1 else width
        op = node["op"]
        if op == "input":
            vector = np.zeros(size); vector[node["index"]] = 1
            return vector, 0.
        if op == "constant":
            return np.zeros(size), float(node["value"])
        if stage < 1:
            raise ValueError("Boolean expression exceeds the four-stage capacity")
        key = canonical_json(node)
        mapping = assigned[stage - 1]
        if key in mapping:
            start = mapping[key]
        else:
            if len(mapping) >= 8:
                raise ValueError("Boolean expression exceeds eight gate triples in one stage")
            start = 10 + 3 * len(mapping)
            mapping[key] = start
            arguments = ([node["arg"]] if op == "not" else [node["condition"], node["then"], node["else"]] if op == "mux" else node["args"] if op == "majority" else [node["left"], node["right"]])
            refs = [reference(arg, stage - 1) if arg["op"] not in ("input", "constant") else reference(arg, stage) for arg in arguments]
            # Each child lives in the preceding layer; input coordinates are
            # copied unchanged, so a literal is addressable at every stage.
            def affine(coefficients, bias=0):
                vector = sum((c * ref[0] for c, ref in zip(coefficients, refs)), np.zeros(size))
                offset = bias + sum(c * ref[1] for c, ref in zip(coefficients, refs))
                return vector, offset
            zero = (np.zeros(size), 0.)
            if op == "and": forms = [affine([1, 1], -1), zero, zero]
            elif op == "or": forms = [affine([1, 1]), zero, affine([1, 1], -1)]
            elif op == "xor": forms = [affine([1, 1]), zero, affine([2, 2], -2)]
            elif op == "not": forms = [affine([-1], 1), zero, zero]
            elif op == "nand": forms = [(np.zeros(size), 1.), zero, affine([1, 1], -1)]
            elif op == "nor": forms = [affine([-1, -1], 1), zero, zero]
            elif op == "mux": forms = [affine([1, 1, 0], -1), affine([-1, 0, 1]), zero]
            elif op == "majority": forms = [affine([1, 1, 1], -1), zero, affine([1, 1, 1], -2)]
            else: raise ValueError("Unknown independent Boolean constructor operation")
            for offset, (vector, bias) in enumerate(forms):
                weights[f"w{stage - 1}"][start + offset] = vector
                weights[f"b{stage - 1}"][start + offset] = bias
        vector = np.zeros(width); vector[start:start + 3] = [1, 1, -1]
        return vector, 0.

    vector, bias = reference(program, 4)
    weights["out"][0] = -vector; weights["bias"][0] = 1 - bias
    weights["out"][1] = vector; weights["bias"][1] = bias
    weights["bias"][2] = -1
    if sum(value.size for value in weights.values()) > 4096:
        raise ValueError("Constructed checkpoint exceeds the public scalar budget")
    return weights


def _synthesize(public, architecture, method):
    specification = public.get("truth_table", public.get("specification"))
    if not isinstance(specification, dict):
        raise ValueError("Synthesis requires the complete public truth table")
    inputs = np.asarray(specification["inputs"], int)
    table = np.asarray(specification["outputs"], int).reshape(-1)
    if not np.array_equal(inputs, _domain()) or table.shape != (1024,) or not np.isin(table, (0, 1)).all():
        raise ValueError("Malformed public truth-table specification")
    if method == "anf_constructor":
        programs = [_anf_program(table)]
    elif method == "dnf_constructor":
        programs = [_cover_program(table), _cover_program(table, True)]
    else:
        influence = [np.mean(table != table[np.arange(1024) ^ (1 << (9 - bit))]) for bit in range(10)]
        orders = [list(range(10)), list(reversed(range(10))), list(np.argsort(influence)), list(np.argsort(influence)[::-1])]
        orders += [list(range(start, 10)) + list(range(start)) for start in range(1, 10)]
        programs = [_bdd_program(table, order) for order in orders]
    failures = []
    for program in programs:
        try:
            weights = construct_relu(program, architecture)
            return {"status": "completed", "submission": {"patch": [
                {"op": "replace", "tensor": name, "values": value.tolist()} for name, value in sorted(weights.items())]},
                "diagnostics": {"method": method, "source": "complete public truth table", "candidate_expressions": len(programs),
                                "persistent_scalars": sum(a.size for a in weights.values()), "private_program_used": False}}
        except ValueError as error:
            failures.append(str(error))
    # A representability refusal is a valid zero-credit attempted construction,
    # not a missing baseline or an invalid architecture-changing artifact.
    return {"status": "completed", "submission": {"patch": []},
            "diagnostics": {"method": method, "capacity_refusal": True,
                            "candidate_expressions": len(programs), "reasons": sorted(set(failures))}}


def run(session, name, reader=None):
    task = session.public_task
    if task["architecture"].get("kind") != "boolean_relu":
        return {"status": "not_applicable", "submission": {}, "diagnostics": {"reason": "not a Boolean challenge"}}
    family = task["family"]
    if name == "structured_reader":
        if reader is None:
            raise ValueError("Balanced independent reader population has not been fitted")
        if family == "rule_recovery":
            answer = {"outputs": reader.predict(session.view_weights)}
        elif family == "edit_effects":
            answer = {"before_outputs": reader.predict(session.checkpoint_weights("before")),
                      "after_outputs": reader.predict(session.checkpoint_weights("after"))}
        else:
            return {"status": "not_applicable", "submission": {}, "diagnostics": {"reason": "reader task type"}}
        return {"status": "completed", "submission": answer, "diagnostics": {"training": reader.training}}
    if name in {"bdd_constructor", "anf_constructor", "dnf_constructor"}:
        return _synthesize(task["public"], task["architecture"], name)
    if name != "boolean_behavior" or session.condition.value != "C":
        return {"status": "not_applicable", "submission": {}, "diagnostics": {"reason": "unknown method or access"}}
    paired = family == "edit_effects"
    checkpoints = ("before", "after") if paired else ("target",)
    collected = {key: ([], []) for key in checkpoints}
    answer = {"before_outputs": [0] * 1024, "after_outputs": [0] * 1024} if paired else {"outputs": [0] * 1024}
    session.commit(answer)
    maximum = 64 if paired else min(128, task["public"]["practice_size"])
    positions = {key: 0 for key in checkpoints}
    diagnostics = {}
    while session.remaining_observations:
        advanced = False
        for checkpoint in checkpoints:
            start = positions[checkpoint]
            size = min(16, maximum - start, session.next_boundary - session.observation_count,
                       session.remaining_observations)
            if size <= 0:
                continue
            result = session.probe(list(range(start, start + size)), checkpoint=checkpoint)
            positions[checkpoint] += size
            inputs, outputs = collected[checkpoint]
            inputs.extend(result["inputs"])
            outputs.extend(np.asarray(result["outputs"]).reshape(-1).tolist())
            table, diagnostic = _learn_table(inputs, outputs)
            answer[checkpoint + "_outputs" if paired else "outputs"] = table
            diagnostics[checkpoint] = diagnostic
            session.commit(answer)
            advanced = True
        if not advanced:
            break
    return {"status": "completed", "submission": answer,
            "diagnostics": {"informative_weights": False, "practice_only": True, "checkpoints": diagnostics}}
