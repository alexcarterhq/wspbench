"""Bounded, declarative sequence-rule grammar; never called by target inference."""
from __future__ import annotations

from itertools import product
import json
import numpy as np

PAD = 3
MAX_LENGTH = 5

def domain_inputs() -> np.ndarray:
    return np.asarray([list(x) + [PAD] * (MAX_LENGTH-len(x))
                       for n in range(1, MAX_LENGTH+1) for x in product(range(3), repeat=n)], dtype=np.int64)


def _readout(node: dict, values: np.ndarray, depth: int = 0) -> np.ndarray:
    if not isinstance(node, dict) or depth > 3:
        raise ValueError("Invalid or overly deep readout")
    if node.get("op") == "mod" and set(node) == {"op", "modulus", "offset"}:
        if node["modulus"] != 3 or type(node["offset"]) is not int or not 0 <= node["offset"] <= 2:
            raise ValueError("Invalid modular readout")
        return (values + node["offset"]) % 3
    if node.get("op") == "threshold" and set(node) == {"op", "cuts"}:
        cuts = node["cuts"]
        if not isinstance(cuts, list) or len(cuts) != 2 or any(type(v) is not int for v in cuts) or not 0 <= cuts[0] < cuts[1] <= 21:
            raise ValueError("Invalid threshold readout")
        return (values >= cuts[0]).astype(int) + (values >= cuts[1]).astype(int)
    if node.get("op") == "conditional" and set(node) == {"op", "threshold", "then", "else"}:
        if type(node["threshold"]) is not int or not 0 <= node["threshold"] <= 20:
            raise ValueError("Invalid conditional threshold")
        left = _readout(node["then"], values, depth+1)
        right = _readout(node["else"], values, depth+1)
        return np.where(values >= node["threshold"], left, right)
    raise ValueError("Unknown readout grammar")


def validate_program(program: dict) -> None:
    if not isinstance(program, dict) or set(program) != {"op", "codes", "positions", "readout"} or program["op"] != "weighted":
        raise ValueError("Invalid sequence program")
    if not isinstance(program["codes"], list) or any(type(v) is not int for v in program["codes"]) or sorted(program["codes"]) != [0, 1, 2]:
        raise ValueError("Codes must permute the three symbols")
    if not isinstance(program["positions"], list) or len(program["positions"]) != 5 or any(type(v) is not int or v not in (1, 2) for v in program["positions"]):
        raise ValueError("Invalid position weights")
    _readout(program["readout"], np.arange(21))


def evaluate_program(program: dict, inputs) -> np.ndarray:
    validate_program(program)
    inputs = np.asarray(inputs)
    if inputs.ndim != 2 or inputs.shape[1] != MAX_LENGTH or inputs.dtype.kind not in "iu" or np.any((inputs < 0) | (inputs > PAD)):
        raise ValueError("Inputs must be padded symbol sequences")
    codes = np.asarray(program["codes"] + [0], dtype=np.int64)
    values = (codes[inputs] * np.asarray(program["positions"])).sum(axis=1)
    return _readout(program["readout"], values).astype(np.int64)


def readout_values(program: dict, values) -> np.ndarray:
    validate_program(program)
    return _readout(program["readout"], np.asarray(values, dtype=np.int64)).astype(np.int64)


def canonical(program: dict) -> str:
    validate_program(program)
    return json.dumps(program, sort_keys=True, separators=(",", ":"))


def candidates(split: str | None = None) -> list[dict]:
    """Four dev structures and eight held-out nested compositions, fixed before scoring."""
    mods = [{"op": "mod", "modulus": 3, "offset": i} for i in range(3)]
    thresholds = [{"op": "threshold", "cuts": c} for c in ([3, 7], [4, 9], [5, 11])]
    programs = []
    for i in range(12):
        if i < 4:
            readout = mods[i % 3] if i < 3 else thresholds[0]
        else:
            readout = {"op": "conditional", "threshold": 4 + i % 4,
                       "then": mods[i % 3], "else": thresholds[i % 3]}
            if i >= 8:
                readout["then"] = {"op": "conditional", "threshold": 9+i%3,
                                   "then": thresholds[(i+1)%3], "else": mods[(i+1)%3]}
        codes = ([0,1,2], [1,2,0], [2,0,1], [2,1,0])[i % 4]
        positions = ([1,1,1,1,1], [1,2,1,2,1], [2,1,2,1,2])[i % 3]
        programs.append({"op": "weighted", "codes": list(codes), "positions": list(positions), "readout": readout})
    if split in ("dev", "baseline_train"):
        return programs[:4]
    if split == "heldout":
        return programs[4:]
    if split is not None and split != "all":
        raise ValueError("Unknown program split")
    return programs
