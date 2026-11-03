"""Numeric-only patches preserve untouched values on the true checkpoint."""

from __future__ import annotations

import numpy as np

from .serialization import validate_weights


def _number(value) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value):
        raise ValueError("Patch values must be finite numbers")
    if abs(float(value)) > 1e6:
        raise ValueError("Patch scalar exceeds magnitude limit")
    return float(value)


def _indices(indices, size: int) -> list[int]:
    if not isinstance(indices, list) or not indices or any(type(i) is not int or i < 0 or i >= size for i in indices):
        raise ValueError("Patch index outside tensor")
    if len(set(indices)) != len(indices):
        raise ValueError("Invalid or duplicate patch indices")
    return indices


def materialize(architecture: dict, starting_weights: dict[str, np.ndarray], patch) -> dict[str, np.ndarray]:
    validate_weights(architecture, starting_weights)
    if isinstance(patch, dict):
        if set(patch) != {"patch"}:
            raise ValueError("A patch submission must contain only the patch field")
        patch = patch["patch"]
    if not isinstance(patch, list) or len(patch) > 8192:
        raise ValueError("A numeric patch must be an ordered list of at most 8192 operations")
    result = {name: np.asarray(value, dtype=np.float32).copy() for name, value in starting_weights.items()}
    for operation in patch:
        if not isinstance(operation, dict) or operation.get("tensor") not in result:
            raise ValueError("Patch names an unknown tensor")
        name = operation["tensor"]
        array = result[name]
        kind = operation.get("op")
        if kind == "set":
            if set(operation) != {"op", "tensor", "index", "value"}:
                raise ValueError("Invalid set operation")
            index = operation["index"]
            if not isinstance(index, list) or len(index) != array.ndim:
                raise ValueError("Patch index rank mismatch")
            for i, size in zip(index, array.shape):
                _indices([i], size)
            array[tuple(index)] = _number(operation["value"])
        elif kind in {"replace", "add"}:
            if set(operation) != {"op", "tensor", "values"}:
                raise ValueError("Invalid array operation")
            value = np.asarray(operation["values"])
            if value.shape != array.shape or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
                raise ValueError("Patch array must be finite numeric and match shape")
            if np.max(np.abs(value.astype(np.float64)), initial=0) > 1e6:
                raise ValueError("Patch array exceeds magnitude limit")
            result[name] = value.astype(np.float32) if kind == "replace" else array + value.astype(np.float32)
        elif kind in {"scale", "zero"}:
            allowed = {"op", "tensor", "factor", "axis", "indices"} if kind == "scale" else {"op", "tensor", "axis", "indices"}
            if set(operation) - allowed or (kind == "scale" and "factor" not in operation):
                raise ValueError("Invalid scaling operation")
            factor = _number(operation["factor"]) if kind == "scale" else 0.0
            if "axis" in operation or "indices" in operation:
                axis = operation.get("axis")
                if type(axis) is not int or axis < 0 or axis >= array.ndim:
                    raise ValueError("Invalid patch axis")
                indices = _indices(operation.get("indices"), array.shape[axis])
                selection = [slice(None)] * array.ndim
                selection[axis] = indices
                array[tuple(selection)] *= factor
            else:
                array *= factor
        elif kind == "rank_one":
            if set(operation) - {"op", "tensor", "left", "right", "alpha", "columns"} or not {"left", "right", "alpha"} <= set(operation):
                raise ValueError("Invalid rank-one operation")
            left, right = np.asarray(operation["left"]), np.asarray(operation["right"])
            if array.ndim != 2 or left.shape != (array.shape[0],) or right.shape != (array.shape[1],):
                raise ValueError("Rank-one vectors do not match matrix")
            if any(v.dtype.kind not in "fiu" or not np.isfinite(v).all() or np.max(np.abs(v.astype(np.float64)), initial=0) > 1e6 for v in (left, right)):
                raise ValueError("Rank-one vectors must be bounded finite numbers")
            right = right.astype(np.float64)
            if "columns" in operation:
                columns = _indices(operation["columns"], array.shape[1])
                masked = np.zeros_like(right)
                masked[columns] = right[columns]
                right = masked
            result[name] = (array.astype(np.float64) + _number(operation["alpha"]) * np.outer(left, right)).astype(np.float32)
        else:
            raise ValueError("Unknown numeric patch operation")
        validate_weights(architecture, result)
    return result


def diff_patch(original: dict[str, np.ndarray], edited: dict[str, np.ndarray]) -> dict:
    if set(original) != set(edited):
        raise ValueError("Edited checkpoint changed tensor keys")
    operations = []
    for name in sorted(original):
        before, after = np.asarray(original[name], dtype=np.float32), np.asarray(edited[name], dtype=np.float32)
        if before.shape != after.shape:
            raise ValueError("Edited checkpoint changed tensor shape")
        for index in np.argwhere(before != after):
            coordinate = tuple(index)
            operations.append({"op": "set", "tensor": name, "index": index.tolist(), "value": float(after[coordinate])})
    return {"patch": operations}
