"""Canonical, round-trip float32 serialization shared by every access condition."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import zipfile

import numpy as np

MAX_PARAMETERS = 4096
MAX_BYTES = 1_000_000


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def serialization_digest(value) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _array(value) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or array.ndim > 4 or array.size > MAX_PARAMETERS:
        raise ValueError("Weights must be bounded real numeric arrays")
    if not np.isfinite(array).all() or np.max(np.abs(array.astype(np.float64)), initial=0) > 1e6:
        raise ValueError("Weights must be finite with magnitude at most 1e6")
    return np.asarray(array, dtype=np.float32)


def encode_weights(weights: dict[str, np.ndarray]) -> dict:
    if not isinstance(weights, dict) or not weights or len(weights) > 64:
        raise ValueError("Invalid weight mapping")
    tensors = {}
    count = 0
    for name, value in sorted(weights.items()):
        if not isinstance(name, str) or not name or any(ord(c) < 32 for c in name):
            raise ValueError("Invalid tensor name")
        array = _array(value)
        count += array.size
        values = [float(format(float(x), ".9g")) for x in array.reshape(-1)]
        restored = np.asarray(values, dtype=np.float32).reshape(array.shape)
        if not np.array_equal(array.view(np.uint32), restored.view(np.uint32)):
            raise ValueError("Weight serialization is not bit-exact")
        tensors[name] = {"shape": list(array.shape), "values": values}
    if count > MAX_PARAMETERS:
        raise ValueError("Checkpoint exceeds 4096 scalar values")
    return {"dtype": "float32", "tensors": tensors}


def decode_weights(payload: dict) -> dict[str, np.ndarray]:
    if not isinstance(payload, dict) or set(payload) != {"dtype", "tensors"} or payload["dtype"] != "float32":
        raise ValueError("Invalid serialized checkpoint")
    tensors = payload["tensors"]
    if not isinstance(tensors, dict) or not tensors or len(tensors) > 64:
        raise ValueError("Invalid serialized tensors")
    result = {}
    count = 0
    for name, entry in tensors.items():
        if not isinstance(name, str) or not isinstance(entry, dict) or set(entry) != {"shape", "values"}:
            raise ValueError("Malformed tensor entry")
        shape = entry["shape"]
        if not isinstance(shape, list) or len(shape) > 4 or any(type(x) is not int or x < 0 for x in shape):
            raise ValueError("Invalid tensor shape")
        size = math.prod(shape)
        values = entry["values"]
        if size > MAX_PARAMETERS or not isinstance(values, list) or len(values) != size:
            raise ValueError("Tensor values do not match shape")
        count += size
        if count > MAX_PARAMETERS:
            raise ValueError("Checkpoint exceeds capacity")
        result[name] = _array(values).reshape(shape)
    encode_weights(result)
    return result


def validate_weights(architecture: dict, weights: dict[str, np.ndarray]) -> None:
    shapes = architecture.get("shapes")
    if not isinstance(shapes, dict) or set(shapes) != set(weights):
        raise ValueError("Tensor names must match the fixed architecture")
    for name, shape in shapes.items():
        if list(np.asarray(weights[name]).shape) != list(shape):
            raise ValueError(f"Wrong tensor shape: {name}")
    encode_weights(weights)


def save_weights(path: Path, weights: dict[str, np.ndarray]) -> None:
    encoded = encode_weights(weights)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as output:
        np.savez_compressed(output, **decode_weights(encoded))
    temporary.replace(path)


def load_weights(path: Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Checkpoint archive too large")
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if not entries or len(entries) > 64 or sum(x.file_size for x in entries) > MAX_BYTES:
            raise ValueError("Expanded checkpoint exceeds size limit")
        if len({x.filename for x in entries}) != len(entries):
            raise ValueError("Duplicate checkpoint members")
        count = 0
        for entry in entries:
            if not entry.filename.endswith(".npy") or "/" in entry.filename or "\\" in entry.filename:
                raise ValueError("Unexpected archive member")
            with archive.open(entry) as member:
                version = np.lib.format.read_magic(member)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(member)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(member)
                else:
                    raise ValueError("Unsupported array version")
                count += math.prod(shape)
                if dtype.hasobject or dtype.kind not in "fiu" or count > MAX_PARAMETERS:
                    raise ValueError("Forbidden dtype or oversized checkpoint")
                if member.tell() + math.prod(shape) * dtype.itemsize != entry.file_size:
                    raise ValueError("Array length does not match header")
    with np.load(path, allow_pickle=False) as data:
        result = {name: data[name].copy() for name in data.files}
    return decode_weights(encode_weights(result))


def weights_digest(weights: dict[str, np.ndarray]) -> str:
    return serialization_digest(encode_weights(weights))
