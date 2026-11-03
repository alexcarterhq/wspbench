"""Small benchmark-owned architectures; submissions contain arrays, never code."""
from __future__ import annotations
from pathlib import Path
from typing import Mapping
import zipfile
import math
import numpy as np


def mlp_architecture(sizes: list[int]) -> dict:
    shapes = {}
    for i, (nin, nout) in enumerate(zip(sizes[:-1], sizes[1:])):
        shapes[f"w{i}"] = [nout, nin]
        shapes[f"b{i}"] = [nout]
    return {"kind": "mlp", "sizes": sizes, "activation": "relu", "shapes": shapes}


def rnn_architecture(hidden: int = 12, length: int = 5) -> dict:
    return {"kind": "rnn", "hidden": hidden, "length": length, "activation": "tanh",
            "shapes": {"rnn.weight_ih": [hidden, 1], "rnn.weight_hh": [hidden, hidden],
                       "rnn.bias": [hidden], "head.weight": [1, hidden], "head.bias": [1]}}


def zeros(architecture: dict) -> dict[str, np.ndarray]:
    return {name: np.zeros(shape, dtype=np.float32) for name, shape in architecture["shapes"].items()}


def validate_weights(architecture: dict, weights: Mapping[str, np.ndarray]) -> None:
    expected = architecture["shapes"]
    if set(weights) != set(expected):
        raise ValueError(f"Tensor names must be exactly {sorted(expected)}")
    total = 0
    for name, shape in expected.items():
        value = np.asarray(weights[name])
        if list(value.shape) != shape:
            raise ValueError(f"Wrong shape for {name}: expected {shape}, got {list(value.shape)}")
        if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError(f"Tensor {name} must contain finite real numbers")
        if np.max(np.abs(value.astype(np.float64, copy=False)), initial=0) > 1e6:
            raise ValueError(f"Tensor {name} exceeds the numerical magnitude limit")
        total += value.size
    if total >= 25_000:
        raise ValueError("The pilot architecture must have fewer than 25,000 parameters")


def forward(architecture: dict, weights: Mapping[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    validate_weights(architecture, weights)
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("Inputs must be finite")
    if architecture["kind"] == "mlp":
        if x.ndim != 2 or x.shape[1] != architecture["sizes"][0]:
            raise ValueError("Input shape does not match the architecture")
        for i in range(len(architecture["sizes"]) - 1):
            x = x @ np.asarray(weights[f"w{i}"], dtype=np.float64).T + weights[f"b{i}"]
            if i < len(architecture["sizes"]) - 2:
                x = np.maximum(x, 0)
        return x
    if architecture["kind"] == "rnn":
        if x.ndim != 2 or x.shape[1] != architecture["length"]:
            raise ValueError("RNN inputs must be batch by sequence length")
        h = np.zeros((len(x), architecture["hidden"]), dtype=np.float64)
        for t in range(x.shape[1]):
            h = np.tanh(x[:, t:t + 1] @ weights["rnn.weight_ih"].T + h @ weights["rnn.weight_hh"].T + weights["rnn.bias"])
        return h @ weights["head.weight"].T + weights["head.bias"]
    raise ValueError(f"Unknown benchmark architecture: {architecture['kind']}")


def load_weights(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if path.stat().st_size > 2_000_000:
        raise ValueError("Tensor artifact exceeds the 2 MB pilot limit")
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) > 16 or sum(m.file_size for m in members) > 2_000_000:
            raise ValueError("Expanded tensor artifact exceeds the pilot limit")
        if len({m.filename for m in members}) != len(members):
            raise ValueError("Tensor artifact has duplicate members")
        for member in members:
            if not member.filename.endswith(".npy") or "/" in member.filename or "\\" in member.filename:
                raise ValueError("Tensor artifact contains unexpected members")
            with archive.open(member) as source:
                version = np.lib.format.read_magic(source)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(source)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(source)
                else:
                    raise ValueError("Unsupported tensor array format")
                if dtype.hasobject or dtype.kind not in "fiu" or math.prod(shape) >= 25_000:
                    raise ValueError("Tensor artifact contains a forbidden dtype or oversized array")
                if source.tell() + math.prod(shape) * dtype.itemsize != member.file_size:
                    raise ValueError("Tensor array length does not match its header")
    with np.load(path, allow_pickle=False) as data:
        result = {name: np.asarray(data[name]).copy() for name in data.files}
    return result


def save_weights(path: str | Path, weights: Mapping[str, np.ndarray]) -> None:
    np.savez(Path(path), **{k: np.asarray(v, dtype=np.float32) for k, v in weights.items()})
