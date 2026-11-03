import io
import zipfile

import numpy as np
import pytest

from wspbench.v2.patches import materialize
from wspbench.v2.serialization import decode_weights, encode_weights, load_weights, save_weights


def test_float32_bytes_survive_all_public_serialization(tmp_path):
    values = np.array([0., -0., np.nextafter(np.float32(0), np.float32(1)),
                       1 / 3, -123.456], dtype=np.float32)
    original = {"matrix": values.reshape(1, -1)}
    restored = decode_weights(encode_weights(original))
    assert np.array_equal(restored["matrix"].view(np.uint32), original["matrix"].view(np.uint32))
    path = tmp_path / "weights.npz"
    save_weights(path, original)
    assert np.array_equal(load_weights(path)["matrix"].view(np.uint32), values.reshape(1, -1).view(np.uint32))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "3", True])
def test_malformed_numeric_patches_reject_before_mutation(bad):
    weights = {"body": np.arange(4, dtype=np.float32).reshape(2, 2)}
    before = weights["body"].copy()
    with pytest.raises(ValueError):
        materialize({"shapes": {"body": [2, 2]}}, weights,
                    [{"op": "set", "tensor": "body", "index": [0, 0], "value": bad}])
    assert np.array_equal(weights["body"], before)


@pytest.mark.parametrize("indices", [[[0]], [True], [-1], [2], [0, 0]])
def test_invalid_axis_selections_are_validation_errors(indices):
    with pytest.raises(ValueError):
        materialize({"shapes": {"body": [2, 2]}}, {"body": np.eye(2, dtype=np.float32)},
                    [{"op": "zero", "tensor": "body", "axis": 0, "indices": indices}])


def test_pickle_and_oversized_declared_array_are_rejected(tmp_path):
    unsafe = tmp_path / "object.npz"
    np.savez(unsafe, value=np.array([{"execute": "no"}], dtype=object))
    with pytest.raises(ValueError, match="dtype"):
        load_weights(unsafe)
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {"descr": "<f4", "fortran_order": False, "shape": (10**12,)})
    oversized = tmp_path / "oversized.npz"
    with zipfile.ZipFile(oversized, "w") as archive:
        archive.writestr("value.npy", header.getvalue())
    with pytest.raises(ValueError, match="oversized"):
        load_weights(oversized)
