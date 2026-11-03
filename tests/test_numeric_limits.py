import numpy as np
import pytest

from wspbench.networks import mlp_architecture, validate_weights, zeros


def test_extreme_signed_integer_cannot_wrap_past_artifact_magnitude_limit():
    architecture = mlp_architecture([1, 2, 1])
    weights = zeros(architecture)
    weights['w0'] = np.full((2, 1), np.iinfo(np.int64).min, dtype=np.int64)
    with pytest.raises(ValueError, match='magnitude limit'):
        validate_weights(architecture, weights)
