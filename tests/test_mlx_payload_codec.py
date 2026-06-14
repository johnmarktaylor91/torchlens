"""Private MLX payload codec tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from torchlens._io.payload_codec import get_payload_codec
from torchlens._io.tensor_policy import FailReason, Ok
from torchlens.backends import BackendRuntimeCompatibilityError

mlx = pytest.importorskip("mlx")
import mlx.core as mx  # noqa: E402


pytestmark = pytest.mark.optional


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32),
            np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        ),
        (mx.array([[1, 2], [3, 4]], dtype=mx.int32), np.asarray([[1, 2], [3, 4]])),
        (
            mx.array([[True, False], [False, True]], dtype=mx.bool_),
            np.asarray([[True, False], [False, True]]),
        ),
    ],
)
def test_mlx_payload_codec_round_trips_arrays(value: Any, expected: np.ndarray) -> None:
    """MLX codec should round-trip dense numeric and bool arrays through NumPy."""

    codec = get_payload_codec("mlx")

    assert codec.can_encode(value)
    assert isinstance(codec.validate_for_save(value, strict=True), Ok)
    encoded = codec.to_numpy(value)
    np.testing.assert_array_equal(encoded.array, expected)
    assert encoded.logical_dtype == str(value.dtype)
    assert encoded.logical_device
    assert encoded.codec_metadata is not None
    assert encoded.codec_metadata["logical_shape"] == [int(dim) for dim in expected.shape]

    restored = codec.from_numpy(
        encoded.array,
        {"logical_dtype": encoded.logical_dtype, "logical_device": encoded.logical_device},
        map_location=None,
        strict_runtime=True,
    )
    mx.eval(restored)

    assert isinstance(restored, mx.array)
    assert str(restored.dtype) == str(value.dtype)
    np.testing.assert_array_equal(np.asarray(restored), expected)


def test_mlx_payload_codec_manifest_fields_are_registered() -> None:
    """MLX codec registration should expose private manifest codec metadata."""

    codec = get_payload_codec("mlx")
    value = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)
    encoded = codec.to_numpy(value)

    fields = codec.manifest_fields(value, encoded)

    assert codec.backend_name == "mlx"
    assert fields["logical_backend"] == "mlx"
    assert fields["codec"] == "numpy_safetensors_v1"
    assert fields["logical_dtype"] == "mlx.core.float32"
    assert fields["logical_device"] == encoded.logical_device
    assert fields["transport_backend"] == "safetensors.torch"
    assert fields["transport_dtype"] == "float32"
    assert isinstance(fields["codec_metadata"], dict)


def test_mlx_payload_codec_rejects_unsupported_dtype() -> None:
    """MLX codec validation should fail closed for unsupported object-like dtypes."""

    fake_array = type(
        "array",
        (),
        {"__module__": "mlx.core", "dtype": "object"},
    )()

    decision = get_payload_codec("mlx").validate_for_save(fake_array, strict=True)

    assert isinstance(decision, FailReason)
    assert "object" in decision.text


def test_mlx_payload_codec_rejects_unresolved_load_dtype() -> None:
    """MLX load should fail closed for unresolvable dtype strings."""

    codec = get_payload_codec("mlx")

    with pytest.raises(BackendRuntimeCompatibilityError, match="not-a-real-dtype"):
        codec.from_numpy(
            np.asarray([1.0], dtype=np.float32),
            {"logical_dtype": "not-a-real-dtype", "logical_device": "unknown"},
            map_location=None,
            strict_runtime=True,
        )


def test_mlx_payload_codec_runtime_missing_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing MLX runtime during decode should raise a targeted compatibility error."""

    def _missing_mlx_runtime() -> Any:
        """Pretend the MLX runtime is unavailable during payload decode."""

        raise BackendRuntimeCompatibilityError("mlx runtime unavailable")

    monkeypatch.setattr("torchlens._io.payload_codec._import_mlx_core", _missing_mlx_runtime)

    with pytest.raises(BackendRuntimeCompatibilityError, match="mlx runtime unavailable"):
        get_payload_codec("mlx").from_numpy(
            np.asarray([1.0], dtype=np.float32),
            {"logical_dtype": "mlx.core.float32", "logical_device": "unknown"},
            map_location=None,
            strict_runtime=True,
        )
