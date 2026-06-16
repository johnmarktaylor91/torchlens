"""Paddle payload codec and portable ``.tlspec`` round-trip tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

paddle = pytest.importorskip("paddle")

import torchlens as tl  # noqa: E402
from torchlens._io.payload_codec import get_payload_codec  # noqa: E402
from torchlens._io.tensor_policy import Ok  # noqa: E402
from torchlens.validation import validate_tlspec  # noqa: E402
from torchlens.validation.invariants import check_metadata_invariants  # noqa: E402

pytestmark = pytest.mark.backend_paddle

paddle.seed(0)


def _tensor_for_dtype(dtype_name: str) -> Any:
    """Return a deterministic Paddle tensor for ``dtype_name``.

    Parameters
    ----------
    dtype_name:
        Paddle dtype name without the ``paddle.`` prefix.

    Returns
    -------
    Any
        Paddle tensor with nontrivial values for the requested dtype.
    """

    if dtype_name == "bool":
        return paddle.to_tensor([[True, False], [False, True]], dtype="bool")
    if dtype_name == "int64":
        return paddle.to_tensor([[-3, 0], [7, 11]], dtype="int64")
    base = paddle.to_tensor([[-1.5, -0.25], [0.125, 3.5]], dtype="float32")
    return base.cast(dtype_name)


def _assign_trace(value: Any) -> Any:
    """Capture a one-op Paddle trace that preserves ``value`` exactly."""

    return tl.trace(lambda tensor: paddle.assign(tensor), value, backend="paddle")


def _saved_output(trace: Any) -> Any:
    """Return the saved output payload from a loaded assign trace."""

    output_label = f"{trace.output_layers[0]}:1"
    return trace[output_label].out


def _assert_paddle_tensor_equal(actual: Any, expected: Any) -> None:
    """Assert that two Paddle tensors have equal dtype and NumPy payload bits."""

    assert type(actual).__module__.startswith("paddle")
    assert str(actual.dtype) == str(expected.dtype)
    np.testing.assert_array_equal(actual.numpy(), expected.numpy())


def test_paddle_payload_codec_manifest_fields_are_registered() -> None:
    """Paddle codec should expose schema-v2 logical and transport metadata."""

    codec = get_payload_codec("paddle")
    value = _tensor_for_dtype("bfloat16")

    assert codec.can_encode(value)
    assert isinstance(codec.validate_for_save(value, strict=True), Ok)
    encoded = codec.to_numpy(value)
    assert encoded.logical_dtype == "paddle.bfloat16"
    assert encoded.logical_device == "Place(cpu)"
    assert encoded.array.dtype == np.uint16
    assert encoded.codec_metadata is not None
    assert encoded.codec_metadata["paddle_bfloat16_transport_bits"] is True

    fields = codec.manifest_fields(value, encoded)

    assert codec.backend_name == "paddle"
    assert fields["logical_backend"] == "paddle"
    assert fields["codec"] == "numpy_safetensors_v1"
    assert fields["logical_dtype"] == "paddle.bfloat16"
    assert fields["logical_device"] == "Place(cpu)"
    assert fields["transport_backend"] == "safetensors.torch"
    assert fields["transport_dtype"] == "uint16"
    assert isinstance(fields["codec_metadata"], dict)


def test_paddle_default_tlspec_round_trip_materializes_payloads(tmp_path: Path) -> None:
    """Default Paddle saves should load metadata and materialized Paddle payloads."""

    expected = _tensor_for_dtype("float32")
    trace = _assign_trace(expected)
    path = tmp_path / "paddle_default.tlspec"

    trace.save(path)
    validate_tlspec(path)
    loaded = tl.load(path)

    assert loaded.backend == "paddle"
    assert check_metadata_invariants(loaded) is True
    _assert_paddle_tensor_equal(_saved_output(loaded), expected)


def test_paddle_tlspec_manifest_runtime_fingerprint(tmp_path: Path) -> None:
    """Paddle manifest should include the real runtime fingerprint."""

    trace = _assign_trace(_tensor_for_dtype("float32"))
    path = tmp_path / "paddle_manifest.tlspec"

    trace.save(path)
    manifest = tl.io.inspect_tlspec(path)

    assert manifest["schema_version"] == 2
    assert manifest["backend"] == "paddle"
    assert manifest["torch_version"] is None
    assert manifest["backend_runtime"]["name"] == "paddle"
    assert manifest["backend_runtime"]["version"] == paddle.__version__
    assert manifest["backend_runtime"]["version"] != "0.0.metadata"
    assert manifest["payload_policy"]["policy"] == "array_payloads"
    assert manifest["payload_policy"]["materialization_supported"] is True
    assert manifest["body_format"] == "safetensors"


def test_paddle_bfloat16_tlspec_round_trip_preserves_bits(tmp_path: Path) -> None:
    """Paddle bf16 payloads should round-trip through ``.tlspec`` without uint16 corruption."""

    expected = _tensor_for_dtype("bfloat16")
    trace = _assign_trace(expected)
    path = tmp_path / "paddle_bfloat16.tlspec"

    trace.save(path)
    loaded = tl.load(path)
    restored = _saved_output(loaded)

    assert str(restored.dtype) == "paddle.bfloat16"
    np.testing.assert_array_equal(restored.numpy(), expected.numpy())


@pytest.mark.parametrize("dtype_name", ["float32", "int64", "bool", "float16", "bfloat16"])
def test_paddle_tlspec_round_trips_dtype_payloads(
    tmp_path: Path,
    dtype_name: str,
) -> None:
    """Paddle dtype payloads should round-trip through public ``.tlspec`` I/O."""

    expected = _tensor_for_dtype(dtype_name)
    trace = _assign_trace(expected)
    path = tmp_path / f"paddle_{dtype_name}.tlspec"

    trace.save(path)
    loaded = tl.load(path)

    _assert_paddle_tensor_equal(_saved_output(loaded), expected)
