"""TensorFlow payload codec and portable ``.tlspec`` round-trip tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from conftest import tensorflow_backend_modules

pytest.importorskip("safetensors")
tf, keras, _TF_BACKEND_SKIP_REASON = tensorflow_backend_modules()
del keras

import torchlens as tl  # noqa: E402
from torchlens.backends import (  # noqa: E402
    BackendPayloadUnsupportedError,
    BackendRuntimeCompatibilityError,
)
from torchlens.validation import validate_tlspec  # noqa: E402
from torchlens.validation.invariants import check_metadata_invariants  # noqa: E402

pytestmark = [
    pytest.mark.tf_backend,
    pytest.mark.skipif(
        _TF_BACKEND_SKIP_REASON is not None,
        reason=_TF_BACKEND_SKIP_REASON or "TensorFlow backend stack is supported",
    ),
]


def _identity_trace(value: Any) -> tl.Trace:
    """Capture a TensorFlow identity trace that preserves ``value``.

    Parameters
    ----------
    value
        TensorFlow tensor passed through ``tf.identity``.

    Returns
    -------
    tl.Trace
        TensorFlow trace with one saved identity output.
    """

    def identity(x: Any) -> Any:
        """Return a TensorFlow identity of ``x``."""

        return tf.identity(x)

    return tl.trace(identity, value, backend="tf")


def _saved_output(trace: Any) -> Any:
    """Return the loaded output payload from a one-output TensorFlow trace.

    Parameters
    ----------
    trace
        Loaded or live TensorFlow trace.

    Returns
    -------
    Any
        Saved output payload.
    """

    output_label = trace.output_layers[0]
    try:
        return trace[output_label].out
    except KeyError:
        return trace[f"{output_label}:1"].out


def _read_manifest(path: Path) -> dict[str, Any]:
    """Read one ``.tlspec`` manifest JSON object.

    Parameters
    ----------
    path
        Bundle directory.

    Returns
    -------
    dict[str, Any]
        Decoded manifest.
    """

    data = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write one ``.tlspec`` manifest JSON object.

    Parameters
    ----------
    path
        Bundle directory.
    manifest
        Manifest object to write.
    """

    (path / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def test_tf_default_tlspec_round_trip_materializes_payloads(tmp_path: Path) -> None:
    """Default TensorFlow saves should load metadata and materialized TF payloads."""

    expected = tf.constant([[-1.5, 0.25], [2.0, 4.5]], dtype=tf.float32)
    trace = _identity_trace(expected)
    path = tmp_path / "tf_default.tlspec"

    trace.save(path)
    validate_tlspec(path)
    loaded = tl.load(path)
    restored = _saved_output(loaded)

    assert loaded.backend == "tf"
    assert check_metadata_invariants(loaded) is True
    assert isinstance(restored, tf.Tensor)
    assert restored.dtype == tf.float32
    np.testing.assert_allclose(restored.numpy(), expected.numpy())


def test_tf_bfloat16_tlspec_round_trip_preserves_logical_dtype_and_bits(
    tmp_path: Path,
) -> None:
    """TensorFlow bf16 payloads should round-trip without uint16 corruption."""

    expected = tf.constant([[-1.5, 0.25], [2.0, 4.5]], dtype=tf.bfloat16)
    trace = _identity_trace(expected)
    path = tmp_path / "tf_bfloat16.tlspec"

    trace.save(path)
    manifest = _read_manifest(path)
    loaded = tl.load(path)
    restored = _saved_output(loaded)

    body_entries = [entry for entry in manifest["body_index"] if entry["intended_use"] == "out"]
    assert body_entries
    assert all(entry["logical_dtype"] == "tf.bfloat16" for entry in body_entries)
    assert all(entry["transport_dtype"] == "uint16" for entry in body_entries)
    assert restored.dtype == tf.bfloat16
    np.testing.assert_array_equal(
        restored.numpy().view(np.uint16),
        expected.numpy().view(np.uint16),
    )


def test_tf_tlspec_manifest_runtime_fingerprint_and_version_mismatch(
    tmp_path: Path,
) -> None:
    """TensorFlow manifests should carry and enforce the runtime fingerprint."""

    trace = _identity_trace(tf.constant([1.0, 2.0], dtype=tf.float32))
    path = tmp_path / "tf_manifest.tlspec"

    trace.save(path)
    manifest = tl.io.inspect_tlspec(path)

    assert manifest["schema_version"] == 2
    assert manifest["backend"] == "tf"
    assert manifest["torch_version"] is None
    assert manifest["backend_runtime"]["name"] == "tf"
    assert manifest["backend_runtime"]["version"] == tf.__version__
    assert manifest["payload_policy"]["policy"] == "array_payloads"
    assert manifest["payload_policy"]["materialization_supported"] is True
    assert manifest["body_format"] == "safetensors"

    manifest["backend_runtime"]["version"] = "0.0.not-the-installed-tensorflow"
    _write_manifest(path, manifest)
    with pytest.raises(BackendRuntimeCompatibilityError, match="TensorFlow runtime version"):
        tl.load(path)


def test_tf_string_payload_save_fails_closed(tmp_path: Path) -> None:
    """TensorFlow string payloads should raise the typed unsupported-payload error."""

    trace = _identity_trace(tf.constant(["left", "right"], dtype=tf.string))

    with pytest.raises(BackendPayloadUnsupportedError, match="not supported"):
        trace.save(tmp_path / "tf_string.tlspec")
