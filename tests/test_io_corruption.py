"""Corruption-path regression tests for portable TorchLens bundles."""

from __future__ import annotations

import io
import json
import pickle
import re
from pathlib import Path
from typing import Any

import pytest
import torch
import torchlens as tl
from torch import nn

pytest.importorskip("safetensors")

from torchlens import load, save
from torchlens._io import TorchLensIOError
from torchlens._io._safe_unpickle import SafeBundleUnpickler
from torchlens._io.manifest import sha256_of_file
from torchlens.data_classes.trace import Trace


class _CorruptionModel(nn.Module):
    """Small model used to create deterministic corruption fixtures."""

    def __init__(self) -> None:
        """Initialize the corruption test model."""

        super().__init__()
        self.linear1 = nn.Linear(4, 6)
        self.linear2 = nn.Linear(6, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the corruption test model.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output tensor.
        """

        return self.linear2(torch.relu(self.linear1(x)))


def _save_bundle(tmp_path: Path, path_name: str = "bundle.tl") -> Path:
    """Create a deterministic portable bundle for corruption tests.

    Parameters
    ----------
    tmp_path:
        Temporary test directory.
    path_name:
        Bundle directory name.

    Returns
    -------
    Path
        Saved bundle path.
    """

    torch.manual_seed(0)
    model = _CorruptionModel()
    inputs = torch.randn(2, 4)
    trace = tl.trace(model, inputs, layers_to_save="all", random_seed=0)
    bundle_path = tmp_path / path_name
    save(trace, bundle_path)
    return bundle_path


def _read_manifest(bundle_path: Path) -> dict[str, Any]:
    """Read one bundle manifest into a mutable dictionary.

    Parameters
    ----------
    bundle_path:
        Bundle directory path.

    Returns
    -------
    dict[str, Any]
        Decoded manifest JSON.
    """

    with (bundle_path / "manifest.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_manifest(bundle_path: Path, manifest: dict[str, Any]) -> None:
    """Overwrite one bundle manifest with JSON content.

    Parameters
    ----------
    bundle_path:
        Bundle directory path.
    manifest:
        JSON-ready manifest content.
    """

    with (bundle_path / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


def _first_tensor_entry(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return the first tensor entry from a decoded manifest.

    Parameters
    ----------
    manifest:
        Decoded manifest mapping.

    Returns
    -------
    dict[str, Any]
        First tensor entry.
    """

    tensors = manifest["tensors"]
    assert isinstance(tensors, list)
    return tensors[0]


def _first_saved_layer(trace: Trace) -> Any:
    """Return the first saved layer from one model log.

    Parameters
    ----------
    trace:
        Model log under test.

    Returns
    -------
    Any
        First saved layer-pass entry.
    """

    return next(layer for layer in trace.layer_list if layer.has_saved_activation)


def test_truncated_safetensors_blob_raises_with_blob_path(tmp_path: Path) -> None:
    """Truncated safetensors payloads should fail materialization with the blob path."""

    bundle_path = _save_bundle(tmp_path)
    manifest = _read_manifest(bundle_path)
    tensor_entry = _first_tensor_entry(manifest)
    blob_path = bundle_path / tensor_entry["relative_path"]
    original_bytes = blob_path.read_bytes()
    blob_path.write_bytes(original_bytes[: len(original_bytes) // 2])
    tensor_entry["sha256"] = sha256_of_file(blob_path)
    _write_manifest(bundle_path, manifest)

    lazy_log = load(bundle_path, lazy=True)
    layer = _first_saved_layer(lazy_log)

    with pytest.raises(
        TorchLensIOError,
        match=rf"Failed to materialize blob at {re.escape(str(blob_path))}\.",
    ):
        layer.materialize_out()


def test_missing_blob_file_raises_with_blob_id(tmp_path: Path) -> None:
    """Missing blobs should fail eager load with the missing blob id."""

    bundle_path = _save_bundle(tmp_path)
    manifest = _read_manifest(bundle_path)
    tensor_entry = _first_tensor_entry(manifest)
    blob_path = bundle_path / tensor_entry["relative_path"]
    blob_path.unlink()

    with pytest.raises(
        TorchLensIOError,
        match=rf"missing blob files for blob_id\(s\): {re.escape(tensor_entry['blob_id'])}",
    ):
        load(bundle_path)


def test_corrupt_metadata_pickle_raises_with_metadata_path(tmp_path: Path) -> None:
    """Garbage metadata should fail load with the metadata file path."""

    bundle_path = _save_bundle(tmp_path)
    metadata_path = bundle_path / "metadata.pkl"
    metadata_path.write_bytes(b"not a pickle")

    with pytest.raises(
        TorchLensIOError,
        match=rf"Failed to load bundle metadata from {re.escape(str(metadata_path))}\.",
    ):
        load(bundle_path)


def test_tampered_manifest_field_raises_with_field_name(tmp_path: Path) -> None:
    """Tampered manifest tensor metadata should fail load with the offending field."""

    bundle_path = _save_bundle(tmp_path)
    manifest = _read_manifest(bundle_path)
    tensor_entry = _first_tensor_entry(manifest)
    tensor_entry["dtype"] = "totally_fake_dtype"
    _write_manifest(bundle_path, manifest)

    with pytest.raises(TorchLensIOError, match=r"Unsupported dtype string in manifest"):
        load(bundle_path, lazy=True)


def test_stale_blob_counts_raise_with_count_field(tmp_path: Path) -> None:
    """Manifest blob-count drift should fail with the mismatched count field name."""

    bundle_path = _save_bundle(tmp_path)
    manifest = _read_manifest(bundle_path)
    manifest["n_out_blobs"] += 1
    _write_manifest(bundle_path, manifest)

    with pytest.raises(TorchLensIOError, match=r"n_out_blobs"):
        load(bundle_path)


def test_unknown_extra_files_warn_but_do_not_raise(tmp_path: Path) -> None:
    """Unreferenced files under ``blobs/`` should warn without aborting load."""

    bundle_path = _save_bundle(tmp_path)
    extra_blob_path = bundle_path / "blobs" / "unexpected.bin"
    extra_blob_path.write_bytes(b"extra")

    with pytest.warns(
        UserWarning,
        match=rf"unreferenced extra files in blobs/: {re.escape(extra_blob_path.name)}",
    ):
        restored = load(bundle_path, lazy=True)

    assert isinstance(restored, Trace)
    assert restored._loaded_from_bundle is True


def test_checksum_mismatch_raises_with_blob_id_and_path(tmp_path: Path) -> None:
    """Checksum mismatches should fail eager load with blob id and path details."""

    bundle_path = _save_bundle(tmp_path)
    manifest = _read_manifest(bundle_path)
    tensor_entry = _first_tensor_entry(manifest)
    blob_path = bundle_path / tensor_entry["relative_path"]
    blob_bytes = bytearray(blob_path.read_bytes())
    blob_bytes[-1] = (blob_bytes[-1] + 1) % 256
    blob_path.write_bytes(bytes(blob_bytes))

    with pytest.raises(
        TorchLensIOError,
        match=(
            rf"Checksum mismatch for blob_id={re.escape(tensor_entry['blob_id'])} at "
            rf"{re.escape(str(blob_path))}\."
        ),
    ):
        load(bundle_path)


# --- secA_1 (r51): corrupt-metadata unpickle exception-family normalization ---------
#
# r49 migrated ``SafeBundleUnpickler`` from the C ``pickle.Unpickler`` to the pure-Python
# ``pickle._Unpickler``. The C VM translated EVERY corrupt/truncated/malformed stream into
# ``pickle.UnpicklingError`` (which the load path maps to ``TorchLensIOError``); the
# pure-Python VM instead leaks bare ``IndexError`` (stack underflow), ``struct.error``
# (truncated fixed-width read), ``KeyError`` (missing memo), ``UnicodeDecodeError`` (bad
# utf-8), ``EOFError`` (truncation), etc. ``_DenyMissingDispatch`` only restored the
# invariant for the UNKNOWN-opcode facet; the ``SafeBundleUnpickler.load()`` boundary now
# restores it for the WHOLE family. This corpus is a per-FACET immunizer -- one payload per
# distinct corruption facet across the whole opcode family, NOT a version-fragile
# enumeration of every affected opcode.
#
# Payloads are built from ``pickle`` opcode constants (version-robust). The comment on each
# names the raw builtin the pure-Python base VM would leak WITHOUT the ``load()`` boundary.
_CORRUPTION_CORPUS: dict[str, bytes] = {
    # Stack underflow on a valid opcode -> bare IndexError from the base VM.
    "build_underflow": pickle.BUILD + pickle.STOP,
    "reduce_underflow": pickle.REDUCE + pickle.STOP,
    "pop_underflow": pickle.POP + pickle.STOP,
    "append_underflow": pickle.APPEND + pickle.STOP,
    "setitem_underflow": pickle.SETITEM + pickle.STOP,
    "tuple_underflow": pickle.TUPLE + pickle.STOP,
    "stack_global_underflow": pickle.STACK_GLOBAL + pickle.STOP,
    # Truncated fixed-width int read -> struct.error.
    "binint_truncated": pickle.BININT + b"\x01\x02",
    # Truncated variable-width read -> EOFError (no STOP after the short read).
    "binunicode_truncated": pickle.BINUNICODE + b"\x05\x00\x00\x00ab",
    # Invalid utf-8 in a SHORT_BINUNICODE -> UnicodeDecodeError.
    "bad_utf8_short_binunicode": pickle.SHORT_BINUNICODE + b"\x01\xff",
    # Missing memo key -> KeyError.
    "missing_memo_binget": pickle.BINGET + b"\x00" + pickle.STOP,
    # Empty stream / no STOP -> EOFError.
    "empty_stream": b"",
    "no_stop": pickle.EMPTY_LIST,
    # Unknown opcode -> already clean via _DenyMissingDispatch (passthrough coverage).
    "unknown_opcode": b"\xff" + pickle.STOP,
}


@pytest.mark.parametrize("facet", sorted(_CORRUPTION_CORPUS))
def test_secA_corpus_unit_normalizes_to_unpickling_error(facet: str) -> None:
    """Every corruption facet raises a clean ``UnpicklingError``, never a raw builtin."""

    payload = _CORRUPTION_CORPUS[facet]
    with pytest.raises(pickle.UnpicklingError) as excinfo:
        SafeBundleUnpickler(io.BytesIO(payload)).load()
    # The raised type must be exactly the pickle contract type, not a leaked builtin
    # (IndexError / struct.error / KeyError / UnicodeDecodeError / EOFError) that the
    # load-site ``except UnpicklingError`` would fail to catch.
    assert not isinstance(excinfo.value, (IndexError, KeyError, UnicodeDecodeError, EOFError))


def test_secA_valid_stream_still_loads_through_boundary() -> None:
    """A VALID pickle stream still round-trips through the ``load()`` boundary unmasked."""

    for value in ({"a": 1, "b": [1, 2, 3]}, ("x", 2, 3.5), [1, {"k": "v"}], "hello"):
        stream = io.BytesIO(pickle.dumps(value))
        assert SafeBundleUnpickler(stream).load() == value


@pytest.mark.parametrize("facet", sorted(_CORRUPTION_CORPUS))
def test_secA_corpus_e2e_metadata_raises_torchlens_io_error(tmp_path: Path, facet: str) -> None:
    """A corrupt ``metadata.pkl`` (any facet) fails ``tl.load`` with a clean IO error."""

    bundle_path = _save_bundle(tmp_path, path_name=f"bundle_{facet}.tl")
    metadata_path = bundle_path / "metadata.pkl"
    metadata_path.write_bytes(_CORRUPTION_CORPUS[facet])

    with pytest.raises(
        TorchLensIOError,
        match=rf"Failed to load bundle metadata from {re.escape(str(metadata_path))}\.",
    ):
        load(bundle_path)


def test_secA_valid_bundle_still_round_trips(tmp_path: Path) -> None:
    """Over-deny pin: a genuine, uncorrupted bundle still loads unchanged."""

    bundle_path = _save_bundle(tmp_path)
    restored = load(bundle_path)
    assert isinstance(restored, Trace)
    assert restored._loaded_from_bundle is True
