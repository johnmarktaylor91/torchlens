"""Corruption-path regression tests for portable TorchLens bundles."""

from __future__ import annotations

import io
import json
import pickle
import re
from pathlib import Path
from typing import Any

import pytest
import safetensors  # noqa: F401
import torch
from _format3_envelope import read_record, redigest, set_cell, tensor_references
from torch import nn

import torchlens as tl
from torchlens import load
from torchlens._data_classes.trace import Trace
from torchlens._io import ArtifactError
from torchlens._io._safe_unpickle import SafeBundleUnpickler


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
    trace = tl.trace(model, inputs, capture=tl.options.CaptureOptions(seed=0), save="all")
    bundle_path = tmp_path / path_name
    trace.save(bundle_path)
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


def _first_tensor_entry(root: Path) -> dict[str, Any]:
    """Return an authoritative tensor cell, whose blob is a chunk-relative path."""

    return next(iter(tensor_references(root).values()))


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

    return next(layer for layer in trace._layer_list if layer.has_saved_activation)


def test_truncated_safetensors_blob_raises_with_blob_path(tmp_path: Path) -> None:
    """Even lazy load refuses a malformed referenced header before publishing a Trace."""

    bundle_path = _save_bundle(tmp_path)
    blob_path = bundle_path / _first_tensor_entry(bundle_path)["blob"]
    original_bytes = blob_path.read_bytes()
    blob_path.write_bytes(original_bytes[: len(original_bytes) // 2])
    redigest(bundle_path, blob_path.relative_to(bundle_path).as_posix())
    with pytest.raises(ArtifactError) as caught:
        load(bundle_path, lazy=True)
    assert caught.value.fields["code"] == "bundle_load_failed"
    assert caught.value.file_path == str(blob_path)


def test_missing_blob_file_raises_with_blob_id(tmp_path: Path) -> None:
    """The manifest's listed chunk cannot disappear before eager load."""

    bundle_path = _save_bundle(tmp_path)
    blob_path = bundle_path / _first_tensor_entry(bundle_path)["blob"]
    blob_path.unlink()
    with pytest.raises(ArtifactError) as caught:
        load(bundle_path)
    assert caught.value.fields["code"] == "bundle_metadata_integrity_refused"
    assert caught.value.file_path == str(blob_path)


def test_corrupt_metadata_pickle_raises_with_metadata_path(tmp_path: Path, monkeypatch) -> None:
    """Format 3 refuses a planted pickle at the JSON door without unpickling it."""

    bundle_path = _save_bundle(tmp_path)
    record_path = bundle_path / "records/trace.json"
    record_path.write_bytes(b"not a pickle")
    redigest(bundle_path, "records/trace.json")
    calls = []
    monkeypatch.setattr(SafeBundleUnpickler, "load", lambda self: calls.append(self))
    with pytest.raises(ArtifactError) as caught:
        load(bundle_path)
    assert caught.value.fields["code"] == "bundle_metadata_integrity_refused"
    assert caught.value.file_path == str(record_path)
    assert calls == []


def test_load_front_door_causes_carry_distinct_codes(tmp_path: Path) -> None:
    """R65: the tl.load front-door causes carry distinct ``fields['code']`` values.

    A caller could only tell the six front-door failure causes apart by parsing one
    content-free message. Each now carries a stable code to branch on.
    """

    # (1) symlinked load path.
    real = _save_bundle(tmp_path, "real.tl")
    link = tmp_path / "link.tl"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ArtifactError) as sym:
        load(link)
    assert sym.value.fields.get("code") == "load_path_symlink_rejected"

    # (2) missing manifest is its own cause, distinct from unreadable (R65).
    empty = tmp_path / "empty.tl"
    empty.mkdir()
    with pytest.raises(ArtifactError) as missing:
        load(empty)
    assert missing.value.fields.get("code") == "manifest_missing"

    # (3) manifest root is not a JSON object.
    not_object = tmp_path / "notobj.tl"
    not_object.mkdir()
    (not_object / "manifest.json").write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ArtifactError) as scalar:
        load(not_object)
    assert scalar.value.fields.get("code") == "manifest_not_json_object"

    # (4) A record no longer agrees with the manifest's digest.
    integrity = _save_bundle(tmp_path, "integrity.tl")
    (integrity / "records/trace.json").write_bytes(b"not valid record JSON")
    with pytest.raises(ArtifactError) as bad_record:
        load(integrity)
    assert bad_record.value.fields.get("code") == "bundle_metadata_integrity_refused"

    # (5) A redigested malformed tensor header reaches the typed payload door.
    generic = _save_bundle(tmp_path, "generic.tl")
    relative = _first_tensor_entry(generic)["blob"]
    (generic / relative).write_bytes(b"invalid safetensors header")
    redigest(generic, relative)
    with pytest.raises(ArtifactError) as generic_fail:
        load(generic)
    assert generic_fail.value.fields.get("code") == "bundle_load_failed"


def test_manifest_schema_violations_refuse_with_stable_code(tmp_path: Path) -> None:
    """Malformed current manifests and the retained Manifest seam refuse typed."""

    bundle_path = _save_bundle(tmp_path)
    manifest = _read_manifest(bundle_path)
    tampered = dict(manifest)
    tampered["records"] = "not an object"
    _write_manifest(bundle_path, tampered)
    with pytest.raises(ArtifactError) as excinfo:
        load(bundle_path)
    assert excinfo.value.fields["code"] == "manifest_schema_invalid"
    assert excinfo.value.fields.get("remedy")
    assert str(excinfo.value).rstrip(".").endswith(excinfo.value.fields["remedy"])

    from torchlens._io.manifest import Manifest

    legacy_path = tmp_path / "legacy-manifest.json"
    legacy = _legacy_manifest(manifest).to_dict()
    legacy["n_out_blobs"] = "three"
    legacy_path.write_text(json.dumps(legacy), encoding="utf-8")
    with pytest.raises(ArtifactError) as via_read:
        Manifest.read(legacy_path)
    assert via_read.value.fields["code"] == "manifest_schema_invalid"
    assert via_read.value.file_path == str(legacy_path)

    tampered = json.loads(json.dumps(manifest))
    del next(iter(tampered["files"].values()))["sha256"]
    _write_manifest(bundle_path, tampered)
    with pytest.raises(ArtifactError) as forged:
        load(bundle_path)
    assert forged.value.fields["code"] == "manifest_schema_invalid"


def _legacy_manifest(identity: dict[str, Any]):
    """Build the retained legacy policy object without misrouting a current envelope."""

    from torchlens._io.manifest import Manifest

    return Manifest(
        **{
            key: identity[key]
            for key in (
                "tlspec_version",
                "torchlens_version",
                "torch_version",
                "python_version",
                "platform",
                "created_at",
            )
        },
        bundle_format="directory",
        n_layers=0,
        n_out_blobs=0,
        n_grad_blobs=0,
        n_auxiliary_blobs=0,
        tensors=[],
        unsupported_tensors=[],
    )


def test_version_policy_refusals_carry_distinct_codes(tmp_path: Path) -> None:
    """R65-1 version half: the three version-policy refusals are branchable.

    Provoked at the policy chokepoint itself (``enforce_version_policy``): the
    ``tl.load`` front door reaches a twin version check in
    ``validation.validate_tlspec`` first for the too-new case, which is a
    separate (relayed) duplication finding.
    """

    from torchlens._io import TLSPEC_VERSION
    from torchlens._io.manifest import enforce_version_policy

    bundle_path = _save_bundle(tmp_path, "versions.tl")
    base = _read_manifest(bundle_path)

    # (1) artifact newer than the runtime.
    with pytest.raises(ArtifactError) as newer:
        enforce_version_policy(_legacy_manifest({**base, "tlspec_version": TLSPEC_VERSION + 1}))
    assert newer.value.fields.get("code") == "artifact_version_above_runtime"
    assert newer.value.fields.get("remedy")

    # (2) torch major mismatch.
    with pytest.raises(ArtifactError) as drift:
        enforce_version_policy(_legacy_manifest({**base, "torch_version": "1.0.0"}))
    assert drift.value.fields.get("code") == "bundle_torch_incompatible"

    # (3) producer version unverifiable under PEP 440.
    with pytest.raises(ArtifactError) as producer:
        enforce_version_policy(_legacy_manifest({**base, "torchlens_version": "not-a-version"}))
    assert producer.value.fields.get("code") == "bundle_producer_unverifiable"


def test_manifest_write_failure_refuses_typed(tmp_path: Path) -> None:
    """R65-1 write half: a failed manifest write carries ``manifest_write_failed``."""

    bundle_path = _save_bundle(tmp_path, "writefail.tl")
    manifest = _legacy_manifest(_read_manifest(bundle_path))
    target_dir = tmp_path / "is_a_directory"
    target_dir.mkdir()
    with pytest.raises(ArtifactError) as excinfo:
        manifest.write(target_dir)
    assert excinfo.value.fields.get("code") == "manifest_write_failed"
    assert excinfo.value.fields.get("remedy")


def test_io_artifact_door_codes_are_provoked(tmp_path: Path) -> None:
    """R25 ratchet shrink: three io artifact-door codes gain live provocations.

    Each of these sat in ``UNPROVOKED_BASELINE`` -- a code swap at any of the
    three doors would have failed zero tests.
    """

    trace = _save_bundle(tmp_path, "seed.tl")  # returns the bundle path
    from torchlens.io import load_intervention_spec

    with pytest.raises(TypeError) as kind:
        load_intervention_spec(trace)
    assert kind.value.fields["code"] == "artifact_kind_mismatch"

    source = tl.trace(_CorruptionModel(), torch.randn(2, 4))
    with pytest.raises(ValueError) as level:
        tl.Bundle({"m": source}).save(tmp_path / "b.tlspec", level="runnable")
    assert level.value.fields["code"] == "artifact_save_level_unsupported"

    with pytest.raises(ValueError) as payload:
        source.save(tmp_path / "weights.tl", level="portable", include_weights=True)
    assert payload.value.fields["code"] == "save_payload_level_conflict"


def test_remedy_field_derives_from_authored_message_tail() -> None:
    """R65 remedy contract: fields['remedy'] exists whenever the message ends
    with an authored "Remedy: ..." sentence; an explicit kwarg always wins."""

    from torchlens.errors._base import TorchLensError

    derived = TorchLensError("Thing failed. Remedy: do the fix.", code="x")
    assert derived.fields["remedy"] == "do the fix"
    assert str(derived).rstrip(".").endswith(derived.fields["remedy"])

    explicit = TorchLensError("Thing failed. Remedy: prose text.", remedy="explicit wins")
    assert explicit.fields["remedy"] == "explicit wins"

    plain = TorchLensError("No remedy sentence here.")
    assert "remedy" not in plain.fields


def test_tampered_manifest_field_raises_with_field_name(tmp_path: Path) -> None:
    """A forged tensor dtype in the authoritative record cell refuses by field."""

    bundle_path = _save_bundle(tmp_path)
    for row, cell in enumerate(read_record(bundle_path, "op")["columns"]["out"]):
        if isinstance(cell, dict) and cell.get("$tl") == "tensor":
            set_cell(bundle_path, "op", "out", row, {**cell, "dtype": "totally_fake_dtype"})
            break
    else:
        pytest.fail("the positive capture produced no saved op tensor")
    with pytest.raises(ArtifactError) as caught:
        load(bundle_path, lazy=True)
    assert caught.value.fields["code"] == "record_schema_invalid"
    assert "dtype" in str(caught.value)


def test_stale_blob_counts_raise_with_count_field(tmp_path: Path) -> None:
    """The manifest's row count must agree with its authoritative columnar table."""

    bundle_path = _save_bundle(tmp_path)
    manifest = _read_manifest(bundle_path)
    manifest["records"]["op"]["rows"] += 1
    _write_manifest(bundle_path, manifest)
    with pytest.raises(ArtifactError) as caught:
        load(bundle_path)
    assert caught.value.fields["code"] == "bundle_metadata_integrity_refused"
    assert "row" in str(caught.value)


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
    """A changed chunk body refuses at its listed digest and names its path."""

    bundle_path = _save_bundle(tmp_path)
    blob_path = bundle_path / _first_tensor_entry(bundle_path)["blob"]
    blob_bytes = bytearray(blob_path.read_bytes())
    blob_bytes[-1] = (blob_bytes[-1] + 1) % 256
    blob_path.write_bytes(blob_bytes)
    with pytest.raises(ArtifactError) as caught:
        load(bundle_path)
    assert caught.value.fields["code"] == "bundle_metadata_integrity_refused"
    assert caught.value.file_path == str(blob_path)


# --- secA_1 (r51): corrupt-metadata unpickle exception-family normalization ---------
#
# r49 migrated ``SafeBundleUnpickler`` from the C ``pickle.Unpickler`` to the pure-Python
# ``pickle._Unpickler``. The C VM translated EVERY corrupt/truncated/malformed stream into
# ``pickle.UnpicklingError`` (which the load path maps to ``ArtifactError``); the
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
def test_secA_corpus_e2e_metadata_raises_torchlens_io_error(
    tmp_path: Path, facet: str, monkeypatch
) -> None:
    """Every corrupt pickle facet refuses at the current JSON door before unpickling."""

    bundle_path = _save_bundle(tmp_path, path_name=f"bundle_{facet}.tl")
    record_path = bundle_path / "records/trace.json"
    record_path.write_bytes(_CORRUPTION_CORPUS[facet])
    redigest(bundle_path, "records/trace.json")
    calls = []
    monkeypatch.setattr(SafeBundleUnpickler, "load", lambda self: calls.append(self))
    with pytest.raises(ArtifactError) as caught:
        load(bundle_path)
    assert caught.value.fields["code"] == "bundle_metadata_integrity_refused"
    assert caught.value.file_path == str(record_path)
    assert calls == []


def test_secA_valid_bundle_still_round_trips(tmp_path: Path) -> None:
    """Over-deny pin: a genuine, uncorrupted bundle still loads unchanged."""

    bundle_path = _save_bundle(tmp_path)
    restored = load(bundle_path)
    assert isinstance(restored, Trace)
    assert restored._loaded_from_bundle is True


# --- secA_1 follow-up (r51): the load() boundary is LOCUS-classified ----------------
#
# The ``SafeBundleUnpickler.load()`` boundary normalizes ONLY failures whose traceback
# stays inside the pickle VM itself (stdlib ``pickle.py`` + ``_safe_unpickle.py``).
# An exception that arose while executing APPLICATION/reconstruction code (a REDUCE
# callable's body, a ``__setstate__``, a ``find_class`` import) propagates RAW with its
# identity intact, so the ``bundle.py`` load sites keep it as the DIRECT ``__cause__``
# of ``ArtifactError`` (the pre-existing legacy-bundle live-resource contract,
# pinned by ``test_io_bundle.py::
# test_legacy_multi_trace_bundle_load_typeerror_raises_torchlens_io_error``). These two
# tests lock the narrowed boundary from BOTH sides.


def test_secA_application_reconstruction_failure_keeps_identity() -> None:
    """A failure INSIDE an admitted reconstructor propagates RAW, never 'corrupt'."""

    import torch._utils

    class _FailsInBody:
        def __reduce__(self) -> Any:
            """Reduce to an admitted reconstructor whose BODY raises on bad args."""

            return (torch._utils._rebuild_tensor_v2, (None, 0, (), (), False, None))

    stream = io.BytesIO(pickle.dumps(_FailsInBody()))
    with pytest.raises(Exception) as excinfo:
        SafeBundleUnpickler(stream).load()
    assert not isinstance(excinfo.value, pickle.UnpicklingError)


def test_secA_callsite_arity_mismatch_still_normalizes() -> None:
    """An arity-mismatch REDUCE (zero application frames ran) stays 'corrupt'."""

    import torch._utils

    class _ArityMismatch:
        def __reduce__(self) -> Any:
            """Reduce to an admitted reconstructor invoked with too few arguments."""

            return (torch._utils._rebuild_tensor_v2, ())

    stream = io.BytesIO(pickle.dumps(_ArityMismatch()))
    with pytest.raises(pickle.UnpicklingError):
        SafeBundleUnpickler(stream).load()


def test_corrupt_manifest_json_refuses_manifest_unreadable(tmp_path: Path) -> None:
    """A manifest that fails JSON parsing carries the manifest_unreadable code."""

    corrupt = tmp_path / "corrupt.tl"
    corrupt.mkdir()
    (corrupt / "manifest.json").write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ArtifactError) as excinfo:
        load(corrupt)
    assert excinfo.value.fields.get("code") == "manifest_unreadable"
