"""Tests for portable TorchLens directory bundle save/load."""

from __future__ import annotations

import json
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import safetensors  # noqa: F401
import torch
from _format3_envelope import redigest
from torch import nn

import torchlens as tl
from torchlens import Trace, func, load, trace as trace_fn
from torchlens._io import (
    TLSPEC_VERSION,
    ArtifactError,
    bundle as bundle_io,
)
from torchlens._io.paths import resolve_bundle_blob_path
from torchlens._io.payload_codec import (
    _raise_for_unsupported_array_dtype,
    _unsupported_array_dtype_reason,
)
from torchlens.io import cleanup_tmp, detect_tlspec_format
from torchlens.types import UnencodableValue


class _ConvBundleModel(nn.Module):
    """Small convolutional model used for bundle round-trip checks."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(4, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the test model."""

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return self.conv3(x)


class _ActivationPostfuncModel(nn.Module):
    """Small linear model for out transform policy tests."""

    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(4, 4)
        self.linear2 = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the test model."""

        return self.linear2(torch.relu(self.linear1(x)))


class _InputTransformModel(nn.Module):
    """Small model for input transform provenance tests."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a simple affine operation.

        Parameters
        ----------
        x:
            Model input tensor.

        Returns
        -------
        torch.Tensor
            Shifted tensor.
        """

        return x + 1


def _build_conv_log(seed: int = 0) -> Trace:
    """Create a deterministic ``Trace`` for bundle tests.

    Parameters
    ----------
    seed:
        Random seed used for model init and inputs.

    Returns
    -------
    Trace
        Logged forward pass with all outs saved.
    """

    torch.manual_seed(seed)
    model = _ConvBundleModel()
    x = torch.randn(1, 3, 8, 8)
    return trace_fn(model, x, capture=tl.options.CaptureOptions(seed=seed), save="all")


def _build_sparse_out_log() -> Trace:
    """Create a live log whose first saved out is converted to sparse post-hoc.

    Returns
    -------
    Trace
        Model log containing an unsupported sparse out tensor.
    """

    trace = _build_conv_log()
    first_saved_layer = next(layer for layer in trace._layer_list if layer.has_saved_activation)
    assert isinstance(first_saved_layer.out, torch.Tensor)
    first_saved_layer.out = first_saved_layer.out.to_sparse()
    return trace


def _build_non_tensor_out_log() -> Trace:
    """Create a live log whose first transformed out becomes a non-tensor post-hoc.

    Returns
    -------
    Trace
        Model log containing a non-tensor transformed out payload.
    """

    trace = _build_conv_log()
    first_saved_layer = next(layer for layer in trace._layer_list if layer.has_saved_activation)
    first_saved_layer._payload_transformed_out = 1.0
    trace._options_activation_transform = lambda tensor: float(tensor.mean().item())
    return trace


def _build_complex_out_log(*, dtype: torch.dtype) -> Trace:
    """Create a live log whose first saved out is converted to a complex dtype post-hoc.

    Parameters
    ----------
    dtype:
        Target complex dtype (e.g. ``torch.complex64``/``torch.complex128``).

    Returns
    -------
    Trace
        Model log containing a complex-dtype out tensor.
    """

    trace = _build_conv_log()
    first_saved_layer = next(layer for layer in trace._layer_list if layer.has_saved_activation)
    assert isinstance(first_saved_layer.out, torch.Tensor)
    real = first_saved_layer.out
    first_saved_layer.out = torch.complex(real, real).to(dtype)
    return trace


def _save_bundle(
    tmp_path: Path, *, seed: int = 0, path_name: str = "bundle.tl"
) -> tuple[Path, Trace]:
    """Save a deterministic bundle and return both path and source log.

    Parameters
    ----------
    tmp_path:
        Temporary test directory.
    seed:
        Random seed used for model init and inputs.
    path_name:
        Bundle directory name.

    Returns
    -------
    tuple[Path, Trace]
        Saved bundle path and the original live log.
    """

    trace = _build_conv_log(seed=seed)
    bundle_path = tmp_path / path_name
    trace.save(bundle_path)
    return bundle_path, trace


def _read_manifest(bundle_path: Path) -> dict[str, Any]:
    """Read a bundle manifest into a mutable dict.

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
    """Overwrite a bundle manifest with JSON content.

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


def _first_blob_path(bundle_path: Path) -> Path:
    """Return the first tensor blob referenced by the bundle manifest.

    Parameters
    ----------
    bundle_path:
        Bundle directory path.

    Returns
    -------
    Path
        Path to the first saved blob.
    """

    manifest = _read_manifest(bundle_path)
    relative_path = next(path for path in manifest["files"] if path.startswith("blobs/"))
    return bundle_path / relative_path


def _corrupt_blob_byte(blob_path: Path) -> None:
    """Modify one byte in a blob file without deleting it.

    Parameters
    ----------
    blob_path:
        Blob file path to modify.
    """

    blob_bytes = bytearray(blob_path.read_bytes())
    blob_bytes[-1] = (blob_bytes[-1] + 1) % 256
    blob_path.write_bytes(bytes(blob_bytes))


def test_manifest_provenance_roundtrip_and_hash_determinism(tmp_path: Path) -> None:
    """Trace saves should carry a compact, deterministic provenance certificate."""

    torch.manual_seed(44)
    model = _InputTransformModel().eval()
    inputs = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    captured = trace_fn(model, inputs, capture=tl.options.CaptureOptions(seed=44), save="all")
    first_path = tmp_path / "first.tlspec"
    second_path = tmp_path / "second.tlspec"

    captured.save(first_path)
    captured.save(second_path)
    restored = load(first_path)
    first = _read_manifest(first_path)
    second = _read_manifest(second_path)

    assert restored._num_ops == captured._num_ops
    assert first["provenance"] is not None
    assert second["provenance"] is not None
    assert first["provenance"]["provenance_version"] == 1
    assert first["provenance"]["capture_devices"] == ["cpu"]
    assert first["provenance"]["dtype_policy"]["default_dtype"] == "torch.float32"
    assert first["provenance"]["dtype_policy"]["observed_autocast"]
    assert first["provenance"]["input_hash"] == second["provenance"]["input_hash"]
    assert (
        first["provenance"]["model_structure_hash"] == second["provenance"]["model_structure_hash"]
    )
    assert first["provenance"]["rng_state_digests"] == second["provenance"]["rng_state_digests"]
    assert len(json.dumps(first["provenance"])) < 16_384


def test_manifest_provenance_sentinel_round_trips_and_tamper_refuses(tmp_path: Path) -> None:
    """Selective saves record the could-not-compute sentinel; tampering still refuses.

    A selective ``save=`` capture discards input payloads before save time, so
    the writer records ``unavailable:<ExceptionName>`` for ``input_hash`` by
    design (could-not-compute stays distinguishable from does-not-apply). Load
    must accept exactly that closed grammar -- and keep refusing any digest
    value that is neither a SHA-256 hex digest nor the sentinel.
    """

    torch.manual_seed(45)
    model = _ActivationPostfuncModel().eval()
    inputs = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    captured = trace_fn(model, inputs, save=func("relu"))
    bundle_path = tmp_path / "selective.tlspec"
    captured.save(bundle_path)

    manifest = _read_manifest(bundle_path)
    recorded = manifest["provenance"]["input_hash"]
    assert recorded.startswith("unavailable:")
    assert recorded.partition(":")[2].isidentifier()
    restored = load(bundle_path)
    assert restored._num_ops == captured._num_ops

    for forged in ("deadbeef", "unavailable:", "unavailable:not an identifier", "x" * 64):
        manifest["provenance"]["input_hash"] = forged
        _write_manifest(bundle_path, manifest)
        with pytest.raises(ArtifactError, match="input_hash"):
            load(bundle_path)


def test_manifest_without_optional_provenance_still_loads(tmp_path: Path) -> None:
    """Pre-change manifests with no provenance block should retain load compatibility."""

    bundle_path, _captured = _save_bundle(tmp_path)
    manifest = _read_manifest(bundle_path)
    manifest.pop("provenance")
    _write_manifest(bundle_path, manifest)

    parsed = _read_manifest(bundle_path)
    restored = load(bundle_path)

    assert parsed.get("provenance") is None
    assert restored._num_ops > 0


def test_manifest_git_commit_is_absent_outside_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No Git hash when the TORCHLENS package dir is not a repository (R21-2).

    The provenance git hash is resolved from the torchlens package directory, not
    the working directory, so it is ``None`` for a released wheel (no ``.git``). The
    working directory is deliberately irrelevant; only the package dir governs.
    """

    import torchlens._io.bundle as bundle_mod

    non_repo = tmp_path / "not_a_repo"
    non_repo.mkdir()
    monkeypatch.setattr(bundle_mod, "_torchlens_package_dir", lambda: non_repo)
    captured = trace_fn(
        _InputTransformModel(),
        torch.ones(2, 3),
        save="all",
    )
    bundle_path = tmp_path / "outside.tlspec"

    captured.save(bundle_path)
    manifest = _read_manifest(bundle_path)

    assert manifest["provenance"] is not None
    assert manifest["provenance"]["git_commit_hash"] is None


def test_input_transform_repr_roundtrips_while_callable_drops(tmp_path: Path) -> None:
    """Input transform repr is kept while the live callable remains non-portable."""

    def double_input(x: torch.Tensor) -> torch.Tensor:
        """Double the input tensor for capture.

        Parameters
        ----------
        x:
            Raw input tensor.

        Returns
        -------
        torch.Tensor
            Doubled tensor.
        """

        return x * 2

    trace = trace_fn(
        _InputTransformModel(),
        torch.ones(1, 2),
        capture=tl.options.CaptureOptions(input_transform=double_input),
    )
    assert trace.options.input_transform == repr(double_input)
    assert trace._transform is double_input

    bundle_path = tmp_path / "input_transform.tlspec"
    trace.save(bundle_path)
    restored = load(bundle_path)

    assert restored.options.input_transform == trace.options.input_transform
    assert restored._transform is None


def test_bundle_save_strict_false_refuses_sparse_tensors(tmp_path: Path) -> None:
    """Format 3 never silently skips an unsupported tensor, even with strict=False."""

    trace = _build_sparse_out_log()
    with pytest.raises(ArtifactError) as caught:
        trace.save(tmp_path / "sparse_bundle.tl", strict=False)
    assert caught.value.fields["code"] == "metadata_value_unportable"


@pytest.mark.parametrize("strict", [True, False])
def test_bundle_save_complex128_round_trips(tmp_path: Path, strict: bool) -> None:
    """Complex128 persists as paired float64 values and reloads exactly."""

    trace = _build_complex_out_log(dtype=torch.complex128)
    path = tmp_path / "complex128.tlspec"
    trace.save(path, strict=strict)
    loaded = load(path)
    for original, restored in zip(trace.ops, loaded.ops, strict=True):
        if isinstance(original.out, torch.Tensor):
            assert torch.equal(restored.out, original.out)
            assert restored.out.dtype == original.out.dtype


# shrink-r4: rows kept carry a sole library arc or a campaign kill; the others were proven redundant (slice 1 evidence)
@pytest.mark.parametrize(
    ("scenario", "mutate_manifest", "expectation", "expected_text"),
    [
        (
            "torchlens_newer",
            lambda manifest: manifest.__setitem__("torchlens_version", "999.0.0"),
            "warning",
            "999.0.0",
        ),
        (
            # A writer from an earlier major release refuses at the load door:
            # one typed error naming the saver.
            "torchlens_earlier_major",
            lambda manifest: manifest.__setitem__("torchlens_version", "0.0.1"),
            "raise",
            "does not read files saved by torchlens 0.x",
        ),
        (
            # A format 3 manifest stamped by a writer older than the first TorchLens that
            # writes format 3 (the 2.38.0 floor) is an impossible pairing: below the
            # floor, before identity.
            "torchlens_2x_writer",
            lambda manifest: manifest.__setitem__("torchlens_version", "2.37.1"),
            "raise",
            "saved by torchlens 2.37",
        ),
        (
            # An older writer at or above the 2.38.0 floor (a pre-release of the
            # floor release) loads and is logged.
            "torchlens_older_supported",
            lambda manifest: manifest.__setitem__("torchlens_version", "2.38.0rc1"),
            "info_log",
            "2.38.0rc1",
        ),
        (
            "python_major_mismatch",
            lambda manifest: manifest.__setitem__("python_version", "999.0.0"),
            "python_major_mismatch",
            "999.0.0",
        ),
    ],
)
def test_bundle_version_policy_rows(
    tmp_path: Path,
    scenario: str,
    mutate_manifest: Callable[[dict[str, Any]], Any],
    expectation: str,
    expected_text: str | Callable[[], str] | None,
) -> None:
    """Fork F version and integrity policy rows should behave as specified."""

    bundle_path, _ = _save_bundle(tmp_path, path_name=f"{scenario}.tl")
    manifest = _read_manifest(bundle_path)
    mutate_manifest(manifest)
    _write_manifest(bundle_path, manifest)
    resolved_expected_text = expected_text() if callable(expected_text) else expected_text

    if expectation == "python_major_mismatch":
        # The 3.0 artifact is JSON plus safetensors: a Python major mismatch is advisory
        # (the 2.x refusal came from its pickle failing under the other interpreter).
        with pytest.warns(UserWarning, match="python_version=999.0.0"):
            loaded = load(bundle_path)
        assert loaded.source.class_name == "_ConvBundleModel"
        return

    if expectation == "extra_blob_warning":
        (bundle_path / "blobs" / "extra.bin").write_bytes(b"extra")
        with pytest.warns(UserWarning, match="unreferenced extra files"):
            loaded = load(bundle_path)
        assert loaded.source.class_name == "_ConvBundleModel"
        return

    if expectation == "raise":
        with pytest.raises(ArtifactError, match=resolved_expected_text or ""):
            load(bundle_path)
        return

    if expectation == "warning":
        with pytest.warns(UserWarning, match=resolved_expected_text or ""):
            loaded = load(bundle_path)
        assert loaded.source.class_name == "_ConvBundleModel"
        return

    loaded = load(bundle_path)
    assert loaded.source.class_name == "_ConvBundleModel"


def test_bundle_load_raises_for_missing_safetensors_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load should wrap backend import failures with an install hint."""

    bundle_path, _ = _save_bundle(tmp_path)

    def _raise_backend_import(*_args: Any, **_kwargs: Any) -> Any:
        raise ImportError("missing backend")

    monkeypatch.setattr("torchlens._io.lazy.safe_open", _raise_backend_import)

    with pytest.raises(ArtifactError, match="Install safetensors>=0.4"):
        load(bundle_path)


def test_bundle_load_raises_on_corrupt_manifest(tmp_path: Path) -> None:
    """Non-JSON manifests should fail with ``ArtifactError``."""

    bundle_path, _ = _save_bundle(tmp_path)
    (bundle_path / "manifest.json").write_text("{not valid json", encoding="utf-8")

    with pytest.raises(ArtifactError) as excinfo:
        load(bundle_path)
    assert excinfo.value.fields["code"] == "manifest_unreadable"


def test_bundle_load_raises_on_missing_blob_file(tmp_path: Path) -> None:
    """Manifest entries without corresponding files should fail and name the blob path."""

    bundle_path, _ = _save_bundle(tmp_path)
    blob_path = _first_blob_path(bundle_path)
    blob_path.unlink()

    with pytest.raises(ArtifactError, match=blob_path.name) as excinfo:
        load(bundle_path)
    assert excinfo.value.fields["code"] == "bundle_blob_missing"


def test_bundle_load_raises_on_truncated_safetensors(tmp_path: Path) -> None:
    """Truncated blob files should fail bundle load."""

    bundle_path, _ = _save_bundle(tmp_path)
    blob_path = _first_blob_path(bundle_path)
    blob_path.write_bytes(blob_path.read_bytes()[:16])

    with pytest.raises(ArtifactError) as excinfo:
        load(bundle_path)
    assert excinfo.value.fields["code"] == "bundle_metadata_integrity_refused"


def test_bundle_load_raises_on_checksum_tamper(tmp_path: Path) -> None:
    """Blob checksum tampering should be detected eagerly."""

    bundle_path, _ = _save_bundle(tmp_path)
    blob_path = _first_blob_path(bundle_path)
    _corrupt_blob_byte(blob_path)

    with pytest.raises(ArtifactError, match=blob_path.name) as excinfo:
        load(bundle_path)
    assert excinfo.value.fields["code"] == "bundle_metadata_integrity_refused"


def test_bundle_save_overwrite_false_refuses_typed(tmp_path: Path) -> None:
    """Existing target paths refuse ``bundle_path_exists`` unless ``overwrite=True``."""

    bundle_path, trace = _save_bundle(tmp_path)

    with pytest.raises(ArtifactError) as excinfo:
        trace.save(bundle_path)

    assert excinfo.value.fields.get("code") == "bundle_path_exists"
    assert str(excinfo.value).startswith(f"Bundle path already exists: {bundle_path}")
    assert "overwrite=True" in excinfo.value.fields.get("remedy", "")
    assert excinfo.value.__cause__ is None


def test_bundle_save_overwrite_typeerror_preserves_original_and_marks_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare ``TypeError`` writing JSON records must preserve the original bundle."""

    bundle_path, first_log = _save_bundle(tmp_path, seed=0)
    second_log = _build_conv_log(seed=1)

    def _poisoned_metadata_dump(*_args: Any, **_kwargs: Any) -> None:
        raise TypeError("simulated record encoding failure")

    monkeypatch.setattr("torchlens._io.format3.writer.write_record_files", _poisoned_metadata_dump)

    with pytest.raises(ArtifactError) as excinfo:
        second_log.save(bundle_path, overwrite=True)

    assert isinstance(excinfo.value.__cause__, TypeError)

    # The original (pre-overwrite) bundle must survive the failed overwrite, but
    # tl.load refuses it while the failed save's PARTIAL staging dir is beside it.
    assert bundle_path.exists()
    assert [p for p in tmp_path.iterdir() if ".bak." in p.name] == []  # before cleanup_tmp sweeps
    (tmp_dir,) = [p for p in tmp_path.iterdir() if p.name.startswith(f"{bundle_path.name}.tmp.")]
    assert (tmp_dir / "PARTIAL").exists()
    with pytest.raises(ArtifactError) as refused:
        load(bundle_path)
    assert refused.value.fields["code"] == "newer_save_unfinished"
    assert cleanup_tmp(bundle_path) == [tmp_dir, bundle_path / "NEWER_SAVE_UNFINISHED.json"]
    restored = load(bundle_path)
    first_output = first_log._layer_list[-1].out
    restored_output = restored._layer_list[-1].out
    assert isinstance(first_output, torch.Tensor)
    assert isinstance(restored_output, torch.Tensor)
    assert torch.equal(first_output, restored_output)

    # No undocumented "<name>.bak.<uuid>" sibling should be left behind --
    # the failed-overwrite path must restore it back onto ``bundle_path``.
    siblings = [p for p in tmp_path.iterdir() if p != bundle_path]
    bak_dirs = [p for p in siblings if ".bak." in p.name]
    assert bak_dirs == []

    # The PARTIAL staging dir was swept by ``cleanup_tmp()`` (no ``force=True``).
    assert not tmp_dir.exists()
    assert [p for p in siblings if ".tmp." in p.name] == []


class _NeverEnumeratedError(Exception):
    """Stand-in for a hypothetical future third-party writer failure mode.

    Deliberately NOT a subclass of ``ArtifactError``, ``BackendPayloadUnsupportedError``,
    or any member of ``save()``'s historical ``(ImportError, OSError, TypeError, ValueError,
    pickle.PickleError)`` tuple, so a test injecting it proves the catch-all safety net
    closes the *class* of "unenumerated exception strands the backup" bug rather than
    merely covering the one exception type (``KeyError``) that happened to surface it.
    """


# shrink-r4: rows kept carry a sole library arc or a campaign kill; the others were proven redundant (slice 1 evidence)
@pytest.mark.parametrize(
    "injected_exception",
    [_NeverEnumeratedError("simulated brand-new writer failure")],
    ids=["unenumerated-exception-type"],
)
def test_bundle_save_overwrite_arbitrary_exception_preserves_original_and_marks_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    injected_exception: Exception,
) -> None:
    """ANY exception mid-write during ``save(overwrite=True)`` must not strand the backup.

    Regression test closing the recurring bug *class*, not just one dtype. Rounds 3-7
    hardened specific exception types one at a time (most recently ``TypeError`` in
    ``ddd9440f``); cert round 8 found a raw ``KeyError`` from
    ``safetensors.torch.save_file()`` (triggered by a ``complex128`` tensor wrongly
    allow-listed in ``tensor_policy``) sailing straight past that hand-enumerated except
    tuple, leaving the live bundle path *missing* after a failed overwrite with only an
    unrestored ``.bak.<uuid>`` copy surviving -- and no automatic recovery. ``save()`` now
    has a ``BaseException`` catch-all after the specific branches, so the same
    backup-restore + ``PARTIAL``-sentinel recovery fires for literally any exception type,
    known or not yet discovered (the parametrized ``_NeverEnumeratedError`` stands in for
    "not yet discovered").
    """

    bundle_path, first_log = _save_bundle(tmp_path, seed=0)
    second_log = _build_conv_log(seed=1)

    def _poisoned_metadata_dump(*_args: Any, **_kwargs: Any) -> None:
        raise injected_exception

    monkeypatch.setattr("torchlens._io.format3.writer.write_record_files", _poisoned_metadata_dump)

    with pytest.raises(ArtifactError) as excinfo:
        second_log.save(bundle_path, overwrite=True)

    assert excinfo.value.__cause__ is injected_exception

    # The original (pre-overwrite) bundle must survive the failed overwrite, but
    # tl.load refuses it while the failed save's PARTIAL staging dir is beside it,
    # regardless of which exception type the write raised.
    assert bundle_path.exists()
    assert [p for p in tmp_path.iterdir() if ".bak." in p.name] == []  # before cleanup_tmp sweeps
    (tmp_dir,) = [p for p in tmp_path.iterdir() if p.name.startswith(f"{bundle_path.name}.tmp.")]
    assert (tmp_dir / "PARTIAL").exists()
    with pytest.raises(ArtifactError) as refused:
        load(bundle_path)
    assert refused.value.fields["code"] == "newer_save_unfinished"
    assert cleanup_tmp(bundle_path) == [tmp_dir, bundle_path / "NEWER_SAVE_UNFINISHED.json"]
    restored = load(bundle_path)
    first_output = first_log._layer_list[-1].out
    restored_output = restored._layer_list[-1].out
    assert isinstance(first_output, torch.Tensor)
    assert isinstance(restored_output, torch.Tensor)
    assert torch.equal(first_output, restored_output)

    # No undocumented "<name>.bak.<uuid>" sibling should be left behind.
    siblings = [p for p in tmp_path.iterdir() if p != bundle_path]
    bak_dirs = [p for p in siblings if ".bak." in p.name]
    assert bak_dirs == []

    # The PARTIAL staging dir was swept by ``cleanup_tmp()`` (no ``force=True``).
    assert not tmp_dir.exists()
    assert [p for p in siblings if ".tmp." in p.name] == []


class _LegacyPicklePlaceholder:
    """Trivial module-level class used to force ``find_class`` during unpickling."""

    def __init__(self, value: int) -> None:
        self.value = value


def test_legacy_multi_trace_bundle_refuses_before_unpickling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 3.0 load door refuses 2.x bundles before reaching any unpickler."""

    bundle_path = tmp_path / "legacy_bundle.tl"
    bundle_path.mkdir()
    (bundle_path / "manifest.json").write_text(
        json.dumps({"kind": "bundle", "tlspec_version": TLSPEC_VERSION}), encoding="utf-8"
    )
    (bundle_path / "metadata.pkl").write_bytes(pickle.dumps(_LegacyPicklePlaceholder(1)))

    reached_unpickler = False

    def _poisoned_unpickler(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal reached_unpickler
        reached_unpickler = True
        raise AssertionError("2.x artifacts must not reach an unpickler")

    monkeypatch.setattr(bundle_io, "_metadata_unpickler", _poisoned_unpickler)
    monkeypatch.setattr(bundle_io._RenameAwareUnpickler, "load", _poisoned_unpickler)

    with pytest.raises(ArtifactError) as excinfo:
        load(bundle_path)

    assert excinfo.value.fields["code"] == "artifact_version_below_floor"
    assert "2.36.x" in str(excinfo.value)
    assert excinfo.value.__cause__ is None
    assert not reached_unpickler


def test_bundle_invalid_record_json_refuses_before_unpickling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A correctly digested, invalid JSON record refuses before object construction."""

    bundle_path, trace = _save_bundle(tmp_path)
    reached_unpickler = False

    def _poisoned_unpickler(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal reached_unpickler
        reached_unpickler = True
        raise AssertionError("JSON artifacts must not reach an unpickler")

    monkeypatch.setattr(bundle_io, "_metadata_unpickler", _poisoned_unpickler)
    monkeypatch.setattr(bundle_io._RenameAwareUnpickler, "load", _poisoned_unpickler)
    assert load(bundle_path)._num_ops == trace._num_ops
    assert not reached_unpickler
    relpath = _read_manifest(bundle_path)["records"]["trace"]["path"]
    (bundle_path / relpath).write_text("{not valid json", encoding="utf-8")
    redigest(bundle_path, relpath)

    with pytest.raises(ArtifactError) as excinfo:
        load(bundle_path)

    assert excinfo.value.fields["code"] == "bundle_metadata_integrity_refused"
    assert not reached_unpickler


def test_bundle_save_raw_input_stringifies_unpicklable_value_keeps_tensors(
    tmp_path: Path,
) -> None:
    """``save_raw_input=True`` must stringify un-picklable objects, not pickle them raw.

    Unmocked, public-API-only repro of the root cause behind the BLOCKER
    above: ``save_raw_input=True``/``save_transformed_output=True`` (documented
    top-level ``tl.trace()`` kwargs) previously routed raw values through
    the unsafe ``_scrub_value()`` passthrough, so a generator (or any other
    live, un-picklable object) nested inside a raw input reached
    ``pickle.dump()`` unmodified. Now the raw-value scrub path stringifies
    genuinely un-picklable leaves while still fully retaining picklable
    ones (e.g. tensors), matching the documented "stores the full object"
    behavior for the common case.
    """

    def make_generator() -> Any:
        yield 1

    model = nn.Linear(3, 3)
    raw_input = {"tensor": torch.randn(3, 3), "meta": make_generator()}
    trace = trace_fn(
        model,
        raw_input,
        capture=tl.options.CaptureOptions(
            input_transform=lambda d: d["tensor"], save_raw_input=True, seed=0
        ),
    )
    assert isinstance(trace.input["meta"], type(make_generator()))

    bundle_path = tmp_path / "raw_input_bundle.tl"
    trace.save(bundle_path)

    loaded = load(bundle_path)
    assert isinstance(loaded.input["tensor"], torch.Tensor)
    assert torch.equal(loaded.input["tensor"], raw_input["tensor"])
    assert isinstance(loaded.input["meta"], UnencodableValue)
    assert loaded.input["meta"].type_name == "builtins.generator"
    assert "input" in loaded.input["meta"].at


def test_bundle_save_raw_input_large_tensor_skips_picklability_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A large tensor in ``save_raw_input=True`` must not be pickle-probed twice.

    Regression test for a MAJOR cost regression: ``_is_safely_picklable``'s
    ``stringify_unknown`` probe called ``pickle.dumps()`` on every unspecced
    value reached via ``save_raw_input``/``save_raw_output`` -- including
    large tensors, which is the entire point of choosing the ``True`` "full
    retention" policy over the default bounded ``"small"`` policy -- doubling
    CPU cost and transiently doubling peak memory. ``torch.Tensor`` and
    ``numpy.ndarray`` are known-serializable via their own ``__reduce_ex__``
    (unlike generators/locks/file handles), so the probe should be skipped
    for them entirely while the value still round-trips exactly.
    """

    tensor_dumps_calls = 0
    real_dumps = pickle.dumps

    def _counting_dumps(value: Any, *args: Any, **kwargs: Any) -> bytes:
        nonlocal tensor_dumps_calls
        if isinstance(value, torch.Tensor):
            tensor_dumps_calls += 1
        return real_dumps(value, *args, **kwargs)

    monkeypatch.setattr(pickle, "dumps", _counting_dumps)

    model = nn.Linear(3, 3)
    large_tensor = torch.randn(200, 200)
    raw_input = {"tensor": large_tensor, "x": torch.randn(1, 3)}
    trace = trace_fn(
        model,
        raw_input,
        capture=tl.options.CaptureOptions(
            input_transform=lambda d: d["x"], save_raw_input=True, seed=0
        ),
    )

    bundle_path = tmp_path / "raw_input_large_tensor_bundle.tl"
    trace.save(bundle_path)

    assert tensor_dumps_calls == 0

    loaded = load(bundle_path)
    assert isinstance(loaded.input["tensor"], torch.Tensor)
    assert torch.equal(loaded.input["tensor"], large_tensor)


def test_bundle_save_raw_input_numeric_ndarray_skips_picklability_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A numeric-dtype ``numpy.ndarray`` in ``save_raw_input=True`` must not be
    pickle-probed twice (the H4 perf win preserved for the numeric-array case).
    """

    ndarray_dumps_calls = 0
    real_dumps = pickle.dumps

    def _counting_dumps(value: Any, *args: Any, **kwargs: Any) -> bytes:
        nonlocal ndarray_dumps_calls
        if isinstance(value, np.ndarray):
            ndarray_dumps_calls += 1
        return real_dumps(value, *args, **kwargs)

    monkeypatch.setattr(pickle, "dumps", _counting_dumps)

    model = nn.Linear(3, 3)
    numeric_array = np.arange(100, dtype=np.float64).reshape(10, 10)
    raw_input = {"array": numeric_array, "x": torch.randn(1, 3)}
    trace = trace_fn(
        model,
        raw_input,
        capture=tl.options.CaptureOptions(
            input_transform=lambda d: d["x"], save_raw_input=True, seed=0
        ),
    )

    bundle_path = tmp_path / "raw_input_numeric_ndarray_bundle.tl"
    trace.save(bundle_path)

    assert ndarray_dumps_calls == 0

    loaded = load(bundle_path)
    assert isinstance(loaded.input["array"], np.ndarray)
    np.testing.assert_array_equal(loaded.input["array"], numeric_array)


def test_bundle_save_raw_input_object_dtype_ndarray_stringifies_unpicklable_element(
    tmp_path: Path,
) -> None:
    """An object-dtype ``numpy.ndarray`` holding a live unpicklable element must
    still be probed and gracefully degraded, not blindly exempted.

    Regression test for the round-7 MAJOR: the round-6 H4 fix exempted ALL
    ``numpy.ndarray`` instances from the picklability probe, which is correct
    for numeric/bool/string dtypes but wrong for ``dtype=object`` arrays --
    those can hold arbitrary live Python objects (generators, locks, ...)
    just like a plain list/dict, so exempting them reintroduces the hard
    ``save()`` failure the probe exists to prevent.
    """

    def make_generator() -> Any:
        yield 1

    model = nn.Linear(3, 3)
    object_array = np.array([make_generator(), 1, "text"], dtype=object)
    raw_input = {"array": object_array, "x": torch.randn(1, 3)}
    trace = trace_fn(
        model,
        raw_input,
        capture=tl.options.CaptureOptions(
            input_transform=lambda d: d["x"], save_raw_input=True, seed=0
        ),
    )

    bundle_path = tmp_path / "raw_input_object_ndarray_bundle.tl"
    # Must NOT raise -- the unpicklable element degrades gracefully instead
    # of reaching the real pickle.dump() over the full scrubbed state.
    trace.save(bundle_path)

    loaded = load(bundle_path)
    assert isinstance(loaded.input["array"], UnencodableValue)
    assert loaded.input["array"].type_name == "numpy.ndarray"
    assert "input" in loaded.input["array"].at


def test_bundle_save_raw_input_structured_ndarray_object_field_stringifies_unpicklable_element(
    tmp_path: Path,
) -> None:
    """A structured/record ``numpy.ndarray`` with an object-dtype field holding a
    live unpicklable element must still be probed and gracefully degraded.

    Regression test for the cert7 MAJOR: the round-7 fix's own guard
    (``value.dtype != np.dtype("object")``) is a top-level dtype-identity
    check, so a structured/record dtype whose *field* is object-typed
    (e.g. ``np.dtype([('a', object), ('b', 'i4')])``) is never itself equal
    to ``np.dtype("object")`` and wrongly slips past the exemption even
    though it genuinely embeds live object references in the ``'a'``
    field's storage -- reintroducing the exact ``save()`` failure the
    round-6/round-7 fixes were written to close, one door narrower. The fix
    is ``not value.dtype.hasobject`` (numpy's recursive containment check,
    a strict superset of the identity comparison it replaces).
    """

    def make_generator() -> Any:
        yield 1

    record_dtype = np.dtype([("a", object), ("b", "i4")])
    structured_array = np.array([(make_generator(), 1)], dtype=record_dtype)
    assert structured_array.dtype != np.dtype("object")
    assert structured_array.dtype.hasobject

    model = nn.Linear(3, 3)
    raw_input = {"array": structured_array, "x": torch.randn(1, 3)}
    trace = trace_fn(
        model,
        raw_input,
        capture=tl.options.CaptureOptions(
            input_transform=lambda d: d["x"], save_raw_input=True, seed=0
        ),
    )

    bundle_path = tmp_path / "raw_input_structured_ndarray_bundle.tl"
    # Must NOT raise -- the unpicklable field element degrades gracefully
    # instead of reaching the real pickle.dump() over the full scrubbed state.
    trace.save(bundle_path)

    loaded = load(bundle_path)
    assert isinstance(loaded.input["array"], UnencodableValue)
    assert loaded.input["array"].type_name == "numpy.ndarray"
    assert "input" in loaded.input["array"].at


def test_bundle_save_rejects_symlink_target(tmp_path: Path) -> None:
    """Bundle save should refuse symlinked target paths."""

    trace = _build_conv_log()
    real_path = tmp_path / "real_bundle"
    real_path.mkdir()
    symlink_path = tmp_path / "bundle_link"
    symlink_path.symlink_to(real_path, target_is_directory=True)

    with pytest.raises(ArtifactError, match="symlinked save target"):
        trace.save(symlink_path)


def _build_unified_bundle(tmp_path: Path) -> Path:
    """Save one valid unified ``.tlspec`` bundle for symlink-guard tests.

    Parameters
    ----------
    tmp_path:
        Base temporary directory.

    Returns
    -------
    Path
        Saved bundle directory that detects as ``v3``.
    """

    trace = _build_conv_log()
    bundle_path = tmp_path / "unified_bundle.tlspec"
    trace.save(bundle_path)
    assert detect_tlspec_format(bundle_path) == "v3"
    return bundle_path


def test_unified_load_rejects_symlinked_record(tmp_path: Path) -> None:
    """Unified loads should refuse a symlinked JSON record member."""

    bundle_path = _build_unified_bundle(tmp_path)
    member = bundle_path / _read_manifest(bundle_path)["records"]["trace"]["path"]
    target = tmp_path / "real_trace.json"
    member.rename(target)
    member.symlink_to(target)

    with pytest.raises(ArtifactError) as excinfo:
        load(bundle_path)
    assert excinfo.value.fields["code"] == "load_path_symlink_rejected"


def test_unified_load_rejects_symlinked_manifest(tmp_path: Path) -> None:
    """Unified loads should refuse a symlinked ``manifest.json`` member."""

    bundle_path = _build_unified_bundle(tmp_path)
    member = bundle_path / "manifest.json"
    target = tmp_path / "real_manifest.json"
    member.rename(target)
    member.symlink_to(target)

    # r29 F-r28-1: a symlinked manifest is now rejected even earlier, at
    # format-detection (before the loader's own symlink guard), so match the
    # symlink-refusal message point-agnostically -- the assertion still verifies
    # that a symlinked manifest.json member is refused with ArtifactError.
    with pytest.raises(ArtifactError, match="Refusing symlinked"):
        load(bundle_path)


def test_unified_load_rejects_symlinked_blobs_dir(tmp_path: Path) -> None:
    """Unified loads should refuse a symlinked ``blobs`` directory."""

    bundle_path = _build_unified_bundle(tmp_path)
    member = bundle_path / "blobs"
    target = tmp_path / "real_blobs"
    member.rename(target)
    member.symlink_to(target, target_is_directory=True)

    with pytest.raises(ArtifactError) as excinfo:
        load(bundle_path)
    assert excinfo.value.fields["code"] == "load_path_symlink_rejected"


def test_resolve_bundle_blob_path_rejects_symlinked_blobs_dir(tmp_path: Path) -> None:
    """The blob resolver should reject the ``blobs`` directory itself being a symlink."""

    bundle_root = tmp_path / "root.tlspec"
    bundle_root.mkdir()
    outside_tree = tmp_path / "outside_tree"
    outside_tree.mkdir()
    (outside_tree / "0000000001.safetensors").write_bytes(b"payload")
    (bundle_root / "blobs").symlink_to(outside_tree, target_is_directory=True)

    with pytest.raises(ArtifactError, match="Refusing symlinked blobs directory"):
        resolve_bundle_blob_path(bundle_root, "blobs/0000000001.safetensors")


def test_bundle_save_rejects_non_tensor_activation_transform_output(tmp_path: Path) -> None:
    """Save should fail before writing blobs when activation_transform returned a non-tensor."""

    trace = _build_non_tensor_out_log()
    bundle_path = tmp_path / "non_tensor_bundle.tl"

    with pytest.raises(ArtifactError, match="activation_transform outputs"):
        trace.save(bundle_path)

    assert not bundle_path.exists()


def test_cleanup_tmp_removes_redundant_backup_when_bundle_exists(tmp_path: Path) -> None:
    """A ``.bak.*`` sibling that is byte-identical to the live bundle is provably redundant."""

    target_path = tmp_path / "bundle.tl"
    target_path.mkdir()
    (target_path / "marker").write_text("current", encoding="utf-8")
    bak_path = tmp_path / f"{target_path.name}.bak.deadbeef"
    bak_path.mkdir()
    (bak_path / "marker").write_text("current", encoding="utf-8")

    removed = cleanup_tmp(target_path)

    assert removed == [bak_path]
    assert not bak_path.exists()
    assert target_path.exists()
    assert (target_path / "marker").read_text(encoding="utf-8") == "current"


def test_cleanup_tmp_force_removes_distinct_backup_when_bundle_exists(tmp_path: Path) -> None:
    """``force=True`` still allows removing a non-provably-redundant backup."""

    target_path = tmp_path / "bundle.tl"
    target_path.mkdir()
    (target_path / "marker").write_text("current", encoding="utf-8")
    bak_path = tmp_path / f"{target_path.name}.bak.deadbeef"
    bak_path.mkdir()
    (bak_path / "marker").write_text("stale-but-distinct", encoding="utf-8")

    with pytest.warns(UserWarning, match="Force-removed"):
        removed = cleanup_tmp(target_path, force=True)

    assert removed == [bak_path]
    assert not bak_path.exists()


def test_cleanup_tmp_does_not_destroy_second_distinct_orphaned_backup(tmp_path: Path) -> None:
    """Two DISTINCT orphaned ``.bak.*`` dirs for the same target: restoring the
    first must not cause the second, unrelated backup to be silently deleted.

    Regression test for the round-7 BLOCKER: the round-6 H4 orphan-.bak sweep
    inferred "redundant" purely from ``bundle_path.exists()`` becoming true
    mid-loop (after an earlier iteration restored a sibling), with no content
    check -- so a second, independently-orphaned backup holding genuinely
    different data was silently destroyed.
    """

    target_path = tmp_path / "bundle.tl"
    bak_path_1 = tmp_path / f"{target_path.name}.bak.aaaaaaaa"
    bak_path_1.mkdir()
    (bak_path_1 / "marker").write_text("first-orphan", encoding="utf-8")
    bak_path_2 = tmp_path / f"{target_path.name}.bak.bbbbbbbb"
    bak_path_2.mkdir()
    (bak_path_2 / "marker").write_text("second-orphan-distinct", encoding="utf-8")

    with pytest.warns(UserWarning, match="not provably redundant"):
        removed = cleanup_tmp(target_path)

    # Exactly one orphan was restored onto the missing bundle path; the other,
    # content-distinct orphan must survive untouched.
    assert removed == [target_path]
    assert target_path.exists()
    restored_marker = (target_path / "marker").read_text(encoding="utf-8")
    assert restored_marker in {"first-orphan", "second-orphan-distinct"}

    surviving = [p for p in (bak_path_1, bak_path_2) if p.exists()]
    assert len(surviving) == 1
    surviving_marker = (surviving[0] / "marker").read_text(encoding="utf-8")
    # The surviving backup must be the one that was NOT restored, and its
    # data must be intact (not silently destroyed).
    assert surviving_marker != restored_marker
    assert surviving_marker in {"first-orphan", "second-orphan-distinct"}


def test_array_transport_admits_both_complex_widths() -> None:
    """Both complex widths have a physical real-view encoding in the format 3 carrier."""

    for dtype in (np.complex64, np.complex128):
        array = np.array([1 + 2j], dtype=dtype)
        assert _unsupported_array_dtype_reason(array) is None
        _raise_for_unsupported_array_dtype(array, backend_name="numpy")
