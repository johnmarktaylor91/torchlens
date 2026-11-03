"""Tests for portable TorchLens directory bundle save/load."""

from __future__ import annotations

import json
import logging
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import safetensors  # noqa: F401
import torch
from torch import nn

import torchlens as tl
from torchlens import Trace, func, load, trace as trace_fn
from torchlens._io import (
    TLSPEC_VERSION,
    ArtifactError,
)
from torchlens._io.manifest import Manifest
from torchlens._io.paths import resolve_bundle_blob_path
from torchlens._io.payload_codec import (
    _raise_for_unsupported_array_dtype,
    _unsupported_array_dtype_reason,
)
from torchlens.io import cleanup_tmp, detect_tlspec_format


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

    manifest = Manifest.read(bundle_path / "manifest.json")
    return bundle_path / manifest.tensors[0].relative_path


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
    manifest = Manifest.read(bundle_path / "manifest.json")

    assert manifest.provenance is not None
    assert manifest.provenance.git_commit_hash is None


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


def test_bundle_save_strict_false_records_unsupported_tensors(tmp_path: Path) -> None:
    """Best-effort save should skip unsupported tensors and record them in the manifest."""

    trace = _build_sparse_out_log()
    bundle_path = tmp_path / "sparse_bundle.tl"

    trace.save(bundle_path, strict=False)

    manifest = Manifest.read(bundle_path / "manifest.json")
    assert manifest.unsupported_tensors
    assert all(entry["kind"] == "out" for entry in manifest.unsupported_tensors)
    assert all("sparse" in entry["reason"] for entry in manifest.unsupported_tensors)


def test_bundle_save_strict_default_raises_on_complex128_tensor(tmp_path: Path) -> None:
    """Strict bundle save must reject ``complex128`` with a clean error, not a raw ``KeyError``.

    Regression test for a data-loss BLOCKER (cert round 8): ``torch.complex128`` was
    listed in ``tensor_policy._SUPPORTED_DTYPES`` even though the actual writer,
    ``safetensors.torch.save_file()``, has no dtype-size-table entry for it. A
    ``complex128`` tensor therefore sailed straight past the allow-list tripwire that
    exists precisely to keep unsupported tensors out of ``save_file()`` and crashed with
    a raw, unwrapped ``KeyError`` from deep inside the third-party library -- bypassing
    every save-path hardening fix from cert rounds 3-7 entirely, since ``KeyError`` was
    never in any of their except tuples. The fix drops ``complex128`` from the allow-list
    so this now fails cleanly, before ``save_file()`` is ever reached.
    """

    trace = _build_complex_out_log(dtype=torch.complex128)
    bundle_path = tmp_path / "complex128_bundle.tl"

    with pytest.raises(ArtifactError, match="complex128"):
        trace.save(bundle_path)

    # The strict allow-list check fires before ``safetensors.torch.save_file()``
    # is ever reached, so no raw ``KeyError`` escapes and ``bundle_path`` itself
    # is never created. A leftover ``.tmp.*`` dir is expected (standard
    # fresh-save failure contract) and must carry the PARTIAL sentinel so
    # ``cleanup_tmp()`` can sweep it.
    assert not bundle_path.exists()
    tmp_dirs = [p for p in tmp_path.iterdir() if ".tmp." in p.name]
    assert tmp_dirs
    for tmp_dir in tmp_dirs:
        assert (tmp_dir / "PARTIAL").exists()
    removed = cleanup_tmp(bundle_path)
    assert set(removed) == set(tmp_dirs)
    assert not any(p.exists() for p in tmp_dirs)


def test_bundle_save_strict_false_records_complex128_as_unsupported(tmp_path: Path) -> None:
    """Best-effort save should skip ``complex128`` tensors and record them, not crash."""

    trace = _build_complex_out_log(dtype=torch.complex128)
    bundle_path = tmp_path / "complex128_bundle.tl"

    trace.save(bundle_path, strict=False)

    manifest = Manifest.read(bundle_path / "manifest.json")
    assert manifest.unsupported_tensors
    assert all(entry["kind"] == "out" for entry in manifest.unsupported_tensors)
    assert all("complex128" in entry["reason"] for entry in manifest.unsupported_tensors)


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
            # An older writer of the same major release loads and is logged.
            "torchlens_older_supported",
            lambda manifest: manifest.__setitem__("torchlens_version", "2.36.1"),
            "info_log",
            "2.36.1",
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
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
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
        (bundle_path / "metadata.pkl").write_bytes(b"not a pickle")
        with pytest.warns(UserWarning, match="python_version=999.0.0"):
            with pytest.raises(ArtifactError, match="python_version=999.0.0"):
                load(bundle_path)
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

    if expectation == "info_log":
        with caplog.at_level(logging.INFO, logger="torchlens._io.manifest"):
            loaded = load(bundle_path)
        assert loaded.source.class_name == "_ConvBundleModel"
        assert resolved_expected_text is not None
        assert resolved_expected_text in caplog.text
        assert "older than runtime torchlens_version" in caplog.text
        return

    loaded = load(bundle_path)
    assert loaded.source.class_name == "_ConvBundleModel"


def test_bundle_load_raises_for_missing_safetensors_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load should wrap backend import failures with an install hint."""

    bundle_path, _ = _save_bundle(tmp_path)

    def _raise_backend_import(path: Path, device: str) -> dict[str, torch.Tensor]:
        raise ImportError("missing backend")

    monkeypatch.setattr("torchlens._io.bundle.load_file", _raise_backend_import)

    with pytest.raises(ArtifactError, match="Install safetensors>=0.4"):
        load(bundle_path)


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
    """A bare ``TypeError`` from ``pickle.dump`` must not destroy the original bundle.

    Regression test for a data-loss BLOCKER: ``pickle.dump()`` raises a bare
    ``TypeError`` (not the ``pickle.PickleError`` subclass) for many live
    objects, e.g. ``TypeError: cannot pickle 'generator' object``. Before the
    fix, ``save()``'s exception handler only caught
    ``(ImportError, OSError, ValueError, pickle.PickleError)``, so a
    ``TypeError`` propagated straight past every handler in the function --
    skipping both the ``PARTIAL`` sentinel (leaving the ``.tmp`` dir
    un-sweepable by ``cleanup_tmp()``) and the pre-overwrite backup restore
    (permanently losing the original bundle under an undocumented
    ``.bak.<uuid>`` directory name with a bare ``TypeError`` propagating to
    the caller instead of a clean ``ArtifactError``).
    """

    bundle_path, first_log = _save_bundle(tmp_path, seed=0)
    second_log = _build_conv_log(seed=1)

    def _poisoned_metadata_dump(*_args: Any, **_kwargs: Any) -> None:
        raise TypeError("simulated live-resource pickling failure")

    # ``save()`` writes metadata.pkl through ``dump_canonical_metadata()``
    # (B3R4-R21-2 canonical container bytes), which drives a ``pickle._Pickler``
    # subclass -- patching bare ``pickle.dump`` would no longer intercept it.
    monkeypatch.setattr("torchlens._io.bundle.dump_canonical_metadata", _poisoned_metadata_dump)

    with pytest.raises(ArtifactError) as excinfo:
        second_log.save(bundle_path, overwrite=True)

    assert isinstance(excinfo.value.__cause__, TypeError)

    # The original (pre-overwrite) bundle must survive the failed overwrite.
    assert bundle_path.exists()
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

    # Any leftover ``.tmp.<uuid>`` dir must carry the PARTIAL sentinel so
    # ``cleanup_tmp()`` (without ``force=True``) can sweep it.
    tmp_dirs = [p for p in siblings if ".tmp." in p.name]
    for tmp_dir in tmp_dirs:
        assert (tmp_dir / "PARTIAL").exists()
    removed = cleanup_tmp(bundle_path)
    assert set(removed) == set(tmp_dirs)
    assert not any(p.exists() for p in tmp_dirs)


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

    # Injected at the metadata.pkl writer (``dump_canonical_metadata()``,
    # B3R4-R21-2): bare ``pickle.dump`` is no longer on the save path.
    monkeypatch.setattr("torchlens._io.bundle.dump_canonical_metadata", _poisoned_metadata_dump)

    with pytest.raises(ArtifactError) as excinfo:
        second_log.save(bundle_path, overwrite=True)

    assert excinfo.value.__cause__ is injected_exception

    # The original (pre-overwrite) bundle must survive the failed overwrite,
    # regardless of which exception type the write raised.
    assert bundle_path.exists()
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

    # Any leftover ``.tmp.<uuid>`` dir must carry the PARTIAL sentinel so
    # ``cleanup_tmp()`` (without ``force=True``) can sweep it.
    tmp_dirs = [p for p in siblings if ".tmp." in p.name]
    for tmp_dir in tmp_dirs:
        assert (tmp_dir / "PARTIAL").exists()
    removed = cleanup_tmp(bundle_path)
    assert set(removed) == set(tmp_dirs)
    assert not any(p.exists() for p in tmp_dirs)


class _LegacyPicklePlaceholder:
    """Trivial module-level class used to force ``find_class`` during unpickling."""

    def __init__(self, value: int) -> None:
        self.value = value


def test_legacy_multi_trace_bundle_load_typeerror_raises_torchlens_io_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare ``TypeError`` unpickling a legacy multi-trace ``Bundle`` must not escape.

    Regression test for a recurrence of the round-5 bug class:
    ``_load_unified_bundle()``'s legacy (pre-``bundle.json``) pickle-load
    path -- reached from public ``tl.load(path)`` whenever ``manifest.json``
    declares ``kind: "bundle"`` and no ``bundle.json`` sibling exists (old-
    format bundles, an explicitly supported backward-compat case) -- was
    missing ``TypeError`` from its except tuple, unlike the three sibling
    call sites round 5 fixed (``bundle.py`` save, main-trace load,
    ``streaming.py`` finalize). A bare ``TypeError`` (the same failure mode
    ``pickle`` raises for many live-resource/incompatible-global objects)
    escaped uncaught instead of the documented ``ArtifactError``.
    """

    bundle_path = tmp_path / "legacy_bundle.tl"
    bundle_path.mkdir()
    (bundle_path / "manifest.json").write_text(
        json.dumps({"kind": "bundle", "tlspec_version": TLSPEC_VERSION}), encoding="utf-8"
    )
    (bundle_path / "metadata.pkl").write_bytes(pickle.dumps(_LegacyPicklePlaceholder(1)))

    def _poisoned_find_class(*_args: Any, **_kwargs: Any) -> Any:
        raise TypeError("simulated live-resource unpickling failure")

    monkeypatch.setattr(
        "torchlens._io.bundle._RenameAwareUnpickler.find_class", _poisoned_find_class
    )

    with pytest.raises(ArtifactError) as excinfo:
        load(bundle_path)

    assert isinstance(excinfo.value.__cause__, TypeError)


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
    assert loaded.input["meta"] == "<scrubbed:generator>"


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

    monkeypatch.setattr("torchlens._io.scrub.pickle.dumps", _counting_dumps)

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
        Saved bundle directory that detects as ``v2.0_unified``.
    """

    trace = _build_conv_log()
    bundle_path = tmp_path / "unified_bundle.tlspec"
    trace.save(bundle_path)
    assert detect_tlspec_format(bundle_path) == "v2.0_unified"
    return bundle_path


def test_unified_load_rejects_symlinked_metadata(tmp_path: Path) -> None:
    """Unified loads should refuse a symlinked ``metadata.pkl`` member."""

    bundle_path = _build_unified_bundle(tmp_path)
    member = bundle_path / "metadata.pkl"
    target = tmp_path / "real_metadata.pkl"
    member.rename(target)
    member.symlink_to(target)

    with pytest.raises(ArtifactError, match="Refusing symlinked metadata"):
        load(bundle_path)


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

    with pytest.raises(ArtifactError, match="Refusing symlinked blobs directory"):
        load(bundle_path)


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


def test_unsupported_array_dtype_reason_rejects_complex128_but_allows_complex64() -> None:
    """Non-torch backends must reject ``complex128`` arrays before the shared transport.

    Regression test for the non-torch half of the cert round 8 BLOCKER:
    ``_raise_for_unsupported_array_dtype()``/``_unsupported_array_dtype_reason()`` gate
    every JAX/TF/Paddle/tinygrad/MLX payload before it reaches
    ``numpy_to_transport_tensor()`` -> ``torch.from_numpy()`` -> the same unguarded
    ``safetensors.torch.save_file()`` call the torch-native path hits. Before this fix,
    the guard only rejected numpy dtype *kinds* ``{"O", "S", "U", "V"}`` -- ``complex128``
    has kind ``"c"`` (shared with the genuinely-supported ``complex64``), so it sailed
    through unblocked. The fix distinguishes by itemsize (complex128 is 16 bytes,
    complex64 is 8) so only the unsupported width is rejected.
    """

    complex128_array = np.array([1 + 2j], dtype=np.complex128)
    complex64_array = np.array([1 + 2j], dtype=np.complex64)

    reason = _unsupported_array_dtype_reason(complex128_array)
    assert reason is not None
    assert "complex128" in reason
    assert _unsupported_array_dtype_reason(complex64_array) is None

    with pytest.raises(TypeError, match="complex128"):
        _raise_for_unsupported_array_dtype(complex128_array, backend_name="jax")

    _raise_for_unsupported_array_dtype(complex64_array, backend_name="jax")
