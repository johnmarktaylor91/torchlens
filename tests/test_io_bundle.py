"""Tests for portable TorchLens directory bundle save/load."""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import torch
from torch import nn

pytest.importorskip("safetensors")

from torchlens import Trace, load, trace as trace_fn, save
from torchlens.io import cleanup_tmp
from torchlens._io import TLSPEC_VERSION, TorchLensIOError
from torchlens._io.manifest import Manifest
from torchlens._io.payload_codec import (
    _raise_for_unsupported_array_dtype,
    _unsupported_array_dtype_reason,
)
from torchlens.data_classes.trace import ResolvedPostprocessing


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
    return trace_fn(model, x, layers_to_save="all", random_seed=seed)


def _build_transform_log(transform: Callable[[torch.Tensor], Any]) -> Trace:
    """Create a deterministic log using a custom out transform.

    Parameters
    ----------
    transform:
        Activation postprocessing function applied during logging.

    Returns
    -------
    Trace
        Logged forward pass with transformed outs.
    """

    torch.manual_seed(0)
    model = _ActivationPostfuncModel()
    x = torch.randn(2, 4)
    return trace_fn(
        model,
        x,
        layers_to_save="all",
        activation_transform=transform,
        random_seed=0,
    )


def _build_sparse_out_log() -> Trace:
    """Create a live log whose first saved out is converted to sparse post-hoc.

    Returns
    -------
    Trace
        Model log containing an unsupported sparse out tensor.
    """

    trace = _build_conv_log()
    first_saved_layer = next(layer for layer in trace.layer_list if layer.has_saved_activation)
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
    first_saved_layer = next(layer for layer in trace.layer_list if layer.has_saved_activation)
    first_saved_layer.transformed_out = 1.0
    trace.activation_transform = lambda tensor: float(tensor.mean().item())
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
    first_saved_layer = next(layer for layer in trace.layer_list if layer.has_saved_activation)
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
    save(trace, bundle_path)
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


def test_bundle_roundtrip_preserves_saved_outs_bit_exactly(tmp_path: Path) -> None:
    """Eager bundle load should restore all saved outs exactly."""

    bundle_path, live_log = _save_bundle(tmp_path)

    restored = load(bundle_path)

    live_by_label = {
        layer.layer_label: layer.out
        for layer in live_log.layer_list
        if layer.has_saved_activation and isinstance(layer.out, torch.Tensor)
    }
    restored_by_label = {
        layer.layer_label: layer.out
        for layer in restored.layer_list
        if layer.has_saved_activation and isinstance(layer.out, torch.Tensor)
    }

    assert restored._loaded_from_bundle is True
    assert isinstance(restored._source_bundle_manifest_sha256, str)
    assert live_by_label.keys() == restored_by_label.keys()
    for layer_label, live_out in live_by_label.items():
        assert torch.equal(live_out, restored_by_label[layer_label])


def test_semantic_output_fields_default_none_roundtrip(tmp_path: Path) -> None:
    """Semantic output provenance fields default to ``None`` and survive bundle load."""

    trace = _build_conv_log()

    assert trace.output_postprocessor is None
    assert trace.transform_repr is None
    assert trace.output_id2label is None
    assert trace.output_num_classes is None
    assert trace.decoded_output is None

    bundle_path = tmp_path / "semantic_defaults.tlspec"
    save(trace, bundle_path)
    restored = load(bundle_path)

    assert restored.output_postprocessor is None
    assert restored.transform_repr is None
    assert restored.output_id2label is None
    assert restored.output_num_classes is None
    assert restored.decoded_output is None


def test_resolved_postprocessing_roundtrips_through_tlspec(tmp_path: Path) -> None:
    """Resolved output postprocessing provenance is portable state."""

    trace = _build_conv_log()
    trace.output_postprocessor = ResolvedPostprocessing(
        source="hf_config",
        identifier="tiny-classifier",
        verified=True,
        config={"num_labels": 2},
        description="HF config labels",
        style="classification",
        selected_output_head="logits",
        label_source="config.id2label",
        label_source_version="test",
        confidence=0.95,
        top_n_captured=5,
        ambiguous=False,
    )
    trace.decoded_output = [{"batch_item": 0, "rank": 1, "label": "cat", "prob": 0.8}]

    bundle_path = tmp_path / "postprocessor.tlspec"
    save(trace, bundle_path)
    restored = load(bundle_path)

    assert restored.output_postprocessor == trace.output_postprocessor
    assert restored.decoded_output == trace.decoded_output


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
        transform=double_input,
    )
    assert trace.transform_repr == repr(double_input)
    assert trace._transform is double_input

    bundle_path = tmp_path / "input_transform.tlspec"
    save(trace, bundle_path)
    restored = load(bundle_path)

    assert restored.transform_repr == trace.transform_repr
    assert restored._transform is None


def test_bundle_save_strict_default_raises_on_sparse_tensor(tmp_path: Path) -> None:
    """Strict bundle save should reject sparse tensors."""

    trace = _build_sparse_out_log()

    with pytest.raises(TorchLensIOError, match="sparse"):
        save(trace, tmp_path / "sparse_bundle.tl")


def test_bundle_save_strict_false_records_unsupported_tensors(tmp_path: Path) -> None:
    """Best-effort save should skip unsupported tensors and record them in the manifest."""

    trace = _build_sparse_out_log()
    bundle_path = tmp_path / "sparse_bundle.tl"

    save(trace, bundle_path, strict=False)

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

    with pytest.raises(TorchLensIOError, match="complex128"):
        save(trace, bundle_path)

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

    save(trace, bundle_path, strict=False)

    manifest = Manifest.read(bundle_path / "manifest.json")
    assert manifest.unsupported_tensors
    assert all(entry["kind"] == "out" for entry in manifest.unsupported_tensors)
    assert all("complex128" in entry["reason"] for entry in manifest.unsupported_tensors)


def test_bundle_save_complex64_still_round_trips(tmp_path: Path) -> None:
    """``complex64`` remains genuinely supported -- ``complex128`` was the only gap.

    Confirms the ``tensor_policy``/payload-codec fix is scoped narrowly: it must not
    regress the dtype that ``safetensors`` 0.8.0 genuinely round-trips.
    """

    trace = _build_complex_out_log(dtype=torch.complex64)
    bundle_path = tmp_path / "complex64_bundle.tl"
    first_saved_layer = next(layer for layer in trace.layer_list if layer.has_saved_activation)
    original_out = first_saved_layer.out
    assert isinstance(original_out, torch.Tensor)
    assert original_out.dtype == torch.complex64

    save(trace, bundle_path)
    restored = load(bundle_path)
    restored_layer = next(layer for layer in restored.layer_list if layer.has_saved_activation)

    assert isinstance(restored_layer.out, torch.Tensor)
    assert restored_layer.out.dtype == torch.complex64
    assert torch.equal(original_out, restored_layer.out)


@pytest.mark.parametrize(
    ("scenario", "mutate_manifest", "expectation", "expected_text"),
    [
        (
            "io_format_newer",
            lambda manifest: manifest.__setitem__("tlspec_version", TLSPEC_VERSION + 1),
            "raise",
            str(TLSPEC_VERSION + 1),
        ),
        (
            "io_format_older",
            lambda manifest: manifest.__setitem__("tlspec_version", TLSPEC_VERSION - 1),
            "deprecation_warning",
            str(TLSPEC_VERSION - 1),
        ),
        (
            "io_format_equal",
            lambda manifest: manifest.__setitem__("tlspec_version", TLSPEC_VERSION),
            "ok",
            None,
        ),
        (
            "torch_major_mismatch",
            lambda manifest: manifest.__setitem__("torch_version", "999.0.0"),
            "raise",
            "999.0.0",
        ),
        (
            "torch_minor_mismatch",
            lambda manifest: manifest.__setitem__("torch_version", _torch_minor_mismatch_version()),
            "warning",
            lambda: _torch_minor_mismatch_version(),
        ),
        (
            "torchlens_newer",
            lambda manifest: manifest.__setitem__("torchlens_version", "999.0.0"),
            "warning",
            "999.0.0",
        ),
        (
            "torchlens_older",
            lambda manifest: manifest.__setitem__("torchlens_version", "0.0.1"),
            "info_log",
            "0.0.1",
        ),
        (
            "python_major_mismatch",
            lambda manifest: manifest.__setitem__("python_version", "999.0.0"),
            "python_major_mismatch",
            "999.0.0",
        ),
        (
            "unknown_extra_blob_file",
            lambda manifest: manifest,
            "extra_blob_warning",
            None,
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
            with pytest.raises(TorchLensIOError, match="python_version=999.0.0"):
                load(bundle_path)
        return

    if expectation == "extra_blob_warning":
        (bundle_path / "blobs" / "extra.bin").write_bytes(b"extra")
        with pytest.warns(UserWarning, match="unreferenced extra files"):
            loaded = load(bundle_path)
        assert loaded.model_class_name == "_ConvBundleModel"
        return

    if expectation == "raise":
        with pytest.raises(TorchLensIOError, match=resolved_expected_text or ""):
            load(bundle_path)
        return

    if expectation == "deprecation_warning":
        with pytest.warns(DeprecationWarning, match=resolved_expected_text or ""):
            loaded = load(bundle_path)
        assert loaded.model_class_name == "_ConvBundleModel"
        return

    if expectation == "warning":
        with pytest.warns(UserWarning, match=resolved_expected_text or ""):
            loaded = load(bundle_path)
        assert loaded.model_class_name == "_ConvBundleModel"
        return

    if expectation == "info_log":
        with caplog.at_level(logging.INFO, logger="torchlens._io.manifest"):
            loaded = load(bundle_path)
        assert loaded.model_class_name == "_ConvBundleModel"
        assert resolved_expected_text is not None
        assert resolved_expected_text in caplog.text
        assert "older than runtime torchlens_version" in caplog.text
        return

    loaded = load(bundle_path)
    assert loaded.model_class_name == "_ConvBundleModel"


def test_bundle_load_raises_for_missing_safetensors_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load should wrap backend import failures with an install hint."""

    bundle_path, _ = _save_bundle(tmp_path)

    def _raise_backend_import(path: Path, device: str) -> dict[str, torch.Tensor]:
        raise ImportError("missing backend")

    monkeypatch.setattr("torchlens._io.bundle.load_file", _raise_backend_import)

    with pytest.raises(TorchLensIOError, match="Install safetensors>=0.4"):
        load(bundle_path)


def test_bundle_load_raises_on_corrupt_manifest(tmp_path: Path) -> None:
    """Non-JSON manifests should fail with ``TorchLensIOError``."""

    bundle_path, _ = _save_bundle(tmp_path)
    (bundle_path / "manifest.json").write_text("{not valid json", encoding="utf-8")

    with pytest.raises(TorchLensIOError, match="Failed to read manifest"):
        load(bundle_path)


def test_bundle_load_raises_on_missing_blob_file(tmp_path: Path) -> None:
    """Manifest entries without corresponding files should fail and name the blob id."""

    bundle_path, _ = _save_bundle(tmp_path)
    manifest = Manifest.read(bundle_path / "manifest.json")
    blob_entry = manifest.tensors[0]
    (bundle_path / blob_entry.relative_path).unlink()

    with pytest.raises(TorchLensIOError, match=blob_entry.blob_id):
        load(bundle_path)


def test_bundle_load_raises_on_truncated_safetensors(tmp_path: Path) -> None:
    """Truncated blob files should fail bundle load."""

    bundle_path, _ = _save_bundle(tmp_path)
    blob_path = _first_blob_path(bundle_path)
    blob_path.write_bytes(blob_path.read_bytes()[:16])

    with pytest.raises(TorchLensIOError):
        load(bundle_path)


def test_bundle_load_raises_on_checksum_tamper(tmp_path: Path) -> None:
    """Blob checksum tampering should be detected eagerly."""

    bundle_path, _ = _save_bundle(tmp_path)
    manifest = Manifest.read(bundle_path / "manifest.json")
    blob_entry = manifest.tensors[0]
    _corrupt_blob_byte(bundle_path / blob_entry.relative_path)

    with pytest.raises(TorchLensIOError, match=blob_entry.blob_id):
        load(bundle_path)


def test_bundle_save_overwrite_false_wraps_file_exists_error(tmp_path: Path) -> None:
    """Existing target paths should fail unless ``overwrite=True``."""

    bundle_path, trace = _save_bundle(tmp_path)

    with pytest.raises(TorchLensIOError) as excinfo:
        save(trace, bundle_path)

    assert isinstance(excinfo.value.__cause__, FileExistsError)


def test_bundle_save_overwrite_true_replaces_existing_bundle(tmp_path: Path) -> None:
    """Overwrite mode should atomically replace an existing bundle."""

    bundle_path, first_log = _save_bundle(tmp_path, seed=0)
    second_log = _build_conv_log(seed=1)

    save(second_log, bundle_path, overwrite=True)
    restored = load(bundle_path)

    first_output = first_log.layer_list[-1].out
    second_output = second_log.layer_list[-1].out
    restored_output = restored.layer_list[-1].out

    assert isinstance(first_output, torch.Tensor)
    assert isinstance(second_output, torch.Tensor)
    assert isinstance(restored_output, torch.Tensor)
    assert not torch.equal(first_output, second_output)
    assert torch.equal(second_output, restored_output)


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
    the caller instead of a clean ``TorchLensIOError``).
    """

    bundle_path, first_log = _save_bundle(tmp_path, seed=0)
    second_log = _build_conv_log(seed=1)

    def _poisoned_pickle_dump(*_args: Any, **_kwargs: Any) -> None:
        raise TypeError("simulated live-resource pickling failure")

    monkeypatch.setattr("torchlens._io.bundle.pickle.dump", _poisoned_pickle_dump)

    with pytest.raises(TorchLensIOError) as excinfo:
        save(second_log, bundle_path, overwrite=True)

    assert isinstance(excinfo.value.__cause__, TypeError)

    # The original (pre-overwrite) bundle must survive the failed overwrite.
    assert bundle_path.exists()
    restored = load(bundle_path)
    first_output = first_log.layer_list[-1].out
    restored_output = restored.layer_list[-1].out
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

    Deliberately NOT a subclass of ``TorchLensIOError``, ``BackendPayloadUnsupportedError``,
    or any member of ``save()``'s historical ``(ImportError, OSError, TypeError, ValueError,
    pickle.PickleError)`` tuple, so a test injecting it proves the catch-all safety net
    closes the *class* of "unenumerated exception strands the backup" bug rather than
    merely covering the one exception type (``KeyError``) that happened to surface it.
    """


@pytest.mark.parametrize(
    "injected_exception",
    [KeyError("torch.complex128"), _NeverEnumeratedError("simulated brand-new writer failure")],
    ids=["keyerror", "unenumerated-exception-type"],
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

    def _poisoned_pickle_dump(*_args: Any, **_kwargs: Any) -> None:
        raise injected_exception

    monkeypatch.setattr("torchlens._io.bundle.pickle.dump", _poisoned_pickle_dump)

    with pytest.raises(TorchLensIOError) as excinfo:
        save(second_log, bundle_path, overwrite=True)

    assert excinfo.value.__cause__ is injected_exception

    # The original (pre-overwrite) bundle must survive the failed overwrite,
    # regardless of which exception type the write raised.
    assert bundle_path.exists()
    restored = load(bundle_path)
    first_output = first_log.layer_list[-1].out
    restored_output = restored.layer_list[-1].out
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
    escaped uncaught instead of the documented ``TorchLensIOError``.
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

    with pytest.raises(TorchLensIOError) as excinfo:
        load(bundle_path)

    assert isinstance(excinfo.value.__cause__, TypeError)


def test_bundle_save_raw_input_stringifies_unpicklable_value_keeps_tensors(
    tmp_path: Path,
) -> None:
    """``save_raw_input=True`` must stringify un-picklable objects, not pickle them raw.

    Unmocked, public-API-only repro of the root cause behind the BLOCKER
    above: ``save_raw_input=True``/``save_raw_output=True`` (documented
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
        transform=lambda d: d["tensor"],
        save_raw_input=True,
        random_seed=0,
    )
    assert isinstance(trace.raw_input["meta"], type(make_generator()))

    bundle_path = tmp_path / "raw_input_bundle.tl"
    save(trace, bundle_path)

    loaded = load(bundle_path)
    assert isinstance(loaded.raw_input["tensor"], torch.Tensor)
    assert torch.equal(loaded.raw_input["tensor"], raw_input["tensor"])
    assert loaded.raw_input["meta"] == "<scrubbed:generator>"


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

    monkeypatch.setattr("torchlens._io.scrub.pickle.dumps", _counting_dumps)

    model = nn.Linear(3, 3)
    large_tensor = torch.randn(200, 200)
    raw_input = {"tensor": large_tensor, "x": torch.randn(1, 3)}
    trace = trace_fn(
        model,
        raw_input,
        transform=lambda d: d["x"],
        save_raw_input=True,
        random_seed=0,
    )

    bundle_path = tmp_path / "raw_input_large_tensor_bundle.tl"
    save(trace, bundle_path)

    assert tensor_dumps_calls == 0

    loaded = load(bundle_path)
    assert isinstance(loaded.raw_input["tensor"], torch.Tensor)
    assert torch.equal(loaded.raw_input["tensor"], large_tensor)


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
        transform=lambda d: d["x"],
        save_raw_input=True,
        random_seed=0,
    )

    bundle_path = tmp_path / "raw_input_numeric_ndarray_bundle.tl"
    save(trace, bundle_path)

    assert ndarray_dumps_calls == 0

    loaded = load(bundle_path)
    assert isinstance(loaded.raw_input["array"], np.ndarray)
    np.testing.assert_array_equal(loaded.raw_input["array"], numeric_array)


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
        transform=lambda d: d["x"],
        save_raw_input=True,
        random_seed=0,
    )

    bundle_path = tmp_path / "raw_input_object_ndarray_bundle.tl"
    # Must NOT raise -- the unpicklable element degrades gracefully instead
    # of reaching the real pickle.dump() over the full scrubbed state.
    save(trace, bundle_path)

    loaded = load(bundle_path)
    assert loaded.raw_input["array"] == "<scrubbed:ndarray>"


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
        transform=lambda d: d["x"],
        save_raw_input=True,
        random_seed=0,
    )

    bundle_path = tmp_path / "raw_input_structured_ndarray_bundle.tl"
    # Must NOT raise -- the unpicklable field element degrades gracefully
    # instead of reaching the real pickle.dump() over the full scrubbed state.
    save(trace, bundle_path)

    loaded = load(bundle_path)
    assert loaded.raw_input["array"] == "<scrubbed:ndarray>"


def test_bundle_save_rejects_symlink_target(tmp_path: Path) -> None:
    """Bundle save should refuse symlinked target paths."""

    trace = _build_conv_log()
    real_path = tmp_path / "real_bundle"
    real_path.mkdir()
    symlink_path = tmp_path / "bundle_link"
    symlink_path.symlink_to(real_path, target_is_directory=True)

    with pytest.raises(TorchLensIOError, match="symlinked save target"):
        save(trace, symlink_path)


def test_bundle_loaded_log_validation_guard_raises(tmp_path: Path) -> None:
    """Portable-loaded logs should reject replay validation APIs."""

    bundle_path, _ = _save_bundle(tmp_path)
    restored = load(bundle_path)

    assert restored._loaded_from_bundle is True
    with pytest.raises(TorchLensIOError, match="portable bundles drop them"):
        restored.validate_forward_pass([])


def test_bundle_save_rejects_non_tensor_activation_transform_output(tmp_path: Path) -> None:
    """Save should fail before writing blobs when activation_transform returned a non-tensor."""

    trace = _build_non_tensor_out_log()
    bundle_path = tmp_path / "non_tensor_bundle.tl"

    with pytest.raises(TorchLensIOError, match="activation_transform outputs"):
        save(trace, bundle_path)

    assert not bundle_path.exists()


def test_cleanup_tmp_removes_partial_temp_directories(tmp_path: Path) -> None:
    """``cleanup_tmp`` should remove sibling temp dirs marked as partial."""

    target_path = tmp_path / "bundle.tl"
    partial_tmp_path = tmp_path / f"{target_path.name}.tmp.partial"
    partial_tmp_path.mkdir()
    (partial_tmp_path / "PARTIAL").write_text("", encoding="utf-8")

    removed = cleanup_tmp(target_path)

    assert removed == [partial_tmp_path]
    assert not partial_tmp_path.exists()


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


def test_cleanup_tmp_leaves_distinct_backup_when_bundle_exists(tmp_path: Path) -> None:
    """A ``.bak.*`` sibling whose contents differ from the live bundle must NOT be
    silently destroyed by default -- it is not provably redundant.

    Regression test for the round-6 H4 over-reach: the previous implementation
    inferred "redundant" purely from ``bundle_path.exists()`` with no content
    check, so a genuinely distinct backup was silently ``shutil.rmtree()``'d.
    """

    target_path = tmp_path / "bundle.tl"
    target_path.mkdir()
    (target_path / "marker").write_text("current", encoding="utf-8")
    bak_path = tmp_path / f"{target_path.name}.bak.deadbeef"
    bak_path.mkdir()
    (bak_path / "marker").write_text("stale-but-distinct", encoding="utf-8")

    with pytest.warns(UserWarning, match="not provably redundant"):
        removed = cleanup_tmp(target_path)

    assert removed == []
    assert bak_path.exists()
    assert (bak_path / "marker").read_text(encoding="utf-8") == "stale-but-distinct"
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


def test_cleanup_tmp_restores_orphaned_backup_when_bundle_missing(tmp_path: Path) -> None:
    """A double-failure orphaned ``.bak.*`` dir should be recovered onto ``bundle_path``.

    Regression test for the round-5 MINOR: if ``save(overwrite=True)`` fails
    and the best-effort ``_restore_backup()`` step also fails (a second,
    independent I/O failure), the ``.bak.<uuid>`` sibling was previously left
    permanently orphaned with no sweep mechanism -- ``cleanup_tmp()`` only
    globbed ``.tmp.*``. Since the backup holds the only surviving copy of the
    pre-overwrite bundle, the fix restores it onto the missing ``bundle_path``
    instead of deleting it.
    """

    target_path = tmp_path / "bundle.tl"
    bak_path = tmp_path / f"{target_path.name}.bak.deadbeef"
    bak_path.mkdir()
    (bak_path / "marker").write_text("recovered", encoding="utf-8")

    removed = cleanup_tmp(target_path)

    assert removed == [target_path]
    assert not bak_path.exists()
    assert target_path.exists()
    assert (target_path / "marker").read_text(encoding="utf-8") == "recovered"


def _torch_minor_mismatch_version() -> str:
    """Return a torch version string with the same major and different minor.

    Returns
    -------
    str
        Version string suitable for the minor-mismatch policy test.
    """

    version_parts = torch.__version__.split("+", maxsplit=1)[0].split(".")
    major = int(version_parts[0])
    minor = int(version_parts[1]) if len(version_parts) > 1 else 0
    patch = int(version_parts[2]) if len(version_parts) > 2 and version_parts[2].isdigit() else 0
    return f"{major}.{minor + 1}.{patch}"


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
