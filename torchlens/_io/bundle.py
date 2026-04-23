"""Portable directory-bundle save/load helpers for TorchLens model logs."""

from __future__ import annotations

import os
import platform
import pickle
import shutil
import sys
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from . import BlobRef, IO_FORMAT_VERSION, TorchLensIOError
from .manifest import Manifest, TensorEntry, enforce_version_policy, sha256_of_file
from .rehydrate import rehydrate_model_log
from .scrub import scrub_for_save
from .tensor_policy import FailReason, Ok, SkipReason, is_supported_for_save
from .. import __version__ as TORCHLENS_VERSION
from ..data_classes.model_log import ModelLog

PARTIAL_SENTINEL = "PARTIAL"
REASON_SENTINEL = "REASON.txt"
_BLOB_TENSOR_KEY = "data"


def save(
    model_log: ModelLog,
    path: str | Path,
    *,
    include_activations: bool = True,
    include_gradients: bool = True,
    include_captured_args: bool = False,
    include_rng_states: bool = False,
    strict: bool = True,
    overwrite: bool = False,
) -> None:
    """Persist a ``ModelLog`` into a portable TorchLens directory bundle.

    Parameters
    ----------
    model_log:
        Completed model log to save.
    path:
        Output bundle directory path.
    include_activations:
        Whether activations should be saved as blobs.
    include_gradients:
        Whether gradients should be saved as blobs.
    include_captured_args:
        Whether captured args/kwargs and related tensor payloads should be saved.
    include_rng_states:
        Whether per-layer RNG state tensors should be saved.
    strict:
        Whether unsupported tensors should abort the save instead of being skipped.
    overwrite:
        Whether an existing bundle at ``path`` may be replaced.

    Raises
    ------
    TorchLensIOError
        If the bundle cannot be created or contains unsupported state.
    """

    bundle_path = Path(path)
    _reject_symlink_path(bundle_path, context="save target")
    _validate_activation_postfunc_outputs(model_log, include_activations=include_activations)

    backup_path: Path | None = None
    tmp_path = _make_tmp_bundle_path(bundle_path)
    try:
        if bundle_path.exists():
            if not overwrite:
                raise FileExistsError(f"Bundle path already exists: {bundle_path}")
            backup_path = _make_backup_path(bundle_path)
            bundle_path.rename(backup_path)

        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.mkdir()
        (tmp_path / "blobs").mkdir()

        scrubbed_state, blob_specs = _scrub_model_log_for_bundle(
            model_log,
            include_activations=include_activations,
            include_gradients=include_gradients,
            include_captured_args=include_captured_args,
            include_rng_states=include_rng_states,
        )

        tensor_entries: list[TensorEntry] = []
        unsupported_tensors: list[dict[str, str]] = []
        skipped_blob_ids: set[str] = set()

        for blob_id, tensor, kind, label in blob_specs:
            decision = is_supported_for_save(tensor, strict=strict)
            if isinstance(decision, Ok):
                tensor_entries.append(
                    _write_tensor_blob(
                        tmp_path=tmp_path, blob_id=blob_id, tensor=tensor, kind=kind, label=label
                    )
                )
                continue
            if isinstance(decision, FailReason):
                raise TorchLensIOError(
                    f"Unsupported tensor for bundle save at {label} ({kind}): {decision.text}"
                )
            unsupported_tensors.append({"label": label, "kind": kind, "reason": decision.text})
            skipped_blob_ids.add(blob_id)

        if skipped_blob_ids:
            _apply_skipped_blobs_to_scrubbed_state(scrubbed_state, skipped_blob_ids)

        manifest = _build_manifest(
            model_log=model_log,
            tensor_entries=tensor_entries,
            unsupported_tensors=unsupported_tensors,
        )
        manifest.write(tmp_path / "manifest.json")
        with (tmp_path / "metadata.pkl").open("wb") as handle:
            pickle.dump(scrubbed_state, handle, protocol=pickle.HIGHEST_PROTOCOL)

        tmp_path.rename(bundle_path)
        if backup_path is not None:
            _remove_path(backup_path)
    except TorchLensIOError:
        _mark_partial(tmp_path)
        if backup_path is not None and not bundle_path.exists() and backup_path.exists():
            _restore_backup(backup_path, bundle_path)
        raise
    except (ImportError, OSError, ValueError, pickle.PickleError) as exc:
        _mark_partial(tmp_path, reason=str(exc))
        if backup_path is not None and not bundle_path.exists() and backup_path.exists():
            _restore_backup(backup_path, bundle_path)
        raise TorchLensIOError(f"Failed to save bundle at {bundle_path}.") from exc


def load(
    path: str | Path,
    *,
    lazy: bool = False,
    map_location: str | torch.device = "cpu",
    materialize_nested: bool = True,
) -> ModelLog:
    """Load a portable TorchLens bundle into a ``ModelLog``.

    Parameters
    ----------
    path:
        Bundle directory path.
    lazy:
        Whether direct activation/gradient blobs should remain lazy placeholders.
    map_location:
        Target device for eager tensor materialization.
    materialize_nested:
        Whether nested blob refs in captured args and RNG states should be
        materialized when ``lazy=True``.

    Returns
    -------
    ModelLog
        Rehydrated model log.

    Raises
    ------
    TorchLensIOError
        If the bundle is invalid, corrupt, or incompatible with this runtime.
    """

    bundle_path = Path(path)
    _reject_symlink_path(bundle_path, context="bundle path")
    manifest_path = bundle_path / "manifest.json"
    metadata_path = bundle_path / "metadata.pkl"
    blobs_path = bundle_path / "blobs"
    _reject_symlink_path(manifest_path, context="manifest")
    _reject_symlink_path(metadata_path, context="metadata")
    _reject_symlink_path(blobs_path, context="blobs directory")

    try:
        manifest = Manifest.read(manifest_path)
        enforce_version_policy(manifest)
        _check_unknown_blob_entries(manifest, blobs_path)
        _validate_manifest_blob_paths(manifest, bundle_path)
        if not lazy:
            _eager_verify_blob_payloads(manifest, bundle_path, map_location)

        python_major_mismatch = _python_major_mismatch(manifest)
        with metadata_path.open("rb") as handle:
            scrubbed_state = pickle.load(handle)
    except TorchLensIOError:
        raise
    except (pickle.UnpicklingError, EOFError) as exc:
        hint = ""
        if python_major_mismatch:
            hint = (
                f" Bundle was written with python_version={manifest.python_version} but runtime is "
                f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}."
            )
        raise TorchLensIOError(
            f"Failed to load bundle metadata from {metadata_path}.{hint}"
        ) from exc
    except (OSError, AttributeError, EOFError, ImportError, ValueError) as exc:
        raise TorchLensIOError(f"Failed to load bundle at {bundle_path}.") from exc

    model_log = rehydrate_model_log(
        scrubbed_state,
        manifest,
        bundle_path,
        lazy=lazy,
        map_location=map_location,
        materialize_nested=materialize_nested,
    )
    setattr(model_log, "_loaded_from_bundle", True)
    setattr(model_log, "_source_bundle_manifest_sha256", sha256_of_file(manifest_path))
    return model_log


def cleanup_tmp(path: str | Path, *, force: bool = False) -> list[Path]:
    """Remove leftover sibling temp bundle directories for one target path.

    Parameters
    ----------
    path:
        Final bundle path whose ``.tmp.*`` siblings should be inspected.
    force:
        Whether temp dirs without a ``PARTIAL`` sentinel should also be removed.

    Returns
    -------
    list[Path]
        Removed temp directory paths.

    Raises
    ------
    TorchLensIOError
        If the requested target path or candidate temp dirs are symlinks.
    """

    bundle_path = Path(path)
    _reject_symlink_path(bundle_path, context="cleanup target")
    removed: list[Path] = []
    pattern = f"{bundle_path.name}.tmp.*"
    for candidate in bundle_path.parent.glob(pattern):
        if candidate.is_symlink():
            raise TorchLensIOError(f"Refusing to clean symlink temp directory {candidate}.")
        if not candidate.is_dir():
            continue
        if force or (candidate / PARTIAL_SENTINEL).exists():
            shutil.rmtree(candidate)
            removed.append(candidate)
            continue
        warnings.warn(
            f"Leaving non-partial temp directory {candidate} in place; pass force=True to remove it.",
            UserWarning,
            stacklevel=2,
        )
    return removed


def _scrub_model_log_for_bundle(
    model_log: ModelLog,
    *,
    include_activations: bool,
    include_gradients: bool,
    include_captured_args: bool,
    include_rng_states: bool,
) -> tuple[dict[str, Any], list[tuple[str, torch.Tensor, str, str]]]:
    """Scrub a model log while excluding transient load-only private attrs.

    Parameters
    ----------
    model_log:
        Model log being saved.
    include_activations:
        Whether activations should be blobified.
    include_gradients:
        Whether gradients should be blobified.
    include_captured_args:
        Whether nested captured args should be blobified.
    include_rng_states:
        Whether nested RNG states should be blobified.

    Returns
    -------
    tuple[dict[str, Any], list[tuple[str, torch.Tensor, str, str]]]
        Scrubbed metadata and blob specs.
    """

    transient_attrs = {}
    for attr_name in ("_loaded_from_bundle", "_source_bundle_manifest_sha256"):
        if hasattr(model_log, attr_name):
            transient_attrs[attr_name] = getattr(model_log, attr_name)
            delattr(model_log, attr_name)
    try:
        return scrub_for_save(
            model_log,
            include_activations=include_activations,
            include_gradients=include_gradients,
            include_captured_args=include_captured_args,
            include_rng_states=include_rng_states,
        )
    finally:
        for attr_name, attr_value in transient_attrs.items():
            setattr(model_log, attr_name, attr_value)


def _write_tensor_blob(
    *,
    tmp_path: Path,
    blob_id: str,
    tensor: torch.Tensor,
    kind: str,
    label: str,
) -> TensorEntry:
    """Write one supported tensor blob and build its manifest entry.

    Parameters
    ----------
    tmp_path:
        Temporary bundle directory root.
    blob_id:
        Opaque blob identifier.
    tensor:
        Tensor payload to persist.
    kind:
        Logical tensor kind.
    label:
        Human-readable TorchLens layer label.

    Returns
    -------
    TensorEntry
        Manifest tensor entry for the written blob.
    """

    contiguous_tensor = tensor.contiguous()
    relative_path = Path("blobs") / f"{blob_id}.safetensors"
    blob_path = tmp_path / relative_path
    save_file({_BLOB_TENSOR_KEY: contiguous_tensor}, str(blob_path))
    return TensorEntry(
        blob_id=blob_id,
        kind=kind,
        label=label,
        relative_path=relative_path.as_posix(),
        backend="safetensors",
        shape=[int(dim) for dim in contiguous_tensor.shape],
        dtype=str(contiguous_tensor.dtype).replace("torch.", ""),
        device_at_save=str(tensor.device),
        layout=str(contiguous_tensor.layout).replace("torch.", ""),
        bytes=int(contiguous_tensor.numel() * contiguous_tensor.element_size()),
        sha256=sha256_of_file(blob_path),
    )


def _build_manifest(
    *,
    model_log: ModelLog,
    tensor_entries: list[TensorEntry],
    unsupported_tensors: list[dict[str, str]],
) -> Manifest:
    """Create a manifest instance for a finished bundle save.

    Parameters
    ----------
    model_log:
        Source model log.
    tensor_entries:
        Persisted tensor entries.
    unsupported_tensors:
        Unsupported tensor records accumulated under ``strict=False``.

    Returns
    -------
    Manifest
        Fully-populated bundle manifest.
    """

    n_activation_blobs = sum(1 for entry in tensor_entries if entry.kind == "activation")
    n_gradient_blobs = sum(1 for entry in tensor_entries if entry.kind == "gradient")
    n_auxiliary_blobs = len(tensor_entries) - n_activation_blobs - n_gradient_blobs
    return Manifest(
        io_format_version=IO_FORMAT_VERSION,
        torchlens_version=TORCHLENS_VERSION,
        torch_version=torch.__version__,
        python_version=(
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        ),
        platform=f"{platform.system().lower()}-{platform.machine().lower()}",
        created_at=datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace(
            "+00:00",
            "Z",
        ),
        bundle_format="directory",
        n_layers=len(model_log.layer_list),
        n_activation_blobs=n_activation_blobs,
        n_gradient_blobs=n_gradient_blobs,
        n_auxiliary_blobs=n_auxiliary_blobs,
        tensors=tensor_entries,
        unsupported_tensors=unsupported_tensors,
    )


def _validate_activation_postfunc_outputs(
    model_log: ModelLog,
    *,
    include_activations: bool,
) -> None:
    """Reject portable saves when activation postprocessing produced non-tensors.

    Parameters
    ----------
    model_log:
        Model log being saved.
    include_activations:
        Whether activation fields will be saved.

    Raises
    ------
    TorchLensIOError
        If a saved activation is not a plain tensor.
    """

    if not include_activations or getattr(model_log, "activation_postfunc", None) is None:
        return
    for layer in model_log.layer_list:
        if not getattr(layer, "has_saved_activations", False):
            continue
        if layer.activation is None:
            continue
        if not isinstance(layer.activation, torch.Tensor):
            raise TorchLensIOError(
                "Portable bundle save requires activation_postfunc outputs to be torch.Tensor "
                f"instances, but layer {layer.layer_label} produced {type(layer.activation).__name__}."
            )


def _apply_skipped_blobs_to_scrubbed_state(
    scrubbed_state: dict[str, Any],
    skipped_blob_ids: set[str],
) -> None:
    """Replace skipped blob refs with ``None`` in scrubbed metadata.

    Parameters
    ----------
    scrubbed_state:
        Scrubbed metadata dict produced by ``scrub_for_save``.
    skipped_blob_ids:
        Blob ids skipped under ``strict=False``.
    """

    for field_name, field_value in list(scrubbed_state.items()):
        scrubbed_state[field_name] = _replace_skipped_blob_refs(field_value, skipped_blob_ids)

    layer_list = scrubbed_state.get("layer_list")
    if isinstance(layer_list, list):
        activation_labels = [
            layer.layer_label
            for layer in layer_list
            if bool(getattr(layer, "has_saved_activations", False))
        ]
        gradient_labels = [
            layer.layer_label for layer in layer_list if bool(getattr(layer, "has_gradient", False))
        ]
        scrubbed_state["layers_with_saved_activations"] = activation_labels
        scrubbed_state["layers_with_saved_gradients"] = gradient_labels
        scrubbed_state["num_tensors_saved"] = len(activation_labels)
        scrubbed_state["saved_activation_memory"] = sum(
            int(getattr(layer, "tensor_memory", 0) or 0)
            for layer in layer_list
            if bool(getattr(layer, "has_saved_activations", False))
        )
        scrubbed_state["has_gradients"] = bool(gradient_labels)


def _replace_skipped_blob_refs(value: Any, skipped_blob_ids: set[str]) -> Any:
    """Walk a scrubbed object graph and null out selected blob refs.

    Parameters
    ----------
    value:
        Scrubbed value to inspect.
    skipped_blob_ids:
        Blob ids skipped under ``strict=False``.

    Returns
    -------
    Any
        Value with matching ``BlobRef`` instances replaced by ``None``.
    """

    if isinstance(value, BlobRef):
        if value.blob_id in skipped_blob_ids:
            return None
        return value
    if isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = _replace_skipped_blob_refs(item, skipped_blob_ids)
        return value
    if isinstance(value, tuple):
        return tuple(_replace_skipped_blob_refs(item, skipped_blob_ids) for item in value)
    if isinstance(value, dict):
        for key, item in list(value.items()):
            value[key] = _replace_skipped_blob_refs(item, skipped_blob_ids)
        return value

    spec = getattr(type(value), "PORTABLE_STATE_SPEC", None)
    if spec is None:
        return value

    for field_name, field_value in list(vars(value).items()):
        replaced_value = _replace_skipped_blob_refs(field_value, skipped_blob_ids)
        setattr(value, field_name, replaced_value)
        if (
            field_name == "activation"
            and replaced_value is None
            and hasattr(value, "has_saved_activations")
        ):
            value.has_saved_activations = False
        if field_name == "gradient" and replaced_value is None and hasattr(value, "has_gradient"):
            value.has_gradient = False
    return value


def _check_unknown_blob_entries(manifest: Manifest, blobs_path: Path) -> None:
    """Warn when ``blobs/`` contains files that are not referenced by the manifest.

    Parameters
    ----------
    manifest:
        Parsed bundle manifest.
    blobs_path:
        Bundle ``blobs/`` directory path.
    """

    expected_names = {Path(entry.relative_path).name for entry in manifest.tensors}
    actual_names: set[str] = set()
    for child in blobs_path.iterdir():
        if child.is_symlink():
            raise TorchLensIOError(f"Refusing to load symlinked blob path {child}.")
        actual_names.add(child.name)
    extra_names = sorted(actual_names - expected_names)
    if extra_names:
        warnings.warn(
            f"Bundle contains unreferenced extra files in blobs/: {', '.join(extra_names)}.",
            UserWarning,
            stacklevel=2,
        )


def _validate_manifest_blob_paths(manifest: Manifest, bundle_path: Path) -> None:
    """Ensure every manifest tensor entry points at a real non-symlink blob file.

    Parameters
    ----------
    manifest:
        Parsed bundle manifest.
    bundle_path:
        Bundle directory root.

    Raises
    ------
    TorchLensIOError
        If any referenced blob is missing or symlinked.
    """

    missing_blob_ids: list[str] = []
    for entry in manifest.tensors:
        blob_path = bundle_path / entry.relative_path
        if blob_path.is_symlink():
            raise TorchLensIOError(f"Refusing to load symlinked blob path {blob_path}.")
        if not blob_path.exists():
            missing_blob_ids.append(entry.blob_id)
    if missing_blob_ids:
        raise TorchLensIOError(
            "Bundle manifest references missing blob files for blob_id(s): "
            f"{', '.join(missing_blob_ids)}."
        )


def _eager_verify_blob_payloads(
    manifest: Manifest,
    bundle_path: Path,
    map_location: str | torch.device,
) -> None:
    """Eagerly checksum and decode every blob for ``lazy=False`` loads.

    Parameters
    ----------
    manifest:
        Parsed bundle manifest.
    bundle_path:
        Bundle directory root.
    map_location:
        Device passed through to ``safetensors`` decoding.
    """

    for entry in manifest.tensors:
        blob_path = bundle_path / entry.relative_path
        observed_sha256 = sha256_of_file(blob_path)
        if observed_sha256 != entry.sha256:
            raise TorchLensIOError(f"Checksum mismatch for blob_id={entry.blob_id} at {blob_path}.")
        tensor_map = _load_safetensors_file(blob_path, map_location)
        if _BLOB_TENSOR_KEY not in tensor_map:
            raise TorchLensIOError(
                f"Blob {blob_path} does not contain the expected {_BLOB_TENSOR_KEY!r} tensor entry."
            )


def _python_major_mismatch(manifest: Manifest) -> bool:
    """Return whether the manifest's Python major version differs from runtime.

    Parameters
    ----------
    manifest:
        Parsed bundle manifest.

    Returns
    -------
    bool
        True when manifest and runtime major versions differ.
    """

    try:
        return int(manifest.python_version.split(".", maxsplit=1)[0]) != sys.version_info.major
    except ValueError:
        return False


def _load_safetensors_file(
    blob_path: Path,
    map_location: str | torch.device,
) -> dict[str, torch.Tensor]:
    """Load one safetensors blob with a TorchLens-specific install hint.

    Parameters
    ----------
    blob_path:
        Blob file path to load.
    map_location:
        Target device for decoded tensors.

    Returns
    -------
    dict[str, torch.Tensor]
        Loaded tensor mapping from the safetensors file.

    Raises
    ------
    TorchLensIOError
        If the safetensors backend is unavailable.
    """

    try:
        return load_file(blob_path, device=str(map_location))
    except ImportError as exc:
        raise TorchLensIOError(
            "Portable bundle load requires the safetensors backend. Install safetensors>=0.4."
        ) from exc


def _reject_symlink_path(path: Path, *, context: str) -> None:
    """Raise when a bundle path that must stay local is a symlink.

    Parameters
    ----------
    path:
        Path to validate.
    context:
        Human-readable context used in the error message.
    """

    if path.is_symlink():
        raise TorchLensIOError(f"Refusing symlinked {context}: {path}.")


def _make_tmp_bundle_path(bundle_path: Path) -> Path:
    """Create the deterministic sibling temp path for one bundle target.

    Parameters
    ----------
    bundle_path:
        Final bundle directory path.

    Returns
    -------
    Path
        Temporary working directory path.
    """

    return bundle_path.parent / f"{bundle_path.name}.tmp.{uuid.uuid4().hex}"


def _make_backup_path(bundle_path: Path) -> Path:
    """Create the sibling backup path used during overwrite replacement.

    Parameters
    ----------
    bundle_path:
        Final bundle directory path.

    Returns
    -------
    Path
        Backup directory path.
    """

    return bundle_path.parent / f"{bundle_path.name}.bak.{uuid.uuid4().hex}"


def _mark_partial(tmp_path: Path, *, reason: str | None = None) -> None:
    """Best-effort mark a temp bundle directory as partial after failure.

    Parameters
    ----------
    tmp_path:
        Temporary bundle directory path.
    reason:
        Optional failure reason string to persist alongside the sentinel.
    """

    try:
        if tmp_path.exists():
            (tmp_path / PARTIAL_SENTINEL).write_text("", encoding="utf-8")
            if reason is not None:
                (tmp_path / REASON_SENTINEL).write_text(reason, encoding="utf-8")
    except OSError:
        return


def _remove_path(path: Path) -> None:
    """Remove a backup path after a successful overwrite replacement.

    Parameters
    ----------
    path:
        Path to remove.
    """

    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _restore_backup(backup_path: Path, bundle_path: Path) -> None:
    """Best-effort restore an overwritten bundle after a failed replacement.

    Parameters
    ----------
    backup_path:
        Backup path holding the previous bundle contents.
    bundle_path:
        Final bundle path to restore.
    """

    try:
        backup_path.rename(bundle_path)
    except OSError:
        return
