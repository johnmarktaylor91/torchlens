"""Unified ``.tlspec`` manifest writer helpers."""

from __future__ import annotations

import contextlib
import json
import os
import platform
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Literal

import torch

from .. import __version__ as TORCHLENS_VERSION
from .._data_classes._trace_missing_attrs import is_live_trace
from .._errors import InvalidArgumentError
from .._utils._record_group_table import record_member
from . import TLSPEC_VERSION, refuse_bundle_exists
from ._durability import _restrict_mode, _restrict_tree_modes, fsync_dir, fsync_tree
from ._staging import begin_staging, settle_failed_staging
from ._tlspec_identity import _backend_v2_fields, _model_fingerprint, _model_signature, _safe_len
from ._tlspec_json_helpers import _json_int_or_none, _json_str_or_none, _utc_timestamp
from .format3.container import write_members
from .format3.writer import EnvelopeSpec, envelope_fields
from .manifest import Manifest, TensorEntry, _runtime_python_version
from .paths import reject_symlink_path

BUNDLE_RELATIONS_FILE = "relations.json"
"""Bundle member list and relation table (3.0 bundle layout)."""
_MEMBER_SLUG_LIMIT = 64
"""Characters of a member name kept in its directory name."""

_PARTIAL_SENTINEL = "PARTIAL"
"""In-progress marker matching ``bundle.py``/``streaming.py``; ``cleanup_tmp``
sweeps a stranded ``{target}.tmp.*`` dir without ``force=`` only when present."""


def _reject_symlink_path(path: Path, *, context: str) -> None:
    """Reject symlink paths before writing a ``.tlspec`` payload."""

    reject_symlink_path(
        path,
        context=context,
        message_prefix="Refusing to write through symlink",
    )


# NOTE: ``TLSPEC_VERSION`` is imported (not redefined) from ``torchlens._io``
# so there is a single source of truth for the on-disk ``tlspec_version``
# manifest field. It previously had a second, independent definition here
# (pinned at 1) that collided with the same JSON key and was silently
# clobbering the correct value in ``write_trace_manifest`` below.
TLSPEC_SCHEMA_VERSION = 1
TLSPEC_MANIFEST_FILENAME = "manifest.json"
TLSPEC_VALID_SAVE_LEVELS = ("audit", "executable_with_callables", "portable", "runnable")
TlspecKind = Literal["intervention", "trace", "bundle", "recipe"]
TlspecSaveLevel = Literal["audit", "executable_with_callables", "portable", "runnable"]


def _member_slug(name: object) -> str:
    """Return the filesystem-safe part of a member directory name."""

    slug = re.sub(r"[^A-Za-z0-9_.-]", "_", str(name)).strip(".")
    return slug[:_MEMBER_SLUG_LIMIT] or "member"


class _TlSpecWriter:
    """Write unified ``.tlspec`` manifests and bundle payloads."""

    @classmethod
    def build_trace_manifest(
        cls,
        *,
        trace: Any,
        legacy_manifest: Manifest,
        save_level: str,
        sparse_run: dict[str, Any] | None = None,
        scrubbed_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the unified manifest fields of a saved ``Trace`` payload.

        The 3.0 writer adds its envelope fields on top and writes the result.

        Parameters
        ----------
        trace:
            Source model log.
        legacy_manifest:
            Existing portable-I/O manifest fields required by the loader.
        save_level:
            Save level recorded in the public schema.
        sparse_run:
            Authoritative sparse runnable descriptor, or ``None`` for all
            pre-existing analysis save levels.
        scrubbed_state:
            Portable trace state whose remapped identities must feed public sites.
        """

        manifest = legacy_manifest.to_dict()
        unified_fields = cls.build_manifest(
            kind="trace",
            source=trace,
            tensor_entries=legacy_manifest.tensors,
            save_level=save_level,
            spec_compat_info=None,
            intervention_compat_metadata=None,
        )
        if scrubbed_state is not None:
            unified_fields["sites"] = cls._sites(scrubbed_state, kind="trace")
        # ``legacy_manifest.to_dict()`` already carries the authoritative
        # ``tlspec_version`` (sourced from the same ``TLSPEC_VERSION`` single
        # source of truth). Drop the duplicate key from the unified fields
        # before merging instead of silently letting ``dict.update`` decide
        # which of two same-named keys wins.
        unified_fields.pop("tlspec_version", None)
        manifest.update(unified_fields)
        backend_name = str(trace._provenance_backend if is_live_trace(trace) else "torch")
        if backend_name != "torch" or sparse_run is not None:
            manifest.update(
                _backend_v2_fields(
                    trace, backend_name=backend_name, tensor_entries=legacy_manifest.tensors
                )
            )
        if sparse_run is not None:
            manifest["run"] = sparse_run
        return manifest

    @classmethod
    def write_trace_manifest(
        cls,
        *,
        path: Path,
        trace: Any,
        legacy_manifest: Manifest,
        save_level: str,
        sparse_run: dict[str, Any] | None = None,
        scrubbed_state: dict[str, Any] | None = None,
    ) -> None:
        """Write the unified manifest of a saved ``Trace`` payload.

        Parameters
        ----------
        path:
            Destination ``manifest.json`` path.
        trace:
            Source model log.
        legacy_manifest:
            Existing portable-I/O manifest fields required by the loader.
        save_level:
            Save level recorded in the public schema.
        sparse_run:
            Authoritative sparse runnable descriptor, or ``None``.
        scrubbed_state:
            Portable trace state whose remapped identities must feed public sites.
        """

        manifest = cls.build_trace_manifest(
            trace=trace,
            legacy_manifest=legacy_manifest,
            save_level=save_level,
            sparse_run=sparse_run,
            scrubbed_state=scrubbed_state,
        )
        cls.write_json(path, manifest)

    @classmethod
    def write_intervention_manifest(
        cls,
        *,
        path: Path,
        log: Any,
        recipe: dict[str, Any],
        target_manifest: list[Any],
        save_level: str,
        envelope: dict[str, Any],
    ) -> None:
        """Write the manifest of a ``kind="intervention"`` artifact.

        Parameters
        ----------
        path:
            Destination ``manifest.json`` path.
        log:
            Source model log carrying the intervention recipe.
        recipe:
            The recipe document written to ``intervention.json``.
        target_manifest:
            Resolved target rows the replay compatibility check compares.
        save_level:
            Save level recorded in the public schema.
        envelope:
            The 3.0 envelope fields (written last, so ``files`` is complete).
        """

        if save_level == "runnable":
            raise InvalidArgumentError(
                "Intervention artifacts do not support save_level='runnable'",
                code="artifact_save_level_unsupported",
                remedy="use 'audit', 'executable_with_callables', or 'portable'",
                artifact_kind="intervention",
            )
        intervention_metadata = cls._intervention_compat_metadata(
            {**recipe, "target_manifest": target_manifest}
        )
        manifest = cls.build_manifest(
            kind="intervention",
            source=log,
            tensor_entries=[],
            save_level=save_level,
            spec_compat_info=intervention_metadata,
            intervention_compat_metadata=intervention_metadata,
        )
        manifest.pop("tensor_entries", None)
        manifest.update(envelope)
        cls.write_json(path, manifest)

    @classmethod
    def write_bundle(
        cls,
        *,
        bundle: Any,
        path: str | Path,
        save_level: str = "portable",
        overwrite: bool = False,
    ) -> None:
        """Write a ``Bundle`` as a unified ``.tlspec`` directory.

        Parameters
        ----------
        bundle:
            Bundle instance to persist.
        path:
            Destination directory.
        save_level:
            Save level recorded in the public schema.
        overwrite:
            Whether an existing destination may be replaced.
        """

        level = coerce_tlspec_save_level(save_level)
        if level == "runnable":
            raise InvalidArgumentError(
                "Bundle artifacts do not support save_level='runnable'",
                code="artifact_save_level_unsupported",
                remedy="use 'audit', 'executable_with_callables', or 'portable'",
                artifact_kind="bundle",
            )
        target_path = Path(path)
        _reject_symlink_path(target_path, context="bundle tlspec target")
        # Target-PREFIXED staging/backup names (one defect two labs):
        # the old target-independent ``tmp.<hex>`` / ``tmp.bak.<hex>`` names
        # were invisible to ``cleanup_tmp(target)`` (it globs
        # ``{name}.tmp.*`` / ``{name}.bak.*``), so a SIGKILL between the two
        # ``os.replace`` calls below stranded the ONLY pre-overwrite copy in
        # an undiscoverable directory that no sweep would ever restore.
        tmp_path = target_path.parent / f"{target_path.name}.tmp.{uuid.uuid4().hex}"
        # ``backup_path`` holds the pre-overwrite bundle *renamed aside* (never
        # deleted) so a failure during the final swap can restore it. It stays
        # ``None`` unless we actually move an existing target out of the way.
        backup_path: Path | None = None
        try:
            if target_path.exists() and not overwrite:
                refuse_bundle_exists(target_path, overwritable=True)
            begin_staging(target_path, tmp_path, overwrite=overwrite)
            _restrict_mode(tmp_path, 0o700)
            # Sweepable from birth: mark the staging dir PARTIAL so a SIGKILL
            # anywhere before publish leaves a directory ``cleanup_tmp()``
            # removes without ``force=True``; the sentinel comes off right
            # before the durability fsync + publish swap.
            (tmp_path / _PARTIAL_SENTINEL).write_text("", encoding="utf-8")
            member_records = cls._write_bundle_members(bundle, tmp_path=tmp_path, save_level=level)
            bundle_metadata: dict[str, Any] = {
                "members": member_records,
                "baseline_name": getattr(bundle, "baseline_name", None),
            }
            cls._add_gated_member_relations(bundle, bundle_metadata)
            cls._add_lineage_sections(bundle, bundle_metadata)
            cls._add_preserved_sections(bundle, bundle_metadata)
            cls.write_json(tmp_path / BUNDLE_RELATIONS_FILE, bundle_metadata)

            manifest = cls._bundle_manifest(bundle, tmp_path=tmp_path, save_level=level)
            cls.write_json(tmp_path / TLSPEC_MANIFEST_FILENAME, manifest)
            _restrict_tree_modes(tmp_path)
            # Durability before publish: fsync every written file and
            # directory so a power/OS crash after the rename below cannot
            # publish a bundle holding zero-length or partial members.
            (tmp_path / _PARTIAL_SENTINEL).unlink()
            fsync_tree(tmp_path)
            # Atomic overwrite. Never ``rmtree`` the only good bundle before
            # the replacement is known installed: move the existing target
            # ASIDE to a sibling backup (rename, not delete), swap the freshly
            # written bundle into place, then remove the backup only after the
            # swap succeeds. Any failure before/during the swap leaves the OLD
            # bundle recoverable (see the ``except`` restore below). All three
            # paths are siblings under ``target_path.parent`` so every rename
            # is same-filesystem/atomic. Mirrors the backup/restore state
            # machine in ``torchlens/_io/bundle.py`` (``_make_backup_path`` /
            # ``_restore_backup``).
            if target_path.exists():
                # Re-check overwrite at swap time, not just at save start
                # (R59 TOCTOU): under ``overwrite=False`` a concurrent writer
                # could have created ``target_path`` AFTER the start-of-save
                # check passed, and the unconditional swap below would then
                # back it aside and destroy it (the backup is rmtree'd on
                # success). Mirror ``torchlens/_io/bundle.py`` and refuse
                # instead of live-destroying a concurrently-published artifact.
                if not overwrite:
                    refuse_bundle_exists(target_path, overwritable=True)
                backup_path = target_path.parent / f"{target_path.name}.bak.{uuid.uuid4().hex}"
                os.replace(target_path, backup_path)
            os.replace(tmp_path, target_path)
            # Make the rename(s) themselves durable: one parent-directory
            # fsync after the final swap persists both the aside-rename and
            # the publish (they are entries of the same directory).
            fsync_dir(target_path.parent)
            if backup_path is not None:
                # The overwrite is complete; discarding the backup can never
                # lose the new bundle, so a cleanup failure must not fail the
                # save nor trigger a spurious restore.
                shutil.rmtree(backup_path, ignore_errors=True)
        except BaseException as exc:
            # ``BaseException`` (not ``Exception``) so an interrupt unwinding
            # mid-swap still restores the old bundle; ``raise`` re-raises the
            # original unchanged, preserving control-flow semantics.
            if backup_path is not None and not target_path.exists() and backup_path.exists():
                # The swap did not install the new bundle; put the old one back
                # exactly where it was. If this restore itself fails, the old
                # bundle survives under ``backup_path`` as a recovery artifact.
                with contextlib.suppress(OSError):
                    os.replace(backup_path, target_path)
            # An older bundle left at the target keeps the staging dir PARTIAL for tl.load.
            settle_failed_staging(tmp_path, target_path, overwrite=overwrite, primary=exc)
            raise

    @classmethod
    def _bundle_manifest(cls, bundle: Any, *, tmp_path: Path, save_level: str) -> dict[str, Any]:
        """Build the container manifest of a staged bundle directory.

        Members are 3.0 trace artifacts under ``members/<index>-<name>/``; the
        container manifest carries the 3.0 envelope and digests every staged
        file, members included, so it is built after the members are written.
        """

        manifest = cls.build_manifest(
            kind="bundle",
            source=bundle,
            tensor_entries=[],
            save_level=save_level,
            spec_compat_info=None,
            intervention_compat_metadata=None,
        )
        manifest.pop("body_index", None)
        manifest.update(envelope_fields(tmp_path, EnvelopeSpec("bundle", save_level)))
        return manifest

    @staticmethod
    def _add_gated_member_relations(bundle: Any, bundle_metadata: dict[str, Any]) -> None:
        """Write the ``member_relations`` key into ``bundle.json``.

        The key persists plainly as of the tlspec v8 coordinated bump; the
        load side validates it against the closed row schema and re-checks
        that every endpoint names a loaded member. An empty table writes
        nothing (absence means a plain bundle).
        """

        relation_table = getattr(bundle, "_member_relations", None)
        if relation_table is None or len(relation_table.rows) == 0:
            return
        bundle_metadata["member_relations"] = relation_table.to_payload()

    @staticmethod
    def _add_lineage_sections(bundle: Any, bundle_metadata: dict[str, Any]) -> None:
        """Write the lineage sections into ``bundle.json``.

        The split-off join shapes: container
        identity (``bundle_id`` / ``forked_from_bundle_id``), per-member
        ``member_construction`` origin anchors, the hash-chained
        ``operations`` chronology, and the ``member_effect_tables`` registry.
        Empty surfaces write nothing (entry-dark on plain bundles; an older
        reader in the same unreleased window sees only known keys).
        """

        bundle_id = getattr(bundle, "_bundle_id", None)
        if bundle_id:
            bundle_metadata["bundle_id"] = bundle_id
        forked_from = getattr(bundle, "_forked_from_bundle_id", None)
        if forked_from:
            bundle_metadata["forked_from_bundle_id"] = forked_from
        anchors = getattr(bundle, "_member_construction", None)
        if anchors:
            bundle_metadata["member_construction"] = {
                name: dict(anchor) for name, anchor in anchors.items()
            }
        operations = getattr(bundle, "_operations", None)
        if operations:
            bundle_metadata["operations"] = [row.to_payload() for row in operations]
        effect_tables = getattr(bundle, "_effect_tables", None)
        if effect_tables:
            bundle_metadata["member_effect_tables"] = {
                operation_id: table.to_payload() for operation_id, table in effect_tables.items()
            }

    @staticmethod
    def _add_preserved_sections(bundle: Any, bundle_metadata: dict[str, Any]) -> None:
        """Re-emit preserved unknown namespaced ``bundle.json`` sections.

        Loader doctrine leg (c) (the amendment): sections a load
        preserved opaque write back verbatim so re-save never silently
        destroys another provider's well-formed data. A preserved key can
        never collide with a writer-owned key — the loader only preserves
        NAMESPACED (dotted) keys and the writer's own keys are bare.
        """

        preserved = getattr(bundle, "preserved_sections", None)
        if not preserved:
            return
        for key, value in preserved.items():
            if key in bundle_metadata:
                # Defensive: never let carried data shadow a writer-owned
                # section (unreachable while writer keys stay bare).
                continue
            bundle_metadata[key] = value

    @classmethod
    def _write_bundle_members(
        cls,
        bundle: Any,
        *,
        tmp_path: Path,
        save_level: str,
    ) -> list[dict[str, str]]:
        """Write bundle members as nested unified Trace specs.

        Parameters
        ----------
        bundle:
            Bundle object.
        tmp_path:
            Temporary bundle directory.
        save_level:
            Save level forwarded to each member save.

        Returns
        -------
        list[dict[str, str]]
            Member metadata records.
        """

        return write_members(
            bundle,
            tmp_path,
            level=save_level,
            write_member=lambda trace, member_path, level: trace.save(member_path, level=level),
        )

    @classmethod
    def build_manifest(
        cls,
        *,
        kind: TlspecKind,
        source: Any,
        tensor_entries: list[TensorEntry],
        save_level: str,
        spec_compat_info: dict[str, Any] | None,
        intervention_compat_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build the public unified manifest fields.

        Parameters
        ----------
        kind:
            Public object kind.
        source:
            Source object used to derive model identity and site descriptors.
        tensor_entries:
            Tensor sidecars written for the payload.
        save_level:
            Save level recorded in the manifest.
        spec_compat_info:
            Intervention compatibility metadata, or ``None``.
        intervention_compat_metadata:
            Intervention replay metadata, or ``None``.

        Returns
        -------
        dict[str, Any]
            JSON-serializable unified manifest fields.
        """

        level = coerce_tlspec_save_level(save_level)
        model_signature = _model_signature(source, kind=kind)
        return {
            "tlspec_version": TLSPEC_VERSION,
            "kind": kind,
            "created_at": getattr(source, "_source_bundle_created_at", None) or _utc_timestamp(),
            "torchlens_version": TORCHLENS_VERSION,
            "python_version": _runtime_python_version(),
            "torch_version": torch.__version__,
            "platform": f"{platform.system().lower()}-{platform.machine().lower()}",
            "schema_version": TLSPEC_SCHEMA_VERSION,
            "model_signature": model_signature,
            "model_fingerprint": _model_fingerprint(source, model_signature, kind=kind),
            "sites": cls._sites(source, kind=kind),
            "spec_compat_info": spec_compat_info,
            "body_format": "safetensors",
            "body_index": cls._body_index(tensor_entries),
            "backward_summary": cls._backward_summary(source, tensor_entries),
            "save_level": level,
            "optional_dependencies": cls._optional_dependencies(source),
            "intervention_compat_metadata": intervention_compat_metadata,
        }

    @staticmethod
    def write_json(path: Path, data: dict[str, Any]) -> None:
        """Write a JSON object with deterministic formatting.

        Parameters
        ----------
        path:
            Destination JSON path.
        data:
            JSON object to write.
        """

        text = json.dumps(data, indent=2, sort_keys=False, allow_nan=False)
        with path.open("w", encoding="utf-8") as handle:
            handle.write(text + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    @classmethod
    def _sites(cls, source: Any, *, kind: TlspecKind) -> list[dict[str, Any]]:
        """Return site descriptors for a source object.

        Parameters
        ----------
        source:
            Source object.
        kind:
            Public object kind.

        Returns
        -------
        list[dict[str, Any]]
            Site descriptor objects.
        """

        if kind == "bundle":
            sites: list[dict[str, Any]] = []
            members = getattr(source, "members", {})
            if isinstance(members, dict):
                for member_name, member in members.items():
                    for site in cls._sites(member, kind="trace"):
                        site["bundle_member"] = member_name
                        sites.append(site)
            return sites
        layers = (
            source.get("_layer_list", [])
            if isinstance(source, dict)
            else (source._layer_list if is_live_trace(source) else [])
        )
        sites = []
        for layer in layers if isinstance(layers, list) else []:
            sites.append(
                {
                    "function_path": _json_str_or_none(getattr(layer, "func_name", None)),
                    "module_path": _json_str_or_none(getattr(layer, "module", None)),
                    "layer_label": str(getattr(layer, "layer_label", "")),
                    "pass_index": _json_int_or_none(getattr(layer, "call_index", None)),
                    "op_kind": _json_str_or_none(
                        record_member(layer, "equivalence_class", None)
                        or getattr(layer, "func_name", None)
                    ),
                }
            )
        return sites

    @staticmethod
    def _body_index(tensor_entries: list[TensorEntry]) -> list[dict[str, Any]]:
        """Build the public body index from tensor entries.

        Parameters
        ----------
        tensor_entries:
            Persisted tensor entries.

        Returns
        -------
        list[dict[str, Any]]
            Body index entries.
        """

        index = []
        for entry in tensor_entries:
            num_elements = 1
            for dim in entry.shape:
                num_elements *= dim
            body_entry = {
                "filename": entry.relative_path,
                "dtype": entry.dtype,
                "shape": entry.shape,
                "num_elements": num_elements,
                "intended_use": entry.kind,
                "sha256": entry.sha256,
            }
            for field_name in (
                "logical_backend",
                "codec",
                "logical_dtype",
                "logical_device",
                "transport_backend",
                "transport_dtype",
                "codec_metadata",
            ):
                field_value = getattr(entry, field_name)
                if field_value is not None:
                    body_entry[field_name] = field_value
            index.append(body_entry)
        return index

    @staticmethod
    def _backward_summary(source: Any, tensor_entries: list[TensorEntry]) -> dict[str, Any]:
        """Build public backward metadata for a unified manifest.

        Parameters
        ----------
        source:
            Saved object whose backward projection may be present.
        tensor_entries:
            Persisted tensor entries.

        Returns
        -------
        dict[str, Any]
            JSON-serializable summary of backward records and gradient blobs.
        """

        grad_kinds = {"grad", "transformed_grad", "grad_fn_grad"}
        backend_name = str(source._provenance_backend if is_live_trace(source) else "torch")
        saved_grad_records = 0
        if backend_name == "torch":
            saved_grad_records = _safe_len(getattr(source, "saved_grad_ops", None))
        return {
            "has_backward_pass": bool(
                source._autograd_has_backward_pass if is_live_trace(source) else False
            ),
            "num_backward_passes": int(
                (source._num_backward_passes if is_live_trace(source) else 0) or 0
            ),
            "num_grad_fns": _safe_len(source._grad_fn_logs if is_live_trace(source) else None),
            "num_grad_fn_calls": _safe_len(
                getattr(getattr(source, "autograd", None), "grad_fn_calls", None)
            ),
            "num_saved_grad_records": saved_grad_records,
            "gradient_blob_count": sum(1 for entry in tensor_entries if entry.kind in grad_kinds),
            "gradient_blob_kinds": sorted(
                {entry.kind for entry in tensor_entries if entry.kind in grad_kinds}
            ),
            "old_bundle_policy": (
                "legacy class remaps are accepted at load; removed backward gradient "
                "configuration fields are dropped during Trace state normalization"
            ),
        }

    @staticmethod
    def _optional_dependencies(source: Any) -> list[str]:
        """Return optional extras needed to load a source object.

        Parameters
        ----------
        source:
            Source object.

        Returns
        -------
        list[str]
            Optional dependency extra names.
        """

        metadata = getattr(source, "metadata", None)
        if isinstance(metadata, dict):
            value = metadata.get("optional_dependencies")
            if isinstance(value, list) and all(isinstance(item, str) for item in value):
                return sorted(set(value))
        return []

    @staticmethod
    def _intervention_compat_metadata(spec_json: dict[str, Any]) -> dict[str, Any]:
        """Extract replay compatibility metadata from an intervention payload.

        Parameters
        ----------
        spec_json:
            Serialized intervention spec payload.

        Returns
        -------
        dict[str, Any]
            Compatibility metadata for replay checks.
        """

        return {
            "format_version": spec_json.get("format_version"),
            "helper_registry_version": spec_json.get("helper_registry_version"),
            "executable": bool(spec_json.get("executable", False)),
            "target_manifest": spec_json.get("target_manifest", []),
            "function_registry_keys": spec_json.get("function_registry_keys", []),
            "append_state": spec_json.get("append_state", {}),
        }


def coerce_tlspec_save_level(save_level: str) -> TlspecSaveLevel:
    """Validate and normalize a ``.tlspec`` save level.

    Parameters
    ----------
    save_level:
        User-supplied save level string.

    Returns
    -------
    TlspecSaveLevel
        Normalized save level.

    Raises
    ------
    ValueError
        If the save level is unsupported.
    """

    if save_level not in TLSPEC_VALID_SAVE_LEVELS:
        levels = ", ".join(repr(level) for level in TLSPEC_VALID_SAVE_LEVELS)
        raise InvalidArgumentError(
            f".tlspec save level {save_level!r} is unsupported",
            code="artifact_save_level_invalid",
            remedy=f"set the save level to one of {levels}",
            argument="save_level",
        )
    return save_level  # type: ignore[return-value]


__all__ = [
    "TLSPEC_MANIFEST_FILENAME",
    "TLSPEC_SCHEMA_VERSION",
    "TLSPEC_VERSION",
    "_TlSpecWriter",
    "coerce_tlspec_save_level",
]
