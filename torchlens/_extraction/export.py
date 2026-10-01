"""Read-only streaming exporters + the self-contained contract (D15).

NPY, HDF5, and MAT converters: complete-only, streaming (one shard of
working memory), READ-ONLY — destructive ``reclaim=`` was withdrawn by its
own proposers (deleting source shard k after writing destination part k
destroys the only complete canonical artifact before the derived one is
durable). Export preflights destination space and refuses with the exact
byte number; the honest disclosure stays in the docs: NSD-scale export
transiently needs 2x disk until the deferred transactional replace verb
exists. Temp + fsync + atomic publish; the exporter writes its OWN
destination ledger as it streams.

The export-owned top-level contract file is SELF-CONTAINED (deliberately
NOT named ``manifest.json`` — a consumer distinguishing a native artifact
from an export by its contract filename must keep that ability): files
present, source manifest/ledger/ID digests, row order and ID resolution,
subsets, the opaque/sanitized key map, shapes, axes, dtypes, conversions,
byte sizes, lossy facts, and per-file sha256, with byte-verbatim copies of
the source manifest and ID sidecar under the labeled provenance subpath.

Ragged members use the values/offsets/shapes triplet in EVERY format; no
object arrays anywhere. bf16/fp8 members widen to fp32 WITH disclosure or
refuse on request (NumPy raises ``TypeError`` for all three natively).

Every spelling here is DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch

from .._errors import InvalidArgumentError
from .._io._durability import fsync_dir, fsync_file
from .context import sanitize_key
from .ragged import RaggedBatch

__tl_layer__ = "L5"

__all__ = ["EXPORT_CONTRACT_FILENAME", "EXPORT_FORMATS", "export_extraction"]

#: The self-contained contract filename (recorded panel preference: NOT
#: ``manifest.json``; final spelling is the naming sprint's).
EXPORT_CONTRACT_FILENAME = "export_contract.json"

#: Labeled provenance subpath holding byte-verbatim source copies.
_PROVENANCE_SUBDIR = "provenance"

#: Closed export-format vocabulary.
EXPORT_FORMATS: tuple[str, ...] = ("npy", "hdf5", "mat")

#: Name of the optional extra covering the hdf5/mat dependencies.
_EXPORT_EXTRA = "extraction-export"

#: MATLAB v5 hard limit on any single variable.
_MAT_V5_LIMIT = 2**31 - 1

#: Dtypes NumPy cannot represent: widened to fp32 with disclosure.
_WIDEN_DTYPES = (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2)


def _sha256_file(path: Path) -> str:
    """Stream one file's sha256 (computed AT EXPORT TIME, always).

    Parameters
    ----------
    path:
        File to hash.

    Returns
    -------
    str
        ``"sha256:..."`` digest.
    """

    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                return f"sha256:{hasher.hexdigest()}"
            hasher.update(chunk)


def _widen_for_export(
    key: str, tensor: torch.Tensor, widen_unsupported: bool
) -> tuple[torch.Tensor, dict[str, Any] | None]:
    """Widen NumPy-unrepresentable dtypes to fp32, or refuse on request.

    Parameters
    ----------
    key:
        Output key (refusal text).
    tensor:
        Stored tensor.
    widen_unsupported:
        Whether widening is permitted (``False`` refuses typed).

    Returns
    -------
    tuple[torch.Tensor, dict | None]
        The export tensor and the conversion disclosure (``None`` when
        exported natively).

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_export_dtype_unsupported`` when widening was refused.
    """

    if tensor.dtype not in _WIDEN_DTYPES:
        return tensor, None
    if not widen_unsupported:
        raise InvalidArgumentError(
            f"Output key {key!r} is stored as {tensor.dtype}, which NumPy "
            "cannot represent, and widen_unsupported=False refused the "
            "fp32 widening.",
            code="extraction_export_dtype_unsupported",
            remedy=(
                "allow the disclosed widening (widen_unsupported=True, the "
                "default), or read the native artifact via open_extraction"
            ),
            key=key,
            stored_dtype=str(tensor.dtype),
        )
    return tensor.to(torch.float32), {
        "stored_dtype": str(tensor.dtype),
        "exported_dtype": "torch.float32",
        "conversion": f"{str(tensor.dtype).removeprefix('torch.')}->float32",
        "lossy": False,
    }


def _preflight_destination(dest: Path, needed_bytes: int) -> None:
    """Refuse an export whose destination lacks the exact byte room (D15).

    Parameters
    ----------
    dest:
        Destination directory.
    needed_bytes:
        Exact bytes about to be written.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_export_space_insufficient`` naming the exact numbers.
    """

    usage = shutil.disk_usage(dest)
    if needed_bytes > usage.free:
        raise InvalidArgumentError(
            f"Export needs {needed_bytes:,} bytes but the destination "
            f"filesystem has {usage.free:,} free. NSD-scale exports "
            "transiently need the source AND destination on disk at once "
            "(the transactional replace verb is deferred; the exporter "
            "never deletes its source).",
            code="extraction_export_space_insufficient",
            remedy="free destination space or export a key subset (keys=)",
            needed_bytes=needed_bytes,
            free_bytes=usage.free,
        )


def _disambiguate_sanitized(keys: list[str]) -> dict[str, str]:
    """Build the injective, case-collision-safe key -> filename-stem map.

    :func:`sanitize_key` is injective, but case-insensitive filesystems
    fold ``a``/``A``; colliding stems get a short content-hash suffix, and
    the FULL map is recorded in the contract file.

    Parameters
    ----------
    keys:
        Output keys being exported.

    Returns
    -------
    dict[str, str]
        ``key -> unique stem``.
    """

    stems = {key: sanitize_key(key) for key in keys}
    folded: dict[str, list[str]] = {}
    for key, stem in stems.items():
        folded.setdefault(stem.lower(), []).append(key)
    for colliders in folded.values():
        if len(colliders) > 1:
            for key in colliders:
                suffix = hashlib.sha256(key.encode()).hexdigest()[:8]
                stems[key] = f"{stems[key]}-{suffix}"
    return stems


class _DestinationLedger:
    """The exporter's own append-only destination ledger (D15)."""

    def __init__(self, path: Path) -> None:
        """Open the ledger for appends.

        Parameters
        ----------
        path:
            Ledger file path inside the temp destination.
        """

        self._path = path

    def append(self, row: dict[str, Any]) -> None:
        """Append one fsynced ledger row.

        Parameters
        ----------
        row:
            JSON-portable row.
        """

        with open(self._path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


@dataclasses.dataclass
class _ExportJob:
    """One export run's shared state (keeps the format writers 1-arg).

    Attributes
    ----------
    reader:
        Open extraction reader.
    selected:
        Keys to export.
    stems:
        Sanitized key -> filename-stem map.
    temp:
        Temp destination directory.
    ledger:
        Destination ledger.
    widen_unsupported:
        Whether bf16/fp8 members widen to fp32 with disclosure.
    conversions:
        Mutable per-key conversion disclosures.
    """

    reader: Any
    selected: list[str]
    stems: dict[str, str]
    temp: Path
    ledger: _DestinationLedger
    widen_unsupported: bool
    conversions: dict[str, Any]


def export_extraction(
    output_dir: str | Path,
    dest: str | Path,
    *,
    format: str = "npy",
    keys: list[str] | None = None,
    widen_unsupported: bool = True,
) -> Path:
    """Export one COMPLETE extraction artifact (read-only, streaming; D15).

    Parameters
    ----------
    output_dir:
        Source artifact directory.
    dest:
        Destination directory (created; published atomically by rename).
    format:
        ``"npy"`` (base install) | ``"hdf5"`` | ``"mat"`` (both behind the
        named optional extra; lazy imports).
    keys:
        Optional output-key subset.
    widen_unsupported:
        Widen bf16/fp8 members to fp32 with disclosure (default) or refuse
        typed.

    Returns
    -------
    pathlib.Path
        The published destination directory.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_export_format_invalid`` outside the format vocabulary;
        ``extraction_export_dependency_missing`` when the extra is absent;
        ``extraction_export_space_insufficient`` on the preflight;
        ``extraction_export_dtype_unsupported`` on refused widening;
        ``extraction_export_mat_v5_limit`` for MAT members over 2 GB.
    """

    if format not in EXPORT_FORMATS:
        raise InvalidArgumentError(
            f"format= value {format!r} is not in the closed vocabulary {EXPORT_FORMATS}.",
            code="extraction_export_format_invalid",
            remedy="pass 'npy', 'hdf5', or 'mat'",
            value=repr(format),
        )
    from .reader import open_extraction

    # Complete-only by construction: open_extraction refuses non-complete
    # artifacts typed unless explicitly opened in_progress (which this
    # exporter never does).
    reader = open_extraction(output_dir)
    selected = keys if keys is not None else reader.keys
    for key in selected:
        reader.logical_shape(key)
    destination = Path(dest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise InvalidArgumentError(
            f"Export destination {str(destination)!r} already exists; the "
            "exporter publishes atomically and never overwrites.",
            code="extraction_export_destination_exists",
            remedy="point dest= at a fresh path (or delete the old export first)",
            dest=str(destination),
        )
    needed = reader.requested_bytes(selected)
    _preflight_destination(destination.parent, needed)
    stems = _disambiguate_sanitized(list(selected))
    temp = destination.parent / (destination.name + ".tmp-export")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    ledger = _DestinationLedger(temp / "export_ledger.jsonl")
    conversions: dict[str, Any] = {}
    job = _ExportJob(
        reader=reader,
        selected=list(selected),
        stems=stems,
        temp=temp,
        ledger=ledger,
        widen_unsupported=widen_unsupported,
        conversions=conversions,
    )
    if format == "npy":
        members = _export_npy(job)
    elif format == "hdf5":
        members = _export_hdf5(job)
    else:
        members = _export_mat(job)
    _write_provenance(reader, temp)
    _write_contract(job, format, members)
    for path in sorted(temp.rglob("*")):
        if path.is_file():
            fsync_file(path)
    os.replace(temp, destination)
    fsync_dir(destination.parent)
    return destination


def _iter_key_batches(reader: Any, key: str) -> Any:
    """Yield one key's per-shard payloads (one shard of working memory).

    Parameters
    ----------
    reader:
        Open extraction reader.
    key:
        Output key.

    Yields
    ------
    torch.Tensor | RaggedBatch
        The key's payload per shard, ledger order.
    """

    for payload in reader.iter_batches(keys=[key]):
        yield payload[key]


def _export_npy(job: _ExportJob) -> list[dict[str, Any]]:
    """Stream keys into preallocated ``.npy`` memmaps (never object arrays).

    Parameters
    ----------
    job:
        The export run's shared state.

    Returns
    -------
    list[dict[str, Any]]
        Contract member rows.
    """

    import numpy as np

    reader, selected, stems = job.reader, job.selected, job.stems
    temp, ledger = job.temp, job.ledger
    widen_unsupported, conversions = job.widen_unsupported, job.conversions
    members: list[dict[str, Any]] = []
    for key in selected:
        stem = stems[key]
        ragged_parts: list[RaggedBatch] = []
        dense_target: Any = None
        cursor = 0
        for payload in _iter_key_batches(reader, key):
            if isinstance(payload, RaggedBatch):
                ragged_parts.append(payload)
                continue
            tensor, conversion = _widen_for_export(key, payload, widen_unsupported)
            if conversion:
                conversions[key] = conversion
            array = tensor.cpu().numpy()
            if dense_target is None:
                shape = (reader.n_stimuli,) + array.shape[1:]
                dense_target = np.lib.format.open_memmap(
                    temp / f"{stem}.npy", mode="w+", dtype=array.dtype, shape=shape
                )
            dense_target[cursor : cursor + array.shape[0]] = array
            cursor += array.shape[0]
        if dense_target is not None:
            dense_target.flush()
            del dense_target
            members.extend(_member_rows(temp, key, [f"{stem}.npy"], "dense", ledger))
        if ragged_parts:
            from .reader import _merge_ragged

            merged = _merge_ragged(ragged_parts)
            values, conversion = _widen_for_export(key, merged.values, widen_unsupported)
            if conversion:
                conversions[key] = conversion
            np.save(temp / f"{stem}__values.npy", values.cpu().numpy(), allow_pickle=False)
            np.save(temp / f"{stem}__offsets.npy", merged.offsets.cpu().numpy(), allow_pickle=False)
            shapes = np.array([list(s) for s in merged.row_shapes], dtype=np.int64)
            np.save(temp / f"{stem}__shapes.npy", shapes, allow_pickle=False)
            members.extend(
                _member_rows(
                    temp,
                    key,
                    [f"{stem}__values.npy", f"{stem}__offsets.npy", f"{stem}__shapes.npy"],
                    "ragged_triplet",
                    ledger,
                )
            )
    return members


def _require_export_extra(module_name: str) -> Any:
    """Import one optional exporter dependency, refusing typed when absent.

    Parameters
    ----------
    module_name:
        ``"h5py"`` or ``"scipy.io"``.

    Returns
    -------
    Any
        The imported module.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_export_dependency_missing`` naming the extra.
    """

    import importlib

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise InvalidArgumentError(
            f"This export format needs {module_name!r}, which is not "
            "installed (it ships behind the optional extra, never on the "
            "default path).",
            code="extraction_export_dependency_missing",
            remedy=f'pip install "torchlens[{_EXPORT_EXTRA}]"',
            module=module_name,
            extra=_EXPORT_EXTRA,
        ) from exc


def _export_hdf5(job: _ExportJob) -> list[dict[str, Any]]:
    """Stream keys into one row-chunked HDF5 file (ragged = numeric groups).

    Parameters
    ----------
    job:
        The export run's shared state.

    Returns
    -------
    list[dict[str, Any]]
        Contract member rows.
    """

    reader, selected, stems = job.reader, job.selected, job.stems
    temp, ledger = job.temp, job.ledger
    widen_unsupported, conversions = job.widen_unsupported, job.conversions

    h5py = _require_export_extra("h5py")
    filename = "extraction.h5"
    with h5py.File(temp / filename, "w") as handle:
        for key in selected:
            stem = stems[key]
            dataset = None
            cursor = 0
            ragged_parts: list[RaggedBatch] = []
            for payload in _iter_key_batches(reader, key):
                if isinstance(payload, RaggedBatch):
                    ragged_parts.append(payload)
                    continue
                tensor, conversion = _widen_for_export(key, payload, widen_unsupported)
                if conversion:
                    conversions[key] = conversion
                array = tensor.cpu().numpy()
                if dataset is None:
                    dataset = handle.create_dataset(
                        stem,
                        shape=(reader.n_stimuli,) + array.shape[1:],
                        dtype=array.dtype,
                        chunks=(min(64, max(1, reader.n_stimuli)),) + array.shape[1:],
                    )
                    dataset.attrs["torchlens_key"] = key
                dataset[cursor : cursor + array.shape[0]] = array
                cursor += array.shape[0]
            if ragged_parts:
                from .reader import _merge_ragged

                merged = _merge_ragged(ragged_parts)
                values, conversion = _widen_for_export(key, merged.values, widen_unsupported)
                if conversion:
                    conversions[key] = conversion
                group = handle.create_group(stem)
                group.attrs["torchlens_key"] = key
                group.create_dataset("values", data=values.cpu().numpy())
                group.create_dataset("offsets", data=merged.offsets.cpu().numpy())
                group.create_dataset(
                    "shapes",
                    data=[list(shape) for shape in merged.row_shapes],
                )
    return _member_rows(temp, "|".join(selected), [filename], "hdf5_file", ledger)


def _export_mat(job: _ExportJob) -> list[dict[str, Any]]:
    """Export keys as MATLAB v5, refusing BEFORE the 2 GB limit (D15).

    Parameters
    ----------
    job:
        The export run's shared state.

    Returns
    -------
    list[dict[str, Any]]
        Contract member rows.
    """

    reader, selected, stems = job.reader, job.selected, job.stems
    temp, ledger = job.temp, job.ledger
    widen_unsupported, conversions = job.widen_unsupported, job.conversions

    scipy_io = _require_export_extra("scipy.io")
    for key in selected:
        member_bytes = reader.requested_bytes([key])
        if member_bytes >= _MAT_V5_LIMIT:
            raise InvalidArgumentError(
                f"Output key {key!r} needs {member_bytes:,} bytes, over the "
                f"MATLAB v5 per-variable limit ({_MAT_V5_LIMIT:,}); an "
                "arbitrary HDF5 file is never mislabeled MAT v7.3.",
                code="extraction_export_mat_v5_limit",
                remedy=(
                    "export format='hdf5' and read it in MATLAB with "
                    "h5read('extraction.h5', '/<key>')"
                ),
                key=key,
                needed_bytes=member_bytes,
            )
    payload: dict[str, Any] = {}
    for key in selected:
        stem = _mat_identifier(stems[key])
        materialized = reader.materialize([key], max_bytes=_MAT_V5_LIMIT)[key]
        if isinstance(materialized, RaggedBatch):
            values, conversion = _widen_for_export(key, materialized.values, widen_unsupported)
            if conversion:
                conversions[key] = conversion
            payload[f"{stem}__values"] = values.cpu().numpy()
            payload[f"{stem}__offsets"] = materialized.offsets.cpu().numpy()
            payload[f"{stem}__shapes"] = [list(shape) for shape in materialized.row_shapes]
        else:
            tensor, conversion = _widen_for_export(key, materialized, widen_unsupported)
            if conversion:
                conversions[key] = conversion
            payload[stem] = tensor.cpu().numpy()
    filename = "extraction.mat"
    scipy_io.savemat(temp / filename, payload, do_compression=False)
    return _member_rows(temp, "|".join(selected), [filename], "mat_v5_file", ledger)


def _mat_identifier(stem: str) -> str:
    """Coerce a sanitized stem into a legal MATLAB identifier.

    Parameters
    ----------
    stem:
        Sanitized filename stem.

    Returns
    -------
    str
        Alphanumeric/underscore identifier starting with a letter.
    """

    cleaned = "".join(char if char.isalnum() else "_" for char in stem)
    if not cleaned or not cleaned[0].isalpha():
        cleaned = f"k_{cleaned}"
    return cleaned[:63]


def _member_rows(
    temp: Path,
    key: str,
    filenames: list[str],
    role: str,
    ledger: _DestinationLedger,
) -> list[dict[str, Any]]:
    """Ledger and describe freshly written export members.

    Parameters
    ----------
    temp:
        Temp destination.
    key:
        Source output key (or joined keys for single-file formats).
    filenames:
        Member filenames.
    role:
        Member role label.
    ledger:
        Destination ledger.

    Returns
    -------
    list[dict[str, Any]]
        Contract member rows with export-time sha256 and byte sizes.
    """

    rows = []
    for filename in filenames:
        path = temp / filename
        row = {
            "file": filename,
            "source_key": key,
            "role": role,
            "byte_size": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        ledger.append(row)
        rows.append(row)
    return rows


def _write_provenance(reader: Any, temp: Path) -> None:
    """Copy source manifest + ID sidecar byte-verbatim under provenance/.

    Parameters
    ----------
    reader:
        Open extraction reader.
    temp:
        Temp destination.
    """

    from .._data_substrate import LEDGER_FILENAME, STIMULUS_IDS_FILENAME

    provenance = temp / _PROVENANCE_SUBDIR
    provenance.mkdir()
    source = Path(reader._container)
    for name in ("manifest.json", STIMULUS_IDS_FILENAME, LEDGER_FILENAME):
        candidate = source / name
        if candidate.exists():
            shutil.copyfile(candidate, provenance / name)


def _write_contract(
    job: _ExportJob,
    export_format: str,
    members: list[dict[str, Any]],
) -> None:
    """Write the self-contained export contract file (D15).

    Parameters
    ----------
    job:
        The export run's shared state.
    export_format:
        Export format.
    members:
        Contract member rows.
    """

    reader, selected, stems = job.reader, job.selected, job.stems
    temp, conversions = job.temp, job.conversions

    from .._data_substrate import LEDGER_FILENAME

    manifest = reader.manifest
    source = Path(reader._container)
    ledger_path = source / LEDGER_FILENAME
    ids = reader.stimulus_ids()
    contract = {
        "schema": "tl_extraction_export_v1",
        "format": export_format,
        "source": {
            "manifest_sha256": _sha256_file(source / "manifest.json"),
            "ledger_sha256": _sha256_file(ledger_path) if ledger_path.exists() else None,
            "stimulus_ids_digest": (manifest.get("signature") or {}).get("stimulus_ids_digest"),
            "n_stimuli": reader.n_stimuli,
            "n_shards": reader.n_shards,
            "unknown_v1_prefix_semantics": bool(manifest.get("unknown_v1_prefix_semantics", False)),
        },
        "row_order": (
            "row i of every exported member is stimulus i in the source "
            "artifact's iteration order; stimulus_ids (when recorded) map "
            "identifiers to row indices, duplicates preserved in order"
        ),
        "stimulus_ids": ids,
        "keys": {
            key: {
                "sanitized_stem": stems[key],
                "logical_shape": reader.logical_shape(key),
                "layer": (manifest.get("layers") or {}).get(key),
                "conversion": conversions.get(key),
            }
            for key in selected
        },
        "members": members,
        "provenance_subpath": _PROVENANCE_SUBDIR,
        "ragged_encoding": (
            "trimmed keys ship as the values/offsets/shapes triplet: row i "
            "is values[offsets[i]:offsets[i+1]] reshaped to shapes[i]"
        ),
    }
    (temp / EXPORT_CONTRACT_FILENAME).write_text(
        json.dumps(contract, indent=1, sort_keys=False), encoding="utf-8"
    )
