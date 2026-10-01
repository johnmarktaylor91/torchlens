"""The lazy extraction reader: the documented read path (extract D14).

Opening reads BOUNDED manifest/ledger metadata only. The reader supplies
metadata, keys, logical shapes, ledger-order ``iter_batches``, integer /
slice / fancy row access via cumulative offsets, duplicate-aware ID lookup,
exact ragged rows, explicit materialization behind an exact byte guard,
trusted-prefix monitoring of in-progress artifacts, and graded verification
(extract D7 read side: structural by default, ``verify="first_access"``
verifies each shard once and caches the verdict for the handle's lifetime,
``verify_all()`` streams everything — never a full-shard scan per row
slice).

Row requests are COALESCED into contiguous spans per shard and safetensors
``safe_open`` handles are cached (safetensors loses 1.8x to .pt-mmap on 64
scattered single-row reads of an already-open shard — the motivation for
both; T-COALESCE gates the wiring).

Every spelling here is DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

import bisect
import zlib
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

from .._data_substrate import ExtractionArtifactError, read_trusted_rows
from .._errors import InvalidArgumentError
from .ragged import RaggedBatch, to_padded
from .shards import read_shard

__tl_layer__ = "L5"

__all__ = ["ExtractionReader", "open_extraction"]

#: Maximum simultaneously cached shard handles.
_HANDLE_CACHE_SIZE = 64

#: Reserved trimmed-carrier subkeys inside safetensors shards.
_RAGGED_SUBKEYS = ("values", "offsets", "shapes")


def _crc32_file(path: Path) -> int:
    """Stream one file's CRC-32 (the D7 accidental-corruption fact).

    Parameters
    ----------
    path:
        File to checksum.

    Returns
    -------
    int
        CRC-32 over the final file bytes.
    """

    crc = 0
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                return crc
            crc = zlib.crc32(chunk, crc)


class _ShardHandles:
    """Bounded cache of per-shard read handles (D14).

    safetensors shards cache ``safe_open`` objects (keyed row slicing
    without materializing the file); ``pt`` shards cache the mmap-loaded
    payload dict.
    """

    def __init__(self, capacity: int = _HANDLE_CACHE_SIZE) -> None:
        """Initialize an empty cache.

        Parameters
        ----------
        capacity:
            Maximum cached handles; the least recently used evicts first.
        """

        self._capacity = capacity
        self._handles: OrderedDict[int, Any] = OrderedDict()

    def get(self, index: int, opener: Any) -> Any:
        """Return the cached handle for one shard, opening on miss.

        Parameters
        ----------
        index:
            Shard index.
        opener:
            Zero-arg callable producing the handle.

        Returns
        -------
        Any
            The cached or fresh handle.
        """

        if index in self._handles:
            self._handles.move_to_end(index)
            return self._handles[index]
        handle = opener()
        self._handles[index] = handle
        if len(self._handles) > self._capacity:
            self._handles.popitem(last=False)
        return handle


class ExtractionReader:
    """Lazy reader over one extraction artifact (extract D14).

    Construct through :func:`open_extraction`. Opening reads bounded
    manifest and ledger metadata only; payload bytes move on demand.
    """

    def __init__(
        self,
        container: Path,
        manifest: dict[str, Any],
        rows: list[dict[str, Any]],
        *,
        verify: str = "structural",
        in_progress: bool = False,
    ) -> None:
        """Bind the reader to its verified artifact state.

        Parameters
        ----------
        container:
            Artifact directory.
        manifest:
            Parsed manifest document.
        rows:
            Trusted ledger rows in ledger order.
        verify:
            ``"structural"`` (default) or ``"first_access"`` (each shard's
            CRC verified once, verdict cached for this handle's lifetime).
        in_progress:
            Whether this handle monitors a still-writing artifact (its
            trusted prefix; :meth:`refresh` rescans).
        """

        self._container = container
        self._manifest = manifest
        self._rows = rows
        self._verify = verify
        self._in_progress = in_progress
        self._handles = _ShardHandles()
        self._verified: dict[int, bool] = {}
        self._id_index: dict[str, list[int]] | None = None
        self._row_starts = [int(row["row_start"]) for row in rows]

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def manifest(self) -> dict[str, Any]:
        """Return the parsed manifest document.

        Returns
        -------
        dict[str, Any]
            The bounded manifest header.
        """

        return self._manifest

    @property
    def shard_format(self) -> str:
        """Return the artifact's recorded shard format.

        Returns
        -------
        str
            ``"safetensors"`` or ``"pt"`` (v1 artifacts read as ``"pt"``).
        """

        return str((self._manifest.get("storage") or {}).get("shard_format", "pt"))

    @property
    def keys(self) -> list[str]:
        """Return the artifact's output keys.

        Returns
        -------
        list[str]
            Keys from the manifest layers block.
        """

        return sorted((self._manifest.get("layers") or {}).keys())

    @property
    def n_stimuli(self) -> int:
        """Return the number of trusted stimulus rows.

        Returns
        -------
        int
            Total rows over the trusted ledger prefix.
        """

        if not self._rows:
            return 0
        last = self._rows[-1]
        return int(last["row_start"]) + int(last["n_rows"])

    @property
    def n_shards(self) -> int:
        """Return the number of trusted shards.

        Returns
        -------
        int
            Trusted ledger-row count.
        """

        return len(self._rows)

    def logical_shape(self, key: str) -> list[Any]:
        """Return one key's per-stimulus logical shape.

        Parameters
        ----------
        key:
            Output key.

        Returns
        -------
        list[Any]
            The manifest's stored per-stimulus shape (``None`` entries mark
            ragged axes).
        """

        layer = (self._manifest.get("layers") or {}).get(key)
        if layer is None:
            raise InvalidArgumentError(
                f"Output key {key!r} is not in this artifact (available: {self.keys}).",
                code="extraction_reader_key_unknown",
                remedy="request a key recorded in the manifest layers block",
                key=key,
                available=self.keys,
            )
        return list(layer.get("stored_per_stimulus_shape") or [])

    def stimulus_ids(self) -> list[str] | None:
        """Return the ordered stimulus-id sidecar, when the run recorded one.

        Returns
        -------
        list[str] | None
            Ordered identifiers, or ``None``.
        """

        from .._data_substrate import STIMULUS_IDS_FILENAME
        from .._io import _json

        sidecar = self._container / STIMULUS_IDS_FILENAME
        if not sidecar.exists():
            return None
        payload = _json.read_bounded(sidecar)
        ids = payload.get("ids") if isinstance(payload, dict) else None
        return [str(item) for item in ids] if isinstance(ids, list) else None

    # ------------------------------------------------------------------
    # ID lookup (duplicate-aware)
    # ------------------------------------------------------------------

    def _ensure_id_index(self) -> dict[str, list[int]]:
        """Build (once) and return the duplicate-aware ID multimap.

        Returns
        -------
        dict[str, list[int]]
            ``id -> ordered row indices``.
        """

        if self._id_index is None:
            ids = self.stimulus_ids()
            if ids is None:
                raise InvalidArgumentError(
                    "This artifact records no stimulus_ids sidecar, so rows "
                    "cannot be addressed by identifier.",
                    code="extraction_reader_ids_unavailable",
                    remedy="extract with stimulus_ids=, or address rows by index",
                )
            index: dict[str, list[int]] = {}
            for row, item in enumerate(ids):
                index.setdefault(item, []).append(row)
            self._id_index = index
        return self._id_index

    def rows_for(self, stimulus_id: str) -> list[int]:
        """Return EVERY row index recorded under one identifier.

        Parameters
        ----------
        stimulus_id:
            Stimulus identifier.

        Returns
        -------
        list[int]
            Ordered row indices (duplicates are legal and all returned).
        """

        index = self._ensure_id_index()
        if stimulus_id not in index:
            raise InvalidArgumentError(
                f"Stimulus id {stimulus_id!r} is not in this artifact's id sidecar.",
                code="extraction_reader_id_unknown",
                remedy="look up a recorded id (reader.stimulus_ids() lists them)",
                stimulus_id=stimulus_id,
            )
        return list(index[stimulus_id])

    def row_for(self, stimulus_id: str) -> int:
        """Return THE row index for one identifier, refusing ambiguity.

        Parameters
        ----------
        stimulus_id:
            Stimulus identifier.

        Returns
        -------
        int
            The unique row index.

        Raises
        ------
        torchlens.errors.InvalidArgumentError
            ``extraction_reader_id_ambiguous`` when the id names several
            rows (``rows_for`` is the multimap spelling).
        """

        rows = self.rows_for(stimulus_id)
        if len(rows) > 1:
            raise InvalidArgumentError(
                f"Stimulus id {stimulus_id!r} names {len(rows)} rows "
                f"({rows[:10]}...); duplicates are legal in artifacts, so "
                "the singular lookup refuses.",
                code="extraction_reader_id_ambiguous",
                remedy="use rows_for(id) to get every row index",
                stimulus_id=stimulus_id,
                n_rows=len(rows),
            )
        return rows[0]

    # ------------------------------------------------------------------
    # Verification (D7 read side)
    # ------------------------------------------------------------------

    def _check_shard(self, shard_index: int) -> None:
        """Apply the graded verification policy before one shard read.

        Parameters
        ----------
        shard_index:
            Shard about to be read.

        Raises
        ------
        torchlens._data_substrate.ExtractionArtifactError
            ``extraction_shard_integrity_mismatch`` on a CRC mismatch under
            ``verify="first_access"``.
        """

        if self._verify != "first_access" or self._verified.get(shard_index):
            return
        self._verify_shard_crc(shard_index)
        self._verified[shard_index] = True

    def _verify_shard_crc(self, shard_index: int) -> None:
        """Verify one shard's CRC-32 against its ledger row.

        Parameters
        ----------
        shard_index:
            Shard to verify.

        Raises
        ------
        torchlens._data_substrate.ExtractionArtifactError
            ``extraction_shard_integrity_mismatch`` when the file's CRC
            differs from the ledgered fact (the trusted prefix ends here).
        """

        row = self._rows[shard_index]
        expected = row.get("crc32")
        if expected is None:
            return
        observed = _crc32_file(self._container / str(row["file"]))
        if observed != expected:
            raise ExtractionArtifactError(
                f"Shard {row['file']!r} fails its ledgered CRC-32 "
                f"(ledgered {expected}, observed {observed}); the file's "
                "bytes changed after commit, and the trusted prefix ends "
                "here.",
                code="extraction_shard_integrity_mismatch",
                remedy=(
                    "restore the shard from a backup, or re-extract; "
                    "checksum level 'fast' detects accidental corruption "
                    "only (see the manifest integrity block)"
                ),
                shard=str(row["file"]),
                index=shard_index,
                ledgered_crc32=expected,
                observed_crc32=observed,
            )

    def verify_all(self) -> dict[str, Any]:
        """Stream-verify every trusted shard (CRC + value reductions).

        Returns
        -------
        dict[str, Any]
            ``{"n_shards": ..., "crc_checked": ..., "values_checked": ...}``.

        Raises
        ------
        torchlens._data_substrate.ExtractionArtifactError
            ``extraction_shard_integrity_mismatch`` on the first mismatch
            of either labeled integrity fact.
        """

        from .._data_substrate import value_reduction

        crc_checked = 0
        values_checked = 0
        for shard_index, row in enumerate(self._rows):
            self._verify_shard_crc(shard_index)
            crc_checked += 1
            payload = read_shard(self._container / str(row["file"]), self.shard_format, keys=None)
            for key, facts in (row.get("keys") or {}).items():
                recorded = facts.get("value_reduction")
                if recorded is None or key not in payload:
                    continue
                stored = payload[key]
                tensor = stored.values if isinstance(stored, RaggedBatch) else stored
                observed = value_reduction(key, tensor)
                values_checked += 1
                if observed != recorded:
                    raise ExtractionArtifactError(
                        f"Shard {row['file']!r} key {key!r} fails its "
                        "ledgered value reduction: the values corrupted "
                        "between the on-device computation and this read "
                        "(the file-level CRC matched, so the file was "
                        "faithfully written).",
                        code="extraction_shard_integrity_mismatch",
                        remedy="re-extract; the artifact's values cannot be trusted",
                        shard=str(row["file"]),
                        key=key,
                    )
        return {
            "n_shards": len(self._rows),
            "crc_checked": crc_checked,
            "values_checked": values_checked,
        }

    # ------------------------------------------------------------------
    # Batch iteration + row access
    # ------------------------------------------------------------------

    def iter_batches(self, keys: list[str] | None = None) -> Any:
        """Iterate shard payloads in LEDGER order (never filename order).

        Parameters
        ----------
        keys:
            Optional output-key subset.

        Yields
        ------
        dict[str, torch.Tensor | RaggedBatch]
            One shard's payload at a time (bounded memory: one shard).
        """

        if keys is not None:
            for key in keys:
                self.logical_shape(key)  # refuses unknown keys typed
        for shard_index in range(len(self._rows)):
            yield self.batch(shard_index, keys=keys)

    def batch(self, shard_index: int, keys: list[str] | None = None) -> dict[str, Any]:
        """Read ONE shard's payload by ledger position (bounded memory).

        Parameters
        ----------
        shard_index:
            Ledger-order shard position.
        keys:
            Optional output-key subset.

        Returns
        -------
        dict[str, torch.Tensor | RaggedBatch]
            The shard's payload.
        """

        if shard_index < 0 or shard_index >= len(self._rows):
            raise InvalidArgumentError(
                f"Shard {shard_index} is out of range for an artifact with "
                f"{len(self._rows)} trusted shards.",
                code="extraction_reader_row_out_of_range",
                remedy="request shards in [0, n_shards)",
                shard=shard_index,
                n_shards=len(self._rows),
            )
        row = self._rows[shard_index]
        self._check_shard(shard_index)
        return read_shard(self._container / str(row["file"]), self.shard_format, keys=keys)

    def _shard_for_row(self, row_index: int) -> tuple[int, int]:
        """Locate the shard holding one global row.

        Parameters
        ----------
        row_index:
            Global row index.

        Returns
        -------
        tuple[int, int]
            ``(shard_index, local_row)``.
        """

        if row_index < 0 or row_index >= self.n_stimuli:
            raise InvalidArgumentError(
                f"Row {row_index} is out of range for an artifact with "
                f"{self.n_stimuli} trusted rows.",
                code="extraction_reader_row_out_of_range",
                remedy="request rows in [0, n_stimuli)",
                row=row_index,
                n_stimuli=self.n_stimuli,
            )
        shard_index = bisect.bisect_right(self._row_starts, row_index) - 1
        return shard_index, row_index - self._row_starts[shard_index]

    def _dense_spans(
        self, key: str, spans_by_shard: dict[int, list[tuple[int, int]]]
    ) -> list[torch.Tensor]:
        """Read contiguous local spans of one dense key per shard.

        safetensors handles slice through ``get_slice`` (no full-file
        materialization); ``pt`` handles index the mmap-loaded payload.

        Parameters
        ----------
        key:
            Dense output key.
        spans_by_shard:
            ``shard_index -> [(local_start, local_stop), ...]``.

        Returns
        -------
        list[torch.Tensor]
            One tensor per span, in (shard, span) order.
        """

        out: list[torch.Tensor] = []
        for shard_index in sorted(spans_by_shard):
            self._check_shard(shard_index)
            row = self._rows[shard_index]
            path = self._container / str(row["file"])
            if self.shard_format == "safetensors":
                from safetensors import safe_open

                handle = self._handles.get(
                    shard_index, lambda p=path: safe_open(str(p), framework="pt")
                )
                shard_keys = set(handle.keys())
                for start, stop in spans_by_shard[shard_index]:
                    if key in shard_keys:
                        out.append(handle.get_slice(key)[start:stop])
                    elif f"{key}/values" in shard_keys:
                        raise InvalidArgumentError(
                            f"Output key {key!r} is stored TRIMMED in shard "
                            f"{row['file']!r}; dense span reads do not apply.",
                            code="extraction_reader_key_unknown",
                            remedy="read ragged rows via reader.row(...) or to_padded(...)",
                            key=key,
                        )
                    else:
                        raise InvalidArgumentError(
                            f"Output key {key!r} is not in shard {row['file']!r}.",
                            code="extraction_reader_key_unknown",
                            remedy="request a key recorded in the manifest layers block",
                            key=key,
                        )
            else:
                handle = self._handles.get(
                    shard_index,
                    lambda p=path: read_shard(p, "pt", keys=None),
                )
                payload = handle[key]
                if isinstance(payload, RaggedBatch):
                    raise InvalidArgumentError(
                        f"Output key {key!r} is stored TRIMMED in shard "
                        f"{row['file']!r}; dense span reads do not apply.",
                        code="extraction_reader_key_unknown",
                        remedy="read ragged rows via reader.row(...) or to_padded(...)",
                        key=key,
                    )
                for start, stop in spans_by_shard[shard_index]:
                    out.append(payload[start:stop])
        return out

    def rows(self, indices: Any, keys: list[str] | None = None) -> dict[str, Any]:
        """Read specific rows, COALESCED into contiguous spans per shard.

        Parameters
        ----------
        indices:
            Integer, slice, or iterable of integers (fancy indexing).
        keys:
            Optional output-key subset.

        Returns
        -------
        dict[str, Any]
            Per-key payload: a stacked dense tensor in REQUEST order, or —
            for trimmed keys — a list of exact ragged rows in request
            order.
        """

        if isinstance(indices, slice):
            index_list = list(range(*indices.indices(self.n_stimuli)))
        elif isinstance(indices, int):
            index_list = [indices]
        else:
            index_list = [int(i) for i in indices]
        located = [self._shard_for_row(i) for i in index_list]
        selected_keys = keys if keys is not None else self.keys
        for key in selected_keys:
            self.logical_shape(key)
        # Coalesce: per shard, sorted local rows -> contiguous spans.
        spans_by_shard: dict[int, list[tuple[int, int]]] = {}
        for shard_index, local in sorted(set(located)):
            spans = spans_by_shard.setdefault(shard_index, [])
            if spans and spans[-1][1] == local:
                spans[-1] = (spans[-1][0], local + 1)
            else:
                spans.append((local, local + 1))
        result: dict[str, Any] = {}
        for key in selected_keys:
            if self._key_is_trimmed(key):
                result[key] = [self._trimmed_row(key, shard, local) for shard, local in located]
                continue
            result[key] = self._assemble_dense_rows(key, spans_by_shard, located)
        return result

    def _assemble_dense_rows(
        self,
        key: str,
        spans_by_shard: dict[int, list[tuple[int, int]]],
        located: list[tuple[int, int]],
    ) -> torch.Tensor:
        """Read coalesced spans and reassemble the caller's request order.

        Parameters
        ----------
        key:
            Dense output key.
        spans_by_shard:
            Coalesced contiguous spans per shard.
        located:
            ``(shard_index, local_row)`` per requested row, request order.

        Returns
        -------
        torch.Tensor
            Rows stacked in request order.
        """

        span_tensors = self._dense_spans(key, spans_by_shard)
        position: dict[tuple[int, int], torch.Tensor] = {}
        cursor = 0
        for shard_index in sorted(spans_by_shard):
            for start, stop in spans_by_shard[shard_index]:
                tensor = span_tensors[cursor]
                cursor += 1
                for offset in range(stop - start):
                    position[(shard_index, start + offset)] = tensor[offset]
        return torch.stack([position[loc] for loc in located])

    def __getitem__(self, index: int | slice) -> dict[str, Any]:
        """Read one row (or slice of rows) across every key.

        Parameters
        ----------
        index:
            Row index or slice.

        Returns
        -------
        dict[str, Any]
            Per-key payloads in request order.
        """

        return self.rows(index)

    def _key_is_trimmed(self, key: str) -> bool:
        """Return whether one key is stored as the trimmed carrier.

        Parameters
        ----------
        key:
            Output key.

        Returns
        -------
        bool
            ``True`` when the manifest layers block marks the key trimmed.
        """

        layer = (self._manifest.get("layers") or {}).get(key) or {}
        return layer.get("layout") == "trimmed"

    def _trimmed_row(self, key: str, shard_index: int, local_row: int) -> torch.Tensor:
        """Read one exact ragged row of a trimmed key.

        Parameters
        ----------
        key:
            Trimmed output key.
        shard_index:
            Shard holding the row.
        local_row:
            Row index within the shard.

        Returns
        -------
        torch.Tensor
            The row at its true extent.
        """

        self._check_shard(shard_index)
        row = self._rows[shard_index]
        payload = read_shard(self._container / str(row["file"]), self.shard_format, keys=[key])
        carrier = payload.get(key)
        if not isinstance(carrier, RaggedBatch):
            raise InvalidArgumentError(
                f"Output key {key!r} in shard {row['file']!r} is not the "
                "trimmed carrier its manifest entry declares.",
                code="extraction_reader_key_unknown",
                remedy="re-extract; the artifact is internally inconsistent",
                key=key,
            )
        return carrier.row(local_row)

    def row(self, index: int, keys: list[str] | None = None) -> dict[str, Any]:
        """Read one row across keys (exact ragged rows for trimmed keys).

        Parameters
        ----------
        index:
            Global row index.
        keys:
            Optional output-key subset.

        Returns
        -------
        dict[str, Any]
            Per-key single-row payloads.
        """

        selected = keys if keys is not None else self.keys
        shard_index, local = self._shard_for_row(index)
        out: dict[str, Any] = {}
        for key in selected:
            if self._key_is_trimmed(key):
                out[key] = self._trimmed_row(key, shard_index, local)
            else:
                out[key] = self.rows([index], keys=[key])[key][0]
        return out

    def to_padded(
        self, key: str, *, pad_value: float = 0.0, max_len: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize one trimmed key as padded values + mask (D4).

        Value-equivalent within the cross-batch tolerance, NOT
        byte-identical to a historical padded pipeline.

        Parameters
        ----------
        key:
            Trimmed output key.
        pad_value:
            Fill value for padded positions.
        max_len:
            Optional fixed width.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(values, mask)``.
        """

        if not self._key_is_trimmed(key):
            raise InvalidArgumentError(
                f"Output key {key!r} is stored dense; to_padded applies to "
                "trimmed keys (dense keys are already rectangular).",
                code="extraction_reader_key_unknown",
                remedy="read dense keys via rows()/materialize()",
                key=key,
            )
        carriers: list[RaggedBatch] = []
        for payload in self.iter_batches(keys=[key]):
            carrier = payload.get(key)
            if isinstance(carrier, RaggedBatch):
                carriers.append(carrier)
        return to_padded(carriers, pad_value=pad_value, max_len=max_len)

    # ------------------------------------------------------------------
    # Materialization (byte-guarded)
    # ------------------------------------------------------------------

    def requested_bytes(self, keys: list[str] | None = None) -> int:
        """Compute the EXACT bytes a materialization would allocate.

        Derived from ledger facts alone — per-key shapes, dtypes, and
        trimmed value counts — never from opening payloads.

        Parameters
        ----------
        keys:
            Optional output-key subset.

        Returns
        -------
        int
            Exact byte total.
        """

        selected = set(keys if keys is not None else self.keys)
        total = 0
        for row in self._rows:
            for key, facts in (row.get("keys") or {}).items():
                if key not in selected:
                    continue
                dtype_size = _dtype_size(str(facts.get("dtype")))
                if facts.get("layout") == "trimmed":
                    total += int(facts.get("n_values", 0)) * dtype_size
                    continue
                numel = int(row.get("n_rows", 0))
                for dim in facts.get("per_stimulus_shape") or []:
                    numel *= int(dim)
                total += numel * dtype_size
        return total

    def materialize(
        self, keys: list[str] | None = None, *, max_bytes: int | None = None
    ) -> dict[str, Any]:
        """Explicitly materialize keys into memory (the one eager door).

        Parameters
        ----------
        keys:
            Optional output-key subset.
        max_bytes:
            Optional explicit budget override; ``None`` applies the
            default guard (half of measurable available host memory, else
            8 GiB).

        Returns
        -------
        dict[str, Any]
            Dense keys concatenate to one tensor; trimmed keys return one
            merged :class:`RaggedBatch`.

        Raises
        ------
        torchlens.errors.InvalidArgumentError
            ``extraction_eager_budget_exceeded`` when the EXACT requested
            bytes exceed the budget; the refusal names the number, this
            reader, and the override.
        """

        needed = self.requested_bytes(keys)
        budget = max_bytes if max_bytes is not None else _default_byte_budget()
        if needed > budget:
            raise InvalidArgumentError(
                f"Materializing {sorted(keys) if keys else 'all keys'} needs "
                f"exactly {needed:,} bytes, over the {budget:,}-byte budget. "
                "The lazy reader (open_extraction) serves batches, rows, and "
                "views without materializing the artifact.",
                code="extraction_eager_budget_exceeded",
                remedy=(
                    "read lazily via open_extraction(...), request fewer "
                    "keys, or override explicitly with max_bytes="
                ),
                requested_bytes=needed,
                budget_bytes=budget,
            )
        dense: dict[str, list[torch.Tensor]] = {}
        ragged: dict[str, list[RaggedBatch]] = {}
        for payload in self.iter_batches(keys=keys):
            for key, value in payload.items():
                if isinstance(value, RaggedBatch):
                    ragged.setdefault(key, []).append(value)
                else:
                    dense.setdefault(key, []).append(value)
        out: dict[str, Any] = {key: torch.cat(parts, dim=0) for key, parts in dense.items()}
        for key, carriers in ragged.items():
            out[key] = _merge_ragged(carriers)
        return out

    # ------------------------------------------------------------------
    # Trusted-prefix monitoring
    # ------------------------------------------------------------------

    def refresh(self) -> int:
        """Rescan an in-progress artifact's ledger for newly committed shards.

        Returns
        -------
        int
            Newly trusted shard count since the last scan.

        Raises
        ------
        torchlens.errors.InvalidArgumentError
            ``extraction_reader_not_monitoring`` when the reader was not
            opened with ``in_progress=True``.
        """

        if not self._in_progress:
            raise InvalidArgumentError(
                "refresh() monitors a still-writing artifact and this "
                "reader was opened for a completed one.",
                code="extraction_reader_not_monitoring",
                remedy="open_extraction(dir, in_progress=True) to monitor",
            )
        before = len(self._rows)
        self._rows = read_trusted_rows(self._container)
        self._row_starts = [int(row["row_start"]) for row in self._rows]
        return len(self._rows) - before


def _merge_ragged(carriers: list[RaggedBatch]) -> RaggedBatch:
    """Concatenate trimmed carriers into one artifact-wide carrier.

    Parameters
    ----------
    carriers:
        Per-shard carriers in ledger order.

    Returns
    -------
    RaggedBatch
        The merged carrier.
    """

    values = torch.cat([carrier.values for carrier in carriers], dim=0)
    row_shapes: tuple[tuple[int, ...], ...] = tuple(
        shape for carrier in carriers for shape in carrier.row_shapes
    )
    offsets = torch.zeros(len(row_shapes) + 1, dtype=torch.int64)
    cursor = 0
    position = 0
    for carrier in carriers:
        for i in range(carrier.row_count):
            cursor += int(carrier.offsets[i + 1] - carrier.offsets[i])
            position += 1
            offsets[position] = cursor
    return RaggedBatch(values=values, offsets=offsets, row_shapes=row_shapes)


def _dtype_size(dtype_name: str) -> int:
    """Return the element size in bytes for a stored dtype name.

    Parameters
    ----------
    dtype_name:
        ``str(torch.dtype)`` form (``"torch.float32"``).

    Returns
    -------
    int
        Element byte size (1 for unknown names — a conservative floor).
    """

    name = dtype_name.removeprefix("torch.")
    dtype = getattr(torch, name, None)
    if isinstance(dtype, torch.dtype):
        return torch.empty((), dtype=dtype).element_size()
    return 1


def _default_byte_budget() -> int:
    """Return the default eager-materialization byte budget.

    Returns
    -------
    int
        Half of measurable available host memory, else 8 GiB.
    """

    import os

    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return (pages * page_size) // 2
    except (ValueError, OSError, AttributeError):
        pass
    return 8 * 1024**3


def open_extraction(
    output_dir: str | Path,
    *,
    verify: str = "structural",
    in_progress: bool = False,
) -> ExtractionReader:
    """Open one extraction artifact lazily (the documented read path, D14).

    Opening reads bounded manifest/ledger metadata only; payload bytes move
    on demand through the reader.

    Parameters
    ----------
    output_dir:
        Artifact directory written by disk-mode ``extract_dataset``.
    verify:
        ``"structural"`` (default: manifest/ledger well-formed, files
        present at their exact ledgered byte sizes) or ``"first_access"``
        (each shard's CRC verified once per handle lifetime).
    in_progress:
        Open a still-writing artifact's trusted committed prefix;
        ``reader.refresh()`` rescans.

    Returns
    -------
    ExtractionReader
        The lazy reader.

    Raises
    ------
    torchlens.dataset_extraction.DatasetExtractionResumeError
        On a missing/invalid manifest or an incomplete artifact opened
        without ``in_progress=True``.
    torchlens.errors.InvalidArgumentError
        ``extraction_reader_verify_invalid`` outside the closed
        verification vocabulary.
    """

    if verify not in ("structural", "first_access"):
        raise InvalidArgumentError(
            f"verify= value {verify!r} is not in the closed vocabulary "
            "('structural', 'first_access'); full verification is the "
            "explicit reader.verify_all() call, never a per-read default.",
            code="extraction_reader_verify_invalid",
            remedy="pass verify='structural' or verify='first_access'",
            value=repr(verify),
        )
    from ..dataset_extraction import (
        MANIFEST_FILENAME,
        MANIFEST_SCHEMA_V2,
        DatasetExtractionResumeError,
        _load_manifest,
    )

    container = Path(output_dir)
    manifest = _load_manifest(container / MANIFEST_FILENAME)
    if manifest.get("schema") != MANIFEST_SCHEMA_V2:
        from .._data_substrate import migrate_v1_artifact

        manifest, _rows = migrate_v1_artifact(container, manifest)
    if manifest.get("status") != "complete" and not in_progress:
        raise DatasetExtractionResumeError(
            f"Extraction artifact at {str(container)!r} has status "
            f"{manifest.get('status')!r}; open the trusted committed prefix "
            "explicitly to monitor a still-writing run.",
            code="extraction_manifest_invalid",
            remedy=(
                "finish the run (extract_dataset(..., resume=True)) or open "
                "with open_extraction(dir, in_progress=True)"
            ),
            status=manifest.get("status"),
        )
    rows = read_trusted_rows(container)
    if manifest.get("status") == "complete":
        n_ledgered = (manifest.get("totals") or {}).get("n_shards")
        if len(rows) != n_ledgered:
            raise DatasetExtractionResumeError(
                f"Extraction artifact at {str(container)!r} ledgers "
                f"{len(rows)} trusted shard rows but its terminal totals "
                f"record {n_ledgered}.",
                code="extraction_manifest_invalid",
                remedy="re-run extract_dataset(..., resume=True) to finish the artifact",
                n_present=len(rows),
                n_ledgered=n_ledgered,
            )
    return ExtractionReader(container, manifest, rows, verify=verify, in_progress=in_progress)
