"""Versioned per-step history artifact: disk-first, append-only (D19/D20).

Directory layout::

    <dir>/
      manifest.json      -- run record + descriptor + schema version (atomic)
      catalog.json       -- site catalog, rewritten atomically on site adds
      index.json         -- committed chunk index (atomic rewrite per commit)
      chunk-000000.npz   -- checksummed NumPy column chunks (no pickle)

A chunk becomes real only when its row enters ``index.json``: the chunk file
is written to a temp name, fsynced, renamed, and only then indexed -- so a
killed writer leaves the artifact readable to the last committed chunk. A
committed chunk whose bytes later fail their recorded sha256 refuses typed:
recovery never silently drops committed data.

The RAM ring (default 256 committed StepBlocks) is a CACHE when the disk
archive is on -- eviction is plain LRU and not an event. RAM-only mode has
three EXPLICIT policies (D19/D20): ``refuse`` (default; typed refusal BEFORE
the next scheduled block naming the remedies), ``drop_oldest`` (with
counters), and ``coarsen`` (exact pairwise count-preserving merges, each
record carrying its [step_lo, step_hi] span).

Spellings are DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

import hashlib
import io as _io_module
import json
import os
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .._io._json import read_bounded
from ._errors import HistoryArtifactError
from ._kernels import HistogramDescriptor, HistogramResult, SpineResult
from ._schema import (
    HISTORY_SCHEMA_VERSION,
    ObservationRecord,
    RunRecord,
    SiteRecord,
    StepBlockRecord,
    merge_observations,
)

__tl_layer__ = "L5"

#: Default RAM ring capacity in committed StepBlocks (D19).
DEFAULT_RING_CAPACITY = 256

#: Closed RAM-only retention policies (D19/D20).
RAM_POLICIES = ("refuse", "drop_oldest", "coarsen")

_SPINE_FLOAT_FIELDS = (
    "finite_min",
    "finite_max",
    "finite_absmax",
    "sum",
    "sum_squares",
    "sum_abs",
    "mean",
    "m2",
)
_SPINE_COUNT_FIELDS = (
    "count_total",
    "count_finite",
    "count_zero",
    "count_negative",
    "count_nan",
    "count_posinf",
    "count_neginf",
)


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON to ``path`` atomically (temp + fsync + rename)."""

    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, indent=1, sort_keys=True)
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _sha256_file(path: Path) -> str:
    """Return the hex sha256 of one file's bytes."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CommittedBlock:
    """One committed StepBlock plus its observations (the ring/merge unit).

    ``step_lo``/``step_hi`` carry the coarsening span: an uncoarsened block
    has ``step_lo == step_hi == block.global_step``; a coarsened record is
    an exact pairwise count-preserving merge labelled with its span (D19).
    """

    block: StepBlockRecord
    observations: tuple[ObservationRecord, ...]
    step_lo: int
    step_hi: int
    coarsened: bool = False


def coarsen_pair(a: CommittedBlock, b: CommittedBlock) -> CommittedBlock:
    """Merge two adjacent committed blocks exactly (count-preserving).

    Observations sharing (site, stream, phase) merge on their integer counts
    exactly; the merged record carries the union [step_lo, step_hi] span and
    is stamped ``coarsened=True``. The step axis coordinate of the merged
    block is the span's low edge -- consumers must read the span, and the
    renderers draw variable-width columns where coarsening happened (D24).
    """

    lo, hi = min(a.step_lo, b.step_lo), max(a.step_hi, b.step_hi)
    by_key: OrderedDict[tuple[str, str, str], ObservationRecord] = OrderedDict()
    for obs in (*a.observations, *b.observations):
        key = (obs.site_id, obs.stream, obs.phase)
        if key in by_key:
            prior = by_key[key]
            rebased = ObservationRecord(
                global_step=prior.global_step,
                site_id=obs.site_id,
                stream=obs.stream,
                phase=obs.phase,
                presence=obs.presence,
                spine=obs.spine,
                sketch=obs.sketch,
                grad_scale=obs.grad_scale,
                estimated=obs.estimated,
                sample_size=obs.sample_size,
                reason=obs.reason,
            )
            by_key[key] = merge_observations(prior, rebased)
        else:
            by_key[key] = obs
    return CommittedBlock(
        block=a.block if a.step_lo <= b.step_lo else b.block,
        observations=tuple(by_key.values()),
        step_lo=lo,
        step_hi=hi,
        coarsened=True,
    )


class RamRing:
    """Bounded in-RAM cache of committed blocks with explicit policies."""

    def __init__(
        self,
        capacity: int = DEFAULT_RING_CAPACITY,
        policy: str = "refuse",
        *,
        disk_backed: bool = False,
    ) -> None:
        if policy not in RAM_POLICIES:
            raise HistoryArtifactError(
                f"RAM-only policy {policy!r} is not one of {RAM_POLICIES}.",
                code="history_vocab_invalid",
                field="policy",
                value=policy,
                remedy=f"Use one of {RAM_POLICIES}.",
            )
        if capacity < 2:
            raise HistoryArtifactError(
                f"Ring capacity {capacity} is too small; coarsening and "
                "drop-oldest both need at least 2 committed blocks.",
                code="history_vocab_invalid",
                field="capacity",
                value=capacity,
                remedy="Use capacity >= 2.",
            )
        self.capacity = capacity
        self.policy = policy
        self.disk_backed = disk_backed
        self.dropped_blocks = 0
        self.coarsen_merges = 0
        self._blocks: list[CommittedBlock] = []

    def __len__(self) -> int:
        return len(self._blocks)

    @property
    def blocks(self) -> tuple[CommittedBlock, ...]:
        """The committed blocks currently resident, oldest first."""

        return tuple(self._blocks)

    def will_admit(self) -> None:
        """Refuse BEFORE the next scheduled block when full under ``refuse``.

        Called by the collector before scheduling a block so the refusal
        lands at the boundary, not after work was done (D20).
        """

        if self.disk_backed or self.policy != "refuse" or len(self._blocks) < self.capacity:
            return
        raise HistoryArtifactError(
            f"The RAM-only history ring is full ({self.capacity} committed "
            "StepBlocks) under the default 'refuse' policy. Nothing was "
            "dropped or degraded (D20: no automatic tier shedding, anywhere).",
            code="history_ram_budget_exceeded",
            capacity=self.capacity,
            remedy=(
                "Attach a disk archive (output_dir=...), raise ring_capacity, "
                "narrow the observed sites, raise the cadence, or opt into "
                "ram_policy='drop_oldest' or 'coarsen' explicitly."
            ),
        )

    def admit(self, block: CommittedBlock) -> None:
        """Admit one committed block, applying the explicit policy."""

        if len(self._blocks) >= self.capacity:
            if self.disk_backed or self.policy == "drop_oldest":
                # Disk-backed eviction is plain LRU and NOT an event; RAM-only
                # drop_oldest counts every dropped block.
                self._blocks.pop(0)
                if not self.disk_backed:
                    self.dropped_blocks += 1
            elif self.policy == "coarsen":
                merged = coarsen_pair(self._blocks[0], self._blocks[1])
                self._blocks[0:2] = [merged]
                self.coarsen_merges += 1
            else:
                self.will_admit()
        self._blocks.append(block)


def _spine_to_row(spine: SpineResult | None) -> dict[str, Any]:
    """Flatten one spine (or absence) into nullable column cells."""

    if spine is None:
        return dict.fromkeys((*_SPINE_COUNT_FIELDS, *_SPINE_FLOAT_FIELDS)) | {
            "reduction_dtype": None
        }
    row = {name: getattr(spine, name) for name in (*_SPINE_COUNT_FIELDS, *_SPINE_FLOAT_FIELDS)}
    row["reduction_dtype"] = spine.reduction_dtype
    return row


def _spine_from_row(row: dict[str, Any]) -> SpineResult | None:
    """Rebuild one spine from column cells (None cells = absent spine)."""

    if row["count_total"] is None:
        return None
    return SpineResult(
        count_total=int(row["count_total"]),
        count_finite=int(row["count_finite"]),
        count_zero=int(row["count_zero"]),
        count_negative=int(row["count_negative"]),
        count_nan=int(row["count_nan"]),
        count_posinf=int(row["count_posinf"]),
        count_neginf=int(row["count_neginf"]),
        finite_min=row["finite_min"],
        finite_max=row["finite_max"],
        finite_absmax=row["finite_absmax"],
        sum=row["sum"],
        sum_squares=row["sum_squares"],
        sum_abs=row["sum_abs"],
        mean=row["mean"],
        m2=row["m2"],
        reduction_dtype=str(row["reduction_dtype"]),
    )


class HistoryWriter:
    """Append-only artifact writer; every commit is atomic."""

    def __init__(self, output_dir: str | os.PathLike[str], run: RunRecord) -> None:
        self.path = Path(output_dir)
        self.run = run
        self.path.mkdir(parents=True, exist_ok=True)
        manifest_path = self.path / "manifest.json"
        if manifest_path.exists():
            raise HistoryArtifactError(
                f"{self.path} already contains a history manifest. Appending a "
                "different run onto an existing artifact would forge its "
                "identity; resume declares a new segment instead.",
                code="history_artifact_invalid",
                path=str(self.path),
                remedy=(
                    "Write to a fresh directory, or open the existing artifact "
                    "with HistoryReader and continue via a new segment "
                    "directory."
                ),
            )
        run_payload = asdict(run)
        run_payload["descriptor"] = asdict(run.descriptor)
        run_payload["package_versions"] = dict(run.package_versions)
        run_payload["reduction_dtypes"] = dict(run.reduction_dtypes)
        run_payload["cadences"] = dict(run.cadences)
        run_payload["budgets"] = dict(run.budgets)
        _atomic_write_json(
            manifest_path,
            {"format": "torchlens.history.v1", "run": run_payload},
        )
        self._sites: dict[str, SiteRecord] = {}
        self._chunk_rows: list[dict[str, Any]] = []
        self._n_committed = 0
        self._write_catalog()
        _atomic_write_json(self.path / "index.json", {"chunks": []})

    def _write_catalog(self) -> None:
        """Rewrite the site catalog atomically."""

        _atomic_write_json(
            self.path / "catalog.json",
            {"sites": [asdict(site) for site in self._sites.values()]},
        )

    def add_site(self, site: SiteRecord) -> None:
        """Register (or update, e.g. drift-ended) one site in the catalog."""

        self._sites[site.site_id] = site
        self._write_catalog()

    def append_block(self, committed: CommittedBlock) -> None:
        """Write one committed block as one chunk; atomic commit.

        The chunk hits the index only after its bytes are fsynced and
        renamed into place, so a crash between the two leaves the artifact
        readable at the previous chunk.
        """

        for obs in committed.observations:
            if obs.site_id not in self._sites:
                raise HistoryArtifactError(
                    f"Observation names site {obs.site_id!r} which is not in "
                    "the catalog; a chunk keyed to an uncataloged site is "
                    "unreadable without the model.",
                    code="history_artifact_invalid",
                    site_id=obs.site_id,
                    remedy="Call add_site(...) before appending observations for it.",
                )
        ordinal = self._n_committed
        chunk_name = f"chunk-{ordinal:06d}.npz"
        chunk_path = self.path / chunk_name
        tmp_path = self.path / (chunk_name + ".tmp")
        columns = self._columns_for(committed)
        buffer = _io_module.BytesIO()
        np.savez(buffer, **columns)  # type: ignore[arg-type]
        with open(tmp_path, "wb") as handle:
            handle.write(buffer.getvalue())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, chunk_path)
        digest = _sha256_file(chunk_path)
        index_path = self.path / "index.json"
        index = read_bounded(index_path)
        index["chunks"].append(
            {
                "file": chunk_name,
                "sha256": digest,
                "n_observations": len(committed.observations),
                "step_lo": committed.step_lo,
                "step_hi": committed.step_hi,
                "coarsened": committed.coarsened,
                "block": asdict(committed.block),
            }
        )
        _atomic_write_json(index_path, index)
        self._n_committed += 1

    def _columns_for(self, committed: CommittedBlock) -> dict[str, np.ndarray]:
        """Build the NumPy column arrays for one committed block."""

        observations = committed.observations
        n = len(observations)
        columns: dict[str, np.ndarray] = {
            "global_step": np.array([obs.global_step for obs in observations], dtype=np.int64),
            "site_id": np.array([obs.site_id for obs in observations], dtype=np.str_),
            "stream": np.array([obs.stream for obs in observations], dtype=np.str_),
            "phase": np.array([obs.phase for obs in observations], dtype=np.str_),
            "presence": np.array([obs.presence for obs in observations], dtype=np.str_),
            "grad_scale": np.array([obs.grad_scale or "" for obs in observations], dtype=np.str_),
            "estimated": np.array([obs.estimated for obs in observations], dtype=np.bool_),
            "sample_size": np.array(
                [-1 if obs.sample_size is None else obs.sample_size for obs in observations],
                dtype=np.int64,
            ),
            "reason": np.array([obs.reason or "" for obs in observations], dtype=np.str_),
        }
        spine_rows = [_spine_to_row(obs.spine) for obs in observations]
        for name in _SPINE_COUNT_FIELDS:
            columns[f"spine_{name}"] = np.array(
                [-1 if row[name] is None else row[name] for row in spine_rows], dtype=np.int64
            )
        for name in _SPINE_FLOAT_FIELDS:
            columns[f"spine_{name}"] = np.array(
                [np.nan if row[name] is None else row[name] for row in spine_rows],
                dtype=np.float64,
            )
        columns["spine_reduction_dtype"] = np.array(
            [row["reduction_dtype"] or "" for row in spine_rows], dtype=np.str_
        )
        # Sketch counts ride one dense int64 matrix; rows without a sketch
        # are all -1 (missing is never zero).
        d = self.run.descriptor
        cells = 2 * d.bins_per_side + 8
        sketch_matrix = np.full((n, cells), -1, dtype=np.int64)
        for row_index, obs in enumerate(observations):
            if obs.sketch is None:
                continue
            specials = obs.sketch.specials
            sketch_matrix[row_index] = np.concatenate(
                [
                    np.asarray(obs.sketch.neg_counts, dtype=np.int64),
                    np.asarray(obs.sketch.pos_counts, dtype=np.int64),
                    np.asarray(
                        [
                            specials.get(key, 0)
                            for key in (
                                "zero",
                                "pos_underflow",
                                "neg_underflow",
                                "pos_overflow",
                                "neg_overflow",
                                "nan",
                                "posinf",
                                "neginf",
                            )
                        ],
                        dtype=np.int64,
                    ),
                ]
            )
        columns["sketch_counts"] = sketch_matrix
        return columns


class HistoryReader:
    """Read one artifact without loading the run or importing a model."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        manifest_path = self.path / "manifest.json"
        if not manifest_path.exists():
            raise HistoryArtifactError(
                f"{self.path} has no manifest.json; not a history artifact.",
                code="history_artifact_invalid",
                path=str(self.path),
                remedy="Point at a directory written by HistoryWriter.",
            )
        manifest = read_bounded(manifest_path)
        if manifest.get("format") != "torchlens.history.v1":
            raise HistoryArtifactError(
                f"Unknown history format {manifest.get('format')!r}.",
                code="history_artifact_invalid",
                path=str(self.path),
                remedy="Use a torchlens.history.v1 artifact.",
            )
        run_payload = dict(manifest["run"])
        run_payload["descriptor"] = HistogramDescriptor(**run_payload["descriptor"])
        self.run = RunRecord(**run_payload)
        if self.run.schema_version > HISTORY_SCHEMA_VERSION:
            raise HistoryArtifactError(
                f"Artifact schema_version {self.run.schema_version} is newer "
                f"than this reader ({HISTORY_SCHEMA_VERSION}).",
                code="history_artifact_invalid",
                path=str(self.path),
                remedy="Upgrade torchlens to read this artifact.",
            )
        catalog = read_bounded(self.path / "catalog.json")
        sites = []
        for row in catalog["sites"]:
            row = dict(row)
            if row.get("shape") is not None:
                row["shape"] = tuple(row["shape"])
            sites.append(SiteRecord(**row))
        self.sites: dict[str, SiteRecord] = {site.site_id: site for site in sites}
        self._index = read_bounded(self.path / "index.json")["chunks"]

    @property
    def n_chunks(self) -> int:
        """Number of committed chunks."""

        return len(self._index)

    def step_blocks(self) -> tuple[StepBlockRecord, ...]:
        """Return the committed StepBlock records, oldest first."""

        return tuple(StepBlockRecord(**entry["block"]) for entry in self._index)

    def _verified_chunk_path(self, entry: dict[str, Any]) -> Path:
        """Return one committed chunk's path after existence + sha256 checks."""

        chunk_path = self.path / entry["file"]
        if not chunk_path.exists():
            raise HistoryArtifactError(
                f"Committed chunk {entry['file']} is missing from "
                f"{self.path}; the index says it was committed, so this is "
                "corruption, not an interrupted write.",
                code="history_artifact_corrupt",
                chunk=entry["file"],
                remedy="Restore the artifact from backup; committed chunks are never optional.",
            )
        digest = _sha256_file(chunk_path)
        if digest != entry["sha256"]:
            raise HistoryArtifactError(
                f"Chunk {entry['file']} fails its recorded sha256; refusing "
                "to serve silently corrupted history.",
                code="history_artifact_corrupt",
                chunk=entry["file"],
                expected=entry["sha256"],
                actual=digest,
                remedy="Restore the chunk from backup or drop the artifact.",
            )
        return chunk_path

    @staticmethod
    def _row_matches(
        data: Any, i: int, *, step: int | None, site_id: str | None, stream: str | None
    ) -> bool:
        """Row-level filter for :meth:`observations`."""

        if site_id is not None and str(data["site_id"][i]) != site_id:
            return False
        if stream is not None and str(data["stream"][i]) != stream:
            return False
        return step is None or int(data["global_step"][i]) == step

    @staticmethod
    def _decode_spine(data: Any, i: int) -> SpineResult | None:
        """Decode row ``i``'s persisted spine columns (absent spine -> None)."""

        spine_row: dict[str, Any] = {}
        for name in _SPINE_COUNT_FIELDS:
            count_cell = int(data[f"spine_{name}"][i])
            spine_row[name] = None if count_cell < 0 else count_cell
        for name in _SPINE_FLOAT_FIELDS:
            float_cell = float(data[f"spine_{name}"][i])
            spine_row[name] = None if np.isnan(float_cell) else float_cell
        dtype_cell = str(data["spine_reduction_dtype"][i])
        spine_row["reduction_dtype"] = dtype_cell or None
        if spine_row["count_total"] is None:
            return None
        # Restore payload-carried NaN extrema honestly: an observed spine
        # keeps float cells; absent spine is keyed on count_total.
        return _spine_from_row(spine_row)

    def _decode_sketch(self, data: Any, i: int) -> HistogramResult | None:
        """Decode row ``i``'s persisted sketch columns (absent sketch -> None)."""

        sketch_cells = data["sketch_counts"][i]
        if int(sketch_cells[0]) < 0:
            return None
        d = self.run.descriptor
        n_side = d.bins_per_side
        neg = tuple(int(x) for x in sketch_cells[:n_side])
        pos = tuple(int(x) for x in sketch_cells[n_side : 2 * n_side])
        special_values = sketch_cells[2 * n_side :]
        specials = dict(
            zip(
                (
                    "zero",
                    "pos_underflow",
                    "neg_underflow",
                    "pos_overflow",
                    "neg_overflow",
                    "nan",
                    "posinf",
                    "neginf",
                ),
                (int(x) for x in special_values),
                strict=False,
            )
        )
        return HistogramResult(descriptor=d, pos_counts=pos, neg_counts=neg, specials=specials)

    def _decode_row(self, data: Any, i: int) -> ObservationRecord:
        """Decode one persisted chunk row back into an ObservationRecord."""

        sample_size = int(data["sample_size"][i])
        return ObservationRecord(
            global_step=int(data["global_step"][i]),
            site_id=str(data["site_id"][i]),
            stream=str(data["stream"][i]),
            phase=str(data["phase"][i]),
            presence=str(data["presence"][i]),
            spine=self._decode_spine(data, i),
            sketch=self._decode_sketch(data, i),
            grad_scale=str(data["grad_scale"][i]) or None,
            estimated=bool(data["estimated"][i]),
            sample_size=None if sample_size < 0 else sample_size,
            reason=str(data["reason"][i]) or None,
        )

    def observations(
        self,
        *,
        step: int | None = None,
        site_id: str | None = None,
        stream: str | None = None,
    ) -> list[ObservationRecord]:
        """Materialize observations, optionally filtered; verifies checksums.

        The artifact is indexed by step AND by site/stream (D19): chunk rows
        carry [step_lo, step_hi], so a step filter skips chunks without
        opening them.
        """

        results: list[ObservationRecord] = []
        for entry in self._index:
            if step is not None and not (entry["step_lo"] <= step <= entry["step_hi"]):
                continue
            chunk_path = self._verified_chunk_path(entry)
            with np.load(chunk_path, allow_pickle=False) as data:
                for i in range(int(data["global_step"].shape[0])):
                    if self._row_matches(data, i, step=step, site_id=site_id, stream=stream):
                        results.append(self._decode_row(data, i))
        return results


__all__ = [
    "DEFAULT_RING_CAPACITY",
    "RAM_POLICIES",
    "CommittedBlock",
    "HistoryReader",
    "HistoryWriter",
    "RamRing",
    "coarsen_pair",
]
