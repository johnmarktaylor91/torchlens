"""The ``ReadTable`` result carrier (M(reads) item 1, decision D8).

Immutable, trace-bound Mapping keyed ``(target_id, kind, address)`` with
honesty as columns: the closed status vocabulary, retention/capture status,
resolution grain, and frozen requested/resolved/reached ride EVERY row, and
table-level provenance carries population counts, excluded-status counts,
the batching plan, and digests -- never only in ``DataFrame.attrs``.

Scalar tables persist portably (``read_table_v1`` JSON, standalone artifact,
loaded through the bounded reader); tensor-bearing tables refuse
(``read_table_not_portable``) until sidecars exist. A loaded table is
``rescorable=False`` because the runtime autograd registry is not serialized.
Dense tensors live on the carrier, not in scalar pandas rows.

The EDGE kind is admitted with kind-specific addresses from day one (deferred
EAP plumbing, section 8) even though v1 only produces ACT rows; ``sample_id``
is nullable from day one (dataset-mean seam).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from ..._io import _json as _bounded_json
from ._errors import ReadError

__all__ = ["ReadRow", "ReadTable", "TableProvenance", "load_read_table"]

READ_TABLE_SCHEMA_VERSION = "read_table_v1"

# Closed row-status vocabulary (D8). A numeric zero is `ok`; autograd None is
# `unreachable`; fabricating zeros for unreachable rows is the defect class
# this carrier exists to prevent.
ROW_STATUSES: tuple[str, ...] = (
    "ok",
    "unreachable",
    "not_differentiable",
    "unsupported_grain",
    "unavailable",
)

ROW_KINDS: tuple[str, ...] = ("ACT", "EDGE")

RESOLUTIONS: tuple[str, ...] = ("exact", "fused_parent", "derived")

GRAINS: tuple[str, ...] = ("site", "element")


@dataclass(frozen=True)
class ReadRow:
    """One read result row; immutable; honesty fields are not optional.

    Attributes
    ----------
    target_id:
        Stable id of the target this row answers, or ``None`` for
        target-free rows (``method='activation'``).
    kind:
        ``'ACT'`` (session address ``(layer_label, pass_index)``) or
        ``'EDGE'`` (kind-specific address; admitted for the deferred EAP
        stage, never nullable-stuffed into ACT rows).
    address:
        Kind-specific session address tuple.
    site_key:
        Portable L1 structural site key as a SEPARATE field (never overloaded
        onto the session address), or ``None`` on keyless records.
    alias_group:
        Ordinal of the ``(node, slot)`` alias group this row's site shares a
        gradient with, or ``None`` when unshared. Rows are per requested
        address; dedup is for the engine (D11).
    method:
        Signed method name (``activation_x_grad`` / ``grad`` /
        ``activation``).
    reduction:
        Named scalar reduction that produced ``score``, or ``None`` at
        element grain.
    grain:
        ``'site'`` (named scalar reduction) or ``'element'``
        (``reduce=None``; dense value on the carrier).
    score:
        Scalar score at site grain; ``None`` at element grain or on non-ok
        rows.
    value:
        Detached dense tensor at element grain; ``None`` otherwise.
    shape, dtype, device:
        Recorded geometry of the site's output (metadata, present even for
        unsaved sites).
    status:
        Closed vocabulary member of ``ROW_STATUSES``.
    status_reason:
        Closed reason detail (for example ``not_upstream_of_target``), or
        ``None`` for ``ok``.
    differentiable:
        Whether the site's output participates in autograd, when known.
    retention:
        ``'saved'`` / ``'unsaved'`` payload retention at the site.
    capture_status:
        The owning capture's settled outcome status string.
    resolution:
        ``'exact'`` | ``'fused_parent'`` | ``'derived'`` (fused computation
        is a hard upper bound on grain; a parent-grain answer is disclosed,
        never copied onto synthetic children).
    frozen_requested / frozen_resolved / frozen_reached:
        Per-row linearization honesty: whether freezing was requested at this
        site, resolved onto it, and actually fired in the target's cone
        (``frozen_reached`` is cone-dependent and may be ``None`` when no
        backward touched the site).
    policy_digest:
        Digest of the resolved frozen policy this row was measured under.
    detached:
        ``True`` when ``value`` is a detached copy (always true in v1 for
        carried tensors).
    sample_id:
        Nullable sample identity (dataset-mean seam; always ``None`` in v1).
    rescorable:
        ``False`` on loaded tables (the runtime registry is not serialized).
    """

    target_id: str | None
    kind: str
    address: tuple[Any, ...]
    site_key: str | None
    alias_group: int | None
    method: str
    reduction: str | None
    grain: str
    score: float | None
    value: Any
    shape: tuple[int, ...] | None
    dtype: str | None
    device: str | None
    status: str
    status_reason: str | None
    differentiable: bool | None
    retention: str | None
    capture_status: str | None
    resolution: str
    frozen_requested: bool
    frozen_resolved: bool
    frozen_reached: bool | None
    policy_digest: str | None
    detached: bool
    sample_id: str | None
    rescorable: bool

    def __post_init__(self) -> None:
        """Validate closed vocabularies at construction."""

        if self.kind not in ROW_KINDS:
            raise ReadError(
                f"ReadRow kind {self.kind!r} is not in the closed kind "
                f"vocabulary {ROW_KINDS}. Remedy: construct rows through the "
                "read engine",
                code="read_row_vocabulary_invalid",
                field_name="kind",
                value=self.kind,
            )
        if self.status not in ROW_STATUSES:
            raise ReadError(
                f"ReadRow status {self.status!r} is not in the closed status "
                f"vocabulary {ROW_STATUSES}. Remedy: construct rows through "
                "the read engine",
                code="read_row_vocabulary_invalid",
                field_name="status",
                value=self.status,
            )
        if self.resolution not in RESOLUTIONS:
            raise ReadError(
                f"ReadRow resolution {self.resolution!r} is not in the closed "
                f"vocabulary {RESOLUTIONS}. Remedy: construct rows through "
                "the read engine",
                code="read_row_vocabulary_invalid",
                field_name="resolution",
                value=self.resolution,
            )
        if self.grain not in GRAINS:
            raise ReadError(
                f"ReadRow grain {self.grain!r} is not in the closed grain "
                f"vocabulary {GRAINS}. Remedy: construct rows through the "
                "read engine",
                code="read_row_vocabulary_invalid",
                field_name="grain",
                value=self.grain,
            )

    @property
    def key(self) -> tuple[str | None, str, tuple[Any, ...]]:
        """Return the table key ``(target_id, kind, address)``."""

        return (self.target_id, self.kind, self.address)


@dataclass(frozen=True)
class TableProvenance:
    """Table-level provenance and disclosure (D8), never only DataFrame.attrs.

    Attributes
    ----------
    schema_version:
        ``read_table_v1``.
    trace_label:
        Human identity of the source trace (model name + capture ordinal).
    capture_outcome:
        The capture's settled outcome status string.
    target_reprs:
        Representation of every target, in preserved order.
    within_digest:
        Digest of the population request (``None`` for implicit).
    population_size:
        Number of addresses in the declared population.
    excluded_counts:
        status/reason -> count for rows EXCLUDED from an implicit population
        (the D10 ancestor-cone counts land here as
        ``not_upstream_of_target``).
    frozen_policy:
        Resolved frozen policy name (``'mlp_out'`` / ``'none'`` /
        ``'explicit'``).
    frozen_digest:
        Digest of the resolved frozen site set.
    frozen_requested_count / frozen_resolved_count / frozen_fired_count:
        Freeze coverage disclosure; fired is cone-dependent and asserted
        ``<=`` resolved, never an exact count (D6).
    alias_group_count / aliased_site_count:
        Disclosure of shared-gradient duplication (D11).
    batching_plan:
        The chosen plan: ``{'batch_size': B, 'reason': ..., 'device': ...}``.
    autograd_calls:
        Exact number of ``autograd.grad`` calls made (``ceil(T/B)``).
    timing_s:
        Wall-clock seconds inside the engine.
    result_bytes:
        Bytes retained on the carrier (dense values).
    sample_count:
        Number of samples folded into this table (dataset-mean seam; 1 in
        v1).
    warnings:
        Warning codes emitted during the read.
    rescorable:
        ``False`` on loaded tables.
    """

    schema_version: str = READ_TABLE_SCHEMA_VERSION
    trace_label: str | None = None
    capture_outcome: str | None = None
    target_reprs: tuple[str, ...] = ()
    within_digest: str | None = None
    population_size: int = 0
    excluded_counts: dict[str, int] = field(default_factory=dict)
    frozen_policy: str | None = None
    frozen_digest: str | None = None
    frozen_requested_count: int = 0
    frozen_resolved_count: int = 0
    frozen_fired_count: int = 0
    alias_group_count: int = 0
    aliased_site_count: int = 0
    batching_plan: dict[str, Any] = field(default_factory=dict)
    autograd_calls: int = 0
    timing_s: float | None = None
    result_bytes: int = 0
    sample_count: int = 1
    warnings: tuple[str, ...] = ()
    rescorable: bool = True


class ReadTable(Mapping):
    """Immutable trace-bound mapping of read rows (D8).

    Keys are ``(target_id, kind, address)``; values are :class:`ReadRow`.
    The table never mutates; every derivation (``for_target``,
    ``aggregate_targets``, alias collapse) returns a new table.
    """

    __slots__ = ("_rows", "_provenance", "_trace_ref", "_trace_token")

    _rows: dict[tuple[str | None, str, tuple[Any, ...]], ReadRow]
    _provenance: TableProvenance
    _trace_ref: Any
    _trace_token: tuple[Any, ...] | None

    def __init__(
        self,
        rows: Mapping[tuple[str | None, str, tuple[Any, ...]], ReadRow],
        provenance: TableProvenance,
        *,
        trace: Any = None,
        trace_token: tuple[Any, ...] | None = None,
    ) -> None:
        """Bind rows and provenance; hold the trace weakly.

        Parameters
        ----------
        rows:
            Ordered mapping from row key to row.
        provenance:
            Table-level provenance block.
        trace:
            Source trace (held weakly; ``None`` for loaded tables).
        trace_token:
            Validity token of the trace at read time (staleness checks).
        """

        import weakref

        object.__setattr__(self, "_rows", dict(rows))
        object.__setattr__(self, "_provenance", provenance)
        ref = None
        if trace is not None:
            try:
                ref = weakref.ref(trace)
            except TypeError:
                ref = None
        object.__setattr__(self, "_trace_ref", ref)
        object.__setattr__(self, "_trace_token", trace_token)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("ReadTable is immutable")

    # -- Mapping protocol -------------------------------------------------

    def __getitem__(self, key: Any) -> ReadRow:
        """Return the row for ``(target_id, kind, address)`` or a label.

        A bare pass-qualified label string is accepted as sugar for the
        single row whose address matches it across all targets; ambiguity
        (multiple targets) refuses with the exact keys.
        """

        if isinstance(key, tuple) and len(key) == 3:
            return self._rows[key]
        if isinstance(key, str):
            matches = [row for row in self._rows.values() if _address_label(row.address) == key]
            if len(matches) == 1:
                return matches[0]
            if not matches:
                raise KeyError(key)
            raise ReadError(
                f"Label {key!r} matches {len(matches)} rows across targets. "
                "Remedy: index with the full (target_id, kind, address) key "
                "or narrow with for_target(target_id) first",
                code="read_table_key_ambiguous",
                label=key,
                n_matches=len(matches),
            )
        raise KeyError(key)

    def __iter__(self) -> Iterator[tuple[str | None, str, tuple[Any, ...]]]:
        return iter(self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    # -- identity and provenance ------------------------------------------

    @property
    def provenance(self) -> TableProvenance:
        """Return the table-level provenance block."""

        return self._provenance

    @property
    def trace(self) -> Any:
        """Return the source trace when still alive, else ``None``."""

        ref = self._trace_ref
        return None if ref is None else ref()

    @property
    def trace_token(self) -> tuple[Any, ...] | None:
        """Return the trace validity token captured at read time."""

        return self._trace_token

    @property
    def rescorable(self) -> bool:
        """Whether the table can be re-scored against its live trace."""

        return self._provenance.rescorable and self.trace is not None

    # -- views -------------------------------------------------------------

    def rows(self) -> tuple[ReadRow, ...]:
        """Return every row in insertion order."""

        return tuple(self._rows.values())

    def target_ids(self) -> tuple[str, ...]:
        """Return the distinct non-None target ids in insertion order."""

        seen: dict[str, None] = {}
        for row in self._rows.values():
            if row.target_id is not None:
                seen.setdefault(row.target_id)
        return tuple(seen)

    def status_counts(self) -> dict[str, int]:
        """Return the status histogram over all rows."""

        counts: dict[str, int] = {}
        for row in self._rows.values():
            counts[row.status] = counts.get(row.status, 0) + 1
        return counts

    def for_target(self, target_id: str) -> ReadTable:
        """Return the single-target slice, provenance preserved.

        Parameters
        ----------
        target_id:
            One of :meth:`target_ids`.

        Raises
        ------
        ReadError
            Code ``read_table_target_unknown`` when no row carries the id.
        """

        rows = {key: row for key, row in self._rows.items() if row.target_id == target_id}
        if not rows:
            raise ReadError(
                f"No rows for target_id {target_id!r}; known targets: "
                f"{list(self.target_ids())!r}. Remedy: pick a target_id from "
                "target_ids()",
                code="read_table_target_unknown",
                target_id=target_id,
                known=list(self.target_ids()),
            )
        return ReadTable(
            rows,
            self._provenance,
            trace=self.trace,
            trace_token=self._trace_token,
        )

    def column(self, name: str) -> dict[tuple[str | None, str, tuple[Any, ...]], Any]:
        """Return one row field as ``key -> value``.

        Parameters
        ----------
        name:
            A :class:`ReadRow` field name.
        """

        if name not in ReadRow.__dataclass_fields__:
            raise ReadError(
                f"Unknown ReadRow column {name!r}. Remedy: pick one of "
                f"{sorted(ReadRow.__dataclass_fields__)}",
                code="read_table_column_unknown",
                column=name,
            )
        return {key: getattr(row, name) for key, row in self._rows.items()}

    def aggregate_targets(
        self, reducer: Callable[[tuple[float, ...]], float], *, name: str | None = None
    ) -> ReadTable:
        """Fold multi-target rows into one row per address, explicitly.

        No implicit mean/max/sum exists anywhere (D9): the caller supplies
        the reducer. Only site-grain ``ok`` rows fold; a non-ok row at any
        target keeps the address's worst status with its reason.

        Parameters
        ----------
        reducer:
            Callable folding the per-target score tuple to one float.
        name:
            Optional reducer name recorded in the aggregated rows'
            ``reduction`` column (defaults to the callable's ``__name__``).
        """

        reducer_name = name or getattr(reducer, "__name__", "reducer")
        by_address: dict[tuple[str, tuple[Any, ...]], list[ReadRow]] = {}
        for row in self._rows.values():
            by_address.setdefault((row.kind, row.address), []).append(row)
        out: dict[tuple[str | None, str, tuple[Any, ...]], ReadRow] = {}
        for (kind, address), group in by_address.items():
            non_ok = [row for row in group if row.status != "ok"]
            template = group[0]
            if non_ok:
                folded = replace(
                    template,
                    target_id=None,
                    reduction=f"{template.reduction or ''}|{reducer_name}".lstrip("|"),
                    score=None,
                    value=None,
                    status=non_ok[0].status,
                    status_reason=non_ok[0].status_reason,
                )
            else:
                if any(row.grain != "site" or row.score is None for row in group):
                    raise ReadError(
                        "aggregate_targets folds site-grain scalar scores; "
                        f"address {address!r} carries element-grain rows. "
                        "Remedy: re-read with a named scalar reduction "
                        "(reduce='sum' etc.) before aggregating targets",
                        code="read_table_grain_unaggregatable",
                        address=list(address),
                    )
                scores = tuple(row.score for row in group if row.score is not None)
                folded = replace(
                    template,
                    target_id=None,
                    reduction=f"{template.reduction or ''}|{reducer_name}".lstrip("|"),
                    score=float(reducer(scores)),
                    value=None,
                )
            out[(None, kind, address)] = folded
        return ReadTable(
            out,
            replace(
                self._provenance,
                target_reprs=self._provenance.target_reprs + (f"aggregate:{reducer_name}",),
            ),
            trace=self.trace,
            trace_token=self._trace_token,
        )

    def collapse_alias_groups(self) -> ReadTable:
        """Explicitly collapse alias groups to one row per ``(node, slot)``.

        The default table reports one row per requested address with the
        duplication disclosed (D11); this is the explicit opt-in collapse.
        The first member address of each group survives; scores must agree
        exactly within a group (they share one gradient by construction), so
        a conflict is an engine defect and raises.
        """

        out: dict[tuple[str | None, str, tuple[Any, ...]], ReadRow] = {}
        seen_groups: dict[tuple[str | None, int], tuple[Any, ...]] = {}
        for key, row in self._rows.items():
            if row.alias_group is None:
                out[key] = row
                continue
            group_key = (row.target_id, row.alias_group)
            first_address = seen_groups.get(group_key)
            if first_address is None:
                seen_groups[group_key] = row.address
                out[key] = row
                continue
            kept = out[(row.target_id, row.kind, first_address)]
            if (
                kept.status == "ok"
                and row.status == "ok"
                and kept.score is not None
                and row.score is not None
                and not math.isclose(kept.score, row.score, rel_tol=0.0, abs_tol=0.0)
            ):
                raise ReadError(
                    "Alias-group score conflict: members of one (node, slot) "
                    "group must share one gradient factor. This is an engine "
                    "defect, not a user error. Remedy: report this as a bug",
                    code="read_alias_conflict",
                    group=row.alias_group,
                    kept=list(kept.address),
                    dropped=list(row.address),
                )
        return ReadTable(
            out,
            self._provenance,
            trace=self.trace,
            trace_token=self._trace_token,
        )

    # -- adapters ----------------------------------------------------------

    def to_pandas(self, values: str = "omit") -> Any:
        """Return a pandas DataFrame view of the table.

        Parameters
        ----------
        values:
            ``'omit'`` drops dense values (scalar rows only; the default),
            ``'object'`` carries detached tensors as object cells,
            ``'explode'`` emits one row per element with a flat index column.

        Notes
        -----
        Dense tensors live on the carrier; the DataFrame is a projection.
        Provenance is duplicated into ``DataFrame.attrs`` but the carrier
        copy is authoritative (D8).
        """

        if values not in ("omit", "object", "explode"):
            raise ReadError(
                f"to_pandas values mode {values!r} is not one of "
                "('omit', 'object', 'explode'). Remedy: pick a supported mode",
                code="read_table_values_mode_invalid",
                mode=values,
            )
        import pandas as pd

        records: list[dict[str, Any]] = []
        for row in self._rows.values():
            base = asdict(row)
            base["address"] = _address_label(row.address)
            if values == "omit":
                base.pop("value", None)
                records.append(base)
            elif values == "object":
                records.append(base)
            else:
                value = base.pop("value", None)
                if value is None:
                    records.append(base)
                else:
                    flat = value.reshape(-1)
                    for flat_index, element in enumerate(flat.tolist()):
                        element_row = dict(base)
                        element_row["flat_index"] = flat_index
                        element_row["element_value"] = element
                        records.append(element_row)
        frame = pd.DataFrame.from_records(records)
        frame.attrs["read_table_provenance"] = asdict(self._provenance)
        return frame

    # -- persistence (scalar-only v1) ---------------------------------------

    def save(self, path: str | Path) -> Path:
        """Persist a scalar table as a standalone ``read_table_v1`` JSON file.

        Tensor-bearing (element-grain) tables refuse until sidecars exist
        (D8). The saved artifact is session-independent: no trace payloads,
        no runtime handles; a loaded table is ``rescorable=False``.

        Parameters
        ----------
        path:
            Destination file path (created or overwritten).

        Raises
        ------
        ReadError
            Code ``read_table_not_portable`` when any row carries a dense
            value.
        """

        dense = [row for row in self._rows.values() if row.value is not None]
        if dense:
            raise ReadError(
                f"{len(dense)} rows carry dense element-grain values; the v1 "
                "artifact is scalar-only. Remedy: save a scalar table "
                "(reduce='sum' etc.) or keep element-grain tables in session",
                code="read_table_not_portable",
                dense_rows=len(dense),
            )
        payload = {
            "schema_version": READ_TABLE_SCHEMA_VERSION,
            "provenance": asdict(replace(self._provenance, rescorable=False)),
            "rows": [
                {**asdict(row), "address": list(row.address), "value": None}
                for row in self._rows.values()
            ],
        }
        destination = Path(path)
        destination.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
        return destination


def _address_label(address: tuple[Any, ...]) -> str:
    """Render an ACT address ``(layer_label, pass_index)`` as ``label:pass``."""

    if len(address) == 2 and isinstance(address[0], str):
        return f"{address[0]}:{address[1]}"
    return "/".join(str(part) for part in address)


def load_read_table(path: str | Path) -> ReadTable:
    """Load a persisted scalar ``read_table_v1`` artifact, fail-closed.

    Validation is load-time and typed: an unknown schema version, a row
    outside the closed vocabularies, or a malformed payload refuses rather
    than degrading. The loaded table is trace-unbound and
    ``rescorable=False``.

    Parameters
    ----------
    path:
        Path written by :meth:`ReadTable.save`.

    Raises
    ------
    ReadError
        Code ``read_table_artifact_invalid`` on any structural violation.
    """

    source = Path(path)
    try:
        payload = _bounded_json.loads_bounded(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReadError(
            f"Read-table artifact at {source} is unreadable or over the "
            "bounded-reader ceilings. Remedy: re-save the table with "
            "ReadTable.save",
            code="read_table_artifact_invalid",
            path=str(source),
        ) from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != READ_TABLE_SCHEMA_VERSION:
        raise ReadError(
            "Read-table artifact does not declare schema_version "
            f"{READ_TABLE_SCHEMA_VERSION!r}. Remedy: re-save the table with "
            "this TorchLens version",
            code="read_table_artifact_invalid",
            path=str(source),
            found=payload.get("schema_version") if isinstance(payload, dict) else None,
        )
    provenance_raw = payload.get("provenance")
    rows_raw = payload.get("rows")
    if not isinstance(provenance_raw, dict) or not isinstance(rows_raw, list):
        raise ReadError(
            "Read-table artifact is missing its provenance block or rows "
            "list. Remedy: re-save the table with ReadTable.save",
            code="read_table_artifact_invalid",
            path=str(source),
        )
    known_provenance = {
        key: value
        for key, value in provenance_raw.items()
        if key in TableProvenance.__dataclass_fields__
    }
    known_provenance["rescorable"] = False
    known_provenance["warnings"] = tuple(known_provenance.get("warnings", ()))
    known_provenance["target_reprs"] = tuple(known_provenance.get("target_reprs", ()))
    provenance = TableProvenance(**known_provenance)
    rows: dict[tuple[str | None, str, tuple[Any, ...]], ReadRow] = {}
    for row_raw in rows_raw:
        if not isinstance(row_raw, dict):
            raise ReadError(
                "Read-table artifact carries a non-object row. Remedy: "
                "re-save the table with ReadTable.save",
                code="read_table_artifact_invalid",
                path=str(source),
            )
        filtered = {
            key: value for key, value in row_raw.items() if key in ReadRow.__dataclass_fields__
        }
        filtered["address"] = tuple(filtered.get("address", ()))
        shape = filtered.get("shape")
        filtered["shape"] = tuple(shape) if shape is not None else None
        filtered["value"] = None
        filtered["rescorable"] = False
        try:
            row = ReadRow(**filtered)
        except (TypeError, ReadError) as exc:
            raise ReadError(
                "Read-table artifact row violates the closed row schema. "
                "Remedy: re-save the table with this TorchLens version",
                code="read_table_artifact_invalid",
                path=str(source),
            ) from exc
        rows[row.key] = row
    return ReadTable(rows, provenance, trace=None, trace_token=None)
