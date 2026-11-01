"""Single Bundle type for intervention-ready TorchLens model logs."""

from __future__ import annotations

import functools
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import torch

from ..errors.episode import BundleRelationError
from ..intervention._super.super_logs import (
    SuperBufferAccessor,
    SuperGradFnAccessor,
    SuperGradFnCallAccessor,
    SuperModuleAccessor,
    SuperModuleCallAccessor,
    SuperParamAccessor,
)
from ..intervention._super.super_op import (
    SuperAtenOp,
    SuperLayerAccessor,
    SuperOp,
    SuperOpAccessor,
)
from ..intervention._topology.topology import Supergraph, build_supergraph
from ..intervention.errors import (
    BaselineUndeterminedError,
    BundleMemberError,
)
from ..intervention.resolver import _resolve_sites
from ..intervention.types import Relationship
from ._aligned_sites import aligned_site_labels, resolve_aligned_site
from ._alternatives import (
    AlternativeCall as AlternativeCall,  # re-export: torchlens.bundle.AlternativeCall
    AlternativeReport as AlternativeReport,  # re-export: torchlens.bundle.AlternativeReport
    _bundle_add_alternative,
)
from ._compare_gate import require_comparable
from ._comparisons import (
    _bundle_store_comparison,
    _bundle_stored_comparison,
    _bundle_stored_comparison_names,
)

# Re-exports (redundant aliases): _distance_value is imported from
# torchlens.bundle by the cert7/cert8 hardening suites; the rest are the
# historical module-level surface of the pre-split file.
from ._deltas import (
    _bundle_aligned_pairs as _bundle_aligned_pairs,
    _bundle_delta_map as _bundle_delta_map,
    _bundle_norm_delta as _bundle_norm_delta,
    _bundle_output_delta as _bundle_output_delta,
    _distance_value as _distance_value,
    _metric_label as _metric_label,
    _resolve_member_name as _resolve_member_name,
    _tensor_field as _tensor_field,
)
from ._draw import bundle_draw as _bundle_draw
from ._labels import label_view
from ._lineage import BundleOperation, MemberEffectTable, mint_bundle_id
from ._members import (
    coerce_member_name,
    coerce_member_name_list,
    coerce_optional_name_list,
    coerce_trace_list,
    derive_name,
    is_list_like,
    parse_members,
)
from ._outcome_fold import _fold_member_outcomes
from ._provenance import (
    WhyReport as WhyReport,  # re-export: torchlens.bundle.WhyReport
    bundle_provenance as _bundle_provenance,
    bundle_why as _bundle_why,
)
from ._relation_carriage import (
    _bundle_derive_episode_status,
    _bundle_downstream,
    _bundle_relate,
    _bundle_upstream,
    settle_relation_rows,
)
from ._relations import MemberRelationRow, MemberRelationTable, OpaqueRelationRow
from ._token_divergence import (
    TokenDivergence as TokenDivergence,  # re-export: torchlens.bundle.TokenDivergence
    _bundle_token_divergence,
)
from ._token_spans import _bundle_token_pairs
from ._vary import bundle_vary as _bundle_vary

if TYPE_CHECKING:
    from torch import nn

    from .._capture.outcome import CaptureOutcome
    from .._data_classes.op import Op
    from .._data_classes.trace import Trace
    from .._literals import VisDirectionLiteral
    from ._draw import BundleGraphLiteral
    from ._token_spans import TokenPairs


def _member_outcome_token(member: Any) -> str:
    """One member's settled outcome-status token, degrading to ``unknown``."""

    from .._utils.fail_open import fail_open

    outcome = fail_open(lambda: getattr(member, "outcome", None), lambda _error: None)
    return getattr(getattr(outcome, "status", None), "value", None) or "unknown"


class AmbiguousLabelError(KeyError):
    """Raised when ``Bundle.at`` finds a label in multiple accessors."""


class _BundleAccessorClassView:
    """Class-level placeholder for Bundle accessor properties."""


class _BundleAccessorProperty:
    """Property that exposes Bundle accessors without inflating method counts."""

    def __init__(self, accessor_cls: type[Any]) -> None:
        """Initialize the property.

        Parameters
        ----------
        accessor_cls:
            Accessor class to instantiate for Bundle instances.
        """

        self._accessor_cls = accessor_cls
        self._class_view = _BundleAccessorClassView()

    def __get__(
        self,
        instance: Bundle | None,
        owner: type[Bundle] | None = None,
    ) -> Any:
        """Return a class view or an accessor on instances.

        Parameters
        ----------
        instance:
            Bundle instance, or ``None`` for class access.
        owner:
            Owning Bundle class.

        Returns
        -------
        Any
            Class view for class access, otherwise the instantiated accessor.
        """

        if instance is None:
            return self._class_view
        return self._accessor_cls(instance)


class _BundleStructuralProperty:
    """Descriptor exposing Bundle predicates without counting as a method."""

    def __init__(self, getter: Callable[[Bundle], Any]) -> None:
        """Initialize the descriptor.

        Parameters
        ----------
        getter:
            Instance getter to call for Bundle instances.
        """

        self._getter = getter
        self.__doc__ = getter.__doc__

    def __get__(self, instance: Bundle | None, owner: type[Bundle]) -> Any:
        """Return the descriptor on classes or computed value on instances.

        Parameters
        ----------
        instance:
            Bundle instance, or ``None`` for class access.
        owner:
            Owning Bundle class.

        Returns
        -------
        Any
            Descriptor for class access, otherwise the computed property value.
        """

        if instance is None:
            return self
        return self._getter(instance)


#: Per-kind name readers behind ``bundle.<kind>.shared`` / ``bundle.<kind>.divergent``.
_KIND_NAME_GETTERS: dict[str, Callable[[Any], Any]] = {
    "ops": lambda trace: getattr(trace, "op_labels", ()),
    "layers": lambda trace: getattr(trace, "layer_labels", ()),
    "modules": lambda trace: trace.modules.keys(),
    "params": lambda trace: trace.params.keys(),
    "buffers": lambda trace: trace.buffers.keys(),
    "grad_fns": lambda trace: trace.autograd.grad_fns.keys(),
}


#: Public Bundle methods removed by the 2026-10-01 shim removal (no alias kept).


_REMOVED_BUNDLE_MEMBERS: dict[str, str] = {
    "replay": "use bundle.do(site, edit) -- do replays every member",
    "rerun": "use bundle.run(model, x) -- Bundle.rerun was renamed",
    "push": "use bundle.do(site, edit) -- do applies and replays the edit on every member",
    "attach_hooks": "use bundle.do(site, hook) -- do is the one editing verb on a bundle",
    "node": "use bundle.at(site)",
    "compare_at": "use bundle.at(site).compare()",
    "joint_metric": "call fn(bundle) directly",
    "set_capacity": "assign bundle.capacity = n",
    "traces": "use bundle.members or bundle[name]",
    "help": "print(bundle) shows the bundle card",
    "cluster": "there is no bundle clustering; compute it over bundle.map(fn)",
    "member_construction": "read bundle.history for how members entered the bundle",
    "measure_members": "use bundle.map(fn) -- released members read 'unmeasured'",
    "show_diff": "use bundle.draw(graph='diff', members=(a, b), metric=...)",
    "delta_map": "use bundle.compare(a, b, metric=..., on=...) per pair, bundle.rank_changes() across members",
    "norm_delta": "use bundle.compare(baseline, member, metric='relative_l2')",
    "output_delta": "use bundle.compare(member, target, metric=...) and read the output sites",
    "aligned_pairs": "use bundle.supergraph, the structural alignment across members",
}


class Bundle:
    """Flat container of Traces with relationship-gated operations.

    CROSS-MEMBER PARAMETER READS ARE CLAIM-GATED (A-CKPT; foldB D7): a
    parameter value/difference/trajectory read across two or more members
    asserts capture-time parameter values, but parameter members carry live
    model handles (or nothing after deserialization), never capture-time
    bytes. Absent immutable parameter evidence on every member (capture-time
    snapshots, R8(b), future), such reads refuse BEFORE tensor lookup with
    the stable code ``checkpoint_series_live_params`` -- keyed on the claim,
    never on Python object identity or the relationship lattice below, which
    never licenses a parameter value claim. A relation row asserting a
    version axis (``successor_of``/``forked_from``/``escalates``) still
    ORDERS members; only the weight claim is refused.

    Parameters
    ----------
    members:
        Mapping of member names to ``Trace`` objects, sequence of logs,
        or sequence of ``(name, log)`` tuples. Empty (the default) builds an
        empty bundle to fill later with :meth:`add`.
    names:
        Optional names for a sequence of logs.
    baseline:
        Optional baseline member name or ``Trace`` reference.
    member_relations:
        Optional S6 member-relation rows (``MemberRelationRow`` instances or
        payload mappings; the artifact loader may additionally pass
        ``OpaqueRelationRow`` instances for preserved unknown namespaced
        kinds). Rows are schema-validated and checked against the initial
        members (R1: no dangling edges). ``None`` means an empty table with
        unchanged plain-Bundle semantics.
    preserved_sections:
        Load-side carriage for unknown NAMESPACED top-level ``bundle.json``
        sections (loader doctrine leg (c), the C07X amendment): preserved
        verbatim, disclosed through :attr:`preserved_sections`, never
        executed, and re-emitted on save. User construction normally leaves
        this ``None``.
    bundle_id:
        Load-side carriage for the persisted container identity (F03
        lineage). ``None`` (the user door) mints a fresh random id; the
        artifact loader passes the saved id verbatim.
    """

    # Bundle outcome authority (foldB D9, instance three of the two-answers
    # disease): the container speaks for itself through the DERIVED
    # worst-of-members fold below, so ``outcome_for(bundle)`` never falls back
    # to a false UNKNOWN with the hand-built-object warning.
    _OUTCOME_SELF_AUTHORITY: ClassVar[bool] = True

    def __init__(
        self,
        members: Mapping[str, Trace] | Sequence[Trace] | Sequence[tuple[str, Trace]] = (),
        *,
        names: Sequence[str] | None = None,
        baseline: str | Trace | None = None,
        relations: (
            Sequence[MemberRelationRow | OpaqueRelationRow | Mapping[str, Any]] | None
        ) = None,
        preserved_sections: Mapping[str, Any] | None = None,
        bundle_id: str | None = None,
    ) -> None:
        """Initialize a flat bundle without eagerly building a supergraph."""

        parsed = parse_members(members, names=names)
        self._members: OrderedDict[str, Trace] = OrderedDict(parsed)
        self._supergraph: Supergraph | None = None
        self._capacity: int | None = None
        self._baseline_name: str | None = self._resolve_baseline_name(baseline)
        self._member_relations: MemberRelationTable = self._build_relation_table(
            (), relations or ()
        )
        self._preserved_sections: dict[str, Any] = dict(preserved_sections or {})
        # F03 lineage (ledger memo 0b): one random container identity, minted
        # here, restored verbatim by the artifact loader — NEVER a content
        # hash (the D1a id law). Anchors record how each member entered THIS
        # container; the operation list is the container-side chronology.
        self._bundle_id: str = bundle_id if bundle_id else mint_bundle_id()
        self._forked_from_bundle_id: str | None = None
        self._member_construction: dict[str, dict[str, Any]] = {
            name: {"origin": "constructed"} for name in self._members
        }
        self._operations: list[BundleOperation] = []
        self._effect_tables: dict[str, MemberEffectTable] = {}

    def __len__(self) -> int:
        """Return the number of bundle members.

        Returns
        -------
        int
            Number of member logs.
        """

        return len(self._members)

    def __iter__(self) -> Iterator[Trace]:
        """Iterate member logs in insertion order.

        Returns
        -------
        Iterator[Trace]
            Iterator over member logs.
        """

        return iter(self._members.values())

    def __contains__(self, name: str) -> bool:
        """Return whether a member name is present.

        Parameters
        ----------
        name:
            Member name.

        Returns
        -------
        bool
            Whether the bundle contains ``name``.
        """

        return name in self._members

    def _outcome_distribution(self) -> dict[str, int]:
        """Count member capture outcomes by settled status (F10)."""

        counts: dict[str, int] = {}
        for member in self._members.values():
            token = _member_outcome_token(member)
            counts[token] = counts.get(token, 0) + 1
        return counts

    def __repr__(self) -> str:
        """One-line bundle card: members, baseline, outcome distribution.

        Poison/divergence facts sit ABOVE the fold (lovely matrix): any
        non-complete member outcome prints in the distribution, never
        behind a member lookup. A missing baseline renders as unquoted
        None, never the string 'None' masquerading as a member name
        (lovely bug 17).
        """

        distribution = self._outcome_distribution()
        outcome_note = ""
        if any(token != "complete" for token in distribution):
            pairs = ", ".join(f"{token}={count}" for token, count in sorted(distribution.items()))
            outcome_note = f", outcomes=({pairs})"
        base = (
            f"Bundle(n_members={len(self)}, names={self.names!r}, "
            f"baseline={self._baseline_name!r}, "
            f"structurally_consistent={self.is_structurally_consistent}"
            f"{outcome_note})"
        )
        if not self._effect_tables:
            return base
        # D3c: silent truncation reading as "covered everything" is the
        # failure mode this sprint keeps finding — an engine-built bundle
        # states its candidate accounting and policy in its own repr.
        table = next(reversed(self._effect_tables.values()))
        counts: dict[str, int] = {}
        for row in table.rows:
            if row.candidate_id == "__baseline__":
                continue
            counts[row.status] = counts.get(row.status, 0) + 1
        attempted = sum(counts.values())
        summary = ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))
        return (
            f"{base[:-1]}, effects=[attempted={attempted}, {summary}, "
            f"retain={table.retain_policy}, lane={table.lane}])"
        )

    def __str__(self) -> str:
        """Bounded bundle view: repr line + per-member one-liners (F10)."""

        from ..stats._envelope import COLLECTION_MAX_CHILDREN

        lines = [self.__repr__()]
        names = list(self.names)
        for name in names[:COLLECTION_MAX_CHILDREN]:
            member = self._members[name]
            marker = " (baseline)" if name == self._baseline_name else ""
            lines.append(f"  {name}{marker}: {member!r}")
        if len(names) > COLLECTION_MAX_CHILDREN:
            lines.append(f"  ... {len(names) - COLLECTION_MAX_CHILDREN} more members")
        return "\n".join(lines)

    def __getitem__(self, name: str) -> Trace:
        """Return a member by name.

        Parameters
        ----------
        name:
            Member name.

        Returns
        -------
        Trace
            Matching member log.
        """

        if not isinstance(name, str):
            raise TypeError(f"Bundle indices must be member names, got {type(name).__name__}.")
        return self._members[name]

    @property
    def outcome(self) -> CaptureOutcome:
        """Return the derived worst-of-members capture-outcome fold.

        A Bundle has no settlement of its own -- members remain the
        settlement authority. This property is the declared container
        DERIVATION (foldB D9): a frozen ``CaptureOutcome`` with
        ``derived=True`` whose status is the most severe member status
        (COMPLETE never blessed above the weakest member; an unsettled
        member folds as UNKNOWN fail-closed) and whose ``settlement_note``
        names the driving member. It is a disclosure reported beside
        results, NEVER a per-member gate: capability gating applies at the
        member whose facts a read cites (D8).

        Returns
        -------
        CaptureOutcome
            Derived worst-of-members fold over the current members.
        """

        from .._capture.outcome import outcome_for

        return _fold_member_outcomes(
            [(name, outcome_for(member)) for name, member in self._members.items()]
        )

    def save(
        self,
        path: str | Path,
        *,
        level: str = "portable",
        overwrite: bool = False,
    ) -> None:
        """Save this bundle as a unified ``.tlspec`` directory.

        Parameters
        ----------
        path:
            Destination ``.tlspec`` directory path.
        level:
            Save level: ``"audit"``, ``"executable_with_callables"``, or
            ``"portable"``.
        overwrite:
            Whether an existing destination may be replaced.
        """

        from .._io.tlspec import _TlSpecWriter

        _TlSpecWriter.write_bundle(
            bundle=self,
            path=path,
            save_level=level,
            overwrite=overwrite,
        )

    def __getattr__(self, name: str) -> Any:
        """Refuse a removed Bundle member with the typed redirect naming its replacement.

        Raises
        ------
        AttributeError
            If ``name`` is not a Bundle attribute.
        """

        from .._utils.facade import refuse_removed_member

        refuse_removed_member("Bundle", name, _REMOVED_BUNDLE_MEMBERS)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    @property
    def names(self) -> list[str]:
        """Return member names in bundle order.

        Returns
        -------
        list[str]
            Member names.
        """

        return list(self._members)

    @property
    def members(self) -> dict[str, Trace]:
        """Return a shallow copy of the member mapping.

        Returns
        -------
        dict[str, Trace]
            Member mapping.
        """

        return dict(self._members)

    @property
    def ops(self) -> SuperOpAccessor:
        """Return cross-member access to pass-qualified Op labels.

        Returns
        -------
        SuperOpAccessor
            Bundle op accessor.
        """

        return SuperOpAccessor(self)

    @property
    def layers(self) -> SuperLayerAccessor:
        """Return cross-member access to aggregate Layer labels.

        Returns
        -------
        SuperLayerAccessor
            Bundle layer accessor.
        """

        return SuperLayerAccessor(self)

    modules = _BundleAccessorProperty(SuperModuleAccessor)
    buffers = _BundleAccessorProperty(SuperBufferAccessor)
    params = _BundleAccessorProperty(SuperParamAccessor)
    grad_fns = _BundleAccessorProperty(SuperGradFnAccessor)
    module_calls = _BundleAccessorProperty(SuperModuleCallAccessor)
    grad_fn_calls = _BundleAccessorProperty(SuperGradFnCallAccessor)

    @_BundleStructuralProperty
    def is_structurally_consistent(self) -> bool:
        """Return whether all members share the same graph-shape hash.

        Returns
        -------
        bool
            Whether every member has an equal ``graph_shape_hash`` value.
        """

        hashes = {getattr(member, "graph_shape_hash", None) for member in self._members.values()}
        return len(hashes) == 1

    def _kind_names(self, kind: str, *, shared: bool) -> list[str]:
        """Return the names of one kind present in every member, or in some but not all.

        Parameters
        ----------
        kind:
            Accessor kind: ``ops``, ``layers``, ``modules``, ``params``, ``buffers`` or
            ``grad_fns``.
        shared:
            ``True`` for names in every member (ordered by the first member), ``False``
            for names present in some but not all members.

        Returns
        -------
        list[str]
            The names. Read through ``bundle.<kind>.shared`` and ``bundle.<kind>.divergent``.
        """

        getter = _KIND_NAME_GETTERS[kind]
        return self._shared(getter) if shared else self._divergent(getter)

    @property
    def id(self) -> str:
        """This container's persisted random identity (F03 lineage, D1a law).

        Minted at construction (never a content hash), restored verbatim on
        load, and re-minted on :meth:`fork` — the forked container anchors
        its source id in its ``member_construction`` anchors and its
        ``fork`` operation row instead of sharing the identity.
        """

        return self._bundle_id

    @property
    def history(self) -> tuple[BundleOperation, ...]:
        """The container-side chronology: hash-chained BundleOperation rows.

        One row per top-level bundle action (fork / sweep / vary / ...),
        append-only, persisted in ``bundle.json``. DISCLOSURE, never
        settlement authority; the trace-side canonical audit remains the
        construction truth the provenance join walks.
        """

        return tuple(self._operations)

    def _record_bundle_operation(
        self,
        kind: str,
        *,
        member_names: Sequence[str] = (),
        params: Mapping[str, Any] | None = None,
    ) -> BundleOperation:
        """Append one chronology row to the container-side operation ledger.

        The ledger is append-only and hash-chained (seq + previous-row
        digest, the same spelling as the v9 EVENT audit rows), so fork
        lineage and semantic chronology can never grow as two orderings on
        one container (foldB F03 brief-delta).
        """

        previous = self._operations[-1] if self._operations else None
        row = BundleOperation(
            operation_id=mint_bundle_id(),
            seq=(previous.seq + 1) if previous is not None else 1,
            kind=kind,
            member_names=tuple(member_names),
            params=dict(params or {}),
            prev_operation_digest=(previous.operation_digest if previous is not None else None),
        )
        self._operations.append(row)
        return row

    @property
    def baseline_name(self) -> str | None:
        """Return the configured baseline member name, if any.

        Returns
        -------
        str | None
            Baseline member name.
        """

        return self._baseline_name

    @_BundleStructuralProperty
    def relations(self) -> tuple[MemberRelationRow | OpaqueRelationRow, ...]:
        """Return the immutable S6 member-relation view (R4).

        Returns
        -------
        tuple[MemberRelationRow | OpaqueRelationRow, ...]
            Identity-stable frozen-row tuple: repeated reads return THE SAME
            object until ``relate`` (or an R5 cascade) installs a new table
            version. In-place mutation is impossible. ``OpaqueRelationRow``
            entries are preserved unknown namespaced kinds (loader doctrine
            leg (a)): disclosed, never executed, re-saved verbatim.
        """

        return self._member_relations.rows

    # Real class methods (ledger memo item 0: reachable by hasattr/IDE
    # completion, no longer __getattr__-only dynamic lookups).
    relate = _bundle_relate
    episode_status = _bundle_derive_episode_status
    # The provenance join (F03 item 4): pure derived views, never stored.
    why = _bundle_why
    provenance = _bundle_provenance
    # One explicit edit or identity per member (F03 item 5); broadcast do()
    # is untouched forever.
    vary = _bundle_vary

    def upstream(self, name: str | Trace) -> tuple[str, ...]:
        """Return the members ``name`` transitively came after (what fed it), nearest first.

        Walks the ordering relation kinds (``fed_by``, ``selected_by``,
        ``successor_of``, ``forked_from``, ``escalates``) from the later
        member to the earlier ones.
        """

        return _bundle_upstream(self, name)

    def downstream(self, name: str | Trace) -> tuple[str, ...]:
        """Return the members that transitively came after ``name`` (what it fed), nearest first.

        Walks the same ordering relation kinds as :meth:`upstream` (``fed_by``,
        ``selected_by``, ``successor_of``, ``forked_from``, ``escalates``) from
        the earlier member to the later ones; the inverse of :meth:`upstream`.
        """

        return _bundle_downstream(self, name)

    def token_pairs(self, reader: str | Trace, source: str | Trace) -> TokenPairs:
        """Return the verified one-to-one token positions of ``reader`` fed by ``source``.

        Parameters
        ----------
        reader:
            The later member (name or Trace) whose prompt holds the fed text.
        source:
            The earlier member (name or Trace) that produced that text.
        """

        return _bundle_token_pairs(self, reader, source)

    def token_divergence(self, original: str | Trace, rerun: str | Trace) -> TokenDivergence:
        """Return where two members' prompts and completions part, by exact token ids.

        Parameters
        ----------
        original:
            The member (name or Trace) from the original run, the reference.
        rerun:
            The member (name or Trace) from the re-run compared against it,
            typically its copy from a steered second run (``"cf_critique"``).
        """

        return _bundle_token_divergence(self, original, rerun)

    def add_alternative(
        self,
        second_run: Bundle,
        *,
        steered: str | Sequence[str],
        prefix: str = "cf_",
    ) -> AlternativeReport:
        """Merge a re-run of the harness with some calls steered; report what changed.

        Parameters
        ----------
        second_run:
            The bundle the same harness built with the ``steered`` calls captured
            under their specs.
        steered:
            The steered call name, or a sequence of names.
        prefix:
            Prefix naming each second-run member's copy (default ``"cf_"``).
        """

        return _bundle_add_alternative(self, second_run, steered=steered, prefix=prefix)

    def effects(self, history_id: str | None = None) -> Any:
        """Serve the stored per-candidate effect table (F03 item 8).

        The table is DATA the engine wrote once (released/refused/failed
        candidates included), persisted in the artifact keyed by operation
        id; this read never recomputes it.
        """

        from ..experiment._engine import bundle_effects

        return bundle_effects(self, history_id)

    @property
    def preserved_sections(self) -> dict[str, Any]:
        """Unknown namespaced ``bundle.json`` sections preserved at load.

        Loader doctrine leg (c) (the C07X amendment): a well-formed unknown
        TOP-LEVEL ``bundle.json`` section under a NAMESPACED key
        (``"<ns>.<name>"``) loads opaque and disclosed here, is never
        executed, and re-emits verbatim on :meth:`save` — a bare-unknown
        section refuses at load instead (the historical silent
        load-then-destroy is banned in both directions).

        Returns
        -------
        dict[str, Any]
            Fresh copy of the preserved sections (empty for bundles built
            in-session).
        """

        # ``__dict__`` read: a legacy-pickled Bundle predating the slot
        # simply has no preserved sections (and the private-getattr census
        # stays clean).
        return dict(self.__dict__.get("_preserved_sections") or {})

    def _build_relation_table(
        self,
        existing_rows: Sequence[MemberRelationRow | OpaqueRelationRow],
        new_rows: Sequence[MemberRelationRow | OpaqueRelationRow | Mapping[str, Any]],
        *,
        assess_unclaimed: bool = False,
    ) -> MemberRelationTable:
        """Return a NEW validated relation table (existing + coerced new rows).

        New ``fed_by`` rows then settle their basis against the members
        (``settle_relation_rows``; ``assess_unclaimed`` is the positional
        ``relate`` door, which also binds and verifies unclaimed rows).

        ``OpaqueRelationRow`` instances pass through unchanged (loader-side
        carriage for preserved unknown namespaced kinds); mapping payloads
        always coerce through the CLOSED grammar — new rows of unknown kinds
        refuse at construction, opaque rows enter from artifacts only.

        Raises
        ------
        BundleRelationError
            ``bundle_relation_schema_invalid`` when a new row is off-schema
            (unknown kind, wrong shape for its kind, undeclared or missing
            param keys, ill-typed values);
            ``bundle_relation_evidence_over_budget`` rides through with its
            own code; R1/R3 refusals ride through from
            ``validate_against_members``.
        """

        rows: list[MemberRelationRow | OpaqueRelationRow] = list(existing_rows)
        for row in new_rows:
            if isinstance(row, (MemberRelationRow, OpaqueRelationRow)):
                rows.append(row)
                continue
            try:
                rows.append(MemberRelationRow.from_payload(row))
            except BundleRelationError:
                # Distinct-code refusals (evidence over budget) keep their
                # own code; re-wrapping would flatten them to schema_invalid.
                raise
            except (TypeError, ValueError) as exc:
                raise BundleRelationError(
                    f"Bundle member-relation row is outside the closed S6 schema: {exc}",
                    code="bundle_relation_schema_invalid",
                ) from exc
        table = MemberRelationTable(rows)
        table.validate_against_members(self._members.keys())
        return settle_relation_rows(
            table, self._members, first_new=len(existing_rows), assess_unclaimed=assess_unclaimed
        )

    def _relation_table_for_removal(
        self,
        removed_names: Sequence[str],
        *,
        cascade_relations: bool,
        operation: str,
    ) -> MemberRelationTable | None:
        """Return the post-removal relation table, refusing typed first (R5).

        Called BEFORE any member is removed so the refusal is atomic.

        Returns
        -------
        MemberRelationTable | None
            The cascaded table to install after removal succeeds, or
            ``None`` when no relation row names a removed member (table
            unchanged, view identity preserved).

        Raises
        ------
        BundleRelationError
            ``bundle_member_has_relations`` when a removed member is named
            in a relation row and ``cascade_relations`` is ``False``
            (silent orphaning is forbidden).
        """

        removed = set(removed_names)
        related = sorted({name for name in removed if self._member_relations.rows_naming(name)})
        if not related:
            return None
        if not cascade_relations:
            raise BundleRelationError(
                f"Bundle.{operation} would orphan member-relation rows naming "
                f"{related}; pass cascade_relations=True to drop those rows "
                "explicitly, or remove the relations first (S6 R5).",
                code="bundle_member_has_relations",
                operation=operation,
                related_members=related,
            )
        return MemberRelationTable(
            tuple(
                row
                for row in self._member_relations.rows
                if not (set(row.named_members()) & removed)
            )
        )

    @property
    def supergraph(self) -> Supergraph:
        """Return the lazily built bundle supergraph.

        Returns
        -------
        Supergraph
            Cached union graph for the bundle members.
        """

        return self._ensure_supergraph()

    def at(self, site: Any) -> Any:
        """Return the cross-member Super view for ``site``.

        A string is a label from any Bundle accessor family. Any other
        selector-like site (``tl.func("relu")``, ``tl.module(address)``)
        resolves to exactly one op in every member, after the model-axis
        floor of the comparison gate passes for every member pair (a view
        read, so different input values are allowed). ``compare()`` on the
        returned view is the member-by-member matrix at this site and first
        runs the comparison gate (shared graph, same input values) over the
        members it reads. To read a subset of members, build a bundle of
        them: ``tl.Bundle({name: bundle[name] for name in names}).at(site)``.

        Parameters
        ----------
        site:
            Label from any Bundle accessor family, or a selector-like site.

        Returns
        -------
        Any
            Matching ``Super*`` view.

        Raises
        ------
        AmbiguousLabelError
            If an unrecognized-format label is present in multiple accessors.
        KeyError
            If the label is absent from every accessor.
        BundleComparisonError
            If a selector-like site's members fail the model-axis floor.
        BundleMemberError
            If a selector-like site fails to resolve in some member.
        """

        view = self._label_view(site) if isinstance(site, str) else self._site_view(site)
        names = list(getattr(view, "members", {}))
        pairs = [(left, right) for index, left in enumerate(names) for right in names[index + 1 :]]
        view._compare_gate = functools.partial(self._require_comparable, "at.compare", pairs)
        return view

    def _site_view(self, site: Any) -> SuperOp:
        """Resolve a selector-like site to one op per member (the model-floor gated read)."""

        names = list(self._members)
        pairs = [(left, right) for index, left in enumerate(names) for right in names[index + 1 :]]
        self._require_comparable("at", pairs=pairs)
        self._ensure_supergraph()
        layer_members: dict[str, Op] = {}
        failures: dict[str, str] = {}
        for name in names:
            try:
                table = _resolve_sites(self._members[name], site, max_fanout=1)
                layer_members[name] = cast("Op", table.first())
            except Exception as exc:  # noqa: BLE001 - rewrapped with member context
                failures[name] = str(exc)
        if failures:
            detail = "; ".join(f"{name}: {message}" for name, message in failures.items())
            raise BundleMemberError(f"site {site!r} failed to resolve for bundle members: {detail}")
        return SuperOp.from_members(site, layer_members)

    def _label_view(self, site: str) -> Any:
        """Return the Super view for a label from any Bundle accessor family."""

        return label_view(self, site)

    def add(
        self,
        traces: Trace | Sequence[Trace],
        names: str | Sequence[str] | None = None,
    ) -> Bundle:
        """Add one or more member logs and invalidate the cached supergraph.

        Parameters
        ----------
        traces:
            Model log or logs to add.
        names:
            Optional member name or names.

        Returns
        -------
        Bundle
            This bundle.
        """

        logs = coerce_trace_list(traces, arg_name="log_or_logs")
        member_names = coerce_optional_name_list(names, count=len(logs))
        for index, log in enumerate(logs):
            member_name = derive_name(
                log,
                name=member_names[index],
                index=len(self._members),
            )
            if member_name in self._members:
                raise ValueError(f"Bundle member names must be unique; duplicate {member_name!r}.")
            self._members[member_name] = log
            self._member_construction[member_name] = {"origin": "added"}
        self._supergraph = None
        self._enforce_capacity()
        return self

    def remove(
        self,
        names: str | Trace | Sequence[str | Trace],
        *,
        cascade_relations: bool = False,
    ) -> Trace | list[Trace]:
        """Remove and return one or more members by name or Trace object.

        Parameters
        ----------
        names:
            Member name, Trace object, or a list of either.
        cascade_relations:
            Whether relation rows naming a removed member are dropped with
            it. ``False`` (default) refuses typed BEFORE any member is
            removed when a removed member is named in a relation row.

        Returns
        -------
        Trace | list[Trace]
            Removed member, or removed members for list input.

        Raises
        ------
        BundleRelationError
            ``bundle_member_has_relations`` when a removed member has
            relation rows and ``cascade_relations`` is ``False`` (S6 R5).
        """

        is_many = is_list_like(names)
        names = coerce_member_name_list(self, names)
        unknown = [name for name in names if name not in self._members]
        if unknown:
            raise KeyError(f"Unknown bundle member(s): {sorted(unknown)}")
        new_table = self._relation_table_for_removal(
            names, cascade_relations=cascade_relations, operation="remove"
        )
        removed: list[Trace] = []
        for name in names:
            log = self._members.pop(name)
            removed.append(log)
            self._member_construction.pop(name, None)
            if self._baseline_name == name:
                self._baseline_name = None
        if new_table is not None:
            self._member_relations = new_table
        self._supergraph = None
        return removed if is_many else removed[0]

    def remove_except(
        self,
        keep: str | Trace | Sequence[str | Trace],
        *,
        cascade_relations: bool = False,
    ) -> None:
        """Remove every member whose name is not listed in ``keep``.

        Parameters
        ----------
        keep:
            Member name, Trace object, or list of either to retain.
        cascade_relations:
            Whether relation rows naming any removed member are dropped
            with it. ``False`` (default) refuses typed BEFORE any member is
            removed when a removed member is named in a relation row.

        Returns
        -------
        None
            The bundle is mutated in place.

        Raises
        ------
        BundleRelationError
            ``bundle_member_has_relations`` when a removed member has
            relation rows and ``cascade_relations`` is ``False`` (S6 R5).
        """

        keep_set = set(coerce_member_name_list(self, keep))
        unknown = keep_set - set(self._members)
        if unknown:
            raise KeyError(f"Unknown bundle member(s): {sorted(unknown)}")
        removed_names = [name for name in self._members if name not in keep_set]
        new_table = self._relation_table_for_removal(
            removed_names, cascade_relations=cascade_relations, operation="remove_except"
        )
        self._members = OrderedDict(
            (name, log) for name, log in self._members.items() if name in keep_set
        )
        if self._baseline_name is not None and self._baseline_name not in self._members:
            self._baseline_name = None
        if new_table is not None:
            self._member_relations = new_table
        self._supergraph = None

    @property
    def capacity(self) -> int | None:
        """Return the configured member capacity.

        Returns
        -------
        int | None
            Member capacity, or ``None`` when uncapped.
        """

        return self._capacity

    @capacity.setter
    def capacity(self, n: int | None) -> None:
        """Set member capacity with LRU-style eviction while preserving the baseline.

        Parameters
        ----------
        n:
            Maximum member count, or ``None`` to remove the cap.
        """

        if n is None:
            self._capacity = None
            return
        if n < 1:
            raise ValueError("Bundle capacity must be at least 1.")
        self._capacity = n
        self._enforce_capacity()

    def clear(self, *, cascade_relations: bool = False) -> None:
        """Remove all non-baseline members.

        Parameters
        ----------
        cascade_relations:
            Whether relation rows naming any removed member are dropped
            with it. ``False`` (default) refuses typed BEFORE any member is
            removed when a removed member is named in a relation row.

        Returns
        -------
        None
            The bundle is mutated in place.

        Raises
        ------
        BundleRelationError
            ``bundle_member_has_relations`` when a removed member has
            relation rows and ``cascade_relations`` is ``False`` (S6 R5).
        """

        keep_baseline = self._baseline_name is not None and self._baseline_name in self._members
        removed_names = [
            name for name in self._members if not (keep_baseline and name == self._baseline_name)
        ]
        new_table = self._relation_table_for_removal(
            removed_names, cascade_relations=cascade_relations, operation="clear"
        )
        if keep_baseline:
            baseline_name = cast("str", self._baseline_name)
            baseline = self._members[baseline_name]
            self._members = OrderedDict([(baseline_name, baseline)])
        else:
            self._members.clear()
            self._baseline_name = None
        for removed_name in removed_names:
            self._member_construction.pop(removed_name, None)
        if new_table is not None:
            self._member_relations = new_table
        self._supergraph = None

    def do(self, *args: Any, **kwargs: Any) -> Bundle:
        """Apply ``Trace.do`` to every member.

        Returns
        -------
        Bundle
            This bundle.
        """

        for member in self._members.values():
            member.do(*args, **kwargs)
        return self

    def fork(self, name: str | None = None) -> Bundle:
        """Fork all member logs into a new bundle.

        Lineage survives the fork (ledger memo item 0b): the relation table
        and preserved sections carry over, the child mints a NEW
        ``bundle_id``, every member's construction anchor records the source
        container/member, and both containers append a ``fork`` chronology
        row — the historical behavior (relation table dropped, no
        ``forked_from`` evidence anywhere) made live-only lineage that could
        not back an artifact-level join.

        Parameters
        ----------
        name:
            Optional suffix prefix for forked member names.

        Returns
        -------
        Bundle
            New bundle containing forked logs.
        """

        forked: OrderedDict[str, Trace] = OrderedDict()
        for member_name, member in self._members.items():
            fork_name = f"{name}_{member_name}" if name is not None else None
            forked[member_name] = member.fork(name=fork_name)
        child = Bundle(
            forked,
            baseline=self._baseline_name,
            relations=self._member_relations.rows,
            preserved_sections=self._preserved_sections,
        )
        child._forked_from_bundle_id = self._bundle_id
        self._record_bundle_operation(
            "fork",
            member_names=tuple(forked),
            params={"child_bundle_id": child._bundle_id},
        )
        child_row = child._record_bundle_operation(
            "fork",
            member_names=tuple(forked),
            params={"source_bundle_id": self._bundle_id},
        )
        child._member_construction = {
            member_name: {
                "origin": "forked",
                "source_bundle_id": self._bundle_id,
                "source_member": member_name,
                "operation_id": child_row.operation_id,
            }
            for member_name in forked
        }
        return child

    def run(self, model: nn.Module, x: Any = None, **kwargs: Any) -> Bundle:
        """Run all member logs with a supplied model and input.

        Parameters
        ----------
        model:
            Model forwarded to each member.
        x:
            Forward input.

        Returns
        -------
        Bundle
            This bundle.
        """

        for member in self._members.values():
            member.run(model, x, **kwargs)
        return self

    def map(self, fn: Callable[..., Any]) -> dict[str, Any]:
        """Apply a function independently to each member; released members read ``"unmeasured"``.

        Parameters
        ----------
        fn:
            Callable receiving one member log. A callable that accepts a
            second positional parameter additionally receives the member's
            NAME (``bundle.map(lambda log, name: ...)``); single-parameter
            callables keep the historical contract unchanged.

        Returns
        -------
        dict[str, Any]
            Results keyed by member name. A candidate the effect table
            records that the bundle no longer holds (released, refused or
            failed) is keyed too, with the value ``"unmeasured"``: it is
            never silently re-scored and never silently dropped.
        """

        pass_name = self._accepts_member_name(fn)
        results: dict[str, Any] = {
            name: (fn(member, name) if pass_name else fn(member))
            for name, member in self._members.items()
        }
        for candidate in self._released_candidates():
            results.setdefault(candidate, "unmeasured")
        return results

    def _released_candidates(self) -> list[str]:
        """Effect-table candidates with no retained member, sorted, each once."""

        released: set[str] = set()
        for table in self._effect_tables.values():
            for row in table.rows:
                if row.candidate_id == "__baseline__":
                    continue
                if row.member_name is None or row.member_name not in self._members:
                    released.add(row.candidate_id)
        return sorted(released)

    @staticmethod
    def _accepts_member_name(fn: Callable[..., Any]) -> bool:
        """Whether ``fn`` can take (member, name) rather than (member,) only.

        Signature inspection failures (builtins, C callables) fall back to
        the historical single-argument call, never a guessed two-argument
        call that would raise mid-iteration.
        """

        import inspect

        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            return False
        positional_kinds = (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        count = 0
        for parameter in signature.parameters.values():
            if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
                return True
            if parameter.kind in positional_kinds:
                count += 1
        return count >= 2

    def show(self, kind: str | None = None, **kwargs: Any) -> dict[str, str | None]:
        """Render each bundle member graph as a member-keyed strip.

        Parameters
        ----------
        kind:
            Picture kind forwarded to each member's :meth:`Trace.show`.
        **kwargs:
            Forwarded to each member's :meth:`Trace.show`. When an output
            path is supplied, member names are appended to produce one artifact
            per log. ``vis_mode='none'`` is accepted and returns without
            rendering, matching ``Trace.show``.

        Returns
        -------
        dict[str, str | None]
            Member-keyed render results. Values are DOT source strings when
            rendering occurs, or ``None`` for skipped ``vis_mode='none'`` calls.
        """

        if kwargs.get("vis_mode") == "none":
            return dict.fromkeys(self._members)

        base_outpath = kwargs.get("path")
        results: dict[str, str | None] = {}
        for name, member in self._members.items():
            member_kwargs = dict(kwargs)
            if isinstance(base_outpath, str):
                member_kwargs["path"] = f"{base_outpath}_{name}"
            if kind == "repr":
                results[name] = repr(member)
            elif kind == "html":
                results[name] = member._repr_html_()
            else:
                results[name] = member.draw(**member_kwargs)
        return results

    def draw(
        self,
        path: str | None = None,
        *,
        graph: BundleGraphLiteral = "forward",
        nodes: Literal["ops", "members"] = "ops",
        rolled: bool = False,
        direction: VisDirectionLiteral | None = None,
        theme: str | None = None,
        show_legend: bool | None = None,
        file_format: str | None = None,
        open_viewer: bool = False,
        graph_overrides: dict[str, Any] | None = None,
        node_overrides: dict[str, Any] | None = None,
        edge_overrides: dict[tuple[str, str], Any] | None = None,
        members: tuple[str | Trace, str | Trace] | None = None,
        metric: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = "relative_l2",
        on: Literal["out", "grad"] = "out",
    ) -> str:
        """Draw the bundle as one picture and return its DOT source.

        Parameters
        ----------
        path:
            Output path; a recognised extension picks the format. A bare call
            writes a collision-safe ``bundle_modelgraph`` file (``bundle_diff``
            for ``graph="diff"``) in the working directory.
        graph:
            Which picture: ``"forward"`` (default), ``"backward"`` or
            ``"combined"`` draw the bundle's union op graph for that pass;
            ``"diff"`` draws two members side by side with their aligned
            sites coloured by ``metric`` distance.
        nodes:
            ``"ops"`` (default) or ``"members"``: one node per member and one
            edge per relation row (``graph`` must stay ``"forward"``).
        rolled:
            Draw the rolled (per-layer) graph instead of every op.
        direction:
            Graphviz layout direction; ``None`` is ``"bottomup"``.
        theme:
            Skin preset, the names ``Trace.draw`` takes. Defaults to
            ``"torchlens"``, or ``"paper"`` for ``graph="diff"``.
        show_legend:
            ``True`` draws the key under the header.
        file_format:
            Output format. Defaults to ``"pdf"``, or ``"svg"`` for ``graph="diff"``.
        open_viewer:
            Open the rendered file; ``False`` only writes it.
        graph_overrides:
            Graph-level Graphviz attributes.
        node_overrides:
            Graphviz node attributes keyed by op label (member name under
            ``nodes="members"``).
        edge_overrides:
            Graphviz edge attributes keyed by ``(source, target)`` labels.
        members:
            For ``graph="diff"``: the ``(left, right)`` members to compare.
            Defaults to the baseline and the next member.
        metric:
            For ``graph="diff"``: the per-site tensor metric.
        on:
            For ``graph="diff"``: the tensor field compared at each site.

        Returns
        -------
        str
            Graphviz DOT source of the rendered picture.
        """

        return _bundle_draw(
            self,
            path,
            graph=graph,
            nodes=nodes,
            rolled=rolled,
            direction=direction,
            theme=theme,
            show_legend=show_legend,
            file_format=file_format,
            open_viewer=open_viewer,
            graph_overrides=graph_overrides,
            node_overrides=node_overrides,
            edge_overrides=edge_overrides,
            members=members,
            metric=metric,
            on=on,
        )

    def rank_changes(
        self,
        baseline: str | Trace | None = None,
        *,
        k: int = 10,
        metric: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = "cosine",
    ) -> list[tuple[str, float]]:
        """Rank sites by out distance from a baseline (the SITE axis).

        Each site's score is the ``metric`` distance from the baseline's
        activation to every other member's, AVERAGED across those members —
        the answer to "where did the bundle diverge", never "which member
        mattered" (the member axis lives on the effect table).

        Sites pair through the structural alignment (:attr:`supergraph`), not
        by literal label; a site some member lacks is not scored.

        Parameters
        ----------
        baseline:
            Optional baseline override.
        k:
            Maximum rows to RETURN. Every site is scored first; this is a
            display cap on the sorted result, not a compute bound.
        metric:
            Pairwise tensor metric.

        Returns
        -------
        list[tuple[str, float]]
            ``(site_label, score)`` rows sorted descending.
        """

        baseline_name = self._baseline_or_raise(baseline)
        # SITE AXIS (ledger memo D3f): most_changed answers "WHERE did the
        # bundle diverge" — per site it AVERAGES the metric from the baseline
        # to every other member. The member axis ("WHICH candidate mattered")
        # lives on the effect table, where most_changed is unambiguous.
        # top_k caps the RETURNED rows after every site is scored (a display
        # cap, never a compute bound or a coverage claim).
        # Operand scoping (A-GATE item 3): most_changed compares baseline
        # against every other member, never other members against each other,
        # so only baseline pairs must pass the comparison gate. Site reads
        # across all members still run the at() model-floor gate.
        self._require_comparable(
            "most_changed",
            pairs=[(baseline_name, name) for name in self._members if name != baseline_name],
        )
        baseline_log = self._members[baseline_name]
        scored: list[tuple[str, float]] = []
        baseline_sites = list(getattr(baseline_log, "layer_list", []))
        if baseline_sites:  # the at() model-floor gate, once instead of per site
            names = list(self._members)
            pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]
            self._require_comparable("at", pairs=pairs)
        aligned = aligned_site_labels(self._ensure_supergraph(), baseline_name)
        for site in baseline_sites:
            members = resolve_aligned_site(self._members, str(site.layer_label), aligned)
            if members is None:
                continue
            outs = {
                name: getattr(member, "out", None)
                for name, member in members.items()
                if getattr(member, "has_saved_activation", False)
            }
            base = outs.get(baseline_name)
            if not isinstance(base, torch.Tensor):
                continue
            distances: list[float] = []
            for name, out in outs.items():
                if name == baseline_name or not isinstance(out, torch.Tensor):
                    continue
                label = str(site.layer_label)
                distances.append(_distance_value(base, out, metric, site=label))
            if distances:
                scored.append((str(site.layer_label), sum(distances) / len(distances)))
        scored.sort(key=lambda row: row[1], reverse=True)
        return scored[:k]

    def compare(
        self,
        a: str,
        b: str,
        *,
        metric: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = "cosine",
        on: Literal["out", "grad"] = "out",
    ) -> list[tuple[str, float]]:
        """Return per-site differences between two members.

        The receiver sets the comparison axis: ``bundle.compare(a, b)`` reads
        one member pair across every shared site; ``bundle.at(site).compare()``
        is the member-by-member matrix at one site. The pair goes through the
        comparison gate first (operand scoping: the two-member form reads
        exactly one pair, so a foreign or ordered third member never disables
        it) and refuses typed when the members cannot be aligned. The grade of
        the pair is :meth:`comparability`.

        Parameters
        ----------
        a:
            First member name.
        b:
            Second member name.
        metric:
            Metric name or callable over ``(left, right)`` tensors; scalar-like
            sites always use ``relative_l1``.
        on:
            Tensor field to compare: ``"out"`` or ``"grad"``.

        Returns
        -------
        list[tuple[str, float]]
            ``(site_label, distance)`` rows for sites both members have,
            paired through the structural alignment (:attr:`supergraph`).
        """

        self._require_comparable("diff", pairs=[(a, b)])
        return self._diff_members(a, b, metric=metric, on=on)

    def comparability(self, a: str, b: str) -> Relationship:
        """Return the derived relationship for a pair of members.

        Parameters
        ----------
        a:
            First member name.
        b:
            Second member name.

        Returns
        -------
        Relationship
            Derived relationship.
        """

        return self._relationship_between(self._members[a], self._members[b])

    def _delta_map(
        self,
        metric: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = "relative_l2",
        *,
        baseline: str | Trace | None = None,
        on: Literal["out", "grad"] = "out",
    ) -> dict[str, dict[str, float]]:
        """Per-node ``metric`` distances of every member from the baseline (gated)."""

        return _bundle_delta_map(self, metric, baseline=baseline, on=on)

    def _norm_delta(
        self, *, baseline: str | Trace | None = None, on: Literal["out", "grad"] = "out"
    ) -> dict[str, dict[str, float]]:
        """Per-node relative L2 distances of every member from the baseline (gated)."""

        return _bundle_norm_delta(self, baseline=baseline, on=on)

    def _output_delta(
        self,
        target: str | Trace,
        *,
        metric: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = "relative_l2",
        on: Literal["out", "grad"] = "out",
    ) -> dict[str, dict[str, float]]:
        """Output-site ``metric`` distances of every member from ``target`` (gated)."""

        return _bundle_output_delta(self, target, metric=metric, on=on)

    def _aligned_pairs(
        self,
        left: str | Trace | None = None,
        right: str | Trace | None = None,
        *,
        min_score: float = 0.45,
    ) -> list[tuple[Any, Any]]:
        """Best-match layer pairs across two members (the diff picture's rows)."""

        return _bundle_aligned_pairs(self, left, right, min_score=min_score)

    def _diff_members(
        self,
        left_name: str,
        right_name: str,
        *,
        metric: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = "cosine",
        on: Literal["out", "grad"] = "out",
    ) -> list[tuple[str, float]]:
        """Return per-site differences between two members.

        Parameters
        ----------
        left_name:
            First member name.
        right_name:
            Compared member name.
        metric:
            Metric name or callable (scalar-like sites use ``relative_l1``).
        on:
            Tensor field read at each site.

        Returns
        -------
        list[tuple[str, float]]
            ``(site_label, distance)`` rows for sites both members have,
            paired through the structural alignment (:attr:`supergraph`).
        """

        left_log = self._members[left_name]
        right_log = self._members[right_name]
        # _distance_value holds the shared per-site rules: scalar fallback
        # only when both operands are scalar-like, identical payloads read
        # 0.0, different shapes refuse (typed) naming the site.
        rows: list[tuple[str, float]] = []
        aligned = aligned_site_labels(self._ensure_supergraph(), left_name)
        for left_site in getattr(left_log, "layer_list", []):
            label = str(left_site.layer_label)
            right_label = aligned.get(label, {}).get(right_name, label)
            try:
                right_site = _resolve_sites(right_log, right_label, max_fanout=1).first()
            except Exception:  # noqa: BLE001 - missing sites are not common sites
                continue
            left_out = getattr(left_site, on, None)
            right_out = getattr(right_site, on, None)
            if not isinstance(left_out, torch.Tensor) or not isinstance(
                right_out,
                torch.Tensor,
            ):
                continue
            rows.append((label, _distance_value(left_out, right_out, metric, site=label)))
        return rows

    @property
    def comparability_table(self) -> dict[tuple[str, str], Relationship]:
        """Return pairwise relationships for all member pairs.

        Returns
        -------
        dict[tuple[str, str], Relationship]
            Pairwise matrix keyed by member-name tuples.
        """

        matrix: dict[tuple[str, str], Relationship] = {}
        for left in self._members:
            for right in self._members:
                matrix[(left, right)] = self.comparability(left, right)
        return matrix

    def store_comparison(
        self,
        metric: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = "relative_l2",
        *,
        name: str | None = None,
        baseline: str | Trace | None = None,
        on: Literal["out", "grad"] = "out",
    ) -> str:
        """Compute one per-node comparison once and store it on the supergraph.

        Returns the stored name (the metric's name unless ``name`` is given);
        :meth:`stored_comparison` reads it back without recomputing.
        """

        return _bundle_store_comparison(self, metric, name=name, baseline=baseline, on=on)

    def stored_comparison(self, name: str) -> dict[str, dict[str, float]]:
        """Return one stored per-node comparison by name, without recomputing."""

        return _bundle_stored_comparison(self, name)

    def stored_comparison_names(self) -> tuple[str, ...]:
        """Return every comparison name stored on this bundle's supergraph, sorted."""

        return _bundle_stored_comparison_names(self)

    def _coerce_member_name(self, value: str | Trace) -> str:
        """Resolve one member reference (a name or the member object) to its name."""

        return coerce_member_name(self, value)

    def _resolve_baseline_name(self, baseline: str | Trace | None) -> str | None:
        """Resolve a baseline constructor argument to a member name.

        Returns
        -------
        str | None
            Baseline member name.
        """

        if baseline is None:
            return self._auto_detect_baseline()
        if isinstance(baseline, str):
            if baseline not in self._members:
                raise KeyError(f"Unknown baseline member {baseline!r}.")
            return baseline
        for name, member in self._members.items():
            if member is baseline:
                return name
        raise KeyError("Baseline Trace is not a member of this Bundle.")

    def _auto_detect_baseline(self) -> str | None:
        """Auto-detect a pristine baseline when exactly one candidate exists.

        A member is a candidate only when ``Trace.is_pristine`` reads ``True``;
        ``None`` (no evidence recorded) never counts as pristine.

        Returns
        -------
        str | None
            Baseline name or ``None`` when ambiguous/not needed yet.
        """

        candidates = [
            name
            for name, member in self._members.items()
            if getattr(member, "is_pristine", None) is True
        ]
        return candidates[0] if len(candidates) == 1 else None

    def _baseline_or_raise(self, baseline: str | Trace | None) -> str:
        """Return a baseline name or raise for ambiguity.

        Returns
        -------
        str
            Baseline member name.
        """

        if baseline is not None:
            resolved = self._resolve_baseline_name(baseline)
            if resolved is not None:
                return resolved
        if self._baseline_name is not None:
            return self._baseline_name
        detected = self._auto_detect_baseline()
        if detected is not None:
            self._baseline_name = detected
            return detected
        raise BaselineUndeterminedError(
            "Bundle operation requires a baseline, but no unique pristine baseline exists."
        )

    def _ensure_supergraph(self) -> Supergraph:
        """Build and cache the supergraph lazily.

        Returns
        -------
        Supergraph
            Cached supergraph.
        """

        if self._supergraph is None:
            self._supergraph = build_supergraph(list(self._members.values()), list(self._members))
        return self._supergraph

    def _shared(self, key_fn: Callable[[Trace], Sequence[Any]]) -> list[str]:
        """Return keys common to every member, ordered by the first member.

        Parameters
        ----------
        key_fn:
            Function returning candidate keys for one member trace.

        Returns
        -------
        list[str]
            Keys present in all members.
        """

        member_keys = self._member_key_lists(key_fn)
        key_sets = [set(keys) for keys in member_keys] or [set()]
        shared = set.intersection(*key_sets)
        return [key for key in (member_keys or [[]])[0] if key in shared]

    def _divergent(self, key_fn: Callable[[Trace], Sequence[Any]]) -> list[str]:
        """Return keys present in some members but not every member.

        Parameters
        ----------
        key_fn:
            Function returning candidate keys for one member trace.

        Returns
        -------
        list[str]
            Keys present in at least one member and absent from at least one member.
        """

        member_keys = self._member_key_lists(key_fn)
        key_sets = [set(keys) for keys in member_keys] or [set()]
        shared = set.intersection(*key_sets)
        union = set.union(*key_sets)
        divergent = union - shared
        ordered: list[str] = []
        seen: set[str] = set()
        for keys in member_keys:
            for key in keys:
                if key in divergent and key not in seen:
                    ordered.append(key)
                    seen.add(key)
        return ordered

    def _member_key_lists(self, key_fn: Callable[[Trace], Sequence[Any]]) -> list[list[str]]:
        """Return normalized per-member key lists with duplicates removed.

        Parameters
        ----------
        key_fn:
            Function returning candidate keys for one member trace.

        Returns
        -------
        list[list[str]]
            String keys for each member in member order.
        """

        member_keys: list[list[str]] = []
        for member in self._members.values():
            keys: list[str] = []
            seen: set[str] = set()
            for raw_key in key_fn(member):
                if raw_key is None:
                    continue
                key = str(raw_key)
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
            member_keys.append(keys)
        return member_keys

    def _enforce_capacity(self) -> None:
        """Evict oldest non-baseline members when above capacity.

        Returns
        -------
        None
            The bundle is mutated in place.
        """

        if self._capacity is None:
            return
        while len(self._members) > self._capacity:
            evictable = next(
                (name for name in self._members if name != self._baseline_name),
                None,
            )
            if evictable is None:
                break
            if self._member_relations.rows_naming(evictable):
                # R5: an IMPLICIT eviction may neither orphan relation rows
                # nor cascade them silently — refuse typed, always.
                raise BundleRelationError(
                    f"Bundle capacity eviction would orphan member-relation "
                    f"rows naming {evictable!r}; remove the member explicitly "
                    "(cascade_relations=True) or raise the capacity (S6 R5).",
                    code="bundle_member_has_relations",
                    operation="eviction",
                    related_members=[evictable],
                )
            self._members.pop(evictable)
            self._member_construction.pop(evictable, None)
            self._supergraph = None

    def _require_comparable(
        self,
        operation: str,
        pairs: Sequence[tuple[str, str]] | None = None,
    ) -> None:
        """Run the two-predicate comparison gate over the operand pairs.

        Thin delegation to :func:`torchlens.bundle._compare_gate.require_comparable`
        (topology -> model-axis floor -> value-level input identity, per
        operand pair; A-GATE, foldB D6/D18).

        Parameters
        ----------
        operation:
            Gated operation name (a ``_GATE_REQUIREMENTS`` key).
        pairs:
            Operand pairs the operation actually reads; ``None`` means every
            i<j member pair.

        Raises
        ------
        BundleComparisonError
            With the stable gate code on ``fields["code"]``.
        """

        require_comparable(
            self._members,
            self._member_relations,
            operation,
            pairs,
            self._relationship_between,
        )

    @classmethod
    def _relationship_between(cls, left: Trace, right: Trace) -> Relationship:
        """Derive relationship evidence for two model logs.

        Identity ranks require LIVE evidence: two distinct Trace objects
        derive ``same_object`` / ``same_model_at_capture`` only through a
        shared live weak model reference. The persisted ``model_object_id``
        is deliberately never sufficient — object ids do not survive (or
        stay unique across) a save/load boundary, so trusting them
        manufactured identity-rank upgrades between loaded artifacts (the
        A-GATE rank-upgrade audit pins this: loaded pairs settle to the
        graph/weight evidence that actually round-trips).

        Returns
        -------
        Relationship
            Highest-confidence relationship.
        """

        if left is right:
            return Relationship.SAME_OBJECT

        left_class = getattr(left, "model_class_qualname", None)
        right_class = getattr(right, "model_class_qualname", None)
        left_weight = cls._weight_fingerprint(left)
        right_weight = cls._weight_fingerprint(right)

        left_model = cls._weak_model(left)
        right_model = cls._weak_model(right)
        if (
            left_model is not None
            and left_model is right_model
            and left_class is not None
            and left_class == right_class
        ):
            if left_weight is not None and left_weight == right_weight:
                return Relationship.SAME_OBJECT
            return Relationship.SAME_MODEL_OBJECT_AT_CAPTURE

        left_graph = getattr(left, "graph_shape_hash", None)
        right_graph = getattr(right, "graph_shape_hash", None)
        left_input = getattr(left, "input_signature_hash", None)
        right_input = getattr(right, "input_signature_hash", None)
        if left_graph is not None and left_graph == right_graph:
            if left_input is not None and left_input == right_input:
                return Relationship.SHARED_GRAPH_SAME_INPUT
            return Relationship.SHARED_GRAPH_DIFFERENT_INPUT

        if left_weight is not None and left_weight == right_weight:
            return Relationship.SAME_PARAM_SHAPES
        if left_class is not None and left_class == right_class:
            return Relationship.SHARED_ARCHITECTURE
        if left_class is not None and right_class is not None and left_class != right_class:
            return Relationship.DIFF_MODEL
        return Relationship.UNKNOWN

    @staticmethod
    def _weight_fingerprint(log: Trace) -> str | None:
        """Return the strongest available weight fingerprint.

        Returns
        -------
        str | None
            Fingerprint value.
        """

        return getattr(log, "param_hash_full", None) or getattr(
            log,
            "param_hash_quick",
            None,
        )

    @staticmethod
    def _weak_model(log: Trace) -> Any | None:
        """Resolve a captured weak model reference.

        Returns
        -------
        Any | None
            Live model object, if available.
        """

        ref = getattr(log, "_source_model_ref", None)
        if ref is None:
            return None
        try:
            return ref()
        except TypeError:
            return None


__all__ = ["AmbiguousLabelError", "SuperAtenOp", "SuperOp"]
