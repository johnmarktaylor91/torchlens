"""Single Bundle type for intervention-ready TorchLens model logs."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import torch

from .._utils.facade import refuse_removed_member
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
from ..intervention._topology.topology import Supergraph
from ..intervention.types import Relationship
from . import (
    _aligned_sites,
    _card,
    _comparisons,
    _lookup,
    _members,
    _operations,
    _relation_carriage,
    _save,
)
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
from ._labels import AmbiguousLabelError, label_view
from ._lineage import BundleOperation, MemberEffectTable, mint_bundle_id
from ._lookup import _BundleAccessorProperty, _BundleStructuralProperty
from ._members import coerce_member_name, parse_members
from ._outcome_fold import _fold_member_outcomes as _fold_member_outcomes  # re-export
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
    from .._data_classes.trace import Trace
    from .._literals import VisDirectionLiteral
    from ..viz._drawing import Drawing
    from ._draw import BundleGraphLiteral
    from ._token_spans import TokenPairs


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

        return _card.outcome_distribution(self)

    def __repr__(self) -> str:
        """One-line bundle card: members, baseline, outcome distribution.

        Poison/divergence facts sit ABOVE the fold (lovely matrix): any
        non-complete member outcome prints in the distribution, never
        behind a member lookup. A missing baseline renders as unquoted
        None, never the string 'None' masquerading as a member name
        (lovely bug 17).
        """

        return _card.bundle_repr(self)

    def __str__(self) -> str:
        """Bounded bundle view: repr line + per-member one-liners (F10)."""

        return _card.bundle_str(self)

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

        return _card.bundle_outcome(self)

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

        _save.save_bundle(self, path, level=level, overwrite=overwrite)

    def __getattr__(self, name: str) -> Any:
        """Refuse a removed Bundle member with the typed redirect naming its replacement.

        Raises
        ------
        AttributeError
            If ``name`` is not a Bundle attribute.
        """

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

        return _lookup.is_structurally_consistent(self)

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

        return _members.kind_names(self, kind, shared=shared)

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

        return _operations.record_bundle_operation(
            self, kind, member_names=member_names, params=params
        )

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

        return _operations.effects_table(self, history_id)

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

        return _relation_carriage.build_relation_table(
            self, existing_rows, new_rows, assess_unclaimed=assess_unclaimed
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

        return _relation_carriage.relation_table_for_removal(
            self, removed_names, cascade_relations=cascade_relations, operation=operation
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
        runs the comparison gate (shared graph; same input values or, for runs
        on different inputs, graphs that line up) over the members it reads. To read a subset of members, build a bundle of
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

        return _lookup.bundle_at(self, site)

    def _site_view(self, site: Any) -> SuperOp:
        """Resolve a selector-like site to one op per member (the model-floor gated read)."""

        return _aligned_sites.site_view(self, site)

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

        return _members.add_members(self, traces, names)

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

        return _members.remove_members(self, names, cascade_relations=cascade_relations)

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

        _members.remove_except(self, keep, cascade_relations=cascade_relations)

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

        _members.set_capacity(self, n)

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

        _members.clear_members(self, cascade_relations=cascade_relations)

    def do(self, *args: Any, **kwargs: Any) -> Bundle:
        """Apply ``Trace.do`` to every member.

        Returns
        -------
        Bundle
            This bundle.
        """

        return _operations.bundle_do(self, *args, **kwargs)

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

        return _operations.bundle_fork(self, name)

    def run(self, model: nn.Module, inputs: Any = None, **kwargs: Any) -> Bundle:
        """Run all member logs with a supplied model and input.

        Parameters
        ----------
        model:
            Model forwarded to each member.
        inputs:
            Forward input.

        Returns
        -------
        Bundle
            This bundle.
        """

        return _operations.bundle_run(self, model, inputs, **kwargs)

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

        return _operations.bundle_map(self, fn)

    @staticmethod
    def _accepts_member_name(fn: Callable[..., Any]) -> bool:
        """Whether ``fn`` can take (member, name) rather than (member,) only.

        Signature inspection failures (builtins, C callables) fall back to
        the historical single-argument call, never a guessed two-argument
        call that would raise mid-iteration.
        """

        return _operations.accepts_member_name(fn)

    def show(self, kind: str | None = None, **kwargs: Any) -> dict[str, Drawing | str]:
        """Render each bundle member graph as a member-keyed strip.

        Parameters
        ----------
        kind:
            Picture kind forwarded to each member's :meth:`Trace.show`.
        **kwargs:
            ``Trace.draw`` keywords, forwarded to each member. When a ``path``
            is supplied, member names are appended to produce one artifact
            per member.

        Returns
        -------
        dict[str, Drawing | str]
            Member-keyed results: each member's :class:`Drawing`, or the
            ``repr`` / HTML text for those kinds.
        """

        return _operations.bundle_show(self, kind, **kwargs)

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

        ``graph`` picks the picture: ``"forward"`` (default), ``"backward"`` or
        ``"combined"`` draw the bundle's union op graph for that pass, and ``"diff"``
        draws the ``members`` pair (default: the baseline and the next member) side by
        side with their aligned sites coloured by ``metric`` distance over the ``on``
        tensor field (``"out"`` or ``"grad"``). ``nodes="members"`` draws one node per
        member and one edge per relation row (``graph`` must stay ``"forward"``);
        ``rolled`` draws the rolled (per-layer) graph instead of every op.

        ``path`` is the output file (a recognised extension picks the format); a bare
        call writes a collision-safe ``bundle_modelgraph`` file (``bundle_diff`` for
        ``graph="diff"``) in the working directory. ``file_format`` defaults to
        ``"pdf"`` (``"svg"`` for diff), ``theme`` to ``"torchlens"`` (``"paper"`` for
        diff; the names ``Trace.draw`` takes) and ``direction`` to ``"bottomup"``.
        ``show_legend=True`` draws the key under the header; ``open_viewer`` opens the
        rendered file. ``graph_overrides``, ``node_overrides`` (keyed by op label, or
        member name under ``nodes="members"``) and ``edge_overrides`` (keyed by
        ``(source, target)`` labels) pass Graphviz attributes through.
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
            ``(site_label, score)`` rows sorted descending. A multi-pass op
            scores one row per pass, keyed by its pass-qualified label
            (``tanh_1_1:2``); a single-pass op keeps its bare layer label.
        """

        return _comparisons.rank_changes(self, baseline, k=k, metric=metric)

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
            paired through the structural alignment (:attr:`supergraph`). A
            multi-pass op gives one row per pass, keyed by its pass-qualified
            label (``tanh_1_1:2``); a single-pass op keeps its bare layer label.
        """

        return _comparisons.compare_members(self, a, b, metric=metric, on=on)

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
            paired through the structural alignment (:attr:`supergraph`). A
            multi-pass op gives one row per pass, keyed by its pass-qualified
            label (``tanh_1_1:2``); a single-pass op keeps its bare layer label.
        """

        return _comparisons.diff_members(self, left_name, right_name, metric=metric, on=on)

    @property
    def comparability_table(self) -> dict[tuple[str, str], Relationship]:
        """Return pairwise relationships for all member pairs.

        Returns
        -------
        dict[tuple[str, str], Relationship]
            Pairwise matrix keyed by member-name tuples.
        """

        return _comparisons.comparability_table(self)

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

        return _members.resolve_baseline_name(self, baseline)

    def _auto_detect_baseline(self) -> str | None:
        """Auto-detect a pristine baseline when exactly one candidate exists.

        A member is a candidate only when ``Trace.is_pristine`` reads ``True``;
        ``None`` (no evidence recorded) never counts as pristine.

        Returns
        -------
        str | None
            Baseline name or ``None`` when ambiguous/not needed yet.
        """

        return _members.auto_detect_baseline(self)

    def _baseline_or_raise(self, baseline: str | Trace | None) -> str:
        """Return a baseline name or raise for ambiguity.

        Returns
        -------
        str
            Baseline member name.
        """

        return _members.baseline_or_raise(self, baseline)

    def _ensure_supergraph(self) -> Supergraph:
        """Build and cache the supergraph lazily.

        Returns
        -------
        Supergraph
            Cached supergraph.
        """

        return _lookup.ensure_supergraph(self)

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

        return _members.shared_keys(self, key_fn)

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

        return _members.divergent_keys(self, key_fn)

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

        return _members.member_key_lists(self, key_fn)

    def _enforce_capacity(self) -> None:
        """Evict oldest non-baseline members when above capacity.

        Returns
        -------
        None
            The bundle is mutated in place.
        """

        _members.enforce_capacity(self)

    def _require_comparable(
        self,
        operation: str,
        pairs: Sequence[tuple[str, str]] | None = None,
    ) -> None:
        """Run the two-predicate comparison gate over the operand pairs.

        Thin delegation to :func:`torchlens.bundle._compare_gate.require_comparable`
        (topology -> model-axis floor -> site line-up: proven-equal input values
        or equal graph-shape hashes, per operand pair; A-GATE, foldB D6/D18).

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

        return _comparisons.relationship_between(left, right)

    @staticmethod
    def _weight_fingerprint(log: Trace) -> str | None:
        """Return the strongest available weight fingerprint.

        Returns
        -------
        str | None
            Fingerprint value.
        """

        return _comparisons.weight_fingerprint(log)

    @staticmethod
    def _weak_model(log: Trace) -> Any | None:
        """Resolve a captured weak model reference.

        Returns
        -------
        Any | None
            Live model object, if available.
        """

        return _comparisons.weak_model(log)


__all__ = ["AmbiguousLabelError", "SuperAtenOp", "SuperOp"]
