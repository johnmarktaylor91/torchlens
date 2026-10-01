"""Op/facet-grain history collection, Route B via fastlog (explorer D15; F25).

The differentiator tier: "we see inside the block, they see the block".
One :func:`torchlens.record` pass per logged training step captures every
selected OP site (567 on gpt2, vs ~125 leaf modules), reduced at the save
point by a declared summary transform (P1) with the raw clone skipped
(P2) -- the observation payload per site is one fused spine vector plus
one sketch count vector, folded into the SAME per-step history artifact
Route A writes (site kinds ``op`` / ``facet``).

The wrapped step runs the model's REAL forward and returns its live
output (graph intact), so a training loop replaces its ``model(x)`` call
with ``collector.observe_step(...)`` inside the step bracket -- one
forward per step, never two. Cost is DISCLOSED, never estimated silently:
``discover()`` runs one metadata-only fastlog pass and prints the same
elements-per-step plan table Route A uses (D21); the per-step wrapper
overhead is a measured gate row (D25), not a docs adjective.

Facet sites: ``facets={selector_or_label: splitter}`` declares slices of
an op site (e.g. attention heads); each declared slice becomes its own
``facet`` site with stable identity derived from the parent op site and
the facet key. Splitters run on the already-detached summary view and
must return a dict of named tensor views.

Spellings are DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import torch

from ._artifact import HistoryWriter, RamRing
from ._collector import PlanRow, StepTruth, WatchPlan, WatchSettings
from ._errors import WatchLifecycleError, WatchPlanError
from ._kernels import (
    SPINE_SLOTS,
    histogram_result_from_vector,
    sketch_vector,
    spine_result_from_vector,
    spine_vector_fused,
)
from ._quantiles import WatchRenderError
from ._schema import (
    ObservationRecord,
    RunRecord,
    SiteRecord,
    StepBlockRecord,
    validate_step_order,
)

__tl_layer__ = "L5"

#: A facet splitter: detached op output -> {facet_name: tensor view}.
FacetSplitter = Callable[[torch.Tensor], Mapping[str, torch.Tensor]]


def _op_site_id(label: str) -> str:
    """Stable op-site identity from the fastlog structural raw label."""

    return f"op:{label}"


def _facet_site_id(label: str, facet: str) -> str:
    """Stable facet-site identity from parent op site + facet key."""

    return f"facet:{label}#{facet}"


class _OpReducer:
    """The per-op summary reducer: fused spine + sketch, facets included.

    Declared summary role (P1) so the reduce-only path skips the raw clone
    (P2), and ctx-aware (``_tl_wants_ctx``) so declared facet splitters
    apply at the one site where the raw view exists. The payload is
    ``{"op": packed, "facets": {name: packed}}`` where ``packed`` is the
    15-slot float64 spine concatenated with the int64 sketch counts (cast
    to float64), one row per observed population, decoded after the pass.
    """

    _tl_transform_role = "summary"
    _tl_wants_ctx = True

    def __init__(self, descriptor: Any, facets: Mapping[str, FacetSplitter]) -> None:
        self.descriptor = descriptor
        self.facets = dict(facets)
        #: label -> facet-name row order of the LAST observed split; the
        #: collector reads it at fold time (single-threaded by design).
        self.facet_names: dict[str, tuple[str, ...]] = {}
        #: label -> reason string for the LAST reducer failure (D14).
        self.failures: dict[str, str] = {}

    def _pack(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reduce one detached view to the packed observation vector.

        Stays ON DEVICE: the collector batches ONE host transfer per step
        over all sites (the D16 transfer invariant; never a per-site
        ``.cpu()``).
        """

        spine = spine_vector_fused(tensor)
        sketch = sketch_vector(tensor, self.descriptor).to(torch.float64)
        return torch.cat([spine, sketch])

    def __call__(self, tensor: torch.Tensor, ctx: Any = None) -> torch.Tensor:
        """Reduce one op output (detached view) plus its declared facets.

        Returns a ``[1 + n_facets, W]`` tensor: row 0 the op population,
        the rest the declared facet slices in the recorded name order (a
        plain tensor keeps every fastlog projection surface happy).

        Total and fail-open (D14): a reducer failure on ONE site (an
        unsupported dtype, a raising splitter) returns the empty ``[0, W]``
        sentinel, which the collector folds as a per-site
        ``capture_failed`` observation -- the user's training step always
        survives. The failure reason lands on the observation record.
        """

        label = None
        if ctx is not None:
            label = getattr(ctx, "raw_label", None) or getattr(ctx, "label", None)
        width = SPINE_SLOTS + 2 * self.descriptor.bins_per_side + 8
        try:
            detached = tensor.detach()
            rows = [self._pack(detached)]
            splitter = self.facets.get(label) if label is not None else None
            if splitter is not None and label is not None:
                views = splitter(detached)
                names = tuple(sorted(views))
                self.facet_names[label] = names
                rows.extend(self._pack(views[name]) for name in names)
            return torch.stack(rows)
        except Exception as exc:  # noqa: BLE001 -- D14: the reducer is total and fail-open by contract (a site failure degrades to a disclosed capture_failed observation, never a crashed training step)
            if label is not None:
                self.failures[label] = f"{type(exc).__name__}: {exc}"
            return torch.zeros((0, width), dtype=torch.float64)


class OpTierCollector:
    """Route-B op/facet-grain collector over one model.

    Lifecycle mirrors Route A: ``discover(inputs)`` (one metadata-only
    fastlog pass catalogs the selected op sites and prices the plan) ->
    ``step(global_step)`` brackets each training step, inside which
    ``observe_step(inputs)`` runs the wrapped forward and returns the live
    model output -> read via :class:`HistoryView.from_collector`.

    Parameters
    ----------
    model:
        The model whose ops are observed.
    save:
        Fastlog ``save=`` predicate/selector scoping the op sites (the
        selector-algebra door: ``tl.func(...)``, ``tl.in_module(...)``,
        composed predicates). ``None`` selects every op.
    facets:
        Optional ``{op_label: splitter}`` mapping declaring facet slices
        of specific op sites (keys match fastlog raw labels).
    settings:
        The shared plumbing knobs (descriptor, run identity, ring,
        output_dir); same object Route A takes.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        save: Any = None,
        facets: Mapping[str, FacetSplitter] | None = None,
        settings: WatchSettings | None = None,
    ) -> None:
        settings = settings if settings is not None else WatchSettings()
        self.model = model
        self.save = save
        self.facets = dict(facets or {})
        self.settings = settings
        self.descriptor = settings.descriptor
        self.run_id = settings.run_id or uuid.uuid4().hex
        self.segment_id = settings.segment_id or uuid.uuid4().hex
        self.run = RunRecord(
            run_id=self.run_id,
            segment_id=self.segment_id,
            descriptor=self.descriptor,
            package_versions={"torch": torch.version.__version__},
            selector_plan=repr(save) if save is not None else None,
        )
        ring_capacity = settings.ram_capacity if settings.ram_capacity is not None else 256
        self.ring = RamRing(
            capacity=ring_capacity,
            policy=settings.ram_policy,
            disk_backed=settings.output_dir is not None,
        )
        self.writer = (
            HistoryWriter(settings.output_dir, self.run)
            if settings.output_dir is not None
            else None
        )
        self._sites_by_id: dict[str, SiteRecord] = {}
        self._plan: WatchPlan | None = None
        self._reducer = _OpReducer(self.descriptor, self.facets)
        self._last_step: int | None = None
        self._block_step: int | None = None
        self._block_truth: StepTruth | None = None
        self._block_observations: list[ObservationRecord] | None = None

    @property
    def site_catalog(self) -> dict[str, SiteRecord]:
        """Public read of the discovered site records, keyed by site_id."""

        return dict(self._sites_by_id)

    @property
    def plan(self) -> WatchPlan:
        """The disclosed-cost plan table; refuses before discovery."""

        if self._plan is None:
            raise WatchLifecycleError(
                "No plan yet: run discover(*inputs) first (one metadata-only "
                "fastlog pass that catalogs the selected op sites).",
                code="watch_lifecycle_invalid",
                remedy="Call collector.discover(inputs) before reading the plan.",
            )
        return self._plan

    def discover(self, input_args: Any, input_kwargs: dict[str, Any] | None = None) -> None:
        """Catalog the selected op sites and price the plan (metadata pass).

        Raises
        ------
        WatchPlanError
            ``watch_plan_empty`` when the selection matches no op (a
            warning at step 0 of a ten-hour run is not disclosure).
        """

        import warnings

        from .. import fastlog

        with warnings.catch_warnings():
            # Fastlog's zero-match WARNING is superseded here by the plan-time
            # TYPED refusal below (strictly stronger: a warning at step 0 of a
            # ten-hour run is not disclosure, D6).
            warnings.filterwarnings("ignore", message=".*matched zero sites.*")
            recording = fastlog.record(
                self.model,
                input_args,
                input_kwargs,
                save=self.save,
                default_op=fastlog.CaptureSpec(save_out=False, save_metadata=True),
            )
        rows: list[PlanRow] = []
        for record in recording.records:
            ctx = record.ctx
            if ctx.kind != "op" or not (record.spec.save_out or record.spec.save_metadata):
                continue
            label = ctx.raw_label or ctx.label
            shape = tuple(ctx.shape) if ctx.shape is not None else ()
            numel = 1
            for dim in shape:
                numel *= dim
            site = SiteRecord(
                site_id=_op_site_id(label),
                kind="op",
                display_label=label,
                module_path=ctx.address,
                shape=shape or None,
                dtype=str(ctx.dtype) if ctx.dtype is not None else None,
                numel=numel,
            )
            self._sites_by_id[site.site_id] = site
            rows.append(
                PlanRow(
                    site_id=site.site_id,
                    stream="activation",
                    elements_per_step=numel,
                    bytes_per_step=(SPINE_SLOTS + 2 * self.descriptor.bins_per_side + 8) * 8,
                    cadence=1,
                    sketch_cadence=1,
                    note="op tier (fastlog)",
                )
            )
            if label in self.facets:
                facet_site = SiteRecord(
                    site_id=_facet_site_id(label, "*"),
                    kind="facet",
                    display_label=f"{label}#<facets>",
                    module_path=label,
                )
                # Facet identities materialize at the first observed split;
                # the placeholder row discloses the declared splitter now.
                rows.append(
                    PlanRow(
                        site_id=facet_site.site_id,
                        stream="activation",
                        elements_per_step=numel,
                        bytes_per_step=0,
                        cadence=1,
                        sketch_cadence=1,
                        note="facet splitter declared; identities land at first step",
                    )
                )
        if self.writer is not None:
            for site in self._sites_by_id.values():
                self.writer.add_site(site)
        if not rows:
            raise WatchPlanError(
                "The op-tier selection matched no op site at plan time; "
                "zero-match selectors refuse HERE, never silently at step N.",
                code="watch_plan_empty",
                remedy="Widen save= (or pass save=None for every op).",
            )
        self._plan = WatchPlan(
            rows=tuple(rows),
            total_elements_per_step=sum(row.elements_per_step for row in rows),
            total_bytes_per_step=sum(row.bytes_per_step for row in rows),
        )

    @contextmanager
    def step(
        self,
        global_step: int,
        *,
        truth: StepTruth | None = None,
        new_segment: bool = False,
    ) -> Iterator[None]:
        """Bracket one training step; commits the StepBlock atomically."""

        if self._plan is None:
            raise WatchLifecycleError(
                "step() before discover(): the op-tier plan (and its "
                "disclosed cost) must exist before observations land.",
                code="watch_lifecycle_invalid",
                remedy="Call collector.discover(inputs) first.",
            )
        if self._block_observations is not None:
            raise WatchLifecycleError(
                "A step transaction is already open; steps never nest.",
                code="watch_step_conflict",
                remedy="Close the open step before starting the next.",
            )
        if new_segment:
            self.segment_id = uuid.uuid4().hex
            self._last_step = None
        validate_step_order(self._last_step, global_step, same_segment=True)
        self.ring.will_admit()
        self._block_step = global_step
        self._block_truth = truth if truth is not None else StepTruth()
        self._block_observations = []
        try:
            yield
        finally:
            self._commit_block()

    def observe_step(
        self,
        input_args: Any,
        input_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Run the wrapped forward, fold op observations, return live output.

        The model's real output is returned with its autograd graph intact:
        the training loop calls this INSTEAD of ``model(x)``, so the tier
        costs one wrapped forward per logged step, never a second forward.
        """

        if self._block_observations is None:
            raise WatchLifecycleError(
                "observe_step() outside a step() bracket: observations need "
                "their atomic StepBlock coordinate.",
                code="watch_lifecycle_invalid",
                remedy="Wrap the training step in collector.step(global_step).",
            )
        from .. import fastlog

        output, recording = fastlog.record(
            self.model,
            input_args,
            input_kwargs,
            save=self.save,
            default_op=True,
            activation_transform=self._reducer,
            save_raw_activations=False,
            return_output=True,
        )
        step = self._open_block_step()
        # Batch the host transfer (D16): one stacked .cpu() per device per
        # step over every packed row, never a per-site sync.
        pending: list[tuple[str, Any, str | None, torch.Tensor]] = []
        observations = self._open_block_observations()
        for record in recording.records:
            ctx = record.ctx
            payload = record.transformed_ram_payload
            if ctx.kind != "op" or not isinstance(payload, torch.Tensor) or payload.dim() != 2:
                continue
            label = ctx.raw_label or ctx.label
            if payload.shape[0] == 0:
                # D14 fail-open sentinel: the reducer failed on this site;
                # record capture_failed (with the reason) and move on -- the
                # training step already survived.
                site_id = _op_site_id(label)
                self._ensure_site(step, site_id, label=label, ctx=ctx)
                observations.append(
                    ObservationRecord(
                        global_step=step,
                        site_id=site_id,
                        stream="activation",
                        phase="forward",
                        presence="capture_failed",
                        reason=self._reducer.failures.get(label),
                    )
                )
                continue
            pending.append((label, ctx, None, payload[0]))
            names = self._reducer.facet_names.get(label, ())
            for row_index, facet_name in enumerate(names, start=1):
                if row_index < payload.shape[0]:
                    pending.append((label, ctx, facet_name, payload[row_index]))
        self._land_and_fold(step, pending)
        return output

    def _land_and_fold(
        self,
        step: int,
        pending: list[tuple[str, Any, str | None, torch.Tensor]],
    ) -> None:
        """Land pending rows with ONE stacked ``.cpu()`` per device (D16), then fold."""

        if not pending:
            return
        by_device: dict[torch.device, list[int]] = {}
        for index, (_label, _ctx, _facet, vector) in enumerate(pending):
            by_device.setdefault(vector.device, []).append(index)
        landed: dict[int, torch.Tensor] = {}
        for device_indexes in by_device.values():
            stacked = torch.stack([pending[i][3] for i in device_indexes]).cpu()
            for row, index in enumerate(device_indexes):
                landed[index] = stacked[row]
        for index, (fold_label, fold_ctx, fold_facet, _vector) in enumerate(pending):
            self._fold_row(step, fold_label, fold_ctx, fold_facet, landed[index])

    def _open_block_step(self) -> int:
        """Return the open bracket's step; refuse typed outside a bracket."""

        if self._block_step is None:
            raise WatchLifecycleError(
                "No step transaction is open; observations need their atomic StepBlock coordinate.",
                code="watch_lifecycle_invalid",
                remedy="Wrap the training step in collector.step(global_step).",
            )
        return self._block_step

    def _open_block_observations(self) -> list[ObservationRecord]:
        """Return the open bracket's observation list; refuse typed outside."""

        if self._block_observations is None:
            raise WatchLifecycleError(
                "No step transaction is open; observations need their atomic StepBlock coordinate.",
                code="watch_lifecycle_invalid",
                remedy="Wrap the training step in collector.step(global_step).",
            )
        return self._block_observations

    def _ensure_site(
        self, step: int, site_id: str, *, label: str, ctx: Any, parent: str | None = None
    ) -> None:
        """Register one site row on first sight (D5: drift adds rows).

        The kind is derived: a site with a ``parent`` op is a ``facet``
        site, a parentless one an ``op`` site.
        """

        if site_id in self._sites_by_id:
            return
        kind = "op" if parent is None else "facet"
        shape = tuple(ctx.shape) if getattr(ctx, "shape", None) is not None else None
        self._sites_by_id[site_id] = SiteRecord(
            site_id=site_id,
            kind=kind,
            display_label=label,
            module_path=parent if parent is not None else getattr(ctx, "address", None),
            shape=shape if kind == "op" else None,
            first_step=step,
        )
        if self.writer is not None:
            self.writer.add_site(self._sites_by_id[site_id])

    def _decode(self, packed: torch.Tensor) -> tuple[Any, Any]:
        """Unpack one payload vector into (SpineResult, HistogramResult)."""

        spine = spine_result_from_vector(packed[:SPINE_SLOTS], "float64")
        sketch = histogram_result_from_vector(packed[SPINE_SLOTS:].to(torch.int64), self.descriptor)
        return spine, sketch

    def _fold_row(
        self,
        step: int,
        label: str,
        ctx: Any,
        facet_name: str | None,
        packed: torch.Tensor,
    ) -> None:
        """Decode one landed vector into its op or facet observation record."""

        observations = self._open_block_observations()
        if facet_name is None:
            site_id = _op_site_id(label)
            self._ensure_site(step, site_id, label=label, ctx=ctx)
        else:
            site_id = _facet_site_id(label, facet_name)
            self._ensure_site(
                step,
                site_id,
                label=f"{label}#{facet_name}",
                ctx=ctx,
                parent=label,
            )
        spine, sketch = self._decode(packed)
        observations.append(
            ObservationRecord(
                global_step=step,
                site_id=site_id,
                stream="activation",
                phase="forward",
                presence="observed",
                spine=spine,
                sketch=sketch,
            )
        )

    def _commit_block(self) -> None:
        """Commit the bracketed step's observations atomically."""

        from ._artifact import CommittedBlock

        observations = self._block_observations
        truth = self._block_truth
        step = self._block_step
        self._block_observations = None
        self._block_truth = None
        self._block_step = None
        if observations is None or truth is None or step is None:
            return
        block = StepBlockRecord(
            segment_id=self.segment_id,
            global_step=step,
            provenance="explicit",
            optimizer_status="applied" if truth.applied else "skipped",
            scale=truth.scale,
            unscaled=truth.unscaled,
            clipped=truth.clipped,
            micro_batches=truth.micro_batches,
        )
        committed = CommittedBlock(
            block=block,
            observations=tuple(observations),
            step_lo=step,
            step_hi=step,
        )
        self.ring.admit(committed)
        if self.writer is not None:
            self.writer.append_block(committed)
        self._last_step = step


def split_heads(n_heads: int, dim: int = -2) -> FacetSplitter:
    """A convenience facet splitter: slice an attention-shaped tensor by head.

    Assumes the head axis is ``dim`` (default -2 fits ``[B, H, S, D]`` at
    ``dim=1``... pass the axis explicitly for your layout). Returns views;
    the reducer copies nothing.

    Raises
    ------
    WatchRenderError
        ``watch_facet_shape_invalid`` when the named axis does not carry
        ``n_heads`` slices -- a silently mis-split head axis would relabel
        every head series.
    """

    def splitter(tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        """Slice one tensor into per-head views along the declared axis."""

        axis = dim if dim >= 0 else tensor.dim() + dim
        if axis < 0 or axis >= tensor.dim() or tensor.shape[axis] != n_heads:
            raise WatchRenderError(
                f"head axis {dim} of shape {tuple(tensor.shape)} does not "
                f"carry {n_heads} slices; a mis-split axis would relabel "
                "every head series.",
                code="watch_facet_shape_invalid",
                shape=tuple(tensor.shape),
                n_heads=n_heads,
                remedy="Pass the correct dim= for your attention layout.",
            )
        return {f"head{i}": tensor.select(axis, i) for i in range(n_heads)}

    return splitter


__all__ = ["FacetSplitter", "OpTierCollector", "split_heads"]
