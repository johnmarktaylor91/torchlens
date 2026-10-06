"""The tier-P/M watch engine over the C06 collector (memo build item 5-6).

``watch(model, to=sink, ...)`` attaches ONCE with disclosed cost tiers:

- **Tier P (default)** -- parameters / gradients / updates through the C06
  optimizer-boundary hooks. Wraps NO forward, runs NO discovery forward:
  parameter sites are catalogued directly, so the default tier never
  executes the user's model at all.
- **Tier M** -- module-output activations through ordinary forward hooks
  (the C06 Route-A collector). Requires an explicit selector AND
  ``example_input=`` (the one real attach-time forward, run in the model's
  CURRENT mode -- never the shipped Lightning callback's eval()/no_grad()
  re-trace flaw); zero matches refuse at attach with the live module census.
- **Tier O (op grain)** -- SHED this queue window: requesting it refuses
  typed with the honest cost teaching (op capture is wrap-dominated;
  ``halt=`` is the time lever, not ``save=``) and the explicit
  ``tl.record``/fastlog recipe as the interim spelling.

Step law (memo 3.7): the caller's ``global_step`` is mandatory; there is no
hidden counter, ever. Sources: the ``with watch.step(n):`` scope
(authoritative), ``step=callable`` at attach for unchanged loops, or the
framework callbacks. An optimizer boundary reached with NO step source
refuses typed naming all three.

Enablement law (memo 3.16): watching is a visible call. The kill switch
``TORCHLENS_WATCH_DISABLE=1`` only ever turns things OFF, and a disabled
watcher still writes its ``torchlens/run`` rows saying it was disabled and
by what.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace as _dc_replace
from typing import Any

import torch

from .. import _state
from ..observability import (
    CommittedBlock,
    EventStream,
    HistogramDescriptor,
    HistoryCollector,
    ObserverEvent,
    WatchSettings,
)
from ..observability._collector import StepTruth
from ..observability._kernels import DEFAULT_DESCRIPTOR, spine_result_from_vector, spine_vector
from ..observability._schema import validate_step_order
from ..utils.env_flags import closed_bool_env
from ._amp import correct_spine, corrected_gradient_block, observed_grad_scale, unscale_stage
from ._errors import SinkProtocolError, TrackersError, WatchConfigError, WatchRuntimeError
from ._protocol import EmissionLedger, require_capabilities, sink_name
from ._records import (
    ScalarPoint,
    StepEmission,
    TagGrammar,
    TextPoint,
    build_manifest,
    emission_from_block,
    spine_scalars,
)

__tl_layer__ = "L8"

#: The kill switch (off-only; registered in torchlens.utils.env_flags).
WATCH_DISABLE_ENV = "TORCHLENS_WATCH_DISABLE"

#: Closed signal vocabulary -> C06 stream (or a typed deferral).
SIGNALS = ("parameters", "gradients", "updates", "activations", "activation_grads")

_SIGNAL_STREAMS = {
    "parameters": "param",
    "gradients": "param_grad",
    "updates": "param_delta",
    "activations": "activation",
}

#: Cadence sentinel meaning "never scheduled" for the collector's
#: ``step % cadence`` test. Steps are validated nonnegative and far below
#: this, so only a forced cadence of 1 ever samples.
_NEVER = 2**62

#: Default scalar cadence (memo 3.8, majority; the G1 gate may tune the
#: constant, never the first/last rule).
DEFAULT_EVERY = 50

#: Live (unclosed) sessions per model, keyed by the tag-grammar components
#: ``(name, namespace)``. A second ``watch()`` on the same model under the
#: same grammar would install a second hook set and emit every point TWICE
#: into a shared sink (AUD-CODE 2.14); it refuses typed, while distinct
#: ``name=`` values keep the documented multi-session spelling.
_ACTIVE_SESSIONS: weakref.WeakKeyDictionary[
    torch.nn.Module, dict[tuple[str | None, str | None], WatchSession]
] = weakref.WeakKeyDictionary()

#: Boundary scale evidence -> the StepBlock ``unscaled`` tri-state (the C06
#: record keeps the numbers it measured; "no" = still scaled, corrected at
#: emission). Anything else is "unknown", never a default.
_EVIDENCE_TO_UNSCALED = {"unscaled_observed": "yes", "unscaled_derived": "no"}

#: Boundary scale evidence -> the per-observation ``grad_scale`` stamp (D7).
_EVIDENCE_TO_GRAD_SCALE = {"unscaled_observed": "unscaled", "unscaled_derived": "scaled"}


def _module_census(model: torch.nn.Module) -> dict[str, int]:
    """Count leaf-module classes: the teaching payload for zero-match errors."""

    census: dict[str, int] = {}
    for _name, module in model.named_modules():
        if next(module.children(), None) is None:
            key = type(module).__name__
            census[key] = census.get(key, 0) + 1
    return census


def _foreign_watch_prefixes(model: torch.nn.Module) -> tuple[str, ...]:
    """Detect a foreign watcher's hooks on this model (G7).

    ``wandb.watch`` installs forward/backward hooks whose callables live in
    the ``wandb`` package; the scan is vendor-agnostic (any non-torchlens
    tracker package claiming hooks) but currently names only the prefixes it
    can prove.
    """

    owners: set[str] = set()
    for module in model.modules():
        hook_maps = (
            getattr(module, "_forward_hooks", {}),
            getattr(module, "_forward_pre_hooks", {}),
            getattr(module, "_backward_hooks", {}),
        )
        for hooks in hook_maps:
            for hook in hooks.values():
                owner = getattr(hook, "__module__", "") or ""
                if owner.split(".", 1)[0] == "wandb":
                    owners.add("wandb")
    if "wandb" in owners:
        return ("gradients/", "parameters/")
    return ()


@dataclass
class CloseReport:
    """The close report: every scheduled observation ends as data, an
    enumerated presence/skip outcome, or an exception (memo 3.13)."""

    requested_signals: tuple[str, ...]
    matched_sites: int
    sampled_steps: tuple[int, ...]
    first_step: int | None
    last_step: int | None
    final_sample_forced: bool
    computed_scalars: int
    computed_histograms: int
    dropped_blocks: int
    sink_rows: tuple[dict[str, Any], ...]
    missing_phases: tuple[str, ...]
    named_skips: tuple[str, ...]
    disabled: bool = False

    def format(self) -> str:
        """Render the report as plain ASCII lines."""

        lines = [
            f"signals={','.join(self.requested_signals)} sites={self.matched_sites}",
            f"steps sampled={len(self.sampled_steps)} "
            f"first={self.first_step} last={self.last_step} "
            f"final_sample_forced={self.final_sample_forced}",
            f"computed scalars={self.computed_scalars} "
            f"histograms={self.computed_histograms} dropped_blocks={self.dropped_blocks}",
        ]
        for row in self.sink_rows:
            lines.append(
                f"sink {row['sink']}: scalars={row['emitted_scalars']} "
                f"histograms={row['emitted_histograms']} texts={row['emitted_texts']} "
                f"failed={row['failed']} relay={row['relay_state']}"
            )
        if self.missing_phases:
            lines.append("missing phases: " + ", ".join(self.missing_phases))
        for skip in self.named_skips:
            lines.append(f"skip: {skip}")
        if self.disabled:
            lines.append(f"DISABLED by {WATCH_DISABLE_ENV}")
        return "\n".join(lines)


#: The honest cost teaching for op-grain deferral refusals (memo 3.9): op
#: capture is WRAP-dominated, so save= is not the lever -- halt= is.
_TIER_O_COST_TEACHING = (
    "Honest cost context before you ask for it: op capture is WRAP-dominated "
    "(~34x forward on the repo's own gpt2 CUDA rows; saving nothing still "
    "costs ~35x), narrowing save= moves only ~14% of the bill, and halt= is "
    "the lever that actually cuts time (10.4x vs 35.0x)."
)

_TIER_O_REMEDY = (
    "Use module grain (signals=('activations',), select=...), or run an "
    "explicit sampled-step capture: tl.record(model, inputs, "
    "save=<predicate>) on the steps you care about."
)


@dataclass(frozen=True)
class _SessionConfig:
    """The session-plumbing bundle (the C06 WatchSettings precedent):
    scheduling, step sourcing, phase policy, and the event seam travel as
    one object so the session constructor stays flat-and-small."""

    signals: tuple[str, ...]
    every: int
    hist_every: int | None = None
    optimizer: torch.optim.Optimizer | None = None
    step_source: Callable[[], int] | None = None
    allow_missing_phases: bool = False
    event_stream: EventStream | None = None
    disabled_by: str | None = None


class WatchSession:
    """One attached watch: step scopes, emission, close report.

    Construct through :func:`watch`. The session is a context manager;
    ``close()`` runs on every exit path and detaches every hook.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        sinks: tuple[Any, ...],
        collector: HistoryCollector | None,
        grammar: TagGrammar,
        config: _SessionConfig,
    ) -> None:
        """Bind the attached collector and the emission plumbing."""

        self.model = model
        self.sinks = sinks
        self.collector = collector
        self.grammar = grammar
        self.signals = config.signals
        self.every = max(1, int(config.every))
        self.hist_every = config.hist_every
        self.optimizer = config.optimizer
        self.step_source = config.step_source
        self.allow_missing_phases = config.allow_missing_phases
        self.event_stream = config.event_stream
        self.disabled_by = config.disabled_by
        self.ledger = EmissionLedger()
        self._drained_blocks = 0
        self._sampled_steps: list[int] = []
        self._seen_steps: list[int] = []
        self._closed = False
        self._optimizer_fired_this_step = False
        self._optimizer_ever_fired = False
        self._scale_at_boundary: float | None = None
        self._entry_scale: float | None = None
        self._scale_evidence = "unavailable"
        self._current_scaler: Any = None
        self._step_open = False
        self._named_skips: list[str] = []
        self._own_optimizer_handle: Any = None
        self._post_handles: list[Any] = []
        self._manifest_emitted = False
        self._forward_handle: Any = None
        self._deferred_attach: list[ScalarPoint | TextPoint] = []
        self._pending_events: list[ObserverEvent] = []
        if self.event_stream is not None:
            # Subscribe a METHOD, never the bound ``append`` of the initial
            # list: the old subscription kept feeding the original list after
            # the first drain rebound ``_pending_events`` to a fresh one, so
            # every verdict after the first drain was orphaned (AUD-CODE 2.14c).
            self.event_stream.subscribe(self._receive_event)

    # -- emission plumbing -------------------------------------------------

    def _emit(self, emission: StepEmission) -> None:
        """Fan one step emission out to every sink; failures latch loudly."""

        self.ledger.computed_scalars += len(emission.scalars)
        self.ledger.computed_histograms += len(emission.histograms)
        for sink in self.sinks:
            row = self.ledger.row(sink)
            if row.failed:
                continue
            try:
                for scalar in emission.scalars:
                    sink.emit_scalar(scalar)
                    row.emitted_scalars += 1
                    if self.grammar.is_data_tag(scalar.tag):
                        row.emitted_data_points += 1
                for histogram in emission.histograms:
                    sink.emit_histogram(histogram)
                    row.emitted_histograms += 1
                    if self.grammar.is_data_tag(histogram.tag):
                        row.emitted_data_points += 1
                for text in emission.texts:
                    sink.emit_text(text)
                    row.emitted_texts += 1
            except Exception as exc:  # noqa: BLE001 - latched + reported, never silent
                row.failed = True
                row.failure = f"{type(exc).__name__}: {exc}"
                self._named_skips.append(
                    f"sink {row.name} latched failed at step {emission.step}: {row.failure}"
                )

    def _stage_attach_rows(self) -> None:
        """Build the manifest (memo 3.5) + heartbeat rows; emission is deferred.

        Both used to land at step 0 the moment ``watch()`` returned. A resumed
        wandb run (already at step N) DROPS every ``log(step=0)`` call, so the
        rows saying "this run is being watched" were exactly the ones that
        vanished (AUD-CODE 2.14). They now ride the first step the session
        actually sees (``_emit_attach_rows``); ``close()`` emits them at the
        last seen step (or 0 when no step ever ran) so no run ends without
        them. Emission order is preserved: they precede the first data row.
        """

        if self._manifest_emitted or self.collector is None:
            return
        manifest = build_manifest(
            self.collector.run,
            dict(self.collector._sites_by_id),
            self.grammar,
            grains={site_id: site.kind for site_id, site in self.collector._sites_by_id.items()},
        )
        heartbeat = ScalarPoint(self.grammar.run_health("attached"), 0, 1.0)
        self._deferred_attach = [manifest, heartbeat]

    def _emit_attach_rows(self, step: int) -> None:
        """Emit the staged manifest + heartbeat exactly once, at ``step``."""

        if self._manifest_emitted or not self._deferred_attach:
            return
        self._manifest_emitted = True
        rows = [_dc_replace(point, step=step) for point in self._deferred_attach]
        self._deferred_attach = []
        texts = tuple(point for point in rows if isinstance(point, TextPoint))
        scalars = tuple(point for point in rows if isinstance(point, ScalarPoint))
        self._emit(StepEmission(step=step, scalars=scalars, histograms=(), texts=texts))

    def _drain(self) -> None:
        """Emit every newly committed block since the last drain."""

        if self.collector is None:
            return
        blocks = self.collector.ring.blocks
        fresh = blocks[self._drained_blocks :]
        self._drained_blocks = len(blocks)
        sites = dict(self.collector._sites_by_id)
        for block in fresh:
            step = block.block.global_step
            self._emit_attach_rows(step)
            block, derived = self._unscale_derived(block)
            emission = emission_from_block(block, sites, self.grammar)
            if derived:
                disclosure = ScalarPoint(self.grammar.run_health("amp_unscale_derived"), step, 1.0)
                emission = _dc_replace(emission, scalars=emission.scalars + (disclosure,))
            self._emit(emission)
            self._drain_events(block)

    def _unscale_derived(self, block: CommittedBlock) -> tuple[CommittedBlock, int]:
        """Correct a block reduced from SCALED grads in closed form (memo 3.11).

        The boundary stamped ``unscaled="no"`` plus the observed power-of-two
        scale when the scaler had NOT unscaled this optimizer (a plain
        ``opt.step()`` after ``scaler.scale(loss).backward()``). The ring
        record keeps the scaled numbers it measured; the EMISSION carries the
        corrected ones (evidence ``unscaled_derived``, disclosed on
        ``torchlens/run/amp_unscale_derived``). Scaled numbers used to ship
        stamped ``unscaled='yes'`` (AUD-CODE 2.14b).
        """

        truth = block.block
        if truth.unscaled != "no" or truth.scale is None or truth.scale <= 0.0:
            return (block, 0)
        return corrected_gradient_block(block, float(truth.scale))

    def _drain_events(self, block: CommittedBlock) -> None:
        """Serialize check findings from the shared event stream (chassis).

        Trackers own serialization; checks own severity. The session
        subscribes at construction and buffers; verdicts serialize at the
        step they were stamped with (falling back to the committing block's
        step -- a hook-only event has no honest x-coordinate of its own).
        """

        if self.event_stream is None:
            return
        self._flush_pending_events(block.block.global_step)

    def _receive_event(self, event: ObserverEvent) -> None:
        """EventStream subscriber: buffer until the next drain."""

        self._pending_events.append(event)

    def _flush_pending_events(self, fallback_step: int) -> None:
        """Serialize every buffered verdict; the buffer is drained IN PLACE."""

        pending = list(self._pending_events)
        del self._pending_events[:]
        for event in pending:
            if event.kind == "verdict" and event.verdict is not None:
                value = {"pass": 0.0, "warn": 1.0, "fail": 2.0}.get(event.verdict)
                if value is not None:
                    step = (
                        event.accepted_step_id
                        if event.accepted_step_id is not None
                        else fallback_step
                    )
                    self._emit_check(event.key, value, step)

    def _emit_check(self, name: str, value: float, step: int) -> None:
        """Emit one check outcome scalar (0 pass / 1 warn / 2 fail)."""

        self._emit_attach_rows(step)
        point = ScalarPoint(self.grammar.check(name), step, value)
        self._emit(StepEmission(step=step, scalars=(point,), histograms=()))

    def emit_check(self, name: str, outcome: int, step: int) -> None:
        """The explicit checks-finding door (memo build item 5).

        ``outcome`` is the closed 0/1/2 vocabulary (pass/warn/fail); a
        failure's retained-trace artifact link rides the close report, not
        the scalar.
        """

        if outcome not in (0, 1, 2):
            raise WatchRuntimeError(
                f"check outcome {outcome!r} is not in the closed 0/1/2 "
                "(pass/warn/fail) vocabulary; dashboards threshold on these "
                "exact values.",
                code="watch_check_outcome_invalid",
                outcome=outcome,
                remedy="Pass 0 (pass), 1 (warn), or 2 (fail).",
            )
        self._emit_check(name, float(outcome), step)

    # -- scheduling ---------------------------------------------------------

    def _scheduled(self, step: int) -> bool:
        """First eligible step always samples; then the caller-axis cadence."""

        return not self._sampled_steps or step % self.every == 0

    def _apply_cadence(self, step: int, sampled: bool) -> None:
        """Force or mute the collector's per-stream cadence for this block.

        The collector consults ``cadences`` live at each observation site;
        watch owns scheduling (first/last law), so blocks are armed with
        cadence 1 when watch schedules them and an unreachable cadence
        otherwise. Nonnegative caller steps are far below the sentinel, so
        the sentinel never accidentally samples.
        """

        if self.collector is None:
            return
        value = 1 if sampled else _NEVER
        for stream in list(self.collector.cadences):
            self.collector.cadences[stream] = value
        hist = self.hist_every
        sketch_scheduled = (
            sampled and hist is not None and (not self._sampled_steps or step % hist == 0)
        )
        sketch_value = 1 if sketch_scheduled else _NEVER
        for stream in list(self.collector.sketch_cadences):
            self.collector.sketch_cadences[stream] = sketch_value

    # -- the optimizer boundary --------------------------------------------

    def _boundary_pre_hook(self, optimizer: Any, args: Any, kwargs: Any) -> None:
        """Registered FIRST: step-source law + AMP scale read (memo 3.7/3.11).

        Fires inside ``scaler.step(optimizer)`` after the scaler's own
        unscale and BEFORE ``scaler.update()`` -- the one moment the scale
        read is exact (update halves it on overflow steps).
        """

        del args, kwargs
        self._optimizer_fired_this_step = True
        self._optimizer_ever_fired = True
        if self._current_scaler is not None:
            self._read_boundary_scale(optimizer)
        if self._step_open or self.collector is None:
            return
        if self.step_source is None:
            raise WatchRuntimeError(
                "An optimizer step ran with NO step source: there is no "
                "hidden step counter, ever (a checkpoint resume with an "
                "internal counter restarting at 0 silently misaligns every "
                "panel).",
                code="watch_step_source_missing",
                remedy=(
                    "Provide the caller's step one of three ways: wrap the "
                    "step in `with watch.step(global_step):`, pass "
                    "step=<callable> at watch(), or drive the watcher from a "
                    "framework callback."
                ),
            )
        step_value = int(self.step_source())
        sampled = self._scheduled(step_value)
        self._apply_cadence(step_value, sampled)
        validate_step_order(self.collector._last_step, step_value, same_segment=True)
        self._open_implicit_block(step_value)
        self._note_step(step_value, sampled)

    def _read_boundary_scale(self, optimizer: Any) -> None:
        """Read the scale AND whether the grads at this boundary are unscaled.

        ``scaler.step(optimizer)`` unscales in place before ``optimizer.step()``
        (evidence ``unscaled_observed``); a plain ``optimizer.step()`` after
        ``scaler.scale(loss).backward()`` leaves the factor on every grad, so
        the reduction is corrected in closed form at emission (evidence
        ``unscaled_derived``); an unreadable stage is ``scaled_unknown_factor``
        and the record says so. A readable ``get_scale()`` alone used to be
        taken as proof of unscaling (AUD-CODE 2.14b).
        """

        scale, evidence = observed_grad_scale(self._current_scaler)
        self._scale_at_boundary = scale
        if scale is None:
            self._scale_evidence = "unavailable"
            return
        if evidence == "unscaled_observed":
            # A disabled scaler: factor 1.0 is an observed fact.
            self._scale_evidence = evidence
            return
        stage = unscale_stage(self._current_scaler, optimizer)
        self._scale_evidence = {
            "unscaled": "unscaled_observed",
            "scaled": "unscaled_derived",
        }.get(stage, "scaled_unknown_factor")

    def _open_implicit_block(self, step_value: int) -> None:
        """Open the implicit block WITHOUT discarding the forward's observations.

        Under ``step=callable`` the module forward hooks fill the collector's
        staging buffers BEFORE ``optimizer.step()`` reaches this boundary, and
        ``_open_block`` zeroes those buffers as part of opening -- so tier M
        emitted ZERO activations on the documented ``step=`` spelling and never
        refused (AUD-CODE 2.14a). The pending forward staging is carried across
        the open here; an open that ADOPTS pending staging on the collector
        side is the durable home (filed).
        """

        collector = self.collector
        if collector is None:
            return
        staging = collector._staging
        touched = list(collector._staging_touched)
        pending = staging.clone() if staging is not None and any(touched) else None
        sketch = collector._sketch_staging
        pending_sketch = sketch.clone() if pending is not None and sketch is not None else None
        overflow = list(collector._staging_overflow)
        collector._open_block(step_value, provenance="implicit")
        if pending is not None and collector._staging is not None:
            collector._staging.copy_(pending)
            collector._staging_touched = touched
            if pending_sketch is not None and collector._sketch_staging is not None:
                collector._sketch_staging.copy_(pending_sketch)
        if overflow:
            collector._staging_overflow.extend(overflow)

    def _forward_entry_hook(self, module: Any, args: Any) -> None:
        """Tier M under ``step=callable``: schedule the forward's observations.

        The collector's module hooks consult the per-stream cadence AT FORWARD
        TIME, but the boundary hook (which reads the step and decides
        sampling) fires inside ``optimizer.step()`` afterwards -- so without
        this hook a step's activations were gathered under the PREVIOUS step's
        decision. Registered only when both ``step=`` and ``activations`` are
        requested; the explicit ``watch.step`` scope already schedules at entry.
        """

        del module, args
        if self._step_open or self.collector is None or self.step_source is None:
            return
        if self.collector._block_open:
            return
        step_value = int(self.step_source())
        self._apply_cadence(step_value, self._scheduled(step_value))

    def _boundary_post_hook(self, optimizer: Any, args: Any, kwargs: Any) -> None:
        """Registered LAST: drain freshly committed implicit blocks."""

        del optimizer, args, kwargs
        if not self._step_open:
            self._drain()

    def _note_step(self, step: int, sampled: bool) -> None:
        """Track seen/sampled step lists for the close report."""

        self._seen_steps.append(step)
        if sampled:
            self._sampled_steps.append(step)

    # -- the explicit step scope ---------------------------------------------

    @contextmanager
    def step(
        self,
        global_step: int,
        *,
        scaler: Any = None,
        new_segment: bool = False,
        micro_batches: int | None = None,
    ) -> Iterator[None]:
        """The authoritative step transaction (memo 3.7).

        Brackets one training step on the CALLER's axis; also the home of
        AMP scale stamping and skipped-step truth. Duplicate or decreasing
        steps refuse unless ``new_segment=True`` declares a resume.
        """

        if self.collector is None:
            yield
            return
        if self._step_open:
            raise WatchRuntimeError(
                "watch.step scopes never nest: one step axis, ever.",
                code="watch_step_conflict",
                remedy="Close the open step scope before opening the next.",
            )
        step_value = int(global_step)
        if step_value < 0:
            raise WatchRuntimeError(
                f"global_step {step_value} is negative; the step axis is the "
                "caller's own nonnegative training coordinate.",
                code="watch_step_invalid",
                step=step_value,
                remedy="Pass the loop's real nonnegative global step.",
            )
        sampled = self._scheduled(step_value)
        self._apply_cadence(step_value, sampled)
        self._current_scaler = scaler
        self._optimizer_fired_this_step = False
        self._scale_at_boundary = None
        self._scale_evidence = "unavailable"
        self._entry_scale, _ = observed_grad_scale(scaler)
        transaction = self.collector.step(
            step_value, truth=StepTruth(micro_batches=micro_batches), new_segment=new_segment
        )
        # A refused open (duplicate/decreasing step) must not wedge the
        # session: the scope only counts as open once the collector accepted.
        transaction.__enter__()
        self._step_open = True
        failure: BaseException | None = None
        try:
            yield
        except BaseException as exc:
            failure = exc
            raise
        finally:
            self._settle_step(step_value, sampled, scaler, transaction, failure)

    def _settle_step(
        self,
        step_value: int,
        sampled: bool,
        scaler: Any,
        transaction: Any,
        failure: BaseException | None,
    ) -> None:
        """Inject observed truth, commit the block, drain, enforce phases.

        The entry-time scale reading rides ``self._entry_scale`` (set by
        ``step()`` before the body ran).
        """

        truth_updates: dict[str, Any] = {}
        skipped = False
        if scaler is not None:
            exit_scale, _ = observed_grad_scale(scaler)
            entry_scale = self._entry_scale
            skipped = (
                not self._optimizer_fired_this_step
                and entry_scale is not None
                and exit_scale is not None
                and exit_scale < entry_scale
            )
            scale = self._scale_at_boundary if self._scale_at_boundary is not None else exit_scale
            truth_updates["scale"] = scale
            truth_updates["unscaled"] = _EVIDENCE_TO_UNSCALED.get(self._scale_evidence, "unknown")
        if skipped or not self._optimizer_fired_this_step:
            truth_updates["optimizer_status"] = "skipped" if skipped else "unknown"
        if self.collector is not None and self.collector._block_open:
            self.collector._block_truth.update(truth_updates)
            if scaler is not None:
                # D7: every gradient observation carries the boundary's scale
                # truth, never a default; the emitter corrects "scaled" rows.
                stamp = _EVIDENCE_TO_GRAD_SCALE.get(self._scale_evidence, "unknown")
                for key in list(self.collector._block_grad_scale):
                    self.collector._block_grad_scale[key] = stamp
        self._step_open = False
        transaction.__exit__(
            type(failure) if failure is not None else None,
            failure,
            failure.__traceback__ if failure is not None else None,
        )
        self._note_step(step_value, sampled)
        if skipped:
            self._named_skips.append(
                f"amp_skipped: step {step_value} (scale decrement observed; no fake update emitted)"
            )
        try:
            self._drain()
        except TrackersError as exc:
            if failure is None:
                raise
            # ``finally`` context: a typed drain refusal (tag safety, scale
            # shift) must never MASK the training loop's own exception; it is
            # recorded and the user's exception keeps propagating.
            self._named_skips.append(
                f"drain refused at step {step_value} while the loop's own exception "
                f"unwound: {exc.fields.get('code')}"
            )
        self._enforce_phases(step_value, failure, skipped)

    def _enforce_phases(
        self,
        step_value: int,
        failure: BaseException | None,
        skipped: bool,
    ) -> None:
        """Requested-phase-absent refusal at step close (demotable).

        An AMP-skipped step is the scaler working, never a phase failure.
        """

        needs_boundary = bool({"gradients", "updates", "parameters"} & set(self.signals))
        if (
            failure is None
            and needs_boundary
            and not self._optimizer_fired_this_step
            and not skipped
        ):
            if self.allow_missing_phases:
                self._named_skips.append(
                    f"phase_missing: no optimizer step inside step scope "
                    f"{step_value} (demoted by allow_missing_phases=True)"
                )
            else:
                raise WatchRuntimeError(
                    f"Step scope {step_value} closed without the optimizer "
                    f"boundary the requested signals "
                    f"{sorted({'gradients', 'updates', 'parameters'} & set(self.signals))} "
                    "need; the series would be silently absent for this step.",
                    code="watch_phase_missing",
                    step=step_value,
                    remedy=(
                        "Call optimizer.step() (or scaler.step(optimizer)) "
                        "inside the scope, or pass "
                        "allow_missing_phases=True for mixed train/eval "
                        "loops (recorded as a named skip)."
                    ),
                )

    # -- lifecycle -----------------------------------------------------------

    def plan_report(self) -> str:
        """The attach-time plan: sites, both cost columns, cadences, tiers."""

        if self.collector is None:
            return f"watch DISABLED by {self.disabled_by}"
        plan = self.collector.plan
        lines = [
            plan.format_table(),
            f"signals={','.join(self.signals)} every={self.every}"
            + (f" hist_every={self.hist_every}" if self.hist_every else ""),
            "tier P wraps no forward; tier M uses ordinary module hooks; "
            "first and last eligible steps always sample",
        ]
        return "\n".join(lines)

    def _final_forced_sample(self) -> None:
        """Force the LAST-step sample at close when it was not scheduled.

        Reduces parameters (and any still-present gradients) directly and
        emits at the last seen step -- so no correct short run ends
        one-point (memo 3.8). Updates cannot be re-derived here (they need
        the pre/post boundary pair) and are not fabricated.
        """

        if self.collector is None or not self._seen_steps:
            return
        last = self._seen_steps[-1]
        if self._sampled_steps and self._sampled_steps[-1] == last:
            return
        include_gradients, derive_scale = self._forced_gradient_policy(last)
        scalars = self._forced_sample_scalars(last, include_gradients, derive_scale)
        if scalars:
            self._sampled_steps.append(last)
            self._emit_attach_rows(last)
            if derive_scale is not None and include_gradients:
                scalars.append(
                    ScalarPoint(self.grammar.run_health("amp_unscale_derived"), last, 1.0)
                )
            self._emit(StepEmission(step=last, scalars=tuple(scalars), histograms=()))
            self._named_skips.append(f"final_sample_forced at step {last}")

    def _forced_gradient_policy(self, last: int) -> tuple[bool, float | None]:
        """Whether the forced sample may read ``p.grad``, and the AMP factor to divide out.

        ``p.grad`` at close is whatever the last backward left: unscaled in
        place under ``scaler.step`` (read as is), still scaled under a plain
        ``optimizer.step`` (corrected by the observed scale), or of unknown
        basis when the scaler's stage was unreadable (skipped, named).
        """

        grad_evidence = self._scale_evidence if self._current_scaler is not None else "unavailable"
        include_gradients = "gradients" in self.signals
        if include_gradients and grad_evidence == "scaled_unknown_factor":
            include_gradients = False
            self._named_skips.append(
                f"final_sample_gradients_skipped at step {last}: AMP scale evidence unknown"
            )
        derive_scale = (
            self._scale_at_boundary
            if grad_evidence == "unscaled_derived" and self._scale_at_boundary
            else None
        )
        return (include_gradients, derive_scale)

    def _forced_sample_scalars(
        self, last: int, include_gradients: bool, derive_scale: float | None
    ) -> list[ScalarPoint]:
        """Reduce parameters (and admitted gradients) directly for the forced sample."""

        scalars: list[ScalarPoint] = []
        if self.collector is None:
            return scalars
        with torch.no_grad(), _state.pause_logging():
            for name, param in self.model.named_parameters():
                slot = self.collector._param_sites.get(name)
                if slot is None:
                    continue
                leaf = slot.record.display_label
                targets = [("parameters", param.detach())]
                if param.grad is not None and include_gradients:
                    targets.append(("gradients", param.grad.detach()))
                for family, tensor in targets:
                    if family == "parameters" and "parameters" not in self.signals:
                        continue
                    spine = spine_result_from_vector(spine_vector(tensor).cpu(), "float64")
                    if family == "gradients" and derive_scale is not None:
                        spine = correct_spine(spine, float(derive_scale))
                    for statistic, value in spine_scalars(spine).items():
                        scalars.append(
                            ScalarPoint(self.grammar.data(family, statistic, leaf), last, value)
                        )
        return scalars

    def close(self, *, unwinding: bool = False) -> CloseReport:
        """Detach everything, force the last sample, settle the report.

        Raises ``watch_close_empty`` when zero requested data was ever
        emitted -- EXCEPT while a user exception is unwinding (``__exit__``
        never masks the training loop's own failure).
        """

        if self._closed:
            return self._report()
        self._closed = True
        _deregister_session(self)
        if self.collector is not None:
            if self._own_optimizer_handle is not None:
                self._own_optimizer_handle.remove()
                self._own_optimizer_handle = None
            for handle in list(getattr(self, "_post_handles", ())):
                handle.remove()
            if self._forward_handle is not None:
                self._forward_handle.remove()
                self._forward_handle = None
            self._final_forced_sample()
            self.collector.detach()
            fallback = self._seen_steps[-1] if self._seen_steps else 0
            self._emit_attach_rows(fallback)
            if self.event_stream is not None:
                self._flush_pending_events(fallback)
        self._flush_and_close_sinks()
        report = self._report()
        # DATA points only: a heartbeat or manifest landing in a sink is not
        # "the dashboard has data" (the old count let run-health rows mask an
        # empty run, AUD-CODE 2.14a).
        emitted_any = any(row.emitted_data_points for row in self.ledger.sinks.values())
        if (
            not unwinding
            and self.collector is not None
            and self._seen_steps
            and not emitted_any
            and not self._data_absence_explained()
        ):
            raise WatchRuntimeError(
                "watch closed with ZERO successfully emitted data points for "
                "the requested signals -- an empty dashboard with no "
                "explanation is the failure class this engine exists to "
                "kill.",
                code="watch_close_empty",
                remedy=(
                    "Read the close report's sink rows and named skips: "
                    + "; ".join(self._named_skips or ["no skips recorded"])
                ),
            )
        return report

    def _flush_and_close_sinks(self) -> None:
        """Flush then close every sink, latching failures on its ledger row.

        flush and close are SEPARATE attempts: a flush that raised used to
        skip close() entirely and leak the handle (AUD-CODE 2.14).
        """

        for sink in self.sinks:
            row = self.ledger.row(sink)
            try:
                sink.flush()
            except Exception as exc:  # noqa: BLE001 - latched + reported, never silent
                row.failed = True
                row.failure = f"flush: {type(exc).__name__}: {exc}"
            try:
                sink.close()
            except Exception as exc:  # noqa: BLE001 - latched + reported, never silent
                row.failed = True
                prefix = f"{row.failure}; " if row.failure else ""
                row.failure = f"{prefix}close: {type(exc).__name__}: {exc}"

    def _data_absence_explained(self) -> bool:
        """True when EVERY seen step carries a named data-absence skip.

        An AMP-skipped step or a demoted missing phase is the loop working
        as declared, not an unexplained empty panel; a latched sink is not
        an explanation of absent data and still raises.
        """

        explained = sum(
            1 for skip in self._named_skips if skip.startswith(("amp_skipped", "phase_missing"))
        )
        return explained >= len(self._seen_steps)

    def _report(self) -> CloseReport:
        """Assemble the close report from the ledgers."""

        for sink in self.sinks:
            row = self.ledger.row(sink)
            relay_state = getattr(sink, "relay_state", None)
            if callable(relay_state) and not row.failed:
                row.relay_state, row.relay_detail = relay_state()
            elif not row.failed and (row.emitted_scalars or row.emitted_histograms):
                row.relay_state = "emitted"
        missing = tuple(skip for skip in self._named_skips if skip.startswith("phase_missing"))
        return CloseReport(
            requested_signals=self.signals,
            matched_sites=len(self.collector._sites_by_id) if self.collector else 0,
            sampled_steps=tuple(self._sampled_steps),
            first_step=self._seen_steps[0] if self._seen_steps else None,
            last_step=self._seen_steps[-1] if self._seen_steps else None,
            final_sample_forced=any(
                skip.startswith("final_sample_forced") for skip in self._named_skips
            ),
            computed_scalars=self.ledger.computed_scalars,
            computed_histograms=self.ledger.computed_histograms,
            dropped_blocks=self.collector.ring.dropped_blocks if self.collector else 0,
            sink_rows=tuple(row.as_dict() for row in self.ledger.sinks.values()),
            missing_phases=missing,
            named_skips=tuple(self._named_skips),
            disabled=self.collector is None,
        )

    def __enter__(self) -> WatchSession:
        """Context form; close() is guaranteed on every exit path."""

        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Close on every exit path; never mask the loop's own exception."""

        self.close(unwinding=exc_type is not None)


def _validate_signals(
    signals: Iterable[str],
    select: Any,
    example_input: Any,
    grain: str | None,
) -> tuple[str, ...]:
    """Validate the signal request; refuse deferred tiers typed."""

    requested = tuple(signals)
    unknown = [signal for signal in requested if signal not in SIGNALS]
    if unknown or not requested:
        raise WatchConfigError(
            f"signals={requested!r} is not a valid request; the closed "
            f"vocabulary is {SIGNALS} and at least one signal is required.",
            code="watch_signals_invalid",
            signals=requested,
            remedy=f"Request a nonempty subset of {SIGNALS}.",
        )
    if grain == "op":
        raise WatchConfigError(
            f"Op-grain watching (grain='op') is not shipped in this build. {_TIER_O_COST_TEACHING}",
            code="watch_tier_unavailable",
            signal="activations",
            grain="op",
            remedy=_TIER_O_REMEDY,
        )
    if "activation_grads" in requested:
        raise WatchConfigError(
            "signals='activation_grads' rides the F24 backward-observer seam, "
            "which has not merged yet; the C06 Route-A collector sees forward "
            "module outputs only. Refusing beats a silently absent series.",
            code="watch_tier_unavailable",
            signal="activation_grads",
            remedy=(
                "Drop 'activation_grads' for now, or capture a step with "
                "capture=tl.options.CaptureOptions(backward_ready=True) and "
                "read gradients from the trace."
            ),
        )
    if "activations" in requested:
        if select is None:
            raise WatchConfigError(
                "signals include 'activations' but no select= was passed: "
                "there is NO default activation selector (a friendly "
                "model-dependent default can be silently empty, huge, or "
                "unexpectedly expensive -- real distilgpt2 has no softmax "
                "and no gelu op).",
                code="watch_signals_invalid",
                signals=requested,
                remedy=(
                    "Pass select=(module path prefixes or parameter names) naming what to watch."
                ),
            )
        if example_input is None:
            raise WatchConfigError(
                "Module-tier activations need example_input= at attach: the "
                "one real discovery forward (run in the model's CURRENT "
                "mode) is what catalogs sites, validates the selector "
                "against reality, and prices the plan before your loop "
                "spends time.",
                code="watch_signals_invalid",
                signals=requested,
                remedy="Pass example_input=<a real batch> (tuple/tensor).",
            )
    return requested


def watch(
    model: torch.nn.Module,
    *,
    to: Any,
    signals: Iterable[str] = ("gradients",),
    select: Iterable[str] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    step: Callable[[], int] | None = None,
    every: int = DEFAULT_EVERY,
    hist_every: int | None = None,
    name: str | None = None,
    namespace: str | None = None,
    example_input: Any = None,
    grain: str | None = None,
    descriptor: HistogramDescriptor | None = None,
    budgets: Mapping[str, int] | None = None,
    allow_missing_phases: bool = False,
    event_stream: EventStream | None = None,
    settings: WatchSettings | None = None,
) -> WatchSession:
    """Attach once; three disclosed cost tiers; loud failures (memo 3.13).

    Every spelling here is DOCUMENTED-UNSTABLE pending the naming sprint.

    Parameters
    ----------
    model:
        The live module to watch. Never wrapped, never re-traced: tier P
        installs optimizer hooks only, tier M ordinary module hooks.
    to:
        One sink or a tuple of sinks implementing the documented protocol
        (:class:`~torchlens.trackers.TensorBoardSink`,
        :class:`~torchlens.trackers.WandbSink`,
        :class:`~torchlens.trackers.JSONLSink`, or your own).
    signals:
        Subset of :data:`SIGNALS`. Default watches parameter gradients.
    select:
        Site selection (module path prefixes / parameter names). REQUIRED
        for activations; optional narrowing for parameter streams.
    optimizer:
        The optimizer whose step boundary anchors parameter streams.
    step:
        Optional zero-argument callable returning the caller's global step
        (the unchanged-loop spelling). The explicit ``watch.step(n)`` scope
        is authoritative and REQUIRED for resume.
    every:
        Scalar cadence on the caller's step axis (first and last eligible
        steps always sample).
    hist_every:
        Histogram cadence; may run coarser than scalars. ``None`` disables
        the sketch tier (scalars remain).
    name:
        Multi-model tag component distinguishing sessions that share a sink
        (memo 3.4).
    namespace:
        Tag re-rooting remedy for layout collisions in a shared sink
        (memo 3.4).
    example_input:
        Attach-time discovery batch (required for activations).
    grain:
        ``"op"`` requests op-grain identity and refuses typed this window.
    descriptor:
        Histogram grid override. When neither this nor ``settings`` names a
        grid, the first sink offering ``default_histogram_descriptor()``
        supplies it (``WandbSink``: ``WANDB_SAFE_DESCRIPTOR``, 497 buckets);
        otherwise the C06 default grid applies.
    budgets:
        Hard caps checked against the plan as preflight FACTS:
        ``max_sites``, ``max_bytes_per_step``, ``max_elements_per_step``.
    allow_missing_phases:
        Demote the missing-optimizer-phase refusal to a named skip (mixed
        train/eval loops).
    event_stream:
        A C06 chassis stream whose check verdicts serialize as
        ``torchlens/check/<name>`` rows.
    settings:
        Advanced C06 collector plumbing (ring capacity, output_dir, ...).
    """

    sinks = tuple(to) if isinstance(to, (tuple, list)) else (to,)
    requested = _validate_signals(signals, select, example_input, grain)
    grammar = TagGrammar(name=name, namespace=namespace)
    disabled_by = WATCH_DISABLE_ENV if closed_bool_env(WATCH_DISABLE_ENV) else None
    needs_histograms = hist_every is not None
    needed = ["scalar", "text_manifest"] + (["raw_histogram"] if needs_histograms else [])
    for sink in sinks:
        require_capabilities(sink, tuple(needed))
    _refuse_duplicate_sinks(sinks)
    if disabled_by is not None:
        session = WatchSession(
            model,
            sinks,
            None,
            grammar,
            _SessionConfig(
                signals=requested,
                every=every,
                hist_every=hist_every,
                allow_missing_phases=allow_missing_phases,
                event_stream=event_stream,
                disabled_by=disabled_by,
            ),
        )
        point = ScalarPoint(grammar.run_health("disabled"), 0, 1.0)
        session._emit(StepEmission(step=0, scalars=(point,), histograms=()))
        return session
    foreign = () if namespace is not None else _foreign_watch_prefixes(model)
    if foreign:
        raise WatchConfigError(
            f"A foreign watcher already owns hook-installed series on this "
            f"model (detected prefixes: {', '.join(foreign)}). Two watchers "
            "interleaving the same dashboard sections is the migration "
            "footgun this refusal exists to catch.",
            code="tracker_namespace_collision",
            prefixes=foreign,
            remedy=(
                "Drop the foreign watcher (e.g. skip wandb.watch), or pass "
                "namespace='torchlens' to re-root every TorchLens series."
            ),
        )
    streams = tuple(_SIGNAL_STREAMS[signal] for signal in requested if signal in _SIGNAL_STREAMS)
    if descriptor is None and needs_histograms:
        descriptor = _sink_default_descriptor(sinks, settings)
    base_settings = _resolve_settings(settings, descriptor, needs_histograms)
    _refuse_second_session(model, grammar)
    _preflight_histogram_sinks(sinks, base_settings.descriptor, needs_histograms)
    collector = HistoryCollector(
        model,
        sites=tuple(select) if select is not None else None,
        streams=streams,
        cadence=1,
        settings=base_settings,
    )
    if example_input is not None:
        args = example_input if isinstance(example_input, tuple) else (example_input,)
        collector.discover(*args)
    else:
        _discover_without_forward(collector)
    _check_budgets(collector, budgets)
    _preflight_tags(grammar, collector, requested)
    _refuse_sparse_gradients(collector, requested)
    session = WatchSession(
        model,
        sinks,
        collector,
        grammar,
        _SessionConfig(
            signals=requested,
            every=every,
            hist_every=hist_every,
            optimizer=optimizer,
            step_source=step,
            allow_missing_phases=allow_missing_phases,
            event_stream=event_stream,
        ),
    )
    if optimizer is not None:
        session._own_optimizer_handle = optimizer.register_step_pre_hook(session._boundary_pre_hook)
    collector.attach(optimizer=optimizer)
    if optimizer is not None:
        session._post_handles = [optimizer.register_step_post_hook(session._boundary_post_hook)]
    else:
        session._post_handles = []
    _arm_forward_scheduling(session, model, step, requested)
    session._stage_attach_rows()
    _register_session(model, grammar, session)
    return session


def _resolve_settings(
    settings: WatchSettings | None,
    descriptor: HistogramDescriptor | None,
    needs_histograms: bool,
) -> WatchSettings:
    """Fold the descriptor override and the sketch tier into the C06 settings."""

    base_settings = settings if settings is not None else WatchSettings()
    if descriptor is not None or needs_histograms:
        base_settings = _dc_replace(
            base_settings,
            descriptor=descriptor if descriptor is not None else base_settings.descriptor,
            sketch_cadence=1 if needs_histograms else base_settings.sketch_cadence,
        )
    return base_settings


def _arm_forward_scheduling(
    session: WatchSession,
    model: torch.nn.Module,
    step: Callable[[], int] | None,
    requested: tuple[str, ...],
) -> None:
    """Install the tier-M forward-entry scheduler when ``step=`` drives the axis."""

    if step is not None and "activations" in requested:
        session._forward_handle = model.register_forward_pre_hook(session._forward_entry_hook)


def _refuse_duplicate_sinks(sinks: tuple[Any, ...]) -> None:
    """One sink OBJECT listed twice would receive every point twice."""

    seen: set[int] = set()
    for sink in sinks:
        if id(sink) in seen:
            raise SinkProtocolError(
                f"{sink_name(sink)} appears more than once in to=: one sink "
                "object listed twice would receive every point twice while its "
                "ledger row counted them once.",
                code="tracker_sink_duplicate",
                sink=sink_name(sink),
                remedy="List each sink object once (two files need two JSONLSink objects).",
            )
        seen.add(id(sink))


def _register_session(model: torch.nn.Module, grammar: TagGrammar, session: WatchSession) -> None:
    """Record ``session`` as the live watcher of ``model`` under its grammar."""

    _ACTIVE_SESSIONS.setdefault(model, {})[(grammar.name, grammar.namespace)] = session


def _deregister_session(session: WatchSession) -> None:
    """Forget ``session`` at close (idempotent; other grammars untouched)."""

    live = _ACTIVE_SESSIONS.get(session.model)
    if not live:
        return
    key = (session.grammar.name, session.grammar.namespace)
    if live.get(key) is session:
        del live[key]


def _refuse_second_session(model: torch.nn.Module, grammar: TagGrammar) -> None:
    """A second open session on one model under one grammar duplicates every point."""

    live = _ACTIVE_SESSIONS.get(model) or {}
    existing = live.get((grammar.name, grammar.namespace))
    if existing is not None and not existing._closed:
        raise WatchConfigError(
            "This model already has an open watch session under the same tag "
            f"grammar (name={grammar.name!r}, namespace={grammar.namespace!r}); "
            "a second session would install a second hook set and emit every "
            "point TWICE into a shared sink.",
            code="tracker_namespace_collision",
            name=grammar.name,
            namespace=grammar.namespace,
            remedy=(
                "close() the open session first, or pass a distinct name= so "
                "the two series stay distinguishable."
            ),
        )


def _sink_default_descriptor(
    sinks: tuple[Any, ...], settings: WatchSettings | None
) -> HistogramDescriptor | None:
    """The first sink-offered histogram grid, when the caller chose none.

    A sink may expose ``default_histogram_descriptor()`` (``WandbSink`` returns
    ``WANDB_SAFE_DESCRIPTOR``, which fits wandb's bucket cap). It applies only
    when ``settings`` is absent or still carries the untouched C06 default
    grid; a caller-chosen grid is kept and preflight judges it.
    """

    if settings is not None and settings.descriptor is not DEFAULT_DESCRIPTOR:
        return None
    for sink in sinks:
        offer = getattr(sink, "default_histogram_descriptor", None)
        if callable(offer):
            offered = offer()
            if isinstance(offered, HistogramDescriptor):
                return offered
    return None


def _preflight_histogram_sinks(
    sinks: tuple[Any, ...], descriptor: HistogramDescriptor, needs_histograms: bool
) -> None:
    """Attach-time histogram admission for sinks that declare a check.

    A sink may expose ``preflight_histograms(descriptor)`` raising typed;
    the same refusal raised inside ``emit_histogram`` used to be swallowed
    by the runtime latch and silently stopped every later row.
    """

    if not needs_histograms:
        return
    for sink in sinks:
        preflight = getattr(sink, "preflight_histograms", None)
        if callable(preflight):
            preflight(descriptor)


def _preflight_tags(
    grammar: TagGrammar, collector: HistoryCollector, requested: tuple[str, ...]
) -> None:
    """Render one tag per catalogued site at ATTACH so an unsafe leaf refuses now.

    The first render used to happen inside the first drain -- inside a
    ``finally`` -- where the typed refusal masked the user's own exception.
    """

    param_families = tuple(f for f in ("parameters", "gradients", "updates") if f in requested)
    for site in collector._sites_by_id.values():
        families = param_families if site.kind == "param" else ("activations",)
        for family in (*families, "nonfinite"):
            grammar.data(family, "norm", site.display_label)


def _refuse_sparse_gradients(collector: HistoryCollector, requested: tuple[str, ...]) -> None:
    """Gradient watching over ``sparse=True`` modules refuses at attach.

    The gradient reduction kernel needs dense grads; a sparse grad used to
    crash UNTYPED inside the optimizer pre-hook, aborting the user's step.
    """

    if "gradients" not in requested:
        return
    sparse: list[str] = []
    for name in collector._param_sites:
        owner, _, _leaf = name.rpartition(".")
        try:
            module = collector.model.get_submodule(owner) if owner else collector.model
        except AttributeError:
            continue
        if getattr(module, "sparse", False) is True:
            sparse.append(name)
    if sparse:
        raise WatchConfigError(
            f"Parameters {sparse} belong to modules constructed with sparse=True "
            "(sparse gradients); the gradient reduction needs dense grads and "
            "would crash INSIDE optimizer.step(), aborting the update.",
            code="watch_sparse_grad_unsupported",
            parameters=tuple(sparse),
            remedy="Exclude them via select=, or construct the module with sparse=False.",
        )


def _discover_without_forward(collector: HistoryCollector) -> None:
    """Tier-P discovery: catalog parameter sites with NO forward pass.

    The C06 ``discover()`` spelling runs one real discovery forward because
    module sites need output geometry; parameter sites do not. This is the
    param-only door tier P uses so the DEFAULT tier never executes the
    user's model. Seam note: uses the collector's internal catalog steps in
    their documented order; a public param-only discovery door on the C06
    collector is filed as a follow-up.
    """

    if any(stream == "activation" for stream in collector.streams):
        raise WatchConfigError(
            "Activation streams need the discovery forward; this door is parameter-streams-only.",
            code="watch_signals_invalid",
            remedy="Pass example_input= so discover() can run.",
        )
    collector._catalog_param_sites()
    if not collector._sites_by_id:
        raise WatchConfigError(
            "The site selection matched NO parameters on this model.",
            code="watch_selector_matched_no_sites",
            requested=collector.requested_sites,
            census=tuple(sorted(_module_census(collector.model))),
            remedy=(
                "Pass select= entries that prefix-match named_parameters() "
                "keys, or select=None for every parameter."
            ),
        )
    device = "cpu"
    for param in collector.model.parameters():
        device = str(param.device)
        break
    collector._allocate_staging(device)
    collector._plan = collector._build_plan()
    if collector.writer is not None:
        for site in collector._sites_by_id.values():
            collector.writer.add_site(site)


def _check_budgets(collector: HistoryCollector, budgets: Mapping[str, int] | None) -> None:
    """Enforce the hard preflight caps against the plan FACTS (memo 3.9)."""

    if not budgets:
        return
    plan = collector.plan
    facts = {
        "max_sites": len(collector._sites_by_id),
        "max_bytes_per_step": plan.total_bytes_per_step,
        "max_elements_per_step": plan.total_elements_per_step,
    }
    unknown = set(budgets) - set(facts)
    if unknown:
        raise WatchConfigError(
            f"Unknown budget keys {sorted(unknown)}; the closed set is {sorted(facts)}.",
            code="watch_budget_invalid",
            unknown=tuple(sorted(unknown)),
            remedy=f"Use budget keys from {sorted(facts)}.",
        )
    for key, cap in budgets.items():
        actual = facts[key]
        if actual > cap:
            raise WatchConfigError(
                f"Budget {key}={cap} exceeded before attach: the resolved "
                f"plan needs {actual}. This is a preflight FACT refusal -- "
                "nothing was emitted and no loop time was spent.",
                code="watch_budget_exceeded",
                budget=key,
                cap=cap,
                actual=actual,
                remedy=(
                    "Narrow select=, reduce requested signals, or raise the budget deliberately."
                ),
            )


__all__ = [
    "DEFAULT_EVERY",
    "SIGNALS",
    "WATCH_DISABLE_ENV",
    "CloseReport",
    "WatchSession",
    "watch",
]
