"""Tier O: op-grain watching through a disclosed, consented forward wrap.

``watch(model, to=sink, signals=("activations",), grain="op", select=...,
example_input=x, allow_forward_wrap=True)`` drives the Route-B op-tier
collector (:class:`~torchlens._observability.OpTierCollector`) from the
ordinary training loop. Per-op activations exist only inside a captured
forward, so this tier replaces the ROOT module's ``forward`` (an instance
attribute, removed again at ``close()``) with a wrapper that runs the
collector's one fastlog pass on SAMPLED steps and the original forward on
every other call. That wrap is the cost the other tiers never pay, so it
needs explicit consent and is priced at attach:

- consent: ``allow_forward_wrap=True``; without it ``grain="op"`` refuses
  typed with the cost teaching;
- calibration: an interleaved A/B (the paired-ratio overhead harness, with an
  output-structure witness) of the plain forward against the wrapped pass on
  ``example_input``, under ``no_grad``; the RNG state and every buffer are restored after discovery
  and calibration, so attach leaves the model's state as it found it;
- disclosure: the calibration lands in ``plan_report()``, in the
  ``torchlens/meta/op_grain_calibration`` text row and the
  ``torchlens/run/op_grain_overhead_median_x`` scalar, and in one
  ``watch_op_grain_cost_disclosed`` warning at attach.

Observation law: only the FIRST forward of a sampled step is observed (a
second forward in the same step runs plain and is a named skip); on observed
steps the root module's own hooks fire twice (the outer call plus the
captured one). Facet sites render under ``<op label>/<facet>`` because ``#``
is not dashboard-safe. Spellings are DOCUMENTED-UNSTABLE.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace as _dc_replace
from typing import TYPE_CHECKING, Any

import torch

from .._observability import (
    ArmSpec,
    CommittedBlock,
    OpTierCollector,
    OverheadMeasurement,
    RefusalPredicate,
    WatchSettings,
    measure_overhead,
)
from .._observability._collector import StepTruth
from ._errors import WatchConfigError

if TYPE_CHECKING:
    from ._watch import WatchSession

__tl_layer__ = "L8"

#: Calibration protocol at attach: few pairs (it runs the user's model), no
#: additivity floor and no dispersion bound -- this is a disclosure, never a
#: published number, so an inadmissible run is reported, not refused.
CALIBRATION_REPS = 3
CALIBRATION_WARMUP = 1
_CALIBRATION_PREDICATE = RefusalPredicate(min_ratio=None, max_relative_iqr=float("inf"))

_MISSING = object()

_OP_GRAIN_COST_TEACHING = (
    "Op-grain watching wraps the root module's forward: on every sampled step "
    "the forward runs inside one fastlog capture (op capture is "
    "wrap-dominated, so narrowing select= moves only part of the bill and "
    "every= is the time lever)."
)


def refuse_without_consent(allow_forward_wrap: bool) -> None:
    """Refuse ``grain="op"`` unless the caller consented to the forward wrap."""

    if allow_forward_wrap is not True:
        raise WatchConfigError(
            f"grain='op' needs explicit consent to wrap the model's forward. {_OP_GRAIN_COST_TEACHING}"
            " The cost is calibrated and disclosed at attach once you consent.",
            code="watch_forward_wrap_consent_required",
            grain="op",
            remedy=(
                "Pass allow_forward_wrap=True to accept the disclosed wrap, or "
                "watch at module grain (omit grain=) with select= module paths."
            ),
        )


def validate_grain_request(
    grain: str | None,
    requested: tuple[str, ...],
    facets: Mapping[str, Any] | None,
    allow_forward_wrap: bool,
) -> None:
    """Refuse an unknown grain, a non-activation op-grain request, or stray op knobs."""

    if grain not in (None, "op"):
        raise WatchConfigError(
            f"grain={grain!r} is not a watch grain; the closed set is None "
            "(module grain, the default) and 'op'.",
            code="watch_grain_invalid",
            grain=repr(grain),
            remedy="Omit grain= for module grain, or pass grain='op' for op grain.",
        )
    if grain is None:
        if facets is not None or allow_forward_wrap:
            raise WatchConfigError(
                "facets= and allow_forward_wrap= belong to op-grain watching "
                "(grain='op'); at module grain they would be silently ignored.",
                code="watch_signals_invalid",
                remedy="Pass grain='op' with them, or drop them.",
            )
        return
    refuse_without_consent(allow_forward_wrap)
    if requested != ("activations",):
        raise WatchConfigError(
            f"grain='op' watches op OUTPUTS, so signals must be ('activations',); "
            f"got {requested!r}. Parameter streams have no op grain.",
            code="watch_signals_invalid",
            signals=requested,
            remedy=(
                "Pass signals=('activations',) here, and watch parameter streams "
                "in a second session with a distinct name=."
            ),
        )


def op_selector(select: Any) -> Any:
    """Return the fastlog ``save=`` selector for an op-grain ``select=``.

    Strings (one or many) are module path prefixes, OR-ed into
    ``tl.in_module(...)`` selectors; any other object is a selector or
    predicate from the selector algebra and passes through unchanged.
    """

    from ..intervention.selectors import in_module

    if isinstance(select, str):
        return in_module(select)
    if isinstance(select, (tuple, list)) and select and all(isinstance(s, str) for s in select):
        combined: Any = in_module(select[0])
        for prefix in select[1:]:
            combined = combined | in_module(prefix)
        return combined
    return select


class OpGrainCollector:
    """The surface ``WatchSession`` drives, over one ``OpTierCollector``.

    Parameter streams have no op grain, so the Route-A parameter catalog,
    cadence tables and grad-scale stamps are empty here; scheduling is the
    ``armed`` flag the session sets per step.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        save: Any,
        facets: Mapping[str, Any] | None,
        settings: WatchSettings,
    ) -> None:
        """Wrap a fresh op-tier collector over ``model`` with the watch's settings."""

        self.model = model
        self.inner = OpTierCollector(model, save=save, facets=facets, settings=settings)
        self.streams: tuple[str, ...] = ("activation",)
        self.cadences: dict[str, int] = {}
        self.sketch_cadences: dict[str, int] = {}
        self._param_sites: dict[str, Any] = {}
        self._block_grad_scale: dict[Any, str] = {}
        self.armed = True
        self.sketch_armed = False
        self.calibration: OverheadMeasurement | None = None
        self._sketch_steps: set[int] = set()
        self._observed_in_block = False
        self._inside = False
        self._prior_forward: Any = _MISSING
        self._wrapper: Callable[..., Any] | None = None
        self._optimizer_handle: Any = None
        self._session: WatchSession | None = None

    # -- the surface the session reads ---------------------------------------

    @property
    def run(self) -> Any:
        """The op tier's run record."""

        return self.inner.run

    @property
    def ring(self) -> Any:
        """The op tier's committed-block ring."""

        return self.inner.ring

    @property
    def plan(self) -> Any:
        """The op tier's disclosed-cost plan."""

        return self.inner.plan

    @property
    def _last_step(self) -> int | None:
        """The last committed step on the op tier (``None`` before any)."""

        return self.inner._last_step

    @property
    def _block_open(self) -> bool:
        """True while a step bracket is open on the op tier."""

        return self.inner._block_observations is not None

    @property
    def _sites_by_id(self) -> dict[str, Any]:
        """Dashboard-safe site view: facet leaves as ``<op>/<facet>``.

        The declared-splitter placeholder row (identities land at the first
        observed step) is plan disclosure, not a series, so it is left out.
        """

        view: dict[str, Any] = {}
        for site_id, site in self.inner.site_catalog.items():
            if site.kind == "facet":
                if site_id.endswith("#*"):
                    continue
                parent, _, facet = site.display_label.partition("#")
                site = _dc_replace(site, display_label=f"{parent}/{facet}")
            view[site_id] = site
        return view

    @contextmanager
    def step(
        self,
        global_step: int,
        *,
        truth: StepTruth | None = None,
        new_segment: bool = False,
    ) -> Iterator[None]:
        """Bracket one explicit step on the op tier (one block per step)."""

        self._observed_in_block = False
        with self.inner.step(global_step, truth=truth, new_segment=new_segment):
            yield

    def arm(self, step: int, sampled: bool, sketch_scheduled: bool) -> None:
        """Arm (or mute) observation for ``step``; sketches only when scheduled."""

        self.armed = sampled
        self.sketch_armed = sketch_scheduled
        if sketch_scheduled:
            self._sketch_steps.add(step)

    def settle(self, truth_updates: Mapping[str, Any], *, optimizer_fired: bool) -> None:
        """Record the step's optimizer truth on the open op block."""

        status = truth_updates.get("optimizer_status", "applied" if optimizer_fired else "unknown")
        self.inner.settle_open_block(
            optimizer_status=status,
            scale=truth_updates.get("scale"),
            unscaled=truth_updates.get("unscaled"),
        )

    def emission_view(self, block: CommittedBlock) -> CommittedBlock:
        """Drop sketches from blocks whose step was not sketch-scheduled.

        The reducer packs a sketch on every observed op; ``hist_every=`` is
        honored at emission so a sink that never advertised histograms is
        never handed one.
        """

        if block.block.global_step in self._sketch_steps:
            return block
        observations = tuple(
            _dc_replace(obs, sketch=None) if obs.sketch is not None else obs
            for obs in block.observations
        )
        return _dc_replace(block, observations=observations)

    def disclosure(self) -> str:
        """The plan-report lines naming the wrap and its calibrated cost."""

        lines = [
            f"tier O: {type(self.model).__name__}.forward is WRAPPED (consented by "
            "allow_forward_wrap=True); sampled steps run one fastlog capture, other "
            "steps the plain forward; the first forward of a sampled step is observed; "
            "root-module hooks fire twice on observed steps",
        ]
        measurement = self.calibration
        if measurement is None:
            lines.append("calibration: not run")
            return "\n".join(lines)
        median = measurement.median_ratio
        floor = measurement.floor_ratio
        lines.append(
            "calibration at attach (example_input, no_grad, RNG and buffers restored): "
            f"median {_ratio(median)} of the plain forward, floor {_ratio(floor)}, "
            f"pairs={measurement.reps}, admissible={measurement.admissible}"
            + (f" ({'; '.join(measurement.refusals)})" if measurement.refusals else "")
        )
        return "\n".join(lines)

    # -- calibration ---------------------------------------------------------

    def discover_and_calibrate(self, example: tuple[Any, ...]) -> OverheadMeasurement:
        """Discover the op sites, then price the wrap; the model is left as found.

        The discovery pass and both calibration arms run real forwards on
        ``example`` (dropout draws, train-mode BatchNorm updates), so the CPU
        and CUDA RNG states and every buffer are snapshotted first and
        restored afterwards. The A/B (plain forward vs the per-step op-tier
        pass, both under ``no_grad``) resets the RNG before each arm so the
        harness's output witness compares like with like.
        """

        model = self.model
        fastlog_args = _fastlog_args(example)
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        buffers = [(buf, buf.detach().clone()) for _, buf in model.named_buffers()]

        def reset_rng() -> None:
            """Put the snapshotted CPU (and CUDA) RNG state back."""

            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

        def plain() -> Any:
            """The baseline arm: the model's own forward."""

            reset_rng()
            with torch.no_grad():
                return model(*example)

        def wrapped() -> Any:
            """The instrumented arm: the per-step op-tier pass."""

            reset_rng()
            with torch.no_grad():
                return self.inner.instrumented_forward(fastlog_args)

        try:
            self.inner.discover(fastlog_args)
            measurement = measure_overhead(
                ArmSpec("plain_forward", plain),
                ArmSpec("op_grain_forward", wrapped),
                reps=CALIBRATION_REPS,
                warmup=CALIBRATION_WARMUP,
                predicate=_CALIBRATION_PREDICATE,
                compare=_same_structure,
            )
        finally:
            reset_rng()
            with torch.no_grad():
                for buffer, saved in buffers:
                    buffer.copy_(saved)
        self.calibration = measurement
        return measurement

    # -- the forward wrap ----------------------------------------------------

    def attach(self, session: WatchSession, optimizer: torch.optim.Optimizer | None) -> None:
        """Install the consented root-forward wrap (and the optimizer probe)."""

        self._session = session
        model = self.model
        self._prior_forward = model.__dict__.get("forward", _MISSING)
        original = model.forward

        def watched_forward(*args: Any, **kwargs: Any) -> Any:
            """Observe the first forward of a sampled step; run plain otherwise."""

            return self._forward(original, args, kwargs)

        self._wrapper = watched_forward
        model.__dict__["forward"] = watched_forward
        if optimizer is not None:
            self._optimizer_handle = optimizer.register_step_pre_hook(self._optimizer_fired)

    def detach(self) -> None:
        """Remove the wrap (unless user code replaced it) and the optimizer probe."""

        model = self.model
        if self._wrapper is not None and model.__dict__.get("forward") is self._wrapper:
            self._restore_prior()
        self._wrapper = None
        if self._optimizer_handle is not None:
            self._optimizer_handle.remove()
            self._optimizer_handle = None

    def _optimizer_fired(self, optimizer: Any, args: Any, kwargs: Any) -> None:
        """Note that the optimizer boundary ran inside the current step."""

        del optimizer, args, kwargs
        session = self._session
        if session is not None:
            session._optimizer_fired_this_step = True
            session._optimizer_ever_fired = True

    def _restore_prior(self) -> None:
        """Put the root module's own ``forward`` binding back."""

        if self._prior_forward is _MISSING:
            self.model.__dict__.pop("forward", None)
        else:
            self.model.__dict__["forward"] = self._prior_forward

    def _forward(self, original: Callable[..., Any], args: tuple[Any, ...], kwargs: Any) -> Any:
        """Route one forward call: observed capture or the plain forward."""

        session = self._session
        if self._inside or session is None or session._closed:
            return original(*args, **kwargs)
        if self._block_open:
            if not self.armed:
                return original(*args, **kwargs)
            if self._observed_in_block:
                session._named_skips.append(
                    f"op_grain_extra_forward at step {self.inner._block_step}: "
                    "only the first forward of a sampled step is observed"
                )
                return original(*args, **kwargs)
            self._observed_in_block = True
            return self._observe(args, kwargs)
        if session.step_source is None:
            return original(*args, **kwargs)
        return self._observe_implicit(session, original, args, kwargs)

    def _observe_implicit(
        self,
        session: WatchSession,
        original: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: Any,
    ) -> Any:
        """``step=callable``: one block per sampled step, opened by its first forward."""

        step_source = session.step_source
        if step_source is None:
            return original(*args, **kwargs)
        step_value = int(step_source())
        if session._seen_steps and session._seen_steps[-1] == step_value:
            return original(*args, **kwargs)
        sampled = session._scheduled(step_value)
        session._apply_cadence(step_value, sampled)
        session._note_step(step_value, sampled)
        if not sampled:
            return original(*args, **kwargs)
        with self.inner.step(step_value):
            output = self._observe(args, kwargs)
            self.inner.settle_open_block(optimizer_status="unknown")
        session._drain()
        return output

    def _observe(self, args: tuple[Any, ...], kwargs: Any) -> Any:
        """Run the captured forward with the root's own ``forward`` restored."""

        self._inside = True
        self._restore_prior()
        try:
            return self.inner.observe_step(_fastlog_args(args), dict(kwargs) or None)
        finally:
            if self._wrapper is not None:
                self.model.__dict__["forward"] = self._wrapper
            self._inside = False


def _same_structure(plain: Any, wrapped: Any) -> bool:
    """Calibration witness: both arms returned outputs of one structure.

    Value agreement is the wrong witness for a COST calibration: a model with
    dropout (or any RNG draw the capture path orders differently) gives the
    arms different values on one input. Shapes and dtypes of every tensor
    leaf must agree, so the two arms ran the same program shape.
    """

    if isinstance(plain, torch.Tensor) or isinstance(wrapped, torch.Tensor):
        return (
            isinstance(plain, torch.Tensor)
            and isinstance(wrapped, torch.Tensor)
            and plain.shape == wrapped.shape
            and plain.dtype == wrapped.dtype
        )
    if isinstance(plain, (tuple, list)) and isinstance(wrapped, (tuple, list)):
        return len(plain) == len(wrapped) and all(
            _same_structure(a, b) for a, b in zip(plain, wrapped, strict=True)
        )
    if isinstance(plain, Mapping) and isinstance(wrapped, Mapping):
        return plain.keys() == wrapped.keys() and all(
            _same_structure(plain[key], wrapped[key]) for key in plain
        )
    return type(plain) is type(wrapped)


def _fastlog_args(args: tuple[Any, ...]) -> Any:
    """Positional forward arguments in the shape ``fastlog.record`` takes."""

    return args[0] if len(args) == 1 else list(args)


def _ratio(value: float | None) -> str:
    """Render one ratio as ``N.NNx`` (or ``n/a``)."""

    return "n/a" if value is None else f"{value:.2f}x"


def disclose_cost(collector: OpGrainCollector) -> None:
    """Warn once at attach with the calibrated cost (the loud disclosure)."""

    from ..errors import TorchLensWarning

    measurement = collector.calibration
    median = measurement.median_ratio if measurement is not None else None
    warnings.warn(
        TorchLensWarning(
            f"Op-grain watch attached with a forward wrap on "
            f"{type(collector.model).__name__}: calibrated at attach, a sampled step's "
            f"forward costs {_ratio(median)} the plain forward (median of paired ratios; "
            "unsampled steps run the plain forward). session.plan_report() has the full "
            "calibration. Remedy: raise every= to sample fewer steps, or narrow select=",
            code="watch_op_grain_cost_disclosed",
            median_ratio=median,
        ),
        stacklevel=3,
    )


__all__ = [
    "CALIBRATION_REPS",
    "CALIBRATION_WARMUP",
    "OpGrainCollector",
    "disclose_cost",
    "op_selector",
    "refuse_without_consent",
    "validate_grain_request",
]
