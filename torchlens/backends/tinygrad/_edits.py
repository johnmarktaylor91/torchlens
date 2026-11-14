"""Edits applied during a tinygrad capture.

The observer sees every Tensor-API call result before the model does, so an
edit (resolved by ``_edit_plan``) replaces a call's result with a lazy
expression over it. Nothing is realized when an edit applies, so the edited
capture still realizes once, and the edit's own ops are never recorded: the
edited op keeps the label, parents and module of the call it replaced, and its
payload is the edited value.

An edited capture runs the forward twice. The first run is a plain capture of
the same call; the selector is matched against its finished ops exactly as
``trace(save=...)`` matches, and each matched op is mapped to the Tensor-API
calls that returned it. The second run applies the edit at those calls.
tinygrad forwards are lazy, so both runs make the same calls in the same
order; a call whose name differs between the runs refuses typed, and so does
an edited capture whose matched ops are not exactly the edited ones, or an
edit that changes shape or dtype.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from ..._ir.intervention import FireResult
from ..._ir.op_record import OpAmendment, amend_module_exit_intervention
from .._options import is_missing
from ..registry import BackendUnsupportedError, get_backend_spec, require_capability_implementation
from ._edited_grads import attach_edited_derived_grads
from ._rules import RNG_STATE_IN_GRAPH_CODE

if TYPE_CHECKING:
    from ..._selection.selectors import BaseSelector

EDIT_REFUSAL = "interventions_unsupported"
_PLAN_ID = "tinygrad_intervene"


@dataclass(frozen=True)
class TinygradEdit:
    """A resolved edit: a lazy ``tensor -> tensor`` application.

    Parameters
    ----------
    identity
        Stable name of the edit (``helper:<name>`` or ``callable:<qualname>``).
    apply
        Maps the call's result tensor to its lazy replacement.
    """

    identity: str
    apply: Callable[[Any], Any]


@dataclass(frozen=True)
class TinygradEditPlan:
    """The condition and edit of one ``trace(intervene=tl.when(condition, edit))``.

    ``matcher(trace, condition)`` gives the raw labels a condition that reads
    predicate contexts (a plain callable or a ``tl.where`` leaf) matches, else
    it is ``None`` and the condition is called on each op. ``grad_site`` is
    set when the condition is a ``tl.grad(...)`` site: it arms the edit on a
    plain capture and attaches the derived gradients it edits
    (``_grad_site_grads.attach_grad_site_derived_grads``).
    """

    selector: BaseSelector | Callable[[Any], bool]
    edit: TinygradEdit
    matcher: Callable[[Any, Any], frozenset[str]] | None = None
    grad_site: Callable[..., Any] | None = None


def refuse_edit(message: str, *, code: str) -> BackendUnsupportedError:
    """A typed refusal of an edit the tinygrad backend does not support."""

    return BackendUnsupportedError(f"tinygrad backend: {message}", code=code)


class TinygradEditSession:
    """Numbers the Tensor-API calls of one forward and edits the chosen ones.

    Parameters
    ----------
    sites
        ``call number -> callable name`` of the calls to edit; empty for the
        planning run, which only numbers the calls.
    edit
        The edit to apply, or ``None`` for the planning run.
    selector
        The ``tl.when`` condition, recorded on each fire.
    """

    def __init__(
        self,
        sites: Mapping[int, str],
        edit: TinygradEdit | None,
        selector: object = None,
    ) -> None:
        """Initialize a session before its forward runs."""

        self.sites = dict(sites)
        self.edit = edit
        self.selector = selector
        self.calls: list[tuple[Any, str]] = []
        self.replaced: dict[int, Any] = {}
        self._pins: list[Any] = []

    def apply(self, observer: Any, result: Any, name: str) -> Any:
        """Number one call and return its result, edited when it is a site.

        Parameters
        ----------
        observer
            The active ``_observe_tensor_ops``; paused while the edit runs so
            the edit's own calls are neither numbered nor recorded.
        result
            Tensor the call returned.
        name
            Name of the callable tinygrad applied.

        Returns
        -------
        Any
            ``result``, or its lazy replacement.
        """

        ordinal = len(self.calls)
        self.calls.append((result.uop, name))
        expected = self.sites.get(ordinal)
        if expected is None or self.edit is None:
            return result
        if expected != name:
            raise refuse_edit(
                f"call {ordinal} was {expected!r} in the planning run and {name!r} in the "
                "edited run; the forward must make the same Tensor calls every run.",
                code=EDIT_REFUSAL,
            )
        observer.paused = True
        try:
            replacement = self.edit.apply(result)
            _check_replacement(result, replacement, self.edit.identity)
            replacement = _fresh_replacement(result, replacement)
        finally:
            observer.paused = False
        observer.alias(result.uop, replacement.uop)
        self.replaced[id(replacement.uop)] = result.uop
        self._pins.append(replacement)
        return replacement

    def edit_for(self, uop: Any) -> TinygradEdit | None:
        """The edit that produced ``uop``, or ``None`` for an unedited UOp."""

        return self.edit if id(uop) in self.replaced else None


def _check_replacement(result: Any, replacement: Any, identity: str) -> None:
    """Refuse a replacement that is not a tinygrad Tensor of the result's shape and dtype."""

    if type(replacement) is not type(result):
        raise refuse_edit(
            f"edit {identity} returned {type(replacement).__name__}, not a Tensor.",
            code=EDIT_REFUSAL,
        )
    if tuple(replacement.shape) != tuple(result.shape) or replacement.dtype != result.dtype:
        raise refuse_edit(
            f"edit {identity} changed {tuple(result.shape)} {result.dtype} to "
            f"{tuple(replacement.shape)} {replacement.dtype}; edits keep shape and dtype.",
            code=EDIT_REFUSAL,
        )


def _fresh_replacement(result: Any, replacement: Any) -> Any:
    """``replacement`` as a new UOp that depends on ``result``, with the same values.

    tinygrad interns UOps, so an edit that builds its value without reading
    the result (``zeros_like``, ``replace_with``) can return a UOp the forward
    already holds elsewhere, such as the zero constant inside relu. Selecting
    through a condition that is false everywhere (NaN included) keeps the
    values and gives the replacement its own UOp in the graph.
    """

    never = (result != result) & (result == result)
    return never.where(result, replacement)


def fire_amendment(event: Any, edit: TinygradEdit, selector: object) -> OpAmendment:
    """The amendment marking an emitted op event as edited.

    Parameters
    ----------
    event
        The appended op event of the edited op.
    edit
        The edit applied at the op.
    selector
        The ``tl.when`` selector that chose the op.

    Returns
    -------
    OpAmendment
        Sets ``intervention_fired``, ``is_edited`` and one fire result.
    """

    tensor = event.output.tensor
    shape = tuple(tensor.shape) if tensor.shape is not None else None
    dtype = str(tensor.dtype) if tensor.dtype is not None else None
    fire = FireResult(
        plan_id=_PLAN_ID,
        site_label=event.label_raw,
        fired_at_capture_index=event.raw_index,
        pre_hook_shape=shape,
        post_hook_shape=shape,
        pre_hook_dtype=dtype,
        post_hook_dtype=dtype,
        replaced=True,
        fire_record={
            "backend": "tinygrad",
            "plan_id": _PLAN_ID,
            "site_label": event.label_raw,
            "level": "op",
            "selector": repr(selector),
            "edit": edit.identity,
            "replaced": True,
        },
    )
    return amend_module_exit_intervention(
        event.seq,
        event.label_raw,
        intervention_fired=True,
        is_edited=True,
        fire_results=(fire,),
    )


def _matched_labels(trace: Any, plan: TinygradEditPlan) -> frozenset[str]:
    """Raw labels of the finished ops the plan's condition matches (the ``save=`` matcher)."""

    if plan.matcher is not None:
        return plan.matcher(trace, plan.selector)
    selector = plan.selector
    return frozenset(str(op._label_raw) for op in trace._layer_list if bool(selector(op)))


def _planned_sites(
    trace: Any, plan: TinygradEditPlan, calls: list[tuple[Any, str]]
) -> dict[int, str]:
    """``call number -> name`` of the calls returning the ops the plan's condition matches."""

    uop_by_label = {capture.label_raw: capture.uop for capture in trace.tinygrad_uop_captures}
    numbers: dict[int, list[int]] = {}
    for number, (uop, _name) in enumerate(calls):
        numbers.setdefault(id(uop), []).append(number)
    sites: dict[int, str] = {}
    for label in sorted(_matched_labels(trace, plan)):
        uop = uop_by_label.get(label)
        found = numbers.get(id(uop), []) if uop is not None else []
        if not found:
            raise refuse_edit(
                f"the selector matched op {label!r}, which no Tensor call returned (an "
                "input, an output or an op tinygrad built inside a call). Edits apply to "
                "Tensor call results; narrow the selector, e.g. with tl.label(...).",
                code=EDIT_REFUSAL,
            )
        sites.update((number, calls[number][1]) for number in found)
    return sites


def capture_with_edits(backend: Any, call: dict[str, Any], intervene: Any, halt: Any) -> Any:
    """Capture ``call`` with ``trace(intervene=...)`` applied.

    The registered ``can_intervene`` capability implementation resolves the
    edit, so a capability table that does not bind it refuses typed.

    Parameters
    ----------
    backend
        The ``TinygradBackend``; ``capture_trace`` reads its ``_edit_session``
        and ``_warn_zero_fire`` (the shared coded zero-fire settlement) warns
        when nothing matched.
    call
        ``capture_trace`` keyword arguments without ``intervene``/``stop_after``.
    intervene
        Public ``trace(intervene=...)`` value.
    halt
        Public ``trace(stop_after=...)`` value.

    Returns
    -------
    Any
        The edited trace, or the plain trace (with a warning) when nothing matched.
    """

    spec = get_backend_spec("tinygrad")
    if not spec.capabilities.can_intervene:
        raise refuse_edit("the capability table declares can_intervene=False.", code=EDIT_REFUSAL)
    resolve = require_capability_implementation(spec, "can_intervene")
    plan = cast(TinygradEditPlan, cast(Any, resolve)(intervene, halt))
    grad_options = _given_grad_options(call)
    if plan.grad_site is not None:
        return _capture_with_grad_site(backend, call, plan, grad_options)
    original, call = call, {**call, "grad_options": None}
    planning = TinygradEditSession({}, None)
    planned = _run_session(backend, call, planning)
    sites = _planned_sites(planned, plan, planning.calls)
    if not sites:
        backend._warn_zero_fire(
            f"Capture-time intervention selector {plan.selector!r} matched zero sites on "
            "the tinygrad forward; no intervention fired.",
            selector=repr(plan.selector),
        )
        return planned if grad_options is None else backend.capture_trace(**original)
    _refuse_lazy_random_draws(planned)
    expected = _matched_labels(planned, plan)
    del planned
    session = TinygradEditSession(sites, plan.edit, plan.selector)
    edited = _run_session(backend, call, session)
    edited_labels = {c.label_raw for c in edited.tinygrad_uop_captures if c.edit is not None}
    if not expected == edited_labels == _matched_labels(edited, plan):
        raise refuse_edit(
            "the edited capture's matched ops differ from the planning capture's; the edit "
            "was not applied where the selector points.",
            code=EDIT_REFUSAL,
        )
    if grad_options is not None:
        replay = TinygradEditSession(fired_sites(session), plan.edit)
        attach_edited_derived_grads(backend, edited, original, session, replay)
    return edited


def _given_grad_options(call: Mapping[str, Any]) -> Any:
    """The call's ``grad_options``, or ``None`` when absent."""

    grad_options = call.get("grad_options")
    return None if grad_options is None or is_missing(grad_options) else grad_options


def _site_labels(trace: Any, selector: Any) -> list[str]:
    """Labels of the finished ops a forward selector matches, in trace order.

    These are the final labels the tinygrad replay adapter names its recipe slots
    with, so the armed sites match the replay graph the gradient walk runs over.
    """

    return [str(op.label) for op in trace._layer_list if bool(selector(op))]


def _capture_with_grad_site(
    backend: Any, call: dict[str, Any], plan: TinygradEditPlan, grad_options: Any
) -> Any:
    """Capture the plain forward and attach derived gradients with the ``tl.grad`` edit armed."""

    if grad_options is None:
        raise refuse_edit(
            f"the site {plan.selector!r} edits a gradient. tinygrad records no backward pass, "
            "so a tl.grad(...) edit applies in derived gradients: pass grad_options=GradOptions(...).",
            code="backward_capture_unsupported",
        )
    trace = backend.capture_trace(**{**call, "grad_options": None})
    attach = cast(Any, plan.grad_site)
    attach(backend, trace, call, plan, _site_labels)
    return trace


def fired_sites(session: TinygradEditSession) -> dict[int, str]:
    """``call number -> callable name`` of the calls ``session`` edited."""

    replaced = {id(site) for site in session.replaced.values()}
    return {number: name for number, (uop, name) in enumerate(session.calls) if id(uop) in replaced}


def _refuse_lazy_random_draws(planned: Any) -> None:
    """Refuse an edit when the forward reads random numbers that are not yet realized.

    An edited capture runs the forward twice. A lazy random draw (such as the
    weights of a default-initialized ``tinygrad.nn`` layer) is evaluated by each
    run's realization and advances tinygrad's random state each time, so the
    two runs would not see the same values.
    """

    seen: set[int] = set()
    stack = [capture.uop for capture in planned.tinygrad_uop_captures]
    while stack:
        uop = stack.pop()
        if id(uop) in seen:
            continue
        seen.add(id(uop))
        if getattr(getattr(uop, "op", None), "name", None) == "THREEFRY":
            raise refuse_edit(
                "the forward reads lazy random values (for example default-initialized "
                "tinygrad.nn weights), which an edited capture would draw twice. Realize "
                "them first, e.g. with Tensor.realize(*nn.state.get_parameters(model)).",
                code=RNG_STATE_IN_GRAPH_CODE,
            )
        stack.extend(getattr(uop, "src", ()) or ())


def _run_session(backend: Any, call: dict[str, Any], session: TinygradEditSession) -> Any:
    """Run one plain ``capture_trace`` of ``call`` with ``session`` observing it."""

    backend._edit_session = session
    try:
        return backend.capture_trace(**call)
    finally:
        backend._edit_session = None
