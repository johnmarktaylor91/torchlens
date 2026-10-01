"""Backward NaN bisector: a fire-order transition ledger with healing.

Observe memo item 10: walk one captured backward pass's ``GradFnCall``
records in MEASURED fire order and classify every transition with five
words -- ``clean`` / ``birth`` / ``propagated`` / ``healed`` / ``root_seed``
(plus ``unchecked`` for calls whose payloads were not saved). Healing is
measured fact, not decoration: a ``sqrt`` backward's NaN birth can be erased
by a downstream ``relu`` backward (its backward zeroes exactly the positions
the sqrt blew up on), so a finite parameter gradient does NOT prove a clean
backward -- the result carries the FULL ledger with the earliest provable
birth headlined. Orientation is settled against ``backward.py``:
``grad_outputs`` ARRIVE at a node, ``grad_inputs`` are EMITTED upstream.

GradFn field spellings are DOCUMENTED-UNSTABLE, so every record read routes
through the small accessor functions here rather than being scattered across
the tool; the naming sprint stays free. Every public spelling here is itself
DOCUMENTED-UNSTABLE pending ratification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from ._first_bad import FirstBadThing, amp_scaled_gradients_hint

if TYPE_CHECKING:
    from torchlens.data_classes.trace import Trace

__all__ = ["BackwardNanResult", "BackwardTransition", "bisect_nan_backward"]


@dataclass(frozen=True)
class BackwardTransition:
    """One backward node execution, classified.

    Parameters
    ----------
    fire_index:
        Measured fire-order position of the grad_fn within its backward pass.
    grad_fn_label:
        TorchLens backward node label.
    grad_fn_class:
        Autograd node class name (``SqrtBackward0``).
    op_label:
        Paired FORWARD op's public label via the grad-fn handle join, when
        the node differentiates a captured op.
    module:
        Module address of the paired forward op, when known.
    source_line:
        ``"file:line"`` of the paired forward call site, when known.
    verdict:
        ``clean`` / ``birth`` / ``propagated`` / ``healed`` / ``root_seed`` /
        ``unchecked``.
    arriving:
        Non-finite kind among ARRIVING gradients (``grad_outputs``):
        ``"none"`` / ``"nan"`` / ``"inf"`` / ``"nan+inf"`` / ``"unchecked"``.
    emitted:
        Non-finite kind among EMITTED gradients (``grad_inputs``), same
        vocabulary.
    checked_leaves:
        Number of tensor leaves actually checked at this call.
    unchecked_leaves:
        Tensor leaves that could not be checked (unsupported dtype); a
        nonzero count keeps this call out of any ``clean`` claim.
    """

    fire_index: int
    grad_fn_label: str
    grad_fn_class: str
    op_label: str | None
    module: str | None
    source_line: str | None
    verdict: str
    arriving: str
    emitted: str
    checked_leaves: int
    unchecked_leaves: int


@dataclass(frozen=True)
class BackwardNanResult:
    """Result of :func:`bisect_nan_backward`.

    Parameters
    ----------
    found:
        Whether any non-finite gradient was observed in the walked pass.
    birth:
        The EARLIEST PROVABLE birth transition (finite checkable in,
        non-finite out), or ``None``. A ``root_seed`` non-finite (non-finite
        already in the seed gradient) is reported through ``transitions``
        and headlined in the message, never silently promoted to a birth.
    verdict_scope:
        ``"complete"`` when every earlier fire was checkable;
        ``"found_first_among_checked"`` when unchecked earlier fires exist
        (the claim is never absolute first); ``"clean_complete"`` /
        ``"clean_among_checked"`` for no-finding walks.
    transitions:
        The FULL ledger in measured fire order.
    healed:
        Labels of transitions where a non-finite arrival was erased
        (non-finite in, finite out) -- a finite loss gradient does not prove
        a clean backward.
    backward_pass:
        The one-based backward pass walked.
    checked_calls:
        Calls with saved, checkable gradient payloads.
    unchecked_calls:
        Calls without saved payloads (not selected by ``save_grads``) or
        with uncheckable dtypes; they lower coverage and are named in
        ``uncertainty_zone``.
    uncertainty_zone:
        Backward-node labels whose payloads could not be checked.
    grad_scale:
        The disclosed AMP loss scale supplied by the caller, or ``None``.
    amp_hint:
        The shared GradScaler disclosure (item 6) when every finding is
        ``inf``-kind and no ``grad_scale`` was supplied; ``None`` otherwise.
    message:
        Human-readable summary headlining the earliest provable birth.
    """

    found: bool
    birth: BackwardTransition | None
    verdict_scope: str
    transitions: tuple[BackwardTransition, ...]
    healed: tuple[str, ...]
    backward_pass: int
    checked_calls: int
    unchecked_calls: int
    uncertainty_zone: tuple[str, ...]
    grad_scale: float | None
    amp_hint: str | None
    message: str

    @property
    def first_bad_thing(self) -> FirstBadThing:
        """Project this result into the shared first-bad-thing vocabulary."""

        headline = self.birth
        if headline is None and self.found:
            headline = next(
                (t for t in self.transitions if t.verdict in ("root_seed", "propagated")),
                None,
            )
        kind = "none"
        if headline is not None:
            kind = headline.emitted if headline.emitted != "none" else headline.arriving
        return FirstBadThing(
            found=self.found,
            tool="bisect_nan_backward",
            kind=kind,
            label=headline.op_label if headline is not None else None,
            label_status="final" if headline is not None and headline.op_label else "unavailable",
            module=headline.module if headline is not None else None,
            source_line=headline.source_line if headline is not None else None,
            backward_pass=self.backward_pass,
            coverage="complete"
            if self.verdict_scope.endswith("complete")
            else ("found_first_among_checked" if self.found else "partial"),
            uncertainty_zone=self.uncertainty_zone,
            detection_basis="saved_gradients",
            message=self.message,
        )


def _tensor_leaves(payload: Any, depth: int = 0) -> list[torch.Tensor]:
    """Collect tensor leaves from a saved gradient payload (None slots are ordinary)."""

    if depth > 6 or payload is None:
        return []
    if isinstance(payload, torch.Tensor):
        return [payload]
    if isinstance(payload, (list, tuple)):
        leaves: list[torch.Tensor] = []
        for item in payload:
            leaves.extend(_tensor_leaves(item, depth + 1))
        return leaves
    if isinstance(payload, dict):
        leaves = []
        for item in payload.values():
            leaves.extend(_tensor_leaves(item, depth + 1))
        return leaves
    return []


def _nonfinite_kind_of(leaves: list[torch.Tensor]) -> tuple[str, int, int]:
    """Classify non-finiteness across leaves; return (kind, checked, unchecked)."""

    has_nan = False
    has_inf = False
    checked = 0
    unchecked = 0
    for leaf in leaves:
        if not (torch.is_floating_point(leaf) or torch.is_complex(leaf)):
            checked += 1
            continue
        try:
            detached = leaf.detach()
            has_nan = has_nan or bool(torch.isnan(detached).any().item())
            has_inf = has_inf or bool(torch.isinf(detached).any().item())
            checked += 1
        except (RuntimeError, TypeError):
            unchecked += 1
    if has_nan and has_inf:
        return "nan+inf", checked, unchecked
    if has_nan:
        return "nan", checked, unchecked
    if has_inf:
        return "inf", checked, unchecked
    return "none", checked, unchecked


def _grad_fn_rows(trace: Trace) -> dict[str, Any]:
    """Index grad_fn records by label through one accessor (unstable spellings)."""

    return {getattr(node, "label", ""): node for node in getattr(trace, "grad_fns", ())}


def _fire_index(node: Any) -> int:
    """Return the measured fire-order position of a grad_fn node."""

    return int(getattr(node, "step_index", 0) or 0)


def _forward_join(trace: Trace, node: Any) -> tuple[str | None, str | None, str | None]:
    """Return (op_label, module, source_line) for a grad_fn's paired forward op."""

    op_label = getattr(node, "op_label", None)
    module = getattr(node, "module_address", None)
    source_line = None
    if isinstance(op_label, str) and op_label:
        try:
            op = trace[op_label]
        except Exception:  # noqa: BLE001 - the join is best-effort disclosure.
            op = None
        if op is not None:
            from ._common import _source_line

            source_line = _source_line(op)
    return (
        op_label if isinstance(op_label, str) and op_label else None,
        module if isinstance(module, str) and module else None,
        source_line,
    )


def _select_backward_pass(trace: Trace, bwd: int | None) -> tuple[Any, int]:
    """Resolve the one backward pass to walk, refusing ambiguity typed.

    Returns
    -------
    tuple[Any, int]
        ``(backward_pass_record, pass_index)``.

    Raises
    ------
    InvalidArgumentError
        ``backward_capture_required`` / ``bwd_selector_required`` /
        ``bwd_selector_unknown`` (passes never collapse).
    """

    from .._errors import InvalidArgumentError

    try:
        backward_passes = list(trace.backward_passes)
    except (ValueError, AttributeError):
        backward_passes = []
    if not backward_passes:
        raise InvalidArgumentError(
            "bisect_nan_backward needs a captured backward pass",
            code="backward_capture_required",
            remedy=(
                "re-trace with CaptureOptions(backward_ready=True, save_grads=...) "
                "and run trace.log_backward(loss) before bisecting"
            ),
        )
    if bwd is None:
        if len(backward_passes) > 1:
            raise InvalidArgumentError(
                f"this trace captured {len(backward_passes)} backward passes; "
                "pass bwd=N to select one (passes never collapse)",
                code="bwd_selector_required",
                remedy="pass bwd=1..N",
            )
        bwd = int(backward_passes[0].pass_index)
    selected = next((bp for bp in backward_passes if int(bp.pass_index) == int(bwd)), None)
    if selected is None:
        raise InvalidArgumentError(
            f"backward pass {bwd} was not captured "
            f"(captured: {[int(bp.pass_index) for bp in backward_passes]})",
            code="bwd_selector_unknown",
            remedy="pass one of the captured pass indexes",
        )
    return selected, int(bwd)


class _LedgerWalk:
    """Mutable bookkeeping for one fire-order ledger walk."""

    def __init__(self) -> None:
        """Start an empty walk."""

        self.transitions: list[BackwardTransition] = []
        self.healed: list[str] = []
        self.uncertainty: list[str] = []
        self.birth: BackwardTransition | None = None
        self.unchecked_before_birth = False
        self.checked_calls = 0
        self.unchecked_calls = 0
        self.any_nonfinite = False

    def add(self, transition: BackwardTransition) -> None:
        """Fold one classified transition into the walk state."""

        self.transitions.append(transition)
        if transition.verdict == "unchecked" and transition.checked_leaves == 0:
            self.unchecked_calls += 1
            self.uncertainty.append(transition.grad_fn_label)
            if self.birth is None:
                self.unchecked_before_birth = True
            return
        self.checked_calls += 1
        if transition.unchecked_leaves:
            self.uncertainty.append(transition.grad_fn_label)
        if transition.arriving not in ("none", "unchecked") or transition.emitted not in (
            "none",
            "unchecked",
        ):
            self.any_nonfinite = True
        if transition.verdict == "healed":
            self.healed.append(transition.grad_fn_label)
        if transition.verdict == "birth" and self.birth is None:
            self.birth = transition
        elif self.birth is None and transition.unchecked_leaves > 0:
            self.unchecked_before_birth = True


def _classify_call(trace: Trace, nodes_by_label: dict[str, Any], call: Any) -> BackwardTransition:
    """Classify one GradFnCall into its five-word transition verdict."""

    label = str(getattr(call, "label", ""))
    node = nodes_by_label.get(label)
    fire_index = _fire_index(node) if node is not None else 0
    grad_fn_class = str(getattr(node, "class_name", "")) if node is not None else ""
    op_label, module, source_line = (
        _forward_join(trace, node) if node is not None else (None, None, None)
    )
    arriving_leaves = _tensor_leaves(getattr(call, "grad_outputs", None))
    emitted_leaves = _tensor_leaves(getattr(call, "grad_inputs", None))
    if not arriving_leaves and not emitted_leaves:
        return BackwardTransition(
            fire_index=fire_index,
            grad_fn_label=label,
            grad_fn_class=grad_fn_class,
            op_label=op_label,
            module=module,
            source_line=source_line,
            verdict="unchecked",
            arriving="unchecked",
            emitted="unchecked",
            checked_leaves=0,
            unchecked_leaves=0,
        )
    arriving_kind, arriving_checked, arriving_unchecked = _nonfinite_kind_of(arriving_leaves)
    emitted_kind, emitted_checked, emitted_unchecked = _nonfinite_kind_of(emitted_leaves)
    unchecked = arriving_unchecked + emitted_unchecked
    arrived_bad = arriving_kind != "none"
    emitted_bad = emitted_kind != "none"
    if not arrived_bad and not emitted_bad:
        # ``clean`` REQUIRES complete checkable coverage at this call.
        verdict = "clean" if unchecked == 0 else "unchecked"
    elif arrived_bad and emitted_bad:
        verdict = "propagated"
    elif arrived_bad and not emitted_bad:
        verdict = "healed"
    elif arriving_leaves and unchecked == 0:
        # Finite CHECKED arrivals, non-finite emission: a provable birth.
        verdict = "birth"
    else:
        # Non-finite emitted with NO (or incompletely checked) arriving
        # gradients: the seed itself may have been non-finite, or an
        # unchecked arrival may have carried it in -- never a birth claim.
        verdict = "root_seed"
    return BackwardTransition(
        fire_index=fire_index,
        grad_fn_label=label,
        grad_fn_class=grad_fn_class,
        op_label=op_label,
        module=module,
        source_line=source_line,
        verdict=verdict,
        arriving=arriving_kind,
        emitted=emitted_kind,
        checked_leaves=arriving_checked + emitted_checked,
        unchecked_leaves=unchecked,
    )


def _compose_message(
    walk: _LedgerWalk, *, verdict_scope: str, bwd: int, amp_hint: str | None
) -> str:
    """Compose the human-readable summary headlining the earliest provable birth."""

    birth = walk.birth
    found = walk.any_nonfinite
    healed = walk.healed
    unchecked_calls = walk.unchecked_calls
    if birth is not None:
        where = f" ({birth.source_line})" if birth.source_line else ""
        forward_name = birth.op_label or "an uncaptured forward op"
        qualifier = (
            ""
            if verdict_scope == "complete"
            else " (first among CHECKED fires; earlier "
            "unchecked fires exist -- see uncertainty_zone)"
        )
        healed_note = (
            f" {len(healed)} downstream fire(s) healed non-finite arrivals; a finite loss "
            "gradient does not prove a clean backward."
            if healed
            else ""
        )
        message = (
            f"Non-finite gradient born in the backward of {forward_name}{where} "
            f"[{birth.grad_fn_class}, fire #{birth.fire_index}, pass {bwd}]{qualifier}."
            f"{healed_note}"
        )
    elif found:
        message = (
            f"Non-finite gradients observed in pass {bwd} but no birth is provable from "
            "the saved payloads (root-seed or propagated-only evidence); see transitions."
        )
    else:
        message = f"No non-finite gradients among checked fires in pass {bwd}." + (
            f" {unchecked_calls} fire(s) were unchecked (save_grads did not select them);"
            " a clean verdict over incomplete coverage is not absolute."
            if unchecked_calls
            else ""
        )
    if amp_hint is not None:
        message += f" NOTE: {amp_hint}."
    return message


def bisect_nan_backward(
    trace: Trace,
    *,
    bwd: int | None = None,
    grad_scale: float | None = None,
) -> BackwardNanResult:
    """Locate where a backward pass's NaN/Inf gradients were BORN.

    Post-hoc over one logged backward; no rerun. Requires a trace captured
    with ``save_grads`` and a logged backward pass (the
    ``trace + save_grads + captured backward`` tier).

    Parameters
    ----------
    trace:
        Completed TorchLens trace with a captured backward.
    bwd:
        One-based backward pass selector; required when multiple passes were
        captured (the ``gradient_flow_audit`` convention verbatim).
    grad_scale:
        Disclosed AMP loss scale (``scaler.get_scale()``); recorded on the
        result. The transition ledger classifies non-finiteness, which no
        positive scale changes, so this is DISCLOSURE, not arithmetic.

    Returns
    -------
    BackwardNanResult
        The full fire-order transition ledger with the earliest provable
        birth headlined.

    Raises
    ------
    InvalidArgumentError
        If ``bwd`` is required and missing (code ``bwd_selector_required``),
        names an unknown pass (code ``bwd_selector_unknown``), or
        ``grad_scale`` is not a positive finite number (code
        ``grad_scale_invalid``).
    """

    from ._gradients import _validate_grad_scale

    grad_scale = _validate_grad_scale(grad_scale)
    selected, bwd = _select_backward_pass(trace, bwd)

    nodes_by_label = _grad_fn_rows(trace)
    calls = sorted(
        selected.grad_fn_calls,
        key=lambda call: _fire_index(nodes_by_label.get(getattr(call, "label", ""), None) or call),
    )

    walk = _LedgerWalk()
    for call in calls:
        walk.add(_classify_call(trace, nodes_by_label, call))
    transitions = walk.transitions
    healed = walk.healed
    uncertainty = walk.uncertainty
    birth = walk.birth
    unchecked_before_birth = walk.unchecked_before_birth
    checked_calls = walk.checked_calls
    unchecked_calls = walk.unchecked_calls

    found = walk.any_nonfinite
    if found:
        verdict_scope = "complete" if not unchecked_before_birth else "found_first_among_checked"
    else:
        verdict_scope = "clean_complete" if unchecked_calls == 0 else "clean_among_checked"

    amp_hint: str | None = None
    if found and grad_scale is None:
        # The GradScaler overflow signature is an INF-kind FIRST bad thing
        # (downstream inf*0 legitimately cascades into nan, so later kinds
        # cannot disqualify the disclosure). A hint, never a verdict.
        first_bad_kind = next(
            (
                kind
                for t in transitions
                for kind in (t.arriving, t.emitted)
                if kind not in ("none", "unchecked")
            ),
            None,
        )
        if first_bad_kind == "inf":
            amp_hint = amp_scaled_gradients_hint(all_nonfinite=True)

    message = _compose_message(walk, verdict_scope=verdict_scope, bwd=bwd, amp_hint=amp_hint)

    return BackwardNanResult(
        found=found,
        birth=birth,
        verdict_scope=verdict_scope,
        transitions=tuple(transitions),
        healed=tuple(healed),
        backward_pass=int(bwd),
        checked_calls=checked_calls,
        unchecked_calls=unchecked_calls,
        uncertainty_zone=tuple(uncertainty),
        grad_scale=grad_scale,
        amp_hint=amp_hint,
        message=message,
    )
