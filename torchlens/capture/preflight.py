"""Address preflight: price a selective-save spelling BEFORE the forward (leverage B9).

The selector compiler already knows which ``layers_to_save`` spellings
resolve LIVE during the forward (module paths, bare type names, negative
ordinals, ``save=`` predicates — one site at near save-nothing cost) and
which need FINAL graph numbering (labels like ``conv2d_5_15``, positive
integer ordinals, ``output``/``identity`` prefixes) or join ``tl.module`` with
a non-module term by ``|`` (every candidate payload escrows until postprocess,
the whole-graph candidate-class escrow). This
module publishes that verdict as a user-facing preflight, and — given a
finished trace of the same model — NAMES the equivalent live-resolvable
spelling for each deferred component ("``conv2d_5_15`` costs a whole-graph
escrow; ``layer1.1.conv2`` is the same site and costs nothing").

No hooks-based tool has a selector compiler to ask, and no static-graph tool
has a capture to price. Every spelling here is DOCUMENTED-UNSTABLE pending
naming-session ratification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._trace_selector_helpers import _selector_component_is_final_only

__all__ = ["AddressPreflight", "AddressVerdict", "address_preflight", "cone_preflight"]


@dataclass(frozen=True)
class AddressVerdict:
    """One selection component's resolvability verdict.

    ``resolution`` is ``"live"`` (resolves during the forward; near
    save-nothing cost) or ``"deferred"`` (needs final graph numbering, or
    joins ``tl.module`` with a non-module term by ``|``; enters the
    candidate-class escrow). ``equivalent_spelling`` names a
    live-resolvable module-path spelling for the SAME site where a trace was
    supplied and the component matched exactly one module-owned site;
    ``matched_labels`` is the trace-side disclosure of what the component
    addresses.
    """

    component: Any
    resolution: str
    reason: str
    equivalent_spelling: str | None = None
    matched_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class AddressPreflight:
    """The settled preflight: per-component verdicts plus the escrow summary."""

    verdicts: tuple[AddressVerdict, ...]
    escrowing: bool

    @property
    def deferred(self) -> tuple[AddressVerdict, ...]:
        """The components that would enter the escrow lane."""

        return tuple(verdict for verdict in self.verdicts if verdict.resolution == "deferred")

    def __repr__(self) -> str:
        lane = "whole-graph candidate-class escrow" if self.escrowing else "live single-pass"
        return f"AddressPreflight({len(self.verdicts)} components, lane: {lane})"


def _classify(component: Any) -> tuple[str, str]:
    """Classify one public selection component (live vs deferred, with reason)."""

    if isinstance(component, int) and not isinstance(component, bool) and component < 0:
        # Negative ordinals resolve through a BOUNDED rolling window — the
        # compiler marks them final-only for numbering, but their escrow is
        # windowed, never the whole-graph candidate class.
        return "live", "negative ordinals resolve through a bounded rolling window"
    if _selector_component_is_final_only(component):
        if isinstance(component, int):
            reason = "positive integer ordinals are FINAL layer numbers"
        elif isinstance(component, str) and component.startswith(("output", "identity")):
            reason = "output/identity spellings exist only after final numbering"
        else:
            reason = "indexed labels embed final type indexes renumbered after capture"
        return "deferred", reason
    if isinstance(component, str):
        return "live", "module paths and bare type names resolve during the forward"
    if isinstance(component, int):
        return "live", "negative ordinals resolve through a bounded rolling window"
    from ..intervention.selectors import BaseSelector
    from ..ir.selector_eval import module_exit_bound_addresses, selector_contains_kind

    if isinstance(component, BaseSelector) and selector_contains_kind(component, "module"):
        # Any selector naming a module resolves post hoc; one whose matches all
        # lie among module-pass outputs limits its escrow to those passes.
        if module_exit_bound_addresses(component) is None:
            verdict = ("deferred", _UNBOUNDED_MODULE_REASON)
        else:
            verdict = ("live", "tl.module selectors escrow only the ops inside matching passes")
        return verdict
    return "live", "predicate selectors resolve per-op during the forward"


_UNBOUNDED_MODULE_REASON = (
    "joining tl.module with a non-module term by | (or negating it) escrows every op"
)


def _escrow_remedy(selector: Any, module_exit_addresses: tuple[str, ...] | None) -> str:
    """Explain why a deferred selection escrows and name the live spelling that avoids it.

    Parameters
    ----------
    selector:
        The capture's deferred retention selector: a selector object, one
        ``layers_to_save`` component, or a list of either.
    module_exit_addresses:
        Covering module addresses when the escrow is already limited to the
        ops inside matching module passes, else ``None``.

    Returns
    -------
    str
        Reason sentences ending in a ``Remedy:`` sentence, for the
        ``escrow_budget_exceeded`` refusal.
    """

    if module_exit_addresses is not None:
        address = module_exit_addresses[0].rsplit(":", 1)[0]
        return (
            "The escrow already holds only the ops inside matching passes of "
            f"{', '.join(module_exit_addresses)}, and those ops alone exceed the bound. "
            "Remedy: when the module's output is produced in its own forward body, save "
            f"that body live with save=tl.where(lambda ctx: ctx.address == {address!r}) "
            f"and read trace.modules[{address!r}].out (no escrow); otherwise select a "
            "smaller submodule or one pass (tl.module('<address>:<pass>'))."
        )
    components = list(selector) if isinstance(selector, (list, tuple)) else [selector]
    verdicts = [(item, _classify(item)) for item in components]
    final_only = [
        item
        for item, (resolution, _reason) in verdicts
        if resolution == "deferred" and _selector_component_is_final_only(item)
    ]
    unbounded = [item for item, verdict in verdicts if verdict[1] == _UNBOUNDED_MODULE_REASON]
    parts: list[str] = []
    remedies: list[str] = []
    if final_only:
        parts.append(
            f"{final_only!r} need FINAL graph numbering (op labels, positive ordinals, "
            "output spellings), so every op's output is escrowed until postprocess."
        )
        remedies.append(
            "address those sites by module path (save=tl.module('<address>') or the "
            "string '<address>'), bare op type, or negative ordinal; "
            "torchlens.capture.preflight.address_preflight(layers, trace) on a finished "
            "trace names the module path for each label"
        )
    if unbounded:
        parts.append(
            f"{', '.join(repr(item) for item in unbounded)} joins tl.module with a "
            "non-module term by | (or negates it), so every op's output is escrowed."
        )
        remedies.append(
            "keep the module part to tl.module terms joined by | or intersected with "
            "other terms by & (escrow then stays inside the matching module passes), "
            "and save the other sites in a separate capture with a live selector "
            "such as tl.func(name) or tl.in_module(path)"
        )
    if not parts:
        return (
            "Every candidate op's output is escrowed until postprocess resolves the "
            "selector. Remedy: address the same sites with a live spelling, such as "
            "tl.module(path), tl.func(name), a module-path or bare type-name string, "
            "or a save= predicate."
        )
    return " ".join(parts) + " Remedy: " + "; ".join(remedies) + "."


def _module_equivalent(trace: Any, component: Any) -> tuple[str | None, tuple[str, ...]]:
    """Name the module-path spelling for one deferred component on ``trace``."""

    matched: list[str] = []
    try:
        layer = trace[component]
    except Exception:
        return None, ()
    for op in getattr(layer, "ops", None) or [layer]:
        label = getattr(op, "label", None)
        if isinstance(label, str):
            matched.append(label)
    modules = tuple(getattr(layer, "modules", ()) or ())
    if not modules:
        return None, tuple(matched)
    # 'conv1:1' -> 'conv1': the pass-free module path IS a live spelling.
    innermost = str(modules[-1]).rsplit(":", 1)[0]
    return (innermost or None), tuple(matched)


def address_preflight(layers_to_save: Any, trace: Any = None) -> AddressPreflight:
    """Price a ``layers_to_save`` selection before running the capture.

    Parameters
    ----------
    layers_to_save:
        The public selection — one component or a sequence of components
        (labels, ordinals, module paths, type names, predicates).
    trace:
        Optional FINISHED trace of the same model. When given, each deferred
        component's verdict names the equivalent live-resolvable module-path
        spelling where one exists, and discloses the labels it addresses.

    Returns
    -------
    AddressPreflight
        Per-component verdicts plus whether the selection as a whole enters
        the whole-graph candidate-class escrow lane.
    """

    if layers_to_save is None or isinstance(layers_to_save, (str, int)) or callable(layers_to_save):
        components: list[Any] = [layers_to_save] if layers_to_save is not None else []
    else:
        try:
            components = list(layers_to_save)
        except TypeError:
            components = [layers_to_save]
    verdicts: list[AddressVerdict] = []
    for component in components:
        resolution, reason = _classify(component)
        equivalent: str | None = None
        matched: tuple[str, ...] = ()
        if resolution == "deferred" and trace is not None:
            equivalent, matched = _module_equivalent(trace, component)
        verdicts.append(
            AddressVerdict(
                component=component,
                resolution=resolution,
                reason=reason,
                equivalent_spelling=equivalent,
                matched_labels=matched,
            )
        )
    escrowing = any(verdict.resolution == "deferred" for verdict in verdicts)
    return AddressPreflight(verdicts=tuple(verdicts), escrowing=escrowing)


def cone_preflight(trace: Any, origins: Any) -> dict[str, Any]:
    """Report the EXACT planned replay work before running it (leverage B11/D-12).

    The returned mapping discloses the replay lane's planned work set — the
    exact pass-qualified labels ``push``/``do`` would recompute for these
    origins (the preflight-equals-actual oracle is pinned by test) — beside
    the structural facts a user needs to price the run: the live lane always
    re-executes the FULL forward regardless of cone size, replay requires an
    ``intervention_ready=True`` capture (a disclosed ~3x capture-time premium
    on CPU transformers), and reachability is NEVER numeric change (movers
    are a subset of the cone, not equal to it). No wall-clock number is ever
    predicted — a host-specific time band is the caller's own measurement.

    Parameters
    ----------
    trace:
        Finished capture whose recorded graph prices the plan.
    origins:
        Origin op records (the same operand contract as
        :func:`cone_of_effect`).

    Returns
    -------
    dict[str, Any]
        ``planned_replay_labels`` (exact, execution order), ``cone_size``,
        ``total_ops``, ``live_lane_ops`` (always the full graph),
        ``replay_ready`` (whether this capture can run the replay lane), and
        ``notes`` (the engine-assumption disclosures).
    """

    from ..intervention.replay import cone_of_effect

    cone = cone_of_effect(trace, origins)
    planned = tuple(op.label for op in cone)
    total_ops = len(list(trace.op_labels))
    return {
        "planned_replay_labels": planned,
        "cone_size": len(planned),
        "total_ops": total_ops,
        "live_lane_ops": total_ops,
        "replay_ready": bool(getattr(trace, "intervention_ready", False)),
        "notes": (
            "replay recomputes exactly the planned labels; the live lane "
            "re-executes the full forward regardless of cone size",
            "intervention_ready=True capture costs a measured ~3x capture "
            "wall-time premium on CPU transformers (disclose it in any "
            "replay journey)",
            "reachability is never numeric change: movers are a subset of "
            "the cone, never asserted equal to it",
        ),
    }
