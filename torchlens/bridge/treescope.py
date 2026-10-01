"""The treescope bridge: external registration, probe, sniff, door (B4, F16).

Makes TorchLens objects render boundedly inside Google DeepMind's treescope
notebook renderer (treescope memo decision 1). Design rules, each earned by
a panel measurement:

- **External registration, all treescope symbols in THIS module.** Core /
  notebook imports never import treescope; the dependency is an extra,
  never hard. Explicit idempotent :func:`register`, scoped
  :func:`registered`, :func:`unregister` by exact handler identity, typed
  conflict refusals on occupied slots, :func:`status` reporting
  version / handlers / tensor capability / suppression state. NEVER
  ``register_as_default()``; NEVER register or replace ``torch.Tensor``
  (treescope owns that slot).
- **Hybrid activation (memo 3.2)**: thin lazy ``__treescope_repr__``
  methods live on ``Trace`` and ``PartialTrace`` ONLY (the two dataclasses
  treescope reflects into megabyte dumps; the method route is the only one
  that fires with zero registration) and delegate to
  :func:`treescope_repr` here; ``Layer``/``Op`` ride the registry. Under
  ``register_as_default()`` at TOP LEVEL the methods are inert because our
  ``_repr_html_`` wins first (F-B) -- they protect the EXPLICIT and NESTED
  render paths.
- **Container rule (memo 3.4, hard)**: a container handler emits scalars,
  strings, and a budgeted foldable index of lookup keys with
  ``shown K of N``; it NEVER passes child TL objects to the subtree
  renderer -- the recursive version measured WORSE than no bridge (F-D).
- **Tensor gate (memo 3.5)**: a tensor leaf renders only when saved,
  resident, non-meta; the adapter is probed BEHAVIORALLY at first use
  (F-A: released 0.1.10 raises on every torch>=2.13 tensor). Healthy: the
  REAL tensor passes through (bounded by construction, F-F). Broken:
  strict degradation naming the reason and pointing at the native card.
- **The repaired duplicate-box sniff (memo 3.6)**: treescope appends any
  object's ``_repr_html_`` in a collapsed "Rich HTML representation" box
  AFTER our handler renders. The cards return a one-line sentinel instead
  of the full card exactly when BOTH (a) a treescope repr-HTML
  postprocessor frame is on the call stack (NARROW predicate:
  ``repr_html_postprocessor`` / ``object_inspection`` modules -- the broad
  "any treescope frame" version measured 26 bytes where the user's card
  should be, because the kernel formatter calls us from inside treescope's
  package) and (b) the bridge recorded ``id(node)`` in a one-shot slot
  AFTER successfully building its rendering (without this, a broken
  handler suppresses the user's only surviving good rendering). The
  predicate couples to private treescope module names and FAILS SAFE (the
  duplicate quietly returns) if upstream renames them.

Every public spelling is DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

import contextlib
import sys
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from .._errors import InvalidArgumentError, MissingDependencyError

__all__ = [
    "BRIDGE_LEDGER",
    "BridgeStatus",
    "consume_repr_suppression",
    "disabled",
    "display",
    "probe_tensor_support",
    "register",
    "registered",
    "status",
    "treescope_repr",
    "unregister",
]

#: The three-way bucket ledger (memo 3.3): a CHECKED-IN decision per
#: exported dataclass type, never a blanket rule. RICH types carry full
#: cards (methods or registry handlers); ONE_LINE types render as
#: ``<ClassName: one-line summary>`` (initial assignments are PROVISIONAL
#: per the memo -- each needs a measured byte row at implementation before
#: its bucket is checked in); everything else is LEAVE-TO-TREESCOPE, with
#: the CI audit pinning LEAVE dataclasses bounded. Keys are dotted import
#: paths; absent optional types are skipped at register() and disclosed in
#: :func:`status`.
BRIDGE_LEDGER: dict[str, str] = {
    "torchlens.data_classes.trace.Trace": "rich",
    "torchlens.partial.PartialTrace": "rich",
    "torchlens.data_classes.layer.Layer": "rich",
    "torchlens.data_classes.op.Op": "rich",
    "torchlens.data_classes.module.Module": "one_line",
    "torchlens.data_classes.module.ModuleCall": "one_line",
    "torchlens.fastlog.types.Recording": "one_line",
    "torchlens.runnable.RunResult": "one_line",
    "torchlens.data_classes.grad_fn.GradFn": "one_line",
    "torchlens.data_classes.grad_fn.GradFnCall": "one_line",
}

#: Budget for the foldable key index a bridged Trace emits (strings only).
_KEY_INDEX_BUDGET = 20


@dataclass
class _BridgeState:
    """Process-local bridge state (single-threaded by design)."""

    installed: dict[type, Any] = field(default_factory=dict)
    disabled_depth: int = 0
    #: One-shot post-success slot for the duplicate-box sniff (memo 3.6).
    last_render_id: int | None = None
    probe_cache: tuple[bool, str | None] | None = None
    skipped_types: tuple[str, ...] = ()
    degraded_warned: bool = False


_STATE = _BridgeState()


@dataclass(frozen=True)
class BridgeStatus:
    """One :func:`status` snapshot.

    Attributes
    ----------
    treescope_importable / treescope_version:
        Whether/which treescope import resolves.
    registered:
        Whether our registry handlers are currently installed.
    handlers:
        Qualified names of the types we hold registry slots for.
    skipped_types:
        Ledger types that could not be resolved at register time.
    tensor_capability:
        Behavioral probe verdict (``None`` = not probed yet).
    tensor_capability_reason:
        Probe failure detail, when broken.
    suppression_armed:
        Whether the one-shot duplicate-suppression slot currently holds an
        object id.
    disabled:
        Whether a :func:`disabled` scope is active.
    """

    treescope_importable: bool
    treescope_version: str | None
    registered: bool
    handlers: tuple[str, ...]
    skipped_types: tuple[str, ...]
    tensor_capability: bool | None
    tensor_capability_reason: str | None
    suppression_armed: bool
    disabled: bool


def _import_treescope() -> Any:
    """Import treescope or raise the typed dependency refusal."""

    try:
        import treescope
    except ImportError as error:
        raise MissingDependencyError(
            "the treescope bridge requires the optional 'treescope' package, "
            "which is not installed",
            code="treescope_bridge_unavailable",
            remedy='pip install treescope (or the "torchlens[treescope]" extra once published)',
        ) from error
    return treescope


def probe_tensor_support() -> tuple[bool, str | None]:
    """Behaviorally probe the treescope torch adapter (memo 3.5 / F-A).

    Released treescope 0.1.10 reads the removed ``Tensor.names`` attribute
    and raises on every torch>=2.13 tensor, so capability is PROBED on a
    tiny tensor at first use -- never version-pinned. The verdict is cached
    per process.

    Returns
    -------
    tuple[bool, str | None]
        ``(True, None)`` when the adapter renders, else ``(False, reason)``.
    """

    if _STATE.probe_cache is not None:
        return _STATE.probe_cache
    treescope = _import_treescope()
    try:
        import torch

        treescope.render_to_html(torch.zeros((2, 2)), ignore_exceptions=False, compressed=False)
        _STATE.probe_cache = (True, None)
    except Exception as error:  # noqa: BLE001 - the probe verdict IS the product
        _STATE.probe_cache = (
            False,
            f"{type(error).__name__}: {error}",
        )
    return _STATE.probe_cache


def _degradation_note() -> str:
    """Strict-degradation message naming the reason and the native door."""

    _probed, reason = _STATE.probe_cache or (None, None)
    detail = reason or "treescope tensor adapter unavailable"
    return (
        f"tensor view unavailable via treescope ({detail}); upgrade treescope, "
        "or view natively via the TorchLens card (_repr_html_)"
    )


def _resolve_ledger_type(dotted: str) -> type | None:
    """Resolve one ledger row's dotted path to a class, or ``None``."""

    module_name, _, attribute = dotted.rpartition(".")
    try:
        import importlib

        module = importlib.import_module(module_name)
        resolved = getattr(module, attribute)
    except Exception:  # noqa: BLE001 - optional rows skip, disclosed in status()
        return None
    return resolved if isinstance(resolved, type) else None


def register() -> None:
    """Install the bridge's registry handlers (idempotent).

    ``Layer``/``Op`` get RICH handlers; ONE_LINE ledger rows get bounded
    one-line handlers. ``Trace``/``PartialTrace`` ride their
    ``__treescope_repr__`` methods and are never registry rows. A foreign
    handler already occupying a slot raises the typed conflict -- the
    bridge never silently replaces user state.

    Raises
    ------
    MissingDependencyError
        ``treescope_bridge_unavailable`` when treescope is not installed.
    InvalidArgumentError
        ``treescope_slot_occupied`` when a foreign handler holds a slot.
    """

    treescope = _import_treescope()
    registry = treescope.type_registries.TREESCOPE_HANDLER_REGISTRY
    skipped: list[str] = []
    planned: list[tuple[type, Any]] = []
    for dotted, bucket in BRIDGE_LEDGER.items():
        if bucket == "rich" and dotted.rsplit(".", 1)[-1] in ("Trace", "PartialTrace"):
            continue  # method-activated (memo 3.2), never registry rows
        target = _resolve_ledger_type(dotted)
        if target is None:
            skipped.append(dotted)
            continue
        handler = _rich_handler if bucket == "rich" else _one_line_handler
        existing = registry.get(target)
        if existing is not None and existing is not _STATE.installed.get(target):
            raise InvalidArgumentError(
                f"treescope handler slot for {dotted} is already occupied by a foreign handler",
                code="treescope_slot_occupied",
                remedy="unregister the existing handler first, or render through "
                "torchlens.bridge.treescope.display() without registering",
                occupied_type=dotted,
            )
        planned.append((target, handler))
    for target, handler in planned:
        registry[target] = handler
        _STATE.installed[target] = handler
    _STATE.skipped_types = tuple(skipped)


def unregister() -> None:
    """Remove exactly OUR handlers by identity; foreign handlers survive."""

    try:
        treescope = _import_treescope()
    except MissingDependencyError:
        _STATE.installed.clear()
        return
    registry = treescope.type_registries.TREESCOPE_HANDLER_REGISTRY
    for target, handler in list(_STATE.installed.items()):
        if registry.get(target) is handler:
            del registry[target]
        del _STATE.installed[target]


@contextlib.contextmanager
def registered() -> Iterator[None]:
    """Scoped registration: register on entry, unregister on exit.

    Entering with the bridge already registered leaves it registered on
    exit (the scope only undoes its own installation).
    """

    was_registered = bool(_STATE.installed)
    register()
    try:
        yield
    finally:
        if not was_registered:
            unregister()


@contextlib.contextmanager
def disabled() -> Iterator[None]:
    """Scope in which every bridge hook falls through to treescope defaults."""

    _STATE.disabled_depth += 1
    try:
        yield
    finally:
        _STATE.disabled_depth -= 1


def status() -> BridgeStatus:
    """Report the bridge's live state (memo 3.1)."""

    try:
        import treescope

        importable, version = True, getattr(treescope, "__version__", "unknown")
    except ImportError:
        importable, version = False, None
    probed = _STATE.probe_cache
    return BridgeStatus(
        treescope_importable=importable,
        treescope_version=version,
        registered=bool(_STATE.installed),
        handlers=tuple(sorted(t.__qualname__ for t in _STATE.installed)),
        skipped_types=_STATE.skipped_types,
        tensor_capability=None if probed is None else probed[0],
        tensor_capability_reason=None if probed is None else probed[1],
        suppression_armed=_STATE.last_render_id is not None,
        disabled=_STATE.disabled_depth > 0,
    )


# ---------------------------------------------------------------------------
# Handlers (registry route: Layer / Op rich, ledger one-liners)


def _record_success(node: Any) -> None:
    """Arm the one-shot suppression slot AFTER a successful build (3.6b)."""

    _STATE.last_render_id = id(node)


def _rich_handler(node: Any, path: str | None, subtree_renderer: Any) -> Any:
    """RICH registry handler for Layer/Op: bounded identity + stats + keys.

    Container rule: strings and scalars only; the ONE non-TL child ever
    passed to ``subtree_renderer`` is the saved TENSOR payload, and only
    when the behavioral probe says the adapter can render it (bounded by
    construction, F-F). Broken adapter: strict degradation text.
    """

    if _STATE.disabled_depth > 0:
        return NotImplemented
    try:
        from treescope import rendering_parts as rp

        lines = _record_summary_lines(node)
        children: list[Any] = [rp.text(line) for line in lines]
        children.extend(_tensor_leaf(node, path, subtree_renderer))
        rendering = rp.build_foldable_tree_node_from_children(
            prefix=f"<TorchLens {type(node).__name__} ",
            children=children,
            suffix=">",
            path=path,
        )
    except Exception:  # noqa: BLE001 - a broken handler degrades, never crashes
        return NotImplemented
    _record_success(node)
    return rendering


def _one_line_handler(node: Any, path: str | None, subtree_renderer: Any) -> Any:
    """ONE_LINE registry handler: ``<ClassName: bounded public summary>``."""

    if _STATE.disabled_depth > 0:
        return NotImplemented
    try:
        from treescope import rendering_parts as rp

        summary = _bounded_summary(node)
        rendering = rp.build_one_line_tree_node(f"<{type(node).__name__}: {summary}>", path=path)
    except Exception:  # noqa: BLE001 - degrade to treescope's default
        return NotImplemented
    _record_success(node)
    return rendering


def _bounded_summary(node: Any) -> str:
    """One bounded summary line with no ``object at 0x`` id leaks."""

    try:
        text = str(node).splitlines()[0]
    except Exception:  # noqa: BLE001 - summaries degrade, never raise
        text = type(node).__name__
    if " at 0x" in text:
        text = text.split(" at 0x", 1)[0] + ">"
    return text[:200]


def _record_summary_lines(record: Any) -> list[str]:
    """Bounded identity/stats/context lines for a Layer/Op handler."""

    lines: list[str] = []
    label = getattr(record, "label", None) or getattr(record, "layer_label", None)
    if label:
        lines.append(f"label: {label}")
    shape = getattr(record, "shape", None)
    dtype = getattr(record, "dtype", None)
    if shape is not None:
        lines.append(f"shape: {tuple(shape)} {dtype}")
    num_passes = getattr(record, "num_passes", None)
    if num_passes and num_passes > 1:
        lines.append(f"passes: {num_passes}")
    parents = tuple(getattr(record, "parents", ()) or ())[:6]
    if parents:
        lines.append("parents: " + ", ".join(repr(str(p)) for p in parents))
    children = tuple(getattr(record, "children", ()) or ())[:6]
    if children:
        lines.append("children: " + ", ".join(repr(str(c)) for c in children))
    return lines


def _tensor_leaf(record: Any, path: str | None, subtree_renderer: Any) -> list[Any]:
    """The tensor gate (memo 3.5): real tensor through, or a named reason."""

    from treescope import rendering_parts as rp

    if not getattr(record, "has_saved_activation", False):
        return []
    try:
        import torch

        tensor = getattr(record, "out", None)
        if type(tensor) is not torch.Tensor or tensor.is_meta:
            return [rp.text("value not resident (disk/offloaded payload)")]
    except Exception:  # noqa: BLE001 - unreadable payloads degrade
        return []
    supported, _reason = probe_tensor_support()
    if not supported:
        _warn_degraded_once()
        return [rp.text(_degradation_note())]
    child_path = f"{path}.out" if path else None
    return [subtree_renderer(tensor, path=child_path)]


def _warn_degraded_once() -> None:
    """Warn once per process when the bridge enters strict degradation.

    Delivery is BEST-EFFORT: under ``-W error`` promotion the warn call
    raises, and a courtesy signal may never sink the rendering it
    accompanies (the binding disclosures are the in-render degradation
    text and :func:`status`), so the raise is swallowed here.
    """

    if _STATE.degraded_warned:
        return
    _STATE.degraded_warned = True
    from ..errors._base import TorchLensWarning

    _probed, reason = _STATE.probe_cache or (None, None)
    # warnings-as-errors must not kill the render the warning accompanies.
    with contextlib.suppress(Exception):
        warnings.warn(
            TorchLensWarning(
                "treescope bridge degraded: the installed treescope cannot render "
                f"torch tensors ({reason}); object cards render without interactive "
                "array views. Remedy: upgrade treescope to a release containing the "
                "Tensor.names fix, or view tensors natively via the TorchLens card",
                code="treescope_bridge_degraded",
            ),
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Method route (Trace / PartialTrace __treescope_repr__ bodies)


def treescope_repr(obj: Any, path: str | None, subtree_renderer: Any) -> Any:
    """Body of the thin ``__treescope_repr__`` methods (memo 3.2).

    Emits identity facts and a budgeted foldable index of lookup keys with
    ``shown K of N`` (container rule 3.4: strings only, never child TL
    objects). Falls through to ``NotImplemented`` on the disabled scope or
    any fault.
    """

    if _STATE.disabled_depth > 0:
        return NotImplemented
    try:
        from treescope import rendering_parts as rp

        facts = _trace_fact_lines(obj)
        children: list[Any] = [rp.text(line) for line in facts]
        keys = [str(k) for k in (getattr(obj, "layer_logs", None) or ())][:64]
        if keys:
            shown = keys[:_KEY_INDEX_BUDGET]
            index_children = [rp.text(f"['{key}']") for key in shown]
            index_children.append(rp.text(f"shown {len(shown)} of {len(keys)}"))
            children.append(
                rp.build_foldable_tree_node_from_children(
                    prefix="lookup keys (",
                    children=index_children,
                    suffix=")",
                )
            )
        rendering = rp.build_foldable_tree_node_from_children(
            prefix=f"<TorchLens {type(obj).__name__} ",
            children=children,
            suffix=">",
            path=path,
        )
    except Exception:  # noqa: BLE001 - degrade to treescope's default rendering
        return NotImplemented
    _record_success(obj)
    return rendering


def _trace_fact_lines(obj: Any) -> list[str]:
    """Bounded fact lines for the Trace/PartialTrace method route."""

    lines = [_bounded_summary(obj)]
    outcome = getattr(obj, "outcome", None)
    outcome_status = getattr(getattr(outcome, "status", None), "name", None)
    if outcome_status:
        lines.append(f"outcome: {outcome_status}")
    num_ops = getattr(obj, "num_ops", None)
    if num_ops is not None:
        lines.append(f"ops: {num_ops} (saved: {getattr(obj, 'num_saved_ops', '?')})")
    return lines


# ---------------------------------------------------------------------------
# The repaired duplicate-box sniff (memo 3.6)

#: NARROW frame predicate: only these treescope-private module names count.
#: Broad "any treescope frame" measured 26 bytes where the user's card
#: should be (the kernel formatter calls _repr_html_ from inside treescope).
_POSTPROCESSOR_MODULE_MARKERS = ("repr_html_postprocessor", "object_inspection")


def _postprocessor_frame_on_stack() -> bool:
    """Whether a treescope repr-HTML postprocessor frame is on the stack."""

    from types import FrameType

    frame: FrameType | None = sys._getframe()
    while frame is not None:
        module_name = frame.f_globals.get("__name__", "")
        if module_name.startswith("treescope") and any(
            marker in module_name for marker in _POSTPROCESSOR_MODULE_MARKERS
        ):
            return True
        frame = frame.f_back
    return False


def consume_repr_suppression(obj: Any) -> bool:
    """Decide whether a card ``_repr_html_`` should return the sentinel.

    True exactly when BOTH halves of the repaired predicate hold: the
    one-shot slot carries ``id(obj)`` (armed only AFTER a successful bridge
    build) and a treescope postprocessor frame is on the call stack. The
    slot is consumed on a hit; any fault answers ``False`` (fail-safe: the
    duplicate quietly returns rather than the card vanishing).
    """

    try:
        if _STATE.last_render_id != id(obj):
            return False
        if not _postprocessor_frame_on_stack():
            return False
        _STATE.last_render_id = None
        return True
    except Exception:  # noqa: BLE001 - fail safe, never suppress on a fault
        return False


# ---------------------------------------------------------------------------
# The explicit bridge door (memo 3.7)


def display(obj: Any) -> None:
    """Explicitly render one object through treescope (the bridge door).

    Because TorchLens cards SHADOW the bridge at top level (F-H: a live
    kernel calls ``_repr_html_`` before treescope's machinery), this door
    is how a treescope user reaches the interactive render: it registers
    the bridge (idempotent), then displays. For a Layer/Op whose saved
    payload is resident and whose adapter probes healthy, the payload
    renders through treescope's automatic array visualizer with OUR
    truncation budgets.

    Spelling is [UI-SPRINT]; ``layer.show(method="treescope")`` is the
    candidate method form.
    """

    treescope = _import_treescope()
    register()
    tensor = None
    if getattr(obj, "has_saved_activation", False):
        try:
            import torch

            candidate = getattr(obj, "out", None)
            if type(candidate) is torch.Tensor and not candidate.is_meta:
                tensor = candidate
        except Exception:  # noqa: BLE001 - fall back to the object render
            tensor = None
    supported, _reason = probe_tensor_support()
    if tensor is not None and supported:
        from ..notebook._truncation import (
            CELL_BUDGET_DEFAULT,
            EDGE_ITEMS_DEFAULT,
            PER_AXIS_BUDGET_DEFAULT,
        )

        autovisualizer = treescope.ArrayAutovisualizer(
            maximum_size=CELL_BUDGET_DEFAULT,
            cutoff_size_per_axis=PER_AXIS_BUDGET_DEFAULT,
            edge_items=EDGE_ITEMS_DEFAULT,
        )
        with treescope.active_autovisualizer.set_scoped(autovisualizer):
            treescope.display(tensor)
        return
    if tensor is not None and not supported:
        _warn_degraded_once()
    treescope.display(obj)
