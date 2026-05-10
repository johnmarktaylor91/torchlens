"""Phase 1 diagnostic comparing thread-replay and hook-stack modules.

Runs immediately after Step 6 (``_fix_modules_for_internal_tensors``) and
before Step 7 in ``postprocess.__init__``. Both ``modules`` and
``_modules_via_stack`` are still raw ``(address, pass_index)`` tuples at this
boundary; Step 12 converts module call tuples to ``"address:pass"`` strings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..data_classes.model_log import Trace


class ModuleContainmentMismatchError(AssertionError):
    """Raised when thread replay and hook stack disagree on op modules."""


def _normalize_modules(value: Any) -> list[tuple[Any, ...]]:
    """Normalize module call containers to tuples.

    Parameters
    ----------
    value:
        Iterable module context value from an OpLog field.

    Returns
    -------
    list[tuple[Any, ...]]
        Tuple-normalized module call list.
    """

    return [tuple(item) for item in list(value)]


def assert_module_stack_equality(trace: "Trace") -> None:
    """Compare or apply raw hook-stack containment.

    Parameters
    ----------
    trace:
        Trace being postprocessed.

    Raises
    ------
    ModuleContainmentMismatchError
        If the active diagnostic engine is ``"both"`` and any op has a different
        raw module stack between the two engines.
    """

    engine = getattr(trace, "_module_containment_engine", "thread_replay")
    if engine == "thread_replay":
        return
    if engine == "hook_stack":
        for op in getattr(trace, "_raw_layer_dict", {}).values():
            op.modules = list(getattr(op, "_modules_via_stack", []))
        return

    mismatches = []
    for op in getattr(trace, "_raw_layer_dict", {}).values():
        thread_norm = _normalize_modules(getattr(op, "modules", []))
        stack_norm = _normalize_modules(getattr(op, "_modules_via_stack", []))
        if thread_norm != stack_norm:
            label = getattr(op, "tl__label_raw", getattr(op, "_label_raw", "<unknown>"))
            mismatches.append((label, thread_norm, stack_norm))

    if mismatches:
        lines = ["module containment mismatch (thread_replay vs hook_stack):"]
        for label, thread_v, stack_v in mismatches[:20]:
            lines.append(f"  {label}: thread={thread_v} stack={stack_v}")
        if len(mismatches) > 20:
            lines.append(f"  ... and {len(mismatches) - 20} more")
        raise ModuleContainmentMismatchError("\n".join(lines))
