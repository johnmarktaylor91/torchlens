"""Per-row facts behind the optional summary columns (walk batch 1).

The ladder reads these once per trace: every member shape of a module's
output in return order, the shapes entering a row, the kernel size and
groups of a row's own conv/pool op, the ``module.training`` mode, and the
live optimizer state the Params footer names. Pure readers over finished
capture records; nothing here mints a FLOP or parameter count.

Spellings DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..capture.flops import ATTENTION_FULL_SQUARE_FUNCS, ATTENTION_FULL_SQUARE_NOTE

if TYPE_CHECKING:
    from .._data_classes.trace import Trace

Shape = tuple[int, ...]


def counted_attention_ops(aggregation: Any) -> int:
    """Attention op rows of a FLOP aggregation whose FLOPs were counted."""

    return sum(
        1
        for row in aggregation.rows
        if row.kind == "op"
        and row.func_name in ATTENTION_FULL_SQUARE_FUNCS
        and row.flops_fma2 is not None
    )


def attention_full_square_line(count: int) -> str:
    """The whole-capture disclosure line for ``count`` counted attention ops."""

    return f"attention: {count} op(s) counted; {ATTENTION_FULL_SQUARE_NOTE}"


def shape_tuple(shape: Any) -> Shape | None:
    """A plain int tuple for one recorded shape, or None when unknown."""

    if shape is None or not hasattr(shape, "__iter__"):
        return None
    try:
        return tuple(int(dim) for dim in shape)
    except (TypeError, ValueError):
        return None


def label_shapes(trace: Trace) -> dict[str, Shape | None]:
    """Op label (pass-qualified and layer label) -> output shape."""

    shapes: dict[str, Shape | None] = {}
    for op in getattr(trace, "layer_list", ()) or ():
        shape = shape_tuple(getattr(op, "shape", None))
        for key in (getattr(op, "label", None), getattr(op, "layer_label", None)):
            if key is not None:
                shapes.setdefault(str(key), shape)
    return shapes


def _distinct_shapes(
    labels: Iterable[Any], shapes: Mapping[str, Shape | None]
) -> tuple[Shape, ...]:
    """Shapes of distinct labeled tensors, in first-seen order."""

    seen: set[str] = set()
    found: list[Shape] = []
    for label in labels:
        key = str(label)
        if key in seen:
            continue
        seen.add(key)
        shape = shapes.get(key)
        if shape is not None:
            found.append(shape)
    return tuple(found)


def op_input_shapes(op: Any, shapes: Mapping[str, Shape | None]) -> tuple[Shape, ...]:
    """Shapes of the distinct tensors feeding one op (parents order)."""

    return _distinct_shapes(getattr(op, "parents", ()) or (), shapes)


def op_kernel_groups(op: Any) -> tuple[Shape | None, int | None]:
    """Kernel size and conv groups from an op's salient-argument record.

    Groups apply to convolutions only (the salient record omits the
    default ``groups=1``); pooling ops carry a kernel and no groups.
    """

    config = getattr(op, "func_config", None) or {}
    kernel = config.get("kernel_size")
    if isinstance(kernel, int):
        kernel = (kernel,)
    kernel_size = shape_tuple(kernel)
    func_name = str(getattr(op, "func_name", "") or "")
    groups: int | None = None
    if func_name.startswith("conv") and kernel_size is not None:
        groups = int(config.get("groups", 1))
    return kernel_size, groups


def _role_hint_candidates(record: Any) -> list[Any]:
    """Role-hint tables to try for one module record (class first)."""

    from .._data_classes._module_role_hints import (
        ROLE_HINTS_BY_MODULE_TYPE,
        role_hints_for_module_class,
    )

    candidates: list[Any] = []
    cls = getattr(record, "cls", None)
    if isinstance(cls, type):
        hints = role_hints_for_module_class(cls)
        if hints is not None:
            candidates.append(hints)
    class_name = str(getattr(record, "class_name", "") or "")
    for module_type, hints in ROLE_HINTS_BY_MODULE_TYPE.items():
        if module_type.__name__ == class_name and hints not in candidates:
            candidates.append(hints)
    candidates.append(None)
    return candidates


def _ordered_output_labels(trace: Trace, record: Any, call: Any) -> list[str]:
    """A call's output op labels in RETURN order (op order when unprovable).

    ``output_ops`` is in execution order while ``output_paths`` is in
    container traversal order; each output op's ``multi_output_name`` was
    minted from its path, so matching names against the paths recovers the
    return order. Any unmatched or ambiguous name keeps execution order.
    """

    from .._data_classes._module_role_hints import multi_output_role_from_path

    labels = [str(label) for label in getattr(call, "output_ops", ()) or ()]
    paths = list(getattr(call, "output_paths", ()) or ())
    if len(labels) < 2 or len(paths) != len(labels):
        return labels
    ops: Any = getattr(trace, "ops", None)
    if ops is None:
        return labels
    try:
        names = [getattr(ops[label], "multi_output_name", None) for label in labels]
    except Exception:  # noqa: BLE001 - an unresolvable label keeps execution order
        return labels
    for hints in _role_hint_candidates(record):
        position = {
            multi_output_role_from_path(tuple(path), index, hints=hints): index
            for index, path in enumerate(paths)
        }
        if len(position) != len(paths) or any(name not in position for name in names):
            continue
        order = sorted(range(len(labels)), key=lambda i: position[names[i]])
        return [labels[i] for i in order]
    return labels


def module_call_shapes(
    trace: Trace, record: Any, shapes: Mapping[str, Shape | None]
) -> tuple[tuple[Shape, ...], tuple[Shape, ...]]:
    """(output member shapes in return order, input shapes) of the last call."""

    call: Any = None
    accessor = getattr(record, "calls", None)
    if accessor is not None:
        for key in (-1, int(getattr(record, "num_calls", 1) or 1)):
            try:
                call = accessor[key]
                break
            except Exception:  # noqa: BLE001, S112 - candidate-key probe over heterogeneous accessors
                continue
    if call is None:
        return (), ()
    outputs: list[Shape] = []
    for label in _ordered_output_labels(trace, record, call):
        shape = shapes.get(label)
        if shape is not None:
            outputs.append(shape)
    inputs = _distinct_shapes(getattr(call, "input_ops", ()) or (), shapes)
    return tuple(outputs), inputs


def merge_mode(modes: Iterable[str | None]) -> str | None:
    """One row's mode over its modules: train, eval, mixed, or unknown."""

    known = {mode for mode in modes if mode is not None}
    if not known:
        return None
    return known.pop() if len(known) == 1 else "mixed"


def training_mode(record: Any) -> str | None:
    """``module.training`` as captured: 'train', 'eval', or None."""

    training = getattr(record, "training", None)
    if training is None:
        return None
    return "train" if bool(training) else "eval"


@dataclass(frozen=True)
class OptimizerStateFacts:
    """Live optimizer state behind a capture given ``optimizer=``.

    ``state_params`` counts every element of the optimizer's state tensors
    (Adam: two moments per trained element plus step counters) and
    ``state_bytes`` their storage; zero before the first ``step()``.
    ``held_params`` counts the captured parameters the optimizer holds.
    """

    state_params: int
    state_bytes: int
    held_params: int


def optimizer_state_facts(trace: Trace) -> OptimizerStateFacts | None:
    """Read the capture's optimizer state; None when none was given.

    The optimizer is a session-time handle (not persisted), so a loaded
    trace has none and the footer omits the line.
    """

    optimizer = trace._optimizer  # noqa: SLF001 -- declared session field (DROP: None on load)
    state = getattr(optimizer, "state", None)
    if optimizer is None or state is None:
        return None
    numel = 0
    nbytes = 0
    for entry in state.values():
        for value in getattr(entry, "values", lambda: ())():
            if hasattr(value, "numel") and hasattr(value, "element_size"):
                numel += int(value.numel())
                nbytes += int(value.numel()) * int(value.element_size())
    held = sum(
        int(getattr(param, "num_params", 0) or 0)
        for param in getattr(trace, "params", ()) or ()
        if getattr(param, "has_optimizer", None)
    )
    return OptimizerStateFacts(state_params=numel, state_bytes=nbytes, held_params=held)
