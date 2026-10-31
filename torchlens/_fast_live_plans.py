"""Collection plans for the guarded fast live run.

The fast live session collects saved sites from a native forward: one plan per
saved module CALL (a module the forward calls twice, such as resnet's shared
``BasicBlock.relu``, gets one plan per saved call) and one per scoped functional
call. Each plan names the op labels it refreshes and the shapes, dtypes and
container paths the live output must match.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

__all__ = [
    "_FastOutputPlan",
    "_module_output_paths",
    "_pair_output_leaves",
    "build_function_plans",
    "build_module_plans",
    "recorded_module_call_counts",
]


@dataclass(frozen=True, slots=True)
class _FastOutputPlan:
    """Expected tensor leaves for one live module or functional call."""

    address_or_name: str
    op_labels: tuple[str, ...]
    shapes: tuple[tuple[int, ...] | None, ...]
    dtypes: tuple[str | None, ...]
    save_labels: frozenset[str]
    # Each op's container path inside the call's output, when capture tied
    # every op to exactly one path; ``None`` keeps the positional pairing.
    paths: tuple[tuple[Any, ...], ...] | None = None
    # The module call (1-based pass number) this plan collects; ``None`` for
    # functional plans. A module the forward calls more than once (resnet's
    # BasicBlock reuses one ``self.relu``) fires its hook on every call, and
    # only the planned calls are collected.
    call_index: int | None = None


def _pair_output_leaves(
    output: Any, captured_paths: Sequence[tuple[Any, ...]] | None, count: int
) -> tuple[list[Any] | None, str]:
    """Pair ``count`` captured output ops with the tensor leaves of a live output.

    The walk is capture's own output walker (plus its bare-attribute fallback),
    so a typed container capture descended into (a Hugging Face ``ModelOutput``
    carrying a ``DynamicCache`` of per-layer key/value tensors, a namedtuple, a
    dataclass) is seen the same way here. With per-op paths every captured path
    must resolve to exactly one leaf and the output must carry no other leaf;
    without them (a single leaf, an opaque root, a role-hinted family) the
    pairing is positional over the same walk. Returns the paired values, or
    ``None`` with the reason when the structure differs from the capture.
    """

    # Module import: the dotted spelling of this path matches the layer lint's
    # torch-private pattern, but the walker is TorchLens's own, not a torch private.
    from ._utils.introspection import get_vars_of_type_from_obj
    from .backends.torch import _ops_containers

    native = [
        (tuple(path), tensor)
        for tensor, path, _spec in _ops_containers._walk_output_tensors_with_paths(output)
    ]
    if not native:
        native = [
            ((), tensor)
            for tensor in get_vars_of_type_from_obj(output, torch.Tensor, search_depth=4)
        ]
    if len(native) != count:
        return None, f"{len(native)} tensor leaves against {count} captured output ops"
    by_path = dict(native)
    if (
        captured_paths is None
        or len(by_path) != len(native)
        or any(path == () for path in captured_paths)
    ):
        return [tensor for _path, tensor in native], ""
    missing = [path for path in captured_paths if path not in by_path]
    if missing:
        return None, f"unmatched captured output paths {missing}"
    return [by_path[path] for path in captured_paths], ""


def _module_output_paths(
    trace: Any, module_call: Any, address: str, resolved: Sequence[Any]
) -> tuple[tuple[Any, ...], ...] | None:
    """Return each output op's container path in a module call's output.

    ``ModuleCall.output_paths`` is in capture traversal order while
    ``output_ops`` is in op order; the documented tie between them is each
    op's ``multi_output_name``, minted from its path (or a role hint) at
    capture. ``None`` when the call has fewer than two leaves or any op lacks
    a unique path, which keeps the positional pairing.
    """

    from ._data_classes._module_role_hints import (
        multi_output_role_from_path,
        role_hints_for_module_class,
    )

    paths = tuple(tuple(path) for path in (getattr(module_call, "output_paths", None) or ()))
    if len(paths) < 2 or len(paths) != len(resolved):
        return None
    try:
        module = trace.modules[address]
    except (KeyError, IndexError, TypeError, AttributeError):
        module = None
    hints = role_hints_for_module_class(getattr(module, "cls", None))
    by_name: dict[str, tuple[Any, ...]] = {}
    for index, path in enumerate(paths):
        name = multi_output_role_from_path(path, index, hints=hints)
        if name is None or name in by_name:
            return None
        by_name[name] = path
    matched: list[tuple[Any, ...]] = []
    for item in resolved:
        name = getattr(item, "multi_output_name", None)
        if name not in by_name:
            return None
        matched.append(by_name[name])
    if len(set(matched)) != len(matched):
        return None
    return tuple(matched)


def build_module_plans(trace: Any) -> tuple[_FastOutputPlan, ...]:
    """Derive the module-call exit sequence whose outputs the capture saved.

    Every module call (leaf or container) whose output ops carry a saved
    activation gets a plan, including the ``interventionreplacement`` op a
    module-boundary intervention leaves as the module's output. Plans are
    ordered as torch fires forward hooks: by module EXIT, i.e. the raw
    index of the module's last output op, with the deeper module first
    when a nested call hands the same tensor to its parent. The root module is
    the input/output boundary and is refreshed separately.
    """

    keyed: list[tuple[int, int, _FastOutputPlan]] = []
    module_calls = trace.module_calls
    for call_label in list(module_calls.keys()):
        module_call = module_calls[call_label]
        address, _, call_suffix = call_label.rpartition(":")
        if not address:
            address, call_suffix = call_suffix, "1"
        if address == "self":
            continue
        resolved = tuple(
            trace.layer_dict_all_keys[label]
            for label in module_call.output_ops
            if label in trace.layer_dict_all_keys
        )
        if not resolved or not any(
            bool(getattr(item, "has_saved_activation", False)) for item in resolved
        ):
            continue
        labels = tuple(item.label for item in resolved)
        # A module exits after its LAST output op: a returned KV cache
        # carries tensors produced long before the module's final op.
        exit_index = max(int(getattr(item, "raw_index", 0) or 0) for item in resolved)
        keyed.append(
            (
                exit_index,
                -address.count("."),
                _FastOutputPlan(
                    address_or_name=address,
                    op_labels=labels,
                    shapes=tuple(
                        tuple(item.shape) if item.shape is not None else None for item in resolved
                    ),
                    dtypes=tuple(
                        str(item.dtype) if item.dtype is not None else None for item in resolved
                    ),
                    save_labels=frozenset(
                        label
                        for label, item in zip(labels, resolved)
                        if bool(getattr(item, "has_saved_activation", False))
                    ),
                    paths=_module_output_paths(trace, module_call, address, resolved),
                    call_index=int(call_suffix) if call_suffix.isdigit() else 1,
                ),
            )
        )
    keyed.sort(key=lambda item: (item[0], item[1]))
    return tuple(plan for _, _, plan in keyed)


def build_function_plans(
    trace: Any, module_saved_labels: frozenset[str]
) -> tuple[_FastOutputPlan, ...]:
    """Derive scoped functional calls needed by saved non-module ops.

    Both save forms qualify. A functional ``save=`` predicate collects the
    functional ops it saved. The default save-all capture collects only the
    saved functional ops that no module output covers (``out += identity``
    inside a residual block, before the block's last op), so a module
    output such as an intervention replacement is never collected twice.
    Collection runs in capture order under the same per-call path, shape
    and dtype guards.

    Parameters
    ----------
    trace:
        Source trace.
    module_saved_labels:
        Saved op labels the module plans already collect.
    """

    explicit = getattr(trace, "_predicate_save_options", None) is not None
    target_names = {
        str(op.func_name)
        for op in trace.layer_list
        if bool(getattr(op, "has_saved_activation", False))
        and not bool(getattr(op, "is_atomic_module", False))
        and str(getattr(op, "func_name", "none")) not in {"none", "identity"}
        and (explicit or op.label not in module_saved_labels)
    }
    plans: list[_FastOutputPlan] = []
    ops = [op for op in trace.layer_list if str(getattr(op, "func_name", "")) in target_names]
    index = 0
    while index < len(ops):
        first = ops[index]
        group = [first]
        spec = getattr(first, "container_spec", None)
        if spec is not None:
            next_index = index + 1
            while (
                next_index < len(ops)
                and ops[next_index].raw_index == group[-1].raw_index + 1
                and getattr(ops[next_index], "container_spec", None) is spec
            ):
                group.append(ops[next_index])
                next_index += 1
        labels = tuple(item.label for item in group)
        plans.append(
            _FastOutputPlan(
                address_or_name=str(first.func_name),
                op_labels=labels,
                shapes=tuple(
                    tuple(item.shape) if item.shape is not None else None for item in group
                ),
                dtypes=tuple(str(item.dtype) if item.dtype is not None else None for item in group),
                save_labels=frozenset(
                    item.label
                    for item in group
                    if bool(getattr(item, "has_saved_activation", False))
                    and not bool(getattr(item, "is_atomic_module", False))
                    and (explicit or item.label not in module_saved_labels)
                ),
            )
        )
        index += len(group)
    return tuple(plans)


def recorded_module_call_counts(trace: Any, addresses: frozenset[str]) -> dict[str, int]:
    """Count the captured calls of each planned module address.

    Parameters
    ----------
    trace:
        Source trace whose ``module_calls`` keys spell ``address:call``.
    addresses:
        Module addresses the fast session collects from.

    Returns
    -------
    dict[str, int]
        Recorded call count per planned address.
    """

    counts: dict[str, int] = dict.fromkeys(addresses, 0)
    for call_label in list(trace.module_calls.keys()):
        address = str(call_label).rpartition(":")[0] or str(call_label)
        if address in counts:
            counts[address] += 1
    return counts
