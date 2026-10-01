"""Namespace derivation from ``module_call_stack`` (memo D1, D19).

The hierarchy is the product: namespaces come from the captured module call
stack, dot-split so unexecuted containers reappear as collapsible levels
(``transformer/h/0/attn``), with consistent pass-qualification -- when a
module address executes more than once in the graph, EVERY call of that
address is qualified (never a bare pass-1 next to ``:2``); single-call
addresses stay bare. Malformed stacks keep the longest verified prefix and
are disclosed (``namespace_fidelity=partial``), with the raw stack preserved
on the node; ``strict=True`` refuses instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._errors import ModelExplorerExportError
from ._ids import escape_component

__tl_layer__ = "L8"

#: Synthetic boundary group namespaces (memo D19): only true graph
#: inputs/outputs get synthetic groups; buffers follow their module stack.
INPUTS_NAMESPACE = "Inputs"
OUTPUTS_NAMESPACE = "Outputs"

#: Episode-only namespace segment for stackless driver ops (memo D11).
DRIVER_NAMESPACE = "driver"


def address_call_counts(entries: list[Any]) -> dict[str, set[int]]:
    """Collect the distinct call indices seen per module address in a graph.

    An address with more than one distinct call index in THIS graph is
    pass-qualified on every call (memo D1 "consistent siblings").
    """

    counts: dict[str, set[int]] = {}
    for entry in entries:
        for stack_entry in getattr(entry, "module_call_stack", ()) or ():
            address, call_index = _split_stack_entry(str(stack_entry))
            if address is not None and call_index is not None:
                counts.setdefault(address, set()).add(call_index)
    return counts


@dataclass(frozen=True)
class NamespaceLevel:
    """One namespace path level and the module address it renders.

    ``executed`` marks levels corresponding to captured module calls;
    intermediate dot-split container levels (a ``ModuleList`` that never
    executes) carry ``executed=False`` and no call index.
    """

    namespace: str
    address: str
    call_index: int | None
    executed: bool


def namespace_for_entry(
    entry: Any,
    call_counts: dict[str, set[int]],
    *,
    strict: bool = False,
    strip_prefix_entry: str | None = None,
) -> tuple[str, bool, list[NamespaceLevel]]:
    """Derive one node's namespace path from its module call stack.

    Parameters
    ----------
    entry:
        Layer-pass entry.
    call_counts:
        Per-address distinct call indices for the graph (qualification rule).
    strict:
        Refuse typed on a malformed stack instead of degrading.
    strip_prefix_entry:
        Exact leading stack entry to strip (episode step graphs strip the
        stepped-module call segment: the level is the graph's identity).

    Returns
    -------
    tuple[str, bool, list[NamespaceLevel]]
        The slash-joined namespace (``""`` for root), whether the full stack
        was verified (``False`` marks a partial, disclosed namespace), and
        the per-level namespace/address records for group-row accounting.
    """

    # Essential complexity (CC>10 named): one linear walk owns the whole
    # derivation -- strip seeding, continuity verification, strict refusal,
    # dot-split, and qualification share loop state a split would smear.
    stack = [str(item) for item in (getattr(entry, "module_call_stack", ()) or ())]
    components: list[str] = []
    levels: list[NamespaceLevel] = []
    previous_address: str | None = None
    if strip_prefix_entry is not None and stack and stack[0] == strip_prefix_entry:
        # The stripped level is the graph's identity (episode step graphs);
        # later deltas stay relative to its address so the container prefix
        # does not resurface as namespace components.
        previous_address, _ = _split_stack_entry(stack[0])
        stack = stack[1:]
    for stack_entry in stack:
        address, call_index = _split_stack_entry(stack_entry)
        delta = _address_delta(address, previous_address) if address is not None else None
        if address is None or call_index is None or delta is None:
            if strict:
                raise ModelExplorerExportError(
                    f"Module call stack entry {stack_entry!r} on op "
                    f"{getattr(entry, 'label', '<unknown>')!r} breaks root-first prefix "
                    "continuity; the namespace cannot be derived faithfully",
                    code="model_explorer_namespace_malformed",
                    remedy=(
                        "export with strict_namespace=False to keep the longest verified "
                        "prefix (disclosed as namespace_fidelity=partial), and report the "
                        "malformed stack to TorchLens"
                    ),
                    stack=tuple(stack),
                )
            return "/".join(components), False, levels
        delta_parts = delta.split(".")
        address_parts = address.split(".")
        base_length = len(address_parts) - len(delta_parts)
        for offset, part in enumerate(delta_parts):
            component = escape_component(part)
            is_last = offset == len(delta_parts) - 1
            if is_last and len(call_counts.get(address, ())) > 1:
                component = f"{component}:{call_index}"
            components.append(component)
            levels.append(
                NamespaceLevel(
                    namespace="/".join(components),
                    address=".".join(address_parts[: base_length + offset + 1]),
                    call_index=call_index if is_last else None,
                    executed=is_last,
                )
            )
        previous_address = address
    return "/".join(components), True, levels


def _split_stack_entry(stack_entry: str) -> tuple[str | None, int | None]:
    """Split one root-first stack entry ``address:call`` (memo D1 rsplit)."""

    address, separator, call_text = stack_entry.rpartition(":")
    if not separator or not call_text.isdigit():
        return None, None
    return address, int(call_text)


def _address_delta(address: str | None, previous_address: str | None) -> str | None:
    """Return the dotted-address delta, or ``None`` on continuity breaks."""

    if address is None or not address:
        return None
    if previous_address is None:
        return address
    if not address.startswith(previous_address + "."):
        return None
    return address[len(previous_address) + 1 :]
