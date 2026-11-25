"""Module-call currency for rolled edges and labels that touch a collapsed module box.

A rolled collapsed module box stands for every call of one module, and its title
counts those calls. The ops inside it are rolled Layers whose pass numbers count op
executions, which differ from module calls whenever an op runs more than once per call
(an inner Python loop) or only on some calls. Anything drawn on the box -- recurrence
self-edges, ``In``/``Out`` end labels, the colon call partition on the title -- must be
stated in module calls, never in the inner op's passes.
"""

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._data_classes.layer import Layer

__all__ = (
    "CallPartitionJoin",
    "collapsed_endpoint_address",
    "end_label_indices",
    "pass_calls",
)


def _module_call_index(module_calls: Iterable[str], address: str) -> int | None:
    """Return the call index of ``address`` in a pass-qualified module stack.

    Parameters
    ----------
    module_calls:
        Pass-qualified module labels (``"address:call"``) of one op.
    address:
        Module address to look up.

    Returns
    -------
    int | None
        The call index, or ``None`` when the op is not inside that module.
    """

    for module_call in module_calls:
        owner, separator, call_text = str(module_call).rpartition(":")
        if separator and owner == address and call_text.isdigit():
            return int(call_text)
    return None


def collapsed_endpoint_address(layer: "Layer", rendered_name: str) -> str | None:
    """Return the collapsed module address a rendered endpoint name stands for.

    Parameters
    ----------
    layer:
        Rolled layer underlying the edge end.
    rendered_name:
        Graphviz node name the end was drawn on.

    Returns
    -------
    str | None
        ``rendered_name`` when it names a module that contains every pass of
        ``layer`` (the end is a collapsed module box), else ``None``.
    """

    trace = getattr(layer, "source_trace", None)
    modules = getattr(trace, "modules", None)
    if modules is None or rendered_name not in modules:
        return None
    ops = getattr(layer, "ops", None) or {}
    if not ops or any(_module_call_index(op.modules, rendered_name) is None for op in ops.values()):
        return None
    return rendered_name


def pass_calls(layer: "Layer", address: str, passes: Sequence[int]) -> list[int]:
    """Translate rolled pass numbers of ``layer`` into calls of module ``address``.

    Parameters
    ----------
    layer:
        Rolled layer whose passes are translated.
    address:
        Collapsed module address containing every pass of ``layer``.
    passes:
        Pass numbers of ``layer``.

    Returns
    -------
    list[int]
        Module call index for each pass, in the given order.
    """

    ops_by_pass = dict(layer.ops.items())  # 1-based pass keys; ops[k] is a 0-based position
    calls: list[int] = []
    for pass_index in passes:
        call_index = _module_call_index(ops_by_pass[pass_index].modules, address)
        if call_index is not None:
            calls.append(call_index)
    return calls


class CallPartitionJoin:
    """Join per-layer call partitions into one partition per module address.

    Two calls of a module belong to one region when ANY layer inside the module
    connects them. Taking the finest single-layer partition instead lets one
    input-independent op (whose passes never feed each other, such as mask
    arithmetic recomputed on every call) split a module whose calls feed one
    another, so a looped layer would read ``:1,2,3`` instead of ``(x3)``.
    """

    def __init__(self) -> None:
        """Start with no recorded partitions."""

        self._parents: dict[str, dict[int, int]] = {}

    def _find(self, parents: dict[int, int], call_index: int) -> int:
        """Return the union-find root of one call."""

        root = parents.setdefault(call_index, call_index)
        while parents[root] != root:
            root = parents[root]
        while parents[call_index] != root:
            parents[call_index], call_index = root, parents[call_index]
        return root

    def add(self, address: str, groups: Iterable[Sequence[int]]) -> None:
        """Record that each group's calls of ``address`` are connected.

        Parameters
        ----------
        address:
            Module address the call indices belong to.
        groups:
            Connected call-index groups reported by one layer.
        """

        parents = self._parents.setdefault(address, {})
        for group in groups:
            first, *rest = group
            for call_index in (first, *rest):
                low, high = sorted((self._find(parents, first), self._find(parents, call_index)))
                parents[high] = low

    def split_partitions(self) -> dict[str, tuple[tuple[int, ...], ...]]:
        """Return every address whose joined calls form more than one region.

        Returns
        -------
        dict[str, tuple[tuple[int, ...], ...]]
            Address to sorted call groups, ordered by first call.
        """

        result: dict[str, tuple[tuple[int, ...], ...]] = {}
        for address, parents in self._parents.items():
            groups: dict[int, list[int]] = {}
            for call_index in sorted(parents):
                groups.setdefault(self._find(parents, call_index), []).append(call_index)
            if len(groups) > 1:
                result[address] = tuple(sorted((tuple(g) for g in groups.values()), key=min))
        return result


def end_label_indices(
    layer: "Layer", rendered_name: str, passes: Sequence[int]
) -> tuple[list[int], int]:
    """Return an edge end's index list and the total it is a subset of.

    Parameters
    ----------
    layer:
        Rolled layer at the edge end.
    rendered_name:
        Graphviz node name the end was drawn on.
    passes:
        Pass numbers of ``layer`` that use the edge.

    Returns
    -------
    tuple[list[int], int]
        ``(passes, layer.num_passes)`` for an op end; for an end drawn on a
        collapsed module box, the distinct module calls that use the edge and
        the module's call count, so the label speaks the box title's currency.
    """

    address = collapsed_endpoint_address(layer, rendered_name)
    if address is None:
        return list(passes), int(layer.num_passes)
    module_calls = int(layer.source_trace.modules[address].num_calls)
    return sorted(set(pass_calls(layer, address, passes))), module_calls
