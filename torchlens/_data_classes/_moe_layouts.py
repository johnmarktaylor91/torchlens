"""Read-time structural candidates for ordinary and batched expert containers."""

from dataclasses import dataclass

from torch import nn


def _join(address: str, child: str) -> str:
    """Join a root-relative module address."""
    return f"{address}.{child}" if address else child


@dataclass(frozen=True)
class _Layout:
    """An expert container candidate, requiring later captured-dataflow proof."""

    block: str
    experts: tuple[str, ...]
    gates: tuple[str, ...]
    parameter_prefixes: tuple[str, ...]
    count: int


def _layouts(model: nn.Module) -> list[_Layout]:
    """Find homogeneous module collections and expert-batched parameter arrays."""
    # Essential structural proof: keep block, sibling-gate and container checks together.
    found = []
    for address, block in model.named_modules():
        children = tuple(block.named_children())
        for name, container in children:
            gates = tuple(
                _join(address, key)
                for key, child in children
                if child is not container and not isinstance(child, (nn.ModuleList, nn.ModuleDict))
            )
            if isinstance(container, (nn.ModuleList, nn.ModuleDict)):
                entries = tuple(container.named_children())
                if len(entries) < 2 or len({type(child) for _, child in entries}) != 1:
                    continue
                experts = tuple(_join(_join(address, name), key) for key, _ in entries)
                found.append(
                    _Layout(
                        address,
                        experts,
                        gates,
                        tuple(expert + "." for expert in experts),
                        len(experts),
                    )
                )
            else:
                candidate = _batched(address, _join(address, name), container, gates)
                if candidate is not None:
                    found.append(candidate)
        direct = _batched(
            address, address, block, tuple(_join(address, key) for key, _ in children)
        )
        if direct is not None:
            found.append(direct)
    return found


def _batched(
    block: str, address: str, container: nn.Module, gates: tuple[str, ...]
) -> _Layout | None:
    """Require at least one matrix bank with a shared leading expert dimension."""
    arrays = [
        (name, value)
        for name, value in container.named_parameters(recurse=False)
        if value.ndim == 3
    ]
    widths = {int(value.shape[0]) for _, value in arrays}
    if len(widths) != 1 or next(iter(widths)) < 2:
        return None
    return _Layout(
        block, (), gates, tuple(_join(address, name) for name, _ in arrays), next(iter(widths))
    )
