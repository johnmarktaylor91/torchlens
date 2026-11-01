"""Output-path occurrences of captured tensors, built at capture and read at output-op build.

One tensor object can sit at several output paths (``return a, a``;
``return x, [x0, x]``; an eval-mode dropout returning its input). Every
occurrence gets its own output op, so the producing op's raw label maps to ALL
its ``(address, path, spec)`` occurrences in walk order and each output op takes
the path at its own address. Keeping one path per label gave every alias the
LAST path, so the earlier paths matched no output op and the container rebuild
ran out of leaves.

This module imports nothing from the package: the torch backend builds the map
(``TorchBackend.extract_and_mark_outputs``) and postprocess reads it
(``_postprocess/graph_traversal.py::_add_output_layers``).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

Occurrence = tuple[str, tuple[Any, ...], Any]
"""One ``(display address, container path, container spec or None)`` occurrence."""


def container_path_to_address(path: tuple[Any, ...]) -> str:
    """Convert an output path tuple to TorchLens' display address string.

    Parameters
    ----------
    path:
        Path components from path-aware output traversal.

    Returns
    -------
    str
        Dot-separated output address suffix.
    """

    parts: list[str] = []
    for component in path:
        if hasattr(component, "index"):
            parts.append(str(component.index))
        elif hasattr(component, "key"):
            parts.append(str(component.key))
        elif hasattr(component, "name"):
            parts.append(str(component.name))
        else:
            parts.append(str(component))
    return ".".join(parts)


def occurrences_by_raw_label(
    labeled_entries: Iterable[tuple[str | None, tuple[Any, ...], Any]],
) -> dict[str, list[Occurrence]]:
    """Group output-walk entries by producing raw label, one occurrence per path.

    Parameters
    ----------
    labeled_entries:
        ``(raw label or None, container path, container spec or None)`` in
        output-walk order; entries without a raw label are skipped.

    Returns
    -------
    dict[str, list[Occurrence]]
        Raw label to its occurrences in walk order.
    """

    occurrences: dict[str, list[Occurrence]] = {}
    for label_raw, path, container_spec in labeled_entries:
        if label_raw is not None:
            occurrences.setdefault(label_raw, []).append(
                (container_path_to_address(path), path, container_spec)
            )
    return occurrences


def output_container_path_meta(
    trace: Any, label_raw: str, output_address: str
) -> tuple[tuple[Any, ...], Any] | None:
    """Return the ``(container_path, container_spec)`` of one output occurrence.

    The output op built for the occurrence at ``output_address`` takes THAT
    occurrence's path, so each aliased slot keeps its own path and the
    container rebuild finds a leaf for every slot.

    Parameters
    ----------
    trace:
        Trace carrying ``_output_container_specs_by_raw_label``.
    label_raw:
        Raw label of the output op's parent (the producing op).
    output_address:
        Display address of the output op being built.

    Returns
    -------
    tuple[tuple[Any, ...], Any] | None
        The matching occurrence's path and spec; the first occurrence when no
        address matches; ``None`` when the tensor sits at no container path.
    """

    occurrences = getattr(trace, "_output_container_specs_by_raw_label", {}).get(label_raw)
    if not occurrences:
        return None
    for address, path, spec in occurrences:
        if address == output_address:
            return path, spec
    _address, path, spec = occurrences[0]
    return path, spec
