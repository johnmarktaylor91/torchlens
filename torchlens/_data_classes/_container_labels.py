"""Pure value helpers for container views.

Split out of ``container.py``: the tree-line key and leaf-reference text of a
container ``repr``, the text of one container path component, the spec
identity test sibling-leaf discovery uses, and the name of a spec's first
opaque member.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .._ir.container import ContainerSpec, OutputPathComponent


def _path_component_text(component: OutputPathComponent) -> str:
    """Return the key, index or field name of one container path component."""

    for attr in ("key", "name", "index"):
        if hasattr(component, attr):
            return str(getattr(component, attr))
    return str(component)


def _same_container_spec(left: ContainerSpec | None, right: ContainerSpec | None) -> bool:
    """Return whether two specs represent the same captured container root.

    Parameters
    ----------
    left:
        Candidate sibling spec.
    right:
        Seed op spec.

    Returns
    -------
    bool
        ``True`` when specs are the same object or equal frozen dataclasses.
    """

    return left is right or (left is not None and right is not None and left == right)


def _display_key(key: Any) -> str:
    """Return a display label for an immediate key.

    Parameters
    ----------
    key:
        User-facing key.

    Returns
    -------
    str
        Readable key label.
    """

    return f"[{key!r}]" if not isinstance(key, str) else key


def _leaf_reference(value: Any) -> str:
    """Return a stable leaf reference for a container tree line.

    Parameters
    ----------
    value:
        Leaf value, usually an ``Op``.

    Returns
    -------
    str
        Output-style layer label when available, otherwise ``repr(value)``.
    """

    layer_label = getattr(value, "layer_label", None)
    if layer_label is not None:
        return str(layer_label)
    return repr(value)


def _opaque_member_text(spec: ContainerSpec | None) -> str | None:
    """Name the first opaque node of a final-output spec, or ``None`` when there is none.

    Parameters
    ----------
    spec:
        Final-output container spec, ``None`` for a path-only view.

    Returns
    -------
    str | None
        ``"the model output"`` for an opaque root, ``"its member 'a.b'"`` for a
        nested opaque node, ``None`` when no node is opaque.
    """

    if spec is None:
        return None
    pending: list[tuple[tuple[str, ...], ContainerSpec]] = [((), spec)]
    while pending:
        path, node = pending.pop(0)
        if node.kind == "opaque":
            if not path:
                return "the model output"
            return f"its member {'.'.join(path)!r}"
        pending.extend(
            ((*path, _path_component_text(component)), child)
            for component, child in node.child_specs
        )
    return None
