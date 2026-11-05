"""Pure value helpers for container views.

Split out of ``container.py``: the tree-line key and leaf-reference text of a
container ``repr``, the text of one container path component, the spec
identity test sibling-leaf discovery uses, the name of a spec's first
opaque member, and the refusal text for a final output that cannot be rebuilt.
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


def _unrebuildable_output_refusal(opaque_member: str | None) -> tuple[str, str, str]:
    """Return the ``container_not_reconstructable`` refusal for a multi-output trace.

    Parameters
    ----------
    opaque_member:
        Text naming the first opaque node of the final-output spec, or ``None``.

    Returns
    -------
    tuple[str, str, str]
        Problem, code and remedy for ``PayloadUnavailableError``.
    """

    if opaque_member is None:
        # No structure was recorded: the capture option persists it.
        return (
            "No reconstructable final-output container was captured",
            "container_not_reconstructable",
            "trace with capture_container_structure=True to persist container structure",
        )
    # The structure is recorded and holds an opaque node: the capture option cannot help.
    return (
        f"The final-output container cannot be rebuilt: {opaque_member} is or holds "
        "an object TorchLens cannot represent (container kind 'opaque', for example a "
        "Hugging Face DynamicCache)",
        "container_not_reconstructable",
        "read the output leaves one by one through trace.output_layers, register "
        "the type with tl.types.register_container, or have the model return only "
        "supported containers (for a Hugging Face model, use_cache=False leaves "
        "the cache out)",
    )
