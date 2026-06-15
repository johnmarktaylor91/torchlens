"""Runtime views over captured TorchLens output containers."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from ..ir.container import (
    ContainerSpec,
    DataclassField,
    DictKey,
    HFKey,
    NamedField,
    OutputPathComponent,
    TupleIndex,
    rebuild_container_from_spec,
)

if TYPE_CHECKING:
    from .op import Op

ContainerRootKind = Literal["call", "final_output", "path"]


@dataclass(frozen=True)
class Container:
    """Computed view over a captured Python output container."""

    spec: ContainerSpec | None
    leaves: tuple["Op", ...]
    root_kind: ContainerRootKind
    root_id: tuple[str, Any]
    path: tuple[OutputPathComponent, ...] = ()
    supports_reconstruct: bool = True

    @property
    def kind(self) -> str | None:
        """Return the container kind for this view.

        Returns
        -------
        str | None
            Spec kind, or ``None`` for path-only degraded views.
        """

        return self.spec.kind if self.spec is not None else None

    @property
    def reconstructable(self) -> bool:
        """Return whether this view can reconstruct the original Python object.

        Returns
        -------
        bool
            ``True`` when a full spec and all leaf payloads are available.
        """

        return (
            self.supports_reconstruct
            and self.spec is not None
            and all(_value_for_op(op, "out") is not None for op in self._leaf_ops())
        )

    def __len__(self) -> int:
        """Return the number of immediate elements in the container.

        Returns
        -------
        int
            Immediate container arity.
        """

        if self.spec is None:
            return 0
        if self.spec.kind in {"tuple", "list", "registered"}:
            return int(self.spec.length or 0)
        if self.spec.kind in {"dict", "hf_model_output"}:
            return len(self.spec.keys)
        if self.spec.kind in {"namedtuple", "dataclass"}:
            return len(self.spec.fields)
        if self.spec.kind == "literal":
            return 1
        return 0

    def __iter__(self) -> Iterator[Any]:
        """Iterate over immediate elements.

        Yields
        ------
        Any
            Immediate leaf ``Op`` objects or nested ``Container`` views.
        """

        if self.spec is None:
            return
        for key in _element_keys(self.spec):
            yield self[key]

    def __getitem__(self, key: int | str | Any) -> Any:
        """Return an immediate element by index, dict key, or field name.

        Parameters
        ----------
        key:
            Tuple/list index, dict/HF key, or dataclass/namedtuple field name.

        Returns
        -------
        Any
            Leaf ``Op`` or nested ``Container`` view.

        Raises
        ------
        KeyError
            If the key is not present in this container.
        """

        if self.spec is None:
            raise KeyError(key)
        component = _component_for_key(self.spec, key)
        child_spec = dict(self.spec.child_specs).get(component)
        child_path = (*self.path, component)
        if child_spec is not None:
            return Container(
                spec=child_spec,
                leaves=self.leaves,
                root_kind=self.root_kind,
                root_id=self.root_id,
                path=child_path,
                supports_reconstruct=self.supports_reconstruct,
            )
        for op in self.leaves:
            if tuple(getattr(op, "container_path", ()) or ()) == child_path:
                return op
        if self.spec.kind == "literal":
            return self.spec.literal_value
        raise KeyError(key)

    def reconstruct(self, values: Literal["out", "transformed"] = "out") -> Any:
        """Reconstruct the original Python container.

        Parameters
        ----------
        values:
            Leaf value source. ``"out"`` uses saved outputs; ``"transformed"``
            uses transformed outputs.

        Returns
        -------
        Any
            Rebuilt Python object.

        Raises
        ------
        ValueError
            If this view is path-only or leaf values are unavailable.
        """

        if self.spec is None:
            raise ValueError("Path-only container views cannot be reconstructed.")
        leaves = []
        for op in self._leaf_ops():
            value = _value_for_op(op, values)
            if value is None:
                raise ValueError(f"Container leaf {op.layer_label!r} has no saved {values} value.")
            leaves.append(value)
        return rebuild_container_from_spec(self.spec, leaves)

    def __repr__(self) -> str:
        """Return an indented tree representation.

        Returns
        -------
        str
            Human-readable container tree.
        """

        lines = [self._repr_line()]
        self._append_repr_lines(lines, indent=2)
        return "\n".join(lines)

    def _leaf_ops(self) -> tuple["Op", ...]:
        """Return leaf ops under this view in spec traversal order.

        Returns
        -------
        tuple[Op, ...]
            Ordered leaf operations.
        """

        if self.spec is None:
            return ()
        by_path = {tuple(getattr(op, "container_path", ()) or ()): op for op in self.leaves}
        return tuple(
            by_path[path] for path in _leaf_paths(self.spec, prefix=self.path) if path in by_path
        )

    def _repr_line(self) -> str:
        """Return the root line for ``repr``.

        Returns
        -------
        str
            Single-line summary.
        """

        return (
            f"Container(kind={self.kind!r}, root_kind={self.root_kind!r}, "
            f"root_id={self.root_id!r}, reconstructable={self.reconstructable})"
        )

    def _append_repr_lines(self, lines: list[str], *, indent: int) -> None:
        """Append child tree lines to a representation buffer.

        Parameters
        ----------
        lines:
            Mutable output line buffer.
        indent:
            Current indentation width.
        """

        if self.spec is None:
            lines.append(f"{' ' * indent}<path-only>")
            return
        for key in _element_keys(self.spec):
            label = _display_key(key)
            try:
                value = self[key]
            except KeyError:
                lines.append(f"{' ' * indent}{label}: <missing>")
                continue
            if isinstance(value, Container):
                lines.append(f"{' ' * indent}{label}: {value.kind}")
                value._append_repr_lines(lines, indent=indent + 2)
            else:
                layer_label = getattr(value, "layer_label", None)
                if layer_label is None:
                    layer_label = repr(value)
                lines.append(f"{' ' * indent}{label}: {layer_label}")


def container_from_op(op: "Op") -> Container | None:
    """Build a container view rooted at an op's captured output container.

    Parameters
    ----------
    op:
        Operation whose output container metadata should be viewed.

    Returns
    -------
    Container | None
        Runtime container view, degraded path-only view, or ``None``.
    """

    spec = getattr(op, "container_spec", None)
    path = tuple(getattr(op, "container_path", ()) or ())
    capability = _container_structure_capability(op)
    if capability == "none" and spec is None:
        return None
    if spec is None and not path:
        return None
    root_kind, root_id = _root_identity(op)
    leaves = _sibling_leaves(op, spec=spec, root_kind=root_kind)
    return Container(
        spec=spec,
        leaves=leaves,
        root_kind=root_kind,
        root_id=root_id,
        supports_reconstruct=spec is not None,
    )


def _container_structure_capability(op: "Op") -> str:
    """Return the owning backend's declared container-structure capability.

    Parameters
    ----------
    op:
        Operation whose source trace may declare a backend.

    Returns
    -------
    str
        ``"none"``, ``"paths_only"``, ``"full_spec"``, or ``"full_spec"``
        when the backend cannot be resolved for legacy in-memory traces.
    """

    trace = getattr(op, "source_trace", None)
    backend = getattr(trace, "backend", None)
    if backend is None:
        return "full_spec"
    try:
        from ..backends import get_backend_spec

        return str(get_backend_spec(str(backend)).capabilities.container_structure)
    except Exception:
        return "full_spec"


def reconstruct_output(trace: Any, values: Literal["out", "transformed"] = "out") -> Any:
    """Reconstruct the traced model's final output container.

    Parameters
    ----------
    trace:
        Trace with synthetic output nodes.
    values:
        Leaf value source passed through to ``Container.reconstruct``.

    Returns
    -------
    Any
        Reconstructed model return value.
    """

    output_labels = tuple(getattr(trace, "output_layers", ()) or ())
    for label in output_labels:
        op = trace.ops[label]
        container = container_from_op(op)
        if container is not None and container.root_kind == "final_output":
            return container.reconstruct(values=values)
    if len(output_labels) == 1:
        return trace.ops[output_labels[0]].out
    raise ValueError(
        "No reconstructable final-output container was captured. "
        "Trace with intervention_ready=True to persist output structure."
    )


def _root_identity(op: "Op") -> tuple[ContainerRootKind, tuple[str, Any]]:
    """Return the public root identity for an op container view.

    Parameters
    ----------
    op:
        Operation carrying container metadata.

    Returns
    -------
    tuple[ContainerRootKind, tuple[str, Any]]
        Root kind and stable root id.
    """

    if _op_is_final_output(op):
        return "final_output", ("final_output", 0)
    func_call_id = getattr(op, "func_call_id", None)
    if func_call_id is not None:
        return "call", ("call", func_call_id)
    return "path", ("path", tuple(getattr(op, "container_path", ()) or ()))


def _sibling_leaves(
    op: "Op",
    *,
    spec: ContainerSpec | None,
    root_kind: ContainerRootKind,
) -> tuple["Op", ...]:
    """Return sibling leaf ops for a container root.

    Parameters
    ----------
    op:
        Seed operation.
    spec:
        Root container spec.
    root_kind:
        Root identity kind.

    Returns
    -------
    tuple[Op, ...]
        Sibling leaves in trace order.
    """

    trace = getattr(op, "source_trace", None)
    if trace is None:
        return (op,)
    candidates = trace.output_layers if root_kind == "final_output" else trace.layer_labels
    leaves = []
    for label in candidates:
        sibling = trace.ops[label]
        if root_kind == "final_output":
            if _same_container_spec(getattr(sibling, "container_spec", None), spec):
                leaves.append(sibling)
            continue
        if getattr(sibling, "func_call_id", None) == getattr(op, "func_call_id", None) and (
            spec is None or _same_container_spec(getattr(sibling, "container_spec", None), spec)
        ):
            leaves.append(sibling)
    return tuple(leaves) or (op,)


def _op_is_final_output(op: "Op") -> bool:
    """Return whether an op belongs to the trace final-output layer set.

    Parameters
    ----------
    op:
        Candidate operation.

    Returns
    -------
    bool
        ``True`` when the op is explicitly marked output or appears in
        ``trace.output_layers``.
    """

    if bool(getattr(op, "is_output", False)):
        return True
    trace = getattr(op, "source_trace", None)
    if trace is None:
        return False
    output_labels = set(getattr(trace, "output_layers", ()) or ())
    if getattr(op, "layer_label", None) in output_labels:
        return True
    if getattr(op, "layer_label_raw", None) in output_labels:
        return True
    return any(trace.ops[label] is op for label in output_labels if label in trace.ops)


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


def _value_for_op(op: "Op", source: str) -> Any:
    """Return a leaf value from an op.

    Parameters
    ----------
    op:
        Leaf operation.
    source:
        Value source name.

    Returns
    -------
    Any
        Saved value or ``None``.
    """

    if source == "out":
        return getattr(op, "out", None)
    if source == "transformed":
        return getattr(op, "transformed_out", None)
    raise ValueError(f"Unsupported container value source {source!r}.")


def _element_keys(spec: ContainerSpec) -> tuple[Any, ...]:
    """Return immediate user-facing keys for a spec.

    Parameters
    ----------
    spec:
        Container spec.

    Returns
    -------
    tuple[Any, ...]
        Immediate keys.
    """

    if spec.kind in {"tuple", "list", "registered"}:
        return tuple(range(spec.length or 0))
    if spec.kind in {"dict", "hf_model_output"}:
        return spec.keys
    if spec.kind in {"namedtuple", "dataclass"}:
        return spec.fields
    if spec.kind == "literal":
        return ("literal",)
    return ()


def _component_for_key(spec: ContainerSpec, key: Any) -> OutputPathComponent:
    """Convert a user key to a typed output path component.

    Parameters
    ----------
    spec:
        Container spec.
    key:
        User-facing key.

    Returns
    -------
    OutputPathComponent
        Typed path component.
    """

    if spec.kind in {"tuple", "list", "registered"} and isinstance(key, int):
        return TupleIndex(key)
    if spec.kind == "dict":
        return DictKey(key)
    if spec.kind == "hf_model_output":
        return HFKey(key)
    if spec.kind == "namedtuple" and isinstance(key, str):
        return NamedField(key)
    if spec.kind == "dataclass" and isinstance(key, str):
        return DataclassField(key)
    if spec.kind == "literal" and key == "literal":
        return TupleIndex(0)
    raise KeyError(key)


def _leaf_paths(
    spec: ContainerSpec,
    *,
    prefix: tuple[OutputPathComponent, ...],
) -> Iterator[tuple[OutputPathComponent, ...]]:
    """Yield tensor leaf paths in spec traversal order.

    Parameters
    ----------
    spec:
        Container spec.
    prefix:
        Path prefix for this spec node.

    Yields
    ------
    tuple[OutputPathComponent, ...]
        Leaf paths.
    """

    child_by_key = dict(spec.child_specs)
    if spec.kind == "literal":
        return
    for key in _element_keys(spec):
        component = _component_for_key(spec, key)
        child_spec = child_by_key.get(component)
        child_path = (*prefix, component)
        if child_spec is None:
            yield child_path
        else:
            yield from _leaf_paths(child_spec, prefix=child_path)


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


__all__ = ["Container", "container_from_op", "reconstruct_output"]
