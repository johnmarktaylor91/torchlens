"""Generate the checked-in per-class StorageBinding tables.

One declared schema, generated artifacts (docs/reference/trace_core_design.md
section 3.4): this tool derives one ``StorageBinding`` per declared field for
all 12 record classes and writes ``torchlens/_data_classes/_schema_bindings.py``
(the ``trace`` table goes to its sibling ``_schema_bindings_trace.py``) as
reviewable, deterministic source. A CI test regenerates and diffs, so the
bindings can never drift from the classes silently.

Classification is mechanical and conservative; the storage kinds refine as
each family columnarizes (BITSET/GROUP/PAYLOAD/EDGE assignments here are the
declared intent from the converged design, not yet physical reality).

Run: python tools/generate_record_schema.py [--check]
"""

from __future__ import annotations

import sys
import types
import typing
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

_OUT_PATH = _REPO_ROOT / "torchlens" / "_data_classes" / "_schema_bindings.py"
#: The ``trace`` table (about a third of all bindings) is emitted into its own
#: generated module, which ``_schema_bindings`` imports, so neither generated
#: file outgrows the file-size cap.
_SPLIT_KEY = "trace"
_SPLIT_NAME = "TRACE_BINDINGS"
_SPLIT_OUT_PATH = _REPO_ROOT / "torchlens" / "_data_classes" / "_schema_bindings_trace.py"

_HEADER = '''"""GENERATED storage bindings for the declared record schema. DO NOT EDIT.

Regenerate with ``python tools/generate_record_schema.py``; a CI test
(`tests/test_record_schema_bindings.py`) diffs this file against a fresh
generation, so hand edits cannot survive. The declared axes live in
``field_policy.StorageBinding``; classification policy lives in the generator.
"""

from __future__ import annotations

from ._schema_bindings_trace import TRACE_BINDINGS
from .field_policy import StorageBinding, StorageKind

'''

_SPLIT_HEADER = '''"""GENERATED storage bindings for the ``trace`` record. DO NOT EDIT.

The ``trace`` table of ``_schema_bindings.STORAGE_BINDINGS``, emitted as its own
module so neither generated file outgrows the file-size cap. Regenerate with
``python tools/generate_record_schema.py``; a CI test
(`tests/test_record_schema_bindings.py`) diffs this file against a fresh
generation, so hand edits cannot survive.
"""

from __future__ import annotations

from .field_policy import StorageBinding, StorageKind

'''

#: Fields whose storage is the payload arena (tensor values / lazy blobs).
_PAYLOAD_FIELDS = {
    "out",
    "grad",
    "transformed_out",
    "transformed_grad",
    "saved_args",
    "saved_kwargs",
    "out_versions_by_child",
}

#: Ancestor-closure fields backed by the interned ancestry pool.
_BITSET_FIELDS = {"root_ancestors", "internal_source_ancestors"}

#: Group-shared fields (content lives once per group block).
_GROUP_FIELDS = {"equivalent_ops", "recurrent_ops"}

#: Edge fields the torch backend ALSO overlays with slot-preserving
#: properties (staging-set/frozen-view accessors on ``Op``); like the
#: bitset pair they must classify before the property check or the
#: result depends on whether the backend module is imported.
_OVERLAY_EDGE_FIELDS = {"input_ancestors", "output_descendants"}

#: Copy-on-read alias-barrier fields (FORK-1 default: fresh copies).
_COPY_ON_READ_FIELDS = {"equivalent_ops", "recurrent_ops"}

#: Variable-cardinality relation fields destined for the edge-occurrence
#: table (labels/ids referencing sibling records).
_EDGE_FIELDS = {
    "children",
    "parents",
    "parent_layers",
    "child_layers",
    "input_ancestors",
    "output_descendants",
    "internally_initialized_ancestors",
    "internally_initialized_parents",
    "internal_descendants",
    "orig_ancestors",
    "modules",
    "module_calls",
    "containing_modules_origin_nested",
    "module_passes_entered",
    "module_passes_exited",
    "output_of_modules",
    "output_of_module_calls",
    "input_of_modules",
    "input_of_module_calls",
    "layer_labels",
    "layers",
    "ops",
    "input_ops",
    "output_ops",
    "input_layers",
    "output_layers",
    "grad_fns",
    "parent_params",
    "parent_param_passes",
    "params",
    "buffers",
}

# Primitive-kind relations are deliberately scoped by schema key: ``module_call_stack``
# is not an edge column on the existing Op schema, and adding it to the global set would
# silently change an installed binding outside L3's fenced kind.
_PRIMITIVE_EDGE_FIELDS = frozenset({"parent_op_refs", "module_call_stack"})

# Intended post-bump intern pool for the primitive kind. The installed wave-0 policies
# are all DROP, so the DROP-before-pool classifier keeps every checked-in binding RUNTIME.
_PRIMITIVE_POOLED_SLOTS = frozenset(
    {
        "label",
        "namespace",
        "operator",
        "overload",
        "schema",
        "schema_fingerprint",
        "autocast_context",
        "dispatch_key_context",
        "execution_context",
    }
)


def _annotation_for(cls: type, name: str) -> str | None:
    """Return the class-declared annotation string for a field, if any."""

    for mro_cls in cls.__mro__:
        annotations = mro_cls.__dict__.get("__annotations__", {})
        if name in annotations:
            return _render_annotation(annotations[name])
    return None


def _render_annotation(annotation: Any) -> str:
    """Render one annotation to a string that is identical on every Python.

    String annotations (modules under ``from __future__ import annotations``)
    pass through, and evaluated non-unions keep their ``repr``. Evaluated
    unions are canonicalized, because ``repr`` of a union differs by
    interpreter (``Any | None`` evaluates to
    ``typing.Optional[typing.Any]`` on 3.10 but to a ``types.UnionType``
    reading ``typing.Any | None`` on 3.11+), which made the generated file
    stale on every interpreter but the one that last wrote it. Unions render
    as their ``|``-joined members in declaration order.

    Parameters
    ----------
    annotation:
        Class-declared annotation object or string.

    Returns
    -------
    str
        Interpreter-independent annotation text.
    """

    if isinstance(annotation, str):
        return annotation
    if _is_union(annotation):
        return " | ".join(_render_union_member(arg) for arg in typing.get_args(annotation))
    return repr(annotation)


def _is_union(annotation: Any) -> bool:
    """Return whether ``annotation`` is a ``typing.Union`` or ``X | Y`` union."""

    return typing.get_origin(annotation) is typing.Union or isinstance(annotation, types.UnionType)


def _render_union_member(member: Any) -> str:
    """Render one union member the way a ``types.UnionType`` repr names it.

    Parameters
    ----------
    member:
        One argument of a union annotation.

    Returns
    -------
    str
        ``None`` for ``NoneType``, a bare name for builtins, a
        module-qualified name for other plain classes, else ``repr``.
    """

    if member is type(None):
        return "None"
    if _is_union(member):
        return " | ".join(_render_union_member(arg) for arg in typing.get_args(member))
    if isinstance(member, type) and not isinstance(member, types.GenericAlias):
        if member.__module__ == "builtins":
            return member.__qualname__
        return f"{member.__module__}.{member.__qualname__}"
    return repr(member)


def _is_property(cls: type, name: str) -> bool:
    """Return whether a declared field resolves to a property descriptor."""

    for mro_cls in cls.__mro__:
        if name in mro_cls.__dict__:
            return isinstance(mro_cls.__dict__[name], property)
    return False


def _classify(
    schema_key: str,
    cls: type,
    name: str,
    policy: Any,
    container_defaults: dict[str, Any],
    pooled_slots: frozenset[str],
) -> tuple[str, str]:
    """Return (StorageKind name, mutability) for one declared field."""

    from torchlens._io import FieldPolicy

    # Ancestor/group fields come FIRST: the torch backend installs lazy
    # property overlays over these slots at import time, so a property check
    # on them would be import-order-dependent.
    if name in _COPY_ON_READ_FIELDS:
        return "GROUP", "copy_on_read"
    if name in _BITSET_FIELDS:
        return "BITSET", "mutable_container"
    if name in _GROUP_FIELDS:
        return "GROUP", "mutable_container"
    if _is_property(cls, name) and name not in _OVERLAY_EDGE_FIELDS:
        return "COMPUTED", "immutable"
    if policy.portable_policy is FieldPolicy.DROP:
        return "RUNTIME", "immutable"
    if name in _PAYLOAD_FIELDS:
        return "PAYLOAD", "immutable"
    default = container_defaults.get(name)
    is_container = isinstance(default, (list, dict, set))
    if name in _EDGE_FIELDS or (schema_key == "primitive_op" and name in _PRIMITIVE_EDGE_FIELDS):
        return "EDGE", "mutable_container" if is_container else "immutable"
    if is_container:
        return "SCALAR", "mutable_container"
    if name in pooled_slots:
        return "INTERNED", "immutable"
    return "SCALAR", "immutable"


def _collect() -> dict[str, list[tuple[str, str, str | None, str]]]:
    """Build the schema-key -> [(field, kind, annotation, mutability)] map."""

    from torchlens._data_classes import op as op_module
    from torchlens._data_classes.aten_op import AtenOp
    from torchlens._data_classes.backward_pass import BackwardPass
    from torchlens._data_classes.buffer import Buffer
    from torchlens._data_classes.func_call_location import FuncCallLocation
    from torchlens._data_classes.grad_fn import GradFn
    from torchlens._data_classes.grad_fn_call import GradFnCall
    from torchlens._data_classes.layer import Layer
    from torchlens._data_classes.module import Module, ModuleCall
    from torchlens._data_classes.op import Op
    from torchlens._data_classes.param import Param
    from torchlens._data_classes.trace import Trace

    op_container_defaults = dict(getattr(op_module, "_LAYER_PASS_LOG_CONTAINER_DEFAULTS", {}))
    op_container_defaults.update(getattr(op_module, "_LAYER_PASS_LOG_DEFAULT_FILL", {}))
    pooled_slots = frozenset(getattr(op_module, "_POOLED_SLOTS", ()))

    classes: list[tuple[str, type, dict[str, Any], frozenset[str]]] = [
        ("trace", Trace, {}, frozenset()),
        ("op", Op, op_container_defaults, pooled_slots),
        ("layer", Layer, {}, frozenset()),
        ("module", Module, {}, frozenset()),
        ("module_call", ModuleCall, {}, frozenset()),
        ("param", Param, {}, frozenset()),
        ("buffer", Buffer, {}, frozenset()),
        ("grad_fn", GradFn, {}, frozenset()),
        ("grad_fn_call", GradFnCall, {}, frozenset()),
        ("backward_pass", BackwardPass, {}, frozenset()),
        ("func_call_location", FuncCallLocation, {}, frozenset()),
        ("primitive_op", AtenOp, {}, _PRIMITIVE_POOLED_SLOTS),
    ]

    schema: dict[str, list[tuple[str, str, str | None, str]]] = {}
    for key, cls, container_defaults, pooled in classes:
        rows: list[tuple[str, str, str | None, str]] = []
        for name, policy in cls._FIELD_POLICY.items():
            kind, mutability = _classify(key, cls, name, policy, container_defaults, pooled)
            rows.append((name, kind, _annotation_for(cls, name), mutability))
        schema[key] = rows
    return schema


def collect_primitive_candidate_bindings() -> dict[str, str]:
    """Return isolated post-bump primitive storage kinds for S3 review.

    Returns
    -------
    dict[str, str]
        Primitive field name to intended ``StorageKind`` name with every
        portable policy virtually promoted from DROP to KEEP. This does not
        alter the installed generated bindings or the registrar switch.
    """

    from dataclasses import replace

    from torchlens._data_classes.aten_op import AtenOp
    from torchlens._io import FieldPolicy

    candidate: dict[str, str] = {}
    for name, policy in AtenOp._FIELD_POLICY.items():
        promoted = replace(policy, portable_policy=FieldPolicy.KEEP)
        kind, _ = _classify(
            "primitive_op",
            AtenOp,
            name,
            promoted,
            {},
            _PRIMITIVE_POOLED_SLOTS,
        )
        candidate[name] = kind
    return candidate


def _binding_line(row: tuple[str, str, str | None, str], indent: str) -> str:
    """Render one ``"field": StorageBinding(...),`` source line."""

    name, kind, annotation, mutability = row
    parts = [f"StorageKind.{kind}"]
    if annotation is not None:
        parts.append(f"annotation={annotation!r}")
    if mutability != "immutable":
        parts.append(f'mutability="{mutability}"')
    return f'{indent}"{name}": StorageBinding({", ".join(parts)}),\n'


def _render(schema: dict[str, list[tuple[str, str, str | None, str]]]) -> str:
    """Render the bindings module source deterministically."""

    lines = [_HEADER]
    lines.append("STORAGE_BINDINGS: dict[str, dict[str, StorageBinding]] = {\n")
    for key, rows in schema.items():
        if key == _SPLIT_KEY:
            lines.append(f'    "{key}": {_SPLIT_NAME},\n')
            continue
        lines.append(f'    "{key}": {{\n')
        lines.extend(_binding_line(row, "        ") for row in rows)
        lines.append("    },\n")
    lines.append("}\n")
    return "".join(lines)


def _render_split(schema: dict[str, list[tuple[str, str, str | None, str]]]) -> str:
    """Render the split-out ``trace`` bindings module source deterministically."""

    lines = [_SPLIT_HEADER]
    lines.append(f"{_SPLIT_NAME}: dict[str, StorageBinding] = {{\n")
    lines.extend(_binding_line(row, "    ") for row in schema[_SPLIT_KEY])
    lines.append("}\n")
    return "".join(lines)


def main() -> int:
    """Generate (or with --check, verify) the bindings modules."""

    schema = _collect()
    outputs = ((_OUT_PATH, _render(schema)), (_SPLIT_OUT_PATH, _render_split(schema)))
    if "--check" in sys.argv:
        stale = [
            path.name
            for path, rendered in outputs
            if (path.read_text() if path.exists() else "") != rendered
        ]
        for name in stale:
            print(f"stale: {name} differs from a fresh generation")
        if stale:
            return 1
        print("ok: _schema_bindings.py and _schema_bindings_trace.py are current")
        return 0
    for path, rendered in outputs:
        path.write_text(rendered)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
