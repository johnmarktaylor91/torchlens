"""Cross-kind key resolution shared by the TorchLens indexing doors.

Every door returns the kind of object its name promises. The plain door
``trace[key]`` returns the kind the key names: an op label gives the Op, a
module address the Module, ``address:N`` the ModuleCall, a parameter address
the Param, a buffer address the Buffer, a grad_fn label the GradFn. A typed
door (``trace.ops``, ``trace.layers``, ``trace.modules``,
``trace.module_calls``, ``trace.params``, ``trace.buffers``,
``trace.autograd.grad_fns``) always returns its own kind and converts a key
that names exactly one such object (``trace.ops["encoder.mlp"]`` is the
module's output Op, ``trace.modules["relu_1_2"]`` the Module that op ran
in). A key naming several objects of the door's kind refuses with
``AmbiguousOpLookupError`` (``ambiguous_op_lookup``) and a remedy spelling
the ``:N`` forms. Each helper returns ``None`` when the key names nothing of
that kind, so the door's own not-found error still fires.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from .._errors import AmbiguousOpLookupError


def owning_trace(records: Iterable[Any]) -> Any:
    """The owning Trace of the first record that still reaches one, else ``None``."""

    for record in records:
        trace = None
        for name in ("_source_trace_ref", "_source_ref"):
            source_ref = getattr(record, name, None)
            trace = source_ref() if callable(source_ref) else None
            if trace is not None:
                break
        if trace is None:
            trace = getattr(record, "_source_trace", None)
        if trace is not None:
            return trace
    return None


def accessor_trace(accessor: Any) -> Any:
    """The Trace an accessor belongs to: its own weak reference, else its records' owner."""

    source_ref = getattr(accessor, "_source_ref", None)
    trace = source_ref() if callable(source_ref) else None
    return trace if trace is not None else owning_trace(getattr(accessor, "_list", ()))


def converted(accessor: Any, key: str, resolver: Any) -> Any:
    """Run a typed-door conversion for ``accessor`` and keep the result only if it is a member.

    A filtered collection (``trace.params.trainable()``, ``trace.ops.in_module(...)``)
    converts a key only to one of its own records, so a key naming a record outside
    the collection still falls through to the door's not-found error.
    """

    exact = getattr(accessor, "_dict", {}).get(key)
    if exact is not None:
        return exact  # a key that names a member of this door exactly is that member
    trace = accessor_trace(accessor)
    if trace is None:
        return None
    record = resolver(trace, key)
    if record is None or not any(record is item for item in accessor._list):
        return None
    return record


def refuse_several(kind: str, key: str, spellings: Sequence[str], noun: str) -> None:
    """Raise the ambiguity refusal naming every object ``key`` reaches."""

    listed = ", ".join(repr(spelling) for spelling in spellings[:10])
    suffix = "..." if len(spellings) > 10 else ""
    raise AmbiguousOpLookupError(
        f"{key!r} names {len(spellings)} {noun} on this trace, so no single {kind} "
        f"answers it. Use one of: {listed}{suffix}.",
        key=key,
        candidates=tuple(spellings),
    )


# --------------------------------------------------------------------------- #
# The plain door: the kind the key names.
# --------------------------------------------------------------------------- #


def named_record(trace: Any, key: str) -> Any:
    """Return the Module, ModuleCall, Param or Buffer an exact key names, else ``None``.

    Exact spellings only: a module address or alias, a module-call label
    (``address:N``), a parameter address or a buffer address. Op labels and
    every inexact spelling are left to the caller's own cascade.
    """

    modules = trace.modules
    if key in getattr(modules, "_dict", {}) or key in getattr(modules, "_alias_dict", {}):
        return modules[key]
    module_call = getattr(modules, "_pass_dict", {}).get(key)
    if module_call is not None:
        return module_call
    params = trace.params
    if key in getattr(params, "_dict", {}):
        return params[key]
    buffers = trace.buffers
    if key in getattr(buffers, "_dict", {}):
        return buffers[key]
    return None


def op_at(trace: Any, key: Any) -> Any:
    """Return the Op ``key`` names on ``trace``, converting a record key to its output Op.

    The internal spelling for callers that accept any lookup key but need an
    Op: ``trace[key]`` for an Op (or a selector result), ``trace.ops[key]`` when
    the plain door answers with a Layer, Module, ModuleCall or Buffer (a bare
    multi-pass layer label refuses there, naming the pass spellings).
    """

    record = trace[key]
    if isinstance(key, str) and type(record).__name__ in {
        "Layer",
        "Module",
        "ModuleCall",
        "Buffer",
    }:
        return trace.ops[key]
    return record


# --------------------------------------------------------------------------- #
# Typed doors: convert a key that names exactly one object of the door's kind.
# --------------------------------------------------------------------------- #


def _module_at(trace: Any, key: str) -> Any:
    """The Module an address, alias or call label names, else ``None``."""

    modules = trace.modules
    if key in getattr(modules, "_dict", {}) or key in getattr(modules, "_alias_dict", {}):
        return modules[key]
    module_call = getattr(modules, "_pass_dict", {}).get(key)
    return None if module_call is None else modules[module_call.address]


def _output_op_candidates(trace: Any, key: str) -> tuple[list[Any], list[str]]:
    """Output ops a record key reaches, with the spellings that name each one.

    A module address reaches the output ops of every call (spelled by call
    label when each call has one output), ``address:N`` the output ops of that
    call, a buffer address one op per version (``address:N`` that version). An op label is not converted:
    the ops door answers it itself or refuses, so validation keeps its tripwire.
    """

    keys = trace._layer_dict_all_keys
    pass_dict = getattr(trace.modules, "_pass_dict", {})
    call = pass_dict.get(key)
    if call is not None:
        labels = [label for label in (call.output_ops or []) if label in keys]
        return [keys[label] for label in labels], labels
    module = _module_at(trace, key)
    if module is not None:
        calls = [pass_dict[label] for label in module.call_labels if label in pass_dict]
        per_call = [[label for label in (c.output_ops or []) if label in keys] for c in calls]
        labels = [label for labels in per_call for label in labels]
        one_each = all(len(labels) == 1 for labels in per_call)
        return [keys[label] for label in labels], [c.label for c in calls] if one_each else labels
    buffers = getattr(trace.buffers, "_dict", {})
    if key in buffers:
        versions = list(trace.buffers[key].versions)
        return versions, [f"{key}:{index}" for index in range(1, len(versions) + 1)]
    address, _, version = key.rpartition(":")
    if address in buffers and version.isdigit():
        versions = list(trace.buffers[address].versions)
        if 1 <= int(version) <= len(versions):
            return [versions[int(version) - 1]], [key]
    return [], []


def output_op(trace: Any, key: str) -> Any:
    """The output Op a module, module-call or buffer key names, else ``None``.

    Refuses a module address called several times (one output Op per call)
    and a buffer address with several versions, naming the ``:N`` spellings.
    """

    if key in (getattr(trace, "_ambiguous_lookup_keys", {}) or {}):
        trace[key]  # the plain door raises the ambiguity refusal for this key
    ops, spellings = _output_op_candidates(trace, key)
    if len(ops) > 1:
        refuse_several("Op", key, spellings, "output ops")
    return ops[0] if ops else None


def layer_for_key(trace: Any, key: str) -> Any:
    """The Layer a module, module-call or buffer key names, else ``None``.

    A module called several times still names one Layer when every call's
    output op belongs to the same Layer; otherwise the refusal lists the calls.
    """

    ops, spellings = _output_op_candidates(trace, key)
    labels = list(dict.fromkeys(op.layer_label for op in ops))
    if len(labels) > 1:
        refuse_several("Layer", key, spellings, "output ops in different Layers")
    return trace.layers[labels[0]] if labels else None


def _ops_named(trace: Any, key: str) -> list[Any]:
    """Every Op a key names: the passes of an op or layer label, else a record key's output ops."""

    try:
        ops = list(trace.ops.resolve_all(key))
    except (KeyError, ValueError, AttributeError):
        ops = []
    return ops or _output_op_candidates(trace, key)[0]


def _call_label_address(call_label: Any) -> str | None:
    """The module address of a ``address:N`` call label."""

    if not isinstance(call_label, str) or not call_label:
        return None
    return call_label.rpartition(":")[0] or call_label


def module_for_key(trace: Any, key: str) -> Any:
    """The Module an op label, parameter address or buffer address names, else ``None``.

    An op names the innermost module whose call produced it; a bare multi-pass
    label names one Module only when every pass ran in the same module.
    """

    ops = _ops_named(trace, key)
    if ops:
        addresses = list(
            dict.fromkeys(
                address
                for op in ops
                if (address := _call_label_address(getattr(op, "module", None))) is not None
            )
        )
        if len(addresses) > 1:
            refuse_several("Module", key, [op.label for op in ops], "passes in different modules")
        return trace.modules[addresses[0]] if addresses else None
    for accessor in (trace.params, trace.buffers):
        if key in getattr(accessor, "_dict", {}):
            address = getattr(accessor[key], "module_address", None)
            return trace.modules[address or "self"]
    return None


def module_call_for_key(trace: Any, key: str) -> Any:
    """The ModuleCall an op label names (the innermost call that produced it), else ``None``."""

    ops = _ops_named(trace, key)
    labels = list(
        dict.fromkeys(
            label for op in ops if isinstance(label := getattr(op, "module", None), str) and label
        )
    )
    if len(labels) > 1:
        refuse_several("ModuleCall", key, labels, "calls")
    return trace.module_calls[labels[0]] if labels else None


def _record_for_module_or_op(trace: Any, key: str, kind: str, attribute: str) -> Any:
    """The single Param or Buffer (``attribute``) a module address or op label reaches."""

    module = _module_at(trace, key)
    if module is not None:
        owned = getattr(module, attribute, None)
        addresses = [str(address) for address in getattr(owned, "keys", list)()]
    else:
        ops = _ops_named(trace, key)
        if not ops:
            return None
        addresses = list(
            dict.fromkeys(
                str(address)
                for op in ops
                for address in getattr(getattr(op, attribute, None), "keys", list)()
            )
        )
    if len(addresses) > 1:
        refuse_several(kind, key, addresses, f"{attribute}")
    if not addresses:
        return None
    return getattr(trace, attribute)[addresses[0]]


def param_for_key(trace: Any, key: str) -> Any:
    """The one Param a module address or op label reaches, else ``None``."""

    return _record_for_module_or_op(trace, key, "Param", "params")


def buffer_for_key(trace: Any, key: str) -> Any:
    """The one Buffer a module address reaches, or the Buffer of a buffer-version op, else ``None``."""

    ops = _ops_named(trace, key)
    if len(ops) == 1 and getattr(ops[0], "is_buffer", False):
        for buffer in trace.buffers:
            if any(version is ops[0] for version in getattr(buffer, "versions", ())):
                return buffer
    return _record_for_module_or_op(trace, key, "Buffer", "buffers")


def grad_fn_for_key(trace: Any, key: str) -> Any:
    """The GradFn an op label names (the op's own grad_fn), else ``None``."""

    ops = _ops_named(trace, key)
    labels = list(
        dict.fromkeys(
            label for op in ops if isinstance(label := getattr(op, "grad_fn_label", None), str)
        )
    )
    if len(labels) > 1:
        refuse_several("GradFn", key, [op.label for op in ops], "passes with their own grad_fn")
    return trace.autograd.grad_fns[labels[0]] if labels else None


# --------------------------------------------------------------------------- #
# Scoped containers: Module[...] and Layer.ops[...]
# --------------------------------------------------------------------------- #


def module_item(module: Any, key: str) -> Any:
    """Resolve ``module[key]`` for a string key: the kind the key names, inside this module.

    A layer label of this module gives what ``trace[label]`` gives (its Layer,
    or the Op of ``label:N``); a descendant module address gives that
    Module and a descendant call label its ModuleCall; a substring naming one
    layer of this module gives that record.
    """

    trace = module._source_trace
    if key in module.layer_labels or key.rsplit(":", 1)[0] in module.layer_labels:
        return trace[key]
    descendant = _module_at(trace, key)
    address = getattr(descendant, "address", None)
    if descendant is not None and (
        module.address == "self"
        or address == module.address
        or (isinstance(address, str) and address.startswith(module.address + "."))
    ):
        module_call = getattr(trace.modules, "_pass_dict", {}).get(key)
        return module_call if module_call is not None else descendant
    matches = [label for label in module.layer_labels if key in label]
    if len(matches) == 1:
        return trace[matches[0]]
    if len(matches) > 1:
        refuse_several("layer", key, matches, f"layers of module '{module.address}'")
    raise KeyError(f"'{key}' not found in module '{module.address}' layers")


def module_call_owning_op(calls: Iterable[Any], key: str) -> Any:
    """The one ModuleCall among ``calls`` whose ops include ``key`` (pass-qualified or bare label)."""

    matches = [
        call
        for call in calls
        if any(label == key or label.rpartition(":")[0] == key for label in call.ops)
    ]
    if len(matches) > 1:
        refuse_several("ModuleCall", key, [call.label for call in matches], "module calls")
    return matches[0] if matches else None


def scoped_op_by_lookup_key(ops: Iterable[Any], key: str) -> Any:
    """The one op among ``ops`` whose lookup keys include ``key`` (a module-call or buffer key)."""

    matches = [op for op in ops if key in (getattr(op, "lookup_keys", None) or ())]
    if len(matches) > 1:
        refuse_several("Op", key, [op.label for op in matches], "ops")
    return matches[0] if matches else None
