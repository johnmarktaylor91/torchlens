"""Module records built on first lookup: the facts step 16 seals and the deferred build.

Step 16 keeps only the work that observes live objects or writes op fields
(multi-output role names, the root call's argument summaries, the uncalled
module census, gradient-record owner naming). Everything else it used to
build (the ``Module`` and ``ModuleCall`` records, the buffer accessor, the
flat module-call accessor and ``_param_logs_by_module``) is built from the
plain facts held in one ``PendingModules`` object the first time any door
reads one of those trace fields. The trace holds no ``Module`` or
``ModuleCall`` object while it is pending.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .._data_classes._module_facade_fields import format_call_arg
from .._data_classes._module_role_hints import (
    multi_output_role_from_path,
    role_hints_for_module_class,
)
from .._data_classes._op_column_scan import call_outputs, ops_with
from .._io._buffer_versions import seal_buffer_versions
from .._ir.workspaces import ModuleCaptureWorkspace, _init_module_hierarchy_data
from ._module_census import uncalled_module_addresses

if TYPE_CHECKING:
    from .._data_classes.op import Op
    from .._data_classes.trace import Trace

#: Trace attribute holding the pending facts; absent once the records are built.
PENDING_ATTR = "_module_pending"
#: Trace fields the deferred build fills; a read of any of them builds first.
LAZY_MODULE_FIELDS = frozenset({"_module_logs", "_buffer_accessor", "_param_logs_by_module"})


#: Marks a trace that had no module workspace when its records were built.
_NO_WORKSPACE = object()


class PendingModules:
    """Plain facts the deferred module build reads, captured at step 16.

    Every value is a dict, list, tuple, string, number, class or argument
    stub; buffer versions are op labels. A fork shares this object by
    reference and builds its own records from it.
    """

    __slots__ = (
        "module_build_data",
        "module_metadata",
        "module_forward_args",
        "root_forward_args",
        "root_summaries",
        "outputs_by_call",
        "input_layers_by_call",
        "buffer_versions",
        "module_count",
        "attribute_writes",
        "build",
    )
    module_build_data: dict[str, Any]
    module_metadata: dict[Any, Any]
    module_forward_args: dict[Any, Any]
    root_forward_args: Any
    root_summaries: tuple[str, str] | None
    outputs_by_call: Any
    input_layers_by_call: dict[str, list[str]]
    buffer_versions: dict[str, list[str]]
    module_count: int
    attribute_writes: dict[str, Any] | None
    build: Callable[[Any, PendingModules], None]

    def __init__(self, **facts: Any) -> None:
        """Store the captured facts by name (every slot is required)."""

        for name in self.__slots__:
            object.__setattr__(self, name, facts[name])

    def __deepcopy__(self, memo: dict[int, Any]) -> PendingModules:
        """Share the immutable captured facts instead of copying them."""

        return self

    def read(self, trace: Any, name: str) -> tuple[bool, Any]:
        """Build ``trace``'s records on a read of a field the build fills.

        Returns ``(True, value)`` for one of ``LAZY_MODULE_FIELDS`` and
        ``(False, None)`` for any other name.
        """

        if name not in LAZY_MODULE_FIELDS:
            return False, None
        self.ensure(trace)
        return True, trace.__dict__[name]

    def ensure(self, trace: Any) -> None:
        """Build the records of ``trace`` from these facts, once."""

        state = trace.__dict__
        if state.get(PENDING_ATTR) is not self:
            return
        del state[PENDING_ATTR]
        # The builders read the sealed facts through the module workspace
        # they came from.
        prior_ws = state.get("_module_capture_ws", _NO_WORKSPACE)
        state["_module_capture_ws"] = ModuleCaptureWorkspace(
            module_build_data=self.module_build_data,
            module_metadata=self.module_metadata,
            module_forward_args=self.module_forward_args,
            root_forward_args=self.root_forward_args,
        )
        try:
            self.build(trace, self)
        except BaseException:
            for name in LAZY_MODULE_FIELDS:
                state.pop(name, None)
            state[PENDING_ATTR] = self
            raise
        finally:
            if prior_ws is _NO_WORKSPACE:
                del state["_module_capture_ws"]
            else:
                state["_module_capture_ws"] = prior_ws


def seal_module_facts(trace: Trace, build: Callable[[Any, PendingModules], None]) -> None:
    """Step 16: do the seal-time module work and leave the rest pending.

    Parameters
    ----------
    trace:
        Trace whose module capture workspace is consumed.
    build:
        The deferred build, called with the trace and these facts.
    """

    ws = trace._module_capture_ws
    mbd = ws.module_build_data
    metadata = ws.module_metadata
    outputs_by_call = call_outputs(trace)
    _name_output_roles(trace, mbd, metadata, outputs_by_call)
    input_layers_by_call: dict[str, list[str]] = defaultdict(list)
    for op in ops_with(trace, "input_to_module_calls"):  # only ops that feed a call
        for call_label in op._module_context_input_to_module_calls:
            input_layers_by_call[call_label].append(op.layer_label)
    root_args, root_summaries = _seal_root_args(trace, ws.root_forward_args)
    state = trace.__dict__
    state["_uncalled_module_addresses"] = uncalled_module_addresses(
        trace, _called_addresses(mbd, metadata)
    )
    state[PENDING_ATTR] = PendingModules(
        module_build_data=mbd,
        module_metadata=metadata,
        module_forward_args=ws.module_forward_args,
        root_forward_args=root_args,
        root_summaries=root_summaries,
        outputs_by_call=outputs_by_call,
        input_layers_by_call=input_layers_by_call,
        buffer_versions=seal_buffer_versions(trace),
        module_count=len(mbd["addresses"]) - ("self" in mbd["addresses"]),
        attribute_writes=None,  # the capture's observer bracket fills it
        build=build,
    )
    for name in LAZY_MODULE_FIELDS:
        state.pop(name, None)
    ws.module_metadata = {}
    ws.module_forward_args = {}
    ws.root_forward_args = None
    ws.module_build_data = _init_module_hierarchy_data()


def _name_output_roles(
    trace: Trace,
    mbd: dict[str, Any],
    metadata: dict[Any, Any],
    outputs_by_call: Any,
) -> None:
    """Write ``multi_output_name`` on the outputs of every call with several outputs.

    Calls are visited in the order the eager build used (module order, then
    call index), so the first call to name a shared output wins as before.
    """

    multi = [label for label, outputs in outputs_by_call.items() if len(outputs) > 1]
    if not multi:
        return
    order = {address: index for index, address in enumerate(mbd["addresses"])}
    by_alias = _metadata_by_alias(metadata)
    keyed = []
    for label in multi:
        address, _, call_index = label.rpartition(":")
        if address != "self" and address in order:
            keyed.append((order[address], int(call_index), address, label))
    ops = trace.ops
    for _order, _index, address, label in sorted(keyed):
        outputs = outputs_by_call[label]
        pass_layers = mbd["module_pass_layers"].get(label, [])
        entries = [ops[op_label] for op_label in pass_layers if op_label in outputs]
        cls = by_alias.get(address, {}).get("cls")
        _assign_output_roles(entries, hints=role_hints_for_module_class(cls))


def _assign_output_roles(
    output_entries: list[Op],
    *,
    hints: Any | None,
) -> None:
    """Populate semantic multi-output roles on output entries.

    Parameters
    ----------
    output_entries:
        Output OpLogs from one module call.
    hints:
        Optional role-hint mapping keyed by primitive output paths.
    """

    if len(output_entries) <= 1:
        return
    for index, output in enumerate(output_entries):
        if output.multi_output_name is not None:
            parent_layer = output.source_trace._layer_logs.get(output.layer_label)
            if parent_layer is not None:
                parent_layer.multi_output_name = output.multi_output_name
            continue
        output.multi_output_name = multi_output_role_from_path(
            tuple(output.container_path or ()),
            output.multi_output_index if output.multi_output_index is not None else index,
            hints=hints,
        )
        parent_layer = output.source_trace._layer_logs.get(output.layer_label)
        if parent_layer is not None:
            parent_layer.multi_output_name = output.multi_output_name


def _metadata_by_alias(metadata: dict[Any, Any]) -> dict[str, dict[str, Any]]:
    """Map every alias of a shared module to its captured metadata."""

    by_alias: dict[str, dict[str, Any]] = {}
    for primary, meta in metadata.items():
        for alias in meta.get("all_addresses", [primary]):
            by_alias[alias] = meta
    return by_alias


def _called_addresses(mbd: dict[str, Any], metadata: dict[Any, Any]) -> set[str]:
    """Return every address and alias of the modules the build will record."""

    called = set(mbd["addresses"])
    called.add("self")
    called.update(metadata.get("self", {}).get("all_addresses", ["self"]))
    by_alias = _metadata_by_alias(metadata)
    for address in mbd["addresses"]:
        if address != "self":
            called.update(by_alias.get(address, {}).get("all_addresses", [address]))
    return called


def _seal_root_args(trace: Trace, root_args: Any) -> tuple[Any, tuple[str, str] | None]:
    """Summarise the root call's live arguments now and keep only their shape.

    The identity modes that keep arguments keep the live objects, as the
    eager build did; otherwise the pending facts hold placeholders of the
    same lengths so the argument counts come out the same.
    """

    if root_args is None:
        return None, None
    args, kwargs = root_args
    summaries = (format_call_arg(args), format_call_arg(kwargs))
    if trace._options_module_identity_mode in {"pytree_module", "object_module"}:
        return root_args, summaries
    shape_args = None if args is None else (None,) * len(args)
    shape_kwargs = None if kwargs is None else dict.fromkeys(kwargs)
    return (shape_args, shape_kwargs), summaries


def ensure_modules_built(trace: Any) -> None:
    """Build the pending module records of ``trace`` once; a no-op when built."""

    pending = trace.__dict__.get(PENDING_ATTR)
    if pending is not None:
        pending.ensure(trace)
