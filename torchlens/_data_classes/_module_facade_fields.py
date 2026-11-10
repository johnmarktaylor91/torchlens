"""Stored-field tables for the ``ModuleCall`` and ``Module`` record facades.

``module.py`` installs these through ``install_module_facades`` at import time.
The two row builders turn a constructor's arguments into one row in the order
of the tables, so a record is written once instead of field by field.
"""

from __future__ import annotations

from typing import Any

from .._quantities import Duration
from .._trace_core.op_store import DetachedOpStore
from .._trace_core.record_refs import install_child_fields
from .._trace_core.record_rows import CORE_KEY, install_record_facade
from ._summary import format_call_arg

# ---------------------------------------------------------------------------
# The facades: declared stored fields become row-cell descriptors; the
# instance dict keeps the store binding, user extras, and the handful of
# names whose access paths hardcode ``__dict__`` (the template/source-trace/
# facets slots poked directly by their properties).
# ---------------------------------------------------------------------------

_MODULE_CALL_STORED_FIELDS: tuple[str, ...] = (
    "address",
    "all_addresses",
    "cls",
    "class_name",
    "class_qualname",
    "call_index",
    "label",
    "ordinal_index",
    "ops",
    "input_layers",
    "output_layers",
    "input_ops",
    "output_ops",
    "output_structure",
    "output_paths",
    "forward_args",
    "forward_kwargs",
    "inputs_before_pre_hooks",
    "inputs_after_pre_hooks",
    "forward_pre_hook_effects",
    "forward_arg_names",
    "num_forward_pos_args",
    "num_forward_kwargs",
    "num_forward_args_total",
    "forward_args_summary",
    "forward_kwargs_summary",
    "forward_duration",
    "code_context",
    "module_call_stack",
    "call_parent",
    "call_children",
)

_MODULE_STORED_FIELDS: tuple[str, ...] = (
    "address",
    "all_addresses",
    "cls",
    "class_name",
    "class_qualname",
    "class_source_file",
    "class_source_line",
    "init_source_file",
    "init_source_line",
    "forward_source_file",
    "forward_source_line",
    "class_docstring",
    "init_signature",
    "init_docstring",
    "forward_signature",
    "forward_docstring",
    "address_parent",
    "address_children",
    "address_depth",
    "call_parent",
    "call_children",
    "call_depth",
    "num_calls",
    "ops",
    "call_labels",
    "layer_labels",
    "input_ops",
    "input_layers",
    "output_ops",
    "output_layers",
    "params",
    "num_params",
    "num_params_trainable",
    "num_params_frozen",
    "buffer_layers",
    "_buffer_accessor",
    "training",
    "forward_pre_hooks",
    "forward_hooks",
    "backward_pre_hooks",
    "backward_hooks",
    "full_backward_pre_hooks",
    "full_backward_hooks",
    "custom_attributes",
    "custom_methods",
)

#: Row positions of the ``ModuleCall`` cells the helpers below read in a detached row.
_IN_LAYERS, _OUT_LAYERS, _IN_OPS, _OUT_OPS, _ARGS, _KWARGS, _ARGS_SUMMARY, _KWARGS_SUMMARY = (
    _MODULE_CALL_STORED_FIELDS.index(name)
    for name in (
        "input_layers",
        "output_layers",
        "input_ops",
        "output_ops",
        "forward_args",
        "forward_kwargs",
        "forward_args_summary",
        "forward_kwargs_summary",
    )
)


def install_module_facades(module_call_cls: type, module_cls: type) -> None:
    """Install the ``ModuleCall`` and ``Module`` row-cell descriptors.

    ``Module.ops`` and ``Module.params`` list child records, so their cells
    take the row-key descriptors of ``_trace_core.record_refs``.
    """

    install_record_facade(module_call_cls, _MODULE_CALL_STORED_FIELDS)
    install_record_facade(module_cls, _MODULE_STORED_FIELDS)
    install_child_fields(module_cls, {"ops": "module_call", "params": "param"})


def call_boundaries(calls: Any) -> tuple[list[str], list[str], list[str], list[str]]:
    """Return a module's input ops, input layers, output ops and output layers.

    Each list is the union over ``calls`` in first-seen call order. A call that
    is still detached (the module build) is read from its own row.
    """

    input_ops: list[str] = []
    input_layers: list[str] = []
    output_ops: list[str] = []
    output_layers: list[str] = []
    for call in calls.values():
        state = call.__dict__
        if CORE_KEY in state and state[CORE_KEY].__class__ is DetachedOpStore:
            row = state[CORE_KEY]._cells
            sides = (row[_IN_OPS], row[_IN_LAYERS], row[_OUT_OPS], row[_OUT_LAYERS])
        else:
            sides = (call.input_ops, call.input_layers, call.output_ops, call.output_layers)
        for labels, union in zip(sides, (input_ops, input_layers, output_ops, output_layers)):
            for label in labels:
                if label not in union:
                    union += (label,)
    return input_ops, input_layers, output_ops, output_layers


def summarize_call_args(calls: Any, keep_args: bool) -> None:
    """Write each call's argument summaries, then drop its arguments unless kept.

    The arguments can hold large tensors from the model's ``forward`` call. A
    call that is still detached (the module build) is written in its own row.
    """

    for call in calls:
        state = call.__dict__
        if CORE_KEY in state and state[CORE_KEY].__class__ is DetachedOpStore:
            row = state[CORE_KEY]._cells
            row[_ARGS_SUMMARY] = format_call_arg(row[_ARGS])
            row[_KWARGS_SUMMARY] = format_call_arg(row[_KWARGS])
            if not keep_args:
                row[_ARGS] = row[_KWARGS] = None
            continue
        call.forward_args_summary = format_call_arg(call.forward_args)
        call.forward_kwargs_summary = format_call_arg(call.forward_kwargs)
        if not keep_args:
            call.forward_args = None
            call.forward_kwargs = None


def module_call_row(given: dict[str, Any]) -> list[Any]:
    """Return a ``ModuleCall`` row from ``ModuleCall.__init__``'s arguments by name.

    The row has one value per field of ``_MODULE_CALL_STORED_FIELDS``, in that order.
    """

    address = given["address"]
    all_addresses = given["all_addresses"]
    input_layers = given["input_layers"]
    output_layers = given["output_layers"]
    output_ops = given["output_ops"]
    output_paths = given["output_paths"]
    forward_args = given["forward_args"]
    forward_kwargs = given["forward_kwargs"]
    effects = given["forward_pre_hook_effects"]
    arg_names = given["forward_arg_names"]
    code_context = given["code_context"]
    call_stack = given["module_call_stack"]
    call_children = given["call_children"]
    num_pos = len(forward_args) if forward_args is not None else 0
    num_kwargs = len(forward_kwargs) if forward_kwargs is not None else 0
    return [
        address,
        all_addresses if all_addresses is not None else [address],
        given["cls"],
        given["class_name"],
        given["class_qualname"],
        given["call_index"],
        given["label"],  # e.g. "features.0:1"
        given["ordinal_index"],
        given["ops"],  # pass-qualified Op labels
        input_layers,
        output_layers,
        list(input_layers),
        output_ops if output_ops is not None else list(output_layers),
        given["output_structure"],
        output_paths if output_paths is not None else (),
        forward_args,
        forward_kwargs,
        given["inputs_before_pre_hooks"],
        given["inputs_after_pre_hooks"],
        effects if effects is not None else (),
        arg_names if arg_names is not None else [],
        num_pos,
        num_kwargs,
        num_pos + num_kwargs,
        "",  # forward_args_summary
        "",  # forward_kwargs_summary
        Duration(given["forward_duration"]),
        code_context if code_context is not None else [],
        call_stack if call_stack is not None else [],
        given["call_parent"],
        call_children if call_children is not None else [],
    ]


def module_row(given: dict[str, Any], calls: Any) -> list[Any]:
    """Return a ``Module`` row from ``Module.__init__``'s arguments by name.

    ``params`` is already defaulted and ``calls`` is the accessor stored as the
    ``ops`` cell. The row has one value per field of ``_MODULE_STORED_FIELDS``,
    in that order.
    """

    address = given["address"]
    all_addresses = given["all_addresses"]
    input_ops, input_layers, output_ops, output_layers = call_boundaries(calls)
    return [
        address,
        all_addresses if all_addresses is not None else [address],
        given["cls"],
        given["class_name"],
        given["class_qualname"],
        given["class_source_file"],
        given["class_source_line"],
        given["init_source_file"],
        given["init_source_line"],
        given["forward_source_file"],
        given["forward_source_line"],
        given["class_docstring"],
        given["init_signature"],
        given["init_docstring"],
        given["forward_signature"],
        given["forward_docstring"],
        given["address_parent"],
        value if (value := given["address_children"]) is not None else [],
        given["address_depth"],
        given["call_parent"],
        value if (value := given["call_children"]) is not None else [],
        given["call_depth"],
        given["num_calls"],
        calls,
        value if (value := given["call_labels"]) is not None else [],
        # layer_labels holds NO-PASS labels (e.g. "conv2d_1_1") -> Layer;
        # ModuleCall.ops holds pass-qualified labels.
        value if (value := given["layer_labels"]) is not None else [],
        input_ops,
        input_layers,
        output_ops,
        output_layers,
        given["params"],
        given["num_params"],
        given["num_params_trainable"],
        given["num_params_frozen"],
        value if (value := given["buffer_layers"]) is not None else [],
        None,  # _buffer_accessor, populated by _build_module_logs
        given["training"],
        value if (value := given["forward_pre_hooks"]) is not None else [],
        value if (value := given["forward_hooks"]) is not None else [],
        value if (value := given["backward_pre_hooks"]) is not None else [],
        value if (value := given["backward_hooks"]) is not None else [],
        value if (value := given["full_backward_pre_hooks"]) is not None else [],
        value if (value := given["full_backward_hooks"]) is not None else [],
        value if (value := given["custom_attributes"]) is not None else {},
        value if (value := given["custom_methods"]) is not None else [],
    ]
