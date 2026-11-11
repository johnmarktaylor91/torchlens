"""Exhaustive freeze ref builders, outside the package's largest import cycle.

The exhaustive commit (``_ops_row_journal.commit_row_record``), the row cell
writer (``_ops_row_cells.op_row_cells``) and the eager freeze
(``_ops_container_base._op_record_from_log``) build the same nested refs from
one op's raw fields: the tensor and output refs, the module frames, the
parameter refs, the shared function-call ref, the capture policy and the copied
parent-argument positions. They live here, importing only IR value types and
leaf helpers, so the row-direct modules can use them without importing a module
of the cycle (``ops`` calls those modules, so such an import would pull them
into the cycle).
"""

from __future__ import annotations

import copy
import dataclasses
from typing import TYPE_CHECKING, Any

import torch

from ..._io import BlobRef
from ..._ir.events import FunctionCallRef, ModuleFrame, OutputRef
from ..._ir.refs import ParamRef, TensorRef
from ..._ir.semantics import BackendSemantics, CapturePolicy
from ..._utils._internal_read import internal_scalar_read
from ..._utils._tensor_label import get_param_meta

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace

__all__ = (
    "_FUNCTION_REF_PER_OUTPUT_FIELDS",
    "_tensor_ref_from_fields",
    "_module_frames_from_fields",
    "_param_refs_from_fields",
    "_function_call_ref_from_fields",
    "_resolve_call_function_ref",
    "_exhaustive_freeze_refs",
    "_exhaustive_output_ref",
    "_exhaustive_capture_policy",
    "_is_plain_position_key",
    "_copy_parent_arg_positions",
)


#: FunctionCallRef fields the per-output logging path genuinely rewrites
#: (``_log_output_tensor_info``): FLOPs derive from the output shape and
#: ``is_inplace`` from the output tensor's version. Everything else on the
#: ref is a call-level fact, so sibling outputs of one wrapped call share
#: ONE frozen ref; a sibling whose per-output facts differ gets a
#: ``dataclasses.replace`` derivative that still shares every container
#: field by reference.
_FUNCTION_REF_PER_OUTPUT_FIELDS = ("flops_forward", "flops_backward", "is_inplace")


def _tensor_ref_from_fields(tensor: torch.Tensor, fields_dict: dict[str, Any]) -> TensorRef:
    """Build an IR tensor reference from one raw fields dictionary.

    Parameters
    ----------
    tensor
        Tensor represented by the operation event.
    fields_dict
        Raw field mapping used to construct the corresponding ``Op``.

    Returns
    -------
    TensorRef
        Backend-neutral tensor metadata and optional payload reference.
    """

    # TorchLens's OWN IR-bookkeeping ``requires_grad`` read runs under the explicit
    # internal-read marker so the state-metadata property observer never mistakes it
    # for a user autograd read on a registered buffer/param receiver.
    with internal_scalar_read():
        _requires_grad = bool(tensor.requires_grad)
    return _tensor_ref_from_facts(fields_dict, _requires_grad, str(tensor.device), str(id(tensor)))


def _tensor_ref_from_facts(
    fields_dict: dict[str, Any], requires_grad: bool, device: str, handle_id: str
) -> TensorRef:
    """Build the output tensor ref from the three live-tensor facts read at commit.

    ``_tensor_ref_from_fields`` reads ``requires_grad``, the device and the
    tensor's ``id`` and calls this; the deferred row commit reads the same three
    at commit and calls it later, so both build the same ref.
    """

    return TensorRef(
        label_raw=fields_dict["_label_raw"],
        shape=fields_dict["shape"],
        dtype=str(fields_dict["dtype"]),
        device=device,
        requires_grad=requires_grad,
        memory=fields_dict["activation_memory"],
        payload=fields_dict["out"],
        blob_ref=(
            BlobRef(blob_id=fields_dict["_pending_blob_id"], kind="out")  # type: ignore[arg-type]
            if fields_dict.get("_pending_blob_id") is not None
            else None
        ),
        backend_handle_id=handle_id,
    )


def _module_frames_from_fields(fields_dict: dict[str, Any]) -> tuple[ModuleFrame, ...]:
    """Convert raw module stack tuples to IR module frames.

    Parameters
    ----------
    fields_dict
        Raw field mapping used to construct the corresponding ``Op``.

    Returns
    -------
    tuple[ModuleFrame, ...]
        Module stack snapshot for the operation event.
    """

    return tuple(
        [
            ModuleFrame(
                address=address,
                address_normalized=None,
                module_type="",
                call_index=pass_index,
                fx_qualpath=None,
                entry_argnames=(),
            )
            for address, pass_index in fields_dict["modules"]
        ]
    )


def _param_refs_from_fields(fields_dict: dict[str, Any]) -> tuple[ParamRef, ...]:
    """Convert raw parameter metadata to IR parameter references.

    Parameters
    ----------
    fields_dict
        Raw field mapping used to construct the corresponding ``Op``.

    Returns
    -------
    tuple[ParamRef, ...]
        Parameter references for the operation event.
    """

    refs: list[ParamRef] = []
    # One ref per RAW parent-param occurrence, each carrying ITS OWN barcode
    # from the param's meta. ``_param_barcodes`` is the DEDUPED key list of
    # ``parent_param_ops`` (a dict), so zipping it against the raw
    # ``parent_params`` misaligned every pairing after a repeated param
    # (weight-tied einsum/hypernetwork class): ``[A, B, A, C]`` persisted
    # ``ParamRef(barcode=C, geometry-of-A)`` and dropped a ref while
    # ``num_params`` said four.
    for param in fields_dict["parent_params"]:
        param_meta = get_param_meta(param)
        if param_meta is None or param_meta.param_barcode is None:
            raise RuntimeError(
                "TorchLens internal error: a parent parameter reached ParamRef "
                "construction without an assigned barcode; param metadata was "
                "not processed for this op event."
            )
        barcode = param_meta.param_barcode
        param_address = (
            ""
            if param_meta is None or param_meta.param_address is None
            else param_meta.param_address
        )
        # TorchLens's OWN per-op parent-param ``requires_grad`` read runs under the
        # explicit internal-read marker so the state-metadata property observer never
        # mistakes it for a user autograd read on the registered parameter (it fires for
        # EVERY op with parent params -- unmarked it would spuriously fact-record every
        # consumed slot of every model).
        with internal_scalar_read():
            trainable = bool(param.requires_grad)
        refs.append(
            ParamRef(
                barcode=barcode,
                address=param_address,
                shape=tuple(param.shape),
                dtype=str(param.dtype),
                trainable=trainable,
                module_address=None,
            )
        )
    return tuple(refs)


def _function_call_ref_from_fields(fields_dict: dict[str, Any]) -> FunctionCallRef:
    """Build the frozen function-call summary for one operation event."""

    return FunctionCallRef(
        func=fields_dict["func"],
        func_name=fields_dict["func_name"],
        func_qualname=fields_dict["func_qualname"],
        func_call_id=fields_dict["func_call_id"],
        code_context=tuple(fields_dict["code_context"]),
        func_duration=fields_dict["func_duration"],
        flops_forward=fields_dict["flops_forward"],
        flops_backward=fields_dict["flops_backward"],
        func_rng_states=fields_dict["func_rng_states"],
        func_autocast_state=fields_dict["func_autocast_state"],
        arg_names=tuple(fields_dict["arg_names"]),
        num_args_total=fields_dict["num_args_total"],
        num_pos_args=fields_dict["num_pos_args"],
        num_kwargs=fields_dict["num_kwargs"],
        non_tensor_pos_args=tuple(fields_dict["non_tensor_pos_args"]),
        non_tensor_kwargs=tuple(fields_dict["non_tensor_kwargs"].items()),
        func_non_tensor_args=tuple(fields_dict["func_non_tensor_args"]),
        is_inplace=fields_dict["is_inplace"],
        func_config=tuple(fields_dict["func_config"].items()),
        func_id=fields_dict.get("func_id"),
    )


def _resolve_call_function_ref(
    fields_dict: dict[str, Any],
    call_ref_box: list[FunctionCallRef] | None,
) -> FunctionCallRef:
    """Return the (shared) function-call ref for one output event.

    The first output of a wrapped call builds the ref and parks it in
    ``call_ref_box``; sibling outputs reuse it verbatim when their per-output
    facts match, otherwise derive via ``dataclasses.replace`` (container
    fields stay shared by reference either way).
    """

    if call_ref_box:
        shared = call_ref_box[0]
        overrides = {
            name: fields_dict[name]
            for name in _FUNCTION_REF_PER_OUTPUT_FIELDS
            if getattr(shared, name) != fields_dict[name]
        }
        return dataclasses.replace(shared, **overrides) if overrides else shared
    ref = _function_call_ref_from_fields(fields_dict)
    if call_ref_box is not None:
        call_ref_box.append(ref)
    return ref


def _exhaustive_freeze_refs(
    fields_dict: dict[str, Any],
    tensor: torch.Tensor,
    module_stack: tuple[ModuleFrame, ...] | None,
) -> tuple[Any, tuple[ModuleFrame, ...], Any, BackendSemantics]:
    """Compute the exhaustive freeze's shared nested refs ONCE per record.

    Both freeze shapes (legacy ``OpEvent`` and decomposed ``OpRecord``) build
    from these exact values, so the shapes cannot drift on the ref layer.
    """

    tensor_ref = _tensor_ref_from_fields(tensor, fields_dict)
    return (tensor_ref, *_exhaustive_side_refs(fields_dict, module_stack))


def _exhaustive_side_refs(
    fields_dict: dict[str, Any], module_stack: tuple[ModuleFrame, ...] | None
) -> tuple[tuple[ModuleFrame, ...], Any, BackendSemantics]:
    """Return the module frames, transformed-output ref and backend semantics.

    The refs of ``_exhaustive_freeze_refs`` that read no live tensor, so the
    deferred row commit builds them from its field snapshot with the same code.
    """

    if module_stack is None:
        module_stack = _module_frames_from_fields(fields_dict)
    transformed_ref = (
        None
        if fields_dict["transformed_out"] is None
        else TensorRef(
            label_raw=fields_dict["_label_raw"],
            shape=fields_dict["transformed_out_shape"],
            dtype=str(fields_dict["transformed_out_dtype"]),
            device=fields_dict["output_device"],
            requires_grad=None,
            memory=fields_dict["transformed_activation_memory"],
            payload=fields_dict["transformed_out"],
            blob_ref=(
                BlobRef(
                    blob_id=fields_dict["_pending_transformed_out_blob_id"],
                    kind="transformed_out",
                )  # type: ignore[arg-type]
                if fields_dict.get("_pending_transformed_out_blob_id") is not None
                else None
            ),
            backend_handle_id=None,
        )
    )
    backend_semantics = fields_dict.get("backend_semantics")
    if backend_semantics is None:
        backend_semantics = BackendSemantics(
            backend_grad_handle=fields_dict["grad_fn_handle"],
            grad_fn_class_name=fields_dict["grad_fn_class_name"],
            autograd_memory=fields_dict["autograd_memory"],
            num_autograd_tensors=fields_dict["num_autograd_tensors"],
            mutated_input_positions=(),
            aliased_output_inputs=(),
            unknown_aliasing=False,
            bytes_delta_at_call=fields_dict["bytes_delta_at_call"],
            bytes_peak_at_call=fields_dict["bytes_peak_at_call"],
        )
    return module_stack, transformed_ref, backend_semantics


def _exhaustive_output_ref(
    fields_dict: dict[str, Any], tensor_ref: Any, transformed_ref: Any
) -> OutputRef:
    """Return the exhaustive freeze's output ref (shared by both shapes)."""

    return OutputRef(
        tensor=tensor_ref,
        transformed_tensor=transformed_ref,
        has_saved_activation=fields_dict["has_saved_activation"],
        output_device=fields_dict["output_device"],
        activation_transform=fields_dict["activation_transform"],
        detach_saved_activations=fields_dict["detach_saved_activations"],
        visualizer_path=fields_dict["visualizer_path"],
        multi_output_index=fields_dict["multi_output_index"],
        in_multi_output=fields_dict["in_multi_output"],
        container_path=tuple(fields_dict["container_path"]),
        container_spec=fields_dict["container_spec"],
        child_versions=tuple(fields_dict["out_versions_by_child"].items()),
    )


def _exhaustive_capture_policy(trace: Trace, fields_dict: dict[str, Any]) -> CapturePolicy:
    """Return the exhaustive freeze's capture policy (shared by both shapes)."""

    return CapturePolicy(
        save_payload=fields_dict["has_saved_activation"],
        save_grad=fields_dict["save_grads"],
        save_mode=trace._options_save_mode,
    )


def _is_plain_position_key(key: Any) -> bool:
    """Return whether an argument-position key is an int, a str, or a tuple of them."""

    kind = type(key)
    if kind is int or kind is str:
        return True
    if kind is not tuple:
        return False
    return all(type(part) in (int, str) for part in key)


def _copy_parent_arg_positions(positions: Any) -> Any:
    """Copy an op's ``{"args": {...}, "kwargs": {...}}`` parent label map.

    The usual map holds int, str or tuple keys and str labels, all immutable, so
    copying the inner dicts gives what ``copy.deepcopy`` gave, without its
    per-object dispatch and memo. Any other shape still goes through ``deepcopy``.
    """

    if type(positions) is not dict:
        return copy.deepcopy(positions)
    copied = {}
    for kind, inner in positions.items():
        if type(inner) is not dict:
            return copy.deepcopy(positions)
        for key, label in inner.items():
            if type(label) is not str or (
                type(key) is not int and type(key) is not str and not _is_plain_position_key(key)
            ):
                return copy.deepcopy(positions)
        copied[kind] = dict(inner)
    return copied
