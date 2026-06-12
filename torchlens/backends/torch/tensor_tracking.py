"""Functions for tracking tensor lineage, family relationships, and operation equivalence.

Handles backward hooks for grad capture, parent-child-sibling-spouse linkage,
parameter pass tracking, and structural fingerprinting of operations for loop detection.

Key concepts:

**Family links** (parent/child/sibling/spouse):
    When a new tensor is created by a function, its input tensors become parents,
    co-parents become spouses, and children of the same parent become siblings.
    All links are bidirectional and updated immediately at creation time.

**Operation equivalence type** (``_get_equivalence_class``):
    A structural fingerprint string that identifies operations as "the same layer"
    across loop iterations.  Used by loop detection to group operations into
    equivalence classes.  For parameterized ops, the fingerprint is based on the
    parameter barcodes + op type (e.g. ``"conv2d_abc123_def456"``).  For
    non-parameterized ops, it hashes non-tensor args, output index, and
    containing module.

**Backward hooks** (``_add_tensor_backward_hook``):
    Uses ``weakref.ref(Trace)`` to avoid preventing garbage collection of the
    Trace after the user is done with it.  The hook closure captures the weakref
    and the raw tensor label (a string, not the tensor itself).

**Parent arg position tracking** (``_locate_parent_tensors_in_args``):
    Records where each parent tensor appeared in the function's args/kwargs,
    supporting up to 2 levels of nesting (e.g., ``args[0]`` or ``args[1][2]``).
    Deeper nesting is not tracked.
"""

import itertools as it
import time
import warnings
import weakref
from typing import TYPE_CHECKING, Any

import torch

from ._tl import get_param_meta, get_tensor_label, increment_param_call_index, set_param_meta
from ...ir.events import BackwardPassStart, OpGradObserved
from ...data_classes.op import Op
from ..._state import pause_logging
from ...intervention.selectors import BaseSelector
from ...utils.hashing import make_random_barcode, make_short_barcode_from_input
from ...utils.tensor_utils import safe_copy
from ...fastlog.types import CaptureSpec

if TYPE_CHECKING:
    from ...data_classes.trace import Trace


def _add_tensor_backward_hook(trace: "Trace", t: torch.Tensor, tensor_label: str) -> None:
    """Register a backward hook on ``t`` that captures its grad into Trace.

    The hook closure captures a ``weakref`` to Trace (not a strong reference)
    so that the hook doesn't prevent GC of the Trace after the user drops it
    (GC-8).  The closure also captures ``tensor_label`` (a string) rather than
    the tensor itself, avoiding circular references.

    Only tensors that participate in autograd (have grad_fn_handle or require_grad)
    get hooks — others would never receive grads.

    Args:
        t: The tensor to hook.
        tensor_label: Raw tensor label (e.g. ``"conv2d_3_47_raw"``) used to
            look up the corresponding log entry when the grad arrives.
    """
    hooked_tensors = trace.__dict__.setdefault("_tl_backward_hooked_tensor_keys", set())
    hook_key = (tensor_label, id(t))
    if hook_key in hooked_tensors:
        return
    hooked_tensors.add(hook_key)

    # Weak reference prevents Trace -> tensor -> hook -> Trace ref cycle.
    trace_ref = weakref.ref(trace)

    def log_grad_to_model_history(grad: torch.Tensor) -> None:
        active_trace = trace_ref()
        if active_trace is not None:
            _emit_tensor_grad_event(active_trace, grad, tensor_label)
            if getattr(active_trace, "save_grads", None) not in (None, False):
                _log_tensor_grad(active_trace, grad, tensor_label)

    if t.grad_fn is not None:
        from .backward import _register_forward_grad_fn

        _register_forward_grad_fn(trace, t.grad_fn, tensor_label)
    if (t.grad_fn is not None) or t.requires_grad:
        t.register_hook(log_grad_to_model_history)  # type: ignore[no-untyped-call]


def _ensure_backward_event_stream(trace: "Trace") -> Any:
    """Return the mutable capture event bundle for backward sidecar emission."""

    events = getattr(trace, "_capture_events", None)
    if events is not None:
        return events
    events = getattr(trace, "capture_events", None)
    if events is not None:
        return events
    from ...ir import CaptureEvents

    events = CaptureEvents()
    trace._capture_events = events
    return events


def _ensure_backward_pass_for_tensor_hook(trace: "Trace") -> int:
    """Return an active backward pass index, opening an implicit pass if needed."""

    pass_index = getattr(trace, "_active_backward_pass_index", None)
    if pass_index is not None:
        return int(pass_index)
    pass_index = int(getattr(trace, "num_backward_passes", 0)) + 1
    trace._active_backward_pass_index = pass_index
    trace._implicit_backward_pass_open = True
    if not getattr(trace, "_warned_implicit_backward_pass", False):
        warnings.warn(
            "TorchLens observed gradients outside a managed backward trigger; recording an "
            "implicit backward pass. Use trace.log_backward(), trace.backward(), or a TorchLens "
            "autograd trigger for precise pass boundaries.",
            RuntimeWarning,
            stacklevel=3,
        )
        trace._warned_implicit_backward_pass = True
    events = _ensure_backward_event_stream(trace)
    events.append_backward(
        BackwardPassStart(
            pass_index=pass_index,
            trigger="implicit",
            implicit=True,
            outer_context=None,
            call_context_ref=None,
            root_meta=(),
            root_grad_arguments=None,
            inputs_subset=(),
            order=None,
            origin_backward_pass=None,
            save_grads_policy_repr=repr(_active_save_grads_policy(trace)),
            engine_flags=None,
            timestamp=time.time(),
        )
    )
    return pass_index


def _emit_tensor_grad_event(trace: "Trace", grad: torch.Tensor, tensor_label: str) -> None:
    """Append an ``OpGradObserved`` event for a tensor hook firing."""

    if getattr(trace, "_tl_backward_triggers_disarmed", False):
        return
    events = _ensure_backward_event_stream(trace)
    pass_index = _ensure_backward_pass_for_tensor_hook(trace)
    final_label = getattr(trace, "_raw_to_final_layer_labels", {}).get(tensor_label, tensor_label)
    with pause_logging():
        memory = int(grad.nelement() * grad.element_size())
        payload, transformed_payload = _build_grad_payloads(trace, grad, final_label)
    events.append_backward(
        OpGradObserved(
            op_label=final_label,
            pass_index=pass_index,
            payload_ref=payload,
            transformed_payload_ref=transformed_payload,
            shape=tuple(grad.shape),
            dtype=str(grad.dtype),
            memory=memory,
            timestamp=time.time(),
            seq=events.next_backward_seq(),
        )
    )


def _build_grad_payloads(
    trace: "Trace", grad: torch.Tensor, layer_label: str
) -> tuple[torch.Tensor | None, Any | None]:
    """Return raw and transformed payloads for one observed op gradient.

    Parameters
    ----------
    trace:
        Trace owning the observed gradient.
    grad:
        Raw gradient tensor emitted by autograd.
    layer_label:
        Final operation label for the hook firing.

    Returns
    -------
    tuple[torch.Tensor | None, Any | None]
        Raw payload and transformed payload retained for this event.
    """

    if not _should_save_grad_payload(trace, layer_label):
        return None, None
    if layer_label not in getattr(trace, "layer_dict_all_keys", {}):
        return _build_fastlog_grad_payloads(trace, grad)
    op = trace[layer_label]
    grad_transform = getattr(trace, "grad_transform", None)
    save_raw_gradients = getattr(trace, "save_raw_gradients", True)
    raw_payload = _copy_grad_payload(grad) if save_raw_gradients or grad_transform is None else None
    if grad_transform is None:
        return raw_payload, None
    writer = getattr(trace, "_out_writer", None)
    transformed_payload = op._apply_transform(
        grad,
        grad_transform,
        transform_kind="grad",
        streaming_active=writer is not None,
    )
    op._validate_train_mode_transform_output(
        grad,
        transformed_payload,
        transform_kind="grad",
    )
    op._validate_streaming_transform_output(
        transformed_payload,
        transform_kind="grad",
        streaming_active=writer is not None,
    )
    return raw_payload, transformed_payload


def _should_save_grad_payload(trace: "Trace", layer_label: str) -> bool:
    """Return whether a tensor-hook gradient should retain its payload."""

    policy = _active_save_grads_policy(trace)
    if policy in [None, False, "none", []]:
        return False
    if policy is True or policy == "all":
        return True
    if layer_label not in getattr(trace, "layer_dict_all_keys", {}):
        ctx = getattr(trace, "_fastlog_grad_contexts", {}).get(layer_label)
        if ctx is None:
            return False
        if callable(policy) or isinstance(policy, BaseSelector):
            decision = policy(
                _FastlogGradPayloadContext(ctx=ctx, pass_index=_current_backward_pass(trace))
            )
            return _grad_payload_decision_saves_out(decision)
        return False
    op = trace[layer_label]
    if isinstance(policy, BaseSelector):
        decision = policy(_GradPayloadContext(op=op, pass_index=_current_backward_pass(trace)))
        return _grad_payload_decision_saves_out(decision)
    if callable(policy):
        decision = policy(_GradPayloadContext(op=op, pass_index=_current_backward_pass(trace)))
        return _grad_payload_decision_saves_out(decision)
    selection = getattr(trace, "_grad_op_nums_to_save", "all")
    if selection in [None, "none", []]:
        return False
    if selection == "all":
        return True
    return op.raw_index in selection


def _build_fastlog_grad_payloads(
    trace: "Trace",
    grad: torch.Tensor,
) -> tuple[torch.Tensor | None, Any | None]:
    """Return gradient payloads for predicate-mode recording contexts."""

    grad_transform = getattr(trace, "grad_transform", None)
    save_raw_gradients = getattr(trace, "save_raw_gradients", True)
    raw_payload = _copy_grad_payload(grad) if save_raw_gradients or grad_transform is None else None
    if grad_transform is None:
        return raw_payload, None
    transformed_payload = grad_transform(grad)
    if not isinstance(transformed_payload, torch.Tensor):
        raise TypeError("grad_transform must return a torch.Tensor for fastlog gradients")
    return raw_payload, transformed_payload


def _copy_grad_payload(grad: torch.Tensor) -> torch.Tensor:
    """Return a detached gradient payload snapshot through the copy chokepoint.

    Parameters
    ----------
    grad:
        Gradient tensor observed by an autograd hook.

    Returns
    -------
    torch.Tensor
        Detached tensor copy suitable for storage in gradient records.
    """

    copied = safe_copy(grad, detach_tensor=True)
    if not isinstance(copied, torch.Tensor):
        raise TypeError("safe_copy returned a non-tensor gradient payload")
    return copied


def _grad_payload_decision_saves_out(decision: Any) -> bool:
    """Return whether a gradient predicate decision retains tensor payload."""

    if isinstance(decision, CaptureSpec):
        return decision.save_out
    return bool(decision)


class _GradPayloadContext:
    """Minimal predicate context for trace-side op gradient retention."""

    def __init__(self, *, op: Op, pass_index: int | None) -> None:
        """Initialize a trace gradient predicate context.

        Parameters
        ----------
        op:
            Operation whose output gradient was observed.
        pass_index:
            One-based backward pass number, when known.
        """

        self.label = op.label
        self.layer_label = op.layer_label
        self.op_label = op.label
        self.raw_label = getattr(op, "raw_tensor_label", None)
        self.func_name = op.func_name
        self.layer_type = op.layer_type
        self.type = op.layer_type
        self.module_stack = tuple(getattr(op, "module_stack", ()) or ())
        self.modules = tuple(getattr(op, "modules", ()) or ())
        self.output_of_module_calls = tuple(getattr(op, "output_of_module_calls", ()) or ())
        self.has_forward_op = True
        self.has_op = True
        self.grad_kind = "grad_output"
        self.pass_index = pass_index
        self.backward_pass_index = pass_index
        self.shape = op.shape
        self.dtype = op.dtype
        self.tensor_device = getattr(op, "output_device", None)


class _FastlogGradPayloadContext:
    """Minimal predicate context for fastlog op gradient retention."""

    def __init__(self, *, ctx: Any, pass_index: int | None) -> None:
        """Initialize a fastlog gradient predicate context."""

        self.label = ctx.label
        self.layer_label = _public_fastlog_label(ctx)
        self.op_label = self.layer_label
        self.raw_label = ctx.raw_label
        self.func_name = ctx.func_name
        self.layer_type = ctx.layer_type
        self.type = ctx.layer_type
        self.module_stack = tuple(getattr(ctx, "module_stack", ()) or ())
        self.modules = tuple(frame.address for frame in self.module_stack)
        self.output_of_module_calls = self.modules
        self.has_forward_op = True
        self.has_op = True
        self.grad_kind = "grad_output"
        self.pass_index = pass_index
        self.backward_pass_index = pass_index
        self.shape = ctx.shape
        self.dtype = ctx.dtype
        self.tensor_device = ctx.tensor_device


def _public_fastlog_label(ctx: Any) -> str:
    """Return the compact fastlog op label for a record context."""

    if ctx.kind == "op" and ctx.layer_type is not None and ctx.type_index is not None:
        return f"{ctx.layer_type}_{ctx.type_index}"
    return ctx.label


def _current_backward_pass(trace: "Trace") -> int | None:
    """Return the currently active backward pass index, if any."""

    pass_index = getattr(trace, "_active_backward_pass_index", None)
    return None if pass_index is None else int(pass_index)


def _active_save_grads_policy(trace: "Trace") -> Any:
    """Return the current gradient retention policy for tensor hooks."""

    if hasattr(trace, "_active_save_grads_policy"):
        return getattr(trace, "_active_save_grads_policy")
    return getattr(trace, "save_grads", None)


def _log_tensor_grad(self: "Trace", grad: torch.Tensor, _label_raw: str) -> None:
    """Callback invoked during backward pass to save a tensor's grad.

    Resolves the raw label to a final label, then saves the grad on the
    layer entry.  If the layer is an output parent, its output-layer children
    also receive the grad (since output layers are identity wrappers that
    share the same grad).

    Args:
        grad: The grad tensor from autograd.
        _label_raw: Raw tensor label used to look up the final label.
    """
    self.has_gradients = True
    if _label_raw not in self._raw_to_final_layer_labels:
        return
    tensor_label = self._raw_to_final_layer_labels[_label_raw]
    layer_log_entry = self[tensor_label]
    layers_to_update = [tensor_label]
    # Output layers are identity wrappers; propagate grad to them too.
    if layer_log_entry.is_output_parent:
        for child_layer in layer_log_entry.children:
            if self[child_layer].is_output:
                layers_to_update.append(child_layer)

    for layer_label in layers_to_update:
        layer = self[layer_label]
        selection = getattr(self, "_grad_op_nums_to_save", "all")
        if selection != "all":
            if selection in [None, "none", []] or layer.raw_index not in selection:
                continue
        if layer_label not in self._saved_grad_labels:
            self._saved_grad_labels.add(layer_label)
        layer.log_tensor_grad(grad)
        self.saved_gradient_memory += layer.gradient_memory
        self.total_gradient_memory += layer.gradient_memory


def _locate_parent_tensors_in_args(
    self: "Trace",
    parent_log_entries: list[Op],
    args: tuple[Any, ...],
    kwargs: dict[Any, Any],
) -> dict[str, dict[Any, str]]:
    """Map each parent tensor to its position in the function's args/kwargs.

    Supports up to 2 levels of nesting:
      - Top-level: ``args[i]`` maps to key ``i``
      - Nested: ``args[i][j]`` maps to key ``(i, j)``
    Deeper nesting is not tracked (would require recursive search).

    This mapping is stored as ``parent_arg_positions`` on the child's log
    entry, and is used by:
      - ``_get_parent_contents``: to retrieve pre-call parent values from arg_copies
      - Validation replay: to reconstruct the function call

    Returns:
        ``{"args": {pos: label, ...}, "kwargs": {key: label, ...}}``
    """
    tensor_all_arg_positions: dict[str, dict[Any, str]] = {"args": {}, "kwargs": {}}
    arg_struct_dict = {"args": args, "kwargs": kwargs}

    for parent_entry in parent_log_entries:
        for arg_type in ["args", "kwargs"]:
            arg_struct = arg_struct_dict[arg_type]
            _find_arg_positions_for_single_parent(
                parent_entry,
                arg_type,
                arg_struct,  # type: ignore[arg-type]
                tensor_all_arg_positions,
            )

    return tensor_all_arg_positions


def _find_arg_positions_for_single_parent(
    parent_entry: Op,
    arg_type: str,
    arg_struct: list[Any] | tuple[Any, ...] | dict[Any, Any],
    tensor_all_arg_positions: dict[str, dict[Any, str]],
) -> None:
    """Locate a single parent tensor within args or kwargs (up to 2 nesting levels).

    Scans the top-level args/kwargs and one level of sub-containers (lists,
    tuples, dicts).  For top-level matches, the key is a scalar (int index or
    kwarg name).  For nested matches, the key is a tuple ``(outer_key, inner_key)``.

    Args:
        parent_entry: The parent tensor's log entry.
        arg_type: ``"args"`` or ``"kwargs"``.
        arg_struct: The actual args tuple or kwargs dict.
        tensor_all_arg_positions: Accumulator dict; mutated in place.
    """
    # Polymorphic iteration: enumerate for positional, .items() for keyword.
    iteration_strategies = {
        "args": enumerate,
        "kwargs": lambda x: x.items(),
        list: enumerate,
        tuple: enumerate,
        dict: lambda x: x.items(),
    }
    iterfunc = iteration_strategies[arg_type]

    for arg_key, arg in iterfunc(arg_struct):  # type: ignore[operator]
        if (
            not isinstance(arg, torch.nn.Parameter)
            and get_tensor_label(arg) == parent_entry._label_raw
        ):
            tensor_all_arg_positions[arg_type][arg_key] = parent_entry._label_raw
        elif type(arg) in [list, tuple, dict]:
            # Second level of nesting (e.g., torch.cat([tensor_a, tensor_b])).
            iterfunc2 = iteration_strategies[type(arg)]
            for sub_arg_key, sub_arg in iterfunc2(arg):  # type: ignore[operator]
                if (
                    not isinstance(sub_arg, torch.nn.Parameter)
                    and get_tensor_label(sub_arg) == parent_entry._label_raw
                ):
                    tensor_all_arg_positions[arg_type][(arg_key, sub_arg_key)] = (
                        parent_entry._label_raw
                    )


def _get_ancestors_from_parents(
    parent_entries: list[Op],
) -> tuple[set[str], set[str]]:
    """Utility function to get the ancestors of a tensor based on those of its parent tensors.

    Args:
        parent_entries: list of parent entries

    Returns:
        List of input ancestors and internally initialized ancestors.
    """
    input_ancestors = set()
    internal_source_ancestors = set()

    for parent_entry in parent_entries:
        input_ancestors.update(parent_entry.input_ancestors)
        internal_source_ancestors.update(parent_entry.internal_source_ancestors)
    return input_ancestors, internal_source_ancestors


def _process_parent_param_ops(
    arg_parameters: list[torch.nn.Parameter],
) -> dict[str, int]:
    """Assign persistent barcodes to parameters and track their pass number.

    On first encounter, each parameter gets a random barcode
    and pass number 1.  On subsequent encounters (same parameter used again in
    a later loop iteration), the pass number is incremented.

    The barcode is a random string that uniquely identifies the parameter tensor
    across the entire logging session.  It's used (together with layer_type) to
    build the ``equivalence_class`` for parameterized operations, which
    is how loop detection recognizes "conv2d with weights W" as the same layer
    on pass 1 and pass 2.

    Args:
        arg_parameters: Parameter tensors found in the function's arguments.

    Returns:
        Dict mapping each parameter's barcode to its current pass number.
    """
    parent_param_ops = {}
    for param in arg_parameters:
        meta = get_param_meta(param)
        if meta is None or meta.param_barcode is None:
            # First time seeing this parameter — assign a unique barcode.
            param_barcode = make_random_barcode()
            set_param_meta(
                param,
                barcode=param_barcode,
                address="",
                requires_grad_before=param.requires_grad,
            )
        else:
            param_barcode = meta.param_barcode
        call_index = increment_param_call_index(param)
        parent_param_ops[param_barcode] = call_index
    return parent_param_ops


def _make_raw_param_group_barcode(
    indiv_param_barcodes: list[str],
    layer_type: str,
    *,
    output_index: int | None = None,
) -> str:
    """Build an equivalence_class string for a parameterized operation.

    Combines the layer type with sorted parameter barcodes to produce a
    canonical fingerprint.  Sorting ensures order-independence (e.g., weight
    and bias can appear in either order).

    The layer_type prefix is critical: different operations using the same
    parameters (e.g., ``isinf(weight)`` vs ``expand(weight)``) must NOT be
    grouped as the same layer.

    Example: ``"conv2d_abc123_def456"``

    Parameters
    ----------
    indiv_param_barcodes:
        Barcodes for each parameter tensor.
    layer_type:
        The normalized operation name.
    output_index:
        Zero-based output index for iterable or structured multi-output
        operations, or ``None`` for single-output operations.

    Returns
    -------
    str
        Canonical fingerprint string for this parameterized operation.
    """
    param_group_barcode = f"{layer_type}_{'_'.join(sorted(indiv_param_barcodes))}"
    if output_index is not None:
        param_group_barcode += f"_outindex{output_index}"
    return param_group_barcode


def _append_module_suffix_to_equivalence_class(
    equivalence_class: str,
    modules: list[tuple[str, int]] | tuple[tuple[str, int], ...],
) -> str:
    """Append the canonical module-stack suffix used by loop detection.

    Parameters
    ----------
    equivalence_class:
        Base operation equivalence string.
    modules:
        Module stack snapshot for the captured op, as ``(address, pass)``
        pairs ordered outer-to-inner.

    Returns
    -------
    str
        ``equivalence_class`` plus the historical module-path suffix. Empty
        module stacks append an empty suffix.
    """

    return equivalence_class + "_".join(module_pass[0] for module_pass in modules)


def _get_equivalence_class(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    i: int,
    layer_type: str,
    fields_dict: dict[str, Any],
) -> str:
    """Build an equivalence_class string for a NON-parameterized operation.

    For ops that don't use parameters (e.g., ``relu``, ``cat``, ``add``), the
    fingerprint is built from:
      1. ``layer_type``: the normalized function name.
      2. ``arg_hash``: hash of non-tensor arguments (shapes, scalar values, etc.).
      3. ``outindex`` suffix: disambiguates outputs of multi-output functions.
      4. ``module`` suffix: disambiguates identical ops in different submodules.

    This fingerprint is used by loop detection.  Two operations with the same
    fingerprint are candidates for being "the same layer on different ops."

    Note: non-parameterized ops default to ``call_index=1`` because without
    parameters to track reuse, there's no reliable way to count ops.

    Args:
        args: Positional arguments to the function call.
        kwargs: Keyword arguments to the function call.
        i: Index of this output tensor within a multi-output call.
        layer_type: The normalized operation name.
        fields_dict: Must contain ``in_multi_output`` and
            ``module``.

    Returns:
        A string key identifying this operation's equivalence class.
    """
    arg_hash = _get_hash_from_args(args, kwargs)
    equivalence_class = f"{layer_type}_{arg_hash}"
    if fields_dict["in_multi_output"]:
        equivalence_class += f"_outindex{i}"
    if fields_dict["module"] is not None:
        module_str = fields_dict["module"][0]
        equivalence_class += f"_module{module_str}"
    return equivalence_class


def _get_hash_from_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Compute a structural hash of non-tensor arguments for equivalence fingerprinting.

    Tensor arguments are excluded (they define graph edges, not structural identity).
    Parameters are also excluded (they have their own barcode system).
    The hash preserves positional indices and kwarg keys to avoid collisions
    between e.g. ``f(a=1, b=2)`` and ``f(a=2, b=1)``.

    Returns:
        A short deterministic hash string, or ``"no_args"`` if no non-tensor
        arguments are present.
    """
    args_to_hash: list[Any] = []
    for a, arg in enumerate(args):
        _append_arg_hash(arg, f"pos{a}", args_to_hash)
    for key, arg in kwargs.items():
        _append_arg_hash(arg, f"kw_{key}", args_to_hash)

    if len(args_to_hash) == 0:
        return "no_args"
    return make_short_barcode_from_input(args_to_hash)


def _append_arg_hash(arg: Any, prefix: str, args_to_hash: list[Any], _depth: int = 0) -> None:
    """Append structural fingerprint tokens for a single argument to the accumulator list.

    Builds an ``equivalence_class`` -- a structural fingerprint of the operation's
    argument types and shapes (not a content hash). This fingerprint is used by loop
    detection to identify operations that are structurally identical across ops.

    For tensors, only shape and dtype are recorded (not values). Containers (dicts, lists,
    tuples, sets) are recursed into with depth-limited traversal. Parameters are excluded.

    Args:
        arg: The argument value to fingerprint.
        prefix: String prefix encoding the argument's position/key path.
        args_to_hash: Accumulator list that fingerprint tokens are appended to.
        _depth: Recursion depth guard; stops at 10 to prevent infinite recursion.
    """
    if _depth > 10:
        args_to_hash.append(f"{prefix}_deep")
        return
    if isinstance(arg, torch.nn.Parameter):
        pass  # exclude parameters from hash — must check before Tensor (Parameter is a subclass)
    elif isinstance(arg, torch.Tensor):
        # Use shape/dtype only — formatting a tensor can trigger wrapped
        # custom_methods (item, __format__) which re-enter logging and cause
        # infinite recursion.
        args_to_hash.append(f"{prefix}_tensor{arg.shape}")
    elif isinstance(arg, dict):
        for k, v in arg.items():
            _append_arg_hash(v, f"{prefix}_dk{k}", args_to_hash, _depth + 1)
    elif isinstance(arg, (list, tuple, set)):
        for i, elem in enumerate(arg):
            _append_arg_hash(elem, f"{prefix}_i{i}", args_to_hash, _depth + 1)
    else:
        args_to_hash.append(f"{prefix}_{arg}")
