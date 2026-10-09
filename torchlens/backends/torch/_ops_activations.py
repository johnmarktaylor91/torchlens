"""Output tensor metadata and activation persistence."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from ..._robustness import UnsupportedTensorVariantError
from ...capture.flops import compute_backward_flops, compute_forward_flops
from ...data_classes.op import (
    _dedup_cached_identity_out,
    _dedup_saved_activation_out,
    _dtype_or_none,
    _effective_activation_save_mode,
    _memory_or_none,
    _recursive_safe_copy,
    _shape_or_none,
    _stamp_reference_out,
    apply_transform,
    train_mode_tripwire_armed,
    validate_streaming_transform_output,
    validate_train_mode_transform_output,
)
from ...utils._torch_compat import (
    tensor_version_or_none,
)
from ...utils.tensor_utils import (
    get_memory_amount_from_metadata,
    safe_copy,
    safe_to,
)
from ._tl import (
    get_tensor_label,
    mark_detached_saved_activation,
    pop_same_object_mutation,
)
from .completeness_witness import internal_scalar_read, record_alias_mutation_candidate
from .tensor_tracking import (
    _append_module_suffix_to_equivalence_class,
    _get_equivalence_class,
    _make_raw_param_group_barcode,
)

if TYPE_CHECKING:
    from ...data_classes.trace import Trace

if TYPE_CHECKING:
    from .ops import (
        _SETTER_MUTATION_FUNC_NAMES,
        _admit_save_budget,
        _admit_saved_args_budget,
        _commit_save_budget,
        _commit_saved_args_budget,
        _is_inplace_augmented_assignment_dunder,
        _label_version_baseline,
        _should_keep_alias_mutation_contract,
    )

__all__ = (
    "_log_output_tensor_info",
    "_save_activation_fields",
    "_stream_activation_fields",
    "_retention_device",
)


def _metadata_shape(t: torch.Tensor, op_label: str | None = None) -> tuple[int, ...]:
    """Return a captured tensor's shape, refusing TYPED for a shapeless variant.

    ``tuple(t.shape)`` is the very first metadata read of every recorded output, and on a
    NESTED tensor it dies with torch's own internal error -- "Internal error:
    NestedTensorImpl doesn't support sizes. Please file an issue." -- so a model doing
    ``torch.nested.as_nested_tensor([...])`` (including the mainstream jagged-SDPA
    layout) aborted mid-forward and told the user to file a TORCH bug about a TorchLens
    limitation. The entry gate covers meta/sparse/fake/functional INPUTS; nested is a
    whole variant class created INSIDE ``forward``, where the gate cannot see it.

    Feature-detected per the ``_torch_compat`` doctrine: the capability probed is "does
    this tensor support ``sizes``", never a version string. Any shapeless variant that
    appears later refuses the same way instead of leaking a raw internal error.
    """

    # Both mid-forward raises honor the class fields contract (R65): stable
    # ``code``, a ``remedy``, and one structured offense naming the op label
    # this output was being recorded at (the mid-forward analog of the entry
    # gate's input-tree path).
    site = op_label if op_label is not None else "<mid-forward output>"
    if getattr(t, "is_nested", False):
        raise UnsupportedTensorVariantError(
            "torchlens cannot log a NESTED tensor created inside forward(): the variant "
            "has no dense shape, so TorchLens can record no shape, memory, or FLOPs "
            "metadata for it. Restructure the forward to build the nested tensor "
            "outside the traced region, or pad to a dense tensor before the ops you "
            "want captured.",
            code="unsupported_tensor_variant",
            remedy=(
                "build the nested tensor outside the traced region, or pad to a "
                "dense tensor before the ops you want captured"
            ),
            offenses=(
                {
                    "name": "nested tensor created inside forward()",
                    "reason": "the variant has no dense shape",
                    "path": site,
                    "shape": None,
                    "dtype": str(getattr(t, "dtype", None)),
                },
            ),
        )
    try:
        return tuple(t.shape)
    except RuntimeError as error:
        raise UnsupportedTensorVariantError(
            "torchlens cannot log a tensor variant that does not support `sizes` "
            f"({type(t).__name__}); TorchLens can record no shape metadata for it. "
            f"Underlying torch error: {error}",
            code="unsupported_tensor_variant",
            remedy=(
                "materialize a dense, strided tensor before the ops you want "
                "captured; this variant exposes no shape metadata"
            ),
            offenses=(
                {
                    "name": f"shapeless tensor variant ({type(t).__name__})",
                    "reason": str(error),
                    "path": site,
                    "shape": None,
                    "dtype": str(getattr(t, "dtype", None)),
                },
            ),
        ) from error


def _log_output_tensor_info(
    self: "Trace",
    t: torch.Tensor,
    i: int,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    parent_param_ops: dict[str, int],
    fields_dict: dict[str, Any],
    autograd_saved_stats: tuple[int | None, int | None],
) -> None:
    """Populate per-tensor fields that differ across outputs of a single function call.

    This includes:
      - Counter-based label generation (``_label_raw``).
      - Operation equivalence type assignment (used by loop detection to
        identify structurally identical operations across forward-pass iterations).
      - FLOPs computation.
      - Shape, dtype, and memory size.

    Label format: ``"{layer_type}_{type_num}_{realtime_num}_raw"``
      - ``layer_type``: normalized function name (e.g. "conv2d")
      - ``type_num``: how many times this layer_type has been seen (monotonic)
      - ``realtime_num``: global operation counter across all types (monotonic)

    Args:
        t: The output tensor.
        i: Index of this tensor in a multi-output function call (0 for single outputs).
        args: Positional args to the function that created the tensor.
        kwargs: Keyword args to the function.
        parent_param_ops: Dict mapping param barcodes to their current pass number.
        fields_dict: Per-tensor fields dict to populate (mutated in place).
    """
    layer_type = fields_dict["type"]
    indiv_param_barcodes = list(parent_param_ops.keys())
    self._raw_graph_ws.layer_counter += 1
    self._raw_graph_ws.raw_layer_type_counter[layer_type] += 1
    raw_index = self._raw_graph_ws.layer_counter
    type_index = self._raw_graph_ws.raw_layer_type_counter[layer_type]
    _label_raw = f"{layer_type}_{type_index}_{raw_index}_raw"

    # Determine operation equivalence type — the fingerprint used by loop detection
    # to group structurally identical operations (same layer across ops).
    if len(parent_param_ops) > 0:
        # Parameterized ops: equivalence is defined by the exact set of parameters
        # used, combined with the operation type.  E.g., two conv2d calls using the
        # same weight+bias tensors are the same layer on different ops.
        output_index = i if fields_dict["in_multi_output"] else None
        equivalence_class = _make_raw_param_group_barcode(
            indiv_param_barcodes,
            layer_type,
            output_index=output_index,
        )
        base_equivalence_class = equivalence_class
        self.layers_with_params[equivalence_class].append(_label_raw)
        fields_dict["pass_index"] = len(self.layers_with_params[equivalence_class])
        equivalence_class = _append_module_suffix_to_equivalence_class(
            equivalence_class, fields_dict["modules"]
        )
        fields_dict["equivalence_class"] = equivalence_class
    else:
        # Non-parameterized ops: equivalence is a hash of the operation type,
        # non-tensor args, output index, and containing module.  Each unique
        # non-param operation is seen only once (pass_index=1).
        logged_func_name = fields_dict["func_name"]
        is_inplace_output = _is_inplace_augmented_assignment_dunder(logged_func_name) or (
            logged_func_name.endswith("_") and not logged_func_name.startswith("__")
        )
        equivalence_layer_type = f"{layer_type}_inplace" if is_inplace_output else layer_type
        equivalence_class = _get_equivalence_class(
            args, kwargs, i, equivalence_layer_type, fields_dict
        )
        base_equivalence_class = equivalence_class
        equivalence_class = _append_module_suffix_to_equivalence_class(
            equivalence_class, fields_dict["modules"]
        )
        fields_dict["equivalence_class"] = equivalence_class
        fields_dict["pass_index"] = 1

    # equivalent_ops is a DIRECT reference to the Trace-level set —
    # all entries sharing this equivalence type point to the same set object.
    # Defensive: if equivalence_class isn't yet registered (e.g. user-injected
    # tensor through intervention/raw-hook with a fresh hash), create the
    # set on demand rather than crashing with KeyError.
    if base_equivalence_class not in self.op_equivalence_classes:
        self.op_equivalence_classes[base_equivalence_class] = set()
    self.op_equivalence_classes[base_equivalence_class].add(_label_raw)
    fields_dict["equivalent_ops"] = self.op_equivalence_classes[base_equivalence_class]

    # In-place ops return the same tensor object, which already has a raw label -- but so
    # does a NON-mutating identity return (``x.cpu()`` on CPU, ``x.contiguous()`` on a
    # contiguous tensor). ``is_inplace`` must reflect ACTUAL mutation, not mere label
    # presence: a false in-place flag makes the runnable descriptor persist a version/alias
    # relation the replay cannot satisfy on the original input (MUTATION_VERSION_MISMATCH).
    # In-place ops return the same tensor object as one of their inputs, so the output already
    # carries a capture label; a NON-mutating identity return (``x.cpu()`` on CPU) ALSO returns
    # its receiver and so ALSO carries a label. ``is_inplace`` must reflect ACTUAL mutation, not
    # mere label presence. A mutation can ALSO target an UNLABELLED alias (``y.data.add_(5.0)``):
    # ``y.data`` is a fresh Python tensor object with no TorchLens metadata, yet ``add_`` genuinely
    # mutates ``y``'s storage. Keying is_inplace on label presence (``prior_label is None -> never
    # in-place``) misses exactly that case, so the runnable descriptor never learns the write and
    # would falsely VERIFY -- the version-bump / mutation-signature test below classifies it right.
    prior_label = get_tensor_label(t)
    # A same-object return is logged against a fresh copy; the wrapper stamps the
    # call's mutation verdict on it (signature incl. ``inplace=True``, refined by the
    # live receiver's version counter). Consume it so it never leaks onto the copy.
    same_object_mutation = pop_same_object_mutation(t)
    if not _should_keep_alias_mutation_contract(self):
        # Default forward-only capture: preserve the historical label-based flag exactly (no
        # runnable replay consumes it, so identity returns are harmless and goldens stay unchanged).
        fields_dict["is_inplace"] = prior_label is not None
    else:
        # Runnable / intervention / backward / validation: require a real version bump on the
        # aliased tensor. The version baseline is the tensor's version when TorchLens last labeled
        # it as an op output; a bump since then is a genuine mutation, an unchanged version is an
        # identity return. When no baseline exists (a raw input / param / buffer first touched here,
        # a setter-style op whose logged output is a reconstructed target, or an UNLABELLED ``.data``
        # alias), fall back to the operator's mutation signature -- an in-place-named op (``add_`` /
        # ``mul_`` / ``copy_`` / ``__i*``), a setter dunder (``__setitem__`` / ``__delitem__``, which
        # mutate but do not end in ``_``), or an ``out=`` tensor kwarg -- so genuine mutation is
        # still detected while a non-mutating identity return (``x.cpu()`` / ``x.contiguous()``,
        # which has no mutation-signature name) is not, and does not crash. Baselines are
        # session-scoped: a snapshot recorded by an EARLIER capture never serves as this
        # session's baseline (the tensor may have been mutated between captures -- W3 F7).
        baseline = _label_version_baseline(t)
        # TorchLens bookkeeping ``_version`` read; on an in-place op ``t`` IS
        # the user's receiver, so an unmarked read on registered state would
        # record a phantom read-kind (r65 unread-bit contract).
        with internal_scalar_read():
            current_version = tensor_version_or_none(t)
        if same_object_mutation is not None:
            fields_dict["is_inplace"] = bool(same_object_mutation)
        elif baseline is not None and current_version is not None:
            fields_dict["is_inplace"] = current_version != baseline
        else:
            name = fields_dict["func_name"]
            name_mutating = (
                _is_inplace_augmented_assignment_dunder(name)
                or name in _SETTER_MUTATION_FUNC_NAMES
                or (name.endswith("_") and not name.startswith("__"))
                # Mutating property setters (``t.real = rhs``, round-31 M6)
                # keep the property's plain name; their descriptor ``__set__``
                # callable is the mutation signature.
                or str(getattr(fields_dict.get("func"), "__name__", "")) == "__set__"
            )
            out_kwarg = kwargs.get("out") if isinstance(kwargs, dict) else None
            has_out_tensor = isinstance(out_kwarg, torch.Tensor) or (
                isinstance(out_kwarg, (list, tuple))
                and any(isinstance(item, torch.Tensor) for item in out_kwarg)
            )
            fields_dict["is_inplace"] = bool(name_mutating or has_out_tensor)
        # A genuine mutation whose op gets ORPHAN-PRUNED silently drops the write, whether the
        # TARGET is an invisible ``.data`` / foreign alias (``y.data.add_(5.0)``, unlabelled) OR a
        # LABELLED VIEW of one (``y.data.view(-1)[0] = 9``, whose ``__setitem__`` targets a labelled
        # view yet is still input-disconnected and pruned). Keying the flag on ``prior_label is None``
        # missed the labelled-view family (r14-H2), so record EVERY in-place op as a candidate and let
        # orphan removal (``_record_pruned_alias_mutation``) keep only the ones ACTUALLY pruned ->
        # UNVERIFIABLE, instead of a false VERIFIED with the mutation gone. An in-place op that stays
        # graph-connected (``y.add_(2); return y``) is never in the pruned set, so it is not flagged
        # and is replayed normally.
        if fields_dict["is_inplace"]:
            record_alias_mutation_candidate(self, _label_raw)
    # Round-31 M5: same-object in-place returns and pass-through container
    # members are logged against a safe copy whose ``grad_fn`` is TorchLens's
    # own ``CloneBackward`` node. The wrapper stamped the USER op's live
    # autograd node on the copy; that snapshot is the operation's metadata.
    user_grad_fn = getattr(t, "tl_user_grad_fn", None)
    if user_grad_fn is not None:
        try:
            delattr(t, "tl_user_grad_fn")
        except AttributeError:
            pass
    # TorchLens bookkeeping ``grad_fn`` read (same r65 receiver aliasing note
    # as the ``_version`` read above).
    with internal_scalar_read():
        op_grad_fn = user_grad_fn if user_grad_fn is not None else t.grad_fn
    grad_fn_cls = type(op_grad_fn) if op_grad_fn is not None else None
    fields_dict["grad_fn_class_name"] = None if grad_fn_cls is None else grad_fn_cls.__name__
    fields_dict["grad_fn_class_qualname"] = (
        None if grad_fn_cls is None else f"{grad_fn_cls.__module__}.{grad_fn_cls.__qualname__}"
    )
    fields_dict["grad_fn_object_id"] = id(op_grad_fn) if op_grad_fn is not None else None
    # Autograd Function objects do not consistently support weak references.
    # Keep the object only until explicit backward capture has registered hooks;
    # the backward finalizer clears these strong refs to avoid pinning graphs.
    fields_dict["grad_fn_handle"] = op_grad_fn
    fields_dict["grad_fn"] = None

    if fields_dict["in_multi_output"]:
        fields_dict["multi_output_index"] = i
    else:
        fields_dict["multi_output_index"] = None

    # ``ndim`` is an unwrapped property: TorchLens's own metadata reads must not
    # enter the call fingerprint (``_state.CallFingerprint``) as model calls.
    if (t.dtype == torch.bool) and t.ndim == 0:
        fields_dict["is_scalar_bool"] = True
        try:
            # TorchLens's own scalar-bool value read is a capture-internal escape:
            # mark it explicitly so the completeness witness never mistakes it for a
            # user host escape (allowlist-by-construction; see internal_scalar_read).
            with internal_scalar_read():
                fields_dict["bool_value"] = t.item()
        except RuntimeError:
            # .item() forbidden inside torch.vmap context
            fields_dict["bool_value"] = None
    else:
        fields_dict["is_scalar_bool"] = False
        fields_dict["bool_value"] = None

    # General info
    fields_dict["_label_raw"] = _label_raw
    fields_dict["step_index"] = None
    fields_dict["recurrent_ops"] = []
    fields_dict["raw_index"] = raw_index
    fields_dict["step_index"] = None
    fields_dict["source_trace"] = self
    fields_dict["_tracing_finished"] = False

    # Other labeling info
    fields_dict["layer_label"] = None
    fields_dict["layer_label_short"] = None
    fields_dict["label"] = None
    fields_dict["label_short"] = None
    fields_dict["layer_label"] = None
    fields_dict["layer_label_short"] = None
    fields_dict["type"] = layer_type
    fields_dict["_layer_label_raw"] = _label_raw
    fields_dict["type_index"] = type_index
    fields_dict["num_passes"] = 1
    fields_dict["lookup_keys"] = []

    # Saved tensor info
    fields_dict["out"] = None
    fields_dict["transformed_out"] = None
    fields_dict["has_saved_activation"] = False
    fields_dict["activation_transform"] = self.activation_transform
    # Collective boundary nodes carry their portable collective_boundary_v1
    # payload in the reserved "collective" annotations namespace; the wrapper
    # stamps it on the replay callable (same channel as the __tl_transform_*
    # dunders). Deep-copied per output so sibling records never share state.
    collective_info = getattr(fields_dict.get("func"), "__tl_collective_info__", None)
    if collective_info:
        import copy as _copy

        fields_dict["annotations"] = {"collective": _copy.deepcopy(collective_info)}
        if collective_info.get("schema") == "functional_collective_boundary_v0":
            # Plane-W settlement: funcol completions are observed AFTER this
            # deep copy is taken (the wait fires later in the same forward),
            # so the capture-scoped session tracks every per-record copy and
            # syncs it from the settled master at session close.
            from .funcol import active_funcol_session

            session = active_funcol_session()
            if session is not None:
                session.register_annotation_copy(fields_dict["annotations"]["collective"])
    else:
        fields_dict["annotations"] = {}
    fields_dict["intervention_replaced"] = False
    fields_dict["fire_results"] = ()
    fields_dict["has_saved_args"] = False
    fields_dict["saved_args"] = None
    fields_dict["saved_kwargs"] = None
    fields_dict["shape"] = _metadata_shape(t, op_label=fields_dict.get("_label_raw"))
    fields_dict["transformed_out_shape"] = None
    fields_dict["dtype"] = t.dtype
    fields_dict["transformed_out_dtype"] = None
    fields_dict["activation_memory"] = get_memory_amount_from_metadata(
        t,
        fields_dict["shape"],
        fields_dict["dtype"],
    )
    fields_dict["transformed_activation_memory"] = None
    fields_dict["visualizer_path"] = None
    # Not measured is None, never a fabricated 0 (observe item 15); the
    # track_device_memory commit-site projection overwrites when sampled.
    fields_dict["bytes_delta_at_call"] = None
    fields_dict["bytes_peak_at_call"] = None
    (
        fields_dict["autograd_memory"],
        fields_dict["num_autograd_tensors"],
    ) = autograd_saved_stats

    # FLOPs computation
    fields_dict["flops_forward"] = compute_forward_flops(
        fields_dict.get("func_name"),  # type: ignore[arg-type]
        fields_dict["shape"],
        fields_dict.get("param_shapes", []),
        args,
        kwargs,
    )
    fields_dict["flops_backward"] = compute_backward_flops(
        fields_dict.get("func_name"),  # type: ignore[arg-type]
        fields_dict["flops_forward"],
    )

    # Child tensor variation tracking
    fields_dict["has_out_variations"] = False
    fields_dict["out_versions_by_child"] = {}

    # If internally initialized, fix this information:
    if len(fields_dict["parents"]) == 0:
        fields_dict["is_internal_source"] = True
        fields_dict["has_internal_source_ancestor"] = True
        fields_dict["internal_source_parents"] = []
        fields_dict["internal_source_ancestors"] = {_label_raw}


def _save_activation_fields(
    trace: "Trace",
    fields_dict: dict[str, Any],
    t: torch.Tensor,
    t_args: tuple[Any, ...],
    t_kwargs: dict[str, Any],
    activation_transform: Callable[..., Any] | None,
) -> None:
    """Save activation data directly into a live field dictionary.

    Parameters
    ----------
    trace
        Active trace.
    fields_dict
        Mutable live field mapping.
    t
        Output tensor to save.
    t_args
        Positional function arguments.
    t_kwargs
        Keyword function arguments.
    activation_transform
        Optional output transform.

    Returns
    -------
    None
        Mutates ``fields_dict``.
    """

    writer = getattr(trace, "_out_writer", None)
    try:
        save_mode = _effective_activation_save_mode(
            trace,
            func_name=fields_dict.get("func_name"),
            is_inplace=bool(fields_dict.get("is_inplace", False)),
        )
        save_raw_activations = getattr(trace, "save_raw_activations", True)
        store_raw = save_raw_activations or activation_transform is None
        # Explorer P2: when nothing raw is retained AND the transform declares
        # the non-differentiable summary role, the transient safe_copy is
        # skipped entirely -- the reducer sees a detached view of the live
        # output at the save point (correct: no later in-place mutation has
        # run yet). Function-local import: this function is rebound into
        # backends.torch.ops globals (_split_rebind), so a new module-level
        # name would not resolve there.
        from ...ir.summary_role import is_summary_transform

        reduce_only = not store_raw and is_summary_transform(activation_transform)
        budget_reservation = (
            None
            if reduce_only
            else _admit_save_budget(
                trace,
                t,
                fields_dict,
                target_device=(
                    torch.device("cpu")
                    if save_mode == "cpu_async"
                    else _retention_device(t, fields_dict.get("output_device"))
                ),
                retain_in_ram=True,
            )
        )
        # Pre-copy identity probe (dedup-after-copy ordering): a hit reuses
        # the already-saved payload and skips the clone entirely. Restricted
        # to plain "copy" mode -- reference/view copies are free and
        # cpu_async has fence side effects.
        dedup_cached_out = (
            _dedup_cached_identity_out(
                trace,
                t,
                fields_dict["annotations"],
                getattr(trace, "save_arg_values", False),
            )
            if store_raw and save_mode == "copy"
            else None
        )
        if reduce_only:
            raw_out = t
        elif dedup_cached_out is not None:
            raw_out = dedup_cached_out
        else:
            # Single-transport retention (r8 b5 R35): ``"copy"`` mode
            # materializes directly on the retention device (one copy); the
            # other save modes ignore the target and keep the follow-up move.
            raw_out = safe_copy(
                t,
                fields_dict["detach_saved_activations"],
                save_mode=save_mode,
                target_device=fields_dict["output_device"],
            )
            if fields_dict["output_device"] not in [str(raw_out.device), "same"]:
                if save_mode == "cpu_async":
                    # R36: a cpu_async copy of THIS tensor may still be in
                    # flight; a cross-device move would read the pinned host
                    # buffer before its D2H fence. Drain pending fences first.
                    from ...utils.tensor_utils import synchronize_pending_cpu_async_copies

                    synchronize_pending_cpu_async_copies()
                raw_out = safe_to(raw_out, fields_dict["output_device"])
            _stamp_reference_out(fields_dict["annotations"], raw_out, save_mode)

        fields_dict["shape"] = tuple(raw_out.shape)
        fields_dict["dtype"] = raw_out.dtype
        fields_dict["activation_memory"] = get_memory_amount_from_metadata(
            raw_out,
            fields_dict["shape"],
            fields_dict["dtype"],
        )

        if store_raw:
            if dedup_cached_out is None:
                raw_out = _dedup_saved_activation_out(
                    trace,
                    t,
                    raw_out,
                    fields_dict["_layer_label_raw"],
                    fields_dict["annotations"],
                    getattr(trace, "save_arg_values", False),
                )
            if isinstance(raw_out, torch.Tensor):
                mark_detached_saved_activation(
                    t,
                    raw_out,
                    fields_dict.get("_layer_label_raw"),
                )
        fields_dict["out"] = raw_out if store_raw else None
        fields_dict["transformed_out"] = None
        fields_dict["transformed_out_shape"] = None
        fields_dict["transformed_out_dtype"] = None
        fields_dict["transformed_activation_memory"] = None
        if activation_transform is not None:
            transformed_out = apply_transform(
                label=fields_dict.get("_layer_label_raw"),
                raw_label=fields_dict.get("_label_raw"),
                func_name=fields_dict.get("func_name"),
                tensor=raw_out,
                transform=activation_transform,
                transform_kind="activation",
                streaming_active=writer is not None,
            )
            validate_train_mode_transform_output(
                raw_tensor=raw_out,
                transformed_tensor=transformed_out,
                transform_kind="activation",
                tripwire_armed=train_mode_tripwire_armed(
                    backward_ready=fields_dict.get(
                        "backward_ready", getattr(trace, "backward_ready", False)
                    ),
                    transform=activation_transform,
                ),
                label=fields_dict.get("_layer_label_raw"),
            )
            validate_streaming_transform_output(
                transformed_tensor=transformed_out,
                transform_kind="activation",
                streaming_active=writer is not None,
                label=fields_dict.get("_layer_label_raw"),
            )
            if reduce_only:
                from ...ir.summary_role import ensure_summary_output_owns_storage

                transformed_out = ensure_summary_output_owns_storage(transformed_out, t)
            fields_dict["transformed_out"] = transformed_out
            fields_dict["transformed_out_shape"] = _shape_or_none(transformed_out)
            fields_dict["transformed_out_dtype"] = _dtype_or_none(transformed_out)
            fields_dict["transformed_activation_memory"] = _memory_or_none(transformed_out)
        fields_dict["has_saved_activation"] = True
        if reduce_only:
            # No source-sized admission happened (no clone allocated); the
            # retained summary output is charged directly so a reduce-only
            # capture stays visible to a tight save_budget.
            try:
                accountant = trace._save_budget_accountant
            except AttributeError:
                accountant = None
            if accountant is not None:
                accountant.charge_retained(
                    str(fields_dict.get("_layer_label_raw") or "<reduce_only>"),
                    (fields_dict["transformed_out"],),
                )
        else:
            _commit_save_budget(trace, fields_dict, budget_reservation)

        _stream_activation_fields(trace, fields_dict)

        out_sink = getattr(trace, "_out_sink", None)
        if out_sink is not None and isinstance(fields_dict["out"], torch.Tensor):
            out_sink(fields_dict["_label_raw"], fields_dict["out"])

        if trace.save_arg_values:
            # Admit BEFORE the clones allocate (r8 R34, sol 2): the snapshot
            # used to clone every tensor argument first and charge after.
            arg_reservations = _admit_saved_args_budget(trace, fields_dict, t_args, t_kwargs)
            fields_dict["has_saved_args"] = True
            fields_dict["saved_args"] = [_recursive_safe_copy(arg) for arg in t_args]
            fields_dict["saved_kwargs"] = {
                key: _recursive_safe_copy(value) for key, value in t_kwargs.items()
            }
            _commit_saved_args_budget(trace, fields_dict, arg_reservations)
        else:
            fields_dict["saved_args"] = None
            fields_dict["saved_kwargs"] = None
    except Exception as exc:
        if writer is not None:
            writer.abort(f"Failed while saving out for {fields_dict['_label_raw']}: {exc}")
        raise


def _stream_activation_fields(trace: "Trace", fields_dict: dict[str, Any]) -> None:
    """Write saved activation tensors during capture when streaming is active.

    Parameters
    ----------
    trace:
        Active trace whose writer receives tensor blobs.
    fields_dict:
        Mutable live field mapping for the captured operation.

    Returns
    -------
    None
        Mutates pending blob-id fields when blobs are written.
    """

    writer = getattr(trace, "_out_writer", None)
    if writer is None or not trace._wrapper_runtime_ws.in_exhaustive_pass:
        return

    # R36: serializing a cpu_async payload to disk is a host-side byte read;
    # the pinned buffer may still be in flight. No-op unless async fence
    # events are pending.
    from ...utils.tensor_utils import synchronize_pending_cpu_async_copies

    synchronize_pending_cpu_async_copies()

    label = fields_dict["_label_raw"]
    for tensor_field, pending_field, kind in (
        ("out", "_pending_blob_id", "out"),
        ("transformed_out", "_pending_transformed_out_blob_id", "transformed_out"),
    ):
        tensor = fields_dict.get(tensor_field)
        if tensor is None:
            continue
        blob_id = writer.next_blob_id()
        fields_dict[pending_field] = blob_id
        writer.submit_blob(blob_id, tensor, kind=kind, label=label)


def _retention_device(tensor: torch.Tensor, configured: Any) -> torch.device:
    """Return the projected RAM retention device for one activation.

    Parameters
    ----------
    tensor:
        Live source tensor.
    configured:
        User-configured output device or ``"same"``.

    Returns
    -------
    torch.device
        Device used for pre-allocation admission.
    """

    if configured in (None, "same", str(tensor.device)):
        return tensor.device
    return torch.device(configured)
