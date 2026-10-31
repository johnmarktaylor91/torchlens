"""Postprocess step executor: the ordered step registry and ``run_pipeline``.

``STEP_REGISTRY`` lists steps 1-20 in the order they run; step order is
load-bearing (warnings and first-exception order depend on it) and is verified
by the byte-identity oracles. Step "0" is the prologue in ``postprocess`` and is
deliberately absent.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..utils.display import _vtimed

if TYPE_CHECKING:
    from ..data_classes.trace import Trace


def _pp() -> Any:
    """Return the live ``torchlens.postprocess`` module namespace.

    Every step body resolves its callable through this namespace AT CALL
    TIME: tests monkeypatch step
    functions on the module — a registry holding imported references would
    break the in-place-mutation tripwire test and the streaming/step-0
    seams silently.
    """

    return sys.modules["torchlens.postprocess"]


@dataclass
class StepContext:
    """Mutable per-run context threaded through the step registry."""

    trace: Trace
    output_tensors: list[Any]
    output_tensor_addresses: list[str]
    output_parent_labels: list[Any]
    capture_session: Any | None
    #: Step-18-position snapshot: 18's should_run IS the
    #: snapshot point — 18 clears _out_writer, so a re-evaluated 19
    #: predicate would always be false and streamed outs would never be
    #: evicted. Context-writing, never trace-writing.
    finalize_streaming: bool | None = None


def _always(ctx: StepContext) -> bool:
    """Default should_run: the step always executes."""

    return True


@dataclass(frozen=True)
class StepSpec:
    """One executable pipeline step.

    ``should_run`` is evaluated EXACTLY ONCE per step per run, in registry
    order (executor invariant, counting-tested).
    """

    step: str
    run: Callable[[StepContext], None]
    should_run: Callable[[StepContext], bool] = _always


def _run_step_1(ctx: StepContext) -> None:
    """Step 1: create dedicated output nodes for each attributable output tensor."""

    with _vtimed(ctx.trace, "  Step 1: Add output layers"):
        _pp()._add_output_layers(
            ctx.trace,
            ctx.output_tensors,
            ctx.output_tensor_addresses,
            ctx.output_parent_labels,
        )


def _run_step_2(ctx: StepContext) -> None:
    """Step 2: mark every node that is an ancestor of a model output."""

    with _vtimed(ctx.trace, "  Step 2: Trace output ancestors"):
        _pp()._find_output_ancestors(ctx.trace)


def _run_step_3(ctx: StepContext) -> None:
    """Step 3: drop raw nodes that reach no model output."""

    with _vtimed(ctx.trace, "  Step 3: Remove orphan nodes"):
        _pp()._remove_orphan_nodes(ctx.trace)


def _run_step_4(ctx: StepContext) -> None:
    """Step 4: write the optional input/output distance metadata."""

    with _vtimed(ctx.trace, "  Step 4: Input/output distances"):
        _pp()._mark_layer_depths(ctx.trace)


def _should_run_step_4(ctx: StepContext) -> bool:
    """Gate step 4 on the opt-in ``Trace.mark_layer_depths`` flag."""

    return bool(ctx.trace.mark_layer_depths)


def _run_step_5(ctx: StepContext) -> None:
    """Step 5: attribute conditional branches (AST, bool, event, and edge evidence)."""

    with _vtimed(ctx.trace, "  Step 5: Mark conditional branches"):
        _pp()._mark_conditional_branches(ctx.trace)


def _run_step_6(ctx: StepContext) -> None:
    """Step 6: deduplicate buffer layers and reconnect their edges."""

    with _vtimed(ctx.trace, "  Step 6: Fix buffer layers"):
        _pp()._fix_buffer_layers(ctx.trace)


def _run_step_7(ctx: StepContext) -> None:
    """Step 7: group recurrent calls into multi-pass layers.

    With ``recurrence_detection`` off this degrades to shared-parameter
    grouping only, which is the historical single-pass layout.
    """

    trace = ctx.trace
    loop_desc = (
        "  Step 7: Loop detection (full)"
        if trace.recurrence_detection
        else "  Step 7: Loop detection (params only)"
    )
    with _vtimed(trace, loop_desc):
        if trace.recurrence_detection:
            _pp()._detect_and_label_loops(trace)
        else:
            _pp()._group_by_shared_params(trace)


def _run_step_8(ctx: StepContext) -> None:
    """Step 8: build the raw-label to final-label map."""

    with _vtimed(ctx.trace, "  Step 8: Map labels"):
        _pp()._map_raw_labels_to_final_labels(ctx.trace)


def _run_step_9(ctx: StepContext) -> None:
    """Step 9: write the final per-layer and per-module fields."""

    with _vtimed(ctx.trace, "  Step 9: Log final info"):
        _pp()._log_final_info_for_layers(ctx.trace)


def _run_step_10(ctx: StepContext) -> None:
    """Step 10: rename global label references to their final names."""

    with _vtimed(ctx.trace, "  Step 10: Rename labels"):
        _pp()._rename_model_history_layer_names(ctx.trace)


def _run_step_11(ctx: StepContext) -> None:
    """Step 11: build lookup keys, finalize retained layers, warn on unattributed args.

    The three calls share one step because the saved-summary refresh and the
    unattributed-arg warning both read the retained-layer lists this step
    finalizes.
    """

    module = _pp()
    with _vtimed(ctx.trace, "  Step 11: Build lookup keys"):
        module._build_lookup_keys_and_finalize_retained_layers(ctx.trace)
        module._refresh_fast_saved_summary(ctx.trace)
        module._warn_unattributed_tensor_args(ctx.trace)


def _run_step_11_5(ctx: StepContext) -> None:
    """Step 11.5: resolve source assignment variable names from the AST index."""

    with _vtimed(ctx.trace, "  Step 11.5: Populate source var names"):
        _pp()._populate_var_names(ctx.trace)


def _run_step_11_75(ctx: StepContext) -> None:
    """Step 11.75: let the capture session resolve deferred retention decisions.

    Runs only when a ``CaptureSession`` is attached; ``should_run`` gates on
    that, so the body may assert it.
    """

    capture_session = ctx.capture_session
    assert capture_session is not None  # should_run gates on this
    with _vtimed(ctx.trace, "  Step 11.75: Resolve deferred retention"):
        capture_session.resolve_deferred_retention(ctx.trace, list(ctx.output_tensors))


def _run_step_12(ctx: StepContext) -> None:
    """Step 12: strip TorchLens ``tl_*`` attributes from every saved tensor."""

    with _vtimed(ctx.trace, "  Step 12: Undecorate tensors"):
        _pp()._undecorate_all_saved_tensors(ctx.trace)


def _run_step_13(ctx: StepContext) -> None:
    """Step 13: clear the CUDA caching allocator for CUDA-touching captures."""

    # Unwrapped by _vtimed (historical). R36-3: additionally gated on the CAPTURE having touched CUDA (the
    # trace-level forward_memory_backend fact) so a CPU-only trace inside a
    # GPU training loop no longer flushes the caller's allocator.
    from ..utils.tensor_utils import capture_touched_cuda

    module = _pp()
    if module._is_cuda_available() and capture_touched_cuda(ctx.trace):
        module.torch.cuda.empty_cache()


def _run_step_14(ctx: StepContext) -> None:
    """Step 14: record elapsed capture timing on the trace."""

    with _vtimed(ctx.trace, "  Step 14: Log timing"):
        _pp()._log_time_elapsed(ctx.trace)


def _run_step_15(ctx: StepContext) -> None:
    """Step 15: build and complete the ``Param`` records."""

    with _vtimed(ctx.trace, "  Step 15: Finalize params"):
        _pp()._finalize_param_logs(ctx.trace)


def _run_step_15_5(ctx: StepContext) -> None:
    """Step 15.5: build aggregate ``Layer`` records and the pass-index index.

    ``Trace.by_pass`` maps each pass index to the ``layer_list`` positions of
    the layers recorded at that pass; layers with no ``pass_index`` are absent
    rather than bucketed.
    """

    trace = ctx.trace
    with _vtimed(trace, "  Step 15.5: Build layer logs"):
        _pp()._build_layer_logs(trace)
        trace.by_pass = {}
        for index, op in enumerate(trace.layer_list):
            pass_index = getattr(op, "pass_index", None)
            if pass_index is not None:
                trace.by_pass.setdefault(pass_index, []).append(index)


def _run_step_16(ctx: StepContext) -> None:
    """Step 16: build ``Module`` records and refresh the saved module-call count."""

    module = _pp()
    with _vtimed(ctx.trace, "  Step 16: Build module logs"):
        module._build_module_logs(ctx.trace)
        module.refresh_saved_module_call_count(ctx.trace)


def _run_step_16_5(ctx: StepContext) -> None:
    """Step 16.5: normalize layer addresses, then hash the graph shape.

    The hash is taken here, before step 17 flips the trace to user-facing
    behavior and changes what the traversal observes.
    """

    module = _pp()
    trace = ctx.trace
    with _vtimed(trace, "  Step 16.5: Graph shape hash"):
        module.populate_normalized_layer_addresses(trace)
        trace.graph_shape_hash = module.compute_graph_shape_hash(trace)


def _run_step_17(ctx: StepContext) -> None:
    """Step 17: switch the trace from build-time to user-facing behavior."""

    with _vtimed(ctx.trace, "  Step 17: Mark pass finished"):
        _pp()._set_tracing_finished(ctx.trace)


def _run_step_17_5(ctx: StepContext) -> None:
    """Step 17.5: adopt container records, then drop the per-phase workspaces.

    This is the contracted terminal consume of ``RawGraphWorkspace``,
    ``ModuleCaptureWorkspace``, and ``WrapperRuntimeWorkspace``: nothing after
    this step may read them. The container registry's records are copied onto
    the trace before its live state is cleared.
    """

    # Unwrapped by _vtimed (historical). Adopt container records, then drop
    # the per-phase workspaces (terminal consumes).
    trace = ctx.trace
    wrapper_ws = trace.__dict__.get("_wrapper_runtime_ws")
    if wrapper_ws is not None:
        registry = getattr(wrapper_ws, "container_registry", None)
        if registry is not None:
            if registry.records:
                trace.__dict__["_containers"] = dict(registry.records)
            registry.clear_live_state()
    for field_name in (
        "_raw_graph_ws",
        "_module_capture_ws",
        "_wrapper_runtime_ws",
        "capture_events",
        "_output_container_specs_by_raw_label",
    ):
        trace.__dict__.pop(field_name, None)


def _should_run_step_18(ctx: StepContext) -> bool:
    """THE streaming snapshot point (deliberately context-writing)."""

    trace = ctx.trace
    ctx.finalize_streaming = getattr(trace, "_out_writer", None) is not None and not getattr(
        trace, "_defer_streaming_bundle_finalization", False
    )
    return ctx.finalize_streaming


def _run_step_18(ctx: StepContext) -> None:
    """Step 18: finalize the streamed out bundle."""

    with _vtimed(ctx.trace, "  Step 18: Finalize streamed bundle"):
        _pp()._finalize_streamed_bundle(ctx.trace)


def _should_run_step_19(ctx: StepContext) -> bool:
    """Gate step 19 on step 18 having run with in-memory outs not requested."""

    return bool(ctx.finalize_streaming) and not ctx.trace._keep_outs_in_memory


def _run_step_19(ctx: StepContext) -> None:
    """Step 19: evict streamed outs from memory."""

    with _vtimed(ctx.trace, "  Step 19: Evict streamed outs"):
        _pp()._evict_streamed_outs(ctx.trace)


def _run_step_20(ctx: StepContext) -> None:
    """Step 20: release live parameter references held for finalization."""

    with _vtimed(ctx.trace, "  Step 20: Release param refs"):
        ctx.trace.release_param_refs(allow_iter_rehydrate=True)


#: The executable registry, in run order.
STEP_REGISTRY: tuple[StepSpec, ...] = (
    StepSpec("1", _run_step_1),
    StepSpec("2", _run_step_2),
    StepSpec("3", _run_step_3),
    StepSpec("4", _run_step_4, should_run=_should_run_step_4),
    StepSpec("5", _run_step_5),
    StepSpec("6", _run_step_6),
    StepSpec("7", _run_step_7),
    StepSpec("8", _run_step_8),
    StepSpec("9", _run_step_9),
    StepSpec("10", _run_step_10),
    StepSpec("11", _run_step_11),
    StepSpec("11.5", _run_step_11_5),
    StepSpec(
        "11.75",
        _run_step_11_75,
        should_run=lambda ctx: ctx.capture_session is not None,
    ),
    StepSpec("12", _run_step_12),
    StepSpec("13", _run_step_13),
    StepSpec("14", _run_step_14),
    StepSpec("15", _run_step_15),
    StepSpec("15.5", _run_step_15_5),
    StepSpec("16", _run_step_16),
    StepSpec("16.5", _run_step_16_5),
    StepSpec("17", _run_step_17),
    StepSpec("17.5", _run_step_17_5),
    StepSpec("18", _run_step_18, should_run=_should_run_step_18),
    StepSpec("19", _run_step_19, should_run=_should_run_step_19),
    StepSpec("20", _run_step_20),
)


def _step_11_violation(trace: Trace) -> str | None:
    """Return the first step-11 lookup-key postcondition violation, if any."""

    for op in trace.layer_list:
        if not op.label:
            return f"Step 11 left {op!r} without a final op label"
        if not op.layer_label:
            return f"Step 11 left {op!r} without a final layer label"
        if not op.lookup_keys:
            return f"Step 11 left {op.label} without lookup keys"
        if trace.layer_dict_all_keys[op.label] is not op:
            return f"Step 11 did not map op label {op.label!r} to its own op"
        # The bare layer label resolves to ONE pass of that layer (the public
        # lookup contract keeps the LAST pass for multi-pass layers), never to
        # a foreign layer's op.
        resolved = trace.layer_dict_all_keys[op.layer_label]
        if resolved.layer_label != op.layer_label:
            return (
                f"Step 11 mapped layer label {op.layer_label!r} to a foreign op {resolved.label!r}"
            )
    return None


def _step_15_5_violation(trace: Trace) -> str | None:
    """Return the step-15.5 aggregate-layer postcondition violation, if any."""

    if not trace.layer_logs:
        return "Step 15.5 must build aggregate layer logs"
    if len(trace.layer_logs) != len(trace.layer_labels):
        return (
            f"Step 15.5 built {len(trace.layer_logs)} layer logs for "
            f"{len(trace.layer_labels)} layer labels"
        )
    if not isinstance(trace.by_pass, dict):
        return "Step 15.5 must build by_pass as a dict"
    return None


def _step_1_violation(ctx: StepContext) -> str | None:
    """Return the step-1 output-registration postcondition violation, if any.

    Step 1 creates exactly one output node per attributable output tensor
    (one with a graph parent) and none for the rest, so a model whose outputs
    are all unattributable (``None``, raw ``torch.func`` escapes) legitimately
    has no output layers; the count, not non-emptiness, is the contract.
    """

    expected = sum(1 for label in ctx.output_parent_labels if label is not None)
    registered = len(ctx.trace.output_layers)
    if registered != expected:
        return (
            f"Step 1 must register one output layer per attributable output: "
            f"registered {registered}, expected {expected}"
        )
    return None


def _step_8_violation(trace: Trace) -> str | None:
    """Return the step-8 label-map postcondition violation, if any."""

    if not trace._raw_to_final_layer_labels:
        return "Step 8 must build raw-to-final layer labels"
    if not trace._raw_to_final_op_labels:
        return "Step 8 must build raw-to-final op labels"
    return None


def _step_16_violation(trace: Trace) -> str | None:
    """Return the step-16 module-log postcondition violation, if any."""

    if trace._module_logs is None:
        return "Step 16 must build module logs"
    return None


def _step_16_5_violation(trace: Trace) -> str | None:
    """Return the step-16.5 graph-hash postcondition violation, if any."""

    if trace.graph_shape_hash is None:
        return "Step 16.5 must compute graph_shape_hash"
    return None


def _step_17_violation(trace: Trace) -> str | None:
    """Return the step-17 tracing-finished postcondition violation, if any."""

    if trace._tracing_finished is not True:
        return "Step 17 must mark tracing finished"
    return None


#: Per-step postconditions checked after the step runs, on every capture.
STEP_POSTCONDITIONS: dict[str, Callable[[StepContext], str | None]] = {
    "1": _step_1_violation,
    "8": lambda ctx: _step_8_violation(ctx.trace),
    "11": lambda ctx: _step_11_violation(ctx.trace),
    "15.5": lambda ctx: _step_15_5_violation(ctx.trace),
    "16": lambda ctx: _step_16_violation(ctx.trace),
    "16.5": lambda ctx: _step_16_5_violation(ctx.trace),
    "17": lambda ctx: _step_17_violation(ctx.trace),
}


def _check_step_postconditions(ctx: StepContext, step: str) -> None:
    """Raise if a completed step left its postcondition unmet.

    Runs on every capture, with no env gate, and raises a typed error rather
    than ``assert`` so ``python -O`` cannot strip it. A violation is a
    postprocess defect, never a user error.

    Raises
    ------
    MetadataInvariantError
        With check name and refusal code ``postprocess_step_postcondition``
        and a message naming the step.
    """

    check = STEP_POSTCONDITIONS.get(step)
    violation = None if check is None else check(ctx)
    if violation is not None:
        from ..validation.invariants import MetadataInvariantError

        raise MetadataInvariantError(
            "postprocess_step_postcondition",
            f"{violation}. Remedy: report this as a TorchLens bug.",
            code="postprocess_step_postcondition",
        )


def run_pipeline(ctx: StepContext) -> None:
    """Execute steps 1-20 in registry order, skipping steps whose gate is false.

    After each step that runs, ``_check_step_postconditions`` checks that
    step's postcondition from ``STEP_POSTCONDITIONS`` (steps 1, 8, 11, 15.5,
    16, 16.5 and 17 carry one).
    """

    for spec in STEP_REGISTRY:
        if spec.should_run(ctx):
            spec.run(ctx)
            _check_step_postconditions(ctx, spec.step)
