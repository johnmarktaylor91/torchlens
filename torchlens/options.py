"""Grouped option dataclasses for public TorchLens APIs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, TypeVar, cast

import torch

from ._deprecations import (
    MISSING,
    MissingType,
)
from ._episode_spec import EpisodeSpec
from ._errors import (
    ArgumentConflictError,
    ArgumentTypeError,
    KeywordConflictError,
)
from ._literals import (
    BufferVisibilityLiteral,
    CollapseLiteral,
    FoldRepeatsLiteral,
    OutputDeviceLiteral,
    VisDirectionLiteral,
    VisInterventionModeLiteral,
    VisModeLiteral,
    VisNodeModeLiteral,
    VisNodePlacementLiteral,
    VisRendererLiteral,
)
from ._option_receipt import (
    EchoOptions,
    OptionReceiptEntry,
    _explicit_fields,
    _field_is_explicit,
    _resolve_option_value,
    _set_frozen_fields,
    option_receipt,
)
from ._options_validation import (
    _normalize_layout,
    _validate_buffer_visibility,
    _validate_capture_values,
    _validate_collapse,
    _validate_fold_repeats,
    _validate_intervention_mode,
    _validate_node_style,
    _validate_visualization_flag_fields,
)
from ._save_budget import SaveBudgetOption
from ._vocab.node_spec import NodeSpec

if TYPE_CHECKING:
    from .data_classes.layer import Layer
    from .data_classes.module import Module

T = TypeVar("T")
ActivationPostfunc = Callable[[torch.Tensor], torch.Tensor]
GradientPostfunc = Callable[[torch.Tensor], torch.Tensor]


_CAPTURE_FIELDS: Final[tuple[str, ...]] = (
    "layers_to_save",
    "transform",
    "save_raw_input",
    "batch_render",
    "output_transform",
    "output_style",
    "output_head",
    "save_raw_output",
    "layer_visualizers",
    "save_visualizations",
    "keep_orphans",
    "output_device",
    "save_arg_values",
    "save_grads",
    "capture_tensor_grad_hooks",
    "save_code_context",
    "save_rng_states",
    "random_seed",
    "source_context_lines",
    "optimizer",
    "compute_input_output_distances",
    "detach_saved_activations",
    "recurrence_detection",
    "intervention_ready",
    "capture_container_structure",
    "hooks",
    "unwrap_when_done",
    "verbose",
    "backward_ready",
    "inference_only",
    "name",
    "cache",
    "cache_dir",
    "module_filter",
    "stop_after",
    "jax_control_flow",
    "jax_max_control_flow_unroll",
    "module_identity_mode",
    "payload_policy",
    "save_preview",
    "emit_nvtx",
    "measure_python_peak_memory",
    "save_budget",
    "distributed_witness",
    "raise_on_nan",
    "track_nonfinite",
    "track_device_memory",
    "structure_only",
    "log_injections",
)
_SAVE_FIELDS: Final[tuple[str, ...]] = (
    "activation_transform",
    "grad_transform",
    "save_raw_activations",
    "save_raw_gradients",
)
_VISUALIZATION_FIELDS: Final[tuple[str, ...]] = (
    "view",
    "depth",
    "container_path",
    "save_only",
    "file_format",
    "show_buffers",
    "direction",
    "graph_overrides",
    "node_style",
    "node_spec_fn",
    "collapsed_node_spec_fn",
    "collapse_fn",
    "collapse",
    "fold_repeats",
    "skip_fn",
    "edge_overrides",
    "grad_edge_overrides",
    "module_overrides",
    "layout",
    "renderer",
    "theme",
    "intervention_mode",
    "show_cone",
    "node_overlay",
    "node_label_fields",
    "show_legend",
    "color_by",
    "size_by",
    "scale",
    "stack_by",
    "show_redundant_args",
    "font_size",
    "dpi",
    "for_paper",
    "return_graph",
    "order_siblings",
)
_REPLAY_FIELDS: Final[tuple[str, ...]] = (
    "strict",
    "hooks",
    "differentiable",
    "append",
    "chunk_size",
)
_INTERVENTION_FIELDS: Final[tuple[str, ...]] = (
    "engine",
    "confirm_mutation",
    "strict",
)
_STREAMING_FIELDS: Final[tuple[str, ...]] = (
    "bundle_path",
    "retain_in_memory",
    "out_callback",
    "include_custom_attributes",
    "include_buffer_values",
    "async_writes",
    "max_pending_bytes",
)
_CAPTURE_FLAT_TO_GROUP: Final[dict[str, str]] = {
    "layers_to_save": "layers_to_save",
    "transform": "transform",
    "save_raw_input": "save_raw_input",
    "batch_render": "batch_render",
    "output_transform": "output_transform",
    "output_style": "output_style",
    "output_head": "output_head",
    "save_raw_output": "save_raw_output",
    "layer_visualizers": "layer_visualizers",
    "save_visualizations": "save_visualizations",
    "keep_orphans": "keep_orphans",
    "output_device": "output_device",
    "save_arg_values": "save_arg_values",
    "save_grads": "save_grads",
    "capture_tensor_grad_hooks": "capture_tensor_grad_hooks",
    "save_code_context": "save_code_context",
    "save_rng_states": "save_rng_states",
    "random_seed": "random_seed",
    "source_context_lines": "source_context_lines",
    "optimizer": "optimizer",
    "compute_input_output_distances": "compute_input_output_distances",
    "detach_saved_activations": "detach_saved_activations",
    "recurrence_detection": "recurrence_detection",
    "intervention_ready": "intervention_ready",
    "capture_container_structure": "capture_container_structure",
    "hooks": "hooks",
    "unwrap_when_done": "unwrap_when_done",
    "verbose": "verbose",
    "backward_ready": "backward_ready",
    "inference_only": "inference_only",
    "name": "name",
    "cache": "cache",
    "cache_dir": "cache_dir",
    "module_filter": "module_filter",
    "stop_after": "stop_after",
    "jax_control_flow": "jax_control_flow",
    "jax_max_control_flow_unroll": "jax_max_control_flow_unroll",
    "module_identity_mode": "module_identity_mode",
    "payload_policy": "payload_policy",
    "save_preview": "save_preview",
    "raise_on_nan": "raise_on_nan",
    "structure_only": "structure_only",
}
_SAVE_FLAT_TO_GROUP: Final[dict[str, str]] = {
    "activation_transform": "activation_transform",
    "grad_transform": "grad_transform",
    "save_raw_activations": "save_raw_activations",
    "save_raw_gradients": "save_raw_gradients",
}
_VISUALIZATION_FLAT_TO_GROUP: Final[dict[str, str]] = {
    "view": "view",
    "depth": "depth",
    "layout": "layout",
    "node_style": "node_style",
    "renderer": "renderer",
    "collapse": "collapse",
    "fold_repeats": "fold_repeats",
    "order_siblings": "order_siblings",
}
_REPLAY_FLAT_TO_GROUP: Final[dict[str, str]] = {
    "strict": "strict",
    "hooks": "hooks",
    "differentiable": "differentiable",
    "append": "append",
    "chunk_size": "chunk_size",
}
_INTERVENTION_FLAT_TO_GROUP: Final[dict[str, str]] = {
    "engine": "engine",
    "confirm_mutation": "confirm_mutation",
    "strict": "strict",
}
_STREAMING_FLAT_TO_GROUP: Final[dict[str, str]] = {
    "bundle_path": "bundle_path",
    "retain_in_memory": "retain_in_memory",
    "out_callback": "out_callback",
}


class _MutateWarningSuppression:
    """Session-level toggle and context manager for mutation warnings."""

    def __init__(self) -> None:
        """Initialize the suppression flag."""

        self._suppress = False
        # Stack of states to restore on ``__exit__``; supports nested ``with``.
        self._restore_stack: list[bool] = []
        # Pre-call state captured by ``__call__`` so ``with suppress(on):``
        # restores the state that existed *before* the call instead of leaking
        # the in-block value. ``None`` means no call-form entry is pending.
        self._pending_prior: bool | None = None

    def __call__(self, on: bool = True) -> _MutateWarningSuppression:
        """Set suppression state and return this context-capable object.

        Toggling immediately keeps the bare ``suppress_mutate_warnings(True)``
        session-level form working, while snapshotting the pre-call state lets a
        ``with suppress_mutate_warnings(on):`` restore it on exit rather than
        leaking the in-block value.

        Parameters
        ----------
        on:
            Whether mutate-in-place warnings should be suppressed.

        Returns
        -------
        _MutateWarningSuppression
            This suppression controller.
        """

        self._pending_prior = self._suppress
        self._suppress = bool(on)
        return self

    def __enter__(self) -> _MutateWarningSuppression:
        """Temporarily suppress mutate-in-place warnings.

        Returns
        -------
        _MutateWarningSuppression
            This suppression controller.
        """

        if self._pending_prior is not None:
            prior = self._pending_prior
            self._pending_prior = None
        else:
            prior = self._suppress
        self._restore_stack.append(prior)
        self._suppress = True
        return self

    def __exit__(self, *exc: object) -> None:
        """Restore the suppression state active before the context.

        Parameters
        ----------
        *exc:
            Exception triple supplied by the context manager protocol.
        """

        self._pending_prior = None
        self._suppress = self._restore_stack.pop() if self._restore_stack else False

    @property
    def is_suppressed(self) -> bool:
        """Whether mutate-in-place warnings are currently suppressed."""

        return self._suppress


suppress_mutate_warnings = _MutateWarningSuppression()


def _merge_grouped_options(
    *,
    option: Any | None,
    option_factory: Callable[[], Any],
    flat_to_group: Mapping[str, str],
    flat_values: Mapping[str, Any],
    group_name: str,
    conflict_message: str,
) -> Any:
    """Merge flat kwargs into a grouped options object (internal plumbing).

    Parameters
    ----------
    option:
        Caller-supplied grouped options, if any.
    option_factory:
        Zero-argument constructor for defaults.
    flat_to_group:
        Mapping from flat kwarg names to canonical option field names.
    flat_values:
        Flat kwarg values or ``MISSING``.
    group_name:
        Public grouped option parameter name.
    conflict_message:
        Message for same-field grouped/flat conflicts.

    Returns
    -------
    Any
        New grouped option object with merged values.

    Raises
    ------
    ValueError
        If a field is supplied by both grouped and flat styles.
    """

    default_option = option_factory()
    option_type = type(default_option)
    if option is not None and not isinstance(option, option_type):
        raise ArgumentTypeError(
            f"Grouped option {group_name!r} received {type(option).__name__}, not "
            f"{option_type.__name__}",
            code="option_group_type_invalid",
            remedy=f"pass a {option_type.__name__} instance or None as {group_name}",
            argument=group_name,
            received_type=type(option).__name__,
        )

    if option is None:
        values = default_option.as_dict()
        specified_fields: frozenset[str] = frozenset()
    else:
        values = option.as_dict()
        specified_fields = _explicit_fields(option)

    for flat_name, group_field in flat_to_group.items():
        flat_value = flat_values.get(flat_name, MISSING)
        if flat_value is MISSING:
            continue
        if option is not None and group_field in specified_fields:
            # ValueError lineage: these five merge doors historically raised
            # `raise ValueError(conflict_message)`. The TypeError-lineage
            # grouped/flat door is merge_visualization_options
            # (`option_group_keyword_conflict`), split per site history.
            raise ArgumentConflictError(
                conflict_message,
                code="option_group_conflict",
                remedy=f"pass either {group_name} or its individual keyword arguments",
                arguments=(flat_name, f"{group_name}.{group_field}"),
            )
        values[group_field] = flat_value
        specified_fields = frozenset((*specified_fields, group_field))
    return option_factory().from_values(values, specified_fields)


@dataclass(frozen=True, init=False)
class CaptureOptions:
    """Grouped capture options for ``trace``.

    Parameters
    ----------
    layers_to_save:
        Activation layer selector to capture.
    transform:
        Optional callable applied once to the user input before ``model.forward``.
    save_raw_input:
        Raw user-input save policy for portable bundles. ``"small"`` stores a
        bounded representation, ``True`` stores the full object, and ``False``
        drops it when saving.
    batch_render:
        Raw-input batch rendering policy for visualization. Supported values are
        ``"auto"``, ``"all"``, ``"first"``, ``"first_n:<N>"``, and
        ``"shape_only"``.
    output_transform:
        Optional callable applied once to the model output after ``model.forward``.
    output_style:
        Optional semantic output decode style.
    output_head:
        Optional live-output head to decode.
    save_raw_output:
        Raw model-output save policy for portable bundles. ``"small"`` stores a
        bounded representation, ``True`` stores the full object, and ``False``
        drops it when saving.
    layer_visualizers:
        Optional mapping from site selectors to per-layer thumbnail callables.
    save_visualizations:
        Whether rendered visualizer image files should be copied into portable bundles.
    keep_orphans:
        Whether island ops (computations unreachable from both the model inputs and
        outputs) are retained in raw metadata and exposed via ``trace.orphans``. Defaults
        to ``False`` (islands pruned); set ``True`` to surface them. Retained orphans do not
        enter ``layer_list``/summaries; they live only on the ``trace.orphans`` accessor.
    output_device:
        Device placement for saved tensors.
    save_arg_values:
        Whether non-tensor function arguments are captured.
    save_grads:
        Backward gradient-retention policy. ``True`` captures all gradients,
        ``False``/``None`` disables capture, and selectors restrict retention:
        label strings/ordinal lists, ``tl.*`` selectors, and bare callables
        are all honored (never collapsed to "all"). A bare callable is
        evaluated once per FINALIZED op (a layer-like ctx, post-postprocess,
        before any backward) and must return a strict ``bool``; only matching
        ops receive gradient hooks.
    save_code_context:
        Whether source-text context is captured in addition to source identity.
    save_rng_states:
        Whether operation-level RNG states are captured.
    random_seed:
        Fixed seed used for deterministic capture. PROCESS-GLOBAL side
        effect: every capture reseeds all four global RNG engines (Python
        ``random``, NumPy, torch CPU, and every CUDA device) with this seed
        at entry and does NOT restore their prior states afterwards, so a
        pipeline that seeds, captures, then samples draws different numbers
        than the same pipeline without the capture. When ``None`` (the
        default) the seed itself is drawn from the entropy-seeded global
        ``random`` stream, so ``torch.manual_seed(k)`` before an unseeded
        capture does NOT make the capture reproducible — pass ``random_seed=``
        explicitly for that. The seed used is recorded on
        ``trace.random_seed``.
    source_context_lines:
        Number of source-context lines to store.
    optimizer:
        Optional optimizer used to annotate optimized parameters.
    compute_input_output_distances:
        Whether input/output graph distances are computed.
    detach_saved_activations:
        Whether saved tensors are detached from autograd. The default
        ``False`` keeps saved activations graph-connected, which retains the
        captured autograd graph alongside the payloads: measured ~1.48x the
        saved-payload bytes live on a resnet18 capture, vs ~1.01x with
        ``True``. Pass ``True`` for forward-only analysis when that
        multiplier matters (R33).
    recurrence_detection:
        Whether repeated graph patterns are detected during postprocess.
    intervention_ready:
        Whether replay-template metadata is captured for intervention APIs.
    capture_container_structure:
        Whether to persist input and output container structure without
        enabling intervention replay metadata.
    hooks:
        Optional live hook plan applied during capture.
    unwrap_when_done:
        Whether Torch functions are unwrapped after this call.
    verbose:
        Whether progress messages are printed.
    backward_ready:
        Whether capture keeps autograd-connected tensors for training workflows.
    inference_only:
        Whether capture wraps the user forward in ``torch.no_grad()``.
    name:
        Optional user-facing name for the returned log.
    cache:
        Whether to use the content-hash capture cache. The key covers model
        tensor content (including per-tensor device and ``requires_grad``),
        training flags, non-persistent buffers, module tree and ``forward``
        code, plain instance attributes (bounded digest), user-registered
        module hooks, inputs, and the capture configuration; closure cells,
        globals referenced by ``forward``, and the interior state of opaque
        attribute objects remain outside the key (documented boundary).
        ``torchlens.clear_capture_cache()`` empties the cache.
    cache_dir:
        Optional directory for content-hash cache entries.
    module_filter:
        Optional THIRD save gate composed (AND) with ``save=`` /
        ``layers_to_save``: an op's payload is retained only when the
        save selection picks it AND this predicate returns truthy. The
        predicate receives an op-record namespace (the legacy-shaped
        ``SimpleNamespace`` of captured op fields such as ``func_name``,
        ``layer_label``, and ``modules``), NEVER an
        ``nn.Module`` instance — a filter written against modules (e.g.
        ``lambda m: isinstance(m, nn.Linear)``) matches nothing and saves
        ZERO payloads. Returning ``False`` keeps metadata but skips payload
        saving; a capture whose every selected payload was suppressed by
        this gate emits a ``module_filter_zero_saved`` warning.
    stop_after:
        Inclusive stop-early site for torch captures (DOCUMENTED-UNSTABLE
        spelling): capture halts immediately AFTER the named site is
        captured, returning a partial halted trace that includes it. The
        site compiles into the halt engine on emission identity: a string
        halts at the first emission whose module address
        (``"encoder.layer.4"``, at that module's exit boundary) or function
        name (``"relu"``) matches, a live selector (``tl.func("relu")``,
        ``tl.module(...)``) halts at its first match, and a callable
        predicate halts when it returns True. Finalized postprocess labels
        (``"relu_1_2"``) do not exist during capture and refuse typed
        (``stop_after_site_not_live``). A selector-shaped site that never
        fires refuses typed (``stop_after_never_fired``); a callable that
        never fires warns. Cannot combine with ``halt=``
        (``stop_after_halt_conflict``) or ``chunk_size``. Torch-only.
    jax_control_flow:
        Declared JAX control-flow policy. Backends other than torch reject
        explicit use until their implementation phase supports it.
    jax_max_control_flow_unroll:
        Declared maximum JAX control-flow unroll count.
    module_identity_mode:
        Declared backend module-mode selection passthrough.
    payload_policy:
        Declared payload materialization/codec policy passthrough.
    save_preview:
        Non-torch preview backends' declared flag reserving extended ``save=``
        semantics; the shipped torch ``save=`` kwarg is independent of it.
    emit_nvtx:
        Whether torch capture emits NVIDIA Tools Extension (NVTX) CUDA profiling
        ranges around each logged operation. Range names carry call identity
        (``torchlens::<op>#<call_id>``) and TorchLens's own bookkeeping calls
        are separated under ``torchlens::internal::`` so the Nsight timeline
        shows model work, not capture plumbing. NVTX markers are visible in
        NVIDIA Nsight Systems and Nsight Compute timelines. The default is ``False``;
        enabling it can add small per-op overhead. The option applies to full
        ``tl.trace`` capture and is preserved through ``tl.record``-style
        options, although sparse recording may only expose ranges for operations
        it actually logs.
    measure_python_peak_memory:
        Whether the host-side forward-pass peak recorded in
        ``Trace.forward_peak_memory`` additionally includes a ``tracemalloc``
        Python-allocation peak. The default is ``False``: the CPU/MPS
        measurement is then the cheap host resident-set-size (or MPS allocator)
        delta alone, which reads ``0`` for models too small to move that
        coarse-grained counter. Enabling it installs CPython's allocator hook
        for the duration of the forward pass, which is precise for small models
        but taxes every traced operation (measured at 1.7x-2.5x total capture
        time on torchvision CNNs and ViTs), so it is opt-in. CUDA captures
        report the true device peak and ignore this option.
    distributed_witness:
        Session-time witness level for collective boundary records captured
        under the distributed opt-in. ``"none"`` (the default) records
        structure and correlation only; ``"digest"`` additionally stores
        byte-exact SHA-256 digests of each contribution at issue and each
        destination at observed completion (redundant evidence that can only
        DEMOTE a merge verdict, never rescue one -- and a synchronization cost
        on accelerator captures). ``"payload"`` is reserved for the merge
        artifact story and currently refuses. Like
        ``measure_python_peak_memory`` this is a session-time knob: it changes
        what capture pays for, not what a trace means, and load restores the
        default. The per-boundary ``witness.policy_resolved`` field IS
        portable evidence of what was captured.
    save_budget:
        Ceiling on the bytes of activation payload a single capture may retain,
        enforced per device. ``"auto"`` (the default)
        allows half of each device's *available* memory measured at that device's
        first save; a float in ``(0, 1]`` sets a different fraction; an int sets
        an absolute per-device byte cap; ``None`` disables budgeting. Crossing the
        budget stops capture with
        :class:`torchlens.errors.SaveBudgetExceededError`. The primary retained copy
        is admitted before allocation from source-tensor bytes and retained storage
        is alias-aware. This is not a general OOM guarantee: the model forward,
        transform-only deltas, and cross-device temporaries can allocate before they
        are knowable. Predicate-selected disk-only saves are exempt, while exhaustive
        ``save="all"`` plus disk streaming remains budgeted until postprocess eviction.
        Devices whose headroom cannot be measured are left unbudgeted and
        emit a ``UserWarning`` on their first non-empty charge; use an absolute byte
        cap to enforce those devices.
    raise_on_nan:
        Whether capture should stop at the first NaN or Inf tensor.
    track_nonfinite:
        Whether torch capture records a per-op finiteness verdict
        (DOCUMENTED-UNSTABLE), served by ``Trace.nonfinite_ops`` /
        ``Trace.nonfinite_coverage``; covers unsaved ops, never changes
        control flow, reads device flags in one post-forward batch. A
        session-time knob. Doc: ``docs/reference/capture_outcomes.md``.
    track_device_memory:
        Whether torch capture samples device allocator counters around each op call, served by
        ``torchlens.observe.device_memory_samples`` (DOCUMENTED-UNSTABLE; a session-time knob):
        opt-in, never resetting; allocated/reserved before/after, signed deltas, high-water ADVANCE
        per initialized CUDA device; unsupported devices (CPU/MPS) get typed absence, never zeros.
    structure_only:
        Whether this capture runs under the structure-only contract
        (DOCUMENTED-UNSTABLE surface, pending naming-session/S2 ratification;
        no deprecation shim owed on rename). Structure-only capture records
        the op graph, module hierarchy, parameter geometry, and per-op
        shape/dtype as HYPOTHESES while every value-bearing claim is refused
        typed or gated; value-dependent branches refuse with the user's
        source line. Torch-only; the capability contract lives in
        ``docs/reference/structure_only_capabilities.md``.
    log_injections:
        Whether computation inside intervention hooks is recorded as anchored
        injected-op records served by ``Trace.injected_ops`` / ``Trace.model_ops``
        (F01 stage 0-1, DOCUMENTED-UNSTABLE; stage-1 saves refuse typed -- F44).

    Examples
    --------
    >>> opts = CaptureOptions(layers_to_save=["fc1"], random_seed=0)
    >>> opts.layers_to_save
    ['fc1']
    """

    layers_to_save: str | list[Any] | None = "all"
    transform: Callable[[Any], Any] | None = None
    save_raw_input: str | bool = "small"
    batch_render: str = "auto"
    output_transform: Callable[[Any], Any] | None = None
    output_style: str | None = None
    output_head: str | None = None
    save_raw_output: str | bool = "small"
    layer_visualizers: Mapping[Any, Callable[..., Any]] | None = None
    save_visualizations: bool = False
    keep_orphans: bool = False
    output_device: OutputDeviceLiteral = "same"
    save_arg_values: bool = False
    save_grads: bool | str | list[Any] | Callable[[Any], Any] | None = None
    capture_tensor_grad_hooks: bool = True
    save_code_context: bool = False
    save_rng_states: bool = False
    random_seed: int | None = None
    source_context_lines: int = 7
    optimizer: Any = None
    compute_input_output_distances: bool = True
    detach_saved_activations: bool = False
    recurrence_detection: bool = True
    intervention_ready: bool = False
    capture_container_structure: bool = False
    hooks: Any | None = None
    unwrap_when_done: bool = False
    verbose: bool = False
    backward_ready: bool = False
    inference_only: bool = False
    name: str | None = None
    cache: bool = False
    cache_dir: str | Path | None = None
    module_filter: Callable[[Any], bool] | None = None
    stop_after: Any | None = None
    jax_control_flow: Literal["reject", "unroll", "region"] = "unroll"
    jax_max_control_flow_unroll: int = 64
    module_identity_mode: str | None = None
    payload_policy: str | None = None
    save_preview: bool = False
    emit_nvtx: bool = False
    measure_python_peak_memory: bool = False
    save_budget: SaveBudgetOption = "auto"
    distributed_witness: str = "none"
    raise_on_nan: bool = False
    track_nonfinite: bool = False
    track_device_memory: bool = False
    structure_only: bool = False
    log_injections: bool = False
    _specified_fields: frozenset[str] = field(default_factory=frozenset, init=False, repr=False)

    def __init__(
        self,
        layers_to_save: str | list[Any] | None | MissingType = MISSING,
        transform: Callable[[Any], Any] | None | MissingType = MISSING,
        save_raw_input: str | bool | MissingType = MISSING,
        batch_render: str | MissingType = MISSING,
        output_transform: Callable[[Any], Any] | None | MissingType = MISSING,
        output_style: str | None | MissingType = MISSING,
        output_head: str | None | MissingType = MISSING,
        save_raw_output: str | bool | MissingType = MISSING,
        layer_visualizers: Mapping[Any, Callable[..., Any]] | None | MissingType = MISSING,
        save_visualizations: bool | MissingType = MISSING,
        keep_orphans: bool | MissingType = MISSING,
        output_device: OutputDeviceLiteral | MissingType = MISSING,
        save_arg_values: bool | MissingType = MISSING,
        save_grads: bool | str | list[Any] | Callable[[Any], Any] | None | MissingType = MISSING,
        capture_tensor_grad_hooks: bool | MissingType = MISSING,
        save_code_context: bool | MissingType = MISSING,
        save_rng_states: bool | MissingType = MISSING,
        random_seed: int | None | MissingType = MISSING,
        source_context_lines: int | MissingType = MISSING,
        optimizer: Any | MissingType = MISSING,
        compute_input_output_distances: bool | MissingType = MISSING,
        detach_saved_activations: bool | MissingType = MISSING,
        recurrence_detection: bool | MissingType = MISSING,
        intervention_ready: bool | MissingType = MISSING,
        capture_container_structure: bool | MissingType = MISSING,
        hooks: Any | MissingType = MISSING,
        unwrap_when_done: bool | MissingType = MISSING,
        verbose: bool | MissingType = MISSING,
        backward_ready: bool | MissingType = MISSING,
        inference_only: bool | MissingType = MISSING,
        name: str | None | MissingType = MISSING,
        cache: bool | MissingType = MISSING,
        cache_dir: str | Path | None | MissingType = MISSING,
        module_filter: Callable[[Any], bool] | None | MissingType = MISSING,
        stop_after: Any | None | MissingType = MISSING,
        jax_control_flow: Literal["reject", "unroll", "region"] | MissingType = MISSING,
        jax_max_control_flow_unroll: int | MissingType = MISSING,
        module_identity_mode: str | None | MissingType = MISSING,
        payload_policy: str | None | MissingType = MISSING,
        save_preview: bool | MissingType = MISSING,
        emit_nvtx: bool | MissingType = MISSING,
        measure_python_peak_memory: bool | MissingType = MISSING,
        save_budget: SaveBudgetOption | MissingType = MISSING,
        distributed_witness: str | MissingType = MISSING,
        raise_on_nan: bool | MissingType = MISSING,
        *,
        track_nonfinite: bool | MissingType = MISSING,
        track_device_memory: bool | MissingType = MISSING,
        structure_only: bool | MissingType = MISSING,
        log_injections: bool | MissingType = MISSING,
    ) -> None:
        """Initialize a frozen capture option bundle."""

        specified_fields: set[str] = set()
        values: dict[str, Any] = {
            "layers_to_save": _resolve_option_value(
                "layers_to_save", layers_to_save, "all", specified_fields
            ),
            "transform": _resolve_option_value("transform", transform, None, specified_fields),
            "save_raw_input": _resolve_option_value(
                "save_raw_input", save_raw_input, "small", specified_fields
            ),
            "batch_render": _resolve_option_value(
                "batch_render", batch_render, "auto", specified_fields
            ),
            "output_transform": _resolve_option_value(
                "output_transform", output_transform, None, specified_fields
            ),
            "output_style": _resolve_option_value(
                "output_style", output_style, None, specified_fields
            ),
            "output_head": _resolve_option_value(
                "output_head", output_head, None, specified_fields
            ),
            "save_raw_output": _resolve_option_value(
                "save_raw_output", save_raw_output, "small", specified_fields
            ),
            "layer_visualizers": _resolve_option_value(
                "layer_visualizers", layer_visualizers, None, specified_fields
            ),
            "save_visualizations": _resolve_option_value(
                "save_visualizations", save_visualizations, False, specified_fields
            ),
            "keep_orphans": _resolve_option_value(
                "keep_orphans", keep_orphans, False, specified_fields
            ),
            "output_device": _resolve_option_value(
                "output_device", output_device, "same", specified_fields
            ),
            "save_arg_values": _resolve_option_value(
                "save_arg_values", save_arg_values, False, specified_fields
            ),
            "save_grads": _resolve_option_value("save_grads", save_grads, None, specified_fields),
            "capture_tensor_grad_hooks": _resolve_option_value(
                "capture_tensor_grad_hooks",
                capture_tensor_grad_hooks,
                True,
                specified_fields,
            ),
            "save_code_context": _resolve_option_value(
                "save_code_context", save_code_context, False, specified_fields
            ),
            "save_rng_states": _resolve_option_value(
                "save_rng_states", save_rng_states, False, specified_fields
            ),
            "random_seed": _resolve_option_value(
                "random_seed", random_seed, None, specified_fields
            ),
            "source_context_lines": _resolve_option_value(
                "source_context_lines", source_context_lines, 7, specified_fields
            ),
            "optimizer": _resolve_option_value("optimizer", optimizer, None, specified_fields),
            "compute_input_output_distances": _resolve_option_value(
                "compute_input_output_distances",
                compute_input_output_distances,
                True,
                specified_fields,
            ),
            "detach_saved_activations": _resolve_option_value(
                "detach_saved_activations", detach_saved_activations, False, specified_fields
            ),
            "recurrence_detection": _resolve_option_value(
                "recurrence_detection", recurrence_detection, True, specified_fields
            ),
            "intervention_ready": _resolve_option_value(
                "intervention_ready", intervention_ready, False, specified_fields
            ),
            "capture_container_structure": _resolve_option_value(
                "capture_container_structure",
                capture_container_structure,
                False,
                specified_fields,
            ),
            "hooks": _resolve_option_value("hooks", hooks, None, specified_fields),
            "unwrap_when_done": _resolve_option_value(
                "unwrap_when_done", unwrap_when_done, False, specified_fields
            ),
            "verbose": _resolve_option_value("verbose", verbose, False, specified_fields),
            "backward_ready": _resolve_option_value(
                "backward_ready", backward_ready, False, specified_fields
            ),
            "inference_only": _resolve_option_value(
                "inference_only", inference_only, False, specified_fields
            ),
            "name": _resolve_option_value("name", name, None, specified_fields),
            "cache": _resolve_option_value("cache", cache, False, specified_fields),
            "cache_dir": _resolve_option_value("cache_dir", cache_dir, None, specified_fields),
            "module_filter": _resolve_option_value(
                "module_filter", module_filter, None, specified_fields
            ),
            "stop_after": _resolve_option_value("stop_after", stop_after, None, specified_fields),
            "jax_control_flow": _resolve_option_value(
                "jax_control_flow", jax_control_flow, "unroll", specified_fields
            ),
            "jax_max_control_flow_unroll": _resolve_option_value(
                "jax_max_control_flow_unroll",
                jax_max_control_flow_unroll,
                64,
                specified_fields,
            ),
            "module_identity_mode": _resolve_option_value(
                "module_identity_mode", module_identity_mode, None, specified_fields
            ),
            "payload_policy": _resolve_option_value(
                "payload_policy", payload_policy, None, specified_fields
            ),
            "save_preview": _resolve_option_value(
                "save_preview", save_preview, False, specified_fields
            ),
            "emit_nvtx": _resolve_option_value("emit_nvtx", emit_nvtx, False, specified_fields),
            "measure_python_peak_memory": _resolve_option_value(
                "measure_python_peak_memory",
                measure_python_peak_memory,
                False,
                specified_fields,
            ),
            "save_budget": _resolve_option_value(
                "save_budget",
                save_budget,
                "auto",
                specified_fields,
            ),
            "distributed_witness": _resolve_option_value(
                "distributed_witness",
                distributed_witness,
                "none",
                specified_fields,
            ),
            "raise_on_nan": _resolve_option_value(
                "raise_on_nan", raise_on_nan, False, specified_fields
            ),
            "track_nonfinite": _resolve_option_value(
                "track_nonfinite", track_nonfinite, False, specified_fields
            ),
            "track_device_memory": _resolve_option_value(
                "track_device_memory", track_device_memory, False, specified_fields
            ),
            "structure_only": _resolve_option_value(
                "structure_only", structure_only, False, specified_fields
            ),
            "log_injections": _resolve_option_value(
                "log_injections", log_injections, False, specified_fields
            ),
        }
        _validate_capture_values(values)
        _set_frozen_fields(self, _CAPTURE_FIELDS, values)
        object.__setattr__(self, "_specified_fields", frozenset(specified_fields))

    def as_dict(self) -> dict[str, Any]:
        """Return the option values as a plain dictionary."""

        return {field_name: getattr(self, field_name) for field_name in _CAPTURE_FIELDS}

    def is_field_explicit(self, field_name: str) -> bool:
        """Return whether a field was explicitly supplied by the caller."""

        return _field_is_explicit(self, field_name)

    @classmethod
    def from_values(
        cls, values: Mapping[str, Any], specified_fields: frozenset[str]
    ) -> CaptureOptions:
        """Build an instance from already-resolved field values.

        Applies the same invariants as ``__init__`` (via
        :func:`_validate_capture_values`) so this construction path -- used by
        the flat-kwarg merge -- cannot accept values the grouped constructor
        rejects, matching the sibling :meth:`VisualizationOptions.from_values`.
        """

        _validate_capture_values(values)
        instance = object.__new__(cls)
        _set_frozen_fields(instance, _CAPTURE_FIELDS, values)
        object.__setattr__(instance, "_specified_fields", specified_fields)
        return instance


@dataclass(frozen=True, init=False)
class SaveOptions:
    """Grouped out-save options for ``trace``.

    Parameters
    ----------
    activation_transform:
        Optional transform applied to each out before storage.
    grad_transform:
        Optional transform applied to each grad before storage.
    save_raw_activations:
        Whether raw outs remain available when transformed.
    save_raw_gradients:
        Whether raw grads remain available when transformed.

    Examples
    --------
    >>> opts = SaveOptions(activation_transform=lambda x: x.detach())
    >>> opts.save_raw_activations
    True
    """

    activation_transform: ActivationPostfunc | None = None
    grad_transform: GradientPostfunc | None = None
    save_raw_activations: bool = True
    save_raw_gradients: bool = True
    _specified_fields: frozenset[str] = field(default_factory=frozenset, init=False, repr=False)

    def __init__(
        self,
        activation_transform: ActivationPostfunc | None | MissingType = MISSING,
        grad_transform: GradientPostfunc | None | MissingType = MISSING,
        save_raw_activations: bool | MissingType = MISSING,
        save_raw_gradients: bool | MissingType = MISSING,
    ) -> None:
        """Initialize a frozen save option bundle."""

        specified_fields: set[str] = set()
        values: dict[str, Any] = {
            "activation_transform": _resolve_option_value(
                "activation_transform", activation_transform, None, specified_fields
            ),
            "grad_transform": _resolve_option_value(
                "grad_transform", grad_transform, None, specified_fields
            ),
            "save_raw_activations": _resolve_option_value(
                "save_raw_activations", save_raw_activations, True, specified_fields
            ),
            "save_raw_gradients": _resolve_option_value(
                "save_raw_gradients", save_raw_gradients, True, specified_fields
            ),
        }
        _set_frozen_fields(self, _SAVE_FIELDS, values)
        object.__setattr__(self, "_specified_fields", frozenset(specified_fields))

    def as_dict(self) -> dict[str, Any]:
        """Return the option values as a plain dictionary."""

        return {field_name: getattr(self, field_name) for field_name in _SAVE_FIELDS}

    def is_field_explicit(self, field_name: str) -> bool:
        """Return whether a field was explicitly supplied by the caller."""

        return _field_is_explicit(self, field_name)

    @classmethod
    def from_values(
        cls, values: Mapping[str, Any], specified_fields: frozenset[str]
    ) -> SaveOptions:
        """Build an instance from already-resolved field values."""

        instance = object.__new__(cls)
        _set_frozen_fields(instance, _SAVE_FIELDS, values)
        object.__setattr__(instance, "_specified_fields", specified_fields)
        return instance


#: Alias properties removed by the 2026-10-01 shim removal (no alias kept).
_REMOVED_VISUALIZATION_OPTION_MEMBERS: dict[str, str] = {
    "mode": "use VisualizationOptions.view -- the mode alias was removed",
    "max_module_depth": "use VisualizationOptions.depth -- the max_module_depth alias was removed",
    "layout_engine": "use VisualizationOptions.layout -- the layout_engine alias was removed",
    "node_mode": "use VisualizationOptions.node_style -- the node_mode alias was removed",
}


@dataclass(frozen=True, init=False)
class VisualizationOptions:
    """Grouped visualization options for capture and graph-rendering APIs.

    Parameters
    ----------
    view:
        Graph view: ``"none"``, ``"rolled"``, or ``"unrolled"``.
    depth:
        Maximum module nesting depth shown in the graph.
    container_path:
        Output path stem for rendered graph files.
    save_only:
        Whether rendering saves without opening a viewer.
    file_format:
        Graph output file format.
    show_buffers:
        Buffer visibility policy.
    direction:
        Graph layout direction.
    graph_overrides:
        Graphviz graph-level style overrides.
    node_style:
        Built-in node label/style preset.
    node_spec_fn:
        Optional layer-node customization callback.
    collapsed_node_spec_fn:
        Optional collapsed module-node customization callback.
    collapse_fn:
        Optional module collapse predicate.
    collapse:
        Smart module-collapse mode: ``"none"``, ``"auto"``, ``"max"``, or a
        float in ``[0.0, 1.0]`` on the public monotone schedule.
    fold_repeats:
        Repeat-fold policy. ``None`` preserves the collapse mode default,
        ``True`` folds every eligible run, and ``False`` disables run folding.
    skip_fn:
        Optional layer skip predicate.
    edge_overrides:
        Forward-edge style overrides.
    grad_edge_overrides:
        Gradient-edge style overrides.
    module_overrides:
        Module cluster style overrides.
    layout:
        Layout engine selector: ``"auto"``, ``"dot"``, or ``"rank"``.
    renderer:
        Renderer backend selector.
    theme:
        Theme name.
    intervention_mode:
        Intervention overlay mode.
    show_cone:
        Whether intervention cones are highlighted.
    node_overlay:
        Built-in overlay name, an external label->value score mapping, or a callable
        invoked as ``fn(node)`` to compute a per-node overlay value.
    node_label_fields:
        Optional explicit label row fields.
    show_legend:
        Tri-state legend visibility (``None`` = auto: legend only when an
        encoding channel is active); see ``Trace.draw``.
    color_by:
        UNSTABLE encoding-channel value source; see ``Trace.draw``.
    size_by:
        UNSTABLE size-channel value source (field, ``"dims"``, or callable);
        see ``Trace.draw``.
    scale:
        UNSTABLE size-channel scale transform (``"sqrt"``/``"linear"``);
        see ``Trace.draw``.
    stack_by:
        UNSTABLE rank-channel annotation source (``True``/``"auto"``,
        field, or callable); see ``Trace.draw``.
    show_redundant_args:
        UNSTABLE: show constructor args the checked-suppression equality
        check proved redundant (default ``False``); see ``Trace.draw``.
    font_size:
        Optional Graphviz font size.
    dpi:
        Optional Graphviz output DPI.
    for_paper:
        Convenience toggle forcing the paper theme preset.
    return_graph:
        Whether rendering returns the renderer object instead of DOT source.
    order_siblings:
        Whether Graphviz ``dot`` renders should verify and apply execution-order
        placement for true parallel siblings.

    Examples
    --------
    >>> opts = VisualizationOptions(view="rolled", depth=2)
    >>> opts.layout
    'auto'
    """

    view: VisModeLiteral = "none"
    depth: int = 1000
    container_path: str = "graph.gv"
    save_only: bool = False
    file_format: str = "pdf"
    show_buffers: BufferVisibilityLiteral = "meaningful"
    direction: VisDirectionLiteral = "bottomup"
    graph_overrides: dict[str, Any] | None = None
    node_style: VisNodeModeLiteral = "default"
    node_spec_fn: Callable[[Layer, NodeSpec], NodeSpec | None] | None = None
    collapsed_node_spec_fn: Callable[[Module, NodeSpec], NodeSpec | None] | None = None
    collapse_fn: Callable[[Module], bool] | None = None
    collapse: CollapseLiteral = "none"
    fold_repeats: FoldRepeatsLiteral = None
    skip_fn: Callable[[Layer], bool] | None = None
    edge_overrides: dict[str, Any] | None = None
    grad_edge_overrides: dict[str, Any] | None = None
    module_overrides: dict[str, Any] | None = None
    layout: VisNodePlacementLiteral = "auto"
    renderer: VisRendererLiteral = "graphviz"
    theme: str = "torchlens"
    intervention_mode: VisInterventionModeLiteral = "node_mark"
    show_cone: bool = True
    node_overlay: str | Mapping[str, Any] | Callable[[Any], Any] | None = None
    node_label_fields: list[str] | None = None
    show_legend: bool | None = None
    color_by: str | Callable[[Any], Any] | None = None
    size_by: str | Callable[[Any], Any] | None = None
    scale: str | None = None
    stack_by: str | bool | Callable[[Any], Any] | None = None
    show_redundant_args: bool = False
    font_size: int | None = None
    dpi: int | None = None
    for_paper: bool = False
    return_graph: bool = False
    order_siblings: bool = True
    _specified_fields: frozenset[str] = field(default_factory=frozenset, init=False, repr=False)

    def __init__(
        self,
        view: VisModeLiteral | MissingType = MISSING,
        depth: int | MissingType = MISSING,
        container_path: str | MissingType = MISSING,
        save_only: bool | MissingType = MISSING,
        file_format: str | MissingType = MISSING,
        show_buffers: BufferVisibilityLiteral | bool | MissingType = MISSING,
        direction: VisDirectionLiteral | MissingType = MISSING,
        graph_overrides: dict[str, Any] | None | MissingType = MISSING,
        node_style: VisNodeModeLiteral | MissingType = MISSING,
        node_spec_fn: (Callable[[Layer, NodeSpec], NodeSpec | None] | None | MissingType) = MISSING,
        collapsed_node_spec_fn: (
            Callable[[Module, NodeSpec], NodeSpec | None] | None | MissingType
        ) = MISSING,
        collapse_fn: Callable[[Module], bool] | None | MissingType = MISSING,
        collapse: CollapseLiteral | MissingType = MISSING,
        fold_repeats: FoldRepeatsLiteral | MissingType = MISSING,
        skip_fn: Callable[[Layer], bool] | None | MissingType = MISSING,
        edge_overrides: dict[str, Any] | None | MissingType = MISSING,
        grad_edge_overrides: dict[str, Any] | None | MissingType = MISSING,
        module_overrides: dict[str, Any] | None | MissingType = MISSING,
        layout: VisNodePlacementLiteral | MissingType = MISSING,
        renderer: VisRendererLiteral | MissingType = MISSING,
        theme: str | MissingType = MISSING,
        intervention_mode: VisInterventionModeLiteral | MissingType = MISSING,
        show_cone: bool | MissingType = MISSING,
        node_overlay: str | Mapping[str, Any] | Callable[[Any], Any] | None | MissingType = MISSING,
        node_label_fields: list[str] | None | MissingType = MISSING,
        show_legend: bool | None | MissingType = MISSING,
        color_by: str | Callable[[Any], Any] | None | MissingType = MISSING,
        size_by: str | Callable[[Any], Any] | None | MissingType = MISSING,
        scale: str | None | MissingType = MISSING,
        stack_by: str | bool | Callable[[Any], Any] | None | MissingType = MISSING,
        show_redundant_args: bool | MissingType = MISSING,
        font_size: int | None | MissingType = MISSING,
        dpi: int | None | MissingType = MISSING,
        for_paper: bool | MissingType = MISSING,
        return_graph: bool | MissingType = MISSING,
        order_siblings: bool | MissingType = MISSING,
    ) -> None:
        """Initialize a frozen visualization option bundle."""

        specified_fields: set[str] = set()
        values: dict[str, Any] = {
            "view": _resolve_option_value("view", view, "none", specified_fields),
            "depth": _resolve_option_value("depth", depth, 1000, specified_fields),
            "container_path": _resolve_option_value(
                "container_path", container_path, "graph.gv", specified_fields
            ),
            "save_only": _resolve_option_value("save_only", save_only, False, specified_fields),
            "file_format": _resolve_option_value(
                "file_format", file_format, "pdf", specified_fields
            ),
            "show_buffers": _resolve_option_value(
                "show_buffers", show_buffers, "meaningful", specified_fields
            ),
            "direction": _resolve_option_value(
                "direction", direction, "bottomup", specified_fields
            ),
            "graph_overrides": _resolve_option_value(
                "graph_overrides", graph_overrides, None, specified_fields
            ),
            "node_style": _resolve_option_value(
                "node_style", node_style, "default", specified_fields
            ),
            "node_spec_fn": _resolve_option_value(
                "node_spec_fn", node_spec_fn, None, specified_fields
            ),
            "collapsed_node_spec_fn": _resolve_option_value(
                "collapsed_node_spec_fn", collapsed_node_spec_fn, None, specified_fields
            ),
            "collapse_fn": _resolve_option_value(
                "collapse_fn", collapse_fn, None, specified_fields
            ),
            "collapse": _resolve_option_value("collapse", collapse, "none", specified_fields),
            "fold_repeats": _resolve_option_value(
                "fold_repeats", fold_repeats, None, specified_fields
            ),
            "skip_fn": _resolve_option_value("skip_fn", skip_fn, None, specified_fields),
            "edge_overrides": _resolve_option_value(
                "edge_overrides", edge_overrides, None, specified_fields
            ),
            "grad_edge_overrides": _resolve_option_value(
                "grad_edge_overrides", grad_edge_overrides, None, specified_fields
            ),
            "module_overrides": _resolve_option_value(
                "module_overrides", module_overrides, None, specified_fields
            ),
            "layout": _normalize_layout(
                _resolve_option_value("layout", layout, "auto", specified_fields)
            ),
            "renderer": _resolve_option_value("renderer", renderer, "graphviz", specified_fields),
            "theme": _resolve_option_value("theme", theme, "torchlens", specified_fields),
            "intervention_mode": _resolve_option_value(
                "intervention_mode", intervention_mode, "node_mark", specified_fields
            ),
            "show_cone": _resolve_option_value("show_cone", show_cone, True, specified_fields),
            "node_overlay": _resolve_option_value(
                "node_overlay", node_overlay, None, specified_fields
            ),
            "node_label_fields": _resolve_option_value(
                "node_label_fields", node_label_fields, None, specified_fields
            ),
            "show_legend": _resolve_option_value(
                "show_legend", show_legend, None, specified_fields
            ),
            "color_by": _resolve_option_value("color_by", color_by, None, specified_fields),
            "size_by": _resolve_option_value("size_by", size_by, None, specified_fields),
            "scale": _resolve_option_value("scale", scale, None, specified_fields),
            "stack_by": _resolve_option_value("stack_by", stack_by, None, specified_fields),
            "show_redundant_args": _resolve_option_value(
                "show_redundant_args", show_redundant_args, False, specified_fields
            ),
            "font_size": _resolve_option_value("font_size", font_size, None, specified_fields),
            "dpi": _resolve_option_value("dpi", dpi, None, specified_fields),
            "for_paper": _resolve_option_value("for_paper", for_paper, False, specified_fields),
            "return_graph": _resolve_option_value(
                "return_graph", return_graph, False, specified_fields
            ),
            "order_siblings": _resolve_option_value(
                "order_siblings", order_siblings, True, specified_fields
            ),
        }
        _validate_buffer_visibility(values["show_buffers"])
        _validate_node_style(cast(VisNodeModeLiteral, values["node_style"]))
        _validate_intervention_mode(cast(VisInterventionModeLiteral, values["intervention_mode"]))
        _validate_collapse(cast(CollapseLiteral, values["collapse"]))
        _validate_fold_repeats(cast(FoldRepeatsLiteral, values["fold_repeats"]))
        _validate_visualization_flag_fields(values)
        _set_frozen_fields(self, _VISUALIZATION_FIELDS, values)
        object.__setattr__(self, "_specified_fields", frozenset(specified_fields))

    # Runtime only: under TYPE_CHECKING the hook would make every attribute
    # type-check as Any, hiding typos and the removed spellings from mypy.
    if not TYPE_CHECKING:

        def __getattr__(self, name: str) -> Any:
            """Name the replacement for a removed public member, else fail as usual."""

            from .utils.facade import refuse_removed_member

            refuse_removed_member(
                "VisualizationOptions", name, _REMOVED_VISUALIZATION_OPTION_MEMBERS
            )
            return object.__getattribute__(self, name)

    def as_dict(self) -> dict[str, Any]:
        """Return the option values as a plain dictionary."""

        return {field_name: getattr(self, field_name) for field_name in _VISUALIZATION_FIELDS}

    def is_field_explicit(self, field_name: str) -> bool:
        """Return whether a field was explicitly supplied by the caller."""

        return _field_is_explicit(self, field_name)

    @classmethod
    def from_values(
        cls,
        values: Mapping[str, Any],
        specified_fields: frozenset[str],
    ) -> VisualizationOptions:
        """Build an instance from already-resolved field values."""

        instance = object.__new__(cls)
        values = dict(values)
        values["layout"] = _normalize_layout(values["layout"])
        _validate_node_style(cast(VisNodeModeLiteral, values["node_style"]))
        _validate_intervention_mode(cast(VisInterventionModeLiteral, values["intervention_mode"]))
        _validate_buffer_visibility(values["show_buffers"])
        _validate_collapse(cast(CollapseLiteral, values["collapse"]))
        _validate_fold_repeats(cast(FoldRepeatsLiteral, values["fold_repeats"]))
        _validate_visualization_flag_fields(values)
        _set_frozen_fields(instance, _VISUALIZATION_FIELDS, values)
        object.__setattr__(instance, "_specified_fields", specified_fields)
        return instance


@dataclass(frozen=True, init=False)
class ReplayOptions:
    """Grouped replay/rerun options for ``Trace`` propagation APIs.

    Parameters
    ----------
    strict:
        Whether divergence warnings should raise.
    hooks:
        Optional replay hook mapping.
    differentiable:
        Whether replay returns a new differentiable Trace over fresh frontier
        leaves instead of mutating the existing Trace in place.
    append:
        Whether rerun appends a compatible batch chunk.
    chunk_size:
        Forward chunk size for rerun chunking sugar. Splits positional input
        along dimension 0 and appends compatible chunks.

    Examples
    --------
    >>> opts = ReplayOptions(strict=True)
    >>> opts.strict
    True
    """

    strict: bool = False
    hooks: dict[Any, Any] | None = None
    differentiable: bool = False
    append: bool = False
    chunk_size: int | None = None
    _specified_fields: frozenset[str] = field(default_factory=frozenset, init=False, repr=False)

    def __init__(
        self,
        strict: bool | MissingType = MISSING,
        hooks: dict[Any, Any] | None | MissingType = MISSING,
        differentiable: bool | MissingType = MISSING,
        append: bool | MissingType = MISSING,
        chunk_size: int | None | MissingType = MISSING,
    ) -> None:
        """Initialize a frozen replay option bundle."""

        specified_fields: set[str] = set()
        values: dict[str, Any] = {
            "strict": _resolve_option_value("strict", strict, False, specified_fields),
            "hooks": _resolve_option_value("hooks", hooks, None, specified_fields),
            "differentiable": _resolve_option_value(
                "differentiable", differentiable, False, specified_fields
            ),
            "append": _resolve_option_value("append", append, False, specified_fields),
            "chunk_size": _resolve_option_value("chunk_size", chunk_size, None, specified_fields),
        }
        _set_frozen_fields(self, _REPLAY_FIELDS, values)
        object.__setattr__(self, "_specified_fields", frozenset(specified_fields))

    def as_dict(self) -> dict[str, Any]:
        """Return the option values as a plain dictionary."""

        return {field_name: getattr(self, field_name) for field_name in _REPLAY_FIELDS}

    def is_field_explicit(self, field_name: str) -> bool:
        """Return whether a field was explicitly supplied by the caller."""

        return _field_is_explicit(self, field_name)

    @classmethod
    def from_values(
        cls, values: Mapping[str, Any], specified_fields: frozenset[str]
    ) -> ReplayOptions:
        """Build an instance from already-resolved field values."""

        instance = object.__new__(cls)
        _set_frozen_fields(instance, _REPLAY_FIELDS, values)
        object.__setattr__(instance, "_specified_fields", specified_fields)
        return instance


@dataclass(frozen=True, init=False)
class InterventionOptions:
    """Grouped intervention options for ``Trace.do``.

    Parameters
    ----------
    engine:
        Propagation engine: ``"auto"``, ``"replay"``, ``"rerun"``, or ``"set_only"``.
    confirm_mutation:
        Whether root-mutation warnings are suppressed for intentional mutation.
    strict:
        Whether selector and propagation checks raise instead of warning.

    Examples
    --------
    >>> opts = InterventionOptions(engine="set_only", strict=True)
    >>> opts.engine
    'set_only'
    """

    engine: str = "auto"
    confirm_mutation: bool = False
    strict: bool = False
    _specified_fields: frozenset[str] = field(default_factory=frozenset, init=False, repr=False)

    def __init__(
        self,
        engine: str | MissingType = MISSING,
        confirm_mutation: bool | MissingType = MISSING,
        strict: bool | MissingType = MISSING,
    ) -> None:
        """Initialize a frozen intervention option bundle."""

        specified_fields: set[str] = set()
        values: dict[str, Any] = {
            "engine": _resolve_option_value("engine", engine, "auto", specified_fields),
            "confirm_mutation": _resolve_option_value(
                "confirm_mutation", confirm_mutation, False, specified_fields
            ),
            "strict": _resolve_option_value("strict", strict, False, specified_fields),
        }
        _set_frozen_fields(self, _INTERVENTION_FIELDS, values)
        object.__setattr__(self, "_specified_fields", frozenset(specified_fields))

    def as_dict(self) -> dict[str, Any]:
        """Return the option values as a plain dictionary."""

        return {field_name: getattr(self, field_name) for field_name in _INTERVENTION_FIELDS}

    def is_field_explicit(self, field_name: str) -> bool:
        """Return whether a field was explicitly supplied by the caller."""

        return _field_is_explicit(self, field_name)

    @classmethod
    def from_values(
        cls,
        values: Mapping[str, Any],
        specified_fields: frozenset[str],
    ) -> InterventionOptions:
        """Build an instance from already-resolved field values."""

        instance = object.__new__(cls)
        _set_frozen_fields(instance, _INTERVENTION_FIELDS, values)
        object.__setattr__(instance, "_specified_fields", specified_fields)
        return instance


@dataclass(frozen=True, init=False)
class StreamingOptions:
    """Grouped streaming-save options for ``trace``.

    Parameters
    ----------
    bundle_path:
        Portable bundle directory for streamed out saves.
    retain_in_memory:
        Whether streamed outs remain in memory.
    out_callback:
        Callback invoked with ``(label, tensor)`` for each saved out.
    include_custom_attributes:
        Whether harvested module attributes (``Module.custom_attributes``)
        are persisted verbatim in the streamed bundle. The streamed-bundle
        counterpart of ``tl.save(..., include_custom_attributes=)`` (R62:
        the streaming path had no opt-out and reopened the token-leak class
        the save-path fix closed).
    include_buffer_values:
        Whether captured pre-forward buffer values
        (``Trace._buffer_initial_values``) are persisted in the streamed
        bundle. The streamed-bundle counterpart of
        ``tl.save(..., include_buffer_values=)`` (R62 buffer extension).
    async_writes:
        Whether ``trace(..., storage=...)`` blob writes overlap forward
        capture through a bounded single-worker pipeline instead of pausing
        capture for each write (DOCUMENTED-UNSTABLE spelling, naming deferred
        to the UI sprint). Tri-state: ``None`` (default) means the consumer
        default — async for ``trace`` captures, synchronous for
        ``tl.record`` streaming; ``False`` forces synchronous writes;
        ``True`` requires the async pipeline and refuses typed on consumers
        that cannot honor it (``tl.record``). Ordering, backpressure,
        failure latching, and the finalize drain barrier are documented on
        ``BundleStreamWriter``.
    max_pending_bytes:
        Pending snapshot byte budget for the async pipeline (``None`` uses
        the 256 MiB default). Once pending writes hold this many bytes,
        capture blocks until the disk catches up, so a slow disk slows
        capture instead of accumulating unbounded RAM.

    Examples
    --------
    >>> opts = StreamingOptions(bundle_path="run.tlspec", retain_in_memory=False)
    >>> opts.retain_in_memory
    False
    """

    bundle_path: str | Path | None = None
    retain_in_memory: bool = True
    out_callback: Callable[[str, torch.Tensor], None] | None = None
    include_custom_attributes: bool = True
    include_buffer_values: bool = True
    async_writes: bool | None = None
    max_pending_bytes: int | None = None
    _specified_fields: frozenset[str] = field(default_factory=frozenset, init=False, repr=False)

    def __init__(
        self,
        bundle_path: str | Path | None | MissingType = MISSING,
        retain_in_memory: bool | MissingType = MISSING,
        out_callback: Callable[[str, torch.Tensor], None] | None | MissingType = MISSING,
        *,
        include_custom_attributes: bool | MissingType = MISSING,
        include_buffer_values: bool | MissingType = MISSING,
        async_writes: bool | None | MissingType = MISSING,
        max_pending_bytes: int | None | MissingType = MISSING,
    ) -> None:
        """Initialize a frozen streaming option bundle."""

        specified_fields: set[str] = set()
        values: dict[str, Any] = {
            "bundle_path": _resolve_option_value(
                "bundle_path", bundle_path, None, specified_fields
            ),
            "retain_in_memory": _resolve_option_value(
                "retain_in_memory", retain_in_memory, True, specified_fields
            ),
            "out_callback": _resolve_option_value(
                "out_callback", out_callback, None, specified_fields
            ),
            "include_custom_attributes": _resolve_option_value(
                "include_custom_attributes", include_custom_attributes, True, specified_fields
            ),
            "include_buffer_values": _resolve_option_value(
                "include_buffer_values", include_buffer_values, True, specified_fields
            ),
            "async_writes": _resolve_option_value(
                "async_writes", async_writes, None, specified_fields
            ),
            "max_pending_bytes": _resolve_option_value(
                "max_pending_bytes", max_pending_bytes, None, specified_fields
            ),
        }
        _set_frozen_fields(self, _STREAMING_FIELDS, values)
        object.__setattr__(self, "_specified_fields", frozenset(specified_fields))

    def as_dict(self) -> dict[str, Any]:
        """Return the option values as a plain dictionary."""

        return {field_name: getattr(self, field_name) for field_name in _STREAMING_FIELDS}

    def is_field_explicit(self, field_name: str) -> bool:
        """Return whether a field was explicitly supplied by the caller."""

        return _field_is_explicit(self, field_name)

    @classmethod
    def from_values(
        cls,
        values: Mapping[str, Any],
        specified_fields: frozenset[str],
    ) -> StreamingOptions:
        """Build an instance from already-resolved field values."""

        instance = object.__new__(cls)
        _set_frozen_fields(instance, _STREAMING_FIELDS, values)
        object.__setattr__(instance, "_specified_fields", specified_fields)
        return instance


def to_disk(
    path: str | Path,
    *,
    retain_in_memory: bool = False,
    include_custom_attributes: bool = True,
    include_buffer_values: bool = True,
    async_writes: bool | None = None,
    max_pending_bytes: int | None = None,
) -> StreamingOptions:
    """Return storage options that stream selected payloads to a bundle.

    Parameters
    ----------
    path:
        Destination bundle directory. The path must not already exist.
    retain_in_memory:
        Whether streamed payloads should also remain as RAM copies.
    include_custom_attributes:
        Whether harvested module attributes are persisted verbatim in the
        streamed bundle (the ``tl.save`` opt-out, mirrored for streaming).
    include_buffer_values:
        Whether captured pre-forward buffer values are persisted in the
        streamed bundle (the ``tl.save`` opt-out, mirrored for streaming).
    async_writes:
        Whether ``trace(..., storage=...)`` blob writes overlap capture on a
        bounded single-worker pipeline instead of pausing the forward for
        each write (DOCUMENTED-UNSTABLE spelling, naming deferred to the UI
        sprint). Tri-state: ``None`` (default) means async for ``trace``
        captures and synchronous for ``tl.record`` streaming; ``False``
        forces synchronous writes; ``True`` requires the async pipeline and
        refuses typed where it cannot be honored (``tl.record``). Writes
        land in submission order, a failed write raises
        ``TorchLensIOError`` and marks the bundle PARTIAL, and finalization
        waits for every pending write before the bundle publishes.
    max_pending_bytes:
        Pending snapshot byte budget for the async pipeline (``None`` uses
        the 256 MiB default). Once pending writes hold this many bytes,
        capture blocks until the disk catches up, bounding the RAM the
        deferred writes may occupy.

    Returns
    -------
    StreamingOptions
        Storage options suitable for ``trace(..., storage=...)`` and
        ``record(..., streaming=...)``.
    """

    return StreamingOptions(
        bundle_path=path,
        retain_in_memory=retain_in_memory,
        include_custom_attributes=include_custom_attributes,
        include_buffer_values=include_buffer_values,
        async_writes=async_writes,
        max_pending_bytes=max_pending_bytes,
    )


def merge_capture_options(
    *,
    capture: CaptureOptions | None,
    **flat_values: Any,
) -> CaptureOptions:
    """Merge individual capture kwargs into a grouped options object."""

    return cast(
        CaptureOptions,
        _merge_grouped_options(
            option=capture,
            option_factory=CaptureOptions,
            flat_to_group=_CAPTURE_FLAT_TO_GROUP,
            flat_values=flat_values,
            group_name="capture",
            conflict_message=(
                "conflicting capture options: pass either CaptureOptions or individual kwargs, not both"
            ),
        ),
    )


def merge_save_options(*, save: SaveOptions | None, **flat_values: Any) -> SaveOptions:
    """Merge individual save kwargs into a grouped options object."""

    return cast(
        SaveOptions,
        _merge_grouped_options(
            option=save,
            option_factory=SaveOptions,
            flat_to_group=_SAVE_FLAT_TO_GROUP,
            flat_values=flat_values,
            group_name="save",
            conflict_message=(
                "conflicting save options: pass either SaveOptions or individual kwargs, not both"
            ),
        ),
    )


def merge_visualization_options(
    *,
    function_default_mode: VisModeLiteral,
    visualization: VisualizationOptions | None,
    view: VisModeLiteral | MissingType = MISSING,
    depth: int | MissingType = MISSING,
    layout: VisNodePlacementLiteral | MissingType = MISSING,
    node_style: VisNodeModeLiteral | MissingType = MISSING,
    renderer: VisRendererLiteral | MissingType = MISSING,
    collapse: CollapseLiteral | MissingType = MISSING,
    fold_repeats: FoldRepeatsLiteral | MissingType = MISSING,
    order_siblings: bool | MissingType = MISSING,
) -> VisualizationOptions:
    """Merge the canonical flat visualization kwargs into a grouped options object."""

    if visualization is None:
        values = VisualizationOptions().as_dict()
        values["view"] = function_default_mode
        specified_fields: frozenset[str] = frozenset()
    else:
        values = visualization.as_dict()
        specified_fields = _explicit_fields(visualization)
        if "view" not in specified_fields:
            # The grouped object's non-explicit view default ("none") must not
            # shadow the calling function's default mode: show_model_graph
            # with a VisualizationOptions that leaves view unset used to merge
            # to view="none" and silently render NOTHING (D04 integration fix).
            values["view"] = function_default_mode

    flat_values: dict[str, Any] = {
        "view": view,
        "depth": depth,
        "layout": layout,
        "node_style": node_style,
        "renderer": renderer,
        "collapse": collapse,
        "fold_repeats": fold_repeats,
        "order_siblings": order_siblings,
    }
    for flat_name, group_name in _VISUALIZATION_FLAT_TO_GROUP.items():
        flat_value = flat_values[flat_name]
        if flat_value is MISSING:
            continue
        if visualization is not None and group_name in specified_fields:
            raise KeywordConflictError(
                f"Do not pass both `{flat_name}` and `visualization.{group_name}`",
                code="option_group_keyword_conflict",
                remedy=f"remove either {flat_name!r} or {f'visualization.{group_name}'!r}",
                arguments=(flat_name, f"visualization.{group_name}"),
            )
        values[group_name] = flat_value
        specified_fields = frozenset((*specified_fields, group_name))
    return VisualizationOptions.from_values(values, specified_fields)


def merge_replay_options(*, replay: ReplayOptions | None, **flat_values: Any) -> ReplayOptions:
    """Merge individual replay kwargs into a grouped options object."""

    return cast(
        ReplayOptions,
        _merge_grouped_options(
            option=replay,
            option_factory=ReplayOptions,
            flat_to_group=_REPLAY_FLAT_TO_GROUP,
            flat_values=flat_values,
            group_name="replay",
            conflict_message=(
                "conflicting replay options: pass either ReplayOptions or individual kwargs, not both"
            ),
        ),
    )


def merge_intervention_options(
    *,
    intervention: InterventionOptions | None,
    **flat_values: Any,
) -> InterventionOptions:
    """Merge individual intervention kwargs into a grouped options object."""

    return cast(
        InterventionOptions,
        _merge_grouped_options(
            option=intervention,
            option_factory=InterventionOptions,
            flat_to_group=_INTERVENTION_FLAT_TO_GROUP,
            flat_values=flat_values,
            group_name="intervention",
            conflict_message=(
                "conflicting intervention options: pass either InterventionOptions or individual "
                "kwargs, not both"
            ),
        ),
    )


def merge_streaming_options(
    *,
    streaming: StreamingOptions | None,
    **flat_values: Any,
) -> StreamingOptions:
    """Merge individual streaming kwargs into a grouped options object."""

    return cast(
        StreamingOptions,
        _merge_grouped_options(
            option=streaming,
            option_factory=StreamingOptions,
            flat_to_group=_STREAMING_FLAT_TO_GROUP,
            flat_values=flat_values,
            group_name="streaming",
            conflict_message=(
                "conflicting streaming options: pass either StreamingOptions or individual kwargs, "
                "not both"
            ),
        ),
    )


def visualization_to_render_kwargs(visualization: VisualizationOptions) -> dict[str, Any]:
    """Translate grouped visualization options into ``Trace.draw`` kwargs.

    Parameters
    ----------
    visualization:
        Resolved grouped visualization options.

    Returns
    -------
    dict[str, Any]
        Keyword arguments expected by ``Trace.draw``.
    """

    kwargs: dict[str, Any] = {
        "vis_mode": visualization.view,
        "vis_call_depth": visualization.depth,
        "vis_outpath": visualization.container_path,
        "vis_graph_overrides": visualization.graph_overrides,
        "node_mode": visualization.node_style,
        "node_spec_fn": visualization.node_spec_fn,
        "collapsed_node_spec_fn": visualization.collapsed_node_spec_fn,
        "collapse_fn": visualization.collapse_fn,
        "collapse": visualization.collapse,
        "skip_fn": visualization.skip_fn,
        "vis_edge_overrides": visualization.edge_overrides,
        "vis_grad_edge_overrides": visualization.grad_edge_overrides,
        "vis_module_overrides": visualization.module_overrides,
        "vis_save_only": visualization.save_only,
        "vis_fileformat": visualization.file_format,
        "show_buffer_layers": visualization.show_buffers,
        "direction": visualization.direction,
        "vis_node_placement": visualization.layout,
        "vis_renderer": visualization.renderer,
        "vis_theme": visualization.theme,
        "vis_intervention_mode": visualization.intervention_mode,
        "vis_show_cone": visualization.show_cone,
    }
    if visualization.fold_repeats is not None or visualization.is_field_explicit("fold_repeats"):
        kwargs["fold_repeats"] = visualization.fold_repeats
    phase7_kwargs = {
        "node_overlay": visualization.node_overlay,
        "node_label_fields": visualization.node_label_fields,
        "show_legend": visualization.show_legend,
        "color_by": visualization.color_by,
        "size_by": visualization.size_by,
        "scale": visualization.scale,
        "stack_by": visualization.stack_by,
        "show_redundant_args": visualization.show_redundant_args,
        "font_size": visualization.font_size,
        "dpi": visualization.dpi,
        "for_paper": visualization.for_paper,
        "return_graph": visualization.return_graph,
        "order_siblings": visualization.order_siblings,
    }
    for field_name, value in phase7_kwargs.items():
        if field_name == "order_siblings":
            should_include = visualization.is_field_explicit(field_name) or value is False
        else:
            should_include = visualization.is_field_explicit(field_name) or (
                value is not None and value is not False
            )
        if should_include:
            kwargs[field_name] = value
    return kwargs


__all__ = [
    "CaptureOptions",
    "EchoOptions",
    "EpisodeSpec",
    "InterventionOptions",
    "OptionReceiptEntry",
    "ReplayOptions",
    "SaveOptions",
    "StreamingOptions",
    "option_receipt",
    "to_disk",
    "VisualizationOptions",
    "suppress_mutate_warnings",
]
