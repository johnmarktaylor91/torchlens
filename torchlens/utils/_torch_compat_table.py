"""Per-flag torch capability table: measured availability and declarative probes.

One row per ``HAS_*`` capability flag in :mod:`torchlens.utils._torch_compat`.
Each row records the torch versions of the measured matrix on which the flag
read ``False`` (``absent_on``), so a support-floor change deletes rows and
probes by reading this table instead of re-measuring. A row with an empty
``absent_on`` is constant since the 2.1 floor on the measured matrix.

Rows with ``attribute_paths`` are probed declaratively: the flag is ``True``
when every ``"module:dotted.attr"`` path resolves to a non-``None`` object.
That keeps feature detection (a torch release that removes a private surface
still flips the flag) at one line per flag. Rows without paths keep a bespoke
behavioral probe in ``_torch_compat``.

This module is data only and imports nothing from torch, so tooling can read
it without importing the capture stack.
"""

from __future__ import annotations

from dataclasses import dataclass

MEASURED_TORCH_VERSIONS: tuple[str, ...] = (
    "2.1.2",
    "2.2.2",
    "2.3.1",
    "2.4.1",
    "2.5.1",
    "2.6.0",
    "2.7.1",
    "2.8.0",
    "2.9.1",
    "2.13.0",
)
"""Torch releases the ``absent_on`` columns were measured on."""

MEASUREMENT_CONDITIONS: str = "Linux, CPU-only wheels, CPython 3.10"
"""Build conditions of the measurement; GPU builds and newer CPython are unmeasured."""


@dataclass(frozen=True)
class CapabilityRow:
    """One capability flag's measured availability and optional declarative probe.

    Attributes
    ----------
    flag:
        Module-level ``HAS_*`` flag name in ``_torch_compat``.
    absent_on:
        Measured torch versions on which the flag read ``False``.
    attribute_paths:
        ``"module:dotted.attr"`` paths that must all resolve for the flag to be
        ``True``; empty when the flag keeps a bespoke probe.
    measured:
        ``False`` when the measurement read only the unprobed placeholder (a
        lazy probe that needs a live profiler, or one the snapshot did not
        force at measurement time), so ``absent_on`` carries no information.
    """

    flag: str
    absent_on: tuple[str, ...] = ()
    attribute_paths: tuple[str, ...] = ()
    measured: bool = True

    @property
    def measured_class(self) -> str:
        """Return ``constant-true``, ``constant-false``, ``varying`` or ``unmeasured``.

        Returns
        -------
        str
            Classification of the flag across :data:`MEASURED_TORCH_VERSIONS`.
        """

        if not self.measured:
            return "unmeasured"
        if not self.absent_on:
            return "constant-true"
        if len(self.absent_on) == len(MEASURED_TORCH_VERSIONS):
            return "constant-false"
        return "varying"


def _row(
    flag: str,
    *,
    absent_on: tuple[str, ...] = (),
    attribute_paths: tuple[str, ...] = (),
    measured: bool = True,
) -> CapabilityRow:
    return CapabilityRow(flag, absent_on, attribute_paths, measured)


_ROWS: tuple[CapabilityRow, ...] = (
    _row("HAS_ACCUMULATE_GRAD_CLASS", attribute_paths=("torch:_C._functions.AccumulateGrad",)),
    _row(
        "HAS_AMP_GRADSCALER",
        absent_on=("2.1.2", "2.2.2"),
        attribute_paths=("torch.amp:GradScaler",),
    ),
    _row("HAS_ASYNC_COLLECTIVE_TENSOR"),
    _row("HAS_ATTENTION_CAUSAL_BIAS", absent_on=("2.1.2", "2.2.2")),
    _row("HAS_AUTOGRAD_ENGINE_QUEUE_CALLBACK"),
    _row("HAS_C10D_ABORT_PG", absent_on=("2.1.2", "2.2.2", "2.3.1", "2.4.1", "2.5.1")),
    _row("HAS_C10D_GROUP_REGISTRY"),
    _row("HAS_C10D_GROUP_SEQ"),
    _row("HAS_CACHED_UNTYPED_STORAGE_WRAPPER", absent_on=("2.1.2",)),
    _row("HAS_CHECKPOINT_HOOK_CLASS"),
    _row(
        "HAS_CHECKPOINT_INTERNAL_HOOK_CLASS",
        absent_on=(
            "2.1.2",
            "2.2.2",
            "2.3.1",
            "2.4.1",
            "2.5.1",
            "2.6.0",
            "2.7.1",
            "2.8.0",
            "2.9.1",
            "2.13.0",
        ),
    ),
    _row(
        "HAS_CODE_POSITIONS",
        absent_on=(
            "2.1.2",
            "2.2.2",
            "2.3.1",
            "2.4.1",
            "2.5.1",
            "2.6.0",
            "2.7.1",
            "2.8.0",
            "2.9.1",
            "2.13.0",
        ),
    ),
    _row(
        "HAS_CODE_QUALNAME",
        absent_on=(
            "2.1.2",
            "2.2.2",
            "2.3.1",
            "2.4.1",
            "2.5.1",
            "2.6.0",
            "2.7.1",
            "2.8.0",
            "2.9.1",
            "2.13.0",
        ),
    ),
    _row(
        "HAS_CPU_FLOAT8_DETERMINISTIC_FILL", absent_on=("2.1.2", "2.2.2", "2.3.1", "2.4.1", "2.5.1")
    ),
    _row("HAS_CPU_HALF_KERNELS", absent_on=("2.1.2", "2.2.2")),
    _row("HAS_CUDA_GRAPHS"),
    _row("HAS_CURRENT_AUTOGRAD_NODE", attribute_paths=("torch:_C._current_autograd_node",)),
    _row("HAS_CURRENT_GRAPH_TASK_ID", attribute_paths=("torch:_C._current_graph_task_id",)),
    _row("HAS_DETERMINISTIC_FILL_FLAG", absent_on=("2.1.2",)),
    _row("HAS_DEVICE_CONSTRUCTORS", attribute_paths=("torch.utils._device:_device_constructors",)),
    _row(
        "HAS_DEVICE_CONTEXT_DISPATCH",
        attribute_paths=(
            "torch.utils._device:DeviceContext",
            "torch.overrides:_get_current_function_mode_stack",
            "torch.overrides:_len_torch_function_stack",
        ),
    ),
    _row("HAS_DEVICE_MESH", absent_on=("2.1.2",)),
    _row("HAS_DISABLED_TORCH_FUNCTION_IMPL"),
    _row("HAS_DISABLE_TORCH_FUNCTION"),
    _row("HAS_DISPATCH_MODE_STACK_QUERY"),
    _row("HAS_DTENSOR"),
    _row("HAS_DTENSOR_SHARD_GEOMETRY", absent_on=("2.1.2", "2.2.2", "2.3.1", "2.4.1")),
    _row("HAS_DYNAMO_COMPILE_COUNTERS"),
    _row("HAS_DYNAMO_EXPLAIN"),
    _row("HAS_DYNAMO_IS_COMPILING"),
    _row("HAS_DYNAMO_OPTIMIZED_MODULE"),
    _row("HAS_DYNAMO_ORIG_CALLABLE_MARKER", absent_on=("2.9.1",)),
    _row("HAS_EXPANDED_WEIGHTS_CONV_PICKER"),
    _row("HAS_FAKE_TENSOR_MODE"),
    _row(
        "HAS_FP32_PRECISION_CONTROLS",
        absent_on=("2.1.2", "2.2.2", "2.3.1", "2.4.1", "2.5.1", "2.6.0", "2.7.1", "2.8.0"),
        attribute_paths=("torch:backends.fp32_precision",),
    ),
    _row("HAS_FP8_DTYPES"),
    _row("HAS_FSDP_WRAPPER"),
    _row(
        "HAS_FUNCOL_GROUP_RESOLUTION",
        absent_on=("2.1.2", "2.2.2", "2.3.1", "2.4.1", "2.5.1", "2.6.0", "2.7.1", "2.8.0", "2.9.1"),
    ),
    _row("HAS_FUNCOL_MODULE"),
    _row("HAS_FUNCOL_WAIT_INTERPOSITION", absent_on=("2.1.2",)),
    _row("HAS_FUNCTORCH_APIS"),
    _row(
        "HAS_FUNCTORCH_LEVEL_API",
        absent_on=("2.1.2", "2.2.2"),
        attribute_paths=("torch:_C._functorch.maybe_current_level",),
    ),
    _row(
        "HAS_FUNCTORCH_WRAPPED_TENSOR_API",
        attribute_paths=("torch:_C._functorch.is_functorch_wrapped_tensor",),
    ),
    _row("HAS_FX_GRAPH_MODULE", attribute_paths=("torch:fx.GraphModule",)),
    _row(
        "HAS_GENERATOR_CLONE_STATE",
        absent_on=("2.1.2", "2.2.2", "2.3.1"),
        attribute_paths=("torch:Generator.clone_state",),
    ),
    _row(
        "HAS_GENERATOR_GRAPHSAFE_GET_STATE",
        absent_on=("2.1.2", "2.2.2", "2.3.1"),
        attribute_paths=("torch:Generator.graphsafe_get_state",),
    ),
    _row(
        "HAS_GENERATOR_GRAPHSAFE_SET_STATE",
        absent_on=("2.1.2", "2.2.2", "2.3.1"),
        attribute_paths=("torch:Generator.graphsafe_set_state",),
    ),
    _row(
        "HAS_GENERATOR_PHILOX_STATE",
        absent_on=(
            "2.1.2",
            "2.2.2",
            "2.3.1",
            "2.4.1",
            "2.5.1",
            "2.6.0",
            "2.7.1",
            "2.8.0",
            "2.9.1",
            "2.13.0",
        ),
        attribute_paths=("torch:Generator.philox_state",),
    ),
    _row("HAS_GRADIENT_EDGE", absent_on=("2.1.2", "2.2.2", "2.3.1", "2.4.1")),
    _row(
        "HAS_JIT_BOOLEAN_DISPATCH_TABLE",
        attribute_paths=("torch._jit_internal:boolean_dispatched",),
    ),
    _row("HAS_JIT_BUILTIN_TABLE", attribute_paths=("torch.jit._builtins:_builtin_table",)),
    _row("HAS_JIT_OVERLOAD_RESOLVER", attribute_paths=("torch.jit._script:_get_overloads",)),
    _row("HAS_JIT_SCHEMA_ENUMERATION"),
    _row("HAS_KINETO_EVENT_SCOPE", measured=False),
    _row("HAS_KINETO_INMEMORY_EVENTS", measured=False),
    _row("HAS_LEGACY_CONSTRUCTOR_NEW_PATCH"),
    _row("HAS_MEMORY_PROFILE", measured=False),
    _row("HAS_META_ITEM_GUARD", measured=False),
    _row("HAS_MHA_FASTPATH_SWITCH", measured=False),
    _row(
        "HAS_NAMED_TENSOR_API",
        absent_on=("2.13.0",),
        attribute_paths=(
            "torch:Tensor.names",
            "torch:Tensor.has_names",
            "torch:Tensor.refine_names",
        ),
    ),
    _row("HAS_NN_ATTENTION_MODULE", absent_on=("2.1.2", "2.2.2")),
    _row("HAS_NODE_PREHOOK", attribute_paths=("torch.autograd.graph:Node.register_prehook",)),
    _row(
        "HAS_PARAMETER_AS_SUBCLASS_IN_DISPATCH_MODE",
        absent_on=("2.1.2", "2.2.2", "2.3.1", "2.4.1", "2.5.1"),
    ),
    _row("HAS_PIPELINING"),
    _row("HAS_REDUCE_TUPLE_DIM", absent_on=("2.1.2",)),
    _row(
        "HAS_RMSNORM_MODULE",
        absent_on=("2.1.2", "2.2.2", "2.3.1"),
        attribute_paths=("torch:nn.RMSNorm",),
    ),
    _row("HAS_SAFE_WEIGHTS_ONLY_LOAD", absent_on=("2.1.2", "2.2.2", "2.3.1", "2.4.1", "2.5.1")),
    _row("HAS_SAVED_TENSORS_HOOKS_PATCHABLE"),
    _row(
        "HAS_SAVED_TENSORS_HOOK_INTROSPECTION",
        absent_on=("2.1.2", "2.2.2", "2.3.1", "2.4.1", "2.5.1", "2.6.0", "2.7.1"),
    ),
    _row("HAS_SCRIPT_FUNCTION_CALL_HOOK"),
    _row("HAS_SET_STANCE", absent_on=("2.1.2", "2.2.2", "2.3.1", "2.4.1", "2.5.1")),
    _row(
        "HAS_SUBCLASS_CTOR_IN_DISPATCH_MODE",
        absent_on=("2.1.2", "2.2.2", "2.3.1", "2.4.1", "2.5.1"),
    ),
    _row("HAS_TENSORBASE_CLASS"),
    _row("HAS_TENSOR_SEQUENCE_SLOT_FIX"),
    _row("HAS_TORCH_FUNC", attribute_paths=("torch:func",)),
    _row("HAS_TORCH_FUNCTION_STACK_SURGERY"),
    _row("HAS_TORCH_VF", attribute_paths=("torch:_VF",)),
    _row("HAS_TRACING_TENSOR_TYPES"),
    _row("HAS_TRANSFORMER_ACTIVATION_FASTPATH_FLAG"),
    _row("HAS_VARIABLE_FUNCTIONS", attribute_paths=("torch:_C._VariableFunctions",)),
    _row("HAS_VARIABLE_FUNCTIONS_CLASS"),
)

CAPABILITY_ROWS: dict[str, CapabilityRow] = {row.flag: row for row in _ROWS}
"""Flag name -> row, in flag-name order."""
