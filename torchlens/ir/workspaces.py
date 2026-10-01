"""Named per-phase capture/postprocess workspaces (M10).

The former ``TraceBuildState`` — one flat 19-field scratchpad shared by every
capture and postprocess phase — dissolves into three named workspaces, each
owned by exactly ONE phase family and dropped at the transient-state cleanup
seam (``_drop_transient_capture_state``). No workspace is portable state:
every field is ``FieldPolicy.DROP``, absent on loaded traces, and never
survives pickling.

Ownership map (docs/reference/trace_core_design.md section 3.7):

* ``RawGraphWorkspace`` — owned by the capture ingress and postprocess
  steps 0-11: the raw op registry the graph passes rewrite until final
  labels exist. The backend ``finalize_forward_session`` protocol uses this
  workspace as its ownership token.
* ``ModuleCaptureWorkspace`` — owned by the backend module-prep/stack
  capture; consumed (and cleared) by step 16's module build.
* ``WrapperRuntimeWorkspace`` — owned by the wrapper hot path during the
  live forward: per-call barcode nesting, the exhaustive-pass flag, and the
  container registry the output-container passes read afterwards.

The former dead fields ``grad_fn_strong_refs``,
``output_container_specs_by_raw_label``, and ``output_container_specs``
(zero readers tree-wide; the live trace-level
``_output_container_specs_by_raw_label`` field is a separate attribute) were
deleted with the dissolution rather than assigned a fictitious owner.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any

from .container_registry import ContainerRegistry

LEGACY_TRACE_BUILD_STATE_KEYS = frozenset(
    {
        "_raw_layer_dict",
        "_raw_layer_labels_list",
        "_layer_counter",
        "_raw_layer_type_counter",
        "_current_func_barcode",
        "_mod_entered",
        "_mod_exited",
        "_mod_call_index",
        "_mod_call_labels",
        "_module_build_data",
        "_module_metadata",
        "_module_forward_args",
        "_module_containment_engine",
        "_exhaustive_module_stack",
        "_grad_fn_strong_refs",
        "_in_exhaustive_pass",
        "_input_tensor_addresses",
    }
)
"""Legacy flat Trace scratch keys accepted only for drop-on-restore compatibility."""


@dataclass(slots=True)
class RawGraphWorkspace:
    """Raw op registry owned by capture ingress + postprocess steps 0-11."""

    raw_layer_dict: dict[str, Any] = field(default_factory=OrderedDict)
    raw_layer_labels_list: list[str] = field(default_factory=list)
    layer_counter: int = 0
    raw_layer_type_counter: dict[str, int] = field(default_factory=lambda: defaultdict(lambda: 0))
    input_tensor_addresses: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ReleasedTensorStub:
    """Payload-free surrogate for a released module-forward-arg tensor.

    F20 W1a (capture-floor release-at-emission): the module-arg stash held a
    strong reference to every module call's live input tensors for the whole
    forward -- one of the two dominant capture-floor holders (brainpipe memo
    section 3.3). At module exit the tensor leaves are replaced with these
    stubs, which carry exactly what the two downstream consumers read: the
    ``shape``/``dtype`` pair for ``format_call_arg`` summaries and the shape
    for quantized-module FLOPs estimation. The payload itself is never
    consumed after module exit on the torch trace path (GC-11 nulls
    ``ModuleCall.forward_args`` before the trace is returned).
    """

    shape: tuple[int, ...]
    dtype: str
    device: str
    label_raw: str | None


@dataclass(slots=True)
class ModuleCaptureWorkspace:
    """Module prep/stack capture state consumed by the step 16 module build."""

    mod_entered: dict[int, list[str]] = field(default_factory=dict)
    mod_exited: dict[int, list[str]] = field(default_factory=dict)
    mod_call_index: dict[int, int] = field(default_factory=dict)
    mod_call_labels: dict[int, list[tuple[str, int]]] = field(default_factory=dict)
    exhaustive_module_stack: list[Any] = field(default_factory=list)
    module_build_data: dict[str, Any] = field(default_factory=dict)
    module_metadata: dict[Any, Any] = field(default_factory=dict)
    module_forward_args: dict[Any, Any] = field(default_factory=dict)


def _init_module_hierarchy_data() -> dict[str, Any]:
    """Create the transient ``module_build_data`` dict for ``ModuleCaptureWorkspace``.

    Consumed by ``_build_module_logs`` (step 16) and then cleared.
    """
    return {
        "addresses": [],
        "module_types": {},
        "module_ops": [],
        "module_num_calls": defaultdict(lambda: 1),
        "top_level_modules": [],
        "top_level_module_ops": [],
        "module_children": defaultdict(list),
        "module_pass_children": defaultdict(list),
        "module_nparams": defaultdict(lambda: 0),
        "module_nparams_trainable": defaultdict(lambda: 0),
        "module_nparams_frozen": defaultdict(lambda: 0),
        "module_num_tensors": defaultdict(lambda: 0),
        "module_call_index_tensors": defaultdict(lambda: 0),
        "module_layers": defaultdict(list),
        "module_pass_layers": defaultdict(list),
        "module_output_structures": {},
        "module_layer_argnames": defaultdict(list),
        "module_training_modes": {},
        "module_forward_start_times": {},
        "module_forward_durations": {},
        "module_code_contexts": {},
        "module_call_stacks": {},
    }


@dataclass(slots=True)
class WrapperRuntimeWorkspace:
    """Wrapper hot-path state owned by the live forward pass."""

    current_func_barcode: Any = None
    in_exhaustive_pass: bool = True
    container_registry: ContainerRegistry = field(default_factory=ContainerRegistry)
