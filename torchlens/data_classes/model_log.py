"""ModelLog: the top-level container for a fully logged forward pass.

ModelLog is the root data structure returned by ``log_forward_pass()``.
It owns every LayerPassLog (per-operation entry), every LayerLog (per-layer
aggregate), the module hierarchy, parameter metadata, and graph-level
bookkeeping.

Key design patterns:

* **_pass_finished behavioural switch** - Many methods (``__len__``, ``__getitem__``,
  ``__str__``, ``__iter__``) behave differently during logging vs after
  postprocessing.  While logging is active (``_pass_finished=False``), the
  model's tensors are keyed by their raw internal barcodes in
  ``_raw_layer_dict``.  After postprocessing flips ``_pass_finished=True``,
  the friendly ``layer_list`` / ``layer_dict_all_keys`` / ``layer_logs``
  structures are populated and used instead.  ``_pass_finished`` also
  persists across the fast pass on purpose: fast-path postprocessing
  relies on the fully-populated lookup dicts from the exhaustive pass.

* **Explicit ModelLog methods** - Public methods are defined directly on
  ``ModelLog``. Heavier implementations may delegate into subpackages
  through local imports, but users still call them as
  ``model_log.render_graph(...)`` or ``model_log.validate_forward_pass(...)``.

* **_module_build_data** - A transient dict that accumulates module hierarchy
  information during the forward pass.  Consumed by ``_build_module_logs``
  (postprocessing step 17) and then cleared.  Initialised via
  ``_init_module_build_data()``.
"""

import copy
from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import difflib
from functools import cached_property
from html import escape
from os import PathLike
from pathlib import Path
import time
import uuid
import weakref
import warnings
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Set, TYPE_CHECKING, Tuple

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    import pandas as pd

    from .._io.streaming import BundleStreamWriter
    from ..experimental.dagua._bridge import TorchLensRenderAudit
    from ..intervention.types import FireRecord
    from ..visualization.code_panel import CodePanelOption
    from .buffer_log import BufferAccessor
    from .layer_log import LayerAccessor
    from .module_log import ModuleLog

from .._deprecations import MISSING, MissingType, warn_deprecated_alias
from .. import _state
from .._run_state import RunState
from .._training_validation import reject_compiled_model
from .._literals import (
    BufferVisibilityLiteral,
    VisDirectionLiteral,
    VisInterventionModeLiteral,
    VisModeLiteral,
    VisNodeModeLiteral,
    VisNodePlacementLiteral,
    VisRendererLiteral,
)
from .._io import FieldPolicy, IO_FORMAT_VERSION, default_fill_state, read_io_format_version
from ..constants import MODEL_LOG_FIELD_ORDER
from ..options import (
    InterventionOptions,
    ReplayOptions,
    merge_intervention_options,
    merge_replay_options,
)
from ..intervention.types import (
    FrozenInterventionSpec,
    ForkFieldPolicy,
    MODEL_LOG_FORK_POLICY,
    InterventionSpec,
    Relationship,
    TargetSpec,
)
from ..types import ActivationPostfunc, GradientPostfunc
from .cleanup import (
    _LIST_FIELDS_TO_CLEAN,
    _clear_entry_attributes,
    _label_for_reference_removal,
    _remove_log_entry_references,
    _scrub_conditional_fields_after_removal,
    cleanup,
)
from .module_log import ModuleAccessor
from .param_log import ParamAccessor
from ..utils.display import human_readable_size
from .interface import (
    _format_conditional_branch_stack,
    _getitem_after_pass,
    _getitem_during_pass,
    _str_after_pass,
    _str_during_pass,
)
from .grad_fn_log import GradFnAccessor, GradFnLog
from .layer_log import LayerLog
from .layer_pass_log import LayerPassLog, TensorLog

_MODEL_LOG_DEFAULT_FILL: dict[str, Any] = {
    "name": None,
    "intervention_ready": False,
    "capture_full_args": False,
    "parent_run": None,
    "_intervention_spec": None,
    "operation_history": [],
    "last_run_ctx": None,
    "_has_direct_writes": False,
    "_warned_direct_write": False,
    "_warned_mutate_in_place": False,
    "_spec_revision": 0,
    "_activation_recipe_revision": 0,
    "_append_sequence_id": 0,
    "_last_hook_handle_ids": (),
    "run_state": RunState.PRISTINE,
    "source_model_id": None,
    "source_model_class": None,
    "weight_fingerprint_at_capture": None,
    "weight_fingerprint_full": None,
    "input_id_at_capture": None,
    "input_shape_hash": None,
    "graph_shape_hash": None,
    "module_filter_fn": None,
    "emit_nvtx": False,
    "raise_on_nan": False,
    "capture_kpis": {},
    "report_values": {},
    "observer_spans": [],
    "manual_tensor_connections": [],
    "forward_lineno": None,
    "capture_cache_hit": False,
    "capture_cache_key": None,
    "capture_cache_path": None,
    "recording_kept": True,
    "streaming_pass_logs": [],
    "num_streamed_passes": 1,
    "_activation_hash_cache": {},
    "is_appended": False,
    "relationship_evidence": {},
}
_MODEL_LOG_DEFAULT_FILL = {
    **{field_name: None for field_name in MODEL_LOG_FIELD_ORDER},
    **_MODEL_LOG_DEFAULT_FILL,
}
_MODEL_LOG_DEFAULT_FILL["io_format_version"] = IO_FORMAT_VERSION


def _init_module_build_data() -> dict:
    """Create the transient dict used to accumulate module hierarchy data during logging.

    Consumed by ``_build_module_logs`` (step 17) and then cleared.
    """
    return {
        "module_addresses": [],
        "module_types": {},
        "module_passes": [],
        "module_num_passes": defaultdict(lambda: 1),
        "top_level_modules": [],
        "top_level_module_passes": [],
        "module_children": defaultdict(list),
        "module_pass_children": defaultdict(list),
        "module_nparams": defaultdict(lambda: 0),
        "module_nparams_trainable": defaultdict(lambda: 0),
        "module_nparams_frozen": defaultdict(lambda: 0),
        "module_num_tensors": defaultdict(lambda: 0),
        "module_pass_num_tensors": defaultdict(lambda: 0),
        "module_layers": defaultdict(list),
        "module_pass_layers": defaultdict(list),
        "module_layer_argnames": defaultdict(list),
        "module_training_modes": {},
    }


@dataclass
class ConditionalEvent:
    """Structured metadata for one conditional event in user source code."""

    id: int
    kind: Literal["if_chain", "ifexp"]
    source_file: str
    function_qualname: str
    function_span: Tuple[int, int]
    if_stmt_span: Tuple[int, int]
    test_span: Tuple[int, int, int, int]
    branch_ranges: Dict[str, Tuple[int, int, int, int]]
    branch_test_spans: Dict[str, Tuple[int, int, int, int]]
    nesting_depth: int
    parent_conditional_id: Optional[int]
    parent_branch_kind: Optional[str]
    bool_layers: List[str] = field(default_factory=list)


class _CallableList(list):
    """List that returns a plain list when called.

    This keeps rare report surfaces callable for user ergonomics without adding
    extra callable methods to the ModelLog method ledger.
    """

    def __call__(self) -> list[Any]:
        """Return a plain-list copy of this report.

        Returns
        -------
        list[Any]
            Plain list containing this report's items.
        """

        return list(self)


class _CallableDict(dict):
    """Dict that returns a plain dict when called.

    This preserves legacy ``log.report_by_type()`` ergonomics for budgeted
    report properties that should not remain inspectable methods.
    """

    def __call__(self) -> dict[Any, Any]:
        """Return a plain-dict copy of this report.

        Returns
        -------
        dict[Any, Any]
            Plain dict containing this report's items.
        """

        return dict(self)


def _legacy_conditional_then_edges(
    conditional_arm_edges: Mapping[Tuple[int, str], List[Tuple[str, str]]],
) -> List[Tuple[str, str]]:
    """Return the legacy THEN-edge view from canonical conditional arm edges.

    Parameters
    ----------
    conditional_arm_edges:
        Canonical ``(cond_id, branch_kind) -> edge list`` mapping.

    Returns
    -------
    List[Tuple[str, str]]
        Legacy ``(parent, child)`` THEN-edge view.
    """

    return [
        edge
        for (_conditional_id, branch_kind), edges in conditional_arm_edges.items()
        if branch_kind == "then"
        for edge in edges
    ]


def _legacy_conditional_elif_edges(
    conditional_arm_edges: Mapping[Tuple[int, str], List[Tuple[str, str]]],
) -> List[Tuple[int, int, str, str]]:
    """Return the legacy ELIF-edge view from canonical conditional arm edges.

    Parameters
    ----------
    conditional_arm_edges:
        Canonical ``(cond_id, branch_kind) -> edge list`` mapping.

    Returns
    -------
    List[Tuple[int, int, str, str]]
        Legacy ``(cond_id, elif_index, parent, child)`` ELIF-edge view.
    """

    return [
        (conditional_id, int(branch_kind.split("_", 1)[1]), parent, child)
        for (conditional_id, branch_kind), edges in conditional_arm_edges.items()
        if branch_kind.startswith("elif_")
        for parent, child in edges
    ]


def _legacy_conditional_else_edges(
    conditional_arm_edges: Mapping[Tuple[int, str], List[Tuple[str, str]]],
) -> List[Tuple[int, str, str]]:
    """Return the legacy ELSE-edge view from canonical conditional arm edges.

    Parameters
    ----------
    conditional_arm_edges:
        Canonical ``(cond_id, branch_kind) -> edge list`` mapping.

    Returns
    -------
    List[Tuple[int, str, str]]
        Legacy ``(cond_id, parent, child)`` ELSE-edge view.
    """

    return [
        (conditional_id, parent, child)
        for (conditional_id, branch_kind), edges in conditional_arm_edges.items()
        if branch_kind == "else"
        for parent, child in edges
    ]


class ModelLog:
    """Top-level container for a logged forward pass.

    Serves double duty: during the forward pass it accumulates raw tensor
    metadata in ``_raw_layer_dict``; after postprocessing (``_pass_finished=True``)
    it presents a clean, user-facing view via ``layer_list``, ``layer_dict_all_keys``,
    ``layer_logs``, ``modules``, ``params``, and ``buffers``.

    Supports ``len()``, iteration, and flexible ``__getitem__`` lookup by
    integer index, layer label, module address, or substring.
    """

    PORTABLE_STATE_SPEC: dict[str, FieldPolicy] = {
        "name": FieldPolicy.KEEP,
        "model_name": FieldPolicy.KEEP,
        "num_context_lines": FieldPolicy.KEEP,
        "_optimizer": FieldPolicy.DROP,
        "io_format_version": FieldPolicy.KEEP,
        "_pass_finished": FieldPolicy.KEEP,
        "logging_mode": FieldPolicy.KEEP,
        "_all_layers_logged": FieldPolicy.KEEP,
        "_all_layers_saved": FieldPolicy.KEEP,
        "keep_unsaved_layers": FieldPolicy.KEEP,
        "intervention_ready": FieldPolicy.KEEP,
        "capture_full_args": FieldPolicy.KEEP,
        "activation_postfunc": FieldPolicy.DROP,
        "activation_postfunc_repr": FieldPolicy.KEEP,
        "save_raw_activation": FieldPolicy.KEEP,
        "input_metadata": FieldPolicy.KEEP,
        "_source_code_blob": FieldPolicy.KEEP,
        "_source_model_ref": FieldPolicy.DROP,
        "parent_run": FieldPolicy.DROP,
        "source_model_id": FieldPolicy.KEEP,
        "source_model_class": FieldPolicy.KEEP,
        "weight_fingerprint_at_capture": FieldPolicy.KEEP,
        "weight_fingerprint_full": FieldPolicy.KEEP,
        "input_id_at_capture": FieldPolicy.KEEP,
        "input_shape_hash": FieldPolicy.KEEP,
        "current_function_call_barcode": FieldPolicy.KEEP,
        "random_seed_used": FieldPolicy.KEEP,
        "output_device": FieldPolicy.KEEP,
        "detach_saved_tensors": FieldPolicy.KEEP,
        "train_mode": FieldPolicy.DROP,
        "module_filter_fn": FieldPolicy.DROP,
        "emit_nvtx": FieldPolicy.KEEP,
        "raise_on_nan": FieldPolicy.KEEP,
        "capture_kpis": FieldPolicy.KEEP,
        "report_values": FieldPolicy.KEEP,
        "observer_spans": FieldPolicy.KEEP,
        "manual_tensor_connections": FieldPolicy.KEEP,
        "forward_lineno": FieldPolicy.KEEP,
        "capture_cache_hit": FieldPolicy.KEEP,
        "capture_cache_key": FieldPolicy.KEEP,
        "capture_cache_path": FieldPolicy.KEEP,
        "recording_kept": FieldPolicy.KEEP,
        "streaming_pass_logs": FieldPolicy.DROP,
        "num_streamed_passes": FieldPolicy.KEEP,
        "_activation_hash_cache": FieldPolicy.DROP,
        "save_function_args": FieldPolicy.KEEP,
        "save_gradients": FieldPolicy.KEEP,
        "gradients_to_save": FieldPolicy.KEEP,
        "_gradient_layer_nums_to_save": FieldPolicy.KEEP,
        "gradient_postfunc": FieldPolicy.DROP,
        "gradient_postfunc_repr": FieldPolicy.KEEP,
        "save_raw_gradient": FieldPolicy.KEEP,
        "save_source_context": FieldPolicy.KEEP,
        "save_rng_states": FieldPolicy.KEEP,
        "detect_loops": FieldPolicy.KEEP,
        "verbose": FieldPolicy.KEEP,
        "has_gradients": FieldPolicy.KEEP,
        "mark_input_output_distances": FieldPolicy.KEEP,
        "graph_shape_hash": FieldPolicy.KEEP,
        "_intervention_spec": FieldPolicy.DROP,
        "operation_history": FieldPolicy.KEEP,
        "last_run_ctx": FieldPolicy.DROP,
        "_has_direct_writes": FieldPolicy.KEEP,
        "_warned_direct_write": FieldPolicy.DROP,
        "_warned_mutate_in_place": FieldPolicy.DROP,
        "_spec_revision": FieldPolicy.KEEP,
        "_activation_recipe_revision": FieldPolicy.KEEP,
        "_append_sequence_id": FieldPolicy.KEEP,
        "_last_hook_handle_ids": FieldPolicy.DROP,
        "run_state": FieldPolicy.KEEP,
        "is_appended": FieldPolicy.KEEP,
        "relationship_evidence": FieldPolicy.KEEP,
        "_output_container_specs_by_raw_label": FieldPolicy.KEEP,
        "layer_list": FieldPolicy.KEEP,
        "layer_dict_main_keys": FieldPolicy.KEEP,
        "layer_dict_all_keys": FieldPolicy.KEEP,
        "layer_logs": FieldPolicy.KEEP,
        "layer_labels": FieldPolicy.KEEP,
        "layer_labels_w_pass": FieldPolicy.KEEP,
        "layer_labels_no_pass": FieldPolicy.KEEP,
        "layer_num_passes": FieldPolicy.KEEP,
        "_raw_layer_dict": FieldPolicy.KEEP,
        "_raw_layer_labels_list": FieldPolicy.KEEP,
        "_layer_nums_to_save": FieldPolicy.KEEP,
        "_layer_counter": FieldPolicy.KEEP,
        "num_operations": FieldPolicy.KEEP,
        "_raw_layer_type_counter": FieldPolicy.KEEP,
        "_unsaved_layers_lookup_keys": FieldPolicy.KEEP,
        "_raw_to_final_layer_labels": FieldPolicy.KEEP,
        "_final_to_raw_layer_labels": FieldPolicy.KEEP,
        "_lookup_keys_to_layer_num_dict": FieldPolicy.KEEP,
        "_layer_num_to_lookup_keys_dict": FieldPolicy.KEEP,
        "input_layers": FieldPolicy.KEEP,
        "output_layers": FieldPolicy.KEEP,
        "buffer_layers": FieldPolicy.KEEP,
        "buffer_num_passes": FieldPolicy.KEEP,
        "_buffer_accessor": FieldPolicy.DROP,
        "internally_initialized_layers": FieldPolicy.KEEP,
        "_layers_where_internal_branches_merge_with_input": FieldPolicy.KEEP,
        "internally_terminated_layers": FieldPolicy.KEEP,
        "internally_terminated_bool_layers": FieldPolicy.KEEP,
        "conditional_branch_edges": FieldPolicy.KEEP,
        "conditional_events": FieldPolicy.KEEP,
        "conditional_arm_edges": FieldPolicy.KEEP,
        "conditional_edge_passes": FieldPolicy.KEEP,
        "layers_with_saved_activations": FieldPolicy.KEEP,
        "orphan_layers": FieldPolicy.KEEP,
        "unlogged_layers": FieldPolicy.KEEP,
        "layers_with_saved_gradients": FieldPolicy.KEEP,
        "_saved_gradients_set": FieldPolicy.DROP,
        "layers_with_params": FieldPolicy.KEEP,
        "equivalent_operations": FieldPolicy.KEEP,
        "total_activation_memory": FieldPolicy.KEEP,
        "total_autograd_saved_bytes": FieldPolicy.KEEP,
        "num_tensors_saved": FieldPolicy.KEEP,
        "saved_activation_memory": FieldPolicy.KEEP,
        "param_logs": FieldPolicy.KEEP,
        "total_param_tensors": FieldPolicy.KEEP,
        "total_param_layers": FieldPolicy.KEEP,
        "total_params": FieldPolicy.KEEP,
        "total_params_trainable": FieldPolicy.KEEP,
        "total_params_frozen": FieldPolicy.KEEP,
        "total_params_memory": FieldPolicy.KEEP,
        "_mod_pass_num": FieldPolicy.DROP,
        "_mod_pass_labels": FieldPolicy.DROP,
        "_mod_entered": FieldPolicy.DROP,
        "_mod_exited": FieldPolicy.DROP,
        "_module_build_data": FieldPolicy.DROP,
        "_module_logs": FieldPolicy.DROP,
        "_module_metadata": FieldPolicy.DROP,
        "_module_forward_args": FieldPolicy.DROP,
        "_param_logs_by_module": FieldPolicy.DROP,
        "_pre_forward_rng_states": FieldPolicy.DROP,
        "_activation_writer": FieldPolicy.DROP,
        "_keep_activations_in_memory": FieldPolicy.DROP,
        "_keep_gradients_in_memory": FieldPolicy.DROP,
        "_defer_streaming_bundle_finalization": FieldPolicy.DROP,
        "_activation_sink": FieldPolicy.DROP,
        "_in_exhaustive_pass": FieldPolicy.DROP,
        "pass_start_time": FieldPolicy.KEEP,
        "pass_end_time": FieldPolicy.KEEP,
        "time_setup": FieldPolicy.KEEP,
        "time_forward_pass": FieldPolicy.KEEP,
        "time_cleanup": FieldPolicy.KEEP,
        "time_function_calls": FieldPolicy.KEEP,
        "has_backward_log": FieldPolicy.KEEP,
        "grad_fn_logs": FieldPolicy.KEEP,
        "grad_fn_order": FieldPolicy.KEEP,
        "backward_root_grad_fn_id": FieldPolicy.KEEP,
        "backward_num_passes": FieldPolicy.KEEP,
        "backward_peak_memory_bytes": FieldPolicy.KEEP,
        "backward_memory_backend": FieldPolicy.KEEP,
    }

    def __init__(
        self,
        model_name: str,
        output_device: str = "same",
        activation_postfunc: Optional[ActivationPostfunc] = None,
        gradient_postfunc: Optional[GradientPostfunc] = None,
        save_raw_activation: bool = True,
        save_raw_gradient: bool = True,
        keep_unsaved_layers: bool = True,
        save_function_args: bool = False,
        save_gradients: bool = False,
        gradients_to_save: Any = "all",
        detach_saved_tensors: bool = False,
        mark_input_output_distances: bool = True,
        num_context_lines: int = 7,
        optimizer=None,
        save_source_context: bool = False,
        save_rng_states: bool = False,
        detect_loops: bool = True,
        verbose: bool = False,
        train_mode: bool = False,
        module_filter_fn: Callable[[Any], bool] | None = None,
        emit_nvtx: bool = False,
    ):
        """Initialise a fresh ModelLog for a new logging session.

        Args:
            model_name: Human-readable name of the model being logged.
            output_device: Device to move saved activations to ("same" keeps original device).
            activation_postfunc: Optional function applied to each tensor before saving.
            gradient_postfunc: Optional function applied to each gradient before saving.
            save_raw_activation: Whether raw activations are retained when a postfunc is set.
            save_raw_gradient: Whether raw gradients are retained when a postfunc is set.
            keep_unsaved_layers: If False, layers without saved activations are removed
                from the final log (but still logged during the pass).
            save_function_args: Whether to deep-copy each operation's input arguments.
            save_gradients: Whether to register gradient hooks for backward pass.
            gradients_to_save: Which layer gradients should be saved.
            detach_saved_tensors: Whether to detach saved tensors from the autograd graph.
            mark_input_output_distances: Whether to compute BFS distances from
                inputs/outputs for each layer.
            num_context_lines: Number of source-code context lines to capture
                around each function call (used by FuncCallLocation).
            optimizer: Optional torch optimizer, used to annotate which params
                have optimizers attached.
            verbose: If True, print timed progress messages at each major pipeline stage.
            train_mode: Session-time flag for training-compatible activation retention.
                Portable bundle load restores the default ``False`` value.
            emit_nvtx: Whether decorated torch operations should emit NVTX ranges.
        """
        # Callables are effectively immutable - deepcopy is unnecessary.

        # General info
        self.name: str | None = None
        self.model_name = model_name
        self.num_context_lines = num_context_lines
        self._optimizer = optimizer
        self.io_format_version = IO_FORMAT_VERSION
        # _pass_finished is the master behavioural switch: False during logging,
        # True after postprocessing.  Many methods (len, getitem, str, iter)
        # branch on this flag to choose raw-barcode vs final-label access.
        # It intentionally persists across the fast pass so fast-path
        # postprocessing can use the exhaustive pass's lookup dicts.
        self._pass_finished = False
        # "exhaustive" captures all metadata; "fast" reuses exhaustive-pass
        # structure, only re-capturing tensor contents.
        self.logging_mode: Literal["exhaustive", "fast", "predicate"] = "exhaustive"
        self._all_layers_logged = False
        self._all_layers_saved = False
        self.keep_unsaved_layers = keep_unsaved_layers
        self.intervention_ready = False
        self.capture_full_args = False
        self.activation_postfunc = activation_postfunc
        self.activation_postfunc_repr = (
            repr(activation_postfunc) if activation_postfunc is not None else None
        )
        self.save_raw_activation = save_raw_activation
        self.input_metadata: Dict[str, Any] = {}
        self.gradient_postfunc = gradient_postfunc
        self.gradient_postfunc_repr = (
            repr(gradient_postfunc) if gradient_postfunc is not None else None
        )
        self.save_raw_gradient = save_raw_gradient
        self._source_code_blob: dict[str, str] = {}
        self._source_model_ref: weakref.ReferenceType[nn.Module] | None = None
        self.parent_run: weakref.ReferenceType["ModelLog"] | None = None
        self.source_model_id: int | None = None
        self.source_model_class: str | None = None
        self.weight_fingerprint_at_capture: str | None = None
        self.weight_fingerprint_full: str | None = None
        self.input_id_at_capture: int | None = None
        self.input_shape_hash: str | None = None
        self.current_function_call_barcode = None
        self.random_seed_used = None
        self.output_device = output_device
        self.detach_saved_tensors = detach_saved_tensors
        self.train_mode = train_mode
        self.module_filter_fn = module_filter_fn
        self.emit_nvtx = emit_nvtx
        self.raise_on_nan: bool = False
        self.capture_kpis: Dict[str, Any] = {}
        self.manual_tensor_connections: List[Tuple[str, str]] = []
        self.forward_lineno: int | None = None
        self.capture_cache_hit: bool = False
        self.capture_cache_key: str | None = None
        self.capture_cache_path: str | None = None
        self.recording_kept: bool = True
        self.streaming_pass_logs: List["ModelLog"] = []
        self.num_streamed_passes: int = 1
        self._activation_hash_cache: Dict[str, Tuple[str, torch.Tensor]] = {}
        self.save_function_args = save_function_args
        self.save_gradients = save_gradients
        self.gradients_to_save = gradients_to_save
        self.save_source_context = save_source_context
        self.save_rng_states = save_rng_states
        self.detect_loops = detect_loops
        self.verbose = verbose
        self.has_gradients = False
        self.mark_input_output_distances = mark_input_output_distances
        self.graph_shape_hash: str | None = None
        self._intervention_spec: InterventionSpec | None = InterventionSpec()
        self.operation_history: list[Any] = []
        self.observer_spans: list[dict[str, Any]] = list(_state._active_record_spans)
        self.report_values: dict[str, Any] = {}
        self.last_run_ctx: Any | None = None
        self._has_direct_writes = False
        self._warned_direct_write = False
        self._warned_mutate_in_place = False
        self._spec_revision = 0
        self._activation_recipe_revision = 0
        self._append_sequence_id = 0
        self._last_hook_handle_ids: tuple[str, ...] = ()
        self.run_state = RunState.PRISTINE
        self.is_appended = False
        self.relationship_evidence: dict[str, Relationship] = {
            "model": Relationship.UNKNOWN,
            "weights": Relationship.UNKNOWN,
            "input": Relationship.UNKNOWN,
            "graph": Relationship.UNKNOWN,
        }
        self._output_container_specs_by_raw_label: dict[str, Any] = {}
        self._activation_writer: Optional["BundleStreamWriter"] = None
        self._keep_activations_in_memory: bool = True
        self._keep_gradients_in_memory: bool = True
        self._defer_streaming_bundle_finalization: bool = False
        self._activation_sink: Optional[Callable[[str, torch.Tensor], None]] = None
        self._in_exhaustive_pass: bool = True

        # Model structure info (computed @properties: is_recurrent,
        # max_recurrent_loops, is_branching, has_conditional_branching)

        # Tensor Tracking - post-processed (populated after _pass_finished=True):
        self.layer_list: List[LayerPassLog] = []  # ordered list of all layer passes
        self.layer_dict_main_keys: Dict[str, LayerPassLog] = OrderedDict()  # primary label -> entry
        self.layer_dict_all_keys: Dict[str, LayerPassLog] = (
            OrderedDict()
        )  # all lookup keys -> entry
        self.layer_logs: Dict[str, LayerLog] = OrderedDict()  # no-pass label -> aggregate LayerLog
        self.layer_labels: List[str] = []  # primary labels in execution order
        self.layer_labels_w_pass: List[str] = []  # pass-qualified labels (e.g. "conv2d_1_1:1")
        self.layer_labels_no_pass: List[str] = []  # pass-stripped labels (e.g. "conv2d_1_1")
        self.layer_num_passes: Dict[str, int] = OrderedDict()  # no-pass label -> pass count
        # Tensor Tracking - raw (populated during the forward pass, before postprocessing):
        self._raw_layer_dict: Dict[str, LayerPassLog] = OrderedDict()  # raw barcode -> entry
        self._raw_layer_labels_list: List[str] = []
        self._layer_nums_to_save: List[int] = []  # ordinal positions of layers to save
        self._gradient_layer_nums_to_save: List[int] | str = []
        self._layer_counter: int = 0  # monotonic counter for operation numbering
        self.num_operations: int = 0  # total operations after postprocessing
        self._raw_layer_type_counter: Dict[str, int] = defaultdict(lambda: 0)
        self._unsaved_layers_lookup_keys: Set[str] = (
            set()
        )  # populated but never consulted (known unused)

        # Mapping between raw barcodes and final human-readable labels
        # (populated during postprocessing's label-assignment step):
        self._raw_to_final_layer_labels: Dict[str, str] = {}
        self._final_to_raw_layer_labels: Dict[str, str] = {}
        self._lookup_keys_to_layer_num_dict: Dict[str, int] = {}
        self._layer_num_to_lookup_keys_dict: Dict[int, List[str]] = defaultdict(list)

        # Special Layers:
        self.input_layers: List[str] = []
        self.output_layers: List[str] = []
        self.buffer_layers: List[str] = []
        self.buffer_num_passes: Dict = {}
        self._buffer_accessor = None
        self.internally_initialized_layers: List[str] = []
        self._layers_where_internal_branches_merge_with_input: List[str] = []
        self.internally_terminated_layers: List[str] = []
        self.internally_terminated_bool_layers: List[str] = []
        self.conditional_branch_edges: List[Tuple[str, str]] = []
        self.conditional_events: List[ConditionalEvent] = []
        self.conditional_arm_edges: Dict[Tuple[int, str], List[Tuple[str, str]]] = {}
        self.conditional_edge_passes: Dict[Tuple[str, str, int, str], List[int]] = {}
        self.layers_with_saved_activations: List[str] = []
        self.orphan_layers: List[str] = []
        self.unlogged_layers: List[str] = []
        self.layers_with_saved_gradients: List[str] = []
        self._saved_gradients_set: set = set()
        self.layers_with_params: Dict[str, List] = defaultdict(list)
        # Maps operation_equivalence_type -> set of layer labels that share
        # that equivalence type (populated by loop_detection.py).
        self.equivalent_operations: Dict[str, set] = defaultdict(set)

        # Aggregate tensor statistics (computed during postprocessing):
        self.total_activation_memory: int = 0
        self.total_autograd_saved_bytes: Optional[int] = None
        self.num_tensors_saved: int = 0  # layers with has_saved_activations=True
        self.saved_activation_memory: int = 0

        # Param info:
        self.param_logs: "ParamAccessor" = ParamAccessor({})
        self.total_param_tensors: int = 0
        self.total_param_layers: int = 0
        self.total_params: int = 0
        self.total_params_trainable: int = 0
        self.total_params_frozen: int = 0
        self.total_params_memory: int = 0

        # Session-scoped per-module tracking dicts (keyed by id(module)).
        # These replace the old tl_* attrs that were set directly on nn.Module
        # instances. Lives on ModelLog so they're GC'd with the log - no cleanup
        # iteration over modules needed.
        self._mod_pass_num: Dict[int, int] = {}  # id(module) -> pass count
        self._mod_pass_labels: Dict[int, list] = {}  # id(module) -> [(addr, pass_num), ...]
        self._mod_entered: Dict[int, list] = {}  # id(module) -> [raw_label, ...]
        self._mod_exited: Dict[int, list] = {}  # id(module) -> [raw_label, ...]

        # Transient module build data (consumed by _build_module_logs, then cleared):
        self._module_build_data: Dict = _init_module_build_data()

        # Structured module info:
        self._module_logs: ModuleAccessor = ModuleAccessor({})

        # Temporary storage for module metadata capture (consumed by _build_module_logs):
        self._module_metadata: Dict = {}
        self._module_forward_args: Dict = {}

        # Time elapsed:
        self.pass_start_time: float = 0
        self.pass_end_time: float = 0
        self.time_setup: float = 0
        self.time_forward_pass: float = 0
        self.time_cleanup: float = 0
        self.time_function_calls: float = 0
        self.has_backward_log: bool = False
        self.grad_fn_logs: Dict[int, GradFnLog] = OrderedDict()
        self.grad_fn_order: List[int] = []
        self.backward_root_grad_fn_id: int | None = None
        self.backward_num_passes: int = 0
        self.backward_peak_memory_bytes: int = 0
        self.backward_memory_backend: str = "unknown"
        _state._register_log(self)

    # ********************************************
    # ************ Built-in Methods **************
    # ********************************************

    def __len__(self):
        """Number of layer-pass entries. Uses final list after postprocessing, raw dict during logging."""
        if self._pass_finished:
            return len(self.layer_list)
        else:
            return len(self._raw_layer_dict)

    def __getitem__(self, ix) -> LayerPassLog:
        """Returns an object logging a model layer given an index. If the pass is finished,
        it'll do this intelligently; if not, it simply queries based on the layer's raw barcode.

        Args:
            ix: desired index

        Returns:
            Tensor log entry object with info about specified layer.
        """
        if self._pass_finished:
            return _getitem_after_pass(self, ix)
        else:
            return _getitem_during_pass(self, ix)

    def find_sites(self, query: Any, *, strict: bool = False, max_fanout: int = 8) -> Any:
        """Return a table of intervention sites matching a query.

        Parameters
        ----------
        query:
            Selector, target spec, frozen target spec, or non-strict bare string.
        strict:
            Whether to reject non-portable query forms.
        max_fanout:
            Maximum number of matching sites.

        Returns
        -------
        SiteTable
            Ordered table of matching layer-pass records.
        """

        from ..intervention.resolver import find_sites

        return find_sites(self, query, strict=strict, max_fanout=max_fanout)

    def resolve_sites(self, query: Any, *, strict: bool = False, max_fanout: int = 8) -> Any:
        """Resolve intervention sites matching a query.

        Parameters
        ----------
        query:
            Selector, target spec, frozen target spec, or non-strict bare string.
        strict:
            Whether to reject non-portable query forms.
        max_fanout:
            Maximum number of matching sites.

        Returns
        -------
        SiteTable
            Ordered table of matching layer-pass records.
        """

        from ..intervention.resolver import resolve_sites

        return resolve_sites(self, query, strict=strict, max_fanout=max_fanout)

    def find_layers(self, query: str, *, limit: int = 10) -> List[str]:
        """Return layer labels matching a fuzzy query.

        Parameters
        ----------
        query:
            Layer-label substring or approximate layer name.
        limit:
            Maximum number of labels to return.

        Returns
        -------
        List[str]
            Matching no-pass layer labels in execution order, followed by close
            fuzzy matches when substring matches are insufficient.
        """

        query_text = str(query).lower()
        labels = list(self.layer_labels_no_pass)
        substring_matches = [label for label in labels if query_text in label.lower()]
        if len(substring_matches) >= limit:
            return substring_matches[:limit]
        fuzzy_matches = difflib.get_close_matches(str(query), labels, n=limit, cutoff=0.25)
        result = substring_matches[:]
        for label in fuzzy_matches:
            if label not in result:
                result.append(label)
            if len(result) >= limit:
                break
        return result

    def suggest(self, query: str, *, limit: int = 3) -> List[str]:
        """Return a short list of suggested layer labels for a failed query.

        Parameters
        ----------
        query:
            Layer query that did not resolve.
        limit:
            Maximum number of suggestions.

        Returns
        -------
        List[str]
            Suggested no-pass layer labels.
        """

        return self.find_layers(query, limit=limit)

    @property
    def unsupported_ops(self) -> _CallableList:
        """Return operations TorchLens could not represent specially.

        Returns
        -------
        _CallableList
            Best-effort list of unsupported operation names. The current
            exhaustive capture path records all observed tensor-producing ops,
            so this returns an empty list unless future capture metadata marks
            a layer with ``unsupported_op=True``.
        """

        unsupported: set[str] = set()
        for layer in self.layer_list:
            if getattr(layer, "unsupported_op", False):
                op_name = getattr(layer, "func_name", None) or getattr(layer, "layer_type", "")
                unsupported.add(str(op_name))
        return _CallableList(sorted(unsupported))

    @property
    def uncalled_modules(self) -> _CallableList:
        """Return registered modules that were not exercised in the captured pass.

        Returns
        -------
        _CallableList
            Module addresses present on the source model but absent from the
            captured module accessor. Returns an empty list when the source
            model is no longer available.
        """

        source_ref = getattr(self, "_source_model_ref", None)
        model = source_ref() if source_ref is not None else None
        if model is None:
            return _CallableList()
        registered = {address or "self" for address, _module in model.named_modules()}
        called = set(getattr(self._module_logs, "_dict", {}).keys())
        called.update(getattr(self._module_logs, "_alias_dict", {}).keys())
        return _CallableList(sorted(registered - called))

    def save_intervention(
        self,
        path: str | Path,
        *,
        level: str = "executable_with_callables",
        allow_direct_writes: bool = False,
        overwrite: bool = False,
    ) -> None:
        """Save this log's intervention recipe to a ``.tlspec`` directory.

        Parameters
        ----------
        path:
            Destination ``.tlspec`` directory path.
        level:
            Save level: ``"audit"``, ``"executable_with_callables"``, or
            ``"portable"``.
        allow_direct_writes:
            Whether executable saves may proceed after direct activation writes.
        overwrite:
            Whether an existing destination may be replaced.
        """

        from ..intervention.save import save_intervention

        save_intervention(
            self,
            path,
            level=level,
            allow_direct_writes=allow_direct_writes,
            overwrite=overwrite,
        )

    @cached_property
    def intervention_spec(self) -> FrozenInterventionSpec:
        """Return an immutable snapshot of this log's intervention recipe.

        Returns
        -------
        FrozenInterventionSpec
            Frozen public view of the current mutable intervention spec.
        """

        return self._ensure_intervention_spec().freeze()

    def set(
        self,
        site: Any,
        value: Any,
        *,
        strict: bool = False,
        confirm_mutation: bool = False,
    ) -> "ModelLog":
        """Set a site activation recipe without propagating it.

        Parameters
        ----------
        site:
            Selector-like target for the activation to replace.
        value:
            Static replacement tensor or one-shot callable accepting the
            matched activation and returning a replacement tensor.
        strict:
            Whether site resolution should reject non-portable selectors.
        confirm_mutation:
            Suppress the once-per-root mutate-in-place warning for callers that
            intentionally mutate this log.

        Returns
        -------
        ModelLog
            This model log, with a stale intervention recipe.
        """

        self._warn_if_root_mutation(confirm_mutation=confirm_mutation)
        self._validate_intervention_site(site, strict=strict)
        metadata = {"created_by": "set_callable_one_shot"} if callable(value) else {}
        self._ensure_intervention_spec().add_set(
            self._target_spec_from_site(site, strict=strict),
            value,
            metadata=metadata,
        )
        self._mark_intervention_spec_mutated()
        self._record_operation(
            "set",
            site=repr(site),
            value_kind=type(value).__name__,
            strict=strict,
            callable=callable(value),
        )
        return self

    def attach_hooks(
        self,
        hooks_or_site: Any,
        hook: Any = None,
        *extra_hooks: Any,
        strict: bool = False,
        prepend: bool = False,
        confirm_mutation: bool = False,
    ) -> Any:
        """Attach sticky hooks to the current intervention spec.

        Parameters
        ----------
        hooks_or_site:
            Mapping/list batch input or selector-like site.
        hook:
            Optional hook for the ``(site, hook)`` input shape.
        *extra_hooks:
            Additional hooks to compose at ``hooks_or_site`` in left-to-right order.
        strict:
            Whether site resolution should reject non-portable selectors.
        prepend:
            Whether new sticky hooks should run before existing sticky hooks.
        confirm_mutation:
            Suppress the once-per-root mutate-in-place warning for callers that
            intentionally mutate this log.

        Returns
        -------
        Any
            Scoped removable hook handle.
        """

        self._warn_if_root_mutation(confirm_mutation=confirm_mutation)
        from ..intervention.errors import HookSignatureError
        from ..intervention.handles import HookHandle
        from ..intervention.hooks import normalize_hook_plan

        if extra_hooks:
            if hook is None:
                raise HookSignatureError("extra hooks require an initial hook argument.")
            entries = normalize_hook_plan(
                [(hooks_or_site, hook_like) for hook_like in (hook, *extra_hooks)]
            )
        else:
            entries = normalize_hook_plan(hooks_or_site, hook)
        for entry in entries:
            self._validate_intervention_site(entry.site_target, strict=strict)
        spec = self._ensure_intervention_spec()
        handle_ids: list[str] = []
        for entry in entries:
            handle_id = f"hook-{uuid.uuid4().hex}"
            handle_ids.append(handle_id)
            metadata = dict(entry.metadata)
            spec.add_hook(
                self._target_spec_from_site(entry.site_target, strict=strict),
                entry.helper_spec if entry.helper_spec is not None else entry.normalized_callable,
                helper=entry.helper_spec,
                handle=handle_id,
                metadata=metadata,
                prepend=prepend,
            )
        self._mark_intervention_spec_mutated()
        self._record_operation(
            "attach_hooks",
            hook_count=len(entries),
            sites=tuple(repr(entry.site_target) for entry in entries),
            strict=strict,
            prepend=prepend,
            handles=tuple(handle_ids),
        )
        self._last_hook_handle_ids = tuple(handle_ids)
        scoped_handle = HookHandle(self, tuple(handle_ids), confirm_mutation=confirm_mutation)
        if extra_hooks:
            return scoped_handle
        return self

    def remove(self) -> None:
        """Remove the most recent legacy-returned hook attachment.

        Returns
        -------
        None
            Hook specs attached by the last single-hook ``attach_hooks`` call
            are detached.
        """

        for handle_id in self._last_hook_handle_ids:
            self.detach_hooks(handle=handle_id, confirm_mutation=True)
        self._last_hook_handle_ids = ()

    def __enter__(self) -> "ModelLog":
        """Enter a legacy scoped hook attachment.

        Returns
        -------
        ModelLog
            This log, acting as the most recent hook handle.
        """

        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Clean up a legacy scoped hook attachment.

        Parameters
        ----------
        exc_type:
            Exception type, if the body raised.
        exc:
            Exception value, if the body raised.
        traceback:
            Exception traceback, if the body raised.
        """

        self.remove()

    def detach_hooks(
        self,
        site: Any = None,
        handle: Any = None,
        *,
        strict: bool = False,
        confirm_mutation: bool = False,
    ) -> "ModelLog":
        """Detach sticky hooks by site or handle.

        Parameters
        ----------
        site:
            Optional selector-like target. When provided, all sticky hooks for
            that target are removed.
        handle:
            Optional hook handle returned by ``attach_hooks``.
        strict:
            Whether no-op detach requests should raise.
        confirm_mutation:
            Suppress the once-per-root mutate-in-place warning for callers that
            intentionally mutate this log.

        Returns
        -------
        ModelLog
            This model log, with a stale recipe if hooks were removed.
        """

        self._warn_if_root_mutation(confirm_mutation=confirm_mutation)
        from ..intervention.errors import SpecMutationError

        if site is None and handle is None:
            if strict:
                raise SpecMutationError("detach_hooks requires a site or handle in strict mode.")
            return self

        target_spec = None
        if site is not None:
            self._validate_intervention_site(site, strict=strict)
            target_spec = self._target_spec_from_site(site, strict=strict)

        handle_values = (
            tuple(getattr(handle, "handle_ids", (str(handle),))) if handle is not None else (None,)
        )
        removed = 0
        for handle_value in handle_values:
            removed += self._ensure_intervention_spec().remove_hook(
                site_target=target_spec,
                handle=handle_value,
            )
        if removed == 0 and strict:
            raise SpecMutationError("detach_hooks did not match any sticky hooks.")
        if removed > 0:
            self._mark_intervention_spec_mutated()
        self._record_operation(
            "detach_hooks",
            site=repr(site) if site is not None else None,
            handle=str(handle) if handle is not None else None,
            removed=removed,
            strict=strict,
        )
        return self

    def clear_hooks(self, *, confirm_mutation: bool = False) -> "ModelLog":
        """Clear all sticky hooks from the current intervention spec.

        Parameters
        ----------
        confirm_mutation:
            Suppress the once-per-root mutate-in-place warning for callers that
            intentionally mutate this log.

        Returns
        -------
        ModelLog
            This model log with hook specs cleared and marked stale.
        """

        self._warn_if_root_mutation(confirm_mutation=confirm_mutation)
        self._ensure_intervention_spec().clear()
        self._mark_intervention_spec_mutated()
        self._record_operation("clear_hooks")
        return self

    def do(
        self,
        hooks_or_site: Any,
        value_or_hook: Any = None,
        *,
        model: nn.Module | None = None,
        x: Any = None,
        engine: str | MissingType = MISSING,
        confirm_mutation: bool | MissingType = MISSING,
        strict: bool | MissingType = MISSING,
        intervention: InterventionOptions | None = None,
    ) -> "ModelLog":
        """Apply an intervention and dispatch to replay, rerun, or set-only.

        Parameters
        ----------
        hooks_or_site:
            Mapping/list batch input or selector-like site.
        value_or_hook:
            Optional hook for the ``(site, hook)`` input shape.
        model:
            Model required when ``engine="rerun"``.
        x:
            Input required when ``engine="rerun"``.
        engine:
            ``"auto"``, ``"replay"``, ``"rerun"``, or ``"set_only"``.
        confirm_mutation:
            Suppress the once-per-root mutate-in-place warning for callers that
            intentionally mutate this log.
        strict:
            Whether selector and propagation checks should raise.

        Returns
        -------
        ModelLog
            This model log after the selected propagation engine runs.
        """

        from ..intervention.errors import EngineDispatchError

        intervention_options = merge_intervention_options(
            intervention=intervention,
            engine=engine,
            confirm_mutation=confirm_mutation,
            strict=strict,
        )
        engine_value = intervention_options.engine
        confirm_mutation_value = intervention_options.confirm_mutation
        strict_value = intervention_options.strict

        if engine_value not in {"auto", "replay", "rerun", "set_only"}:
            raise ValueError(
                "do(..., engine=...) must be 'auto', 'replay', 'rerun', or 'set_only'."
            )

        selected_engine = self._select_do_engine(engine_value, model=model, x=x)
        if selected_engine == "rerun":
            if model is None:
                raise EngineDispatchError("do(..., engine='rerun') requires model= and x=.")
            self._validate_supplied_model_matches_capture(model)
        mutation_kind = self._apply_do_mutation(
            hooks_or_site,
            value_or_hook,
            engine=selected_engine,
            strict=strict_value,
            confirm_mutation=confirm_mutation_value,
        )
        self._record_operation(
            "do",
            mutation_kind=mutation_kind,
            engine=selected_engine,
            requested_engine=engine_value,
            model_supplied=model is not None,
            x_supplied=x is not None,
            strict=strict_value,
        )

        if selected_engine == "set_only":
            return self
        if selected_engine == "replay":
            return self.replay(strict=strict_value)
        assert model is not None
        return self.rerun(model, x, strict=strict_value)

    def fork(self, name: str | None = None) -> "ModelLog":
        """Create a copy-on-write intervention fork of this log.

        Parameters
        ----------
        name:
            Optional name for the forked log.

        Returns
        -------
        ModelLog
            Forked model log.
        """

        fork = self._fork_model_log(name=name)
        self._record_operation("fork", source_id=id(self), name=fork.name)
        return fork

    def _record_operation(self, op: str, **payload: Any) -> None:
        """Append a structured operation record to ``operation_history``.

        Parameters
        ----------
        op:
            Operation name.
        **payload:
            Operation-specific metadata.

        Returns
        -------
        None
            The history list is mutated in place.
        """

        self.operation_history.append(
            {
                "op": op,
                "spec_revision": self._spec_revision,
                "timestamp": time.monotonic(),
                **payload,
            }
        )

    def _warn_if_root_mutation(self, *, confirm_mutation: bool) -> None:
        """Emit the once-per-root mutate-in-place warning when appropriate.

        Parameters
        ----------
        confirm_mutation:
            Whether the caller explicitly accepted in-place mutation.
        """

        if confirm_mutation or self.parent_run is not None or self._warned_mutate_in_place:
            return
        from ..intervention.errors import MutateInPlaceWarning
        from ..options import suppress_mutate_warnings

        if suppress_mutate_warnings.is_suppressed:
            return
        warnings.warn(
            "MutateInPlaceWarning: ModelLog mutators modify root logs in place. "
            "Use log.fork(...) for isolated edits or pass confirm_mutation=True.",
            MutateInPlaceWarning,
            stacklevel=3,
        )
        self._warned_mutate_in_place = True

    def _select_do_engine(self, engine: str, *, model: nn.Module | None, x: Any) -> str:
        """Resolve the concrete ``do`` engine from caller arguments.

        Parameters
        ----------
        engine:
            Requested engine name.
        model:
            Optional model supplied for rerun.
        x:
            Optional input supplied for rerun.

        Returns
        -------
        str
            Concrete engine name.

        Raises
        ------
        EngineDispatchError
            If the engine cannot be inferred from an incomplete model/input pair.
        """

        from ..intervention.errors import EngineDispatchError

        if engine != "auto":
            if engine == "rerun" and (model is None or x is None):
                raise EngineDispatchError(
                    "do(..., engine='rerun') requires both model= and x=. "
                    "Pass both, or use engine='replay' if full rerun is not intended."
                )
            return engine
        if (model is None) != (x is None):
            raise EngineDispatchError(
                "do(engine='auto') needs both model= and x= for rerun, or neither for "
                "replay. Pass both, or use engine='replay' if rerun is not intended."
            )
        return "rerun" if model is not None else "replay"

    def _apply_do_mutation(
        self,
        hooks_or_site: Any,
        value_or_hook: Any,
        *,
        engine: str,
        strict: bool,
        confirm_mutation: bool,
    ) -> str:
        """Apply the mutation part of ``do`` and report its kind.

        Parameters
        ----------
        hooks_or_site:
            Mapping/list batch input or selector-like site.
        value_or_hook:
            Optional value or hook.
        engine:
            Concrete engine selected by ``_select_do_engine``.
        strict:
            Whether selector checks should be strict.
        confirm_mutation:
            Whether root mutation warnings should be suppressed.

        Returns
        -------
        str
            ``"set"`` or ``"attach_hooks"``.
        """

        if engine == "set_only" and value_or_hook is not None:
            self.set(
                hooks_or_site,
                value_or_hook,
                strict=strict,
                confirm_mutation=confirm_mutation,
            )
            return "set"
        if value_or_hook is not None and not callable(value_or_hook):
            self.set(
                hooks_or_site,
                value_or_hook,
                strict=strict,
                confirm_mutation=confirm_mutation,
            )
            return "set"
        self.attach_hooks(
            hooks_or_site,
            value_or_hook,
            strict=strict,
            confirm_mutation=confirm_mutation,
        )
        return "attach_hooks"

    def _validate_supplied_model_matches_capture(self, model: nn.Module) -> None:
        """Validate rerun model evidence against the captured source model.

        Parameters
        ----------
        model:
            Candidate model for rerun.

        Raises
        ------
        ModelMismatchError
            If available class or weight-fingerprint evidence differs.
        """

        from ..intervention.errors import ModelMismatchError
        from ..user_funcs import _fingerprint_model_weights, _qualname_for_model

        expected_class = getattr(self, "source_model_class", None)
        actual_class = _qualname_for_model(model)
        if expected_class is not None and actual_class != expected_class:
            raise ModelMismatchError(
                "Supplied model class does not match captured model class: "
                f"expected {expected_class!r}, got {actual_class!r}."
            )

        expected_fingerprint = getattr(self, "weight_fingerprint_at_capture", None)
        if expected_fingerprint is None:
            return
        actual_fingerprint = _fingerprint_model_weights(model)
        if actual_fingerprint != expected_fingerprint:
            raise ModelMismatchError(
                "Supplied model weight fingerprint does not match captured model weights."
            )

    def _fork_model_log(self, *, name: str | None) -> "ModelLog":
        """Build a forked ModelLog with policy-driven field handling.

        Parameters
        ----------
        name:
            Optional fork name.

        Returns
        -------
        ModelLog
            Forked log whose mutable containers are independent.
        """

        fork = object.__new__(type(self))
        fork_state = {
            field_name: self._fork_model_field(field_name, value)
            for field_name, value in self.__dict__.items()
        }
        fork.__dict__.update(fork_state)
        fork.parent_run = weakref.ref(self)
        fork.name = name or self._next_fork_name()
        fork._intervention_spec = copy.deepcopy(self._ensure_intervention_spec())
        fork.operation_history = copy.deepcopy(self.operation_history)
        fork.relationship_evidence = copy.deepcopy(self.relationship_evidence)
        fork._activation_recipe_revision = self._activation_recipe_revision
        fork._spec_revision = self._spec_revision
        fork.run_state = self.run_state
        fork._warned_mutate_in_place = False
        fork._warned_direct_write = False

        layer_map = fork._fork_layer_passes_from(self)
        fork._rebuild_fork_layer_collections(self, layer_map)
        fork._rebind_fork_owner_refs()
        _state._register_log(fork)
        return fork

    def _next_fork_name(self) -> str:
        """Return a deterministic default fork name for this parent log."""

        base_name = self.name or "model_log"
        fork_count = sum(
            1
            for record in self.operation_history
            if isinstance(record, dict) and record.get("op") == "fork"
        )
        return f"{base_name}_fork_{fork_count + 1}"

    def _fork_model_field(self, field_name: str, value: Any) -> Any:
        """Apply the ModelLog fork policy to a single field.

        Parameters
        ----------
        field_name:
            Field being copied.
        value:
            Current field value.

        Returns
        -------
        Any
            Field value for the fork.
        """

        policy = MODEL_LOG_FORK_POLICY.get(field_name, self._default_fork_policy(value))
        if policy is ForkFieldPolicy.FORK_SHARE:
            return value
        if policy is ForkFieldPolicy.FORK_RECONSTRUCT:
            return None
        return self._copy_fork_value(value)

    def _fork_layer_passes_from(self, parent: "ModelLog") -> dict[int, LayerPassLog]:
        """Fork every LayerPassLog and return an old-object-id map.

        Parameters
        ----------
        parent:
            Parent log whose layer passes are being forked.

        Returns
        -------
        dict[int, LayerPassLog]
            Mapping from ``id(parent_pass)`` to forked pass.
        """

        layer_map: dict[int, LayerPassLog] = {}
        fork_equivalent_operations = self.equivalent_operations
        for parent_pass in parent.layer_list:
            fork_pass = object.__new__(LayerPassLog)
            fork_pass.__dict__.update(
                {
                    field_name: self._fork_layer_pass_field(field_name, value)
                    for field_name, value in parent_pass.__dict__.items()
                }
            )
            fork_pass.source_model_log = self
            eq_type = getattr(fork_pass, "operation_equivalence_type", None)
            if eq_type in fork_equivalent_operations:
                fork_pass.equivalent_operations = fork_equivalent_operations[eq_type]
            object.__setattr__(fork_pass, "_construction_done", True)
            layer_map[id(parent_pass)] = fork_pass
        return layer_map

    def _fork_layer_pass_field(self, field_name: str, value: Any) -> Any:
        """Apply the LayerPassLog fork policy to a single field.

        Parameters
        ----------
        field_name:
            LayerPassLog field being copied.
        value:
            Current field value.

        Returns
        -------
        Any
            Field value for the forked pass.
        """

        if field_name in {"_source_model_log_ref", "parent_layer_log"}:
            return None
        policy = LayerPassLog.FORK_POLICY.get(field_name, self._default_fork_policy(value))
        if policy is ForkFieldPolicy.FORK_SHARE:
            return value
        if policy is ForkFieldPolicy.FORK_RECONSTRUCT:
            return None
        return self._copy_fork_value(value)

    def _rebuild_fork_layer_collections(
        self, parent: "ModelLog", layer_map: dict[int, LayerPassLog]
    ) -> None:
        """Rebuild layer lookup containers so they point at forked passes.

        Parameters
        ----------
        parent:
            Parent log whose containers are being mirrored.
        layer_map:
            Mapping from parent pass object id to forked pass.
        """

        def remap_pass(value: Any) -> Any:
            return layer_map.get(id(value), value)

        self.layer_list = [remap_pass(layer) for layer in parent.layer_list]
        self.layer_dict_main_keys = OrderedDict(
            (key, remap_pass(layer)) for key, layer in parent.layer_dict_main_keys.items()
        )
        self.layer_dict_all_keys = OrderedDict(
            (key, remap_pass(layer)) for key, layer in parent.layer_dict_all_keys.items()
        )
        self._raw_layer_dict = OrderedDict(
            (key, remap_pass(layer)) for key, layer in parent._raw_layer_dict.items()
        )

        fork_layer_logs: dict[str, LayerLog] = OrderedDict()
        for label, parent_layer_log in parent.layer_logs.items():
            fork_layer_log = copy.copy(parent_layer_log)
            fork_layer_log.__dict__ = {
                key: self._copy_fork_value(value)
                for key, value in parent_layer_log.__dict__.items()
            }
            fork_layer_log.source_model_log = self
            fork_layer_log.passes = OrderedDict(
                (pass_num, remap_pass(layer_pass))
                for pass_num, layer_pass in parent_layer_log.passes.items()
            )
            for layer_pass in fork_layer_log.passes.values():
                layer_pass.parent_layer_log = fork_layer_log
            if (
                getattr(fork_layer_log, "operation_equivalence_type", None)
                in self.equivalent_operations
            ):
                fork_layer_log.equivalent_operations = self.equivalent_operations[
                    fork_layer_log.operation_equivalence_type
                ]
            fork_layer_logs[label] = fork_layer_log
        self.layer_logs = fork_layer_logs

    def _rebind_fork_owner_refs(self) -> None:
        """Rebind weak owner references on forked child objects to this fork."""

        for layer_pass in self.layer_list:
            layer_pass.source_model_log = self
            parent_layer_log = self.layer_logs.get(layer_pass.layer_label_no_pass)
            if parent_layer_log is not None:
                layer_pass.parent_layer_log = parent_layer_log
        for layer_log in self.layer_logs.values():
            layer_log.source_model_log = self

    @staticmethod
    def _copy_fork_value(value: Any) -> Any:
        """Copy a fork field while preserving tensor and callable identity.

        Parameters
        ----------
        value:
            Value to copy.

        Returns
        -------
        Any
            Fork-safe copy.
        """

        if isinstance(value, torch.Tensor) or callable(value):
            return value
        try:
            return copy.deepcopy(value)
        except Exception:
            return copy.copy(value)

    @staticmethod
    def _default_fork_policy(value: Any) -> ForkFieldPolicy:
        """Choose a conservative fork policy for fields outside policy tables.

        Parameters
        ----------
        value:
            Field value without an explicit policy.

        Returns
        -------
        ForkFieldPolicy
            Default share/copy decision.
        """

        if isinstance(value, (str, bytes, int, float, bool, type(None), tuple)):
            return ForkFieldPolicy.FORK_SHARE
        if isinstance(value, torch.Tensor) or callable(value):
            return ForkFieldPolicy.FORK_SHARE
        return ForkFieldPolicy.FORK_COPY

    def _recipe_is_clean(self) -> bool:
        """Return whether propagated activations match the current spec revision.

        Returns
        -------
        bool
            ``True`` when the current activation recipe revision equals the
            mutable intervention spec revision.
        """

        return self._spec_revision == self._activation_recipe_revision

    def _ensure_intervention_spec(self) -> InterventionSpec:
        """Return the mutable intervention spec, creating one if needed.

        Returns
        -------
        InterventionSpec
            Mutable intervention recipe owned by this log.
        """

        if self._intervention_spec is None:
            self._intervention_spec = InterventionSpec()
        return self._intervention_spec

    def _mark_intervention_spec_mutated(self) -> None:
        """Invalidate cached frozen views and mark the spec stale.

        Returns
        -------
        None
            This model log is mutated in place.
        """

        self._spec_revision += 1
        self.__dict__.pop("intervention_spec", None)
        self.__dict__.pop("_frozen_intervention_spec", None)
        self.__dict__.pop("_cached_frozen_intervention_spec", None)
        self.run_state = RunState.SPEC_STALE

    def _validate_intervention_site(self, site: Any, *, strict: bool) -> None:
        """Validate that a mutator site resolves on this log.

        Parameters
        ----------
        site:
            Selector-like target to validate.
        strict:
            Whether selector resolution should be strict.

        Returns
        -------
        None
            Raises when the site cannot resolve.
        """

        max_fanout = max(1, len(self.layer_list))
        self.resolve_sites(site, strict=strict, max_fanout=max_fanout)

    def _target_spec_from_site(self, site: Any, *, strict: bool) -> TargetSpec:
        """Convert a selector-like site to a mutable target spec.

        Parameters
        ----------
        site:
            Selector-like target, target spec, or layer pass.
        strict:
            Whether the resulting target should carry strict resolution.

        Returns
        -------
        TargetSpec
            Mutable target spec stored in the intervention recipe.
        """

        if isinstance(site, TargetSpec):
            target = copy.copy(site)
            target.strict = strict or target.strict
            return target
        if hasattr(site, "to_target_spec"):
            target = site.to_target_spec()
            target.strict = strict or target.strict
            return target
        if hasattr(site, "layer_label"):
            return TargetSpec("label", str(site.layer_label), strict=strict)
        return TargetSpec("label", site, strict=strict)

    def __str__(self) -> str:
        """Human-readable summary; delegates to post-pass or mid-pass formatter."""
        if self._pass_finished:
            return _str_after_pass(self)
        else:
            return _str_during_pass(self)

    def __repr__(self) -> str:
        """Short identity-card representation for REPL display."""
        from ..visualization._summary_internal import format_model_repr

        return format_model_repr(self)

    def _repr_html_(self) -> str:
        """Return the notebook HTML representation for this model log.

        Returns
        -------
        str
            HTML fragment for IPython/Jupyter display.

        Falls back to ``repr(self)`` when the notebook extra is unavailable.
        """
        try:
            import IPython  # noqa: F401
        except ImportError:
            return repr(self)

        from html import escape

        layers = len(getattr(self, "layer_logs", {}) or {})
        ops = getattr(self, "num_operations", 0)
        save_level = "all" if getattr(self, "_all_layers_saved", False) else "selected"
        if getattr(self, "num_tensors_saved", 0) == 0:
            save_level = "metadata only"
        nonfinite = self.first_nonfinite()
        nonfinite_summary = (
            "No non-finite saved activations"
            if nonfinite.startswith("No non-finite")
            else escape(nonfinite)
        )
        title = escape(str(getattr(self, "name", None) or self.model_name))
        run_state = escape(str(getattr(getattr(self, "run_state", None), "name", "UNKNOWN")))
        return (
            "<div style='border:1px solid #d0d7de;border-radius:8px;"
            "padding:10px 12px;font-family:system-ui,sans-serif;max-width:560px'>"
            f"<div style='font-weight:700;margin-bottom:6px'>TorchLens ModelLog: {title}</div>"
            "<div style='display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:4px 12px'>"
            f"<div><b>Layers</b>: {layers}</div>"
            f"<div><b>Ops</b>: {ops}</div>"
            f"<div><b>Save level</b>: {escape(save_level)}</div>"
            f"<div><b>Run state</b>: {run_state}</div>"
            "</div>"
            f"<div style='margin-top:8px'><b>NaN/Inf</b>: {nonfinite_summary}</div>"
            "</div>"
        )

    def __iter__(self):
        """Loops through all tensors in the log."""
        if self._pass_finished:
            return iter(self.layer_list)
        else:
            return iter(list(self._raw_layer_dict.values()))

    def save(self, path: str | Path, **kwargs: Any) -> None:
        """Call :func:`torchlens.save` for this model log.

        Warning
        -------
        Portable bundles contain a pickle file. Only load bundles from trusted
        sources. Loading an untrusted bundle can execute arbitrary code.
        """

        from .._io.bundle import save as save_bundle

        save_bundle(self, path, **kwargs)

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> "ModelLog":
        """Call :func:`torchlens.load` and return the loaded model log.

        Warning
        -------
        Portable bundles contain a pickle file. Only load bundles from trusted
        sources. Loading an untrusted bundle can execute arbitrary code.
        """

        from .._io.bundle import load as load_bundle

        loaded = load_bundle(path, **kwargs)
        if not isinstance(loaded, cls):
            raise TypeError(f"ModelLog.load expected a ModelLog, got {type(loaded).__name__}.")
        return loaded

    def __getstate__(self) -> Dict[str, Any]:
        """Return pickle state with non-picklable weakref-backed accessors stripped."""
        state = self.__dict__.copy()
        state["_module_logs"] = None
        state["_buffer_accessor"] = None
        state["_module_build_data"] = None
        state["_source_model_ref"] = None
        state["parent_run"] = None
        state["last_run_ctx"] = None
        state["streaming_pass_logs"] = []
        state["_activation_hash_cache"] = {}
        state["_raw_layer_type_counter"] = dict(self._raw_layer_type_counter)
        state["activation_postfunc_repr"] = (
            repr(self.activation_postfunc) if self.activation_postfunc is not None else None
        )
        state["io_format_version"] = IO_FORMAT_VERSION
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore pickle state and rebuild weakref-backed links."""
        read_io_format_version(state, cls_name=type(self).__name__)
        default_fill_state(
            state,
            defaults={
                **_MODEL_LOG_DEFAULT_FILL,
                "io_format_version": IO_FORMAT_VERSION,
                "activation_postfunc_repr": None,
                "save_raw_activation": True,
                "input_metadata": {},
                "gradient_postfunc": None,
                "gradient_postfunc_repr": None,
                "save_raw_gradient": True,
                "gradients_to_save": "all",
                "_gradient_layer_nums_to_save": [],
                "has_backward_log": False,
                "grad_fn_logs": OrderedDict(),
                "grad_fn_order": [],
                "backward_root_grad_fn_id": None,
                "backward_num_passes": 0,
                "backward_peak_memory_bytes": 0,
                "backward_memory_backend": "unknown",
                "total_autograd_saved_bytes": None,
                "_buffer_accessor": None,
                "_module_logs": None,
                "_module_build_data": None,
                "_activation_writer": None,
                "_keep_activations_in_memory": True,
                "_keep_gradients_in_memory": True,
                "_defer_streaming_bundle_finalization": False,
                "_activation_sink": None,
                "_in_exhaustive_pass": False,
                "_source_code_blob": {},
                "_source_model_ref": None,
                "train_mode": False,
                "module_filter_fn": None,
                "raise_on_nan": False,
                "capture_kpis": {},
                "report_values": {},
                "observer_spans": [],
                "manual_tensor_connections": [],
                "forward_lineno": None,
                "capture_cache_hit": False,
                "capture_cache_key": None,
                "capture_cache_path": None,
                "recording_kept": True,
                "streaming_pass_logs": [],
                "num_streamed_passes": 1,
                "_activation_hash_cache": {},
                "_last_hook_handle_ids": (),
            },
        )
        if state.get("_intervention_spec") is None:
            state["_intervention_spec"] = InterventionSpec()
        if not state.get("relationship_evidence"):
            state["relationship_evidence"] = {
                "model": Relationship.UNKNOWN,
                "weights": Relationship.UNKNOWN,
                "input": Relationship.UNKNOWN,
                "graph": Relationship.UNKNOWN,
            }
        if state["train_mode"] is None:
            state["train_mode"] = False
        conditional_arm_edges = dict(state.get("conditional_arm_edges") or {})
        for parent, child in state.pop("conditional_then_edges", []) or []:
            conditional_arm_edges.setdefault((0, "then"), []).append((parent, child))
        for conditional_id, elif_index, parent, child in (
            state.pop("conditional_elif_edges", []) or []
        ):
            conditional_arm_edges.setdefault((conditional_id, f"elif_{elif_index}"), []).append(
                (parent, child)
            )
        for conditional_id, parent, child in state.pop("conditional_else_edges", []) or []:
            conditional_arm_edges.setdefault((conditional_id, "else"), []).append((parent, child))
        state["conditional_arm_edges"] = conditional_arm_edges
        self.__dict__.update(state)
        if self.__dict__.get("_module_logs") is None:
            self._module_logs = ModuleAccessor({})
        if "_buffer_accessor" not in self.__dict__:
            self._buffer_accessor = None
        if self.__dict__.get("_module_build_data") is None:
            self._module_build_data = _init_module_build_data()

        for layer_log in self.layer_logs.values():
            layer_log.source_model_log = self
            for layer_pass in layer_log.passes.values():
                layer_pass.source_model_log = self
                layer_pass.parent_layer_log = layer_log
        for layer_pass in self.layer_list:
            layer_pass.source_model_log = self
            parent_layer_log = self.layer_logs.get(layer_pass.layer_label_no_pass)
            if parent_layer_log is not None:
                layer_pass.parent_layer_log = parent_layer_log
        for grad_fn in self.grad_fn_logs.values():
            if isinstance(grad_fn.corresponding_layer, str):
                grad_fn.corresponding_layer = self.layer_logs.get(grad_fn.corresponding_layer)
            if grad_fn.corresponding_layer is not None:
                grad_fn.corresponding_layer.corresponding_grad_fn = grad_fn
                if hasattr(grad_fn.corresponding_layer, "passes"):
                    for layer_pass in grad_fn.corresponding_layer.passes.values():
                        layer_pass.corresponding_grad_fn = grad_fn
        _state._register_log(self)

    def replace_run_state_from(self, new_log: "ModelLog") -> None:
        """Atomically replace this log's run-state from another ``ModelLog``.

        This method is intended for intervention rerun. The rerun engine builds
        ``new_log`` off to the side and calls this only after validation passes.
        The final state replacement uses one ``self.__dict__.update(...)`` call
        to minimize torn-state windows. Concurrent reads during rerun are
        unsupported; no lock is taken.

        Parameters
        ----------
        new_log:
            Fully postprocessed fresh log whose graph, layer containers,
            accessors, output metadata, shape/hash fields, and per-pass entries
            should replace this log's current run-state.

        Returns
        -------
        None
            This log is mutated in place.
        """

        preserved_fields = (
            "name",
            "parent_run",
            "_intervention_spec",
            "operation_history",
            "_warned_direct_write",
            "_warned_mutate_in_place",
            "source_model_id",
            "source_model_class",
            "weight_fingerprint_at_capture",
            "weight_fingerprint_full",
            "input_id_at_capture",
            "input_shape_hash",
            "is_appended",
            "relationship_evidence",
            "_source_model_ref",
            "_has_direct_writes",
            "_spec_revision",
            "_activation_recipe_revision",
            "_append_sequence_id",
        )
        preserved_state = {
            field_name: self.__dict__.get(field_name) for field_name in preserved_fields
        }
        replacement_state = dict(new_log.__dict__)
        replacement_state.update(preserved_state)
        self.__dict__.update(replacement_state)

    def append_run_state_from(self, new_log: "ModelLog") -> None:
        """Merge compatible chunk activations from ``new_log`` into this log.

        Parameters
        ----------
        new_log:
            Freshly captured append chunk whose topology and tensor metadata
            have already been validated against this log.
        """

        new_by_raw = {layer.layer_label_raw: layer for layer in new_log.layer_list}
        for layer in self.layer_list:
            new_layer = new_by_raw[layer.layer_label_raw]
            layer._append_tensor_from(new_layer, "activation")
            layer._append_tensor_from(new_layer, "transformed_activation")
            self._copy_append_last_chunk_fields(layer, new_layer)
            self._refresh_appended_tensor_metadata(layer)
        self.has_gradients = self.has_gradients or new_log.has_gradients
        self.random_seed_used = new_log.random_seed_used
        self.input_id_at_capture = new_log.input_id_at_capture
        self.input_shape_hash = new_log.input_shape_hash
        self._rebind_fork_owner_refs()

    def _copy_append_last_chunk_fields(self, layer: Any, new_layer: Any) -> None:
        """Copy per-call metadata fields from the last appended chunk.

        Parameters
        ----------
        layer:
            Existing accumulated layer pass.
        new_layer:
            New chunk layer pass supplying per-call state.
        """

        for field_name in (
            "func_time",
            "flops_forward",
            "flops_backward",
            "func_rng_states",
            "func_autocast_state",
            "func_argnames",
            "num_args",
            "num_positional_args",
            "num_keyword_args",
            "func_positional_args_non_tensor",
            "func_kwargs_non_tensor",
            "func_non_tensor_args",
            "func_is_inplace",
            "grad_fn_name",
            "grad_fn_id",
            "intervention_log",
            "extra_data",
        ):
            if hasattr(new_layer, field_name):
                layer._internal_set(
                    field_name, self._copy_append_metadata_value(getattr(new_layer, field_name))
                )

    def _copy_append_metadata_value(self, value: Any) -> Any:
        """Copy metadata from the last chunk without failing on non-leaf tensors.

        Parameters
        ----------
        value:
            Metadata value from the new chunk.

        Returns
        -------
        Any
            Best-effort copied value.
        """

        if isinstance(value, torch.Tensor):
            from ..utils.tensor_utils import safe_copy

            return safe_copy(value, detach_tensor=True)
        if isinstance(value, list):
            return [self._copy_append_metadata_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._copy_append_metadata_value(item) for item in value)
        if isinstance(value, dict):
            return {
                self._copy_append_metadata_value(key): self._copy_append_metadata_value(item)
                for key, item in value.items()
            }
        try:
            return copy.deepcopy(value)
        except RuntimeError:
            return value

    def _refresh_appended_tensor_metadata(self, layer: Any) -> None:
        """Refresh shape, dtype, and memory fields after tensor concatenation.

        Parameters
        ----------
        layer:
            Layer pass whose tensor fields may have been concatenated.
        """

        for tensor_field, shape_field, dtype_field, memory_field in (
            ("activation", "tensor_shape", "tensor_dtype", "tensor_memory"),
            (
                "transformed_activation",
                "transformed_activation_shape",
                "transformed_activation_dtype",
                "transformed_activation_memory",
            ),
            ("gradient", "grad_shape", "grad_dtype", "grad_memory"),
            (
                "transformed_gradient",
                "transformed_gradient_shape",
                "transformed_gradient_dtype",
                "transformed_gradient_memory",
            ),
        ):
            value = getattr(layer, tensor_field, None)
            if isinstance(value, torch.Tensor):
                from ..utils.tensor_utils import get_tensor_memory_amount

                layer._internal_set(shape_field, tuple(value.shape))
                layer._internal_set(dtype_field, value.dtype)
                layer._internal_set(memory_field, get_tensor_memory_amount(value))
            else:
                layer._internal_set(shape_field, None)
                layer._internal_set(dtype_field, None)
                layer._internal_set(memory_field, None)

    # ********************************************
    # ********** Computed Properties *************
    # ********************************************

    @property
    def activation_transform(self) -> Optional[ActivationPostfunc]:
        """Canonical activation transform callable used during capture.

        Returns
        -------
        Optional[ActivationPostfunc]
            Transform callable, or ``None`` when activations are stored unchanged.
        """

        return self.activation_postfunc

    @activation_transform.setter
    def activation_transform(self, value: Optional[ActivationPostfunc]) -> None:
        """Set the canonical activation transform callable.

        Parameters
        ----------
        value:
            Transform callable, or ``None``.
        """

        self.activation_postfunc = value

    @property
    def conditional_then_edges(self) -> List[Tuple[str, str]]:
        """Deprecated THEN-edge view derived from ``conditional_arm_edges``.

        Returns
        -------
        List[Tuple[str, str]]
            Legacy ``(parent, child)`` edge view.
        """

        warn_deprecated_alias("conditional_then_edges", "conditional_arm_edges")
        return _legacy_conditional_then_edges(self.conditional_arm_edges)

    @conditional_then_edges.setter
    def conditional_then_edges(self, value: List[Tuple[str, str]]) -> None:
        """Set the deprecated THEN-edge view by updating canonical arm edges.

        Parameters
        ----------
        value:
            Legacy ``(parent, child)`` edge list. Edges are assigned to
            conditional id 0 because the legacy view did not carry ids.
        """

        warn_deprecated_alias("conditional_then_edges", "conditional_arm_edges")
        self.conditional_arm_edges = {
            key: edges for key, edges in self.conditional_arm_edges.items() if key[1] != "then"
        }
        if value:
            self.conditional_arm_edges[(0, "then")] = list(value)

    @property
    def conditional_elif_edges(self) -> List[Tuple[int, int, str, str]]:
        """Deprecated ELIF-edge view derived from ``conditional_arm_edges``.

        Returns
        -------
        List[Tuple[int, int, str, str]]
            Legacy ``(cond_id, elif_index, parent, child)`` edge view.
        """

        warn_deprecated_alias("conditional_elif_edges", "conditional_arm_edges")
        return _legacy_conditional_elif_edges(self.conditional_arm_edges)

    @conditional_elif_edges.setter
    def conditional_elif_edges(self, value: List[Tuple[int, int, str, str]]) -> None:
        """Set the deprecated ELIF-edge view by updating canonical arm edges.

        Parameters
        ----------
        value:
            Legacy ``(cond_id, elif_index, parent, child)`` edge list.
        """

        warn_deprecated_alias("conditional_elif_edges", "conditional_arm_edges")
        self.conditional_arm_edges = {
            key: edges
            for key, edges in self.conditional_arm_edges.items()
            if not key[1].startswith("elif_")
        }
        for conditional_id, elif_index, parent, child in value:
            self.conditional_arm_edges.setdefault(
                (conditional_id, f"elif_{elif_index}"), []
            ).append((parent, child))

    @property
    def conditional_else_edges(self) -> List[Tuple[int, str, str]]:
        """Deprecated ELSE-edge view derived from ``conditional_arm_edges``.

        Returns
        -------
        List[Tuple[int, str, str]]
            Legacy ``(cond_id, parent, child)`` edge view.
        """

        warn_deprecated_alias("conditional_else_edges", "conditional_arm_edges")
        return _legacy_conditional_else_edges(self.conditional_arm_edges)

    @conditional_else_edges.setter
    def conditional_else_edges(self, value: List[Tuple[int, str, str]]) -> None:
        """Set the deprecated ELSE-edge view by updating canonical arm edges.

        Parameters
        ----------
        value:
            Legacy ``(cond_id, parent, child)`` edge list.
        """

        warn_deprecated_alias("conditional_else_edges", "conditional_arm_edges")
        self.conditional_arm_edges = {
            key: edges for key, edges in self.conditional_arm_edges.items() if key[1] != "else"
        }
        for conditional_id, parent, child in value:
            self.conditional_arm_edges.setdefault((conditional_id, "else"), []).append(
                (parent, child)
            )

    @property
    def is_recurrent(self) -> bool:
        """Whether any layer has more than one pass."""
        return any(v > 1 for v in self.layer_num_passes.values())

    @property
    def max_recurrent_loops(self) -> int:
        """Maximum number of passes for any layer."""
        return max(self.layer_num_passes.values(), default=1)

    @property
    def is_branching(self) -> bool:
        """Whether any layer has more than one child."""
        return any(len(entry.child_layers) > 1 for entry in self.layer_list)

    @property
    def has_conditional_branching(self) -> bool:
        """Whether any layer is in a conditional branch."""
        return any(entry.in_cond_branch for entry in self.layer_list)

    @property
    def num_tensors_total(self) -> int:
        """Total number of tensor operations."""
        return len(self)

    @property
    def total_activation_memory_str(self) -> str:
        """Human-readable total tensor size."""
        return human_readable_size(self.total_activation_memory)

    @property
    def saved_activation_memory_str(self) -> str:
        """Human-readable saved tensor size."""
        return human_readable_size(self.saved_activation_memory)

    @property
    def time_total(self) -> float:
        """Total time from start to end of pass."""
        if not self.pass_start_time or not self.pass_end_time:
            return 0
        return self.pass_end_time - self.pass_start_time

    @property
    def time_logging(self) -> float:
        """Time spent on TorchLens overhead (total minus function calls)."""
        return self.time_total - self.time_function_calls

    # ********************************************
    # ************* FLOPs Properties *************
    # ********************************************
    # FLOPs are estimated per-operation during logging (flops_forward,
    # flops_backward on each LayerPassLog).  These properties aggregate
    # across the entire model.  Layers with None FLOPs (unknown ops) are
    # skipped, so the totals may undercount.

    @property
    def total_params_memory_str(self) -> str:
        return human_readable_size(self.total_params_memory)

    @property
    def total_flops_forward(self) -> int:
        """Total forward FLOPs across all layers (skipping None/unknown)."""
        return sum(
            entry.flops_forward for entry in self.layer_list if entry.flops_forward is not None
        )

    @property
    def total_flops_backward(self) -> int:
        """Total backward FLOPs across all layers (skipping None/unknown)."""
        return sum(
            entry.flops_backward for entry in self.layer_list if entry.flops_backward is not None
        )

    @property
    def total_flops(self) -> int:
        """Total FLOPs (forward + backward)."""
        return self.total_flops_forward + self.total_flops_backward

    @property
    def flops_by_type(self) -> _CallableDict:
        """Group FLOPs by layer type.

        Returns:
            Callable dict mapping layer_type to forward/backward/count totals.
        """
        result: Dict[str, Dict[str, int]] = {}
        for entry in self.layer_list:
            lt = entry.layer_type
            if lt not in result:
                result[lt] = {"forward": 0, "backward": 0, "count": 0}
            result[lt]["count"] += 1
            if entry.flops_forward is not None:
                result[lt]["forward"] += entry.flops_forward
            if entry.flops_backward is not None:
                result[lt]["backward"] += entry.flops_backward
        return _CallableDict(result)

    # ********************************************
    # ************** MACs Properties *************
    # ********************************************
    # MACs (multiply-accumulate operations) = FLOPs / 2.

    @property
    def total_macs_forward(self) -> int:
        """Total forward MACs across all layers (skipping None/unknown)."""
        return self.total_flops_forward // 2

    @property
    def total_macs_backward(self) -> int:
        """Total backward MACs across all layers (skipping None/unknown)."""
        return self.total_flops_backward // 2

    @property
    def total_macs(self) -> int:
        """Total MACs (forward + backward)."""
        return self.total_flops // 2

    @property
    def macs_by_type(self) -> _CallableDict:
        """Group MACs by layer type.

        Returns:
            Callable dict mapping layer_type to forward/backward/count totals.
        """
        result: Dict[str, Dict[str, int]] = {}
        for entry in self.layer_list:
            lt = entry.layer_type
            if lt not in result:
                result[lt] = {"forward": 0, "backward": 0, "count": 0}
            result[lt]["count"] += 1
            if entry.flops_forward is not None:
                result[lt]["forward"] += entry.flops_forward // 2
            if entry.flops_backward is not None:
                result[lt]["backward"] += entry.flops_backward // 2
        return _CallableDict(result)

    # ********************************************
    # ************* Params Accessor **************
    # ********************************************

    @property
    def params(self) -> ParamAccessor:
        """Access parameter metadata by address, short name, or index."""
        return self.param_logs

    @property
    def layers(self) -> "LayerAccessor":
        """Access aggregate per-layer metadata by label, index, or pass notation."""
        from .layer_log import LayerAccessor

        return LayerAccessor(self.layer_logs, source_model_log=self)

    @property
    def modules(self) -> "ModuleAccessor":
        """Access structured per-module metadata by address, index, or pass notation."""
        return self._module_logs

    @property
    def root_module(self):
        """The root module (the model itself)."""
        return self._module_logs["self"]

    @property
    def buffers(self) -> "BufferAccessor":
        """Access buffer metadata by address, short name, or index."""
        return self._buffer_accessor  # type: ignore[return-value]

    @property
    def grad_fns(self) -> GradFnAccessor:
        """Access backward grad_fn metadata by label, index, pass label, or substring."""
        return GradFnAccessor(self.grad_fn_logs, self.grad_fn_order)

    @property
    def num_grad_fns(self) -> int:
        """Number of unique autograd grad_fn nodes discovered."""
        return len(self.grad_fn_logs)

    @property
    def num_intervening_grad_fns(self) -> int:
        """Number of grad_fn nodes without a corresponding forward LayerLog."""
        return sum(1 for grad_fn in self.grad_fn_logs.values() if grad_fn.is_intervening)

    # ********************************************
    # ******** Public Convenience Methods ********
    # ********************************************

    def render_graph(
        self,
        vis_mode: VisModeLiteral = "unrolled",
        vis_nesting_depth: int = 1000,
        vis_outpath: str = "modelgraph",
        vis_graph_overrides: Optional[Dict] = None,
        module: "ModuleLog | str | None" = None,
        node_mode: VisNodeModeLiteral = "default",
        node_spec_fn: Optional[Callable] = None,
        collapsed_node_spec_fn: Optional[Callable] = None,
        collapse_fn: Optional[Callable] = None,
        skip_fn: Optional[Callable] = None,
        vis_edge_overrides: Optional[Dict] = None,
        vis_gradient_edge_overrides: Optional[Dict] = None,
        vis_module_overrides: Optional[Dict] = None,
        vis_save_only: bool = False,
        vis_fileformat: str = "pdf",
        show_buffer_layers: BufferVisibilityLiteral | bool = "meaningful",
        direction: VisDirectionLiteral = "bottomup",
        vis_node_placement: VisNodePlacementLiteral = "auto",
        vis_renderer: VisRendererLiteral = "graphviz",
        vis_theme: str = "torchlens",
        vis_intervention_mode: VisInterventionModeLiteral = "node_mark",
        vis_show_cone: bool = True,
        code_panel: "CodePanelOption" = False,
        node_overlay: str | Mapping[str, Any] | None = None,
        node_label_fields: list[str] | None = None,
        show_legend: bool = False,
        font_size: int | None = None,
        dpi: int | None = None,
        for_paper: bool = False,
        return_graph: bool = False,
    ) -> Any:
        """Render the computational graph for this model log.

        Parameters
        ----------
        vis_mode, vis_nesting_depth, vis_outpath, vis_graph_overrides, module, node_mode, \
        node_spec_fn, collapsed_node_spec_fn, collapse_fn, skip_fn, vis_edge_overrides, \
        vis_gradient_edge_overrides, vis_module_overrides, vis_save_only, vis_fileformat, \
        show_buffer_layers, direction, vis_node_placement, vis_renderer, vis_theme, \
        vis_intervention_mode, vis_show_cone, code_panel:
            Forwarded unchanged to :func:`torchlens.visualization.rendering.render_graph`.
            ``show_buffer_layers`` accepts ``"never"``, ``"meaningful"``, or
            ``"always"``. Legacy bools are deprecated but supported by the
            Graphviz renderer.

        Returns
        -------
        Any
            Graphviz DOT source, renderer-specific output, or renderer object
            when ``return_graph=True``.
        """
        from ..visualization.rendering import render_graph as _impl

        return _impl(
            self,
            vis_mode=vis_mode,
            vis_nesting_depth=vis_nesting_depth,
            vis_outpath=vis_outpath,
            vis_graph_overrides=vis_graph_overrides,
            module=module,
            node_mode=node_mode,
            node_spec_fn=node_spec_fn,
            collapsed_node_spec_fn=collapsed_node_spec_fn,
            collapse_fn=collapse_fn,
            skip_fn=skip_fn,
            vis_edge_overrides=vis_edge_overrides,
            vis_gradient_edge_overrides=vis_gradient_edge_overrides,
            vis_module_overrides=vis_module_overrides,
            vis_save_only=vis_save_only,
            vis_fileformat=vis_fileformat,
            show_buffer_layers=show_buffer_layers,
            direction=direction,
            vis_node_placement=vis_node_placement,
            vis_renderer=vis_renderer,
            vis_theme=vis_theme,
            vis_intervention_mode=vis_intervention_mode,
            vis_show_cone=vis_show_cone,
            code_panel=code_panel,
            node_overlay=node_overlay,
            node_label_fields=node_label_fields,
            show_legend=show_legend,
            font_size=font_size,
            dpi=dpi,
            for_paper=for_paper,
            return_graph=return_graph,
        )

    def add_node_overlay(
        self,
        scores: Mapping[str, Any],
        *,
        name: str = "overlay",
    ) -> "ModelLog":
        """Register external per-node overlay scores for later rendering.

        Parameters
        ----------
        scores:
            Mapping from layer labels to scalar or displayable values.
        name:
            Overlay name stored on the log for discoverability.

        Returns
        -------
        ModelLog
            This log, allowing chained calls before ``render_graph``.
        """

        self._node_overlay_scores = dict(scores)
        self._node_overlay_name = name
        return self

    def animate_passes(self, site: Any) -> str:
        """Return a minimal HTML animation for repeated passes at ``site``.

        Parameters
        ----------
        site:
            Layer label, pass-qualified layer label, or object with a
            ``layer_label`` attribute.

        Returns
        -------
        str
            Self-contained HTML fragment with play/pause controls.

        Raises
        ------
        KeyError
            If the requested site cannot be resolved.
        """

        label = str(getattr(site, "layer_label", site))
        base_label = label.split(":", 1)[0]
        if base_label not in self.layer_logs:
            raise KeyError(f"Unknown layer site {label!r}.")
        layer = self.layer_logs[base_label]
        pass_entries = list(getattr(layer, "passes", ()) or [])
        if not pass_entries:
            pass_entries = [self[label]]
        frames = [
            {
                "pass": int(getattr(entry, "pass_num", index + 1) or index + 1),
                "label": str(getattr(entry, "layer_label", base_label)),
                "shape": "x".join(str(dim) for dim in getattr(entry, "tensor_shape", ()) or ()),
                "memory": str(getattr(entry, "tensor_memory_str", "")),
            }
            for index, entry in enumerate(pass_entries)
        ]
        frame_markup = "".join(
            "<li data-frame='{idx}'>{label} pass {pass_num}: {shape} {memory}</li>".format(
                idx=index,
                label=escape(str(frame["label"])),
                pass_num=frame["pass"],
                shape=escape(str(frame["shape"] or "scalar")),
                memory=escape(str(frame["memory"])),
            )
            for index, frame in enumerate(frames)
        )
        return (
            "<div class='tl-pass-animation' data-site='"
            + escape(base_label)
            + "'><button type='button' data-action='play'>Play</button>"
            + "<button type='button' data-action='pause'>Pause</button><ol>"
            + frame_markup
            + "</ol><script>(function(){var root=document.currentScript.parentElement;"
            + "var items=root.querySelectorAll('li');var i=0,t=null;"
            + "function show(){items.forEach(function(x,j){x.style.display=j===i?'':'none';});}"
            + "show();root.querySelector('[data-action=play]').onclick=function(){"
            + "if(t)return;t=setInterval(function(){i=(i+1)%items.length;show();},500);};"
            + "root.querySelector('[data-action=pause]').onclick=function(){clearInterval(t);t=null;};"
            + "})();</script></div>"
        )

    def first_nonfinite(self) -> str:
        """Return a text answer describing the first saved non-finite activation.

        Returns
        -------
        str
            Human-readable single-paragraph answer naming the layer, operation,
            module, shape, dtype, parents, and source location.
        """

        for layer in getattr(self, "layer_list", []) or []:
            activation = getattr(layer, "activation", None)
            if not isinstance(activation, torch.Tensor) or activation.numel() == 0:
                continue
            try:
                has_nonfinite = bool((~torch.isfinite(activation.detach())).any().item())
            except (RuntimeError, TypeError):
                continue
            if not has_nonfinite:
                continue
            stack = getattr(layer, "func_call_stack", None) or []
            location = "source unavailable"
            if stack:
                frame = stack[0]
                location = (
                    f"{getattr(frame, 'file', 'unknown')}:"
                    f"{getattr(frame, 'line_number', 'unknown')}"
                )
            parents = ", ".join(getattr(layer, "parent_layers", None) or []) or "none"
            module = getattr(layer, "containing_module", None) or "no module"
            return (
                f"First non-finite saved activation is in layer {layer.layer_label} "
                f"(op {getattr(layer, 'func_name', 'unknown')}, module {module}), "
                f"shape={getattr(layer, 'tensor_shape', None)}, "
                f"dtype={getattr(layer, 'tensor_dtype', None)}, parents={parents}, "
                f"source={location}."
            )
        return "No non-finite tensor values found in saved activations."

    def show(self, method: Literal["graph", "repr", "html"] = "graph", **kwargs: Any) -> str | None:
        """Render this model log with intervention visualization defaults.

        Parameters
        ----------
        method:
            ``"graph"`` renders the model graph. ``"repr"`` prints the compact
            representation. ``"html"`` returns the notebook HTML fragment.
        **kwargs:
            Forwarded to :meth:`render_graph`. The legacy ``vis_opt`` alias is
            accepted for parity with logging APIs.

        Returns
        -------
        str | None
            DOT source when rendering occurs, otherwise ``None`` for
            ``vis_opt='none'`` / ``vis_mode='none'``.
        """

        if method == "repr":
            return repr(self)
        if method == "html":
            return self._repr_html_()
        if method != "graph":
            raise ValueError("method must be 'graph', 'repr', or 'html'.")
        if "vis_opt" in kwargs and "vis_mode" not in kwargs:
            kwargs["vis_mode"] = kwargs.pop("vis_opt")
        elif "vis_opt" in kwargs:
            kwargs.pop("vis_opt")
        if kwargs.get("vis_mode") == "none":
            return None
        return self.render_graph(**kwargs)

    def show_backward_graph(
        self,
        vis_outpath: str = "backward_modelgraph",
        vis_graph_overrides: Optional[Dict] = None,
        node_spec_fn: Optional[Callable] = None,
        collapsed_node_spec_fn: Optional[Callable] = None,
        vis_node_mode: VisNodeModeLiteral = "default",
        vis_edge_overrides: Optional[Dict] = None,
        vis_save_only: bool = False,
        vis_fileformat: str = "pdf",
        vis_direction: VisDirectionLiteral = "topdown",
        code_panel: "CodePanelOption" = False,
    ) -> str:
        """Render the captured backward grad_fn graph.

        Parameters
        ----------
        vis_outpath, vis_graph_overrides, node_spec_fn, collapsed_node_spec_fn, \
        vis_node_mode, vis_edge_overrides, vis_save_only, vis_fileformat, \
        vis_direction, code_panel:
            Forwarded unchanged to
            :func:`torchlens.visualization.rendering.render_backward_graph`.
            ``collapsed_node_spec_fn`` and ``vis_node_mode`` are accepted for
            forward-visualization API symmetry but are not applied because
            backward graphs do not render collapsed module nodes.

        Returns
        -------
        str
            Graphviz DOT source.
        """
        from ..visualization.rendering import render_backward_graph as _impl

        return _impl(
            self,
            vis_outpath=vis_outpath,
            vis_graph_overrides=vis_graph_overrides,
            node_spec_fn=node_spec_fn,
            collapsed_node_spec_fn=collapsed_node_spec_fn,
            vis_node_mode=vis_node_mode,
            vis_edge_overrides=vis_edge_overrides,
            vis_save_only=vis_save_only,
            vis_fileformat=vis_fileformat,
            direction=vis_direction,
            code_panel=code_panel,
        )

    def preview_fastlog(
        self,
        predicate: Optional[Callable] = None,
        keep_op: Optional[Callable] = None,
        keep_module: Optional[Callable] = None,
        **kwargs: Any,
    ) -> str:
        """Render a fastlog predicate preview for this model graph.

        Parameters
        ----------
        predicate, keep_op, keep_module:
            Predicate callables that receive synthesized fastlog ``RecordContext``
            objects.
        **kwargs:
            Forwarded to :func:`torchlens.visualization.fastlog_preview.preview_fastlog`.

        Returns
        -------
        str
            Graphviz DOT source.
        """

        from ..visualization.fastlog_preview import preview_fastlog as _impl

        return _impl(
            self,
            predicate=predicate,
            keep_op=keep_op,
            keep_module=keep_module,
            **kwargs,
        )

    def last_run_records(self) -> tuple["FireRecord", ...]:
        """Return fire records from the most recent replay, rerun, or live capture.

        Returns
        -------
        tuple[FireRecord, ...]
            Immutable snapshot of matching intervention fire records.
        """

        ctx = getattr(self, "last_run_ctx", None)
        if not isinstance(ctx, dict):
            return ()
        timestamp = ctx.get("timestamp")
        if not isinstance(timestamp, (int, float)):
            return ()
        records = []
        for layer in getattr(self, "layer_list", []) or []:
            for record in getattr(layer, "intervention_log", []) or []:
                record_timestamp = getattr(record, "timestamp", None)
                if isinstance(record_timestamp, (int, float)) and record_timestamp >= timestamp:
                    records.append(record)
        return tuple(records)

    def summary(
        self,
        level: Literal[
            "overview", "graph", "memory", "control_flow", "compute", "cost", "waterfall"
        ] = "overview",
        *,
        fields: Optional[List[str]] = None,
        mode: Literal["auto", "rolled", "unrolled"] = "auto",
        show_ops: bool = False,
        preset: Optional[
            Literal["overview", "graph", "memory", "control_flow", "compute", "cost", "waterfall"]
        ] = None,
        columns: Optional[List[str]] = None,
        include_ops: Optional[bool] = None,
        max_rows: Optional[int] = 200,
        print_to: Optional[Callable[[str], None]] = None,
        count_fma_as_two: bool = False,
    ) -> str:
        """Render a concise text summary of the logged model.

        Parameters
        ----------
        level:
            Summary level to render.
        fields:
            Explicit column selection for the primary table.
        mode:
            Operation aggregation mode. ``"rolled"`` uses aggregate layer rows,
            while ``"unrolled"`` uses per-pass operation rows.
        show_ops:
            Whether to append an operation table.
        preset:
            Alias for ``level`` retained for compatibility with the design spec.
        columns:
            Alias for ``fields``.
        include_ops:
            Alias for ``show_ops`` retained for compatibility with the design spec.
        max_rows:
            Maximum number of rows to render per table. ``None`` disables truncation.
        print_to:
            Optional callable that receives the rendered summary text.
        count_fma_as_two:
            FLOP/MAC convention marker for summary consumers. Current captured
            counts are displayed as stored; this flag reserves the public
            convention toggle without changing saved metadata.

        Returns
        -------
        str
            Rendered summary string.
        """
        from ..visualization._summary_internal import render_model_summary

        del count_fma_as_two
        return render_model_summary(
            self,
            level=level,
            preset=preset,
            fields=fields,
            columns=columns,
            mode=mode,
            show_ops=show_ops,
            include_ops=include_ops,
            max_rows=max_rows,
            print_to=print_to,
        )

    def render_dagua_graph(
        self,
        vis_mode: str = "unrolled",
        vis_nesting_depth: int = 1000,
        vis_outpath: str = "graph.gv",
        vis_save_only: bool = False,
        vis_fileformat: str = "pdf",
        vis_buffer_layers: bool = False,
        vis_direction: str = "bottomup",
        vis_theme: str = "torchlens",
    ) -> str:
        """Render this model log with the experimental Dagua backend.

        Parameters
        ----------
        vis_mode, vis_nesting_depth, vis_outpath, vis_save_only, vis_fileformat, \
        vis_buffer_layers, vis_direction, vis_theme:
            Forwarded unchanged to
            :func:`torchlens.experimental.dagua.render_model_log_with_dagua`.

        Returns
        -------
        str
            Serialized Dagua graph output or the rendered artifact path.
        """
        from ..experimental.dagua import render_model_log_with_dagua as _impl

        return _impl(
            self,
            vis_mode=vis_mode,
            vis_nesting_depth=vis_nesting_depth,
            vis_outpath=vis_outpath,
            vis_save_only=vis_save_only,
            vis_fileformat=vis_fileformat,
            vis_buffer_layers=vis_buffer_layers,
            vis_direction=vis_direction,
            vis_theme=vis_theme,
        )

    def to_dagua_graph(
        self,
        vis_mode: str = "unrolled",
        vis_nesting_depth: int = 1000,
        show_buffer_layers: bool = False,
        direction: str = "bottomup",
        include_gradient_edges: Optional[bool] = None,
    ) -> Any:
        """Translate this model log into an experimental Dagua graph.

        Parameters
        ----------
        vis_mode, vis_nesting_depth, show_buffer_layers, direction, include_gradient_edges:
            Forwarded unchanged to
            :func:`torchlens.experimental.dagua.model_log_to_dagua_graph`.

        Returns
        -------
        Any
            Dagua graph object.
        """
        from ..experimental.dagua import model_log_to_dagua_graph as _impl

        return _impl(
            self,
            vis_mode=vis_mode,
            vis_nesting_depth=vis_nesting_depth,
            show_buffer_layers=show_buffer_layers,
            direction=direction,
            include_gradient_edges=include_gradient_edges,
        )

    def visualization_field_audit(self) -> "TorchLensRenderAudit":
        """Return the visualization field-usage audit for this model log.

        Returns
        -------
        TorchLensRenderAudit
            Audit of used and unused fields in the visualization bridge.
        """
        from ..experimental.dagua import build_render_audit as _impl

        return _impl(self)

    def to_pandas(self) -> "pd.DataFrame":
        """Return a dataframe containing one row per layer pass.

        Returns
        -------
        pd.DataFrame
            Layer-pass table for this model log.

        Raises
        ------
        RuntimeError
            If called before the forward pass has completed.
        """
        if not self._pass_finished:
            raise RuntimeError(
                "to_pandas() cannot be called before the forward pass is complete. "
                "Please wait until log_forward_pass has returned."
            )
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for this feature. Install with `pip install torchlens[tabular]`."
            ) from e

        fields_for_df = [
            "layer_label",
            "layer_label_w_pass",
            "layer_label_no_pass",
            "layer_label_short",
            "layer_label_w_pass_short",
            "layer_label_no_pass_short",
            "layer_type",
            "layer_type_num",
            "layer_total_num",
            "num_passes",
            "pass_num",
            "operation_num",
            "tensor_shape",
            "tensor_dtype",
            "tensor_memory",
            "tensor_memory_str",
            "func_name",
            "func_config",
            "func_time",
            "func_is_inplace",
            "grad_fn_name",
            "is_input_layer",
            "is_output_layer",
            "is_buffer_layer",
            "is_part_of_iterable_output",
            "iterable_output_index",
            "parent_layers",
            "has_parents",
            "root_ancestors",
            "child_layers",
            "has_children",
            "output_descendants",
            "sibling_layers",
            "has_siblings",
            "co_parent_layers",
            "has_co_parents",
            "is_internally_initialized",
            "min_distance_from_input",
            "max_distance_from_input",
            "min_distance_from_output",
            "max_distance_from_output",
            "uses_params",
            "num_params_total",
            "parent_param_shapes",
            "params_memory",
            "params_memory_str",
            "modules_entered",
            "modules_exited",
            "is_submodule_input",
            "is_submodule_output",
            "containing_module",
            "containing_modules",
            "conditional_branch_depth",
            "bool_is_branch",
            "bool_context_kind",
            "bool_wrapper_kind",
            "bool_conditional_id",
            "conditional_branch_stack",
            "cond_branch_then_children",
            "cond_branch_elif_children",
            "cond_branch_else_children",
        ]

        fields_to_change_type = {
            "layer_type_num": int,
            "layer_total_num": int,
            "num_passes": int,
            "pass_num": int,
            "operation_num": int,
            "func_is_inplace": bool,
            "is_input_layer": bool,
            "is_output_layer": bool,
            "is_buffer_layer": bool,
            "is_part_of_iterable_output": bool,
            "has_parents": bool,
            "has_children": bool,
            "has_siblings": bool,
            "has_co_parents": bool,
            "uses_params": bool,
            "num_params_total": int,
            "params_memory": int,
            "tensor_memory": int,
            "is_submodule_input": bool,
            "is_submodule_output": bool,
            "conditional_branch_depth": int,
            "bool_is_branch": bool,
        }

        model_df_dictlist = []
        for layer_entry in self.layer_list:
            layer_dict = {}
            for field_name in fields_for_df:
                if field_name == "conditional_branch_stack":
                    layer_dict[field_name] = _format_conditional_branch_stack(
                        layer_entry.conditional_branch_stack
                    )
                else:
                    layer_dict[field_name] = getattr(layer_entry, field_name)
            model_df_dictlist.append(layer_dict)
        model_df = pd.DataFrame(model_df_dictlist)

        for column_name in fields_to_change_type:
            model_df[column_name] = model_df[column_name].astype(fields_to_change_type[column_name])
        model_df["bool_conditional_id"] = model_df["bool_conditional_id"].astype("Int64")

        return model_df

    def to_csv(self, filepath: str | PathLike[str], **kwargs: Any) -> None:
        """Write the layer table to CSV.

        Parameters
        ----------
        filepath:
            Output CSV path.
        **kwargs:
            Additional keyword arguments forwarded to ``DataFrame.to_csv``.
        """
        warnings.warn(
            "ModelLog.to_csv() is deprecated; use torchlens.export.csv(log, path) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from ..export import csv as export_csv

        export_csv(self, Path(filepath), **kwargs)

    def to_parquet(self, filepath: str | PathLike[str], **kwargs: Any) -> None:
        """Write the layer table to Parquet.

        Parameters
        ----------
        filepath:
            Output Parquet path.
        **kwargs:
            Additional keyword arguments forwarded to ``DataFrame.to_parquet``.

        Raises
        ------
        ImportError
            If ``pyarrow`` is unavailable.
        """
        warnings.warn(
            "ModelLog.to_parquet() is deprecated; use torchlens.export.parquet(log, path) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from ..export import parquet as export_parquet

        try:
            export_parquet(self, Path(filepath), **kwargs)
        except ImportError as exc:
            raise ImportError(
                "to_parquet requires pyarrow. Install with: pip install torchlens[io]"
            ) from exc

    def to_json(
        self,
        filepath: str | PathLike[str],
        *,
        orient: Literal["split", "records", "index", "columns", "values", "table"] = "records",
        **kwargs: Any,
    ) -> None:
        """Write the layer table to JSON.

        Parameters
        ----------
        filepath:
            Output JSON path.
        orient:
            JSON orientation passed to ``DataFrame.to_json``.
        **kwargs:
            Additional keyword arguments forwarded to ``DataFrame.to_json``.
        """
        warnings.warn(
            "ModelLog.to_json() is deprecated; use torchlens.export.json(log, path) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from ..export import json as export_json

        export_json(self, Path(filepath), orient=orient, **kwargs)

    def save_new_activations(
        self,
        model: torch.nn.Module,
        input_args: torch.Tensor | List[Any],
        input_kwargs: Optional[Dict[Any, Any]] = None,
        layers_to_save: str | List = "all",
        gradients_to_save: str | List | None = "all",
        random_seed: Optional[int] = None,
        train_mode: bool | None = None,
    ) -> None:
        """Re-run the model with new inputs, saving only activations.

        Parameters
        ----------
        model, input_args, input_kwargs, layers_to_save, gradients_to_save, random_seed, train_mode:
            Forwarded unchanged to
            :func:`torchlens.capture.trace.save_new_activations`.
        """
        from ..capture.trace import save_new_activations as _impl

        if train_mode is True:
            reject_compiled_model(model, api_name="ModelLog.save_new_activations")

        return _impl(
            self,
            model=model,
            input_args=input_args,
            input_kwargs=input_kwargs,
            layers_to_save=layers_to_save,
            gradients_to_save=gradients_to_save,
            random_seed=random_seed,
            train_mode=train_mode,
        )

    def validate_saved_activations(
        self,
        ground_truth_output_tensors: List[torch.Tensor],
        verbose: bool = False,
        validate_metadata: bool = True,
    ) -> bool:
        """Deprecated alias for :meth:`validate_forward_pass`.

        Parameters
        ----------
        ground_truth_output_tensors, verbose, validate_metadata:
            Forwarded unchanged to :meth:`validate_forward_pass`.

        Returns
        -------
        bool
            ``True`` if validation succeeds.
        """
        warn_deprecated_alias(
            "ModelLog.validate_saved_activations",
            "ModelLog.validate_forward_pass",
        )
        return self.validate_forward_pass(
            ground_truth_output_tensors=ground_truth_output_tensors,
            verbose=verbose,
            validate_metadata=validate_metadata,
        )

    def validate_forward_pass(
        self,
        ground_truth_output_tensors: List[torch.Tensor],
        verbose: bool = False,
        validate_metadata: bool = True,
    ) -> bool:
        """Validate saved activations against ground-truth model outputs.

        Parameters
        ----------
        ground_truth_output_tensors, verbose, validate_metadata:
            Forwarded unchanged to
            :func:`torchlens.validation.core.validate_saved_activations`.

        Returns
        -------
        bool
            ``True`` if validation succeeds.
        """
        from ..validation.core import validate_saved_activations as _impl

        return _impl(
            self,
            ground_truth_output_tensors=ground_truth_output_tensors,
            verbose=verbose,
            validate_metadata=validate_metadata,
        )

    def replay(
        self,
        strict: bool | MissingType = MISSING,
        hooks: dict[Any, Any] | None | MissingType = MISSING,
        replay: ReplayOptions | None = None,
    ) -> "ModelLog":
        """Replay the saved DAG cone affected by hooks.

        Parameters
        ----------
        strict:
            Whether replay divergence warnings should raise.
        hooks:
            Optional mapping from selector-like targets to hook callables.

        Returns
        -------
        ModelLog
            This model log, mutated in place.
        """

        replay_options = merge_replay_options(replay=replay, strict=strict, hooks=hooks)

        from ..intervention.replay import replay as _impl

        return _impl(self, replay=replay_options)

    def replay_from(
        self,
        site: Any,
        strict: bool | MissingType = MISSING,
        replay: ReplayOptions | None = None,
    ) -> "ModelLog":
        """Replay downstream from a pre-mutated site.

        Parameters
        ----------
        site:
            Layer pass or selector resolving to one origin. The origin's
            current activation is preserved and used as the override.
        strict:
            Whether replay divergence warnings should raise.

        Returns
        -------
        ModelLog
            This model log, mutated in place.
        """

        replay_options = merge_replay_options(replay=replay, strict=strict)

        from ..intervention.replay import replay_from as _impl

        return _impl(self, site, replay=replay_options)

    def rerun(
        self,
        model: nn.Module,
        x: Any = None,
        *,
        append: bool | MissingType = MISSING,
        strict: bool | MissingType = MISSING,
        replay: ReplayOptions | None = None,
    ) -> "ModelLog":
        """Re-execute a model with this log's active intervention spec.

        Parameters
        ----------
        model:
            Model to execute through TorchLens decorated wrappers.
        x:
            Forward input. Phase 7 requires callers to pass this explicitly.
        append:
            If true, append a compatible chunk along batch dimension 0.
        strict:
            Whether graph-shape divergence should raise instead of warn.

        Returns
        -------
        ModelLog
            This model log, mutated in place after a validated atomic swap.
        """

        replay_options = merge_replay_options(replay=replay, append=append, strict=strict)

        from ..intervention.rerun import rerun as _impl

        return _impl(self, model, x, replay=replay_options)

    def check_metadata_invariants(self) -> bool:
        """Run metadata invariant checks on this completed model log.

        Returns
        -------
        bool
            ``True`` if all invariants pass.
        """
        from ..validation.invariants import check_metadata_invariants as _impl

        return _impl(self)

    def cleanup(self) -> None:
        """Delete log data, break cycles, and free cached GPU memory.

        Returns
        -------
        None
            This method mutates the model log in place.
        """
        return cleanup(self)

    def release_param_refs(self) -> None:
        """Release live ``nn.Parameter`` references held by ParamLogs.

        Returns
        -------
        None
            This method mutates ParamLogs in place.
        """
        for param_log in self.param_logs:
            param_log.release_param_ref()

    def _postprocess(
        self,
        output_tensors: List[torch.Tensor],
        output_tensor_addresses: List[str],
    ) -> None:
        """Run postprocessing on a completed raw capture pass.

        Parameters
        ----------
        output_tensors:
            Output tensors returned by the model.
        output_tensor_addresses:
            Hierarchical addresses for those outputs.
        """
        from ..postprocess import postprocess as _impl

        return _impl(
            self,
            output_tensors=output_tensors,
            output_tensor_addresses=output_tensor_addresses,
        )

    def _run_and_log_inputs_through_model(
        self,
        model: torch.nn.Module,
        input_args: torch.Tensor | List[Any],
        input_kwargs: Optional[Dict[Any, Any]] = None,
        layers_to_save: Optional[str | List[str | int]] = "all",
        gradients_to_save: Optional[str | List[str | int]] = "all",
        random_seed: Optional[int] = None,
    ) -> None:
        """Run a forward pass and capture it into this model log.

        Parameters
        ----------
        model, input_args, input_kwargs, layers_to_save, gradients_to_save, random_seed:
            Forwarded unchanged to
            :func:`torchlens.capture.trace.run_and_log_inputs_through_model`.
        """
        from ..capture.trace import run_and_log_inputs_through_model as _impl

        return _impl(
            self,
            model=model,
            input_args=input_args,
            input_kwargs=input_kwargs,
            layers_to_save=layers_to_save,
            gradients_to_save=gradients_to_save,
            random_seed=random_seed,
        )

    def log_backward(self, loss: torch.Tensor, **backward_kwargs: Any) -> "ModelLog":
        """Run backward from ``loss`` while capturing first-class backward metadata.

        Parameters
        ----------
        loss:
            Tensor whose ``grad_fn`` roots the backward graph.
        **backward_kwargs:
            Keyword arguments forwarded to ``torch.Tensor.backward``.

        Returns
        -------
        ModelLog
            This model log, for chaining.
        """
        from ..capture.backward import log_backward as _impl

        return _impl(self, loss, **backward_kwargs)

    def recording_backward(self) -> Any:
        """Return a context manager that captures user-managed backward calls.

        Returns
        -------
        Any
            Backward recording context manager.
        """
        from ..capture.backward import recording_backward as _impl

        return _impl(self)

    def _remove_log_entry(
        self,
        log_entry: LayerPassLog,
        remove_references: bool = True,
    ) -> None:
        """Remove a single layer-pass entry and scrub graph references.

        Parameters
        ----------
        log_entry:
            Entry to remove.
        remove_references:
            Whether to scrub all graph references to the removed entry.
        """
        tensor_label = _label_for_reference_removal(log_entry, self._pass_finished)
        if remove_references:
            _remove_log_entry_references(self, tensor_label)
        _clear_entry_attributes(log_entry)

    def _batch_remove_log_entries(
        self,
        entries_to_remove: Iterable[LayerPassLog],
        remove_references: bool = True,
    ) -> None:
        """Remove multiple layer-pass entries using single-pass filtering.

        Parameters
        ----------
        entries_to_remove:
            Entries to remove.
        remove_references:
            Whether to scrub all graph references to the removed entries.
        """
        entries_to_remove = list(entries_to_remove)
        surviving_entries = [entry for entry in self if entry not in entries_to_remove]

        labels_to_remove = set()
        for entry in entries_to_remove:
            labels_to_remove.add(_label_for_reference_removal(entry, self._pass_finished))
            _clear_entry_attributes(entry)

        if not remove_references:
            return

        _scrub_conditional_fields_after_removal(self, labels_to_remove, surviving_entries)

        for field_name in _LIST_FIELDS_TO_CLEAN:
            collection = getattr(self, field_name)
            collection[:] = [label for label in collection if label not in labels_to_remove]

        self.conditional_branch_edges = [
            edge
            for edge in self.conditional_branch_edges
            if edge[0] not in labels_to_remove and edge[1] not in labels_to_remove
        ]

        for param_group, tensor_labels in list(self.layers_with_params.items()):
            self.layers_with_params[param_group] = [
                label for label in tensor_labels if label not in labels_to_remove
            ]
        self.layers_with_params = {
            param_group: tensor_labels
            for param_group, tensor_labels in self.layers_with_params.items()
            if len(tensor_labels) > 0
        }

        for equiv_group, equivalent_label_set in list(self.equivalent_operations.items()):
            equivalent_label_set -= labels_to_remove
        self.equivalent_operations = {
            equiv_group: equivalent_label_set
            for equiv_group, equivalent_label_set in self.equivalent_operations.items()
            if len(equivalent_label_set) > 0
        }


ModelLog.FORK_POLICY = MODEL_LOG_FORK_POLICY  # type: ignore[attr-defined]
ModelLog.DEFAULT_FILL_STATE = _MODEL_LOG_DEFAULT_FILL  # type: ignore[attr-defined]
