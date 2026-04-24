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
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Set, TYPE_CHECKING, Tuple

import numpy as np
import pandas as pd
import torch

if TYPE_CHECKING:
    from .._io.streaming import BundleStreamWriter
    from .buffer_log import BufferAccessor
    from .layer_log import LayerAccessor
    from ..visualization.dagua_bridge import TorchLensRenderAudit

from .._io import FieldPolicy, IO_FORMAT_VERSION, default_fill_state, read_io_format_version
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
from .layer_log import LayerLog
from .layer_pass_log import LayerPassLog, TensorLog


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
        "model_name": FieldPolicy.KEEP,
        "num_context_lines": FieldPolicy.KEEP,
        "_optimizer": FieldPolicy.DROP,
        "io_format_version": FieldPolicy.KEEP,
        "_pass_finished": FieldPolicy.KEEP,
        "logging_mode": FieldPolicy.KEEP,
        "_all_layers_logged": FieldPolicy.KEEP,
        "_all_layers_saved": FieldPolicy.KEEP,
        "keep_unsaved_layers": FieldPolicy.KEEP,
        "activation_postfunc": FieldPolicy.DROP,
        "activation_postfunc_repr": FieldPolicy.KEEP,
        "current_function_call_barcode": FieldPolicy.KEEP,
        "random_seed_used": FieldPolicy.KEEP,
        "output_device": FieldPolicy.KEEP,
        "detach_saved_tensors": FieldPolicy.KEEP,
        "save_function_args": FieldPolicy.KEEP,
        "save_gradients": FieldPolicy.KEEP,
        "save_source_context": FieldPolicy.KEEP,
        "save_rng_states": FieldPolicy.KEEP,
        "detect_loops": FieldPolicy.KEEP,
        "verbose": FieldPolicy.KEEP,
        "has_gradients": FieldPolicy.KEEP,
        "mark_input_output_distances": FieldPolicy.KEEP,
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
        "conditional_then_edges": FieldPolicy.KEEP,
        "conditional_elif_edges": FieldPolicy.KEEP,
        "conditional_else_edges": FieldPolicy.KEEP,
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
        "_activation_sink": FieldPolicy.DROP,
        "_in_exhaustive_pass": FieldPolicy.DROP,
        "pass_start_time": FieldPolicy.KEEP,
        "pass_end_time": FieldPolicy.KEEP,
        "time_setup": FieldPolicy.KEEP,
        "time_forward_pass": FieldPolicy.KEEP,
        "time_cleanup": FieldPolicy.KEEP,
        "time_function_calls": FieldPolicy.KEEP,
    }

    def __init__(
        self,
        model_name: str,
        output_device: str = "same",
        activation_postfunc: Optional[Callable] = None,
        keep_unsaved_layers: bool = True,
        save_function_args: bool = False,
        save_gradients: bool = False,
        detach_saved_tensors: bool = False,
        mark_input_output_distances: bool = True,
        num_context_lines: int = 7,
        optimizer=None,
        save_source_context: bool = False,
        save_rng_states: bool = False,
        detect_loops: bool = True,
        verbose: bool = False,
    ):
        """Initialise a fresh ModelLog for a new logging session.

        Args:
            model_name: Human-readable name of the model being logged.
            output_device: Device to move saved activations to ("same" keeps original device).
            activation_postfunc: Optional function applied to each tensor before saving.
            keep_unsaved_layers: If False, layers without saved activations are removed
                from the final log (but still logged during the pass).
            save_function_args: Whether to deep-copy each operation's input arguments.
            save_gradients: Whether to register gradient hooks for backward pass.
            detach_saved_tensors: Whether to detach saved tensors from the autograd graph.
            mark_input_output_distances: Whether to compute BFS distances from
                inputs/outputs for each layer.
            num_context_lines: Number of source-code context lines to capture
                around each function call (used by FuncCallLocation).
            optimizer: Optional torch optimizer, used to annotate which params
                have optimizers attached.
            verbose: If True, print timed progress messages at each major pipeline stage.
        """
        # Callables are effectively immutable - deepcopy is unnecessary.

        # General info
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
        self.logging_mode = "exhaustive"
        self._all_layers_logged = False
        self._all_layers_saved = False
        self.keep_unsaved_layers = keep_unsaved_layers
        self.activation_postfunc = activation_postfunc
        self.activation_postfunc_repr = (
            repr(activation_postfunc) if activation_postfunc is not None else None
        )
        self.current_function_call_barcode = None
        self.random_seed_used = None
        self.output_device = output_device
        self.detach_saved_tensors = detach_saved_tensors
        self.save_function_args = save_function_args
        self.save_gradients = save_gradients
        self.save_source_context = save_source_context
        self.save_rng_states = save_rng_states
        self.detect_loops = detect_loops
        self.verbose = verbose
        self.has_gradients = False
        self.mark_input_output_distances = mark_input_output_distances
        self._activation_writer: Optional["BundleStreamWriter"] = None
        self._keep_activations_in_memory: bool = True
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
        self.conditional_then_edges: List[Tuple[str, str]] = []
        self.conditional_elif_edges: List[Tuple[int, int, str, str]] = []
        self.conditional_else_edges: List[Tuple[int, str, str]] = []
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

    def __str__(self) -> str:
        """Human-readable summary; delegates to post-pass or mid-pass formatter."""
        if self._pass_finished:
            return _str_after_pass(self)
        else:
            return _str_during_pass(self)

    def __repr__(self) -> str:
        """Short identity-card representation for REPL display."""
        from .._summary import format_model_repr

        return format_model_repr(self)

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
        return loaded

    def __getstate__(self) -> Dict[str, Any]:
        """Return pickle state with non-picklable weakref-backed accessors stripped."""
        state = self.__dict__.copy()
        state["_module_logs"] = None
        state["_buffer_accessor"] = None
        state["_module_build_data"] = None
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
                "io_format_version": IO_FORMAT_VERSION,
                "activation_postfunc_repr": None,
                "_buffer_accessor": None,
                "_module_logs": None,
                "_module_build_data": None,
                "_activation_writer": None,
                "_keep_activations_in_memory": True,
                "_activation_sink": None,
                "_in_exhaustive_pass": False,
            },
        )
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

    # ********************************************
    # ********** Computed Properties *************
    # ********************************************

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

    def flops_by_type(self) -> Dict[str, Dict[str, int]]:
        """Group FLOPs by layer type.

        Returns:
            Dict mapping layer_type to {"forward": int, "backward": int, "count": int}.
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
        return result

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

    def macs_by_type(self) -> Dict[str, Dict[str, int]]:
        """Group MACs by layer type.

        Returns:
            Dict mapping layer_type to {"forward": int, "backward": int, "count": int}.
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
        return result

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

    # ********************************************
    # ******** Public Convenience Methods ********
    # ********************************************

    def render_graph(
        self,
        vis_mode: str = "unrolled",
        vis_nesting_depth: int = 1000,
        vis_outpath: str = "modelgraph",
        vis_graph_overrides: Optional[Dict] = None,
        vis_node_overrides: Optional[Dict] = None,
        vis_nested_node_overrides: Optional[Dict] = None,
        vis_edge_overrides: Optional[Dict] = None,
        vis_gradient_edge_overrides: Optional[Dict] = None,
        vis_module_overrides: Optional[Dict] = None,
        vis_save_only: bool = False,
        vis_fileformat: str = "pdf",
        show_buffer_layers: bool = False,
        direction: str = "bottomup",
        vis_node_placement: str = "auto",
        vis_renderer: str = "graphviz",
        vis_theme: str = "torchlens",
    ) -> str:
        """Render the computational graph for this model log.

        Parameters
        ----------
        vis_mode, vis_nesting_depth, vis_outpath, vis_graph_overrides, vis_node_overrides, \
        vis_nested_node_overrides, vis_edge_overrides, vis_gradient_edge_overrides, \
        vis_module_overrides, vis_save_only, vis_fileformat, show_buffer_layers, direction, \
        vis_node_placement, vis_renderer, vis_theme:
            Forwarded unchanged to :func:`torchlens.visualization.rendering.render_graph`.

        Returns
        -------
        str
            Graphviz DOT source or renderer-specific output.
        """
        from ..visualization.rendering import render_graph as _impl

        return _impl(
            self,
            vis_mode=vis_mode,
            vis_nesting_depth=vis_nesting_depth,
            vis_outpath=vis_outpath,
            vis_graph_overrides=vis_graph_overrides,
            vis_node_overrides=vis_node_overrides,
            vis_nested_node_overrides=vis_nested_node_overrides,
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
        )

    def summary(
        self,
        level: Literal[
            "overview", "graph", "memory", "control_flow", "compute", "cost"
        ] = "overview",
        *,
        fields: Optional[List[str]] = None,
        mode: Literal["auto", "rolled", "unrolled"] = "auto",
        show_ops: bool = False,
        preset: Optional[
            Literal["overview", "graph", "memory", "control_flow", "compute", "cost"]
        ] = None,
        columns: Optional[List[str]] = None,
        include_ops: Optional[bool] = None,
        max_rows: Optional[int] = 200,
        print_to: Optional[Callable[[str], None]] = None,
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

        Returns
        -------
        str
            Rendered summary string.
        """
        from .._summary import render_model_summary

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
        """Render this model log with the Dagua backend.

        Parameters
        ----------
        vis_mode, vis_nesting_depth, vis_outpath, vis_save_only, vis_fileformat, \
        vis_buffer_layers, vis_direction, vis_theme:
            Forwarded unchanged to
            :func:`torchlens.visualization.dagua_bridge.render_model_log_with_dagua`.

        Returns
        -------
        str
            Serialized Dagua graph output or the rendered artifact path.
        """
        from ..visualization.dagua_bridge import render_model_log_with_dagua as _impl

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
        """Translate this model log into a Dagua graph.

        Parameters
        ----------
        vis_mode, vis_nesting_depth, show_buffer_layers, direction, include_gradient_edges:
            Forwarded unchanged to
            :func:`torchlens.visualization.dagua_bridge.model_log_to_dagua_graph`.

        Returns
        -------
        Any
            Dagua graph object.
        """
        from ..visualization.dagua_bridge import model_log_to_dagua_graph as _impl

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
        from ..visualization.dagua_bridge import build_render_audit as _impl

        return _impl(self)

    def print_all_fields(self) -> None:
        """Print all public non-callable fields on this model log.

        Returns
        -------
        None
            This method prints directly to stdout.
        """
        fields_to_exclude = [
            "layer_list",
            "layer_dict_main_keys",
            "layer_dict_all_keys",
            "raw_layer_dict",
            "decorated_to_orig_funcs_dict",
        ]

        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if not any([attr_name.startswith("_"), attr_name in fields_to_exclude, callable(attr)]):
                print(f"{attr_name}: {attr}")

    def to_pandas(self) -> pd.DataFrame:
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
        self.to_pandas().to_csv(filepath, index=False, **kwargs)

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
        try:
            import pyarrow  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "to_parquet requires pyarrow. Install with: pip install torchlens[io]"
            ) from exc
        self.to_pandas().to_parquet(filepath, **kwargs)

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
        self.to_pandas().to_json(filepath, orient=orient, **kwargs)

    def save_new_activations(
        self,
        model: torch.nn.Module,
        input_args: torch.Tensor | List[Any],
        input_kwargs: Optional[Dict[Any, Any]] = None,
        layers_to_save: str | List = "all",
        random_seed: Optional[int] = None,
    ) -> None:
        """Re-run the model with new inputs, saving only activations.

        Parameters
        ----------
        model, input_args, input_kwargs, layers_to_save, random_seed:
            Forwarded unchanged to
            :func:`torchlens.capture.trace.save_new_activations`.
        """
        from ..capture.trace import save_new_activations as _impl

        return _impl(
            self,
            model=model,
            input_args=input_args,
            input_kwargs=input_kwargs,
            layers_to_save=layers_to_save,
            random_seed=random_seed,
        )

    def validate_saved_activations(
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

    def validate_forward_pass(
        self,
        ground_truth_output_tensors: List[torch.Tensor],
        verbose: bool = False,
        validate_metadata: bool = True,
    ) -> bool:
        """Alias for :meth:`validate_saved_activations`.

        Parameters
        ----------
        ground_truth_output_tensors, verbose, validate_metadata:
            Forwarded unchanged to :meth:`validate_saved_activations`.

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
        random_seed: Optional[int] = None,
    ) -> None:
        """Run a forward pass and capture it into this model log.

        Parameters
        ----------
        model, input_args, input_kwargs, layers_to_save, random_seed:
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
            random_seed=random_seed,
        )

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
        self.conditional_then_edges = [
            edge
            for edge in self.conditional_then_edges
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
