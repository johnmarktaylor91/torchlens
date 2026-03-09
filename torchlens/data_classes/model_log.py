"""ModelLog: the top-level container for a fully logged forward pass.

ModelLog is the root data structure returned by ``log_forward_pass()``.
It owns every LayerPassLog (per-operation entry), every LayerLog (per-layer
aggregate), the module hierarchy, parameter metadata, and graph-level
bookkeeping.

Key design patterns:

* **_pass_finished behavioural switch** — Many methods (``__len__``, ``__getitem__``,
  ``__str__``, ``__iter__``) behave differently during logging vs after
  postprocessing.  While logging is active (``_pass_finished=False``), the
  model's tensors are keyed by their raw internal barcodes in
  ``_raw_layer_dict``.  After postprocessing flips ``_pass_finished=True``,
  the friendly ``layer_list`` / ``layer_dict_all_keys`` / ``layer_logs``
  structures are populated and used instead.  ``_pass_finished`` also
  persists across the fast pass on purpose: fast-path postprocessing
  relies on the fully-populated lookup dicts from the exhaustive pass.

* **Method importation** — Several heavy methods (``render_graph``,
  ``save_new_activations``, ``validate_saved_activations``, etc.) are
  defined in other modules and bound to ModelLog as class attributes at
  the bottom of this file.  This keeps the class body small while giving
  users ``model_log.render_graph(...)`` syntax.

* **_module_build_data** — A transient dict that accumulates module hierarchy
  information during the forward pass.  Consumed by ``_build_module_logs``
  (postprocessing step 17) and then cleared.  Initialised via
  ``_init_module_build_data()``.
"""

import copy
from collections import OrderedDict, defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from .buffer_log import BufferAccessor
    from .layer_log import LayerAccessor

from .cleanup import _remove_log_entry, _batch_remove_log_entries, cleanup, release_param_refs
from .module_log import ModuleAccessor
from .param_log import ParamAccessor
from ..utils.display import human_readable_size
from .interface import (
    _getitem_after_pass,
    _getitem_during_pass,
    _str_after_pass,
    _str_during_pass,
    to_pandas,
    print_all_fields,
)
from ..capture.trace import save_new_activations
from ..postprocess import postprocess
from .layer_log import LayerLog
from .layer_pass_log import LayerPassLog, TensorLog
from ..capture.trace import run_and_log_inputs_through_model
from ..validation import validate_saved_activations
from ..validation import check_metadata_invariants
from ..visualization.rendering import render_graph


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


class ModelLog:
    """Top-level container for a logged forward pass.

    Serves double duty: during the forward pass it accumulates raw tensor
    metadata in ``_raw_layer_dict``; after postprocessing (``_pass_finished=True``)
    it presents a clean, user-facing view via ``layer_list``, ``layer_dict_all_keys``,
    ``layer_logs``, ``modules``, ``params``, and ``buffers``.

    Supports ``len()``, iteration, and flexible ``__getitem__`` lookup by
    integer index, layer label, module address, or substring.
    """

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
        # Callables are effectively immutable — deepcopy is unnecessary.

        # General info
        self.model_name = model_name
        self.num_context_lines = num_context_lines
        self._optimizer = optimizer
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
        self.has_saved_gradients = False
        self.mark_input_output_distances = mark_input_output_distances

        # Model structure info (computed @properties: model_is_recurrent,
        # model_max_recurrent_loops, model_is_branching, model_has_conditional_branching)

        # Tensor Tracking — post-processed (populated after _pass_finished=True):
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
        # Tensor Tracking — raw (populated during the forward pass, before postprocessing):
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
        self.layers_with_saved_activations: List[str] = []
        self.orphan_layers: List[str] = []
        self.unlogged_layers: List[str] = []
        self.layers_with_saved_gradients: List[str] = []
        self._saved_gradients_set: set = set()
        self.layers_computed_with_params: Dict[str, List] = defaultdict(list)
        # Maps operation_equivalence_type -> set of layer labels that share
        # that equivalence type (populated by loop_detection.py).
        self.equivalent_operations: Dict[str, set] = defaultdict(set)

        # Aggregate tensor statistics (computed during postprocessing):
        self.tensor_fsize_total: int = 0
        self.num_tensors_saved: int = 0  # layers with has_saved_activations=True
        self.tensor_fsize_saved: int = 0

        # Param info:
        self.param_logs: "ParamAccessor" = ParamAccessor({})
        self.total_param_tensors: int = 0
        self.total_param_layers: int = 0
        self.total_params: int = 0
        self.total_params_trainable: int = 0
        self.total_params_frozen: int = 0
        self.total_params_fsize: int = 0

        # Session-scoped per-module tracking dicts (keyed by id(module)).
        # These replace the old tl_* attrs that were set directly on nn.Module
        # instances. Lives on ModelLog so they're GC'd with the log — no cleanup
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
        self.elapsed_time_setup: float = 0
        self.elapsed_time_forward_pass: float = 0
        self.elapsed_time_cleanup: float = 0
        self.elapsed_time_function_calls: float = 0

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
        """Same as __str__."""
        return self.__str__()

    def __iter__(self):
        """Loops through all tensors in the log."""
        if self._pass_finished:
            return iter(self.layer_list)
        else:
            return iter(list(self._raw_layer_dict.values()))

    # ********************************************
    # ********** Computed Properties *************
    # ********************************************

    @property
    def model_is_recurrent(self) -> bool:
        """Whether any layer has more than one pass."""
        return any(v > 1 for v in self.layer_num_passes.values())

    @property
    def model_max_recurrent_loops(self) -> int:
        """Maximum number of passes for any layer."""
        return max(self.layer_num_passes.values(), default=1)

    @property
    def model_is_branching(self) -> bool:
        """Whether any layer has more than one child."""
        return any(len(entry.child_layers) > 1 for entry in self.layer_list)

    @property
    def model_has_conditional_branching(self) -> bool:
        """Whether any layer is in a conditional branch."""
        return any(entry.in_cond_branch for entry in self.layer_list)

    @property
    def num_tensors_total(self) -> int:
        """Total number of tensor operations."""
        return len(self)

    @property
    def tensor_fsize_total_nice(self) -> str:
        """Human-readable total tensor size."""
        return human_readable_size(self.tensor_fsize_total)

    @property
    def tensor_fsize_saved_nice(self) -> str:
        """Human-readable saved tensor size."""
        return human_readable_size(self.tensor_fsize_saved)

    @property
    def elapsed_time_total(self) -> float:
        """Total time from start to end of pass."""
        if not self.pass_start_time or not self.pass_end_time:
            return 0
        return self.pass_end_time - self.pass_start_time

    @property
    def elapsed_time_torchlens_logging(self) -> float:
        """Time spent on TorchLens overhead (total minus function calls)."""
        return self.elapsed_time_total - self.elapsed_time_function_calls

    # ********************************************
    # ************* FLOPs Properties *************
    # ********************************************
    # FLOPs are estimated per-operation during logging (flops_forward,
    # flops_backward on each LayerPassLog).  These properties aggregate
    # across the entire model.  Layers with None FLOPs (unknown ops) are
    # skipped, so the totals may undercount.

    @property
    def total_params_fsize_nice(self) -> str:
        return human_readable_size(self.total_params_fsize)

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
    # ******** Assign Imported Methods ***********
    # ********************************************
    # These are functions defined in other modules, bound here as class
    # attributes so they can be called as instance methods
    # (e.g. ``model_log.render_graph(...)``).  This pattern keeps the
    # ModelLog class body small while co-locating heavy logic in its
    # own module (visualization, validation, capture, etc.).

    render_graph = render_graph
    print_all_fields = print_all_fields
    to_pandas = to_pandas
    save_new_activations = save_new_activations
    validate_saved_activations = validate_saved_activations
    validate_forward_pass = validate_saved_activations  # user-facing alias
    check_metadata_invariants = check_metadata_invariants
    cleanup = cleanup
    release_param_refs = release_param_refs
    _postprocess = postprocess
    _run_and_log_inputs_through_model = run_and_log_inputs_through_model
    _remove_log_entry = _remove_log_entry
    _batch_remove_log_entries = _batch_remove_log_entries
