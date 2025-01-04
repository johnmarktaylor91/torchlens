# This file is for defining the ModelHistory class that stores the representation of the forward pass.
import copy
from collections import OrderedDict, defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .cleanup import _remove_log_entry, cleanup
from .decorate_torch import decorate_pytorch
from .helper_funcs import (
    human_readable_size,
)
from .interface import (_getitem_after_pass, _getitem_during_pass, _str_after_pass,
                        _str_during_pass, to_pandas, print_all_fields)
from .logging_funcs import save_new_activations
from .model_funcs import cleanup_model, prepare_model
from .postprocess import postprocess
from .tensor_log import RolledTensorLogEntry, TensorLogEntry
from .trace_model import run_and_log_inputs_through_model
from .validation import validate_saved_activations
from .vis import render_graph


# todo add saved_layer field, remove the option to only keep saved layers


class ModelHistory:
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
    ):
        """Object that stores the history of a model's forward pass.
        Both logs the history in real time, and stores a nice
        representation of the full history for the user afterward.
        """
        # Setup:
        activation_postfunc = copy.deepcopy(activation_postfunc)

        # General info
        self.model_name = model_name
        self._pass_finished = False
        self._track_tensors = False
        self.logging_mode = "exhaustive"
        self._pause_logging = False
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
        self.has_saved_gradients = False
        self.mark_input_output_distances = mark_input_output_distances

        # Model structure info
        self.model_is_recurrent = False
        self.model_max_recurrent_loops = 1
        self.model_has_conditional_branching = False
        self.model_is_branching = False

        # Tensor Tracking:
        self.layer_list: List[TensorLogEntry] = []
        self.layer_list_rolled: List[RolledTensorLogEntry] = []
        self.layer_dict_main_keys: Dict[str, TensorLogEntry] = OrderedDict()
        self.layer_dict_all_keys: Dict[str, TensorLogEntry] = OrderedDict()
        self.layer_dict_rolled: Dict[str, RolledTensorLogEntry] = OrderedDict()
        self.layer_labels: List[str] = []
        self.layer_labels_w_pass: List[str] = []
        self.layer_labels_no_pass: List[str] = []
        self.layer_num_passes: Dict[str, int] = OrderedDict()
        self._raw_tensor_dict: Dict[str, TensorLogEntry] = OrderedDict()
        self._raw_tensor_labels_list: List[str] = []
        self._tensor_nums_to_save: List[int] = []
        self._tensor_counter: int = 0
        self.num_operations: int = 0
        self._raw_layer_type_counter: Dict[str, int] = defaultdict(lambda: 0)
        self._unsaved_layers_lookup_keys: Set[str] = set()

        # Mapping from raw to final layer labels:
        self._raw_to_final_layer_labels: Dict[str, str] = {}
        self._final_to_raw_layer_labels: Dict[str, str] = {}
        self._lookup_keys_to_tensor_num_dict: Dict[str, int] = {}
        self._tensor_num_to_lookup_keys_dict: Dict[int, List[str]] = defaultdict(list)

        # Special Layers:
        self.input_layers: List[str] = []
        self.output_layers: List[str] = []
        self.buffer_layers: List[str] = []
        self.buffer_num_passes: Dict = {}
        self.internally_initialized_layers: List[str] = []
        self._layers_where_internal_branches_merge_with_input: List[str] = []
        self.internally_terminated_layers: List[str] = []
        self.internally_terminated_bool_layers: List[str] = []
        self.conditional_branch_edges: List[Tuple[str, str]] = []
        self.layers_with_saved_activations: List[str] = []
        self.orphan_layers: List[str] = []
        self.unlogged_layers: List[str] = []
        self.layers_with_saved_gradients: List[str] = []
        self.layers_computed_with_params: Dict[str, List] = defaultdict(list)
        self.equivalent_operations: Dict[str, set] = defaultdict(set)
        self.same_layer_operations: Dict[str, list] = defaultdict(list)

        # Tensor info:
        self.num_tensors_total: int = 0
        self.tensor_fsize_total: int = 0
        self.tensor_fsize_total_nice: str = human_readable_size(0)
        self.num_tensors_saved: int = 0
        self.tensor_fsize_saved: int = 0
        self.tensor_fsize_saved_nice: str = human_readable_size(0)

        # Param info:
        self.total_param_tensors: int = 0
        self.total_param_layers: int = 0
        self.total_params: int = 0
        self.total_params_fsize: int = 0
        self.total_params_fsize_nice: str = human_readable_size(0)

        # Module info:
        self.module_addresses: List[str] = []
        self.module_types: Dict[str, Any] = {}
        self.module_passes: List = []
        self.module_num_passes: Dict = defaultdict(lambda: 1)
        self.top_level_modules: List = []
        self.top_level_module_passes: List = []
        self.module_children: Dict = defaultdict(list)
        self.module_pass_children: Dict = defaultdict(list)
        self.module_nparams: Dict = defaultdict(lambda: 0)
        self.module_num_tensors: Dict = defaultdict(lambda: 0)
        self.module_pass_num_tensors: Dict = defaultdict(lambda: 0)
        self.module_layers: Dict = defaultdict(list)
        self.module_pass_layers: Dict = defaultdict(list)
        self.module_layer_argnames = defaultdict(list)

        # Time elapsed:
        self.pass_start_time: float = 0
        self.pass_end_time: float = 0
        self.elapsed_time_setup: float = 0
        self.elapsed_time_forward_pass: float = 0
        self.elapsed_time_cleanup: float = 0
        self.elapsed_time_total: float = 0
        self.elapsed_time_function_calls: float = 0
        self.elapsed_time_torchlens_logging: float = 0

        # Reference info
        self.func_argnames: Dict[str, tuple] = defaultdict(lambda: tuple([]))

    # ********************************************
    # ************ Built-in Methods **************
    # ********************************************

    def __len__(self):
        if self._pass_finished:
            return len(self.layer_list)
        else:
            return len(self._raw_tensor_dict)

    def __getitem__(self, ix) -> TensorLogEntry:
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
        if self._pass_finished:
            return _str_after_pass(self)
        else:
            return _str_during_pass(self)

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        """Loops through all tensors in the log."""
        if self._pass_finished:
            return iter(self.layer_list)
        else:
            return iter(list(self._raw_tensor_dict.values()))

    # ********************************************
    # ******** Assign Imported Methods ***********
    # ********************************************

    render_graph = render_graph
    print_all_fields = print_all_fields
    to_pandas = to_pandas
    save_new_activations = save_new_activations
    validate_saved_activations = validate_saved_activations
    cleanup = cleanup
    _postprocess = postprocess
    _decorate_pytorch = decorate_pytorch
    _prepare_model = prepare_model
    _cleanup_model = cleanup_model
    _run_and_log_inputs_through_model = run_and_log_inputs_through_model
    _remove_log_entry = _remove_log_entry
