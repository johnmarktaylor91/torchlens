import copy
import random
from collections import defaultdict, OrderedDict
import multiprocessing as mp
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from torch_func_handling import log_tensor_metadata
from xray_utils import get_vars_of_type_from_obj, remove_list_duplicates


def get_module_from_address(model: nn.Module, address: str) -> nn.Module:
    """Given a model and an address to a layer, returns the module at that address.
    The address gives nested instructions for going from the top-level module to the desired module,
    in the format 'level1.level2.level3', etc. If a level is a string, then it'll be
    indexed by looking it up as an attribute; if an integer greater than or equal to zero,
    it'll be indexed like a list. For example, 'classifier.1' will first grab
    the 'classifier' attribute of the network, and then go into the second element of that list.

    Args:
        model: PyTorch model.
        address: String address.

    Returns:
        The module at the given address.
    """
    address = address.split('.')
    module = model
    for a in address:
        if (a.isdigit()) and (int(a) >= 0):
            module = module[int(a)]
        else:
            module = getattr(module, a)
    return module


def make_layer_name(layer_type: str,
                    layer_type_num: int,
                    layer_num: int,
                    pass_num: Optional[int] = None) -> str:
    """Makes a string name for a layer, given its type, type number, layer number, and pass number."""
    layer_name = f"{layer_type}{layer_type_num}_{layer_num}"
    if pass_num is not None:
        layer_name += f":{pass_num}"
    return layer_name


def text_num_split(s: str) -> Tuple[str, int]:
    """Utility function that takes in a string that begins with letters and ends in a number,
    and splits it into the letter part and the number part, returning both in a tuple.

    Args:
        s: String

    Returns:
        Tuple containing the beginning string and ending number.
    """
    s = s.strip()
    num = ''
    while s[-1].isdigit():
        num = s[-1] + num
        s = s[:-1]
    return s, int(num)


def parse_layer_name(layer_name: str) -> OrderedDict:
    """Given layer name, decomposes it into the relevant values.

    Args:
        layer_name: Name of the layer in format {layer_type}{layer_type_num}_{layer_num}:{pass_num},
            with pass_num optional. For example, conv4_9:2 is the second pass through the 4th convolutional
            layer, which is the 9th layer overall.
    Returns:
        Dict with layer_type, layer_type_num, layer_num, and pass_num if it's there.
    """
    layer_dict = OrderedDict()
    if ':' in layer_name:
        layer_name, pass_num = layer_name.split(':')
        layer_dict['pass_num'] = int(pass_num)
    else:
        pass_num = None

    type_label, layer_num = layer_name.split('_')
    layer_type, layer_type_num = text_num_split(type_label)
    layer_dict['layer_type'] = layer_type
    layer_dict['layer_type_num'] = layer_type_num
    layer_dict['layer_num'] = layer_num
    if pass_num:
        layer_dict['pass_num'] = pass_num
    return layer_dict


def get_bottom_level_modules(model: nn.Module) -> Dict[str, nn.Module]:
    """Recursively crawls through a given model, and returns a dict of the bottom-level
    modules, with keys corresponding to their address, and values corresponding to the
    modules themselves.

    Args:
        model: PyTorch model.

    Returns:
        Dict of bottom-level modules.
    """
    module_stack = [('', model)]
    module_dict = OrderedDict()
    while len(module_stack) > 0:
        module_tuple = module_stack.pop(0)
        module_address, module = module_tuple
        module_children = list(module.named_children())
        if len(module_children) == 0:
            module_dict[module_address] = module
        else:
            children_to_add = []
            for name, child in module_children:
                child_address = f"{module_address}.{name}" if module_address else name
                children_to_add.append((child_address, child))
            module_stack = children_to_add + module_stack
    return module_dict


def get_all_submodules(model: nn.Module) -> List[nn.Module]:
    """Recursively gets list of all submodules for given module, no matter their level in the
    hierarchy; this includes the model itself.

    Args:
        model: PyTorch model.

    Returns:
        List of all submodules.
    """
    submodules = [model]
    for module in model.children():
        submodules.append(module)
        submodules += get_all_submodules(module)
    return submodules


def hook_bottom_level_modules(model: nn.Module,
                              hook_fns: List[Callable],
                              prehook_fns: List[Callable],
                              hook_handles: List) -> List:
    """Hook all bottom-level modules in a model with given function and return
    list of hook handles (so as to easily clear later).

    Args:
        model: PyTorch model.
        hook_fns: List of hook functions to add.
        prehook_fns: List of pre-hook functions to add.
        hook_handles: Pre-allocated list for storing the hook handles.

    Returns:
        List of tuples (module_pointer, hook_handle) for each module.
    """
    bottom_level_modules = get_bottom_level_modules(model)
    for module_address, module in bottom_level_modules.items():
        for hook_fn in hook_fns:
            hook_handle = module.register_forward_hook(hook_fn)
            hook_handles.append(hook_handle)
        for prehook_fn in prehook_fns:
            hook_handle = module.register_forward_pre_hook(prehook_fn)
            hook_handles.append(hook_handle)
    return hook_handles


def find_tensor_parent_modules_and_operations(t: Union[torch.Tensor]) -> Tuple[List[str], List[str]]:
    """Given an input tensor, traverses backwards through the computation graph, until it finds
    its immediate module parents; returns none if it has no module parents.

    Args:
        t: Input tensor.

    Returns:
        parent_modules
    """
    if t.grad_fn is None:
        return []
    if 'xray_module_parent' in t.grad_fn.metadata:
        return [t.grad_fn.metadata['xray_module_parent']], [t.grad_fn.metadata['xray_module_operation_parent']]

    parent_modules = []
    parent_operations = []
    gfn_stack = list(t.grad_fn.next_functions)
    while len(gfn_stack) > 0:
        op = gfn_stack.pop(0)[0]
        if op is None:
            continue

        # Check if it's annotated with a parent module.
        if 'xray_module_parent' in op.metadata:
            parent_modules.append(op.metadata['xray_module_parent'])
            parent_operations.append(op.metadata['xray_module_operation_parent'])
            continue
        if 'xray_origin_parent' in op.metadata:  # covers cases where it's the input tensor or a newly made tensor.
            origin_parent = op.metadata['xray_origin_parent']
            parent_modules.append(origin_parent)
            parent_operations.append(origin_parent)
            continue

        new_ops = list(op.next_functions)
        gfn_stack = new_ops + gfn_stack

    return parent_modules, parent_operations


def find_tensor_list_parent_modules_and_operations(tensor_list: List[torch.Tensor]) -> Tuple[List[str], List[str]]:
    """Convenience function to take a list of tensors, and return the union of all their parents.

    Args:
        tensor_list: List of tensors.

    Returns:
        Union of all parent modules, with duplicates removed.
    """
    parent_modules = []
    parent_operations = []
    for t in tensor_list:
        new_parent_modules, new_parent_operations = find_tensor_parent_modules_and_operations(t)
        parent_modules += new_parent_modules
        parent_operations += new_parent_operations

    parent_modules = remove_list_duplicates(parent_modules)
    parent_operations = remove_list_duplicates(parent_operations)

    return parent_modules, parent_operations


def module_pre_hook(module: nn.Module,
                    input_: tuple):
    """Pre-hook to attach to the modules: it marks the tensors as currently being inside a module, and
    indicates which module it is.

    Args:
        module: PyTorch module.
        input_: The input.

    Returns:
        The input, now marked with information about the module it's entering.
    """
    module_address = module.xray_module_address
    module.xray_module_pass_num += 1
    input_tensors = get_vars_of_type_from_obj(input_, torch.Tensor)
    for t in input_tensors:
        t.xray_containing_modules_nested.append(module_address)
        t.xray_containing_module = module_address
        t.xray_entered_module = True
        t.xray_last_module_seen_address = module_address
        t.xray_last_module_seen = module

        log_tensor_metadata(t)  # Update tensor log with this new information.


def module_post_hook(module: nn.Module,
                     input_,
                     output_):
    """Hook to run after the module is executed: it marks the tensors as no longer being inside a module,
    and indicates which module it is.

    Args:
        module: The module.
        input_: The input.
        output_: The output.

    Returns:
        Nothing, but records all relevant data.
    """

    output_tensors = get_vars_of_type_from_obj(output_, torch.Tensor)
    for t in output_tensors:
        t.xray_is_module_output = True
        t.xray_containing_modules_nested.pop()  # remove the last module address.
        if module.xray_is_bottom_level_module:
            t.xray_is_bottom_level_module_output = True
        else:
            t.xray_is_bottom_level_module_output = False

        if len(t.xray_tensor_in_these_modules) == 0:
            t.xray_entered_module = False
            t.xray_containing_module = None
        else:
            t.xray_containing_module = t.xray_tensor_in_these_modules[-1]

        log_tensor_metadata(t)  # Update tensor log with this new information.

    return output_


def prepare_model(model: nn.Module,
                  hook_handles: List,
                  mode: str = 'modules_only',
                  which_layers: Union[str, List] = 'all') -> List:
    """Adds annotations and hooks to the model.

    Args:
        model: Model to prepare.
        hook_handles: Pre-allocated list to store the hooks so they can be cleared even if execution fails.
        mode: Either 'modules_only' for just the modules, or 'exhaustive' for all function calls.
        which_layers: List of layers to include.

    Returns:
        Model with hooks and attributes added.
    """
    if mode not in ['modules_only', 'exhaustive']:
        raise ValueError("Mode must be either 'modules_only' or 'exhaustive'.")

    module_stack = list(model.named_children())  # list of tuples (name, module)
    while len(module_stack) > 0:
        address, module = module_stack.pop()
        module_children = list(module.named_children())
        # Annotate the children with the full address.
        for child_name, child_module in module_children:
            child_module.xray_module_address = f"{address}.{child_name}"
        module_stack = module_children + module_stack
        if len(module_children) == 0:
            is_bottom_level_module = True
        else:
            is_bottom_level_module = False

        module.xray_module_address = address
        module.xray_module_type = str(type(module).__name__).lower()
        module.xray_is_bottom_level_module = is_bottom_level_module
        module.xray_module_pass_num = 0

        # Add hooks.

        hook_handles.append(module.register_forward_pre_hook(module_pre_hook))
        hook_handles.append(module.register_forward_hook(module_post_hook))


def clear_hooks(hook_handles: List):
    """Takes in a list of tuples (module, hook_handle), and clears the hook at that
    handle for each module.

    Args:
        hook_handles: List of tuples (module, hook_handle)

    Returns:
        Nothing.
    """
    for hook_handle in hook_handles:
        hook_handle.remove()


def clear_model_keyword_attributes(model: nn.Module, attribute_keyword: str = 'xray'):
    """Recursively clears the given attribute from all modules in the model.

    Args:
        model: PyTorch model.
        attribute_keyword: Any attribute with this keyword will be cleared.

    Returns:
        Nothing.
    """
    for module in get_all_submodules(model):
        for attribute_name in dir(module):
            if attribute_keyword in attribute_name:
                delattr(module, attribute_name)


def cleanup_model(model: nn.Module, hook_handles: List) -> nn.Module:
    """Reverses all temporary changes to the model (namely, the forward hooks and added
    model attributes) that were added for PyTorch x-ray (scout's honor; leave no trace).

    Args:
        model: PyTorch model.
        hook_handles: List of hooks.

    Returns:
        Original version of the model.
    """
    clear_hooks(hook_handles)
    clear_model_keyword_attributes(model, attribute_keyword='xray')
    return model


def xray_model(model: nn.Module,
               x: torch.Tensor,
               mode: str = 'modules_only',
               which_layers: Union[str, List] = 'all',
               visualize_opt: str = 'none') -> OrderedDict[str, OrderedDict]:
    """Run a forward pass through a model, and return activations of desired hidden layers.
    Specify mode as 'modules_only' to do so only for proper modules, or as 'exhaustive' to
    also return activations from non-module functions. If only a subset of layers
    is desired, specify the list of layer names (e.g., 'conv1_5') in which_layers; if you wish to
    further specify that only certain passes through a layer should be saved
    (i.e., in a recurrent network, only save the third pass through a layer), then
    add :{pass_number} to the layer name (e.g., 'conv1_5:3'). Additionally, the graph
    can be visualized if desire to see the architecture and easily reference the names.

    Args:
        model: PyTorch model
        x: desired Tensor input.
        mode: 'modules_only' to return activations only for module objects, or
            'exhaustive' to do it for ALL tensor operations.
        which_layers: List of layers to include. If 'all', then include all layers.
        visualize_opt: Whether, and how, to visualize the network; 'none' for
            no visualization, 'rolled' to show the graph in rolled-up format (i.e.,
            one node per layer if a recurrent network), or 'unrolled' to show the graph
            in unrolled format (i.e., one node per pass through a layer if a recurrent)

    Returns:
        activations: Dict of dicts with the activations from each layer.
    """
    if mp.current_process().name != 'MainProcess':
        print("WARNING: It looks like you are using parallel execution; it is strongly advised"
              "to only run pytorch-xray in the main process, since certain operations "
              "depend on execution order.")

    x = copy.deepcopy(x)

    if mode not in ['modules_only', 'exhaustive']:
        raise ValueError("Mode must be either 'modules_only' or 'exhaustive'.")
    if visualize_opt not in ['none', 'rolled', 'unrolled']:
        raise ValueError("Visualization option must be either 'none', 'rolled', or 'unrolled'.")

    hook_handles = []
    HistoryTensor.clear_history()
    x_history = HistoryTensor(x, mode, which_layers)

    # Wrap everything in a try-except block to guarantee the model remains unchanged in case of error.
    try:
        hook_handles = prepare_model(model, hook_handles)
        output = model(x_history)
        module_output_dict = model.xray_all_modules_dict
        module_output_dict = postprocess_module_output_dict(module_output_dict)
        HistoryTensor.postprocess_tensor_history()
        tensor_history = HistoryTensor.copy_tidy_tensor_history()
        if visualize_opt != 'none':
            model_graph = make_model_graph(module_output_dict,
                                           tensor_history,
                                           mode,
                                           visualize_opt)
            render_model_graph(model_graph)
        cleanup_model(model, hook_handles)
        HistoryTensor.clear_history()
        if mode == 'modules_only':
            return module_output_dict
        elif mode == 'exhaustive':
            return tensor_history
    finally:  # if anything fails, clean up the model and re-raise the error.
        print("Execution failed somewhere, returning model to original state...")
        cleanup_model(model, hook_handles)
        HistoryTensor.clear_history()
        raise e


def show_model_graph(model: nn.Module,
                     x: torch.Tensor,
                     mode: str = 'modules_only',
                     visualize_opt: str = 'rolled') -> None:
    """Visualize the model graph without saving any activations.

    Args:
        model: PyTorch model.
        x: Input for which you want to visualize the graph (this is needed in case the graph varies based on input)
        mode: 'modules_only' to only view modules, 'exhaustive' to
            view all tensor operations.
        visualize_opt: 'rolled' to show the graph in rolled-up format (one node
            per layer, even if multiple passes), or 'unrolled' to view with
            one node per operation.

    Returns:
        Nothing.
    """
    if mode not in ['modules_only', 'exhaustive']:
        raise ValueError("Mode must be either 'modules_only' or 'exhaustive'.")
    if visualize_opt not in ['none', 'rolled', 'unrolled']:
        raise ValueError("Visualization option must be either 'none', 'rolled', or 'unrolled'.")

    # Simply call xray_model without saving any layers.
    _ = xray_model(model, x, mode, which_layers=[], visualize_opt=visualize_opt)


def list_model_layers(model: nn.Module,
                      x: torch.Tensor,
                      mode: str = 'modules_only',
                      unrolled: bool = False) -> List[str]:
    """List the layers in a model.

    Args:
        model: PyTorch model.
        x: Input for which you want to visualize the graph (this is needed in case the graph varies based on input)
        mode: 'modules_only' to only view modules, 'exhaustive' to
            view all tensor operations.
        unrolled: Whether to list each layer once, or list each computational step (if recurrent).

    Returns:
        List of layer names.
    """
    if mode not in ['modules_only', 'exhaustive']:
        raise ValueError("Mode must be either 'modules_only' or 'exhaustive'.")

    # Simply call xray_model without saving any layers.
    layer_dict = xray_model(model, x, mode, which_layers=[], visualize_opt='none')
    layer_list = list(layer_dict.keys())

    if not unrolled:
        layer_list = [layer.split(':')[0] for layer in layer_list]
        layer_list = remove_list_duplicates(layer_list)

    return layer_list


class TransparentModel(nn.Module):
    def __init__(self, network, which_activations='modules_only'):
        """Alters a network to make it "transparent": it remains identical, but now
        stores a list of the layer activations after each forward pass, along with an option to delete
        these activations to save memory.

        Args:
            network: Any PyTorch network.
        """
        super(TransparentModel, self).__init__()
        network = copy.deepcopy(network)  # deep copy so user can keep the original.
        self.network = network
        self.input_identity = nn.Identity()  # This is for simplicity of coding to make everything equal.
        self.output_identity = nn.Identity()
        self.which_activations = which_activations

        # Now we have to crawl through the network and add the forward hook to all of them.
        # The logic is, if it has children, crawl into those, and if it has no children, add the hook.

        layer_activations = []  # list of layer activations
        layer_activations_dict = {}  # dictionary of layer activations
        layer_type_tracker = defaultdict(
            list)  # keeps hashes for each type of layer as it's encountered (to count layer types)
        layer_instance_counter = defaultdict(lambda: 0)  # for each layer, counts how many forward passes it's seen
        layer_tracker = []
        layer_sequence = []
        current_layer = [None]
        parent_layer_dict = defaultdict(list)
        parent_operation_dict = defaultdict(list)
        module_stack = [(self, ())]
        while len(module_stack) > 0:
            module_tuple = module_stack.pop(0)
            module, module_address = module_tuple
            if len(list(module.children())) > 0:
                module_children = []
                for name, child in module.named_children():
                    if name == 'input_identity':
                        name = 'input'
                    elif name == 'output_identity':
                        name = 'output'
                    if name.isdigit():
                        name = int(name)
                    child_address = module_address + (name,)
                    if child_address[0] == 'network':
                        child_address = child_address[1:]
                    module_children.append((child, child_address))
                module_stack = module_children + module_stack
            else:
                module.layer_type_tracker = layer_type_tracker
                module.layer_instance_counter = layer_instance_counter
                module.layer_tracker = layer_tracker
                module.layer_sequence = layer_sequence
                module.layer_activations = layer_activations
                module.layer_activations_dict = layer_activations_dict
                module.parent_layer_dict = parent_layer_dict
                module.parent_operation_dict = parent_operation_dict
                module.current_layer = current_layer
                module.layer_num = None
                module.barcode = random.getrandbits(128)
                module.module_address = module_address
                module.register_forward_hook(make_layer_return_activations)

        self._layer_activations = layer_activations
        self.layer_activations_dict = layer_activations_dict
        self.layer_type_tracker = layer_type_tracker
        self.parent_layer_dict = parent_layer_dict
        self.parent_operation_dict = parent_operation_dict
        self.layer_instance_counter = layer_instance_counter

    def clear_activations(self):
        self._layer_activations.clear()
        self.layer_type_tracker.clear()
        self.layer_tracker.clear()
        self.layer_instance_counter.clear()

    @property
    def layer_activations(self):
        if not self._layer_activations:
            raise ValueError("No layer activations yet. Run the network first.")
        return self._layer_activations

    # NEW IDEA: subclass Tensor and make it save every new tensor that's created, this IS the activations,
    # cross-reference with the modules that receive the tensor. Explore whether best way is subclass,
    # vs. bolting on new methods. All this ugliness can be contained inside the network without frightening users.

    def fill_out_next_layers_and_operations(self):
        """
        Fills in the next layers and operations for each layer.

        """
        for layer in self.layer_activations:
            if layer['layer_name'] not in self.parent_layer_dict:
                layer['output_layers'] = []
                layer['output_operations'] = []
            else:
                layer['output_layers'] = self.parent_layer_dict[layer['layer_name']]
                layer['output_operations'] = self.parent_operation_dict[layer['operation_name']]

    def forward(self, x):
        x = self.input_identity(x)
        output = self.network(x)
        output = self.output_identity(output)
        self.fill_out_next_layers_and_operations()
        if self.which_activations == 'exhaustive':
            self.get_activations_between_modules()
        return output, self.layer_activations


def visualize_model(model, x, mode='modules'):
    """Visualize the computational graph of a model given an input (since the graph can be
    input-specific in the case of any if/then branching in the forward pass).

    Args:
        model: Model
        x: Input
        mode: Either 'modules' (to show at the module level, in "rolled up" format if recurrent),
            'modules-unrolled' to show the unrolled module-level graph, showing each pass
            of each module separately, or 'functions' to visualize the full function graph.

    Returns:

    """
    pass


def list_model(model, x, mode='modules'):
    """Visualize the computational graph of a model given an input (since the graph can be
    input-specific in the case of any if/then branching in the forward pass).

    Args:
        model: Model
        x: Input
        mode: Either 'modules' (to show at the module level, in "rolled up" format if recurrent),
            'modules-unrolled' to show the unrolled module-level graph, showing each pass
            of each module separately, or 'functions' to visualize the full function graph.

    Returns:

    """
    pass


def make_model_transparent(model: nn.Module):
    """Prepares a model for returning the internal activations.

    Args:
        model: PyTorch model.

    Returns:
        Transparent model.
    """
    transparent_model = TransparentModel(model)
    return transparent_model
