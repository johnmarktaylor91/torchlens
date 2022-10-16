import copy
import random
from collections import OrderedDict
from sys import getsizeof
from typing import Callable, Dict, List, Tuple, Union

import torch
from torch import nn
from torch import Tensor


def make_barcode() -> int:
    """Generates a random integer hash for a layer to use as internal label (invisible from user side).

    Returns:
        Random hash.
    """
    return random.getrandbits(128)


def get_tensor_memory_amount(t: torch.Tensor) -> int:
    """Returns the size of a tensor in bytes.

    Args:
        t: Tensor.

    Returns:
        Size of tensor in bytes.
    """
    return getsizeof(t.storage())


def human_readable_size(size: int, decimal_places: int = 2) -> str:
    """Utility function to convert a size in bytes to a human-readable format.

    Args:
        size: Number of bytes.
        decimal_places: Number of decimal places to use.

    Returns:
        String with human-readable size.
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
        if size < 1024.0 or unit == 'PB':
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"


def readable_tensor_size(t: torch.Tensor) -> str:
    """Returns the size of a tensor in human-readable format.

    Args:
        t: Tensor.

    Returns:
        Human-readable size of tensor.
    """
    return human_readable_size(get_tensor_memory_amount(t))


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
                              hook_fn: Callable,
                              hook_handles: List) -> List:
    """Hook all bottom-level modules in a model with given function and return
    list of hook handles (so as to easily clear later).

    Args:
        model: PyTorch model.
        hook_fn: Hook function.
        hook_handles: Pre-allocated list for storing the hook handles.

    Returns:
        List of tuples (module_pointer, hook_handle) for each module.
    """
    bottom_level_modules = get_bottom_level_modules(model)
    for module_address, module in bottom_level_modules.items():
        hook_handle = module.register_forward_hook(hook_fn)
        hook_handles.append(hook_handle)
    return hook_handles


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


def prepare_model(model: nn.Module,
                  hook_handles: List,
                  mode: str = 'modules_only',
                  which_layers: Union[str, List] = 'all') -> List:
    """Prepares the model to return its inner layer activations.

    Args:
        model: Model to prepare.
        hook_handles: Pre-allocated list to store the hooks.
        mode: Either 'modules_only' for just the modules, or 'exhaustive' for all function calls.
        which_layers: List of layers to include.

    Returns:
        Model with hooks and attributes added.
    """
    if mode not in ['modules_only', 'exhaustive']:
        raise ValueError("Mode must be either 'modules_only' or 'exhaustive'.")

    bottom_level_modules = get_bottom_level_modules(model)
    xray_all_modules_dict = OrderedDict({'which_layers': which_layers,
                                         'mode': mode})
    for module_address, module in bottom_level_modules.items():
        setattr(module, f'xray_this_module_dict', OrderedDict())  # Info for just this module.
        setattr(module, f'xray_all_modules_dict',
                xray_all_modules_dict)  # Pointer to dictionary of info for whole model.
    hook_handles = hook_bottom_level_modules(model,
                                             hook_fn=record_module_forward_pass,
                                             hook_handles=hook_handles)
    return hook_handles


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
        tensor_history = HistoryTensor.tidy_tensor_history
        if visualize_opt != 'none':
            model_graph = make_model_graph(module_output_dict,
                                           tensor_history,
                                           mode,
                                           visualize_opt)
            render_model_graph(model_graph)
        cleanup_model(model, hook_handles)
        if mode == 'modules_only':
            return module_output_dict
        elif mode == 'exhaustive':
            return tensor_history
    except Exception as e:  # if anything fails, clean up the model and re-raise the error.
        print("Execution failed somewhere, returning model to original state...")
        cleanup_model(model, hook_handles)
        HistoryTensor.clear_history()
        raise e


def show_model_graph(model: nn.Module,
                     x: torch.Tensor,
                     mode: str = 'modules_only',
                     visualize_opt: str = 'rolled') -> None:
    """Visualize the model graph.

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

    hook_handles = []
    try:
        hook_handles = prepare_model(model, hook_handles)
        module_output_dict = model.xray_all_modules_dict
        module_output_dict = postprocess_module_output_dict(module_output_dict)
        model_graph = make_model_graph(module_output_dict,
                                       tensor_history=None,
                                       mode=mode,
                                       visualize_opt=visualize_opt)
        render_model_graph(model_graph)
        cleanup_model(model, hook_handles)
    except Exception as e:  # if anything fails, clean up the model and re-raise the error.
        print("Execution failed somewhere, returning model to original state...")
        cleanup_model(model, hook_handles)
        raise e


class HistoryTensor(torch.Tensor):
    tensor_history = OrderedDict()

    def __init__(self, *args, add_to_history=True):
        super().__init__()
        if add_to_history:
            HistoryTensor.add_record(t)

    @staticmethod
    def strip_alias(gfn):
        """Strip away the tedious aliasbackward operations from a grad_fn.

        Args:
            gfn: grad_fn object

        Returns:
            The gfn rolled back to the next "real" operation.
        """
        if gfn is None:
            return gfn

        while type(gfn).__name__ == 'AliasBackward':
            gfn = gfn.next_functions[0][0]

        return gfn

    @classmethod
    def add_record(cls, t):
        barcode = make_barcode()
        real_grad_fn = cls.strip_alias(t.grad_fn)
        if real_grad_fn is not None:
            real_grad_fn.metadata['barcode'] = barcode
        HistoryTensor.tensor_history[barcode] = {'tensor': t,
                                                 'real_grad_fn': real_grad_fn}

    @classmethod
    def clear_tensor_history(cls):
        cls.tensor_history.clear()

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        func_output = super().__torch_function__(func, types, args, kwargs)

        if (func in [torch.Tensor.__repr__, torch.Tensor.__str__]):
            return func_output

        # Do nothing if not a tensor or history tensor.
        if type(func_output) not in [torch.Tensor, HistoryTensor]:
            return func_output

        # Make it a history tensor if it's a tensor.
        if type(func_output) == torch.Tensor:
            func_output = HistoryTensor(func_output, add_to_history=False)

        HistoryTensor.add_record(func_output)
        return func_output


def find_parent_layers_and_operations(t: torch.Tensor):
    """Given an input tensor, crawls back through the computation graph until it finds parameters that have
    been annotated with the layer and operation that birthed them. It should crawl through the tree
    until it hits operations with no parents, or until it finds labeled parents.

    Args:
        t: Input tensor.

    Returns:
        parent_layers, parent_operations
    """
    if t.grad_fn is None:
        return [], []
    if 'layer_name' in t.grad_fn.metadata:
        return [t.grad_fn.metadata['layer_name']], [t.grad_fn.metadata['operation_name']]

    parent_layers = []
    parent_operations = []

    op_stack = list(t.grad_fn.next_functions)
    while len(op_stack) > 0:
        op = op_stack.pop(0)[0]
        if op is None:
            continue

        # Check if it has parameters, and if they're annotated with a layer/operation; if so, grab them.
        if 'layer_name' in op.metadata:
            parent_layers.append(op.metadata['layer_name'])
            parent_operations.append(op.metadata['operation_name'])
            continue

        new_ops = list(op.next_functions)
        op_stack = new_ops + op_stack
    return parent_layers, parent_operations


# TODO: Add the explicit inputs and outputs to the output dict, since these can fail to match if there's non-modules.
def make_layer_return_activations(module, input_, output_):
    """Forward hook to add to a layer to make it return the activations, and also gives it a nice name.

    Args:
        module: The module.
        input_: The input.
        output_: The output.

    Returns:
        Nothing, but it adds to the layer_activation list that can be accessed from the outermost module.
    """

    module.current_layer[0] = module

    # Get previous layers and operations.

    prev_layers = []
    prev_operations = []

    is_first_layer = False
    for i in input_:
        if hasattr(i, 'barcode') and i.barcode == 'INPUT_TENSOR':
            prev_layers = ['input']
            prev_operations = ['input']
            delattr(i, 'barcode')
            is_first_layer = True

    if not is_first_layer:
        for input_entry in input_:
            prev_layers_, prev_operations_ = find_parent_layers_and_operations(input_entry)
            prev_layers += prev_layers_
            prev_operations += prev_operations_

    if module.module_address[0] == 'input':  # tag the input on the way past
        output_.barcode = 'INPUT_TENSOR'

    # Get module type.
    module_type = str(type(module).__name__)

    # Tally how many operations have happened both total and for this network.
    operation_num = len(module.layer_activations)  # how many computation steps have happened  in the network
    module.layer_instance_counter[module.barcode] += 1  # how many times forward has been called on this module

    # Add to the list of layers that the forward pass has seen so far and get the index of this layer.
    if module.layer_num is None:
        module.layer_num = len(module.layer_tracker)
        module.layer_tracker.append(module.barcode)

    # Get how many times this layer type has happened.
    if module.barcode not in module.layer_type_tracker[module_type]:
        module.layer_type_tracker[module_type].append(module.barcode)
        module.layer_type_num = len(module.layer_type_tracker[module_type])

    layer_name = f"{module_type}_{module.layer_type_num}_{module.layer_num}"
    operation_name = f"{layer_name}_{module.layer_instance_counter[module.barcode]}"
    module.layer_sequence.append(layer_name)

    # Make dictionary to flip the parent/child mappings.

    for layer in prev_layers:
        module.parent_layer_dict[layer].append(layer_name)

    for op in prev_operations:
        module.parent_operation_dict[op].append(operation_name)

    if module.module_address[0] == 'input':
        operation_num = 0
        layer_name = 'input'
        module.layer_num = 0
        module_type = 'input'
        module.layer_type_num = 0
        module.module_address = 'input'
        prev_layers = []
        layer_operation_num = 0
        operation_name = 'input'
        input_operations = []
    elif module.module_address[0] == 'output':
        layer_name = 'output'
        module_type = 'output'
        module.layer_type_num = 0
        module.module_address = 'output'
        layer_operation_num = 0
        operation_name = 'output'
    else:
        output_.grad_fn.metadata['layer_name'] = layer_name
        output_.grad_fn.metadata['operation_name'] = operation_name

    module.module_dict = {'operation_num': operation_num,
                          'layer_num': module.layer_num,
                          'layer_type': module_type,
                          'layer_type_num': module.layer_type_num,
                          'layer_name': layer_name,
                          'layer_address': module.module_address,
                          'input_layers': prev_layers,
                          'output_layers': 'output',
                          'layer_operation_num': module.layer_instance_counter[module.barcode],
                          'operation_name': operation_name,
                          'input_operations': prev_operations,
                          'output_operations': 'output',
                          'layer_input': input_,
                          'layer_output': output_
                          }

    module.layer_activations.append(module.module_dict)
    module.layer_activations_dict[operation_name] = module.module_dict

    return output_


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
