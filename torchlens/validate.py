# This module contains functions for validating the rest of the package and ensuring that they return
# correct outputs. The idea is to exhaustively save everything, and backtracking from the output,
# repeatedly run my own forward pass and make sure it matches the real output, and do this for
# multiple inputs per model.

# Validation steps: first, make sure that all possible key entries in the pretty final output
# corresponding to that tensor are in fact the same tensor data. Then plug that data in, and run
# the feedforward pass from there, and check that it matches the output.

# As a "debug mode", keep ALL functions applied and their arguments without discarding (this might
# require tweaking the logic of expand_multiple_functions).

from typing import Dict, List

import torch
from tqdm import tqdm

from torchlens import ModelHistory, get_all_tensor_lookup_keys, rough_barcode_to_final_barcode
from torchlens import get_rng_states, set_rng_states, tuple_assign


def validate_lookup_keys(history_dict: Dict,
                         pretty_history: ModelHistory,
                         verbose: bool = False) -> bool:
    """Checks whether the saved activations in the raw history_dict data structure match those in the final
    pretty history, based on all possible lookup keys. This is neceessary to make sure that the activations
    being checked are the same as those that the user sees.

    Args:
        history_dict: Raw history dict.
        pretty_history: The pretty user-facing history_dict.
        verbose: True to print the error message if there's no match, false otherwise.

    Returns:
        True if the activations all match, false otherwise.
    """
    tensor_log = history_dict['tensor_log']
    num_nodes = len(tensor_log)
    lookup_keys_valid = True
    for n, (barcode, node) in enumerate(tensor_log.items()):
        lookup_keys = get_all_tensor_lookup_keys(node, n, num_nodes, history_dict)
        raw_activations = node['tensor_contents']
        for lookup_key in lookup_keys:
            user_activations = pretty_history[lookup_key].tensor_contents
            if not torch.equal(raw_activations, user_activations):
                if verbose:
                    print(f"Saved activations for {lookup_key} do not match those in the raw history_dict.")
                lookup_keys_valid = False

    return lookup_keys_valid


def get_starting_node_sets(history_dict: Dict) -> List[List]:
    """Helper function to get a list of all the possible sets of starting nodes that are necessary and
    sufficient to regenerate all outputs if the forward pass starts from those nodes.

    Args:
        history_dict: The history dict.

    Returns:
        List of lists of all starting nodes that are sufficient to regenerate all outputs.
    """
    tensor_log = history_dict['tensor_log']

    # Get the initial node stack: all immediate predecessors of the output nodes.

    output_tensor_barcodes = history_dict['output_tensors']
    node_stack = []
    nodes_seen = set(output_tensor_barcodes)
    for output_tensor_barcode in output_tensor_barcodes:
        output_tensor = tensor_log[output_tensor_barcode]
        for parent_barcode in output_tensor['parent_tensor_barcodes']:
            node_stack.append(parent_barcode)
            nodes_seen.add(parent_barcode)
    starting_node_sets = [node_stack[:]]

    # Now go through and replace each node in the stack with its parents (breadth-first)

    while True:
        node_stack_is_all_inputs = True  # assume true until we find a node that's not an input node.
        for n, node_barcode in enumerate(node_stack):
            node = tensor_log[node_barcode]
            if len(node['parent_tensor_barcodes']) == 0:  # leave an input node alone.
                continue
            else:
                node_stack_is_all_inputs = False
                node_to_remove_barcode = node_stack.pop(n)
                break
        if node_stack_is_all_inputs:  # we're done; no more parent nodes. Add this last one and quit.
            break
        node_to_remove = tensor_log[node_to_remove_barcode]
        for parent_barcode in node_to_remove['parent_tensor_barcodes']:
            if parent_barcode not in nodes_seen:  # only add a node if not seen yet.
                node_stack.append(parent_barcode)
                nodes_seen.add(parent_barcode)
        starting_node_sets.append(node_stack[:])

    return starting_node_sets


def compute_forward_step(node_barcode: str,
                         new_forward_pass_activations: Dict,
                         history_dict: Dict) -> torch.Tensor:
    """Computes a tensor at a given step based on the saved model activations by plugging in the saved
    activations in the relevant parts of the function used to compute the tensor at that step.

    Args:
        node_barcode: barcode of node in question
        new_forward_pass_activations: Dict mapping tensor barcodes to their new forward pass values.
        history_dict: Dictionary of history

    Returns:
        New values of the tensor based on the saved activation values.
    """
    tensor_log = history_dict['tensor_log']
    node = tensor_log[node_barcode]
    node_func = node['funcs_applied'][-1]
    node_funcname = node['funcs_applied_names'][-1]
    node_rng_states = node['func_rng_states']

    if (node['is_model_output']) or (node_funcname == 'identity'):  # just get the child activation.
        parent_barcode = node['parent_tensor_barcodes'][0]
        child_activation = new_forward_pass_activations[parent_barcode].clone()
        return child_activation

    func_args = list(node['creation_args'][:])
    func_kwargs = node['creation_kwargs'].copy()

    # Now make new versions of the args and kwargs based on the saved tensors.

    for arg_loc, arg_barcode in node['parent_tensor_arg_locs']['args'].items():
        if type(arg_loc) == tuple:
            if type(func_args[arg_loc[0]]) == tuple:
                func_args[arg_loc[0]] = tuple_assign(func_args[arg_loc[0]],
                                                     arg_loc[1],
                                                     new_forward_pass_activations[arg_barcode].clone())
            else:
                func_args[arg_loc[0]][arg_loc[1]] = new_forward_pass_activations[arg_barcode].clone()
        else:
            func_args[arg_loc] = new_forward_pass_activations[arg_barcode].clone()

    for kwarg_key, kwarg_barcode in node['parent_tensor_arg_locs']['kwargs'].items():
        if type(kwarg_key) == tuple:
            if type(func_kwargs[kwarg_key[0]]) == tuple:
                func_kwargs[kwarg_key[0]] = tuple_assign(func_kwargs[kwarg_key[0]],
                                                         kwarg_key[1],
                                                         new_forward_pass_activations[kwarg_barcode].clone())
            else:
                func_kwargs[kwarg_key[0]][kwarg_key[1]] = new_forward_pass_activations[kwarg_barcode].clone()
        else:
            func_kwargs[kwarg_key] = new_forward_pass_activations[kwarg_barcode].clone()

    current_rng_states = get_rng_states()
    set_rng_states(node_rng_states)
    new_tensor_val = node_func(*func_args, **func_kwargs)
    set_rng_states(current_rng_states)
    if type(new_tensor_val) in [list, tuple]:
        new_tensor_val = new_tensor_val[node['out_index']]

    for arg in func_args:
        del arg
    for kwarg in func_kwargs.values():
        del kwarg
    return new_tensor_val


def update_node_children(node_barcode: str,
                         new_forward_pass_activations: Dict,
                         new_node_stack: List,
                         history_dict: Dict):
    """Checks the children of a node for whether all their parents have been added; if so, computes their values
    and adds them to the stack, and if not all children are computable yet, adds the current node to the stack too.

    Args:
        node_barcode: barcode of node in question
        new_forward_pass_activations: Dict of activations for each node.
        new_node_stack: Node stack for the next pass
        history_dict: Dict of history
    """
    tensor_log = history_dict['tensor_log']
    node = tensor_log[node_barcode]
    for child_node_barcode in node['child_tensor_barcodes']:
        if child_node_barcode in new_forward_pass_activations:  # if child computed already, we're done
            continue
        child_node = tensor_log[child_node_barcode]
        child_node_parent_barcodes = child_node['parent_tensor_barcodes']

        # If all child's parents are added, compute the new value, add to saved activations, and add to stack.
        if all([child_parent_barcode in new_forward_pass_activations
                for child_parent_barcode in child_node_parent_barcodes]):
            child_activations_orig = compute_forward_step(child_node_barcode,
                                                          new_forward_pass_activations,
                                                          history_dict)
            if type(child_activations_orig) not in [list, tuple, set]:
                child_activations = [child_activations_orig]
            else:
                child_activations = child_activations_orig[:]

            for child_activation in child_activations:
                new_forward_pass_activations[child_node_barcode] = child_activation.clone()
                new_node_stack.append(child_node_barcode)

    # keep the node around if not all children added yet.
    if not all([child_node_barcode in new_forward_pass_activations
                for child_node_barcode in node['child_tensor_barcodes']]) and (node_barcode not in new_node_stack):
        new_node_stack.append(node_barcode)


def validate_single_forward_pass(history_dict: Dict,
                                 starting_node_set: List[str],
                                 perturb: bool = False) -> bool:
    """Checks whether a forward pass beginning from a specified set of starting nodes matches the ground truth output
    values.

    Args:
        history_dict: Raw history dict.
        starting_node_set: Barcodes of the layers to start from.
        perturb: Whether to swap out saved activations for fake activations.

    Returns:
        Whether the outputs of the forward pass from the saved activations match the ground-truth outputs.
    """
    tensor_log = history_dict['tensor_log']
    node_stack = starting_node_set[:]
    # dict mapping tensor barcodes to their saved activations; start with saved values of the starting nodes.

    if not perturb:
        new_forward_pass_activations = {barcode: tensor_log[barcode]['tensor_contents'].clone() for barcode in
                                        node_stack}
    else:
        new_forward_pass_activations = {barcode: torch.rand(tensor_log[barcode]['tensor_contents'].shape,
                                                            device=tensor_log[barcode]['tensor_contents'].device)
                                        for barcode in node_stack}
    output_node_barcodes = history_dict['output_tensors']

    # Keep populating the regenereated forward pass values from the starting nodes, until all output
    # nodes have been populated.

    while True:

        # Check if the forward pass has generated values for all the output nodes yet; if so, break.
        if all([output_node_barcode in new_forward_pass_activations for output_node_barcode in output_node_barcodes]):
            break

        # Go through each node in the stack and check for each child whether all its parents are added;
        # if so, compute the child's value and add node to the stack. That is, being in the stack means
        # that a node's value has been computed and added to the saved activations for the forward pass, and
        # that its children may not have been added yet.

        new_node_stack = []

        for node_barcode in node_stack:
            node = tensor_log[node_barcode]
            if len(node['child_tensor_barcodes']) == 0:  # we've hit an output node, forget about it.
                continue
            update_node_children(node_barcode, new_forward_pass_activations, new_node_stack, history_dict)

        # And start it again, with any newly updated nodes included, keeping around nodes whose children aren't added.
        node_stack = new_node_stack

    # Next, check whether these new values of the output nodes match those of the ground truth.

    for output_node_barcode in output_node_barcodes:
        orig_output_node_values = tensor_log[output_node_barcode]['tensor_contents']
        new_output_node_values = new_forward_pass_activations[output_node_barcode]
        if not torch.equal(orig_output_node_values, new_output_node_values):
            return False

    for t in new_forward_pass_activations.values():
        del t
    new_forward_pass_activations.clear()

    return True


def validate_all_forward_passes(history_dict: Dict,
                                verbose: bool = False) -> bool:
    """Starting from the output, crawls backward and re-runs the forward pass from the saved activations at
    each preceding step.

    Args:
        history_dict: Raw history dict
        verbose: Whether to print where it fails

    Returns:
        True if the forward passes from all saved activations match the ground-truth output, False otherwise.
    """
    starting_node_sets = get_starting_node_sets(history_dict)

    # Now go through each possible set of starting nodes, and check whether it produces outputs
    # that match the ground truth outputs.

    if verbose:  # tqdm indicator if verbose indicated.
        iterator = tqdm(starting_node_sets, desc='Validating forward passes for saved activations')
    else:
        iterator = starting_node_sets

    for starting_node_set in iterator:
        pass_is_valid = validate_single_forward_pass(history_dict, starting_node_set)
        if not pass_is_valid:
            if verbose:
                pretty_starting_nodes = [rough_barcode_to_final_barcode(barcode, history_dict)
                                         for barcode in starting_node_set]
                print(f"\nForward pass failed starting from tensors {pretty_starting_nodes}")
            return False

    return True


def perturb_all_forward_passes(history_dict: Dict,
                               verbose: bool = False):
    """Runs the forward pass from all saved activations, but changes the saved activations, and ensures that
    it in fact results in a wrong output.

    Args:
        history_dict: Raw dict of history.
        verbose: Whether to print results
    """
    starting_node_sets = get_starting_node_sets(history_dict)

    # Now go through each possible set of starting nodes, and check whether it produces outputs
    # that match the ground truth outputs.

    if verbose:  # tqdm indicator if verbose indicated.
        iterator = tqdm(starting_node_sets, desc='Perturbing forward passes for saved activations')
    else:
        iterator = starting_node_sets

    num_starting_sets = len(starting_node_sets)
    num_consequential_perturbations = 0
    for starting_node_set in iterator:
        pass_matches_output = validate_single_forward_pass(history_dict, starting_node_set, perturb=True)
        if not pass_matches_output:
            num_consequential_perturbations += 1

    proportion_consequential_layers = num_consequential_perturbations / num_starting_sets
    return proportion_consequential_layers


def saved_activations_match_child_args(history_dict: Dict,
                                       verbose: bool) -> bool:
    """Checks whether the saved activations at each step match what's fed into the function at the step after.

    Args:
        history_dict: History dict
        verbose: Whether to print when errors happen

    Returns:
        True if all tensors match input into the next step, false otherwise.
    """
    tensor_log = history_dict['tensor_log']
    for barcode, node in tensor_log.items():
        tensor_args = node['creation_args']
        tensor_kwargs = node['creation_kwargs']
        parent_tensor_arg_locs = node['parent_tensor_arg_locs']['args']
        parent_tensor_kwarg_locs = node['parent_tensor_arg_locs']['kwargs']
        for arg_loc, arg_barcode in parent_tensor_arg_locs.items():
            saved_arg_val = tensor_log[arg_barcode]['tensor_contents']
            if type(arg_loc) == tuple:
                actual_arg_val = tensor_args[arg_loc[0]][arg_loc[1]]
            else:
                actual_arg_val = tensor_args[arg_loc]
            if not torch.equal(saved_arg_val, actual_arg_val):
                if verbose:
                    node_pretty_barcode = rough_barcode_to_final_barcode(barcode, history_dict)
                    parent_pretty_barcode = rough_barcode_to_final_barcode(arg_barcode, history_dict)
                    print(f"Activations for {parent_pretty_barcode} do not match input into "
                          f"its child {node_pretty_barcode}")
                return False

        for kwarg_key, kwarg_barcode in parent_tensor_kwarg_locs.items():
            saved_kwarg_val = tensor_log[kwarg_barcode]['tensor_contents']
            if type(kwarg_key) == tuple:
                actual_kwarg_val = tensor_kwargs[kwarg_key[0]][kwarg_key[1]]
            else:
                actual_kwarg_val = tensor_kwargs[kwarg_key]
            if not torch.equal(saved_kwarg_val, actual_kwarg_val):
                if verbose:
                    node_pretty_barcode = rough_barcode_to_final_barcode(barcode, history_dict)
                    parent_pretty_barcode = rough_barcode_to_final_barcode(kwarg_barcode, history_dict)
                    print(f"Activations for {parent_pretty_barcode} do not match input into "
                          f"its child {node_pretty_barcode}")
                return False

    return True


def validate_model_history(history_dict: Dict,
                           pretty_history: ModelHistory,
                           min_proportion_consequential_layers: float = 0,
                           verbose: bool = False) -> bool:
    """Given the raw history_dict and saved user-facing pretty_history, confirms 1) that all saved activations
    in the pretty history match those in the raw history_dict, based on all possible lookup keys, and
    2) that running a forward pass of the model from any of the saved activations yields the exact same
    outputs as the original model. Returns True if this succeeds, False if it fails, along with printing
    where it fails if "verbose" is specified.

    Args:
        history_dict: The raw history_dict
        pretty_history: The pretty, user-facing version of the history dict.
        min_proportion_consequential_layers: Proportion of layers for which changing them affects the output;
            if zero, it doesn't check them at all.
        verbose: Whether to print where the validation fails, if indeed it does.

    Returns:
        True if all activations are validated and yield the same output, False otherwise.
    """

    # First check that for every saved tensor in the raw history_dict, it matches the one in the final, pretty
    # history_dict, for all possible lookup keys. This guarantees that the saved tensor being checked is the
    # one that the user actually sees.

    if not validate_lookup_keys(history_dict, pretty_history, verbose):
        return False

    # Second, check that the saved activations at each step match the
    # corresponding input argument of the step afterwards.

    if not saved_activations_match_child_args(history_dict, verbose):
        return False

    # Third, swap in the saved activations from the final pretty history into each step of the computation graph,
    # crawling back from the final outputs, re-run the forward pass from those activations, and check that they
    # in fact yield the exactly correct final output.

    if not validate_all_forward_passes(history_dict, verbose):
        return False

    # Finally, check that perturbing the saved activations at each step yields a different output, for a minimum
    # threshold of such perturbations.

    if min_proportion_consequential_layers > 0:
        proportion_consequential_layers = perturb_all_forward_passes(history_dict, verbose)
        if proportion_consequential_layers < min_proportion_consequential_layers:
            if verbose:
                print(f"Only {proportion_consequential_layers} of saved activations are consequential, "
                      f"which is less than the minimum {min_proportion_consequential_layers}")
            return False

    return True
