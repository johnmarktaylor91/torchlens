# ruff: noqa: E741  (verbatim upstream uses single-letter `l` for a tensor-length var)
# SOURCE: vendored from peterbjorgensen/DeepDFT @ main
# https://github.com/peterbjorgensen/DeepDFT/blob/main/densitymodel.py
# https://github.com/peterbjorgensen/DeepDFT/blob/main/layer.py
#
# DeepDFT: message-passing neural network for charge density prediction (Jorgensen &
# Bhowmik, npj Comput Mater 2022, https://github.com/peterbjorgensen/DeepDFT). This file
# inlines the two upstream source files verbatim -- `layer.py` (SchNet-style message-sum
# interaction blocks + PaiNN scalar/vector interaction blocks + Gaussian/sinc edge
# expansions) and `densitymodel.py` (the `DensityModel` entry point that combines an
# `AtomRepresentationModel` with a `ProbeMessageModel` to predict electron density at
# query "probe" points around a molecule/crystal). Only change: relative import
# `import layer` / `from layer import ShiftedSoftplus` collapsed to one module, and the
# one `ase.data.atomic_numbers` lookup (used only for `len(...)` to size the atom-type
# embedding table) is replaced with its literal length (118, the standard periodic-table
# element count including the dummy index 0) so the file has no `ase` import -- no
# architectural change.
import itertools
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"

# len(ase.data.atomic_numbers) in the real repo -- ASE's table has indices 0..118
# (0 = dummy "X" placeholder, 1..118 = real elements). Inlined as a literal so this
# file has no `ase` import; identical embedding-table size to the real model.
_NUM_ATOMIC_NUMBERS = 119


# --- layer.py (verbatim helpers + interaction blocks) ---
def pad_and_stack(tensors: List[torch.Tensor]):
    """Pad list of tensors if tensors are arrays and stack if they are scalars"""
    if tensors[0].shape:
        return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0)
    return torch.stack(tensors)


def shifted_softplus(x):
    r"""
    Compute shifted soft-plus activation function.
    .. math::
       y = \ln\left(1 + e^{-x}\right) - \ln(2)
    """
    return nn.functional.softplus(x) - np.log(2.0)


class ShiftedSoftplus(nn.Module):
    def forward(self, x):
        return shifted_softplus(x)


def unpad_and_cat(stacked_seq: torch.Tensor, seq_len: torch.Tensor):
    """Unpad and concatenate by removing batch dimension"""
    unstacked = stacked_seq.unbind(0)
    unpadded = [torch.narrow(t, 0, 0, l) for (t, l) in zip(unstacked, seq_len.unbind(0))]
    return torch.cat(unpadded, dim=0)


def sum_splits(values: torch.Tensor, splits: torch.Tensor):
    """Sum across dimension 0 of the tensor `values` in chunks defined in `splits`"""
    ind = torch.zeros(splits.sum(), dtype=splits.dtype, device=splits.device)
    ind[torch.cumsum(splits, dim=0)[:-1]] = 1
    ind = torch.cumsum(ind, dim=0)
    sum_y = torch.zeros(splits.shape + values.shape[1:], dtype=values.dtype, device=values.device)
    sum_y.index_add_(0, ind, values)
    return sum_y


def calc_distance(
    positions: torch.Tensor,
    cells: torch.Tensor,
    edges: torch.Tensor,
    edges_displacement: torch.Tensor,
    splits: torch.Tensor,
    return_diff=False,
):
    """Calculate distance of edges"""
    unitcell_repeat = torch.repeat_interleave(cells, splits, dim=0)  # num_edges, 3, 3
    displacement = torch.matmul(
        torch.unsqueeze(edges_displacement, 1), unitcell_repeat
    )  # num_edges, 1, 3
    displacement = torch.squeeze(displacement, dim=1)
    neigh_pos = positions[edges[:, 0]]  # num_edges, 3
    neigh_abs_pos = neigh_pos + displacement  # num_edges, 3
    this_pos = positions[edges[:, 1]]  # num_edges, 3
    diff = this_pos - neigh_abs_pos  # num_edges, 3
    dist = torch.sqrt(torch.sum(torch.square(diff), dim=1, keepdim=True))  # num_edges, 1

    if return_diff:
        return dist, diff
    return dist


def calc_distance_to_probe(
    positions: torch.Tensor,
    positions_probe: torch.Tensor,
    cells: torch.Tensor,
    edges: torch.Tensor,
    edges_displacement: torch.Tensor,
    splits: torch.Tensor,
    return_diff=False,
):
    """Calculate distance of edges"""
    unitcell_repeat = torch.repeat_interleave(cells, splits, dim=0)  # num_edges, 3, 3
    displacement = torch.matmul(
        torch.unsqueeze(edges_displacement, 1), unitcell_repeat
    )  # num_edges, 1, 3
    displacement = torch.squeeze(displacement, dim=1)
    neigh_pos = positions[edges[:, 0]]  # num_edges, 3
    neigh_abs_pos = neigh_pos + displacement  # num_edges, 3
    this_pos = positions_probe[edges[:, 1]]  # num_edges, 3
    diff = this_pos - neigh_abs_pos  # num_edges, 3
    dist = torch.sqrt(torch.sum(torch.square(diff), dim=1, keepdim=True))  # num_edges, 1
    if return_diff:
        return dist, diff
    return dist


def gaussian_expansion(input_x: torch.Tensor, expand_params: List[Tuple]):
    """Expand each feature in a number of Gaussian basis functions."""
    feat_list = torch.unbind(input_x, dim=1)
    expanded_list = []
    for step_tuple, feat in itertools.zip_longest(expand_params, feat_list):
        assert feat is not None, "Too many expansion parameters given"
        if step_tuple:
            start, step, stop = step_tuple
            feat_expanded = torch.unsqueeze(feat, dim=1)
            sigma = step
            basis_mu = torch.arange(start, stop, step, device=input_x.device, dtype=input_x.dtype)
            expanded_list.append(torch.exp(-((feat_expanded - basis_mu) ** 2) / (2.0 * sigma**2)))
        else:
            expanded_list.append(torch.unsqueeze(feat, 1))
    return torch.cat(expanded_list, dim=1)


class SchnetMessageFunction(nn.Module):
    def __init__(self, input_size, edge_size, output_size, hard_cutoff):
        super().__init__()
        self.msg_function_edge = nn.Sequential(
            nn.Linear(edge_size, output_size),
            ShiftedSoftplus(),
            nn.Linear(output_size, output_size),
        )
        self.msg_function_node = nn.Sequential(
            nn.Linear(input_size, input_size),
            ShiftedSoftplus(),
            nn.Linear(input_size, output_size),
        )

        self.soft_cutoff_func = lambda x: 1.0 - torch.sigmoid(5 * (x - (hard_cutoff - 1.5)))

    def forward(self, node_state, edge_state, edge_distance):
        gates = self.msg_function_edge(edge_state) * self.soft_cutoff_func(edge_distance)
        nodes = self.msg_function_node(node_state)
        return nodes * gates


class Interaction(nn.Module):
    def __init__(self, node_size, edge_size, cutoff, include_receiver=False):
        super().__init__()

        self.message_sum_module = MessageSum(node_size, edge_size, cutoff, include_receiver)

        self.state_transition_function = nn.Sequential(
            nn.Linear(node_size, node_size),
            ShiftedSoftplus(),
            nn.Linear(node_size, node_size),
        )

    def forward(self, node_state, edges, edge_state, edges_distance):
        message_sum = self.message_sum_module(node_state, edges, edge_state, edges_distance)
        new_state = node_state + self.state_transition_function(message_sum)
        return new_state


class MessageSum(nn.Module):
    def __init__(self, node_size, edge_size, cutoff, include_receiver):
        super().__init__()

        self.include_receiver = include_receiver

        if include_receiver:
            input_size = node_size * 2
        else:
            input_size = node_size

        self.message_function = SchnetMessageFunction(input_size, edge_size, node_size, cutoff)

    def forward(self, node_state, edges, edge_state, edges_distance, receiver_nodes=None):
        # Compute all messages
        if self.include_receiver:
            if receiver_nodes is not None:
                senders = node_state[edges[:, 0]]
                receivers = receiver_nodes[edges[:, 1]]
                nodes = torch.cat((senders, receivers), dim=1)
            else:
                num_edges = edges.shape[0]
                nodes = torch.reshape(node_state[edges], (num_edges, -1))
        else:
            nodes = node_state[edges[:, 0]]  # Only include sender in messages
        messages = self.message_function(nodes, edge_state, edges_distance)

        # Sum messages
        if receiver_nodes is not None:
            message_sum = torch.zeros_like(receiver_nodes)
        else:
            message_sum = torch.zeros_like(node_state)
        message_sum.index_add_(0, edges[:, 1], messages)

        return message_sum


class PaiNNUpdate(nn.Module):
    """PaiNN style update network. Models the interaction between scalar and vectorial part"""

    def __init__(self, node_size):
        super().__init__()

        self.linearU = nn.Linear(node_size, node_size, bias=False)
        self.linearV = nn.Linear(node_size, node_size, bias=False)
        self.combined_mlp = nn.Sequential(
            nn.Linear(2 * node_size, node_size),
            nn.SiLU(),
            nn.Linear(node_size, 3 * node_size),
        )

    def forward(self, node_state_scalar, node_state_vector):
        Uv = self.linearU(node_state_vector)  # num_nodes, 3, node_size
        Vv = self.linearV(node_state_vector)  # num_nodes, 3, node_size

        Vv_norm = torch.linalg.norm(Vv, dim=1, keepdim=False)  # num_nodes, node_size

        mlp_input = torch.cat((node_state_scalar, Vv_norm), dim=1)  # num_nodes, node_size*2
        mlp_output = self.combined_mlp(mlp_input)

        a_ss, a_sv, a_vv = torch.split(mlp_output, node_state_scalar.shape[1], dim=1)

        inner_prod = torch.sum(Uv * Vv, dim=1)  # num_nodes, node_size

        delta_v = torch.unsqueeze(a_vv, 1) * Uv  # num_nodes, 3, node_size
        delta_s = a_ss + a_sv * inner_prod  # num_nodes, node_size

        return node_state_scalar + delta_s, node_state_vector + delta_v


class PaiNNInteraction(nn.Module):
    """Interaction network"""

    def __init__(self, node_size, edge_size, cutoff):
        super().__init__()

        self.filter_layer = nn.Linear(edge_size, 3 * node_size)
        self.cutoff = cutoff
        self.scalar_message_mlp = nn.Sequential(
            nn.Linear(node_size, node_size),
            nn.SiLU(),
            nn.Linear(node_size, 3 * node_size),
        )

    def forward(
        self, node_state_scalar, node_state_vector, edge_state, edge_vector, edge_distance, edges
    ):
        edge_vector_normalised = edge_vector / torch.maximum(
            torch.linalg.norm(edge_vector, dim=1, keepdim=True), torch.tensor(1e-12)
        )  # num_edges, 3

        filter_weight = self.filter_layer(edge_state)  # num_edges, 3*node_size
        filter_weight = filter_weight * cosine_cutoff(edge_distance, self.cutoff)

        scalar_output = self.scalar_message_mlp(node_state_scalar)  # num_nodes, 3*node_size
        scalar_output = scalar_output[edges[:, 0]]  # num_edges, 3*node_size
        filter_output = filter_weight * scalar_output  # num_edges, 3*node_size

        gate_state_vector, gate_edge_vector, gate_node_state = torch.split(
            filter_output, node_state_scalar.shape[1], dim=1
        )

        gate_state_vector = torch.unsqueeze(gate_state_vector, 1)  # num_edges, 1, node_size
        gate_edge_vector = torch.unsqueeze(gate_edge_vector, 1)  # num_edges, 1, node_size

        messages_scalar = node_state_scalar[edges[:, 0]] * gate_node_state
        messages_state_vector = node_state_vector[
            edges[:, 0]
        ] * gate_state_vector + gate_edge_vector * torch.unsqueeze(edge_vector_normalised, 2)

        message_sum_scalar = torch.zeros_like(node_state_scalar)
        message_sum_scalar.index_add_(0, edges[:, 1], messages_scalar)
        message_sum_vector = torch.zeros_like(node_state_vector)
        message_sum_vector.index_add_(0, edges[:, 1], messages_state_vector)

        new_state_scalar = node_state_scalar + message_sum_scalar
        new_state_vector = node_state_vector + message_sum_vector

        return new_state_scalar, new_state_vector


class PaiNNInteractionOneWay(nn.Module):
    """Same as Interaction network, but the receiving nodes are differently indexed from the sending nodes"""

    def __init__(self, node_size, edge_size, cutoff):
        super().__init__()

        self.filter_layer = nn.Linear(edge_size, 3 * node_size)
        self.cutoff = cutoff
        self.scalar_message_mlp = nn.Sequential(
            nn.Linear(node_size, node_size),
            nn.SiLU(),
            nn.Linear(node_size, 3 * node_size),
        )

        # Ignore messages gate (not part of original PaiNN network)
        self.update_gate_mlp = nn.Sequential(
            nn.Linear(node_size, 2 * node_size),
            nn.SiLU(),
            nn.Linear(2 * node_size, 2 * node_size),
            nn.Sigmoid(),
        )

    def forward(
        self,
        sender_node_state_scalar,
        sender_node_state_vector,
        receiver_node_state_scalar,
        receiver_node_state_vector,
        edge_state,
        edge_vector,
        edge_distance,
        edges,
    ):
        edge_vector_normalised = edge_vector / torch.maximum(
            torch.linalg.norm(edge_vector, dim=1, keepdim=True), torch.tensor(1e-12)
        )  # num_edges, 3

        filter_weight = self.filter_layer(edge_state)  # num_edges, 3*node_size
        filter_weight = filter_weight * cosine_cutoff(edge_distance, self.cutoff)

        scalar_output = self.scalar_message_mlp(sender_node_state_scalar)  # num_nodes, 3*node_size
        scalar_output = scalar_output[edges[:, 0]]  # num_edges, 3*node_size
        filter_output = filter_weight * scalar_output  # num_edges, 3*node_size

        gate_state_vector, gate_edge_vector, gate_node_state = torch.split(
            filter_output, sender_node_state_scalar.shape[1], dim=1
        )

        gate_state_vector = torch.unsqueeze(gate_state_vector, 1)  # num_edges, 1, node_size
        gate_edge_vector = torch.unsqueeze(gate_edge_vector, 1)  # num_edges, 1, node_size

        messages_scalar = sender_node_state_scalar[edges[:, 0]] * gate_node_state
        messages_state_vector = sender_node_state_vector[
            edges[:, 0]
        ] * gate_state_vector + gate_edge_vector * torch.unsqueeze(edge_vector_normalised, 2)

        message_sum_scalar = torch.zeros_like(receiver_node_state_scalar)
        message_sum_scalar.index_add_(0, edges[:, 1], messages_scalar)
        message_sum_vector = torch.zeros_like(receiver_node_state_vector)
        message_sum_vector.index_add_(0, edges[:, 1], messages_state_vector)

        update_gate_scalar, update_gate_vector = torch.split(
            self.update_gate_mlp(message_sum_scalar), receiver_node_state_scalar.shape[1], dim=1
        )
        update_gate_vector = torch.unsqueeze(update_gate_vector, 1)  # num_nodes, 1, node_size
        new_state_scalar = (
            update_gate_scalar * receiver_node_state_scalar
            + (1.0 - update_gate_scalar) * message_sum_scalar
        )
        new_state_vector = (
            update_gate_vector * receiver_node_state_vector
            + (1.0 - update_gate_vector) * message_sum_vector
        )

        return new_state_scalar, new_state_vector


def sinc_expansion(input_x: torch.Tensor, expand_params: List[Tuple]):
    """Expand each feature in a sinc-like basis function expansion (arXiv:2003.03123)."""
    feat_list = torch.unbind(input_x, dim=1)
    expanded_list = []
    for step_tuple, feat in itertools.zip_longest(expand_params, feat_list):
        assert feat is not None, "Too many expansion parameters given"
        if step_tuple:
            n, cutoff = step_tuple
            feat_expanded = torch.unsqueeze(feat, dim=1)
            n_range = torch.arange(n, device=input_x.device, dtype=input_x.dtype) + 1
            out = torch.sinc(n_range / cutoff * feat_expanded) * np.pi * n_range / cutoff
            expanded_list.append(out)
        else:
            expanded_list.append(torch.unsqueeze(feat, 1))
    return torch.cat(expanded_list, dim=1)


def cosine_cutoff(distance: torch.Tensor, cutoff: float):
    """f(d) = 0.5*(cos(pi*d/d_cut)+1) for d < d_cut and 0 otherwise (Behler-Parinello cutoff)."""
    return torch.where(
        distance < cutoff,
        0.5 * (torch.cos(np.pi * distance / cutoff) + 1),
        torch.tensor(0.0, device=distance.device, dtype=distance.dtype),
    )


# --- densitymodel.py (verbatim entry point; using the PaiNN variant, PainnDensityModel) ---
class DensityModel(nn.Module):
    def __init__(
        self, num_interactions, hidden_state_size, cutoff, gaussian_expansion_step=0.1, **kwargs
    ):
        super().__init__(**kwargs)

        self.atom_model = AtomRepresentationModel(
            num_interactions, hidden_state_size, cutoff, gaussian_expansion_step
        )
        self.probe_model = ProbeMessageModel(
            num_interactions, hidden_state_size, cutoff, gaussian_expansion_step
        )

    def forward(self, input_dict):
        atom_representation = self.atom_model(input_dict)
        probe_result = self.probe_model(input_dict, atom_representation)
        return probe_result


class PainnDensityModel(nn.Module):
    def __init__(
        self, num_interactions, hidden_state_size, cutoff, distance_embedding_size=30, **kwargs
    ):
        super().__init__(**kwargs)

        self.atom_model = PainnAtomRepresentationModel(
            num_interactions, hidden_state_size, cutoff, distance_embedding_size
        )
        self.probe_model = PainnProbeMessageModel(
            num_interactions, hidden_state_size, cutoff, distance_embedding_size
        )

    def forward(self, input_dict):
        atom_representation_scalar, atom_representation_vector = self.atom_model(input_dict)
        probe_result = self.probe_model(
            input_dict, atom_representation_scalar, atom_representation_vector
        )
        return probe_result


class ProbeMessageModel(nn.Module):
    def __init__(
        self, num_interactions, hidden_state_size, cutoff, gaussian_expansion_step, **kwargs
    ):
        super().__init__(**kwargs)

        self.num_interactions = num_interactions
        self.hidden_state_size = hidden_state_size
        self.cutoff = cutoff
        self.gaussian_expansion_step = gaussian_expansion_step

        edge_size = int(np.ceil(self.cutoff / self.gaussian_expansion_step))

        self.messagesum_layers = nn.ModuleList(
            [
                layer_MessageSum(hidden_state_size, edge_size, self.cutoff, include_receiver=True)
                for _ in range(num_interactions)
            ]
        )

        self.probe_state_gate_functions = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_state_size, hidden_state_size),
                    ShiftedSoftplus(),
                    nn.Linear(hidden_state_size, hidden_state_size),
                    nn.Sigmoid(),
                )
                for _ in range(num_interactions)
            ]
        )
        self.probe_state_transition_functions = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_state_size, hidden_state_size),
                    ShiftedSoftplus(),
                    nn.Linear(hidden_state_size, hidden_state_size),
                )
                for _ in range(num_interactions)
            ]
        )

        self.readout_function = nn.Sequential(
            nn.Linear(hidden_state_size, hidden_state_size),
            ShiftedSoftplus(),
            nn.Linear(hidden_state_size, 1),
        )

    def forward(
        self,
        input_dict: Dict[str, torch.Tensor],
        atom_representation: List[torch.Tensor],
        compute_iri=False,
        compute_dori=False,
        compute_hessian=False,
    ):
        if compute_iri or compute_dori or compute_hessian:
            input_dict["probe_xyz"].requires_grad_()

        atom_xyz = unpad_and_cat(input_dict["atom_xyz"], input_dict["num_nodes"])
        probe_xyz = unpad_and_cat(input_dict["probe_xyz"], input_dict["num_probes"])
        edge_offset = torch.cumsum(
            torch.cat(
                (
                    torch.tensor([0], device=input_dict["num_nodes"].device),
                    input_dict["num_nodes"][:-1],
                )
            ),
            dim=0,
        )
        edge_offset = edge_offset[:, None, None]

        probe_edges_displacement = unpad_and_cat(
            input_dict["probe_edges_displacement"], input_dict["num_probe_edges"]
        )
        edge_probe_offset = torch.cumsum(
            torch.cat(
                (
                    torch.tensor([0], device=input_dict["num_probes"].device),
                    input_dict["num_probes"][:-1],
                )
            ),
            dim=0,
        )
        edge_probe_offset = edge_probe_offset[:, None, None]
        edge_probe_offset = torch.cat((edge_offset, edge_probe_offset), dim=2)
        probe_edges = input_dict["probe_edges"] + edge_probe_offset
        probe_edges = unpad_and_cat(probe_edges, input_dict["num_probe_edges"])

        probe_edges_features = calc_distance_to_probe(
            atom_xyz,
            probe_xyz,
            input_dict["cell"],
            probe_edges,
            probe_edges_displacement,
            input_dict["num_probe_edges"],
        )

        probe_edge_state = gaussian_expansion(
            probe_edges_features, [(0.0, self.gaussian_expansion_step, self.cutoff)]
        )

        probe_state = torch.zeros(
            (torch.sum(input_dict["num_probes"]), self.hidden_state_size),
            device=atom_representation[0].device,
        )
        for msg_layer, gate_layer, state_layer, nodes in zip(
            self.messagesum_layers,
            self.probe_state_gate_functions,
            self.probe_state_transition_functions,
            atom_representation,
        ):
            msgsum = msg_layer(
                nodes, probe_edges, probe_edge_state, probe_edges_features, probe_state
            )
            gates = gate_layer(probe_state)
            probe_state = probe_state * gates + (1 - gates) * state_layer(msgsum)

        probe_output = self.readout_function(probe_state).squeeze(1)
        probe_output = pad_and_stack(
            torch.split(probe_output, list(input_dict["num_probes"].detach().cpu().numpy()), dim=0)
        )

        if compute_iri or compute_dori or compute_hessian:
            dp_dxyz = torch.autograd.grad(
                probe_output,
                input_dict["probe_xyz"],
                grad_outputs=torch.ones_like(probe_output),
                retain_graph=True,
                create_graph=True,
            )[0]

        grad_probe_outputs = {}

        if compute_iri:
            iri = torch.linalg.norm(dp_dxyz, dim=2) / (torch.pow(probe_output, 1.1))
            grad_probe_outputs["iri"] = iri

        if compute_dori:
            norm_grad_2 = torch.linalg.norm(dp_dxyz / torch.unsqueeze(probe_output, 2), dim=2) ** 2
            grad_norm_grad_2 = torch.autograd.grad(
                norm_grad_2,
                input_dict["probe_xyz"],
                grad_outputs=torch.ones_like(norm_grad_2),
                only_inputs=True,
                retain_graph=True,
                create_graph=True,
            )[0].detach()
            phi_r = torch.linalg.norm(grad_norm_grad_2, dim=2) ** 2 / (norm_grad_2**3)
            dori = phi_r / (1 + phi_r)
            grad_probe_outputs["dori"] = dori

        if compute_hessian:
            hessian_shape = (
                input_dict["probe_xyz"].shape[0],
                input_dict["probe_xyz"].shape[1],
                3,
                3,
            )
            hessian = torch.zeros(hessian_shape, device=probe_xyz.device, dtype=probe_xyz.dtype)
            for dim_idx, grad_out in enumerate(torch.unbind(dp_dxyz, dim=-1)):
                dp2_dxyz2 = torch.autograd.grad(
                    grad_out,
                    input_dict["probe_xyz"],
                    grad_outputs=torch.ones_like(grad_out),
                    only_inputs=True,
                    retain_graph=True,
                    create_graph=True,
                )[0]
                hessian[:, :, dim_idx] = dp2_dxyz2
            grad_probe_outputs["hessian"] = hessian

        if grad_probe_outputs:
            return probe_output, grad_probe_outputs
        return probe_output


# NOTE: local aliases so ProbeMessageModel above (which upstream refers to via
# `layer.MessageSum`) resolves to the same MessageSum class defined earlier in this file.
layer_MessageSum = MessageSum


class AtomRepresentationModel(nn.Module):
    def __init__(
        self, num_interactions, hidden_state_size, cutoff, gaussian_expansion_step, **kwargs
    ):
        super().__init__(**kwargs)

        self.num_interactions = num_interactions
        self.hidden_state_size = hidden_state_size
        self.cutoff = cutoff
        self.gaussian_expansion_step = gaussian_expansion_step

        edge_size = int(np.ceil(self.cutoff / self.gaussian_expansion_step))

        self.interactions = nn.ModuleList(
            [
                Interaction(hidden_state_size, edge_size, self.cutoff, include_receiver=True)
                for _ in range(num_interactions)
            ]
        )

        self.atom_embeddings = nn.Embedding(_NUM_ATOMIC_NUMBERS, self.hidden_state_size)

    def forward(self, input_dict):
        edges_displacement = unpad_and_cat(
            input_dict["atom_edges_displacement"], input_dict["num_atom_edges"]
        )
        edge_offset = torch.cumsum(
            torch.cat(
                (
                    torch.tensor([0], device=input_dict["num_nodes"].device),
                    input_dict["num_nodes"][:-1],
                )
            ),
            dim=0,
        )
        edge_offset = edge_offset[:, None, None]
        edges = input_dict["atom_edges"] + edge_offset
        edges = unpad_and_cat(edges, input_dict["num_atom_edges"])

        atom_xyz = unpad_and_cat(input_dict["atom_xyz"], input_dict["num_nodes"])
        nodes = unpad_and_cat(input_dict["nodes"], input_dict["num_nodes"])
        nodes = self.atom_embeddings(nodes)

        edges_features = calc_distance(
            atom_xyz, input_dict["cell"], edges, edges_displacement, input_dict["num_atom_edges"]
        )

        edge_state = gaussian_expansion(
            edges_features, [(0.0, self.gaussian_expansion_step, self.cutoff)]
        )

        nodes_list = []
        for int_layer in self.interactions:
            nodes = int_layer(nodes, edges, edge_state, edges_features)
            nodes_list.append(nodes)

        return nodes_list


class PainnAtomRepresentationModel(nn.Module):
    def __init__(
        self, num_interactions, hidden_state_size, cutoff, distance_embedding_size, **kwargs
    ):
        super().__init__(**kwargs)

        self.num_interactions = num_interactions
        self.hidden_state_size = hidden_state_size
        self.cutoff = cutoff
        self.distance_embedding_size = distance_embedding_size

        self.interactions = nn.ModuleList(
            [
                PaiNNInteraction(hidden_state_size, self.distance_embedding_size, self.cutoff)
                for _ in range(num_interactions)
            ]
        )
        self.scalar_vector_update = nn.ModuleList(
            [PaiNNUpdate(hidden_state_size) for _ in range(num_interactions)]
        )

        self.atom_embeddings = nn.Embedding(_NUM_ATOMIC_NUMBERS, self.hidden_state_size)

    def forward(self, input_dict):
        edges_displacement = unpad_and_cat(
            input_dict["atom_edges_displacement"], input_dict["num_atom_edges"]
        )
        edge_offset = torch.cumsum(
            torch.cat(
                (
                    torch.tensor([0], device=input_dict["num_nodes"].device),
                    input_dict["num_nodes"][:-1],
                )
            ),
            dim=0,
        )
        edge_offset = edge_offset[:, None, None]
        edges = input_dict["atom_edges"] + edge_offset
        edges = unpad_and_cat(edges, input_dict["num_atom_edges"])

        atom_xyz = unpad_and_cat(input_dict["atom_xyz"], input_dict["num_nodes"])
        nodes_scalar = unpad_and_cat(input_dict["nodes"], input_dict["num_nodes"])
        nodes_scalar = self.atom_embeddings(nodes_scalar)
        nodes_vector = torch.zeros(
            (nodes_scalar.shape[0], 3, self.hidden_state_size),
            dtype=nodes_scalar.dtype,
            device=nodes_scalar.device,
        )

        edges_distance, edges_diff = calc_distance(
            atom_xyz,
            input_dict["cell"],
            edges,
            edges_displacement,
            input_dict["num_atom_edges"],
            return_diff=True,
        )

        edge_state = sinc_expansion(edges_distance, [(self.distance_embedding_size, self.cutoff)])

        nodes_list_scalar = []
        nodes_list_vector = []
        for int_layer, update_layer in zip(self.interactions, self.scalar_vector_update):
            nodes_scalar, nodes_vector = int_layer(
                nodes_scalar, nodes_vector, edge_state, edges_diff, edges_distance, edges
            )
            nodes_scalar, nodes_vector = update_layer(nodes_scalar, nodes_vector)
            nodes_list_scalar.append(nodes_scalar)
            nodes_list_vector.append(nodes_vector)

        return nodes_list_scalar, nodes_list_vector


class PainnProbeMessageModel(nn.Module):
    def __init__(
        self, num_interactions, hidden_state_size, cutoff, distance_embedding_size, **kwargs
    ):
        super().__init__(**kwargs)

        self.num_interactions = num_interactions
        self.hidden_state_size = hidden_state_size
        self.cutoff = cutoff
        self.distance_embedding_size = distance_embedding_size

        self.message_layers = nn.ModuleList(
            [
                PaiNNInteractionOneWay(hidden_state_size, self.distance_embedding_size, self.cutoff)
                for _ in range(num_interactions)
            ]
        )
        self.scalar_vector_update = nn.ModuleList(
            [PaiNNUpdate(hidden_state_size) for _ in range(num_interactions)]
        )

        self.readout_function = nn.Sequential(
            nn.Linear(hidden_state_size, hidden_state_size),
            nn.SiLU(),
            nn.Linear(hidden_state_size, 1),
        )

    def forward(
        self,
        input_dict: Dict[str, torch.Tensor],
        atom_representation_scalar: List[torch.Tensor],
        atom_representation_vector: List[torch.Tensor],
        compute_iri=False,
        compute_dori=False,
        compute_hessian=False,
    ):
        if compute_iri or compute_dori or compute_hessian:
            input_dict["probe_xyz"].requires_grad_()

        atom_xyz = unpad_and_cat(input_dict["atom_xyz"], input_dict["num_nodes"])
        probe_xyz = unpad_and_cat(input_dict["probe_xyz"], input_dict["num_probes"])
        edge_offset = torch.cumsum(
            torch.cat(
                (
                    torch.tensor([0], device=input_dict["num_nodes"].device),
                    input_dict["num_nodes"][:-1],
                )
            ),
            dim=0,
        )
        edge_offset = edge_offset[:, None, None]

        probe_edges_displacement = unpad_and_cat(
            input_dict["probe_edges_displacement"], input_dict["num_probe_edges"]
        )
        edge_probe_offset = torch.cumsum(
            torch.cat(
                (
                    torch.tensor([0], device=input_dict["num_probes"].device),
                    input_dict["num_probes"][:-1],
                )
            ),
            dim=0,
        )
        edge_probe_offset = edge_probe_offset[:, None, None]
        edge_probe_offset = torch.cat((edge_offset, edge_probe_offset), dim=2)
        probe_edges = input_dict["probe_edges"] + edge_probe_offset
        probe_edges = unpad_and_cat(probe_edges, input_dict["num_probe_edges"])

        probe_edges_distance, probe_edges_diff = calc_distance_to_probe(
            atom_xyz,
            probe_xyz,
            input_dict["cell"],
            probe_edges,
            probe_edges_displacement,
            input_dict["num_probe_edges"],
            return_diff=True,
        )

        edge_state = sinc_expansion(
            probe_edges_distance, [(self.distance_embedding_size, self.cutoff)]
        )

        probe_state_scalar = torch.zeros(
            (torch.sum(input_dict["num_probes"]), self.hidden_state_size),
            device=atom_representation_scalar[0].device,
        )
        probe_state_vector = torch.zeros(
            (torch.sum(input_dict["num_probes"]), 3, self.hidden_state_size),
            device=atom_representation_scalar[0].device,
        )

        for msg_layer, update_layer, atom_nodes_scalar, atom_nodes_vector in zip(
            self.message_layers,
            self.scalar_vector_update,
            atom_representation_scalar,
            atom_representation_vector,
        ):
            probe_state_scalar, probe_state_vector = msg_layer(
                atom_nodes_scalar,
                atom_nodes_vector,
                probe_state_scalar,
                probe_state_vector,
                edge_state,
                probe_edges_diff,
                probe_edges_distance,
                probe_edges,
            )
            probe_state_scalar, probe_state_vector = update_layer(
                probe_state_scalar, probe_state_vector
            )

        probe_output = self.readout_function(probe_state_scalar).squeeze(1)
        probe_output = pad_and_stack(
            torch.split(probe_output, list(input_dict["num_probes"].detach().cpu().numpy()), dim=0)
        )

        if compute_iri or compute_dori or compute_hessian:
            dp_dxyz = torch.autograd.grad(
                probe_output,
                input_dict["probe_xyz"],
                grad_outputs=torch.ones_like(probe_output),
                retain_graph=True,
                create_graph=True,
            )[0]

        grad_probe_outputs = {}

        if compute_iri:
            iri = torch.linalg.norm(dp_dxyz, dim=2) / (torch.pow(probe_output, 1.1))
            grad_probe_outputs["iri"] = iri

        if compute_dori:
            norm_grad_2 = (
                torch.linalg.norm(dp_dxyz / (torch.unsqueeze(probe_output, 2)), dim=2) ** 2
            )
            grad_norm_grad_2 = torch.autograd.grad(
                norm_grad_2,
                input_dict["probe_xyz"],
                grad_outputs=torch.ones_like(norm_grad_2),
                only_inputs=True,
                retain_graph=True,
                create_graph=True,
            )[0].detach()
            phi_r = torch.linalg.norm(grad_norm_grad_2, dim=2) ** 2 / (norm_grad_2**3)
            dori = phi_r / (1 + phi_r)
            grad_probe_outputs["dori"] = dori

        if compute_hessian:
            hessian_shape = (
                input_dict["probe_xyz"].shape[0],
                input_dict["probe_xyz"].shape[1],
                3,
                3,
            )
            hessian = torch.zeros(hessian_shape, device=probe_xyz.device, dtype=probe_xyz.dtype)
            for dim_idx, grad_out in enumerate(torch.unbind(dp_dxyz, dim=-1)):
                dp2_dxyz2 = torch.autograd.grad(
                    grad_out,
                    input_dict["probe_xyz"],
                    grad_outputs=torch.ones_like(grad_out),
                    only_inputs=True,
                    retain_graph=True,
                    create_graph=True,
                )[0]
                hessian[:, :, dim_idx] = dp2_dxyz2
            grad_probe_outputs["hessian"] = hessian

        if grad_probe_outputs:
            return probe_output, grad_probe_outputs
        return probe_output


def build_deepdft():
    # Tiny menagerie-scale PainnDensityModel config (PaiNN variant is the model actually
    # shipped/recommended in the repo's example configs): 2 interaction layers,
    # hidden_state_size=16, cutoff kept physically meaningful (Angstrom-scale).
    return PainnDensityModel(
        num_interactions=2, hidden_state_size=16, cutoff=4.0, distance_embedding_size=8
    )


def example_input_deepdft():
    # Hand-built batch-1 `input_dict`, matching the exact tensor keys/shapes produced by
    # the real `atoms_and_probe_sample_to_graph_dict` + `collate_list_of_dicts` pipeline
    # in dataset.py (padded/stacked with a leading batch dim of 1, since batch size is 1
    # here no padding is actually needed). A small synthetic 4-atom "molecule" with 6
    # probe query points and a fully-connected edge set within the cutoff.
    torch.manual_seed(0)
    num_atoms = 4
    num_probes = 6

    atom_xyz = torch.rand(num_atoms, 3) * 2.0
    cell = torch.eye(3) * 10.0
    nodes = torch.randint(1, 10, (num_atoms,))

    # Fully connect all atom pairs (i != j) as "edges" with zero periodic displacement --
    # a faithful stand-in for a real neighbor list within a generous cutoff.
    atom_edges = torch.tensor(
        [[i, j] for i in range(num_atoms) for j in range(num_atoms) if i != j], dtype=torch.long
    )
    atom_edges_displacement = torch.zeros(atom_edges.shape[0], 3)

    probe_xyz = torch.rand(num_probes, 3) * 2.0
    probe_edges = torch.tensor(
        [[i, j] for i in range(num_atoms) for j in range(num_probes)], dtype=torch.long
    )
    probe_edges_displacement = torch.zeros(probe_edges.shape[0], 3)

    input_dict = {
        "nodes": nodes.unsqueeze(0),
        "atom_edges": atom_edges.unsqueeze(0),
        "atom_edges_displacement": atom_edges_displacement.unsqueeze(0),
        "probe_edges": probe_edges.unsqueeze(0),
        "probe_edges_displacement": probe_edges_displacement.unsqueeze(0),
        "num_nodes": torch.tensor([num_atoms]),
        "num_atom_edges": torch.tensor([atom_edges.shape[0]]),
        "num_probe_edges": torch.tensor([probe_edges.shape[0]]),
        "num_probes": torch.tensor([num_probes]),
        "probe_xyz": probe_xyz.unsqueeze(0),
        "atom_xyz": atom_xyz.unsqueeze(0),
        "cell": cell.unsqueeze(0),
    }
    return (input_dict,)


MENAGERIE_ENTRIES = [
    (
        "DeepDFT",
        "build_deepdft",
        "example_input_deepdft",
        2022,
        MENAGERIE_ZOO,
    ),
]
