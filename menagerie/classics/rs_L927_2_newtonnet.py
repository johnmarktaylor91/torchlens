# SOURCE: vendored from THGLab/NewtonNet @ fc1ac987a2cc88efdf58c73ce7db9cc98648afe2
#   (repo: https://github.com/THGLab/NewtonNet)
#   newtonnet/models/newtonnet.py (NewtonNet, EmbeddingNet, InteractionNet) +
#   newtonnet/models/output.py (CustomOutputSet, DerivativeProperty,
#   EnergyOutput, GradientForceOutput, SumAggregator, NullAggregator) +
#   newtonnet/layers/representations.py (get_representation_by_string,
#   ScaledNorm, PolynomialCutoff, RadialBesselLayer) + newtonnet/layers/
#   scalers.py (ScaleShift, get_scaler_by_string) + newtonnet/layers/
#   activations.py (get_activation_by_string).
# NewtonNet (Haghighatlari, Li, Guan, Zhang, Das, Stein, Heidar-Zadeh,
# Liu, Head-Gordon, Bertels, Hao & Head-Gordon, "NewtonNet: A Newtonian
# message passing network for deep learning of interatomic potentials and
# forces", Digital Discovery 2022) message-passes invariant atomic ("a") and
# equivariant force-direction ("f") embeddings through radial-Bessel edge
# features gated by a polynomial cutoff, alternating invariant and
# Newtonian-mechanics-inspired vector updates each interaction layer, then
# reads out per-atom energy contributions summed to a molecular energy
# (`SumAggregator`/`scatter`).
#
# Pinned to the last commit BEFORE `newtonnet/models/output.py` added a hard
# `from les import Les` import (commit `9a79477`, "Les (#22)", 2025-05-18);
# `les` (git+https://github.com/ChengUCB/les`, an unconditional
# `Les()` instantiation inside `EnergyAggregator.__init__` on the current
# `main`) is a non-base long-range-electrostatics package this environment
# does not have and the task rules forbid installing for a vendor. The
# pre-LES commit is the same NewtonNet architecture (message passing +
# energy/force heads; only BEC/charge outputs, added later, are absent) and
# needs only base torch + torch_geometric, both installed -- so it is
# vendored verbatim at that ref rather than reimplemented. Code copied
# verbatim; only import paths were flattened into this single file.

import numpy as np
import torch
from torch import nn
from torch_geometric.utils import scatter

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# Verbatim from newtonnet/layers/activations.py @ fc1ac987
# ---------------------------------------------------------------------------
def get_activation_by_string(key):
    if key == "swish":
        activation = nn.SiLU()
    elif key == "silu":
        activation = nn.SiLU()
    elif key == "relu":
        activation = nn.ReLU()
    elif key == "elu":
        activation = nn.ELU()
    elif key == "leaky_relu":
        activation = nn.LeakyReLU()
    elif key == "tanh":
        activation = nn.Tanh()
    elif key == "sigmoid":
        activation = nn.Sigmoid()
    elif key == "softplus":
        activation = nn.Softplus()
    elif key == "gelu":
        activation = nn.GELU()
    elif key == "ssp":
        activation = ShiftedSoftplus()
    elif key == "swiglu":
        activation = SwiGLU()
    else:
        raise NotImplementedError("The activation function '%s' is unknown." % str(key))
    return activation


class ShiftedSoftplus(nn.Module):
    """y = ln(1 + e^(-x)) - ln(2). Copied from schnetpack (MIT)."""

    def __init__(self):
        super().__init__()
        self.softplus = nn.Softplus()
        self.shift = torch.log(torch.tensor(2.0))

    def forward(self, x):
        return self.softplus(x) - self.shift


class SwiGLU(nn.Module):
    """y = gate(x) * out(x) = swish(linear(x)) * linear(x)"""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(in_features, out_features)
        self.gate = nn.SiLU()

    def forward(self, x):
        return self.gate(self.linear1(x)) * self.linear2(x)


# ---------------------------------------------------------------------------
# Verbatim from newtonnet/layers/representations.py @ fc1ac987
# ---------------------------------------------------------------------------
def get_representation_by_string(
    cutoff, cutoff_network="poly", radial_network="bessel", n_basis=20
):
    representations = {}
    representations["norm"] = ScaledNorm(r=cutoff)

    if cutoff_network == "poly":
        representations["cutoff"] = PolynomialCutoff(p=9)
    elif cutoff_network == "cos":
        representations["cutoff"] = CosineCutoff()
    else:
        raise NotImplementedError(f"The cutoff function {cutoff_network} is unknown.")

    if radial_network == "bessel":
        representations["radial"] = RadialBesselLayer(n_basis=n_basis)
    else:
        raise NotImplementedError(f"The radial function {radial_network} is unknown.")

    return representations


class ScaledNorm(nn.Module):
    """Compute scaled norm of interatomic distances.
    Based on Klicpera, Grob, Gunnemann. Directional Message Passing for
    Molecular Graphs. ICLR 2020."""

    def __init__(self, r, **kwargs):
        super().__init__()
        self.r = r

    def forward(self, disp):
        dist = torch.norm(disp, dim=-1, keepdim=True)
        dir = disp / dist
        dist = dist / self.r
        return dist, dir


class PolynomialCutoff(nn.Module):
    """Based on Klicpera, Grob, Gunnemann. ICLR 2020.
    y = 1 - 0.5(p+1)(p+2)x^p + p(p+2)x^(p+1) - 0.5p(p+1)x^(p+2); y(0)=1, y(1)=0.
    """

    def __init__(self, p, **kwargs):
        super().__init__()
        self.p = p

    def forward(self, dist):
        cutoffs = (
            1
            - 0.5 * (self.p + 1) * (self.p + 2) * dist.pow(self.p)
            + self.p * (self.p + 2) * dist.pow(self.p + 1)
            - 0.5 * self.p * (self.p + 1) * dist.pow(self.p + 2)
        )
        return cutoffs


class CosineCutoff(nn.Module):
    """Behler cosine cutoff function. Copied from schnetpack (MIT).
    y = 0.5(1 + cos(pi*x)); y(0)=1, y(1)=0. (Unused by this staging entry --
    `poly` is selected -- kept for architectural completeness; the real
    repo's version of this class omits `import numpy as np` even though it
    references `np.pi`, so instantiating it there would already NameError.)
    """

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, dist):
        cutoffs = 0.5 * (torch.cos(dist * np.pi) + 1.0)
        return cutoffs


class RadialBesselLayer(nn.Module):
    """Radial Bessel functions, DimeNet (klicperajo/dimenet).
    y = sin(pi*r) / (pi*r)."""

    def __init__(self, n_basis):
        super().__init__()
        self.n_basis = n_basis
        self.frequencies = nn.Parameter(
            torch.arange(1, self.n_basis + 1) * torch.pi, requires_grad=False
        )
        self.epsilon = 1.0e-8

    def forward(self, dist):
        out = torch.sin(self.frequencies * dist) / dist
        return out


# ---------------------------------------------------------------------------
# Verbatim from newtonnet/layers/scalers.py @ fc1ac987
# ---------------------------------------------------------------------------
def get_scaler_by_string(key):
    if key == "energy":
        scaler = ScaleShift(scale=True, shift=True)
    elif key == "gradient_force":
        scaler = ScaleShift(scale=False, shift=False)
    elif key == "direct_force":
        scaler = ScaleShift(scale=True, shift=False)
    elif key == "hessian":
        scaler = ScaleShift(scale=False, shift=False)
    elif key == "virial":
        scaler = ScaleShift(scale=False, shift=False)
    else:
        raise NotImplementedError(f"Scaler type {key} is not implemented yet")
    return scaler


class ScaleShift(nn.Module):
    """Node-level scale and shift layer."""

    def __init__(self, scale=True, shift=True):
        super().__init__()
        self.scale = (
            nn.Embedding.from_pretrained(torch.ones(118 + 1, 1), freeze=False, padding_idx=0)
            if scale
            else None
        )
        self.shift = (
            nn.Embedding.from_pretrained(torch.zeros(118 + 1, 1), freeze=False, padding_idx=0)
            if shift
            else None
        )

    def forward(self, output, outputs):
        if self.scale is not None:
            output = output * self.scale(outputs.z)
        if self.shift is not None:
            output = output + self.shift(outputs.z)
        return output

    def set_scale(self, scale):
        self.scale.weight.data = scale.reshape(-1, 1)

    def set_shift(self, shift):
        self.shift.weight.data = shift.reshape(-1, 1)


# ---------------------------------------------------------------------------
# Verbatim from newtonnet/models/output.py @ fc1ac987
# ---------------------------------------------------------------------------
def get_output_by_string(key, n_features=None, activation=None):
    if key == "energy":
        output_layer = EnergyOutput(n_features, activation)
    elif key == "gradient_force":
        output_layer = GradientForceOutput()
    elif key == "direct_force":
        output_layer = DirectForceOutput(n_features, activation)
    elif key == "hessian":
        output_layer = HessianOutput()
    elif key == "virial":
        output_layer = VirialOutput()
    else:
        raise NotImplementedError(f"Output type {key} is not implemented yet")
    return output_layer


def get_aggregator_by_string(key):
    if key == "energy":
        aggregator = SumAggregator()
    elif key == "gradient_force":
        aggregator = NullAggregator()
    elif key == "direct_force":
        aggregator = NullAggregator()
    elif key == "hessian":
        aggregator = NullAggregator()
    elif key == "virial":
        aggregator = NullAggregator()
    else:
        raise NotImplementedError(f"Aggregate type {key} is not implemented yet")
    return aggregator


class CustomOutputSet:
    def __init__(self, **outputs):
        for key, value in outputs.items():
            setattr(self, key, value)


class DirectProperty(nn.Module):
    def __init__(self):
        super().__init__()


class DerivativeProperty(nn.Module):
    def __init__(self):
        super().__init__()
        self.create_graph = False  # Set by the model with train() or eval()

    def get_pairwise_force(self, outputs):
        if not hasattr(outputs, "pairwise_force"):
            outputs.pairwise_force = -torch.autograd.grad(
                outputs.energy,
                outputs.disp,
                grad_outputs=torch.ones_like(outputs.energy),
                create_graph=self.create_graph,
                retain_graph=self.create_graph,
            )[0]
        return outputs.pairwise_force


class SecondDerivativeProperty(DerivativeProperty):
    def __init__(self):
        super().__init__()


class EnergyOutput(DirectProperty):
    """Energy prediction."""

    def __init__(self, n_features, activation):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, n_features),
            activation,
            nn.Linear(n_features, n_features),
            activation,
            nn.Linear(n_features, 1),
        )

    def forward(self, outputs):
        energy = self.layers(outputs.atom_node)
        return energy


class GradientForceOutput(DerivativeProperty):
    """Gradient force prediction."""

    def __init__(self):
        super().__init__()

    def forward(self, outputs):
        pairwise_force = self.get_pairwise_force(outputs)
        force = scatter(
            pairwise_force,
            outputs.edge_index[0],
            dim=0,
            reduce="sum",
            dim_size=outputs.atom_node.size(0),
        ) - scatter(
            pairwise_force,
            outputs.edge_index[1],
            dim=0,
            reduce="sum",
            dim_size=outputs.atom_node.size(0),
        )
        return force


class DirectForceOutput(DirectProperty):
    """Direct force prediction."""

    def __init__(self, n_features, activation):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, n_features),
            activation,
            nn.Linear(n_features, n_features),
            activation,
            nn.Linear(n_features, n_features),
        )

    def forward(self, outputs):
        force = (
            self.layers(outputs.atom_node).unsqueeze(1) * outputs.force_node
        )  # n_nodes, 3, n_features
        force = force.sum(dim=-1)  # n_nodes, 3
        return force


class HessianOutput(SecondDerivativeProperty):
    """Hessian prediction."""

    def __init__(self):
        super().__init__()

    def forward(self, outputs):
        hessian = torch.vmap(
            lambda vec: torch.autograd.grad(
                -outputs.gradient_force.flatten(),
                outputs.disp,
                grad_outputs=vec,
                create_graph=self.create_graph,
                retain_graph=True,
            )[0],
        )(torch.eye(outputs.gradient_force.numel(), device=outputs.gradient_force.device))
        hessian = hessian.reshape(*outputs.gradient_force.shape, *outputs.disp.shape)
        hessian = scatter(
            hessian, outputs.edge_index[0], dim=2, reduce="sum", dim_size=outputs.atom_node.size(0)
        ) - scatter(
            hessian, outputs.edge_index[1], dim=2, reduce="sum", dim_size=outputs.atom_node.size(0)
        )
        return hessian


class VirialOutput(DerivativeProperty):
    """Virial prediction."""

    def __init__(self):
        super().__init__()

    def forward(self, outputs):
        pairwise_force = self.get_pairwise_force(outputs)
        virial = outputs.disp[:, :, None] * pairwise_force[:, None, :]
        virial = scatter(
            virial, outputs.edge_index[0], dim=0, reduce="sum", dim_size=outputs.atom_node.size(0)
        ) + scatter(
            virial, outputs.edge_index[1], dim=0, reduce="sum", dim_size=outputs.atom_node.size(0)
        )
        virial = scatter(virial, outputs.batch, dim=0, reduce="sum")
        return virial


class SumAggregator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, outputs):
        return scatter(output, outputs.batch, dim=0, reduce="sum").reshape(-1)


class NullAggregator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, outputs):
        return output


# ---------------------------------------------------------------------------
# Verbatim from newtonnet/models/newtonnet.py @ fc1ac987
# ---------------------------------------------------------------------------
class NewtonNet(nn.Module):
    """
    Molecular Newtonian Message Passing

    Parameters:
        n_features (int): Number of features in the latent layer. Default: 128.
        n_interactions (int): Number of message passing layers. Default: 3.
        activation (str): Activation function. Default: 'swish'.
        layer_norm (bool): Whether to use layer normalization. Default: False.
        infer_properties (list): The properties to predict. Default: [].
        representations (dict): The distance transformation functions.
    """

    def __init__(
        self,
        n_features: int = 128,
        n_interactions: int = 3,
        activation: str = "swish",
        layer_norm: bool = False,
        infer_properties: list = [],
        representations: nn.Module = None,
    ) -> None:
        super().__init__()
        activation = get_activation_by_string(activation)

        # embedding layer
        self.embedding_layer = EmbeddingNet(
            n_features=n_features,
            representations=representations,
        )

        # message passing
        self.interaction_layers = nn.ModuleList(
            [
                InteractionNet(
                    n_features=n_features,
                    n_basis=self.embedding_layer.n_basis,
                    activation=activation,
                    layer_norm=layer_norm,
                )
                for _ in range(n_interactions)
            ]
        )

        # final output layer
        self.infer_properties = infer_properties
        self.output_layers = nn.ModuleList()
        self.scalers = nn.ModuleList()
        self.aggregators = nn.ModuleList()
        for key in self.infer_properties:
            output_layer = get_output_by_string(key, n_features, activation)
            self.output_layers.append(output_layer)
            if isinstance(output_layer, DerivativeProperty):
                self.embedding_layer.requires_dr = True
            scaler = get_scaler_by_string(key)
            self.scalers.append(scaler)
            aggregator = get_aggregator_by_string(key)
            self.aggregators.append(aggregator)

    def forward(self, z, disp, edge_index, batch):
        """
        Network forward pass

        Parameters:
            z (torch.Tensor): Atomic numbers, shape (n_nodes,).
            disp (torch.Tensor): Displacement vectors, shape (n_edges, 3).
            edge_index (torch.Tensor): Edge index, shape (2, n_edges).
            batch (torch.Tensor): Batch index of the atoms, shape (n_nodes,).

        Returns:
            outputs (CustomOutputSet): The outputs of the network.
        """
        # initialize node and edge representations
        atom_node, force_node, disp_node, dir_edge, dist_edge = self.embedding_layer(z, disp)

        # compute interaction block and update atomic embeddings
        for interaction_layer in self.interaction_layers:
            atom_node, force_node, disp_node = interaction_layer(
                atom_node, force_node, disp_node, dir_edge, dist_edge, edge_index
            )

        # output net
        outputs = CustomOutputSet(
            z=z,
            disp=disp,
            atom_node=atom_node,
            force_node=force_node,
            edge_index=edge_index,
            batch=batch,
        )
        for key, output_layer, scaler, aggregator in zip(
            self.infer_properties, self.output_layers, self.scalers, self.aggregators
        ):
            output = output_layer(outputs)
            output = scaler(output, outputs)
            output = aggregator(output, outputs)
            setattr(outputs, key, output)

        return outputs

    def train(self, mode=True):
        """Set the network to training mode."""
        super().train(mode)
        for output_layer in self.output_layers:
            if isinstance(output_layer, DerivativeProperty):
                output_layer.create_graph = mode


class EmbeddingNet(nn.Module):
    """
    Embedding layer of the network

    Parameters:
        n_features (int): Number of features in the hidden layer.
        representations (dict): The distance transformation functions.
    """

    def __init__(self, n_features, representations):
        super().__init__()

        # atomic embedding
        self.n_features = n_features
        self.node_embedding = nn.Embedding(118 + 1, n_features, padding_idx=0)

        # edge embedding
        self.norm = representations["norm"]
        self.cutoff = representations["cutoff"]
        self.edge_embedding = representations["radial"]
        self.n_basis = self.edge_embedding.n_basis

        # requires dr
        self.requires_dr = False

    def forward(self, z, disp):
        # initialize node representations
        atom_node = self.node_embedding(z)  # n_nodes, n_features
        force_node = torch.zeros(
            z.size(0), 3, self.n_features, dtype=disp.dtype, device=disp.device
        )  # n_nodes, 3, n_features
        disp_node = torch.zeros(
            z.size(0), 3, self.n_features, dtype=disp.dtype, device=disp.device
        )  # n_nodes, 3, n_features

        # recompute distances and distance vectors
        if self.requires_dr:
            disp.requires_grad = True

        # initialize edge representations
        dist_edge, dir_edge = self.norm(disp)  # n_edges, 1; n_edges, 3
        dist_edge = self.cutoff(dist_edge) * self.edge_embedding(dist_edge)  # n_edges, n_basis

        return atom_node, force_node, disp_node, dir_edge, dist_edge


class InteractionNet(nn.Module):
    """
    Message passing layer of the network

    Parameters:
        n_features (int): Number of features in the hidden layer.
        n_basis (int): Number of radial basis functions.
        activation (nn.Module): Activation function.
        layer_norm (bool): Whether to use layer normalization.
    """

    def __init__(self, n_features, n_basis, activation, layer_norm):
        super().__init__()

        self.n_features = n_features

        # invariant message passing
        self.message_nodepart = nn.Sequential(
            nn.Linear(n_features, n_features),
            activation,
            nn.Linear(n_features, n_features),
        )
        self.message_edgepart = nn.Linear(n_basis, n_features, bias=False)

        self.equiv_message1 = nn.Sequential(
            nn.Linear(n_features, n_features, bias=False),
            activation,
            nn.Linear(n_features, n_features, bias=False),
        )
        self.equiv_message2 = nn.Sequential(
            nn.Linear(n_features, n_features, bias=False),
            activation,
            nn.Linear(n_features, n_features, bias=False),
        )

        self.equiv_update = nn.Linear(n_features, n_features, bias=False)

        # layer norm
        if layer_norm:
            self.layer_norm = nn.LayerNorm(n_features)
        else:
            self.layer_norm = None

    def forward(self, atom_node, force_node, disp_node, dir_edge, dist_edge, edge_index):
        # a
        message_nodepart = self.message_nodepart(atom_node)  # n_nodes, n_features
        message_edgepart = self.message_edgepart(dist_edge)  # n_edges, n_features
        message = (
            message_edgepart * message_nodepart[edge_index[0]] * message_nodepart[edge_index[1]]
        )  # n_edges, n_features

        inv_message1 = message  # n_nodes, n_features
        inv_update1 = scatter(
            inv_message1, edge_index[0], dim=0, dim_size=atom_node.size(0)
        )  # n_nodes, n_features
        atom_node = atom_node + inv_update1  # n_nodes, n_features

        # f
        equiv_message1_invpart = self.equiv_message1(message).unsqueeze(1)  # n_edges, 1, n_features
        equiv_message1_equivpart = dir_edge.unsqueeze(2)  # n_edges, 3, 1
        equiv_message1 = equiv_message1_invpart * equiv_message1_equivpart  # n_edges, 3, n_features

        equiv_message2_invpart = self.equiv_message2(message).unsqueeze(1)  # n_edges, 1, n_features
        equiv_message2_equivpart = force_node[edge_index[1]]  # n_edges, 3, n_features
        equiv_message2 = equiv_message2_invpart * equiv_message2_equivpart  # n_edges, 3, n_features

        force_update = scatter(
            equiv_message1 + equiv_message2, edge_index[0], dim=0, dim_size=force_node.size(0)
        )  # n_nodes, 3, n_features
        force_node = force_node + force_update  # n_nodes, 3, n_features

        # update energy
        inv_update2 = torch.sum(
            force_node * self.equiv_update(force_node), dim=1
        )  # n_nodes, n_features
        atom_node = atom_node + inv_update2

        # layer norm
        if self.layer_norm is not None:
            atom_node = self.layer_norm(atom_node)

        return atom_node, force_node, disp_node


# ---------------------------------------------------------------------------
# Menagerie staging entry points
# ---------------------------------------------------------------------------

_CUTOFF = 5.0
_N_BASIS = 8
_N_FEATURES = 16


def build_newtonnet():
    representations = get_representation_by_string(
        cutoff=_CUTOFF, cutoff_network="poly", radial_network="bessel", n_basis=_N_BASIS
    )
    return NewtonNet(
        n_features=_N_FEATURES,
        n_interactions=2,
        activation="swish",
        layer_norm=False,
        infer_properties=["energy", "gradient_force"],
        representations=representations,
    )


def example_input_newtonnet():
    """A tiny 5-atom toy molecule (atomic numbers for C, H, H, H, H, e.g.
    a rough methane skeleton) with an all-pairs edge list (10 directed
    edges within the 5.0 cutoff) and real interatomic displacement vectors,
    matching the `(z, disp, edge_index, batch)` signature the real
    `NewtonNet.forward` expects."""
    torch.manual_seed(0)
    z = torch.tensor([6, 1, 1, 1, 1], dtype=torch.long)
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.63, 0.63, 0.63],
            [-0.63, -0.63, 0.63],
            [-0.63, 0.63, -0.63],
            [0.63, -0.63, -0.63],
        ],
        dtype=torch.float32,
    )

    n = z.shape[0]
    src, dst = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    disp = pos[edge_index[0]] - pos[edge_index[1]]
    batch = torch.zeros(n, dtype=torch.long)

    return (z, disp, edge_index, batch)


MENAGERIE_ENTRIES = [
    (
        "NewtonNet",
        build_newtonnet,
        example_input_newtonnet,
        2022,
        "CODE",
    ),
]
