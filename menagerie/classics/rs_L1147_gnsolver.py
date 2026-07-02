# FAITHFUL PORT of bdonon/GraphNeuralSolver @ master (original framework: TensorFlow 1.x)
#   models/layers.py  (custom_gather, custom_scatter, FullyConnected)
#   models/models.py  (GraphNeuralSolver.build_weights / build_graph)
# Graph Neural Solver for physical/power-system equilibrium problems (Donon et al.,
# "Neural networks for power flow: Graph neural solver", 2019/2020). The original code is
# TF1.x graph-mode (tf.compat.v1.get_variable, session-based training loop, TFRecord data
# pipeline) and cannot run in a base torch env, so the architecture is transcribed
# faithfully into torch: an iterative graph-correction network that repeatedly (1) gathers
# latent node messages onto edges, (2) computes three edge-conditioned MLP transforms
# (phi_from / phi_to / phi_loop, gated by a self-loop mask), (3) scatter-sums each back to
# nodes, (4) concatenates with the running node latent + exogenous node features B and
# feeds a per-update correction MLP, and (5) decodes the updated latent state to the
# physical output via a per-update decoder MLP (offset by a fixed initial guess U0).
"""FAITHFUL PORT of the GraphNeuralSolver TF1.x graph-correction network."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FullyConnected(nn.Module):
    """Port of layers.py::FullyConnected. Applies an MLP independently to every
    node/edge slot of a [n_samples, n_elem, d_in] tensor."""

    def __init__(
        self,
        latent_dimension: int = 10,
        hidden_layers: int = 3,
        non_lin: str = "leaky_relu",
        input_dim: int = None,
        output_dim: int = None,
    ) -> None:
        super().__init__()
        self.latent_dimension = latent_dimension
        self.hidden_layers = hidden_layers
        self.input_dim = input_dim
        self.output_dim = output_dim

        if non_lin == "tanh":
            self.non_lin = torch.tanh
        elif non_lin == "leaky_relu":
            self.non_lin = F.leaky_relu
        else:
            raise ValueError(f"Unsupported non_lin: {non_lin}")

        self.linears = nn.ModuleList()
        for layer in range(hidden_layers):
            left_dim = latent_dimension
            right_dim = latent_dimension
            if layer == 0 and input_dim is not None:
                left_dim = input_dim
            if layer == hidden_layers - 1 and output_dim is not None:
                right_dim = output_dim
            self.linears.append(nn.Linear(left_dim, right_dim))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        n_samples, n_elem, _ = h.shape
        h = h.reshape(-1, h.shape[-1])
        for layer, lin in enumerate(self.linears):
            if layer == self.hidden_layers - 1:
                h = lin(h)
            else:
                h = self.non_lin(lin(h))
        return h.reshape(n_samples, n_elem, -1)


def custom_gather(params: torch.Tensor, indices_edges: torch.Tensor) -> torch.Tensor:
    """Port of layers.py::custom_gather. Batched node -> edge gather.

    params: [n_samples, n_nodes, d_out]
    indices_edges: [n_samples, n_edges] (long)
    returns: [n_samples, n_edges, d_out]
    """
    n_samples, n_edges = indices_edges.shape
    batch_idx = (
        torch.arange(n_samples, device=params.device).view(n_samples, 1).expand(n_samples, n_edges)
    )
    return params[batch_idx, indices_edges]


def custom_scatter(indices_edges: torch.Tensor, params: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """Port of layers.py::custom_scatter. Batched edge -> node scatter-add.

    indices_edges: [n_samples, n_edges] (long)
    params: [n_samples, n_edges, d_F]
    returns: [n_samples, n_nodes, d_F]
    """
    n_samples, n_edges, d_F = params.shape
    out = params.new_zeros(n_samples, n_nodes, d_F)
    idx = indices_edges.unsqueeze(-1).expand(n_samples, n_edges, d_F)
    out.scatter_add_(1, idx, params)
    return out


class GraphNeuralSolver(nn.Module):
    """Port of models.py::GraphNeuralSolver.build_weights / build_graph.

    Iterative graph-correction network. Given edge tensor A (indices_from,
    indices_to, edge characteristics a_ij) and node tensor B (exogenous node
    features), repeatedly refines a latent node state H and decodes a physical
    prediction X at every correction step.
    """

    def __init__(
        self,
        d_in_a: int = 1,
        d_in_b: int = 1,
        d_out: int = 1,
        latent_dimension: int = 10,
        hidden_layers: int = 3,
        correction_updates: int = 5,
        alpha: float = 1e-3,
        non_lin: str = "leaky_relu",
        initial_u: Tuple[float, ...] = (0.0,),
    ) -> None:
        super().__init__()
        self.d_in_a = d_in_a
        self.d_in_b = d_in_b
        self.d_out = d_out
        self.latent_dimension = latent_dimension
        self.correction_updates = correction_updates
        self.alpha = alpha
        self.register_buffer("initial_u", torch.tensor(initial_u, dtype=torch.float32))

        self.correction_block = nn.ModuleList()
        self.phi_from = nn.ModuleList()
        self.phi_to = nn.ModuleList()
        self.phi_loop = nn.ModuleList()
        for _ in range(correction_updates):
            self.correction_block.append(
                FullyConnected(
                    non_lin=non_lin,
                    latent_dimension=latent_dimension,
                    hidden_layers=hidden_layers,
                    input_dim=4 * latent_dimension + d_in_b,
                )
            )
            self.phi_from.append(
                FullyConnected(
                    non_lin=non_lin,
                    latent_dimension=latent_dimension,
                    hidden_layers=hidden_layers,
                    input_dim=2 * latent_dimension + d_in_a,
                )
            )
            self.phi_to.append(
                FullyConnected(
                    non_lin=non_lin,
                    latent_dimension=latent_dimension,
                    hidden_layers=hidden_layers,
                    input_dim=2 * latent_dimension + d_in_a,
                )
            )
            self.phi_loop.append(
                FullyConnected(
                    non_lin=non_lin,
                    latent_dimension=latent_dimension,
                    hidden_layers=hidden_layers,
                    input_dim=2 * latent_dimension + d_in_a,
                )
            )

        self.D = nn.ModuleList()
        for _ in range(correction_updates + 1):
            self.D.append(
                FullyConnected(
                    non_lin=non_lin,
                    latent_dimension=latent_dimension,
                    hidden_layers=hidden_layers,
                    input_dim=latent_dimension,
                    output_dim=d_out,
                )
            )

    def forward(
        self,
        indices_from: torch.Tensor,
        indices_to: torch.Tensor,
        a_ij: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """
        indices_from, indices_to: [n_samples, n_edges] (long)
        a_ij: [n_samples, n_edges, d_in_a]
        b: [n_samples, n_nodes, d_in_b]
        returns: X_final [n_samples, n_nodes, d_out]
        """
        n_samples, n_nodes, _ = b.shape

        mask_loop = (indices_from == indices_to).float().unsqueeze(-1)

        initial_u_tf = self.initial_u.view(1, 1, -1).expand(n_samples, n_nodes, -1)

        h = b.new_zeros(n_samples, n_nodes, self.latent_dimension)
        x = self.D[0](h) + initial_u_tf

        for update in range(self.correction_updates):
            h_from = custom_gather(h, indices_from)
            h_to = custom_gather(h, indices_to)

            phi_input = torch.cat([h_from, h_to, a_ij], dim=2)

            phi_from_val = self.phi_from[update](phi_input) * (1.0 - mask_loop)
            phi_to_val = self.phi_to[update](phi_input) * (1.0 - mask_loop)
            phi_loop_val = self.phi_loop[update](phi_input) * mask_loop

            phi_from_sum = custom_scatter(indices_from, phi_from_val, n_nodes)
            phi_to_sum = custom_scatter(indices_to, phi_to_val, n_nodes)
            phi_loop_sum = custom_scatter(indices_to, phi_loop_val, n_nodes)

            correction_input = torch.cat([h, phi_from_sum, phi_to_sum, phi_loop_sum, b], dim=2)
            correction = self.correction_block[update](correction_input)

            h = h + correction * self.alpha
            x = self.D[update + 1](h) + initial_u_tf

        return x


class _GNSolverTraceWrapper(nn.Module):
    """Tuple-in wrapper so TorchLens sees a single positional call."""

    def __init__(self, solver: GraphNeuralSolver) -> None:
        super().__init__()
        self.solver = solver

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        indices_from, indices_to, a_ij, b = inputs
        return self.solver(indices_from, indices_to, a_ij, b)


_D_IN_A = 1  # spring-constant edge characteristic (datasets/spring/default)
_D_IN_B = 1  # exogenous node forcing term
_D_OUT = 1  # scalar node state (1D spring displacement)


def build_gnsolver() -> nn.Module:
    solver = GraphNeuralSolver(
        d_in_a=_D_IN_A,
        d_in_b=_D_IN_B,
        d_out=_D_OUT,
        latent_dimension=8,
        hidden_layers=2,
        correction_updates=3,
        alpha=1e-3,
        non_lin="leaky_relu",
        initial_u=(0.0,),
    )
    solver.eval()
    return _GNSolverTraceWrapper(solver)


def example_input_gnsolver() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_samples = 1
    n_nodes = 10
    n_edges = 15
    indices_from = torch.randint(0, n_nodes, (n_samples, n_edges), dtype=torch.long)
    indices_to = torch.randint(0, n_nodes, (n_samples, n_edges), dtype=torch.long)
    a_ij = torch.rand(n_samples, n_edges, _D_IN_A) * 0.9 + 0.1
    b = torch.rand(n_samples, n_nodes, _D_IN_B) * 0.9 + 0.1
    return (indices_from, indices_to, a_ij, b)


MENAGERIE_ZOO = "ported-pytorch"

MENAGERIE_ENTRIES = [
    (
        "Graph Neural Solver for Power Systems",
        build_gnsolver,
        example_input_gnsolver,
        2019,
        MENAGERIE_ZOO,
    ),
]
