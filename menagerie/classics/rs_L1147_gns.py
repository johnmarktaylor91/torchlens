# SOURCE: vendored from geoelements/gns @ main
#   gns/graph_network.py (EncodeProcessDecode + Encoder/Processor/Decoder/InteractionNetwork)
# Graph Network-based Simulator (GNS): Encode-Process-Decode graph network built on
# torch_geometric MessagePassing, used to learn particle-based physics simulators
# (Sanchez-Gonzalez et al., "Learning to Simulate Complex Physics with Graph Networks",
# ICML 2020). This is the reusable graph-network backbone from learned_simulator.py's
# `EncodeProcessDecode`; the surrounding `LearnedSimulator` class only adds
# physics-specific (non-tensor / numpy) preprocessing around this core network.
"""Vendored GNS Encode-Process-Decode graph network (geoelements/gns)."""

from typing import List, Tuple

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


def build_mlp(
    input_size: int,
    hidden_layer_sizes: List[int],
    output_size: int = None,
    output_activation: nn.Module = nn.Identity,
    activation: nn.Module = nn.ReLU,
) -> nn.Module:
    """Build a MultiLayer Perceptron."""
    layer_sizes = [input_size] + hidden_layer_sizes
    if output_size:
        layer_sizes.append(output_size)

    nlayers = len(layer_sizes) - 1

    act = [activation for i in range(nlayers)]
    act[-1] = output_activation

    mlp = nn.Sequential()
    for i in range(nlayers):
        mlp.add_module("NN-" + str(i), nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        mlp.add_module("Act-" + str(i), act[i]())

    return mlp


class Encoder(nn.Module):
    """Graph network encoder. Encode nodes and edges states to an MLP."""

    def __init__(
        self,
        nnode_in_features: int,
        nnode_out_features: int,
        nedge_in_features: int,
        nedge_out_features: int,
        nmlp_layers: int,
        mlp_hidden_dim: int,
    ):
        super(Encoder, self).__init__()
        self.node_fn = nn.Sequential(
            *[
                build_mlp(
                    nnode_in_features,
                    [mlp_hidden_dim for _ in range(nmlp_layers)],
                    nnode_out_features,
                ),
                nn.LayerNorm(nnode_out_features),
            ]
        )
        self.edge_fn = nn.Sequential(
            *[
                build_mlp(
                    nedge_in_features,
                    [mlp_hidden_dim for _ in range(nmlp_layers)],
                    nedge_out_features,
                ),
                nn.LayerNorm(nedge_out_features),
            ]
        )

    def forward(self, x: torch.Tensor, edge_features: torch.Tensor):
        return self.node_fn(x), self.edge_fn(edge_features)


class InteractionNetwork(MessagePassing):
    def __init__(
        self,
        nnode_in: int,
        nnode_out: int,
        nedge_in: int,
        nedge_out: int,
        nmlp_layers: int,
        mlp_hidden_dim: int,
    ):
        """InteractionNetwork derived from torch_geometric MessagePassing class"""
        super(InteractionNetwork, self).__init__(aggr="add")
        self.node_fn = nn.Sequential(
            *[
                build_mlp(
                    nnode_in + nedge_out,
                    [mlp_hidden_dim for _ in range(nmlp_layers)],
                    nnode_out,
                ),
                nn.LayerNorm(nnode_out),
            ]
        )
        self.edge_fn = nn.Sequential(
            *[
                build_mlp(
                    nnode_in + nnode_in + nedge_in,
                    [mlp_hidden_dim for _ in range(nmlp_layers)],
                    nedge_out,
                ),
                nn.LayerNorm(nedge_out),
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ):
        x_residual = x
        edge_features_residual = edge_features
        x, edge_features = self.propagate(edge_index=edge_index, x=x, edge_features=edge_features)

        return x + x_residual, edge_features + edge_features_residual

    def message(
        self, x_i: torch.Tensor, x_j: torch.Tensor, edge_features: torch.Tensor
    ) -> torch.Tensor:
        edge_features = torch.cat([x_i, x_j, edge_features], dim=-1)
        self._edge_features = self.edge_fn(edge_features)
        return self._edge_features

    def update(self, x_updated: torch.Tensor, x: torch.Tensor, edge_features: torch.Tensor):
        x_updated = torch.cat([x_updated, x], dim=-1)
        x_updated = self.node_fn(x_updated)
        return x_updated, self._edge_features


class Processor(MessagePassing):
    """The Processor computes interactions among nodes via M steps of learned
    message-passing, producing a sequence of updated latent graphs."""

    def __init__(
        self,
        nnode_in: int,
        nnode_out: int,
        nedge_in: int,
        nedge_out: int,
        nmessage_passing_steps: int,
        nmlp_layers: int,
        mlp_hidden_dim: int,
    ):
        super(Processor, self).__init__(aggr="max")
        self.gnn_stacks = nn.ModuleList(
            [
                InteractionNetwork(
                    nnode_in=nnode_in,
                    nnode_out=nnode_out,
                    nedge_in=nedge_in,
                    nedge_out=nedge_out,
                    nmlp_layers=nmlp_layers,
                    mlp_hidden_dim=mlp_hidden_dim,
                )
                for _ in range(nmessage_passing_steps)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ):
        for gnn in self.gnn_stacks:
            x, edge_features = gnn(x, edge_index, edge_features)
        return x, edge_features


class Decoder(nn.Module):
    """The Decoder extracts the dynamics information from the nodes of the
    final latent graph."""

    def __init__(
        self,
        nnode_in: int,
        nnode_out: int,
        nmlp_layers: int,
        mlp_hidden_dim: int,
    ):
        super(Decoder, self).__init__()
        self.node_fn = build_mlp(nnode_in, [mlp_hidden_dim for _ in range(nmlp_layers)], nnode_out)

    def forward(self, x: torch.Tensor):
        return self.node_fn(x)


class EncodeProcessDecode(nn.Module):
    def __init__(
        self,
        nnode_in_features: int,
        nnode_out_features: int,
        nedge_in_features: int,
        latent_dim: int,
        nmessage_passing_steps: int,
        nmlp_layers: int,
        mlp_hidden_dim: int,
    ):
        """Encode-Process-Decode function approximator for learnable simulator."""
        super(EncodeProcessDecode, self).__init__()
        self._encoder = Encoder(
            nnode_in_features=nnode_in_features,
            nnode_out_features=latent_dim,
            nedge_in_features=nedge_in_features,
            nedge_out_features=latent_dim,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )
        self._processor = Processor(
            nnode_in=latent_dim,
            nnode_out=latent_dim,
            nedge_in=latent_dim,
            nedge_out=latent_dim,
            nmessage_passing_steps=nmessage_passing_steps,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )
        self._decoder = Decoder(
            nnode_in=latent_dim,
            nnode_out=nnode_out_features,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ):
        x, edge_features = self._encoder(x, edge_features)
        x, edge_features = self._processor(x, edge_index, edge_features)
        x = self._decoder(x)
        return x


class _GNSTraceWrapper(nn.Module):
    """Tuple-in wrapper so TorchLens sees a single positional call."""

    def __init__(self, epd: EncodeProcessDecode) -> None:
        super().__init__()
        self.epd = epd

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        x, edge_index, edge_features = inputs
        return self.epd(x, edge_index, edge_features)


_NNODE_IN = 30  # 5 velocity steps * 2 dims + 4 boundary distances + 16 particle-type embed
_NEDGE_IN = 3  # 2D relative displacement + relative distance
_LATENT_DIM = 16
_PARTICLE_DIM = 2  # 2D simulation


def build_gns() -> nn.Module:
    epd = EncodeProcessDecode(
        nnode_in_features=_NNODE_IN,
        nnode_out_features=_PARTICLE_DIM,
        nedge_in_features=_NEDGE_IN,
        latent_dim=_LATENT_DIM,
        nmessage_passing_steps=3,
        nmlp_layers=2,
        mlp_hidden_dim=16,
    )
    epd.eval()
    return _GNSTraceWrapper(epd)


def example_input_gns() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_particles = 12
    n_edges = 30
    x = torch.randn(n_particles, _NNODE_IN)
    edge_index = torch.randint(0, n_particles, (2, n_edges), dtype=torch.long)
    edge_features = torch.randn(n_edges, _NEDGE_IN)
    return (x, edge_index, edge_features)


MENAGERIE_ZOO = "vendored-pytorch"

MENAGERIE_ENTRIES = [
    ("Graph Network-based Simulator (GNS)", build_gns, example_input_gns, 2020, MENAGERIE_ZOO),
]
