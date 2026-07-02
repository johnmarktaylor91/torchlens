# SOURCE: vendored from MIT-REALM/gcbf-pytorch @ main
# Files: gcbf/nn/mlp.py, gcbf/nn/utils.py, gcbf/nn/gnn.py (CBFGNNLayer only), gcbf/algo/gcbf.py (CBFGNN only)
# https://github.com/MIT-REALM/gcbf-pytorch
#
# CBFGNN is the neural Control Barrier Function network from "Graph Neural Network-based Control
# Barrier Functions for Distributed, Safe Multi-Agent Control" (Zhang, Zhang, Fan; CoRL 2023).
# It is a graph-attention message-passing network operating over torch_geometric `Data` batches
# of agent states (node features), pairwise relative info (edge features), and the graph
# connectivity (edge_index); it outputs a scalar CBF value per agent node.
#
# Only the actual nn.Module architecture (MLP, CBFGNNLayer message-passing layer, CBFGNN) is
# vendored here; the training loop / RL environment / buffer machinery in gcbf/algo/gcbf.py is
# dropped as harness plumbing, not architecture.

import torch
import torch.nn as nn

from typing import Optional, Tuple
from torch.nn.utils import spectral_norm
from torch_geometric.nn.conv.message_passing import MessagePassing
from torch import Tensor, cat
from torch_sparse import SparseTensor
from torch_geometric.nn.aggr.attention import AttentionalAggregation
from torch_geometric.nn import Sequential
from torch_geometric.data import Data
from torch_geometric.utils import softmax


def init_param(module: nn.Module, gain: float = 1.0):
    nn.init.orthogonal_(module.weight.data, gain=gain)
    nn.init.constant_(module.bias.data, 0)
    return module


class MLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_layers: tuple,
        hidden_activation: nn.Module = nn.ReLU(),
        output_activation: Optional[nn.Module] = None,
        init: bool = True,
        gain: float = 1.0,
        limit_lip: bool = False,
    ):
        super().__init__()

        layers = []
        units = in_channels
        for next_units in hidden_layers:
            if init:
                if limit_lip:
                    layers.append(
                        init_param(spectral_norm(nn.Linear(units, next_units)), gain=gain)
                    )
                else:
                    layers.append(init_param(nn.Linear(units, next_units), gain=gain))
            else:
                if limit_lip:
                    layers.append(spectral_norm(nn.Linear(units, next_units)))
                else:
                    layers.append(nn.Linear(units, next_units))
            layers.append(hidden_activation)
            units = next_units
        if init:
            if limit_lip:
                layers.append(init_param(spectral_norm(nn.Linear(units, out_channels)), gain=gain))
            else:
                layers.append(init_param(nn.Linear(units, out_channels), gain=gain))
        else:
            if limit_lip:
                layers.append(spectral_norm(nn.Linear(units, out_channels)))
            else:
                layers.append(nn.Linear(units, out_channels))
        if output_activation is not None:
            layers.append(output_activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CBFGNNLayer(MessagePassing):
    def __init__(self, node_dim: int, edge_dim: int, output_dim: int, phi_dim: int):
        super(CBFGNNLayer, self).__init__(
            aggr=AttentionalAggregation(
                gate_nn=MLP(
                    in_channels=phi_dim, out_channels=1, hidden_layers=(128, 128), limit_lip=False
                )
            )
        )
        self.phi = MLP(
            in_channels=2 * node_dim + edge_dim,
            out_channels=phi_dim,
            hidden_layers=(2048, 2048),
            limit_lip=True,
        )
        self.gamma = MLP(
            in_channels=phi_dim + node_dim,
            out_channels=output_dim,
            hidden_layers=(2048, 2048),
            limit_lip=True,
        )

    def forward(self, x: Tensor, edge_attr: Tensor, edge_index: Tensor) -> Tensor:
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: Tensor, x_i: Tensor = None, edge_attr: Tensor = None) -> Tensor:
        info_ij = cat([x_i, x_j, edge_attr], dim=1)
        return self.phi(info_ij)

    def update(self, aggr_out: Tensor, x: Tensor = None) -> Tensor:
        gamma_input = cat([aggr_out, x], dim=1)
        return self.gamma(gamma_input)

    def message_and_aggregate(self, adj_t: SparseTensor) -> Tensor:
        raise NotImplementedError

    def edge_update(self) -> Tensor:
        raise NotImplementedError

    def attention(self, data: Data):
        kwargs = {"x": data.x, "edge_attr": data.edge_attr}
        size = self._check_input(data.edge_index, None)
        coll_dict = self._collect(self._user_args, data.edge_index, size, kwargs)
        msg_kwargs = self.inspector.distribute("message", coll_dict)
        out = self.message(**msg_kwargs)
        aggr_kwargs = self.inspector.distribute("aggregate", coll_dict)
        gate = self.aggr_module.gate_nn(out)
        attention = softmax(
            gate, aggr_kwargs["index"], aggr_kwargs["ptr"], aggr_kwargs["dim_size"], dim=-2
        )
        return attention


class CBFGNN(nn.Module):
    """Graph-attention neural Control Barrier Function network (MIT-REALM/gcbf-pytorch)."""

    def __init__(self, num_agents: int, node_dim: int, edge_dim: int, phi_dim: int):
        super(CBFGNN, self).__init__()
        self.num_agents = num_agents
        self.feat_transformer = Sequential(
            "x, edge_attr, edge_index",
            [
                (
                    CBFGNNLayer(
                        node_dim=node_dim, edge_dim=edge_dim, output_dim=1024, phi_dim=phi_dim
                    ),
                    "x, edge_attr, edge_index -> x",
                ),
            ],
        )
        self.feat_2_CBF = MLP(
            in_channels=1024,
            out_channels=1,
            hidden_layers=(512, 128, 32),
            output_activation=nn.Tanh(),
        )

    def forward(self, data: Data) -> Tensor:
        """
        Get the CBF value for the input states.

        Parameters
        ----------
        data: Data
            batched data using Batch.from_data_list().

        Returns
        -------
        h: Tensor (bs x n,)
            CBF values for all agents
        """
        x = self.feat_transformer(data.x, data.edge_attr, data.edge_index)
        if hasattr(data, "agent_mask"):
            x = x[data.agent_mask]
        h = self.feat_2_CBF(x)
        return h

    def attention(self, data: Data) -> Tensor:
        return self.feat_transformer.module_0.attention(data)


# ==================================================================================================
# MENAGERIE staging entry points

MENAGERIE_ZOO = "vendored-pytorch"

_NUM_AGENTS = 4
_NODE_DIM = 4
_EDGE_DIM = 4
_PHI_DIM = 32  # reduced from paper default (256) for a tiny trace-sized instance


def build_neural_cbf():
    return CBFGNN(num_agents=_NUM_AGENTS, node_dim=_NODE_DIM, edge_dim=_EDGE_DIM, phi_dim=_PHI_DIM)


def example_input_neural_cbf():
    # Build a small fully-connected graph over `_NUM_AGENTS` agents (excluding self-loops), with
    # random node/edge features, matching what `CBFGNN.forward` expects from a `Data` batch.
    n = _NUM_AGENTS
    src = []
    dst = []
    for i in range(n):
        for j in range(n):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    x = torch.randn(n, _NODE_DIM)
    edge_attr = torch.randn(edge_index.shape[1], _EDGE_DIM)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


MENAGERIE_ENTRIES = [
    ("Neural CBF", build_neural_cbf, example_input_neural_cbf, 2023, "vendored-pytorch"),
]
