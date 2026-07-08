# SOURCE: vendored from JungWoo-Chae/GCN_Elliptic_dataset @ HEAD
# SOURCE FILE: Elliptic_dataset_GCN.ipynb, cells defining class GCN
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

MENAGERIE_ZOO = "vendored-pytorch"


class GCN(nn.Module):
    """Two-layer GCN from the Elliptic anti-money-laundering notebook."""

    def __init__(self, num_node_features: int, hidden_channels: list[int]) -> None:
        """Initialize the notebook's two GCNConv layers."""
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_node_features, hidden_channels[0])
        self.conv2 = GCNConv(hidden_channels[0], 2)

    def forward(self, data: Data) -> Tensor:
        """Run the notebook GCN forward pass over a PyG Data object."""
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x


def build_elliptic_gcn() -> GCN:
    """Build a tiny traceable Elliptic GCN."""
    return GCN(num_node_features=6, hidden_channels=[8])


def example_input_elliptic_gcn() -> Data:
    """Return a toy transaction graph for the Elliptic GCN."""
    x = torch.randn(4, 6)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 0, 2], [1, 0, 3, 2, 2, 0]],
        dtype=torch.long,
    )
    return Data(x=x, edge_index=edge_index)


MENAGERIE_ENTRIES = [
    (
        "Anti-Money Laundering GCN (Elliptic / Weber et al.)",
        build_elliptic_gcn,
        example_input_elliptic_gcn,
        2019,
        "CV2a-elliptic-gcn",
    ),
]
