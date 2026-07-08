# SOURCE: vendored from Mohamedelrefaie/DrivAerNet @ main
# https://github.com/Mohamedelrefaie/DrivAerNet/blob/main/DeepSurrogates/DeepSurrogate_models.py
# "DrivAerNet++: A Large-Scale Multimodal Car Dataset with Computational
# Fluid Dynamics Simulations and Deep Learning Benchmarks" (Elrefaie et al.
# 2024). `DragGNN` (3-layer GCNConv message passing + global mean pool +
# 2-layer MLP head) and `DragGNN_XL` (4-layer GCNConv + per-layer BatchNorm
# + dropout + deeper MLP head) are transcribed VERBATIM from the real
# repo's `DeepSurrogate_models.py`; both predict the aerodynamic drag
# coefficient from a graph built over a 3D car-surface point cloud. Only
# the module-level `trimesh` import (unused by these two classes; needed
# elsewhere in the file for mesh I/O) and the unrelated point-cloud
# (`RegDGCNN`, `RegPointNet`) and attention-augmented (`EnhancedDragGNN`)
# classes are dropped for a self-contained staging module -- no
# architectural changes to `DragGNN` / `DragGNN_XL` themselves.
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout, Linear, ReLU, Sequential
from torch_geometric.data import Data
from torch_geometric.nn import BatchNorm, GCNConv, global_mean_pool


class DragGNN(torch.nn.Module):
    """
    Graph Neural Network for predicting drag coefficients using GCNConv layers.

    Args:
        None

    Methods:
        forward(data): Forward pass through the network.
    """

    def __init__(self):
        super(DragGNN, self).__init__()
        self.conv1 = GCNConv(3, 512)
        self.conv2 = GCNConv(512, 1024)
        self.conv3 = GCNConv(1024, 512)
        self.fc1 = torch.nn.Linear(512, 128)
        self.fc2 = torch.nn.Linear(128, 1)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            data (Data): Input graph data containing node features, edge indices, and batch indices.

        Returns:
            torch.Tensor: Output predictions for drag coefficients.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DragGNN_XL(torch.nn.Module):
    """
    Extended Graph Neural Network for predicting drag coefficients using GCNConv layers and BatchNorm layers.

    Args:
        None

    Methods:
        forward(data): Forward pass through the network.
    """

    def __init__(self):
        super(DragGNN_XL, self).__init__()
        self.conv1 = GCNConv(3, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 128)
        self.conv4 = GCNConv(128, 256)

        self.bn1 = BatchNorm(64)
        self.bn2 = BatchNorm(128)
        self.bn3 = BatchNorm(128)
        self.bn4 = BatchNorm(256)

        self.dropout = Dropout(0.4)

        self.fc = Sequential(
            Linear(256, 128), ReLU(), Dropout(0.4), Linear(128, 64), ReLU(), Linear(64, 1)
        )

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            data (Data): Input graph data containing node features, edge indices, and batch indices.

        Returns:
            torch.Tensor: Output predictions for drag coefficients.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.conv4(x, edge_index)))
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x


MENAGERIE_ZOO = "vendored-pytorch"


def _tiny_car_graph():
    torch.manual_seed(0)
    num_nodes = 24
    x = torch.randn(num_nodes, 3)
    # ring + a few chords for a connected, non-trivial graph
    src = list(range(num_nodes))
    dst = [(i + 1) % num_nodes for i in range(num_nodes)]
    src += dst
    dst += list(range(num_nodes))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    batch = torch.zeros(num_nodes, dtype=torch.long)
    return Data(x=x, edge_index=edge_index, batch=batch)


def build_draggnn():
    torch.manual_seed(0)
    model = DragGNN()
    model.eval()
    return model


def example_input_draggnn():
    return (_tiny_car_graph(),)


def build_draggnn_xl():
    torch.manual_seed(0)
    model = DragGNN_XL()
    model.eval()
    return model


def example_input_draggnn_xl():
    return (_tiny_car_graph(),)


MENAGERIE_ENTRIES = [
    ("DrivAerNet_DragGNN", "build_draggnn", "example_input_draggnn", 2024, MENAGERIE_ZOO),
    ("DrivAerNet_DragGNN_XL", "build_draggnn_xl", "example_input_draggnn_xl", 2024, MENAGERIE_ZOO),
]
