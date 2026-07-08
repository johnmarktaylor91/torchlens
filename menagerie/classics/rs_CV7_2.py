# SOURCE: vendored from viannegao/ChromaFold @ HEAD, hehaodele/circuit-gnn @ HEAD, intelligent-environments-lab/CityLearn @ HEAD
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn


class CityLearnLSTM(nn.Module):
    """Vendored CityLearn end-use load profile LSTM."""

    def __init__(
        self,
        n_features: int,
        n_output: int,
        drop_prob: float,
        seq_len: int,
        num_hidden: int,
        num_layers: int,
        weight_decay: float,
    ) -> None:
        """Initialize the CityLearn LSTM."""
        super().__init__()
        self.n_features = n_features
        self.n_output = n_output
        self.seq_len = seq_len
        self.n_hidden = num_hidden
        self.n_layers = num_layers
        self.weight_decay = weight_decay
        self.l_lstm = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.n_hidden,
            num_layers=self.n_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(drop_prob)
        self.l_linear = nn.Linear(self.n_hidden, n_output)
        self.l_linear.weight_decay = self.weight_decay

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden and cell states."""
        hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden, device=device)
        cell_state = torch.zeros(self.n_layers, batch_size, self.n_hidden, device=device)
        return hidden_state, cell_state

    def forward(
        self, input_tensor: torch.Tensor, hidden_cell_tuple: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run the LSTM and return the last-step prediction plus hidden state."""
        lstm_out, hidden_cell_tuple = self.l_lstm(input_tensor, hidden_cell_tuple)
        lstm_out = self.dropout(lstm_out)
        out = lstm_out[:, -1, :]
        out_linear = self.l_linear(out)
        return out_linear, hidden_cell_tuple


class CityLearnTraceWrapper(nn.Module):
    """Single-input wrapper for the vendored CityLearn LSTM."""

    def __init__(self) -> None:
        """Initialize the wrapper."""
        super().__init__()
        self.model = CityLearnLSTM(3, 2, 0.0, 4, 5, 1, 0.0)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run the wrapped LSTM with zero initial state."""
        hidden = self.model.init_hidden(input_tensor.shape[0], input_tensor.device)
        output, _ = self.model(input_tensor, hidden)
        return output


def circuit_complex_linear(
    layer: tuple[nn.Module, nn.Module], x: tuple[torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a vendored complex-valued linear block."""
    return layer[0](x[0]) - layer[1](x[1]), layer[0](x[1]) - layer[1](x[0])


def circuit_complex_apply(
    layer: tuple[nn.Module, nn.Module] | tuple[object, object], x: tuple[torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply two functions or modules to a complex-valued tuple."""
    return layer[0](x[0]), layer[1](x[1])  # type: ignore[operator]


def circuit_complex_add(
    x: tuple[torch.Tensor, torch.Tensor], y: tuple[torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add two complex-valued tuples."""
    return x[0] + y[0], x[1] + y[1]


class CircuitComplexNet(nn.Module):
    """Vendored Circuit-GNN complex prediction network."""

    def __init__(self, nin: int = 5, nh: int = 32, dropout: float = 0.0) -> None:
        """Initialize the complex predictor."""
        super().__init__()
        self.fc1 = nn.ModuleList([nn.Linear(nin, nh), nn.Linear(nin, nh)])
        self.fc2 = nn.ModuleList([nn.Linear(nh, nh), nn.Linear(nh, nh)])
        self.fc22 = nn.ModuleList([nn.Linear(nh, nh), nn.Linear(nh, nh)])
        self.fc2_bn = nn.ModuleList([nn.BatchNorm1d(nh), nn.BatchNorm1d(nh)])
        self.fc22_bn = nn.ModuleList([nn.BatchNorm1d(nh), nn.BatchNorm1d(nh)])
        self.fc3 = nn.ModuleList([nn.Linear(nh, nh), nn.Linear(nh, nh)])
        self.fc32 = nn.ModuleList([nn.Linear(nh, nh), nn.Linear(nh, nh)])
        self.fc3_bn = nn.ModuleList([nn.BatchNorm1d(nh), nn.BatchNorm1d(nh)])
        self.fc32_bn = nn.ModuleList([nn.BatchNorm1d(nh), nn.BatchNorm1d(nh)])
        self.fc4 = nn.ModuleList([nn.Linear(nh, 5001), nn.Linear(nh, 5001)])
        self.dropout_2 = (
            nn.ModuleList([nn.Dropout(dropout), nn.Dropout(dropout)]) if dropout > 0 else None
        )
        self.dropout_3 = (
            nn.ModuleList([nn.Dropout(dropout), nn.Dropout(dropout)]) if dropout > 0 else None
        )

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the complex predictor."""
        x1 = circuit_complex_linear((self.fc1[0], self.fc1[1]), x)
        x1 = F.leaky_relu(x1[0], 0.2), F.leaky_relu(x1[1], 0.2)
        x = circuit_complex_apply(
            (F.relu, F.relu),
            circuit_complex_apply(
                (self.fc2_bn[0], self.fc2_bn[1]),
                circuit_complex_linear((self.fc2[0], self.fc2[1]), x1),
            ),
        )
        x2 = circuit_complex_apply(
            (F.relu, F.relu),
            circuit_complex_apply(
                (self.fc22_bn[0], self.fc22_bn[1]),
                circuit_complex_add(x1, circuit_complex_linear((self.fc22[0], self.fc22[1]), x)),
            ),
        )
        if self.dropout_2 is not None:
            x2 = circuit_complex_apply((self.dropout_2[0], self.dropout_2[1]), x2)
        x = circuit_complex_apply(
            (F.relu, F.relu),
            circuit_complex_apply(
                (self.fc3_bn[0], self.fc3_bn[1]),
                circuit_complex_linear((self.fc3[0], self.fc3[1]), x2),
            ),
        )
        x3 = circuit_complex_apply(
            (F.relu, F.relu),
            circuit_complex_apply(
                (self.fc32_bn[0], self.fc32_bn[1]),
                circuit_complex_add(x2, circuit_complex_linear((self.fc32[0], self.fc32[1]), x)),
            ),
        )
        if self.dropout_3 is not None:
            x3 = circuit_complex_apply((self.dropout_3[0], self.dropout_3[1]), x3)
        return circuit_complex_linear((self.fc4[0], self.fc4[1]), x3)


class CircuitGraphInteractionLayer(nn.Module):
    """Vendored Circuit-GNN graph interaction layer."""

    def __init__(
        self, n_node_attr: int, n_node_code: int, n_edge_attr: int, n_edge_code: int
    ) -> None:
        """Initialize graph interaction processors."""
        super().__init__()
        self.edge_processor = nn.Linear(n_edge_attr + (n_node_attr + n_node_code) * 2, n_edge_code)
        self.node_processor = nn.Linear(n_node_attr + n_node_code + n_edge_code, n_node_code)

    def forward(
        self,
        node_code: torch.Tensor,
        node_attr: torch.Tensor,
        edge_attr: torch.Tensor,
        adj: torch.Tensor,
        return_edge_code: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run one graph interaction update."""
        batch_size, num_nodes = node_code.size(0), node_code.size(1)
        node_info = torch.cat([node_code, node_attr], 2)
        receiver_info = node_info[:, :, None, :].repeat(1, 1, num_nodes, 1)
        sender_info = node_info[:, None, :, :].repeat(1, num_nodes, 1, 1)
        edge_input = torch.cat([edge_attr, receiver_info, sender_info], 3)
        edge_code = F.leaky_relu(
            self.edge_processor(edge_input.reshape(batch_size * num_nodes * num_nodes, -1)).reshape(
                batch_size, num_nodes, num_nodes, -1
            )
        )
        edge_agg = (edge_code * adj[:, :, :, None]).sum(2)
        node_input = torch.cat([node_info, edge_agg], 2)
        new_node_code = self.node_processor(node_input.reshape(batch_size * num_nodes, -1)).reshape(
            batch_size, num_nodes, -1
        )
        if return_edge_code:
            return new_node_code, edge_code
        return new_node_code


class CircuitGIN(nn.Module):
    """Vendored Circuit-GNN graph interaction network."""

    def __init__(
        self,
        n_node_attr: int,
        n_node_code: int,
        n_edge_attr: int,
        n_edge_code: int,
        n_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        """Initialize the graph interaction network."""
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CircuitGraphInteractionLayer(n_node_attr, n_node_code, n_edge_attr, n_edge_code)
                for _ in range(n_layers)
            ]
        )
        self.node_encoder = nn.Sequential(nn.Linear(n_node_attr, n_node_code), nn.LeakyReLU(0.1))
        self.drop_layers = (
            nn.ModuleList([nn.Dropout(p=dropout) for _ in range(n_layers)]) if dropout > 0 else None
        )

    def forward(
        self, node_attr: torch.Tensor, edge_attr: torch.Tensor, adj: torch.Tensor
    ) -> torch.Tensor:
        """Encode graph node attributes."""
        x = self.node_encoder(node_attr)
        for i, layer in enumerate(self.layers):
            x = layer(x, node_attr, edge_attr, adj)
            if not isinstance(x, torch.Tensor):
                x = x[0]
            x = F.leaky_relu(x)
            if self.drop_layers is not None:
                x = self.drop_layers[i](x)
        return x


class CircuitGNN(nn.Module):
    """Vendored Circuit-GNN model."""

    def __init__(self, args: SimpleNamespace) -> None:
        """Initialize Circuit-GNN."""
        super().__init__()
        nhid = args.len_hidden
        self.gnn_encoder = CircuitGIN(
            n_node_code=nhid,
            n_edge_code=nhid,
            n_node_attr=args.len_node_attr,
            n_edge_attr=args.len_edge_attr,
            n_layers=args.gnn_layers,
            dropout=args.dropout,
        )
        self.predictor = CircuitComplexNet(
            nin=nhid * 2, nh=args.len_hidden_predictor, dropout=args.dropout
        )

    def forward(self, model_input: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Run Circuit-GNN over node, edge, and adjacency tensors."""
        node_attr, edge_attr, adj = model_input
        gnn_node_codes = self.gnn_encoder(node_attr, edge_attr, adj)
        gnn_code = torch.cat([gnn_node_codes[:, 0, :], gnn_node_codes[:, -1, :]], 1)
        pred = self.predictor((gnn_code, gnn_code))
        pred_tensor = torch.cat([pred[0][:, None, :], pred[1][:, None, :]], 1)
        return torch.tanh(pred_tensor)


class CircuitTraceWrapper(nn.Module):
    """Single-input wrapper for Circuit-GNN."""

    def __init__(self) -> None:
        """Initialize a tiny Circuit-GNN."""
        super().__init__()
        args = SimpleNamespace(
            len_hidden=8,
            len_node_attr=3,
            len_edge_attr=2,
            gnn_layers=1,
            dropout=0.0,
            len_hidden_predictor=8,
        )
        self.model = CircuitGNN(args)

    def forward(self, flat_input: torch.Tensor) -> torch.Tensor:
        """Split a flat tensor into node, edge, and adjacency inputs."""
        node_attr = flat_input[:, :12].reshape(flat_input.shape[0], 4, 3)
        edge_attr = flat_input[:, 12:44].reshape(flat_input.shape[0], 4, 4, 2)
        adj = torch.sigmoid(flat_input[:, 44:60].reshape(flat_input.shape[0], 4, 4))
        return self.model((node_attr, edge_attr, adj))


class ChromaResBlock(nn.Module):
    """Vendored ChromaFold residual 1D block."""

    def __init__(self, ni: int) -> None:
        """Initialize the residual block."""
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv1d(ni, ni, 3, 1, 1),
            nn.BatchNorm1d(ni),
            nn.ReLU(),
            nn.Conv1d(ni, ni, 3, 1, 1),
            nn.BatchNorm1d(ni),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the residual block."""
        return self.blocks(x) + x


class ChromaSymmetrizeBulk(nn.Module):
    """Vendored ChromaFold bulk symmetrization layer."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert a 1D track into a symmetric 2D representation."""
        batch, channels, length = x.shape
        x = x.reshape(batch, channels, 1, length).repeat(1, 1, length, 1)
        x_t = x.permute(0, 1, 3, 2)
        return torch.concat((x, x_t), axis=1)


class ChromaBranchPbulk(nn.Module):
    """Vendored ChromaFold bulk branch."""

    def __init__(self) -> None:
        """Initialize the ChromaFold bulk branch."""
        super().__init__()
        self.bulk_summed_2d = nn.Sequential(nn.AvgPool1d(kernel_size=200), ChromaSymmetrizeBulk())
        self.bulk_extractor_2d = nn.Sequential(
            nn.Conv1d(2, 16, 11, 1, padding="same"),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 7, 1, padding="same"),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 32, 5, 1, padding="same"),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 32, 5, 1, padding="same"),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, 5, 1, padding="same"),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, 5, 1, dilation=2, padding="same"),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, 5, 1, dilation=3, padding="same"),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, 5, 1, dilation=5, padding="same"),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, 5, 1, dilation=5, padding="same"),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, 5, 1, dilation=7, padding="same"),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, 5, 1, dilation=11, padding="same"),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, 5, 1, dilation=11, padding="same"),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(5),
            nn.Conv1d(32, 16, 3, 1, padding="same"),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(5),
            nn.Conv1d(16, 16, 3, 1, padding="same"),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 16, 3, 1, padding="same"),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            ChromaSymmetrizeBulk(),
        )
        self.total_extractor_2d = nn.Sequential(
            nn.Conv2d(36, 64, 3, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 16, 3, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(nn.Linear(in_features=1936, out_features=512))

    def forward(self, x2: torch.Tensor) -> torch.Tensor:
        """Run the ChromaFold bulk branch."""
        x3_2d = self.bulk_summed_2d(x2)
        x2_2d = self.bulk_extractor_2d(x2)
        x4 = torch.cat((x3_2d, x2_2d), 1)
        x4 = self.total_extractor_2d(x4)
        x4 = torch.flatten(x4, 1)
        return self.classifier(x4)


class ChromaBranchCov(nn.Module):
    """Vendored ChromaFold coverage branch."""

    def __init__(self) -> None:
        """Initialize the ChromaFold coverage branch."""
        super().__init__()
        self.cov_extractor = nn.Sequential(
            nn.Conv1d(20, 16, 5, 1, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 16, 5, 1, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 16, 3, 1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            ChromaResBlock(16),
            nn.MaxPool1d(2),
            ChromaResBlock(16),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 16, 3, 1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 16, 3, 1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 16, 3, 1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(nn.Linear(in_features=992, out_features=512))

    def forward(self, x: torch.Tensor, x_pb: torch.Tensor) -> torch.Tensor:
        """Run the ChromaFold coverage branch."""
        del x_pb
        x = self.cov_extractor(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class ChromaTrunk(nn.Module):
    """Vendored ChromaFold trunk."""

    def __init__(self, branch_pbulk: ChromaBranchPbulk, branch_cov: ChromaBranchCov) -> None:
        """Initialize the ChromaFold trunk."""
        super().__init__()
        self.branch_pbulk = branch_pbulk
        self.branch_cov = branch_cov
        self.out = nn.Sequential(
            nn.Linear(in_features=512 * 2, out_features=512),
            nn.Linear(in_features=512, out_features=200),
        )

    def forward(self, x: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Run ChromaFold from coverage and bulk tracks."""
        x = self.branch_cov(x, x2)
        with torch.no_grad():
            x2 = self.branch_pbulk(x2)
        return self.out(torch.cat((x, x2), 1))


class ChromaTraceWrapper(nn.Module):
    """Single-input wrapper for the two-input ChromaFold trunk."""

    def __init__(self) -> None:
        """Initialize ChromaFold trunk wrapper."""
        super().__init__()
        self.model = ChromaTrunk(ChromaBranchPbulk(), ChromaBranchCov())

    def forward(self, tracks: torch.Tensor) -> torch.Tensor:
        """Split packed tracks into coverage and pbulk inputs."""
        x = tracks[:, :20, :8000]
        x2 = tracks[:, 20:22, :]
        return self.model(x, x2)


def build_citylearn_rlagent() -> nn.Module:
    """Build the vendored CityLearn LSTM wrapper."""
    return CityLearnTraceWrapper()


def example_input_citylearn_rlagent() -> torch.Tensor:
    """Return an example CityLearn LSTM input."""
    return torch.randn(2, 4, 3)


def build_circuit_gnn() -> nn.Module:
    """Build the vendored Circuit-GNN wrapper."""
    return CircuitTraceWrapper()


def example_input_circuit_gnn() -> torch.Tensor:
    """Return an example packed Circuit-GNN input."""
    return torch.randn(2, 60)


def build_chromafold() -> nn.Module:
    """Build the vendored ChromaFold wrapper."""
    return ChromaTraceWrapper()


def example_input_chromafold() -> torch.Tensor:
    """Return an example packed ChromaFold input."""
    return torch.randn(1, 22, 80000)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("Chromafold", "build_chromafold", "example_input_chromafold", 2024, "CV7-218"),
    ("Circuit-GNN", "build_circuit_gnn", "example_input_circuit_gnn", 2023, "CV7-223"),
    (
        "CityLearn-RLAgent",
        "build_citylearn_rlagent",
        "example_input_citylearn_rlagent",
        2021,
        "CV7-224",
    ),
]
