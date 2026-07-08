# SOURCE: vendored from tsinghua-fib-lab/Traffic-Benchmark @ b9f8e40, uber-research/differentiable-plasticity @ 5bd29a1,
# SOURCE: vendored from ChFrenkel/DirectRandomTargetProjection @ dfe02b8, marco-rudolph/differnet @ 9bdf026

from __future__ import annotations

from collections import OrderedDict
from math import exp
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models import alexnet


class GConvRNN(nn.Module):
    """Graph convolution used by DGCRN recurrent gates."""

    def forward(self, x: Tensor, a: Tensor) -> Tensor:
        """Apply a batch-specific adjacency matrix.

        Parameters
        ----------
        x
            Node feature tensor.
        a
            Batch adjacency tensor.

        Returns
        -------
        Tensor
            Convolved node features.
        """
        return torch.einsum("nvc,nvw->nwc", (x, a)).contiguous()


class GConvHyper(nn.Module):
    """Graph convolution used by DGCRN hypernetwork filters."""

    def forward(self, x: Tensor, a: Tensor) -> Tensor:
        """Apply a shared adjacency matrix.

        Parameters
        ----------
        x
            Node feature tensor.
        a
            Shared adjacency tensor.

        Returns
        -------
        Tensor
            Convolved node features.
        """
        return torch.einsum("nvc,vw->nwc", (x, a)).contiguous()


class GCN(nn.Module):
    """DGCRN graph convolution block."""

    def __init__(
        self,
        dims: list[int],
        gdep: int,
        dropout: float,
        alpha: float,
        beta: float,
        gamma: float,
        type_: str | None = None,
    ) -> None:
        """Initialize the graph convolution block."""
        super().__init__()
        del dropout
        if type_ == "RNN":
            self.gconv = GConvRNN()
            self.gconv_pre_a = GConvHyper()
            self.mlp: nn.Module = nn.Linear((gdep + 1) * dims[0], dims[1])
        else:
            self.gconv = GConvHyper()
            self.gconv_pre_a = GConvHyper()
            self.mlp = nn.Sequential(
                OrderedDict(
                    [
                        ("fc1", nn.Linear((gdep + 1) * dims[0], dims[1])),
                        ("sigmoid1", nn.Sigmoid()),
                        ("fc2", nn.Linear(dims[1], dims[2])),
                        ("sigmoid2", nn.Sigmoid()),
                        ("fc3", nn.Linear(dims[2], dims[3])),
                    ]
                )
            )
        self.gdep = gdep
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.type_gnn = type_

    def forward(self, x: Tensor, adj: Tensor | list[Tensor]) -> Tensor:
        """Run the DGCRN graph convolution."""
        h = x
        out = [h]
        if self.type_gnn == "RNN":
            assert isinstance(adj, list)
            for _ in range(self.gdep):
                h = (
                    self.alpha * x
                    + self.beta * self.gconv(h, adj[0])
                    + self.gamma * self.gconv_pre_a(h, adj[1])
                )
                out.append(h)
        else:
            assert isinstance(adj, Tensor)
            for _ in range(self.gdep):
                h = self.alpha * x + self.gamma * self.gconv(h, adj)
                out.append(h)
        return self.mlp(torch.cat(out, dim=-1))


class DGCRN(nn.Module):
    """Dynamic graph convolution recurrent network."""

    def __init__(
        self,
        gcn_depth: int,
        num_nodes: int,
        predefined_a: list[Tensor],
        dropout: float = 0.3,
        node_dim: int = 8,
        middle_dim: int = 2,
        seq_length: int = 3,
        in_dim: int = 2,
        out_dim: int = 2,
        list_weight: list[float] | None = None,
        tanhalpha: float = 3.0,
        cl_decay_steps: int = 4000,
        rnn_size: int = 8,
        hypergnn_dim: int = 8,
    ) -> None:
        """Initialize DGCRN with the original source parameters at tiny size."""
        super().__init__()
        del out_dim
        weights = [0.05, 0.95, 0.95] if list_weight is None else list_weight
        self.output_dim = 1
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_a = predefined_a
        self.seq_length = seq_length
        self.emb1 = nn.Embedding(self.num_nodes, node_dim)
        self.emb2 = nn.Embedding(self.num_nodes, node_dim)
        self.idx = torch.arange(self.num_nodes)
        self.rnn_size = rnn_size
        self.hidden_size = rnn_size
        dims_hyper = [self.hidden_size + in_dim, hypergnn_dim, middle_dim, node_dim]
        self.gcn1_tg = GCN(dims_hyper, gcn_depth, dropout, *weights, "hyper")
        self.gcn2_tg = GCN(dims_hyper, gcn_depth, dropout, *weights, "hyper")
        self.gcn1_tg_de = GCN(dims_hyper, gcn_depth, dropout, *weights, "hyper")
        self.gcn2_tg_de = GCN(dims_hyper, gcn_depth, dropout, *weights, "hyper")
        self.gcn1_tg_1 = GCN(dims_hyper, gcn_depth, dropout, *weights, "hyper")
        self.gcn2_tg_1 = GCN(dims_hyper, gcn_depth, dropout, *weights, "hyper")
        self.gcn1_tg_de_1 = GCN(dims_hyper, gcn_depth, dropout, *weights, "hyper")
        self.gcn2_tg_de_1 = GCN(dims_hyper, gcn_depth, dropout, *weights, "hyper")
        self.fc_final = nn.Linear(self.hidden_size, self.output_dim)
        self.alpha = tanhalpha
        dims = [in_dim + self.hidden_size, self.hidden_size]
        self.gz1 = GCN(dims, gcn_depth, dropout, *weights, "RNN")
        self.gz2 = GCN(dims, gcn_depth, dropout, *weights, "RNN")
        self.gr1 = GCN(dims, gcn_depth, dropout, *weights, "RNN")
        self.gr2 = GCN(dims, gcn_depth, dropout, *weights, "RNN")
        self.gc1 = GCN(dims, gcn_depth, dropout, *weights, "RNN")
        self.gc2 = GCN(dims, gcn_depth, dropout, *weights, "RNN")
        self.gz1_de = GCN(dims, gcn_depth, dropout, *weights, "RNN")
        self.gz2_de = GCN(dims, gcn_depth, dropout, *weights, "RNN")
        self.gr1_de = GCN(dims, gcn_depth, dropout, *weights, "RNN")
        self.gr2_de = GCN(dims, gcn_depth, dropout, *weights, "RNN")
        self.gc1_de = GCN(dims, gcn_depth, dropout, *weights, "RNN")
        self.gc2_de = GCN(dims, gcn_depth, dropout, *weights, "RNN")
        self.use_curriculum_learning = False
        self.cl_decay_steps = cl_decay_steps

    def preprocessing(self, adj: Tensor, predefined_a: Tensor) -> list[Tensor]:
        """Normalize the dynamic adjacency and pair it with predefined adjacency."""
        norm_adj = adj + torch.eye(self.num_nodes, device=adj.device)
        norm_adj = norm_adj / torch.unsqueeze(norm_adj.sum(-1), -1)
        return [norm_adj, predefined_a.to(adj.device)]

    def step(
        self,
        input_: Tensor,
        hidden_state: Tensor,
        cell_state: Tensor,
        predefined_a: list[Tensor],
        type_: str = "encoder",
    ) -> tuple[Tensor, Tensor]:
        """Run one DGCRN recurrent step."""
        x = input_.transpose(1, 2).contiguous()
        nodevec1 = self.emb1(self.idx.to(x.device))
        nodevec2 = self.emb2(self.idx.to(x.device))
        hyper_input = torch.cat((x, hidden_state.view(-1, self.num_nodes, self.hidden_size)), 2)
        if type_ == "encoder":
            filter1 = self.gcn1_tg(hyper_input, predefined_a[0]) + self.gcn1_tg_1(
                hyper_input, predefined_a[1]
            )
            filter2 = self.gcn2_tg(hyper_input, predefined_a[0]) + self.gcn2_tg_1(
                hyper_input, predefined_a[1]
            )
        else:
            filter1 = self.gcn1_tg_de(hyper_input, predefined_a[0]) + self.gcn1_tg_de_1(
                hyper_input, predefined_a[1]
            )
            filter2 = self.gcn2_tg_de(hyper_input, predefined_a[0]) + self.gcn2_tg_de_1(
                hyper_input, predefined_a[1]
            )
        nodevec1 = torch.tanh(self.alpha * torch.mul(nodevec1, filter1))
        nodevec2 = torch.tanh(self.alpha * torch.mul(nodevec2, filter2))
        adj = F.relu(
            torch.tanh(
                self.alpha
                * (
                    torch.matmul(nodevec1, nodevec2.transpose(2, 1))
                    - torch.matmul(nodevec2, nodevec1.transpose(2, 1))
                )
            )
        )
        adp = self.preprocessing(adj, predefined_a[0])
        adpt = self.preprocessing(adj.transpose(1, 2), predefined_a[1])
        hidden_state = hidden_state.view(-1, self.num_nodes, self.hidden_size)
        cell_state = cell_state.view(-1, self.num_nodes, self.hidden_size)
        combined = torch.cat((x, hidden_state), -1)
        if type_ == "encoder":
            z = torch.sigmoid(self.gz1(combined, adp) + self.gz2(combined, adpt))
            r = torch.sigmoid(self.gr1(combined, adp) + self.gr2(combined, adpt))
            cell_state = torch.tanh(
                self.gc1(torch.cat((x, r * hidden_state), -1), adp)
                + self.gc2(torch.cat((x, r * hidden_state), -1), adpt)
            )
        else:
            z = torch.sigmoid(self.gz1_de(combined, adp) + self.gz2_de(combined, adpt))
            r = torch.sigmoid(self.gr1_de(combined, adp) + self.gr2_de(combined, adpt))
            cell_state = torch.tanh(
                self.gc1_de(torch.cat((x, r * hidden_state), -1), adp)
                + self.gc2_de(torch.cat((x, r * hidden_state), -1), adpt)
            )
        hidden_state = z * hidden_state + (1 - z) * cell_state
        return hidden_state.view(-1, self.hidden_size), cell_state.view(-1, self.hidden_size)

    def forward(self, input_: Tensor, ycl: Tensor) -> Tensor:
        """Run encoder-decoder DGCRN forecasting."""
        predefined_a = [a.to(input_.device) for a in self.predefined_a]
        batch_size = input_.size(0)
        hidden_state = torch.zeros(
            batch_size * self.num_nodes, self.hidden_size, device=input_.device
        )
        cell_state = torch.zeros_like(hidden_state)
        for i in range(self.seq_length):
            hidden_state, cell_state = self.step(
                torch.squeeze(input_[..., i]), hidden_state, cell_state, predefined_a, "encoder"
            )
        decoder_input = torch.zeros(
            (batch_size, self.output_dim, self.num_nodes), device=input_.device
        )
        timeofday = ycl[:, 1:, :, :]
        outputs_final = []
        for i in range(2):
            decoder_input = torch.cat([decoder_input, timeofday[..., i]], dim=1)
            hidden_state, cell_state = self.step(
                decoder_input, hidden_state, cell_state, predefined_a, "decoder"
            )
            decoder_output = self.fc_final(hidden_state)
            decoder_input = decoder_output.view(
                batch_size, self.num_nodes, self.output_dim
            ).transpose(1, 2)
            outputs_final.append(decoder_output)
        outputs = torch.stack(outputs_final, dim=1)
        return outputs.view(batch_size, self.num_nodes, 2, self.output_dim).transpose(1, 2)


class PlasticNetwork(nn.Module):
    """Differentiable-plasticity recurrent network from the simple task."""

    def __init__(self, nb_neur: int = 8) -> None:
        """Initialize the plastic recurrent network."""
        super().__init__()
        self.nb_neur = nb_neur
        self.w = nn.Parameter(0.01 * torch.randn(nb_neur, nb_neur))
        self.alpha = nn.Parameter(0.01 * torch.randn(nb_neur, nb_neur))
        self.eta = nn.Parameter(0.01 * torch.ones(1))

    def forward(self, input_: Tensor, yin: Tensor, hebb: Tensor) -> tuple[Tensor, Tensor]:
        """Run one differentiable-plasticity timestep."""
        yout = torch.tanh(yin.mm(self.w + torch.mul(self.alpha, hebb)) + input_)
        hebb = (1 - self.eta) * hebb + self.eta * torch.bmm(yin.unsqueeze(2), yout.unsqueeze(1))[0]
        return yout, hebb


class FeedbackAlignmentWrapper(nn.Module):
    """Fixed random feedback wrapper from DRTP/FA."""

    def __init__(
        self,
        module: nn.Module,
        layer_type: str,
        dim: torch.Size,
        stride: int | None = None,
        padding: int | None = None,
    ) -> None:
        """Initialize the feedback-alignment wrapper."""
        super().__init__()
        self.module = module
        self.layer_type = layer_type
        self.stride = stride
        self.padding = padding
        self.output_grad: Tensor | None = None
        self.x_shape: torch.Size | None = None
        self.fixed_fb_weights = nn.Parameter(torch.empty(dim), requires_grad=False)
        torch.nn.init.kaiming_uniform_(self.fixed_fb_weights)

    def forward(self, x: Tensor) -> Tensor:
        """Run the wrapped layer."""
        return self.module(x)


class TrainingHook(nn.Module):
    """DRTP training hook; forward is identity for tracing."""

    def __init__(self, label_features: int, dim_hook: list[int] | None, train_mode: str) -> None:
        """Initialize the hook and its fixed feedback weights."""
        super().__init__()
        self.train_mode = train_mode
        if self.train_mode in ["DFA", "DRTP", "sDFA"]:
            assert dim_hook is not None
            self.fixed_fb_weights = nn.Parameter(
                torch.empty(torch.Size(dim_hook)), requires_grad=False
            )
            torch.nn.init.kaiming_uniform_(self.fixed_fb_weights)
        else:
            self.fixed_fb_weights = None
        self.label_features = label_features

    def forward(self, input_: Tensor, labels: Tensor, y: Tensor | None) -> Tensor:
        """Return activations unchanged in the forward pass."""
        del labels, y
        return input_


class Activation(nn.Module):
    """Activation wrapper from DRTP."""

    def __init__(self, activation: str) -> None:
        """Initialize the selected activation."""
        super().__init__()
        if activation == "tanh":
            self.act: nn.Module | None = nn.Tanh()
        elif activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "none":
            self.act = None
        else:
            raise NameError(f"activation {activation} not supported")

    def forward(self, x: Tensor) -> Tensor:
        """Apply the selected activation."""
        return x if self.act is None else self.act(x)


class FCBlock(nn.Module):
    """DRTP fully connected block."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str,
        dropout: float,
        dim_hook: list[int] | None,
        label_features: int,
        fc_zero_init: bool,
        train_mode: str,
    ) -> None:
        """Initialize the fully connected DRTP block."""
        super().__init__()
        self.dropout = dropout
        self.fc: nn.Module = nn.Linear(
            in_features=in_features, out_features=out_features, bias=True
        )
        if fc_zero_init:
            torch.zero_(self.fc.weight.data)
        if train_mode == "FA":
            self.fc = FeedbackAlignmentWrapper(self.fc, "fc", self.fc.weight.shape)
        self.act = Activation(activation)
        if dropout != 0:
            self.drop = nn.Dropout(p=dropout)
        self.hook = TrainingHook(
            label_features=label_features, dim_hook=dim_hook, train_mode=train_mode
        )

    def forward(self, x: Tensor, labels: Tensor, y: Tensor | None) -> Tensor:
        """Run the fully connected block."""
        if self.dropout != 0:
            x = self.drop(x)
        return self.hook(self.act(self.fc(x)), labels, y)


class NetworkBuilder(nn.Module):
    """DRTP arbitrary topology network builder."""

    def __init__(
        self,
        topology: str,
        input_size: int,
        input_channels: int,
        label_features: int,
        train_batch_size: int,
        train_mode: str,
        dropout: float,
        hidden_act: str,
        output_act: str,
        fc_zero_init: bool,
        loss: str,
    ) -> None:
        """Initialize a DRTP network from a topology string."""
        super().__init__()
        self.apply_softmax = (output_act == "none") and (loss == "CE")
        self.layers = nn.ModuleList()
        self.y = (
            torch.zeros(train_batch_size, label_features) if train_mode in ["DFA", "sDFA"] else None
        )
        topology_tokens = topology.split("_")
        topology_layers: list[list[str]] = []
        num_layers = 0
        for elem in topology_tokens:
            if not any(char.isdigit() for char in elem):
                num_layers += 1
                topology_layers.append([])
            topology_layers[num_layers - 1].append(elem)
        self.conv_to_fc = 0
        output_dim = input_size * input_size * input_channels
        for i, layer in enumerate(topology_layers):
            input_dim = input_size * input_size * input_channels if i == 0 else output_dim
            output_dim = int(layer[1])
            output_layer = i == (len(topology_layers) - 1)
            self.layers.append(
                FCBlock(
                    input_dim,
                    output_dim,
                    output_act if output_layer else hidden_act,
                    dropout,
                    None if output_layer else [label_features, output_dim],
                    label_features,
                    fc_zero_init,
                    "BP" if output_layer and train_mode != "FA" else train_mode,
                )
            )

    def forward(self, x: Tensor, labels: Tensor) -> Tensor:
        """Run the DRTP network."""
        x = x.reshape(x.size(0), -1)
        for layer in self.layers:
            x = layer(x, labels, self.y)
        if x.requires_grad and (self.y is not None):
            self.y.data.copy_(F.softmax(input=x.data, dim=1) if self.apply_softmax else x.data)
        return x


class FFullyConnected(nn.Module):
    """DifferNet fully connected transform for coupling layers."""

    def __init__(
        self, size_in: int, size: int, internal_size: int | None = None, dropout: float = 0.0
    ) -> None:
        """Initialize the fully connected coupling subnetwork."""
        super().__init__()
        hidden = 2 * size if internal_size is None else internal_size
        self.d1 = nn.Dropout(p=dropout)
        self.d2 = nn.Dropout(p=dropout)
        self.d2b = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(size_in, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc2b = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, size)
        self.nl1 = nn.ReLU()
        self.nl2 = nn.ReLU()
        self.nl2b = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """Run the coupling subnetwork."""
        out = self.nl1(self.d1(self.fc1(x)))
        out = self.nl2(self.d2(self.fc2(out)))
        out = self.nl2b(self.d2b(self.fc2b(out)))
        return self.fc3(out)


class PermuteLayer(nn.Module):
    """DifferNet fixed permutation layer."""

    def __init__(self, channels: int, seed: int) -> None:
        """Initialize the fixed permutation."""
        super().__init__()
        rng = np.random.default_rng(seed)
        perm = rng.permutation(channels)
        perm_inv = np.zeros_like(perm)
        for i, p in enumerate(perm):
            perm_inv[p] = i
        self.register_buffer("perm", torch.LongTensor(perm))
        self.register_buffer("perm_inv", torch.LongTensor(perm_inv))

    def forward(self, x: Tensor, rev: bool = False) -> Tensor:
        """Permute the feature vector."""
        return x[:, self.perm_inv] if rev else x[:, self.perm]


class GlowCouplingLayer(nn.Module):
    """DifferNet Glow-style affine coupling layer."""

    def __init__(
        self, channels: int, internal_size: int = 16, dropout: float = 0.0, clamp: float = 2.0
    ) -> None:
        """Initialize the coupling layer."""
        super().__init__()
        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2
        self.clamp = clamp
        self.max_s = exp(clamp)
        self.min_s = exp(-clamp)
        self.s1 = FFullyConnected(self.split_len1, self.split_len2 * 2, internal_size, dropout)
        self.s2 = FFullyConnected(self.split_len2, self.split_len1 * 2, internal_size, dropout)

    def log_e(self, s: Tensor) -> Tensor:
        """Compute clamped log scale."""
        return self.clamp * 0.636 * torch.atan(s / self.clamp)

    def e(self, s: Tensor) -> Tensor:
        """Compute clamped scale."""
        return torch.exp(self.log_e(s))

    def forward(self, x: Tensor, rev: bool = False) -> Tensor:
        """Run affine coupling."""
        x1, x2 = x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2)
        if not rev:
            r2 = self.s2(x2)
            s2, t2 = r2[:, : self.split_len1], r2[:, self.split_len1 :]
            y1 = self.e(s2) * x1 + t2
            r1 = self.s1(y1)
            s1, t1 = r1[:, : self.split_len2], r1[:, self.split_len2 :]
            y2 = self.e(s1) * x2 + t1
        else:
            r1 = self.s1(x1)
            s1, t1 = r1[:, : self.split_len2], r1[:, self.split_len2 :]
            y2 = (x2 - t1) / self.e(s1)
            r2 = self.s2(y2)
            s2, t2 = r2[:, : self.split_len1], r2[:, self.split_len1 :]
            y1 = (x1 - t2) / self.e(s2)
        return torch.clamp(torch.cat((y1, y2), 1), -1e6, 1e6)


class NormalizingFlowHead(nn.Module):
    """DifferNet normalizing-flow head."""

    def __init__(self, input_dim: int = 384, n_coupling_blocks: int = 2) -> None:
        """Initialize a compact DifferNet flow head."""
        super().__init__()
        blocks: list[nn.Module] = []
        for k in range(n_coupling_blocks):
            blocks.append(PermuteLayer(input_dim, k))
            blocks.append(GlowCouplingLayer(input_dim))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: Tensor) -> Tensor:
        """Run the flow head."""
        for block in self.blocks:
            x = block(x)
        return x


class DifferNet(nn.Module):
    """DifferNet feature extractor plus normalizing flow."""

    def __init__(self) -> None:
        """Initialize DifferNet without downloading pretrained weights."""
        super().__init__()
        self.feature_extractor = alexnet(weights=None)
        self.n_scales = 1
        self.nf = NormalizingFlowHead(256)

    def forward(self, x: Tensor) -> Tensor:
        """Run DifferNet."""
        y_cat = []
        for s in range(self.n_scales):
            x_scaled = F.interpolate(x, size=64 // (2**s)) if s > 0 else x
            feat_s = self.feature_extractor.features(x_scaled)
            y_cat.append(torch.mean(feat_s, dim=(2, 3)))
        return self.nf(torch.cat(y_cat, dim=1))


def build_dgcrn() -> nn.Module:
    """Build a tiny DGCRN model."""
    adj = torch.eye(4)
    return DGCRN(gcn_depth=1, num_nodes=4, predefined_a=[adj, adj])


def example_input_dgcrn() -> tuple[Tensor, Tensor]:
    """Build DGCRN example inputs."""
    return torch.randn(2, 2, 4, 3), torch.randn(2, 2, 4, 2)


def build_differentiable_plasticity() -> nn.Module:
    """Build a tiny differentiable-plasticity model."""
    return PlasticNetwork()


def example_input_differentiable_plasticity() -> tuple[Tensor, Tensor, Tensor]:
    """Build differentiable-plasticity example inputs."""
    return torch.randn(1, 8), torch.zeros(1, 8), torch.zeros(8, 8)


def build_drtp() -> nn.Module:
    """Build a tiny direct-random-target-projection network."""
    return NetworkBuilder("FC_16_FC_4", 1, 8, 4, 1, "DRTP", 0.0, "relu", "none", False, "CE")


def example_input_drtp() -> tuple[Tensor, Tensor]:
    """Build DRTP example inputs."""
    return torch.randn(1, 8, 1, 1), torch.zeros(1, 4)


def build_differnet() -> nn.Module:
    """Build a compact DifferNet model."""
    return DifferNet()


def example_input_differnet() -> Tensor:
    """Build DifferNet example input."""
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES: list[tuple[str, str, str, int, str]] = [
    ("DGCRN", "build_dgcrn", "example_input_dgcrn", 2023, "CV18"),
    (
        "Differentiable Plasticity network",
        "build_differentiable_plasticity",
        "example_input_differentiable_plasticity",
        2018,
        "CV18",
    ),
    ("Direct Feedback Alignment network", "build_drtp", "example_input_drtp", 2021, "CV18"),
    ("DifferNet", "build_differnet", "example_input_differnet", 2021, "CV18"),
]
