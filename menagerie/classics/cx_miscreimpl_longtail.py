"""Compact dependency-free classics for CX long-tail dependency-gated models.

Sources checked for architectural primitives:

MoViNet: Kondratyuk et al. 2021, arXiv:2103.11511.
AASIST: Jung et al. 2021/2022, arXiv:2110.01200 and clovaai/aasist.
BigVGAN: Lee et al. 2022/2023, arXiv:2206.04658 and NVIDIA/BigVGAN.
Bonito: Oxford Nanopore Technologies research basecaller, nanoporetech/bonito.
BitNet/BitLinear: Wang et al. 2023, arXiv:2310.11453.
Bayes by Backprop: Blundell et al. 2015, arXiv:1505.05424.
Borzoi: Linder et al. 2025, RNA-seq genome model, Calico/Borzoi.
ALIGNN: Choudhary and DeCost 2021, arXiv:2106.01829.
Avalanche examples: continual-learning benchmark MLP/CNN patterns.

All implementations are random-init, CPU-friendly reductions that preserve the
distinctive computation graph instead of requiring their original packages.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcite3d(nn.Module):
    """Squeeze-and-excitation gate for compact MoViNet-style 3D blocks."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        """Initialize the channel gate.

        Parameters
        ----------
        channels:
            Number of input and output channels.
        reduction:
            Channel reduction factor for the hidden gate.
        """

        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Conv3d(channels, hidden, 1)
        self.fc2 = nn.Conv3d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply global 3D pooling and sigmoid channel reweighting."""

        scale = F.adaptive_avg_pool3d(x, 1)
        scale = F.silu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class MoViNetBlock(nn.Module):
    """Mobile inverted 3D bottleneck with depthwise temporal-spatial filtering."""

    def __init__(self, in_ch: int, out_ch: int, stride: tuple[int, int, int]) -> None:
        """Initialize a compact MoViNet block.

        Parameters
        ----------
        in_ch:
            Input channel count.
        out_ch:
            Output channel count.
        stride:
            Temporal, height, and width stride.
        """

        super().__init__()
        mid = out_ch * 2
        self.use_skip = in_ch == out_ch and stride == (1, 1, 1)
        self.expand = nn.Conv3d(in_ch, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid)
        self.dw = nn.Conv3d(mid, mid, 3, stride=stride, padding=1, groups=mid, bias=False)
        self.bn2 = nn.BatchNorm3d(mid)
        self.se = SqueezeExcite3d(mid)
        self.project = nn.Conv3d(mid, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run inverted 3D bottleneck and optional residual connection."""

        y = F.silu(self.bn1(self.expand(x)))
        y = F.silu(self.bn2(self.dw(y)))
        y = self.se(y)
        y = self.bn3(self.project(y))
        if self.use_skip:
            y = y + x
        return y


class CompactMoViNet(nn.Module):
    """Small MoViNet-A style streaming video CNN."""

    def __init__(self, num_classes: int = 8) -> None:
        """Initialize the compact video classifier."""

        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(3, 8, (3, 3, 3), stride=(1, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(8),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            MoViNetBlock(8, 12, (1, 1, 1)),
            MoViNetBlock(12, 16, (1, 2, 2)),
            MoViNetBlock(16, 24, (2, 2, 2)),
            MoViNetBlock(24, 24, (1, 1, 1)),
        )
        self.head = nn.Sequential(nn.Conv3d(24, 48, 1), nn.SiLU(), nn.AdaptiveAvgPool3d(1))
        self.classifier = nn.Linear(48, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a short video tensor of shape ``(B, C, T, H, W)``."""

        y = self.blocks(self.stem(x))
        y = self.head(y).flatten(1)
        return self.classifier(y)


class AASISTGraphAttention(nn.Module):
    """Heterogeneous graph attention over temporal, spectral, and stack nodes."""

    def __init__(self, dim: int) -> None:
        """Initialize graph-attention projections."""

        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        """Run scaled dot-product graph attention over node embeddings."""

        q = self.q(nodes)
        k = self.k(nodes)
        v = self.v(nodes)
        attn = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), dim=-1)
        return self.out(attn @ v)


class CompactAASIST(nn.Module):
    """AASIST-like spoofing detector with Res2Net front end and graph attention."""

    def __init__(self, num_classes: int = 2) -> None:
        """Initialize the compact detector."""

        super().__init__()
        self.front = nn.Sequential(
            nn.Conv1d(1, 16, 7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.SiLU(),
            nn.Conv1d(16, 32, 3, padding=1, groups=4),
            nn.BatchNorm1d(32),
            nn.SiLU(),
        )
        self.temporal = nn.Conv1d(32, 32, 5, padding=2, groups=4)
        self.spectral = nn.Conv1d(32, 32, 1)
        self.stack_token = nn.Parameter(torch.zeros(1, 1, 32))
        self.gat1 = AASISTGraphAttention(32)
        self.gat2 = AASISTGraphAttention(32)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """Classify waveform tensors of shape ``(B, T)``."""

        x = self.front(wav.unsqueeze(1))
        temporal_nodes = F.adaptive_avg_pool1d(self.temporal(x), 6).transpose(1, 2)
        spectral_nodes = F.adaptive_max_pool1d(self.spectral(x), 6).transpose(1, 2)
        stack = self.stack_token.expand(wav.shape[0], -1, -1)
        nodes = torch.cat([temporal_nodes, spectral_nodes, stack], dim=1)
        nodes = nodes + self.gat1(nodes)
        nodes = nodes + self.gat2(nodes)
        pooled = torch.cat([nodes.mean(dim=1), nodes.amax(dim=1)], dim=-1)
        return self.classifier(pooled)


class SnakeBeta(nn.Module):
    """BigVGAN periodic SnakeBeta activation."""

    def __init__(self, channels: int) -> None:
        """Initialize learned periodic parameters."""

        super().__init__()
        self.alpha = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``x + sin(alpha*x)^2 / beta`` with channel broadcasting."""

        alpha = self.alpha.view(1, -1, 1).abs() + 1e-4
        beta = self.beta.view(1, -1, 1).abs() + 1e-4
        return x + torch.sin(alpha * x).pow(2) / beta


class AMPBlock(nn.Module):
    """Anti-aliased multi-periodicity residual block from BigVGAN."""

    def __init__(self, channels: int) -> None:
        """Initialize dilated periodic residual convolutions."""

        super().__init__()
        self.act1 = SnakeBeta(channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.act2 = SnakeBeta(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=3, dilation=3)
        self.lowpass = nn.AvgPool1d(3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run periodic residual block with a simple anti-alias low-pass filter."""

        y = self.conv1(self.lowpass(self.act1(x)))
        y = self.conv2(self.lowpass(self.act2(y)))
        return x + y


class CompactBigVGAN(nn.Module):
    """Compact BigVGAN vocoder generator."""

    def __init__(self, mel_bins: int = 80) -> None:
        """Initialize the vocoder."""

        super().__init__()
        self.pre = nn.Conv1d(mel_bins, 32, 7, padding=3)
        self.up1 = nn.ConvTranspose1d(32, 24, 8, stride=4, padding=2)
        self.amp1 = AMPBlock(24)
        self.up2 = nn.ConvTranspose1d(24, 16, 4, stride=2, padding=1)
        self.amp2 = AMPBlock(16)
        self.post = nn.Sequential(SnakeBeta(16), nn.Conv1d(16, 1, 7, padding=3), nn.Tanh())

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Generate waveform from a mel spectrogram ``(B, 80, T)``."""

        x = self.pre(mel)
        x = self.amp1(F.silu(self.up1(x)))
        x = self.amp2(F.silu(self.up2(x)))
        return self.post(x)


class BonitoBlock(nn.Module):
    """Depthwise-separable convolutional block used in compact Bonito."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        """Initialize the basecaller block."""

        super().__init__()
        self.dw = nn.Conv1d(
            channels, channels, kernel_size, padding=kernel_size // 2, groups=channels
        )
        self.pw = nn.Conv1d(channels, channels, 1)
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run residual separable convolution."""

        return x + F.silu(self.bn(self.pw(self.dw(x))))


class CompactBonito(nn.Module):
    """Bonito-like nanopore CTC basecaller."""

    def __init__(self, alphabet: int = 5) -> None:
        """Initialize the basecaller."""

        super().__init__()
        self.stem = nn.Conv1d(1, 32, 5, stride=2, padding=2)
        self.blocks = nn.Sequential(BonitoBlock(32, 5), BonitoBlock(32, 7), BonitoBlock(32, 9))
        self.rnn = nn.LSTM(32, 24, num_layers=1, bidirectional=True, batch_first=True)
        self.proj = nn.Linear(48, alphabet)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Return CTC logits from raw nanopore signal ``(B, 1, T)``."""

        x = self.blocks(F.silu(self.stem(signal))).transpose(1, 2)
        y, _ = self.rnn(x)
        return self.proj(y)


class BitLinear(nn.Linear):
    """BitNet-style ternary-weight linear layer with activation quantization."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS-normalized activation quantization and ternary weights."""

        scale_x = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
        x_q = (x / scale_x * 127).round().clamp(-128, 127) / 127 * scale_x
        x_ste = x + (x_q - x).detach()
        scale_w = self.weight.abs().mean().clamp_min(1e-5)
        w_q = self.weight.sign() * scale_w
        w_ste = self.weight + (w_q - self.weight).detach()
        return F.linear(x_ste, w_ste, self.bias)


class CompactBitNetMLP(nn.Module):
    """Small BitLinear network for the BitLinear_BitNet target."""

    def __init__(self) -> None:
        """Initialize the quantized MLP."""

        super().__init__()
        self.net = nn.Sequential(BitLinear(32, 48), nn.SiLU(), BitLinear(48, 16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the BitLinear MLP."""

        return self.net(x)


class BayesianLinearBBB(nn.Module):
    """Bayes-by-Backprop linear layer with Gaussian posterior parameters."""

    def __init__(self, in_features: int, out_features: int) -> None:
        """Initialize variational weight and bias parameters."""

        super().__init__()
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features).normal_(0, 0.05))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features).fill_(-4.0))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.empty(out_features).fill_(-4.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sample weights with reparameterization and apply a linear map."""

        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)
        weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
        bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
        return F.linear(x, weight, bias)


class CompactBayesByBackprop(nn.Module):
    """Two-layer Bayes-by-Backprop classifier."""

    def __init__(self) -> None:
        """Initialize the compact Bayesian neural network."""

        super().__init__()
        self.fc1 = BayesianLinearBBB(20, 32)
        self.fc2 = BayesianLinearBBB(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the variational MLP."""

        return self.fc2(F.relu(self.fc1(x)))


class BorzoiBlock(nn.Module):
    """Conv-attention block for a compact Borzoi-style genomic model."""

    def __init__(self, channels: int) -> None:
        """Initialize the sequence block."""

        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 5, padding=2, groups=channels)
        self.point = nn.Conv1d(channels, channels, 1)
        self.attn = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run convolutional mixing followed by self-attention."""

        y = x + self.point(self.conv(x))
        seq = y.transpose(1, 2)
        attn, _ = self.attn(seq, seq, seq, need_weights=False)
        seq = seq + attn
        seq = seq + self.ff(seq)
        return seq.transpose(1, 2)


class CompactBorzoi(nn.Module):
    """Borzoi-like DNA sequence-to-coverage model."""

    def __init__(self, tracks: int = 8) -> None:
        """Initialize the compact genomic model."""

        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(4, 32, 15, padding=7), nn.GELU(), nn.MaxPool1d(2))
        self.blocks = nn.Sequential(BorzoiBlock(32), BorzoiBlock(32))
        self.unet_skip = nn.Conv1d(32, 32, 1)
        self.head = nn.Sequential(
            nn.Conv1d(32, 32, 1), nn.GELU(), nn.Conv1d(32, tracks, 1), nn.Softplus()
        )

    def forward(self, dna: torch.Tensor) -> torch.Tensor:
        """Predict coverage tracks from one-hot DNA ``(B, L, 4)``."""

        x = self.stem(dna.transpose(1, 2))
        skip = self.unet_skip(x)
        x = self.blocks(x)
        return self.head(x + skip).transpose(1, 2)


class ALIGNNLayer(nn.Module):
    """One ALIGNN update over atom and line-graph features."""

    def __init__(self, dim: int) -> None:
        """Initialize atom, bond, and angle message functions."""

        super().__init__()
        self.edge_msg = nn.Linear(dim * 3, dim)
        self.atom_msg = nn.Linear(dim * 2, dim)
        self.norm_atom = nn.LayerNorm(dim)
        self.norm_edge = nn.LayerNorm(dim)

    def forward(
        self,
        atom: torch.Tensor,
        edge: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        line_adj: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update edge features from line graph and atom features from edges."""

        edge_neigh = line_adj @ edge
        edge_up = F.silu(
            self.edge_msg(torch.cat([edge, edge_neigh, atom[src] + atom[dst]], dim=-1))
        )
        edge = self.norm_edge(edge + edge_up)
        atom_acc = torch.zeros_like(atom)
        atom_acc.index_add_(0, dst, edge)
        atom_up = F.silu(self.atom_msg(torch.cat([atom, atom_acc], dim=-1)))
        atom = self.norm_atom(atom + atom_up)
        return atom, edge


class CompactALIGNN(nn.Module):
    """Atomistic Line Graph Neural Network for crystal-property prediction."""

    def __init__(self) -> None:
        """Initialize the compact ALIGNN."""

        super().__init__()
        self.atom_embed = nn.Linear(8, 24)
        self.edge_embed = nn.Linear(4, 24)
        self.layers = nn.ModuleList([ALIGNNLayer(24), ALIGNNLayer(24)])
        self.readout = nn.Linear(24, 1)

    def forward(
        self,
        atom_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        line_adj: torch.Tensor,
    ) -> torch.Tensor:
        """Predict a scalar from atom, bond, and line-graph tensors."""

        atom = self.atom_embed(atom_feats)
        edge = self.edge_embed(edge_feats)
        for layer in self.layers:
            atom, edge = layer(atom, edge, src, dst, line_adj)
        return self.readout(atom.mean(dim=0, keepdim=True))


class OccupancyDecoder(nn.Module):
    """Original occupancy-network style conditional ResNet MLP decoder."""

    def __init__(self, code_dim: int = 16, hidden_dim: int = 32) -> None:
        """Initialize the decoder."""

        super().__init__()
        self.fc_p = nn.Linear(3, hidden_dim)
        self.fc_c = nn.Linear(code_dim, hidden_dim)
        self.blocks = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(3)])
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, points: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        """Decode occupancy logits from query points and latent code."""

        x = self.fc_p(points) + self.fc_c(code).unsqueeze(1)
        for block in self.blocks:
            x = x + F.relu(block(F.relu(x)))
        return self.out(F.relu(x)).squeeze(-1)


class AvalancheMTSimpleMLP(nn.Module):
    """Avalanche-style multi-task MLP with a task-conditioned head bank."""

    def __init__(self, num_tasks: int = 3) -> None:
        """Initialize shared trunk and per-task heads."""

        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(20, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(32, 5) for _ in range(num_tasks)])

    def forward(self, x: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        """Run all heads and select by integer task id."""

        h = self.trunk(x)
        stacked = torch.stack([head(h) for head in self.heads], dim=1)
        gather_id = task_id.view(-1, 1, 1).expand(-1, 1, stacked.shape[-1])
        return stacked.gather(1, gather_id).squeeze(1)


def build_movinet_a0_pytorch() -> nn.Module:
    """Build a compact MoViNet-A0-style classifier."""

    return CompactMoViNet()


def example_movinet_input() -> torch.Tensor:
    """Return a small video input."""

    return torch.randn(1, 3, 8, 32, 32)


def build_aasist() -> nn.Module:
    """Build a compact AASIST detector."""

    return CompactAASIST()


def example_aasist_input() -> torch.Tensor:
    """Return a short waveform input."""

    return torch.randn(1, 4096)


def build_bigvgan() -> nn.Module:
    """Build a compact BigVGAN generator."""

    return CompactBigVGAN()


def example_bigvgan_input() -> torch.Tensor:
    """Return a small mel spectrogram."""

    return torch.randn(1, 80, 12)


def build_bonito() -> nn.Module:
    """Build a compact Bonito basecaller."""

    return CompactBonito()


def example_bonito_input() -> torch.Tensor:
    """Return a short raw nanopore signal."""

    return torch.randn(1, 1, 256)


def build_bitlinear_bitnet() -> nn.Module:
    """Build a compact BitLinear network."""

    return CompactBitNetMLP()


def example_bitlinear_input() -> torch.Tensor:
    """Return a token-feature batch for BitLinear."""

    return torch.randn(1, 10, 32)


def build_bayesian_nn_bbb() -> nn.Module:
    """Build a compact Bayes-by-Backprop network."""

    return CompactBayesByBackprop()


def example_bayesian_input() -> torch.Tensor:
    """Return a tabular feature batch."""

    return torch.randn(2, 20)


def build_borzoi() -> nn.Module:
    """Build a compact Borzoi-like genomic model."""

    return CompactBorzoi()


def example_borzoi_input() -> torch.Tensor:
    """Return a small one-hot-like DNA sequence."""

    idx = torch.randint(0, 4, (1, 256))
    return F.one_hot(idx, num_classes=4).float()


def build_alignn() -> nn.Module:
    """Build a compact ALIGNN graph network."""

    return CompactALIGNN()


def example_alignn_input() -> list[torch.Tensor]:
    """Return atom, edge, incidence, and line-graph tensors."""

    atom = torch.randn(5, 8)
    edge = torch.randn(8, 4)
    src = torch.tensor([0, 1, 2, 3, 4, 0, 2, 1], dtype=torch.long)
    dst = torch.tensor([1, 2, 3, 4, 0, 2, 4, 3], dtype=torch.long)
    line_adj = torch.eye(8).roll(1, dims=0) + torch.eye(8).roll(-1, dims=0)
    line_adj = line_adj / line_adj.sum(dim=-1, keepdim=True)
    return [atom, edge, src, dst, line_adj]


def build_occupancy_network_decoder() -> nn.Module:
    """Build an occupancy-network decoder."""

    return OccupancyDecoder()


def example_occupancy_input() -> list[torch.Tensor]:
    """Return query points and latent code."""

    return [torch.randn(1, 32, 3) * 0.4, torch.randn(1, 16)]


def build_avalanche_mt_mlp() -> nn.Module:
    """Build a compact Avalanche multi-task MLP."""

    return AvalancheMTSimpleMLP()


def example_avalanche_input() -> list[torch.Tensor]:
    """Return features and task IDs."""

    return [torch.randn(2, 20), torch.tensor([0, 2], dtype=torch.long)]


MENAGERIE_ENTRIES = [
    (
        "MoViNet-A0 PyTorch compact",
        "build_movinet_a0_pytorch",
        "example_movinet_input",
        "2021",
        "DC",
    ),
    (
        "AASIST compact graph-attention anti-spoofing",
        "build_aasist",
        "example_aasist_input",
        "2021",
        "DC",
    ),
    ("BigVGAN compact periodic vocoder", "build_bigvgan", "example_bigvgan_input", "2022", "DC"),
    (
        "Bonito compact nanopore CTC basecaller",
        "build_bonito",
        "example_bonito_input",
        "2019",
        "DC",
    ),
    (
        "BitLinear BitNet compact layer",
        "build_bitlinear_bitnet",
        "example_bitlinear_input",
        "2023",
        "DC",
    ),
    (
        "Bayesian NN Bayes-by-Backprop compact",
        "build_bayesian_nn_bbb",
        "example_bayesian_input",
        "2015",
        "DC",
    ),
    ("Borzoi compact genomic sequence model", "build_borzoi", "example_borzoi_input", "2025", "DC"),
    (
        "ALIGNN compact atomistic line graph network",
        "build_alignn",
        "example_alignn_input",
        "2021",
        "DC",
    ),
    (
        "Occupancy Network decoder compact",
        "build_occupancy_network_decoder",
        "example_occupancy_input",
        "2019",
        "DC",
    ),
    (
        "Avalanche multi-task simple MLP compact",
        "build_avalanche_mt_mlp",
        "example_avalanche_input",
        "2021",
        "DC",
    ),
]
