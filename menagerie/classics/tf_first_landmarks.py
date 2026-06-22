"""TensorFlow-first research landmarks from DeepMind/Sonnet and Magenta.

Compact PyTorch reimplementations of notable architectures whose original
reference implementations were TensorFlow, TF-Slim, Sonnet, or Magenta and
which were not already covered in the TorchLens catalog under searched aliases.
All modules are random-initialized, CPU-friendly, and sized for trace/draw.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class GenerativeQueryNetwork(nn.Module):
    """Compact Generative Query Network with context aggregation and renderer."""

    def __init__(self, channels: int = 24, pose_dim: int = 7) -> None:
        """Initialize context encoder and query-conditioned renderer.

        Parameters
        ----------
        channels:
            Representation width.
        pose_dim:
            Camera/query pose feature width.
        """
        super().__init__()
        self.pose_dim = pose_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3 + pose_dim, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.query = nn.Linear(pose_dim, channels)
        self.renderer = nn.Sequential(
            nn.ConvTranspose2d(channels * 2, channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(channels, channels // 2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels // 2, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, views: Tensor) -> Tensor:
        """Render a query view from context images and poses.

        Parameters
        ----------
        views:
            Tensor of shape ``(B, V, 3 + pose_dim, H, W)``. The last view is the
            query pose; earlier views are context image-pose pairs.

        Returns
        -------
        Tensor
            Reconstructed query image ``(B, 3, H, W)``.
        """
        context = views[:, :-1]
        query_pose = views[:, -1, 3:, 0, 0]
        batch, n_context, _, height, width = context.shape
        encoded = self.encoder(context.reshape(batch * n_context, 3 + self.pose_dim, height, width))
        encoded = encoded.reshape(
            batch, n_context, encoded.shape[1], encoded.shape[2], encoded.shape[3]
        )
        scene = encoded.sum(dim=1)
        query = self.query(query_pose).unsqueeze(-1).unsqueeze(-1).expand_as(scene)
        return self.renderer(torch.cat([scene, query], dim=1))


class GraphNetworkBlock(nn.Module):
    """DeepMind Graph Nets-style edge, node, and global update block."""

    def __init__(self, node_dim: int = 8, edge_dim: int = 6, global_dim: int = 8) -> None:
        """Initialize MLP updates and a fixed toy graph.

        Parameters
        ----------
        node_dim:
            Node feature width.
        edge_dim:
            Edge feature width.
        global_dim:
            Global graph feature width.
        """
        super().__init__()
        senders = torch.tensor([0, 1, 2, 3, 0, 2], dtype=torch.long)
        receivers = torch.tensor([1, 2, 3, 0, 2, 0], dtype=torch.long)
        self.register_buffer("senders", senders)
        self.register_buffer("receivers", receivers)
        self.edge_attr = nn.Parameter(torch.randn(senders.numel(), edge_dim) * 0.05)
        self.global_attr = nn.Parameter(torch.zeros(global_dim))
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim + global_dim, 32),
            nn.ReLU(),
            nn.Linear(32, edge_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim + global_dim, 32),
            nn.ReLU(),
            nn.Linear(32, node_dim),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(global_dim + node_dim + edge_dim, 32),
            nn.ReLU(),
            nn.Linear(32, global_dim),
        )

    def forward(self, nodes: Tensor) -> Tensor:
        """Apply one edge-node-global graph-network update.

        Parameters
        ----------
        nodes:
            Node features ``(N, node_dim)`` for the fixed graph.

        Returns
        -------
        Tensor
            Updated node features.
        """
        global_b = self.global_attr.expand(self.senders.numel(), -1)
        edge_in = torch.cat(
            [nodes[self.senders], nodes[self.receivers], self.edge_attr, global_b],
            dim=-1,
        )
        edges = self.edge_mlp(edge_in)
        aggregated = nodes.new_zeros(nodes.shape[0], edges.shape[-1])
        aggregated.index_add_(0, self.receivers, edges)
        node_global = self.global_attr.expand(nodes.shape[0], -1)
        node_updates = self.node_mlp(torch.cat([nodes, aggregated, node_global], dim=-1))
        graph_summary = torch.cat(
            [self.global_attr, node_updates.mean(dim=0), edges.mean(dim=0)],
            dim=-1,
        )
        global_update = self.global_mlp(graph_summary)
        return node_updates + global_update[: node_updates.shape[-1]]


class SketchRNN(nn.Module):
    """Magenta Sketch-RNN variational encoder-decoder for vector strokes."""

    def __init__(self, stroke_dim: int = 5, hidden: int = 32, latent: int = 16) -> None:
        """Initialize bidirectional encoder and autoregressive decoder.

        Parameters
        ----------
        stroke_dim:
            Stroke feature width ``dx, dy, pen_down, pen_up, eos``.
        hidden:
            Recurrent hidden width.
        latent:
            Latent code width.
        """
        super().__init__()
        self.encoder = nn.LSTM(stroke_dim, hidden, batch_first=True, bidirectional=True)
        self.to_mu = nn.Linear(hidden * 2, latent)
        self.to_logvar = nn.Linear(hidden * 2, latent)
        self.decoder = nn.LSTM(stroke_dim + latent, hidden, batch_first=True)
        self.mixture = nn.Linear(hidden, 6 * 5)
        self.pen = nn.Linear(hidden, 3)

    def forward(self, strokes: Tensor) -> Tensor:
        """Encode strokes and decode mixture-density pen outputs.

        Parameters
        ----------
        strokes:
            Stroke sequence ``(B, T, 5)``.

        Returns
        -------
        Tensor
            Concatenated Gaussian-mixture parameters and pen logits.
        """
        enc, _ = self.encoder(strokes)
        summary = enc[:, -1]
        mu = self.to_mu(summary)
        logvar = self.to_logvar(summary)
        z = mu + torch.zeros_like(logvar)
        dec_in = torch.cat([strokes, z.unsqueeze(1).expand(-1, strokes.shape[1], -1)], dim=-1)
        dec, _ = self.decoder(dec_in)
        return torch.cat([self.mixture(dec), self.pen(dec)], dim=-1)


class MusicVAE(nn.Module):
    """Magenta MusicVAE with a conductor hierarchy over bars."""

    def __init__(self, vocab: int = 64, hidden: int = 32, latent: int = 16, bars: int = 4) -> None:
        """Initialize encoder, latent bottleneck, conductor, and decoder.

        Parameters
        ----------
        vocab:
            Symbol vocabulary size.
        hidden:
            Recurrent hidden width.
        latent:
            Latent code width.
        bars:
            Number of conductor segments.
        """
        super().__init__()
        self.bars = bars
        self.embed = nn.Embedding(vocab, hidden)
        self.encoder = nn.LSTM(hidden, hidden, batch_first=True, bidirectional=True)
        self.to_mu = nn.Linear(hidden * 2, latent)
        self.to_logvar = nn.Linear(hidden * 2, latent)
        self.conductor = nn.LSTM(latent, hidden, batch_first=True)
        self.decoder = nn.LSTM(hidden * 2, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, tokens: Tensor) -> Tensor:
        """Decode a note sequence through hierarchical latent segments.

        Parameters
        ----------
        tokens:
            Integer note/event ids ``(B, T)``.

        Returns
        -------
        Tensor
            Per-step vocabulary logits.
        """
        embedded = self.embed(tokens)
        enc, _ = self.encoder(embedded)
        mu = self.to_mu(enc[:, -1])
        logvar = self.to_logvar(enc[:, -1])
        z = mu + torch.zeros_like(logvar)
        conductor_in = z.unsqueeze(1).expand(-1, self.bars, -1)
        bar_codes, _ = self.conductor(conductor_in)
        repeats = max(tokens.shape[1] // self.bars, 1)
        bar_context = bar_codes.repeat_interleave(repeats, dim=1)[:, : tokens.shape[1]]
        dec, _ = self.decoder(torch.cat([embedded, bar_context], dim=-1))
        return self.head(dec)


class Coconet(nn.Module):
    """Magenta Coconet-style masked chorale infilling with dilated convolutions."""

    def __init__(self, pitches: int = 48, channels: int = 32, layers: int = 4) -> None:
        """Initialize voice/pitch embedding and dilated convolutional stack.

        Parameters
        ----------
        pitches:
            Pitch vocabulary size.
        channels:
            Hidden channel width.
        layers:
            Number of dilated residual convolution blocks.
        """
        super().__init__()
        self.embed = nn.Embedding(pitches, channels)
        self.blocks = nn.ModuleList(
            [
                nn.Conv2d(channels, channels, 3, padding=2**idx, dilation=2**idx)
                for idx in range(layers)
            ]
        )
        self.norms = nn.ModuleList([nn.BatchNorm2d(channels) for _ in range(layers)])
        self.head = nn.Conv2d(channels, pitches, 1)

    def forward(self, chorale: Tensor) -> Tensor:
        """Predict masked pitches for four-part chorale grids.

        Parameters
        ----------
        chorale:
            Integer pitch ids ``(B, voices, time)``.

        Returns
        -------
        Tensor
            Pitch logits ``(B, pitches, voices, time)``.
        """
        hidden = self.embed(chorale).permute(0, 3, 1, 2)
        for conv, norm in zip(self.blocks, self.norms):
            hidden = hidden + F.relu(norm(conv(hidden)))
        return self.head(hidden)


class NSynthWaveNetAutoencoder(nn.Module):
    """Magenta NSynth-style WaveNet autoencoder with temporal latent codes."""

    def __init__(self, channels: int = 32, latent: int = 16, layers: int = 4) -> None:
        """Initialize convolutional encoder and dilated causal decoder.

        Parameters
        ----------
        channels:
            Hidden channel width.
        latent:
            Bottleneck width.
        layers:
            Number of dilated decoder layers.
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, latent, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.pre = nn.Conv1d(latent, channels, 1)
        self.filters = nn.ModuleList(
            [
                nn.Conv1d(channels, channels * 2, 3, padding=2**idx, dilation=2**idx)
                for idx in range(layers)
            ]
        )
        self.post = nn.Conv1d(channels, 1, 1)

    def forward(self, audio: Tensor) -> Tensor:
        """Reconstruct audio from a downsampled latent sequence.

        Parameters
        ----------
        audio:
            Mono waveform ``(B, 1, T)``.

        Returns
        -------
        Tensor
            Reconstructed waveform with the input length.
        """
        latent = self.encoder(audio)
        hidden = F.interpolate(self.pre(latent), size=audio.shape[-1], mode="linear")
        for layer in self.filters:
            gate, value = layer(hidden)[..., : hidden.shape[-1]].chunk(2, dim=1)
            hidden = hidden + torch.tanh(value) * torch.sigmoid(gate)
        return self.post(hidden)


def build_gqn() -> nn.Module:
    """Build a compact Generative Query Network.

    Returns
    -------
    nn.Module
        Random-initialized GQN.
    """
    return GenerativeQueryNetwork()


def example_input_gqn() -> Tensor:
    """Create GQN context/query view tensor.

    Returns
    -------
    Tensor
        Tensor of shape ``(1, 3, 10, 16, 16)``.
    """
    return torch.randn(1, 3, 10, 16, 16)


def build_graph_network() -> nn.Module:
    """Build a compact Graph Nets interaction block.

    Returns
    -------
    nn.Module
        Random-initialized graph network block.
    """
    return GraphNetworkBlock()


def example_input_graph_network() -> Tensor:
    """Create node features for the fixed graph.

    Returns
    -------
    Tensor
        Node feature tensor ``(4, 8)``.
    """
    return torch.randn(4, 8)


def build_sketch_rnn() -> nn.Module:
    """Build a compact Sketch-RNN.

    Returns
    -------
    nn.Module
        Random-initialized Sketch-RNN.
    """
    return SketchRNN()


def example_input_sketch_rnn() -> Tensor:
    """Create stroke features for Sketch-RNN.

    Returns
    -------
    Tensor
        Stroke tensor ``(1, 12, 5)``.
    """
    return torch.randn(1, 12, 5)


def build_music_vae() -> nn.Module:
    """Build a compact MusicVAE.

    Returns
    -------
    nn.Module
        Random-initialized MusicVAE.
    """
    return MusicVAE()


def example_input_music_vae() -> Tensor:
    """Create symbolic music ids for MusicVAE.

    Returns
    -------
    Tensor
        Integer token tensor ``(1, 16)``.
    """
    return torch.randint(0, 64, (1, 16), dtype=torch.long)


def build_coconet() -> nn.Module:
    """Build a compact Coconet.

    Returns
    -------
    nn.Module
        Random-initialized Coconet.
    """
    return Coconet()


def example_input_coconet() -> Tensor:
    """Create four-voice chorale pitch ids.

    Returns
    -------
    Tensor
        Integer chorale tensor ``(1, 4, 16)``.
    """
    return torch.randint(0, 48, (1, 4, 16), dtype=torch.long)


def build_nsynth_wavenet_autoencoder() -> nn.Module:
    """Build a compact NSynth WaveNet autoencoder.

    Returns
    -------
    nn.Module
        Random-initialized NSynth WaveNet autoencoder.
    """
    return NSynthWaveNetAutoencoder()


def example_input_nsynth_wavenet_autoencoder() -> Tensor:
    """Create mono audio for the NSynth WaveNet autoencoder.

    Returns
    -------
    Tensor
        Audio tensor ``(1, 1, 64)``.
    """
    return torch.randn(1, 1, 64)


MENAGERIE_ENTRIES = [
    (
        "Generative Query Network (GQN)",
        "build_gqn",
        "example_input_gqn",
        "2018",
        "DC",
    ),
    (
        "DeepMind Graph Nets interaction block",
        "build_graph_network",
        "example_input_graph_network",
        "2018",
        "DC",
    ),
    (
        "Sketch-RNN (Magenta vector drawing VAE)",
        "build_sketch_rnn",
        "example_input_sketch_rnn",
        "2017",
        "DC",
    ),
    (
        "MusicVAE (Magenta hierarchical music VAE)",
        "build_music_vae",
        "example_input_music_vae",
        "2018",
        "DC",
    ),
    (
        "Coconet (Magenta counterpoint by convolution)",
        "build_coconet",
        "example_input_coconet",
        "2017",
        "DC",
    ),
    (
        "NSynth WaveNet autoencoder",
        "build_nsynth_wavenet_autoencoder",
        "example_input_nsynth_wavenet_autoencoder",
        "2017",
        "DC",
    ),
]
