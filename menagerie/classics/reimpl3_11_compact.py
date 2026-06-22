"""Compact classics for dependency-gated reimplementation shard 3.11.

These random-initialized PyTorch reconstructions keep each target family's
load-bearing primitive while shrinking dimensions and inputs for base
TorchLens rendering.  Source architecture notes checked for this shard include
Whisper (Radford et al., 2022), YOLOX (Ge et al., 2021), YOLOv6 (Li et al.,
2022), RetNet (Sun et al., 2023), xLSTM (Beck et al., 2024), WaveGAN (Donahue
et al., 2019), ParticleNet (Qu and Gouskos, 2019), UFold (Fu et al., 2022),
USAD (Audibert et al., 2020), and WIRE (Saragadam et al., 2023).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """Convolution, batch normalization, and SiLU activation block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """Initialize the block.

        Parameters
        ----------
        in_channels:
            Number of input feature channels.
        out_channels:
            Number of output feature channels.
        stride:
            Spatial convolution stride.
        """

        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolutional feature extraction.

        Parameters
        ----------
        x:
            Image feature tensor.

        Returns
        -------
        torch.Tensor
            Activated feature tensor.
        """

        return F.silu(self.bn(self.conv(x)))


class CSPBlock(nn.Module):
    """Cross-stage partial block used by compact YOLOX-style backbones."""

    def __init__(self, channels: int) -> None:
        """Initialize branch projections and merge convolution.

        Parameters
        ----------
        channels:
            Number of feature channels.
        """

        super().__init__()
        half = channels // 2
        self.left = nn.Conv2d(half, half, 1)
        self.right = nn.Sequential(ConvBNAct(half, half), ConvBNAct(half, half))
        self.merge = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CSP split-transform-concatenate fusion.

        Parameters
        ----------
        x:
            Feature tensor.

        Returns
        -------
        torch.Tensor
            Fused feature tensor.
        """

        left, right = x.chunk(2, dim=1)
        return F.silu(self.merge(torch.cat((self.left(left), self.right(right)), dim=1)))


class RepBlock(nn.Module):
    """Compact RepVGG-style block for YOLOv6 EfficientRep/RepPAN."""

    def __init__(self, channels: int) -> None:
        """Initialize parallel 3x3 and 1x1 branches.

        Parameters
        ----------
        channels:
            Number of channels processed by the block.
        """

        super().__init__()
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.conv1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply re-parameterizable residual convolution branches.

        Parameters
        ----------
        x:
            Feature tensor.

        Returns
        -------
        torch.Tensor
            Activated feature tensor.
        """

        return F.silu(self.bn(self.conv3(x) + self.conv1(x)))


class DecoupledHead(nn.Module):
    """Anchor-free decoupled detection head."""

    def __init__(self, channels: int, classes: int = 5) -> None:
        """Initialize classification, regression, and objectness branches.

        Parameters
        ----------
        channels:
            Input feature channel count.
        classes:
            Number of object classes.
        """

        super().__init__()
        self.cls = nn.Sequential(ConvBNAct(channels, channels), nn.Conv2d(channels, classes, 1))
        self.reg = nn.Sequential(ConvBNAct(channels, channels), nn.Conv2d(channels, 4, 1))
        self.obj = nn.Sequential(ConvBNAct(channels, channels), nn.Conv2d(channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class logits, box offsets, and objectness.

        Parameters
        ----------
        x:
            Feature map at one detection scale.

        Returns
        -------
        torch.Tensor
            Concatenated detection logits.
        """

        return torch.cat((self.reg(x), self.obj(x), self.cls(x)), dim=1)


class TinyYOLOX(nn.Module):
    """YOLOX-style CSPDarknet, PAFPN, and decoupled anchor-free detector."""

    def __init__(self, width: int = 12, depth: int = 1) -> None:
        """Initialize compact YOLOX detector.

        Parameters
        ----------
        width:
            Base channel width.
        depth:
            Number of repeated CSP blocks at each stage.
        """

        super().__init__()
        self.stem = ConvBNAct(3, width, stride=2)
        self.stage2 = nn.Sequential(
            ConvBNAct(width, width * 2, stride=2), *[CSPBlock(width * 2) for _ in range(depth)]
        )
        self.stage3 = nn.Sequential(
            ConvBNAct(width * 2, width * 4, stride=2), *[CSPBlock(width * 4) for _ in range(depth)]
        )
        self.stage4 = nn.Sequential(ConvBNAct(width * 4, width * 8, stride=2), CSPBlock(width * 8))
        self.spp = nn.ModuleList([nn.MaxPool2d(k, stride=1, padding=k // 2) for k in (3, 5)])
        self.lat4 = nn.Conv2d(width * 8 * 3, width * 4, 1)
        self.pan3 = ConvBNAct(width * 8, width * 4)
        self.down3 = ConvBNAct(width * 4, width * 4, stride=2)
        self.pan4 = ConvBNAct(width * 12, width * 8)
        self.head3 = DecoupledHead(width * 4)
        self.head4 = DecoupledHead(width * 8)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run two-scale YOLOX detection.

        Parameters
        ----------
        x:
            RGB image tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Detection tensors for medium and coarse scales.
        """

        p2 = self.stage2(self.stem(x))
        p3 = self.stage3(p2)
        p4 = self.stage4(p3)
        p4 = torch.cat((p4, *(pool(p4) for pool in self.spp)), dim=1)
        top = F.interpolate(self.lat4(p4), size=p3.shape[-2:], mode="nearest")
        p3_out = self.pan3(torch.cat((top, p3), dim=1))
        p4_out = self.pan4(torch.cat((self.down3(p3_out), p4[:, : p3_out.size(1) * 2]), dim=1))
        return self.head3(p3_out), self.head4(p4_out)


class TinyYOLOv6(nn.Module):
    """YOLOv6-style EfficientRep, RepPAN, and efficient decoupled head."""

    def __init__(self, width: int = 12, depth: int = 1) -> None:
        """Initialize compact YOLOv6 detector.

        Parameters
        ----------
        width:
            Base channel width.
        depth:
            Number of repeated RepBlocks.
        """

        super().__init__()
        self.stem = ConvBNAct(3, width, stride=2)
        self.backbone = nn.Sequential(
            ConvBNAct(width, width * 2, stride=2),
            *[RepBlock(width * 2) for _ in range(depth)],
            ConvBNAct(width * 2, width * 4, stride=2),
            *[RepBlock(width * 4) for _ in range(depth)],
        )
        self.down = ConvBNAct(width * 4, width * 8, stride=2)
        self.neck_reduce = nn.Conv2d(width * 8, width * 4, 1)
        self.neck_rep = RepBlock(width * 4)
        self.head = DecoupledHead(width * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run YOLOv6 detection.

        Parameters
        ----------
        x:
            RGB image tensor.

        Returns
        -------
        torch.Tensor
            Detection tensor.
        """

        mid = self.backbone(self.stem(x))
        high = self.down(mid)
        top = F.interpolate(self.neck_reduce(high), size=mid.shape[-2:], mode="nearest")
        return self.head(self.neck_rep(top + mid))


class TinyAttention(nn.Module):
    """Small explicit multi-head attention."""

    def __init__(self, dim: int, heads: int = 4) -> None:
        """Initialize projections.

        Parameters
        ----------
        dim:
            Token width.
        heads:
            Number of heads.
        """

        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        """Apply self- or cross-attention.

        Parameters
        ----------
        x:
            Query tokens.
        context:
            Optional key/value tokens.

        Returns
        -------
        torch.Tensor
            Attended query tokens.
        """

        source = x if context is None else context
        batch, tokens, dim = x.shape
        src_tokens = source.size(1)
        q = self.q(x).view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(source).view(batch, src_tokens, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(source).view(batch, src_tokens, self.heads, self.head_dim).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        out = torch.matmul(torch.softmax(logits, dim=-1), v)
        return self.o(out.transpose(1, 2).reshape(batch, tokens, dim))


class WhisperBlock(nn.Module):
    """Pre-activation Transformer block with optional encoder cross-attention."""

    def __init__(self, dim: int, cross: bool = False) -> None:
        """Initialize the block.

        Parameters
        ----------
        dim:
            Token width.
        cross:
            Whether to include encoder-decoder cross-attention.
        """

        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = TinyAttention(dim)
        self.cross_norm = nn.LayerNorm(dim)
        self.cross_attn = TinyAttention(dim) if cross else None
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        """Apply Transformer residual updates.

        Parameters
        ----------
        x:
            Input tokens.
        context:
            Optional encoder context for decoder blocks.

        Returns
        -------
        torch.Tensor
            Updated tokens.
        """

        x = x + self.self_attn(self.self_norm(x))
        if self.cross_attn is not None and context is not None:
            x = x + self.cross_attn(self.cross_norm(x), context)
        return x + self.ff(self.ff_norm(x))


class TinyWhisper(nn.Module):
    """Whisper-style log-Mel encoder and autoregressive text decoder."""

    def __init__(self, dim: int = 32, layers: int = 2, vocab: int = 128) -> None:
        """Initialize compact Whisper model.

        Parameters
        ----------
        dim:
            Shared encoder/decoder width.
        layers:
            Number of encoder and decoder blocks.
        vocab:
            Text vocabulary size.
        """

        super().__init__()
        self.audio1 = nn.Conv1d(80, dim, 3, padding=1)
        self.audio2 = nn.Conv1d(dim, dim, 3, stride=2, padding=1)
        self.audio_pos = nn.Parameter(torch.randn(1, 16, dim) * 0.02)
        self.encoder = nn.ModuleList([WhisperBlock(dim) for _ in range(layers)])
        self.token = nn.Embedding(vocab, dim)
        self.text_pos = nn.Parameter(torch.randn(1, 8, dim) * 0.02)
        self.decoder = nn.ModuleList([WhisperBlock(dim, cross=True) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Decode tokens conditioned on log-Mel features.

        Parameters
        ----------
        inputs:
            Pair of log-Mel tensor ``(batch, 80, frames)`` and token ids.

        Returns
        -------
        torch.Tensor
            Decoder vocabulary logits.
        """

        mel, tokens = inputs
        audio = F.gelu(self.audio1(mel))
        enc = F.gelu(self.audio2(audio)).transpose(1, 2) + self.audio_pos[:, : audio.size(-1) // 2]
        for block in self.encoder:
            enc = block(enc)
        dec = self.token(tokens) + self.text_pos[:, : tokens.size(1)]
        for block in self.decoder:
            dec = block(dec, enc)
        return self.lm_head(self.norm(dec))


class RetentionBlock(nn.Module):
    """Multi-scale retention block with decay-weighted causal accumulation."""

    def __init__(self, dim: int = 32, heads: int = 4) -> None:
        """Initialize retention projections.

        Parameters
        ----------
        dim:
            Token width.
        heads:
            Number of retention scales.
        """

        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )
        self.register_buffer("decay", torch.tensor([0.55, 0.7, 0.82, 0.92]).view(1, heads, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply parallel causal retention and gated output projection.

        Parameters
        ----------
        x:
            Token tensor.

        Returns
        -------
        torch.Tensor
            Updated token tensor.
        """

        residual = x
        x_norm = self.norm(x)
        batch, tokens, dim = x_norm.shape
        q = self.q(x_norm).view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x_norm).view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(x_norm).view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        steps = torch.arange(tokens, device=x.device)
        distance = (steps[:, None] - steps[None, :]).clamp_min(0)
        causal = (steps[:, None] >= steps[None, :]).to(x.dtype)
        decay = self.decay[:, :, :1, :1].to(x.dtype) ** distance.view(1, 1, tokens, tokens)
        weights = torch.matmul(q, k.transpose(-1, -2)) * decay * causal.view(1, 1, tokens, tokens)
        retained = torch.matmul(weights / math.sqrt(self.head_dim), v)
        retained = retained.transpose(1, 2).reshape(batch, tokens, dim)
        x = residual + self.out(retained * torch.sigmoid(self.gate(x_norm)))
        return x + self.ff(x)


class TinyRetNet(nn.Module):
    """RetNet-style language model with stacked multi-scale retention."""

    def __init__(self, vocab: int = 128, dim: int = 32, layers: int = 2) -> None:
        """Initialize compact RetNet.

        Parameters
        ----------
        vocab:
            Vocabulary size.
        dim:
            Token width.
        layers:
            Number of retention blocks.
        """

        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([RetentionBlock(dim) for _ in range(layers)])
        self.head = nn.Linear(dim, vocab)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run retention language modeling.

        Parameters
        ----------
        tokens:
            Token ids.

        Returns
        -------
        torch.Tensor
            Vocabulary logits.
        """

        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class xLSTMCell(nn.Module):
    """Exponential-gated scalar-memory xLSTM cell."""

    def __init__(self, dim: int) -> None:
        """Initialize input projections.

        Parameters
        ----------
        dim:
            Hidden width.
        """

        super().__init__()
        self.mix = nn.Linear(dim * 2, dim * 4)
        self.memory_mix = nn.Linear(dim, dim)

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update xLSTM hidden, memory, and normalizer states.

        Parameters
        ----------
        x:
            Input vector.
        state:
            Tuple of previous hidden, memory, and normalizer states.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Updated hidden, memory, and normalizer states.
        """

        hidden, memory, normalizer = state
        i_raw, f_raw, z_raw, o_raw = self.mix(torch.cat((x, hidden), dim=-1)).chunk(4, dim=-1)
        input_gate = torch.exp(torch.clamp(i_raw, max=4.0))
        forget_gate = torch.exp(torch.clamp(f_raw, max=4.0))
        candidate = torch.tanh(z_raw + self.memory_mix(memory))
        memory = forget_gate * memory + input_gate * candidate
        normalizer = forget_gate * normalizer + input_gate
        hidden = torch.sigmoid(o_raw) * torch.tanh(memory / normalizer.clamp_min(1e-4))
        return hidden, memory, normalizer


class TinyXLSTM(nn.Module):
    """xLSTM residual sequence model with exponential recurrent gates."""

    def __init__(self, vocab: int = 128, dim: int = 32) -> None:
        """Initialize compact xLSTM language model.

        Parameters
        ----------
        vocab:
            Vocabulary size.
        dim:
            Hidden width.
        """

        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.norm = nn.LayerNorm(dim)
        self.cell = xLSTMCell(dim)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )
        self.head = nn.Linear(dim, vocab)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run xLSTM over a token sequence.

        Parameters
        ----------
        tokens:
            Token ids.

        Returns
        -------
        torch.Tensor
            Vocabulary logits.
        """

        x = self.embed(tokens)
        batch, steps, dim = x.shape
        hidden = x.new_zeros(batch, dim)
        memory = x.new_zeros(batch, dim)
        normalizer = x.new_ones(batch, dim)
        outs = []
        for idx in range(steps):
            hidden, memory, normalizer = self.cell(
                self.norm(x[:, idx]), (hidden, memory, normalizer)
            )
            outs.append(hidden)
        y = torch.stack(outs, dim=1)
        return self.head(y + self.ff(y))


class TinyWaveGANGenerator(nn.Module):
    """WaveGAN generator using 1D transposed convolutions over raw audio."""

    def __init__(self, latent: int = 32) -> None:
        """Initialize compact WaveGAN generator.

        Parameters
        ----------
        latent:
            Latent vector width.
        """

        super().__init__()
        self.fc = nn.Linear(latent, 64 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose1d(64, 32, 25, stride=4, padding=11, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, 25, stride=4, padding=11, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, 25, stride=4, padding=11, output_padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate a raw waveform.

        Parameters
        ----------
        z:
            Latent tensor.

        Returns
        -------
        torch.Tensor
            Generated waveform.
        """

        return self.net(self.fc(z).view(z.size(0), 64, 4))


class WavKANLayer(nn.Module):
    """Wavelet Kolmogorov-Arnold layer with learned basis mixing."""

    def __init__(self, in_dim: int, out_dim: int, bases: int = 4) -> None:
        """Initialize wavelet basis parameters.

        Parameters
        ----------
        in_dim:
            Input feature width.
        out_dim:
            Output feature width.
        bases:
            Number of wavelet bases per input dimension.
        """

        super().__init__()
        self.scale = nn.Parameter(torch.ones(in_dim, bases))
        self.shift = nn.Parameter(torch.linspace(-1.0, 1.0, bases).repeat(in_dim, 1))
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim, bases) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Mexican-hat wavelet basis expansion and linear mixing.

        Parameters
        ----------
        x:
            Input features.

        Returns
        -------
        torch.Tensor
            Output features.
        """

        u = (x.unsqueeze(-1) - self.shift) / self.scale.clamp_min(1e-3)
        basis = (1.0 - u.pow(2)) * torch.exp(-0.5 * u.pow(2))
        return torch.einsum("...ib,oib->...o", basis, self.weight) + self.bias


class TinyWavKAN(nn.Module):
    """Wav-KAN network built from wavelet basis layers."""

    def __init__(self) -> None:
        """Initialize compact Wav-KAN."""

        super().__init__()
        self.layer1 = WavKANLayer(8, 16)
        self.layer2 = WavKANLayer(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run wavelet KAN regression.

        Parameters
        ----------
        x:
            Input feature tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """

        return self.layer2(torch.tanh(self.layer1(x)))


class WireINR(nn.Module):
    """WIRE implicit neural representation with complex Gabor activations."""

    def __init__(self, hidden: int = 32) -> None:
        """Initialize WIRE coordinate network.

        Parameters
        ----------
        hidden:
            Hidden width.
        """

        super().__init__()
        self.inp = nn.Linear(2, hidden)
        self.mid = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 3)

    def gabor(self, x: torch.Tensor) -> torch.Tensor:
        """Apply real-valued Gabor wave activation.

        Parameters
        ----------
        x:
            Pre-activation tensor.

        Returns
        -------
        torch.Tensor
            Activated tensor.
        """

        return torch.cos(10.0 * x) * torch.exp(-2.0 * x.pow(2))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Map coordinates to RGB values.

        Parameters
        ----------
        coords:
            Coordinate tensor.

        Returns
        -------
        torch.Tensor
            RGB predictions.
        """

        x = self.gabor(self.inp(coords))
        return self.out(self.gabor(self.mid(x)))


class EdgeConv(nn.Module):
    """Dynamic graph EdgeConv block for particle clouds."""

    def __init__(self, in_dim: int, out_dim: int, k: int = 3) -> None:
        """Initialize edge MLP.

        Parameters
        ----------
        in_dim:
            Particle feature width.
        out_dim:
            Output feature width.
        k:
            Number of nearest neighbors.
        """

        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, points: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Apply dynamic k-nearest-neighbor edge convolution.

        Parameters
        ----------
        points:
            Particle coordinates.
        features:
            Particle features.

        Returns
        -------
        torch.Tensor
            Updated particle features.
        """

        dist = torch.cdist(points, points)
        idx = dist.topk(self.k + 1, largest=False).indices[:, :, 1:]
        batch, nodes, feat_dim = features.shape
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, -1, feat_dim)
        neigh = torch.gather(features.unsqueeze(1).expand(-1, nodes, -1, -1), 2, gather_idx)
        center = features.unsqueeze(2).expand_as(neigh)
        edge = torch.cat((center, neigh - center), dim=-1)
        return self.mlp(edge).max(dim=2).values


class TinyParticleNet(nn.Module):
    """ParticleNet-style jet classifier using dynamic EdgeConv blocks."""

    def __init__(self) -> None:
        """Initialize compact ParticleNet."""

        super().__init__()
        self.edge1 = EdgeConv(6, 16)
        self.edge2 = EdgeConv(16, 24)
        self.head = nn.Sequential(nn.Linear(24, 16), nn.ReLU(), nn.Linear(16, 5))

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Classify a particle cloud.

        Parameters
        ----------
        inputs:
            Coordinates and per-particle features.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        points, feats = inputs
        x = self.edge1(points, feats)
        x = self.edge2(points, x)
        return self.head(x.mean(dim=1))


class TinyUFold(nn.Module):
    """UFold-style RNA contact predictor with image-like pair features."""

    def __init__(self, channels: int = 17) -> None:
        """Initialize compact U-Net scorer.

        Parameters
        ----------
        channels:
            Pairwise RNA feature channels.
        """

        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
        )
        self.down2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU())
        self.up = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        """Predict a symmetric RNA base-pair score matrix.

        Parameters
        ----------
        pair_features:
            Image-like RNA pair features.

        Returns
        -------
        torch.Tensor
            Symmetric contact scores.
        """

        skip = self.down1(pair_features)
        up = self.up(self.down2(skip))
        score = self.out(torch.cat((skip, up), dim=1)).squeeze(1)
        return 0.5 * (score + score.transpose(-1, -2))


class TinyUSAD(nn.Module):
    """USAD two-decoder adversarial autoencoder for anomaly detection."""

    def __init__(self, window_features: int = 24, latent: int = 8) -> None:
        """Initialize encoder and two decoders.

        Parameters
        ----------
        window_features:
            Flattened window feature count.
        latent:
            Latent width.
        """

        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(window_features, 16), nn.ReLU(), nn.Linear(16, latent)
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(latent, 16), nn.ReLU(), nn.Linear(16, window_features), nn.Sigmoid()
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(latent, 16), nn.ReLU(), nn.Linear(16, window_features), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return USAD direct and adversarial reconstructions.

        Parameters
        ----------
        x:
            Flattened time-window tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            First decoder, second decoder, and second decoder after first reconstruction.
        """

        z = self.encoder(x)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        return w1, w2, w3


class TinyRecursiveModel(nn.Module):
    """Tiny Recursive Model with repeated latent refinement."""

    def __init__(self, dim: int = 32, steps: int = 4) -> None:
        """Initialize recursive refinement model.

        Parameters
        ----------
        dim:
            Hidden width.
        steps:
            Number of recurrent refinement iterations.
        """

        super().__init__()
        self.steps = steps
        self.inp = nn.Linear(16, dim)
        self.update = nn.GRUCell(dim, dim)
        self.halt = nn.Linear(dim, 1)
        self.out = nn.Linear(dim, 10)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Refine a latent state recursively and emit logits plus halting scores.

        Parameters
        ----------
        x:
            Input feature tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Final logits and per-step halting probabilities.
        """

        drive = torch.tanh(self.inp(x))
        state = torch.zeros_like(drive)
        halts = []
        for _ in range(self.steps):
            state = self.update(drive, state)
            halts.append(torch.sigmoid(self.halt(state)))
        return self.out(state), torch.stack(halts, dim=1)


class FirstVAE(nn.Module):
    """Small VAE MLP for MNIST-style flat inputs."""

    def __init__(self, input_dim: int = 784, latent: int = 8) -> None:
        """Initialize encoder and decoder.

        Parameters
        ----------
        input_dim:
            Flattened input width.
        latent:
            Latent width.
        """

        super().__init__()
        self.enc = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU())
        self.mu = nn.Linear(64, latent)
        self.logvar = nn.Linear(64, latent)
        self.dec = nn.Sequential(
            nn.Linear(latent, 64), nn.ReLU(), nn.Linear(64, input_dim), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode and decode with deterministic reparameterization mean.

        Parameters
        ----------
        x:
            Flattened input image.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Reconstruction, mean, and log variance.
        """

        hidden = self.enc(x)
        mu = self.mu(hidden)
        logvar = self.logvar(hidden)
        return self.dec(mu), mu, logvar


class TinyPanguWeather(nn.Module):
    """Pangu-Weather-style 3D Earth-specific Transformer forecast block."""

    def __init__(self, channels: int = 5, dim: int = 24) -> None:
        """Initialize pressure-level patch embedding and local 3D mixing.

        Parameters
        ----------
        channels:
            Number of meteorological variables.
        dim:
            Token width.
        """

        super().__init__()
        self.patch = nn.Conv3d(channels, dim, kernel_size=(2, 2, 2), stride=(1, 2, 2))
        self.level_bias = nn.Parameter(torch.randn(1, dim, 3, 1, 1) * 0.02)
        self.lat_bias = nn.Parameter(torch.randn(1, dim, 1, 4, 1) * 0.02)
        self.lon_bias = nn.Parameter(torch.randn(1, dim, 1, 1, 4) * 0.02)
        self.qkv = nn.Linear(dim, dim * 3)
        self.local = nn.Conv3d(dim, dim, 3, padding=1, groups=dim)
        self.mix = nn.Conv3d(dim, dim, 1)
        self.unpatch = nn.ConvTranspose3d(dim, channels, kernel_size=(2, 2, 2), stride=(1, 2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forecast gridded weather variables from a compact 3D cube.

        Parameters
        ----------
        x:
            Weather tensor with pressure-level, latitude, and longitude axes.

        Returns
        -------
        torch.Tensor
            Forecast tensor on the original grid.
        """

        tokens = self.patch(x) + self.level_bias + self.lat_bias + self.lon_bias
        batch, dim, levels, lat, lon = tokens.shape
        flat = tokens.flatten(2).transpose(1, 2)
        q, k, v = self.qkv(flat).chunk(3, dim=-1)
        earth_bias = torch.linspace(-1, 1, flat.shape[1], device=x.device, dtype=x.dtype)
        attn = torch.softmax(q @ k.transpose(1, 2) / dim**0.5 + earth_bias.view(1, 1, -1), dim=-1)
        globe = (attn @ v).transpose(1, 2).reshape(batch, dim, levels, lat, lon)
        tokens = tokens + globe + F.gelu(self.mix(self.local(tokens)))
        return self.unpatch(tokens)[..., : x.size(-2), : x.size(-1)]


class TinyTransFusion(nn.Module):
    """TransFusion-style LiDAR BEV queries with image-guided decoder fusion."""

    def __init__(self, dim: int = 32, queries: int = 6) -> None:
        """Initialize LiDAR/image backbones and decoder heads.

        Parameters
        ----------
        dim:
            Shared feature width.
        queries:
            Number of object queries.
        """

        super().__init__()
        self.queries = queries
        self.lidar = nn.Conv2d(4, dim, 3, padding=1)
        self.image = nn.Conv2d(3, dim, 3, stride=2, padding=1)
        self.query_proj = nn.Linear(dim, dim)
        self.lidar_dec = TinyAttention(dim)
        self.image_dec = TinyAttention(dim)
        self.box = nn.Linear(dim, 7)
        self.cls = nn.Linear(dim, 4)

    def forward(
        self, inputs: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict 3D boxes after LiDAR then camera soft-association decoding.

        Parameters
        ----------
        inputs:
            LiDAR BEV feature map and camera image.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Box and class logits for object queries.
        """

        bev, image = inputs
        lidar_tokens = self.lidar(bev).flatten(2).transpose(1, 2)
        image_tokens = self.image(image).flatten(2).transpose(1, 2)
        scores = lidar_tokens.norm(dim=-1)
        top_idx = scores.topk(self.queries, dim=1).indices
        gather = top_idx.unsqueeze(-1).expand(-1, -1, lidar_tokens.size(-1))
        queries = self.query_proj(torch.gather(lidar_tokens, 1, gather))
        queries = queries + self.lidar_dec(queries, lidar_tokens)
        queries = queries + self.image_dec(queries, image_tokens)
        return self.box(queries), self.cls(queries)


class TinyTransLOB(nn.Module):
    """TransLOB-style causal convolution and masked self-attention forecaster."""

    def __init__(self, levels: int = 20, dim: int = 32) -> None:
        """Initialize LOB temporal feature extractor.

        Parameters
        ----------
        levels:
            Number of order-book features per time step.
        dim:
            Hidden width.
        """

        super().__init__()
        self.causal1 = nn.Conv1d(levels, dim, 3, padding=2, dilation=1)
        self.causal2 = nn.Conv1d(dim, dim, 3, padding=4, dilation=2)
        self.attn = TinyAttention(dim)
        self.head = nn.Linear(dim, 3)

    def forward(self, lob: torch.Tensor) -> torch.Tensor:
        """Predict price movement classes from a limit order book sequence.

        Parameters
        ----------
        lob:
            Limit-order-book tensor with shape ``(batch, time, levels)``.

        Returns
        -------
        torch.Tensor
            Movement logits.
        """

        x = lob.transpose(1, 2)
        x = F.relu(self.causal1(x)[..., : lob.size(1)])
        x = F.relu(self.causal2(x)[..., : lob.size(1)]).transpose(1, 2)
        return self.head(self.attn(x)[:, -1])


class TinyTTTLinear(nn.Module):
    """TTT-Linear layer with a fast linear hidden model updated online."""

    def __init__(self, vocab: int = 128, dim: int = 24, lr: float = 0.05) -> None:
        """Initialize token projections and fast-weight seed.

        Parameters
        ----------
        vocab:
            Vocabulary size.
        dim:
            Hidden width.
        lr:
            Inner-loop update rate.
        """

        super().__init__()
        self.lr = lr
        self.embed = nn.Embedding(vocab, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.query = nn.Linear(dim, dim)
        self.fast_seed = nn.Parameter(torch.eye(dim))
        self.head = nn.Linear(dim, vocab)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run TTT-style sequence modeling with online linear-model updates.

        Parameters
        ----------
        tokens:
            Token ids.

        Returns
        -------
        torch.Tensor
            Vocabulary logits.
        """

        x = self.embed(tokens)
        batch, _steps, dim = x.shape
        fast = self.fast_seed.unsqueeze(0).expand(batch, -1, -1)
        outputs = []
        for xt in x.unbind(dim=1):
            key = self.key(xt)
            value = self.value(xt)
            pred = torch.bmm(fast, key.unsqueeze(-1)).squeeze(-1)
            err = pred - value
            fast = fast - self.lr * torch.bmm(err.unsqueeze(-1), key.unsqueeze(1)) / math.sqrt(dim)
            out = torch.bmm(fast, self.query(xt).unsqueeze(-1)).squeeze(-1)
            outputs.append(out)
        return self.head(torch.stack(outputs, dim=1))


class TinyPlanetRSSM(nn.Module):
    """PlaNet-style recurrent state-space model with stochastic latent state."""

    def __init__(self, obs_dim: int = 12, action_dim: int = 4, latent: int = 16) -> None:
        """Initialize encoder, recurrent dynamics, and decoder.

        Parameters
        ----------
        obs_dim:
            Observation feature width.
        action_dim:
            Action feature width.
        latent:
            Deterministic and stochastic state width.
        """

        super().__init__()
        self.obs = nn.Linear(obs_dim, latent)
        self.act = nn.Linear(action_dim, latent)
        self.rnn = nn.GRUCell(latent * 2, latent)
        self.prior = nn.Linear(latent, latent * 2)
        self.post = nn.Linear(latent * 2, latent * 2)
        self.dec = nn.Linear(latent * 2, obs_dim)
        self.reward = nn.Linear(latent * 2, 1)

    def forward(
        self, inputs: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Roll a compact RSSM over observations and actions.

        Parameters
        ----------
        inputs:
            Observation and action sequences.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Reconstructions, rewards, prior means, and posterior means.
        """

        obs, actions = inputs
        batch, steps, latent = obs.size(0), obs.size(1), self.act.out_features
        deter = obs.new_zeros(batch, latent)
        stoch = obs.new_zeros(batch, latent)
        recons = []
        rewards = []
        priors = []
        posts = []
        for idx in range(steps):
            deter = self.rnn(torch.cat((stoch, self.act(actions[:, idx])), dim=-1), deter)
            prior_mu, _prior_logvar = self.prior(deter).chunk(2, dim=-1)
            enc = self.obs(obs[:, idx])
            post_mu, _post_logvar = self.post(torch.cat((deter, enc), dim=-1)).chunk(2, dim=-1)
            stoch = post_mu
            state = torch.cat((deter, stoch), dim=-1)
            recons.append(self.dec(state))
            rewards.append(self.reward(state))
            priors.append(prior_mu)
            posts.append(post_mu)
        return (
            torch.stack(recons, dim=1),
            torch.stack(rewards, dim=1),
            torch.stack(priors, dim=1),
            torch.stack(posts, dim=1),
        )


class TinyFlowMatchingUNet(nn.Module):
    """Flow-matching U-Net conditioned on continuous time."""

    def __init__(self, channels: int = 3) -> None:
        """Initialize compact image vector-field network.

        Parameters
        ----------
        channels:
            Image channel count.
        """

        super().__init__()
        self.time = nn.Linear(1, 16)
        self.down = nn.Conv2d(channels + 16, 24, 3, padding=1)
        self.mid = nn.Conv2d(24, 24, 3, padding=1)
        self.up = nn.Conv2d(48, channels, 3, padding=1)

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Predict a flow vector field for image samples.

        Parameters
        ----------
        inputs:
            Image tensor and scalar time tensor.

        Returns
        -------
        torch.Tensor
            Velocity field with image shape.
        """

        image, time = inputs
        emb = (
            self.time(time)
            .view(time.size(0), 16, 1, 1)
            .expand(-1, -1, image.size(-2), image.size(-1))
        )
        down = F.silu(self.down(torch.cat((image, emb), dim=1)))
        mid = F.silu(self.mid(F.avg_pool2d(down, 2)))
        mid = F.interpolate(mid, size=image.shape[-2:], mode="nearest")
        return self.up(torch.cat((down, mid), dim=1))


def build_yolox_nano() -> nn.Module:
    """Build compact YOLOX-nano.

    Returns
    -------
    nn.Module
        Random-init YOLOX-style detector.
    """

    return TinyYOLOX(width=8, depth=1)


def build_yolox_s() -> nn.Module:
    """Build compact YOLOX-S.

    Returns
    -------
    nn.Module
        Random-init YOLOX-style detector.
    """

    return TinyYOLOX(width=10, depth=1)


def build_yolox_m() -> nn.Module:
    """Build compact YOLOX-M.

    Returns
    -------
    nn.Module
        Random-init YOLOX-style detector.
    """

    return TinyYOLOX(width=12, depth=1)


def build_yolox_l() -> nn.Module:
    """Build compact YOLOX-L.

    Returns
    -------
    nn.Module
        Random-init YOLOX-style detector.
    """

    return TinyYOLOX(width=14, depth=2)


def build_yolox_x() -> nn.Module:
    """Build compact YOLOX-X.

    Returns
    -------
    nn.Module
        Random-init YOLOX-style detector.
    """

    return TinyYOLOX(width=16, depth=2)


def build_yolox_tiny() -> nn.Module:
    """Build compact YOLOX-tiny.

    Returns
    -------
    nn.Module
        Random-init YOLOX-style detector.
    """

    return TinyYOLOX(width=8, depth=1)


def build_yolov6_lite_s() -> nn.Module:
    """Build compact YOLOv6-lite-S.

    Returns
    -------
    nn.Module
        Random-init YOLOv6-style detector.
    """

    return TinyYOLOv6(width=8, depth=1)


def build_yolov6_s() -> nn.Module:
    """Build compact YOLOv6-S.

    Returns
    -------
    nn.Module
        Random-init YOLOv6-style detector.
    """

    return TinyYOLOv6(width=10, depth=1)


def build_yolov6_m() -> nn.Module:
    """Build compact YOLOv6-M.

    Returns
    -------
    nn.Module
        Random-init YOLOv6-style detector.
    """

    return TinyYOLOv6(width=12, depth=1)


def build_yolov6_l() -> nn.Module:
    """Build compact YOLOv6-L.

    Returns
    -------
    nn.Module
        Random-init YOLOv6-style detector.
    """

    return TinyYOLOv6(width=14, depth=2)


def build_whisper_tiny() -> nn.Module:
    """Build compact Whisper tiny/base-style model.

    Returns
    -------
    nn.Module
        Random-init Whisper-style model.
    """

    return TinyWhisper(dim=24, layers=1)


def build_whisper_base() -> nn.Module:
    """Build compact Whisper base-style model.

    Returns
    -------
    nn.Module
        Random-init Whisper-style model.
    """

    return TinyWhisper(dim=32, layers=2)


def build_whisper_medium() -> nn.Module:
    """Build compact Whisper medium-style model.

    Returns
    -------
    nn.Module
        Random-init Whisper-style model.
    """

    return TinyWhisper(dim=40, layers=2)


def build_whisper_large() -> nn.Module:
    """Build compact Whisper large-style model.

    Returns
    -------
    nn.Module
        Random-init Whisper-style model.
    """

    return TinyWhisper(dim=48, layers=2)


def build_retnet() -> nn.Module:
    """Build compact RetNet.

    Returns
    -------
    nn.Module
        Random-init RetNet-style language model.
    """

    return TinyRetNet()


def build_xlstm() -> nn.Module:
    """Build compact xLSTM.

    Returns
    -------
    nn.Module
        Random-init xLSTM-style language model.
    """

    return TinyXLSTM()


def build_wavegan() -> nn.Module:
    """Build compact WaveGAN generator.

    Returns
    -------
    nn.Module
        Random-init WaveGAN generator.
    """

    return TinyWaveGANGenerator()


def build_wavkan() -> nn.Module:
    """Build compact Wav-KAN.

    Returns
    -------
    nn.Module
        Random-init wavelet KAN.
    """

    return TinyWavKAN()


def build_wire_inr() -> nn.Module:
    """Build compact WIRE INR.

    Returns
    -------
    nn.Module
        Random-init WIRE coordinate network.
    """

    return WireINR()


def build_particlenet() -> nn.Module:
    """Build compact ParticleNet.

    Returns
    -------
    nn.Module
        Random-init ParticleNet-style classifier.
    """

    return TinyParticleNet()


def build_ufold() -> nn.Module:
    """Build compact UFold.

    Returns
    -------
    nn.Module
        Random-init UFold-style RNA contact scorer.
    """

    return TinyUFold()


def build_usad() -> nn.Module:
    """Build compact USAD.

    Returns
    -------
    nn.Module
        Random-init USAD anomaly autoencoder.
    """

    return TinyUSAD()


def build_trm() -> nn.Module:
    """Build compact Tiny Recursive Model.

    Returns
    -------
    nn.Module
        Random-init recursive model.
    """

    return TinyRecursiveModel()


def build_first_vae() -> nn.Module:
    """Build compact first VAE MLP.

    Returns
    -------
    nn.Module
        Random-init VAE MLP.
    """

    return FirstVAE()


def build_pangu_weather() -> nn.Module:
    """Build compact Pangu-Weather.

    Returns
    -------
    nn.Module
        Random-init 3DEST-style weather model.
    """

    return TinyPanguWeather()


def build_transfusion() -> nn.Module:
    """Build compact TransFusion detector.

    Returns
    -------
    nn.Module
        Random-init LiDAR-camera fusion detector.
    """

    return TinyTransFusion()


def build_translob() -> nn.Module:
    """Build compact TransLOB.

    Returns
    -------
    nn.Module
        Random-init limit-order-book Transformer.
    """

    return TinyTransLOB()


def build_ttt_linear() -> nn.Module:
    """Build compact TTT-Linear sequence model.

    Returns
    -------
    nn.Module
        Random-init TTT-Linear model.
    """

    return TinyTTTLinear()


def build_planet_rssm() -> nn.Module:
    """Build compact PlaNet RSSM.

    Returns
    -------
    nn.Module
        Random-init recurrent state-space world model.
    """

    return TinyPlanetRSSM()


def build_flow_matching_unet() -> nn.Module:
    """Build compact flow-matching U-Net.

    Returns
    -------
    nn.Module
        Random-init time-conditioned vector-field U-Net.
    """

    return TinyFlowMatchingUNet()


def example_image() -> torch.Tensor:
    """Create a small RGB image.

    Returns
    -------
    torch.Tensor
        Image tensor.
    """

    return torch.randn(1, 3, 64, 64)


def example_whisper() -> tuple[torch.Tensor, torch.Tensor]:
    """Create Whisper log-Mel features and decoder tokens.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Log-Mel tensor and token ids.
    """

    return torch.randn(1, 80, 32), torch.randint(0, 128, (1, 8))


def example_tokens() -> torch.Tensor:
    """Create token ids for sequence models.

    Returns
    -------
    torch.Tensor
        Token id tensor.
    """

    return torch.randint(0, 128, (1, 10))


def example_latent() -> torch.Tensor:
    """Create a WaveGAN latent vector.

    Returns
    -------
    torch.Tensor
        Latent tensor.
    """

    return torch.randn(1, 32)


def example_features() -> torch.Tensor:
    """Create generic feature vectors.

    Returns
    -------
    torch.Tensor
        Feature tensor.
    """

    return torch.randn(2, 8)


def example_coords() -> torch.Tensor:
    """Create coordinate samples for INR models.

    Returns
    -------
    torch.Tensor
        Coordinate tensor.
    """

    return torch.rand(1, 16, 2) * 2.0 - 1.0


def example_particles() -> tuple[torch.Tensor, torch.Tensor]:
    """Create particle cloud coordinates and features.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Coordinates and particle features.
    """

    return torch.randn(1, 8, 3), torch.randn(1, 8, 6)


def example_ufold() -> torch.Tensor:
    """Create image-like RNA pair features.

    Returns
    -------
    torch.Tensor
        Pairwise feature tensor.
    """

    return torch.randn(1, 17, 16, 16)


def example_usad() -> torch.Tensor:
    """Create flattened time-window features for USAD.

    Returns
    -------
    torch.Tensor
        Window feature tensor.
    """

    return torch.rand(2, 24)


def example_trm() -> torch.Tensor:
    """Create recursive-model input features.

    Returns
    -------
    torch.Tensor
        Feature tensor.
    """

    return torch.randn(2, 16)


def example_vae() -> torch.Tensor:
    """Create flattened MNIST-like input.

    Returns
    -------
    torch.Tensor
        Flattened image tensor.
    """

    return torch.rand(2, 784)


def example_weather() -> torch.Tensor:
    """Create compact pressure-level weather fields.

    Returns
    -------
    torch.Tensor
        Weather tensor.
    """

    return torch.randn(1, 5, 4, 8, 8)


def example_transfusion() -> tuple[torch.Tensor, torch.Tensor]:
    """Create LiDAR BEV and camera image inputs.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        BEV and image tensors.
    """

    return torch.randn(1, 4, 16, 16), torch.randn(1, 3, 32, 32)


def example_lob() -> torch.Tensor:
    """Create limit-order-book sequence features.

    Returns
    -------
    torch.Tensor
        LOB tensor.
    """

    return torch.randn(1, 12, 20)


def example_planet() -> tuple[torch.Tensor, torch.Tensor]:
    """Create PlaNet observation and action sequences.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Observation and action tensors.
    """

    return torch.randn(1, 5, 12), torch.randn(1, 5, 4)


def example_flow() -> tuple[torch.Tensor, torch.Tensor]:
    """Create flow-matching image and time inputs.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Image and time tensors.
    """

    return torch.randn(1, 3, 32, 32), torch.rand(1, 1)


MENAGERIE_ENTRIES = [
    (
        "openai_whisper_tiny (encoder-decoder speech Transformer)",
        "build_whisper_tiny",
        "example_whisper",
        "2022",
        "DC",
    ),
    (
        "openai_whisper_base (encoder-decoder speech Transformer)",
        "build_whisper_base",
        "example_whisper",
        "2022",
        "DC",
    ),
    (
        "openai_whisper_small (encoder-decoder speech Transformer)",
        "build_whisper_base",
        "example_whisper",
        "2022",
        "DC",
    ),
    (
        "openai_whisper_medium (encoder-decoder speech Transformer)",
        "build_whisper_medium",
        "example_whisper",
        "2022",
        "DC",
    ),
    (
        "openai_whisper_large (encoder-decoder speech Transformer)",
        "build_whisper_large",
        "example_whisper",
        "2022",
        "DC",
    ),
    (
        "megvii_yolox_nano (CSPDarknet PAFPN anchor-free detector)",
        "build_yolox_nano",
        "example_image",
        "2021",
        "DC",
    ),
    (
        "megvii_yolox_tiny (CSPDarknet PAFPN anchor-free detector)",
        "build_yolox_tiny",
        "example_image",
        "2021",
        "DC",
    ),
    (
        "megvii_yolox_s (CSPDarknet PAFPN anchor-free detector)",
        "build_yolox_s",
        "example_image",
        "2021",
        "DC",
    ),
    (
        "megvii_yolox_m (CSPDarknet PAFPN anchor-free detector)",
        "build_yolox_m",
        "example_image",
        "2021",
        "DC",
    ),
    (
        "megvii_yolox_l (CSPDarknet PAFPN anchor-free detector)",
        "build_yolox_l",
        "example_image",
        "2021",
        "DC",
    ),
    (
        "megvii_yolox_x (CSPDarknet PAFPN anchor-free detector)",
        "build_yolox_x",
        "example_image",
        "2021",
        "DC",
    ),
    (
        "ByteTrack-YOLOX-X (YOLOX-X detector used for tracking)",
        "build_yolox_x",
        "example_image",
        "2021",
        "DC",
    ),
    (
        "meituan_yolov6_lite_s (EfficientRep RepPAN detector)",
        "build_yolov6_lite_s",
        "example_image",
        "2022",
        "DC",
    ),
    (
        "meituan_yolov6_lite_m (EfficientRep RepPAN detector)",
        "build_yolov6_s",
        "example_image",
        "2022",
        "DC",
    ),
    (
        "meituan_yolov6_lite_l (EfficientRep RepPAN detector)",
        "build_yolov6_m",
        "example_image",
        "2022",
        "DC",
    ),
    (
        "meituan_yolov6n (EfficientRep RepPAN detector)",
        "build_yolov6_lite_s",
        "example_image",
        "2022",
        "DC",
    ),
    (
        "meituan_yolov6n6 (EfficientRep RepPAN detector)",
        "build_yolov6_lite_s",
        "example_image",
        "2022",
        "DC",
    ),
    (
        "meituan_yolov6s (EfficientRep RepPAN detector)",
        "build_yolov6_s",
        "example_image",
        "2022",
        "DC",
    ),
    (
        "meituan_yolov6s6 (EfficientRep RepPAN detector)",
        "build_yolov6_s",
        "example_image",
        "2022",
        "DC",
    ),
    (
        "meituan_yolov6m (EfficientRep RepPAN detector)",
        "build_yolov6_m",
        "example_image",
        "2022",
        "DC",
    ),
    (
        "meituan_yolov6m6 (EfficientRep RepPAN detector)",
        "build_yolov6_m",
        "example_image",
        "2022",
        "DC",
    ),
    (
        "meituan_yolov6l (EfficientRep RepPAN detector)",
        "build_yolov6_l",
        "example_image",
        "2022",
        "DC",
    ),
    (
        "meituan_yolov6l6 (EfficientRep RepPAN detector)",
        "build_yolov6_l",
        "example_image",
        "2022",
        "DC",
    ),
    (
        "retnet_yet_another (multi-scale retention language model)",
        "build_retnet",
        "example_tokens",
        "2023",
        "DC",
    ),
    (
        "xlstm_simple (exponential-gated scalar-memory xLSTM)",
        "build_xlstm",
        "example_tokens",
        "2024",
        "DC",
    ),
    (
        "xlstm_block (exponential-gated scalar-memory xLSTM)",
        "build_xlstm",
        "example_tokens",
        "2024",
        "DC",
    ),
    (
        "xlstm_official (exponential-gated scalar-memory xLSTM)",
        "build_xlstm",
        "example_tokens",
        "2024",
        "DC",
    ),
    (
        "xLSTM_BlockStack (exponential-gated scalar-memory xLSTM)",
        "build_xlstm",
        "example_tokens",
        "2024",
        "DC",
    ),
    (
        "wavegan_generator (1D transposed-convolution audio GAN generator)",
        "build_wavegan",
        "example_latent",
        "2019",
        "DC",
    ),
    (
        "wavkan (wavelet Kolmogorov-Arnold network)",
        "build_wavkan",
        "example_features",
        "2024",
        "DC",
    ),
    (
        "wire_inr (Gabor wavelet implicit neural representation)",
        "build_wire_inr",
        "example_coords",
        "2023",
        "DC",
    ),
    (
        "ParticleNet (dynamic EdgeConv particle-cloud jet tagger)",
        "build_particlenet",
        "example_particles",
        "2019",
        "DC",
    ),
    ("UFold (RNA contact-map U-Net)", "build_ufold", "example_ufold", "2022", "DC"),
    (
        "USAD reference UsadModel (two-decoder anomaly autoencoder)",
        "build_usad",
        "example_usad",
        "2020",
        "DC",
    ),
    (
        "TinyRecursiveModel-TRM (latent recursive refinement model)",
        "build_trm",
        "example_trm",
        "2025",
        "DC",
    ),
    (
        "FirstVAE-MLP-MNIST (fully connected variational autoencoder)",
        "build_first_vae",
        "example_vae",
        "2013",
        "DC",
    ),
    (
        "pangu_weather.PanguWeather (3D Earth-specific Transformer)",
        "build_pangu_weather",
        "example_weather",
        "2022",
        "DC",
    ),
    (
        "Transfusion (two-stage LiDAR-camera transformer decoder)",
        "build_transfusion",
        "example_transfusion",
        "2022",
        "DC",
    ),
    (
        "TransLOB (causal-conv masked-attention order-book model)",
        "build_translob",
        "example_lob",
        "2020",
        "DC",
    ),
    (
        "ttt_linear (test-time-training fast linear state)",
        "build_ttt_linear",
        "example_tokens",
        "2024",
        "DC",
    ),
    (
        "planet_rssm (PlaNet recurrent stochastic state-space model)",
        "build_planet_rssm",
        "example_planet",
        "2019",
        "DC",
    ),
    (
        "flow_matching_unet (time-conditioned vector-field U-Net)",
        "build_flow_matching_unet",
        "example_flow",
        "2023",
        "DC",
    ),
]
