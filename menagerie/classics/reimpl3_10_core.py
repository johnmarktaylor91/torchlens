"""Compact faithful classics for dependency-gated reimpl3 shard 10 models.

These are small random-initialized PyTorch reconstructions of install-hostile
or heavyweight models.  Each keeps the load-bearing primitive of the named
architecture rather than serving as a generic stand-in: RivaGAN attention-based
video watermark embedding/extraction, RMVSNet recurrent cost-volume depth
regularization, DARTS recurrent cells, Routing Transformer k-means sparse
attention, RWKV time/channel mixing, spherical spectral/group-style filtering,
S4/S5 state-space scans, SAGAN self-attention with spectral-normalized
upsampling/downsampling, LaMa Fourier convolutions, SAINT row/column attention,
SAM-HQ mask decoding, Stable-Baselines3 feature extractors, and scVI/scGen-style
single-cell encoders.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


class ResidualBlock2d(nn.Module):
    """Small residual convolution block."""

    def __init__(self, channels: int) -> None:
        """Initialize the residual block.

        Parameters
        ----------
        channels:
            Number of feature channels.
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(1, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual convolution.

        Parameters
        ----------
        x:
            Feature tensor.

        Returns
        -------
        torch.Tensor
            Updated feature tensor.
        """

        return F.silu(x + self.net(x))


class RivaGANCompact(nn.Module):
    """RivaGAN-style attention video watermark embedder and decoder."""

    def __init__(self, width: int = 12) -> None:
        """Initialize video, message, attention, and decoder branches.

        Parameters
        ----------
        width:
            Compact hidden channel count.
        """

        super().__init__()
        self.video = nn.Conv2d(3, width, 3, padding=1)
        self.message = nn.Conv2d(1, width, 1)
        self.attn = nn.Sequential(
            ResidualBlock2d(width * 2), nn.Conv2d(width * 2, 1, 1), nn.Sigmoid()
        )
        self.embed = nn.Sequential(
            ResidualBlock2d(width * 2), nn.Conv2d(width * 2, 3, 3, padding=1)
        )
        self.decode = nn.Sequential(
            nn.Conv3d(3, width, (3, 3, 3), padding=1),
            nn.SiLU(),
            nn.Conv3d(width, width, (3, 3, 3), padding=1),
            nn.SiLU(),
        )
        self.readout = nn.Linear(width, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed a message map in video frames and decode a compact watermark.

        Parameters
        ----------
        x:
            Tensor shaped ``(batch, time, 4, height, width)``; the first three
            channels are video RGB and the fourth is a broadcast message map.

        Returns
        -------
        torch.Tensor
            Decoded watermark logits.
        """

        batch, time, _, height, width = x.shape
        video = x[:, :, :3].reshape(batch * time, 3, height, width)
        message = x[:, :, 3:4].reshape(batch * time, 1, height, width)
        vf = self.video(video)
        mf = self.message(message)
        joined = torch.cat([vf, mf], dim=1)
        alpha = self.attn(joined)
        watermarked = video + 0.08 * alpha * torch.tanh(self.embed(joined))
        decoded = self.decode(watermarked.reshape(batch, time, 3, height, width).transpose(1, 2))
        return self.readout(decoded.mean(dim=(2, 3, 4)))


class RMVSNetDepth(nn.Module):
    """RMVSNet-style recurrent regularization over a plane-sweep cost volume."""

    def __init__(self, depth_planes: int = 5, width: int = 8) -> None:
        """Initialize feature extractor and ConvGRU depth regularizer.

        Parameters
        ----------
        depth_planes:
            Number of compact depth hypotheses.
        width:
            Feature width.
        """

        super().__init__()
        self.depth_planes = depth_planes
        self.feat = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1), nn.ReLU(), ResidualBlock2d(width)
        )
        self.gates = nn.Conv2d(width + 2, 2 * width, 3, padding=1)
        self.cand = nn.Conv2d(width + 2, width, 3, padding=1)
        self.score = nn.Conv2d(width, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate depth probabilities from a tiny multi-view image stack.

        Parameters
        ----------
        x:
            Image tensor shaped ``(batch, views, 3, height, width)``.

        Returns
        -------
        torch.Tensor
            Soft argmin depth map.
        """

        batch, views, channels, height, width = x.shape
        feat = self.feat(x.reshape(batch * views, channels, height, width)).reshape(
            batch, views, -1, height, width
        )
        ref = feat[:, 0]
        hidden = torch.zeros_like(ref)
        scores: list[torch.Tensor] = []
        for plane in range(self.depth_planes):
            shift = plane - self.depth_planes // 2
            warped = torch.roll(feat[:, 1:], shifts=shift, dims=-1).mean(dim=1)
            cost = (ref - warped).pow(2).mean(dim=1, keepdim=True)
            depth_code = cost.new_full((batch, 1, height, width), float(plane) / self.depth_planes)
            gates = torch.sigmoid(self.gates(torch.cat([cost, depth_code, hidden], dim=1)))
            reset, update = gates.chunk(2, dim=1)
            cand = torch.tanh(self.cand(torch.cat([cost, depth_code, reset * hidden], dim=1)))
            hidden = (1.0 - update) * hidden + update * cand
            scores.append(self.score(hidden))
        prob = torch.softmax(torch.cat(scores, dim=1), dim=1)
        planes = torch.linspace(0.0, 1.0, self.depth_planes, device=x.device).view(1, -1, 1, 1)
        return (prob * planes).sum(dim=1, keepdim=True)


class DARTSRNNCell(nn.Module):
    """Differentiable Architecture Search recurrent cell."""

    def __init__(self, size: int = 16, search: bool = False) -> None:
        """Initialize DARTS recurrent edges.

        Parameters
        ----------
        size:
            Hidden size.
        search:
            Whether to keep mixed softmax architecture weights instead of a
            fixed derived genotype.
        """

        super().__init__()
        self.search = search
        self.xh = nn.Linear(size * 2, size)
        self.edges = nn.ModuleList([nn.Linear(size, size) for _ in range(4)])
        self.alpha = nn.Parameter(torch.randn(4, 4) * 0.01)

    def _op(self, x: torch.Tensor, op_idx: int) -> torch.Tensor:
        """Apply one DARTS candidate activation.

        Parameters
        ----------
        x:
            Edge input.
        op_idx:
            Operation index.

        Returns
        -------
        torch.Tensor
            Operation output.
        """

        if op_idx == 0:
            return torch.tanh(x)
        if op_idx == 1:
            return torch.relu(x)
        if op_idx == 2:
            return torch.sigmoid(x)
        return x

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Run one DARTS recurrent step.

        Parameters
        ----------
        x:
            Input projection for the current token.
        h:
            Previous hidden state.

        Returns
        -------
        torch.Tensor
            Next hidden state.
        """

        s = torch.tanh(self.xh(torch.cat([x, h], dim=-1)))
        states = [s]
        for edge_idx, edge in enumerate(self.edges):
            base = edge(states[-1])
            if self.search:
                weights = torch.softmax(self.alpha[edge_idx], dim=0)
                nxt = sum(weights[i] * self._op(base, i) for i in range(4))
            else:
                nxt = self._op(base, edge_idx % 4)
            states.append(nxt)
        return torch.stack(states[1:], dim=0).mean(dim=0)


class DARTSRNN(nn.Module):
    """Tiny DARTS recurrent language model."""

    def __init__(self, search: bool = False) -> None:
        """Initialize embedding, DARTS cell, and decoder.

        Parameters
        ----------
        search:
            Whether to use the continuous search network cell.
        """

        super().__init__()
        self.embed = nn.Embedding(32, 16)
        self.cell = DARTSRNNCell(16, search)
        self.out = nn.Linear(16, 32)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Run the recurrent model over token ids.

        Parameters
        ----------
        ids:
            Token ids shaped ``(batch, time)``.

        Returns
        -------
        torch.Tensor
            Token logits.
        """

        emb = self.embed(ids)
        h = emb.new_zeros(emb.shape[0], emb.shape[-1])
        outs: list[torch.Tensor] = []
        for step in range(emb.shape[1]):
            h = self.cell(emb[:, step], h)
            outs.append(self.out(h))
        return torch.stack(outs, dim=1)


class ElmanRNNLM(nn.Module):
    """Classic Elman simple recurrent language model."""

    def __init__(self) -> None:
        """Initialize SRN language model layers."""

        super().__init__()
        self.embed = nn.Embedding(32, 12)
        self.inp = nn.Linear(12, 12)
        self.rec = nn.Linear(12, 12, bias=False)
        self.out = nn.Linear(12, 32)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Run the Elman recurrence.

        Parameters
        ----------
        ids:
            Token ids shaped ``(batch, time)``.

        Returns
        -------
        torch.Tensor
            Logits per token.
        """

        emb = self.embed(ids)
        h = emb.new_zeros(emb.shape[0], 12)
        outs: list[torch.Tensor] = []
        for step in range(emb.shape[1]):
            h = torch.tanh(self.inp(emb[:, step]) + self.rec(h))
            outs.append(self.out(h))
        return torch.stack(outs, dim=1)


class LIFNet(nn.Module):
    """Rockpool LIFTorch-style leaky integrate-and-fire sequence model."""

    def __init__(self) -> None:
        """Initialize LIF recurrent layer and readout."""

        super().__init__()
        self.input = nn.Linear(6, 10)
        self.recurrent = nn.Linear(10, 10, bias=False)
        self.readout = nn.Linear(10, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run LIF membrane dynamics with hard reset.

        Parameters
        ----------
        x:
            Event sequence shaped ``(batch, time, 6)``.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        v = x.new_zeros(x.shape[0], 10)
        spike = x.new_zeros(x.shape[0], 10)
        acc = x.new_zeros(x.shape[0], 10)
        for step in range(x.shape[1]):
            v = 0.75 * v + self.input(x[:, step]) + self.recurrent(spike)
            spike = (v > 1.0).to(x.dtype)
            v = v * (1.0 - spike)
            acc = acc + spike
        return self.readout(acc / x.shape[1])


class RoutingAttention(nn.Module):
    """Routing Transformer attention with learned online-k-means centroids."""

    def __init__(self, dim: int = 24, heads: int = 3, clusters: int = 3) -> None:
        """Initialize projections and centroids.

        Parameters
        ----------
        dim:
            Model dimension.
        heads:
            Attention head count.
        clusters:
            Number of routing clusters.
        """

        super().__init__()
        self.heads = heads
        self.clusters = clusters
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.centroids = nn.Parameter(torch.randn(heads, clusters, dim // heads))
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply cluster-masked sparse self-attention.

        Parameters
        ----------
        x:
            Sequence tensor shaped ``(batch, time, dim)``.

        Returns
        -------
        torch.Tensor
            Routed-attention output.
        """

        batch, time, dim = x.shape
        head_dim = dim // self.heads
        q, k, v = self.to_qkv(x).view(batch, time, 3, self.heads, head_dim).unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        route = torch.cdist(
            q.reshape(batch * self.heads, time, head_dim), self.centroids.repeat(batch, 1, 1)
        )
        labels = route.argmin(dim=-1).view(batch, self.heads, time)
        same = labels[:, :, :, None] == labels[:, :, None, :]
        local = torch.arange(time, device=x.device)
        local_mask = (local[:, None] - local[None, :]).abs() <= 1
        mask = same | local_mask
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
        scores = scores.masked_fill(~mask, -1e4)
        y = torch.matmul(torch.softmax(scores, dim=-1), v)
        return self.out(y.transpose(1, 2).reshape(batch, time, dim))


class RoutingTransformerLM(nn.Module):
    """Compact Routing Transformer language model."""

    def __init__(self) -> None:
        """Initialize embedding, routed attention block, and head."""

        super().__init__()
        self.embed = nn.Embedding(48, 24)
        self.norm1 = nn.LayerNorm(24)
        self.attn = RoutingAttention()
        self.norm2 = nn.LayerNorm(24)
        self.ff = nn.Sequential(nn.Linear(24, 48), nn.GELU(), nn.Linear(48, 24))
        self.head = nn.Linear(24, 48)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Run routed sparse attention over token ids.

        Parameters
        ----------
        ids:
            Token ids.

        Returns
        -------
        torch.Tensor
            Token logits.
        """

        x = self.embed(ids)
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return self.head(x)


class RWKVBlock(nn.Module):
    """RWKV time-mix and channel-mix block with recurrent weighted KV scan."""

    def __init__(self, dim: int = 24, version: int = 6) -> None:
        """Initialize RWKV mixing projections.

        Parameters
        ----------
        dim:
            Hidden dimension.
        version:
            RWKV version hint; version 7 adds a delta-rule state update.
        """

        super().__init__()
        self.version = version
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.time_mix = nn.Parameter(torch.linspace(0.0, 1.0, dim))
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.decay = nn.Parameter(torch.linspace(-3.0, -1.0, dim))
        self.gate = nn.Linear(dim, dim, bias=False)
        self.up = nn.Linear(dim, dim * 3)
        self.down = nn.Linear(dim * 3, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RWKV recurrent time and channel mixing.

        Parameters
        ----------
        x:
            Sequence features.

        Returns
        -------
        torch.Tensor
            Updated sequence features.
        """

        z = self.ln1(x)
        prev = torch.cat([z[:, :1] * 0.0, z[:, :-1]], dim=1)
        mixed = self.time_mix * z + (1.0 - self.time_mix) * prev
        k = torch.exp(torch.clamp(self.key(mixed), max=4.0))
        v = self.value(mixed)
        r = torch.sigmoid(self.receptance(mixed))
        state = torch.zeros_like(k[:, 0])
        denom = torch.zeros_like(k[:, 0])
        outs: list[torch.Tensor] = []
        decay = torch.exp(-torch.exp(self.decay))
        for step in range(x.shape[1]):
            if self.version >= 7:
                delta = k[:, step] * (v[:, step] - state)
                state = decay * state + torch.sigmoid(self.gate(mixed[:, step])) * delta
                denom = decay * denom + k[:, step]
            else:
                state = decay * state + k[:, step] * v[:, step]
                denom = decay * denom + k[:, step]
            outs.append(r[:, step] * state / denom.clamp_min(1e-4))
        y = x + torch.stack(outs, dim=1)
        c = self.ln2(y)
        cprev = torch.cat([c[:, :1] * 0.0, c[:, :-1]], dim=1)
        cmix = 0.5 * c + 0.5 * cprev
        return y + torch.sigmoid(self.gate(cmix)) * self.down(F.gelu(self.up(cmix)))


class RWKVLM(nn.Module):
    """Tiny language model using a single RWKV block."""

    def __init__(self, version: int = 6) -> None:
        """Initialize embedding, RWKV block, and head.

        Parameters
        ----------
        version:
            RWKV block variant.
        """

        super().__init__()
        self.embed = nn.Embedding(48, 24)
        self.block = RWKVBlock(24, version)
        self.head = nn.Linear(24, 48)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Run the RWKV model.

        Parameters
        ----------
        ids:
            Token ids.

        Returns
        -------
        torch.Tensor
            Token logits.
        """

        return self.head(self.block(self.embed(ids)))


class SphericalCNN(nn.Module):
    """Spherical CNN with spectral smoothing and longitude group pooling."""

    def __init__(self) -> None:
        """Initialize spherical spectral filters."""

        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 3, 5, 5) * 0.05)
        self.proj = nn.Linear(4, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply compact spherical convolutional filtering.

        Parameters
        ----------
        x:
            Equirectangular spherical signal ``(batch, channels, lat, lon)``.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        spec = torch.fft.rfft2(x, norm="ortho")
        lat = torch.linspace(-1.0, 1.0, x.shape[-2], device=x.device).view(1, 1, -1, 1)
        smooth = torch.exp(-4.0 * lat.pow(2))
        low = torch.fft.irfft2(spec * smooth, s=x.shape[-2:], norm="ortho")
        y = F.conv2d(F.pad(low, (2, 2, 2, 2), mode="circular"), self.weight)
        y = torch.stack(
            [y, torch.roll(y, shifts=1, dims=-1), torch.roll(y, shifts=2, dims=-1)]
        ).mean(dim=0)
        return self.proj(F.silu(y).mean(dim=(2, 3)))


class DiagonalSSMLayer(nn.Module):
    """S4/S5-style diagonal state-space scan."""

    def __init__(self, dim: int = 16, state: int = 8, mimo: bool = False) -> None:
        """Initialize diagonal SSM parameters.

        Parameters
        ----------
        dim:
            Input/output dimension.
        state:
            State size.
        mimo:
            Whether to use one MIMO S5-style state instead of channelwise S4.
        """

        super().__init__()
        self.mimo = mimo
        self.inp = nn.Linear(dim, state if mimo else dim * state)
        self.out = nn.Linear(state if mimo else dim * state, dim)
        self.log_decay = nn.Parameter(torch.linspace(-3.0, -0.5, state))
        self.skip = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run recurrent state-space filtering.

        Parameters
        ----------
        x:
            Sequence features.

        Returns
        -------
        torch.Tensor
            Filtered sequence.
        """

        batch, time, dim = x.shape
        decay = torch.exp(-torch.exp(self.log_decay))
        if self.mimo:
            state = x.new_zeros(batch, decay.numel())
        else:
            state = x.new_zeros(batch, dim, decay.numel())
        ys: list[torch.Tensor] = []
        for step in range(time):
            u = self.inp(x[:, step])
            if self.mimo:
                state = decay * state + u
                y = self.out(state)
            else:
                u = u.view(batch, dim, decay.numel())
                state = decay * state + u
                y = self.out(state.reshape(batch, -1))
            ys.append(y + self.skip * x[:, step])
        return torch.stack(ys, dim=1)


class SSMClassifier(nn.Module):
    """Compact S4/S5 sequence classifier."""

    def __init__(self, mimo: bool = False) -> None:
        """Initialize projection, SSM layer, and classifier.

        Parameters
        ----------
        mimo:
            Use S5 MIMO state-space layer when true.
        """

        super().__init__()
        self.proj = nn.Linear(6, 16)
        self.ssm = DiagonalSSMLayer(16, 8, mimo)
        self.head = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a short sequence.

        Parameters
        ----------
        x:
            Sequence tensor.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        return self.head(self.ssm(self.proj(x)).mean(dim=1))


class SelfAttention2d(nn.Module):
    """SAGAN non-local self-attention block."""

    def __init__(self, channels: int) -> None:
        """Initialize attention projections.

        Parameters
        ----------
        channels:
            Feature channel count.
        """

        super().__init__()
        hidden = max(1, channels // 8)
        self.q = spectral_norm(nn.Conv2d(channels, hidden, 1))
        self.k = spectral_norm(nn.Conv2d(channels, hidden, 1))
        self.v = spectral_norm(nn.Conv2d(channels, channels, 1))
        self.gamma = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial self-attention.

        Parameters
        ----------
        x:
            Image features.

        Returns
        -------
        torch.Tensor
            Attended features.
        """

        batch, channels, height, width = x.shape
        q = self.q(x).flatten(2).transpose(1, 2)
        k = self.k(x).flatten(2)
        attn = torch.softmax(torch.bmm(q, k) / math.sqrt(max(1, q.shape[-1])), dim=-1)
        v = self.v(x).flatten(2).transpose(1, 2)
        out = torch.bmm(attn, v).transpose(1, 2).view(batch, channels, height, width)
        return x + self.gamma * out


class SAGANGenerator(nn.Module):
    """Self-Attention GAN generator with spectral-normalized upsampling."""

    def __init__(self) -> None:
        """Initialize compact generator."""

        super().__init__()
        self.fc = spectral_norm(nn.Linear(16, 32 * 4 * 4))
        self.up1 = spectral_norm(nn.Conv2d(32, 16, 3, padding=1))
        self.attn = SelfAttention2d(16)
        self.up2 = spectral_norm(nn.Conv2d(16, 8, 3, padding=1))
        self.to_rgb = spectral_norm(nn.Conv2d(8, 3, 3, padding=1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate a small image from latent vectors.

        Parameters
        ----------
        z:
            Latent tensor.

        Returns
        -------
        torch.Tensor
            Generated image.
        """

        x = self.fc(z).view(z.shape[0], 32, 4, 4)
        x = F.interpolate(F.relu(x), scale_factor=2.0, mode="nearest")
        x = F.relu(self.up1(x))
        x = self.attn(x)
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return torch.tanh(self.to_rgb(F.relu(self.up2(x))))


class SAGANDiscriminator(nn.Module):
    """Self-Attention GAN discriminator with spectral normalization."""

    def __init__(self) -> None:
        """Initialize compact discriminator."""

        super().__init__()
        self.c1 = spectral_norm(nn.Conv2d(3, 8, 3, stride=2, padding=1))
        self.attn = SelfAttention2d(8)
        self.c2 = spectral_norm(nn.Conv2d(8, 16, 3, stride=2, padding=1))
        self.out = spectral_norm(nn.Linear(16, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score input images.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        torch.Tensor
            Real/fake logits.
        """

        y = F.leaky_relu(self.c1(x), 0.2)
        y = self.attn(y)
        y = F.leaky_relu(self.c2(y), 0.2)
        return self.out(y.mean(dim=(2, 3)))


class FourierUnit(nn.Module):
    """LaMa fast Fourier convolution unit."""

    def __init__(self, channels: int) -> None:
        """Initialize spectral mixing projection.

        Parameters
        ----------
        channels:
            Feature channel count.
        """

        super().__init__()
        self.mix = nn.Conv2d(channels * 2, channels * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix real and imaginary Fourier coefficients.

        Parameters
        ----------
        x:
            Feature tensor.

        Returns
        -------
        torch.Tensor
            Fourier-filtered features.
        """

        freq = torch.fft.rfft2(x, norm="ortho")
        packed = torch.cat([freq.real, freq.imag], dim=1)
        mixed = self.mix(packed)
        real, imag = mixed.chunk(2, dim=1)
        return torch.fft.irfft2(torch.complex(real, imag), s=x.shape[-2:], norm="ortho")


class LaMaFourierGenerator(nn.Module):
    """LaMa inpainting generator using fast Fourier convolutions."""

    def __init__(self) -> None:
        """Initialize compact Fourier generator."""

        super().__init__()
        self.inp = nn.Conv2d(4, 16, 3, padding=1)
        self.local = ResidualBlock2d(16)
        self.global_filter = FourierUnit(16)
        self.out = nn.Conv2d(16, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inpaint an image with a mask channel.

        Parameters
        ----------
        x:
            Tensor shaped ``(batch, 4, height, width)``.

        Returns
        -------
        torch.Tensor
            Inpainted RGB image.
        """

        y = F.silu(self.inp(x))
        y = self.local(y) + self.global_filter(y)
        return torch.tanh(self.out(y))


class SAINTTabular(nn.Module):
    """SAINT tabular Transformer with row and column attention."""

    def __init__(self) -> None:
        """Initialize column embeddings and attention blocks."""

        super().__init__()
        self.col_embed = nn.Parameter(torch.randn(1, 5, 12))
        self.value = nn.Linear(1, 12)
        self.col_attn = nn.MultiheadAttention(12, 3, batch_first=True)
        self.row_attn = nn.MultiheadAttention(12, 3, batch_first=True)
        self.head = nn.Linear(12, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run SAINT over tabular rows and columns.

        Parameters
        ----------
        x:
            Tabular tensor shaped ``(batch, columns)``.

        Returns
        -------
        torch.Tensor
            Row logits.
        """

        y = self.value(x.unsqueeze(-1)) + self.col_embed
        y, _ = self.col_attn(y, y, y, need_weights=False)
        cols = y.shape[1]
        r = y.transpose(0, 1)
        r, _ = self.row_attn(r, r, r, need_weights=False)
        y = r.transpose(0, 1).reshape(x.shape[0], cols, -1)
        return self.head(y.mean(dim=1))


class SAMHQTiny(nn.Module):
    """SAM-HQ style ViT image encoder and high-quality mask decoder."""

    def __init__(self) -> None:
        """Initialize patch embedding, transformer, prompt, and HQ token head."""

        super().__init__()
        self.patch = nn.Conv2d(3, 24, 4, stride=4)
        enc_layer = nn.TransformerEncoderLayer(24, 3, 48, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, 1)
        self.prompt = nn.Linear(4, 24)
        self.hq_token = nn.Parameter(torch.randn(1, 1, 24))
        self.mask = nn.Linear(24, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode an HQ mask from an image and box prompt.

        Parameters
        ----------
        x:
            Tensor shaped ``(batch, 7, 16, 16)``; first three channels are image,
            remaining channels encode a broadcast box prompt.

        Returns
        -------
        torch.Tensor
            Compact mask logits.
        """

        image = x[:, :3]
        box = x[:, 3:, 0, 0]
        tokens = self.patch(image).flatten(2).transpose(1, 2)
        enc = self.encoder(tokens)
        prompt = self.prompt(box).unsqueeze(1)
        hq = self.hq_token.expand(x.shape[0], -1, -1)
        fused = enc.mean(dim=1, keepdim=True) + prompt + hq
        return self.mask(fused.squeeze(1)).view(x.shape[0], 1, 4, 4)


class NatureCNN(nn.Module):
    """Stable-Baselines3 NatureCNN visual feature extractor."""

    def __init__(self) -> None:
        """Initialize Atari-style convolution stack."""

        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 16, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2),
            nn.ReLU(),
        )
        self.fc = nn.Linear(32 * 2 * 2, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract visual features.

        Parameters
        ----------
        x:
            Atari frame stack.

        Returns
        -------
        torch.Tensor
            Feature vector.
        """

        return F.relu(self.fc(self.net(x).flatten(1)))


class MlpExtractor(nn.Module):
    """Stable-Baselines3 shared MLP with separate policy and value branches."""

    def __init__(self) -> None:
        """Initialize shared, policy, and value networks."""

        super().__init__()
        self.shared = nn.Sequential(nn.Linear(10, 16), nn.Tanh())
        self.policy = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.value = nn.Sequential(nn.Linear(16, 12), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return concatenated policy and value latent features.

        Parameters
        ----------
        x:
            Flat observation features.

        Returns
        -------
        torch.Tensor
            Concatenated actor/critic features.
        """

        shared = self.shared(x)
        return torch.cat([self.policy(shared), self.value(shared)], dim=-1)


class SingleCellVAE(nn.Module):
    """scVI/scGen-style single-cell variational autoencoder."""

    def __init__(self, conditional: bool = False) -> None:
        """Initialize encoder and decoder.

        Parameters
        ----------
        conditional:
            Whether to inject a perturbation/batch condition as scGen does.
        """

        super().__init__()
        self.conditional = conditional
        self.enc = nn.Sequential(nn.Linear(12 + (2 if conditional else 0), 24), nn.ReLU())
        self.mu = nn.Linear(24, 6)
        self.logvar = nn.Linear(24, 6)
        self.dec = nn.Sequential(
            nn.Linear(6 + (2 if conditional else 0), 24), nn.ReLU(), nn.Linear(24, 12)
        )
        self.dispersion = nn.Parameter(torch.zeros(12))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and decode normalized gene counts.

        Parameters
        ----------
        x:
            Gene counts, optionally with two condition columns at the end.

        Returns
        -------
        torch.Tensor
            Negative-binomial mean parameters.
        """

        if self.conditional:
            genes, cond = x[:, :12], x[:, 12:]
            h = self.enc(torch.cat([torch.log1p(genes.clamp_min(0.0)), cond], dim=-1))
        else:
            genes = x
            cond = x.new_zeros(x.shape[0], 0)
            h = self.enc(torch.log1p(genes.clamp_min(0.0)))
        z = self.mu(h) + torch.exp(0.5 * self.logvar(h)) * 0.0
        dec_in = torch.cat([z, cond], dim=-1) if self.conditional else z
        return F.softplus(self.dec(dec_in)) * torch.exp(self.dispersion)


def build_rivagan() -> nn.Module:
    """Build compact RivaGAN.

    Returns
    -------
    nn.Module
        RivaGAN reconstruction.
    """

    return RivaGANCompact()


def example_rivagan() -> torch.Tensor:
    """Return RivaGAN example input.

    Returns
    -------
    torch.Tensor
        Video plus message map.
    """

    return torch.randn(1, 3, 4, 16, 16)


def build_rmvsnet_depth() -> nn.Module:
    """Build compact RMVSNet depth estimator.

    Returns
    -------
    nn.Module
        RMVSNet reconstruction.
    """

    return RMVSNetDepth()


def example_rmvsnet_depth() -> torch.Tensor:
    """Return RMVSNet example input.

    Returns
    -------
    torch.Tensor
        Multi-view image stack.
    """

    return torch.randn(1, 3, 3, 16, 16)


def build_darts_rnn_derived() -> nn.Module:
    """Build DARTS derived recurrent network.

    Returns
    -------
    nn.Module
        Fixed-genotype DARTS RNN.
    """

    return DARTSRNN(search=False)


def build_darts_rnn_search() -> nn.Module:
    """Build DARTS search recurrent network.

    Returns
    -------
    nn.Module
        Continuous-relaxation DARTS RNN.
    """

    return DARTSRNN(search=True)


def example_tokens() -> torch.Tensor:
    """Return token id sequence.

    Returns
    -------
    torch.Tensor
        Token ids.
    """

    return torch.randint(0, 32, (1, 6))


def build_elman_rnn_lm() -> nn.Module:
    """Build Elman RNN language model.

    Returns
    -------
    nn.Module
        Elman model.
    """

    return ElmanRNNLM()


def build_rockpool_lif_torch() -> nn.Module:
    """Build Rockpool LIFTorch-style model.

    Returns
    -------
    nn.Module
        LIF model.
    """

    return LIFNet()


def example_lif() -> torch.Tensor:
    """Return LIF event sequence.

    Returns
    -------
    torch.Tensor
        Event tensor.
    """

    return torch.randn(1, 8, 6)


def build_routing_transformer() -> nn.Module:
    """Build compact Routing Transformer.

    Returns
    -------
    nn.Module
        Routed attention language model.
    """

    return RoutingTransformerLM()


def example_routing_tokens() -> torch.Tensor:
    """Return Routing Transformer token ids.

    Returns
    -------
    torch.Tensor
        Token ids.
    """

    return torch.randint(0, 48, (1, 9))


def build_rwkv6_block() -> nn.Module:
    """Build RWKV-6 block model.

    Returns
    -------
    nn.Module
        RWKV-6 model.
    """

    return RWKVLM(version=6)


def build_rwkv7_block() -> nn.Module:
    """Build RWKV-7 block model.

    Returns
    -------
    nn.Module
        RWKV-7 model.
    """

    return RWKVLM(version=7)


def example_rwkv_tokens() -> torch.Tensor:
    """Return RWKV token ids.

    Returns
    -------
    torch.Tensor
        Token ids.
    """

    return torch.randint(0, 48, (1, 7))


def build_spherical_cnn() -> nn.Module:
    """Build compact spherical CNN.

    Returns
    -------
    nn.Module
        Spherical CNN.
    """

    return SphericalCNN()


def example_spherical() -> torch.Tensor:
    """Return spherical image sample.

    Returns
    -------
    torch.Tensor
        Spherical signal.
    """

    return torch.randn(1, 3, 12, 24)


def build_s4_s4torch() -> nn.Module:
    """Build S4-style sequence model.

    Returns
    -------
    nn.Module
        S4 model.
    """

    return SSMClassifier(mimo=False)


def build_s5() -> nn.Module:
    """Build S5-style sequence model.

    Returns
    -------
    nn.Module
        S5 model.
    """

    return SSMClassifier(mimo=True)


def example_sequence() -> torch.Tensor:
    """Return sequence input.

    Returns
    -------
    torch.Tensor
        Sequence features.
    """

    return torch.randn(1, 10, 6)


def build_sagan_generator() -> nn.Module:
    """Build SAGAN generator.

    Returns
    -------
    nn.Module
        SAGAN generator.
    """

    return SAGANGenerator()


def build_sagan_discriminator() -> nn.Module:
    """Build SAGAN discriminator.

    Returns
    -------
    nn.Module
        SAGAN discriminator.
    """

    return SAGANDiscriminator()


def example_latent() -> torch.Tensor:
    """Return GAN latent sample.

    Returns
    -------
    torch.Tensor
        Latent vector.
    """

    return torch.randn(1, 16)


def example_image16() -> torch.Tensor:
    """Return 16x16 image.

    Returns
    -------
    torch.Tensor
        Image tensor.
    """

    return torch.randn(1, 3, 16, 16)


def build_lama_fourier_generator() -> nn.Module:
    """Build LaMa Fourier generator.

    Returns
    -------
    nn.Module
        LaMa-style generator.
    """

    return LaMaFourierGenerator()


def example_lama() -> torch.Tensor:
    """Return masked image tensor.

    Returns
    -------
    torch.Tensor
        RGB plus mask.
    """

    return torch.randn(1, 4, 16, 16)


def build_saint_tabular() -> nn.Module:
    """Build SAINT tabular model.

    Returns
    -------
    nn.Module
        SAINT model.
    """

    return SAINTTabular()


def example_tabular() -> torch.Tensor:
    """Return tabular input.

    Returns
    -------
    torch.Tensor
        Tabular features.
    """

    return torch.randn(2, 5)


def build_sam_hq_vit_b_source() -> nn.Module:
    """Build compact SAM-HQ ViT-B source model.

    Returns
    -------
    nn.Module
        SAM-HQ reconstruction.
    """

    return SAMHQTiny()


def example_sam_hq() -> torch.Tensor:
    """Return SAM-HQ image and box prompt tensor.

    Returns
    -------
    torch.Tensor
        Image plus prompt channels.
    """

    return torch.randn(1, 7, 16, 16)


def build_sb3_nature_cnn() -> nn.Module:
    """Build Stable-Baselines3 NatureCNN.

    Returns
    -------
    nn.Module
        NatureCNN extractor.
    """

    return NatureCNN()


def example_atari() -> torch.Tensor:
    """Return Atari observation stack.

    Returns
    -------
    torch.Tensor
        Observation tensor.
    """

    return torch.randn(1, 4, 32, 32)


def build_sb3_mlp_extractor() -> nn.Module:
    """Build Stable-Baselines3 MLP extractor.

    Returns
    -------
    nn.Module
        Actor-critic MLP extractor.
    """

    return MlpExtractor()


def example_mlp() -> torch.Tensor:
    """Return flat observation.

    Returns
    -------
    torch.Tensor
        Observation tensor.
    """

    return torch.randn(1, 10)


def build_scdeepcluster() -> nn.Module:
    """Build scDeepCluster-style zero-inflated autoencoder.

    Returns
    -------
    nn.Module
        Single-cell VAE.
    """

    return SingleCellVAE(False)


def build_scgen() -> nn.Module:
    """Build scGen-style conditional single-cell VAE.

    Returns
    -------
    nn.Module
        Conditional single-cell VAE.
    """

    return SingleCellVAE(True)


def example_genes() -> torch.Tensor:
    """Return gene-count input.

    Returns
    -------
    torch.Tensor
        Gene features.
    """

    return torch.rand(2, 12) * 5.0


def example_genes_cond() -> torch.Tensor:
    """Return conditioned gene-count input.

    Returns
    -------
    torch.Tensor
        Gene and condition features.
    """

    genes = torch.rand(2, 12) * 5.0
    cond = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    return torch.cat([genes, cond], dim=-1)


MENAGERIE_ENTRIES = [
    ("RivaGAN", "build_rivagan", "example_rivagan", "2019", "video/watermarking"),
    ("rmvsnet_depth", "build_rmvsnet_depth", "example_rmvsnet_depth", "2019", "vision/depth"),
    ("DARTS RNN derived network", "build_darts_rnn_derived", "example_tokens", "2019", "nas/rnn"),
    ("DARTS RNN search network", "build_darts_rnn_search", "example_tokens", "2019", "nas/rnn"),
    ("ElmanRNNLM", "build_elman_rnn_lm", "example_tokens", "1990", "sequence/rnn"),
    ("rockpool_lif_torch", "build_rockpool_lif_torch", "example_lif", "2020", "spiking"),
    (
        "Routing-Transformer",
        "build_routing_transformer",
        "example_routing_tokens",
        "2020",
        "attention",
    ),
    ("rwkv6_block", "build_rwkv6_block", "example_rwkv_tokens", "2024", "sequence/rwkv"),
    ("rwkv7_block", "build_rwkv7_block", "example_rwkv_tokens", "2025", "sequence/rwkv"),
    ("s2cnn (Spherical CNN)", "build_spherical_cnn", "example_spherical", "2018", "geometric"),
    ("spherical_cnn", "build_spherical_cnn", "example_spherical", "2018", "geometric"),
    ("s4_s4torch", "build_s4_s4torch", "example_sequence", "2021", "state-space"),
    ("s5", "build_s5", "example_sequence", "2022", "state-space"),
    ("s5_layer", "build_s5", "example_sequence", "2022", "state-space"),
    (
        "sagan_heykeetae_discriminator",
        "build_sagan_discriminator",
        "example_image16",
        "2018",
        "gan/vision",
    ),
    ("sagan_heykeetae_generator", "build_sagan_generator", "example_latent", "2018", "gan/vision"),
    (
        "self_attention_gan_johndpope_generator",
        "build_sagan_generator",
        "example_latent",
        "2018",
        "gan/vision",
    ),
    (
        "lama_fourier_generator",
        "build_lama_fourier_generator",
        "example_lama",
        "2021",
        "vision/inpainting",
    ),
    ("SAINT-tabular", "build_saint_tabular", "example_tabular", "2021", "tabular"),
    (
        "SAM_HQ_ViT_B_Source",
        "build_sam_hq_vit_b_source",
        "example_sam_hq",
        "2023",
        "vision/segmentation",
    ),
    ("sb3_NatureCNN", "build_sb3_nature_cnn", "example_atari", "2015", "rl"),
    ("sb3_MlpExtractor", "build_sb3_mlp_extractor", "example_mlp", "2017", "rl"),
    ("scDeepCluster", "build_scdeepcluster", "example_genes", "2019", "single-cell"),
    ("scGen", "build_scgen", "example_genes_cond", "2019", "single-cell"),
]
