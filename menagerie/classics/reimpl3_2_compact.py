"""Compact faithful classics for dependency-gated reimpl3_2 targets.

This module keeps the load-bearing primitives of each source family while using
small random-initialized PyTorch modules suitable for TorchLens rendering:
spiking decision transformers, DSINE ray-conditioned normal refinement, DUSt3R
pointmap regression, EdgeSAM prompt-conditioned masks, EG3D/GRAM triplanes,
Mip-NeRF integrated positional encoding, Basenji/Enformer genomics towers,
ESM protein transformers, ESPnet Branchformer/FastSpeech/DPTNet/JETS blocks,
ExcelFormer semi-permeable attention, GATr-style geometric products, GEARS
perturbation graph propagation, Metric3D depth-normal heads, InGram relation
graph updates, and SE(3)-style scalar/vector attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpikeFn(torch.autograd.Function):
    """Straight-through binary spike function."""

    @staticmethod
    def forward(ctx: object, x: torch.Tensor) -> torch.Tensor:
        """Return binary spikes from membrane values."""

        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx: object, grad_output: torch.Tensor) -> torch.Tensor:
        """Pass gradients through the threshold."""

        return grad_output


def spike(x: torch.Tensor) -> torch.Tensor:
    """Apply a straight-through spike nonlinearity."""

    return SpikeFn.apply(x)


class SpikingAttentionBlock(nn.Module):
    """Spike-driven temporal and positional attention block."""

    def __init__(self, dim: int = 32, heads: int = 4) -> None:
        """Initialize TSSA/PSSA-style projections."""

        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.norm = nn.BatchNorm1d(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.pos_gate = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.proj = nn.Linear(dim, dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run spike-gated self-attention over a sequence."""

        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        pos = spike(self.pos_gate(x.transpose(1, 2)).transpose(1, 2))
        q, k, v = self.qkv(spike(x + pos)).chunk(3, dim=-1)
        bsz, length, dim = q.shape
        q = q.view(bsz, length, self.heads, dim // self.heads).transpose(1, 2)
        k = k.view(bsz, length, self.heads, dim // self.heads).transpose(1, 2)
        v = v.view(bsz, length, self.heads, dim // self.heads).transpose(1, 2)
        attn = torch.softmax(torch.matmul(spike(q), spike(k).transpose(-1, -2)) * self.scale, -1)
        out = torch.matmul(attn, spike(v)).transpose(1, 2).reshape(bsz, length, dim)
        return x + self.proj(out) + self.ff(x)


class SoftmaxFreeSpikeSelfAttention(nn.Module):
    """Spikformer SPS plus softmax-free spike self-attention."""

    def __init__(self, dim: int = 32) -> None:
        """Initialize spiking patch stem and SSA projections."""

        super().__init__()
        self.sps = nn.Sequential(
            nn.Conv2d(3, dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(dim),
        )
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.head = nn.Linear(dim, 10)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Classify images using softmax-free binary spike attention."""

        feat = spike(self.sps(image)).flatten(2).transpose(1, 2)
        q = spike(self.q(feat))
        k = spike(self.k(feat))
        v = spike(self.v(feat))
        ssa = torch.matmul(q, k.transpose(1, 2)) @ v / q.shape[-1]
        return self.head((feat + ssa).mean(1))


class EfficientSpikeFormer(nn.Module):
    """SDT-v3/E-SpikeFormer with linear spike-driven attention."""

    def __init__(self, dim: int = 32) -> None:
        """Initialize patch projection and linear spike-driven attention."""

        super().__init__()
        self.patch = nn.Conv2d(3, dim, 4, 4)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.head = nn.Linear(dim, 10)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Run linear spike-driven attention without quadratic softmax maps."""

        x = spike(self.patch(image)).flatten(2).transpose(1, 2)
        q, k, v = self.qkv(x).chunk(3, -1)
        q = spike(q)
        k = spike(k)
        v = spike(v)
        kv = torch.bmm(k.transpose(1, 2), v)
        z = torch.bmm(q, k.sum(1, keepdim=True).transpose(1, 2)).clamp_min(1.0)
        h = self.proj(torch.bmm(q, kv) / z)
        return self.head(h.mean(1))


class DecisionSpikeFormer(nn.Module):
    """Decision SpikeFormer for offline RL trajectories."""

    def __init__(self, state_dim: int = 17, act_dim: int = 6, dim: int = 32) -> None:
        """Initialize return/state/action token embeddings and spiking blocks."""

        super().__init__()
        self.ret = nn.Linear(1, dim)
        self.state = nn.Linear(state_dim, dim)
        self.action = nn.Linear(act_dim, dim)
        self.pos = nn.Parameter(torch.randn(1, 64, dim) * 0.02)
        self.blocks = nn.ModuleList([SpikingAttentionBlock(dim), SpikingAttentionBlock(dim)])
        self.head = nn.Linear(dim, act_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict actions from ``(return, state, previous-action)`` tokens."""

        ret = self.ret(x[..., :1])
        state = self.state(x[..., 1:18])
        act = self.action(torch.zeros(x.shape[0], x.shape[1], 6, device=x.device, dtype=x.dtype))
        h = torch.stack((ret, state, act), dim=2).flatten(1, 2)
        h = h + self.pos[:, : h.shape[1]]
        for block in self.blocks:
            h = block(h)
        return self.head(h[:, 1::3])


class DSINENormalNet(nn.Module):
    """DSINE ray-conditioned ConvGRU surface-normal estimator."""

    def __init__(self, channels: int = 24, iters: int = 3) -> None:
        """Initialize image/ray stem and recurrent normal refinement."""

        super().__init__()
        self.iters = iters
        self.stem = nn.Sequential(
            nn.Conv2d(6, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
        )
        self.gru = nn.Conv2d(channels + 3, channels * 3, 3, padding=1)
        self.out = nn.Conv2d(channels, 3, 1)
        self.rel_rot = nn.Conv2d(12, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate camera-facing normals from RGB images."""

        bsz, _, height, width = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, height, device=x.device),
            torch.linspace(-1, 1, width, device=x.device),
            indexing="ij",
        )
        rays = F.normalize(torch.stack((xx, yy, torch.ones_like(xx)), 0), dim=0)
        rays = rays.expand(bsz, -1, -1, -1)
        h = self.stem(torch.cat((x, rays), dim=1))
        normal = F.normalize(rays, dim=1)
        for _ in range(self.iters):
            z, r, q = self.gru(torch.cat((h, normal), dim=1)).chunk(3, dim=1)
            h = (1 - torch.sigmoid(z)) * h + torch.sigmoid(z) * torch.tanh(q + r * h)
            normal = F.normalize(self.out(h) + rays, dim=1)
            north = F.pad(normal[:, :, :-1], (0, 0, 1, 0))
            west = F.pad(normal[:, :, :, :-1], (1, 0, 0, 0))
            rel = self.rel_rot(torch.cat((normal, north, west, rays), dim=1))
            normal = F.normalize(normal + torch.cross(rel, normal, dim=1) + rays, dim=1)
            facing = torch.where((normal * rays).sum(1, keepdim=True) < 0, -normal, normal)
            normal = F.normalize(facing, dim=1)
        return normal


class PointmapTransformer(nn.Module):
    """DUSt3R-style two-view transformer regressing dense 3D pointmaps."""

    def __init__(self, dim: int = 32) -> None:
        """Initialize patch encoders, cross decoder, point and confidence heads."""

        super().__init__()
        self.patch = nn.Conv2d(3, dim, 4, 4)
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 1)
        self.decoder = nn.TransformerEncoder(layer, 1)
        self.point = nn.Linear(dim, 3)
        self.conf = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return pointmaps and confidence for a pair of images."""

        bsz, views, channels, height, width = x.shape
        tokens = (
            self.patch(x.reshape(bsz * views, channels, height, width)).flatten(2).transpose(1, 2)
        )
        enc = self.encoder(tokens).reshape(bsz, views, -1, tokens.shape[-1])
        fused = torch.cat((enc[:, 0], enc[:, 1]), dim=1)
        dec = self.decoder(fused)
        pts = self.point(dec).transpose(1, 2).reshape(bsz, 3, 2, height // 4, width // 4)
        conf = (
            torch.sigmoid(self.conf(dec))
            .transpose(1, 2)
            .reshape(bsz, 1, 2, height // 4, width // 4)
        )
        return torch.cat((pts, conf), dim=1)


class EdgeSAMCompact(nn.Module):
    """EdgeSAM with compact CNN image encoder and prompt-conditioned mask decoder."""

    def __init__(self, dim: int = 32) -> None:
        """Initialize distilled encoder, prompt encoder, and mask decoder."""

        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, dim, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, stride=2, padding=1, groups=dim),
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
        )
        self.prompt = nn.Linear(4, dim)
        self.mask = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, padding=1), nn.GELU(), nn.Conv2d(dim, 1, 1)
        )
        self.iou = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Segment an image with an internally supplied box prompt."""

        feat = self.encoder(x)
        prompt = torch.tensor([[0.2, 0.2, 0.8, 0.8]], device=x.device, dtype=x.dtype).expand(
            x.shape[0], -1
        )
        p = (
            self.prompt(prompt)
            .view(x.shape[0], -1, 1, 1)
            .expand(-1, -1, feat.shape[2], feat.shape[3])
        )
        mask = self.mask(torch.cat((feat, p), dim=1))
        score = self.iou((feat * p).mean((2, 3))).view(x.shape[0], 1, 1, 1)
        return F.interpolate(mask, size=x.shape[-2:], mode="bilinear") * torch.sigmoid(score)


class DynamicFilterNetwork(nn.Module):
    """Dynamic Filter Network predicting per-pixel local filters."""

    def __init__(self, channels: int = 12, kernel: int = 3) -> None:
        """Initialize filter-generating and feature branches."""

        super().__init__()
        self.kernel = kernel
        self.features = nn.Conv2d(3, channels, 3, padding=1)
        self.filter_gen = nn.Conv2d(3, kernel * kernel, 3, padding=1)
        self.head = nn.Linear(channels, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatially varying generated filters and classify."""

        feat = self.features(x)
        patches = F.unfold(feat, self.kernel, padding=self.kernel // 2).view(
            x.shape[0], feat.shape[1], self.kernel * self.kernel, x.shape[2], x.shape[3]
        )
        filt = torch.softmax(self.filter_gen(x), dim=1).unsqueeze(1)
        y = (patches * filt).sum(2)
        return self.head(y.mean((2, 3)))


class EMSYOLO(nn.Module):
    """EMS-YOLO compact event/spiking detector over timesteps."""

    def __init__(self, classes: int = 5) -> None:
        """Initialize event membrane stem and YOLO heads."""

        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3, padding=1)
        self.neck = nn.Sequential(nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.SiLU())
        self.obj = nn.Conv2d(24, 3, 1)
        self.box = nn.Conv2d(24, 12, 1)
        self.cls = nn.Conv2d(24, classes * 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Accumulate spiking event frames and emit dense YOLO predictions."""

        mem = torch.zeros(x.shape[1], 16, x.shape[-2], x.shape[-1], device=x.device, dtype=x.dtype)
        for frame in x:
            mem = 0.7 * mem + self.stem(frame)
            mem = spike(mem - 0.25)
        feat = self.neck(mem)
        return torch.cat((self.obj(feat), self.box(feat), self.cls(feat)), dim=1)


class EG3DGenerator(nn.Module):
    """EG3D triplane generator plus OSG decoder."""

    def __init__(self, z_dim: int = 32, channels: int = 8) -> None:
        """Initialize mapping, triplane synthesis, and renderer heads."""

        super().__init__()
        self.channels = channels
        self.mapping = nn.Sequential(nn.Linear(z_dim, 64), nn.LeakyReLU(0.2), nn.Linear(64, 64))
        self.to_planes = nn.Linear(64, 3 * channels * 16 * 16)
        self.decoder = nn.Sequential(nn.Linear(channels, 32), nn.Softplus(), nn.Linear(32, 4))

    def _sample_planes(self, planes: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Bilinearly sample XY, XZ, and YZ planes."""

        xy = coords[..., [0, 1]]
        xz = coords[..., [0, 2]]
        yz = coords[..., [1, 2]]
        grids = torch.stack((xy, xz, yz), dim=1)
        vals = []
        for idx in range(3):
            vals.append(
                F.grid_sample(
                    planes[:, idx], grids[:, idx].unsqueeze(2), align_corners=False
                ).squeeze(-1)
            )
        return torch.stack(vals, dim=0).sum(0).transpose(1, 2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Render RGB-density samples from latent vectors."""

        planes = self.to_planes(self.mapping(z)).view(z.shape[0], 3, self.channels, 16, 16)
        coords = torch.rand(z.shape[0], 32, 3, device=z.device, dtype=z.dtype) * 2 - 1
        rgba = self.decoder(self._sample_planes(planes, coords))
        return rgba.mean(1)


class MipNeRFEncoding(nn.Module):
    """Mip-NeRF integrated positional encoding for conical frustums."""

    def __init__(self, bands: int = 6) -> None:
        """Store frequency bands."""

        super().__init__()
        self.register_buffer("freq", 2.0 ** torch.arange(bands).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode mean and variance with Gaussian attenuation."""

        mean, var = x[..., :3], x[..., 3:].clamp_min(1e-6)
        scaled = mean.unsqueeze(-2) * self.freq.view(1, 1, -1, 1)
        atten = torch.exp(-0.5 * var.unsqueeze(-2) * self.freq.view(1, 1, -1, 1).pow(2))
        return torch.cat((torch.sin(scaled) * atten, torch.cos(scaled) * atten), dim=-1).flatten(-2)


class GenomicTower(nn.Module):
    """Basenji2/Enformer compact sequence-to-track predictor."""

    def __init__(self, transformer: bool = True, targets: int = 8) -> None:
        """Initialize convolutional, dilated, and optional attention blocks."""

        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(4, 32, 15, padding=7),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 48, 5, padding=2),
            nn.GELU(),
            nn.MaxPool1d(2),
        )
        self.dilated = nn.ModuleList(
            [nn.Conv1d(48, 48, 3, padding=2**i, dilation=2**i) for i in range(3)]
        )
        layer = nn.TransformerEncoderLayer(48, 4, 96, batch_first=True)
        self.attn = nn.TransformerEncoder(layer, 1) if transformer else nn.Identity()
        self.head = nn.Conv1d(48, targets, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict genomic tracks from one-hot DNA."""

        h = self.conv(x.transpose(1, 2) if x.shape[-1] == 4 else x)
        for conv in self.dilated:
            h = h + F.gelu(conv(h))
        h = self.attn(h.transpose(1, 2)).transpose(1, 2)
        return self.head(h).mean(-1)


class ESMProteinModel(nn.Module):
    """ESM-style protein encoder with optional multimodal tracks."""

    def __init__(self, tracks: int = 1, vocab: int = 33, dim: int = 48) -> None:
        """Initialize token-track embeddings and transformer encoder."""

        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(vocab, dim) for _ in range(tracks)])
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.lm = nn.Linear(dim, vocab)
        self.structure = nn.Linear(dim, 3)
        self.function = nn.Linear(dim, 8)
        self.geom = nn.Linear(4, dim)
        self.geom_value = nn.Linear(3, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode protein tokens and return masked-LM logits."""

        if x.ndim == 2:
            x = x.unsqueeze(1).expand(-1, len(self.embeds), -1)
        h = sum(
            embed(x[:, idx].clamp(0, embed.num_embeddings - 1))
            for idx, embed in enumerate(self.embeds)
        )
        pos = torch.linspace(-1, 1, h.shape[1], device=h.device, dtype=h.dtype)
        rel = (pos[:, None] - pos[None, :]).abs()
        geom_bias = self.geom(torch.stack((rel, rel.sin(), rel.cos(), rel.square()), dim=-1)).mean(
            1
        )
        struct_tokens = x[:, min(1, x.shape[1] - 1)].to(h.dtype)
        phase = struct_tokens / max(self.embeds[0].num_embeddings - 1, 1)
        coords = torch.stack((pos.unsqueeze(0).expand_as(phase), phase.sin(), phase.cos()), dim=-1)
        dist = torch.cdist(coords, coords)
        geom_attn = torch.softmax(-dist + rel.unsqueeze(0), dim=-1)
        geom_context = torch.bmm(geom_attn, self.geom_value(coords))
        h = self.encoder(h + geom_bias.unsqueeze(0) + geom_context)
        seq_logits = self.lm(h)
        coords = self.structure(h)
        func = self.function(h.mean(1))
        return torch.cat((seq_logits.mean(1), coords.mean(1), func), dim=-1)


class BranchformerTransducer(nn.Module):
    """Branchformer encoder with RNNT predictor and joiner."""

    def __init__(self, vocab: int = 32, dim: int = 48) -> None:
        """Initialize encoder, predictor, and transducer joiner."""

        super().__init__()
        self.encoder = BranchformerEncoder(dim)
        self.predict = nn.GRU(vocab, dim, batch_first=True)
        self.joiner = nn.Linear(dim, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Produce compact RNNT joiner logits from speech frames."""

        enc = self.encoder(x)
        blank_tokens = torch.zeros(x.shape[0], enc.shape[1], 32, device=x.device, dtype=x.dtype)
        pred = self.predict(blank_tokens)[0]
        return self.joiner(torch.tanh(enc + pred))


class BranchformerCTC(nn.Module):
    """Branchformer encoder with CTC projection head."""

    def __init__(self, vocab: int = 32) -> None:
        """Initialize encoder and framewise CTC classifier."""

        super().__init__()
        self.encoder = BranchformerEncoder()
        self.ctc = nn.Linear(48, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return log probabilities for CTC alignment."""

        return F.log_softmax(self.ctc(self.encoder(x)), dim=-1)


class ConformerFastSpeech2Compact(nn.Module):
    """FastSpeech2 with Conformer convolutional encoder and length regulator."""

    def __init__(self) -> None:
        """Initialize Conformer-style convolution and FastSpeech2 decoder."""

        super().__init__()
        self.core = FastSpeech2Compact()
        self.conv = nn.Conv1d(48, 48, 7, padding=3, groups=48)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode tokens with Conformer convolution before regulated decoding."""

        emb = self.core.embed(x.long().clamp(0, 79))
        h = self.core.encoder(emb)
        h = h + self.conv(h.transpose(1, 2)).transpose(1, 2)
        return self.core.decode_regulated(h)


class CgMLP(nn.Module):
    """Convolutional gated MLP branch used by Branchformer."""

    def __init__(self, dim: int) -> None:
        """Initialize channel projections and depthwise convolution."""

        super().__init__()
        self.proj = nn.Linear(dim, dim * 2)
        self.dw = nn.Conv1d(dim, dim, 7, padding=3, groups=dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the cgMLP local-context branch."""

        u, v = self.proj(x).chunk(2, -1)
        v = self.dw(v.transpose(1, 2)).transpose(1, 2)
        return self.out(u * torch.sigmoid(v))


class BranchformerEncoder(nn.Module):
    """Branchformer encoder with attention and cgMLP branches."""

    def __init__(self, dim: int = 48) -> None:
        """Initialize subsampling and parallel-branch encoder layers."""

        super().__init__()
        self.inp = nn.Linear(80, dim)
        self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.cgmlp = CgMLP(dim)
        self.merge = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode speech features with merged global/local context."""

        h = self.inp(x)
        a = self.attn(h, h, h)[0]
        c = self.cgmlp(h)
        return self.norm(h + self.merge(torch.cat((a, c), dim=-1)))


class FastSpeech2Compact(nn.Module):
    """FastSpeech/FastSpeech2 non-autoregressive TTS acoustic model."""

    def __init__(self, vocab: int = 80, dim: int = 48) -> None:
        """Initialize encoder, duration/pitch/energy predictors, and decoder."""

        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 1)
        self.duration = nn.Conv1d(dim, 1, 3, padding=1)
        self.pitch = nn.Conv1d(dim, 1, 3, padding=1)
        self.energy = nn.Conv1d(dim, 1, 3, padding=1)
        self.var_embed = nn.Linear(3, dim)
        self.decoder = nn.TransformerEncoder(layer, 1)
        self.mel = nn.Linear(dim, 80)

    def decode_regulated(self, h: torch.Tensor) -> torch.Tensor:
        """Expand hidden states with a duration length regulator and decode mels."""

        hc = h.transpose(1, 2)
        dur = F.softplus(self.duration(hc)).transpose(1, 2)
        pitch = self.pitch(hc).transpose(1, 2)
        energy = self.energy(hc).transpose(1, 2)
        h = h + self.var_embed(torch.cat((dur, pitch, energy), dim=-1))
        repeat_a = h
        repeat_b = h.repeat_interleave(2, dim=1)
        selector = (dur.mean(2, keepdim=True) > 0.75).repeat_interleave(2, dim=1)
        regulated = torch.where(selector, repeat_b, repeat_a.repeat_interleave(2, dim=1))
        return self.mel(self.decoder(regulated))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate mel frames with variance-adaptor conditioning."""

        h = self.encoder(self.embed(x.long().clamp(0, 79)))
        return self.decode_regulated(h)


class ConvTasNetCompact(nn.Module):
    """ConvTasNet encoder, TCN separator, and mask decoder."""

    def __init__(self, dim: int = 32) -> None:
        """Initialize temporal convolutional mask network."""

        super().__init__()
        self.enc = nn.Conv1d(1, dim, 16, stride=8, padding=4)
        self.tcn = nn.ModuleList(
            [nn.Conv1d(dim, dim, 3, padding=2**idx, dilation=2**idx) for idx in range(4)]
        )
        self.mask = nn.Conv1d(dim, dim * 2, 1)
        self.dec = nn.ConvTranspose1d(dim, 1, 16, stride=8, padding=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Separate waveform with learned TCN masks."""

        enc = F.relu(self.enc(x.unsqueeze(1)))
        h = enc
        for conv in self.tcn:
            h = h + F.relu(conv(h))
        masks = torch.sigmoid(self.mask(h)).chunk(2, 1)
        return torch.stack([self.dec(enc * mask).squeeze(1) for mask in masks], dim=1)


class RNNLanguageModel(nn.Module):
    """ESPnet-style recurrent language model."""

    def __init__(self, vocab: int = 32, dim: int = 48) -> None:
        """Initialize embedding, LSTM, and token projection."""

        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.rnn = nn.LSTM(dim, dim, batch_first=True)
        self.out = nn.Linear(dim, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict next-token logits with recurrent hidden updates."""

        h = self.rnn(self.embed(x.long().clamp(0, 31)))[0]
        return self.out(h)


class TransformerLanguageModel(nn.Module):
    """Token Transformer language model."""

    def __init__(self, vocab: int = 32, dim: int = 48) -> None:
        """Initialize causal token Transformer."""

        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 1)
        self.out = nn.Linear(dim, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict token logits with a causal attention mask."""

        h = self.embed(x.long().clamp(0, 31))
        mask = torch.triu(torch.ones(h.shape[1], h.shape[1], device=h.device, dtype=torch.bool), 1)
        return self.out(self.encoder(h, mask=mask))


class DPTNetCompact(nn.Module):
    """Dual-path Transformer speech separation/enhancement network."""

    def __init__(self, dim: int = 32) -> None:
        """Initialize encoder, intra/inter transformers, and mask head."""

        super().__init__()
        self.enc = nn.Conv1d(1, dim, 8, stride=4, padding=2)
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True)
        self.intra = nn.TransformerEncoder(layer, 1)
        self.inter = nn.TransformerEncoder(layer, 1)
        self.mask = nn.Conv1d(dim, dim * 2, 1)
        self.dec = nn.ConvTranspose1d(dim, 1, 8, stride=4, padding=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Separate a waveform with chunked dual-path attention."""

        h = self.enc(x.unsqueeze(1))
        chunks = h.unfold(2, 8, 4).permute(0, 2, 3, 1)
        bsz, nchunks, clen, dim = chunks.shape
        intra = self.intra(chunks.reshape(bsz * nchunks, clen, dim)).reshape(
            bsz, nchunks, clen, dim
        )
        inter = self.inter(intra.transpose(1, 2).reshape(bsz * clen, nchunks, dim)).reshape(
            bsz, clen, nchunks, dim
        )
        merged = inter.permute(0, 3, 2, 1).mean(-1)
        merged = F.interpolate(merged, size=h.shape[-1], mode="linear", align_corners=False)
        masks = torch.sigmoid(self.mask(merged)).chunk(2, dim=1)
        return torch.stack([self.dec(h * mask).squeeze(1) for mask in masks], dim=1)


class JETSCompact(nn.Module):
    """JETS: FastSpeech2 acoustic model jointly feeding a HiFi-GAN-like vocoder."""

    def __init__(self) -> None:
        """Initialize text-to-mel and waveform generator."""

        super().__init__()
        self.tts = FastSpeech2Compact()
        self.up = nn.Sequential(
            nn.ConvTranspose1d(80, 32, 8, stride=4, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 1, 7, padding=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Synthesize waveform samples from token ids."""

        mel = self.tts(x).transpose(1, 2)
        return self.up(mel).squeeze(1)


class VITSCompact(nn.Module):
    """VITS VAE/flow prior, stochastic duration, and HiFi-GAN-like decoder."""

    def __init__(self, dim: int = 48) -> None:
        """Initialize posterior/prior encoders, flow, duration, and decoder."""

        super().__init__()
        self.text = nn.Embedding(80, dim)
        self.posterior = nn.Conv1d(80, dim * 2, 3, padding=1)
        self.flow_scale = nn.Linear(dim, dim)
        self.flow_shift = nn.Linear(dim, dim)
        self.duration = nn.Conv1d(dim, 1, 3, padding=1)
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(dim, 32, 8, stride=4, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 1, 7, padding=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Synthesize waveform through a compact VITS latent flow path."""

        text = self.text(x.long().clamp(0, 79)).transpose(1, 2)
        mel_hint = F.one_hot(x.long().clamp(0, 79), num_classes=80).float().transpose(1, 2)
        mean, log_scale = self.posterior(mel_hint).chunk(2, 1)
        z = mean + torch.tanh(log_scale) * 0.1
        scale = torch.tanh(self.flow_scale(text.transpose(1, 2))).transpose(1, 2)
        shift = self.flow_shift(text.transpose(1, 2)).transpose(1, 2)
        flowed = z * torch.exp(0.1 * scale) + shift
        dur = F.softplus(self.duration(text))
        regulated = flowed.repeat_interleave(2, dim=2) * dur.repeat_interleave(2, dim=2).sigmoid()
        return self.dec(regulated).squeeze(1)


class PatchFusionDepth(nn.Module):
    """PatchFusion tiled depth estimation with global-local fusion."""

    def __init__(self, dim: int = 24) -> None:
        """Initialize global and tile depth branches."""

        super().__init__()
        self.global_enc = nn.Conv2d(3, dim, 3, stride=2, padding=1)
        self.tile_enc = nn.Conv2d(3, dim, 3, padding=1)
        self.fuse = nn.Conv2d(dim * 2, dim, 1)
        self.depth = nn.Conv2d(dim, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fuse overlapping tile predictions with a global depth prior."""

        global_feat = F.interpolate(F.gelu(self.global_enc(x)), x.shape[-2:], mode="bilinear")
        tiles = F.unfold(x, kernel_size=16, stride=8).transpose(1, 2)
        tiles = tiles.reshape(-1, 3, 16, 16)
        tile_depth = self.depth(F.gelu(self.tile_enc(tiles)))
        columns = tile_depth.flatten(1).unsqueeze(0).transpose(1, 2)
        folded = F.fold(columns, x.shape[-2:], kernel_size=16, stride=8)
        norm_cols = torch.ones_like(tile_depth).flatten(1).unsqueeze(0).transpose(1, 2)
        norm = F.fold(norm_cols, x.shape[-2:], 16, stride=8)
        local = folded / norm.clamp_min(1.0)
        fused = self.fuse(
            torch.cat((global_feat, local.expand(x.shape[0], global_feat.shape[1], -1, -1)), dim=1)
        )
        return self.depth(F.gelu(fused))


class ExcelFormerCompact(nn.Module):
    """ExcelFormer tabular model with semi-permeable attention."""

    def __init__(self, fields: int = 16, dim: int = 32) -> None:
        """Initialize field embeddings and causal/semi-permeable attention."""

        super().__init__()
        self.field = nn.Parameter(torch.randn(fields, dim) * 0.02)
        self.value = nn.Linear(1, dim)
        self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.head = nn.Linear(dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify tabular rows with ordered feature attention."""

        h = self.value(x.unsqueeze(-1)) + self.field[: x.shape[1]]
        mask = torch.triu(torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool), 1)
        h = self.attn(h, h, h, attn_mask=mask)[0]
        return self.head(h.mean(1))


class EnergyTransformerCompact(nn.Module):
    """Energy Transformer with recurrent energy minimization steps."""

    def __init__(self, dim: int = 32, steps: int = 3) -> None:
        """Initialize attention energy and update projections."""

        super().__init__()
        self.steps = steps
        self.inp = nn.Linear(32, dim)
        self.energy = nn.Linear(dim, 1)
        self.update = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.head = nn.Linear(dim, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Refine tokens by descending an attention-defined energy."""

        h = self.inp(x)
        for _ in range(self.steps):
            attn = self.update(h, h, h)[0]
            h = h - 0.1 * torch.tanh(attn) - 0.01 * self.energy(h)
        return self.head(h.mean(1))


class GateNetCompact(nn.Module):
    """GateNet salient-object detector with gated multi-level features."""

    def __init__(self, channels: int = 24) -> None:
        """Initialize backbone features and gated decoder."""

        super().__init__()
        self.low = nn.Conv2d(3, channels, 3, padding=1)
        self.high = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        self.gate = nn.Conv2d(channels * 2, channels, 1)
        self.out = nn.Conv2d(channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict a saliency map with high-to-low feature gates."""

        low = F.gelu(self.low(x))
        high = F.gelu(self.high(low))
        high_up = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        gate = torch.sigmoid(self.gate(torch.cat((low, high_up), dim=1)))
        return self.out(low * gate + high_up)


class GATrCompact(nn.Module):
    """Geometric Algebra Transformer over 16D projective multivectors."""

    def __init__(self, dim: int = 16) -> None:
        """Initialize equivariant bilinear product and token mixer."""

        super().__init__()
        self.left = nn.Linear(dim, dim, bias=False)
        self.right = nn.Linear(dim, dim, bias=False)
        table = torch.zeros(dim, dim, dim)
        for i in range(dim):
            for j in range(dim):
                table[i, j, i ^ j] = -1.0 if ((i & j).bit_count() % 2) else 1.0
        self.register_buffer("gp_table", table)
        self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.head = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix multivectors using a geometric-product proxy."""

        left = self.left(x)
        right = self.right(x)
        gp = torch.einsum("bni,bnj,ijk->bnk", left, right, self.gp_table)
        h = self.attn(gp, gp, gp)[0]
        return self.head(h)


class GEARSCompact(nn.Module):
    """GEARS perturbation-response graph neural network."""

    def __init__(self, genes: int = 16, dim: int = 24) -> None:
        """Initialize gene embeddings, perturb embeddings, and message passing."""

        super().__init__()
        self.gene = nn.Embedding(genes, dim)
        self.pert = nn.Embedding(genes, dim)
        self.msg = nn.Linear(dim * 2, dim)
        self.out = nn.Linear(dim, 1)
        idx = torch.arange(genes)
        adj = (idx[:, None] - idx[None, :]).abs().le(2).float()
        self.register_buffer("go_graph", adj / adj.sum(1, keepdim=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict expression shifts from expression and perturbation mask."""

        expr, pert = x[..., 0], x[..., 1].long().clamp(0, 1)
        idx = torch.arange(expr.shape[1], device=x.device).expand(expr.shape[0], -1)
        h = self.gene(idx) * expr.unsqueeze(-1) + self.pert(idx) * pert.unsqueeze(-1)
        graph_msg = torch.matmul(self.go_graph[: h.shape[1], : h.shape[1]], h)
        return self.out(torch.relu(self.msg(torch.cat((h, graph_msg), dim=-1)))).squeeze(-1)


class GRAMGenerator(nn.Module):
    """GRAM generator with manifold radiance anchors and implicit renderer."""

    def __init__(self) -> None:
        """Initialize latent mapping and radiance manifold decoder."""

        super().__init__()
        self.anchor = nn.Linear(32, 8 * 3)
        self.radiance = nn.Sequential(nn.Linear(32 + 3, 32), nn.Softplus(), nn.Linear(32, 4))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Render a compact GRAM sample."""

        anchors = self.anchor(z).view(z.shape[0], 8, 3)
        coords = (
            torch.linspace(-1, 1, 16, device=z.device, dtype=z.dtype)
            .view(1, 16, 1)
            .expand(z.shape[0], -1, 3)
        )
        dist = (coords[:, :, None] - anchors[:, None]).pow(2).sum(-1)
        weights = torch.softmax(-dist, dim=-1)
        manifold = torch.bmm(weights, anchors)
        latent = z[:, None].expand(-1, manifold.shape[1], -1)
        return self.radiance(torch.cat((latent, manifold), dim=-1)).mean(1)


class Metric3DNormalHead(nn.Module):
    """Metric3D depth model with depth and surface-normal heads."""

    def __init__(self, dim: int = 24) -> None:
        """Initialize hourglass-like encoder and metric heads."""

        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, dim, 3, padding=1), nn.GELU(), nn.Conv2d(dim, dim, 3, stride=2, padding=1)
        )
        self.up = nn.ConvTranspose2d(dim, dim, 4, stride=2, padding=1)
        self.depth = nn.Conv2d(dim, 1, 1)
        self.normal = nn.Conv2d(dim, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict metric depth and normalized surface normals."""

        h = F.gelu(self.up(F.gelu(self.enc(x))))
        return torch.cat((F.softplus(self.depth(h)), F.normalize(self.normal(h), dim=1)), dim=1)


class InGramCompact(nn.Module):
    """InGram inductive knowledge-graph embedding with relation graph updates."""

    def __init__(self, ents: int = 32, rels: int = 8, dim: int = 24) -> None:
        """Initialize entity/relation embeddings and update layers."""

        super().__init__()
        self.ent = nn.Embedding(ents, dim)
        self.rel = nn.Embedding(rels, dim)
        self.rel_update = nn.Linear(dim * 2, dim)
        self.update = nn.Linear(dim * 3, dim)
        self.score = nn.Linear(dim, 1)

    def forward(self, triples: torch.Tensor) -> torch.Tensor:
        """Score triples after relation-aware entity updates."""

        h = self.ent(triples[..., 0].clamp(0, 31))
        r = self.rel(triples[..., 1].clamp(0, 7))
        t = self.ent(triples[..., 2].clamp(0, 31))
        rel_affinity = torch.softmax(torch.bmm(r, r.transpose(1, 2)) / r.shape[-1] ** 0.5, dim=-1)
        rel_neighbors = torch.bmm(rel_affinity, r)
        rel_graph = torch.relu(self.rel_update(torch.cat((r, rel_neighbors), dim=-1)))
        ent_affinity = torch.softmax(
            torch.bmm(h + rel_graph, t.transpose(1, 2)) / h.shape[-1] ** 0.5, dim=-1
        )
        ent_neighbors = torch.bmm(ent_affinity, t)
        ent_graph = torch.relu(self.update(torch.cat((h, rel_graph, ent_neighbors), dim=-1)))
        updated = ent_graph + h * torch.sigmoid(rel_graph)
        return self.score(updated).squeeze(-1)


class SE3TransformerCompact(nn.Module):
    """SE(3)-Transformer-style scalar/vector attention on point clouds."""

    def __init__(self, dim: int = 24) -> None:
        """Initialize scalar and vector channels."""

        super().__init__()
        self.scalar = nn.Linear(4, dim)
        self.vector = nn.Linear(3, dim)
        self.tensor = nn.Linear(9, dim)
        self.out = nn.Linear(dim, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply distance-weighted scalar/vector attention."""

        pos, feat = x[..., :3], x[..., 3:4]
        rel = pos[:, :, None] - pos[:, None, :]
        dist = rel.pow(2).sum(-1).add(1e-4).rsqrt()
        sh = torch.cat(
            (
                rel,
                rel.square(),
                rel[..., :1] * rel[..., 1:2],
                rel[..., 1:2] * rel[..., 2:3],
                rel[..., :1],
            ),
            dim=-1,
        )
        attn = torch.softmax(dist + self.tensor(sh).mean(-1), dim=-1)
        vector_msg = torch.matmul(attn, pos)
        h = self.scalar(torch.cat((pos, feat), dim=-1)) + self.vector(vector_msg)
        return self.out(h.mean(1))


class PointNet2SemSegCompact(nn.Module):
    """PointNet++ semantic segmentation with set abstraction and FP-style upsampling."""

    def __init__(self, classes: int = 6) -> None:
        """Initialize pointwise abstraction and segmentation head."""

        super().__init__()
        self.sa = nn.Sequential(nn.Conv1d(3, 32, 1), nn.ReLU(), nn.Conv1d(32, 64, 1), nn.ReLU())
        self.fp = nn.Sequential(nn.Conv1d(64 + 3, 32, 1), nn.ReLU(), nn.Conv1d(32, classes, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Segment each input point."""

        pts = x.transpose(1, 2)
        global_feat = self.sa(pts).amax(-1, keepdim=True).expand(-1, -1, pts.shape[-1])
        return self.fp(torch.cat((pts, global_feat), dim=1)).transpose(1, 2)


class PointNet2SSGCompact(nn.Module):
    """PointNet++ SSG classifier with compact local grouping."""

    def __init__(self, classes: int = 10) -> None:
        """Initialize local set abstraction and classifier."""

        super().__init__()
        self.local = nn.Sequential(nn.Conv2d(6, 24, 1), nn.ReLU(), nn.Conv2d(24, 48, 1), nn.ReLU())
        self.global_sa = nn.Sequential(
            nn.Conv1d(48, 64, 1), nn.ReLU(), nn.Conv1d(64, 64, 1), nn.ReLU()
        )
        self.head = nn.Linear(64, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify points after one single-scale grouping abstraction."""

        centers = x[:, ::4]
        rel = x[:, None, :, :] - centers[:, :, None, :]
        dist = rel.pow(2).sum(-1)
        idx = dist.topk(8, dim=-1, largest=False).indices
        grouped = torch.gather(
            x[:, None].expand(-1, centers.shape[1], -1, -1),
            2,
            idx.unsqueeze(-1).expand(-1, -1, -1, 3),
        )
        feats = torch.cat((grouped, grouped - centers[:, :, None]), dim=-1).permute(0, 3, 1, 2)
        local = self.local(feats).amax(-1)
        return self.head(self.global_sa(local).amax(-1))


def example_decision() -> torch.Tensor:
    """Return offline-RL trajectory features."""

    return torch.randn(1, 6, 18)


def example_image() -> torch.Tensor:
    """Return a small RGB image."""

    return torch.randn(1, 3, 64, 64)


def example_pair() -> torch.Tensor:
    """Return a pair of RGB images."""

    return torch.randn(1, 2, 3, 64, 64)


def example_events() -> torch.Tensor:
    """Return event frames in ``(T,B,C,H,W)`` format."""

    return torch.randn(4, 1, 3, 64, 64)


def example_latent() -> torch.Tensor:
    """Return latent vectors."""

    return torch.randn(1, 32)


def example_ipe() -> torch.Tensor:
    """Return Gaussian ray means and variances."""

    return torch.randn(1, 16, 6).abs()


def example_dna() -> torch.Tensor:
    """Return one-hot DNA sequence."""

    return F.one_hot(torch.randint(0, 4, (1, 128)), num_classes=4).float()


def example_tokens() -> torch.Tensor:
    """Return token ids."""

    return torch.randint(0, 30, (1, 16))


def example_tracks() -> torch.Tensor:
    """Return multimodal protein token tracks."""

    return torch.randint(0, 30, (1, 3, 16))


def example_speech() -> torch.Tensor:
    """Return log-mel speech features."""

    return torch.randn(1, 32, 80)


def example_wave() -> torch.Tensor:
    """Return a short waveform."""

    return torch.randn(1, 256)


def example_tabular() -> torch.Tensor:
    """Return numerical tabular features."""

    return torch.randn(1, 16)


def example_kan() -> torch.Tensor:
    """Return KAN input features."""

    return torch.randn(1, 8)


def example_tokens32() -> torch.Tensor:
    """Return dense token features."""

    return torch.randn(1, 12, 32)


def example_multivectors() -> torch.Tensor:
    """Return projective-geometric-algebra multivector tokens."""

    return torch.randn(1, 8, 16)


def example_gears() -> torch.Tensor:
    """Return expression and perturbation-mask features."""

    expr = torch.randn(1, 16, 1)
    pert = torch.randint(0, 2, (1, 16, 1)).float()
    return torch.cat((expr, pert), dim=-1)


def example_triples() -> torch.Tensor:
    """Return knowledge-graph triples."""

    return torch.randint(0, 8, (1, 10, 3))


def example_points() -> torch.Tensor:
    """Return point-cloud coordinates plus scalar features."""

    return torch.randn(1, 12, 4)


def example_point_xyz() -> torch.Tensor:
    """Return XYZ point clouds."""

    return torch.randn(1, 32, 3)


def example_video_4() -> torch.Tensor:
    """Return a four-frame video clip."""

    return torch.randn(1, 3, 4, 32, 32)


def example_video_8() -> torch.Tensor:
    """Return an eight-frame video clip."""

    return torch.randn(1, 3, 8, 32, 32)


def example_video_16() -> torch.Tensor:
    """Return a compact sixteen-frame video clip."""

    return torch.randn(1, 3, 16, 32, 32)


def example_video_32() -> torch.Tensor:
    """Return a compact thirty-two-frame video clip."""

    return torch.randn(1, 3, 32, 32, 32)


def example_destine() -> torch.Tensor:
    """Return categorical field ids for DESTINE."""

    from menagerie.classics.fuxictr_ctr import example_input_destine

    return example_input_destine()


def example_geowizard() -> torch.Tensor:
    """Return GeoWizard packed image/time/domain input."""

    from menagerie.classics.geowizard import example_input

    return example_input()


def build_decision_spikeformer() -> nn.Module:
    """Build Decision SpikeFormer."""

    return DecisionSpikeFormer().eval()


def build_spikingformer() -> nn.Module:
    """Build a compact spiking transformer classifier."""

    return SoftmaxFreeSpikeSelfAttention().eval()


def build_sdt_v3() -> nn.Module:
    """Build SDT-v3/E-SpikeFormer linear spike-driven attention."""

    return EfficientSpikeFormer().eval()


def build_dsine() -> nn.Module:
    """Build compact DSINE."""

    return DSINENormalNet().eval()


def build_dynamic_conv() -> nn.Module:
    """Build DyConv-ResNet-style dynamic convolution network."""

    from menagerie.classics.dynamic_convolution import build

    return build().eval()


def build_dust3r() -> nn.Module:
    """Build compact DUSt3R pointmap regressor."""

    return PointmapTransformer().eval()


def build_edgesam() -> nn.Module:
    """Build compact EdgeSAM."""

    return EdgeSAMCompact().eval()


def build_dynamic_filter() -> nn.Module:
    """Build Dynamic Filter Network."""

    return DynamicFilterNetwork().eval()


def build_ems_yolo() -> nn.Module:
    """Build EMS-YOLO."""

    return EMSYOLO().eval()


def build_eg3d() -> nn.Module:
    """Build EG3D triplane generator."""

    return EG3DGenerator().eval()


def build_mipnerf_ipe() -> nn.Module:
    """Build Mip-NeRF integrated positional encoder."""

    return MipNeRFEncoding().eval()


def build_kan_linear() -> nn.Module:
    """Build compact KAN linear layer stack."""

    from menagerie.classics.reimpl2_kan import build_kan

    return build_kan().eval()


def build_basenji2() -> nn.Module:
    """Build Basenji2-style dilated genomics tower."""

    return GenomicTower(transformer=False).eval()


def build_enformer() -> nn.Module:
    """Build Enformer-style convolution plus transformer genomics tower."""

    return GenomicTower(transformer=True).eval()


def build_esm1b() -> nn.Module:
    """Build ESM-1b-style protein encoder."""

    return ESMProteinModel(tracks=1).eval()


def build_esm3() -> nn.Module:
    """Build ESM3-style sequence/structure/function track model."""

    return ESMProteinModel(tracks=3).eval()


def build_esmc() -> nn.Module:
    """Build ESM-Cambrian sequence protein model."""

    return ESMProteinModel(tracks=1, dim=64).eval()


def build_branchformer() -> nn.Module:
    """Build ESPnet Branchformer encoder."""

    return BranchformerEncoder().eval()


def build_branchformer_transducer() -> nn.Module:
    """Build ESPnet Branchformer Transducer."""

    return BranchformerTransducer().eval()


def build_branchformer_ctc() -> nn.Module:
    """Build ESPnet Branchformer CTC."""

    return BranchformerCTC().eval()


def build_fastspeech2() -> nn.Module:
    """Build FastSpeech2."""

    return FastSpeech2Compact().eval()


def build_conformer_fastspeech2() -> nn.Module:
    """Build Conformer FastSpeech2."""

    return ConformerFastSpeech2Compact().eval()


def build_dptnet() -> nn.Module:
    """Build DPTNet speech separator."""

    return DPTNetCompact().eval()


def build_convtasnet() -> nn.Module:
    """Build ConvTasNet speech separator."""

    return ConvTasNetCompact().eval()


def build_jets() -> nn.Module:
    """Build JETS text-to-waveform model."""

    return JETSCompact().eval()


def build_vits() -> nn.Module:
    """Build VITS text-to-waveform model."""

    return VITSCompact().eval()


def build_parallel_wavegan() -> nn.Module:
    """Build a Parallel WaveGAN-like vocoder core."""

    return _ParallelWaveGANCompact().eval()


def build_s4_decoder() -> nn.Module:
    """Build an ESPnet S4-style diagonal state-space decoder."""

    return _S4Decoder().eval()


def build_rnn_lm() -> nn.Module:
    """Build ESPnet recurrent language model."""

    return RNNLanguageModel().eval()


def build_transformer_lm() -> nn.Module:
    """Build ESPnet Transformer language model."""

    return TransformerLanguageModel().eval()


def build_excelformer() -> nn.Module:
    """Build ExcelFormer classifier."""

    return ExcelFormerCompact().eval()


def build_energy_transformer() -> nn.Module:
    """Build Energy Transformer."""

    return EnergyTransformerCompact().eval()


def build_uniformer_video() -> nn.Module:
    """Build compact UniFormer video model."""

    from menagerie.classics.uniformer_xxs4_160_k400 import CompactVideoUniFormer

    return CompactVideoUniFormer(depths=(1, 1, 1, 1)).eval()


def build_destine2() -> nn.Module:
    """Build DESTINE/DESTINE2 compact CTR model."""

    from menagerie.classics.fuxictr_ctr import build_destine

    return build_destine().eval()


def build_gatenet() -> nn.Module:
    """Build GateNet compact saliency model."""

    return GateNetCompact().eval()


def build_pointnet2_ssg() -> nn.Module:
    """Build PointNet++ SSG classifier."""

    return PointNet2SSGCompact().eval()


def build_pointnet2_msg() -> nn.Module:
    """Build PointNet++ MSG classifier."""

    from menagerie.classics.pointnet2_msg import build_pointnet2_msg

    return build_pointnet2_msg().eval()


def build_pointnet2_semseg() -> nn.Module:
    """Build PointNet++ semantic segmentation model."""

    return PointNet2SemSegCompact().eval()


def build_gatr() -> nn.Module:
    """Build GATr compact model."""

    return GATrCompact().eval()


def build_gears() -> nn.Module:
    """Build GEARS compact model."""

    return GEARSCompact().eval()


def build_gram() -> nn.Module:
    """Build GRAM generator."""

    return GRAMGenerator().eval()


def build_metric3d() -> nn.Module:
    """Build Metric3D normal-head model."""

    return Metric3DNormalHead().eval()


def build_patchfusion() -> nn.Module:
    """Build PatchFusion tiled depth fusion model."""

    return PatchFusionDepth().eval()


def build_geowizard() -> nn.Module:
    """Build GeoWizard-style depth-normal estimator."""

    from menagerie.classics.geowizard import build_geowizard_depth_normal

    return build_geowizard_depth_normal().eval()


def build_ingram() -> nn.Module:
    """Build InGram."""

    return InGramCompact().eval()


def build_se3_transformer() -> nn.Module:
    """Build SE(3)-Transformer compact model."""

    return SE3TransformerCompact().eval()


class _TokenSpikeHead(nn.Module):
    """Token head for spiking vision transformer variants."""

    def __init__(self) -> None:
        """Initialize token mixer and classifier."""

        super().__init__()
        self.block = SpikingAttentionBlock(32)
        self.head = nn.Linear(32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify flattened image tokens."""

        return self.head(self.block(x.transpose(1, 2)).mean(1))


class _S4Decoder(nn.Module):
    """Diagonal S4-style state-space decoder."""

    def __init__(self, dim: int = 32) -> None:
        """Initialize diagonal state parameters."""

        super().__init__()
        self.inp = nn.Linear(16, dim)
        self.a = nn.Parameter(-torch.linspace(0.1, 1.0, dim))
        self.b = nn.Parameter(torch.randn(dim) * 0.02)
        self.out = nn.Linear(dim, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scan a diagonal state-space recurrence."""

        u = self.inp(F.one_hot(x.long().clamp(0, 15), num_classes=16).float())
        state = torch.zeros(x.shape[0], u.shape[-1], device=x.device, dtype=u.dtype)
        outs = []
        for step in range(u.shape[1]):
            state = torch.exp(self.a).unsqueeze(0) * state + self.b.unsqueeze(0) * u[:, step]
            outs.append(self.out(state))
        return torch.stack(outs, dim=1)


class _ParallelWaveGANCompact(nn.Module):
    """Parallel WaveGAN-style non-autoregressive mel vocoder."""

    def __init__(self) -> None:
        """Initialize transposed-convolution residual generator."""

        super().__init__()
        self.inp = nn.ConvTranspose1d(80, 32, 8, 4, 2)
        self.residual = nn.ModuleList(
            [nn.Conv1d(32, 32, 3, padding=2**idx, dilation=2**idx) for idx in range(4)]
        )
        self.skip = nn.Conv1d(32, 32, 1)
        self.out = nn.Conv1d(32, 1, 7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate waveform samples from mel frames."""

        h = F.leaky_relu(self.inp(x.transpose(1, 2)), 0.2)
        skips = []
        for conv in self.residual:
            res = torch.tanh(conv(h))
            h = h + res
            skips.append(self.skip(res))
        return self.out(sum(skips) / len(skips)).squeeze(1)


def _entry(
    name: str,
    build: str,
    example: str,
    year: str,
    code: str,
) -> tuple[str, str, str, str, str]:
    """Create a menagerie entry tuple."""

    return (name, build, example, year, code)


MENAGERIE_ENTRIES = [
    _entry("decision_spikeformer", "build_decision_spikeformer", "example_decision", "2025", "DC"),
    _entry("spikingformer_dvsgesture", "build_spikingformer", "example_image", "2022", "DC"),
    _entry("sdt_v3_efficient_spiking_transformer", "build_sdt_v3", "example_image", "2024", "DC"),
    _entry("dsine_convnext_base", "build_dsine", "example_image", "2024", "DC"),
    _entry("dsine_convnext_tiny", "build_dsine", "example_image", "2024", "DC"),
    _entry("DUSt3R-Pointmap", "build_dust3r", "example_pair", "2024", "DC"),
    _entry("DyConv-ResNet50", "build_dynamic_conv", "example_image", "2020", "DC"),
    _entry("DynamicFilterNetwork", "build_dynamic_filter", "example_image", "2016", "DC"),
    _entry("EdgeSAM_3x", "build_edgesam", "example_image", "2023", "DC"),
    _entry("kan_linear_layer", "build_kan_linear", "example_kan", "2024", "CA"),
    _entry("ems_yolo", "build_ems_yolo", "example_events", "2023", "DC"),
    _entry("eg3d_triplane_generator", "build_eg3d", "example_latent", "2022", "DG"),
    _entry("eg3d_osg_decoder", "build_eg3d", "example_latent", "2022", "DG"),
    _entry("mipnerf_IntegratedPosEnc", "build_mipnerf_ipe", "example_ipe", "2021", "DG"),
    _entry("EnergyTransformer", "build_energy_transformer", "example_tokens32", "2024", "CD"),
    _entry("Basenji2-pytorch", "build_basenji2", "example_dna", "2018", "DA"),
    _entry("enformer", "build_enformer", "example_dna", "2021", "DA"),
    _entry("pointnet2_cls_msg", "build_pointnet2_msg", "example_point_xyz", "2017", "CG"),
    _entry("pointnet2_cls_ssg", "build_pointnet2_ssg", "example_point_xyz", "2017", "CG"),
    _entry("pointnet2_semseg_ssg", "build_pointnet2_semseg", "example_point_xyz", "2017", "CG"),
    _entry("ESM-1b style encoder", "build_esm1b", "example_tokens", "2021", "DA"),
    _entry("ESM3 open small", "build_esm3", "example_tracks", "2024", "DA"),
    _entry("esm3_open", "build_esm3", "example_tracks", "2024", "DA"),
    _entry("ESM-C Cambrian 300M", "build_esmc", "example_tokens", "2024", "DA"),
    _entry("esmc", "build_esmc", "example_tokens", "2024", "DA"),
    _entry("ESMC-300m", "build_esmc", "example_tokens", "2024", "DA"),
    _entry(
        "espnet_asr_branchformer_transducer",
        "build_branchformer_transducer",
        "example_speech",
        "2022",
        "DE",
    ),
    _entry("espnet_asr_branchformer_ctc", "build_branchformer_ctc", "example_speech", "2022", "DE"),
    _entry("espnet_branchformer_encoder", "build_branchformer", "example_speech", "2022", "DE"),
    _entry(
        "espnet_tts_conformer_fastspeech2",
        "build_conformer_fastspeech2",
        "example_tokens",
        "2020",
        "DE",
    ),
    _entry("espnet_fastspeech", "build_fastspeech2", "example_tokens", "2019", "DE"),
    _entry("espnet_tts_fastspeech", "build_fastspeech2", "example_tokens", "2019", "DE"),
    _entry("espnet_fastspeech2", "build_fastspeech2", "example_tokens", "2020", "DE"),
    _entry("espnet_tts_fastspeech2", "build_fastspeech2", "example_tokens", "2020", "DE"),
    _entry("espnet_jets", "build_jets", "example_tokens", "2022", "DE"),
    _entry("espnet_tts_jets", "build_jets", "example_tokens", "2022", "DE"),
    _entry("espnet_enh_convtasnet", "build_convtasnet", "example_wave", "2019", "DE"),
    _entry("espnet_enh_dptnet", "build_dptnet", "example_wave", "2020", "DE"),
    _entry(
        "espnet_vocoder_parallel_wavegan", "build_parallel_wavegan", "example_speech", "2020", "DE"
    ),
    _entry("espnet_s4_decoder", "build_s4_decoder", "example_tokens", "2022", "DE"),
    _entry("espnet_lm_rnn", "build_rnn_lm", "example_tokens", "2014", "DE"),
    _entry("espnet_lm_transformer", "build_transformer_lm", "example_tokens", "2017", "DE"),
    _entry("espnet_tts_vits", "build_vits", "example_tokens", "2021", "DE"),
    _entry("espnet_vits", "build_vits", "example_tokens", "2021", "DE"),
    _entry("patchfusion", "build_patchfusion", "example_image", "2024", "DC"),
    _entry("ExcelFormer-Classifier", "build_excelformer", "example_tabular", "2023", "CA"),
    _entry("ExcelFormer", "build_excelformer", "example_tabular", "2023", "CA"),
    _entry(
        "uniformer_b16_sthv1_prek400", "build_uniformer_video", "example_video_16", "2022", "DC"
    ),
    _entry(
        "uniformer_b16_sthv2_prek400", "build_uniformer_video", "example_video_16", "2022", "DC"
    ),
    _entry("uniformer_b16x4_k400", "build_uniformer_video", "example_video_16", "2022", "DC"),
    _entry("uniformer_b16x8_k400", "build_uniformer_video", "example_video_16", "2022", "DC"),
    _entry(
        "uniformer_b32_sthv1_prek400", "build_uniformer_video", "example_video_32", "2022", "DC"
    ),
    _entry(
        "uniformer_b32_sthv2_prek400", "build_uniformer_video", "example_video_32", "2022", "DC"
    ),
    _entry("uniformer_b32x4_k400", "build_uniformer_video", "example_video_32", "2022", "DC"),
    _entry("uniformer_b32x4_k600", "build_uniformer_video", "example_video_32", "2022", "DC"),
    _entry("uniformer_b8x8_k400", "build_uniformer_video", "example_video_8", "2022", "DC"),
    _entry(
        "uniformer_s16_sthv1_prek400", "build_uniformer_video", "example_video_16", "2022", "DC"
    ),
    _entry(
        "uniformer_s16_sthv2_prek400", "build_uniformer_video", "example_video_16", "2022", "DC"
    ),
    _entry(
        "uniformer_s16x4_hmdb51_prek400", "build_uniformer_video", "example_video_16", "2022", "DC"
    ),
    _entry("uniformer_s16x4_k400", "build_uniformer_video", "example_video_16", "2022", "DC"),
    _entry("uniformer_s16x4_k600", "build_uniformer_video", "example_video_16", "2022", "DC"),
    _entry(
        "uniformer_s16x4_ucf101_prek400", "build_uniformer_video", "example_video_16", "2022", "DC"
    ),
    _entry("uniformer_s16x8_k400", "build_uniformer_video", "example_video_16", "2022", "DC"),
    _entry(
        "uniformer_s32_sthv1_prek400", "build_uniformer_video", "example_video_32", "2022", "DC"
    ),
    _entry(
        "uniformer_s32_sthv2_prek400", "build_uniformer_video", "example_video_32", "2022", "DC"
    ),
    _entry("uniformer_s32x4_k400", "build_uniformer_video", "example_video_32", "2022", "DC"),
    _entry("uniformer_s8x8_k400", "build_uniformer_video", "example_video_8", "2022", "DC"),
    _entry("uniformer_xs32_192_k400", "build_uniformer_video", "example_video_32", "2022", "DC"),
    _entry("uniformer_xxs16_128_k400", "build_uniformer_video", "example_video_16", "2022", "DC"),
    _entry("uniformer_xxs16_160_k400", "build_uniformer_video", "example_video_16", "2022", "DC"),
    _entry("uniformer_xxs32_160_k400", "build_uniformer_video", "example_video_32", "2022", "DC"),
    _entry("uniformer_xxs4_128_k400", "build_uniformer_video", "example_video_4", "2022", "DC"),
    _entry("FuxiCTR-DESTINE2", "build_destine2", "example_destine", "2021", "CA"),
    _entry("GateNet", "build_gatenet", "example_image", "2020", "DC"),
    _entry("GATr-GeometricAlgebra", "build_gatr", "example_multivectors", "2023", "CG"),
    _entry("GEARS", "build_gears", "example_gears", "2023", "DA"),
    _entry("gram_generator", "build_gram", "example_latent", "2022", "DG"),
    _entry("geowizard_depth_normal", "build_geowizard", "example_geowizard", "2024", "DC"),
    _entry("geowizard_v2_depth_normal", "build_geowizard", "example_geowizard", "2024", "DC"),
    _entry("metric3d_normal_head", "build_metric3d", "example_image", "2024", "DC"),
    _entry("InGram", "build_ingram", "example_triples", "2023", "CG"),
    _entry(
        "SE(3)-Transformer (DGL/official)", "build_se3_transformer", "example_points", "2020", "CG"
    ),
]
