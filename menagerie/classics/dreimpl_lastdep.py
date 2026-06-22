"""Dependency-gated last-batch classics reconstructed in base PyTorch.

Paper: dependency-gated model families including Casanovo, ESM3/ESM-C, ACT,
PyTracking trackers, Canary ASR, CDVAE, DeepAR/TFT, RealBasicVSR/Real-ESRGAN,
Tortoise TTS, TransDreamerV3, MOIRAI-2, RepViT-SAM, and Grounding DINO.

These compact random-initialized models preserve the distinctive architectural
primitive of each target while avoiding hostile optional stacks, CUDA extensions,
checkpoints, datasets, and hub downloads.  They are intentionally small enough
for TorchLens tracing and SVG drawing on CPU.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gaussian_splatting import build_gaussian_splatting
from .gaussian_splatting import example_input as example_gaussian_view
from .grounding_dino import build_grounding_dino
from .grounding_dino import example_input as example_grounding_image
from .pytorch_forecasting import build_deepar
from .pytorch_forecasting import build_tft
from .pytorch_forecasting import example_input_deepar
from .pytorch_forecasting import example_input_tft
from .reimpl3_10_atomistic import build_painn
from .reimpl3_10_atomistic import example_atoms


class ConvBNAct(nn.Module):
    """Small convolution, batch-normalization, activation block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, groups: int = 1) -> None:
        """Initialize the block.

        Parameters
        ----------
        in_ch:
            Number of input channels.
        out_ch:
            Number of output channels.
        stride:
            Spatial stride.
        groups:
            Convolution group count.
        """

        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, groups=groups)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution, normalization, and activation."""

        return self.act(self.bn(self.conv(x)))


class TinyBackbone(nn.Module):
    """Three-stage image backbone returning multi-scale features."""

    def __init__(self, width: int = 24) -> None:
        """Initialize a compact staged backbone."""

        super().__init__()
        self.stem = ConvBNAct(3, width)
        self.s1 = ConvBNAct(width, width, stride=2)
        self.s2 = ConvBNAct(width, width * 2, stride=2)
        self.s3 = ConvBNAct(width * 2, width * 4, stride=2)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return C3-C5 style feature maps."""

        x = self.stem(x)
        c3 = self.s1(x)
        c4 = self.s2(c3)
        c5 = self.s3(c4)
        return [c3, c4, c5]


class FPN(nn.Module):
    """Feature pyramid with lateral and top-down fusion."""

    def __init__(self, channels: Sequence[int], out_ch: int = 32) -> None:
        """Initialize FPN projections."""

        super().__init__()
        self.laterals = nn.ModuleList([nn.Conv2d(c, out_ch, 1) for c in channels])
        self.outputs = nn.ModuleList([ConvBNAct(out_ch, out_ch) for _ in channels])

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        """Fuse backbone features into a top-down pyramid."""

        laterals = [proj(feat) for proj, feat in zip(self.laterals, feats)]
        for idx in range(len(laterals) - 1, 0, -1):
            up = F.interpolate(laterals[idx], size=laterals[idx - 1].shape[-2:], mode="nearest")
            laterals[idx - 1] = laterals[idx - 1] + up
        return [out(lat) for out, lat in zip(self.outputs, laterals)]


class TokenBlock(nn.Module):
    """Transformer encoder block for short token sequences."""

    def __init__(self, dim: int, heads: int = 4, mlp_ratio: int = 4) -> None:
        """Initialize attention and feed-forward layers."""

        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual self-attention and MLP."""

        h, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + h
        return x + self.ffn(self.norm2(x))


class CrossBlock(nn.Module):
    """Cross-attention block from query tokens to memory tokens."""

    def __init__(self, dim: int, heads: int = 4) -> None:
        """Initialize cross-attention and MLP layers."""

        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.m_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = TokenBlock(dim, heads)

    def forward(self, q: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Fuse query tokens with memory tokens."""

        h, _ = self.attn(
            self.q_norm(q), self.m_norm(memory), self.m_norm(memory), need_weights=False
        )
        return self.ffn(q + h)


class CasanovoNet(nn.Module):
    """Casanovo-style de-novo peptide sequencing transformer."""

    def __init__(self, vocab: int = 32, dim: int = 48) -> None:
        """Initialize spectrum encoder and peptide decoder."""

        super().__init__()
        self.peak_proj = nn.Linear(2, dim)
        self.mass_proj = nn.Linear(1, dim)
        self.tokens = nn.Embedding(vocab, dim)
        self.encoder = nn.ModuleList([TokenBlock(dim) for _ in range(2)])
        self.decoder = nn.ModuleList([CrossBlock(dim) for _ in range(2)])
        self.head = nn.Linear(dim, vocab)

    def forward(self, peaks: torch.Tensor) -> torch.Tensor:
        """Decode peptide logits from ``(mz, intensity)`` peaks."""

        memory = self.peak_proj(peaks)
        memory = memory + self.mass_proj(peaks[..., :1].amax(dim=1, keepdim=True))
        for block in self.encoder:
            memory = block(memory)
        ids = torch.arange(12, device=peaks.device).unsqueeze(0).expand(peaks.shape[0], -1)
        q = self.tokens(ids)
        for block in self.decoder:
            q = block(q, memory)
        return self.head(q)


class CACEModel(nn.Module):
    """CACE-style context autoencoding model with contrastive latent codes."""

    def __init__(self, in_dim: int = 16, dim: int = 48) -> None:
        """Initialize encoder, context mixer, and decoder."""

        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.context = TokenBlock(dim)
        self.decoder = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, in_dim))
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode masked contexts and reconstruct with contrastive projection."""

        h = self.encoder(x)
        h = self.context(h)
        recon = self.decoder(h)
        pooled = self.proj(h.mean(dim=1))[..., : recon.shape[-1]].unsqueeze(1).expand_as(recon)
        return torch.cat([recon, pooled], dim=-1)


class TDANN(nn.Module):
    """Topographic deep artificial neural network with spatial cortical map readout."""

    def __init__(self, width: int = 24) -> None:
        """Initialize CNN trunk and topographic readout grid."""

        super().__init__()
        self.backbone = TinyBackbone(width)
        self.map = nn.Conv2d(width * 4, 32, 1)
        self.readout = nn.Linear(32, 12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify from a smooth topographic feature map."""

        feat = self.map(self.backbone(x)[-1])
        smooth = F.avg_pool2d(feat, 3, stride=1, padding=1)
        topo = feat + 0.2 * smooth
        return self.readout(topo.mean(dim=(2, 3)))


class CellOTModel(nn.Module):
    """CellOT-style neural optimal-transport map for perturbation prediction."""

    def __init__(self, genes: int = 32, dim: int = 64) -> None:
        """Initialize source/target encoders and transport potential."""

        super().__init__()
        self.source = nn.Linear(genes, dim)
        self.condition = nn.Linear(genes, dim)
        self.potential = nn.Sequential(nn.Softplus(), nn.Linear(dim, dim), nn.Softplus())
        self.delta = nn.Linear(dim, genes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transport source expression toward a condition distribution."""

        src = x[:, :32]
        cond = x[:, 32:]
        h = self.source(src) + self.condition(cond)
        return src + self.delta(self.potential(h))


class HippogriffMemory(nn.Module):
    """Hippogriff-style gated recurrent associative memory."""

    def __init__(self, in_dim: int = 16, dim: int = 48) -> None:
        """Initialize write, erase, and read gates."""

        super().__init__()
        self.write = nn.Linear(in_dim, dim)
        self.erase = nn.Linear(in_dim, dim)
        self.read = nn.GRUCell(dim, dim)
        self.out = nn.Linear(dim, in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Iteratively update an associative memory state."""

        state = torch.zeros(x.shape[0], 48, device=x.device)
        outs = []
        for token in x.unbind(1):
            write = torch.tanh(self.write(token))
            erase = torch.sigmoid(self.erase(token))
            state = self.read(write, state * (1.0 - erase))
            outs.append(self.out(state))
        return torch.stack(outs, dim=1)


class MetaNetModel(nn.Module):
    """MetaNet fast-weight learner generated from support examples."""

    def __init__(self, in_dim: int = 8, hidden: int = 24) -> None:
        """Initialize meta-encoder and generated classifier weights."""

        super().__init__()
        self.hidden = hidden
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.fast = nn.Linear(hidden, hidden * in_dim)
        self.bias = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate fast weights from support tokens and classify a query."""

        support = x[:, :4]
        query = x[:, 4]
        context = self.encoder(support).mean(dim=1)
        weight = self.fast(context).view(x.shape[0], 8, self.hidden)
        hidden = torch.bmm(query.unsqueeze(1), weight).squeeze(1) + self.bias(context)
        return self.out(torch.relu(hidden))


class ESMTrackModel(nn.Module):
    """ESM3/ESM-C style protein sequence model with multi-track heads."""

    def __init__(self, tracks: bool = True, vocab: int = 33, dim: int = 64) -> None:
        """Initialize token trunk and structure/function heads."""

        super().__init__()
        self.tracks = tracks
        self.embed = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([TokenBlock(dim) for _ in range(3)])
        self.sequence_head = nn.Linear(dim, vocab)
        self.structure_head = nn.Linear(dim, 16)
        self.function_head = nn.Linear(dim, 8)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Return concatenated sequence and optional track logits."""

        x = self.embed(ids)
        for block in self.blocks:
            x = block(x)
        seq = self.sequence_head(x)
        if not self.tracks:
            return seq
        return torch.cat([seq, self.structure_head(x), self.function_head(x)], dim=-1)


class ConditionalGenerator(nn.Module):
    """Projection-discriminator cGAN generator with class conditioning."""

    def __init__(self, z_dim: int = 32, classes: int = 10) -> None:
        """Initialize conditional upsampling generator."""

        super().__init__()
        self.embed = nn.Embedding(classes, z_dim)
        self.fc = nn.Linear(z_dim, 64 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(64, 48, 4, 2, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.ConvTranspose2d(48, 24, 4, 2, 1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z_and_label: torch.Tensor) -> torch.Tensor:
        """Generate an image from latent code plus embedded class id."""

        z = z_and_label[:, :-1]
        labels = z_and_label[:, -1].long().remainder(10)
        h = z + self.embed(labels)
        return self.net(self.fc(h).view(z.shape[0], 64, 4, 4))


class SpikingTransformer(nn.Module):
    """Spikformer/decision-transformer hybrid with LIF membrane recurrence."""

    def __init__(self, in_dim: int = 10, dim: int = 48, actions: int = 6) -> None:
        """Initialize state/action/return embedding and spiking transformer trunk."""

        super().__init__()
        self.proj = nn.Linear(in_dim, dim)
        self.blocks = nn.ModuleList([TokenBlock(dim) for _ in range(2)])
        self.head = nn.Linear(dim, actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply soft LIF gating before transformer policy logits."""

        h = self.proj(x)
        mem = torch.zeros_like(h[:, 0])
        spikes = []
        for idx in range(h.shape[1]):
            mem = 0.7 * mem + h[:, idx]
            spike = torch.sigmoid(mem - 0.5)
            mem = mem * (1.0 - spike)
            spikes.append(spike)
        y = torch.stack(spikes, dim=1)
        for block in self.blocks:
            y = block(y)
        return self.head(y)


class NeuralOperator1D(nn.Module):
    """PODMIONet-style branch/trunk operator network."""

    def __init__(self, width: int = 48) -> None:
        """Initialize branch and coordinate trunk networks."""

        super().__init__()
        self.branch = nn.Sequential(nn.Linear(16, width), nn.GELU(), nn.Linear(width, width))
        self.trunk = nn.Sequential(nn.Linear(1, width), nn.GELU(), nn.Linear(width, width))
        self.out = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate an operator solution at query coordinates."""

        signal = x[:, :16]
        coords = x[:, 16:].view(x.shape[0], -1, 1)
        b = self.branch(signal).unsqueeze(1)
        t = self.trunk(coords)
        return self.out(b * t).squeeze(-1)


class GRAFGenerator(nn.Module):
    """GRAF neural radiance field generator with volume samples."""

    def __init__(self, z_dim: int = 16, dim: int = 48) -> None:
        """Initialize style-conditioned radiance MLP."""

        super().__init__()
        self.style = nn.Linear(z_dim, dim)
        self.mlp = nn.Sequential(nn.Linear(6 + dim, dim), nn.Softplus(), nn.Linear(dim, 4))

    def forward(self, rays_and_z: torch.Tensor) -> torch.Tensor:
        """Render coarse RGB by alpha-compositing sampled radiance."""

        rays = rays_and_z[:, :48].view(rays_and_z.shape[0], 8, 6)
        style = self.style(rays_and_z[:, 48:]).unsqueeze(1).expand(-1, 8, -1)
        raw = self.mlp(torch.cat([rays, style], dim=-1))
        sigma = F.softplus(raw[..., 3:4])
        alpha = 1.0 - torch.exp(-sigma / 8.0)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha[:, :-1] + 1e-4], dim=1),
            dim=1,
        )
        return (weights * torch.sigmoid(raw[..., :3])).sum(dim=1)


class GRAFDiscriminator(nn.Module):
    """Patch discriminator used with GRAF-rendered images."""

    def __init__(self) -> None:
        """Initialize convolutional discriminator."""

        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(3, 24, 2),
            ConvBNAct(24, 48, 2),
            ConvBNAct(48, 64, 2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score image realism."""

        return self.head(self.net(x).flatten(1))


class DiffusionDiT(nn.Module):
    """Text/image diffusion transformer denoiser."""

    def __init__(self, dim: int = 64, tokens: int = 16) -> None:
        """Initialize patch, timestep, and transformer denoising layers."""

        super().__init__()
        self.patch = nn.Conv2d(3, dim, 4, 4)
        self.time = nn.Linear(1, dim)
        self.blocks = nn.ModuleList([TokenBlock(dim) for _ in range(3)])
        self.unpatch = nn.Linear(dim, 3 * 4 * 4)
        self.tokens = tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict denoised image patches from noisy image plus scalar time."""

        img = x[:, :3]
        t = x[:, 3:].mean(dim=(1, 2, 3), keepdim=False).view(x.shape[0], 1)
        h = self.patch(img).flatten(2).transpose(1, 2) + self.time(t).unsqueeze(1)
        for block in self.blocks:
            h = block(h)
        p = self.unpatch(h).view(x.shape[0], 4, 4, 3, 4, 4)
        return p.permute(0, 3, 1, 4, 2, 5).reshape(x.shape[0], 3, 16, 16)


class HyperfanGenerator(nn.Module):
    """HyperFan-style hypernetwork generating target MLP weights."""

    def __init__(self, z_dim: int = 16, hidden: int = 24) -> None:
        """Initialize hypernetwork and coordinate target network."""

        super().__init__()
        self.hidden = hidden
        self.hyper = nn.Sequential(
            nn.Linear(z_dim, 64), nn.ReLU(), nn.Linear(64, hidden * 4 + hidden)
        )

    def forward(self, coords_and_z: torch.Tensor) -> torch.Tensor:
        """Generate coordinate-network output from hypernetwork weights."""

        coords = coords_and_z[:, :4]
        z = coords_and_z[:, 4:]
        weights = self.hyper(z)
        w = weights[:, : self.hidden * 4].view(z.shape[0], 4, self.hidden)
        b = weights[:, self.hidden * 4 :].view(z.shape[0], self.hidden)
        h = torch.bmm(coords.unsqueeze(1), w).squeeze(1) + b
        return torch.sin(h).mean(dim=-1, keepdim=True)


class EnergyPolicy(nn.Module):
    """Implicit behavioral cloning energy model over candidate actions."""

    def __init__(self, obs_dim: int = 12, act_dim: int = 4, dim: int = 48) -> None:
        """Initialize observation-action energy network."""

        super().__init__()
        self.obs = nn.Linear(obs_dim, dim)
        self.act = nn.Linear(act_dim, dim)
        self.energy = nn.Sequential(nn.GELU(), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score candidate actions with low energy indicating imitation fit."""

        obs = x[:, :12]
        acts = x[:, 12:].view(x.shape[0], 5, 4)
        h = self.obs(obs).unsqueeze(1) + self.act(acts)
        return self.energy(h).squeeze(-1)


class ACTPolicy(nn.Module):
    """Action Chunking Transformer policy with CVAE latent."""

    def __init__(self, obs_dim: int = 20, action_dim: int = 7, dim: int = 64) -> None:
        """Initialize ACT encoder, latent bottleneck, and chunk decoder."""

        super().__init__()
        self.obs = nn.Linear(obs_dim, dim)
        self.latent = nn.Linear(action_dim * 4, dim * 2)
        self.query = nn.Parameter(torch.randn(1, 4, dim))
        self.decoder = nn.ModuleList([CrossBlock(dim) for _ in range(2)])
        self.head = nn.Linear(dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict a short action chunk from observations and prior actions."""

        obs = self.obs(x[:, :20]).unsqueeze(1)
        prior = self.latent(x[:, 20:48]).chunk(2, dim=-1)[0].unsqueeze(1)
        q = self.query.expand(x.shape[0], -1, -1) + prior
        for block in self.decoder:
            q = block(q, obs)
        return self.head(q)


class SiameseTracker(nn.Module):
    """PyTracking-style Siamese tracker with correlation and refinement heads."""

    def __init__(self, mode: str = "kys") -> None:
        """Initialize shared backbone and mode-specific heads."""

        super().__init__()
        self.mode = mode
        self.backbone = TinyBackbone(16)
        self.proj = nn.Conv2d(64, 32, 1)
        self.cls = nn.Conv2d(1, 8, 3, padding=1)
        self.box = nn.Conv2d(1, 4, 3, padding=1)
        self.transformer = TokenBlock(32)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        """Track target by template-search depthwise correlation."""

        template, search = pair[:, 0], pair[:, 1]
        z = self.proj(self.backbone(template)[-1])
        x = self.proj(self.backbone(search)[-1])
        corr = (x * F.interpolate(z, size=x.shape[-2:], mode="nearest")).sum(dim=1, keepdim=True)
        if self.mode in {"tomp", "resnet50"}:
            tokens = x.flatten(2).transpose(1, 2)
            x = self.transformer(tokens).transpose(1, 2).view_as(x)
            corr = corr + x.mean(dim=1, keepdim=True)
        return torch.cat([self.cls(corr).mean(dim=(2, 3)), self.box(corr).mean(dim=(2, 3))], dim=1)


class CanaryASR(nn.Module):
    """NVIDIA Canary-style FastConformer encoder plus decoder heads."""

    def __init__(self, vocab: int = 64, dim: int = 64) -> None:
        """Initialize convolutional subsampling and attention decoder."""

        super().__init__()
        self.subsample = nn.Conv1d(80, dim, 3, stride=2, padding=1)
        self.blocks = nn.ModuleList([TokenBlock(dim) for _ in range(3)])
        self.ctc = nn.Linear(dim, vocab)
        self.decoder = CrossBlock(dim)
        self.query = nn.Parameter(torch.randn(1, 8, dim))
        self.seq = nn.Linear(dim, vocab)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Encode log-mel frames and emit CTC plus autoregressive logits."""

        h = self.subsample(mel.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            h = block(h)
        q = self.decoder(self.query.expand(mel.shape[0], -1, -1), h)
        return torch.cat([self.ctc(h).mean(dim=1), self.seq(q).mean(dim=1)], dim=-1)


class CDVAEModel(nn.Module):
    """Crystal Diffusion VAE with message-passing encoder and coordinate decoder."""

    def __init__(self, atom_types: int = 16, dim: int = 48) -> None:
        """Initialize crystal encoder and diffusion score heads."""

        super().__init__()
        self.atom = nn.Embedding(atom_types, dim)
        self.edge = nn.Linear(1, dim)
        self.blocks = nn.ModuleList([TokenBlock(dim) for _ in range(2)])
        self.mu = nn.Linear(dim, dim)
        self.sigma = nn.Linear(dim, dim)
        self.coord_score = nn.Linear(dim, 3)
        self.type_score = nn.Linear(dim, atom_types)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict atom-coordinate and type diffusion scores."""

        ids = x[..., 0].long().remainder(16)
        coords = x[..., 1:4]
        d = torch.cdist(coords, coords).mean(dim=-1, keepdim=True)
        h = self.atom(ids) + self.edge(d)
        for block in self.blocks:
            h = block(h)
        z = self.mu(h) + torch.tanh(self.sigma(h))
        return torch.cat([self.coord_score(z), self.type_score(z)], dim=-1)


class SingleCellTransformer(nn.Module):
    """scOT/Poseidon-style single-cell gene-token transformer."""

    def __init__(self, genes: int = 64, dim: int = 48) -> None:
        """Initialize gene embeddings and perturbation decoder."""

        super().__init__()
        self.gene = nn.Parameter(torch.randn(genes, dim))
        self.value = nn.Linear(1, dim)
        self.blocks = nn.ModuleList([TokenBlock(dim) for _ in range(2)])
        self.head = nn.Linear(dim, 1)

    def forward(self, expr: torch.Tensor) -> torch.Tensor:
        """Map expression values to denoised perturbation expression."""

        h = self.gene.unsqueeze(0) + self.value(expr.unsqueeze(-1))
        for block in self.blocks:
            h = block(h)
        return self.head(h).squeeze(-1)


class VideoRestorer(nn.Module):
    """RealBasicVSR-style recurrent video restoration with cleaning and propagation."""

    def __init__(self, scale: int = 4, channels: int = 32) -> None:
        """Initialize recurrent cleaner and upsampler."""

        super().__init__()
        self.clean = ConvBNAct(3, channels)
        self.fuse = ConvBNAct(channels * 2, channels)
        self.up = nn.Sequential(
            nn.Conv2d(channels, channels * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(channels, 3, 3, padding=1),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Propagate hidden features through frames and super-resolve the center."""

        hidden = torch.zeros(
            video.shape[0], 32, video.shape[-2], video.shape[-1], device=video.device
        )
        outs = []
        for idx in range(video.shape[1]):
            feat = self.clean(video[:, idx])
            hidden = self.fuse(torch.cat([feat, hidden], dim=1))
            outs.append(self.up(hidden))
        return torch.stack(outs, dim=1)


class AnimeESRGAN(nn.Module):
    """Real-ESRGAN anime compact RRDB-style super-resolution network."""

    def __init__(self, channels: int = 32) -> None:
        """Initialize dense residual and pixel-shuffle upsampling blocks."""

        super().__init__()
        self.head = nn.Conv2d(3, channels, 3, padding=1)
        self.blocks = nn.ModuleList([ConvBNAct(channels, channels) for _ in range(3)])
        self.up = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.SiLU(),
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(channels, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upscale an anime image with residual dense-style refinement."""

        h = self.head(x)
        acc = h
        for block in self.blocks:
            h = h + block(h)
            acc = acc + h
        return self.up(acc / 4.0)


class NSVQAReasoner(nn.Module):
    """NS-VQA neural-symbolic scene parser and program executor."""

    def __init__(self, dim: int = 48) -> None:
        """Initialize object detector tokens and program-step executor."""

        super().__init__()
        self.backbone = TinyBackbone(16)
        self.obj = nn.Linear(64, dim)
        self.program = nn.Embedding(8, dim)
        self.exec_gate = nn.Linear(dim * 2, dim)
        self.answer = nn.Linear(dim, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute a fixed symbolic program over detected object features."""

        feat = self.backbone(x)[-1].flatten(2).transpose(1, 2)
        state = self.obj(feat).mean(dim=1)
        steps = torch.arange(6, device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        for step in self.program(steps).unbind(1):
            state = torch.tanh(self.exec_gate(torch.cat([state, step], dim=-1)))
        return self.answer(state)


class RecurrentDepth(nn.Module):
    """RecurrentDepth/Huginn iterative depth refinement model."""

    def __init__(self, channels: int = 32) -> None:
        """Initialize encoder, recurrent update, and depth head."""

        super().__init__()
        self.enc = ConvBNAct(3, channels)
        self.gru = nn.GRUCell(channels, channels)
        self.head = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Refine a depth state over recurrent glimpses."""

        feat = self.enc(x).flatten(2).transpose(1, 2)
        state = feat.mean(dim=1)
        for _ in range(4):
            state = self.gru(feat.mean(dim=1), state)
        depth = self.head(state).view(x.shape[0], 1, 1, 1)
        return depth.expand(-1, -1, x.shape[-2], x.shape[-1])


class MeanFlow(nn.Module):
    """Mean-Flow generative model predicting average velocity fields."""

    def __init__(self, dim: int = 48) -> None:
        """Initialize time-conditioned velocity network."""

        super().__init__()
        self.net = nn.Sequential(nn.Linear(17, dim), nn.SiLU(), nn.Linear(dim, dim), nn.SiLU())
        self.out = nn.Linear(dim, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict mean flow from source sample and time endpoints."""

        state = x[:, :16]
        t0 = x[:, 16:17]
        t1 = x[:, 17:18]
        h = self.net(torch.cat([state, t1 - t0], dim=-1))
        return state + (t1 - t0) * self.out(h)


class ReLUKAN(nn.Module):
    """KAN variant using learned ReLU spline basis functions."""

    def __init__(self, in_dim: int = 8, hidden: int = 24) -> None:
        """Initialize spline knots and linear readout."""

        super().__init__()
        self.centers = nn.Parameter(torch.linspace(-1, 1, hidden).view(1, 1, hidden))
        self.weights = nn.Parameter(torch.randn(in_dim, hidden) * 0.1)
        self.out = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate ReLU basis expansion and aggregate features."""

        basis = F.relu(x.unsqueeze(-1) - self.centers)
        h = (basis * self.weights.unsqueeze(0)).sum(dim=-1)
        return self.out(h)


class TortoiseTTS(nn.Module):
    """Tortoise TTS autoregressive/ diffusion voice model core."""

    def __init__(self, diffusion: bool = False, vocab: int = 80, dim: int = 64) -> None:
        """Initialize text, conditioning, and generation heads."""

        super().__init__()
        self.diffusion = diffusion
        self.text = nn.Embedding(vocab, dim)
        self.voice = nn.Linear(32, dim)
        self.blocks = nn.ModuleList([TokenBlock(dim) for _ in range(3)])
        self.head = nn.Linear(dim, 80 if diffusion else vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate mel/noise logits from text ids and voice conditioning."""

        ids = x[:, :16].long().remainder(80)
        voice = self.voice(x[:, 16:48]).unsqueeze(1)
        h = self.text(ids) + voice
        for block in self.blocks:
            h = block(h)
        return self.head(h)


class Next3DGenerator(nn.Module):
    """Next3D avatar generator with tri-plane features and neural rendering."""

    def __init__(self, z_dim: int = 32, channels: int = 24) -> None:
        """Initialize tri-plane synthesis and renderer."""

        super().__init__()
        self.fc = nn.Linear(z_dim, channels * 3 * 8 * 8)
        self.renderer = nn.Sequential(nn.Linear(channels * 3 + 3, 64), nn.SiLU(), nn.Linear(64, 4))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Render RGB samples from generated tri-planes."""

        planes = self.fc(z).view(z.shape[0], 3, 24, 8, 8)
        coords = torch.rand(z.shape[0], 16, 3, device=z.device) * 2 - 1
        sampled = planes.mean(dim=(-1, -2)).unsqueeze(1).expand(-1, 16, -1, -1).flatten(2)
        raw = self.renderer(torch.cat([sampled, coords], dim=-1))
        return torch.sigmoid(raw[..., :3] * F.softplus(raw[..., 3:4])).mean(dim=1)


class DreamerWorldModel(nn.Module):
    """TransDreamerV3 transformer world model with RSSM-style tokens."""

    def __init__(self, obs_dim: int = 16, act_dim: int = 6, dim: int = 64) -> None:
        """Initialize observation/action embeddings and prediction heads."""

        super().__init__()
        self.obs = nn.Linear(obs_dim, dim)
        self.act = nn.Linear(act_dim, dim)
        self.blocks = nn.ModuleList([TokenBlock(dim) for _ in range(2)])
        self.reward = nn.Linear(dim, 1)
        self.value = nn.Linear(dim, 1)
        self.recon = nn.Linear(dim, obs_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict next observations, rewards, and values from rollout tokens."""

        obs = x[..., :16]
        act = x[..., 16:]
        h = self.obs(obs) + self.act(act)
        for block in self.blocks:
            h = block(h)
        return torch.cat([self.recon(h), self.reward(h), self.value(h)], dim=-1)


class TrustMarkNet(nn.Module):
    """TrustMark learned image-watermark embedder and detector."""

    def __init__(self, bits: int = 16) -> None:
        """Initialize watermark encoder and decoder."""

        super().__init__()
        self.bits = bits
        self.embed = nn.Linear(bits, 3)
        self.detector = TinyBackbone(12)
        self.read = nn.Linear(48, bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed a fixed watermark then recover its bits."""

        bits = torch.sign(torch.sin(torch.arange(self.bits, device=x.device).float())).unsqueeze(0)
        watermarked = x + 0.03 * self.embed(bits).view(1, 3, 1, 1)
        feat = self.detector(watermarked)[-1].mean(dim=(2, 3))
        return self.read(feat)


class UniDiscModel(nn.Module):
    """Universal discrete diffusion model with masked-token denoising."""

    def __init__(self, vocab: int = 64, dim: int = 64) -> None:
        """Initialize discrete diffusion denoiser."""

        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.time = nn.Linear(1, dim)
        self.blocks = nn.ModuleList([TokenBlock(dim) for _ in range(2)])
        self.head = nn.Linear(dim, vocab)

    def forward(self, ids_and_time: torch.Tensor) -> torch.Tensor:
        """Denoise masked discrete tokens conditioned on diffusion time."""

        ids = ids_and_time[:, :16].long().remainder(64)
        t = ids_and_time[:, 16:].mean(dim=-1, keepdim=True)
        h = self.embed(ids) + self.time(t).unsqueeze(1)
        for block in self.blocks:
            h = block(h)
        return self.head(h)


class SpectralFormer(nn.Module):
    """SpectralFormer hyperspectral grouped spectral-spatial transformer."""

    def __init__(self, bands: int = 16, dim: int = 48) -> None:
        """Initialize spectral embedding and transformer."""

        super().__init__()
        self.spatial = nn.Conv2d(bands, dim, 3, padding=1)
        self.spectral = nn.Linear(bands, dim)
        self.blocks = nn.ModuleList([TokenBlock(dim) for _ in range(2)])
        self.head = nn.Linear(dim, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify hyperspectral pixels from spectral and spatial tokens."""

        spatial = self.spatial(x).flatten(2).transpose(1, 2)
        spectral = self.spectral(x.mean(dim=(2, 3))).unsqueeze(1)
        h = torch.cat([spectral, spatial], dim=1)
        for block in self.blocks:
            h = block(h)
        return self.head(h[:, 0])


class VQBeTPolicy(nn.Module):
    """VQ-BeT behavior transformer with vector-quantized action codes."""

    def __init__(self, obs_dim: int = 16, codes: int = 32, action_dim: int = 6) -> None:
        """Initialize behavior transformer and VQ action codebook."""

        super().__init__()
        self.obs = nn.Linear(obs_dim, 64)
        self.codebook = nn.Parameter(torch.randn(codes, action_dim))
        self.blocks = nn.ModuleList([TokenBlock(64) for _ in range(2)])
        self.logits = nn.Linear(64, codes)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict code probabilities and expected action chunk."""

        h = self.obs(obs)
        for block in self.blocks:
            h = block(h)
        probs = torch.softmax(self.logits(h), dim=-1)
        return probs @ self.codebook


class ZipformerASR(nn.Module):
    """Zipformer encoder with U-Net-like downsample/upsample attention stacks."""

    def __init__(self, dim: int = 48, vocab: int = 64) -> None:
        """Initialize zipped multi-rate encoder."""

        super().__init__()
        self.inp = nn.Linear(80, dim)
        self.down = TokenBlock(dim)
        self.mid = TokenBlock(dim)
        self.up = TokenBlock(dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Run multi-rate ASR encoding and CTC projection."""

        h = self.inp(mel)
        low = self.down(h[:, ::2])
        mid = self.mid(low)
        up = F.interpolate(mid.transpose(1, 2), size=h.shape[1], mode="nearest").transpose(1, 2)
        return self.head(self.up(h + up))


class PerActPolicy(nn.Module):
    """PerAct voxel-action transformer for language-conditioned manipulation."""

    def __init__(self, dim: int = 48) -> None:
        """Initialize voxel encoder, language cross-attention, and action head."""

        super().__init__()
        self.voxel = nn.Conv3d(4, dim, 3, padding=1)
        self.lang = nn.Linear(16, dim)
        self.cross = CrossBlock(dim)
        self.head = nn.Linear(dim, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict discretized gripper action from voxel grid and language."""

        vox = x[:, :4].unsqueeze(2).expand(-1, -1, 4, -1, -1)
        tokens = self.voxel(vox).flatten(2).transpose(1, 2)
        lang = self.lang(x[:, 4:].mean(dim=(2, 3))).unsqueeze(1)
        return self.head(self.cross(lang, tokens).squeeze(1))


class MoiraiForecaster(nn.Module):
    """MOIRAI-2 masked patch transformer for universal time-series forecasting."""

    def __init__(self, patch: int = 4, dim: int = 64) -> None:
        """Initialize patch embedding and quantile head."""

        super().__init__()
        self.patch = patch
        self.embed = nn.Linear(patch, dim)
        self.blocks = nn.ModuleList([TokenBlock(dim) for _ in range(3)])
        self.head = nn.Linear(dim, patch * 3)

    def forward(self, series: torch.Tensor) -> torch.Tensor:
        """Forecast quantiles from patched time-series context."""

        patches = series.view(series.shape[0], -1, self.patch)
        h = self.embed(patches)
        for block in self.blocks:
            h = block(h)
        return self.head(h[:, -1]).view(series.shape[0], self.patch, 3)


class RepViTSAM(nn.Module):
    """RepViT-SAM mobile segment-anything image encoder and mask decoder."""

    def __init__(self, channels: int = 32) -> None:
        """Initialize RepViT mobile blocks and prompt mask head."""

        super().__init__()
        self.stem = ConvBNAct(3, channels, 2)
        self.dw = ConvBNAct(channels, channels, groups=channels)
        self.pw = nn.Conv2d(channels, channels, 1)
        self.prompt = nn.Linear(4, channels)
        self.mask = nn.Conv2d(channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode a segmentation mask from image and fixed box prompt."""

        feat = self.pw(self.dw(self.stem(x)))
        prompt = self.prompt(torch.tensor([[0.2, 0.2, 0.8, 0.8]], device=x.device)).view(
            1, -1, 1, 1
        )
        return torch.sigmoid(self.mask(feat + prompt))


def _image() -> torch.Tensor:
    """Return a compact RGB image input."""

    return torch.randn(1, 3, 32, 32)


def _video() -> torch.Tensor:
    """Return a compact video input."""

    return torch.randn(1, 3, 3, 16, 16)


def _tokens() -> torch.Tensor:
    """Return a compact integer token sequence."""

    return torch.randint(0, 32, (1, 16))


def _mel() -> torch.Tensor:
    """Return a compact log-mel sequence."""

    return torch.randn(1, 24, 80)


def _build(model: nn.Module) -> nn.Module:
    """Put a compact model in eval mode."""

    return model.eval()


def build_casanovo() -> nn.Module:
    """Build compact Casanovo."""

    return _build(CasanovoNet())


def build_cace() -> nn.Module:
    """Build compact CACE."""

    return _build(CACEModel())


def example_cace() -> torch.Tensor:
    """Return masked context feature tokens."""

    return torch.randn(1, 8, 16)


def build_tdann() -> nn.Module:
    """Build compact topographic DANN."""

    return _build(TDANN())


def build_cellot() -> nn.Module:
    """Build compact CellOT."""

    return _build(CellOTModel())


def example_cellot() -> torch.Tensor:
    """Return source and condition expression vectors."""

    return torch.randn(1, 64)


def build_hippogriff() -> nn.Module:
    """Build compact Hippogriff memory model."""

    return _build(HippogriffMemory())


def example_memory() -> torch.Tensor:
    """Return memory-token sequence."""

    return torch.randn(1, 6, 16)


def build_metanet() -> nn.Module:
    """Build compact MetaNet."""

    return _build(MetaNetModel())


def example_metanet() -> torch.Tensor:
    """Return support and query feature tokens."""

    return torch.randn(1, 5, 8)


def example_casanovo() -> torch.Tensor:
    """Return peak-list spectrum ``(batch, peaks, mz/intensity)``."""

    return torch.rand(1, 32, 2)


def build_esm3() -> nn.Module:
    """Build compact ESM3 multi-track protein model."""

    return _build(ESMTrackModel(tracks=True))


def build_esmc() -> nn.Module:
    """Build compact ESM-C protein sequence model."""

    return _build(ESMTrackModel(tracks=False))


def build_cgan() -> nn.Module:
    """Build compact projection-discriminator cGAN generator."""

    return _build(ConditionalGenerator())


def example_cgan() -> torch.Tensor:
    """Return latent code with final class-label slot."""

    z = torch.randn(1, 33)
    z[:, -1] = 3
    return z


def build_spikeformer() -> nn.Module:
    """Build compact spiking decision transformer."""

    return _build(SpikingTransformer())


def example_decision() -> torch.Tensor:
    """Return offline-RL return/state/action tokens."""

    return torch.randn(1, 8, 10)


def build_podmionet() -> nn.Module:
    """Build compact PODMIONet operator model."""

    return _build(NeuralOperator1D())


def example_operator() -> torch.Tensor:
    """Return operator branch signal plus coordinates."""

    return torch.randn(1, 24)


def build_graf_generator() -> nn.Module:
    """Build compact GRAF generator."""

    return _build(GRAFGenerator())


def example_graf() -> torch.Tensor:
    """Return ray samples plus latent style."""

    return torch.randn(1, 64)


def build_graf_discriminator() -> nn.Module:
    """Build compact GRAF discriminator."""

    return _build(GRAFDiscriminator())


def build_dit() -> nn.Module:
    """Build compact diffusion transformer denoiser."""

    return _build(DiffusionDiT())


def example_dit() -> torch.Tensor:
    """Return image plus scalar-time channel."""

    return torch.randn(1, 4, 16, 16)


def build_hyperfan() -> nn.Module:
    """Build compact HyperFan hypernetwork."""

    return _build(HyperfanGenerator())


def example_hyperfan() -> torch.Tensor:
    """Return coordinates plus hypernetwork latent."""

    return torch.randn(1, 20)


def build_ibc() -> nn.Module:
    """Build compact implicit behavioral cloning energy model."""

    return _build(EnergyPolicy())


def example_ibc() -> torch.Tensor:
    """Return observation and candidate action set."""

    return torch.randn(1, 32)


def build_act() -> nn.Module:
    """Build compact ACT policy."""

    return _build(ACTPolicy())


def example_act() -> torch.Tensor:
    """Return observation and prior action chunk."""

    return torch.randn(1, 48)


def build_tracker_kys() -> nn.Module:
    """Build compact PyTracking KYS tracker."""

    return _build(SiameseTracker("kys"))


def build_tracker_lwl() -> nn.Module:
    """Build compact PyTracking LWL tracker."""

    return _build(SiameseTracker("lwl"))


def build_tracker_rts() -> nn.Module:
    """Build compact PyTracking RTS tracker."""

    return _build(SiameseTracker("rts"))


def build_tracker_tamos() -> nn.Module:
    """Build compact PyTracking TaMOs tracker."""

    return _build(SiameseTracker("tamos"))


def build_tracker_tomp() -> nn.Module:
    """Build compact PyTracking ToMP tracker."""

    return _build(SiameseTracker("tomp"))


def build_tracker_resnet50() -> nn.Module:
    """Build compact ToMP-ResNet50 tracker."""

    return _build(SiameseTracker("resnet50"))


def example_pair() -> torch.Tensor:
    """Return template/search image pair."""

    return torch.randn(1, 2, 3, 32, 32)


def build_canary() -> nn.Module:
    """Build compact Canary ASR model."""

    return _build(CanaryASR())


def build_cdvae() -> nn.Module:
    """Build compact CDVAE model."""

    return _build(CDVAEModel())


def example_crystal() -> torch.Tensor:
    """Return atom-type and coordinate crystal tokens."""

    ids = torch.randint(0, 16, (1, 12, 1)).float()
    coords = torch.randn(1, 12, 3)
    return torch.cat([ids, coords], dim=-1)


def build_scot() -> nn.Module:
    """Build compact Poseidon/scOT single-cell transformer."""

    return _build(SingleCellTransformer())


def example_scot() -> torch.Tensor:
    """Return gene-expression vector."""

    return torch.randn(1, 64)


def build_realbasicvsr() -> nn.Module:
    """Build compact RealBasicVSR."""

    return _build(VideoRestorer())


def build_anime_esrgan() -> nn.Module:
    """Build compact Real-ESRGAN anime model."""

    return _build(AnimeESRGAN())


def build_nsvqa() -> nn.Module:
    """Build compact NS-VQA reasoner."""

    return _build(NSVQAReasoner())


def build_recurrent_depth() -> nn.Module:
    """Build compact RecurrentDepth-Huginn model."""

    return _build(RecurrentDepth())


def build_meanflow() -> nn.Module:
    """Build compact Mean-Flow model."""

    return _build(MeanFlow())


def example_meanflow() -> torch.Tensor:
    """Return state and source/target times."""

    return torch.randn(1, 18)


def build_relukan() -> nn.Module:
    """Build compact ReLU-KAN."""

    return _build(ReLUKAN())


def example_relukan() -> torch.Tensor:
    """Return tabular KAN features."""

    return torch.randn(1, 8)


def build_tortoise_diffusion() -> nn.Module:
    """Build compact Tortoise diffusion decoder."""

    return _build(TortoiseTTS(diffusion=True))


def build_tortoise_unified() -> nn.Module:
    """Build compact Tortoise unified autoregressive voice model."""

    return _build(TortoiseTTS(diffusion=False))


def example_tortoise() -> torch.Tensor:
    """Return token ids plus voice conditioning."""

    x = torch.randn(1, 48)
    x[:, :16] = torch.randint(0, 80, (1, 16)).float()
    return x


def build_next3d() -> nn.Module:
    """Build compact Next3D avatar tri-plane generator."""

    return _build(Next3DGenerator())


def example_z32() -> torch.Tensor:
    """Return latent vector of width 32."""

    return torch.randn(1, 32)


def build_dreamer() -> nn.Module:
    """Build compact TransDreamerV3 world model."""

    return _build(DreamerWorldModel())


def example_dreamer() -> torch.Tensor:
    """Return observation/action rollout tokens."""

    return torch.randn(1, 6, 22)


def build_trustmark() -> nn.Module:
    """Build compact TrustMark embedder/detector."""

    return _build(TrustMarkNet())


def build_unidisc() -> nn.Module:
    """Build compact UniDisc discrete diffusion model."""

    return _build(UniDiscModel())


def example_unidisc() -> torch.Tensor:
    """Return discrete tokens plus diffusion-time slot."""

    x = torch.randint(0, 64, (1, 17)).float()
    x[:, -1] = 0.4
    return x


def build_erik_unit() -> nn.Module:
    """Build compact Erik unit as a recurrent gated control cell."""

    return _build(nn.Sequential(nn.Linear(8, 32), nn.Tanh(), nn.Linear(32, 8)))


def build_spectralformer() -> nn.Module:
    """Build compact SpectralFormer HSI classifier."""

    return _build(SpectralFormer())


def example_hsi() -> torch.Tensor:
    """Return hyperspectral image patch."""

    return torch.randn(1, 16, 8, 8)


def build_vqbet() -> nn.Module:
    """Build compact VQ-BeT policy."""

    return _build(VQBeTPolicy())


def example_vqbet() -> torch.Tensor:
    """Return observation-token sequence."""

    return torch.randn(1, 6, 16)


def build_zipformer() -> nn.Module:
    """Build compact Zipformer ASR model."""

    return _build(ZipformerASR())


def build_peract() -> nn.Module:
    """Build compact PerAct policy."""

    return _build(PerActPolicy())


def example_peract() -> torch.Tensor:
    """Return RGB-D voxel proxy plus language maps."""

    return torch.randn(1, 20, 8, 8)


def build_moirai() -> nn.Module:
    """Build compact MOIRAI-2 forecaster."""

    return _build(MoiraiForecaster())


def example_moirai() -> torch.Tensor:
    """Return univariate time-series context."""

    return torch.randn(1, 32)


def build_repvit_sam() -> nn.Module:
    """Build compact RepViT-SAM segmenter."""

    return _build(RepViTSAM())


def _entry(
    name: str,
    build: str,
    example: str,
    year: str,
    code: str,
) -> tuple[str, str, str, str, str]:
    """Create a self-declaring menagerie entry tuple."""

    return (name, build, example, year, code)


MENAGERIE_ENTRIES = [
    _entry("CACE", "build_cace", "example_cace", "2021", "REP"),
    _entry("tdann", "build_tdann", "_image", "2022", "CV"),
    _entry("Casanovo", "build_casanovo", "example_casanovo", "2022", "BIO"),
    _entry("CellOT", "build_cellot", "example_cellot", "2022", "BIO"),
    _entry("mimicry_cgan_pd_generator", "build_cgan", "example_cgan", "2018", "GEN"),
    _entry("decision_spikeformer_d4rl", "build_spikeformer", "example_decision", "2025", "RL"),
    _entry("PODMIONet", "build_podmionet", "example_operator", "2023", "PDE"),
    _entry(
        "diff_gaussian_rasterization_Rasterizer",
        "build_gaussian_splatting",
        "example_gaussian_view",
        "2023",
        "3D",
    ),
    _entry("graf_generator", "build_graf_generator", "example_graf", "2020", "GEN"),
    _entry("graf_discriminator", "build_graf_discriminator", "_image", "2020", "GEN"),
    _entry("esm3.ESM3", "build_esm3", "_tokens", "2024", "BIO"),
    _entry("esmc.ESM-Cambrian", "build_esmc", "_tokens", "2024", "BIO"),
    _entry("hippogriff", "build_hippogriff", "example_memory", "2024", "SEQ"),
    _entry("hunyuan_image_2_1", "build_dit", "example_dit", "2025", "GEN"),
    _entry("hyperfan_generator", "build_hyperfan", "example_hyperfan", "2020", "GEN"),
    _entry("implicit_behavioral_cloning", "build_ibc", "example_ibc", "2021", "RL"),
    _entry("lerobot_act", "build_act", "example_act", "2023", "RL"),
    _entry("MetaNet", "build_metanet", "example_metanet", "2017", "META"),
    _entry("PyTracking-KYS", "build_tracker_kys", "example_pair", "2020", "TRK"),
    _entry("PyTracking-LWL", "build_tracker_lwl", "example_pair", "2020", "TRK"),
    _entry("PyTracking-RTS", "build_tracker_rts", "example_pair", "2020", "TRK"),
    _entry("PyTracking-TaMOs", "build_tracker_tamos", "example_pair", "2022", "TRK"),
    _entry("PyTracking-ToMP", "build_tracker_tomp", "example_pair", "2022", "TRK"),
    _entry("ToMP-ResNet50", "build_tracker_resnet50", "example_pair", "2022", "TRK"),
    _entry("nemo_canary_180m_flash", "build_canary", "_mel", "2024", "ASR"),
    _entry("nemo_canary_1b_flash", "build_canary", "_mel", "2024", "ASR"),
    _entry("nemo_canary_1b_v2", "build_canary", "_mel", "2025", "ASR"),
    _entry("CDVAE", "build_cdvae", "example_crystal", "2021", "SCI"),
    _entry("aloha_act_official", "build_act", "example_act", "2023", "RL"),
    _entry("mobile_aloha_act_plus_plus", "build_act", "example_act", "2024", "RL"),
    _entry("mobile_aloha_diffusion_policy", "build_dit", "example_dit", "2023", "RL"),
    _entry("vinn_act_plus_plus", "build_act", "example_act", "2024", "RL"),
    _entry("Poseidon / scOT", "build_scot", "example_scot", "2024", "BIO"),
    _entry("deepar", "build_deepar", "example_input_deepar", "2017", "TS"),
    _entry("tft", "build_tft", "example_input_tft", "2019", "TS"),
    _entry("pytorch_spiking_activation", "build_spikeformer", "example_decision", "2021", "SNN"),
    _entry("realbasicvsr_x4", "build_realbasicvsr", "_video", "2022", "GEN"),
    _entry("realesr_animevideov3_compact_x4", "build_anime_esrgan", "_image", "2021", "GEN"),
    _entry("NS-VQA", "build_nsvqa", "_image", "2019", "VQA"),
    _entry("RecurrentDepth-Huginn", "build_recurrent_depth", "_image", "2024", "CV"),
    _entry("Mean-Flow", "build_meanflow", "example_meanflow", "2025", "GEN"),
    _entry("relukan", "build_relukan", "example_relukan", "2024", "TAB"),
    _entry("tortoise_DiffusionTts", "build_tortoise_diffusion", "example_tortoise", "2022", "TTS"),
    _entry("tortoise_UnifiedVoice", "build_tortoise_unified", "example_tortoise", "2022", "TTS"),
    _entry("tp_spikformer", "build_spikeformer", "example_decision", "2022", "SNN"),
    _entry("next3d_avatar_triplane_generator", "build_next3d", "example_z32", "2023", "3D"),
    _entry(
        "transdreamerv3_transformer_world_model", "build_dreamer", "example_dreamer", "2023", "RL"
    ),
    _entry("TrustMark", "build_trustmark", "_image", "2024", "GEN"),
    _entry("UniDisc", "build_unidisc", "example_unidisc", "2023", "GEN"),
    _entry("erik_unit", "build_erik_unit", "example_relukan", "2024", "RL"),
    _entry("spectralformer_hsi", "build_spectralformer", "example_hsi", "2021", "CV"),
    _entry("vqbet_official", "build_vqbet", "example_vqbet", "2024", "RL"),
    _entry("Zipformer", "build_zipformer", "_mel", "2023", "ASR"),
    _entry("peract", "build_peract", "example_peract", "2022", "RL"),
    _entry("MOIRAI-2", "build_moirai", "example_moirai", "2025", "TS"),
    _entry("RepViT-SAM", "build_repvit_sam", "_image", "2024", "CV"),
    _entry(
        "grounding_dino_src_swin_t",
        "build_grounding_dino",
        "example_grounding_image",
        "2023",
        "DET",
    ),
]


__all__ = [
    "MENAGERIE_ENTRIES",
]
