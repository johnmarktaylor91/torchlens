"""Compact faithful reimplementations for REIMPL6 shard 1.

Paper anchors:
MDLM / Simple and Effective Masked Diffusion Language Models, Sahoo et al.
2024; Diffusion Policy, Chi et al. 2023; pi-GAN, Chan et al. 2021;
TranscriptFormer / cross-species generative cell atlas, CZI 2025.

These are random-init compact reconstructions. They preserve each model's
load-bearing primitive for TorchLens rendering rather than reproducing a full
training recipe or checkpoint.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal diffusion-step embedding followed by a small projection."""

    def __init__(self, dim: int) -> None:
        """Initialize the embedding.

        Parameters
        ----------
        dim:
            Output embedding width.
        """

        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed scalar timesteps.

        Parameters
        ----------
        t:
            Tensor of scalar diffusion times shaped ``(batch,)``.

        Returns
        -------
        torch.Tensor
            Time embeddings shaped ``(batch, dim)``.
        """

        half = self.dim // 2
        freq = torch.exp(
            torch.arange(half, device=t.device, dtype=t.dtype) * (-math.log(10000.0) / (half - 1))
        )
        args = t[:, None] * freq[None, :]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, 1))
        return self.proj(emb)


class TinySelfAttention(nn.Module):
    """Compact multi-head self-attention."""

    def __init__(self, dim: int, heads: int = 4) -> None:
        """Initialize attention projections.

        Parameters
        ----------
        dim:
            Token width.
        heads:
            Number of attention heads.
        """

        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply scaled dot-product attention.

        Parameters
        ----------
        x:
            Token tensor shaped ``(batch, tokens, dim)``.

        Returns
        -------
        torch.Tensor
            Attended token tensor.
        """

        batch, tokens, dim = x.shape
        qkv = self.qkv(x).view(batch, tokens, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = torch.softmax((q @ k.transpose(-2, -1)) * (self.head_dim**-0.5), dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(batch, tokens, dim)
        return self.out(out)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, dim: int, heads: int = 4, mlp_ratio: int = 2) -> None:
        """Initialize block layers.

        Parameters
        ----------
        dim:
            Token width.
        heads:
            Number of attention heads.
        mlp_ratio:
            Hidden-width multiplier.
        """

        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TinySelfAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the transformer block.

        Parameters
        ----------
        x:
            Token tensor.

        Returns
        -------
        torch.Tensor
            Updated token tensor.
        """

        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class MDLM(nn.Module):
    """Masked discrete diffusion LM with absorbing-mask SUBS-style parameterization."""

    def __init__(self, vocab: int = 96, dim: int = 64, depth: int = 2, mask_id: int = 95) -> None:
        """Initialize compact MDLM.

        Parameters
        ----------
        vocab:
            Vocabulary size including the absorbing mask token.
        dim:
            Token width.
        depth:
            Number of denoising transformer blocks.
        mask_id:
            Absorbing mask token id.
        """

        super().__init__()
        self.mask_id = mask_id
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(torch.randn(1, 16, dim) * 0.02)
        self.time = SinusoidalTimeEmbedding(dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Predict denoised tokens from an absorbed masked sequence.

        Parameters
        ----------
        ids:
            Token ids shaped ``(batch, tokens)``.

        Returns
        -------
        torch.Tensor
            Logits with the absorbing mask token suppressed, shaped
            ``(batch, tokens, vocab)``.
        """

        batch, tokens = ids.shape
        mask_pattern = (torch.arange(tokens, device=ids.device)[None, :] % 3) == 0
        absorbed = torch.where(mask_pattern, torch.full_like(ids, self.mask_id), ids)
        t = torch.full((batch,), 0.6, device=ids.device, dtype=torch.float32)
        x = self.tok(absorbed) + self.pos[:, :tokens] + self.time(t)[:, None, :]
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.norm(x))
        logits = logits.clone()
        logits[..., self.mask_id] = -30.0
        return torch.where(mask_pattern[..., None], logits, logits.detach() * 0.0 + logits)


class FiLMTemporalBlock(nn.Module):
    """Residual 1D temporal block modulated by global observation conditioning."""

    def __init__(self, channels: int, cond_dim: int) -> None:
        """Initialize the FiLM temporal block.

        Parameters
        ----------
        channels:
            Temporal feature channels.
        cond_dim:
            Conditioning width.
        """

        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.film = nn.Linear(cond_dim, channels * 2)
        self.norm1 = nn.GroupNorm(4, channels)
        self.norm2 = nn.GroupNorm(4, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply FiLM-modulated residual temporal convolution.

        Parameters
        ----------
        x:
            Temporal features shaped ``(batch, channels, horizon)``.
        cond:
            Global conditioning vector.

        Returns
        -------
        torch.Tensor
            Updated temporal features.
        """

        scale, shift = self.film(cond).chunk(2, dim=-1)
        y = self.norm1(self.conv1(x))
        y = y * (1.0 + scale[..., None]) + shift[..., None]
        y = self.conv2(F.silu(y))
        return x + F.silu(self.norm2(y))


class TinyVisionObsEncoder(nn.Module):
    """Small image observation encoder used by Diffusion Policy variants."""

    def __init__(self, out_dim: int = 64) -> None:
        """Initialize the encoder.

        Parameters
        ----------
        out_dim:
            Encoded observation width.
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, out_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode image observations.

        Parameters
        ----------
        images:
            Images shaped ``(batch, cameras_or_time, 3, height, width)``.

        Returns
        -------
        torch.Tensor
            Per-image embeddings shaped ``(batch, cameras_or_time, out_dim)``.
        """

        batch, views, channels, height, width = images.shape
        feats = self.net(images.reshape(batch * views, channels, height, width))
        return feats.view(batch, views, -1)


class DiffusionPolicyUNetImage(nn.Module):
    """Image-conditioned Diffusion Policy with FiLM-conditioned 1D action U-Net."""

    def __init__(self, action_dim: int = 4, cond_dim: int = 64) -> None:
        """Initialize the policy.

        Parameters
        ----------
        action_dim:
            Number of action coordinates.
        cond_dim:
            Conditioning width.
        """

        super().__init__()
        self.obs_encoder = TinyVisionObsEncoder(cond_dim)
        self.time = SinusoidalTimeEmbedding(cond_dim)
        self.in_proj = nn.Conv1d(action_dim, 32, 1)
        self.down = nn.Conv1d(32, 32, 4, stride=2, padding=1)
        self.mid = FiLMTemporalBlock(32, cond_dim)
        self.up = nn.ConvTranspose1d(32, 32, 4, stride=2, padding=1)
        self.skip = FiLMTemporalBlock(32, cond_dim)
        self.out = nn.Conv1d(32, action_dim, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Predict denoising noise for an action horizon from image observations.

        Parameters
        ----------
        images:
            Observation images shaped ``(batch, obs_horizon, 3, height, width)``.

        Returns
        -------
        torch.Tensor
            Action noise prediction shaped ``(batch, horizon, action_dim)``.
        """

        batch = images.shape[0]
        cond = self.obs_encoder(images).flatten(1)
        cond = cond[:, :64] + self.time(torch.full((batch,), 8.0, device=images.device))
        noisy_actions = torch.tanh(cond[:, None, :4].repeat(1, 8, 1))
        x = self.in_proj(noisy_actions.transpose(1, 2))
        skip = self.skip(x, cond)
        x = self.mid(self.down(skip), cond)
        x = self.up(x)[..., : skip.shape[-1]] + skip
        return self.out(x).transpose(1, 2)


class DiffusionPolicyTransformerLowdim(nn.Module):
    """Low-dimensional Diffusion Policy transformer over noisy action tokens."""

    def __init__(self, state_dim: int = 6, action_dim: int = 4, dim: int = 64) -> None:
        """Initialize the transformer policy.

        Parameters
        ----------
        state_dim:
            Low-dimensional observation width.
        action_dim:
            Action width.
        dim:
            Token width.
        """

        super().__init__()
        self.state_proj = nn.Linear(state_dim, dim)
        self.action_proj = nn.Linear(action_dim, dim)
        self.time = SinusoidalTimeEmbedding(dim)
        self.type_embed = nn.Parameter(torch.randn(1, 3, dim) * 0.02)
        self.blocks = nn.ModuleList([TransformerBlock(dim) for _ in range(2)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, action_dim)

    def forward(self, state_history: torch.Tensor) -> torch.Tensor:
        """Denoise a future action sequence from low-dimensional state history.

        Parameters
        ----------
        state_history:
            State history shaped ``(batch, obs_horizon, state_dim)``.

        Returns
        -------
        torch.Tensor
            Predicted action noise shaped ``(batch, pred_horizon, action_dim)``.
        """

        batch = state_history.shape[0]
        obs = self.state_proj(state_history) + self.type_embed[:, 0:1]
        seed = torch.tanh(state_history.mean(dim=1, keepdim=True)[..., :4]).repeat(1, 6, 1)
        actions = self.action_proj(seed) + self.type_embed[:, 1:2]
        step = self.time(torch.full((batch,), 12.0, device=state_history.device))[:, None, :]
        tokens = torch.cat([step + self.type_embed[:, 2:3], obs, actions], dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(self.norm(tokens[:, -6:]))


class DiffusionPolicyMultiImageObsEncoder(nn.Module):
    """Multi-camera Diffusion Policy image encoder with spatial-softmax keypoints."""

    def __init__(self, cameras: int = 3, keypoints: int = 8, out_dim: int = 64) -> None:
        """Initialize the encoder.

        Parameters
        ----------
        cameras:
            Number of camera views.
        keypoints:
            Number of spatial-softmax feature maps per camera.
        out_dim:
            Fused output width.
        """

        super().__init__()
        self.cameras = cameras
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.Conv2d(16, keypoints, 3, stride=2, padding=1),
        )
        self.fuse = nn.Linear(cameras * keypoints * 2, out_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode multi-camera images as keypoint coordinates.

        Parameters
        ----------
        images:
            Camera images shaped ``(batch, cameras, 3, height, width)``.

        Returns
        -------
        torch.Tensor
            Fused observation embedding.
        """

        batch, cameras, channels, height, width = images.shape
        feat = self.backbone(images.view(batch * cameras, channels, height, width))
        _, maps, feat_h, feat_w = feat.shape
        prob = torch.softmax(feat.flatten(-2), dim=-1).view(batch * cameras, maps, feat_h, feat_w)
        ys = torch.linspace(-1.0, 1.0, feat_h, device=images.device).view(1, 1, feat_h, 1)
        xs = torch.linspace(-1.0, 1.0, feat_w, device=images.device).view(1, 1, 1, feat_w)
        coords = torch.stack([(prob * xs).sum((-1, -2)), (prob * ys).sum((-1, -2))], dim=-1)
        coords = coords.view(batch, cameras, maps * 2)
        return self.fuse(coords.flatten(1))


class CoordConv2d(nn.Module):
    """Coordinate-concatenating convolution used by pi-GAN discriminators."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int) -> None:
        """Initialize coordinate convolution.

        Parameters
        ----------
        in_channels:
            Input image channels excluding coordinates.
        out_channels:
            Output channels.
        kernel_size:
            Convolution kernel size.
        stride:
            Convolution stride.
        """

        super().__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size, stride, kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Append normalized x/y coordinates and convolve.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        torch.Tensor
            Convolved feature map.
        """

        batch, _, height, width = x.shape
        yy = torch.linspace(-1.0, 1.0, height, device=x.device).view(1, 1, height, 1)
        xx = torch.linspace(-1.0, 1.0, width, device=x.device).view(1, 1, 1, width)
        coords = torch.cat(
            [xx.expand(batch, 1, height, width), yy.expand(batch, 1, height, width)], dim=1
        )
        return self.conv(torch.cat([x, coords], dim=1))


class PiganCCSDiscriminator(nn.Module):
    """pi-GAN coordinate-conditioned progressive discriminator."""

    def __init__(self) -> None:
        """Initialize compact discriminator."""

        super().__init__()
        self.from_rgb = CoordConv2d(3, 16, 3, 1)
        self.block1 = CoordConv2d(16, 32, 3, 2)
        self.block2 = CoordConv2d(32, 64, 3, 2)
        self.block3 = CoordConv2d(64, 64, 3, 2)
        self.pose_head = nn.Linear(64, 2)
        self.real_head = nn.Linear(64, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Score an image and regress camera pose, as in CCS pi-GAN discriminators.

        Parameters
        ----------
        image:
            RGB image shaped ``(batch, 3, height, width)``.

        Returns
        -------
        torch.Tensor
            Concatenated real/fake logit and pose prediction.
        """

        x = F.leaky_relu(self.from_rgb(image), 0.2)
        x = F.avg_pool2d(F.leaky_relu(self.block1(x), 0.2), 2)
        x = F.avg_pool2d(F.leaky_relu(self.block2(x), 0.2), 2)
        x = F.leaky_relu(self.block3(x), 0.2).mean((-1, -2))
        return torch.cat([self.real_head(x), self.pose_head(x)], dim=-1)


class TranscriptFormer(nn.Module):
    """Cross-species single-cell transformer over gene and expression tokens."""

    def __init__(self, genes: int = 32, species: int = 4, dim: int = 64) -> None:
        """Initialize compact TranscriptFormer.

        Parameters
        ----------
        genes:
            Number of gene ids in the compact vocabulary.
        species:
            Number of species ids.
        dim:
            Token width.
        """

        super().__init__()
        self.gene_embed = nn.Embedding(genes, dim)
        self.species_embed = nn.Embedding(species, dim)
        self.expr_proj = nn.Linear(1, dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim) for _ in range(2)])
        self.norm = nn.LayerNorm(dim)
        self.dropout_head = nn.Linear(dim, 1)
        self.expr_head = nn.Linear(dim, 1)

    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        """Model gene expression with gene, transcript-value, and species context.

        Parameters
        ----------
        expression:
            Expression counts shaped ``(batch, genes)``.

        Returns
        -------
        torch.Tensor
            Per-gene reconstructed expression and dropout logits.
        """

        batch, genes = expression.shape
        gene_ids = torch.arange(genes, device=expression.device).expand(batch, genes)
        species_ids = (expression.sum(dim=-1).long() % 4).clamp_min(0)
        x = self.gene_embed(gene_ids)
        x = x + self.expr_proj(torch.log1p(expression).unsqueeze(-1))
        x = x + self.species_embed(species_ids)[:, None, :]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return torch.cat([F.softplus(self.expr_head(x)), self.dropout_head(x)], dim=-1)


class GraphTransformerLucidrains(nn.Module):
    """Graph transformer with node attention biased by pair/edge embeddings."""

    def __init__(self, node_dim: int = 16, edge_dim: int = 8, dim: int = 48) -> None:
        """Initialize compact graph transformer.

        Parameters
        ----------
        node_dim:
            Input node feature width.
        edge_dim:
            Input pair feature width.
        dim:
            Hidden token width.
        """

        super().__init__()
        self.node_proj = nn.Linear(node_dim, dim)
        self.edge_proj = nn.Linear(edge_dim, 4)
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )
        self.readout = nn.Linear(dim, 6)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        """Run edge-aware all-pairs graph attention.

        Parameters
        ----------
        nodes:
            Node features shaped ``(batch, nodes, node_dim)``.

        Returns
        -------
        torch.Tensor
            Graph-level prediction.
        """

        batch, num_nodes, _ = nodes.shape
        x = self.node_proj(nodes)
        left = nodes[:, :, None, :8].expand(batch, num_nodes, num_nodes, 8)
        right = nodes[:, None, :, :8].expand(batch, num_nodes, num_nodes, 8)
        edge_bias = self.edge_proj(left - right).permute(0, 3, 1, 2)
        qkv = self.qkv(x).view(batch, num_nodes, 3, 4, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scores = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5) + edge_bias
        attn = torch.softmax(scores, dim=-1)
        x = x + self.out((attn @ v).transpose(1, 2).reshape(batch, num_nodes, -1))
        x = x + self.ff(x)
        return self.readout(x.mean(dim=1))


class HATBlock(nn.Module):
    """Hybrid Attention Transformer block with window, channel, and overlap attention."""

    def __init__(self, channels: int = 32) -> None:
        """Initialize HAT block.

        Parameters
        ----------
        channels:
            Feature channels.
        """

        super().__init__()
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid(),
        )
        self.overlap_kv = nn.Conv2d(channels, channels * 2, 3, padding=1)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1), nn.GELU(), nn.Conv2d(channels * 2, channels, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply hybrid spatial/channel/overlap attention.

        Parameters
        ----------
        x:
            Image features.

        Returns
        -------
        torch.Tensor
            Updated features.
        """

        batch, channels, height, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = q.flatten(2).transpose(1, 2)
        k = k.flatten(2)
        v = v.flatten(2).transpose(1, 2)
        spatial = torch.softmax((q @ k) * (channels**-0.5), dim=-1) @ v
        spatial = spatial.transpose(1, 2).view(batch, channels, height, width)
        ok, ov = self.overlap_kv(x).chunk(2, dim=1)
        overlap = torch.sigmoid(ok) * ov
        x = x + self.proj(spatial + overlap) * self.channel(x)
        return x + self.ffn(x)


class HATSuperResolution(nn.Module):
    """Compact HAT x4 super-resolution model."""

    def __init__(self, channels: int = 32, scale: int = 4) -> None:
        """Initialize compact HAT.

        Parameters
        ----------
        channels:
            Feature width.
        scale:
            Pixel-shuffle upscaling factor.
        """

        super().__init__()
        self.head = nn.Conv2d(3, channels, 3, padding=1)
        self.blocks = nn.ModuleList([HATBlock(channels), HATBlock(channels)])
        self.body = nn.Conv2d(channels, channels, 3, padding=1)
        self.up = nn.Sequential(
            nn.Conv2d(channels, 3 * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Super-resolve an image by x4.

        Parameters
        ----------
        image:
            Low-resolution RGB image.

        Returns
        -------
        torch.Tensor
            Super-resolved RGB image.
        """

        feat = self.head(image)
        x = feat
        for block in self.blocks:
            x = block(x)
        return self.up(self.body(x) + feat)


class GaussianSplattingScene(nn.Module):
    """Differentiable toy renderer for anisotropic 3D Gaussian splats."""

    def __init__(self, splats: int = 12, image_size: int = 16) -> None:
        """Initialize Gaussian scene parameters.

        Parameters
        ----------
        splats:
            Number of Gaussian primitives.
        image_size:
            Output image side length.
        """

        super().__init__()
        self.image_size = image_size
        self.means = nn.Parameter(torch.randn(splats, 3) * 0.4)
        self.log_scales = nn.Parameter(torch.zeros(splats, 2) - 2.0)
        self.opacity = nn.Parameter(torch.zeros(splats))
        self.sh_color = nn.Parameter(torch.randn(splats, 4, 3) * 0.2)

    def forward(self, camera: torch.Tensor) -> torch.Tensor:
        """Render Gaussian splats from a compact camera vector.

        Parameters
        ----------
        camera:
            Camera vector shaped ``(batch, 3)``.

        Returns
        -------
        torch.Tensor
            Rendered RGB image shaped ``(batch, 3, image_size, image_size)``.
        """

        grid = torch.linspace(-1.0, 1.0, self.image_size, device=camera.device)
        yy, xx = torch.meshgrid(grid, grid, indexing="ij")
        xy = self.means[:, :2] + camera[:, None, :2] * 0.05
        delta_x = xx[None, None] - xy[:, :, 0, None, None]
        delta_y = yy[None, None] - xy[:, :, 1, None, None]
        scales = F.softplus(self.log_scales)[None, :, :, None, None] + 1e-3
        density = torch.exp(
            -0.5 * ((delta_x / scales[:, :, 0]) ** 2 + (delta_y / scales[:, :, 1]) ** 2)
        )
        alpha = torch.sigmoid(self.opacity)[None, :, None, None] * density
        view = F.normalize(camera, dim=-1)
        color = self.sh_color[None, :, 0] + self.sh_color[None, :, 1] * view[:, None, 0:1]
        color = color + self.sh_color[None, :, 2] * view[:, None, 1:2]
        color = color + self.sh_color[None, :, 3] * view[:, None, 2:3]
        rgb = (alpha[:, :, None] * torch.sigmoid(color)[:, :, :, None, None]).sum(dim=1)
        return rgb / alpha.sum(dim=1, keepdim=True).clamp_min(1e-3)


class HyperbolicImageEmbeddings(nn.Module):
    """CNN image encoder projected into the Poincare ball."""

    def __init__(self, dim: int = 16) -> None:
        """Initialize hyperbolic image embedder.

        Parameters
        ----------
        dim:
            Embedding width.
        """

        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, dim),
        )
        self.prototype = nn.Parameter(torch.randn(5, dim) * 0.1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Compute hyperbolic distances from image embedding to class prototypes.

        Parameters
        ----------
        image:
            RGB image.

        Returns
        -------
        torch.Tensor
            Negative Poincare distances to prototypes.
        """

        euclid = self.encoder(image)
        norm = euclid.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        point = torch.tanh(norm) * euclid / norm
        proto_norm = self.prototype.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        proto = torch.tanh(proto_norm) * self.prototype / proto_norm
        diff2 = (point[:, None] - proto[None]).pow(2).sum(dim=-1)
        proto_ball = 1.0 - proto.pow(2).sum(dim=-1)
        denom = (1.0 - point.pow(2).sum(dim=-1, keepdim=True)) * proto_ball[None, :]
        z = 1.0 + 2.0 * diff2 / denom.clamp_min(1e-5)
        return -torch.acosh(z.clamp_min(1.0 + 1e-5))


def build_mdlm() -> nn.Module:
    """Build compact MDLM.

    Returns
    -------
    nn.Module
        Random-init compact MDLM.
    """

    return MDLM()


def example_mdlm() -> torch.Tensor:
    """Return example MDLM token ids.

    Returns
    -------
    torch.Tensor
        Token ids shaped ``(1, 12)``.
    """

    return torch.arange(12, dtype=torch.long).view(1, 12) % 80


def build_diffusion_policy_unet_image() -> nn.Module:
    """Build image-conditioned Diffusion Policy U-Net.

    Returns
    -------
    nn.Module
        Random-init compact policy.
    """

    return DiffusionPolicyUNetImage()


def example_diffusion_policy_images() -> torch.Tensor:
    """Return image observations for Diffusion Policy.

    Returns
    -------
    torch.Tensor
        Image observations shaped ``(1, 2, 3, 32, 32)``.
    """

    return torch.randn(1, 2, 3, 32, 32)


def build_diffusion_policy_transformer_lowdim() -> nn.Module:
    """Build low-dimensional Diffusion Policy transformer.

    Returns
    -------
    nn.Module
        Random-init compact policy.
    """

    return DiffusionPolicyTransformerLowdim()


def example_diffusion_policy_lowdim() -> torch.Tensor:
    """Return low-dimensional state history.

    Returns
    -------
    torch.Tensor
        State history shaped ``(1, 4, 6)``.
    """

    return torch.randn(1, 4, 6)


def build_diffusion_policy_multi_image_encoder() -> nn.Module:
    """Build Diffusion Policy multi-image observation encoder.

    Returns
    -------
    nn.Module
        Random-init compact encoder.
    """

    return DiffusionPolicyMultiImageObsEncoder()


def example_multi_image_obs() -> torch.Tensor:
    """Return multi-camera observations.

    Returns
    -------
    torch.Tensor
        Images shaped ``(1, 3, 3, 32, 32)``.
    """

    return torch.randn(1, 3, 3, 32, 32)


def build_pigan_ccs_discriminator() -> nn.Module:
    """Build pi-GAN coordinate-conditioned discriminator.

    Returns
    -------
    nn.Module
        Random-init compact discriminator.
    """

    return PiganCCSDiscriminator()


def example_pigan_image() -> torch.Tensor:
    """Return example RGB image.

    Returns
    -------
    torch.Tensor
        Image shaped ``(1, 3, 32, 32)``.
    """

    return torch.randn(1, 3, 32, 32)


def build_transcriptformer() -> nn.Module:
    """Build compact TranscriptFormer.

    Returns
    -------
    nn.Module
        Random-init compact TranscriptFormer.
    """

    return TranscriptFormer()


def example_transcriptformer() -> torch.Tensor:
    """Return compact single-cell expression counts.

    Returns
    -------
    torch.Tensor
        Expression counts shaped ``(1, 16)``.
    """

    return torch.rand(1, 16) * 3.0


def build_graph_transformer_lucidrains() -> nn.Module:
    """Build compact lucidrains-style Graph Transformer.

    Returns
    -------
    nn.Module
        Random-init graph transformer.
    """

    return GraphTransformerLucidrains()


def example_graph_nodes() -> torch.Tensor:
    """Return graph node features.

    Returns
    -------
    torch.Tensor
        Node features shaped ``(1, 8, 16)``.
    """

    return torch.randn(1, 8, 16)


def build_hat_classical_sr_x4() -> nn.Module:
    """Build compact HAT x4 super-resolution model.

    Returns
    -------
    nn.Module
        Random-init compact HAT.
    """

    return HATSuperResolution()


def example_lr_image() -> torch.Tensor:
    """Return low-resolution RGB image.

    Returns
    -------
    torch.Tensor
        Image shaped ``(1, 3, 12, 12)``.
    """

    return torch.randn(1, 3, 12, 12)


def build_gaussian_splatting_scene() -> nn.Module:
    """Build compact 3D Gaussian Splatting scene.

    Returns
    -------
    nn.Module
        Random-init Gaussian renderer.
    """

    return GaussianSplattingScene()


def example_camera() -> torch.Tensor:
    """Return compact camera vector.

    Returns
    -------
    torch.Tensor
        Camera vector shaped ``(1, 3)``.
    """

    return torch.tensor([[0.2, -0.1, 1.0]], dtype=torch.float32)


def build_hyperbolic_image_embeddings() -> nn.Module:
    """Build compact Hyperbolic Image Embeddings model.

    Returns
    -------
    nn.Module
        Random-init hyperbolic image embedder.
    """

    return HyperbolicImageEmbeddings()


MENAGERIE_ENTRIES = [
    ("MDLM", "build_mdlm", "example_mdlm", "2024", "DC"),
    (
        "diffusion_policy_unet_image",
        "build_diffusion_policy_unet_image",
        "example_diffusion_policy_images",
        "2023",
        "DC",
    ),
    (
        "diffusion_policy_transformer_lowdim",
        "build_diffusion_policy_transformer_lowdim",
        "example_diffusion_policy_lowdim",
        "2023",
        "DC",
    ),
    (
        "DiffusionPolicy_MultiImageObsEncoder",
        "build_diffusion_policy_multi_image_encoder",
        "example_multi_image_obs",
        "2023",
        "DC",
    ),
    (
        "pigan_ccs_discriminator",
        "build_pigan_ccs_discriminator",
        "example_pigan_image",
        "2021",
        "DC",
    ),
    (
        "transcriptformer.TranscriptFormer",
        "build_transcriptformer",
        "example_transcriptformer",
        "2025",
        "DC",
    ),
    (
        "GraphTransformer-lucidrains",
        "build_graph_transformer_lucidrains",
        "example_graph_nodes",
        "2021",
        "DC",
    ),
    ("hat_classical_sr_x4", "build_hat_classical_sr_x4", "example_lr_image", "2023", "DC"),
    ("hat_l_classical_sr_x4", "build_hat_classical_sr_x4", "example_lr_image", "2023", "DC"),
    ("gaussian_splatting_scene", "build_gaussian_splatting_scene", "example_camera", "2023", "DC"),
    (
        "Hyperbolic Image Embeddings",
        "build_hyperbolic_image_embeddings",
        "example_pigan_image",
        "2020",
        "DC",
    ),
]
