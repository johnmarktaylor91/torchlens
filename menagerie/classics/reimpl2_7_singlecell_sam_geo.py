"""Compact dependency-gated reimplementations for REIMPL2 shard 7.

Sources checked: scvi-tools model docs for scVI/scANVI/totalVI/MultiVI/PeakVI/
DestVI/veloVI, Fuchs et al. 2020 SE(3)-Transformer, Kirillov et al. 2023 SAM,
and Linsley et al. 2018 hGRU.  These models are small random-init PyTorch
reconstructions that preserve the architecture-defining data flow while avoiding
dependency-heavy packages.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Small fully connected network used by compact probabilistic models."""

    def __init__(self, dims: list[int], final_activation: bool = False) -> None:
        """Initialize linear blocks.

        Parameters
        ----------
        dims:
            Consecutive feature dimensions.
        final_activation:
            Whether to apply GELU after the final linear layer.
        """
        super().__init__()
        layers: list[nn.Module] = []
        for index, (din, dout) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            layers.append(nn.Linear(din, dout))
            if final_activation or index < len(dims) - 2:
                layers.append(nn.LayerNorm(dout))
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the multilayer perceptron.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        Tensor
            Transformed tensor.
        """
        return self.net(x)


class NormalEncoder(nn.Module):
    """Encoder returning mean and positive scale for a diagonal normal."""

    def __init__(self, in_dim: int, latent_dim: int, hidden: int = 32) -> None:
        """Initialize an inference encoder.

        Parameters
        ----------
        in_dim:
            Input feature dimension.
        latent_dim:
            Latent variable dimension.
        hidden:
            Hidden width.
        """
        super().__init__()
        self.body = MLP([in_dim, hidden, hidden], final_activation=True)
        self.mean = nn.Linear(hidden, latent_dim)
        self.scale = nn.Linear(hidden, latent_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Encode features into a deterministic random-init posterior sample.

        Parameters
        ----------
        x:
            Observed features.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Mean, positive scale, and reparameterized sample.
        """
        h = self.body(torch.log1p(x.clamp_min(0)))
        mean = self.mean(h)
        scale = F.softplus(self.scale(h)) + 1e-4
        sample = mean + 0.1 * scale
        return mean, scale, sample


class ScVIVAE(nn.Module):
    """scVI-style gene-count VAE with library-size latent and NB decoder."""

    def __init__(self, genes: int = 24, latent: int = 8, batches: int = 3) -> None:
        """Initialize compact scVI.

        Parameters
        ----------
        genes:
            Number of gene-count features.
        latent:
            Latent dimension.
        batches:
            Number of batch covariate categories.
        """
        super().__init__()
        self.z_encoder = NormalEncoder(genes + batches, latent)
        self.l_encoder = NormalEncoder(genes, 1)
        self.decoder = MLP([latent + batches, 32, genes])
        self.dispersion = nn.Parameter(torch.zeros(genes))

    def forward(self, counts: Tensor, batch_onehot: Tensor) -> Tensor:
        """Return reconstructed NB mean, dispersion, and latent statistics.

        Parameters
        ----------
        counts:
            Gene count matrix.
        batch_onehot:
            Batch covariate one-hot matrix.

        Returns
        -------
        Tensor
            Concatenated decoder statistics.
        """
        z_mean, z_scale, z = self.z_encoder(torch.cat((counts, batch_onehot), dim=-1))
        lib_mean, lib_scale, lib = self.l_encoder(counts)
        proportions = torch.softmax(self.decoder(torch.cat((z, batch_onehot), dim=-1)), dim=-1)
        mean = torch.exp(lib).clamp_max(1e4) * proportions
        theta = F.softplus(self.dispersion).expand_as(mean)
        return torch.cat((mean, theta, z_mean, z_scale, lib_mean, lib_scale), dim=-1)


class ScANVI(nn.Module):
    """scANVI-style semi-supervised VAE with classifier and label-conditioned decoder."""

    def __init__(self, genes: int = 24, latent: int = 8, labels: int = 4) -> None:
        """Initialize compact scANVI.

        Parameters
        ----------
        genes:
            Number of gene-count features.
        latent:
            Latent dimension.
        labels:
            Cell-label categories.
        """
        super().__init__()
        self.encoder = NormalEncoder(genes, latent)
        self.classifier = MLP([genes, 32, labels])
        self.decoder = MLP([latent + labels, 32, genes])

    def forward(self, counts: Tensor) -> Tensor:
        """Classify cells and decode with soft label conditioning.

        Parameters
        ----------
        counts:
            Gene count matrix.

        Returns
        -------
        Tensor
            Concatenated class probabilities and reconstructed expression.
        """
        mean, scale, z = self.encoder(counts)
        label_probs = torch.softmax(self.classifier(torch.log1p(counts)), dim=-1)
        px = F.softplus(self.decoder(torch.cat((z, label_probs), dim=-1)))
        return torch.cat((label_probs, px, mean, scale), dim=-1)


class TotalVI(nn.Module):
    """totalVI-style joint RNA/protein VAE with protein background branch."""

    def __init__(self, genes: int = 20, proteins: int = 6, latent: int = 8) -> None:
        """Initialize compact totalVI.

        Parameters
        ----------
        genes:
            RNA feature count.
        proteins:
            Protein feature count.
        latent:
            Latent dimension.
        """
        super().__init__()
        self.encoder = NormalEncoder(genes + proteins, latent)
        self.rna_decoder = MLP([latent, 32, genes])
        self.protein_foreground = MLP([latent, 24, proteins])
        self.protein_background = MLP([latent, 24, proteins])
        self.mixing = MLP([latent, 24, proteins])

    def forward(self, rna: Tensor, protein: Tensor) -> Tensor:
        """Decode joint CITE-seq observations.

        Parameters
        ----------
        rna:
            RNA counts.
        protein:
            Protein counts.

        Returns
        -------
        Tensor
            RNA mean, protein mixture parameters, and latent statistics.
        """
        mean, scale, z = self.encoder(torch.cat((rna, protein), dim=-1))
        rna_mean = F.softplus(self.rna_decoder(z))
        foreground = F.softplus(self.protein_foreground(z))
        background = F.softplus(self.protein_background(z))
        mix = torch.sigmoid(self.mixing(z))
        return torch.cat((rna_mean, foreground, background, mix, mean, scale), dim=-1)


class MultiVI(nn.Module):
    """MultiVI-style paired RNA/ATAC VAE with modality-specific encoders."""

    def __init__(self, genes: int = 18, peaks: int = 16, latent: int = 8) -> None:
        """Initialize compact MultiVI.

        Parameters
        ----------
        genes:
            RNA feature count.
        peaks:
            ATAC peak feature count.
        latent:
            Latent dimension.
        """
        super().__init__()
        self.rna_encoder = NormalEncoder(genes, latent)
        self.atac_encoder = NormalEncoder(peaks, latent)
        self.joint_encoder = NormalEncoder(genes + peaks, latent)
        self.rna_decoder = MLP([latent, 32, genes])
        self.atac_decoder = MLP([latent, 32, peaks])

    def forward(self, rna: Tensor, atac: Tensor, modality_mask: Tensor) -> Tensor:
        """Fuse modality-specific posteriors and decode both modalities.

        Parameters
        ----------
        rna:
            RNA counts.
        atac:
            ATAC accessibility counts.
        modality_mask:
            Three columns selecting RNA, ATAC, or paired posterior.

        Returns
        -------
        Tensor
            Concatenated modality reconstructions and posterior statistics.
        """
        rna_mean, rna_scale, zr = self.rna_encoder(rna)
        atac_mean, atac_scale, za = self.atac_encoder(atac)
        joint_mean, joint_scale, zj = self.joint_encoder(torch.cat((rna, atac), dim=-1))
        z = modality_mask[:, 0:1] * zr + modality_mask[:, 1:2] * za + modality_mask[:, 2:3] * zj
        rna_out = F.softplus(self.rna_decoder(z))
        atac_out = torch.sigmoid(self.atac_decoder(z))
        return torch.cat(
            (
                rna_out,
                atac_out,
                rna_mean,
                atac_mean,
                joint_mean,
                rna_scale,
                atac_scale,
                joint_scale,
            ),
            dim=-1,
        )


class PeakVI(nn.Module):
    """PeakVI-style chromatin accessibility VAE with Bernoulli decoder."""

    def __init__(self, peaks: int = 32, latent: int = 8) -> None:
        """Initialize compact PeakVI.

        Parameters
        ----------
        peaks:
            Number of ATAC peaks.
        latent:
            Latent dimension.
        """
        super().__init__()
        self.encoder = NormalEncoder(peaks, latent)
        self.decoder = MLP([latent, 32, peaks])
        self.depth_decoder = MLP([latent, 16, 1])

    def forward(self, peaks: Tensor) -> Tensor:
        """Decode accessibility probabilities and cell depth.

        Parameters
        ----------
        peaks:
            Binary or count accessibility matrix.

        Returns
        -------
        Tensor
            Peak probabilities, depth, and posterior statistics.
        """
        mean, scale, z = self.encoder(peaks)
        prob = torch.sigmoid(self.decoder(z))
        depth = F.softplus(self.depth_decoder(z))
        return torch.cat((prob, depth, mean, scale), dim=-1)


class DestVI(nn.Module):
    """DestVI-style spatial deconvolution using cell-type decoder programs."""

    def __init__(self, genes: int = 18, cell_types: int = 4, latent: int = 6) -> None:
        """Initialize compact DestVI.

        Parameters
        ----------
        genes:
            Spatial transcript feature count.
        cell_types:
            Number of deconvolved cell types.
        latent:
            Cell-type variation latent dimension.
        """
        super().__init__()
        self.proportion_net = MLP([genes, 32, cell_types])
        self.gamma_encoder = NormalEncoder(genes, latent)
        self.cell_type_decoder = nn.ModuleList(
            [MLP([latent, 24, genes]) for _ in range(cell_types)]
        )

    def forward(self, spatial_counts: Tensor) -> Tensor:
        """Infer cell-type proportions and mix decoded expression programs.

        Parameters
        ----------
        spatial_counts:
            Spatial gene-count matrix.

        Returns
        -------
        Tensor
            Mixed expression and inferred cell-type proportions.
        """
        prop = torch.softmax(self.proportion_net(torch.log1p(spatial_counts)), dim=-1)
        _, _, gamma = self.gamma_encoder(spatial_counts)
        programs = torch.stack([F.softplus(dec(gamma)) for dec in self.cell_type_decoder], dim=1)
        mixed = (prop.unsqueeze(-1) * programs).sum(dim=1)
        return torch.cat((mixed, prop), dim=-1)


class VeloVI(nn.Module):
    """veloVI-style dynamical VAE producing spliced/unspliced RNA rates."""

    def __init__(self, genes: int = 16, latent: int = 6) -> None:
        """Initialize compact veloVI.

        Parameters
        ----------
        genes:
            Number of genes with spliced and unspliced observations.
        latent:
            Latent dimension.
        """
        super().__init__()
        self.encoder = NormalEncoder(genes * 2, latent)
        self.time_head = nn.Linear(latent, 1)
        self.rate_head = MLP([latent + 1, 32, genes * 4])

    def forward(self, unspliced: Tensor, spliced: Tensor) -> Tensor:
        """Infer latent time and RNA velocity rate parameters.

        Parameters
        ----------
        unspliced:
            Unspliced counts.
        spliced:
            Spliced counts.

        Returns
        -------
        Tensor
            Latent time, rate parameters, and posterior statistics.
        """
        mean, scale, z = self.encoder(torch.cat((unspliced, spliced), dim=-1))
        time = torch.sigmoid(self.time_head(z))
        rates = F.softplus(self.rate_head(torch.cat((z, time), dim=-1)))
        alpha, beta, gamma, switch = rates.chunk(4, dim=-1)
        velocity = alpha - beta * unspliced + gamma * spliced - switch
        return torch.cat((time, velocity, alpha, beta, gamma, switch, mean, scale), dim=-1)


class EquivariantGraphAttention(nn.Module):
    """Compact SE(3)-Transformer scalar/vector attention layer."""

    def __init__(self, channels: int = 16) -> None:
        """Initialize invariant attention and equivariant vector update.

        Parameters
        ----------
        channels:
            Scalar feature channel count.
        """
        super().__init__()
        self.q = nn.Linear(channels, channels, bias=False)
        self.k = nn.Linear(channels, channels, bias=False)
        self.v = nn.Linear(channels, channels, bias=False)
        self.radial = MLP([1, 16, channels])
        self.spherical = nn.Linear(9, channels)
        self.tensor_gate = nn.Linear(channels, channels)
        self.vector_gate = nn.Linear(channels, 1)
        self.out = nn.Linear(channels, channels)

    def forward(self, h: Tensor, xyz: Tensor) -> tuple[Tensor, Tensor]:
        """Apply attention over relative coordinates.

        Parameters
        ----------
        h:
            Scalar node features ``(batch, nodes, channels)``.
        xyz:
            Coordinates ``(batch, nodes, 3)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Updated scalar features and equivariant vector features.
        """
        rel = xyz[:, :, None, :] - xyz[:, None, :, :]
        dist = rel.square().sum(dim=-1, keepdim=True).sqrt()
        unit = rel / dist.clamp_min(1e-6)
        harmonics = torch.cat(
            (
                unit,
                unit[..., :1] * unit,
                unit[..., 1:2] * unit,
            ),
            dim=-1,
        )
        radial = self.radial(dist)
        tensor_field = self.spherical(harmonics) * self.tensor_gate(radial)
        score = (self.q(h)[:, :, None, :] * self.k(h)[:, None, :, :] * (radial + tensor_field)).sum(
            dim=-1
        )
        attn = torch.softmax(score / math.sqrt(h.shape[-1]), dim=-1)
        values = self.v(h)
        scalar = torch.einsum("bij,bjc->bic", attn, values) + torch.einsum(
            "bij,bijc->bic", attn, tensor_field
        )
        gate = torch.tanh(self.vector_gate(radial))
        vector = torch.einsum("bij,bijc->bic", attn, gate * unit)
        return self.out(scalar), vector


class SE3TransformerFuchs(nn.Module):
    """Fuchs-style SE(3)-Transformer for point clouds with scalar/vector channels."""

    def __init__(self, in_dim: int = 6, channels: int = 16, layers: int = 2) -> None:
        """Initialize compact SE(3)-Transformer.

        Parameters
        ----------
        in_dim:
            Input scalar feature dimension.
        channels:
            Hidden scalar channels.
        layers:
            Number of equivariant attention layers.
        """
        super().__init__()
        self.embed = nn.Linear(in_dim, channels)
        self.layers = nn.ModuleList([EquivariantGraphAttention(channels) for _ in range(layers)])
        self.readout = nn.Linear(channels + 3, 5)

    def forward(self, features: Tensor, xyz: Tensor) -> Tensor:
        """Run equivariant graph attention and invariant pooling.

        Parameters
        ----------
        features:
            Node scalar features.
        xyz:
            Node coordinates.

        Returns
        -------
        Tensor
            Graph-level prediction.
        """
        h = self.embed(features)
        vector_sum = torch.zeros_like(xyz)
        for layer in self.layers:
            dh, dv = layer(h, xyz)
            h = h + F.gelu(dh)
            vector_sum = vector_sum + dv
        pooled = torch.cat(
            (h.mean(dim=1), vector_sum.norm(dim=-1).mean(dim=1, keepdim=True).expand(-1, 3)), dim=-1
        )
        return self.readout(pooled)


class CompactSAM(nn.Module):
    """Compact SAM source-style ViT encoder, prompt encoder, and mask decoder."""

    def __init__(self, dim: int = 48, heads: int = 4, blocks: int = 3) -> None:
        """Initialize a compact SAM variant.

        Parameters
        ----------
        dim:
            Token width.
        heads:
            Attention heads.
        blocks:
            Number of image-encoder transformer blocks.
        """
        super().__init__()
        self.dim = dim
        self.patch = nn.Conv2d(3, dim, kernel_size=8, stride=8)
        self.pos = nn.Parameter(torch.zeros(1, 64, dim))
        encoder = nn.TransformerEncoderLayer(
            dim, heads, dim * 4, batch_first=True, activation="gelu"
        )
        self.image_encoder = nn.TransformerEncoder(encoder, blocks)
        self.point_encoder = nn.Linear(3, dim)
        self.iou_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.cross_a = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.0)
        self.cross_b = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.0)
        self.up = nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2)
        self.hyper = MLP([dim, dim, dim])
        self.iou = nn.Linear(dim, 1)

    def forward(self, image: Tensor, points: Tensor) -> Tensor:
        """Predict prompt-conditioned mask logits and IoU token output.

        Parameters
        ----------
        image:
            Input image ``(batch, 3, 64, 64)``.
        points:
            Prompt points with ``x, y, label`` columns.

        Returns
        -------
        Tensor
            Mask logits with an appended IoU row.
        """
        batch = image.shape[0]
        tokens = self.patch(image).flatten(2).transpose(1, 2) + self.pos
        image_tokens = self.image_encoder(tokens)
        prompt = torch.cat(
            (
                self.iou_token.expand(batch, -1, -1),
                self.mask_token.expand(batch, -1, -1),
                self.point_encoder(points),
            ),
            dim=1,
        )
        prompt = prompt + self.cross_a(prompt, image_tokens, image_tokens, need_weights=False)[0]
        image_tokens = (
            image_tokens + self.cross_b(image_tokens, prompt, prompt, need_weights=False)[0]
        )
        fmap = self.up(image_tokens.transpose(1, 2).reshape(batch, self.dim, 8, 8))
        weights = self.hyper(prompt[:, 1])
        mask = torch.einsum("bc,bchw->bhw", weights, fmap).unsqueeze(1)
        iou = self.iou(prompt[:, 0]).view(batch, 1, 1, 1).expand(-1, -1, 1, mask.shape[-1])
        return torch.cat((mask, iou), dim=2)


class HGRUCell(nn.Module):
    """Horizontal gated recurrent unit cell over convolutional feature maps."""

    def __init__(self, channels: int = 16) -> None:
        """Initialize hGRU gates and horizontal kernels.

        Parameters
        ----------
        channels:
            Feature-map channels.
        """
        super().__init__()
        self.g1 = nn.Conv2d(channels, channels, 1)
        self.g2 = nn.Conv2d(channels, channels, 1)
        self.horizontal_inhibit = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.horizontal_excite = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.mix = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor, state: Tensor) -> Tensor:
        """Update recurrent state with inhibitory and excitatory horizontal gates.

        Parameters
        ----------
        x:
            Feedforward drive.
        state:
            Previous recurrent state.

        Returns
        -------
        Tensor
            Updated state.
        """
        gate1 = torch.sigmoid(self.g1(state))
        inhibited = F.relu(x - self.horizontal_inhibit(gate1 * state))
        gate2 = torch.sigmoid(self.g2(inhibited))
        excited = F.relu(self.horizontal_excite(inhibited))
        candidate = self.mix(excited)
        return (1.0 - gate2) * state + gate2 * candidate


class BaseHGRU(nn.Module):
    """Serre-lab BasehGRU-style recurrent contour model."""

    def __init__(self, channels: int = 16, steps: int = 4) -> None:
        """Initialize compact hGRU network.

        Parameters
        ----------
        channels:
            Recurrent channel count.
        steps:
            Number of recurrent updates.
        """
        super().__init__()
        self.steps = steps
        self.stem = nn.Conv2d(1, channels, 7, padding=3)
        self.cell = HGRUCell(channels)
        self.head = nn.Conv2d(channels, 2, 1)

    def forward(self, image: Tensor) -> Tensor:
        """Run recurrent horizontal grouping over an image.

        Parameters
        ----------
        image:
            Single-channel image.

        Returns
        -------
        Tensor
            Pixel logits.
        """
        drive = F.relu(self.stem(image))
        state = torch.zeros_like(drive)
        for _ in range(self.steps):
            state = self.cell(drive, state)
        return self.head(state)


def build_scvi() -> nn.Module:
    """Build compact scVI."""
    return ScVIVAE()


def example_scvi() -> tuple[Tensor, Tensor]:
    """Return scVI count and batch example."""
    return torch.poisson(torch.ones(3, 24) * 2.0), F.one_hot(torch.tensor([0, 1, 2]), 3).float()


def build_scanvi() -> nn.Module:
    """Build compact scANVI."""
    return ScANVI()


def example_scanvi() -> Tensor:
    """Return scANVI count example."""
    return torch.poisson(torch.ones(3, 24) * 2.0)


def build_totalvi() -> nn.Module:
    """Build compact totalVI."""
    return TotalVI()


def example_totalvi() -> tuple[Tensor, Tensor]:
    """Return totalVI RNA/protein example."""
    return torch.poisson(torch.ones(3, 20) * 2.0), torch.poisson(torch.ones(3, 6))


def build_multivi() -> nn.Module:
    """Build compact MultiVI."""
    return MultiVI()


def example_multivi() -> tuple[Tensor, Tensor, Tensor]:
    """Return MultiVI RNA/ATAC/mask example."""
    return (
        torch.poisson(torch.ones(3, 18) * 2.0),
        torch.bernoulli(torch.full((3, 16), 0.25)),
        F.one_hot(torch.tensor([0, 1, 2]), 3).float(),
    )


def build_peakvi() -> nn.Module:
    """Build compact PeakVI."""
    return PeakVI()


def example_peakvi() -> Tensor:
    """Return PeakVI accessibility example."""
    return torch.bernoulli(torch.full((3, 32), 0.2))


def build_destvi() -> nn.Module:
    """Build compact DestVI."""
    return DestVI()


def example_destvi() -> Tensor:
    """Return DestVI spatial count example."""
    return torch.poisson(torch.ones(3, 18) * 3.0)


def build_velovi() -> nn.Module:
    """Build compact veloVI."""
    return VeloVI()


def example_velovi() -> tuple[Tensor, Tensor]:
    """Return veloVI unspliced/spliced count example."""
    return torch.poisson(torch.ones(3, 16)), torch.poisson(torch.ones(3, 16) * 2.0)


def build_se3_transformer_fuchs() -> nn.Module:
    """Build compact SE(3)-Transformer."""
    return SE3TransformerFuchs()


def example_se3_transformer_fuchs() -> tuple[Tensor, Tensor]:
    """Return point-cloud features and coordinates."""
    return torch.randn(1, 8, 6), torch.randn(1, 8, 3)


def build_sam_vit_b_source() -> nn.Module:
    """Build compact SAM ViT-B-style source reconstruction."""
    return CompactSAM(dim=48, heads=4, blocks=3)


def build_sam_vit_h_source() -> nn.Module:
    """Build compact SAM ViT-H-style source reconstruction."""
    return CompactSAM(dim=64, heads=4, blocks=4)


def example_sam_source() -> tuple[Tensor, Tensor]:
    """Return image and prompt-point inputs for compact SAM."""
    return torch.randn(1, 3, 64, 64), torch.tensor([[[0.25, 0.25, 1.0], [0.75, 0.5, 0.0]]])


def build_basehgru() -> nn.Module:
    """Build compact BasehGRU."""
    return BaseHGRU()


def example_basehgru() -> Tensor:
    """Return a small grayscale image."""
    return torch.randn(1, 1, 32, 32)


MENAGERIE_ENTRIES = [
    ("scVI", "build_scvi", "example_scvi", "2018", "DC"),
    ("scANVI", "build_scanvi", "example_scanvi", "2020", "DC"),
    ("totalVI", "build_totalvi", "example_totalvi", "2020", "DC"),
    ("MultiVI", "build_multivi", "example_multivi", "2021", "DC"),
    ("peakVI", "build_peakvi", "example_peakvi", "2021", "DC"),
    ("DestVI", "build_destvi", "example_destvi", "2022", "DC"),
    ("veloVI", "build_velovi", "example_velovi", "2022", "DC"),
    (
        "se3_transformer_fuchs",
        "build_se3_transformer_fuchs",
        "example_se3_transformer_fuchs",
        "2020",
        "DC",
    ),
    ("SAM_ViT_B_Source", "build_sam_vit_b_source", "example_sam_source", "2023", "DC"),
    ("SAM_ViT_H_Source", "build_sam_vit_h_source", "example_sam_source", "2023", "DC"),
    ("serrelab_BasehGRU", "build_basehgru", "example_basehgru", "2018", "DC"),
]
