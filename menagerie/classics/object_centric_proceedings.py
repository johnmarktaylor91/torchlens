"""Object-centric architectures from major 2019-2020 proceedings.

Compact random-init PyTorch reimplementations for the TorchLens menagerie:

* PSGNet, Bear et al., "Learning Physical Graph Representations from Visual
  Scenes", NeurIPS 2020.
* Slot Attention, Locatello et al., "Object-Centric Learning with Slot
  Attention", NeurIPS 2020.
* IODINE, Greff et al., "Multi-Object Representation Learning with Iterative
  Variational Inference", ICML 2019.

The implementations keep the distinctive architectural primitives visible in a
small forward graph: PSGNet's recurrent visual feedback plus graph pooling and
message passing, Slot Attention's iterative slot binding, and IODINE's repeated
decode/residual/refinement loop over object slots.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvFeedbackBlock(nn.Module):
    """Convolutional encoder block with recurrent top-down feedback."""

    def __init__(self, channels: int) -> None:
        """Initialize the convolutional feedback block.

        Parameters
        ----------
        channels:
            Number of feature channels.
        """

        super().__init__()
        self.bottom_up = nn.Conv2d(channels, channels, 3, padding=1)
        self.feedback = nn.Conv2d(channels, channels, 3, padding=1)
        self.gate = nn.Conv2d(2 * channels, channels, 1)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Update visual features using bottom-up and feedback terms.

        Parameters
        ----------
        x:
            Current feature map ``(B, C, H, W)``.
        state:
            Previous recurrent state ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Updated recurrent feature map.
        """

        proposal = torch.tanh(self.bottom_up(x) + self.feedback(state))
        gate = torch.sigmoid(self.gate(torch.cat([x, state], dim=1)))
        return gate * state + (1.0 - gate) * proposal


class PSGNet(nn.Module):
    """Physical Scene Graph Network with graph-structured bottleneck."""

    def __init__(self, in_channels: int = 3, channels: int = 24, nodes: int = 5) -> None:
        """Initialize a compact PSGNet.

        Parameters
        ----------
        in_channels:
            Number of image channels.
        channels:
            Width of visual and graph features.
        nodes:
            Number of latent physical scene graph nodes.
        """

        super().__init__()
        self.nodes = nodes
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
        )
        self.feedback = ConvFeedbackBlock(channels)
        self.assignment = nn.Conv2d(channels, nodes, 1)
        self.node_proj = nn.Linear(channels + 2, channels)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.ReLU(),
            nn.Linear(channels, 1),
        )
        self.message = nn.Linear(channels, channels)
        self.decoder = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Infer object-part graph nodes and reconstruct the image.

        Parameters
        ----------
        image:
            Input image tensor ``(B, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Reconstructed image tensor ``(B, 3, H, W)``.
        """

        feat = self.stem(image)
        state = torch.zeros_like(feat)
        for _ in range(2):
            state = self.feedback(feat, state)
            feat = feat + state

        batch, channels, height, width = feat.shape
        logits = self.assignment(feat).flatten(2)
        masks = torch.softmax(logits, dim=1)
        flat_feat = feat.flatten(2).transpose(1, 2)
        node_feat = torch.bmm(masks, flat_feat) / masks.sum(-1, keepdim=True).clamp_min(1e-6)

        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=image.device, dtype=image.dtype),
            torch.linspace(-1.0, 1.0, width, device=image.device, dtype=image.dtype),
            indexing="ij",
        )
        coords = torch.stack([yy, xx], dim=-1).view(1, height * width, 2).expand(batch, -1, -1)
        node_pos = torch.bmm(masks, coords) / masks.sum(-1, keepdim=True).clamp_min(1e-6)
        nodes = self.node_proj(torch.cat([node_feat, node_pos], dim=-1))

        left = nodes.unsqueeze(2).expand(-1, -1, self.nodes, -1)
        right = nodes.unsqueeze(1).expand(-1, self.nodes, -1, -1)
        edge_logits = self.edge_mlp(torch.cat([left, right], dim=-1)).squeeze(-1)
        adjacency = torch.softmax(edge_logits, dim=-1)
        nodes = nodes + torch.bmm(adjacency, self.message(nodes))

        pixel_nodes = torch.bmm(masks.transpose(1, 2), nodes)
        graph_map = pixel_nodes.transpose(1, 2).view(batch, channels, height, width)
        return self.decoder(graph_map + feat)


class SlotAttention(nn.Module):
    """Iterative attention module mapping perceptual tokens to object slots."""

    def __init__(self, dim: int = 32, slots: int = 4, iters: int = 3) -> None:
        """Initialize Slot Attention.

        Parameters
        ----------
        dim:
            Slot and token feature dimension.
        slots:
            Number of exchangeable slots.
        iters:
            Number of attention/refinement iterations.
        """

        super().__init__()
        self.slots = slots
        self.iters = iters
        self.slot_mu = nn.Parameter(torch.zeros(1, slots, dim))
        self.slot_logsigma = nn.Parameter(torch.zeros(1, slots, dim))
        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, dim)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Bind input tokens to slots with iterative soft attention.

        Parameters
        ----------
        tokens:
            Perceptual tokens ``(B, N, D)``.

        Returns
        -------
        torch.Tensor
            Slot tensor ``(B, K, D)``.
        """

        batch = tokens.shape[0]
        inputs = self.norm_inputs(tokens)
        keys = self.to_k(inputs)
        values = self.to_v(inputs)
        slots = self.slot_mu + torch.exp(self.slot_logsigma) * torch.zeros(
            batch, self.slots, self.slot_mu.shape[-1], device=tokens.device, dtype=tokens.dtype
        )
        for _ in range(self.iters):
            prev = slots
            queries = self.to_q(self.norm_slots(slots))
            attn = torch.softmax(
                torch.matmul(keys, queries.transpose(1, 2)) / (queries.shape[-1] ** 0.5), dim=-1
            )
            attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-6)
            updates = torch.bmm(attn.transpose(1, 2), values)
            slots = self.gru(
                updates.reshape(-1, updates.shape[-1]), prev.reshape(-1, prev.shape[-1])
            )
            slots = slots.view(batch, self.slots, -1)
            slots = slots + self.mlp(slots)
        return slots


class SlotAttentionAutoEncoder(nn.Module):
    """Slot Attention autoencoder with spatial-broadcast-style decoder."""

    def __init__(self, channels: int = 32, slots: int = 4) -> None:
        """Initialize the compact Slot Attention autoencoder.

        Parameters
        ----------
        channels:
            Feature width.
        slots:
            Number of object slots.
        """

        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, channels, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 5, padding=2),
            nn.ReLU(),
        )
        self.position = nn.Linear(2, channels)
        self.slot_attention = SlotAttention(channels, slots)
        self.slot_decoder = nn.Sequential(
            nn.Linear(channels + 2, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, 4),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Reconstruct an image by decoding and alpha-compositing slots.

        Parameters
        ----------
        image:
            Input image tensor ``(B, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Reconstructed image tensor ``(B, 3, H, W)``.
        """

        feat = self.encoder(image)
        batch, channels, height, width = feat.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=image.device, dtype=image.dtype),
            torch.linspace(-1.0, 1.0, width, device=image.device, dtype=image.dtype),
            indexing="ij",
        )
        coords = torch.stack([yy, xx], dim=-1).view(1, height * width, 2).expand(batch, -1, -1)
        tokens = feat.flatten(2).transpose(1, 2) + self.position(coords)
        slots = self.slot_attention(tokens)
        slot_grid = slots[:, :, None, :].expand(-1, -1, height * width, -1)
        coord_grid = coords[:, None].expand(-1, slots.shape[1], -1, -1)
        decoded = self.slot_decoder(torch.cat([slot_grid, coord_grid], dim=-1))
        rgb = torch.sigmoid(decoded[..., :3])
        masks = torch.softmax(decoded[..., 3:4], dim=1)
        recon = (rgb * masks).sum(dim=1).transpose(1, 2).view(batch, 3, height, width)
        return recon


class IodineRefiner(nn.Module):
    """Amortized refinement network for IODINE object slots."""

    def __init__(self, slot_dim: int) -> None:
        """Initialize the slot refinement MLP.

        Parameters
        ----------
        slot_dim:
            Slot latent dimension.
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(slot_dim + 3, 2 * slot_dim),
            nn.ReLU(),
            nn.Linear(2 * slot_dim, slot_dim),
        )

    def forward(self, slots: torch.Tensor, residual_summary: torch.Tensor) -> torch.Tensor:
        """Predict slot updates from image residual summaries.

        Parameters
        ----------
        slots:
            Current slots ``(B, K, D)``.
        residual_summary:
            Per-slot residual statistics ``(B, K, 3)``.

        Returns
        -------
        torch.Tensor
            Slot update tensor ``(B, K, D)``.
        """

        return self.net(torch.cat([slots, residual_summary], dim=-1))


class IODINE(nn.Module):
    """Iterative Object Decomposition Inference Network."""

    def __init__(self, slots: int = 4, slot_dim: int = 32, iters: int = 3) -> None:
        """Initialize compact IODINE.

        Parameters
        ----------
        slots:
            Number of object slots.
        slot_dim:
            Latent dimension per slot.
        iters:
            Number of iterative variational refinement steps.
        """

        super().__init__()
        self.slots = slots
        self.slot_dim = slot_dim
        self.iters = iters
        self.initial_slots = nn.Parameter(torch.zeros(1, slots, slot_dim))
        self.decoder = nn.Sequential(
            nn.Linear(slot_dim + 2, 2 * slot_dim),
            nn.ReLU(),
            nn.Linear(2 * slot_dim, 4),
        )
        self.refiner = IodineRefiner(slot_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Run iterative object decomposition and reconstruct the image.

        Parameters
        ----------
        image:
            Input image tensor ``(B, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Reconstructed image tensor ``(B, 3, H, W)``.
        """

        batch, _, height, width = image.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=image.device, dtype=image.dtype),
            torch.linspace(-1.0, 1.0, width, device=image.device, dtype=image.dtype),
            indexing="ij",
        )
        coords = (
            torch.stack([yy, xx], dim=-1)
            .view(1, 1, height * width, 2)
            .expand(batch, self.slots, -1, -1)
        )
        target = image.flatten(2).transpose(1, 2)
        slots = self.initial_slots.expand(batch, -1, -1)
        recon = torch.zeros_like(target)
        for _ in range(self.iters):
            slot_grid = slots[:, :, None, :].expand(-1, -1, height * width, -1)
            decoded = self.decoder(torch.cat([slot_grid, coords], dim=-1))
            rgb = torch.sigmoid(decoded[..., :3])
            masks = torch.softmax(decoded[..., 3:4], dim=1)
            recon = (rgb * masks).sum(dim=1)
            residual = target[:, None] - rgb
            summary = (masks * residual).sum(dim=2) / masks.sum(dim=2).clamp_min(1e-6)
            slots = slots + self.refiner(slots, summary)
        return recon.transpose(1, 2).view(batch, 3, height, width)


def build_psgnet() -> nn.Module:
    """Build compact PSGNet.

    Returns
    -------
    nn.Module
        Random-init PSGNet module.
    """

    return PSGNet()


def build_slot_attention_autoencoder() -> nn.Module:
    """Build compact Slot Attention autoencoder.

    Returns
    -------
    nn.Module
        Random-init Slot Attention autoencoder.
    """

    return SlotAttentionAutoEncoder()


def build_iodine() -> nn.Module:
    """Build compact IODINE.

    Returns
    -------
    nn.Module
        Random-init IODINE module.
    """

    return IODINE()


def example_image() -> torch.Tensor:
    """Create a small RGB image input.

    Returns
    -------
    torch.Tensor
        Example tensor ``(1, 3, 24, 24)``.
    """

    return torch.randn(1, 3, 24, 24)


MENAGERIE_ENTRIES = [
    (
        "PSGNet (Physical Scene Graph Network)",
        "build_psgnet",
        "example_image",
        "2020",
        "DC",
    ),
    (
        "Slot Attention object-centric autoencoder",
        "build_slot_attention_autoencoder",
        "example_image",
        "2020",
        "DC",
    ),
    (
        "IODINE (Iterative Object Decomposition Inference Network)",
        "build_iodine",
        "example_image",
        "2019",
        "DC",
    ),
]
