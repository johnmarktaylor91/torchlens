"""Additional dependency-gated models from reimpl3_4.

Paper: Basset (Kelley et al., 2016), Remora (Oxford Nanopore), LigandMPNN
(Dauparas et al., 2025), LieConv (Finzi et al., 2020), Moshi (Defossez et al.,
2024), HULC (Mees et al., 2022), Lumina-NextDiT (Zhuo et al., 2024), and
BoTNet (Srinivas et al., 2021).

These are compact random-init PyTorch reconstructions of the distinctive
architecture primitives rather than package-dependent checkpoint loads.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BassetNet(nn.Module):
    """Basset-style genomic CNN with wide motif filters and dense heads."""

    def __init__(self, targets: int = 12) -> None:
        """Initialize the compact Basset network."""

        super().__init__()
        self.conv1 = nn.Conv1d(4, 24, kernel_size=19, padding=9)
        self.conv2 = nn.Conv1d(24, 32, kernel_size=11, padding=5)
        self.conv3 = nn.Conv1d(32, 48, kernel_size=7, padding=3)
        self.fc1 = nn.Linear(48 * 16, 64)
        self.fc2 = nn.Linear(64, targets)

    def forward(self, dna_one_hot: torch.Tensor) -> torch.Tensor:
        """Predict regulatory activity from one-hot DNA sequence."""

        x = F.max_pool1d(F.relu(self.conv1(dna_one_hot)), 2)
        x = F.max_pool1d(F.relu(self.conv2(x)), 2)
        x = F.max_pool1d(F.relu(self.conv3(x)), 2)
        return self.fc2(F.dropout(F.relu(self.fc1(x.flatten(1))), p=0.1, training=self.training))


class RemoraNet(nn.Module):
    """Remora-style modified-base caller using signal, k-mer, and move table."""

    def __init__(self, bases: int = 5, width: int = 32) -> None:
        """Initialize signal/base alignment encoders."""

        super().__init__()
        self.base = nn.Embedding(bases, width)
        self.signal = nn.Conv1d(2, width, kernel_size=5, padding=2)
        self.mix = nn.GRU(width, width, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * width, 3)

    def forward(
        self, signal: torch.Tensor, bases: torch.Tensor, moves: torch.Tensor
    ) -> torch.Tensor:
        """Classify a modified base from normalized signal and base mapping."""

        sig = self.signal(torch.stack([signal, moves], dim=1)).transpose(1, 2)
        base_tokens = self.base(bases)
        aligned = sig + F.interpolate(base_tokens.transpose(1, 2), size=sig.shape[1]).transpose(
            1, 2
        )
        hidden, _ = self.mix(aligned)
        return self.head(hidden[:, hidden.shape[1] // 2])


class LieConvLayer(nn.Module):
    """LieConv layer using relative Lie algebra coordinates for point data."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        """Initialize a learned kernel over relative coordinates."""

        super().__init__()
        self.kernel = nn.Sequential(nn.Linear(4, 32), nn.SiLU(), nn.Linear(32, in_dim * out_dim))
        self.out_dim = out_dim

    def forward(self, coords: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Apply equivariant continuous convolution over pairwise group offsets."""

        rel = coords[:, :, None, :] - coords[:, None, :, :]
        dist = rel.norm(dim=-1, keepdim=True)
        algebra = torch.cat([rel, dist], dim=-1)
        weights = self.kernel(algebra).view(*algebra.shape[:3], values.shape[-1], self.out_dim)
        messages = torch.einsum("bsi,btsio->btso", values, weights)
        gate = torch.softmax(-dist.squeeze(-1), dim=-1).unsqueeze(-1)
        return (messages * gate).sum(dim=2)


class LieResNet(nn.Module):
    """LieConv bottleneck residual network with global pooling."""

    def __init__(self) -> None:
        """Initialize the compact LieConv ResNet."""

        super().__init__()
        self.inp = nn.Linear(3, 16)
        self.conv1 = LieConvLayer(16, 16)
        self.conv2 = LieConvLayer(16, 16)
        self.head = nn.Linear(16, 5)

    def forward(self, coords: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Classify point-cloud values with LieConv residual blocks."""

        x = self.inp(values)
        x = x + F.silu(self.conv1(coords, x))
        x = x + F.silu(self.conv2(coords, x))
        return self.head(x.mean(dim=1))


class LigandMPNN(nn.Module):
    """Ligand-aware ProteinMPNN-style graph encoder and autoregressive decoder."""

    def __init__(self, width: int = 32, vocab: int = 21, context_atoms: int = 3) -> None:
        """Initialize residue, ligand, and decoder message passing blocks."""

        super().__init__()
        self.vocab = vocab
        self.context_atoms = context_atoms
        self.residue = nn.Linear(6, width)
        self.ligand = nn.Linear(5, width)
        self.edge = nn.Linear(4, width)
        self.ligand_edge = nn.Linear(4, width)
        self.protein_ligand = nn.Linear(width * 3, width)
        self.encoder_gru = nn.GRUCell(width, width)
        self.token = nn.Embedding(vocab, width)
        self.decoder_gru = nn.GRUCell(width * 2, width)
        self.head = nn.Linear(width, vocab)

    def _ligand_context(
        self, residue_geom: torch.Tensor, ligand_atoms: torch.Tensor
    ) -> torch.Tensor:
        """Build per-residue nearest-ligand atomic context graph features.

        Parameters
        ----------
        residue_geom:
            Residue geometry tensor shaped ``(batch, residues, 6)``.
        ligand_atoms:
            Ligand atom tensor shaped ``(batch, atoms, 5)``.

        Returns
        -------
        torch.Tensor
            Per-residue context features shaped ``(batch, residues, width)``.
        """

        residue_xyz = residue_geom[..., :3]
        ligand_xyz = ligand_atoms[..., :3]
        distances = torch.cdist(residue_xyz, ligand_xyz)
        nearest = distances.topk(self.context_atoms, largest=False).indices
        batch_ids = torch.arange(ligand_atoms.shape[0], device=ligand_atoms.device)[:, None, None]
        selected = ligand_atoms[batch_ids, nearest]
        lig_feat = self.ligand(selected)
        rel = selected[..., :3] - residue_xyz.unsqueeze(2)
        rel_dist = rel.norm(dim=-1, keepdim=True)
        pl_edges = self.ligand_edge(torch.cat([rel, rel_dist], dim=-1))
        atom_rel = selected.unsqueeze(3)[..., :3] - selected.unsqueeze(2)[..., :3]
        atom_dist = atom_rel.norm(dim=-1, keepdim=True)
        ligand_graph = self.ligand_edge(torch.cat([atom_rel, atom_dist], dim=-1)).mean(dim=3)
        return self.protein_ligand(torch.cat([lig_feat, pl_edges, ligand_graph], dim=-1)).mean(
            dim=2
        )

    def forward(
        self, residue_geom: torch.Tensor, ligand_atoms: torch.Tensor, edge_geom: torch.Tensor
    ) -> torch.Tensor:
        """Decode amino-acid logits conditioned on backbone and ligand atoms."""

        res = self.residue(residue_geom)
        lig = self._ligand_context(residue_geom, ligand_atoms)
        edge = self.edge(edge_geom).mean(dim=2)
        hidden = res
        for _ in range(2):
            msg = edge + lig
            hidden = self.encoder_gru(msg.flatten(0, 1), hidden.flatten(0, 1)).view_as(hidden)
        prev = torch.zeros(hidden.shape[0], dtype=torch.long, device=hidden.device)
        state = hidden[:, 0]
        logits = []
        for idx in range(hidden.shape[1]):
            dec_in = torch.cat([hidden[:, idx], self.token(prev)], dim=-1)
            state = self.decoder_gru(dec_in, state)
            step_logits = self.head(state)
            logits.append(step_logits)
            prev = step_logits.argmax(dim=-1).clamp_max(self.vocab - 1)
        return torch.stack(logits, dim=1)


class SpatialMHSA(nn.Module):
    """BoTNet spatial multi-head self-attention over a feature map."""

    def __init__(self, channels: int = 32, heads: int = 4) -> None:
        """Initialize channel-preserving spatial attention."""

        super().__init__()
        self.heads = heads
        self.qkv = nn.Conv2d(channels, 3 * channels, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.rel_h = nn.Parameter(torch.randn(heads, 8, channels // heads) * 0.02)
        self.rel_w = nn.Parameter(torch.randn(heads, 8, channels // heads) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend globally over spatial positions with relative height/width bias."""

        batch, channels, height, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = q.view(batch, self.heads, channels // self.heads, height * width).transpose(-1, -2)
        k = k.view(batch, self.heads, channels // self.heads, height * width)
        v = v.view(batch, self.heads, channels // self.heads, height * width).transpose(-1, -2)
        logits = torch.matmul(q, k) / ((channels // self.heads) ** 0.5)
        rel = (self.rel_h[:, :height].mean(dim=1) + self.rel_w[:, :width].mean(dim=1)).sum(-1)
        attn = torch.softmax(logits + rel.view(1, self.heads, 1, 1), dim=-1)
        out = torch.matmul(attn, v).transpose(-1, -2).reshape(batch, channels, height, width)
        return self.proj(out)


class BoTNet(nn.Module):
    """ResNet-like classifier whose final bottleneck replaces 3x3 conv with MHSA."""

    def __init__(self) -> None:
        """Initialize compact BoTNet."""

        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.conv = nn.Sequential(nn.Conv2d(16, 32, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1))
        self.attn = SpatialMHSA(32)
        self.head = nn.Linear(32, 10)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Classify an image with a bottleneck transformer block."""

        x = self.stem(image)
        x = self.conv(x) + self.attn(self.conv(x))
        return self.head(x.mean(dim=(-2, -1)))


class MoshiFullDuplex(nn.Module):
    """Moshi-style multi-stream speech/text model with Mimi-like audio tokens."""

    def __init__(self, width: int = 40, vocab: int = 64) -> None:
        """Initialize audio codec embeddings and duplex transformer."""

        super().__init__()
        self.audio = nn.Embedding(vocab, width)
        self.text = nn.Embedding(vocab, width)
        layer = nn.TransformerEncoderLayer(width, 4, 96, batch_first=True, norm_first=True)
        self.trunk = nn.TransformerEncoder(layer, 2)
        self.audio_head = nn.Linear(width, vocab)
        self.text_head = nn.Linear(width, vocab)

    def forward(
        self, user_audio: torch.Tensor, assistant_audio: torch.Tensor, text: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Model simultaneous user audio, assistant audio, and inner-monologue text."""

        user = self.audio(user_audio)
        assistant = self.audio(assistant_audio)
        words = self.text(text)
        hidden = self.trunk(torch.cat([user, assistant, words], dim=1))
        return self.audio_head(hidden[:, : user.shape[1]]), self.text_head(
            hidden[:, -words.shape[1] :]
        )


class HULCPolicy(nn.Module):
    """Hierarchical universal language-conditioned imitation policy."""

    def __init__(self, width: int = 40) -> None:
        """Initialize multimodal encoder, discrete plan, and 7-DoF controller."""

        super().__init__()
        self.image = nn.Conv2d(3, width, 5, stride=4, padding=2)
        self.lang = nn.Embedding(64, width)
        self.plan_logits = nn.Linear(width, 8)
        self.plan_codes = nn.Parameter(torch.randn(8, width) * 0.02)
        self.ctrl = nn.GRU(width + 8, width, batch_first=True)
        self.action = nn.Linear(width, 7)

    def forward(
        self, image: torch.Tensor, language: torch.Tensor, proprio: torch.Tensor
    ) -> torch.Tensor:
        """Predict 7-DoF actions from image, language, and discrete latent plan."""

        visual = self.image(image).mean(dim=(-2, -1))
        words = self.lang(language).mean(dim=1)
        logits = self.plan_logits(visual * words)
        plan = torch.softmax(logits, dim=-1) @ self.plan_codes
        seq = torch.cat([plan[:, None, :].expand(-1, 4, -1), proprio], dim=-1)
        hidden, _ = self.ctrl(seq)
        return self.action(hidden)


class LuminaNextDiT(nn.Module):
    """Flow-based Next-DiT with sandwich norms and RoPE-like time/space phases."""

    def __init__(self, width: int = 48) -> None:
        """Initialize compact Lumina Next-DiT denoiser."""

        super().__init__()
        self.patch = nn.Linear(3 * 4 * 4, width)
        self.text = nn.Embedding(80, width)
        self.time = nn.Linear(1, width)
        layer = nn.TransformerEncoderLayer(width, 4, 96, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, 2)
        self.norm_in = nn.LayerNorm(width)
        self.norm_out = nn.LayerNorm(width)
        self.out = nn.Linear(width, 3 * 4 * 4)

    def forward(self, image: torch.Tensor, text: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict flow velocity for image patches conditioned on text."""

        patches = F.unfold(image, kernel_size=4, stride=4).transpose(1, 2)
        tokens = self.patch(patches)
        positions = torch.arange(tokens.shape[1], device=image.device).float()
        rope = torch.sin(positions[None, :, None] / 10.0 + t[:, None, None])
        cond = self.text(text).mean(dim=1, keepdim=True) + self.time(t[:, None]).unsqueeze(1)
        hidden = self.blocks(self.norm_in(tokens + rope + cond))
        out = self.out(self.norm_out(hidden)).transpose(1, 2)
        return F.fold(out, output_size=image.shape[-2:], kernel_size=4, stride=4)


def build_basset() -> nn.Module:
    """Build Basset."""

    return BassetNet()


def example_basset() -> torch.Tensor:
    """Return Basset DNA example input."""

    return F.one_hot(torch.randint(0, 4, (1, 128)), 4).float().transpose(1, 2)


def build_remora() -> nn.Module:
    """Build Remora."""

    return RemoraNet()


def example_remora() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Remora signal, bases, and moves."""

    return torch.randn(1, 80), torch.randint(0, 5, (1, 12)), torch.rand(1, 80)


def build_lieconv() -> nn.Module:
    """Build LieConv ResNet."""

    return LieResNet()


def example_lieconv() -> tuple[torch.Tensor, torch.Tensor]:
    """Return LieConv coordinates and values."""

    return torch.randn(1, 12, 3), torch.randn(1, 12, 3)


def build_ligandmpnn() -> nn.Module:
    """Build LigandMPNN."""

    return LigandMPNN()


def example_ligandmpnn() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return residue, ligand, and edge geometry tensors."""

    return torch.randn(1, 10, 6), torch.randn(1, 4, 5), torch.randn(1, 10, 3, 4)


def build_botnet() -> nn.Module:
    """Build BoTNet."""

    return BoTNet()


def example_image() -> torch.Tensor:
    """Return a small image tensor."""

    return torch.randn(1, 3, 32, 32)


def build_moshi() -> nn.Module:
    """Build Moshi full-duplex model."""

    return MoshiFullDuplex()


def example_moshi() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Moshi audio/text token streams."""

    return torch.randint(0, 64, (1, 5)), torch.randint(0, 64, (1, 5)), torch.randint(0, 64, (1, 4))


def build_hulc() -> nn.Module:
    """Build HULC."""

    return HULCPolicy()


def example_hulc() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return HULC image, language, and proprioception."""

    return torch.randn(1, 3, 32, 32), torch.randint(0, 64, (1, 5)), torch.randn(1, 4, 8)


def build_lumina() -> nn.Module:
    """Build Lumina Next-DiT."""

    return LuminaNextDiT()


def example_lumina() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Lumina image, text, and time."""

    return torch.randn(1, 3, 16, 16), torch.randint(0, 80, (1, 6)), torch.full((1,), 0.4)


MENAGERIE_ENTRIES = [
    ("Basset", "build_basset", "example_basset", "2016", "GN"),
    ("Remora", "build_remora", "example_remora", "2022", "GN"),
    ("lieconv_lie_resnet", "build_lieconv", "example_lieconv", "2020", "GE"),
    ("LigandMPNN", "build_ligandmpnn", "example_ligandmpnn", "2025", "MO"),
    ("BOTNet", "build_botnet", "example_image", "2021", "CV"),
    ("kyutai_moshi_full_duplex", "build_moshi", "example_moshi", "2024", "SP"),
    ("hulc", "build_hulc", "example_hulc", "2022", "RL"),
    ("lumina_nextdit", "build_lumina", "example_lumina", "2024", "DG"),
]
