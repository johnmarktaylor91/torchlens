"""Wave-D compact faithful reimplementations for dependency-heavy menagerie rows.

These modules keep the source families' defining computation paths while avoiding
packages that are unavailable, ABI-broken, or configured by placeholder catalog recipes.
They are random-init validation targets, not checkpoint-compatible replacements.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ProjectionHead(nn.Module):
    """Lightly-style projection head with configurable DINO or DenseCL output path."""

    def __init__(self, mode: str) -> None:
        """Initialize the projection head.

        Parameters
        ----------
        mode:
            Either ``"dino"`` for bottleneck-normalized prototypes or ``"densecl"`` for
            dense contrastive projection.
        """

        super().__init__()
        self.mode = mode
        hidden = 512
        out_dim = 1024 if mode == "dino" else 128
        bottleneck = 256 if mode == "dino" else hidden
        self.mlp = nn.Sequential(
            nn.Linear(2048, hidden),
            nn.GELU(),
            nn.Linear(hidden, bottleneck),
        )
        self.prototypes = nn.Linear(bottleneck, out_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Project features through the SSL head.

        Parameters
        ----------
        x:
            Feature tensor with trailing dimension 2048.

        Returns
        -------
        Tensor
            Projected contrastive/prototype features.
        """

        h = self.mlp(x)
        if self.mode == "dino":
            h = F.normalize(h, dim=-1)
        return self.prototypes(h)


class QuickNATCompact(nn.Module):
    """QuickNAT-like encoder-decoder segmentation network."""

    def __init__(self) -> None:
        """Initialize QuickNAT encoder, decoder, and classifier."""

        super().__init__()
        self.enc1 = self._block(1, 8)
        self.enc2 = self._block(8, 16)
        self.enc3 = self._block(16, 32)
        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2)
        self.dec3 = self._block(32, 16)
        self.dec2 = self._block(16, 8)
        self.out = nn.Conv2d(8, 2, 1)

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        """Build a QuickNAT convolution-normalization block.

        Parameters
        ----------
        in_channels:
            Input channel count.
        out_channels:
            Output channel count.

        Returns
        -------
        nn.Sequential
            Convolutional block.
        """

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Segment a 2-D medical image.

        Parameters
        ----------
        x:
            Image tensor ``(batch, 1, height, width)``.

        Returns
        -------
        Tensor
            Per-class logits.
        """

        e1 = self.enc1(x)
        p1, i1 = self.pool(e1)
        e2 = self.enc2(p1)
        p2, i2 = self.pool(e2)
        e3 = self.enc3(p2)
        _ = (i1, i2)
        d2 = F.interpolate(e3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec3(d2) + e2
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(self.dec2(d1) + e1)


class DeepMedicCompact(nn.Module):
    """DeepMedic-style dual-pathway 3-D CNN for lesion segmentation."""

    def __init__(self) -> None:
        """Initialize normal-resolution and subsampled-resolution pathways."""

        super().__init__()
        self.normal_path = nn.Sequential(
            nn.Conv3d(2, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 24, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(24, 32, 3, padding=1),
            nn.ReLU(),
        )
        self.sub_path = nn.Sequential(
            nn.AvgPool3d(3, stride=2, padding=1),
            nn.Conv3d(2, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 24, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(24, 32, 3, padding=1),
            nn.ReLU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv3d(64, 32, 1),
            nn.ReLU(),
            nn.Conv3d(32, 16, 1),
            nn.ReLU(),
            nn.Conv3d(16, 5, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Segment a 3-D medical volume patch.

        Parameters
        ----------
        x:
            Volume tensor ``(batch, 2, depth, height, width)``.

        Returns
        -------
        Tensor
            Per-class volumetric logits.
        """

        normal = self.normal_path(x)
        sub = self.sub_path(x)
        sub = F.interpolate(sub, size=normal.shape[-3:], mode="trilinear", align_corners=False)
        return self.fuse(torch.cat([normal, sub], dim=1))


class TranschexCompact(nn.Module):
    """TransCheX-style image and language transformer fusion model."""

    def __init__(self) -> None:
        """Initialize vision patches, text embeddings, fusion layers, and classifier."""

        super().__init__()
        dim = 64
        self.patch = nn.Conv2d(3, dim, kernel_size=16, stride=16)
        self.token = nn.Embedding(256, dim)
        self.vision = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, 4, dim_feedforward=128, batch_first=True),
            num_layers=1,
        )
        self.language = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, 4, dim_feedforward=128, batch_first=True),
            num_layers=1,
        )
        self.cross = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.head = nn.Linear(dim, 2)

    def forward(self, inputs: list[Tensor] | tuple[Tensor, ...] | Tensor) -> Tensor:
        """Classify fused image-text features.

        Parameters
        ----------
        inputs:
            Either an image tensor or ``(image, input_ids, token_type_ids)``.

        Returns
        -------
        Tensor
            Class logits.
        """

        if isinstance(inputs, (list, tuple)):
            image = inputs[0]
            ids = inputs[1].long()
        else:
            image = inputs
            ids = torch.zeros(image.shape[0], 16, dtype=torch.long, device=image.device)
        vision = self.patch(image).flatten(2).transpose(1, 2)
        vision = self.vision(vision)
        text = self.language(self.token(ids.clamp_min(0).clamp_max(255)))
        fused, _ = self.cross(text, vision, vision, need_weights=False)
        return self.head((fused + text).mean(dim=1))


class UCTransNetCompact(nn.Module):
    """UCTransNet-style U-Net with channel-transformer skip refinement."""

    def __init__(self) -> None:
        """Initialize encoder, channel transformer, decoder, and segmentation head."""

        super().__init__()
        self.enc1 = QuickNATCompact._block(1, 16)
        self.enc2 = QuickNATCompact._block(16, 32)
        self.enc3 = QuickNATCompact._block(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.channel_attn = nn.MultiheadAttention(64, 4, batch_first=True)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = QuickNATCompact._block(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = QuickNATCompact._block(32, 16)
        self.out = nn.Conv2d(16, 9, 1)

    def _refine(self, feat: Tensor) -> Tensor:
        """Refine feature channels with token attention.

        Parameters
        ----------
        feat:
            Feature map.

        Returns
        -------
        Tensor
            Refined feature map.
        """

        batch, channels, height, width = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        tokens, _ = self.channel_attn(tokens, tokens, tokens, need_weights=False)
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)

    def forward(self, x: Tensor) -> Tensor:
        """Segment an image with channel-refined skip connections.

        Parameters
        ----------
        x:
            Image tensor ``(batch, 1, height, width)``.

        Returns
        -------
        Tensor
            Segmentation logits.
        """

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self._refine(self.enc3(self.pool(e2)))
        d2 = self.up2(e3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        return self.out(self.dec1(torch.cat([d1, e1], dim=1)))


class GraphEncoderCompact(nn.Module):
    """Graph neural model with positional encoding, link attention, or pooling modes."""

    def __init__(self, mode: str = "message") -> None:
        """Initialize graph layers.

        Parameters
        ----------
        mode:
            Graph family mode: ``"gpse"``, ``"lpformer"``, ``"panpool"``, ``"exphormer"``,
            or ``"recbole"``.
        """

        super().__init__()
        self.mode = mode
        dim = 32
        self.node = nn.Linear(16, dim)
        self.pos = nn.Linear(3, dim)
        self.msg = nn.Linear(dim, dim)
        self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.score = nn.Linear(dim, 1)
        self.edge_head = nn.Bilinear(dim, dim, 1)
        self.out = nn.Linear(dim, 8)
        self.user_emb = nn.Embedding(8, dim)
        self.item_emb = nn.Embedding(16, dim)

    def _default_graph(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Create a small graph from catalog input.

        Parameters
        ----------
        x:
            Catalog input tensor.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Node features, positions, and adjacency matrix.
        """

        device = x.device
        features = torch.randn(8, 16, device=device) + x.reshape(-1)[0] * 0.0
        pos = torch.linspace(-1.0, 1.0, 8, device=device).unsqueeze(1).repeat(1, 3)
        adj = torch.eye(8, device=device)
        adj = adj + torch.roll(torch.eye(8, device=device), shifts=1, dims=0)
        return features, pos, adj.clamp_max(1.0)

    def forward(self, x: Tensor) -> Tensor:
        """Run the selected compact graph computation.

        Parameters
        ----------
        x:
            Ignored sentinel or graph feature tensor.

        Returns
        -------
        Tensor
            Graph-level output.
        """

        feat, pos, adj = self._default_graph(x)
        h = self.node(feat)
        if self.mode == "gpse":
            h = h + self.pos(pos)
        msg = adj @ h / adj.sum(-1, keepdim=True).clamp_min(1.0)
        h = F.gelu(h + self.msg(msg))
        if self.mode in {"lpformer", "exphormer"}:
            h = self.attn(h.unsqueeze(0), h.unsqueeze(0), h.unsqueeze(0), need_weights=False)[0][0]
        if self.mode == "panpool":
            keep = torch.topk(self.score(h).squeeze(-1), k=4).indices
            h = h[keep]
        if self.mode == "recbole":
            users = self.user_emb(torch.arange(4, device=x.device))
            items = self.item_emb(torch.arange(4, device=x.device) + 1)
            h = users + items + h[:4]
            return self.edge_head(h, items).mean(0)
        if self.mode == "lpformer":
            return self.edge_head(h[0:1], h[1:2]).squeeze(0)
        return self.out(h.mean(dim=0))


class ELICCompact(nn.Module):
    """ELIC-style learned image compression analysis/synthesis transform."""

    def __init__(self) -> None:
        """Initialize analysis, hyperprior, context, and synthesis transforms."""

        super().__init__()
        self.analysis = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 48, 5, stride=2, padding=2),
            nn.GELU(),
        )
        self.hyper = nn.Sequential(nn.Conv2d(48, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 48, 1))
        self.context = nn.Conv2d(48, 48, 5, padding=2, groups=3)
        self.synthesis = nn.Sequential(
            nn.ConvTranspose2d(48, 32, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Compress and reconstruct an image tensor.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        Tensor
            Reconstructed image.
        """

        y = self.analysis(x)
        params = self.hyper(y)
        y_hat = y + 0.05 * torch.tanh(params + self.context(y))
        return self.synthesis(y_hat)


class HFTextCompact(nn.Module):
    """Compact text model covering encoder-decoder, decoder, MoE, and SSM families."""

    def __init__(self, family: str = "encoder") -> None:
        """Initialize the text architecture.

        Parameters
        ----------
        family:
            Architecture family selector.
        """

        super().__init__()
        self.family = family
        dim = 64
        self.embed = nn.Embedding(512, dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, 4, dim_feedforward=128, batch_first=True),
            num_layers=2,
        )
        self.decoder_layer = nn.TransformerDecoderLayer(
            dim, 4, dim_feedforward=128, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=1)
        self.router = nn.Linear(dim, 4)
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, 128), nn.SiLU(), nn.Linear(128, dim)) for _ in range(4)]
        )
        self.conv = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.out = nn.Linear(dim, 128)

    def _moe(self, h: Tensor) -> Tensor:
        """Apply dense top-k mixture-of-experts routing.

        Parameters
        ----------
        h:
            Token states.

        Returns
        -------
        Tensor
            Mixed token states.
        """

        probs = self.router(h).softmax(-1)
        topv, topi = probs.topk(2, dim=-1)
        gate = torch.zeros_like(probs).scatter(-1, topi, topv)
        gate = gate / gate.sum(-1, keepdim=True).clamp_min(1e-6)
        stacked = torch.stack([expert(h) for expert in self.experts], dim=-2)
        return (stacked * gate.unsqueeze(-1)).sum(-2)

    def _ssm(self, h: Tensor) -> Tensor:
        """Apply convolutional state-space-style recurrence.

        Parameters
        ----------
        h:
            Token states.

        Returns
        -------
        Tensor
            Recurrent token states.
        """

        conv = self.conv(h.transpose(1, 2)).transpose(1, 2)
        state = torch.zeros_like(conv[:, 0])
        outs = []
        for token in conv.unbind(1):
            state = 0.75 * state + torch.tanh(token)
            outs.append(state)
        return torch.stack(outs, dim=1)

    def forward(self, ids: Tensor) -> Tensor:
        """Encode token ids.

        Parameters
        ----------
        ids:
            Token id tensor.

        Returns
        -------
        Tensor
            Token logits or pooled features.
        """

        tokens = ids.long().reshape(ids.shape[0], -1).clamp(0, 511)
        h = self.embed(tokens)
        if self.family in {"bart", "t5", "marian", "fsmt"}:
            memory = self.encoder(h)
            h = self.decoder(h, memory)
        elif self.family in {"mamba", "hybrid"}:
            h = self._ssm(h)
            if self.family == "hybrid":
                h = h + self.encoder(self.embed(tokens))
        else:
            h = self.encoder(h)
        if self.family in {"moe", "hybrid"}:
            h = h + self._moe(h)
        return self.out(h)


class DETRSegmentationCompact(nn.Module):
    """DETR-style convolutional backbone with transformer segmentation decoder."""

    def __init__(self) -> None:
        """Initialize DETR segmentation components."""

        super().__init__()
        dim = 64
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, dim, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.query = nn.Parameter(torch.randn(8, dim))
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(dim, 4, dim_feedforward=128, batch_first=True),
            num_layers=2,
        )
        self.mask = nn.Conv2d(dim, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Predict segmentation masks from an image.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        Tensor
            Query mask logits.
        """

        feat = self.backbone(x)
        memory = feat.flatten(2).transpose(1, 2)
        query = self.query.unsqueeze(0).expand(x.shape[0], -1, -1)
        decoded = self.decoder(query, memory)
        masks = self.mask(feat)
        return torch.einsum("bqd,bdhw->bqhw", decoded, masks)


class MaskRCNNCompact(nn.Module):
    """Mask R-CNN-style backbone, FPN fusion, box head, and mask head."""

    def __init__(self) -> None:
        """Initialize compact Mask R-CNN components."""

        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.fpn = nn.Conv2d(32, 32, 1)
        self.box = nn.Linear(32, 4)
        self.cls = nn.Linear(32, 3)
        self.mask = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 3, 1))

    def forward(self, x: Tensor | list[Tensor] | tuple[Tensor, ...]) -> dict[str, Tensor]:
        """Run detection heads.

        Parameters
        ----------
        x:
            Image tensor or torchvision-style list of image tensors.

        Returns
        -------
        dict[str, Tensor]
            Box, class, and mask outputs.
        """

        image = x[0].unsqueeze(0) if isinstance(x, (list, tuple)) else x
        feat = F.gelu(self.c2(F.gelu(self.c1(image))))
        feat = self.fpn(feat)
        pooled = feat.mean(dim=(2, 3))
        return {"boxes": self.box(pooled), "scores": self.cls(pooled), "masks": self.mask(feat)}


class SpeechSequenceCompact(nn.Module):
    """SpeechBrain-style encoder, TTS, vocoder, codec, or separator."""

    def __init__(self, mode: str) -> None:
        """Initialize speech modules.

        Parameters
        ----------
        mode:
            Speech architecture family selector.
        """

        super().__init__()
        self.mode = mode
        dim = 64
        self.in_proj = nn.Linear(257, dim)
        self.token = nn.Embedding(256, dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, 4, dim_feedforward=128, batch_first=True),
            num_layers=2,
        )
        self.conv = nn.Conv1d(dim, dim, 5, padding=2, groups=4)
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.vocoder = nn.Sequential(
            nn.ConvTranspose1d(80, 64, 8, stride=4, padding=2),
            nn.GELU(),
            nn.ConvTranspose1d(64, 1, 8, stride=4, padding=2),
        )
        self.audio_in = nn.Conv1d(1, dim, 7, padding=3)
        self.audio_out = nn.Conv1d(dim, 2, 7, padding=3)
        self.head = nn.Linear(dim, 80 if mode in {"fastspeech", "tacotron"} else 16)

    def forward(self, x: Tensor) -> Tensor:
        """Run a compact speech architecture.

        Parameters
        ----------
        x:
            Audio, spectrogram, or token tensor.

        Returns
        -------
        Tensor
            Speech model output.
        """

        if self.mode == "hifigan":
            return self.vocoder(x)
        if self.mode in {"codec", "separator"}:
            audio = x.unsqueeze(1) if x.ndim == 2 else x
            h = F.gelu(self.audio_in(audio))
            return self.audio_out(self.conv(h).tanh())
        if not x.dtype.is_floating_point:
            h = self.token(x.long().reshape(x.shape[0], -1).clamp(0, 255))
        else:
            y = x.reshape(x.shape[0], x.shape[1], -1)
            if y.shape[-1] < 257:
                y = F.pad(y, (0, 257 - y.shape[-1]))
            h = self.in_proj(y[..., :257])
        h = self.encoder(h)
        if self.mode in {"branchformer", "conformer", "contextnet"}:
            h = h + self.conv(h.transpose(1, 2)).transpose(1, 2)
        if self.mode in {"tacotron", "fastspeech"}:
            h, _ = self.gru(h)
        return self.head(h)


class MelFeatureAdapter(nn.Module):
    """Adapt SpeechBrain feature tensors to wave-C log-mel encoder builders."""

    def __init__(self, core: nn.Module, input_bins: int) -> None:
        """Initialize a feature-bin projection in front of a speech encoder.

        Parameters
        ----------
        core:
            Wave-C speech encoder/head module that consumes ``(batch, 80, time)`` mel features.
        input_bins:
            Number of frequency or model-width bins in the catalog input tensor.
        """

        super().__init__()
        self.core = core
        self.to_mel = nn.Linear(input_bins, 80)

    def forward(self, x: Tensor) -> Tensor:
        """Project time-major features to mel-major features and run the core model.

        Parameters
        ----------
        x:
            Speech feature tensor shaped ``(batch, time, bins)`` or ``(batch, 80, time)``.

        Returns
        -------
        Tensor
            Core model output.
        """

        if x.ndim != 3:
            raise ValueError("MelFeatureAdapter expects a 3D speech feature tensor")
        if x.shape[1] == 80 and x.shape[-1] != self.to_mel.in_features:
            mel = x
        else:
            mel = self.to_mel(x[..., : self.to_mel.in_features]).transpose(1, 2)
        return self.core(mel)


class FastSpeech2Compact(nn.Module):
    """FastSpeech2-style non-autoregressive TTS with variance adaptation."""

    def __init__(self, vocab: int = 128, dim: int = 64) -> None:
        """Initialize text encoder, duration/pitch/energy predictors, and mel decoder.

        Parameters
        ----------
        vocab:
            Token vocabulary size.
        dim:
            Hidden width.
        """

        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(torch.randn(1, 128, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 4, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.duration = nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(dim, 1, 3, padding=1),
        )
        self.pitch = nn.Linear(dim, dim)
        self.energy = nn.Linear(dim, dim)
        self.decoder = nn.TransformerEncoder(layer, 1)
        self.mel = nn.Linear(dim, 80)

    def forward(self, tokens: Tensor) -> Tensor:
        """Generate mel frames with a compact variance-adaptor path.

        Parameters
        ----------
        tokens:
            Text token ids shaped ``(batch, length)``.

        Returns
        -------
        Tensor
            Mel sequence shaped ``(batch, length, 80)``.
        """

        x = self.embed(tokens.long().clamp(0, self.embed.num_embeddings - 1))
        x = x + self.pos[:, : x.shape[1]]
        encoded = self.encoder(x)
        log_duration = self.duration(encoded.transpose(1, 2)).transpose(1, 2)
        duration_gate = torch.sigmoid(log_duration)
        variance = self.pitch(encoded) + self.energy(encoded)
        adapted = encoded + duration_gate * variance
        return self.mel(self.decoder(adapted))


class MetricGANGeneratorCompact(nn.Module):
    """MetricGAN speech-enhancement generator with BLSTM mask prediction."""

    def __init__(self, freq_bins: int = 257, hidden: int = 64) -> None:
        """Initialize BLSTM and learnable-sigmoid mask head.

        Parameters
        ----------
        freq_bins:
            Input spectral magnitude bins.
        hidden:
            BLSTM hidden width.
        """

        super().__init__()
        self.norm = nn.LayerNorm(freq_bins)
        self.blstm = nn.LSTM(freq_bins, hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.mask = nn.Sequential(nn.Linear(hidden * 2, freq_bins), nn.Sigmoid())
        self.sharpness = nn.Parameter(torch.tensor(1.0))

    def forward(self, spectrum: Tensor) -> Tensor:
        """Enhance a magnitude spectrogram by predicting a soft mask.

        Parameters
        ----------
        spectrum:
            Magnitude spectrum shaped ``(batch, time, freq)``.

        Returns
        -------
        Tensor
            Masked enhanced spectrum.
        """

        features = self.norm(spectrum)
        hidden, _ = self.blstm(features)
        mask = torch.sigmoid(self.sharpness.clamp_min(0.1) * (self.mask(hidden) * 2.0 - 1.0))
        return spectrum * mask


class CNNTransformerSECompact(nn.Module):
    """CNN-Transformer speech-enhancement network with spectral mask output."""

    def __init__(self, freq_bins: int = 257, dim: int = 64) -> None:
        """Initialize convolutional frontend, Transformer core, and mask head.

        Parameters
        ----------
        freq_bins:
            Input spectral feature count.
        dim:
            Transformer width.
        """

        super().__init__()
        self.front = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 16, 3, stride=(1, 2), padding=1),
            nn.GELU(),
        )
        reduced_bins = (freq_bins + 1) // 2
        self.project = nn.Linear(16 * reduced_bins, dim)
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 4, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, 2)
        self.mask = nn.Linear(dim, freq_bins)

    def forward(self, spectrum: Tensor) -> Tensor:
        """Estimate and apply a time-frequency enhancement mask.

        Parameters
        ----------
        spectrum:
            Spectral features shaped ``(batch, time, freq)``.

        Returns
        -------
        Tensor
            Enhanced spectral features.
        """

        feat = self.front(spectrum.unsqueeze(1)).transpose(1, 2).flatten(2)
        tokens = self.project(feat)
        hidden = self.transformer(tokens)
        return spectrum * torch.sigmoid(self.mask(hidden))


class ODEBlockCompact(nn.Module):
    """Neural ODE block using an explicit Euler integration path."""

    def __init__(self) -> None:
        """Initialize ODE function layers."""

        super().__init__()
        self.func = nn.Sequential(
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Integrate the ODE function with fixed Euler steps.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        Tensor
            Integrated feature map.
        """

        h = x
        for _ in range(4):
            h = h + 0.25 * self.func(h)
        return h


def build_projection_head(mode: str) -> nn.Module:
    """Build a Lightly projection head.

    Parameters
    ----------
    mode:
        Projection head mode.

    Returns
    -------
    nn.Module
        Projection head module.
    """

    return ProjectionHead(mode).eval()


def build_quicknat() -> nn.Module:
    """Build compact QuickNAT.

    Returns
    -------
    nn.Module
        QuickNAT model.
    """

    return QuickNATCompact().eval()


def build_deepmedic() -> nn.Module:
    """Build compact DeepMedic.

    Returns
    -------
    nn.Module
        DeepMedic model.
    """

    return DeepMedicCompact().eval()


def build_transchex() -> nn.Module:
    """Build compact TransCheX.

    Returns
    -------
    nn.Module
        TransCheX model.
    """

    return TranschexCompact().eval()


def build_uctransnet() -> nn.Module:
    """Build compact UCTransNet.

    Returns
    -------
    nn.Module
        UCTransNet model.
    """

    return UCTransNetCompact().eval()


def build_graph(mode: str) -> nn.Module:
    """Build a compact graph model.

    Parameters
    ----------
    mode:
        Graph mode.

    Returns
    -------
    nn.Module
        Graph model.
    """

    return GraphEncoderCompact(mode).eval()


def build_elic() -> nn.Module:
    """Build compact ELIC.

    Returns
    -------
    nn.Module
        Learned compression model.
    """

    return ELICCompact().eval()


def build_hf_text(family: str) -> nn.Module:
    """Build compact HuggingFace-style text model.

    Parameters
    ----------
    family:
        Text model family.

    Returns
    -------
    nn.Module
        Text model.
    """

    return HFTextCompact(family).eval()


def build_detr_segmentation() -> nn.Module:
    """Build compact DETR segmentation model.

    Returns
    -------
    nn.Module
        DETR segmentation model.
    """

    return DETRSegmentationCompact().eval()


def build_maskrcnn() -> nn.Module:
    """Build compact Mask R-CNN.

    Returns
    -------
    nn.Module
        Mask R-CNN model.
    """

    return MaskRCNNCompact().eval()


def build_speech(mode: str) -> nn.Module:
    """Build compact speech model.

    Parameters
    ----------
    mode:
        Speech model family.

    Returns
    -------
    nn.Module
        Speech model.
    """

    if mode == "branchformer":
        from menagerie.classics.wave_c_audio_misc import build_ebranchformer

        return MelFeatureAdapter(build_ebranchformer(), input_bins=256).eval()
    if mode == "conformer":
        from menagerie.classics.wave_c_audio_misc import build_conformer

        return MelFeatureAdapter(build_conformer(), input_bins=256).eval()
    if mode == "contextnet":
        from menagerie.classics.wave_c_audio_misc import build_contextnet

        return MelFeatureAdapter(build_contextnet(), input_bins=80).eval()
    if mode == "fastspeech":
        return FastSpeech2Compact().eval()
    if mode == "tacotron":
        from menagerie.classics.wave_c_audio_misc import build_tacotron2

        return build_tacotron2()
    if mode == "mstacotron":
        from menagerie.classics.wave_c_audio_misc import build_tacotron2_gst

        return build_tacotron2_gst()
    if mode == "hifigan":
        from menagerie.classics.wave_c_audio_misc import build_hifigan

        return build_hifigan()
    if mode == "codec":
        from menagerie.classics.descript_dac_codec import build

        return build()
    if mode == "separator":
        from menagerie.classics.wave_c_audio_misc import build_dprnn

        return build_dprnn()
    if mode == "metricgan":
        return MetricGANGeneratorCompact().eval()
    if mode == "cnntransformer_se":
        return CNNTransformerSECompact().eval()
    return SpeechSequenceCompact(mode).eval()


def build_odeblock() -> nn.Module:
    """Build compact neural ODE block.

    Returns
    -------
    nn.Module
        ODE block.
    """

    return ODEBlockCompact().eval()


def image_input(channels: int = 3, size: int = 64) -> Tensor:
    """Build an image input.

    Parameters
    ----------
    channels:
        Channel count.
    size:
        Spatial size.

    Returns
    -------
    Tensor
        Image tensor.
    """

    return torch.randn(1, channels, size, size)


def token_input(length: int = 16) -> Tensor:
    """Build token input.

    Parameters
    ----------
    length:
        Token sequence length.

    Returns
    -------
    Tensor
        Token ids.
    """

    return torch.randint(0, 128, (1, length))


def feature_input(width: int = 2048) -> Tensor:
    """Build feature input.

    Parameters
    ----------
    width:
        Feature width.

    Returns
    -------
    Tensor
        Feature tensor.
    """

    return torch.randn(2, width)


def spectrogram_input(width: int = 257) -> Tensor:
    """Build spectrogram input.

    Parameters
    ----------
    width:
        Frequency-bin count.

    Returns
    -------
    Tensor
        Spectrogram tensor.
    """

    return torch.randn(1, 64, width)


def audio_input(length: int = 512) -> Tensor:
    """Build audio input.

    Parameters
    ----------
    length:
        Sample count.

    Returns
    -------
    Tensor
        Audio tensor.
    """

    return torch.randn(1, 1, length)


def graph_input() -> Tensor:
    """Build graph sentinel input.

    Returns
    -------
    Tensor
        Sentinel tensor.
    """

    return torch.zeros(1)
