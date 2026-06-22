"""Compact faithful MMOCR-family text recognition and KIE building blocks."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Small residual block used by compact ResNet31-style backbones."""

    def __init__(self, channels: int) -> None:
        """Initialize the residual block.

        Parameters
        ----------
        channels:
            Number of feature channels.
        """

        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a two-convolution residual update.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            Residual feature map.
        """

        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class ResNet31Backbone(nn.Module):
    """Compact text-line ResNet31 analogue with anisotropic downsampling."""

    def __init__(self, in_channels: int = 1, channels: int = 48, extra: bool = False) -> None:
        """Initialize the backbone.

        Parameters
        ----------
        in_channels:
            Number of image channels.
        channels:
            Base feature width.
        extra:
            Whether to add the deeper ResNetExtra-style refinement stage.
        """

        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.stage1 = nn.Sequential(BasicBlock(channels), nn.MaxPool2d(2, 2))
        self.proj2 = nn.Conv2d(channels, channels * 2, 1)
        self.stage2 = nn.Sequential(BasicBlock(channels * 2), nn.MaxPool2d((2, 2), (2, 2)))
        self.proj3 = nn.Conv2d(channels * 2, channels * 2, 1)
        self.stage3 = nn.Sequential(
            BasicBlock(channels * 2),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            BasicBlock(channels * 2),
        )
        self.extra = (
            nn.Sequential(BasicBlock(channels * 2), BasicBlock(channels * 2))
            if extra
            else nn.Identity()
        )
        self.out_channels = channels * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an input word image into a low-height feature map.

        Parameters
        ----------
        x:
            Image tensor of shape ``(batch, channels, height, width)``.

        Returns
        -------
        torch.Tensor
            Feature map preserving a horizontal text sequence.
        """

        x = self.stem(x)
        x = self.stage1(x)
        x = self.proj2(x)
        x = self.stage2(x)
        x = self.proj3(x)
        x = self.stage3(x)
        return self.extra(x)


class ThinPlateSplineLite(nn.Module):
    """Thin-plate-spline STN rectifier with learned fiducial control points."""

    def __init__(self, in_channels: int = 1, points_per_side: int = 5) -> None:
        """Initialize a small localization network.

        Parameters
        ----------
        in_channels:
            Number of image channels.
        points_per_side:
            Number of top and bottom TPS fiducial points.
        """

        super().__init__()
        self.num_points = points_per_side * 2
        self.loc = nn.Sequential(
            nn.Conv2d(in_channels, 8, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(16, self.num_points * 2)
        nn.init.zeros_(self.fc.weight)
        xs = torch.linspace(-0.9, 0.9, points_per_side)
        top = torch.stack([xs, torch.full_like(xs, -0.8)], dim=-1)
        bottom = torch.stack([xs, torch.full_like(xs, 0.8)], dim=-1)
        base = torch.cat([top, bottom], dim=0)
        self.register_buffer("source_points", base)
        self.fc.bias.data.copy_(base.flatten())

    @staticmethod
    def _phi(radius: torch.Tensor) -> torch.Tensor:
        """Evaluate the TPS radial basis ``r^2 log(r)`` safely.

        Parameters
        ----------
        radius:
            Pairwise Euclidean distances.

        Returns
        -------
        torch.Tensor
            TPS radial basis values.
        """

        safe = radius.clamp_min(1e-6)
        return safe.square() * safe.log()

    def _solve_tps(self, target_points: torch.Tensor) -> torch.Tensor:
        """Solve TPS parameters mapping regular points to target points.

        Parameters
        ----------
        target_points:
            Learned target fiducials shaped ``(batch, points, 2)``.

        Returns
        -------
        torch.Tensor
            TPS weights and affine terms shaped ``(batch, points + 3, 2)``.
        """

        points = self.source_points
        pair = torch.cdist(points, points)
        kernel = self._phi(pair)
        ones = torch.ones(points.shape[0], 1, device=points.device, dtype=points.dtype)
        p = torch.cat([ones, points], dim=1)
        zeros = torch.zeros(3, 3, device=points.device, dtype=points.dtype)
        system = torch.cat([torch.cat([kernel, p], dim=1), torch.cat([p.T, zeros], dim=1)], dim=0)
        rhs_tail = torch.zeros(
            target_points.shape[0], 3, 2, device=target_points.device, dtype=target_points.dtype
        )
        rhs = torch.cat([target_points, rhs_tail], dim=1)
        return torch.linalg.solve(system.unsqueeze(0).expand(target_points.shape[0], -1, -1), rhs)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Rectify a word image with a TPS control-point sampling grid.

        Parameters
        ----------
        image:
            Input image.

        Returns
        -------
        torch.Tensor
            Rectified image.
        """

        batch, _, height, width = image.shape
        target_points = self.fc(self.loc(image).flatten(1)).view(batch, self.num_points, 2)
        coeffs = self._solve_tps(target_points)
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=image.device, dtype=image.dtype),
            torch.linspace(-1.0, 1.0, width, device=image.device, dtype=image.dtype),
            indexing="ij",
        )
        base_grid = torch.stack([xx, yy], dim=-1).view(-1, 2)
        distances = torch.cdist(base_grid, self.source_points)
        design = torch.cat(
            [
                self._phi(distances),
                torch.ones(base_grid.shape[0], 1, device=image.device, dtype=image.dtype),
                base_grid,
            ],
            dim=1,
        )
        grid = torch.matmul(design.unsqueeze(0), coeffs).view(batch, height, width, 2)
        return F.grid_sample(image, grid, align_corners=False)


class SequenceAttentionDecoder(nn.Module):
    """Autoregressive decoder with additive attention over visual features."""

    def __init__(self, dim: int, vocab: int = 38, steps: int = 8) -> None:
        """Initialize the decoder.

        Parameters
        ----------
        dim:
            Hidden dimension.
        vocab:
            Output vocabulary size.
        steps:
            Number of compact decoding steps.
        """

        super().__init__()
        self.steps = steps
        self.token = nn.Embedding(vocab, dim)
        self.cell = nn.LSTMCell(dim * 2, dim)
        self.attn = nn.Linear(dim * 2, 1)
        self.out = nn.Linear(dim, vocab)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        """Decode visual memory into character logits.

        Parameters
        ----------
        memory:
            Encoded visual sequence ``(batch, length, dim)``.

        Returns
        -------
        torch.Tensor
            Character logits ``(batch, steps, vocab)``.
        """

        batch, _, dim = memory.shape
        hx = memory.mean(dim=1)
        cx = torch.zeros_like(hx)
        token = torch.zeros(batch, dtype=torch.long, device=memory.device)
        outs = []
        for _ in range(self.steps):
            emb = self.token(token)
            query = hx.unsqueeze(1).expand_as(memory)
            alpha = torch.softmax(self.attn(torch.cat([memory, query], dim=-1)).squeeze(-1), dim=-1)
            ctx = torch.bmm(alpha.unsqueeze(1), memory).squeeze(1)
            hx, cx = self.cell(torch.cat([emb, ctx], dim=-1), (hx, cx))
            logits = self.out(hx)
            outs.append(logits)
            token = logits.argmax(dim=-1)
        return torch.stack(outs, dim=1)


class CRNNSTNResNet31(nn.Module):
    """TPS/STN, ResNet31, BiLSTM, and CTC head text recognizer."""

    def __init__(self, vocab: int = 38, dim: int = 48) -> None:
        """Initialize CRNN-STN-ResNet31.

        Parameters
        ----------
        vocab:
            CTC vocabulary size.
        dim:
            Base feature width.
        """

        super().__init__()
        self.rectifier = ThinPlateSplineLite()
        self.backbone = ResNet31Backbone(channels=dim)
        self.rnn = nn.LSTM(dim * 2, dim, batch_first=True, bidirectional=True)
        self.ctc = nn.Linear(dim * 2, vocab)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Recognize a rectified text line with CTC logits.

        Parameters
        ----------
        image:
            Grayscale word image.

        Returns
        -------
        torch.Tensor
            CTC logits.
        """

        feat = self.backbone(self.rectifier(image)).mean(dim=2).transpose(1, 2)
        seq, _ = self.rnn(feat)
        return self.ctc(seq)


class MultiAspectGCAttention(nn.Module):
    """MASTER multi-aspect global-context/non-local attention block."""

    def __init__(self, channels: int) -> None:
        """Initialize channel, height, and width context gates.

        Parameters
        ----------
        channels:
            Feature channel count.
        """

        super().__init__()
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1), nn.Sigmoid()
        )
        self.height_gate = nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0))
        self.width_gate = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1))
        self.mix = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-aspect global-context refinement.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            Refined feature map.
        """

        gate = self.channel_gate(x)
        h_ctx = torch.sigmoid(self.height_gate(x.mean(dim=3, keepdim=True))).expand_as(x)
        w_ctx = torch.sigmoid(self.width_gate(x.mean(dim=2, keepdim=True))).expand_as(x)
        return x + self.mix(x * gate * h_ctx * w_ctx)


class MASTERRecognizer(nn.Module):
    """MASTER self-attention recognizer with a ResNet and target decoder."""

    def __init__(self, vocab: int = 38, dim: int = 48, extra: bool = False) -> None:
        """Initialize MASTER.

        Parameters
        ----------
        vocab:
            Output vocabulary size.
        dim:
            Base feature width.
        extra:
            Whether to use a deeper ResNetExtra-style backbone.
        """

        super().__init__()
        self.backbone = ResNet31Backbone(channels=dim, extra=extra)
        out_dim = self.backbone.out_channels
        self.gc = MultiAspectGCAttention(out_dim)
        enc_layer = nn.TransformerEncoderLayer(out_dim, 4, out_dim * 2, batch_first=True)
        dec_layer = nn.TransformerDecoderLayer(out_dim, 4, out_dim * 2, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, 1)
        self.decoder = nn.TransformerDecoder(dec_layer, 1)
        self.query = nn.Parameter(torch.randn(1, 8, out_dim) * 0.02)
        self.head = nn.Linear(out_dim, vocab)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Recognize text with image-image and target-image attention.

        Parameters
        ----------
        image:
            Grayscale word image.

        Returns
        -------
        torch.Tensor
            Character logits.
        """

        feat = self.gc(self.backbone(image))
        memory = feat.flatten(2).transpose(1, 2)
        memory = self.encoder(memory)
        query = self.query.expand(image.shape[0], -1, -1)
        return self.head(self.decoder(query, memory))


class NRTRModalityTransform(nn.Module):
    """NRTR modality transform that converts 2D image features to 1D tokens."""

    def __init__(self, dim: int = 64) -> None:
        """Initialize the transform block.

        Parameters
        ----------
        dim:
            Token dimension.
        """

        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, dim // 2, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim // 2, dim, 3, stride=(2, 1), padding=1),
            nn.ReLU(),
        )
        self.row_gate = nn.Linear(dim, dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Transform image features into a sequence.

        Parameters
        ----------
        image:
            Word image.

        Returns
        -------
        torch.Tensor
            Sequence tokens.
        """

        feat = self.conv(image)
        seq = feat.mean(dim=2).transpose(1, 2)
        return seq + torch.tanh(self.row_gate(seq))


class NRTRRecognizer(nn.Module):
    """NRTR Transformer text recognizer."""

    def __init__(
        self, vocab: int = 38, dim: int = 64, with_modality_transform: bool = True
    ) -> None:
        """Initialize NRTR.

        Parameters
        ----------
        vocab:
            Output vocabulary size.
        dim:
            Token dimension.
        with_modality_transform:
            Whether to include the convolutional modality-transform block.
        """

        super().__init__()
        self.modality = (
            NRTRModalityTransform(dim) if with_modality_transform else nn.Conv2d(1, dim, 4, 4)
        )
        enc_layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True)
        dec_layer = nn.TransformerDecoderLayer(dim, 4, dim * 2, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, 1)
        self.decoder = nn.TransformerDecoder(dec_layer, 1)
        self.query = nn.Parameter(torch.randn(1, 8, dim) * 0.02)
        self.head = nn.Linear(dim, vocab)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Recognize text using a Transformer encoder-decoder.

        Parameters
        ----------
        image:
            Word image.

        Returns
        -------
        torch.Tensor
            Character logits.
        """

        tokens = self.modality(image)
        if tokens.ndim == 4:
            tokens = tokens.mean(dim=2).transpose(1, 2)
        memory = self.encoder(tokens)
        query = self.query.expand(image.shape[0], -1, -1)
        return self.head(self.decoder(query, memory))


class SARRecognizer(nn.Module):
    """Show-Attend-and-Read recognizer with holistic LSTM and 2D attention."""

    def __init__(self, vocab: int = 38, dim: int = 48) -> None:
        """Initialize SAR.

        Parameters
        ----------
        vocab:
            Output vocabulary size.
        dim:
            Base feature width.
        """

        super().__init__()
        self.backbone = ResNet31Backbone(channels=dim)
        self.holistic = nn.LSTM(dim * 2, dim, batch_first=True, bidirectional=True)
        self.decoder = SequenceAttentionDecoder(dim * 2, vocab=vocab)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Decode a word image with 2D attention over CNN features.

        Parameters
        ----------
        image:
            Grayscale word image.

        Returns
        -------
        torch.Tensor
            Character logits.
        """

        feat = self.backbone(image)
        seq = feat.flatten(2).transpose(1, 2)
        holistic, _ = self.holistic(seq)
        return self.decoder(holistic)


class RobustScannerRecognizer(nn.Module):
    """RobustScanner with hybrid and position-enhancement branches."""

    def __init__(self, vocab: int = 38, dim: int = 48, attention_backbone: bool = False) -> None:
        """Initialize RobustScanner.

        Parameters
        ----------
        vocab:
            Output vocabulary size.
        dim:
            Base feature width.
        attention_backbone:
            Whether to add a global-context attention block after the backbone.
        """

        super().__init__()
        self.backbone = ResNet31Backbone(channels=dim)
        self.gc = MultiAspectGCAttention(dim * 2) if attention_backbone else nn.Identity()
        self.hybrid_decoder = SequenceAttentionDecoder(dim * 2, vocab=vocab)
        self.pos_embed = nn.Parameter(torch.randn(1, 8, dim * 2) * 0.02)
        self.pos_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim * 2), nn.ReLU(), nn.Linear(dim * 2, vocab)
        )
        self.fuse = nn.Linear(vocab * 2, vocab)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Recognize text with dynamic context/position fusion.

        Parameters
        ----------
        image:
            Grayscale word image.

        Returns
        -------
        torch.Tensor
            Fused character logits.
        """

        feat = self.gc(self.backbone(image))
        memory = feat.flatten(2).transpose(1, 2)
        hybrid = self.hybrid_decoder(memory)
        pos = self.pos_mlp(self.pos_embed.expand(image.shape[0], -1, -1))
        gate = torch.sigmoid(self.fuse(torch.cat([hybrid, pos], dim=-1)))
        return gate * hybrid + (1.0 - gate) * pos


class SATRNRecognizer(nn.Module):
    """SATRN recognizer with 2D self-attention encoder and Transformer decoder."""

    def __init__(self, vocab: int = 38, dim: int = 48) -> None:
        """Initialize SATRN.

        Parameters
        ----------
        vocab:
            Output vocabulary size.
        dim:
            Hidden dimension.
        """

        super().__init__()
        self.patch = nn.Conv2d(1, dim, 4, stride=4)
        enc_layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True)
        dec_layer = nn.TransformerDecoderLayer(dim, 4, dim * 2, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, 1)
        self.decoder = nn.TransformerDecoder(dec_layer, 1)
        self.row = nn.Parameter(torch.randn(1, 8, 1, dim) * 0.02)
        self.col = nn.Parameter(torch.randn(1, 1, 24, dim) * 0.02)
        self.query = nn.Parameter(torch.randn(1, 8, dim) * 0.02)
        self.head = nn.Linear(dim, vocab)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Recognize arbitrary-shaped text through 2D spatial self-attention.

        Parameters
        ----------
        image:
            Grayscale word image.

        Returns
        -------
        torch.Tensor
            Character logits.
        """

        feat = self.patch(image).permute(0, 2, 3, 1)
        _, height, width, _ = feat.shape
        feat = feat + self.row[:, :height] + self.col[:, :, :width]
        memory = self.encoder(feat.reshape(image.shape[0], height * width, -1))
        query = self.query.expand(image.shape[0], -1, -1)
        return self.head(self.decoder(query, memory))


class SDMGRKIE(nn.Module):
    """Spatial Dual-Modality Graph Reasoning for key information extraction."""

    def __init__(self, text_vocab: int = 64, classes: int = 8, dim: int = 48) -> None:
        """Initialize SDMGR.

        Parameters
        ----------
        text_vocab:
            Text-token vocabulary size.
        classes:
            Node label count.
        dim:
            Hidden node dimension.
        """

        super().__init__()
        self.text = nn.Embedding(text_vocab, dim)
        self.visual = nn.Linear(8, dim)
        self.edge = nn.Linear(5, dim)
        self.msg = nn.Linear(dim * 3, dim)
        self.norm = nn.LayerNorm(dim)
        self.cls = nn.Linear(dim, classes)

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Classify document text boxes with dual-modality graph messages.

        Parameters
        ----------
        inputs:
            Tuple of text ids, visual region descriptors, and normalized boxes.

        Returns
        -------
        torch.Tensor
            Node classification logits.
        """

        text_ids, visual_feats, boxes = inputs
        node = self.text(text_ids).mean(dim=2) + self.visual(visual_feats)
        center = (boxes[..., :2] + boxes[..., 2:]) * 0.5
        delta = center.unsqueeze(2) - center.unsqueeze(1)
        wh = (boxes[..., 2:] - boxes[..., :2]).clamp_min(1e-3)
        area = (wh[..., 0] * wh[..., 1]).unsqueeze(-1)
        dist = delta.square().sum(dim=-1, keepdim=True).sqrt()
        edge_attr = torch.cat(
            [
                delta,
                dist,
                area.unsqueeze(2).expand(-1, -1, boxes.shape[1], -1),
                area.unsqueeze(1).expand(-1, boxes.shape[1], -1, -1),
            ],
            dim=-1,
        )
        edge = self.edge(edge_attr)
        src = node.unsqueeze(1).expand(-1, boxes.shape[1], -1, -1)
        dst = node.unsqueeze(2).expand_as(src)
        weights = torch.softmax(-dist.squeeze(-1), dim=-1).unsqueeze(-1)
        messages = self.msg(torch.cat([src, dst, edge], dim=-1))
        node = self.norm(node + (weights * messages).sum(dim=2))
        return self.cls(node)


def text_image() -> torch.Tensor:
    """Create a compact grayscale word image.

    Returns
    -------
    torch.Tensor
        Image tensor with shape ``(1, 1, 32, 96)``.
    """

    return torch.randn(1, 1, 32, 96)


def kie_input() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create compact document nodes for SDMGR.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Text ids, visual features, and bounding boxes.
    """

    text_ids = torch.randint(0, 64, (1, 6, 5))
    visual = torch.randn(1, 6, 8)
    xs = torch.linspace(0.05, 0.75, 6).view(1, 6, 1)
    ys = torch.linspace(0.1, 0.6, 6).flip(0).view(1, 6, 1)
    boxes = torch.cat([xs, ys, xs + 0.18, ys + 0.08], dim=-1)
    return text_ids, visual, boxes


def sinusoidal_2d_tokens(height: int, width: int, dim: int, device: torch.device) -> torch.Tensor:
    """Create 2D sinusoidal tokens for small text models.

    Parameters
    ----------
    height:
        Grid height.
    width:
        Grid width.
    dim:
        Embedding dimension.
    device:
        Target device.

    Returns
    -------
    torch.Tensor
        Positional token tensor.
    """

    ys = torch.arange(height, device=device).float().view(height, 1)
    xs = torch.arange(width, device=device).float().view(1, width)
    scale = torch.arange(dim, device=device).float().view(1, 1, dim).clamp_min(1.0)
    return torch.sin((ys.unsqueeze(-1) + xs.unsqueeze(-1)) / torch.pow(10000.0, scale / dim))
