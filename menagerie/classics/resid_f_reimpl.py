"""Slice-F dependency-hostile residual architectures in base PyTorch.

Paper: representative source papers for PyG pooling/link prediction, Longformer,
LUKE, MPNet, I-BERT/QDQ-BERT, GPTSAN, Jina embeddings v3, GLM-style decoders,
Mega, MiDaS/DPT, DeepLabV3/MAnet, SAM2/Hiera, FocalNet, MaxViT, MSG-ProGAN, and
ProGAN.

These are compact random-initialized reimplementations for catalog validation.
They preserve the load-bearing structure of each family while avoiding optional
packages, custom CUDA kernels, hub downloads, and very large default configs.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _as_tokens(x: Tensor, vocab: int = 256) -> Tensor:
    """Convert an arbitrary input tensor into token ids.

    Parameters
    ----------
    x:
        Input tensor.
    vocab:
        Vocabulary size.

    Returns
    -------
    Tensor
        Integer token ids with shape ``(batch, time)``.
    """

    if x.dtype.is_floating_point or x.dtype.is_complex:
        flat = (x.reshape(x.shape[0], -1).abs() * 997).long()
    else:
        flat = x.reshape(x.shape[0], -1).long()
    if flat.shape[1] < 8:
        flat = F.pad(flat, (0, 8 - flat.shape[1]))
    return flat[:, :32].remainder(vocab)


def _causal_mask(length: int, device: torch.device) -> Tensor:
    """Create an additive causal attention mask.

    Parameters
    ----------
    length:
        Sequence length.
    device:
        Tensor device.

    Returns
    -------
    Tensor
        Boolean upper-triangular mask.
    """

    return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)


class FakeQuantLinear(nn.Module):
    """Linear layer with I-BERT/QDQ-style fake quantization around matmul."""

    def __init__(self, in_features: int, out_features: int) -> None:
        """Initialize the quantized linear projection.

        Parameters
        ----------
        in_features:
            Input feature width.
        out_features:
            Output feature width.
        """

        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.act_scale = nn.Parameter(torch.ones(1))
        self.weight_scale = nn.Parameter(torch.ones(1))

    def _fake_quant(self, value: Tensor, scale: Tensor) -> Tensor:
        """Apply straight-through fake int8 quantization.

        Parameters
        ----------
        value:
            Floating point tensor.
        scale:
            Learnable positive scale.

        Returns
        -------
        Tensor
            Fake-quantized tensor.
        """

        safe_scale = F.softplus(scale) + 1e-4
        quant = torch.clamp(torch.round(value / safe_scale), -127, 127) * safe_scale
        return value + (quant - value).detach()

    def forward(self, x: Tensor) -> Tensor:
        """Project fake-quantized activations with fake-quantized weights.

        Parameters
        ----------
        x:
            Input activations.

        Returns
        -------
        Tensor
            Projected activations.
        """

        qx = self._fake_quant(x, self.act_scale)
        qw = self._fake_quant(self.linear.weight, self.weight_scale)
        return F.linear(qx, qw, self.linear.bias)


class ResidualAttentionBlock(nn.Module):
    """Transformer block with optional pre-layernorm and fake quantized MLP."""

    def __init__(self, dim: int = 48, heads: int = 4, pre_norm: bool = False, quant: bool = False):
        """Initialize the attention block.

        Parameters
        ----------
        dim:
            Hidden width.
        heads:
            Attention heads.
        pre_norm:
            Whether to use pre-layer normalization.
        quant:
            Whether to use fake-quantized feed-forward projections.
        """

        super().__init__()
        self.pre_norm = pre_norm
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        linear: type[nn.Linear] | type[FakeQuantLinear] = FakeQuantLinear if quant else nn.Linear
        self.ff = nn.Sequential(linear(dim, dim * 4), nn.GELU(), linear(dim * 4, dim))

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        """Apply self-attention and feed-forward residual updates.

        Parameters
        ----------
        x:
            Sequence tensor.
        attn_mask:
            Optional attention mask.

        Returns
        -------
        Tensor
            Updated sequence tensor.
        """

        if self.pre_norm:
            y = self.norm1(x)
            x = x + self.attn(y, y, y, attn_mask=attn_mask, need_weights=False)[0]
            return x + self.ff(self.norm2(x))
        y = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)[0]
        x = self.norm1(x + y)
        return self.norm2(x + self.ff(x))


class TextFamilyModel(nn.Module):
    """Compact text-model family switch for dependency-hostile transformer rows."""

    def __init__(self, variant: str, vocab: int = 256, dim: int = 48) -> None:
        """Initialize embeddings and variant-specific blocks.

        Parameters
        ----------
        variant:
            Text architecture variant.
        vocab:
            Vocabulary size.
        dim:
            Hidden width.
        """

        super().__init__()
        self.variant = variant
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, dim)
        self.entity_embed = nn.Embedding(32, dim)
        self.pos = nn.Parameter(torch.zeros(1, 96, dim))
        self.blocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    dim,
                    pre_norm=variant in {"roberta_prelayernorm", "glm5"},
                    quant=variant in {"ibert", "qdqbert"},
                )
                for _ in range(2)
            ]
        )
        self.local_qkv = nn.Linear(dim, dim * 3)
        self.global_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.entity_gate = nn.Linear(dim * 2, dim)
        self.permutation = nn.Linear(dim, dim)
        self.prefix = nn.GRU(dim, dim, batch_first=True)
        self.lora_a = nn.Linear(dim, 8, bias=False)
        self.lora_b = nn.Linear(8, dim, bias=False)
        self.ema_conv = nn.Conv1d(dim, dim, 5, padding=4, groups=dim)
        self.gate = nn.Linear(dim, dim)
        self.head = nn.Linear(dim, vocab)
        self.pool = nn.Linear(dim, 64)

    def _base(self, ids: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        """Run the shared Transformer encoder.

        Parameters
        ----------
        ids:
            Token ids.
        attn_mask:
            Optional attention mask.

        Returns
        -------
        Tensor
            Contextual token features.
        """

        x = self.embed(ids) + self.pos[:, : ids.shape[1]]
        for block in self.blocks:
            x = block(x, attn_mask)
        return x

    def _longformer(self, ids: Tensor) -> Tensor:
        """Apply sliding-window plus global-token Longformer attention.

        Parameters
        ----------
        ids:
            Token ids.

        Returns
        -------
        Tensor
            Token logits.
        """

        x = self.embed(ids) + self.pos[:, : ids.shape[1]]
        global_token = self.global_token.expand(ids.shape[0], -1, -1)
        x = torch.cat([global_token, x], dim=1)
        length = x.shape[1]
        local_mask = torch.ones(length, length, dtype=torch.bool, device=x.device)
        for offset in range(-2, 3):
            local_mask.diagonal(offset=offset).fill_(False)
        local_mask[0, :] = False
        local_mask[:, 0] = False
        for block in self.blocks:
            x = block(x, local_mask)
        return self.head(x[:, 1:])

    def _luke(self, ids: Tensor) -> Tensor:
        """Apply LUKE-style word/entity-aware contextualization.

        Parameters
        ----------
        ids:
            Token ids.

        Returns
        -------
        Tensor
            Token logits.
        """

        word = self._base(ids)
        ent_ids = ids[:, : min(4, ids.shape[1])].remainder(32)
        ent = self.entity_embed(ent_ids)
        ent_context = ent.mean(dim=1, keepdim=True).expand(-1, word.shape[1], -1)
        mixed = word + torch.tanh(self.entity_gate(torch.cat([word, ent_context], dim=-1)))
        return self.head(mixed)

    def _mpnet(self, ids: Tensor) -> Tensor:
        """Apply MPNet-style permuted context prediction.

        Parameters
        ----------
        ids:
            Token ids.

        Returns
        -------
        Tensor
            Token logits.
        """

        perm = torch.arange(ids.shape[1] - 1, -1, -1, device=ids.device)
        x = self._base(ids[:, perm])
        x = self.permutation(x)
        return self.head(x[:, perm])

    def _gptsan(self, ids: Tensor) -> Tensor:
        """Apply GPTSAN-style prefix-LM sparse seq2seq processing.

        Parameters
        ----------
        ids:
            Token ids.

        Returns
        -------
        Tensor
            Decoder logits.
        """

        prefix_len = max(2, ids.shape[1] // 2)
        prefix, _ = self.prefix(self.embed(ids[:, :prefix_len]))
        decoder = self.embed(ids[:, prefix_len:]) + prefix[:, -1:, :]
        for block in self.blocks:
            decoder = block(decoder, _causal_mask(decoder.shape[1], decoder.device))
        return self.head(decoder)

    def _jina(self, ids: Tensor) -> Tensor:
        """Apply XLM-R plus task-LoRA and Matryoshka-style normalized pooling.

        Parameters
        ----------
        ids:
            Token ids.

        Returns
        -------
        Tensor
            Embedding vector with nested-prefix dimensions.
        """

        hidden = self._base(ids)
        hidden = hidden + self.lora_b(self.lora_a(hidden))
        pooled = self.pool(hidden.mean(dim=1))
        nested = torch.cat(
            [
                F.normalize(pooled[:, :16], dim=-1),
                F.normalize(pooled[:, :32], dim=-1)[:, 16:],
                F.normalize(pooled, dim=-1)[:, 32:],
            ],
            dim=-1,
        )
        return nested

    def _mega(self, ids: Tensor) -> Tensor:
        """Apply Mega-style moving-average gated attention.

        Parameters
        ----------
        ids:
            Token ids.

        Returns
        -------
        Tensor
            Token logits.
        """

        x = self.embed(ids)
        ema = self.ema_conv(x.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2)
        gated = torch.sigmoid(self.gate(x)) * torch.tanh(ema)
        for block in self.blocks:
            gated = block(gated)
        return self.head(gated)

    def forward(self, x: Tensor) -> Tensor:
        """Run the selected text architecture.

        Parameters
        ----------
        x:
            Token ids or tensor convertible to token ids.

        Returns
        -------
        Tensor
            Model output tensor.
        """

        ids = _as_tokens(x, self.vocab)
        if self.variant == "longformer":
            return self._longformer(ids)
        if self.variant == "luke":
            return self._luke(ids)
        if self.variant == "mpnet":
            return self._mpnet(ids)
        if self.variant == "gptsan":
            return self._gptsan(ids)
        if self.variant == "jina":
            return self._jina(ids)
        if self.variant == "mega":
            return self._mega(ids)
        mask = _causal_mask(ids.shape[1], ids.device) if self.variant == "glm5" else None
        return self.head(self._base(ids, mask))


class GraphPoolingModel(nn.Module):
    """Torch-only graph pooling/link-prediction primitives for PyG residual rows."""

    def __init__(self, mode: str, nodes: int = 8, dim: int = 16) -> None:
        """Initialize graph operators.

        Parameters
        ----------
        mode:
            ``"asa"``, ``"edge"``, or ``"arlink"``.
        nodes:
            Number of compact graph nodes.
        dim:
            Node feature width.
        """

        super().__init__()
        self.mode = mode
        self.nodes = nodes
        self.dim = dim
        self.embed = nn.Embedding(nodes, dim)
        self.score = nn.Linear(dim, 1)
        self.gnn = nn.Linear(dim * 2, dim)
        self.edge_score = nn.Linear(dim * 2, 1)
        self.link_rnn = nn.GRU(dim * 2, dim, batch_first=True)
        self.link_head = nn.Linear(dim, 1)
        src = torch.tensor([0, 1, 2, 3, 4, 5, 6, 0, 2, 4, 1, 3])
        dst = torch.tensor([1, 2, 3, 4, 5, 6, 7, 2, 4, 6, 5, 7])
        self.register_buffer("src", src)
        self.register_buffer("dst", dst)

    def _node_features(self, x: Tensor) -> Tensor:
        """Construct node features from input or embeddings.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        Tensor
            Node features.
        """

        if x.dtype.is_floating_point and x.ndim == 2 and x.shape[-1] == self.dim:
            return x[: self.nodes]
        ids = torch.arange(self.nodes, device=x.device)
        return self.embed(ids)

    def _message_pass(self, feats: Tensor) -> Tensor:
        """Aggregate one neighborhood message-passing step.

        Parameters
        ----------
        feats:
            Node feature matrix.

        Returns
        -------
        Tensor
            Updated node features.
        """

        msg = torch.tanh(self.gnn(torch.cat([feats[self.src], feats[self.dst]], dim=-1)))
        agg = torch.zeros_like(feats).index_add(0, self.dst, msg)
        deg = torch.zeros(feats.shape[0], device=feats.device).index_add(
            0, self.dst, torch.ones_like(self.dst, dtype=feats.dtype)
        )
        return feats + agg / deg.clamp_min(1).unsqueeze(-1)

    def _asapool(self, feats: Tensor) -> Tensor:
        """Apply ASAPooling-style attentive cluster pooling.

        Parameters
        ----------
        feats:
            Node feature matrix.

        Returns
        -------
        Tensor
            Graph embedding after top-k pooling.
        """

        propagated = self._message_pass(feats)
        scores = self.score(torch.tanh(propagated)).squeeze(-1)
        keep = torch.topk(scores, k=max(2, feats.shape[0] // 2)).indices
        pooled = propagated[keep] * torch.sigmoid(scores[keep]).unsqueeze(-1)
        return pooled.mean(dim=0, keepdim=True)

    def _edgepool(self, feats: Tensor) -> Tensor:
        """Apply EdgePooling-style scored edge contraction.

        Parameters
        ----------
        feats:
            Node feature matrix.

        Returns
        -------
        Tensor
            Contracted graph embedding.
        """

        edge_feat = torch.cat([feats[self.src], feats[self.dst]], dim=-1)
        weights = torch.softmax(self.edge_score(edge_feat).squeeze(-1), dim=0)
        contracted = feats.clone()
        for index, (src, dst) in enumerate(zip(self.src.tolist(), self.dst.tolist())):
            mix = weights[index] * 0.5 * (feats[src] + feats[dst])
            contracted[dst] = contracted[dst] + mix
        return contracted.mean(dim=0, keepdim=True)

    def _arlink(self, pairs: Tensor) -> Tensor:
        """Apply autoregressive link scoring over queried node pairs.

        Parameters
        ----------
        pairs:
            Integer pair tensor.

        Returns
        -------
        Tensor
            Link logits.
        """

        pair_ids = pairs.reshape(-1, 2).long().remainder(self.nodes)
        pair_emb = torch.cat([self.embed(pair_ids[:, 0]), self.embed(pair_ids[:, 1])], dim=-1)
        seq = pair_emb.unsqueeze(0)
        hidden, _ = self.link_rnn(seq)
        return self.link_head(hidden.squeeze(0)).squeeze(-1)

    def forward(self, x: Tensor) -> Tensor:
        """Run graph pooling or link prediction.

        Parameters
        ----------
        x:
            Node features for pooling or integer pairs for link prediction.

        Returns
        -------
        Tensor
            Graph output tensor.
        """

        if self.mode == "arlink":
            return self._arlink(x)
        feats = self._node_features(x)
        if self.mode == "edge":
            return self._edgepool(feats)
        return self._asapool(feats)


class CompactDPT(nn.Module):
    """Dense Prediction Transformer / MiDaS-style monocular depth model."""

    def __init__(self, hybrid: bool = False, dim: int = 48) -> None:
        """Initialize patch encoder and DPT fusion decoder.

        Parameters
        ----------
        hybrid:
            Whether to include a convolutional ResNet-style stem.
        dim:
            Token width.
        """

        super().__init__()
        self.hybrid = hybrid
        self.stem = nn.Sequential(
            nn.Conv2d(3, dim, 7, stride=2, padding=3),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.patch = nn.Conv2d(dim if hybrid else 3, dim, 4, stride=4)
        enc = nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, num_layers=2)
        self.fuse1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.fuse2 = nn.Conv2d(dim, dim // 2, 3, padding=1)
        self.head = nn.Conv2d(dim // 2, 1, 3, padding=1)

    def forward(self, image: Tensor) -> Tensor:
        """Predict relative inverse depth.

        Parameters
        ----------
        image:
            RGB image tensor.

        Returns
        -------
        Tensor
            Dense depth map.
        """

        base = self.stem(image) if self.hybrid else image
        patches = self.patch(base)
        batch, channels, height, width = patches.shape
        tokens = patches.flatten(2).transpose(1, 2)
        tokens = self.encoder(tokens)
        grid = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        grid = F.gelu(self.fuse1(grid))
        grid = F.interpolate(grid, scale_factor=2, mode="bilinear", align_corners=False)
        grid = F.gelu(self.fuse2(grid))
        depth = self.head(grid)
        return F.interpolate(depth, size=image.shape[-2:], mode="bilinear", align_corners=False)


class AtrousSegmentationModel(nn.Module):
    """DeepLabV3 or MAnet semantic segmentation over compact backbones."""

    def __init__(self, mode: str = "deeplab", classes: int = 21, width: int = 32) -> None:
        """Initialize segmentation backbone and head.

        Parameters
        ----------
        mode:
            ``"deeplab"`` or ``"manet"``.
        classes:
            Number of segmentation classes.
        width:
            Base channel count.
        """

        super().__init__()
        self.mode = mode
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=False),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=False),
        )
        self.aspp = nn.ModuleList(
            [
                nn.Conv2d(width * 2, width, 1),
                nn.Conv2d(width * 2, width, 3, padding=3, dilation=3),
                nn.Conv2d(width * 2, width, 3, padding=6, dilation=6),
            ]
        )
        self.manet_gate = nn.Sequential(nn.Conv2d(width * 3, width * 3, 1), nn.Sigmoid())
        self.head = nn.Sequential(
            nn.Conv2d(width * 3, width, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(width, classes, 1),
        )

    def forward(self, image: Tensor) -> Tensor:
        """Predict semantic logits.

        Parameters
        ----------
        image:
            RGB image tensor.

        Returns
        -------
        Tensor
            Segmentation logits.
        """

        features = self.stem(image)
        pyramid = torch.cat([branch(features) for branch in self.aspp], dim=1)
        if self.mode == "manet":
            context = F.adaptive_avg_pool2d(pyramid, 1)
            pyramid = pyramid * self.manet_gate(context)
        logits = self.head(pyramid)
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)


class HierarchicalVisionModel(nn.Module):
    """Compact FocalNet, MaxViT, SAM2-Hiera, or large CNN image tower."""

    def __init__(self, mode: str = "focal", classes: int = 1000, width: int = 32) -> None:
        """Initialize hierarchical image tower.

        Parameters
        ----------
        mode:
            Vision architecture mode.
        classes:
            Output class count.
        width:
            Base channel count.
        """

        super().__init__()
        self.mode = mode
        self.stem = nn.Conv2d(3, width, 4, stride=4)
        self.dw3 = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.dw5 = nn.Conv2d(width, width, 5, padding=2, groups=width)
        self.dw7 = nn.Conv2d(width, width, 7, padding=3, groups=width)
        self.channel = nn.Sequential(
            nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width)
        )
        self.attn = nn.MultiheadAttention(width, 4, batch_first=True)
        self.down = nn.Conv2d(width, width * 2, 3, stride=2, padding=1)
        self.head = nn.Linear(width * 2, classes)

    def forward(self, image: Tensor) -> Tensor:
        """Classify an image with the selected hierarchy.

        Parameters
        ----------
        image:
            RGB image tensor.

        Returns
        -------
        Tensor
            Class logits or image embedding.
        """

        x = F.gelu(self.stem(image))
        if self.mode == "focal":
            context = self.dw3(x) + self.dw5(x) + self.dw7(x)
            x = x * torch.sigmoid(context)
        else:
            tokens = x.flatten(2).transpose(1, 2)
            attn, _ = self.attn(tokens, tokens, tokens, need_weights=False)
            x = (tokens + attn + self.channel(tokens)).transpose(1, 2).reshape_as(x)
        x = F.gelu(self.down(x))
        pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(pooled)


class SmallGanGenerator(nn.Module):
    """Progressive/MS-GAN-style generator with multi-scale RGB outputs."""

    def __init__(self, mode: str = "progan", latent: int = 32) -> None:
        """Initialize generator.

        Parameters
        ----------
        mode:
            ``"progan"`` or ``"msg"``.
        latent:
            Latent vector width.
        """

        super().__init__()
        self.mode = mode
        self.fc = nn.Linear(latent, 64 * 4 * 4)
        self.block1 = nn.ConvTranspose2d(64, 48, 4, stride=2, padding=1)
        self.block2 = nn.ConvTranspose2d(48, 32, 4, stride=2, padding=1)
        self.block3 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.rgb1 = nn.Conv2d(48, 3, 1)
        self.rgb2 = nn.Conv2d(32, 3, 1)
        self.rgb3 = nn.Conv2d(16, 3, 1)

    def forward(self, z: Tensor) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        """Generate an image or multi-scale image pyramid.

        Parameters
        ----------
        z:
            Latent tensor.

        Returns
        -------
        Tensor | tuple[Tensor, Tensor, Tensor]
            Final image or MSG-GAN scale pyramid.
        """

        z = z.reshape(z.shape[0], -1)
        if z.shape[1] < 32:
            z = F.pad(z, (0, 32 - z.shape[1]))
        x = self.fc(z[:, :32]).view(z.shape[0], 64, 4, 4)
        x1 = F.leaky_relu(self.block1(x), 0.2)
        x2 = F.leaky_relu(self.block2(x1), 0.2)
        x3 = F.leaky_relu(self.block3(x2), 0.2)
        out1 = torch.tanh(self.rgb1(x1))
        out2 = torch.tanh(self.rgb2(x2))
        out3 = torch.tanh(self.rgb3(x3))
        if self.mode == "msg":
            return out1, out2, out3
        return out3


class _SE(nn.Module):
    """Squeeze-and-excitation channel gate."""

    def __init__(self, channels: int, ratio: int = 4) -> None:
        """Initialize the channel gate.

        Parameters
        ----------
        channels:
            Channel count.
        ratio:
            Bottleneck ratio.
        """

        super().__init__()
        hidden = max(4, channels // ratio)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Apply squeeze-and-excitation.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        Tensor
            Gated feature map.
        """

        gate = F.adaptive_avg_pool2d(x, 1)
        gate = torch.sigmoid(self.fc2(F.silu(self.fc1(gate))))
        return x * gate


class _ResidualConvBlock(nn.Module):
    """Family-configurable residual bottleneck block."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        groups: int = 1,
        preact: bool = False,
        se: bool = False,
        norm: str = "batch",
    ) -> None:
        """Initialize the residual block.

        Parameters
        ----------
        in_ch:
            Input channels.
        out_ch:
            Output channels.
        stride:
            Spatial stride.
        groups:
            Group count for the 3x3 convolution.
        preact:
            Whether to use ResNetV2-style pre-activation.
        se:
            Whether to include squeeze-and-excitation.
        norm:
            ``"batch"``, ``"group"``, or ``"none"`` normalization.
        """

        super().__init__()
        self.preact = preact
        mid = max(8, out_ch // 2)
        self.norm1 = self._norm(in_ch, norm)
        self.conv1 = nn.Conv2d(in_ch, mid, 1, bias=norm == "none")
        self.norm2 = self._norm(mid, norm)
        self.conv2 = nn.Conv2d(
            mid,
            mid,
            3,
            stride=stride,
            padding=1,
            groups=max(1, min(groups, mid)),
            bias=norm == "none",
        )
        self.norm3 = self._norm(mid, norm)
        self.conv3 = nn.Conv2d(mid, out_ch, 1, bias=norm == "none")
        self.se = _SE(out_ch) if se else nn.Identity()
        self.skip = (
            nn.Identity()
            if stride == 1 and in_ch == out_ch
            else nn.Conv2d(in_ch, out_ch, 1, stride=stride)
        )

    def _norm(self, channels: int, norm: str) -> nn.Module:
        """Create a normalization layer.

        Parameters
        ----------
        channels:
            Channel count.
        norm:
            Normalization type.

        Returns
        -------
        nn.Module
            Normalization module.
        """

        if norm == "group":
            return nn.GroupNorm(min(8, channels), channels)
        if norm == "none":
            return nn.Identity()
        return nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        """Apply residual convolution.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        Tensor
            Output feature map.
        """

        if self.preact:
            y = self.conv1(F.silu(self.norm1(x)))
        else:
            y = F.silu(self.norm2(self.conv1(x)))
        y = F.silu(self.norm3(self.conv2(y)))
        y = self.conv3(y)
        return F.silu(self.se(y) + self.skip(x))


class RegNetYCompact(nn.Module):
    """RegNetY-style quantized-width grouped-convolution network with SE."""

    def __init__(self) -> None:
        """Initialize the compact RegNetY model."""

        super().__init__()
        widths = [24, 40, 64]
        self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 3, stride=2, padding=1), nn.SiLU())
        self.stage1 = _ResidualConvBlock(widths[0], widths[0], groups=4, se=True)
        self.stage2 = _ResidualConvBlock(widths[0], widths[1], stride=2, groups=5, se=True)
        self.stage3 = _ResidualConvBlock(widths[1], widths[2], stride=2, groups=8, se=True)
        self.head = nn.Linear(widths[2], 1000)

    def forward(self, image: Tensor) -> Tensor:
        """Classify an image.

        Parameters
        ----------
        image:
            RGB image tensor.

        Returns
        -------
        Tensor
            Class logits.
        """

        x = self.stage3(self.stage2(self.stage1(self.stem(image))))
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))


class ResNetV2BitCompact(nn.Module):
    """BiT/ResNetV2-style pre-activation residual image tower."""

    def __init__(self) -> None:
        """Initialize the compact ResNetV2/BiT model."""

        super().__init__()
        self.root = nn.Conv2d(3, 32, 7, stride=2, padding=3)
        self.blocks = nn.Sequential(
            _ResidualConvBlock(32, 48, stride=2, preact=True, norm="group"),
            _ResidualConvBlock(48, 64, stride=2, preact=True, norm="group"),
            _ResidualConvBlock(64, 96, stride=2, preact=True, norm="group"),
        )
        self.norm = nn.GroupNorm(8, 96)
        self.head = nn.Linear(96, 1000)

    def forward(self, image: Tensor) -> Tensor:
        """Classify an image with pre-activation residual stages.

        Parameters
        ----------
        image:
            RGB image tensor.

        Returns
        -------
        Tensor
            Class logits.
        """

        x = self.blocks(self.root(image))
        x = F.silu(self.norm(x))
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))


class ClipResNetCompact(nn.Module):
    """CLIP modified ResNet image tower with attention pooling."""

    def __init__(self, dim: int = 64) -> None:
        """Initialize the compact CLIP ResNet.

        Parameters
        ----------
        dim:
            Attention-pool width.
        """

        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.Conv2d(24, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.stage1 = _ResidualConvBlock(32, 48, stride=2)
        self.stage2 = _ResidualConvBlock(48, dim, stride=2)
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pool = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.proj = nn.Linear(dim, 512)

    def forward(self, image: Tensor) -> Tensor:
        """Encode an image with attention pooling.

        Parameters
        ----------
        image:
            RGB image tensor.

        Returns
        -------
        Tensor
            CLIP-style image embedding.
        """

        feats = self.stage2(self.stage1(self.stem(image)))
        tokens = feats.flatten(2).transpose(1, 2)
        cls = self.cls.expand(image.shape[0], -1, -1)
        pooled, _ = self.pool(cls, torch.cat([cls, tokens], dim=1), torch.cat([cls, tokens], dim=1))
        return self.proj(pooled.squeeze(1))


class NFNetCompact(nn.Module):
    """Normalization-free residual network with scaled residual branches."""

    def __init__(self) -> None:
        """Initialize compact NFNet."""

        super().__init__()
        self.stem = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.block1 = _ResidualConvBlock(32, 48, stride=2, se=True, norm="none")
        self.block2 = _ResidualConvBlock(48, 64, stride=2, se=True, norm="none")
        self.block3 = _ResidualConvBlock(64, 96, stride=2, se=True, norm="none")
        self.gain = nn.Parameter(torch.tensor(0.2))
        self.head = nn.Linear(96, 1000)

    def forward(self, image: Tensor) -> Tensor:
        """Classify an image with normalization-free residual blocks.

        Parameters
        ----------
        image:
            RGB image tensor.

        Returns
        -------
        Tensor
            Class logits.
        """

        x = F.gelu(self.stem(image))
        for block in (self.block1, self.block2, self.block3):
            x = block(x) * (1.0 + self.gain)
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))


class MViTv2ImageCompact(nn.Module):
    """MViTv2-style image transformer with pooled query/key/value tokens."""

    def __init__(self, dim: int = 64) -> None:
        """Initialize compact MViTv2 image model.

        Parameters
        ----------
        dim:
            Token width.
        """

        super().__init__()
        self.patch = nn.Conv2d(3, dim, 4, stride=4)
        self.pool = nn.Conv2d(dim, dim, 3, stride=2, padding=1, groups=dim)
        self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.head = nn.Linear(dim, 1000)

    def forward(self, image: Tensor) -> Tensor:
        """Classify an image with multiscale token pooling.

        Parameters
        ----------
        image:
            RGB image tensor.

        Returns
        -------
        Tensor
            Class logits.
        """

        grid = F.gelu(self.patch(image))
        pooled_grid = self.pool(grid)
        tokens = grid.flatten(2).transpose(1, 2)
        pooled = pooled_grid.flatten(2).transpose(1, 2)
        mixed, _ = self.attn(tokens, pooled, pooled, need_weights=False)
        tokens = self.norm(tokens + mixed)
        tokens = tokens + self.ff(tokens)
        return self.head(tokens.mean(dim=1))


class RNNTDecoderCompact(nn.Module):
    """RNNT decoder/joint network without tuple-returning submodules."""

    def __init__(self, vocab: int = 64, dim: int = 48) -> None:
        """Initialize compact RNNT components.

        Parameters
        ----------
        vocab:
            Vocabulary size.
        dim:
            Hidden width.
        """

        super().__init__()
        self.acoustic = nn.Linear(32, dim)
        self.pred_embed = nn.Embedding(vocab, dim)
        self.pred_cell = nn.GRUCell(dim, dim)
        self.joint = nn.Sequential(nn.Linear(dim * 2, dim), nn.Tanh(), nn.Linear(dim, vocab))

    def forward(self, feats: Tensor) -> Tensor:
        """Run RNNT prediction and joint scoring.

        Parameters
        ----------
        feats:
            Acoustic feature tensor ``(batch, time, features)``.

        Returns
        -------
        Tensor
            Joint network logits.
        """

        acoustic = torch.tanh(self.acoustic(feats))
        batch = feats.shape[0]
        state = torch.zeros(batch, acoustic.shape[-1], device=feats.device, dtype=feats.dtype)
        outputs = []
        tokens = torch.arange(min(4, feats.shape[1]), device=feats.device).expand(batch, -1)
        for token in tokens.unbind(1):
            state = self.pred_cell(self.pred_embed(token), state)
            joint_in = torch.cat([acoustic.mean(dim=1), state], dim=-1)
            outputs.append(self.joint(joint_in))
        return torch.stack(outputs, dim=1)


def build_asapooling() -> nn.Module:
    """Build compact ASAPooling graph model.

    Returns
    -------
    nn.Module
        Graph pooling module.
    """

    return GraphPoolingModel("asa")


def build_edgepooling() -> nn.Module:
    """Build compact EdgePooling graph model.

    Returns
    -------
    nn.Module
        Graph pooling module.
    """

    return GraphPoolingModel("edge")


def build_arlink_predictor() -> nn.Module:
    """Build compact autoregressive link predictor.

    Returns
    -------
    nn.Module
        Link predictor module.
    """

    return GraphPoolingModel("arlink")


def build_ibert() -> nn.Module:
    """Build compact I-BERT model.

    Returns
    -------
    nn.Module
        Text model.
    """

    return TextFamilyModel("ibert")


def build_qdqbert() -> nn.Module:
    """Build compact QDQ-BERT model.

    Returns
    -------
    nn.Module
        Text model.
    """

    return TextFamilyModel("qdqbert")


def build_longformer() -> nn.Module:
    """Build compact Longformer model.

    Returns
    -------
    nn.Module
        Text model.
    """

    return TextFamilyModel("longformer")


def build_luke() -> nn.Module:
    """Build compact LUKE model.

    Returns
    -------
    nn.Module
        Text model.
    """

    return TextFamilyModel("luke")


def build_mpnet() -> nn.Module:
    """Build compact MPNet model.

    Returns
    -------
    nn.Module
        Text model.
    """

    return TextFamilyModel("mpnet")


def build_roberta() -> nn.Module:
    """Build compact RoBERTa model.

    Returns
    -------
    nn.Module
        Text model.
    """

    return TextFamilyModel("roberta")


def build_roberta_prelayernorm() -> nn.Module:
    """Build compact RoBERTa-PreLayerNorm model.

    Returns
    -------
    nn.Module
        Text model.
    """

    return TextFamilyModel("roberta_prelayernorm")


def build_gptsan() -> nn.Module:
    """Build compact GPTSAN Japanese model.

    Returns
    -------
    nn.Module
        Text model.
    """

    return TextFamilyModel("gptsan")


def build_jina_embeddings_v3() -> nn.Module:
    """Build compact Jina embeddings v3 model.

    Returns
    -------
    nn.Module
        Text embedding model.
    """

    return TextFamilyModel("jina")


def build_glm5() -> nn.Module:
    """Build compact GLM5-style decoder model.

    Returns
    -------
    nn.Module
        Text model.
    """

    return TextFamilyModel("glm5")


def build_mega_causal() -> nn.Module:
    """Build compact Mega causal language model.

    Returns
    -------
    nn.Module
        Text model.
    """

    return TextFamilyModel("mega")


def build_midas_dpt_hybrid() -> nn.Module:
    """Build compact MiDaS DPT-Hybrid model.

    Returns
    -------
    nn.Module
        Depth model.
    """

    return CompactDPT(hybrid=True)


def build_midas_dpt_large() -> nn.Module:
    """Build compact MiDaS DPT-Large model.

    Returns
    -------
    nn.Module
        Depth model.
    """

    return CompactDPT(hybrid=False, dim=64)


def build_midas_v21() -> nn.Module:
    """Build compact MiDaS v2.1 CNN-hybrid model.

    Returns
    -------
    nn.Module
        Depth model.
    """

    return CompactDPT(hybrid=True, dim=40)


def build_deeplabv3() -> nn.Module:
    """Build compact DeepLabV3 model.

    Returns
    -------
    nn.Module
        Segmentation model.
    """

    return AtrousSegmentationModel("deeplab")


def build_manet() -> nn.Module:
    """Build compact MAnet model.

    Returns
    -------
    nn.Module
        Segmentation model.
    """

    return AtrousSegmentationModel("manet")


def build_focalnet() -> nn.Module:
    """Build compact FocalNet-style classifier.

    Returns
    -------
    nn.Module
        Vision model.
    """

    return HierarchicalVisionModel("focal")


def build_maxvit() -> nn.Module:
    """Build compact MaxViT-style classifier.

    Returns
    -------
    nn.Module
        Vision model.
    """

    return HierarchicalVisionModel("maxvit")


def build_sam2_hiera() -> nn.Module:
    """Build compact SAM2-Hiera image encoder.

    Returns
    -------
    nn.Module
        Vision model.
    """

    return HierarchicalVisionModel("sam2")


def build_large_resnet_tower() -> nn.Module:
    """Build compact large-ResNet-family image tower.

    Returns
    -------
    nn.Module
        Vision model.
    """

    return HierarchicalVisionModel("resnet", width=40)


def build_regnety() -> nn.Module:
    """Build compact RegNetY model.

    Returns
    -------
    nn.Module
        RegNetY-style image classifier.
    """

    return RegNetYCompact()


def build_resnetv2_bit() -> nn.Module:
    """Build compact ResNetV2/BiT model.

    Returns
    -------
    nn.Module
        ResNetV2-style image classifier.
    """

    return ResNetV2BitCompact()


def build_clip_resnet() -> nn.Module:
    """Build compact CLIP modified ResNet image tower.

    Returns
    -------
    nn.Module
        CLIP-style image encoder.
    """

    return ClipResNetCompact()


def build_nfnet() -> nn.Module:
    """Build compact NFNet model.

    Returns
    -------
    nn.Module
        Normalization-free image classifier.
    """

    return NFNetCompact()


def build_mvitv2_image() -> nn.Module:
    """Build compact MViTv2 image model.

    Returns
    -------
    nn.Module
        Multiscale vision transformer image classifier.
    """

    return MViTv2ImageCompact()


def build_progan_generator() -> nn.Module:
    """Build compact ProGAN generator.

    Returns
    -------
    nn.Module
        Generator model.
    """

    return SmallGanGenerator("progan")


def build_msg_gan_generator() -> nn.Module:
    """Build compact MSG-GAN generator.

    Returns
    -------
    nn.Module
        Generator model.
    """

    return SmallGanGenerator("msg")


def build_nemo_rnnt_decoder() -> nn.Module:
    """Build compact NeMo RNNT decoder and joint network.

    Returns
    -------
    nn.Module
        RNNT decoder module.
    """

    return RNNTDecoderCompact()


def example_tokens() -> Tensor:
    """Create token ids for compact language rows.

    Returns
    -------
    Tensor
        Token id tensor.
    """

    return torch.arange(16, dtype=torch.long).unsqueeze(0)


def example_graph_features() -> Tensor:
    """Create compact graph node features.

    Returns
    -------
    Tensor
        Node feature matrix.
    """

    return torch.randn(8, 16)


def example_link_pairs() -> Tensor:
    """Create compact link-prediction pairs.

    Returns
    -------
    Tensor
        Node-pair tensor.
    """

    return torch.tensor([[0, 1], [2, 7], [3, 5]], dtype=torch.long)


def example_image() -> Tensor:
    """Create compact RGB image input.

    Returns
    -------
    Tensor
        Image tensor.
    """

    return torch.randn(1, 3, 64, 64)


def example_acoustic_features() -> Tensor:
    """Create compact acoustic features for RNNT rows.

    Returns
    -------
    Tensor
        Acoustic feature tensor.
    """

    return torch.randn(1, 32, 32)


def example_latent() -> Tensor:
    """Create compact generator latent input.

    Returns
    -------
    Tensor
        Latent tensor.
    """

    return torch.randn(1, 32)
