"""Compact classics for dependency-gated reimplementation shard 2.4.

The module covers install-hostile public architectures with random-initialized,
small PyTorch reconstructions suitable for TorchLens rendering in the base
environment.  Each model keeps the architecture's defining computation while
shrinking widths, depths, and input sizes.

Sources used for the reconstructions include SDMGR (arXiv:2103.14470), SVTR
(arXiv:2205.00159), ViTSTR (arXiv:2105.08582), MobileSAM, Mixture-of-Depths
(arXiv:2404.02258), MagFace (CVPR 2021), MeshGraphNet, NequIP, Moonshine ASR,
Morpheus astronomy segmentation, and DeepSEA/Beluga-style regulatory genomics.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallSelfAttention(nn.Module):
    """Small multi-head self-attention without fused kernels."""

    def __init__(self, dim: int, heads: int = 4) -> None:
        """Initialize attention projections.

        Parameters
        ----------
        dim:
            Token embedding width.
        heads:
            Number of attention heads.
        """

        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention.

        Parameters
        ----------
        x:
            Token tensor with shape ``(batch, tokens, dim)``.

        Returns
        -------
        torch.Tensor
            Attended tokens with the same shape as ``x``.
        """

        batch, tokens, dim = x.shape
        qkv = self.qkv(x).view(batch, tokens, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim**0.5)
        out = torch.matmul(torch.softmax(logits, dim=-1), v)
        return self.proj(out.transpose(1, 2).reshape(batch, tokens, dim))


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block."""

    def __init__(self, dim: int, heads: int = 4, mlp_ratio: int = 2) -> None:
        """Initialize attention and feed-forward layers.

        Parameters
        ----------
        dim:
            Token embedding width.
        heads:
            Number of attention heads.
        mlp_ratio:
            Hidden expansion ratio for the feed-forward network.
        """

        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SmallSelfAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention and feed-forward residual updates.

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


class SDMGRNodeEdge(nn.Module):
    """Spatial Dual-Modality Graph Reasoning for document KIE."""

    def __init__(self, dim: int = 32, classes: int = 6) -> None:
        """Initialize node, edge, and recurrent graph-reasoning layers.

        Parameters
        ----------
        dim:
            Hidden feature width.
        classes:
            Number of node labels.
        """

        super().__init__()
        self.text = nn.Linear(16, dim)
        self.layout = nn.Linear(4, dim)
        self.edge = nn.Linear(dim * 2 + 4, dim)
        self.node_update = nn.GRUCell(dim, dim)
        self.node_head = nn.Linear(dim, classes)
        self.edge_head = nn.Linear(dim, 2)

    def forward(
        self, inputs: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Classify nodes and pairwise relations.

        Parameters
        ----------
        inputs:
            Tuple of OCR token features ``(batch, nodes, 16)`` and normalized
            boxes ``(batch, nodes, 4)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Node logits and edge logits.
        """

        feats, boxes = inputs
        nodes = F.relu(self.text(feats) + self.layout(boxes))
        src = nodes.unsqueeze(2).expand(-1, -1, nodes.size(1), -1)
        dst = nodes.unsqueeze(1).expand(-1, nodes.size(1), -1, -1)
        rel = boxes.unsqueeze(2) - boxes.unsqueeze(1)
        edges = F.relu(self.edge(torch.cat((src, dst, rel), dim=-1)))
        msg = edges.mean(dim=2)
        batch, count, dim = nodes.shape
        nodes = self.node_update(msg.reshape(-1, dim), nodes.reshape(-1, dim)).view(
            batch, count, dim
        )
        return self.node_head(nodes), self.edge_head(edges)


def build_sdmgr_nodeedge() -> nn.Module:
    """Build compact SDMGR node-edge model.

    Returns
    -------
    nn.Module
        Random-init SDMGR reconstruction.
    """

    return SDMGRNodeEdge()


def example_sdmgr_nodeedge() -> tuple[torch.Tensor, torch.Tensor]:
    """Create SDMGR example OCR features and boxes.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Feature and box tensors.
    """

    return torch.randn(1, 6, 16), torch.rand(1, 6, 4)


class TextConvStem(nn.Module):
    """ResNet31-like convolutional stem for scene text images."""

    def __init__(self, dim: int = 48) -> None:
        """Initialize convolutional downsampling stem.

        Parameters
        ----------
        dim:
            Output channel count.
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 24, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(24, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image into a width-wise feature sequence.

        Parameters
        ----------
        x:
            Grayscale text image.

        Returns
        -------
        torch.Tensor
            Sequence tensor with shape ``(batch, width, channels)``.
        """

        feat = self.net(x).mean(dim=2)
        return feat.transpose(1, 2)


class SegOCRResNet31(nn.Module):
    """SegOCR-style ResNet31OCR + FPNOCR + segmentation head recognizer."""

    def __init__(self, vocab: int = 38) -> None:
        """Initialize recognizer.

        Parameters
        ----------
        vocab:
            Character vocabulary size.
        """

        super().__init__()
        self.stage1 = nn.Sequential(nn.Conv2d(1, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.stage2 = nn.Sequential(
            nn.Conv2d(24, 48, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1))
        )
        self.stage3 = nn.Sequential(nn.Conv2d(48, 48, 3, padding=1), nn.ReLU())
        self.lateral1 = nn.Conv2d(24, 48, 1)
        self.lateral2 = nn.Conv2d(48, 48, 1)
        self.seg_head = nn.Conv2d(48, vocab, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Recognize character logits through FPN and segmentation maps.

        Parameters
        ----------
        x:
            Grayscale image tensor.

        Returns
        -------
        torch.Tensor
            Per-position character logits.
        """

        c2 = self.stage1(x)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        p3 = self.lateral2(c3) + c4
        p2 = self.lateral1(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        seg = self.seg_head(p2)
        return seg.mean(dim=2).transpose(1, 2)


def build_segocr_resnet31() -> nn.Module:
    """Build compact SegOCR-ResNet31.

    Returns
    -------
    nn.Module
        Random-init recognizer.
    """

    return SegOCRResNet31()


def example_text_image() -> torch.Tensor:
    """Create a compact grayscale text-line image.

    Returns
    -------
    torch.Tensor
        Image tensor with shape ``(1, 1, 32, 96)``.
    """

    return torch.randn(1, 1, 32, 96)


class SVTRBlock(nn.Module):
    """SVTR local/global mixing block."""

    def __init__(self, dim: int, local: bool) -> None:
        """Initialize one SVTR mixing block.

        Parameters
        ----------
        dim:
            Channel width.
        local:
            Whether to use local depthwise convolution instead of global attention.
        """

        super().__init__()
        self.local = local
        self.norm = nn.LayerNorm(dim)
        self.attn = SmallSelfAttention(dim, heads=4)
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Apply SVTR token mixing.

        Parameters
        ----------
        x:
            Token tensor.
        height:
            Token-grid height.
        width:
            Token-grid width.

        Returns
        -------
        torch.Tensor
            Updated token tensor.
        """

        if self.local:
            feat = self.norm(x).transpose(1, 2).reshape(x.size(0), -1, height, width)
            mixed = self.dwconv(feat).flatten(2).transpose(1, 2)
        else:
            mixed = self.attn(self.norm(x))
        x = x + mixed
        return x + self.ffn(x)


class SVTRRecognizer(nn.Module):
    """Scene Text Recognition with a Single Visual Model."""

    def __init__(self, dim: int = 48, depth: int = 3, vocab: int = 38) -> None:
        """Initialize compact SVTR variant.

        Parameters
        ----------
        dim:
            Token width.
        depth:
            Number of local/global mixer blocks.
        vocab:
            Character vocabulary size.
        """

        super().__init__()
        self.patch = nn.Conv2d(1, dim, kernel_size=4, stride=4)
        self.blocks = nn.ModuleList([SVTRBlock(dim, local=(idx % 2 == 0)) for idx in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Recognize text using SVTR visual mixing.

        Parameters
        ----------
        x:
            Grayscale text-line image.

        Returns
        -------
        torch.Tensor
            Width-pooled character logits.
        """

        feat = self.patch(x)
        height, width = feat.shape[-2:]
        tokens = feat.flatten(2).transpose(1, 2)
        for block in self.blocks:
            tokens = block(tokens, height, width)
        grid = self.norm(tokens).view(x.size(0), height, width, -1)
        return self.head(grid.mean(dim=1))


def build_svtr_tiny() -> nn.Module:
    """Build compact SVTR-Tiny.

    Returns
    -------
    nn.Module
        Random-init SVTR-Tiny reconstruction.
    """

    return SVTRRecognizer(dim=32, depth=2)


def build_svtr_small() -> nn.Module:
    """Build compact SVTR-Small.

    Returns
    -------
    nn.Module
        Random-init SVTR-Small reconstruction.
    """

    return SVTRRecognizer(dim=40, depth=3)


def build_svtr_base() -> nn.Module:
    """Build compact SVTR-Base.

    Returns
    -------
    nn.Module
        Random-init SVTR-Base reconstruction.
    """

    return SVTRRecognizer(dim=48, depth=4)


class ViTSTR(nn.Module):
    """Single-stage ViT scene text recognizer."""

    def __init__(self, dim: int = 48, depth: int = 3, vocab: int = 38) -> None:
        """Initialize compact ViTSTR.

        Parameters
        ----------
        dim:
            Token width.
        depth:
            Transformer depth.
        vocab:
            Character vocabulary size.
        """

        super().__init__()
        self.patch = nn.Conv2d(1, dim, kernel_size=(4, 8), stride=(4, 8))
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.blocks = nn.ModuleList([TransformerBlock(dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict text tokens from a text-line image.

        Parameters
        ----------
        x:
            Grayscale text image.

        Returns
        -------
        torch.Tensor
            Per-token logits.
        """

        tokens = self.patch(x).flatten(2).transpose(1, 2)
        tokens = torch.cat((self.cls.expand(x.size(0), -1, -1), tokens), dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(self.norm(tokens[:, 1:]))


def build_vitstr_base() -> nn.Module:
    """Build compact ViTSTR-Base.

    Returns
    -------
    nn.Module
        Random-init ViTSTR reconstruction.
    """

    return ViTSTR()


class MobileSAMViTT(nn.Module):
    """MobileSAM with TinyViT-like encoder and SAM-style prompt decoder."""

    def __init__(self, dim: int = 48) -> None:
        """Initialize image encoder, prompt encoder, and mask decoder.

        Parameters
        ----------
        dim:
            Shared embedding width.
        """

        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(24, 24, 3, padding=1, groups=24),
            nn.Conv2d(24, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, stride=2, padding=1, groups=dim),
        )
        self.image_block = TransformerBlock(dim)
        self.prompt = nn.Linear(3, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.decoder = TransformerBlock(dim)
        self.mask_head = nn.ConvTranspose2d(dim, 1, 4, stride=4)
        self.iou_head = nn.Linear(dim, 1)

    def forward(
        self, inputs: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict masks from image and point prompts.

        Parameters
        ----------
        inputs:
            Image tensor and point prompts ``(x, y, label)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Low-resolution mask and IoU score.
        """

        image, points = inputs
        feat = self.stem(image)
        height, width = feat.shape[-2:]
        image_tokens = feat.flatten(2).transpose(1, 2)
        image_tokens = self.image_block(image_tokens)
        prompt_tokens = self.prompt(points)
        mask_token = self.mask_token.expand(image.size(0), -1, -1)
        decoded = self.decoder(torch.cat((mask_token, prompt_tokens, image_tokens), dim=1))
        mask_context = decoded[:, 0:1] + image_tokens
        mask_feat = mask_context.transpose(1, 2).reshape(image.size(0), -1, height, width)
        return self.mask_head(mask_feat), self.iou_head(decoded[:, 0])


def build_mobilesam_vit_t() -> nn.Module:
    """Build compact MobileSAM ViT-T.

    Returns
    -------
    nn.Module
        Random-init MobileSAM reconstruction.
    """

    return MobileSAMViTT()


def example_mobilesam_vit_t() -> tuple[torch.Tensor, torch.Tensor]:
    """Create MobileSAM image and point prompts.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Image and point tensors.
    """

    return torch.randn(1, 3, 32, 32), torch.tensor([[[0.25, 0.25, 1.0], [0.7, 0.6, 0.0]]])


class MixtureOfDepthsTransformer(nn.Module):
    """Transformer layer stack with top-k token depth routing."""

    def __init__(self, vocab: int = 64, dim: int = 48, depth: int = 4, capacity: int = 6) -> None:
        """Initialize MoD language model.

        Parameters
        ----------
        vocab:
            Vocabulary size.
        dim:
            Token width.
        depth:
            Number of routed blocks.
        capacity:
            Number of tokens processed by each block.
        """

        super().__init__()
        self.capacity = capacity
        self.embed = nn.Embedding(vocab, dim)
        self.router = nn.ModuleList([nn.Linear(dim, 1) for _ in range(depth)])
        self.blocks = nn.ModuleList([TransformerBlock(dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Apply capacity-limited routed Transformer blocks.

        Parameters
        ----------
        ids:
            Token ids.

        Returns
        -------
        torch.Tensor
            Language-model logits.
        """

        x = self.embed(ids)
        for router, block in zip(self.router, self.blocks, strict=True):
            scores = router(x).squeeze(-1)
            _, top_idx = torch.topk(scores, k=min(self.capacity, x.size(1)), dim=1)
            gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, x.size(-1))
            routed_tokens = torch.gather(x, 1, gather_idx)
            processed = block(routed_tokens)
            x = x.scatter(1, gather_idx, processed)
        return self.head(self.norm(x))


def build_mixtureofdepths_transformer() -> nn.Module:
    """Build compact Mixture-of-Depths Transformer.

    Returns
    -------
    nn.Module
        Random-init MoD model.
    """

    return MixtureOfDepthsTransformer()


def example_tokens() -> torch.Tensor:
    """Create a short token sequence.

    Returns
    -------
    torch.Tensor
        Token id tensor.
    """

    return torch.randint(0, 64, (1, 12))


class IRBlock(nn.Module):
    """InsightFace IR residual block."""

    def __init__(self, channels: int, stride: int = 1) -> None:
        """Initialize IR residual block.

        Parameters
        ----------
        channels:
            Channel count.
        stride:
            Spatial stride.
        """

        super().__init__()
        self.body = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, 3, stride=stride, padding=1, bias=False),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual IR block.

        Parameters
        ----------
        x:
            Face feature map.

        Returns
        -------
        torch.Tensor
            Updated feature map.
        """

        return x + self.body(x)


class MagFaceIR50(nn.Module):
    """IR50-style face encoder exposing MagFace feature magnitude quality."""

    def __init__(self, classes: int = 16, embedding: int = 64) -> None:
        """Initialize compact face-recognition network.

        Parameters
        ----------
        classes:
            Identity classifier size.
        embedding:
            Embedding dimension.
        """

        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.PReLU(32))
        self.blocks = nn.Sequential(
            IRBlock(32), IRBlock(32), nn.MaxPool2d(2), IRBlock(32), IRBlock(32)
        )
        self.embed = nn.Linear(32 * 16 * 16, embedding)
        self.classifier = nn.Linear(embedding, classes, bias=False)
        self.margin_scale = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return adaptive angular-margin logits, magnitude, and regularizer.

        Parameters
        ----------
        x:
            Face crop.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Margin logits, MagFace quality magnitude, and magnitude regularizer.
        """

        feat = self.blocks(self.stem(x)).flatten(1)
        emb = self.embed(feat)
        quality = emb.norm(dim=-1, keepdim=True)
        cosine = F.linear(
            F.normalize(emb, dim=-1), F.normalize(self.classifier.weight, dim=-1)
        ).clamp(-0.99, 0.99)
        clipped_quality = quality.clamp(10.0, 110.0)
        margin = 0.35 + 0.1 * torch.sigmoid(self.margin_scale((clipped_quality - 10.0) / 100.0))
        angular_margin = torch.cos(torch.acos(cosine) + margin)
        regularizer = 1.0 / clipped_quality + clipped_quality / 100.0
        return angular_margin * 32.0, quality, regularizer


def build_magface_ir50() -> nn.Module:
    """Build compact MagFace IR50.

    Returns
    -------
    nn.Module
        Random-init MagFace reconstruction.
    """

    return MagFaceIR50()


def example_face() -> torch.Tensor:
    """Create a compact aligned face crop.

    Returns
    -------
    torch.Tensor
        Image tensor.
    """

    return torch.randn(1, 3, 32, 32)


class MessagePassingNet(nn.Module):
    """Edge/node/global message-passing graph network."""

    def __init__(self, equivariant: bool = False, dim: int = 32) -> None:
        """Initialize graph network.

        Parameters
        ----------
        equivariant:
            Whether to compute NequIP-style radial geometric edge features.
        dim:
            Hidden width.
        """

        super().__init__()
        self.equivariant = equivariant
        edge_in = 4 if equivariant else 6
        self.node_in = nn.Linear(5, dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(dim * 2 + edge_in, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.node_mlp = nn.Sequential(nn.Linear(dim * 2, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.out = nn.Linear(dim, 3 if equivariant else 2)

    def forward(
        self, graph: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Run message passing over a small graph.

        Parameters
        ----------
        graph:
            Tuple of node, edge, sender, and receiver tensors.

        Returns
        -------
        torch.Tensor
            Node-level outputs.
        """

        node_features, edge_features, senders, receivers = graph
        node = self.node_in(node_features)
        edge_attr = edge_features
        if self.equivariant:
            radius = edge_attr.norm(dim=-1, keepdim=True)
            edge_attr = torch.cat(
                (radius, torch.sin(radius), torch.cos(radius), radius.square()), dim=-1
            )
        for _ in range(2):
            msg_in = torch.cat((node[senders], node[receivers], edge_attr), dim=-1)
            msg = self.edge_mlp(msg_in)
            agg = node.new_zeros(node.shape).index_add(0, receivers, msg)
            node = node + self.node_mlp(torch.cat((node, agg), dim=-1))
        return self.out(node)


def _example_graph(edge_dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a small directed graph.

    Parameters
    ----------
    edge_dim:
        Edge feature width.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Graph input tensors.
    """

    senders = torch.tensor([0, 1, 1, 2, 2, 3, 3, 0], dtype=torch.long)
    receivers = torch.tensor([1, 0, 2, 1, 3, 2, 0, 3], dtype=torch.long)
    return torch.randn(4, 5), torch.randn(8, edge_dim), senders, receivers


def build_meshgraphnet() -> nn.Module:
    """Build compact MeshGraphNet.

    Returns
    -------
    nn.Module
        Random-init message-passing simulator.
    """

    return MessagePassingNet(equivariant=False)


class NequIPEquivariantPotential(nn.Module):
    """Compact NequIP with scalar/vector irreps and equivariant tensor products."""

    def __init__(self, dim: int = 32) -> None:
        """Initialize scalar, vector, radial, and ZBL pair-potential paths.

        Parameters
        ----------
        dim:
            Scalar irrep width.
        """

        super().__init__()
        self.node_in = nn.Linear(5, dim)
        self.radial = nn.Sequential(nn.Linear(4, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.scalar_update = nn.Linear(dim * 2, dim)
        self.vector_gate = nn.Linear(dim, 1)
        self.energy = nn.Linear(dim, 1)

    def forward(
        self, graph: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run E(3)-equivariant message passing and ZBL screened repulsion.

        Parameters
        ----------
        graph:
            Node features, relative edge vectors, senders, and receivers.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Total energy and equivariant vector features.
        """

        node_features, rel_vec, senders, receivers = graph
        dist = rel_vec.norm(dim=-1, keepdim=True).clamp_min(1e-4)
        unit = rel_vec / dist
        radial = self.radial(
            torch.cat((dist, torch.sin(dist), torch.cos(dist), dist.square()), dim=-1)
        )
        scalar = self.node_in(node_features)
        vector = rel_vec.new_zeros(scalar.shape[0], scalar.shape[1], 3)
        for _ in range(2):
            tensor_product = scalar[senders].unsqueeze(-1) * unit.unsqueeze(1)
            vector_msg = self.vector_gate(radial).unsqueeze(-1) * tensor_product
            tensor_scalar = (vector[senders] * unit.unsqueeze(1)).sum(dim=-1)
            msg = F.silu(
                self.scalar_update(torch.cat((scalar[senders], radial + tensor_scalar), dim=-1))
            )
            scalar = scalar + torch.zeros_like(scalar).index_add(0, receivers, msg)
            vector = vector + torch.zeros_like(vector).index_add(0, receivers, vector_msg)
        zbl_pair = torch.exp(-dist.squeeze(-1)) / dist.squeeze(-1)
        return self.energy(scalar).sum() + 0.01 * zbl_pair.sum(), vector.sum(dim=1)


def example_meshgraphnet() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create MeshGraphNet example graph.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Graph tensors.
    """

    return _example_graph(6)


def build_nequip() -> nn.Module:
    """Build compact NequIP-style equivariant graph potential.

    Returns
    -------
    nn.Module
        Random-init radial message-passing model.
    """

    return NequIPEquivariantPotential()


def example_nequip() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create NequIP example atomic graph.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Atomic graph tensors.
    """

    return _example_graph(3)


class MoonshineASR(nn.Module):
    """Moonshine-like duration-flexible encoder-decoder ASR model."""

    def __init__(self, dim: int = 48, depth: int = 3, vocab: int = 64) -> None:
        """Initialize compact Moonshine.

        Parameters
        ----------
        dim:
            Hidden width.
        depth:
            Encoder/decoder Transformer depth.
        vocab:
            Text vocabulary size.
        """

        super().__init__()
        self.audio = nn.Sequential(
            nn.Conv1d(80, dim, 3, stride=2, padding=1), nn.GELU(), nn.Conv1d(dim, dim, 3, padding=1)
        )
        self.encoder = nn.ModuleList([TransformerBlock(dim) for _ in range(depth)])
        self.embed = nn.Embedding(vocab, dim)
        self.decoder = nn.ModuleList([TransformerBlock(dim) for _ in range(depth)])
        self.cross = SmallSelfAttention(dim)
        self.head = nn.Linear(dim, vocab)

    def _rope(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embedding to variable-length tokens.

        Parameters
        ----------
        x:
            Token tensor.

        Returns
        -------
        torch.Tensor
            RoPE-rotated token tensor.
        """

        half = x.shape[-1] // 2
        pos = torch.arange(x.shape[1], device=x.device, dtype=x.dtype).unsqueeze(-1)
        freq = torch.arange(half, device=x.device, dtype=x.dtype).unsqueeze(0) / max(half, 1)
        angle = pos / (10000.0**freq)
        first = x[..., :half]
        second = x[..., half : half * 2]
        rotated = torch.cat(
            (
                first * torch.cos(angle) - second * torch.sin(angle),
                first * torch.sin(angle) + second * torch.cos(angle),
            ),
            dim=-1,
        )
        if x.shape[-1] > half * 2:
            rotated = torch.cat((rotated, x[..., half * 2 :]), dim=-1)
        return rotated

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Transcribe mel frames conditioned on prefix tokens.

        Parameters
        ----------
        inputs:
            Mel spectrogram ``(batch, 80, time)`` and token ids.

        Returns
        -------
        torch.Tensor
            Decoder logits.
        """

        mel, ids = inputs
        enc = self.audio(mel).transpose(1, 2)
        for block in self.encoder:
            enc = block(self._rope(enc))
        dec = self._rope(self.embed(ids))
        for block in self.decoder:
            dec = block(self._rope(dec))
            dec = dec + self.cross(torch.cat((dec, enc), dim=1))[:, : dec.size(1)]
        return self.head(dec)


def build_moonshine_tiny_native() -> nn.Module:
    """Build compact Moonshine tiny.

    Returns
    -------
    nn.Module
        Random-init ASR model.
    """

    return MoonshineASR(dim=32, depth=2)


def build_moonshine_base_native() -> nn.Module:
    """Build compact Moonshine base.

    Returns
    -------
    nn.Module
        Random-init ASR model.
    """

    return MoonshineASR(dim=48, depth=3)


def example_moonshine() -> tuple[torch.Tensor, torch.Tensor]:
    """Create Moonshine mel frames and decoder prefix.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Mel and token tensors.
    """

    return torch.randn(1, 80, 24), torch.randint(0, 64, (1, 8))


class MorpheusAstro(nn.Module):
    """Pixel-level astronomical morphology segmentation model."""

    def __init__(self, classes: int = 5) -> None:
        """Initialize compact U-Net-style Morpheus model.

        Parameters
        ----------
        classes:
            Pixel morphology classes.
        """

        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(5, 24, 3, padding=1), nn.ReLU(), nn.Conv2d(24, 24, 3, padding=1), nn.ReLU()
        )
        self.down = nn.Conv2d(24, 40, 3, stride=2, padding=1)
        self.mid = nn.Sequential(nn.ReLU(), nn.Conv2d(40, 40, 3, padding=1), nn.ReLU())
        self.up = nn.ConvTranspose2d(40, 24, 2, stride=2)
        self.seg = nn.Conv2d(48, classes, 1)
        self.detect = nn.Conv2d(48, 1, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Segment source morphology and objectness.

        Parameters
        ----------
        x:
            Multi-band astronomy image.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Morphology logits and source mask logits.
        """

        skip = self.enc1(x)
        up = self.up(self.mid(self.down(skip)))
        feat = torch.cat((skip, up), dim=1)
        return self.seg(feat), self.detect(feat)


def build_morpheus_astro() -> nn.Module:
    """Build compact Morpheus astronomy model.

    Returns
    -------
    nn.Module
        Random-init pixel classifier.
    """

    return MorpheusAstro()


def example_morpheus_astro() -> torch.Tensor:
    """Create a small five-band astronomy tile.

    Returns
    -------
    torch.Tensor
        Image tensor.
    """

    return torch.randn(1, 5, 32, 32)


class DeepSEABeluga(nn.Module):
    """Beluga-style deeper DeepSEA regulatory genomics CNN."""

    def __init__(self, outputs: int = 32) -> None:
        """Initialize convolutional regulatory sequence predictor.

        Parameters
        ----------
        outputs:
            Number of genomic feature outputs.
        """

        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(4, 64, 8),
            nn.ReLU(),
            nn.Conv1d(64, 64, 8),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 96, 8),
            nn.ReLU(),
            nn.Conv1d(96, 96, 8),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        self.head = nn.Sequential(
            nn.Linear(96 * 11, 128), nn.ReLU(), nn.Linear(128, outputs), nn.Sigmoid()
        )

    def forward(self, onehot: torch.Tensor) -> torch.Tensor:
        """Predict chromatin features from one-hot DNA.

        Parameters
        ----------
        onehot:
            One-hot sequence tensor.

        Returns
        -------
        torch.Tensor
            Multi-label probabilities.
        """

        return self.head(self.features(onehot).flatten(1))


def build_deepsea_beluga() -> nn.Module:
    """Build compact DeepSEA-Beluga.

    Returns
    -------
    nn.Module
        Random-init Beluga-style model.
    """

    return DeepSEABeluga()


def example_deepsea_beluga() -> torch.Tensor:
    """Create one-hot DNA sequence input.

    Returns
    -------
    torch.Tensor
        One-hot DNA tensor.
    """

    ids = torch.randint(0, 4, (1, 256))
    return F.one_hot(ids, num_classes=4).float().transpose(1, 2)


MENAGERIE_ENTRIES = [
    ("SDMGR-NodeEdge", "build_sdmgr_nodeedge", "example_sdmgr_nodeedge", "2021", "E7"),
    ("SegOCR-ResNet31", "build_segocr_resnet31", "example_text_image", "2021", "E7"),
    ("SVTR-Base-MMOCR", "build_svtr_base", "example_text_image", "2022", "E7"),
    ("SVTR-Small-MMOCR", "build_svtr_small", "example_text_image", "2022", "E7"),
    ("SVTR-Tiny", "build_svtr_tiny", "example_text_image", "2022", "E7"),
    ("ViTSTR-Base", "build_vitstr_base", "example_text_image", "2021", "E7"),
    ("MobileSAM_ViT_T", "build_mobilesam_vit_t", "example_mobilesam_vit_t", "2023", "E7"),
    (
        "MixtureOfDepths-Transformer",
        "build_mixtureofdepths_transformer",
        "example_tokens",
        "2024",
        "E7",
    ),
    ("magface_ir50", "build_magface_ir50", "example_face", "2021", "E7"),
    ("MeshGraphNet", "build_meshgraphnet", "example_meshgraphnet", "2020", "E7"),
    ("moonshine_base_native", "build_moonshine_base_native", "example_moonshine", "2024", "E7"),
    ("moonshine_tiny_native", "build_moonshine_tiny_native", "example_moonshine", "2024", "E7"),
    ("Morpheus-astro", "build_morpheus_astro", "example_morpheus_astro", "2019", "E7"),
    ("DeepSEA-Beluga", "build_deepsea_beluga", "example_deepsea_beluga", "2018", "E5"),
    ("nequip", "build_nequip", "example_nequip", "2021", "E7"),
    ("NequIP small", "build_nequip", "example_nequip", "2021", "E7"),
    ("nequip_InteractionBlock_R6", "build_nequip", "example_nequip", "2021", "E7"),
    ("nequip_NequIPGNNModel", "build_nequip", "example_nequip", "2021", "E7"),
    ("nequip_PresetNequIPGNNModel", "build_nequip", "example_nequip", "2021", "E7"),
    ("nequip_FullNequIPGNNModel", "build_nequip", "example_nequip", "2021", "E7"),
    ("nequip_ZBLPairPotential", "build_nequip", "example_nequip", "2021", "E7"),
]
