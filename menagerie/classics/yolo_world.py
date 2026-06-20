"""YOLO-World: real-time open-vocabulary object detection.

Cheng et al. 2024, arXiv:2401.17270 (and YOLO-World-v2).  Source: AILab-CVC/YOLO-World.

YOLO-World fuses a YOLOv8 CSPDarknet image backbone with frozen CLIP text
embeddings via the **RepVL-PAN** neck, then detects with a text-contrastive head.
The distinctive, load-bearing pieces reproduced here:

  - **Max-Sigmoid text-guided attention** (T-CSPLayer): image features are gated by
    the *max over text classes* of an image-text dot product, passed through a
    sigmoid (NOT softmax) -- this is the defining open-vocabulary fusion op.
  - **Image-Pooling Attention**: the text embeddings are updated from pooled
    multi-scale image features (standard multi-head cross-attention, text=query).
  - **Contrastive detection head**: classification is replaced by a normalized
    image-embedding x text-embedding dot product with a learnable logit scale and
    bias.  v1 uses an L2-norm on the image embed; v2 uses a BatchNorm instead.

The 10 catalog rows (v1/v2 x S/M/L/X/XL) collapse to this one parametric module:
sizes are YOLOv8 width/depth multipliers; v1/v2 differ only in the contrastive
head normalization.  CLIP is frozen + external, so feeding random text embeddings
is faithful.  Random init, CPU, forward-only; reduced widths for a compact graph.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv(in_ch: int, out_ch: int, k: int = 1, s: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, stride=s, padding=k // 2, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.SiLU(inplace=True),
    )


class DarknetBottleneck(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.cv1 = _conv(ch, ch, 3)
        self.cv2 = _conv(ch, ch, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x))


class C2f(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n: int = 1) -> None:
        super().__init__()
        self.mid = out_ch // 2
        self.cv1 = _conv(in_ch, 2 * self.mid, 1)
        self.blocks = nn.ModuleList([DarknetBottleneck(self.mid) for _ in range(n)])
        self.cv2 = _conv((2 + n) * self.mid, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))
        for b in self.blocks:
            y.append(b(y[-1]))
        return self.cv2(torch.cat(y, dim=1))


class SPPF(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        mid = in_ch // 2
        self.cv1 = _conv(in_ch, mid, 1)
        self.pool = nn.MaxPool2d(5, stride=1, padding=2)
        self.cv2 = _conv(4 * mid, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class CSPDarknet(nn.Module):
    """YOLOv8 backbone -> P3/P4/P5."""

    def __init__(self, w: int = 32) -> None:
        super().__init__()
        self.stem = _conv(3, w, 3, 2)  # stride 2
        self.d1 = _conv(w, 2 * w, 3, 2)  # stride 4
        self.c1 = C2f(2 * w, 2 * w, 1)
        self.d2 = _conv(2 * w, 4 * w, 3, 2)  # stride 8 -> P3
        self.c2 = C2f(4 * w, 4 * w, 2)
        self.d3 = _conv(4 * w, 8 * w, 3, 2)  # stride 16 -> P4
        self.c3 = C2f(8 * w, 8 * w, 2)
        self.d4 = _conv(8 * w, 16 * w, 3, 2)  # stride 32
        self.c4 = C2f(16 * w, 16 * w, 1)
        self.sppf = SPPF(16 * w, 16 * w)  # -> P5

    def forward(self, x: torch.Tensor):
        x = self.c1(self.d1(self.stem(x)))
        p3 = self.c2(self.d2(x))
        p4 = self.c3(self.d3(p3))
        p5 = self.sppf(self.c4(self.d4(p4)))
        return [p3, p4, p5]


class MaxSigmoidAttn(nn.Module):
    """Text-guided max-sigmoid attention: gate image features by max-over-class sim."""

    def __init__(self, in_ch: int, embed_ch: int, text_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.m = num_heads
        self.c = embed_ch // num_heads
        self.guide_fc = nn.Linear(text_dim, embed_ch)
        self.embed_conv = nn.Sequential(
            nn.Conv2d(in_ch, embed_ch, 1, bias=False), nn.BatchNorm2d(embed_ch)
        )
        self.bias = nn.Parameter(torch.zeros(num_heads))
        self.project = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False), nn.BatchNorm2d(in_ch)
        )

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        T = guide.shape[1]
        g = self.guide_fc(guide).view(B, T, self.m, self.c)  # (B,T,m,c)
        e = self.embed_conv(x).view(B, self.m, self.c, H, W)  # (B,m,c,H,W)
        aw = torch.einsum("bmchw,bnmc->bmhwn", e, g)  # (B,m,H,W,T)
        aw = aw.max(dim=-1).values  # MAX over T classes
        aw = aw / (self.c**0.5) + self.bias.view(1, self.m, 1, 1)
        aw = aw.sigmoid()  # SIGMOID gate (not softmax)
        out_ch = x.shape[1]
        px = self.project(x).view(B, self.m, out_ch // self.m, H, W)
        return (px * aw.unsqueeze(2)).reshape(B, out_ch, H, W)


class TextGuidedCSP(nn.Module):
    """C2f whose last appended chunk is a max-sigmoid text-attention block."""

    def __init__(
        self, in_ch: int, out_ch: int, text_dim: int, n: int = 1, num_heads: int = 4
    ) -> None:
        super().__init__()
        self.mid = out_ch // 2
        self.cv1 = _conv(in_ch, 2 * self.mid, 1)
        self.blocks = nn.ModuleList([DarknetBottleneck(self.mid) for _ in range(n)])
        self.attn = MaxSigmoidAttn(self.mid, self.mid, text_dim, num_heads)
        self.cv2 = _conv((3 + n) * self.mid, out_ch, 1)

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))
        for b in self.blocks:
            y.append(b(y[-1]))
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, dim=1))


class ImagePoolingAttn(nn.Module):
    """Update text embeddings from pooled multi-scale image features."""

    def __init__(self, channels, text_dim: int, embed: int = 128, num_heads: int = 4) -> None:
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv2d(c, embed, 1) for c in channels])
        self.pools = nn.ModuleList([nn.AdaptiveMaxPool2d((3, 3)) for _ in channels])
        self.query = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, embed))
        self.key = nn.Sequential(nn.LayerNorm(embed), nn.Linear(embed, embed))
        self.value = nn.Sequential(nn.LayerNorm(embed), nn.Linear(embed, embed))
        self.proj = nn.Linear(embed, text_dim)
        self.attn = nn.MultiheadAttention(embed, num_heads, batch_first=True)
        self.embed = embed

    def forward(self, text: torch.Tensor, feats) -> torch.Tensor:
        toks = []
        for proj, pool, f in zip(self.projections, self.pools, feats):
            toks.append(pool(proj(f)).flatten(2).transpose(1, 2))  # (B,9,embed)
        toks = torch.cat(toks, dim=1)  # (B, 27, embed)
        q = self.query(text)
        k = self.key(toks)
        v = self.value(toks)
        out, _ = self.attn(q, k, v)
        return text + self.proj(out)


class BNContrastiveHead(nn.Module):
    """v2 contrastive head: BatchNorm image embed, L2 text, dot product + scale/bias."""

    def __init__(self, embed: int, v2: bool = True) -> None:
        super().__init__()
        self.v2 = v2
        if v2:
            self.norm = nn.BatchNorm2d(embed)
        self.bias = nn.Parameter(torch.zeros(1))
        self.logit_scale = nn.Parameter(torch.tensor(-1.0 if v2 else 2.659))

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if self.v2:
            x = self.norm(x)
        else:
            x = F.normalize(x, dim=1)
        w = F.normalize(w, dim=-1)
        sim = torch.einsum("bchw,bkc->bkhw", x, w)
        return sim * self.logit_scale.exp() + self.bias


class YOLOWorld(nn.Module):
    def __init__(
        self, w: int = 24, text_dim: int = 512, num_heads: int = 4, v2: bool = True
    ) -> None:
        super().__init__()
        self.backbone = CSPDarknet(w)
        ch = [4 * w, 8 * w, 16 * w]  # P3/P4/P5 channels
        # RepVL-PAN neck (top-down + image-pooling text update + bottom-up)
        self.td1 = TextGuidedCSP(ch[2] + ch[1], ch[1], text_dim, 1, num_heads)
        self.td0 = TextGuidedCSP(ch[1] + ch[0], ch[0], text_dim, 1, num_heads)
        self.img_pool = ImagePoolingAttn(ch, text_dim)
        self.down0 = _conv(ch[0], ch[0], 3, 2)
        self.bu1 = TextGuidedCSP(ch[0] + ch[1], ch[1], text_dim, 1, num_heads)
        self.down1 = _conv(ch[1], ch[1], 3, 2)
        self.bu2 = TextGuidedCSP(ch[1] + ch[2], ch[2], text_dim, 1, num_heads)
        # contrastive detection head per level
        embed = 128
        self.cls_embed = nn.ModuleList(
            [nn.Sequential(_conv(c, c, 3), nn.Conv2d(c, embed, 1)) for c in ch]
        )
        self.reg = nn.ModuleList(
            [nn.Sequential(_conv(c, c, 3), nn.Conv2d(c, 4 * 16, 1)) for c in ch]
        )
        self.contrast = nn.ModuleList([BNContrastiveHead(embed, v2) for _ in ch])
        self.text_proj = nn.Linear(text_dim, embed)

    def forward(self, image: torch.Tensor, text: torch.Tensor):
        p3, p4, p5 = self.backbone(image)
        # top-down
        up5 = F.interpolate(p5, size=p4.shape[2:], mode="nearest")
        n4 = self.td1(torch.cat([up5, p4], dim=1), text)
        up4 = F.interpolate(n4, size=p3.shape[2:], mode="nearest")
        n3 = self.td0(torch.cat([up4, p3], dim=1), text)
        # image-pooling attention updates text
        text = self.img_pool(text, [n3, n4, p5])
        # bottom-up
        d3 = self.down0(n3)
        m4 = self.bu1(torch.cat([d3, n4], dim=1), text)
        d4 = self.down1(m4)
        m5 = self.bu2(torch.cat([d4, p5], dim=1), text)
        feats = [n3, m4, m5]
        text_embed = self.text_proj(text)
        outs = []
        for f, cls_e, reg, con in zip(feats, self.cls_embed, self.reg, self.contrast):
            ce = cls_e(f)
            cls_logit = con(ce, text_embed)
            box = reg(f)
            outs.append((cls_logit, box))
        return outs


def build_yolo_world_v2_s() -> nn.Module:
    return YOLOWorld(w=24, num_heads=4, v2=True)


def build_yolo_world_v2_m() -> nn.Module:
    return YOLOWorld(w=32, num_heads=4, v2=True)


def build_yolo_world_v2_l() -> nn.Module:
    return YOLOWorld(w=40, num_heads=8, v2=True)


def build_yolo_world_v1_l() -> nn.Module:
    return YOLOWorld(w=40, num_heads=8, v2=False)


def example_input():
    """(image ``(1,3,256,256)``, text embeddings ``(1,6,512)``)."""
    return (torch.randn(1, 3, 256, 256), torch.randn(1, 6, 512))


MENAGERIE_ENTRIES = [
    (
        "YOLO-World-v2-S (open-vocab detection, RepVL-PAN max-sigmoid text gate)",
        "build_yolo_world_v2_s",
        "example_input",
        "2024",
        "DC",
    ),
    (
        "YOLO-World-v2-M (open-vocab detection, RepVL-PAN)",
        "build_yolo_world_v2_m",
        "example_input",
        "2024",
        "DC",
    ),
    (
        "YOLO-World-v2-L (open-vocab detection, RepVL-PAN)",
        "build_yolo_world_v2_l",
        "example_input",
        "2024",
        "DC",
    ),
    (
        "YOLO-World-v1-L (open-vocab detection, L2-norm contrastive head)",
        "build_yolo_world_v1_l",
        "example_input",
        "2024",
        "DC",
    ),
]
