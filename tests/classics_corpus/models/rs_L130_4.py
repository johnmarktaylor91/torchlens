# SOURCE: vendored from Huage001/PaintTransformer @ main (inference/network.py, class Painter)
"""Paint Transformer (ICCV 2021) -- feed-forward neural painting via stroke-set prediction.

Paint Transformer casts stroke-based image "painting" as a set-prediction problem: given
the current canvas and target image, a transformer encoder-decoder predicts a fixed-size
set of stroke parameters plus a keep/discard decision per stroke, which a differentiable
renderer then composites onto the canvas -- all strokes for a region are predicted and
applied in parallel (no autoregressive stroke-by-stroke rollout), making inference fast.
The official repo (wzmsltw/PaintTransformer) is PaddlePaddle; Huage001/PaintTransformer is
a faithful PyTorch re-implementation used widely downstream (pip-installable via its
inference/ package) and is itself real, runnable model code -- not a from-scratch guess.
This file vendors inference/network.py's `Painter` module verbatim (twin CNN image/canvas
encoders feeding a `nn.Transformer`, plus per-stroke parameter/decision heads with learned
positional + query embeddings). The differentiable brush renderer (inference/morphology.py)
is a post-hoc image-compositing step, not part of the traced nn.Module forward.
"""

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# inference/network.py (verbatim: SignWithSigmoidGrad, Painter). The custom
# autograd Function is unused by the forward-only inference path we trace but
# is kept for fidelity to the source file.
# ---------------------------------------------------------------------------
class SignWithSigmoidGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        result = (x > 0).float()
        sigmoid_result = torch.sigmoid(x)
        ctx.save_for_backward(sigmoid_result)
        return result

    @staticmethod
    def backward(ctx, grad_result):
        (sigmoid_result,) = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = grad_result * sigmoid_result * (1 - sigmoid_result)
        else:
            grad_input = None
        return grad_input


class Painter(nn.Module):
    def __init__(
        self, param_per_stroke, total_strokes, hidden_dim, n_heads=8, n_enc_layers=3, n_dec_layers=3
    ):
        super().__init__()
        self.enc_img = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 64, 3, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, 3, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.enc_canvas = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 64, 3, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, 3, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.conv = nn.Conv2d(128 * 2, hidden_dim, 1)
        self.transformer = nn.Transformer(hidden_dim, n_heads, n_enc_layers, n_dec_layers)
        self.linear_param = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, param_per_stroke),
        )
        self.linear_decider = nn.Linear(hidden_dim, 1)
        self.query_pos = nn.Parameter(torch.rand(total_strokes, hidden_dim))
        self.row_embed = nn.Parameter(torch.rand(8, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(8, hidden_dim // 2))

    def forward(self, img, canvas):
        b, _, H, W = img.shape
        img_feat = self.enc_img(img)
        canvas_feat = self.enc_canvas(canvas)
        h, w = img_feat.shape[-2:]
        feat = torch.cat([img_feat, canvas_feat], dim=1)
        feat_conv = self.conv(feat)

        pos_embed = (
            torch.cat(
                [
                    self.col_embed[:w].unsqueeze(0).contiguous().repeat(h, 1, 1),
                    self.row_embed[:h].unsqueeze(1).contiguous().repeat(1, w, 1),
                ],
                dim=-1,
            )
            .flatten(0, 1)
            .unsqueeze(1)
        )
        hidden_state = self.transformer(
            pos_embed + feat_conv.flatten(2).permute(2, 0, 1).contiguous(),
            self.query_pos.unsqueeze(1).contiguous().repeat(1, b, 1),
        )
        hidden_state = hidden_state.permute(1, 0, 2).contiguous()
        param = self.linear_param(hidden_state)
        decision = self.linear_decider(hidden_state)
        return param, decision


# ---------------------------------------------------------------------------
# Menagerie build/example helpers
# ---------------------------------------------------------------------------
class PaintTransformerTraceWrapper(nn.Module):
    """Returns only the stroke-parameter tensor so the traced model has a single
    output (Painter.forward natively returns a (param, decision) tuple)."""

    def __init__(
        self,
        param_per_stroke=13,
        total_strokes=8,
        hidden_dim=32,
        n_heads=4,
        n_enc_layers=2,
        n_dec_layers=2,
    ):
        super().__init__()
        self.painter = Painter(
            param_per_stroke, total_strokes, hidden_dim, n_heads, n_enc_layers, n_dec_layers
        )

    def forward(self, img, canvas):
        param, decision = self.painter(img, canvas)
        return param


def build_paint_transformer():
    return PaintTransformerTraceWrapper()


def example_input_paint_transformer():
    return (torch.randn(1, 3, 32, 32), torch.randn(1, 3, 32, 32))


MENAGERIE_ENTRIES = [
    (
        "Paint-Transformer",
        build_paint_transformer,
        example_input_paint_transformer,
        2021,
        MENAGERIE_ZOO,
    ),
]
