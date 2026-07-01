# SOURCE: vendored from yael-vinker/SceneSketch @ master
# https://github.com/yael-vinker/SceneSketch
# https://raw.githubusercontent.com/yael-vinker/SceneSketch/master/models/painter_params.py
#
# Vinker et al. 2023 (ICCV) "CLIPascene: Scene Sketching with Different
# Types and Levels of Abstraction" -- the official repo (`models/
# painter_params.py`) optimizes a set of Bezier-curve strokes to sketch a
# scene under CLIP-similarity + geometric losses (differentiable rendering
# via `pydiffvg`, driving the main `Painter`/`PainterOptimizer` classes).
# Within that pipeline, `MLP` and `WidthMLP` are the two dedicated
# refinement networks: `MLP` maps a flattened vector of stroke control-point
# coordinates through two SELU-activated hidden linear layers back to a
# same-shaped delta, which is added (scaled by 0.1) to the input points to
# predict refined stroke placements (`Painter`'s `mlp_points`, driven by the
# CLIP-based losses at optimization time; see `Painter.__init__`'s
# `self.mlp = MLP(...)` and the `x.flatten() + 0.1 * deltas` residual
# update); `WidthMLP` performs the analogous refinement for per-stroke
# width scalars, ending in a `Sigmoid` to keep widths in [0, 1]. Both are
# pure `torch.nn` MLPs with no dependency on `pydiffvg`/CLIP internally --
# they only consume/produce plain tensors of stroke parameters, which is
# why they can be constructed and traced standalone without the
# differentiable-vector-graphics rendering stack (`pydiffvg`, a
# custom-CUDA-compiled package that is not installed in this base
# environment and is not needed to exercise these two architecture pieces).
#
# `MLP`, `WidthMLP` are copied VERBATIM (architecture completely unchanged,
# including the commented-out `width_optim` branch left as-is in the
# original source) from `models/painter_params.py` above. No import-path or
# other changes were needed; both classes only ever import `torch`/
# `torch.nn` at module scope (the file-level `CLIP_`/`pydiffvg`/
# `sketch_utils` imports at the top of the original file are only used by
# the `Painter`/`PainterOptimizer`/`Hook`/`XDoG_` classes, which are not
# vendored here).

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, num_strokes, num_cp, width_optim=False):
        super().__init__()
        outdim = 1000
        self.width_optim = width_optim
        self.layers_points = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_strokes * num_cp * 2, outdim),
            nn.SELU(inplace=True),
            nn.Linear(outdim, outdim),
            nn.SELU(inplace=True),
            # nn.ReLU(),
            # nn.Linear(1000, 1000),
            # nn.ReLU(),
            nn.Linear(outdim, num_strokes * num_cp * 2),
            # nn.Tanh()
        )

        # if width_optim:
        #     self.layers_width = nn.Sequential(
        #         nn.Linear(num_strokes, num_strokes),
        #         nn.ReLU(),
        #         nn.Linear(num_strokes, num_strokes),
        #         nn.ReLU(),
        #         nn.Linear(num_strokes, num_strokes),
        #         nn.Sigmoid()
        #     )

    def forward(self, x, widths=None):
        """Forward pass"""
        deltas = self.layers_points(x)
        # if self.width_optim:
        #     return x.flatten() + 0.1 * deltas, self.layers_width(widths)
        return x.flatten() + 0.1 * deltas


class WidthMLP(nn.Module):
    def __init__(self, num_strokes, num_cp, width_optim=False):
        super().__init__()
        outdim = 1000
        self.width_optim = width_optim

        self.layers_width = nn.Sequential(
            nn.Linear(num_strokes, outdim),
            nn.SELU(inplace=True),
            nn.Linear(outdim, outdim),
            nn.SELU(inplace=True),
            nn.Linear(outdim, num_strokes),
            nn.Sigmoid(),
        )

    def forward(self, widths=None):
        """Forward pass"""
        return self.layers_width(widths)


def build_stroke_mlp():
    return MLP(num_strokes=8, num_cp=4).eval()


def example_input_stroke_mlp():
    num_strokes = 8
    num_cp = 4
    x = torch.rand(1, num_strokes, num_cp, 2)
    return (x,)


def build_stroke_width_mlp():
    return WidthMLP(num_strokes=8, num_cp=4).eval()


def example_input_stroke_width_mlp():
    widths = torch.rand(1, 8)
    return (widths,)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("CLIPascene-StrokeMLP", "build_stroke_mlp", "example_input_stroke_mlp", 2023, "vendored"),
    (
        "CLIPascene-WidthMLP",
        "build_stroke_width_mlp",
        "example_input_stroke_width_mlp",
        2023,
        "vendored",
    ),
]
