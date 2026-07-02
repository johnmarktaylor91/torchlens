# SOURCE: real class from diffusers (UNet2DConditionModel), configured per
# https://github.com/Nota-NetsPresso/BK-SDM @ main
#
# BK-SDM (Kim, Lee, Ham, Han. 2023, ECCV'24, "BK-SDM: A Lightweight, Fast, and Cheap
# Version of Stable Diffusion") block-removes redundant residual/attention blocks from
# the Stable Diffusion v1.4/v1.5 UNet and knowledge-distills a compact student. The
# repo's own configs (`src/unet_config/bk_base/config.json`,
# `src/unet_config/bk_small/config.json`, `src/unet_config/bk_tiny/config.json`)
# confirm the student is built with `_class_name: "UNet2DConditionModel"` -- the real,
# unmodified diffusers class -- via a shrunk `down_block_types`/`up_block_types`
# (bk_tiny drops an entire down/up stage) and `layers_per_block=1` (vs the standard
# SD v1.x `layers_per_block=2`). The contribution is architecture-space block removal
# + feature/output-level knowledge distillation from the SD teacher, not a new module;
# construction here uses the actual `diffusers.UNet2DConditionModel` class with a tiny
# instance of the bk_base block-type/layers_per_block signature (channel widths and
# attention/cross-attention dims shrunk only for fast tracing).
#
# Needs 3 required forward inputs (sample, timestep, encoder_hidden_states), so this is
# staged as a module (real-library-model path) rather than a single-tensor recipe row.
#
# MENAGERIE_ZOO = "vendored-pytorch"

from __future__ import annotations

import torch

from diffusers import UNet2DConditionModel

MENAGERIE_ZOO = "vendored-pytorch"


def build_bk_sdm():
    # Real, unmodified diffusers.UNet2DConditionModel. Block types + layers_per_block=1
    # mirror BK-SDM's bk_base config.json exactly; channel widths / cross_attention_dim /
    # attention_head_dim are shrunk from the repo's real 320/640/1280/1280 + 768 + 8 only
    # for fast tracing at random init -- the block-removal topology (the actual BK-SDM
    # contribution) is preserved unmodified.
    return UNet2DConditionModel(
        sample_size=8,
        in_channels=4,
        out_channels=4,
        down_block_types=(
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        block_out_channels=(8, 16, 32, 32),
        layers_per_block=1,
        cross_attention_dim=16,
        attention_head_dim=4,
        norm_num_groups=4,
    ).eval()


def example_input_bk_sdm():
    sample = torch.randn(1, 4, 8, 8)
    timestep = torch.tensor(1)
    encoder_hidden_states = torch.randn(1, 4, 16)
    return (sample, timestep, encoder_hidden_states)


MENAGERIE_ENTRIES = [
    ("BK-SDM", "build_bk_sdm", "example_input_bk_sdm", 2023, "vendored-pytorch"),
]
