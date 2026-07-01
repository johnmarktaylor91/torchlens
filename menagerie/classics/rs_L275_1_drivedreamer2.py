# SOURCE: vendored from f1yfisher/DriveDreamer2 @ main
# (dreamer-models/dreamer_models/pipelines/drivedreamer2/pipeline_drivedreamer2.py)
"""DriveDreamer-2 (AAAI 2025) -- LLM-enhanced world model for driving video generation.

The official repo's `DriveDreamer2Pipeline` (dreamer-models/dreamer_models/pipelines/
drivedreamer2/pipeline_drivedreamer2.py) imports its denoising backbone directly from the
library: `from diffusers.models import AutoencoderKLTemporalDecoder,
UNetSpatioTemporalConditionModel` -- i.e. the actual generative core used at inference time
is the real, unmodified diffusers Stable-Video-Diffusion UNet (same class DriveDreamer-2
loads at `self.unet = unet` and calls at `self.unet(...)` in the pipeline's `__call__`). A
near-duplicate copy of this class also ships inside the repo's own
`dreamer_models/models/drivedreamer2/unet_spatio_temporal_condition.py`, but that copy is
dead code -- never imported by the pipeline, which uses the diffusers original. So this is
real, unmodified library-model material (rung 1), staged as a module (not a recipe) because
`UNetSpatioTemporalConditionModel.forward` takes multiple positional/keyword tensor inputs
(noisy video-latent sample, diffusion timestep, image-conditioning encoder_hidden_states,
added_time_ids) rather than a single input tensor.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from diffusers import UNetSpatioTemporalConditionModel

MENAGERIE_ZOO = "vendored-pytorch"


class DriveDreamer2UNetWrapper(nn.Module):
    """Wraps the real diffusers UNetSpatioTemporalConditionModel exactly as constructed and
    called by DriveDreamer2Pipeline, tiny-sized for tracing, returning a single tensor
    (native forward returns a UNetSpatioTemporalConditionOutput dataclass with `.sample`)."""

    def __init__(self):
        super().__init__()
        self.unet = UNetSpatioTemporalConditionModel(
            sample_size=8,  # real repo default is much larger (e.g. 72/96 latent px)
            in_channels=4,  # unmodified from repo (VAE latent channels)
            out_channels=4,
            down_block_types=(
                "CrossAttnDownBlockSpatioTemporal",
                "DownBlockSpatioTemporal",
            ),
            up_block_types=(
                "UpBlockSpatioTemporal",
                "CrossAttnUpBlockSpatioTemporal",
            ),
            block_out_channels=(32, 32),  # real repo uses (320, 640, 1280, 1280)
            addition_time_embed_dim=8,
            projection_class_embeddings_input_dim=24,
            layers_per_block=1,  # real repo uses 2
            cross_attention_dim=8,  # real repo uses 1024
            transformer_layers_per_block=1,
            num_attention_heads=(2, 2),
            num_frames=2,  # real repo uses many more video frames
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.unet(
            sample, timestep, encoder_hidden_states, added_time_ids=added_time_ids
        ).sample


def build_drivedreamer2():
    return DriveDreamer2UNetWrapper()


def example_input_drivedreamer2():
    num_frames = 2
    sample = torch.randn(1, num_frames, 4, 8, 8)
    timestep = torch.tensor(1.0)
    encoder_hidden_states = torch.randn(1, 1, 8)
    added_time_ids = torch.zeros(1, 3)
    return (sample, timestep, encoder_hidden_states, added_time_ids)


MENAGERIE_ENTRIES = [
    ("DriveDreamer-2", build_drivedreamer2, example_input_drivedreamer2, 2025, "vendored-pytorch"),
]
