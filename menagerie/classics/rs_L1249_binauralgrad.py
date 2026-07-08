# SOURCE: vendored from microsoft/NeuralSpeech @ 8cf4fbcbf451c3affc68ecfa6c970327e927062f
# https://raw.githubusercontent.com/microsoft/NeuralSpeech/8cf4fbcbf451c3affc68ecfa6c970327e927062f/BinauralGrad/src/binauralgrad/model.py
# https://raw.githubusercontent.com/microsoft/NeuralSpeech/8cf4fbcbf451c3affc68ecfa6c970327e927062f/BinauralGrad/src/binauralgrad/params.py
#
# Leng, Chen, Guo, Liu, Chen, Le, Tan, Mandic, Zhao, Song, Bian, "BinauralGrad:
# A Two-Stage Conditional Diffusion Probabilistic Model for Binaural Audio
# Synthesis" (NeurIPS 2022, arXiv:2205.14807) -- official Microsoft repo. The
# denoiser network `BinauralGrad(nn.Module)` (real `model.py`, vendored
# verbatim below: `DiffusionEmbedding`, `BinauralPreNet`, `ResidualBlock`,
# `BinauralGrad`) is a WaveNet-style dilated-conv1d diffusion denoiser
# conditioned on a sinusoidal diffusion-step embedding and a geometric-warp
# mel-spectrogram-shaped conditioner produced by `BinauralPreNet` (twin
# Conv1d towers over the geometric-warp signal + camera/head "view" vector,
# concatenated and projected to `n_mels` channels). `ResidualBlock` is the
# standard DiffWave-family gated dilated-conv residual unit: diffusion-step
# FiLM-add, dilated Conv1d, conditioner-projection add, gated-tanh
# activation, 1x1 output projection split into residual+skip. Only mechanical
# edits: dropped the unused `scipy`/`Rotation` import (used elsewhere in the
# package for dataset geometric-warp precomputation, not by the model
# forward path) and the module-level `Linear`/`ConvTranspose2d` aliases are
# kept as in the source. No architecture line altered.
#
# `params.py`'s `AttrDict` + `params_stage_one` config (`predict_mean_condition=True`,
# `use_mono=True`, real `n_mels=80`, `dilation_cycle_length=10`,
# `unconditional=False`) is used, with `residual_layers`/`residual_channels`
# shrunk for a tiny trace-verification build (kept `dilation_cycle_length=10`
# and `unconditional=False` so the real conditioner-projection + dilation-cycle
# control flow are both exercised).

from __future__ import annotations

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def override(self, attrs):
        if isinstance(attrs, dict):
            self.__dict__.update(**attrs)
        elif isinstance(attrs, (list, tuple, set)):
            for attr in attrs:
                self.override(attr)
        elif attrs is not None:
            raise NotImplementedError
        return self


class DiffusionEmbedding(nn.Module):
    def __init__(self, max_steps):
        super().__init__()
        self.register_buffer("embedding", self._build_embedding(max_steps), persistent=False)
        self.projection1 = Linear(128, 512)
        self.projection2 = Linear(512, 512)

    def forward(self, diffusion_step):
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(diffusion_step)
        x = self.projection1(x)
        x = silu(x)
        x = self.projection2(x)
        x = silu(x)
        return x

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)

    def _build_embedding(self, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(64).unsqueeze(0)  # [1,64]
        table = steps * 10.0 ** (dims * 4.0 / 63.0)  # [T,64]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class BinauralPreNet(nn.Module):
    def __init__(
        self,
        n_mels,
        binaural_type="",
        addmono=False,
        use_mean_condition=False,
        predict_mean_condition=False,
    ):
        super().__init__()
        self.conv_view1 = torch.nn.Conv1d(7, 20, 3, padding=1)
        self.conv_view2 = torch.nn.Conv1d(20, 40, 3, padding=1)
        self.addmono = addmono
        self.use_mean_condition = use_mean_condition
        self.predict_mean_condition = predict_mean_condition
        if addmono:
            self.conv_dsp1 = torch.nn.Conv1d(
                3 + (0 if not use_mean_condition else 1), 20, 3, padding=1
            )
        else:
            self.conv_dsp1 = torch.nn.Conv1d(2 if not use_mean_condition else 3, 20, 3, padding=1)
        self.conv_dsp2 = torch.nn.Conv1d(20, 40, 3, padding=1)
        self.conv = torch.nn.Conv1d(80, n_mels, 3, padding=1)

    def forward(self, geowarp, view, mono, mean_condition):
        # geowarp = torch.unsqueeze(geowarp, 1)
        if self.addmono:
            if self.use_mean_condition:
                geowarp = torch.cat([geowarp, mono, mean_condition], axis=1)
            else:
                geowarp = torch.cat([geowarp, mono], axis=1)
        geowarp = self.conv_dsp1(geowarp)
        geowarp = F.leaky_relu(geowarp, 0.4)
        geowarp = self.conv_dsp2(geowarp)
        geowarp = F.leaky_relu(geowarp, 0.4)

        view = self.conv_view1(view)
        view = F.leaky_relu(view, 0.4)
        view = self.conv_view2(view)
        view = F.leaky_relu(view, 0.4)

        x = self.conv(torch.cat([geowarp, view], axis=1))
        x = F.leaky_relu(x, 0.4)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, n_mels, residual_channels, dilation, uncond=False):
        """
        :param n_mels: inplanes of conv1x1 for spectrogram conditional
        :param residual_channels: audio conv
        :param dilation: audio conv dilation
        :param uncond: disable spectrogram conditional
        """
        super().__init__()
        self.dilated_conv = Conv1d(
            residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation
        )
        self.diffusion_projection = Linear(512, residual_channels)
        if not uncond:  # conditional model
            self.conditioner_projection = Conv1d(n_mels, 2 * residual_channels, 1)
        else:  # unconditional model
            self.conditioner_projection = None

        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, diffusion_step, conditioner=None):
        assert (conditioner is None and self.conditioner_projection is None) or (
            conditioner is not None and self.conditioner_projection is not None
        )

        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        y = x + diffusion_step
        if self.conditioner_projection is None:  # using a unconditional model
            y = self.dilated_conv(y)
        else:
            conditioner = self.conditioner_projection(conditioner)
            y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class BinauralGrad(nn.Module):
    def __init__(self, params, binaural_type=""):
        super().__init__()
        self.params = params
        self.binaural_type = binaural_type
        self.loss_per_layer = getattr(params, "loss_per_layer", 0)
        self.use_mean_condition = getattr(params, "use_mean_condition", False)
        self.predict_mean_condition = getattr(params, "predict_mean_condition", False)
        self.warper = None
        if not self.predict_mean_condition:
            self.input_projection = Conv1d(2, params.residual_channels, 1)
            self.output_projection = Conv1d(params.residual_channels, 2, 1)
        else:
            self.input_projection = Conv1d(1, params.residual_channels, 1)
            self.output_projection = Conv1d(params.residual_channels, 1, 1)
        self.diffusion_embedding = DiffusionEmbedding(len(params.noise_schedule))

        self.binaural_pre_net = BinauralPreNet(
            params.n_mels,
            binaural_type=binaural_type,
            addmono=getattr(params, "use_mono", False),
            use_mean_condition=self.use_mean_condition,
            predict_mean_condition=self.predict_mean_condition,
        )
        self.spectrogram_upsampler = None

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    params.n_mels,
                    params.residual_channels,
                    2 ** (i % params.dilation_cycle_length),
                    uncond=params.unconditional,
                )
                for i in range(params.residual_layers)
            ]
        )
        self.skip_projection = Conv1d(params.residual_channels, params.residual_channels, 1)

        nn.init.zeros_(self.output_projection.weight)

    def forward(
        self,
        audio,
        diffusion_step,
        spectrogram=None,
        geowarp=None,
        view=None,
        mono=None,
        mean_condition=None,
    ):
        # x = audio.unsqueeze(1)
        x = audio
        x = self.input_projection(x)
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)
        spectrogram = self.binaural_pre_net(geowarp, view, mono, mean_condition)

        skip = None
        extra_output = []
        for l_id, layer in enumerate(self.residual_layers):
            x, skip_connection = layer(x, diffusion_step, spectrogram)
            if self.loss_per_layer != 0 and l_id % self.loss_per_layer == self.loss_per_layer - 1:
                extra_output.append(
                    self.output_projection(F.relu(self.skip_projection(skip / sqrt(l_id))))
                )
            skip = skip_connection if skip is None else skip_connection + skip

        x = skip / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        if self.loss_per_layer != 0:
            return x, extra_output, geowarp
        else:
            return x, geowarp


def _tiny_params_stage_one() -> AttrDict:
    # Shrunk from the real `params_stage_one` (real: residual_layers=30,
    # residual_channels=128, noise_schedule len 200) for a fast trace build;
    # n_mels / dilation_cycle_length / unconditional / use_mono /
    # predict_mean_condition kept at their real values.
    return AttrDict(
        use_mono=True,
        n_mels=80,
        residual_layers=3,
        residual_channels=8,
        dilation_cycle_length=10,
        unconditional=False,
        noise_schedule=[1e-4, 5e-3, 1e-2, 2e-2],
        loss_per_layer=0,
        use_mean_condition=False,
        predict_mean_condition=True,
    )


def build_binauralgrad() -> BinauralGrad:
    return BinauralGrad(_tiny_params_stage_one(), binaural_type="").eval()


def example_input_binauralgrad():
    p = _tiny_params_stage_one()
    batch = 1
    audio_len = 64
    audio = torch.randn(batch, 1, audio_len)
    diffusion_step = torch.randint(0, len(p.noise_schedule), (batch,))
    # BinauralPreNet.addmono=True (use_mono=True), use_mean_condition=False:
    # conv_dsp1 expects 3 input channels (geowarp channel handled by caller
    # concatenating [geowarp, mono]); conv_view1 expects 7 channels.
    geowarp = torch.randn(batch, 2, audio_len)
    mono = torch.randn(batch, 1, audio_len)
    view = torch.randn(batch, 7, audio_len)
    # positional order matches BinauralGrad.forward(audio, diffusion_step,
    # spectrogram=None, geowarp=None, view=None, mono=None, mean_condition=None)
    return (audio, diffusion_step, None, geowarp, view, mono, None)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("BinauralGrad", "build_binauralgrad", "example_input_binauralgrad", 2022, "vendored-pytorch"),
]
