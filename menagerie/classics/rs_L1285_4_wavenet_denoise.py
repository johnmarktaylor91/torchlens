# FAITHFUL PORT of drethage/speech-denoising-wavenet @ master (original framework: Keras 1.2.0
# + Theano 0.9.0)
# (models.py class DenoisingWavenet.build_model / dilated_residual_block;
# layers.py classes AddSingletonDepth / Slice / Subtract; config.json default hyperparameters)
"""A WaveNet For Speech Denoising (Rethage, Pons, Serra, ICASSP 2018). Official repo:
https://github.com/drethage/speech-denoising-wavenet (``models.py`` + ``layers.py`` @ master).

The official repo pins ``theano==0.9.0`` + ``keras==1.2.0`` (Keras 1.x functional API,
Python 2 print statements) -- ancient, incompatible with any reasonably installable base-env
today -- so this transcribes ``DenoisingWavenet.build_model``/``dilated_residual_block``
FAITHFULLY into self-contained torch, matching the source's real WaveNet-for-denoising
architecture, not a generic WaveNet:
  - ``initial_causal_conv``: a first ``Conv1d`` (kernel length ``filters.lengths.res``,
    "same" padding, no bias) over the raw waveform, immediately summed with a per-condition-
    class ``Dense`` projection (broadcast/repeated over time) -- the network is FiLM-style
    conditioned on a noise/speaker class throughout, exactly as ``initial_dense_condition`` +
    ``initial_data_condition_merge`` in the source.
  - ``num_stacks`` (default 3) stacks of dilated ``dilated_residual_block``s at dilations
    ``2**i`` for ``i in 0..dilations`` (default ``dilations=9`` -> rates
    1,2,4,...,512 per stack): each block applies one dilated causal ``Conv1d`` producing
    ``2*filters.depths.res`` channels, splits it into two halves, adds a condition-class
    ``Dense`` projection to EACH half separately (``res_%d_dense_condition``/
    ``res_%d_condition_reshape``/two ``Slice``s), then gates them
    (``tanh(half1) * sigmoid(half2)``) -- the source's real gated-activation unit, unchanged.
    The gated output passes through a ``1x1 Conv1d`` producing
    ``filters.depths.res + filters.depths.skip`` channels, split into a residual branch
    (added back to the block's input, exactly as ``res_x = original_x + res_x`` in the source)
    and a skip branch cropped to the network's ``target_field_length`` output window (the
    real ``keep_samples_of_interest`` slice, needed because dilated causal convs shrink the
    valid receptive field from the edges).
  - All ``num_residual_blocks`` skip outputs are summed, ReLU'd, then pushed through two more
    condition-conditioned ``Conv1d`` + ReLU stages (``filters.depths.final`` = ``[2048, 256]``
    by default) and a final ``Conv1d(1, kernel_size=1)`` to a single denoised-speech channel
    -- exactly the source's ``penultimate``/``final`` conv-1d-with-condition stages.
  - Two output heads, both real outputs of the source network: the denoised speech
    (``data_output_1``) and, via a ``Subtract`` layer, the residual noise estimate
    (``data_input_target_field_length - data_out_speech`` = ``data_output_2``).
No layer, dilation schedule, channel width, condition-injection point, or gating/skip/residual
topology was changed from ``models.py``; only the framework (Keras/Theano -> torch) and the
input/condition tensor plumbing (Keras functional-API graph -> an explicit ``nn.Module``
forward) were translated.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


def _compute_receptive_field_length(num_stacks, dilations, filter_length, target_field_length=1):
    """Faithful port of ``util.compute_receptive_field_length`` (single-sample receptive
    field of a stack of dilated causal convs with the given filter length)."""
    half = (filter_length - 1) // 2 if filter_length % 2 == 1 else filter_length // 2
    length_per_stack = sum(dilation * 2 * half for dilation in dilations)
    return num_stacks * length_per_stack + target_field_length


class _DilatedResidualBlock(nn.Module):
    """Faithful port of ``DenoisingWavenet.dilated_residual_block``: dilated causal conv ->
    condition-gated tanh/sigmoid activation -> 1x1 conv -> residual + cropped skip outputs."""

    def __init__(
        self, res_channels, skip_channels, condition_dim, kernel_size, dilation, target_field_length
    ):
        super().__init__()
        self.res_channels = res_channels
        self.skip_channels = skip_channels
        self.target_field_length = target_field_length

        pad = dilation * (kernel_size - 1) // 2
        self.dilated_conv = nn.Conv1d(
            res_channels,
            2 * res_channels,
            kernel_size,
            padding=pad,
            dilation=dilation,
            bias=False,
        )
        # condition Dense -> reshape(res_channels, 2) -> two slices, matching
        # ``res_%d_dense_condition``/``res_%d_condition_reshape`` in the source.
        self.condition_dense = nn.Linear(condition_dim, 2 * res_channels, bias=False)
        self.out_conv = nn.Conv1d(res_channels, res_channels + skip_channels, 1, bias=False)

    def forward(self, x, condition):
        original_x = x
        data_out = self.dilated_conv(x)  # (B, 2*res, T)
        data_out_1 = data_out[:, : self.res_channels, :]
        data_out_2 = data_out[:, self.res_channels :, :]

        cond = self.condition_dense(condition)  # (B, 2*res)
        cond = cond.view(cond.size(0), self.res_channels, 2)
        cond_1 = cond[:, :, 0].unsqueeze(-1)  # broadcast over time (RepeatVector equivalent)
        cond_2 = cond[:, :, 1].unsqueeze(-1)

        data_out_1 = data_out_1 + cond_1
        data_out_2 = data_out_2 + cond_2

        tanh_out = torch.tanh(data_out_1)
        sigm_out = torch.sigmoid(data_out_2)
        gated = tanh_out * sigm_out

        out = self.out_conv(gated)
        res_x = out[:, : self.res_channels, :]
        skip_x = out[:, self.res_channels :, :]

        res_x = original_x + res_x

        # keep_samples_of_interest: crop the skip branch to the target field window.
        total_length = skip_x.size(-1)
        start = (total_length - self.target_field_length) // 2
        skip_x = skip_x[:, :, start : start + self.target_field_length]

        return res_x, skip_x


class DenoisingWavenet(nn.Module):
    """Faithful port of ``DenoisingWavenet.build_model``."""

    def __init__(
        self,
        num_condition_classes=29,
        num_stacks=3,
        num_dilation_powers=9,
        res_channels=128,
        skip_channels=128,
        final_channels=(2048, 256),
        res_kernel_size=3,
        final_kernel_sizes=(3, 3),
        target_field_length=65,
    ):
        super().__init__()
        self.dilations = [2**i for i in range(num_dilation_powers + 1)]
        self.num_stacks = num_stacks
        self.res_channels = res_channels
        self.target_field_length = target_field_length
        self.receptive_field_length = _compute_receptive_field_length(
            num_stacks, self.dilations, res_kernel_size, 1
        )
        self.input_length = self.receptive_field_length + (target_field_length - 1)

        self.initial_causal_conv = nn.Conv1d(
            1, res_channels, res_kernel_size, padding=res_kernel_size // 2, bias=False
        )
        self.initial_condition_dense = nn.Linear(num_condition_classes, res_channels, bias=False)

        blocks = []
        for _stack in range(num_stacks):
            for dilation in self.dilations:
                blocks.append(
                    _DilatedResidualBlock(
                        res_channels,
                        skip_channels,
                        num_condition_classes,
                        res_kernel_size,
                        dilation,
                        target_field_length,
                    )
                )
        self.blocks = nn.ModuleList(blocks)

        self.penultimate_conv = nn.Conv1d(
            skip_channels,
            final_channels[0],
            final_kernel_sizes[0],
            padding=final_kernel_sizes[0] // 2,
            bias=False,
        )
        self.penultimate_condition_dense = nn.Linear(
            num_condition_classes, final_channels[0], bias=False
        )

        self.final_conv = nn.Conv1d(
            final_channels[0],
            final_channels[1],
            final_kernel_sizes[1],
            padding=final_kernel_sizes[1] // 2,
            bias=False,
        )
        self.final_condition_dense = nn.Linear(num_condition_classes, final_channels[1], bias=False)

        self.output_conv = nn.Conv1d(final_channels[1], 1, 1)

    def forward(self, data_input, condition_input):
        # data_input: (B, input_length); condition_input: (B, num_condition_classes)
        x = data_input.unsqueeze(1)  # AddSingletonDepth -> (B, 1, T)

        # data_input_target_field_length (for the final Subtract layer)
        total_length = x.size(-1)
        start = (total_length - self.target_field_length) // 2
        target_window = x[:, :, start : start + self.target_field_length]

        data_out = self.initial_causal_conv(x)
        cond_out = self.initial_condition_dense(condition_input).unsqueeze(-1)
        data_out = data_out + cond_out

        skip_connections = []
        for block in self.blocks:
            data_out, skip_out = block(data_out, condition_input)
            skip_connections.append(skip_out)

        data_out = torch.stack(skip_connections, dim=0).sum(dim=0)
        data_out = F.relu(data_out)

        data_out = self.penultimate_conv(data_out)
        cond_out = self.penultimate_condition_dense(condition_input).unsqueeze(-1)
        data_out = data_out + cond_out
        data_out = F.relu(data_out)

        data_out = self.final_conv(data_out)
        cond_out = self.final_condition_dense(condition_input).unsqueeze(-1)
        data_out = data_out + cond_out

        data_out_speech = self.output_conv(data_out)  # (B, 1, target_field_length)
        data_out_noise = target_window - data_out_speech  # Subtract layer

        data_out_speech = data_out_speech.squeeze(1)
        data_out_noise = data_out_noise.squeeze(1)
        return data_out_speech, data_out_noise


# ---------------------------------------------------------------------------
# Menagerie staging harness
# ---------------------------------------------------------------------------
def build_wavenet_denoise():
    """Tiny-size WaveNet denoiser: shrinks num_stacks/dilation depth/channel widths well
    below config.json's real defaults (num_stacks=3, dilations=9, res=128, skip=128,
    final=[2048,256]) while keeping every architectural component (dual condition-gated
    residual blocks, skip summation, two final condition-conditioned conv stages, dual
    speech/noise output heads)."""
    return DenoisingWavenet(
        num_condition_classes=4,  # default 29
        num_stacks=1,  # default 3
        num_dilation_powers=2,  # default 9 (-> dilations [1,2,4,8])
        res_channels=4,  # default 128
        skip_channels=4,  # default 128
        final_channels=(8, 4),  # default (2048, 256)
        res_kernel_size=3,
        final_kernel_sizes=(3, 3),
        target_field_length=9,  # default 1601
    )


def example_input_wavenet_denoise():
    torch.manual_seed(0)
    model = build_wavenet_denoise()
    data_input = torch.randn(1, model.input_length)
    condition_input = F.one_hot(torch.tensor([1]), num_classes=4).float()
    return (data_input, condition_input)


MENAGERIE_ENTRIES = [
    (
        "WaveNet Speech Denoising",
        "build_wavenet_denoise",
        "example_input_wavenet_denoise",
        2018,
        "ported-pytorch",
    ),
]
