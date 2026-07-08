# FAITHFUL PORT of auspicious3000/WaveNet-Enhancement @ master (original framework: TensorFlow 1.x)
# https://github.com/auspicious3000/WaveNet-Enhancement/blob/master/bawn.py
#
# "BaWN" (Bayesian WaveNet) speech-enhancement model. bawn.py is TF1-only
# (tf.get_variable / tf.layers.conv1d / tf.variable_scope graph-mode API,
# not installable/runnable in a modern base-lib torch env), so this is a
# faithful architectural port to torch of the REAL model-building functions:
# `_dilated_conv1d` (gated dilated causal conv block with residual + skip
# paths, "together" 2x-width conv split into filter/gate halves exactly as
# in the original, `tanh` gate on filter half / `sigmoid` on gate half),
# `_wavnet` (stack of `_dilated_conv1d` blocks across blocks/layers with
# doubling dilation rate per layer, matching `check_boundries` bottom/top
# handling), and `_post_processing` (ReLU + stack of 1x1 convs).
# `model_denoise` is the actual full BaWN inference model used for speech
# enhancement (bawn.py):
#   - "clean" WaveNet (prior speech model, filter_width=2, one-hot 256-way
#     mu-law input, causal `ind_in = rate:`)
#   - "noisy" WaveNet (conditioned on raw noisy waveform, filter_width=3,
#     `ind_in = rate:-rate` non-causal padding)
#   - a frozen COPY of the clean/"prior" WaveNet (trainable=False in the
#     original, used as a fixed language-model prior added via
#     `tf.stop_gradient`)
#   - skip outputs from clean+noisy WaveNets are summed and passed through
#     `_post_processing` to produce the "likelihood" logits over 256 mu-law
#     classes; the frozen "prior" WaveNet's own post-processed logits are
#     added on top (`outputs_loglik = stop_gradient(outputs_pr) + outputs_ll`).
# Every gated-conv / skip / residual / post-processing mechanism is
# reproduced; only the TF1 variable/session/graph plumbing is dropped in
# favor of ordinary nn.Module state, and `tf.layers.conv1d(..., padding=
# 'valid', dilation_rate=rate)` becomes `nn.Conv1d(..., padding=0,
# dilation=rate)` (numerically identical "valid" dilated convolution).
import torch
import torch.nn as nn
import torch.nn.functional as F


def _check_boundaries(num_blocks, num_layers, block, layer):
    bottom = block == 0 and layer == 0
    top = (block + 1 == num_blocks) and (layer + 1 == num_layers)
    return bottom, top


class DilatedConv1dBlock(nn.Module):
    """Port of bawn.py `_dilated_conv1d`: gated dilated causal conv with an
    optional bottom-layer input projection, and residual + skip 1x1 convs.
    """

    def __init__(
        self,
        residual_channels,
        skip_channels,
        skip_width,
        filter_width,
        rate,
        causality,
        bottom,
        top,
        bottom_in_channels=None,
    ):
        super().__init__()
        self.residual_channels = residual_channels
        self.skip_width = skip_width
        self.filter_width = filter_width
        self.rate = rate
        self.causality = causality
        self.bottom = bottom
        self.top = top

        if bottom:
            # bawn.py: "pre" conv, tanh activation, projects the raw
            # one-hot (clean/prior) or raw-channel (noisy) input up to
            # `residual_channels`.
            self.pre = nn.Conv1d(
                bottom_in_channels,
                residual_channels,
                kernel_size=filter_width,
                padding=0,
                bias=True,
            )

        # "together": replaces two separate filter/gate convs with one conv
        # of 2x width (bawn.py comment: "replace 2 separate conv with 1 conv
        # with 2x residual channels"), no bias, dilated, valid padding.
        self.together = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            kernel_size=filter_width,
            padding=0,
            dilation=rate,
            bias=False,
        )

        self.skip = nn.Conv1d(residual_channels, skip_channels, kernel_size=1, bias=True)

        if not top:
            self.residual = nn.Conv1d(
                residual_channels, residual_channels, kernel_size=1, bias=True
            )

    def forward(self, inputs):
        if self.bottom:
            inputs_proc = torch.tanh(self.pre(inputs))
        else:
            inputs_proc = inputs

        outputs_together = self.together(inputs_proc)
        r = self.residual_channels
        outputs_filter = outputs_together[:, :r, :]
        outputs_gate = outputs_together[:, r:, :]

        outputs_filter = torch.tanh(outputs_filter)
        outputs_gate = torch.sigmoid(outputs_gate)
        outputs_gated = outputs_filter * outputs_gate

        # bawn.py builds `ind_in`/`ind_out` as plain Python slice objects
        # (np.s_[...]) applied relative to each tensor's own width -- NOT as
        # absolute indices computed from `outputs_together`'s width. Mirror
        # that with relative (possibly negative) start/stop here so the
        # slice is always applied against the tensor it actually indexes.
        width = outputs_together.shape[-1]
        if self.causality in ("clean", "prior"):
            ind_in = (self.rate, None)  # rate:
            ind_out = (width - self.skip_width, None)  # -skip_width:
        else:  # 'noisy'
            ind_in = (self.rate, -self.rate)  # rate:-rate
            len_cut = (width - self.skip_width) // 2
            if len_cut == 0:
                ind_out = (0, None)
            else:
                ind_out = (len_cut, -len_cut)

        gated_for_skip = outputs_gated[:, :, ind_out[0] : ind_out[1]]
        outputs_skip = self.skip(gated_for_skip)

        if not self.top:
            outputs_residual = self.residual(outputs_gated)
            inputs_slice = inputs_proc[:, :, ind_in[0] : ind_in[1]]
            outputs_dense = inputs_slice + outputs_residual
        else:
            outputs_dense = None

        return outputs_dense, outputs_skip, inputs_proc


class WaveNetStack(nn.Module):
    """Port of bawn.py `_wavnet`: a stack of `DilatedConv1dBlock`s across
    `num_blocks` x `num_layers`, dilation rate doubling per layer within a
    block (`rate = 2 ** i`), matching `check_boundries` for bottom/top.
    """

    def __init__(
        self,
        num_blocks,
        num_layers,
        residual_channels,
        skip_channels,
        skip_width,
        filter_width,
        causality,
        bottom_in_channels,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        blocks = []
        for b in range(num_blocks):
            for i in range(num_layers):
                rate = 2**i
                bottom, top = _check_boundaries(num_blocks, num_layers, b, i)
                blocks.append(
                    DilatedConv1dBlock(
                        residual_channels=residual_channels,
                        skip_channels=skip_channels,
                        skip_width=skip_width,
                        filter_width=filter_width,
                        rate=rate,
                        causality=causality,
                        bottom=bottom,
                        top=top,
                        bottom_in_channels=bottom_in_channels if bottom else None,
                    )
                )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, inputs):
        h = inputs
        skips = []
        for block in self.blocks:
            h, skip, _pre = block(h)
            skips.append(skip)
        return skips


class PostProcessing(nn.Module):
    """Port of bawn.py `_post_processing`: sum the skip outputs, ReLU, then
    a stack of 1x1 convs (all but the last with ReLU activation)."""

    def __init__(self, in_channels, num_layers, num_classes):
        super().__init__()
        layers = []
        c_in = in_channels
        for _ in range(num_layers - 1):
            layers.append(nn.Conv1d(c_in, num_classes, kernel_size=1, bias=True))
            c_in = num_classes
        self.hidden = nn.ModuleList(layers)
        self.out = nn.Conv1d(c_in, num_classes, kernel_size=1, bias=True)

    def forward(self, skips):
        h = torch.stack(skips, dim=0).sum(dim=0)
        h = F.relu(h)
        for conv in self.hidden:
            h = F.relu(conv(h))
        return self.out(h)


class WaveNetSE(nn.Module):
    """Port of bawn.py `model_denoise`: clean/prior WaveNets (one-hot
    256-way mu-law clean speech, filter_width=2) + noisy WaveNet (raw noisy
    waveform, filter_width=3), skip sums -> post-processing -> logits over
    mu-law classes; frozen prior branch added on top exactly as
    `stop_gradient(outputs_pr) + outputs_ll` in the original.
    """

    def __init__(
        self,
        num_blocks=1,
        num_layers=3,
        residual_channels=8,
        skip_channels=8,
        skip_width=4,
        num_classes=16,
        num_post_layers=2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.skip_width = skip_width

        self.clean_net = WaveNetStack(
            num_blocks,
            num_layers,
            residual_channels,
            skip_channels,
            skip_width,
            filter_width=2,
            causality="clean",
            bottom_in_channels=num_classes,
        )
        self.noisy_net = WaveNetStack(
            num_blocks,
            num_layers,
            residual_channels,
            skip_channels,
            skip_width,
            filter_width=3,
            causality="noisy",
            bottom_in_channels=1,
        )
        self.prior_net = WaveNetStack(
            num_blocks,
            num_layers,
            residual_channels,
            skip_channels,
            skip_width,
            filter_width=2,
            causality="prior",
            bottom_in_channels=num_classes,
        )
        self.post_likli = PostProcessing(skip_channels, num_post_layers, num_classes)
        self.post_prior = PostProcessing(skip_channels, num_post_layers, num_classes)

    def forward(self, clean_indices, noisy_wave):
        # clean_indices: (B, T_clean) int64 mu-law bin ids -> one-hot (B, C, T_clean)
        clean_onehot = F.one_hot(clean_indices, self.num_classes).permute(0, 2, 1).float()
        # noisy_wave: (B, T_noisy) float raw waveform -> add channel dim (B, 1, T_noisy)
        noisy_in = noisy_wave.unsqueeze(1)

        skips_clean = self.clean_net(clean_onehot)
        skips_noisy = self.noisy_net(noisy_in)
        skips_prior = self.prior_net(clean_onehot)

        outputs_ll = self.post_likli(skips_clean + skips_noisy)
        with torch.no_grad():
            outputs_pr = self.post_prior(skips_prior)
        outputs_loglik = outputs_pr.detach() + outputs_ll
        return outputs_loglik


MENAGERIE_ZOO = "ported-pytorch"


def build_wavenet_se():
    return WaveNetSE(
        num_blocks=1,
        num_layers=3,
        residual_channels=8,
        skip_channels=8,
        skip_width=4,
        num_classes=16,
        num_post_layers=2,
    ).eval()


def example_input_wavenet_se():
    # Each layer's residual path (`inputs_proc[:, :, rate:]` for clean/prior,
    # `inputs_proc[:, :, rate:-rate]` for noisy) shrinks the running length by
    # `rate*(filter_width-1)` per layer, plus a one-time `filter_width-1`
    # shrink from the bottom "pre" projection conv. With num_layers=3 (rates
    # 1, 2, 4): clean/prior (filter_width=2) needs start length 12 to reach
    # skip_width=4; noisy (filter_width=3) needs start length 20.
    clean_len = 12
    noisy_len = 20
    clean_indices = torch.randint(0, 16, (1, clean_len), dtype=torch.long)
    noisy_wave = torch.randn(1, noisy_len)
    return (clean_indices, noisy_wave)


MENAGERIE_ENTRIES = [
    (
        "WAVENET-SE (WaveNet for speech enhancement)",
        "build_wavenet_se",
        "example_input_wavenet_se",
        2017,
        MENAGERIE_ZOO,
    ),
]
