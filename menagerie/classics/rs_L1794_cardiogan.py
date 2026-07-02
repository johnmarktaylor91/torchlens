# FAITHFUL PORT of pritamqu/ppg2ecg-cardiogan @ main (original framework: TensorFlow 2.2 / Keras)
# https://raw.githubusercontent.com/pritamqu/ppg2ecg-cardiogan/main/codes/module.py
# https://raw.githubusercontent.com/pritamqu/ppg2ecg-cardiogan/main/codes/layers.py
#
# Sarkar & Etemad, 2021 (AAAI 2021) "CardioGAN: Attentive Generative Adversarial
# Network with Dual Discriminators for Synthesis of ECG from PPG". The
# released, deployable artifact (test_cardiogan.py / cardiogan_realtime.py) is
# only the trained PPG->ECG generator, `module.generator_attention()`: a
# 1D U-Net-style encoder-decoder over 512-sample PPG windows with 6
# downsample stages (Conv1D/LeakyReLU/LayerNorm), 6 mirrored upsample stages
# (Deconv1D/ReLU/LayerNorm) fused with the *encoder's* corresponding
# activations through a learned soft-attention gate (`attention_block_1d`,
# adapted in the original repo from a Keras U-Net/Attention-U-Net
# implementation) rather than a plain skip-concat -- that attention-gated
# skip connection is CardioGAN generator's architectural signature. The dual
# discriminators (ECG-domain + PPG-domain PatchGANs) are training-only and are
# not part of the shipped inference model, so they are not ported here.
#
# The original code is TensorFlow2.2/Keras (`tensorflow.keras`,
# `tensorflow_addons`), a different framework from base-env torch, so per the
# build ladder this is a faithful architectural PORT into self-contained
# torch, not a vendor-as-is. Every mechanism below (per-stage filter/kernel
# schedule, TF `'same'` padding semantics, LayerNorm-over-channels,
# leaky_relu/relu/tanh placement, and the additive attention-gated skip) is
# transcribed 1:1 from the real `module.py`/`layers.py` source (reproduced in
# the docstrings), translated from Keras NHWC/"Conv1D-as-Conv2D-with-height-1"
# convention into idiomatic torch NCL Conv1d/ConvTranspose1d with manual
# TF-'same'-equivalent padding (computed at __init__ time from the fixed
# input length, since TF's 'same' padding is computed per-call from the
# runtime spatial size while torch's padding is fixed at module-construction
# time -- this repo always runs at the one fixed window size of 512 samples,
# so a static padding schedule reproduces TF 'same' exactly for that size).

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


def _tf_same_pad_conv1d(in_len: int, kernel_size: int, stride: int) -> tuple[int, int]:
    """Left/right zero-padding torch needs so that a stride-`stride`,
    kernel-`kernel_size` Conv1d over an input of length `in_len` reproduces
    TensorFlow `padding='same'` (output length = ceil(in_len / stride))."""
    out_len = -(-in_len // stride)  # ceil division
    total_pad = max((out_len - 1) * stride + kernel_size - in_len, 0)
    pad_left = total_pad // 2
    pad_right = total_pad - pad_left
    return pad_left, pad_right


def _tf_same_pad_deconv1d(in_len: int, kernel_size: int, stride: int) -> tuple[int, int, int]:
    """(torch `padding`, torch `output_padding`, crop) for a ConvTranspose1d so
    that, after cropping `crop` samples total (split from the two ends the way
    TF's `'same'`-padded Conv2DTranspose effectively does), the result matches
    TF `padding='same'` transposed-conv output length (`in_len * stride`)."""
    target_out = in_len * stride
    # Un-padded ("valid") transposed-conv output length:
    valid_out = (in_len - 1) * stride + kernel_size
    total_crop = valid_out - target_out
    total_crop = max(total_crop, 0)
    pad = total_crop // 2
    output_padding = 0
    crop_right = total_crop - pad
    # torch's ConvTranspose1d `padding` trims `padding` samples off *each* end
    # of the valid-mode output; if pad != crop_right (odd total_crop) apply the
    # extra 1-sample crop explicitly after the op.
    return pad, output_padding, crop_right - pad


class _Downsample(nn.Module):
    """Port of layers.py Conv1D (=Conv2D kernel (1,k)) + normalization('layer_norm'
    or 'none') + Activation('leaky_relu'), as used by module.py::_downsample."""

    def __init__(self, in_ch, out_ch, kernel_size, in_len, norm="layer_norm"):
        super().__init__()
        pad_l, pad_r = _tf_same_pad_conv1d(in_len, kernel_size, stride=2)
        self._pad = (pad_l, pad_r)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=2, bias=False)
        self.norm = nn.LayerNorm(out_ch) if norm == "layer_norm" else None

    def forward(self, x):
        x = F.pad(x, self._pad)
        x = self.conv(x)
        if self.norm is not None:
            # Keras LayerNormalization(axis=-1) normalizes over the channel
            # axis of NHWC/NWC data; torch's x is (N, C, L) so transpose to
            # (N, L, C), normalize, transpose back.
            x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = F.leaky_relu(x, negative_slope=0.2)
        return x


class _Upsample(nn.Module):
    """Port of layers.py DeConv1D (=Conv2DTranspose kernel (1,k)) + normalization
    + Activation('relu'), as used by module.py::_upsample (dropout branch
    unused at inference: apply_dropout defaults False in the real call sites)."""

    def __init__(self, in_ch, out_ch, kernel_size, in_len, stride, norm="layer_norm"):
        super().__init__()
        pad, output_padding, extra_crop = _tf_same_pad_deconv1d(in_len, kernel_size, stride)
        self._extra_crop = extra_crop
        self.deconv = nn.ConvTranspose1d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            output_padding=output_padding,
            bias=False,
        )
        self.norm = nn.LayerNorm(out_ch) if norm == "layer_norm" else None

    def forward(self, x):
        x = self.deconv(x)
        if self._extra_crop > 0:
            x = x[..., : x.shape[-1] - self._extra_crop]
        if self.norm is not None:
            x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = F.relu(x)
        return x


class _AttentionBlock1D(nn.Module):
    """Port of layers.py::attention_block_1d (additive-attention skip gate,
    adopted in the real repo from a Keras U-Net/Attention-U-Net
    implementation): 1x1 convs project the decoder ("curr_layer") and encoder
    ("conn_layer") activations to the same channel width, sum + ReLU, a 1x1
    conv + sigmoid produces a per-position gate, and the encoder activation is
    multiplied by that gate."""

    def __init__(self, channels):
        super().__init__()
        self.theta = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.phi = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.psi = nn.Conv1d(channels, 1, kernel_size=1, bias=True)

    def forward(self, curr_layer, conn_layer):
        theta_x = self.theta(conn_layer)
        phi_g = self.phi(curr_layer)
        f = F.relu(theta_x + phi_g)
        rate = torch.sigmoid(self.psi(f))
        return conn_layer * rate


class GeneratorAttention(nn.Module):
    """Port of module.py::generator_attention(input_shape=512,
    filter_size=[64,128,256,512,512,512], kernel_size=[16]*6, n_downsample=6,
    norm='layer_norm', skip_connection=True). Real repo call sites
    (test_cardiogan.py, cardiogan_realtime.py) always use the defaults."""

    def __init__(
        self,
        input_len: int = 512,
        filter_size=(64, 128, 256, 512, 512, 512),
        kernel_size=(16, 16, 16, 16, 16, 16),
        n_downsample: int = 6,
        norm: str = "layer_norm",
        skip_connection: bool = True,
    ):
        super().__init__()
        self.n_downsample = n_downsample
        self.skip_connection = skip_connection

        # ---- downsample stack ----
        down_lens = [input_len]
        for _ in range(n_downsample):
            down_lens.append(-(-down_lens[-1] // 2))  # ceil(len/2), matches TF 'same' stride-2
        self._down_lens = down_lens

        downs = []
        in_ch = 1
        for k in range(n_downsample):
            stage_norm = "none" if k == 0 else norm
            downs.append(
                _Downsample(in_ch, filter_size[k], kernel_size[k], down_lens[k], norm=stage_norm)
            )
            in_ch = filter_size[k]
        self.downs = nn.ModuleList(downs)

        # ---- first upsample stage (stride=1, from the deepest features) ----
        deepest_len = down_lens[n_downsample]
        deepest_ch = filter_size[n_downsample - 1]
        self.up_first = _Upsample(
            deepest_ch, deepest_ch, kernel_size[n_downsample - 1], deepest_len, stride=1, norm=norm
        )
        if skip_connection:
            self.attn_first = _AttentionBlock1D(deepest_ch)

        # ---- remaining n_downsample-1 upsample stages (stride=2, mirrored channel schedule) ----
        ups = []
        attns = []
        cur_len = deepest_len
        cur_ch = deepest_ch
        for stage in range(1, n_downsample):
            out_ch = filter_size[n_downsample - 1 - stage]
            k_size = kernel_size[n_downsample - 1 - stage]
            ups.append(_Upsample(cur_ch, out_ch, k_size, cur_len, stride=2, norm=norm))
            if skip_connection:
                attns.append(_AttentionBlock1D(out_ch))
            cur_len = cur_len * 2
            cur_ch = out_ch
        self.ups = nn.ModuleList(ups)
        self.attns = nn.ModuleList(attns) if skip_connection else None

        # ---- output projection: DeConv1D(filters=1, kernel=kernel_size[last], stride=2, 'same') + tanh ----
        last_kernel = kernel_size[0]
        pad, output_padding, extra_crop = _tf_same_pad_deconv1d(cur_len, last_kernel, stride=2)
        self._out_extra_crop = extra_crop
        self.out_conv = nn.ConvTranspose1d(
            cur_ch,
            1,
            kernel_size=last_kernel,
            stride=2,
            padding=pad,
            output_padding=output_padding,
            bias=True,
        )

    def forward(self, ppg):
        # ppg: (batch, 512) 1D PPG window -> (batch, 1, 512) channel-first
        h = ppg.unsqueeze(1)

        connections = []
        for down in self.downs:
            h = down(h)
            connections.append(h)

        h = self.up_first(h)
        if self.skip_connection:
            attn = self.attn_first(h, connections[self.n_downsample - 1])
            h = h + attn

        for stage, up in enumerate(self.ups, start=1):
            h = up(h)
            if self.skip_connection:
                attn = self.attns[stage - 1](h, connections[self.n_downsample - 1 - stage])
                h = h + attn

        h = self.out_conv(h)
        if self._out_extra_crop > 0:
            h = h[..., : h.shape[-1] - self._out_extra_crop]
        h = torch.tanh(h)
        h = h.squeeze(1)  # (batch, 512) synthetic ECG
        return h


# ============================================================================
# build_/example_input_ harness
# ============================================================================


def build_cardiogan_generator():
    model = GeneratorAttention(input_len=512)
    model.eval()
    return model


def example_input_cardiogan_generator():
    torch.manual_seed(0)
    batch = 2
    # matches test_cardiogan.py's 128Hz*4s PPG window normalized to [-1,1]
    return torch.rand(batch, 512) * 2 - 1


MENAGERIE_ENTRIES = [
    (
        "CardioGAN",
        build_cardiogan_generator,
        example_input_cardiogan_generator,
        2021,
        "ported-pytorch",
    ),
]
