# FAITHFUL PORT of coqui-ai/TTS @ dev (original framework: PyTorch + TTS/trainer/coqpit
# framework), transcribing the real YourTTS architecture:
#
# YourTTS (Casanova et al., "YourTTS: Towards Zero-Shot Multi-Speaker TTS and Zero-Shot
# Voice Conversion for everyone", ICML 2022, arXiv:2112.02418) is the VITS conditional-VAE
# + normalizing-flow + HiFi-GAN single-stage TTS architecture (Kim et al., VITS, ICML 2021)
# conditioned on EXTERNAL speaker d-vectors (no learned speaker embedding table) with the
# small architecture deltas the YourTTS paper/recipe actually uses:
#   - `use_d_vector_file=True`, `d_vector_dim=512` (external speaker-encoder d-vectors feed
#     every speaker-conditioned submodule directly as `g`, no `nn.Embedding` speaker table);
#   - `num_layers_text_encoder=10` (vs VITS-base's 6);
#   - `resblock_type_decoder="2"` (single-conv HiFi-GAN ResBlock, per the official YourTTS
#     recipe's docstring: "we accidentally trained the YourTTS using ResNet blocks type 2").
# Reference recipe (exact `VitsArgs(...)` used to train the paper's released checkpoint):
#   https://github.com/coqui-ai/TTS/blob/dev/recipes/vctk/yourtts/train_yourtts.py
#
# Ported files (`Vits`/`VitsArgs` class + its real submodules), each transcribed verbatim
# with only the config/coqpit/data-loading/training-loop machinery stripped (kept: __init__
# module wiring + eval-mode `inference()` forward; dropped: train_step, loss computation,
# checkpoint I/O, TTSDataset/tokenizer/AudioProcessor/SpeakerManager/LanguageManager glue):
#   TTS/tts/models/vits.py                          (Vits.__init__, Vits.inference,
#                                                      Vits._set_cond_input, Vits._set_x_lengths,
#                                                      Vits.upsampling_z, Vits.init_multispeaker)
#   TTS/tts/layers/vits/networks.py                  (TextEncoder, PosteriorEncoder,
#                                                      ResidualCouplingBlock, ResidualCouplingBlocks)
#   TTS/tts/layers/glow_tts/transformer.py            (RelativePositionTransformer,
#                                                      RelativePositionMultiHeadAttention,
#                                                      FeedForwardNetwork)
#   TTS/tts/layers/generic/normalization.py           (LayerNorm, LayerNorm2)
#   TTS/tts/layers/generic/wavenet.py                 (WN, fused_add_tanh_sigmoid_multiply)
#   TTS/tts/layers/vits/stochastic_duration_predictor.py (StochasticDurationPredictor,
#                                                      DilatedDepthSeparableConv,
#                                                      ElementwiseAffine, ConvFlow)
#   TTS/tts/layers/glow_tts/duration_predictor.py     (DurationPredictor -- unused by
#                                                      YourTTS's use_sdp=True default, ported
#                                                      for architectural completeness)
#   TTS/tts/layers/vits/transforms.py                 (piecewise_rational_quadratic_transform
#                                                      and the rational-quadratic spline flow
#                                                      the stochastic duration predictor uses)
#   TTS/vocoder/models/hifigan_generator.py           (HifiganGenerator, ResBlock1, ResBlock2)
#   TTS/tts/utils/helpers.py                          (sequence_mask, generate_path,
#                                                      maximum_path_numpy)
# https://github.com/coqui-ai/TTS/tree/dev
#
# The real `Vits` class imports `coqpit.Coqpit` (config base class), `trainer` (training
# loop utilities), `librosa` (mel filterbank for spectrogram loss, not used at inference),
# and `TTS.tts.datasets`/`TTS.tts.utils.text.tokenizer`/`TTS.utils.samplers` (data-loading
# infra) -- none of which are installed base libs here (torch/torchvision/timm/transformers/
# torch_geometric/torchaudio/einops/monai/segmentation_models_pytorch/snntorch/diffusers),
# hence a faithful port instead of a rung-2 vendor of the framework file directly. The
# UNDERLYING ARCHITECTURE (every nn.Module the paper contributes: TextEncoder with
# relative-position self-attention, PosteriorEncoder/flow WaveNet stacks, stochastic
# duration predictor with rational-quadratic spline flows, HiFi-GAN generator) lives in
# lower-level `TTS/tts/layers/*` files that import ONLY torch/numpy -- those are transcribed
# verbatim below; only the config-object/data-loader/training-loop wrapper is stripped.
#
# `maximum_path` (used only by the training forward, `forward_mas`) is NOT ported --
# `inference()` uses `generate_path` directly on the duration predictor's own output and
# never calls it; `HifiganGenerator.load_checkpoint` (a checkpoint-I/O convenience method,
# not architecture) was also dropped since it is not on the forward path.

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations

MENAGERIE_ZOO = "ported-pytorch"

LRELU_SLOPE = 0.1


# ---------------------------------------------------------------------------
# from TTS/tts/utils/helpers.py
# ---------------------------------------------------------------------------
def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.max()
    seq_range = torch.arange(max_len, dtype=sequence_length.dtype, device=sequence_length.device)
    return seq_range.unsqueeze(0) < sequence_length.unsqueeze(1)


def convert_pad_shape(pad_shape):
    inverted_shape = pad_shape[::-1]
    pad_shape = [item for sublist in inverted_shape for item in sublist]
    return pad_shape


def generate_path(duration, mask):
    """
    Shapes:
        - duration: :math:`[B, T_en]`
        - mask: :math:'[B, T_en, T_de]`
        - path: :math:`[B, T_en, T_de]`
    """
    b, t_x, t_y = mask.shape
    cum_duration = torch.cumsum(duration, 1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path * mask
    return path


# ---------------------------------------------------------------------------
# from TTS/tts/layers/generic/normalization.py
# ---------------------------------------------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-4):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(1, channels, 1) * 0.1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x):
        mean = torch.mean(x, 1, keepdim=True)
        variance = torch.mean((x - mean) ** 2, 1, keepdim=True)
        x = (x - mean) * torch.rsqrt(variance + self.eps)
        x = x * self.gamma + self.beta
        return x


class LayerNorm2(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = torch.nn.functional.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


# ---------------------------------------------------------------------------
# from TTS/tts/layers/generic/wavenet.py
# ---------------------------------------------------------------------------
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


class WN(torch.nn.Module):
    """Wavenet layers with weight norm and no input conditioning."""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        num_layers,
        c_in_channels=0,
        dropout_p=0,
        weight_norm_flag=True,
    ):
        super().__init__()
        assert kernel_size % 2 == 1
        assert hidden_channels % 2 == 0
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.num_layers = num_layers
        self.c_in_channels = c_in_channels
        self.dropout_p = dropout_p

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.dropout = nn.Dropout(dropout_p)

        if c_in_channels > 0:
            cond_layer = torch.nn.Conv1d(c_in_channels, 2 * hidden_channels * num_layers, 1)
            self.cond_layer = torch.nn.utils.parametrizations.weight_norm(cond_layer, name="weight")

        for i in range(num_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            if i == 0:
                in_layer = torch.nn.Conv1d(
                    in_channels,
                    2 * hidden_channels,
                    kernel_size,
                    dilation=dilation,
                    padding=padding,
                )
            else:
                in_layer = torch.nn.Conv1d(
                    hidden_channels,
                    2 * hidden_channels,
                    kernel_size,
                    dilation=dilation,
                    padding=padding,
                )
            in_layer = torch.nn.utils.parametrizations.weight_norm(in_layer, name="weight")
            self.in_layers.append(in_layer)

            if i < num_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.parametrizations.weight_norm(
                res_skip_layer, name="weight"
            )
            self.res_skip_layers.append(res_skip_layer)

        if not weight_norm_flag:
            self.remove_weight_norm()

    def forward(self, x, x_mask=None, g=None, **kwargs):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])
        x_mask = 1.0 if x_mask is None else x_mask
        if g is not None:
            g = self.cond_layer(g)
        for i in range(self.num_layers):
            x_in = self.in_layers[i](x)
            x_in = self.dropout(x_in)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)
            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.num_layers - 1:
                x = (x + res_skip_acts[:, : self.hidden_channels, :]) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self):
        from torch.nn.utils import parametrize

        if self.c_in_channels != 0:
            parametrize.remove_parametrizations(self.cond_layer, "weight")
        for layer in self.in_layers:
            parametrize.remove_parametrizations(layer, "weight")
        for layer in self.res_skip_layers:
            parametrize.remove_parametrizations(layer, "weight")


# ---------------------------------------------------------------------------
# from TTS/tts/layers/glow_tts/transformer.py
# ---------------------------------------------------------------------------
class RelativePositionMultiHeadAttention(nn.Module):
    """Multi-head attention with Relative Positional embedding. https://arxiv.org/pdf/1809.04281.pdf"""

    def __init__(
        self,
        channels,
        out_channels,
        num_heads,
        rel_attn_window_size=None,
        heads_share=True,
        dropout_p=0.0,
        input_length=None,
        proximal_bias=False,
        proximal_init=False,
    ):
        super().__init__()
        assert channels % num_heads == 0, " [!] channels should be divisible by num_heads."
        self.channels = channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.rel_attn_window_size = rel_attn_window_size
        self.heads_share = heads_share
        self.input_length = input_length
        self.proximal_bias = proximal_bias
        self.dropout_p = dropout_p
        self.attn = None
        self.k_channels = channels // num_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout_p)
        if rel_attn_window_size is not None:
            n_heads_rel = 1 if heads_share else num_heads
            rel_stddev = self.k_channels**-0.5
            emb_rel_k = nn.Parameter(
                torch.randn(n_heads_rel, rel_attn_window_size * 2 + 1, self.k_channels) * rel_stddev
            )
            emb_rel_v = nn.Parameter(
                torch.randn(n_heads_rel, rel_attn_window_size * 2 + 1, self.k_channels) * rel_stddev
            )
            self.register_parameter("emb_rel_k", emb_rel_k)
            self.register_parameter("emb_rel_v", emb_rel_v)

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        if proximal_init:
            self.conv_k.weight.data.copy_(self.conv_q.weight.data)
            self.conv_k.bias.data.copy_(self.conv_q.bias.data)
        nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)
        x, self.attn = self.attention(q, k, v, mask=attn_mask)
        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(b, self.num_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.num_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.num_heads, self.k_channels, t_s).transpose(2, 3)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.k_channels)
        if self.rel_attn_window_size is not None:
            assert t_s == t_t, "Relative attention is only available for self-attention."
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(query, key_relative_embeddings)
            rel_logits = self._relative_position_to_absolute_position(rel_logits)
            scores_local = rel_logits / math.sqrt(self.k_channels)
            scores = scores + scores_local
        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + self._attn_proximity_bias(t_s).to(
                device=scores.device, dtype=scores.dtype
            )
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
            if self.input_length is not None:
                block_mask = (
                    torch.ones_like(scores).triu(-1 * self.input_length).tril(self.input_length)
                )
                scores = scores * block_mask + -1e4 * (1 - block_mask)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        output = torch.matmul(p_attn, value)
        if self.rel_attn_window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
            output = output + self._matmul_with_relative_values(
                relative_weights, value_relative_embeddings
            )
        output = output.transpose(2, 3).contiguous().view(b, d, t_t)
        return output, p_attn

    @staticmethod
    def _matmul_with_relative_values(p_attn, re):
        logits = torch.matmul(p_attn, re.unsqueeze(0))
        return logits

    @staticmethod
    def _matmul_with_relative_keys(query, re):
        logits = torch.matmul(query, re.unsqueeze(0).transpose(-2, -1))
        return logits

    def _get_relative_embeddings(self, relative_embeddings, length):
        pad_length = max(length - (self.rel_attn_window_size + 1), 0)
        slice_start_position = max((self.rel_attn_window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded_relative_embeddings = F.pad(
                relative_embeddings, [0, 0, pad_length, pad_length, 0, 0]
            )
        else:
            padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[
            :, slice_start_position:slice_end_position
        ]
        return used_relative_embeddings

    @staticmethod
    def _relative_position_to_absolute_position(x):
        batch, heads, length, _ = x.size()
        x = F.pad(x, [0, 1, 0, 0, 0, 0, 0, 0])
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(x_flat, [0, length - 1, 0, 0, 0, 0])
        x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])[
            :, :, :length, length - 1 :
        ]
        return x_final

    @staticmethod
    def _absolute_position_to_relative_position(x):
        batch, heads, length, _ = x.size()
        x = F.pad(x, [0, length - 1, 0, 0, 0, 0, 0, 0])
        x_flat = x.view([batch, heads, length**2 + length * (length - 1)])
        x_flat = F.pad(x_flat, [length, 0, 0, 0, 0, 0])
        x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]
        return x_final

    @staticmethod
    def _attn_proximity_bias(length):
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        diff = -torch.log1p(torch.abs(diff))
        return diff.unsqueeze(0).unsqueeze(0)


class FeedForwardNetwork(nn.Module):
    def __init__(
        self, in_channels, out_channels, hidden_channels, kernel_size, dropout_p=0.0, causal=False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dropout_p = dropout_p

        if causal:
            self.padding = self._causal_padding
        else:
            self.padding = self._same_padding

        self.conv_1 = nn.Conv1d(in_channels, hidden_channels, kernel_size)
        self.conv_2 = nn.Conv1d(hidden_channels, out_channels, kernel_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, x_mask):
        x = self.conv_1(self.padding(x * x_mask))
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv_2(self.padding(x * x_mask))
        return x * x_mask

    def _causal_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = self.kernel_size - 1
        pad_r = 0
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, self._pad_shape(padding))
        return x

    def _same_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, self._pad_shape(padding))
        return x

    @staticmethod
    def _pad_shape(padding):
        inverted_padding = padding[::-1]
        pad_shape = [item for sublist in inverted_padding for item in sublist]
        return pad_shape


class RelativePositionTransformer(nn.Module):
    """Transformer with Relative Potional Encoding. https://arxiv.org/abs/1803.02155"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        hidden_channels_ffn: int,
        num_heads: int,
        num_layers: int,
        kernel_size=1,
        dropout_p=0.0,
        rel_attn_window_size: int = None,
        input_length: int = None,
        layer_norm_type: str = "1",
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.hidden_channels_ffn = hidden_channels_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout_p = dropout_p
        self.rel_attn_window_size = rel_attn_window_size

        self.dropout = nn.Dropout(dropout_p)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()

        for idx in range(self.num_layers):
            self.attn_layers.append(
                RelativePositionMultiHeadAttention(
                    hidden_channels if idx != 0 else in_channels,
                    hidden_channels,
                    num_heads,
                    rel_attn_window_size=rel_attn_window_size,
                    dropout_p=dropout_p,
                    input_length=input_length,
                )
            )
            if layer_norm_type == "1":
                self.norm_layers_1.append(LayerNorm(hidden_channels))
            elif layer_norm_type == "2":
                self.norm_layers_1.append(LayerNorm2(hidden_channels))
            else:
                raise ValueError(" [!] Unknown layer norm type")

            if hidden_channels != out_channels and (idx + 1) == self.num_layers:
                self.proj = nn.Conv1d(hidden_channels, out_channels, 1)

            self.ffn_layers.append(
                FeedForwardNetwork(
                    hidden_channels,
                    hidden_channels if (idx + 1) != self.num_layers else out_channels,
                    hidden_channels_ffn,
                    kernel_size,
                    dropout_p=dropout_p,
                )
            )

            if layer_norm_type == "1":
                self.norm_layers_2.append(
                    LayerNorm(hidden_channels if (idx + 1) != self.num_layers else out_channels)
                )
            elif layer_norm_type == "2":
                self.norm_layers_2.append(
                    LayerNorm2(hidden_channels if (idx + 1) != self.num_layers else out_channels)
                )
            else:
                raise ValueError(" [!] Unknown layer norm type")

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        for i in range(self.num_layers):
            x = x * x_mask
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.dropout(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.dropout(y)

            if (i + 1) == self.num_layers and hasattr(self, "proj"):
                x = self.proj(x)

            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x


# ---------------------------------------------------------------------------
# from TTS/tts/layers/vits/networks.py
# ---------------------------------------------------------------------------
class TextEncoder(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int,
        hidden_channels_ffn: int,
        num_heads: int,
        num_layers: int,
        kernel_size: int,
        dropout_p: float,
        language_emb_dim: int = None,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.emb = nn.Embedding(n_vocab, hidden_channels)

        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        if language_emb_dim:
            hidden_channels += language_emb_dim

        self.encoder = RelativePositionTransformer(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            hidden_channels_ffn=hidden_channels_ffn,
            num_heads=num_heads,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout_p=dropout_p,
            layer_norm_type="2",
            rel_attn_window_size=4,
        )

        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, lang_emb=None):
        assert x.shape[0] == x_lengths.shape[0]
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]

        if lang_emb is not None:
            x = torch.cat((x, lang_emb.transpose(2, 1).expand(x.size(0), x.size(1), -1)), dim=-1)

        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)  # [b, 1, t]

        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        num_layers,
        dropout_p=0,
        cond_channels=0,
        mean_only=False,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.half_channels = channels // 2
        self.mean_only = mean_only
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels,
            hidden_channels,
            kernel_size,
            dilation_rate,
            num_layers,
            dropout_p=dropout_p,
            c_in_channels=cond_channels,
        )
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, log_scale = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            log_scale = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(log_scale) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(log_scale, [1, 2])
            return x, logdet
        x1 = (x1 - m) * torch.exp(-log_scale) * x_mask
        x = torch.cat([x0, x1], 1)
        return x


class ResidualCouplingBlocks(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        num_layers: int,
        num_flows=4,
        cond_channels=0,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.num_layers = num_layers
        self.num_flows = num_flows
        self.cond_channels = cond_channels

        self.flows = nn.ModuleList()
        for _ in range(num_flows):
            self.flows.append(
                ResidualCouplingBlock(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    num_layers,
                    cond_channels=cond_channels,
                    mean_only=True,
                )
            )

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
                x = torch.flip(x, [1])
        else:
            for flow in reversed(self.flows):
                x = torch.flip(x, [1])
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class PosteriorEncoder(nn.Module):
    """x -> conv1x1() -> WaveNet() (non-causal) -> conv1x1() -> split() -> [m, s] -> sample(m, s) -> z"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        num_layers: int,
        cond_channels=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.num_layers = num_layers
        self.cond_channels = cond_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels,
            hidden_channels,
            kernel_size,
            dilation_rate,
            num_layers,
            c_in_channels=cond_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        mean, log_scale = torch.split(stats, self.out_channels, dim=1)
        z = (mean + torch.randn_like(mean) * torch.exp(log_scale)) * x_mask
        return z, mean, log_scale, x_mask


# ---------------------------------------------------------------------------
# from TTS/tts/layers/vits/transforms.py
# ---------------------------------------------------------------------------
DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def piecewise_rational_quadratic_transform(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails=None,
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    if tails is None:
        spline_fn = rational_quadratic_spline
        spline_kwargs = {}
    else:
        spline_fn = unconstrained_rational_quadratic_spline
        spline_kwargs = {"tails": tails, "tail_bound": tail_bound}

    outputs, logabsdet = spline_fn(
        inputs=inputs,
        unnormalized_widths=unnormalized_widths,
        unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=unnormalized_derivatives,
        inverse=inverse,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
        **spline_kwargs,
    )
    return outputs, logabsdet


def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def unconstrained_rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails="linear",
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    if tails == "linear":
        unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives[..., 0] = constant
        unnormalized_derivatives[..., -1] = constant

        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    else:
        raise RuntimeError("{} tails are not implemented.".format(tails))

    outputs[inside_interval_mask], logabsdet[inside_interval_mask] = rational_quadratic_spline(
        inputs=inputs[inside_interval_mask],
        unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
        unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
        inverse=inverse,
        left=-tail_bound,
        right=tail_bound,
        bottom=-tail_bound,
        top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )

    return outputs, logabsdet


def rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise ValueError("Input to a transform is not within its domain")

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, -logabsdet
    theta = (inputs - input_cumwidths) / input_bin_widths
    theta_one_minus_theta = theta * (1 - theta)

    numerator = input_heights * (
        input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
    )
    denominator = input_delta + (
        (input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta
    )
    outputs = input_cumheights + numerator / denominator

    derivative_numerator = input_delta.pow(2) * (
        input_derivatives_plus_one * theta.pow(2)
        + 2 * input_delta * theta_one_minus_theta
        + input_derivatives * (1 - theta).pow(2)
    )
    logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

    return outputs, logabsdet


# ---------------------------------------------------------------------------
# from TTS/tts/layers/vits/stochastic_duration_predictor.py
# ---------------------------------------------------------------------------
class DilatedDepthSeparableConv(nn.Module):
    """x |-> DDSConv(x) -> LayerNorm(x) -> GeLU(x) -> Conv1x1(x) -> LayerNorm(x) -> GeLU(x) -> + -> o"""

    def __init__(self, channels, kernel_size, num_layers, dropout_p=0.0):
        super().__init__()
        self.num_layers = num_layers

        self.convs_sep = nn.ModuleList()
        self.convs_1x1 = nn.ModuleList()
        self.norms_1 = nn.ModuleList()
        self.norms_2 = nn.ModuleList()
        for i in range(num_layers):
            dilation = kernel_size**i
            padding = (kernel_size * dilation - dilation) // 2
            self.convs_sep.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    groups=channels,
                    dilation=dilation,
                    padding=padding,
                )
            )
            self.convs_1x1.append(nn.Conv1d(channels, channels, 1))
            self.norms_1.append(LayerNorm2(channels))
            self.norms_2.append(LayerNorm2(channels))
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, x_mask, g=None):
        if g is not None:
            x = x + g
        for i in range(self.num_layers):
            y = self.convs_sep[i](x * x_mask)
            y = self.norms_1[i](y)
            y = F.gelu(y)
            y = self.convs_1x1[i](y)
            y = self.norms_2[i](y)
            y = F.gelu(y)
            y = self.dropout(y)
            x = x + y
        return x * x_mask


class ElementwiseAffine(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.translation = nn.Parameter(torch.zeros(channels, 1))
        self.log_scale = nn.Parameter(torch.zeros(channels, 1))

    def forward(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            y = (x * torch.exp(self.log_scale) + self.translation) * x_mask
            logdet = torch.sum(self.log_scale * x_mask, [1, 2])
            return y, logdet
        x = (x - self.translation) * torch.exp(-self.log_scale) * x_mask
        return x


class ConvFlow(nn.Module):
    """Dilated depth separable convolutional based spline flow."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
        num_layers: int,
        num_bins=10,
        tail_bound=5.0,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.hidden_channels = hidden_channels
        self.half_channels = in_channels // 2

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.convs = DilatedDepthSeparableConv(
            hidden_channels, kernel_size, num_layers, dropout_p=0.0
        )
        self.proj = nn.Conv1d(hidden_channels, self.half_channels * (num_bins * 3 - 1), 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0)
        h = self.convs(h, x_mask, g=g)
        h = self.proj(h) * x_mask

        b, c, t = x0.shape
        h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2)  # [b, cx?, t] -> [b, c, t, ?]

        unnormalized_widths = h[..., : self.num_bins] / math.sqrt(self.hidden_channels)
        unnormalized_heights = h[..., self.num_bins : 2 * self.num_bins] / math.sqrt(
            self.hidden_channels
        )
        unnormalized_derivatives = h[..., 2 * self.num_bins :]

        x1, logabsdet = piecewise_rational_quadratic_transform(
            x1,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=reverse,
            tails="linear",
            tail_bound=self.tail_bound,
        )

        x = torch.cat([x0, x1], 1) * x_mask
        logdet = torch.sum(logabsdet * x_mask, [1, 2])
        if not reverse:
            return x, logdet
        return x


class StochasticDurationPredictor(nn.Module):
    """Stochastic duration predictor with Spline Flows. arXiv:2106.06103."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dropout_p: float,
        num_flows=4,
        cond_channels=0,
        language_emb_dim=0,
    ):
        super().__init__()

        if language_emb_dim:
            in_channels += language_emb_dim

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.convs = DilatedDepthSeparableConv(
            hidden_channels, kernel_size, num_layers=3, dropout_p=dropout_p
        )
        self.proj = nn.Conv1d(hidden_channels, hidden_channels, 1)

        self.flows = nn.ModuleList()
        self.flows.append(ElementwiseAffine(2))
        self.flows += [
            ConvFlow(2, hidden_channels, kernel_size, num_layers=3) for _ in range(num_flows)
        ]

        self.post_pre = nn.Conv1d(1, hidden_channels, 1)
        self.post_convs = DilatedDepthSeparableConv(
            hidden_channels, kernel_size, num_layers=3, dropout_p=dropout_p
        )
        self.post_proj = nn.Conv1d(hidden_channels, hidden_channels, 1)

        self.post_flows = nn.ModuleList()
        self.post_flows.append(ElementwiseAffine(2))
        self.post_flows += [
            ConvFlow(2, hidden_channels, kernel_size, num_layers=3) for _ in range(num_flows)
        ]

        if cond_channels != 0 and cond_channels is not None:
            self.cond = nn.Conv1d(cond_channels, hidden_channels, 1)

        if language_emb_dim != 0 and language_emb_dim is not None:
            self.cond_lang = nn.Conv1d(language_emb_dim, hidden_channels, 1)

    def forward(self, x, x_mask, dr=None, g=None, lang_emb=None, reverse=False, noise_scale=1.0):
        x = self.pre(x)
        if g is not None:
            x = x + self.cond(g)

        if lang_emb is not None:
            x = x + self.cond_lang(lang_emb)

        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask

        if not reverse:
            flows = self.flows
            assert dr is not None

            h = self.post_pre(dr)
            h = self.post_convs(h, x_mask)
            h = self.post_proj(h) * x_mask
            noise = (
                torch.randn(dr.size(0), 2, dr.size(2)).to(device=x.device, dtype=x.dtype) * x_mask
            )
            z_q = noise

            logdet_tot_q = 0.0
            for idx, flow in enumerate(self.post_flows):
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h))
                logdet_tot_q = logdet_tot_q + logdet_q
                if idx > 0:
                    z_q = torch.flip(z_q, [1])

            z_u, z_v = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (dr - u) * x_mask

            logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2])
            nll_posterior_encoder = (
                torch.sum(-0.5 * (math.log(2 * math.pi) + (noise**2)) * x_mask, [1, 2])
                - logdet_tot_q
            )

            z0 = torch.log(torch.clamp_min(z0, 1e-5)) * x_mask
            logdet_tot = torch.sum(-z0, [1, 2])
            z = torch.cat([z0, z_v], 1)

            for idx, flow in enumerate(flows):
                z, logdet = flow(z, x_mask, g=x, reverse=reverse)
                logdet_tot = logdet_tot + logdet
                if idx > 0:
                    z = torch.flip(z, [1])

            nll_flow_layers = (
                torch.sum(0.5 * (math.log(2 * math.pi) + (z**2)) * x_mask, [1, 2]) - logdet_tot
            )
            return nll_flow_layers + nll_posterior_encoder

        flows = list(reversed(self.flows))
        flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
        z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale
        for flow in flows:
            z = torch.flip(z, [1])
            z = flow(z, x_mask, g=x, reverse=reverse)

        z0, _ = torch.split(z, [1, 1], 1)
        logw = z0
        return logw


# ---------------------------------------------------------------------------
# from TTS/tts/layers/glow_tts/duration_predictor.py (not used by YourTTS's
# use_sdp=True default; ported for architectural completeness)
# ---------------------------------------------------------------------------
class DurationPredictor(nn.Module):
    """Glow-TTS duration prediction model: [2 x (conv1d_kxk -> relu -> layer_norm -> dropout)] -> conv1d_1x1 -> durs"""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        kernel_size,
        dropout_p,
        cond_channels=None,
        language_emb_dim=None,
    ):
        super().__init__()

        if language_emb_dim:
            in_channels += language_emb_dim

        self.in_channels = in_channels
        self.filter_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dropout_p = dropout_p
        self.drop = nn.Dropout(dropout_p)
        self.conv_1 = nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = LayerNorm(hidden_channels)
        self.conv_2 = nn.Conv1d(
            hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = LayerNorm(hidden_channels)
        self.proj = nn.Conv1d(hidden_channels, 1, 1)
        if cond_channels is not None and cond_channels != 0:
            self.cond = nn.Conv1d(cond_channels, in_channels, 1)

        if language_emb_dim != 0 and language_emb_dim is not None:
            self.cond_lang = nn.Conv1d(language_emb_dim, in_channels, 1)

    def forward(self, x, x_mask, g=None, lang_emb=None):
        if g is not None:
            x = x + self.cond(g)

        if lang_emb is not None:
            x = x + self.cond_lang(lang_emb)

        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


# ---------------------------------------------------------------------------
# from TTS/vocoder/models/hifigan_generator.py
# ---------------------------------------------------------------------------
def get_padding(k, d):
    return int((k * d - d) / 2)


class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for layer in self.convs1:
            remove_parametrizations(layer, "weight")
        for layer in self.convs2:
            remove_parametrizations(layer, "weight")


class ResBlock2(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
            ]
        )

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for layer in self.convs:
            remove_parametrizations(layer, "weight")


class HifiganGenerator(torch.nn.Module):
    """HiFiGAN Generator with Multi-Receptive Field Fusion (MRF)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        resblock_type,
        resblock_dilation_sizes,
        resblock_kernel_sizes,
        upsample_kernel_sizes,
        upsample_initial_channel,
        upsample_factors,
        inference_padding=5,
        cond_channels=0,
        conv_pre_weight_norm=True,
        conv_post_weight_norm=True,
        conv_post_bias=True,
    ):
        super().__init__()
        self.inference_padding = inference_padding
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_factors)
        self.conv_pre = weight_norm(Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if resblock_type == "1" else ResBlock2
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_factors, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for _, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))
        self.conv_post = weight_norm(Conv1d(ch, out_channels, 7, 1, padding=3, bias=conv_post_bias))
        if cond_channels > 0:
            self.cond_layer = nn.Conv1d(cond_channels, upsample_initial_channel, 1)

        if not conv_pre_weight_norm:
            remove_parametrizations(self.conv_pre, "weight")

        if not conv_post_weight_norm:
            remove_parametrizations(self.conv_post, "weight")

    def forward(self, x, g=None):
        o = self.conv_pre(x)
        if hasattr(self, "cond_layer"):
            o = o + self.cond_layer(g)
        for i in range(self.num_upsamples):
            o = F.leaky_relu(o, LRELU_SLOPE)
            o = self.ups[i](o)
            z_sum = None
            for j in range(self.num_kernels):
                if z_sum is None:
                    z_sum = self.resblocks[i * self.num_kernels + j](o)
                else:
                    z_sum += self.resblocks[i * self.num_kernels + j](o)
            o = z_sum / self.num_kernels
        o = F.leaky_relu(o)
        o = self.conv_post(o)
        o = torch.tanh(o)
        return o

    def remove_weight_norm(self):
        for layer in self.ups:
            remove_parametrizations(layer, "weight")
        for layer in self.resblocks:
            layer.remove_weight_norm()
        remove_parametrizations(self.conv_pre, "weight")
        remove_parametrizations(self.conv_post, "weight")


# ---------------------------------------------------------------------------
# from TTS/tts/models/vits.py -- module wiring + eval-mode inference(), the
# real YourTTS/VITS architecture stripped of the Coqpit config object,
# trainer/data-loading plumbing, and training-only code paths (train_step,
# forward_mas, checkpoint I/O). Every nn.Module below is the real class the
# paper contributes; only the constructor no longer reads from a Coqpit
# config object (values are passed directly, matching the exact values the
# official YourTTS recipe's `VitsArgs(...)`/`VitsArgs` defaults resolve to).
# ---------------------------------------------------------------------------
class YourTTSModel(nn.Module):
    """VITS conditional-VAE + normalizing-flow + HiFi-GAN TTS model, configured
    with YourTTS's real external-d-vector speaker conditioning."""

    def __init__(
        self,
        num_chars: int,
        out_channels: int = 513,
        hidden_channels: int = 192,
        hidden_channels_ffn_text_encoder: int = 768,
        num_heads_text_encoder: int = 2,
        num_layers_text_encoder: int = 10,  # YourTTS recipe: 10 (vs VITS-base's 6)
        kernel_size_text_encoder: int = 3,
        dropout_p_text_encoder: float = 0.1,
        kernel_size_posterior_encoder: int = 5,
        dilation_rate_posterior_encoder: int = 1,
        num_layers_posterior_encoder: int = 16,
        kernel_size_flow: int = 5,
        dilation_rate_flow: int = 1,
        num_layers_flow: int = 4,
        resblock_type_decoder: str = "2",  # YourTTS recipe: "2" (accidental swap, kept faithfully)
        resblock_kernel_sizes_decoder=(3, 7, 11),
        resblock_dilation_sizes_decoder=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        upsample_rates_decoder=(8, 8, 2, 2),
        upsample_initial_channel_decoder: int = 512,
        upsample_kernel_sizes_decoder=(16, 16, 4, 4),
        use_sdp: bool = True,
        length_scale: float = 1,
        inference_noise_scale: float = 0.667,
        inference_noise_scale_dp: float = 1.0,
        dropout_p_duration_predictor: float = 0.5,
        d_vector_dim: int = 512,  # YourTTS recipe: external speaker-encoder d-vectors
        condition_dp_on_speaker: bool = True,
        max_inference_len: int = None,
    ):
        super().__init__()

        self.length_scale = length_scale
        self.inference_noise_scale = inference_noise_scale
        self.inference_noise_scale_dp = inference_noise_scale_dp
        self.max_inference_len = max_inference_len
        self.use_sdp = use_sdp
        self.condition_dp_on_speaker = condition_dp_on_speaker

        # `use_d_vector_file=True` -> no learned speaker-embedding nn.Embedding table;
        # `embedded_speaker_dim` is the external d-vector width and `g` is supplied by
        # the caller directly (see forward()/YourTTS's `Vits._set_cond_input`).
        self.embedded_speaker_dim = d_vector_dim

        self.text_encoder = TextEncoder(
            num_chars,
            hidden_channels,
            hidden_channels,
            hidden_channels_ffn_text_encoder,
            num_heads_text_encoder,
            num_layers_text_encoder,
            kernel_size_text_encoder,
            dropout_p_text_encoder,
            language_emb_dim=0,
        )

        self.posterior_encoder = PosteriorEncoder(
            out_channels,
            hidden_channels,
            hidden_channels,
            kernel_size=kernel_size_posterior_encoder,
            dilation_rate=dilation_rate_posterior_encoder,
            num_layers=num_layers_posterior_encoder,
            cond_channels=self.embedded_speaker_dim,
        )

        self.flow = ResidualCouplingBlocks(
            hidden_channels,
            hidden_channels,
            kernel_size=kernel_size_flow,
            dilation_rate=dilation_rate_flow,
            num_layers=num_layers_flow,
            cond_channels=self.embedded_speaker_dim,
        )

        if use_sdp:
            self.duration_predictor = StochasticDurationPredictor(
                hidden_channels,
                192,
                3,
                dropout_p_duration_predictor,
                4,
                cond_channels=self.embedded_speaker_dim if condition_dp_on_speaker else 0,
                language_emb_dim=0,
            )
        else:
            self.duration_predictor = DurationPredictor(
                hidden_channels,
                256,
                3,
                dropout_p_duration_predictor,
                cond_channels=self.embedded_speaker_dim,
                language_emb_dim=0,
            )

        self.waveform_decoder = HifiganGenerator(
            hidden_channels,
            1,
            resblock_type_decoder,
            list(resblock_dilation_sizes_decoder),
            list(resblock_kernel_sizes_decoder),
            list(upsample_kernel_sizes_decoder),
            upsample_initial_channel_decoder,
            list(upsample_rates_decoder),
            inference_padding=0,
            cond_channels=self.embedded_speaker_dim,
            conv_pre_weight_norm=False,
            conv_post_weight_norm=False,
            conv_post_bias=False,
        )

    @staticmethod
    def _set_cond_input(d_vectors):
        """Set the speaker conditioning input based on the multi-speaker mode.
        (`Vits._set_cond_input`, specialized to the d-vector-only code path.)"""
        g = F.normalize(d_vectors).unsqueeze(-1)
        if g.ndim == 2:
            g = g.unsqueeze_(0)
        return g

    @torch.no_grad()
    def forward(self, x, d_vectors, x_lengths=None):
        """Real `Vits.inference()` eval-mode forward path.

        Shapes:
            - x: :math:`[B, T_seq]` (token ids)
            - d_vectors: :math:`[B, C]` (external speaker-encoder embeddings)
        Returns:
            - waveform: :math:`[B, 1, T_wav]`
        """
        g = self._set_cond_input(d_vectors)
        if x_lengths is None:
            x_lengths = torch.tensor(x.shape[1:2]).to(x.device)

        x, m_p, logs_p, x_mask = self.text_encoder(x, x_lengths, lang_emb=None)

        if self.use_sdp:
            logw = self.duration_predictor(
                x,
                x_mask,
                g=g if self.condition_dp_on_speaker else None,
                reverse=True,
                noise_scale=self.inference_noise_scale_dp,
                lang_emb=None,
            )
        else:
            logw = self.duration_predictor(
                x, x_mask, g=g if self.condition_dp_on_speaker else None, lang_emb=None
            )
        w = torch.exp(logw) * x_mask * self.length_scale

        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = sequence_mask(y_lengths, None).to(x_mask.dtype).unsqueeze(1)  # [B, 1, T_dec]

        attn_mask = x_mask * y_mask.transpose(1, 2)  # [B, 1, T_enc] * [B, T_dec, 1]
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1).transpose(1, 2))

        m_p = torch.matmul(attn.transpose(1, 2), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.transpose(1, 2), logs_p.transpose(1, 2)).transpose(1, 2)

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * self.inference_noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)

        o = self.waveform_decoder((z * y_mask)[:, :, : self.max_inference_len], g=g)
        return o


# ---------------------------------------------------------------------------
# menagerie build/example glue
# ---------------------------------------------------------------------------
def build_yourtts() -> nn.Module:
    """Build a tiny real YourTTS (VITS + external d-vector conditioning) for
    tracing: small vocab/hidden/flow-depth/text-encoder-length so the forward
    pass (and its unrolled trace) is fast."""

    model = YourTTSModel(
        num_chars=32,
        out_channels=8,
        hidden_channels=16,
        hidden_channels_ffn_text_encoder=32,
        num_heads_text_encoder=2,
        num_layers_text_encoder=2,
        kernel_size_posterior_encoder=5,
        num_layers_posterior_encoder=2,
        num_layers_flow=1,
        resblock_kernel_sizes_decoder=(3, 7),
        resblock_dilation_sizes_decoder=((1, 3), (1, 3)),
        upsample_rates_decoder=(8, 8),
        upsample_initial_channel_decoder=16,
        upsample_kernel_sizes_decoder=(16, 16),
        d_vector_dim=32,
    )
    model.eval()
    return model


def example_yourtts() -> tuple[torch.Tensor, torch.Tensor]:
    """(token ids [B, T_seq], external speaker d-vector [B, C])."""

    x = torch.randint(0, 32, (1, 12))
    d_vectors = torch.randn(1, 32)
    return x, d_vectors


MENAGERIE_ENTRIES = [
    ("YourTTS", "build_yourtts", "example_yourtts", "2022", "speech/audio"),
]
