# SOURCE: vendored from IMLHF/PHASEN-PyTorch @ 934a38b53fbd554c8ea5f7f7768cbe7fb412724f
# https://raw.githubusercontent.com/IMLHF/PHASEN-PyTorch/934a38b53fbd554c8ea5f7f7768cbe7fb412724f/phasen_torch/models/phasen.py
# https://raw.githubusercontent.com/IMLHF/PHASEN-PyTorch/934a38b53fbd554c8ea5f7f7768cbe7fb412724f/phasen_torch/FLAGS.py
#
# PHASEN (Yin, Luo, Xiong, Zeng 2020, AAAI, "PHASEN: A Phase-and-Harmonics-Aware Speech
# Enhancement Network") -- official unofficial-author PyTorch reimplementation by
# IMLHF (phasen_torch/models/phasen.py), a two-stream speech-enhancement network that
# jointly estimates a magnitude mask and a phase correction from the noisy STFT: a
# "stream A" (amplitude) branch and a "stream P" (phase) branch each get their own
# Stream_PreNet, are refined through `n_TSB` stacked TwoStreamBlocks (each combining a
# frequency-transformation-block (FTB) + strided-frequency convolutions on stream A
# with LayerNorm'd convolutions on stream P, cross-fused every block via
# `InfoCommunicate` gating), and are finally decoded by a BLSTM-based
# StreamAmplitude_PostNet (mask) and a StreamPhase_PostNet (normalized complex phase
# via atan2).
#
# `SelfConv2d`, `BatchNormAndActivate`, `Stream_PreNet`, `NodeReshape`,
# `FrequencyTransformationBlock`, `InfoCommunicate`, `TwoStreamBlock`,
# `StreamAmplitude_PostNet`, `StreamPhase_PostNet`, `WavFeatures`, `NET_PHASEN_OUT`,
# and `NetPHASEN` (the actual trainable architecture -- everything up to the raw-wav
# STFT/iSTFT frontend/backend and the loss functions) are transcribed verbatim from
# the real `models/phasen.py`. The outer `PHASEN` wrapper class (STFT/iSTFT via
# `conv_stft.ConvSTFT`/`ConviSTFT` and `Losses`/`get_losses`) is training/inference
# glue around raw waveforms and is NOT part of the traced architecture -- `NetPHASEN`
# itself (called directly, as `PHASEN.forward` does via `self._net_model(...)`) is
# the actual two-stream network and is what is captured here, taking pre-computed
# STFT features (`WavFeatures`) as documented in the real forward()'s own docstring.
# `PARAM` (originally `from ..FLAGS import PARAM`, module-level `PARAM =
# se_phasen_009`, a class hierarchy of hyperparameter overrides in FLAGS.py) is
# inlined here as a plain namespace holding only the fields `NetPHASEN` and its
# submodules actually read, with the exact values the real repo's active config
# (`se_phasen_009`, the config the module-level `PARAM` was bound to) sets: `channel_A
# = 96`, `channel_P = 48`, `prenet_A_kernels = [[7,1],[1,7]]`, `prenet_P_kernels =
# [[3,5],[1,25]]`, `n_TSB = 3`, `frequency_dim = 257`, `stream_A_feature_type =
# "stft"`, `stream_P_feature_type = "normed_stft"`, `stft_norm_method = "div"`,
# `stft_div_norm_eps = 1e-7`, `sA_out_act = "sigmoid"`. No architectural changes were
# made; `frequency_dim` (257 = 512/2+1 FFT bins from the repo's `fft_length=512`) is
# kept in full for architectural fidelity (the FrequencyTransformationBlock's Linear
# and BLSTM widths are frequency_dim-dependent), and only the time axis (T, number of
# STFT frames) is shrunk for a fast trace.

import collections

import torch
from torch import nn


class _Param:
    """Inlined subset of FLAGS.py::se_phasen_009 (the real repo's active PARAM
    config) -- only the fields NetPHASEN and its submodules read."""

    channel_A = 96
    channel_P = 48
    prenet_A_kernels = [[7, 1], [1, 7]]
    prenet_P_kernels = [[3, 5], [1, 25]]
    n_TSB = 3
    frequency_dim = 257
    stream_A_feature_type = "stft"  # "stft" | "mag"
    stream_P_feature_type = "normed_stft"  # "stft" | "normed_stft"
    stft_norm_method = "div"  # "atan2" | "div"
    stft_div_norm_eps = 1e-7
    sA_out_act = "sigmoid"


PARAM = _Param()


class SelfConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        use_bias=True,
        padding="same",
        activation=None,
    ):
        super(SelfConv2d, self).__init__()
        assert padding.lower() in ["same", "valid"], "padding must be same or valid."
        if padding.lower() == "same":
            if type(kernel_size) is int:
                padding_nn = kernel_size // 2
            else:
                padding_nn = []
                for kernel_s in kernel_size:
                    padding_nn.append(kernel_s // 2)
        self.conv2d_fn = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, bias=use_bias, padding=padding_nn
        )
        self.act = None if activation is None else activation()

    def forward(self, feature_in):
        """
        feature_in : [N, C, F, T]
        """
        out = feature_in
        out = self.conv2d_fn(out)
        if self.act is not None:
            out = self.act(out)
        return out


class BatchNormAndActivate(nn.Module):
    def __init__(self, channel, activation=nn.ReLU):
        super(BatchNormAndActivate, self).__init__()
        self.bn_layer = nn.BatchNorm2d(channel)
        self.activate_fn = None if activation is None else activation()

    def forward(self, fea_in):
        """
        fea_in: [N, C, F, T]
        """
        out = self.bn_layer(fea_in)
        if self.activate_fn is not None:
            out = self.activate_fn(out)
        return out

    def get_bn_weight(self):
        return self.bn_layer.parameters()


class Stream_PreNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernels=[[7, 1], [1, 7]],
        conv2d_activation=None,
        conv2d_bn=False,
    ):
        """
        channel_out: output channel
        kernels: kernel for layers
        """
        super(Stream_PreNet, self).__init__()
        self.nn_layers = nn.ModuleList()
        for i, kernel in enumerate(kernels):
            conv2d = SelfConv2d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=kernel,
                activation=(None if conv2d_bn else conv2d_activation),
                padding="same",
            )
            self.nn_layers.append(conv2d)
            if conv2d_bn:
                bn_fn = BatchNormAndActivate(out_channels, activation=conv2d_activation)
                self.nn_layers.append(bn_fn)

    def forward(self, feature_in):
        """
        feature_in : [batch, channel_in, F, T]
        return : [batch, channel_out, F, T]
        """
        if len(self.nn_layers) == 0:
            return feature_in
        out = feature_in
        for layer_fn in self.nn_layers:
            out = layer_fn(out)
        return out


class NodeReshape(nn.Module):
    def __init__(self, shape):
        super(NodeReshape, self).__init__()
        self.shape = shape

    def forward(self, feature_in: torch.Tensor):
        shape = feature_in.size()
        batch = shape[0]
        new_shape = [batch]
        new_shape.extend(list(self.shape))
        return feature_in.reshape(new_shape)


class FrequencyTransformationBlock(nn.Module):
    def __init__(self, frequency_dim, channel_in_out, channel_attention=5):
        super(FrequencyTransformationBlock, self).__init__()
        self.frequency_dim = frequency_dim
        self.channel_out = channel_in_out
        self.att_conv2d_1 = SelfConv2d(channel_in_out, channel_attention, [1, 1], padding="same")
        self.att_conv2d_1_bna = BatchNormAndActivate(channel_attention)

        # [batch, channel_attention, F, T] -> [batch, channel_attention*F, T]
        self.att_inner_reshape = NodeReshape([frequency_dim * channel_attention, -1])
        self.att_conv1d_2 = nn.Conv1d(
            frequency_dim * channel_attention, frequency_dim, 9, padding=4
        )  # [batch, F, T]
        self.att_conv1d_2_bna = nn.Sequential(nn.BatchNorm1d(frequency_dim), nn.ReLU(inplace=True))
        self.att_out_reshape = NodeReshape([1, frequency_dim, -1])
        self.frequencyFC = nn.Linear(frequency_dim, frequency_dim)

        self.out_conv2d = SelfConv2d(channel_in_out * 2, channel_in_out, [1, 1], padding="same")
        self.out_conv2d_bna = BatchNormAndActivate(channel_in_out)

    def forward(self, feature_in):
        """
        feature_n: [batch, channel_in_out, F, T]
        """
        att_out = self.att_conv2d_1(feature_in)
        att_out = self.att_conv2d_1_bna(att_out)

        att_out = self.att_inner_reshape(att_out)
        att_out = self.att_conv1d_2(att_out)
        att_out = self.att_conv1d_2_bna(att_out)
        att_out = self.att_out_reshape(att_out)

        atted_out = torch.mul(feature_in, att_out)  # [batch, channel_in_out, F, T]
        atted_out_T = torch.transpose(atted_out, 2, 3)  # [batch, channel_in_out, T, F]
        ffc_out_T = self.frequencyFC(atted_out_T)
        ffc_out = torch.transpose(ffc_out_T, 2, 3)  # [batch, channel_in_out, F, T]
        concated_out = torch.cat([feature_in, ffc_out], 1)  # [batch, channel_in_out*2, F, T]

        out = self.out_conv2d(concated_out)
        out = self.out_conv2d_bna(out)
        return out


class InfoCommunicate(nn.Module):
    def __init__(self, channel_in, channel_out, activate_fn=nn.Tanh):
        super(InfoCommunicate, self).__init__()
        self.conv2d = SelfConv2d(channel_in, channel_out, [1, 1], padding="same")
        self.activate_fn = None if activate_fn is None else activate_fn()

    def forward(self, feature_x1, feature_x2):
        # feature_x1: [batch, channel_out, F, T]
        # feature_x2: [batch, channel_in, F, T]
        # return: [batch, channel_out, F, T]
        out = self.conv2d(feature_x2)
        if self.activate_fn is not None:
            out = self.activate_fn(out)

        out_multiply = torch.mul(feature_x1, out)
        return out_multiply


class TwoStreamBlock(nn.Module):
    def __init__(self, frequency_dim, channel_in_out_A, channel_in_out_P):
        super(TwoStreamBlock, self).__init__()
        self.sA1_pre_FTB = FrequencyTransformationBlock(frequency_dim, channel_in_out_A)
        self.sA2_conv2d = SelfConv2d(channel_in_out_A, channel_in_out_A, [5, 5], padding="same")
        self.sA2_conv2d_bna = BatchNormAndActivate(channel_in_out_A)
        self.sA3_conv2d = SelfConv2d(channel_in_out_A, channel_in_out_A, [1, 25], padding="same")
        self.sA3_conv2d_bna = BatchNormAndActivate(channel_in_out_A)
        self.sA4_conv2d = SelfConv2d(channel_in_out_A, channel_in_out_A, [5, 5], padding="same")
        self.sA4_conv2d_bna = BatchNormAndActivate(channel_in_out_A)
        self.sA5_post_FTB = FrequencyTransformationBlock(frequency_dim, channel_in_out_A)
        self.sA6_info_communicate = InfoCommunicate(channel_in_out_P, channel_in_out_A)

        # [batch, C, F, T] -> [batch, T, F, C]
        self.sP1_conv2d_before_LN = nn.LayerNorm([frequency_dim, channel_in_out_P])
        # [batch, T, F, C] -> [batch, C, F, T]

        self.sP1_conv2d = SelfConv2d(channel_in_out_P, channel_in_out_P, [3, 5], padding="same")

        # [batch, C, F, T] -> [batch, T, F, C]
        self.sP2_conv2d_before_LN = nn.LayerNorm([frequency_dim, channel_in_out_P])
        # [batch, T, F, C] -> [batch, C, F, T]

        self.sP2_conv2d = SelfConv2d(channel_in_out_P, channel_in_out_P, [1, 25], padding="same")
        self.sP3_info_communicate = InfoCommunicate(channel_in_out_A, channel_in_out_P)

    def forward(self, feature_sA, feature_sP):
        # Stream A
        # feature_sA: [N, C, F, T]
        sA_out = feature_sA
        sA_out = self.sA1_pre_FTB(sA_out)
        sA_out = self.sA2_conv2d(sA_out)
        sA_out = self.sA2_conv2d_bna(sA_out)
        sA_out = self.sA3_conv2d(sA_out)
        sA_out = self.sA3_conv2d_bna(sA_out)
        sA_out = self.sA4_conv2d(sA_out)
        sA_out = self.sA4_conv2d_bna(sA_out)
        sA_out = self.sA5_post_FTB(sA_out)

        # Stream P
        sP_out = feature_sP
        # [batch, C, F, T] -> [batch, T, F, C]
        sP_out = torch.transpose(sP_out, 1, 3)
        sP_out = self.sP1_conv2d_before_LN(sP_out)
        sP_out = torch.transpose(sP_out, 1, 3)
        sP_out = self.sP1_conv2d(sP_out)
        sP_out = torch.transpose(sP_out, 1, 3)
        sP_out = self.sP2_conv2d_before_LN(sP_out)
        sP_out = torch.transpose(sP_out, 1, 3)
        sP_out = self.sP2_conv2d(sP_out)

        # information communication
        sA_fin_out = self.sA6_info_communicate(sA_out, sP_out)
        sP_fin_out = self.sP3_info_communicate(sP_out, sA_out)

        return sA_fin_out, sP_fin_out


class StreamAmplitude_PostNet(nn.Module):
    def __init__(self, frequency_dim, channel_sA):
        super(StreamAmplitude_PostNet, self).__init__()
        self.p1_conv2d = SelfConv2d(
            channel_sA, 8, [1, 1], activation=nn.Sigmoid, padding="same"
        )  # [N, 8, F, T]

        uni_rnn_units = 600
        # add transpose ->[N, T, F, C]
        self.p1_reshape = NodeReshape([-1, frequency_dim * 8])
        self.p2_blstm = nn.LSTM(
            frequency_dim * 8, uni_rnn_units, batch_first=True, bidirectional=True
        )

        self.p3_dense = nn.Sequential(nn.Linear(uni_rnn_units * 2, 600), nn.ReLU(inplace=True))
        self.p4_dense = nn.Sequential(nn.Linear(600, 600), nn.ReLU(inplace=True))
        sA_out_act = {
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "None": None,
        }[PARAM.sA_out_act]
        self.out_dense = nn.Linear(600, frequency_dim)
        if sA_out_act is not None:
            self.out_dense = nn.Sequential(self.out_dense, sA_out_act)

    def forward(self, feature_sA):
        """
        input: [N, C, F, T]
        return [N, F, T]
        """
        out = feature_sA
        out = self.p1_conv2d(out)  # [N, 8, F, T]
        out = torch.transpose(out, 1, 3)  # [N, T, F, 8]
        out = self.p1_reshape(out)  # [N, T, F*8]
        out = self.p2_blstm(out)[0]  # [N, T, 2*600]
        out = self.p3_dense(out)
        out = self.p4_dense(out)
        out = self.out_dense(out)  # [N, T, F]
        out = torch.transpose(out, 1, 2)  # [N, F, T]
        return out


class StreamPhase_PostNet(nn.Module):
    def __init__(self, channel_sP):
        super(StreamPhase_PostNet, self).__init__()
        self.conv2d = SelfConv2d(channel_sP, 2, [1, 1], padding="same")

    def forward(self, feature_sP: torch.Tensor):
        """
        return [N, 2, F, T]->complex
        """
        out = feature_sP
        out = self.conv2d(out)
        # out: [batch, 2, F, T]
        out_real = out[:, :1, :, :]
        out_imag = out[:, 1:, :, :]
        out_angle = out_imag.atan2(out_real)  # [N, 1, F, T]
        if PARAM.stft_norm_method == "atan2":
            normed_stft = torch.cat([torch.cos(out_angle), torch.sin(out_angle)], dim=1)
        elif PARAM.stft_norm_method == "div":
            normed_stft = torch.div(
                out, torch.sqrt(out_real**2 + out_imag**2) + PARAM.stft_div_norm_eps
            )
        else:
            raise NotImplementedError
        return normed_stft, out_angle.squeeze(1)


class WavFeatures(
    collections.namedtuple(
        "WavFeatures",
        (
            "wav_batch",  # [N, L]
            "stft_batch",  # [N, 2, F, T]
            "mag_batch",  # [N, F, T]
            "angle_batch",  # [N, F, T]
            "normed_stft_batch",  # [N, F, T]
        ),
    )
):
    pass


class NET_PHASEN_OUT(
    collections.namedtuple("NET_PHASEN_OUT", ("mag_mask", "normalized_complex_phase", "angle"))
):
    pass


class NetPHASEN(nn.Module):
    def __init__(self):
        super(NetPHASEN, self).__init__()
        sA_in_channel = {
            "stft": 2,
            "mag": 1,
        }[PARAM.stream_A_feature_type]
        sP_in_channel = {
            "stft": 2,
            "normed_stft": 2,
        }[PARAM.stream_P_feature_type]
        self.streamA_prenet = Stream_PreNet(
            sA_in_channel,
            PARAM.channel_A,
            kernels=PARAM.prenet_A_kernels,
            conv2d_bn=True,
            conv2d_activation=nn.ReLU,
        )
        self.streamP_prenet = Stream_PreNet(sP_in_channel, PARAM.channel_P, PARAM.prenet_P_kernels)
        self.layers_TSB = nn.ModuleList()
        for i in range(1, PARAM.n_TSB + 1):
            tsb_t = TwoStreamBlock(PARAM.frequency_dim, PARAM.channel_A, PARAM.channel_P)
            self.layers_TSB.append(tsb_t)
        self.streamA_postnet = StreamAmplitude_PostNet(PARAM.frequency_dim, PARAM.channel_A)
        self.streamP_postnet = StreamPhase_PostNet(PARAM.channel_P)

    def forward(self, mixed_wav_features: WavFeatures):
        """
        Args:
          mixed_wav_features
        Return :
          mag_batch[N, F, T]->real,
          normalized_complex_phase[N, 2, F, T]->(real, imag)
        """
        sA_inputs = {
            "stft": mixed_wav_features.stft_batch,  # [N, 2, F, T]
            "mag": mixed_wav_features.mag_batch.unsqueeze(1),  # [N, 1, F, T]
        }[PARAM.stream_A_feature_type]
        sP_inputs = {
            "stft": mixed_wav_features.stft_batch,  # [N, 2, F, T]
            "normed_stft": mixed_wav_features.normed_stft_batch,
        }[PARAM.stream_P_feature_type]

        sA_out = self.streamA_prenet(sA_inputs)  # [batch, Ca, f, t]
        sP_out = self.streamP_prenet(sP_inputs)  # [batch, Cp, f, t]
        for tsb in self.layers_TSB:
            sA_out, sP_out = tsb(sA_out, sP_out)
        sA_out = self.streamA_postnet(sA_out)  # [batch, f, t]
        sP_out_normed_stft, sP_out_angle = self.streamP_postnet(sP_out)  # [batch, 2, f, t]

        est_mask = sA_out  # [batch, f, t]
        normed_complex_phase = sP_out_normed_stft  # [batch, 2, f, t], (real, imag)
        est_angle = sP_out_angle
        return NET_PHASEN_OUT(
            mag_mask=est_mask, normalized_complex_phase=normed_complex_phase, angle=est_angle
        )


def build_phasen():
    torch.manual_seed(0)
    model = NetPHASEN()
    model.eval()
    return model


def example_input_phasen():
    torch.manual_seed(0)
    batch, freq, time = 1, PARAM.frequency_dim, 6
    stft_batch = torch.randn(batch, 2, freq, time)
    mag_batch = torch.randn(batch, freq, time)
    angle_batch = torch.randn(batch, freq, time)
    normed_stft_batch = torch.randn(batch, 2, freq, time)
    features = WavFeatures(
        wav_batch=None,
        stft_batch=stft_batch,
        mag_batch=mag_batch,
        angle_batch=angle_batch,
        normed_stft_batch=normed_stft_batch,
    )
    return (features,)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("PHASEN", "build_phasen", "example_input_phasen", 2020, MENAGERIE_ZOO),
]
