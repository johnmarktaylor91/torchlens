# SOURCE: vendored from https://github.com/SoulProficiency/speechseparation-GALR @ main (GALR.py)
#
# GALR: "Effective Low-Cost Time-Domain Audio Separation Using Globally
# Attentive Locally Recurrent Networks" (Rixin Luo et al., arXiv:2101.05014).
# https://github.com/SoulProficiency/speechseparation-GALR
#
# The classes below are the REAL GALR model code, copied verbatim from the
# official repo's GALR.py (the module's `pad_segment`/`split_feature`
# implementation is itself carried over from the DPTNet dual-path-transformer
# repo, per the original author's own docstring credit). Only the
# `if __name__ == "__main__"` CLI/argparse-free smoke-test block at the
# bottom of the original file was dropped -- it is not part of the traced
# forward path. No architecture was altered.

import torch.nn as nn
import torch
from torch.autograd import Variable
import math
from torch.nn.modules.activation import MultiheadAttention

MENAGERIE_ZOO = "vendored-pytorch"


# ------------------------------------------------over_lap_and_add----------------------------------------
def overlap_and_add(signal, frame_step):
    """
    Author: Kaituo XU
    :param signal:
    :param frame_step:
    :return:
    """
    """Reconstructs a signal from a framed representation.

    Adds potentially overlapping frames of a signal with shape
    `[..., frames, frame_length]`, offsetting subsequent frames by `frame_step`.
    The resulting tensor has shape `[..., output_size]` where

        output_size = (frames - 1) * frame_step + frame_length

    Args:
        signal: A [..., frames, frame_length] Tensor. All dimensions may be unknown, and rank must be at least 2.
        frame_step: An integer denoting overlap offsets. Must be less than or equal to frame_length.

    Returns:
        A Tensor with shape [..., output_size] containing the overlap-added frames of signal's inner-most two dimensions.
        output_size = (frames - 1) * frame_step + frame_length

    Based on https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/signal/python/ops/reconstruction_ops.py
    """
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(frame_length, frame_step)  # gcd=Greatest Common Divisor
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.reshape(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes).unfold(0, subframes_per_frame, subframe_step)
    frame = signal.new_tensor(frame).long()  # signal may in GPU or CPU
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result


# ------------------------------------------------2.1 encodering raw signals-------------------------------
class Encoder(nn.Module):
    """
    INPUT [B,T] B is batch,L is length of every input

    :arg channel_size ->the D-dimesional of 1D gated convolutional
    """

    def __init__(self, basic_signal, feature_dim, kernel_dim=2):
        super(Encoder, self).__init__()
        self.kernel_dim = kernel_dim
        self.feature_dim = feature_dim
        self.ConvLayer1 = nn.Conv1d(
            1,
            self.feature_dim,
            kernel_size=self.kernel_dim,
            stride=self.kernel_dim // 2,
            padding=0,
            dilation=1,
            bias=False,
        )
        self.ReLU = nn.ReLU()

    def forward(self, input):
        # in papers,author promise the result form conv has the same feature dim with input's
        input = input.unsqueeze(dim=1)  # ->[B,1,T] FOR 1D CONV

        # take basic signal into high dims
        conv_out = self.ConvLayer1(input)
        # ReLu active
        out = self.ReLU(conv_out)  # non-negative [B,N,T]
        return out

    # in papers author don't mention the para normal,we can choose the parameters
    @staticmethod
    def normal_para(input):
        pass


# ----------------------------------------------2.2 GALR Blocks--------------------------------------
class Separator(nn.Module):
    """
    note: pad_segment,splite_feature,merge_feature come from dual path transformer
    github:https://github.com/ujscjj/DPTNet

    :args
    segment_size to split our input
    num_layer is the num_layer in local_recurrent
    """

    def __init__(self, basic_signal, feature_dim, hidden_size, num_layer, segment_size):
        super(Separator, self).__init__()
        self.basic_signal = basic_signal
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.segment_size = segment_size

        self.local_recurrent = local_recurrent(self.feature_dim, self.hidden_size, self.num_layer)

        self.Global_attentive = Global_attentive(feature_dim)

    def pad_segment(self, input, segment_size):
        # input is the features: (B, N, T)
        batch_size, dim, seq_len = input.shape
        # 50% overlap
        segment_stride = segment_size // 2

        # calculate the rest length if the inputs can not divided by stride
        rest = segment_size - (segment_stride + seq_len % segment_size) % segment_size
        if rest > 0:
            # zero pad for segment
            pad = Variable(torch.zeros(batch_size, dim, rest)).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(batch_size, dim, segment_stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest

    def split_feature(self, input, segment_size):
        # split the feature into chunks of segment size
        # input is the features: (B, N, T)

        input, rest = self.pad_segment(input, segment_size)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        #  torch.cat([pad_aux, input, pad_aux(ignore)], 2)
        segments1 = (
            input[:, :, :-segment_stride].contiguous().view(batch_size, dim, -1, segment_size)
        )
        #  torch.cat([pad_aux(ignore), input, pad_aux], 2)
        segments2 = (
            input[:, :, segment_stride:].contiguous().view(batch_size, dim, -1, segment_size)
        )
        segments = (
            torch.cat([segments1, segments2], 3)
            .view(batch_size, dim, -1, segment_size)
            .transpose(2, 3)
        )
        # segments [batch,dim,segment_size,segment_num]
        return segments.contiguous(), rest

    def merge_feature(self, input, rest):
        # merge the splitted features into full utterance
        # input is the features: (B, N, L, K)
        # B*nspk, N, L, K
        # the opposite operation of padding
        batch_size, dim, segment_size, _ = input.shape
        segment_stride = segment_size // 2
        # B*2, N, K, L*2
        input = (
            input.transpose(2, 3).contiguous().view(batch_size, dim, -1, segment_size * 2)
        )  # B, N, K, L

        # get speaker1
        input1 = (
            input[:, :, :, :segment_size]
            .contiguous()
            .view(batch_size, dim, -1)[:, :, segment_stride:]
        )  # ignore the zero pad
        # get speaker2
        input2 = (
            input[:, :, :, segment_size:]
            .contiguous()
            .view(batch_size, dim, -1)[:, :, :-segment_stride]
        )
        output = input1 + input2
        if rest > 0:
            output = output[:, :, :-rest]

        return output.contiguous()  # B, N, T

    def forward(self, input):
        pass


class GALR_Block(Separator):
    def __init__(self, *args, **kwargs):
        """
        SUCCEED FROM Separator ,all parameters define in Separator
        """
        super(GALR_Block, self).__init__(*args, **kwargs)

    def forward(self, input):
        """
        :param input: [B,N,T] from encoder
        :return:
        """
        # segment
        enc_segments, enc_rest = self.split_feature(
            input, self.segment_size
        )  # B, D, L, K: L is the segment_size
        B, D, L, K = enc_segments.shape

        PE = poision_encoding(L * K, 8000)
        position_information = PE(enc_segments)

        # i'm not sure the position encoding
        position = torch.zeros(0)
        for i in range(self.feature_dim):
            position = torch.cat([position, position_information], dim=1)
        position = position.view(B, D, L, -1)
        # local recurrent
        output = self.local_recurrent(enc_segments)  # [B,D,L,K]
        output = output + position

        # global attentive
        out = self.Global_attentive(output)
        out = self.merge_feature(out, enc_rest)
        # [B,D,T]

        return out


class local_recurrent(nn.Module):
    def __init__(self, feature_dim, hidden_size, num_layer, Bi_LSTM=True, batch_first=True):
        """
        input [B,D,L,K] L is segment size
        """
        super(local_recurrent, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.bi_lstm = Bi_LSTM
        self.batch_first = batch_first
        self.num_layer = num_layer
        self.Lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layer,
            batch_first=True,
            bidirectional=True,
        )
        self.liner = nn.Linear(2 * hidden_size, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, input):
        # intra_LSTM
        inputs = input
        B, D, L, K = inputs.shape
        input = input.permute(0, 3, 2, 1).contiguous().view(B * K, L, -1)  # [B,K,L,D]

        inputs = inputs.permute(0, 3, 2, 1).contiguous()  # [B,K,L,D]
        inputs = inputs.view(B * K, L, -1)
        local_recurrent, _ = self.Lstm(inputs)
        liner_out = self.liner(local_recurrent)  # [B*K,L,D]
        out = liner_out + input
        out = self.norm(out)
        out = out.view(B, K, L, -1).permute(0, 3, 2, 1).contiguous()  # [B,D,L,K]
        return out


class Global_attentive(nn.Module):
    def __init__(self, feature_dim, nhead=4, dropout=True):
        """
        input [B,D,L,K] L is segment size
        """
        super(Global_attentive, self).__init__()
        self.attn = MultiheadAttention(feature_dim, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout()

    def forward(self, input):
        B, D, L, K = input.shape
        input = input.permute(0, 2, 3, 1).contiguous()  # [B,L,K,D]
        input = input.view(B * L, K, -1)  # [B*L,K,D]
        output = self.attn(input, input, input, attn_mask=None, key_padding_mask=None)[0]
        out = self.norm(self.dropout(output) + input)
        out = out.view(B, L, K, D)
        out = out.permute(0, 3, 1, 2).contiguous()
        return out


class poision_encoding(nn.Module):
    def __init__(self, dim_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, dim_model)
        po = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2) * (-math.log(10000.0) / dim_model))
        pe[:, 0::2] = torch.sin(po * div_term)
        pe[:, 1::2] = torch.cos(po * div_term)
        self.register_buffer("pe", pe)

    def forward(self, input):
        l, *_ = input.shape  # noqa: E741 (verbatim upstream variable name)
        return self.pe[:l, :].unsqueeze(1)


# --------------------------------------------2.3 signals Reconstruction------------------------
class Mask_estimation(nn.Module):
    """
    input [B,D,T]->2DCONV->[B,C,D,T]->[B*C,D,T]->[B*C,N,T]

    """

    def __init__(self, feature_dim, basic_signal, nspk):
        super(Mask_estimation, self).__init__()
        self.nspk = nspk
        self.feature_dim = feature_dim
        # the number of basic signals
        self.basic_signal = basic_signal
        self.conv_layer = nn.Conv2d(1, self.nspk, kernel_size=1, stride=1, bias=False)
        self.conv_tanh = nn.Conv1d(
            self.feature_dim, self.feature_dim, kernel_size=1, stride=1, bias=False
        )
        self.tanh = nn.Tanh()
        self.conv_sigmoid = nn.Conv1d(
            self.feature_dim, self.feature_dim, kernel_size=1, stride=1, bias=False
        )
        self.sigmoid = nn.Sigmoid()
        self.conv_relu = nn.Conv1d(self.feature_dim, self.feature_dim, kernel_size=1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, input):
        B, D, T = input.shape
        input = input.unsqueeze(dim=1)  # [B,1,D,T]
        conv2d_out = self.conv_layer(input)  # [B,2,D,T]
        est_src = conv2d_out.view(B * self.nspk, D, -1)  # [B*2,D,T]
        conv_tanh = self.tanh(self.conv_tanh(est_src))
        conv_sigmoid = self.sigmoid(self.conv_sigmoid(est_src))
        est_mask_c = conv_tanh * conv_sigmoid
        mask_c = self.relu(self.conv_relu(est_mask_c))
        mask_c = mask_c.view(B, self.nspk, -1, T)
        return mask_c  # [B,nspk,N,T]


# ----------------------------------------Decoder---------------------------------------
class Decoder(nn.Module):
    def __init__(self, basic_signal, feature_dim):
        super(Decoder, self).__init__()
        self.basic_signal = basic_signal
        self.feature_dim = feature_dim
        self.liner = nn.Linear(self.feature_dim, self.basic_signal)

    def forward(self, mixture_w, est_mask):
        """
        :param input: [B,N,T]
        :param mask_c: [B,C,N,T]
        :return:
        """
        source_w = torch.unsqueeze(mixture_w, 1) * est_mask  # [B, C, N, L]

        est_source = torch.transpose(source_w, 2, 3)  # [B, C, L, N]
        # don't be confused by the name of basic_signal,it just use for convenience
        # we want to use the function had been written before and do the galr_block for n times!
        # please set basic_signal is 2 to avoid error!
        est_source = self.liner(est_source)
        est_source = overlap_and_add(est_source, self.basic_signal // 2)  # B x C x T
        return est_source


class GALR_model(nn.Module):
    def __init__(
        self, basic_signal, feature_dim, hidden_size, num_layer, segment_size, nspk, galr_block_num
    ):
        super(GALR_model, self).__init__()
        self.basic_signal = basic_signal
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.segment_size = segment_size
        self.nspk = nspk
        self.galr_block_num = galr_block_num

        self.encoder = Encoder(basic_signal=self.basic_signal, feature_dim=self.feature_dim)
        self.Galr_blcok = GALR_Block(
            self.basic_signal, self.feature_dim, self.hidden_size, self.num_layer, self.segment_size
        )
        GALR = []
        for i in range(self.galr_block_num):
            GALR.append(self.Galr_blcok)
        self.GALR = nn.Sequential(*GALR)
        self.est_mask = Mask_estimation(self.feature_dim, self.basic_signal, self.nspk)

        self.decoder = Decoder(self.basic_signal, self.feature_dim)

    def forward(self, inputs):
        encoder_out = self.encoder(inputs)
        GALR_out = self.GALR(encoder_out)
        est_mask_out = self.est_mask(GALR_out)
        decoder_out = self.decoder(encoder_out, est_mask_out)

        return decoder_out

    @classmethod
    def load_model(cls, path):
        # Load to CPU
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls.load_model_from_package(package)
        return model

    @classmethod
    def load_model_from_package(cls, package):
        model = cls(
            package["basic_signal"],
            package["feature_dim"],
            package["hidden_size"],
            package["num_layer"],
            package["segment_size"],
            package["nspk"],
            package["galr_block_num"],
        )
        model.load_state_dict(package["state_dict"])
        return model

    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):
        package = {
            # hyper-parameter
            "basic_signal": model.basic_signal,
            "feature_dim": model.feature_dim,
            "hidden_size": model.hidden_size,
            "num_layer": model.num_layer,
            "segment_size": model.segment_size,
            "nspk": model.nspk,
            "galr_block_num": model.galr_block_num,
            # state
            "state_dict": model.state_dict(),
            "optim_dict": optimizer.state_dict(),
            "epoch": epoch,
        }
        if tr_loss is not None:
            package["tr_loss"] = tr_loss
            package["cv_loss"] = cv_loss
        return package


# ---------------------------------------------------------------------------
# Tiny random-init build/example for TorchLens tracing. The repo's own
# `if __name__ == "__main__"` block uses
# GALR_model(basic_signal=2, feature_dim=64, hidden_size=128, num_layer=2,
#            segment_size=64, nspk=2, galr_block_num=6) fed torch.ones(1, 1024).
# We shrink feature_dim/hidden_size/segment_size/galr_block_num for a fast
# CPU trace while keeping basic_signal=2 and nspk=2 (the repo's Decoder
# docstring explicitly requires basic_signal=2 "to avoid error").
# ---------------------------------------------------------------------------
def build_galr():
    torch.manual_seed(0)
    model = GALR_model(
        basic_signal=2,
        feature_dim=8,
        hidden_size=8,
        num_layer=1,
        segment_size=8,
        nspk=2,
        galr_block_num=1,
    )
    model.eval()
    return model


def example_input_galr():
    torch.manual_seed(0)
    # [B, T] raw waveform samples.
    return torch.randn(1, 128)


MENAGERIE_ENTRIES = [
    ("GALR", "build_galr", "example_input_galr", 2021, MENAGERIE_ZOO),
]
