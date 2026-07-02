# SOURCE: vendored from yluo42/TAC @ master
# Files: FaSNet.py + utility/models.py
# https://github.com/yluo42/TAC/blob/master/FaSNet.py
# https://github.com/yluo42/TAC/blob/master/utility/models.py
#
# Two model classes are defined in FaSNet.py:
#   - FaSNet_origin: the original two-stage FaSNet beamformer (Luo et al. 2019, arXiv:1909.13387)
#   - FaSNet_TAC: single-stage FaSNet with Transform-Average-Concatenate (TAC) cross-channel module
# Both share the DPRNN-based beamforming machinery in utility/models.py, vendored below unmodified
# apart from import-path fixes (relative `utility.models` import removed; classes inlined in one file).
#
# MENAGERIE_ZOO = "vendored-pytorch"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# vendored from utility/models.py
# ---------------------------------------------------------------------------


class SingleRNN(nn.Module):
    """
    Container module for a single RNN layer.

    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        dropout: float, dropout ratio. Default is 0.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, rnn_type, input_size, hidden_size, dropout=0, bidirectional=False):
        super(SingleRNN, self).__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1

        self.rnn = getattr(nn, rnn_type)(
            input_size,
            hidden_size,
            1,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # linear projection layer
        self.proj = nn.Linear(hidden_size * self.num_direction, input_size)

    def forward(self, input):
        # input shape: batch, seq, dim
        output = input
        rnn_output, _ = self.rnn(output)
        rnn_output = self.proj(rnn_output.contiguous().view(-1, rnn_output.shape[2])).view(
            output.shape
        )
        return rnn_output


# dual-path RNN
class DPRNN(nn.Module):
    """
    Deep duaL-path RNN.

    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_layers: int, number of stacked RNN layers. Default is 1.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(
        self,
        rnn_type,
        input_size,
        hidden_size,
        output_size,
        dropout=0,
        num_layers=1,
        bidirectional=True,
    ):
        super(DPRNN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # dual-path RNN
        self.row_rnn = nn.ModuleList([])
        self.col_rnn = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        for i in range(num_layers):
            self.row_rnn.append(
                SingleRNN(rnn_type, input_size, hidden_size, dropout, bidirectional=True)
            )  # intra-segment RNN is always noncausal
            self.col_rnn.append(
                SingleRNN(rnn_type, input_size, hidden_size, dropout, bidirectional=bidirectional)
            )
            self.row_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))
            # default is to use noncausal LayerNorm for inter-chunk RNN. For causal setting change it to causal normalization techniques accordingly.
            self.col_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))

        # output layer
        self.output = nn.Sequential(nn.PReLU(), nn.Conv2d(input_size, output_size, 1))

    def forward(self, input):
        # input shape: batch, N, dim1, dim2
        # apply RNN on dim1 first and then dim2

        batch_size, _, dim1, dim2 = input.shape
        output = input
        for i in range(len(self.row_rnn)):
            row_input = (
                output.permute(0, 3, 2, 1).contiguous().view(batch_size * dim2, dim1, -1)
            )  # B*dim2, dim1, N
            row_output = self.row_rnn[i](row_input)  # B*dim2, dim1, H
            row_output = (
                row_output.view(batch_size, dim2, dim1, -1).permute(0, 3, 2, 1).contiguous()
            )  # B, N, dim1, dim2
            row_output = self.row_norm[i](row_output)
            output = output + row_output

            col_input = (
                output.permute(0, 2, 3, 1).contiguous().view(batch_size * dim1, dim2, -1)
            )  # B*dim1, dim2, N
            col_output = self.col_rnn[i](col_input)  # B*dim1, dim2, H
            col_output = (
                col_output.view(batch_size, dim1, dim2, -1).permute(0, 3, 1, 2).contiguous()
            )  # B, N, dim1, dim2
            col_output = self.col_norm[i](col_output)
            output = output + col_output

        output = self.output(output)

        return output


# dual-path RNN with transform-average-concatenate (TAC)
class DPRNN_TAC(nn.Module):
    """
    Deep duaL-path RNN with transform-average-concatenate (TAC) applied to each layer/block.

    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_layers: int, number of stacked RNN layers. Default is 1.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(
        self,
        rnn_type,
        input_size,
        hidden_size,
        output_size,
        dropout=0,
        num_layers=1,
        bidirectional=True,
    ):
        super(DPRNN_TAC, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # DPRNN + TAC for 3D input (ch, N, T)
        self.row_rnn = nn.ModuleList([])
        self.col_rnn = nn.ModuleList([])
        self.ch_transform = nn.ModuleList([])
        self.ch_average = nn.ModuleList([])
        self.ch_concat = nn.ModuleList([])

        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        self.ch_norm = nn.ModuleList([])

        for i in range(num_layers):
            self.row_rnn.append(
                SingleRNN(rnn_type, input_size, hidden_size, dropout, bidirectional=True)
            )  # intra-segment RNN is always noncausal
            self.col_rnn.append(
                SingleRNN(rnn_type, input_size, hidden_size, dropout, bidirectional=bidirectional)
            )
            self.ch_transform.append(
                nn.Sequential(nn.Linear(input_size, hidden_size * 3), nn.PReLU())
            )
            self.ch_average.append(
                nn.Sequential(nn.Linear(hidden_size * 3, hidden_size * 3), nn.PReLU())
            )
            self.ch_concat.append(nn.Sequential(nn.Linear(hidden_size * 6, input_size), nn.PReLU()))

            self.row_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))
            # default is to use noncausal LayerNorm for inter-chunk RNN and TAC modules. For causal setting change them to causal normalization techniques accordingly.
            self.col_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))
            self.ch_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))

        # output layer
        self.output = nn.Sequential(nn.PReLU(), nn.Conv2d(input_size, output_size, 1))

    def forward(self, input, num_mic):
        # input shape: batch, ch, N, dim1, dim2
        # num_mic shape: batch,
        # apply RNN on dim1 first, then dim2, then ch

        batch_size, ch, N, dim1, dim2 = input.shape
        output = input
        for i in range(len(self.row_rnn)):
            # intra-segment RNN
            output = output.view(batch_size * ch, N, dim1, dim2)  # B*ch, N, dim1, dim2
            row_input = (
                output.permute(0, 3, 2, 1).contiguous().view(batch_size * ch * dim2, dim1, -1)
            )  # B*ch*dim2, dim1, N
            row_output = self.row_rnn[i](row_input)  # B*ch*dim2, dim1, N
            row_output = (
                row_output.view(batch_size * ch, dim2, dim1, -1).permute(0, 3, 2, 1).contiguous()
            )  # B*ch, N, dim1, dim2
            row_output = self.row_norm[i](row_output)
            output = output + row_output  # B*ch, N, dim1, dim2

            # inter-segment RNN
            col_input = (
                output.permute(0, 2, 3, 1).contiguous().view(batch_size * ch * dim1, dim2, -1)
            )  # B*ch*dim1, dim2, N
            col_output = self.col_rnn[i](col_input)  # B*dim1, dim2, N
            col_output = (
                col_output.view(batch_size * ch, dim1, dim2, -1).permute(0, 3, 1, 2).contiguous()
            )  # B*ch, N, dim1, dim2
            col_output = self.col_norm[i](col_output)
            output = output + col_output  # B*ch, N, dim1, dim2

            # TAC for cross-channel communication
            ch_input = output.view(input.shape)  # B, ch, N, dim1, dim2
            ch_input = ch_input.permute(0, 3, 4, 1, 2).contiguous().view(-1, N)  # B*dim1*dim2*ch, N
            ch_output = self.ch_transform[i](ch_input).view(
                batch_size, dim1 * dim2, ch, -1
            )  # B, dim1*dim2, ch, H
            # mean pooling across channels
            if num_mic.max() == 0:
                # fixed geometry array
                ch_mean = ch_output.mean(2).view(batch_size * dim1 * dim2, -1)  # B*dim1*dim2, H
            else:
                # only consider valid channels
                ch_mean = [
                    ch_output[b, :, : num_mic[b]].mean(1).unsqueeze(0) for b in range(batch_size)
                ]  # 1, dim1*dim2, H
                ch_mean = torch.cat(ch_mean, 0).view(batch_size * dim1 * dim2, -1)  # B*dim1*dim2, H
            ch_output = ch_output.view(batch_size * dim1 * dim2, ch, -1)  # B*dim1*dim2, ch, H
            ch_mean = (
                self.ch_average[i](ch_mean).unsqueeze(1).expand_as(ch_output).contiguous()
            )  # B*dim1*dim2, ch, H
            ch_output = torch.cat([ch_output, ch_mean], 2)  # B*dim1*dim2, ch, 2H
            ch_output = self.ch_concat[i](
                ch_output.view(-1, ch_output.shape[-1])
            )  # B*dim1*dim2*ch, N
            ch_output = (
                ch_output.view(batch_size, dim1, dim2, ch, -1).permute(0, 3, 4, 1, 2).contiguous()
            )  # B, ch, N, dim1, dim2
            ch_output = self.ch_norm[i](
                ch_output.view(batch_size * ch, N, dim1, dim2)
            )  # B*ch, N, dim1, dim2
            output = output + ch_output

        output = self.output(output)  # B*ch, N, dim1, dim2

        return output


# base module for deep DPRNN
class DPRNN_base(nn.Module):
    def __init__(
        self,
        input_dim,
        feature_dim,
        hidden_dim,
        output_dim,
        num_spk=2,
        layer=4,
        segment_size=100,
        bidirectional=True,
        model_type="DPRNN",
        rnn_type="LSTM",
    ):
        super(DPRNN_base, self).__init__()

        assert model_type in ["DPRNN", "DPRNN_TAC"], (
            "model_type can only be 'DPRNN' or 'DPRNN_TAC'."
        )

        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.layer = layer
        self.segment_size = segment_size
        self.num_spk = num_spk

        self.model_type = model_type

        self.eps = 1e-8

        # bottleneck
        self.BN = nn.Conv1d(self.input_dim, self.feature_dim, 1, bias=False)

        # DPRNN model
        _dprnn_cls = {"DPRNN": DPRNN, "DPRNN_TAC": DPRNN_TAC}[model_type]
        self.DPRNN = _dprnn_cls(
            rnn_type,
            self.feature_dim,
            self.hidden_dim,
            self.feature_dim * self.num_spk,
            num_layers=layer,
            bidirectional=bidirectional,
        )

    def pad_segment(self, input, segment_size):
        # input is the features: (B, N, T)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        rest = segment_size - (segment_stride + seq_len % segment_size) % segment_size
        if rest > 0:
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

        segments1 = (
            input[:, :, :-segment_stride].contiguous().view(batch_size, dim, -1, segment_size)
        )
        segments2 = (
            input[:, :, segment_stride:].contiguous().view(batch_size, dim, -1, segment_size)
        )
        segments = (
            torch.cat([segments1, segments2], 3)
            .view(batch_size, dim, -1, segment_size)
            .transpose(2, 3)
        )

        return segments.contiguous(), rest

    def merge_feature(self, input, rest):
        # merge the splitted features into full utterance
        # input is the features: (B, N, L, K)

        batch_size, dim, segment_size, _ = input.shape
        segment_stride = segment_size // 2
        input = (
            input.transpose(2, 3).contiguous().view(batch_size, dim, -1, segment_size * 2)
        )  # B, N, K, L

        input1 = (
            input[:, :, :, :segment_size]
            .contiguous()
            .view(batch_size, dim, -1)[:, :, segment_stride:]
        )
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


# ---------------------------------------------------------------------------
# vendored from FaSNet.py
# ---------------------------------------------------------------------------


# DPRNN for beamforming filter estimation
class BF_module(DPRNN_base):
    def __init__(self, *args, **kwargs):
        super(BF_module, self).__init__(*args, **kwargs)

        # gated output layer
        self.output = nn.Sequential(nn.Conv1d(self.feature_dim, self.output_dim, 1), nn.Tanh())
        self.output_gate = nn.Sequential(
            nn.Conv1d(self.feature_dim, self.output_dim, 1), nn.Sigmoid()
        )

    def forward(self, input, num_mic):
        if self.model_type == "DPRNN":
            # input: (B, N, T)
            batch_size, N, seq_length = input.shape
            ch = 1
        elif self.model_type == "DPRNN_TAC":
            # input: (B, ch, N, T)
            batch_size, ch, N, seq_length = input.shape

        input = input.view(batch_size * ch, N, seq_length)  # B*ch, N, T
        enc_feature = self.BN(input)

        # split the encoder output into overlapped, longer segments
        enc_segments, enc_rest = self.split_feature(enc_feature, self.segment_size)  # B*ch, N, L, K

        # pass to DPRNN
        if self.model_type == "DPRNN":
            output = self.DPRNN(enc_segments).view(
                batch_size * ch * self.num_spk, self.feature_dim, self.segment_size, -1
            )  # B*ch*nspk, N, L, K
        elif self.model_type == "DPRNN_TAC":
            enc_segments = enc_segments.view(
                batch_size, ch, -1, enc_segments.shape[2], enc_segments.shape[3]
            )  # B, ch, N, L, K
            output = self.DPRNN(enc_segments, num_mic).view(
                batch_size * ch * self.num_spk, self.feature_dim, self.segment_size, -1
            )  # B*ch*nspk, N, L, K

        # overlap-and-add of the outputs
        output = self.merge_feature(output, enc_rest)  # B*ch*nspk, N, T

        # gated output layer for filter generation
        bf_filter = self.output(output) * self.output_gate(output)  # B*ch*nspk, K, T
        bf_filter = (
            bf_filter.transpose(1, 2)
            .contiguous()
            .view(batch_size, ch, self.num_spk, -1, self.output_dim)
        )  # B, ch, nspk, L, N

        return bf_filter


# base module for FaSNet
class FaSNet_base(nn.Module):
    def __init__(
        self,
        enc_dim,
        feature_dim,
        hidden_dim,
        layer,
        segment_size=50,
        nspk=2,
        win_len=4,
        context_len=16,
        sr=16000,
    ):
        super(FaSNet_base, self).__init__()

        # parameters
        self.window = int(sr * win_len / 1000)
        self.context = int(sr * context_len / 1000)
        self.stride = self.window // 2

        self.filter_dim = self.context * 2 + 1
        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.segment_size = segment_size

        self.layer = layer
        self.num_spk = nspk
        self.eps = 1e-8

        # waveform encoder
        self.encoder = nn.Conv1d(1, self.enc_dim, self.context * 2 + self.window, bias=False)
        self.enc_LN = nn.GroupNorm(1, self.enc_dim, eps=1e-8)

    def pad_input(self, input, window):
        """
        Zero-padding input according to window/stride size.
        """
        batch_size, nmic, nsample = input.shape
        stride = window // 2

        # pad the signals at the end for matching the window/stride size
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            pad = torch.zeros(batch_size, nmic, rest).type(input.type())
            input = torch.cat([input, pad], 2)
        pad_aux = torch.zeros(batch_size, nmic, stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest

    def seg_signal_context(self, x, window, context):
        """
        Segmenting the signal into chunks with specific context.
        input:
            x: size (B, ch, T)
            window: int
            context: int

        """

        # pad input accordingly
        # first pad according to window size
        input, rest = self.pad_input(x, window)
        batch_size, nmic, nsample = input.shape
        stride = window // 2

        # pad another context size
        pad_context = torch.zeros(batch_size, nmic, context).type(input.type())
        input = torch.cat([pad_context, input, pad_context], 2)  # B, ch, L

        # calculate index for each chunk
        nchunk = 2 * nsample // window - 1
        begin_idx = np.arange(nchunk) * stride
        begin_idx = (
            torch.from_numpy(begin_idx).type(input.type()).long().view(1, 1, -1)
        )  # 1, 1, nchunk
        begin_idx = begin_idx.expand(batch_size, nmic, nchunk)  # B, ch, nchunk
        # select entries from index
        chunks = [
            torch.gather(input, 2, begin_idx + i).unsqueeze(3) for i in range(2 * context + window)
        ]  # B, ch, nchunk, 1
        chunks = torch.cat(chunks, 3)  # B, ch, nchunk, chunk_size

        # center frame
        center_frame = chunks[:, :, :, context : context + window]

        return center_frame, chunks, rest

    def seq_cos_sim(self, ref, target):
        """
        Cosine similarity between some reference mics and some target mics
        ref: shape (nmic1, L, seg1)
        target: shape (nmic2, L, seg2)
        """

        assert ref.size(1) == target.size(1), "Inputs should have same length."
        assert ref.size(2) >= target.size(2), (
            "Reference input should be no smaller than the target input."
        )

        seq_length = ref.size(1)

        larger_ch = ref.size(0)
        if target.size(0) > ref.size(0):
            ref = ref.expand(
                target.size(0), ref.size(1), ref.size(2)
            ).contiguous()  # nmic2, L, seg1
            larger_ch = target.size(0)
        elif target.size(0) < ref.size(0):
            target = target.expand(
                ref.size(0), target.size(1), target.size(2)
            ).contiguous()  # nmic1, L, seg2

        # L2 norms
        ref_norm = F.conv1d(
            ref.view(1, -1, ref.size(2)).pow(2),
            torch.ones(ref.size(0) * ref.size(1), 1, target.size(2)).type(ref.type()),
            groups=larger_ch * seq_length,
        )  # 1, larger_ch*L, seg1-seg2+1
        ref_norm = ref_norm.sqrt() + self.eps
        target_norm = target.norm(2, dim=2).view(1, -1, 1) + self.eps  # 1, larger_ch*L, 1
        # cosine similarity
        cos_sim = F.conv1d(
            ref.view(1, -1, ref.size(2)),
            target.view(-1, 1, target.size(2)),
            groups=larger_ch * seq_length,
        )  # 1, larger_ch*L, seg1-seg2+1
        cos_sim = cos_sim / (ref_norm * target_norm)

        return cos_sim.view(larger_ch, seq_length, -1)

    def forward(self, input, num_mic):
        """
        input: shape (batch, max_num_ch, T)
        num_mic: shape (batch, ), the number of channels for each input. Zero for fixed geometry configuration.
        """
        pass


# original FaSNet
class FaSNet_origin(FaSNet_base):
    def __init__(self, *args, **kwargs):
        super(FaSNet_origin, self).__init__(*args, **kwargs)

        # DPRNN for ref mic
        self.ref_BF = BF_module(
            self.filter_dim + self.enc_dim,
            self.feature_dim,
            self.hidden_dim,
            self.filter_dim,
            self.num_spk,
            self.layer,
            self.segment_size,
            model_type="DPRNN",
        )

        # DPRNN for other mics
        self.other_BF = BF_module(
            self.filter_dim + self.enc_dim,
            self.feature_dim,
            self.hidden_dim,
            self.filter_dim,
            1,
            self.layer,
            self.segment_size,
            model_type="DPRNN",
        )

    def forward(self, input, num_mic):
        batch_size = input.size(0)
        nmic = input.size(1)

        # split input into chunks
        all_seg, all_mic_context, rest = self.seg_signal_context(
            input, self.window, self.context
        )  # B, nmic, L, win/chunk
        seq_length = all_seg.size(2)

        # first step: filtering the ref mic to create a clean estimate
        # calculate cosine similarity
        ref_context = (
            all_mic_context[:, 0].contiguous().view(1, -1, self.context * 2 + self.window)
        )  # 1, B*L, 3*win
        other_segment = (
            all_seg[:, 1:].contiguous().transpose(0, 1).contiguous().view(nmic - 1, -1, self.window)
        )  # nmic-1, B*L, win
        ref_cos_sim = self.seq_cos_sim(ref_context, other_segment)  # nmic-1, B*L, 2*win+1
        ref_cos_sim = ref_cos_sim.view(
            nmic - 1, batch_size, seq_length, self.filter_dim
        )  # nmic-1, B, L, 2*win+1
        if num_mic.max() == 0:
            ref_cos_sim = ref_cos_sim.mean(0)  # B, L, 2*win+1
            ref_cos_sim = ref_cos_sim.transpose(1, 2).contiguous()  # B, 2*win+1, L
        else:
            # consider only the valid channels
            ref_cos_sim = [
                ref_cos_sim[: num_mic[b], b, :].mean(0).unsqueeze(0) for b in range(batch_size)
            ]  # 1, L, 2*win+1
            ref_cos_sim = torch.cat(ref_cos_sim, 0).transpose(1, 2).contiguous()  # B, 2*win+1, L

        # pass to a DPRNN
        ref_feature = (
            all_mic_context[:, 0]
            .contiguous()
            .view(batch_size * seq_length, 1, self.context * 2 + self.window)
        )
        ref_feature = self.encoder(ref_feature)  # B*L, N, 1
        ref_feature = (
            ref_feature.view(batch_size, seq_length, self.enc_dim).transpose(1, 2).contiguous()
        )  # B, N, L
        ref_filter = self.ref_BF(
            torch.cat([self.enc_LN(ref_feature), ref_cos_sim], 1), num_mic
        )  # B, 1, nspk, L, 2*win+1

        # convolve with ref mic context segments
        ref_context = torch.cat(
            [all_mic_context[:, 0].unsqueeze(1)] * self.num_spk, 1
        )  # B, nspk, L, 3*win
        ref_output = F.conv1d(
            ref_context.view(1, -1, self.context * 2 + self.window),
            ref_filter.view(-1, 1, self.filter_dim),
            groups=batch_size * self.num_spk * seq_length,
        )  # 1, B*nspk*L, win
        ref_output = ref_output.view(
            batch_size * self.num_spk, seq_length, self.window
        )  # B*nspk, L, win

        # second step: use the ref output as the cue, beamform other mics
        # calculate cosine similarity
        other_context = torch.cat(
            [all_mic_context[:, 1:].unsqueeze(1)] * self.num_spk, 1
        )  # B, nspk, nmic-1, L, 3*win
        other_context_saved = other_context.view(
            batch_size * self.num_spk, nmic - 1, seq_length, self.context * 2 + self.window
        )  # B*nspk, nmic-1, L, 3*win
        other_context = (
            other_context_saved.transpose(0, 1)
            .contiguous()
            .view(nmic - 1, -1, self.context * 2 + self.window)
        )  # nmic-1, B*nspk*L, 3*win
        ref_segment = ref_output.view(1, -1, self.window)  # 1, B*nspk*L, win
        other_cos_sim = self.seq_cos_sim(other_context, ref_segment)  # nmic-1, B*nspk*L, 2*win+1
        other_cos_sim = other_cos_sim.view(
            nmic - 1, batch_size * self.num_spk, seq_length, self.filter_dim
        )  # nmic-1, B*nspk, L, 2*win+1
        other_cos_sim = (
            other_cos_sim.permute(1, 0, 3, 2).contiguous().view(-1, self.filter_dim, seq_length)
        )  # B*nspk*(nmic-1), 2*win+1, L

        # pass to another DPRNN
        other_feature = self.encoder(
            other_context_saved.view(-1, 1, self.context * 2 + self.window)
        ).view(-1, seq_length, self.enc_dim)  # B*nspk*(nmic-1), L, N
        other_feature = other_feature.transpose(1, 2).contiguous()  # B*nspk*(nmic-1), N, L
        other_filter = self.other_BF(
            torch.cat([self.enc_LN(other_feature), other_cos_sim], 1), num_mic
        )  # B*nspk*(nmic-1), 1, 1, L, 2*win+1

        # convolve with other mic context segments
        other_output = F.conv1d(
            other_context_saved.view(1, -1, self.context * 2 + self.window),
            other_filter.view(-1, 1, self.filter_dim),
            groups=batch_size * self.num_spk * (nmic - 1) * seq_length,
        )  # 1, B*nspk*(nmic-1)*L, win
        other_output = other_output.view(
            batch_size * self.num_spk, nmic - 1, seq_length, self.window
        )  # B*nspk, nmic-1, L, win

        all_bf_output = torch.cat(
            [ref_output.unsqueeze(1), other_output], 1
        )  # B*nspk, nmic, L, win

        # reshape to utterance
        bf_signal = all_bf_output.view(batch_size * self.num_spk * nmic, -1, self.window * 2)
        bf_signal1 = (
            bf_signal[:, :, : self.window]
            .contiguous()
            .view(batch_size * self.num_spk * nmic, 1, -1)[:, :, self.stride :]
        )
        bf_signal2 = (
            bf_signal[:, :, self.window :]
            .contiguous()
            .view(batch_size * self.num_spk * nmic, 1, -1)[:, :, : -self.stride]
        )
        bf_signal = bf_signal1 + bf_signal2  # B*nspk*nmic, 1, T
        if rest > 0:
            bf_signal = bf_signal[:, :, :-rest]

        bf_signal = bf_signal.view(batch_size, self.num_spk, nmic, -1)  # B, nspk, nmic, T
        # consider only the valid channels
        if num_mic.max() == 0:
            bf_signal = bf_signal.mean(2)  # B, nspk, T
        else:
            bf_signal = [
                bf_signal[b, :, : num_mic[b]].mean(1).unsqueeze(0) for b in range(batch_size)
            ]  # nspk, T
            bf_signal = torch.cat(bf_signal, 0)  # B, nspk, T

        return bf_signal


# single-stage FaSNet + TAC
class FaSNet_TAC(FaSNet_base):
    def __init__(self, *args, **kwargs):
        super(FaSNet_TAC, self).__init__(*args, **kwargs)

        # DPRNN + TAC for estimation
        self.all_BF = BF_module(
            self.filter_dim + self.enc_dim,
            self.feature_dim,
            self.hidden_dim,
            self.filter_dim,
            self.num_spk,
            self.layer,
            self.segment_size,
            model_type="DPRNN_TAC",
        )

    def forward(self, input, num_mic):
        batch_size = input.size(0)
        nmic = input.size(1)

        # split input into chunks
        all_seg, all_mic_context, rest = self.seg_signal_context(
            input, self.window, self.context
        )  # B, nmic, L, win/chunk
        seq_length = all_seg.size(2)

        # embeddings for all channels
        enc_output = (
            self.encoder(all_mic_context.view(-1, 1, self.context * 2 + self.window))
            .view(batch_size * nmic, seq_length, self.enc_dim)
            .transpose(1, 2)
            .contiguous()
        )  # B*nmic, N, L
        enc_output = self.enc_LN(enc_output).view(
            batch_size, nmic, self.enc_dim, seq_length
        )  # B, nmic, N, L

        # calculate the cosine similarities for ref channel's center frame with all channels' context

        ref_seg = all_seg[:, 0].contiguous().view(1, -1, self.window)  # 1, B*L, win
        all_context = (
            all_mic_context.transpose(0, 1)
            .contiguous()
            .view(nmic, -1, self.context * 2 + self.window)
        )  # 1, B*L, 3*win
        all_cos_sim = self.seq_cos_sim(all_context, ref_seg)  # nmic, B*L, 2*win+1
        all_cos_sim = (
            all_cos_sim.view(nmic, batch_size, seq_length, self.filter_dim)
            .permute(1, 0, 3, 2)
            .contiguous()
        )  # B, nmic, 2*win+1, L

        input_feature = torch.cat([enc_output, all_cos_sim], 2)  # B, nmic, N+2*win+1, L

        # pass to DPRNN
        all_filter = self.all_BF(input_feature, num_mic)  # B, ch, nspk, L, 2*win+1

        # convolve with all mic's context
        mic_context = torch.cat(
            [all_mic_context.view(batch_size * nmic, 1, seq_length, self.context * 2 + self.window)]
            * self.num_spk,
            1,
        )  # B*nmic, nspk, L, 3*win
        all_bf_output = F.conv1d(
            mic_context.view(1, -1, self.context * 2 + self.window),
            all_filter.view(-1, 1, self.filter_dim),
            groups=batch_size * nmic * self.num_spk * seq_length,
        )  # 1, B*nmic*nspk*L, win
        all_bf_output = all_bf_output.view(
            batch_size, nmic, self.num_spk, seq_length, self.window
        )  # B, nmic, nspk, L, win

        # reshape to utterance
        bf_signal = all_bf_output.view(batch_size * nmic * self.num_spk, -1, self.window * 2)
        bf_signal1 = (
            bf_signal[:, :, : self.window]
            .contiguous()
            .view(batch_size * nmic * self.num_spk, 1, -1)[:, :, self.stride :]
        )
        bf_signal2 = (
            bf_signal[:, :, self.window :]
            .contiguous()
            .view(batch_size * nmic * self.num_spk, 1, -1)[:, :, : -self.stride]
        )
        bf_signal = bf_signal1 + bf_signal2  # B*nmic*nspk, 1, T
        if rest > 0:
            bf_signal = bf_signal[:, :, :-rest]

        bf_signal = bf_signal.view(batch_size, nmic, self.num_spk, -1)  # B, nmic, nspk, T
        # consider only the valid channels
        if num_mic.max() == 0:
            bf_signal = bf_signal.mean(1)  # B, nspk, T
        else:
            bf_signal = [
                bf_signal[b, : num_mic[b]].mean(0).unsqueeze(0) for b in range(batch_size)
            ]  # nspk, T
            bf_signal = torch.cat(bf_signal, 0)  # B, nspk, T

        return bf_signal


# ---------------------------------------------------------------------------
# menagerie staging entrypoints
# ---------------------------------------------------------------------------


def build_fasnet():
    """Tiny FaSNet_origin (two-stage filter-and-sum network), matching the repo's own
    `test_model` smoke-test hyperparameters at reduced size for fast tracing. `sr` is lowered
    from the repo default (16000) to 2000, which shrinks `window`/`context`/`filter_dim`
    (derived from `sr*win_len/1000` and `sr*context_len/1000`) proportionally -- the DPRNN
    beamforming path's grouped-conv1d ops (`groups=batch*nmic*seq_length`) are the tracing-cost
    bottleneck, so this keeps the architecture identical while making per-op instrumentation
    tractable."""
    return FaSNet_origin(
        enc_dim=16,
        feature_dim=16,
        hidden_dim=32,
        layer=2,
        segment_size=4,
        nspk=2,
        win_len=4,
        context_len=16,
        sr=2000,
    )


def example_input_fasnet():
    # (batch, num_mic, T); num_mic tensor with zero entry selects the fixed-geometry branch
    # (matches repo's own `test_model` smoke test: `none_mic = torch.zeros(1)`). T kept small
    # (a handful of window/context chunks) since seq_cos_sim's grouped conv1d cost scales with
    # the number of chunks.
    x = torch.rand(1, 3, 200)
    num_mic = torch.zeros(1).long()
    return (x, num_mic)


def build_fasnet_tac():
    """Tiny FaSNet_TAC (single-stage FaSNet with Transform-Average-Concatenate cross-channel
    module), matching the repo's own `test_model` smoke-test hyperparameters at reduced size.
    See `build_fasnet` for why `sr` is lowered from the repo default."""
    return FaSNet_TAC(
        enc_dim=16,
        feature_dim=16,
        hidden_dim=32,
        layer=2,
        segment_size=4,
        nspk=2,
        win_len=4,
        context_len=16,
        sr=2000,
    )


def example_input_fasnet_tac():
    x = torch.rand(1, 3, 200)
    num_mic = torch.zeros(1).long()
    return (x, num_mic)


MENAGERIE_ENTRIES = [
    ("FaSNet", "build_fasnet", "example_input_fasnet", "2019", "audio"),
    ("FaSNet-TAC", "build_fasnet_tac", "example_input_fasnet_tac", "2020", "audio"),
]
