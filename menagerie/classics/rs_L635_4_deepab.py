# SOURCE: vendored from RosettaCommons/DeepAb @ 70488c1a010ebac40263ef4d768c26abc900fd9a
# https://raw.githubusercontent.com/RosettaCommons/DeepAb/main/deepab/models/AbResNet/AbResNet.py
# https://raw.githubusercontent.com/RosettaCommons/DeepAb/main/deepab/models/PairedSeqLSTM/PairedSeqLSTM.py
# https://raw.githubusercontent.com/RosettaCommons/DeepAb/main/deepab/resnets/ResNet1D.py
# https://raw.githubusercontent.com/RosettaCommons/DeepAb/main/deepab/resnets/ResNet2D.py
# https://raw.githubusercontent.com/RosettaCommons/DeepAb/main/deepab/resnets/CrissCrossResNet2D.py
# https://raw.githubusercontent.com/RosettaCommons/DeepAb/main/deepab/layers/OuterConcatenation2D.py
# https://raw.githubusercontent.com/RosettaCommons/DeepAb/main/deepab/util/tensor.py
#
# Ruffolo, Sulam, Gray. "Antibody structure prediction using interpretable deep learning"
# (Patterns, 2022) -- DeepAb / `AbResNet`. The real model (`deepab/models/AbResNet/AbResNet.py`)
# takes a one-hot heavy+light-chain sequence, runs a 1D ResNet stem (`ResNet1D`), fuses in
# a frozen bidirectional-LSTM sequence-pair encoding (`PairedSeqLSTM.encoder`, called under
# `torch.no_grad()` exactly as in the real `get_lstm_encoding`), projects to pairwise space
# via `OuterConcatenation2D` (einops-based outer concatenation), runs a deep dilated 2D
# ResNet (`ResNet2D`, `torch.utils.checkpoint.checkpoint`-wrapped as in the real code) and
# finally 6 output heads (ca_dist/cb_dist/no_dist/omega/theta/phi distance & dihedral bins),
# each an `RCCAModule` criss-cross-attention refinement block from `CrissCrossResNet2D.py`.
# All files copied verbatim; only the unused `load_model` checkpoint-loading helper (needs
# a real .pt file on disk, irrelevant to construction/tracing) was dropped from this
# staging copy, and the CUDA-only `.cuda()` device_ids branch is dead on CPU exactly as in
# the original.

import math
import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.utils.checkpoint import checkpoint


# --- util/tensor.py (verbatim) ---


def max_shape(data):
    """Gets the maximum length along all dimensions in a list of Tensors"""
    shapes = torch.Tensor([_.shape for _ in data])
    return torch.max(shapes.transpose(0, 1), dim=1)[0].int()


def pad_data_to_same_shape(tensor_list, pad_value=0):
    target_shape = max_shape(tensor_list)

    padded_dataset_shape = [len(tensor_list)] + list(target_shape)
    padded_dataset = torch.Tensor(*padded_dataset_shape).type_as(tensor_list[0])

    for i, data in enumerate(tensor_list):
        padding = reversed(target_shape - torch.Tensor(list(data.shape)).int())
        padding = F.pad(padding.unsqueeze(0).t(), (1, 0, 0, 0)).view(-1, 1)
        padding = padding.view(1, -1)[0].tolist()

        padded_data = F.pad(data, padding, value=pad_value)
        padded_dataset[i] = padded_data

    return padded_dataset


# --- layers/OuterConcatenation2D.py (verbatim) ---


class OuterConcatenation2D(nn.Module):
    """Transforms sequential data to pairwise data using an outer concatenation (similar to an outer product)."""

    def __init__(self):
        super(OuterConcatenation2D, self).__init__()

    def forward(self, x: torch.FloatTensor):
        if len(x.shape) != 3:
            raise ValueError("Expected three dimensional shape, got shape {}".format(x.shape))

        seq_len = x.shape[-1]
        row_exp = repeat(x, "b c l -> b c x l", x=seq_len)
        col_exp = repeat(x, "b c l -> b c l x", x=seq_len)
        out = torch.cat([col_exp, row_exp], dim=1)

        return out


# --- resnets/ResNet1D.py (verbatim) ---


class ResBlock1D(nn.Module):
    def __init__(self, in_planes, planes, kernel_size=3, stride=1, shortcut=None):
        super(ResBlock1D, self).__init__()

        padding = kernel_size // 2

        self.activation = F.relu
        self.conv1 = nn.Conv1d(
            in_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(
            planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm1d(planes)

        if shortcut is None and stride == 1:
            self.shortcut = lambda x: F.pad(x, pad=(0, 0, 0, planes - x.shape[1], 0, 0))
        elif shortcut is None and stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes),
            )
        else:
            self.shortcut = shortcut

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class ResNet1D(nn.Module):
    def __init__(self, in_channels, block, num_blocks, planes=64, kernel_size=3):
        super(ResNet1D, self).__init__()
        if not (planes != 0 and ((planes & (planes - 1)) == 0)):
            raise ValueError("The initial number of planes must be a power of 2")

        self.activation = F.relu
        self.kernel_size = kernel_size
        self.planes = planes

        self.conv1 = nn.Conv1d(
            in_channels,
            self.planes,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(self.planes)

        resnet = self._make_layer(block, self.planes, num_blocks, stride=1, kernel_size=kernel_size)

        self.layers = [resnet]
        setattr(self, "layer0", resnet)

    def _make_layer(self, block, planes, num_blocks, stride, kernel_size):
        layers = []
        for i in range(num_blocks):
            layers.append(block(planes, planes, stride=stride, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layers[0](out)
        return out


# --- resnets/ResNet2D.py (verbatim) ---


class ResBlock2D(nn.Module):
    def __init__(self, in_planes, planes, kernel_size=5, dilation=1, stride=1, shortcut=None):
        super(ResBlock2D, self).__init__()

        padding = ((kernel_size - 1) * dilation) // 2

        self.activation = F.relu
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        if shortcut is None and stride == 1:
            self.shortcut = lambda x: F.pad(x, pad=(0, 0, 0, 0, 0, planes - x.shape[1], 0, 0))
        elif shortcut is None and stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = shortcut

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class ResNet2D(nn.Module):
    def __init__(self, in_channels, block, num_blocks, planes=64, kernel_size=5, dilation_cycle=5):
        super(ResNet2D, self).__init__()
        if not (planes != 0 and ((planes & (planes - 1)) == 0)):
            raise ValueError("The initial number of planes must be a power of 2")

        self.activation = F.relu
        self.kernel_size = kernel_size
        self.planes = planes

        self.conv1 = nn.Conv2d(
            in_channels,
            self.planes,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.planes)

        resnet = self._make_layer(
            block,
            self.planes,
            num_blocks,
            stride=1,
            kernel_size=kernel_size,
            dilation_cycle=dilation_cycle,
        )

        self.layers = [resnet]
        setattr(self, "layer0", resnet)

    def _make_layer(self, block, planes, num_blocks, stride, kernel_size, dilation_cycle):
        layers = []
        for i in range(num_blocks):
            dilation = int(math.pow(2, i % dilation_cycle)) if dilation_cycle > 0 else 1
            layers.append(
                block(planes, planes, stride=stride, kernel_size=kernel_size, dilation=dilation)
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layers[0](out)
        return out


# --- resnets/CrissCrossResNet2D.py (verbatim) ---


class CrissCrossAttention(nn.Module):
    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        device = x.device
        b, _, h, w = x.shape

        q = self.query_conv(x)
        q_h = rearrange(q, "b c h w -> (b w) h c")
        q_w = rearrange(q, "b c h w -> (b h) w c")

        k = self.key_conv(x)
        k_h = rearrange(k, "b c h w -> (b w) c h")
        k_w = rearrange(k, "b c h w -> (b h) c w")

        v = self.value_conv(x)
        v_h = rearrange(v, "b c h w -> (b w) c h")
        v_w = rearrange(v, "b c h w -> (b h) c w")

        inf = repeat(
            torch.diag(torch.tensor(float("-inf"), device=device).repeat(h), 0),
            "h1 h2 -> (b w) h1 h2",
            b=b,
            w=w,
        )
        e_h = rearrange(torch.bmm(q_h, k_h) + inf, "(b w) h1 h2 -> b h1 w h2", b=b)
        e_w = rearrange(torch.bmm(q_w, k_w), "(b h) w1 w2 -> b h w1 w2", b=b)

        attn = self.softmax(torch.cat([e_h, e_w], 3))
        attn_h, attn_w = attn.chunk(2, dim=-1)
        attn_h = rearrange(attn_h, "b h1 w h2 -> (b w) h1 h2")
        attn_w = rearrange(attn_w, "b h w1 w2 -> (b h) w1 w2")

        out_h = torch.bmm(v_h, rearrange(attn_h, "bw h1 h2 -> bw h2 h1"))
        out_h = rearrange(out_h, "(b w) c h -> b c h w", b=b)
        out_w = torch.bmm(v_w, rearrange(attn_w, "bh w1 w2 -> bh w2 w1"))
        out_w = rearrange(out_w, "(b h) c w -> b c h w", b=b)

        return_attn = torch.stack(
            [
                rearrange(attn_h, "(b w) h1 h2 -> b h2 h1 w", b=b),
                rearrange(attn_w, "(b h) w1 w2 -> b w2 h w1", b=b),
            ],
            dim=1,
        )

        return self.gamma * (out_h + out_w) + x, return_attn


class RCCAModule(nn.Module):
    def __init__(self, in_channels, kernel_size=3, return_attn=False):
        super(RCCAModule, self).__init__()
        self.return_attn = return_attn
        inter_channels = in_channels // 4
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                inter_channels,
                kernel_size=(kernel_size, kernel_size),
                stride=(1, 1),
                padding=((kernel_size - 1) // 2, (kernel_size - 1) // 2),
                bias=False,
            ),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
        )
        self.cca = CrissCrossAttention(inter_channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                inter_channels,
                in_channels,
                kernel_size=(kernel_size, kernel_size),
                stride=(1, 1),
                padding=((kernel_size - 1) // 2, (kernel_size - 1) // 2),
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        output = self.conv1(x)
        attns = []
        for _ in range(2):
            output, attn = checkpoint(self.cca, output)
            attns.append(attn)
        output = self.conv2(output)

        if self.return_attn:
            return output, attns
        else:
            return output


# --- models/PairedSeqLSTM/PairedSeqLSTM.py (verbatim, load_model dropped) ---


class Encoder(nn.Module):
    def __init__(self, seq_dim: int, enc_hid_dim: int, dec_hid_dim: int):
        super().__init__()

        self.seq_dim = seq_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.rnn = nn.LSTM(seq_dim, enc_hid_dim, bidirectional=True, num_layers=2)
        self.fc1 = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.fc2 = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor]:
        outputs, (hidden, cell) = self.rnn(src.float())
        hidden = torch.tanh(self.fc1(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        cell = torch.tanh(self.fc2(torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1)))

        return outputs, (hidden, cell)


class Decoder(nn.Module):
    def __init__(self, seq_dim: int, enc_hid_dim: int, dec_hid_dim: int):
        super().__init__()

        self.seq_dim = seq_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.rnn = nn.LSTM(enc_hid_dim + seq_dim, dec_hid_dim, num_layers=2)
        self.out = nn.Linear(dec_hid_dim + seq_dim, seq_dim)

    def forward(
        self,
        input: torch.Tensor,
        decoder_hidden: torch.Tensor,
        decoder_cell: torch.Tensor,
        encoder_hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        input = input.unsqueeze(0).float()
        encoder_hidden = encoder_hidden.unsqueeze(0).float()

        if type(decoder_hidden) != type(None):
            output, (decoder_hidden, decoder_cell) = self.rnn(
                torch.cat((input, encoder_hidden), dim=2), (decoder_hidden, decoder_cell)
            )
        else:
            output, (decoder_hidden, decoder_cell) = self.rnn(
                torch.cat((input, encoder_hidden), dim=2)
            )

        input = input.squeeze(0)
        output = output.squeeze(0)

        output = self.out(torch.cat((output, input), dim=1))

        return output, (decoder_hidden, decoder_cell)


class PairedSeqLSTM(nn.Module):
    def __init__(self, seq_dim: int = 23, enc_hid_dim: int = 64, dec_hid_dim: int = 64):
        super().__init__()

        self.encoder = Encoder(seq_dim, enc_hid_dim, dec_hid_dim)
        self.decoder = Decoder(seq_dim, enc_hid_dim, dec_hid_dim)

    def forward(
        self, src: torch.Tensor, trg: torch.Tensor, teacher_forcing_ratio: float = 0.5
    ) -> torch.Tensor:
        device = src.device

        batch_size = src.shape[1]
        max_len = src.shape[0]
        seq_dim = src.shape[2]
        outputs = torch.zeros(max_len, batch_size, seq_dim).to(device)

        encoder_outputs, (encoder_hidden, _) = self.encoder(src)

        output = trg[0, :]
        hidden, cell = None, None
        for t in range(1, max_len):
            output, (hidden, cell) = self.decoder(output, hidden, cell, encoder_hidden)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = F.one_hot(output.argmax(-1), num_classes=output.shape[1])
            output = trg[t] if teacher_force else top1

        return outputs


# --- models/AbResNet/AbResNet.py (verbatim, load_model dropped) ---


def create_output_block(out_planes2D, num_out_bins, kernel_size):
    return nn.Sequential(
        nn.Conv2d(out_planes2D, num_out_bins, kernel_size=kernel_size, padding=kernel_size // 2),
        RCCAModule(in_channels=num_out_bins, kernel_size=kernel_size, return_attn=True),
    )


class AbResNet(nn.Module):
    """
    Predicts binned output distributions for CA-distance, CB-distance, NO-distance,
    omega and theta dihedrals, and phi planar angle from a one-hot encoded sequence
    of heavy and light chain resides.
    """

    def __init__(
        self,
        in_planes,
        lstm_model,
        rnn_planes=128,
        num_out_bins=37,
        num_blocks1D=3,
        num_blocks2D=25,
        dilation_cycle=5,
        dropout_proportion=0.2,
        lstm_mean=None,
        lstm_scale=None,
    ):
        super(AbResNet, self).__init__()

        self.output_names = ["ca_dist", "cb_dist", "no_dist", "omega", "theta", "phi"]

        self.lstm_model = lstm_model
        self.lstm_mean = (
            torch.zeros(
                1,
            )
            if lstm_mean is None
            else lstm_mean
        )
        self.lstm_scale = (
            torch.ones(
                1,
            )
            if lstm_scale is None
            else lstm_scale
        )

        self._num_out_bins = num_out_bins
        self.resnet1D = ResNet1D(in_planes, ResBlock1D, num_blocks1D, planes=32, kernel_size=17)
        self.seq2pairwise = OuterConcatenation2D()

        out_planes1D = self.resnet1D.planes
        in_planes2D = 2 * (out_planes1D + rnn_planes)

        self.resnet2D = ResNet2D(
            in_planes2D,
            ResBlock2D,
            num_blocks2D,
            planes=64,
            kernel_size=5,
            dilation_cycle=dilation_cycle,
        )

        out_planes2D = self.resnet2D.planes

        self.out_dropout = nn.Dropout2d(p=dropout_proportion)

        self.out_ca_dist = create_output_block(
            out_planes2D, num_out_bins, self.resnet2D.kernel_size
        )
        self.out_cb_dist = create_output_block(
            out_planes2D, num_out_bins, self.resnet2D.kernel_size
        )
        self.out_no_dist = create_output_block(
            out_planes2D, num_out_bins, self.resnet2D.kernel_size
        )
        self.out_omega = create_output_block(out_planes2D, num_out_bins, self.resnet2D.kernel_size)
        self.out_theta = create_output_block(out_planes2D, num_out_bins, self.resnet2D.kernel_size)
        self.out_phi = create_output_block(out_planes2D, num_out_bins, self.resnet2D.kernel_size)

    def get_lstm_input(self, x):
        device = x.device
        seq_start, seq_end, seq_delim = (
            torch.tensor([20]).byte().to(device),
            torch.tensor([21]).byte().to(device),
            torch.tensor([22]).byte().to(device),
        )

        input_seqs = x.transpose(1, 2)[:, :, :-1].argmax(-1).to(device)
        input_delims = x.transpose(1, 2)[:, :, -1].argmax(-1).to(device)

        lstm_input = [
            torch.cat([seq_start, seq[: d + 1].byte(), seq_delim, seq[d + 1 :].byte(), seq_end])
            for seq, d in zip(input_seqs, input_delims)
        ]
        lstm_input = pad_data_to_same_shape(lstm_input, pad_value=22)
        lstm_input = torch.stack([nn.functional.one_hot(seq.long()) for seq in lstm_input])
        lstm_input = lstm_input.transpose(0, 1)

        return lstm_input, input_delims

    def get_lstm_encoding(self, inputs):
        with torch.no_grad():
            lstm_input, input_delims = self.get_lstm_input(inputs)

            enc = self.lstm_model.encoder(src=lstm_input)[0].detach()
            enc = enc.permute(1, 0, 2)

            no_delim_enc = []
            for i in range(len(enc)):
                no_delim_enc.append(
                    torch.cat([enc[i][1 : input_delims[i]], enc[i][input_delims[i] + 1 : -1]])
                )
            enc = torch.stack(no_delim_enc).permute(0, 2, 1)

            enc = (enc - self.lstm_mean.view(1, -1, 1)) / self.lstm_scale.view(1, -1, 1)

            return enc

    def forward(self, x):
        out = self.resnet1D(x)
        lstm_enc = self.get_lstm_encoding(x)
        out = torch.cat([out, lstm_enc], dim=1)

        out = self.seq2pairwise(out)
        out = checkpoint(self.resnet2D, out)
        out = self.out_dropout(out)

        out_ca_dist = self.out_ca_dist(out)[0]
        out_cb_dist = self.out_cb_dist(out)[0]
        out_no_dist = self.out_no_dist(out)[0]
        out_omega = self.out_omega(out)[0]
        out_theta = self.out_theta(out)[0]
        out_phi = self.out_phi(out)[0]

        out_ca_dist = out_ca_dist + out_ca_dist.transpose(2, 3)
        out_cb_dist = out_cb_dist + out_cb_dist.transpose(2, 3)
        out_omega = out_omega + out_omega.transpose(2, 3)

        return [out_ca_dist, out_cb_dist, out_no_dist, out_omega, out_theta, out_phi]


# --- staging harness (tiny sizes; not part of the real repo) ---


def build_deepab_abresnet() -> nn.Module:
    # in_planes=21 mirrors the real load_model's hardcoded `in_planes = 21` (one-hot
    # amino-acid channels); num_blocks1D/num_blocks2D shrunk from the real defaults
    # (3->1, 25->2) to keep the checkpoint-wrapped ResNet2D/RCCAModule trace fast --
    # dilation_cycle/dropout_proportion left at real defaults. num_out_bins is kept at
    # 32 (real default 37, rounded down to a power of 4 x 8): RCCAModule computes
    # `inter_channels = in_channels // 4` then `query_conv`/`key_conv` project to
    # `inter_channels // 8`, so num_out_bins must stay >= 32 or the criss-cross attention
    # QK convs collapse to zero output channels -- an architecture constraint of the real
    # code (create_output_block(..., num_out_bins, ...) feeds num_out_bins straight into
    # RCCAModule(in_channels=num_out_bins, ...)), not a staging shortcut. lstm_model is a
    # real (tiny) PairedSeqLSTM, matching how the real AbResNet always carries a
    # lstm_model submodule for get_lstm_encoding(); rnn_planes must equal the LSTM
    # encoder's actual output width (2*enc_hid_dim, bidirectional) since
    # get_lstm_encoding's output is channel-concatenated with resnet1D's output before
    # seq2pairwise -- this is architecture-as-written (AbResNet.__init__ never validates
    # rnn_planes against lstm_model, it is the caller's responsibility to match them; the
    # real load_model always uses the paired real default enc_hid_dim=64 -> rnn_planes=128).
    lstm_model = PairedSeqLSTM(seq_dim=23, enc_hid_dim=8, dec_hid_dim=8)
    model = AbResNet(
        in_planes=21,
        lstm_model=lstm_model,
        rnn_planes=2 * 8,
        num_out_bins=32,
        num_blocks1D=1,
        num_blocks2D=2,
    )
    model.eval()
    return model


def example_input_deepab_abresnet():
    # (batch, 21, seq_len) one-hot encoded heavy+light-chain residue sequence (20 amino
    # acids + 1 chain-delimiter channel), exactly what AbResNet.forward(x) consumes;
    # get_lstm_input's argmax-based delimiter detection needs the delimiter channel (index
    # 20) to actually be the max at exactly one position per sequence, so it is set
    # explicitly here rather than left random.
    batch, channels, seq_len = 2, 21, 14
    x = torch.rand(batch, channels, seq_len)
    delim_pos = seq_len // 2
    x[:, :, delim_pos] = 0.0
    x[:, 20, delim_pos] = 1.0
    return (x,)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("DeepAb", "build_deepab_abresnet", "example_input_deepab_abresnet", 2022, "vendored"),
]
