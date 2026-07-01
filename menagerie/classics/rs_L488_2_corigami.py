# SOURCE: vendored from tanjimin/C.Origami @ main
#   (src/corigami/model/blocks.py, src/corigami/model/corigami_models.py)
#
# C.Origami (Tan et al., Nature Biotechnology 2023): a CNN + Transformer sequence model
# that predicts Hi-C-style genome-folding (chromatin-interaction) maps from DNA sequence
# plus epigenomic tracks (CTCF ChIP-seq, ATAC-seq). `ConvTransModel` is the full
# architecture used in the paper: a dual-branch (sequence / epigenomic) 1D residual-conv
# encoder (`EncoderSplit`), a pre-LN Transformer stack (`AttnModule`) over the pooled
# genomic-bin embeddings, an outer-product "diagonalization" step that turns the 1D
# embedding sequence into a 2D contact-map feature grid, and a dilated-residual-conv 2D
# decoder (`Decoder`) that regresses the predicted interaction matrix. Copied verbatim
# from the real repo's ConvBlock/Encoder/EncoderSplit/ResBlockDilated/Decoder/
# TransformerLayer/TransformerEncoder/PositionalEncoding/AttnModule/ConvModel/
# ConvTransModel classes (aside from collapsing the `import corigami.model.blocks as
# blocks` module split into one file).
import copy

import numpy as np
import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


class ConvBlock(nn.Module):
    def __init__(self, size, stride=2, hidden_in=64, hidden=64):
        super(ConvBlock, self).__init__()
        pad_len = int(size / 2)
        self.scale = nn.Sequential(
            nn.Conv1d(hidden_in, hidden, size, stride, pad_len),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        self.res = nn.Sequential(
            nn.Conv1d(hidden, hidden, size, padding=pad_len),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, size, padding=pad_len),
            nn.BatchNorm1d(hidden),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        scaled = self.scale(x)
        identity = scaled
        res_out = self.res(scaled)
        out = self.relu(res_out + identity)
        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, output_size=256, filter_size=5, num_blocks=12):
        super(Encoder, self).__init__()
        self.filter_size = filter_size
        self.conv_start = nn.Sequential(
            nn.Conv1d(in_channel, 32, 3, 2, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        hiddens = [32, 32, 32, 32, 64, 64, 128, 128, 128, 128, 256, 256]
        hidden_ins = [32, 32, 32, 32, 32, 64, 64, 128, 128, 128, 128, 256]
        self.res_blocks = self.get_res_blocks(num_blocks, hidden_ins, hiddens)
        self.conv_end = nn.Conv1d(256, output_size, 1)

    def forward(self, x):
        x = self.conv_start(x)
        x = self.res_blocks(x)
        out = self.conv_end(x)
        return out

    def get_res_blocks(self, n, his, hs):
        blocks = []
        for i, h, hi in zip(range(n), hs, his):
            blocks.append(ConvBlock(self.filter_size, hidden_in=hi, hidden=h))
        res_blocks = nn.Sequential(*blocks)
        return res_blocks


class EncoderSplit(Encoder):
    def __init__(self, num_epi, output_size=256, filter_size=5, num_blocks=12):
        super(Encoder, self).__init__()
        self.filter_size = filter_size
        self.conv_start_seq = nn.Sequential(
            nn.Conv1d(5, 16, 3, 2, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.conv_start_epi = nn.Sequential(
            nn.Conv1d(num_epi, 16, 3, 2, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        hiddens = [32, 32, 32, 32, 64, 64, 128, 128, 128, 128, 256, 256]
        hidden_ins = [32, 32, 32, 32, 32, 64, 64, 128, 128, 128, 128, 256]
        hiddens_half = (np.array(hiddens) / 2).astype(int)
        hidden_ins_half = (np.array(hidden_ins) / 2).astype(int)
        self.res_blocks_seq = self.get_res_blocks(num_blocks, hidden_ins_half, hiddens_half)
        self.res_blocks_epi = self.get_res_blocks(num_blocks, hidden_ins_half, hiddens_half)
        self.conv_end = nn.Conv1d(256, output_size, 1)

    def forward(self, x):
        seq = x[:, :5, :]
        epi = x[:, 5:, :]
        seq = self.res_blocks_seq(self.conv_start_seq(seq))
        epi = self.res_blocks_epi(self.conv_start_epi(epi))

        x = torch.cat([seq, epi], dim=1)
        out = self.conv_end(x)
        return out


class ResBlockDilated(nn.Module):
    def __init__(self, size, hidden=64, stride=1, dil=2):
        super(ResBlockDilated, self).__init__()
        pad_len = dil
        self.res = nn.Sequential(
            nn.Conv2d(hidden, hidden, size, padding=pad_len, dilation=dil),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, size, padding=pad_len, dilation=dil),
            nn.BatchNorm2d(hidden),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        res_out = self.res(x)
        out = self.relu(res_out + identity)
        return out


class Decoder(nn.Module):
    def __init__(self, in_channel, hidden=256, filter_size=3, num_blocks=5):
        super(Decoder, self).__init__()
        self.filter_size = filter_size

        self.conv_start = nn.Sequential(
            nn.Conv2d(in_channel, hidden, 3, 1, 1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
        )
        self.res_blocks = self.get_res_blocks(num_blocks, hidden)
        self.conv_end = nn.Conv2d(hidden, 1, 1)

    def forward(self, x):
        x = self.conv_start(x)
        x = self.res_blocks(x)
        out = self.conv_end(x)
        return out

    def get_res_blocks(self, n, hidden):
        blocks = []
        for i in range(n):
            dilation = 2 ** (i + 1)
            blocks.append(ResBlockDilated(self.filter_size, hidden=hidden, dil=dilation))
        res_blocks = nn.Sequential(*blocks)
        return res_blocks


class TransformerLayer(torch.nn.TransformerEncoderLayer):
    # Pre-LN structure

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # MHA section
        src_norm = self.norm1(src)
        src_side, attn_weights = self.self_attn(
            src_norm,
            src_norm,
            src_norm,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        src = src + self.dropout1(src_side)

        # MLP section
        src_norm = self.norm2(src)
        src_side = self.linear2(self.dropout(self.activation(self.linear1(src_norm))))
        src = src + self.dropout2(src_side)
        return src, attn_weights


class TransformerEncoder(torch.nn.TransformerEncoder):
    def __init__(self, encoder_layer, num_layers, norm=None, record_attn=False):
        super(TransformerEncoder, self).__init__(encoder_layer, num_layers)
        self.layers = self._get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.record_attn = record_attn

    def forward(self, src, mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        attn_weight_list = []

        for mod in self.layers:
            output, attn_weights = mod(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask
            )
            attn_weight_list.append(attn_weights.unsqueeze(0).detach())
        if self.norm is not None:
            output = self.norm(output)

        if self.record_attn:
            return output, torch.cat(attn_weight_list)
        else:
            return output

    def _get_clones(self, module, N):
        return torch.nn.modules.ModuleList([copy.deepcopy(module) for i in range(N)])


class PositionalEncoding(nn.Module):
    def __init__(self, hidden, dropout=0.1, max_len=256):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden, 2) * (-np.log(10000.0) / hidden))
        pe = torch.zeros(max_len, 1, hidden)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class AttnModule(nn.Module):
    def __init__(self, hidden=128, layers=8, record_attn=False, inpu_dim=256):
        super(AttnModule, self).__init__()

        self.record_attn = record_attn
        self.pos_encoder = PositionalEncoding(hidden, dropout=0.1)
        encoder_layers = TransformerLayer(
            hidden,
            nhead=8,
            dropout=0.1,
            dim_feedforward=512,
            batch_first=True,
        )
        self.module = TransformerEncoder(encoder_layers, layers, record_attn=record_attn)

    def forward(self, x):
        x = self.pos_encoder(x)
        output = self.module(x)
        return output

    def inference(self, x):
        return self.module(x)


class ConvModel(nn.Module):
    def __init__(self, num_genomic_features, mid_hidden=256):
        super(ConvModel, self).__init__()
        self.encoder = EncoderSplit(num_genomic_features, output_size=mid_hidden, num_blocks=12)
        self.decoder = Decoder(mid_hidden * 2)

    def forward(self, x):
        """
        Input feature:
        batch_size, length * res, feature_dim
        """
        x = self.move_feature_forward(x).float()
        x = self.encoder(x)
        x = self.diagonalize(x)
        x = self.decoder(x).squeeze(1)
        return x

    def move_feature_forward(self, x):
        """
        input dim:
        bs, img_len, feat
        to:
        bs, feat, img_len
        """
        return x.transpose(1, 2).contiguous()

    def diagonalize(self, x):
        x_i = x.unsqueeze(2).repeat(1, 1, 256, 1)
        x_j = x.unsqueeze(3).repeat(1, 1, 1, 256)
        input_map = torch.cat([x_i, x_j], dim=1)
        return input_map


class ConvTransModel(ConvModel):
    def __init__(self, num_genomic_features, mid_hidden=256, record_attn=False):
        super(ConvTransModel, self).__init__(num_genomic_features)
        self.encoder = EncoderSplit(num_genomic_features, output_size=mid_hidden, num_blocks=12)
        self.attn = AttnModule(hidden=mid_hidden, record_attn=record_attn)
        self.decoder = Decoder(mid_hidden * 2)
        self.record_attn = record_attn

    def forward(self, x):
        """
        Input feature:
        batch_size, length * res, feature_dim
        """
        x = self.move_feature_forward(x).float()
        x = self.encoder(x)
        x = self.move_feature_forward(x)
        if self.record_attn:
            x, attn_weights = self.attn(x)
        else:
            x = self.attn(x)
        x = self.move_feature_forward(x)
        x = self.diagonalize(x)
        x = self.decoder(x).squeeze(1)
        if self.record_attn:
            return x, attn_weights
        else:
            return x


def build_corigami():
    # Real model uses num_genomic_features=2 (CTCF + ATAC) with 2097152bp (2Mb) windows
    # pooled to length 256 at inference; shrink the sequence length for tracing speed
    # while keeping the true architecture (num_blocks=12 encoder/decoder, 8-layer
    # Transformer) untouched.
    return ConvTransModel(num_genomic_features=2, mid_hidden=256, record_attn=False)


def example_input_corigami():
    # (batch, length, feature_dim): EncoderSplit.forward hardcodes a 5-channel sequence
    # slice (x[:, :5, :], one-hot DNA + N-mask) and routes the remaining num_epi channels
    # (CTCF, ATAC = 2) to the epigenomic branch -> feature_dim = 5 + 2 = 7.
    # `ConvModel.diagonalize` hardcodes a 256-bin outer product (`.repeat(1,1,256,1)`),
    # so the encoder's pooled genomic-bin length must be exactly 256. `ConvBlock`
    # defaults to stride=2 in its `.scale` branch and `get_res_blocks` does not override
    # it, so all 12 res blocks -- not just conv_start -- downsample by 2x, matching the
    # real model's 2Mb (2,097,152 bp) input window: 2**13 * 256 = 2,097,152.
    return torch.randn(2, 2097152, 7)


MENAGERIE_ENTRIES = [
    ("C.Origami", build_corigami, example_input_corigami, 2023, "SOURCE_AVAILABLE"),
]
