# SOURCE: vendored from krpratik/RE-MIMO @ master (fully_correlated_channels/re-mimo/)
# https://raw.githubusercontent.com/krpratik/RE-MIMO/master/fully_correlated_channels/re-mimo/iterative_classifier.py
# https://raw.githubusercontent.com/krpratik/RE-MIMO/master/fully_correlated_channels/re-mimo/EncoderDecoderBlock.py
# https://raw.githubusercontent.com/krpratik/RE-MIMO/master/fully_correlated_channels/re-mimo/MultiheadAttention.py
# https://raw.githubusercontent.com/krpratik/RE-MIMO/master/fully_correlated_channels/re-mimo/NumTransmitterEncoding.py
# https://raw.githubusercontent.com/krpratik/RE-MIMO/master/fully_correlated_channels/re-mimo/TransformerEncoder.py
# https://raw.githubusercontent.com/krpratik/RE-MIMO/master/fully_correlated_channels/re-mimo/TransformerEncoderLayer.py
# https://raw.githubusercontent.com/krpratik/RE-MIMO/master/fully_correlated_channels/re-mimo/TransformerDecoderLayer.py
#
# RE-MIMO ("RE-MIMO: Recurrent and Permutation Equivariant Neural MIMO Detection",
# Pratik, Rao & Welling, arXiv:2007.00140). A permutation-equivariant, variable-size
# iterative MIMO symbol detector: `iterative_classifier` stacks `nlayers` custom
# `EncoderDecoderBlock`s, each running one custom `TransformerEncoder` (self-attention
# over transmitter-antenna tokens via a from-scratch `MultiheadAttention` reimplementing
# `F.multi_head_attention_forward` with a single fused `in_proj_weight`, mirroring old
# PyTorch's internal MHA) followed by a small feed-forward `TransformerDecoderLayer`
# that regresses per-transmitter QAM-symbol logits. `NumTransmitterEncoding` is a
# sinusoidal positional-style encoding over the (variable) number of transmit
# antennas, letting one trained model generalize across problem sizes. Every module,
# its constructor arguments, and its forward-pass tensor algebra (the delta_y residual
# update, the QAM soft-symbol expectation via `real_QAM_const`/`imag_QAM_const`, the
# fused in_proj self-attention) is transcribed verbatim from the six source files
# above. Only the pickle-based `save_attention_weight` debug hook in
# `TransformerEncoderLayer` and the deprecated `np.int`/`np.float` aliases (replaced
# with `int`/`float`, numerically identical) are adjusted; no architecture changed.

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm


# ---- MultiheadAttention.py ----
class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        output_dim,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
    ):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads"
        )

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.vdim))

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = nn.Linear(embed_dim, output_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        if hasattr(self, "_qkv_same_embed_dim") and self._qkv_same_embed_dim is False:
            return F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
            )
        else:
            return F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
            )


# ---- NumTransmitterEncoding.py ----
class NumTransmitterEncoding(nn.Module):
    def __init__(self, d_model, d_transmitter_encoding, max_transmitter, dropout=0.0):
        super(NumTransmitterEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_transmitter_encoding = d_transmitter_encoding

        NT = torch.zeros(int(max_transmitter), d_transmitter_encoding)
        num_transmitters = torch.arange(1.0, max_transmitter + 1.0, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0.0, d_transmitter_encoding, 2).float()
            * (-math.log(float(2 * max_transmitter)) / d_transmitter_encoding)
        )
        NT[:, 0::2] = torch.sin(num_transmitters * div_term)
        NT[:, 1::2] = torch.cos(num_transmitters * div_term)
        NT = NT / math.sqrt(d_model)
        self.register_buffer("NT", NT)

    def forward(self, x):
        num_transmitter = x.shape[0]
        batch_size = x.shape[1]
        num_transmitter_encoding = self.NT[num_transmitter - 1, :].expand(
            size=(num_transmitter, batch_size, self.d_transmitter_encoding)
        )
        x = torch.cat((x, num_transmitter_encoding), dim=2)
        return self.dropout(x)


# ---- TransformerEncoderLayer.py ----
class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        encoder_input_dim,
        mod_n,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(encoder_input_dim, nhead, d_model, dropout=dropout)
        # Implementation of Feedforward model
        initial_dim = d_model
        self.linear1 = nn.Linear(initial_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(
        self, src, common_input, index, save_attn_weight, src_mask=None, src_key_padding_mask=None
    ):
        src_concat = torch.cat((src, common_input), dim=-1)
        del common_input

        # NOTE: the original repo optionally pickles attention weights to disk here
        # (`save_attention_weight`) for offline visualization; that debug-only I/O
        # hook is dropped since it is not part of the traced architecture.
        src2 = self.self_attn(
            src_concat,
            src_concat,
            src_concat,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0]

        del src_concat
        src = src + self.dropout1(src2)
        del src2
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        del src2
        src = self.norm2(src)
        return src


# ---- TransformerEncoder.py ----
def _get_clones(module, N):
    import copy

    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self, src, common_input, index, save_attn_weight, mask=None, src_key_padding_mask=None
    ):
        output = src
        for mod in self.layers:
            output = mod(
                output,
                common_input,
                index,
                save_attn_weight,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


# ---- TransformerDecoderLayer.py ----
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, NR, mod_n):
        super(TransformerDecoderLayer, self).__init__()
        # Implementation of Feedforward model
        initial_dim = d_model + 4 * NR + mod_n + 1
        interim_dim_1 = (initial_dim + 1) // 2
        interim_dim_2 = (interim_dim_1 + 1) // 2
        final_dim = mod_n

        self.linear1 = nn.Linear(initial_dim, interim_dim_1)
        self.linear2 = nn.Linear(interim_dim_1, interim_dim_2)
        self.linear3 = nn.Linear(interim_dim_2, final_dim)
        self.activation_1 = nn.ReLU()
        self.activation_2 = nn.ReLU()
        self.mod_n = mod_n

    def gen_decoder_input(self, st, common_input, noise_sigma, NT):
        noise_sigma_normalized = (
            ((noise_sigma) / np.sqrt(2.0 * NT)).expand(NT, -1).unsqueeze(dim=-1)
        )
        decoder_embed = torch.cat((st, common_input, noise_sigma_normalized), dim=-1)
        del noise_sigma_normalized
        return decoder_embed

    def forward(self, st, common_input, noise_sigma, NT):
        decoder_embed = self.gen_decoder_input(st, common_input, noise_sigma, NT)
        del st, common_input, noise_sigma
        out = self.linear1(decoder_embed)
        out = self.linear2(self.activation_1(out))
        out = self.linear3(self.activation_2(out))
        del decoder_embed
        return out


# ---- EncoderDecoderBlock.py ----
class EncoderDecoderBlock(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        NR,
        mod_n,
        real_QAM_const,
        imag_QAM_const,
        constel,
        device,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
    ):
        super(EncoderDecoderBlock, self).__init__()

        encoder_input_dim = 4 * NR + mod_n + d_model

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, encoder_input_dim, mod_n, dim_feedforward, dropout
        )
        self.encoder = TransformerEncoder(encoder_layer, 1)
        self.decoder = TransformerDecoderLayer(d_model, NR, mod_n)
        self.theta = nn.Parameter(torch.Tensor([1.0]))
        nn.init.normal_(self.theta, mean=1.0, std=0.1)
        self.register_buffer("real_QAM_const", real_QAM_const.to(device=device))
        self.register_buffer("imag_QAM_const", imag_QAM_const.to(device=device))
        self.register_buffer("constel", constel.to(device=device))
        self.NR = NR
        self.device = device
        self.mod_n = mod_n
        self.constel_size = int(np.sqrt(mod_n))

    def gen_common_input(self, xt, H, y, noise_sigma, NT):
        xt_probs = xt.softmax(dim=-1)

        x_real = (xt_probs * self.real_QAM_const).sum(dim=-1)
        x_imag = (xt_probs * self.imag_QAM_const).sum(dim=-1)

        xt_val = torch.cat((x_real, x_imag), dim=0).permute(1, 0)
        delta_y = y - torch.einsum(("ijk,ik->ij"), (H, xt_val))

        del y, xt_probs, xt_val, x_imag, x_real

        tgt = torch.chunk(H, 2, dim=2)[0].permute(2, 0, 1)
        # Normalizing y
        del H

        delta_y = delta_y / np.sqrt(2.0 * NT)
        delta_y = torch.unsqueeze(delta_y, dim=0).expand(NT, -1, -1)

        final_repr_encoder = torch.cat((delta_y, tgt, xt), dim=-1)
        final_repr_decoder = torch.cat((delta_y, tgt, xt), dim=-1)

        del delta_y, tgt, xt
        return final_repr_encoder, final_repr_decoder

    def forward(self, st, xt, H, y, noise_sigma, NT, index, save_attn_weight):
        encoder_input, decoder_input = self.gen_common_input(xt, H, y, noise_sigma, NT)
        del xt, H, y
        encoder_out = self.encoder.forward(st, encoder_input, index, save_attn_weight)
        del encoder_input, st

        decoder_out = self.decoder.forward(encoder_out, decoder_input, noise_sigma, NT)
        del decoder_input, noise_sigma
        return encoder_out, decoder_out


# ---- iterative_classifier.py ----
class iterative_classifier(nn.Module):
    def __init__(
        self,
        d_model,
        n_head,
        nhid,
        nlayers,
        mod_n,
        NR,
        d_transmitter_encoding,
        real_QAM_const,
        imag_QAM_const,
        constel,
        device,
        dropout=0.0,
    ):
        super(iterative_classifier, self).__init__()
        self.d_model = d_model
        self.mod_n = mod_n
        self.device = device
        self.register_buffer("constel", constel.to(device=device))
        self.constel_size = constel.numel()
        self.NR = NR

        # source embeddings
        initial_dim = 4 * NR + d_transmitter_encoding + 1
        interim_dim = d_model * 4

        self.encoder_embed = nn.Sequential(
            nn.Linear(initial_dim, interim_dim), nn.ReLU(), nn.Linear(interim_dim, d_model)
        )

        # Iterative Encoding-Decoding Blocks
        self.iterative_blocks = nn.ModuleList(
            [
                EncoderDecoderBlock(
                    d_model,
                    n_head,
                    NR,
                    mod_n,
                    real_QAM_const,
                    imag_QAM_const,
                    constel,
                    device,
                    nhid,
                    dropout,
                )
                for i in range(nlayers)
            ]
        )

        # Num Transmitter encoder
        self.num_transmitter_encoder = NumTransmitterEncoding(
            d_model, d_transmitter_encoding, max_transmitter=NR, dropout=dropout
        )

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_input(self, H, y, noise_sigma, NT):
        tgt = torch.chunk(H, 2, dim=2)[0].permute(2, 0, 1)
        # Normalizing y
        y = y / np.sqrt(2.0 * NT)
        y = torch.unsqueeze(y, dim=0).expand(NT, -1, -1)
        noise_sigma_normalized = (
            ((noise_sigma) / np.sqrt(2.0 * NT)).expand(NT, -1).unsqueeze(dim=-1)
        )
        src = torch.cat((y, tgt, noise_sigma_normalized), dim=-1)
        del H, y, noise_sigma, tgt
        return src

    def forward(self, H, y, noise_sigma, save_attn_weight=False):
        NT = H.shape[-1] // 2
        sout = self.generate_input(H, y, noise_sigma, NT)
        sout = self.num_transmitter_encoder(sout)
        sout = self.encoder_embed(sout) * math.sqrt(self.d_model)
        xout = torch.zeros(NT, H.shape[0], self.mod_n)
        sout = sout.to(device=self.device)
        xout = xout.to(device=self.device)

        x_list = []

        for index, encoder_decoder in enumerate(self.iterative_blocks):
            sout, xout = encoder_decoder.forward(
                sout, xout, H, y, noise_sigma, NT, index, save_attn_weight
            )
            x_list.append(xout)

        return x_list


# ---- Menagerie build/example plumbing (not part of the original repo) ----
def _qam_const(mod_n):
    """Faithful transcription of `sample_generator.QAM_const`/`QAM_N_const`
    (fully_correlated_channels/re-mimo/sample_generator.py), used only to
    synthesize the QAM-constellation constant buffers the model needs at
    construction time.
    """
    sqrt_mod_n = int(np.sqrt(mod_n))
    constellation = np.linspace(
        int(-np.sqrt(mod_n) + 1), int(np.sqrt(mod_n) - 1), int(np.sqrt(mod_n))
    )
    alpha = np.sqrt((constellation**2).mean())
    constellation /= alpha * np.sqrt(2)
    constellation = torch.tensor(constellation).to(dtype=torch.float32)

    real_qam_consts = torch.empty((mod_n,), dtype=torch.int64)
    imag_qam_consts = torch.empty((mod_n,), dtype=torch.int64)
    for i in range(sqrt_mod_n):
        for j in range(sqrt_mod_n):
            index = sqrt_mod_n * i + j
            real_qam_consts[index] = i
            imag_qam_consts[index] = j
    real_qam_const = constellation[real_qam_consts]
    imag_qam_const = constellation[imag_qam_consts]
    return constellation, real_qam_const, imag_qam_const


_NR = 8
_MOD_N = 16
_D_MODEL = 32
_N_HEAD = 4
_NHID = _D_MODEL * 4
_NLAYERS = 2
_D_TRANSMITTER_ENCODING = _NR


def build_re_mimo():
    torch.manual_seed(0)
    constel, real_qam_const, imag_qam_const = _qam_const(_MOD_N)
    model = iterative_classifier(
        d_model=_D_MODEL,
        n_head=_N_HEAD,
        nhid=_NHID,
        nlayers=_NLAYERS,
        mod_n=_MOD_N,
        NR=_NR,
        d_transmitter_encoding=_D_TRANSMITTER_ENCODING,
        real_QAM_const=real_qam_const,
        imag_QAM_const=imag_qam_const,
        constel=constel,
        device="cpu",
        dropout=0.0,
    )
    model.eval()
    return model


def example_input_re_mimo():
    torch.manual_seed(0)
    NT = 4
    batch_size = 2
    # H: real-valued 2*NR x 2*NT MIMO channel matrix (real/imag stacked block form,
    # as produced by `sample_generator.channel`).
    H = torch.randn(batch_size, 2 * _NR, 2 * NT)
    # y: received signal, 2*NR real-valued dimensions.
    y = torch.randn(batch_size, 2 * _NR)
    # noise_sigma: per-sample noise std.
    noise_sigma = torch.rand(batch_size) * 0.1 + 0.01
    return (H, y, noise_sigma)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("RE-MIMO", "build_re_mimo", "example_input_re_mimo", 2020, MENAGERIE_ZOO),
]
