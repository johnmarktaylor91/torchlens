# SOURCE: vendored from hebbarashwin/neural_polar_decoder @ main (models.py)
# https://raw.githubusercontent.com/hebbarashwin/neural_polar_decoder/main/models.py
#
# CRISP (ICML 2023, "CRISP: Curriculum based Sequential neural decoders for Polar
# codes"): the GPT-style autoregressive Transformer decoder used by the official repo
# (`--model gpt`, the default `MODEL = 'gpt'` in models.py). `XFormerEndToEndGPT`
# embeds the noisy channel LLRs, feeds them through a causal-masked stack of
# `EncoderLayer` blocks (self-attention + position-wise feed-forward, GPT-style
# despite the `EncoderLayer` name in the source) built from `MultiHeadAttention` /
# `ScaledDotProductAttention` / `PositionwiseFeedForward`, and projects to a per-bit
# sigmoid decision (`Lin_Decoder`). Classes below (`get_pad_mask`, `get_subsequent_
# mask`, `ScaledDotProductAttention`, `ScalarMult`, `MultiHeadAttention`,
# `PositionwiseFeedForward`, `EncoderLayer`, `PositionalEncoding`, `XFormerGPT`,
# `XFormerEndToEndGPT`) are copied verbatim from models.py, including the in-place
# `trg_seq[:, 0] = start_emb` assignment in `forward`; only the unused sibling
# classes (RNN/conv/encoder-decoder variants, `polar.py`/`pac_code.py`/`utils.py`
# training-only imports) are dropped since `XFormerEndToEndGPT.forward` does not
# reference them.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

MENAGERIE_ZOO = "vendored-pytorch"


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    """For masking out the subsequent info."""
    sz_b, len_s = seq.size()
    subsequent_mask = (
        1 - torch.triu(torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)
    ).bool()
    return subsequent_mask


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None, causal=False):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class ScalarMult(nn.Module):
    """scalar multiplication layer"""

    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(1e-10 * torch.ones(1))

    def forward(self, x):
        out = self.alpha * x
        return out


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k**0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scalar = ScalarMult()

    def forward(self, q, k, v, mask=None, causal=False):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module"""

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.scalar = ScalarMult()

    def forward(self, x):
        residual = x

        x = self.w_2(F.gelu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class EncoderLayer(nn.Module):
    """Compose with two layers"""

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200, num=10000):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer("pos_table", self._get_sinusoid_encoding_table(n_position, d_hid, num))

    def _get_sinusoid_encoding_table(self, n_position, d_hid, num):
        """Sinusoid position encoding table"""

        def get_position_angle_vec(position, num):
            return [position / np.power(num, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i, num) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, : x.size(1)].clone().detach()


class XFormerGPT(nn.Module):
    def __init__(self, config, layer_idx=None):
        super(XFormerGPT, self).__init__()
        self.embed_dim = config.embed_dim
        self.block_len = config.max_len
        self.position_enc_auto = PositionalEncoding(self.embed_dim, n_position=self.block_len)
        self.dropout = nn.Dropout(p=config.dropout)
        self.layer_stack = nn.ModuleList(
            [
                EncoderLayer(
                    config.embed_dim,
                    config.embed_dim * 4,
                    config.n_head,
                    config.embed_dim // config.n_head,
                    config.embed_dim // config.n_head,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.layer_norm_cross = nn.LayerNorm(config.embed_dim, eps=1e-6)

    def forward(self, trg_seq, trg_mask, device, return_attns=False, return_layer=None):
        dec_slf_attn_list, dec_enc_attn_list = [], []  # noqa: F841 (verbatim from source)
        dec_output = self.position_enc_auto(trg_seq)
        dec_output = self.dropout(dec_output)
        layer = 1
        intermediate_layer_out = None
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(dec_output, slf_attn_mask=trg_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            if return_layer is not None:
                if layer == return_layer:
                    intermediate_layer_out = dec_output
            layer += 1
        if return_attns:
            return dec_output, dec_slf_attn_list
        if return_layer is not None:
            return dec_output, intermediate_layer_out
        return dec_output  # [b_size,block_len,embed_dim]


class XFormerEndToEndGPT(nn.Module):
    def __init__(self, config):
        super(XFormerEndToEndGPT, self).__init__()
        self.embed_dim = config.embed_dim
        self.block_len = config.max_len
        self.trg_pad_idx = 2
        MODEL = config.model  # noqa: F841 (verbatim from source; module-level MODEL not reassigned)
        self.start_embed_layer = nn.Sequential(
            nn.Linear(config.N, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.learnt_pos = True
        if not self.learnt_pos:
            self.emb_inputs = nn.Embedding(2, self.embed_dim)
        else:
            self.pos_emb = nn.Embedding(self.block_len, config.embed_dim)
        self.layer_norm_inp = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.layer_norm_out = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.Decoder = XFormerGPT(config)
        self.Lin_Decoder = nn.Linear(config.embed_dim, 1)

    def forward(self, noisy_enc, mask, trg_seq, device, return_layer=None):
        src_mask = mask  # noqa: F841 (verbatim from source)
        trg_seq = trg_seq[:, :-1]
        if not self.learnt_pos:
            trg_seq = torch.cat(
                (torch.ones((trg_seq.size(0), 1), device=device).long(), (trg_seq == -1).long()), -1
            )
            trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)
            trg_seq = self.emb_inputs(trg_seq)
        else:
            trg_seq = torch.cat((torch.ones((trg_seq.size(0), 1), device=device), trg_seq), -1)
            trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)
            trg_seq = torch.ones(self.embed_dim, device=device) * trg_seq.unsqueeze(-1)
            position_indices = torch.arange(self.block_len, device=device)
            pos_enc = self.pos_emb(position_indices)
            trg_seq = trg_seq * pos_enc

        start_emb = self.start_embed_layer(noisy_enc)
        trg_seq[:, 0] = start_emb
        if return_layer is not None:
            output, intermediate_layer_out = self.Decoder(
                trg_seq, trg_mask, device, return_layer=return_layer
            )
        else:
            output = self.Decoder(trg_seq, trg_mask, device)
        logits = self.Lin_Decoder(output)

        decoded_msg_bits = logits.sign()
        output = torch.sigmoid(logits)
        output = torch.cat((1 - output, output), -1)
        out_mask = mask

        if return_layer is not None:
            return output, decoded_msg_bits, out_mask, logits, intermediate_layer_out

        return output, decoded_msg_bits, out_mask, logits  # [b_size,block_len,2]


class _CRISPConfig:
    """Minimal stand-in for the argparse.Namespace `config`/`args` object that
    models.py's XFormerEndToEndGPT expects (embed_dim, max_len, dropout, n_head,
    n_layers, N, model). Field names match run_models.py's argparse flags exactly."""

    def __init__(self, N=16, max_len=16, embed_dim=32, n_head=4, n_layers=2, dropout=0.1):
        self.N = N
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.n_layers = n_layers
        self.dropout = dropout
        self.model = "gpt"


def build_crisp():
    # Tiny menagerie-scale polar-code block length (N=16, matching --N in run_crisp.sh
    # style invocations, shrunk from the paper's N=64/128) with embed_dim divisible by
    # n_head as models.py requires (embed_dim // n_head used as d_k/d_v).
    config = _CRISPConfig(N=16, max_len=16, embed_dim=32, n_head=4, n_layers=2, dropout=0.0)
    model = XFormerEndToEndGPT(config)
    model.eval()
    return model


def example_input_crisp():
    # noisy_enc: [batch, N] received LLRs/channel outputs. mask: [batch, 1, max_len]
    # boolean attention mask (all-True = attend everywhere, matching the source's
    # `get_pad_mask`/full-block usage during training). trg_seq: [batch, max_len]
    # teacher-forced target bit sequence (+/-1 valued per the source's BPSK convention).
    # device: the source's forward() takes device explicitly rather than inferring it.
    torch.manual_seed(0)
    N = 16
    max_len = 16
    batch = 2
    noisy_enc = torch.randn(batch, N)
    mask = torch.ones(batch, 1, max_len, dtype=torch.bool)
    trg_seq = torch.where(torch.rand(batch, max_len) > 0.5, 1.0, -1.0)
    device = torch.device("cpu")
    return (noisy_enc, mask, trg_seq, device)


MENAGERIE_ENTRIES = [
    (
        "CRISP (Curriculum Sequential Neural Decoder for Polar codes, GPT variant)",
        "build_crisp",
        "example_input_crisp",
        2023,
        "vendored-pytorch",
    ),
]
