# SOURCE: vendored from YuemingJin/Trans-SVNet_Journal @ main
# https://raw.githubusercontent.com/YuemingJin/Trans-SVNet_Journal/main/code_80/transformer2_3_1.py
# https://raw.githubusercontent.com/YuemingJin/Trans-SVNet_Journal/main/code_80/tecno_trans.py
#
# Jin, Yu, Dou, Heng, 2022 (IJCARS 2022, MICCAI 2021 Best Paper follow-up journal
# extension) "Trans-SVNet: hybrid embedding aggregation Transformer for surgical
# workflow analysis with phase and step anticipation". Trans-SVNet's novel
# architectural contribution is the small hybrid Transformer fusion head
# (`Transformer2_3_1`, a from-scratch encoder-decoder cross-attention Transformer
# implementing scaled-dot-product `MultiHeadAttention` + `PoswiseFeedForwardNet` +
# stacked `EncoderLayer`/`DecoderLayer`) that fuses TeCNO's per-frame spatial CNN
# embedding with its own long-range MS-TCN temporal embedding, wrapped by the
# `Transformer` class in `tecno_trans.py` that windows the spatial-embedding
# sequence into `len_q`-length queries and feeds the temporal embedding through
# as encoder input / spatial windows as decoder input. Vendored verbatim below
# (`ScaledDotProductAttention`, `MultiHeadAttention`, `PoswiseFeedForwardNet`,
# `EncoderLayer`, `Encoder`, `DecoderLayer`, `Decoder`, `Transformer2_3_1` from
# `transformer2_3_1.py`, plus the `Transformer` wrapper class from
# `tecno_trans.py`). Fixes are limited to portability: every hardcoded
# `.cuda()` call inside `MultiHeadAttention.forward`/`PoswiseFeedForwardNet.forward`/
# `Transformer2_3_1.__init__`/the `Transformer` wrapper's `torch.zeros(...).cuda()`
# padding tensor is dropped (device-follows-input instead) -- no architectural change,
# purely device portability so the module runs on CPU.

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"

# ============================================================================
# transformer2_3_1.py (verbatim, `.cuda()` calls dropped for CPU portability)
# ============================================================================


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, n_heads):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.n_heads = n_heads

    def forward(self, Q, K, V):
        """
        Q: [batch_size, n_heads, len_q=1, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.d_k**0.5)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, len_q, len_k):
        super(MultiHeadAttention, self).__init__()

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.ScaledDotProductAttention = ScaledDotProductAttention(self.d_k, n_heads)
        self.len_q = len_q
        self.len_k = len_k

    def forward(self, input_Q, input_K, input_V):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        """
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        context, attn = self.ScaledDotProductAttention(Q, K, V)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        output = self.fc(context)
        return self.layer_norm(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False),
        )
        self.d_model = d_model
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, d_model]
        """
        residual = inputs
        output = self.fc(inputs)
        return self.layer_norm(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_heads, len_q):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads, 1, len_q)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs):
        """
        enc_inputs: [batch_size, src_len, d_model]
        """
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_layers, n_heads, len_q):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, d_ff, d_k, d_v, n_heads, len_q) for _ in range(n_layers)]
        )

    def forward(self, enc_inputs):
        """
        enc_inputs: [batch_size, src_len, d_model]
        """
        enc_outputs = enc_inputs
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_heads, len_q):
        super(DecoderLayer, self).__init__()
        self.dec_enc_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads, 1, len_q)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, dec_inputs, enc_outputs):
        """
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        """
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_inputs, enc_outputs, enc_outputs)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_layers, n_heads, len_q):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, d_ff, d_k, d_v, n_heads, len_q) for _ in range(n_layers)]
        )

    def forward(self, dec_inputs, enc_outputs):
        """
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_intpus: [batch_size, src_len, d_model]
        enc_outputs: [batsh_size, src_len, d_model]
        """
        dec_outputs = dec_inputs
        dec_enc_attns = []
        for layer in self.layers:
            dec_outputs, dec_enc_attn = layer(dec_outputs, enc_outputs)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs


class Transformer2_3_1(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_layers, n_heads, len_q):
        super(Transformer2_3_1, self).__init__()
        self.encoder = Encoder(d_model, d_ff, d_k, d_v, n_layers, n_heads, len_q)
        self.decoder = Decoder(d_model, d_ff, d_k, d_v, 1, n_heads, len_q)

    def forward(self, enc_inputs, dec_inputs):
        """
        enc_inputs: [batch_size, src_len, d_model]
        """
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs = self.decoder(dec_inputs, enc_outputs)
        return dec_outputs


# ============================================================================
# tecno_trans.py (Transformer wrapper class, verbatim except the padding
# tensor's `.cuda()` -> `.to(long_feature.device)`)
# ============================================================================


class Transformer(nn.Module):
    def __init__(self, mstcn_f_maps, mstcn_f_dim, out_features, len_q, sequence_length):
        super(Transformer, self).__init__()
        self.num_f_maps = mstcn_f_maps
        self.dim = mstcn_f_dim
        self.num_classes = out_features
        self.len_q = len_q

        self.transformer = Transformer2_3_1(
            d_model=out_features,
            d_ff=mstcn_f_maps,
            d_k=mstcn_f_maps,
            d_v=mstcn_f_maps,
            n_layers=1,
            n_heads=8,
            len_q=sequence_length,
        )
        self.fc = nn.Linear(mstcn_f_dim, out_features, bias=False)

    def forward(self, x, long_feature):
        out_features = x.transpose(1, 2)
        inputs = []
        for i in range(out_features.size(1)):
            if i < self.len_q - 1:
                pad = torch.zeros((1, self.len_q - 1 - i, self.num_classes), device=x.device)
                input_i = torch.cat([pad, out_features[:, 0 : i + 1]], dim=1)
            else:
                input_i = out_features[:, i - self.len_q + 1 : i + 1]
            inputs.append(input_i)
        inputs = torch.stack(inputs, dim=0).squeeze(1)
        feas = torch.tanh(self.fc(long_feature).transpose(0, 1))
        output = self.transformer(inputs, feas)
        return output


# ============================================================================
# build_/example_input_ harness
# ============================================================================


def build_transsvnet():
    """Real `tecno_trans.py` driver constructs
    `Transformer(mstcn_f_maps=32, mstcn_f_dim=2048, out_features=14,
    sequence_length=30)`; shrunk here (mstcn_f_maps=8, mstcn_f_dim=16,
    out_features=7, sequence_length=6) for a fast trace -- n_heads=8,
    n_layers=1 kept exactly as the real constructor call."""
    sequence_length = 6
    model = Transformer(
        mstcn_f_maps=8,
        mstcn_f_dim=16,
        out_features=7,
        len_q=sequence_length,
        sequence_length=sequence_length,
    )
    model.eval()
    return model


def example_input_transsvnet():
    torch.manual_seed(0)
    sequence_length = 6
    out_features = 7
    mstcn_f_dim = 16
    # x: TeCNO's MS-TCN classification logits, [batch, out_features, seq_len]
    x = torch.randn(1, out_features, sequence_length)
    # long_feature: the long-range spatial-embedding memory bank, [batch, seq_len, mstcn_f_dim]
    long_feature = torch.randn(1, sequence_length, mstcn_f_dim)
    return (x, long_feature)


MENAGERIE_ENTRIES = [
    (
        "Trans-SVNet",
        build_transsvnet,
        example_input_transsvnet,
        2022,
        "vendored-pytorch",
    ),
]
