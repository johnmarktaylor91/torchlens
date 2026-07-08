# SOURCE: vendored from a96123155/TransPHLA-AOMP @ 3ed2260292934170757507a71e645d0bcadfc44b
# https://raw.githubusercontent.com/a96123155/TransPHLA-AOMP/master/TransPHLA-AOMP/model.py
#
# Chu, Zhang, Yang, Zhao, Xie, Chen 2022 (Nature Machine Intelligence) "A transformer-based
# model to predict peptide-HLA class I binding and optimize mutated peptides for vaccine
# design". `Transformer` (`TransPHLA-AOMP/model.py`, referred to as "pHLAIformer" in the
# paper) is the real PyTorch `nn.Module`: two independent `Encoder` towers (one for the
# peptide sequence, one for the HLA pseudo-sequence), each a stack of sinusoidal
# `PositionalEncoding` + `EncoderLayer` (custom `MultiHeadAttention` with an explicit
# `ScaledDotProductAttention` submodule + `PoswiseFeedForwardNet`, both post-residual
# `nn.LayerNorm`, matching a from-scratch "Attention is All You Need"-style encoder, not
# `nn.TransformerEncoder`), whose outputs are concatenated along the sequence dimension,
# passed through a `Decoder` (reusing the same `EncoderLayer`/positional-encoding stack
# over the concatenated peptide+HLA sequence -- a "decoder" in name only, with no
# cross-attention to a separate encoder output), flattened, and projected through a
# 3-layer MLP head (`Linear->ReLU->BatchNorm1d->Linear->ReLU->Linear`) to a 2-way
# binding/non-binding logit. `MultiHeadAttention`/`PoswiseFeedForwardNet`/`EncoderLayer`/
# `Encoder`/`DecoderLayer`/`Decoder`/`Transformer` and the `get_attn_pad_mask` masking
# helper are copied verbatim from the real repo, including the module-level
# hyperparameters (`d_model=64`, `d_ff=512`, `d_k=d_v=64`, `n_layers=1`, `n_heads=9`,
# `pep_max_len=15`, `hla_max_len=34`) used by the released `pHLAIformer.pkl` checkpoint.
# The real file loads its amino-acid vocabulary from a repo-local `vocab_dict.npy`
# (Git-LFS pointer, not fetched here) and reads training/eval CSVs; this staging module
# drops those data-loading and training-script portions (`read_predict_data`,
# `make_data`, `MyDataSet`, `eval_step`, `transfer`, plotting/sklearn imports) since they
# are not part of the traced `Transformer.forward()` call, and substitutes a fixed
# vocab_size=25 (the real vocab is a small char->int map over the 20 standard amino acids
# plus a handful of ambiguity/padding symbols) for the on-disk `vocab_dict.npy`. No
# architectural changes were made to the `Transformer` module itself.

import numpy as np
import torch
import torch.nn as nn
import math

pep_max_len = 15  # peptide; enc_input max sequence length
hla_max_len = 34  # hla; dec_input(=dec_output) max sequence length
tgt_len = pep_max_len + hla_max_len

# The real repo loads this from a Git-LFS `vocab_dict.npy` (char -> int map over the 20
# standard amino acids + a small number of ambiguity/padding symbols); vocab_size=25
# stands in for that fixed small vocabulary (used only to size nn.Embedding for this trace).
vocab_size = 25

# Transformer Parameters (real repo module-level constants, matching the released
# pHLAIformer.pkl checkpoint hyperparameters)
d_model = 64  # Embedding Size
d_ff = 512  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V

n_layers, n_heads = 1, 9

use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
    """
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
            d_k
        )  # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(
            attn_mask, -1e9
        )  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.use_cuda = use_cuda
        device = torch.device("cuda" if self.use_cuda else "cpu")  # noqa: F841 (verbatim from real repo; unused there too)
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = (
            self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        )  # Q: [batch_size, n_heads, len_q, d_k]
        K = (
            self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        )  # K: [batch_size, n_heads, len_k, d_k]
        V = (
            self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
        )  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(
            1, n_heads, 1, 1
        )  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(
            batch_size, -1, n_heads * d_v
        )  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).to(device)(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.use_cuda = use_cuda
        device = torch.device("cuda" if self.use_cuda else "cpu")  # noqa: F841 (verbatim from real repo; unused there too)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False), nn.ReLU(), nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, d_model]
        """
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).to(device)(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        """
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(
            enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask
        )  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        """
        enc_inputs: [batch_size, src_len]
        """
        enc_outputs = self.src_emb(enc_inputs)  # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(
            0, 1
        )  # [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(
            enc_inputs, enc_inputs
        )  # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, dec_self_attn_mask):  # dec_inputs = enc_outputs
        """
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        """
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(
            dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask
        )
        dec_outputs = self.pos_ffn(dec_outputs)  # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.use_cuda = use_cuda
        device = torch.device("cuda" if self.use_cuda else "cpu")  # noqa: F841 (verbatim from real repo; unused there too)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])
        self.tgt_len = tgt_len

    def forward(
        self, dec_inputs
    ):  # dec_inputs = enc_outputs (batch_size, peptide_hla_maxlen_sum, d_model)
        """
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batsh_size, src_len, d_model]
        """
        dec_outputs = (
            self.pos_emb(dec_inputs.transpose(0, 1)).transpose(0, 1).to(device)
        )  # [batch_size, tgt_len, d_model]
        dec_self_attn_pad_mask = (
            torch.LongTensor(np.zeros((dec_inputs.shape[0], tgt_len, tgt_len))).bool().to(device)
        )

        dec_self_attns = []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
            dec_outputs, dec_self_attn = layer(dec_outputs, dec_self_attn_pad_mask)
            dec_self_attns.append(dec_self_attn)

        return dec_outputs, dec_self_attns


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.use_cuda = use_cuda
        device = torch.device("cuda" if use_cuda else "cpu")
        self.pep_encoder = Encoder().to(device)
        self.hla_encoder = Encoder().to(device)
        self.decoder = Decoder().to(device)
        self.tgt_len = tgt_len
        self.projection = nn.Sequential(
            nn.Linear(tgt_len * d_model, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(True),
            # output layer
            nn.Linear(64, 2),
        ).to(device)

    def forward(self, pep_inputs, hla_inputs):
        """
        pep_inputs: [batch_size, pep_len]
        hla_inputs: [batch_size, hla_len]
        """
        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        pep_enc_outputs, pep_enc_self_attns = self.pep_encoder(pep_inputs)
        hla_enc_outputs, hla_enc_self_attns = self.hla_encoder(hla_inputs)
        enc_outputs = torch.cat((pep_enc_outputs, hla_enc_outputs), 1)  # concat pep & hla embedding

        # dec_outpus: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attns = self.decoder(enc_outputs)
        dec_outputs = dec_outputs.view(
            dec_outputs.shape[0], -1
        )  # Flatten [batch_size, tgt_len * d_model]
        dec_logits = self.projection(
            dec_outputs
        )  # dec_logits: [batch_size, tgt_len, tgt_vocab_size]

        return (
            dec_logits.view(-1, dec_logits.size(-1)),
            pep_enc_self_attns,
            hla_enc_self_attns,
            dec_self_attns,
        )


# --- staging harness (tiny sizes; not part of the real repo) ---


def build_transphla():
    return Transformer()


def example_input_transphla():
    # pep_max_len=15 and hla_max_len=34 are the real repo's fixed padded lengths
    # (padded with vocab index 0, the "-" pad character in the real vocab_dict.npy).
    batch = 4
    pep_inputs = torch.randint(1, vocab_size, (batch, pep_max_len))
    hla_inputs = torch.randint(1, vocab_size, (batch, hla_max_len))
    return (pep_inputs, hla_inputs)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("TransPHLA", "build_transphla", "example_input_transphla", 2022, "vendored"),
]
