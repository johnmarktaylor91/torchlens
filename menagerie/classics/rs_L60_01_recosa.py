# SOURCE: vendored from devjwsong/recosa-dialogue-generation-pytorch @ 4ddedaef45f31d75e88bdb909a4451173faec4c8
# Files: src/recosa_transformer.py + src/layers.py (MIT License, Jaewoo Song, 2021)
#
# ReCoSa (Detecting the Relevant Contexts with Self-Attention for Multi-turn Dialogue
# Generation, Zhang et al. ACL 2019). The official zhanghainan/ReCoSa repo is TensorFlow
# 1.x (tf.variable_scope-based, Kyubyong Park transformer template) and cannot run in the
# base torch env. This is the widely-used community PyTorch re-implementation
# (devjwsong/recosa-dialogue-generation-pytorch), a faithful transformer
# encoder/decoder over per-utterance word-level GRU context embeddings, matching the
# architecture described in the paper. Vendored verbatim (only import paths and a
# `torchlens`-friendly forward-arg surface were adapted; no architectural changes).
#
# Original forward signature (recosa_transformer.py):
#   forward(self, src_inputs, trg_inputs, src_poses, trg_poses, e_masks, d_masks)
#     src_inputs: (B, T, S_L) per-utterance token ids for T context utterances
#     trg_inputs: (B, T_L) response token ids
#     src_poses:  (B, T) utterance position ids (for context-level positional embedding)
#     trg_poses:  (B, T_L) response token position ids
#     e_masks:    (B, T) context-utterance attention mask
#     d_masks:    (B, T_L, T_L) causal + padding response self-attention mask

import math

import torch
from torch import nn
from torch.nn import functional as F

MENAGERIE_ZOO = "vendored-pytorch"


class LayerNormalization(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.layer = nn.LayerNorm([d_model], elementwise_affine=True, eps=eps)

    def forward(self, x):
        x = self.layer(x)

        return x


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.inf = 1e9
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # W^Q, W^K, W^V in the paper
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # Final output linear transformation
        self.w_0 = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        input_shape = q.shape

        # Linear calculation +  split into num_heads
        q = self.w_q(q).view(input_shape[0], -1, self.num_heads, self.d_k)  # (B, L, H, d_k)
        k = self.w_k(k).view(input_shape[0], -1, self.num_heads, self.d_k)  # (B, L, H, d_k)
        v = self.w_v(v).view(input_shape[0], -1, self.num_heads, self.d_k)  # (B, L, H, d_k)

        # For convenience, convert all tensors in size (B, H, L, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Conduct self-attention
        attn_values = self.self_attention(q, k, v, mask=mask)  # (B, H, L, d_k)
        concat_output = (
            attn_values.transpose(1, 2).contiguous().view(input_shape[0], -1, self.d_model)
        )  # (B, L, d_model)

        return self.w_0(concat_output)

    def self_attention(self, q, k, v, mask=None):
        # Calculate attention scores with scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # (B, H, L, L)
        attn_scores = attn_scores / math.sqrt(self.d_k)

        # If there is a mask, make masked spots -INF
        if mask is not None:
            mask = mask.unsqueeze(1)  # (B, 1, L) => (B, 1, 1, L) or (B, L, L) => (B, 1, L, L)
            attn_scores = attn_scores.masked_fill(mask == 0, -1 * self.inf)

        # Softmax and multiplying K to calculate attention value
        attn_distribs = F.softmax(attn_scores, dim=-1)

        attn_distribs = self.dropout(attn_distribs)
        attn_values = torch.matmul(attn_distribs, v)  # (B, H, L, d_k)

        return attn_values


class FeedFowardLayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff, bias=True)
        self.linear_2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.linear_1(x))  # (B, L, d_ff)
        x = self.dropout(x)
        x = self.linear_2(x)  # (B, L, d_model)

        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout):
        super().__init__()
        self.layer_norm_1 = LayerNormalization(d_model)
        self.multihead_attention = MultiheadAttention(d_model, num_heads, dropout)
        self.drop_out_1 = nn.Dropout(dropout)

        self.layer_norm_2 = LayerNormalization(d_model)
        self.feed_forward = FeedFowardLayer(d_model, d_ff, dropout)
        self.drop_out_2 = nn.Dropout(dropout)

    def forward(self, x, e_masks):  # x: (B, T, d_model), e_masks: (B, T)
        x_1 = self.layer_norm_1(x)  # (B, T, d_model)
        x = x + self.drop_out_1(
            self.multihead_attention(x_1, x_1, x_1, mask=e_masks)
        )  # (B, L, d_model)
        x_2 = self.layer_norm_2(x)  # (B, L, d_model)
        x = x + self.drop_out_2(self.feed_forward(x_2))  # (B, L, d_model)

        return x  # (B, L, d_model)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout):
        super().__init__()
        self.layer_norm_1 = LayerNormalization(d_model)
        self.masked_multihead_attention = MultiheadAttention(d_model, num_heads, dropout)
        self.drop_out_1 = nn.Dropout(dropout)

        self.layer_norm_2 = LayerNormalization(d_model)
        self.multihead_attention = MultiheadAttention(d_model, num_heads, dropout)
        self.drop_out_2 = nn.Dropout(dropout)

        self.layer_norm_3 = LayerNormalization(d_model)
        self.feed_forward = FeedFowardLayer(d_model, d_ff, dropout)
        self.drop_out_3 = nn.Dropout(dropout)

    def forward(self, x, e_outputs, e_masks, d_masks):
        # x: (B, L, d_model), e_outputs: (B, T, d_model), e_masks: (B, T), d_masks: (B, L, L)
        x_1 = self.layer_norm_1(x)  # (B, L, d_model)
        x = x + self.drop_out_1(
            self.masked_multihead_attention(x_1, x_1, x_1, mask=d_masks)
        )  # (B, L, d_model)
        x_2 = self.layer_norm_2(x)  # (B, L, d_model)
        x = x + self.drop_out_2(
            self.multihead_attention(x_2, e_outputs, e_outputs, mask=e_masks)
        )  # (B, L, d_model)
        x_3 = self.layer_norm_3(x)  # (B, L, d_model)
        x = x + self.drop_out_3(self.feed_forward(x_3))  # (B, L, d_model)

        return x  # (B, L, d_model)


class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout, num_layers):
        super().__init__()
        self.num_layers = num_layers

        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, d_ff, num_heads, dropout) for _ in range(num_layers)]
        )
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, x, e_masks):  # x: (B, T, d_model), e_masks: (B, T)
        for i in range(self.num_layers):
            x = self.layers[i](x, e_masks)

        return self.layer_norm(x)


class Decoder(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout, num_layers):
        super().__init__()
        self.num_layers = num_layers

        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, d_ff, num_heads, dropout) for _ in range(num_layers)]
        )
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, x, e_outputs, e_masks, d_masks):
        for i in range(self.num_layers):
            x = self.layers[i](x, e_outputs, e_masks, d_masks)

        return self.layer_norm(x)


class _ReCoSaArgs:
    """Tiny stand-in for the original repo's argparse Namespace."""

    def __init__(
        self,
        vocab_size=64,
        d_model=16,
        d_pos=4,
        trg_max_len=12,
        num_gru_layers=1,
        gru_dropout=0.0,
        d_ff=32,
        num_heads=2,
        dropout=0.0,
        num_encoder_layers=2,
        num_decoder_layers=2,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_pos = d_pos
        self.trg_max_len = trg_max_len
        self.num_gru_layers = num_gru_layers
        self.gru_dropout = gru_dropout
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers


class ReCoSaTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()

        d_emb = args.d_model - args.d_pos
        self.word_embedding = nn.Embedding(args.vocab_size, d_emb)
        self.pos_embedding = nn.Embedding(args.trg_max_len, args.d_pos)

        # Word Level GRU components
        self.gru = nn.GRU(
            input_size=d_emb,
            hidden_size=d_emb,
            num_layers=args.num_gru_layers,
            dropout=(0.0 if args.num_gru_layers == 1 else args.gru_dropout),
            batch_first=True,
        )

        # Encoder & Decoder
        self.encoder = Encoder(
            args.d_model,
            args.d_ff,
            args.num_heads,
            args.dropout,
            args.num_encoder_layers,
        )
        self.decoder = Decoder(
            args.d_model,
            args.d_ff,
            args.num_heads,
            args.dropout,
            args.num_decoder_layers,
        )

        self.output_linear = nn.Linear(args.d_model, args.vocab_size)

    def init_model(self):
        # Initialize parameters
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, src_inputs, trg_inputs, src_poses, trg_poses, e_masks, d_masks):
        # src_inputs: (B, T, S_L), trg_inputs: (B, T_L), src_poses: (B, T),
        # trg_poses: (B, T_L), e_masks: (B, T), d_masks: (B, T_L, T_L)
        # Embeddings & Masking
        src_embs = self.src_embedding(src_inputs, src_poses)  # (B, T, d_model)
        trg_embs = self.trg_embedding(trg_inputs, trg_poses)  # (B, T_L, d_model)

        # Encoding phase
        e_outputs = self.encoder(src_embs, e_masks)  # (B, T, d_model)

        # Decoding phase
        d_outputs = self.decoder(trg_embs, e_outputs, e_masks, d_masks)  # (B, L, d_model)

        return self.output_linear(d_outputs)  # (B, L, vocab_size)

    def src_embedding(self, src_inputs, src_poses):  # src_inputs: (B, T, S_L), src_poses: (B, T)
        src_embs = self.word_embedding(src_inputs)  # (B, T, L, d_emb)
        max_len, d_emb = src_embs.shape[2], src_embs.shape[3]
        last_hiddens = self.gru(src_embs.view(-1, max_len, d_emb))[1][-1]  # (B*T, d_emb)

        batch_size = src_embs.shape[0]
        src_embs = last_hiddens.view(batch_size, -1, d_emb)  # (B, T, d_emb)
        pos_embs = self.pos_embedding(src_poses)  # (B, T, d_pos)
        src_embs = torch.cat((src_embs, pos_embs), dim=-1)  # (B, T, d_model)

        return src_embs  # (B, T, d_model)

    def trg_embedding(self, trg_inputs, trg_poses):  # trg_inputs: (B, T_L), trg_poses: (B, T_L)
        trg_embs = self.word_embedding(trg_inputs)  # (B, T_L, d_emb)
        pos_embs = self.pos_embedding(trg_poses)  # (B, T_L, d_pos)
        trg_embs = torch.cat((trg_embs, pos_embs), dim=-1)  # (B, T_L, d_model)

        return trg_embs  # (B, T_L, d_model)


def build_recosa():
    args = _ReCoSaArgs()
    m = ReCoSaTransformer(args)
    m.init_model()
    m.eval()
    return m


def example_input_recosa():
    args = _ReCoSaArgs()
    batch = 1
    n_ctx_utts = 4
    utt_len = 6
    trg_len = 5

    src_inputs = torch.randint(1, args.vocab_size, (batch, n_ctx_utts, utt_len))
    trg_inputs = torch.randint(1, args.vocab_size, (batch, trg_len))
    src_poses = torch.arange(n_ctx_utts).unsqueeze(0).expand(batch, -1).clone()
    trg_poses = torch.arange(trg_len).unsqueeze(0).expand(batch, -1).clone()
    e_masks = torch.ones(batch, n_ctx_utts)
    d_masks = torch.tril(torch.ones(trg_len, trg_len)).unsqueeze(0).expand(batch, -1, -1).clone()

    return (src_inputs, trg_inputs, src_poses, trg_poses, e_masks, d_masks)


MENAGERIE_ENTRIES = [
    (
        "ReCoSa (Relevant Context Self-Attention Dialogue)",
        build_recosa,
        example_input_recosa,
        2019,
        "vendored-pytorch",
    ),
]
