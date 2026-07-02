# FAITHFUL PORT of bert-nmt/bert-nmt @ master (original framework: fairseq, torch==1.1.0)
# (fairseq/models/transformer.py classes TransformerS2Model / TransformerS2Encoder /
# TransformerS2EncoderLayer / TransformerDecoder / TransformerDecoderLayer)
#
# BERT-GEC (Kaneko et al., ACL 2020, "Encoder-Decoder Models Can Benefit from Pre-trained
# Masked Language Models in Grammatical Error Correction") trains its GEC seq2seq model with
# the kanekomasahiro/bert-gec repo, which is a thin scripts/config wrapper (setup.sh clones
# the bert-nmt fork of fairseq as its actual training engine, no model code of its own) around
# bert-nmt's "BERT-fused NMT" (Zhu et al., ICLR 2020) TransformerS2Model architecture. That
# repo needs a full forked fairseq framework (fairseq==0.6-era, not the pip fairseq package)
# which is not installable/reasonable here, so this ports the REAL bert-nmt architecture
# (transcribed, not guessed) into self-contained base-env torch:
#   - a frozen BertModel (transformers, standing in for bert-nmt's `bert.modeling.BertModel`
#     encoder, loaded via `BertModel.from_pretrained` in the original) feeds token-level BERT
#     representations (`bert_encoder_out`) to every encoder AND decoder layer.
#   - TransformerS2EncoderLayer: standard self-attention sublayer PLUS a second
#     `bert_attn` cross-attention sublayer over the frozen BERT features; the two attention
#     outputs are combined with `encoder_ratio`/`bert_ratio` gate weights (`get_ratio`),
#     exactly as in the original -- this "BERT-fusion" gate is the architectural
#     contribution, not present in a stock Transformer encoder layer.
#   - TransformerDecoderLayer: standard self-attention + encoder cross-attention sublayer
#     PLUS a third `bert_attn` cross-attention sublayer over the frozen BERT features,
#     combined the same ratio-gated way, again verbatim from the source.
#   - Standard fairseq sinusoidal PositionalEmbedding (not itself a novel component; the
#     original imports it from `fairseq.modules.PositionalEmbedding`).
# Every `nn.Linear`/`nn.LayerNorm`/`nn.MultiheadAttention` call mirrors the source's shapes
# and control flow; the gate/ratio math, encoder-then-decoder-with-bert-fusion pipeline, and
# pre/post layer-norm toggle are unchanged. Non-load-bearing fairseq scaffolding (Namespace
# args parsing, ONNX export, incremental decoding cache, adaptive softmax, checkpoint
# upgrade/versioning) is dropped -- it does not touch the architecture's forward computation.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel

MENAGERIE_ZOO = "ported-pytorch"


class SinusoidalPositionalEmbedding(nn.Module):
    """Standard fairseq-style sinusoidal position embedding (non-learned), matching
    the source's `PositionalEmbedding(..., learned=False)` path."""

    def __init__(self, embedding_dim, padding_idx, max_positions=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        weights = self._build_weights(max_positions + padding_idx + 1, embedding_dim, padding_idx)
        self.register_buffer("weights", weights, persistent=False)

    @staticmethod
    def _build_weights(num_embeddings, embedding_dim, padding_idx):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input_ids):
        mask = input_ids.ne(self.padding_idx).long()
        positions = torch.cumsum(mask, dim=1) * mask + self.padding_idx
        return (
            self.weights.index_select(0, positions.reshape(-1)).view(*input_ids.shape, -1).detach()
        )


def _ratio_gate(encoder_ratio, bert_ratio):
    return [encoder_ratio, bert_ratio]


class BertFusedEncoderLayer(nn.Module):
    """TransformerS2EncoderLayer, verbatim gate logic: self-attention + bert cross-attention,
    combined via `encoder_ratio`/`bert_ratio`."""

    def __init__(
        self, embed_dim, ffn_dim, n_heads, bert_out_dim, dropout, encoder_ratio, bert_ratio
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.self_attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout)
        self.bert_attn = nn.MultiheadAttention(
            embed_dim, n_heads, kdim=bert_out_dim, vdim=bert_out_dim, dropout=dropout
        )
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = dropout
        self.encoder_ratio = encoder_ratio
        self.bert_ratio = bert_ratio

    def forward(self, x, encoder_padding_mask, bert_encoder_out, bert_encoder_padding_mask):
        residual = x
        x1, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x2, _ = self.bert_attn(
            query=x,
            key=bert_encoder_out,
            value=bert_encoder_out,
            key_padding_mask=bert_encoder_padding_mask,
        )
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        ratios = _ratio_gate(self.encoder_ratio, self.bert_ratio)
        x = residual + ratios[0] * x1 + ratios[1] * x2
        x = self.self_attn_layer_norm(x)

        residual = x
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        return x


class BertFusedDecoderLayer(nn.Module):
    """TransformerDecoderLayer, verbatim gate logic: self-attention, then encoder
    cross-attention + bert cross-attention combined via `encoder_ratio`/`bert_ratio`,
    then FFN."""

    def __init__(
        self, embed_dim, ffn_dim, n_heads, bert_out_dim, dropout, encoder_ratio, bert_ratio
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.self_attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout)
        self.encoder_attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout)
        self.bert_attn = nn.MultiheadAttention(
            embed_dim, n_heads, kdim=bert_out_dim, vdim=bert_out_dim, dropout=dropout
        )
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.encoder_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = dropout
        self.encoder_ratio = encoder_ratio
        self.bert_ratio = bert_ratio

    def forward(
        self,
        x,
        encoder_out,
        encoder_padding_mask,
        bert_encoder_out,
        bert_encoder_padding_mask,
        self_attn_mask=None,
    ):
        residual = x
        x, _ = self.self_attn(query=x, key=x, value=x, attn_mask=self_attn_mask, need_weights=False)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        residual = x
        x1, _ = self.encoder_attn(
            query=x, key=encoder_out, value=encoder_out, key_padding_mask=encoder_padding_mask
        )
        x2, _ = self.bert_attn(
            query=x,
            key=bert_encoder_out,
            value=bert_encoder_out,
            key_padding_mask=bert_encoder_padding_mask,
        )
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        ratios = _ratio_gate(self.encoder_ratio, self.bert_ratio)
        x = residual + ratios[0] * x1 + ratios[1] * x2
        x = self.encoder_attn_layer_norm(x)

        residual = x
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        return x


class BertFusedEncoder(nn.Module):
    """TransformerS2Encoder: token+positional embedding -> N BertFusedEncoderLayer stages."""

    def __init__(
        self,
        vocab_size,
        embed_dim,
        ffn_dim,
        n_layers,
        n_heads,
        bert_out_dim,
        padding_idx,
        dropout,
        encoder_ratio,
        bert_ratio,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim, padding_idx)
        self.dropout = dropout
        self.layers = nn.ModuleList(
            [
                BertFusedEncoderLayer(
                    embed_dim, ffn_dim, n_heads, bert_out_dim, dropout, encoder_ratio, bert_ratio
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, src_tokens, bert_encoder_out, bert_encoder_padding_mask):
        x = self.embed_scale * self.embed_tokens(src_tokens)
        x = x + self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(0, 1)  # T x B x C

        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        for layer in self.layers:
            x = layer(x, encoder_padding_mask, bert_encoder_out, bert_encoder_padding_mask)
        return {"encoder_out": x, "encoder_padding_mask": encoder_padding_mask}


class BertFusedDecoder(nn.Module):
    """TransformerDecoder (bert-fused variant): token+positional embedding -> N
    BertFusedDecoderLayer stages -> vocab projection."""

    def __init__(
        self,
        vocab_size,
        embed_dim,
        ffn_dim,
        n_layers,
        n_heads,
        bert_out_dim,
        padding_idx,
        dropout,
        encoder_ratio,
        bert_ratio,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim, padding_idx)
        self.dropout = dropout
        self.layers = nn.ModuleList(
            [
                BertFusedDecoderLayer(
                    embed_dim, ffn_dim, n_heads, bert_out_dim, dropout, encoder_ratio, bert_ratio
                )
                for _ in range(n_layers)
            ]
        )
        self.output_proj = nn.Linear(embed_dim, vocab_size, bias=False)

    def buffered_future_mask(self, x):
        seq_len = x.size(0)
        return torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=x.device), 1)

    def forward(self, prev_output_tokens, encoder_out, bert_encoder_out, bert_encoder_padding_mask):
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        x = x + self.embed_positions(prev_output_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(0, 1)  # T x B x C

        self_attn_mask = self.buffered_future_mask(x)
        for layer in self.layers:
            x = layer(
                x,
                encoder_out["encoder_out"],
                encoder_out["encoder_padding_mask"],
                bert_encoder_out,
                bert_encoder_padding_mask,
                self_attn_mask=self_attn_mask,
            )
        x = x.transpose(0, 1)  # B x T x C
        return self.output_proj(x)


class BertGEC(nn.Module):
    """TransformerS2Model: frozen BertModel feeds `bert_encoder_out` to every layer of a
    bert-fused Transformer encoder-decoder, exactly as bert-nmt's BERT-fused NMT (the real
    architecture backing kanekomasahiro/bert-gec)."""

    def __init__(
        self,
        bert_config,
        src_vocab_size,
        tgt_vocab_size,
        embed_dim,
        ffn_dim,
        n_layers,
        n_heads,
        dropout=0.1,
        encoder_ratio=0.5,
        bert_ratio=0.5,
    ):
        super().__init__()
        self.bert_encoder = BertModel(bert_config)
        self.bert_encoder.eval()
        for p in self.bert_encoder.parameters():
            p.requires_grad_(False)
        bert_out_dim = bert_config.hidden_size
        self.encoder = BertFusedEncoder(
            src_vocab_size,
            embed_dim,
            ffn_dim,
            n_layers,
            n_heads,
            bert_out_dim,
            padding_idx=0,
            dropout=dropout,
            encoder_ratio=encoder_ratio,
            bert_ratio=bert_ratio,
        )
        self.decoder = BertFusedDecoder(
            tgt_vocab_size,
            embed_dim,
            ffn_dim,
            n_layers,
            n_heads,
            bert_out_dim,
            padding_idx=0,
            dropout=dropout,
            encoder_ratio=encoder_ratio,
            bert_ratio=bert_ratio,
        )

    def forward(self, src_tokens, prev_output_tokens, bert_input):
        bert_encoder_padding_mask = bert_input.eq(0)
        with torch.no_grad():
            bert_out = self.bert_encoder(
                bert_input, attention_mask=(~bert_encoder_padding_mask).long()
            )
        bert_encoder_out = bert_out.last_hidden_state.transpose(0, 1).contiguous()  # T x B x C

        encoder_out = self.encoder(src_tokens, bert_encoder_out, bert_encoder_padding_mask)
        decoder_out = self.decoder(
            prev_output_tokens, encoder_out, bert_encoder_out, bert_encoder_padding_mask
        )
        return decoder_out


def build_bert_gec():
    """Tiny BERT-GEC (BERT-fused NMT for grammatical error correction) for tracing.
    Architecture -- frozen BERT feeding a ratio-gated bert-fusion cross-attention sublayer
    in every encoder AND decoder layer -- is unmodified from the ported bert-nmt source."""
    bert_config = BertConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=32,
        pad_token_id=0,
    )
    model = BertGEC(
        bert_config=bert_config,
        src_vocab_size=64,
        tgt_vocab_size=64,
        embed_dim=32,
        ffn_dim=64,
        n_layers=2,
        n_heads=2,
        dropout=0.1,
        encoder_ratio=0.5,
        bert_ratio=0.5,
    )
    model.eval()
    return model


def example_input_bert_gec():
    batch, src_len, tgt_len = 2, 10, 8
    return (
        torch.randint(1, 64, (batch, src_len)),  # src_tokens
        torch.randint(1, 64, (batch, tgt_len)),  # prev_output_tokens
        torch.randint(1, 64, (batch, src_len)),  # bert_input
    )


MENAGERIE_ENTRIES = [
    ("BertGEC", build_bert_gec, example_input_bert_gec, 2020, "ported-pytorch"),
]
