"""Compact faithful reimplementations of four dialogue / retrieval NLP families.

Sources checked (paper + official source, no clone/pip-install; reimplemented from scratch):
  - MTTOD (Multi-Task TOD with Span Prediction): arxiv:2109.14739 (Lee, EMNLP 2021); official
    repo github.com/bepoetree/MTTOD (`model.py::T5WithSpan`, `T5WithTokenSpan`) -- a shared T5
    encoder feeds *two* separate T5 decoders: a belief-state decoder (predicts the dialogue
    state / DB-query span, the standard T5 `decoder`) and a dedicated response decoder
    (`resp_decoder`, an independently-weighted T5 decoder stack initialized as a clone of the
    belief decoder) that cross-attends the same encoder states to generate the system
    response -- i.e. one encoder, two decoding heads specialized per sub-task. In parallel, a
    token-level linear span head (`span_head`) reads the raw encoder hidden states directly
    (before any decoding) to predict a BIO-style span label per input token, an auxiliary task
    that grounds slot values to input spans without extra annotation cost. All three modules
    (encoder, both decoders, span head) train jointly.
  - NCI (Neural Corpus Indexer): arxiv:2206.02743 (Wang et al., NeurIPS 2022); official repo
    github.com/solidsea98/Neural-Corpus-Indexer-NCI (`NCI_model/transformers/modeling_t5.py`,
    the `adaptor_efficient` branch of `T5ForConditionalGeneration.forward`) -- generative
    document retrieval: a T5 encoder maps a query to a fixed-size prefix embedding, and a T5
    decoder autoregressively emits docid tokens (a semantic docid built from hierarchical
    k-means over document embeddings, decoded via a prefix trie in the original repo; here we
    keep the trie out-of-scope and reimplement the *decoder's own* distinctive mechanism). The
    key novelty is the **prefix-aware weight-adaptive (PAWA) decoder**: instead of a single
    static `lm_head` shared across every decode step, a small adaptor Transformer reads the
    decoder's already-generated prefix and emits a *per-position* additive correction to the
    lm_head projection matrix (`adaptor_weight`, reshaped to `(batch, seq, d_model, d_model)`
    and composed with the shared `lm_head.weight`), so the vocabulary projection at each
    position is conditioned on exactly which docid prefix has been generated so far -- letting
    the same shared token vocabulary mean something different depending on tree position.
  - PAML (Personalizing Dialogue Agents via Meta-Learning): arxiv:1905.10033 (Madotto et al.,
    ACL 2019); official repo github.com/HLTCHKUST/PAML (`model/transformer.py::Transformer`,
    `Encoder`/`Decoder`/`Generator`) -- MAML is a *training procedure* (per-persona few-shot
    inner-loop adaptation, outer-loop meta-update) applied to a fixed persona-conditioned
    seq2seq Transformer architecture; we reimplement that base architecture, which is the
    actual `nn.Module`. Persona sentences and dialogue history are concatenated and encoded by
    a standard multi-head-attention Transformer encoder; the decoder cross-attends the encoder
    states and additionally computes a **pointer-generator copy mechanism**
    (`Generator.forward`): a learned gate (from decoder state + context + input embedding)
    interpolates between the softmax vocabulary distribution and an attention-derived copy
    distribution over the source (persona+history) tokens, letting the model copy
    persona-specific tokens verbatim into the response.
  - Pangu-Bot (Chinese Dialogue LM): arxiv:2203.17090 (Mi et al., 2022); official repo
    github.com/huawei-noah/Pretrained-Language-Model (`PanGu-Bot/`, fine-tunes the `PanGu-alpha`
    backbone at `PanGu-alpha/pangu_alpha.py::PANGUALPHA_Model`, `QueryLayer`,
    `QueryLayerAttention`) -- Pangu-Bot is PanGu-alpha (a GPT-style decoder-only Transformer)
    fine-tuned on 51.5M Chinese dialogue sessions; its distinctive architectural signature
    (carried over unchanged from the backbone) is the **top query layer**: after N standard
    causal self-attention blocks, one final block replaces the usual "hidden-state-attends-
    hidden-state" self-attention with a learned per-position query embedding
    (`top_query_embedding`, indexed by absolute position, independent of token identity) that
    attends as Q against the stacked blocks' K/V, so the final-layer output for a position is
    computed from "what token comes next at this position" rather than "what token is here" --
    designed for more direct next-token generation supervision.
  - PAQ (Probably Asked Questions) / RePAQ: arxiv:2102.07033 (Lewis et al., 2021); official
    repo github.com/facebookresearch/PAQ (`paq/retrievers/embed.py::embed`,
    `paq/rerankers/rerank.py::predict`) -- the trainable component behind the 65M-QA-pair
    cache is **RePAQ**, a dual-encoder + cross-encoder retrieval pipeline: a query encoder
    (BERT-style, mean/CLS-pooled) embeds the input question into the same space as pre-embedded
    cached QA-pairs for fast nearest-neighbour lookup (`embed.py`), and a second cross-encoder
    (`AutoModelForMultipleChoice`-style: concatenate query with each candidate QA pair,
    multiple-choice classification head) reranks the top retrieved candidates jointly with
    full cross-attention. We reimplement both stages as one compact module: a shared BERT
    query/passage encoder for retrieval scoring plus a cross-encoder scoring head for
    reranking the retrieved top-k.

Skipped: cand_00093 "Neural Belief Tracker (NBT)" (arxiv:1606.03777,
github.com/nmrksic/neural-belief-tracker) -- already in the catalog, built earlier under the
identical canonical name "DSTC (Neural Belief Tracker / NBT)" in
`menagerie/classics/gen_w1a4.py` (`build_neural_belief_tracker` /
`example_input_neural_belief_tracker`).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import BertConfig, BertModel, GPT2Config, GPT2Model, T5Config
from transformers.models.t5.modeling_t5 import T5Stack

# ---------------------------------------------------------------------------
# MTTOD: shared T5 encoder -> two independently-weighted T5 decoders (belief
# state + response), plus a token-level span-prediction auxiliary head reading
# raw encoder hidden states.
# ---------------------------------------------------------------------------


class MTTOD(nn.Module):
    """T5 encoder with dual decoders (belief + response) and a span-prediction head."""

    def __init__(self, config: T5Config, num_span_labels: int) -> None:
        """Build shared embeddings/encoder, two decoder stacks, lm heads, and a span head."""

        super().__init__()
        self.config = config
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = config
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = config
        self.decoder = T5Stack(decoder_config, self.shared)
        self.resp_decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.resp_lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.span_head = nn.Linear(config.d_model, num_span_labels)
        self.model_dim = config.d_model

    def forward(
        self,
        input_ids: Tensor,
        belief_decoder_input_ids: Tensor,
        resp_decoder_input_ids: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return (belief_logits, response_logits, span_logits) for one forward pass."""

        encoder_hidden = self.encoder(input_ids=input_ids).last_hidden_state
        scaled_hidden = encoder_hidden * (self.model_dim**-0.5)
        span_logits = self.span_head(scaled_hidden)

        belief_out = self.decoder(
            input_ids=belief_decoder_input_ids, encoder_hidden_states=encoder_hidden
        ).last_hidden_state
        belief_logits = self.lm_head(belief_out * (self.model_dim**-0.5))

        resp_out = self.resp_decoder(
            input_ids=resp_decoder_input_ids, encoder_hidden_states=encoder_hidden
        ).last_hidden_state
        resp_logits = self.resp_lm_head(resp_out * (self.model_dim**-0.5))

        return belief_logits, resp_logits, span_logits


def build_mttod() -> nn.Module:
    """Build the compact MTTOD (dual-decoder T5 + span head) model.

    Returns
    -------
    nn.Module
        ``MTTOD`` in eval mode.
    """

    config = T5Config(
        vocab_size=128,
        d_model=32,
        d_ff=64,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=2,
        d_kv=16,
        is_gated_act=False,
        dense_act_fn="relu",
        use_cache=False,
        is_encoder_decoder=True,
    )
    return MTTOD(config, num_span_labels=6).eval()


def example_input_mttod() -> tuple[Tensor, Tensor, Tensor]:
    """Example dialogue-context, belief-decoder, and response-decoder token ids for MTTOD.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Encoder input ids ``(1, 12)``, belief decoder ids ``(1, 6)``, response
        decoder ids ``(1, 8)``.
    """

    input_ids = torch.randint(0, 128, (1, 12))
    belief_ids = torch.randint(0, 128, (1, 6))
    resp_ids = torch.randint(0, 128, (1, 8))
    return input_ids, belief_ids, resp_ids


# ---------------------------------------------------------------------------
# NCI (Neural Corpus Indexer): T5 encoder/decoder generative retrieval with a
# prefix-aware weight-adaptive (PAWA) decoder: an adaptor transformer reads
# the already-generated docid prefix and emits a per-position additive
# correction to the shared lm_head projection matrix.
# ---------------------------------------------------------------------------


class PAWADecoderHead(nn.Module):
    """Prefix-aware weight-adaptive lm-head: per-position additive correction to lm_head."""

    def __init__(self, d_model: int, vocab_size: int) -> None:
        """Build the shared lm_head plus a small adaptor transformer + linear projector."""

        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.adaptor_embed = nn.Embedding(vocab_size, d_model)
        adaptor_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=2, dim_feedforward=2 * d_model, batch_first=True
        )
        self.adaptor = nn.TransformerEncoder(adaptor_layer, num_layers=1)
        # Adaptor output -> a flattened (d_model, d_model) additive correction per position.
        self.adaptor_linear = nn.Linear(d_model, d_model * d_model)

    def forward(self, decoder_hidden: Tensor, decoder_input_ids: Tensor) -> Tensor:
        """Return position-adaptive vocabulary logits ``(batch, seq, vocab_size)``."""

        batch, seq_len, _ = decoder_hidden.shape
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(
            decoder_hidden.device
        )
        prefix_embed = self.adaptor_embed(decoder_input_ids)
        adaptor_out = self.adaptor(prefix_embed, mask=causal_mask, is_causal=True)

        adaptor_weight = self.adaptor_linear(adaptor_out).view(
            batch, seq_len, self.d_model, self.d_model
        )
        base_weight = self.lm_head.weight.t()  # (d_model, vocab_size)
        # Per-position additive correction composed with the shared static lm_head weight.
        position_weight = torch.matmul(adaptor_weight, base_weight)  # (b, s, d_model, vocab)
        logits = torch.matmul(decoder_hidden.unsqueeze(-2), position_weight).squeeze(-2)
        logits = logits + self.lm_head(decoder_hidden)
        return logits


class NeuralCorpusIndexer(nn.Module):
    """Generative document retrieval: T5 encoder/decoder with a PAWA decode head."""

    def __init__(self, config: T5Config) -> None:
        """Build the shared T5 encoder/decoder and the prefix-aware weight-adaptive head."""

        super().__init__()
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = T5Stack(config, self.shared)
        self.decoder = T5Stack(config, self.shared)
        self.pawa_head = PAWADecoderHead(config.d_model, config.vocab_size)
        self.model_dim = config.d_model

    def forward(self, input_ids: Tensor, decoder_input_ids: Tensor) -> Tensor:
        """Return PAWA-adapted docid-token logits ``(batch, seq, vocab_size)``."""

        encoder_hidden = self.encoder(input_ids=input_ids).last_hidden_state
        decoder_hidden = self.decoder(
            input_ids=decoder_input_ids, encoder_hidden_states=encoder_hidden
        ).last_hidden_state
        decoder_hidden = decoder_hidden * (self.model_dim**-0.5)
        return self.pawa_head(decoder_hidden, decoder_input_ids)


def build_nci() -> nn.Module:
    """Build the compact NCI (prefix-aware weight-adaptive generative retriever) model.

    Returns
    -------
    nn.Module
        ``NeuralCorpusIndexer`` in eval mode.
    """

    config = T5Config(
        vocab_size=64,
        d_model=32,
        d_ff=64,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=2,
        d_kv=16,
        is_gated_act=False,
        dense_act_fn="relu",
        use_cache=False,
        is_encoder_decoder=True,
    )
    return NeuralCorpusIndexer(config).eval()


def example_input_nci() -> tuple[Tensor, Tensor]:
    """Example query tokens and partially-decoded semantic docid tokens for NCI.

    Returns
    -------
    tuple[Tensor, Tensor]
        Query input ids ``(1, 10)`` and decoder docid-prefix ids ``(1, 5)``.
    """

    query_ids = torch.randint(0, 64, (1, 10))
    docid_prefix_ids = torch.randint(0, 64, (1, 5))
    return query_ids, docid_prefix_ids


# ---------------------------------------------------------------------------
# PAML: persona-conditioned Transformer encoder/decoder with a pointer-
# generator copy mechanism (learned gate blends vocabulary softmax with an
# attention-derived copy distribution over the persona+history source
# tokens). MAML itself is an outer training loop, not part of the module.
# ---------------------------------------------------------------------------


class PamlEncoder(nn.Module):
    """Standard multi-head-attention Transformer encoder over persona + dialogue history."""

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int) -> None:
        """Build token/position embeddings and a stack of self-attention encoder layers."""

        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(64, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=2 * d_model, batch_first=True
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Return contextualized source representations ``(batch, seq, d_model)``."""

        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        hidden = self.embed(input_ids) + self.pos_embed(positions).unsqueeze(0)
        return self.layers(hidden)


class PointerGeneratorDecoder(nn.Module):
    """Transformer decoder with a copy-gate that blends vocab softmax with source-copy attn."""

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int) -> None:
        """Build decoder embeddings, cross-attn decoder layers, copy-attn, and gate/vocab heads."""

        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(64, d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=2 * d_model, batch_first=True
        )
        self.layers = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.copy_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.vocab_proj = nn.Linear(d_model, vocab_size)
        self.gate_proj = nn.Linear(3 * d_model, 1)
        self.vocab_size = vocab_size

    def forward(
        self, decoder_input_ids: Tensor, encoder_hidden: Tensor, source_ids: Tensor
    ) -> Tensor:
        """Return blended vocab-softmax + copy-distribution probabilities over the vocabulary."""

        positions = torch.arange(decoder_input_ids.size(1), device=decoder_input_ids.device)
        dec_embed = self.embed(decoder_input_ids) + self.pos_embed(positions).unsqueeze(0)

        seq_len = decoder_input_ids.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(dec_embed.device)
        dec_hidden = self.layers(
            dec_embed, encoder_hidden, tgt_mask=causal_mask, tgt_is_causal=True
        )

        context, copy_weights = self.copy_attn(dec_hidden, encoder_hidden, encoder_hidden)

        vocab_logits = self.vocab_proj(dec_hidden)
        vocab_dist = F.softmax(vocab_logits, dim=-1)

        gate_input = torch.cat([dec_hidden, context, dec_embed], dim=-1)
        p_gen = torch.sigmoid(self.gate_proj(gate_input))

        batch, tgt_len, src_len = copy_weights.shape
        copy_dist = torch.zeros(batch, tgt_len, self.vocab_size, device=dec_hidden.device)
        src_index = source_ids.unsqueeze(1).expand(-1, tgt_len, -1)
        copy_dist.scatter_add_(2, src_index, copy_weights)

        return p_gen * vocab_dist + (1.0 - p_gen) * copy_dist


class PamlTransformer(nn.Module):
    """Persona-conditioned pointer-generator Transformer (base architecture behind PAML)."""

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int) -> None:
        """Build the shared-vocab encoder and pointer-generator decoder."""

        super().__init__()
        self.encoder = PamlEncoder(vocab_size, d_model, n_heads, n_layers)
        self.decoder = PointerGeneratorDecoder(vocab_size, d_model, n_heads, n_layers)

    def forward(self, source_ids: Tensor, decoder_input_ids: Tensor) -> Tensor:
        """Return final blended output-token probabilities ``(batch, tgt_len, vocab_size)``."""

        encoder_hidden = self.encoder(source_ids)
        return self.decoder(decoder_input_ids, encoder_hidden, source_ids)


def build_paml() -> nn.Module:
    """Build the compact PAML (persona-conditioned pointer-generator Transformer) model.

    Returns
    -------
    nn.Module
        ``PamlTransformer`` in eval mode.
    """

    return PamlTransformer(vocab_size=100, d_model=32, n_heads=2, n_layers=2).eval()


def example_input_paml() -> tuple[Tensor, Tensor]:
    """Example persona+history source tokens and decoder-side response tokens for PAML.

    Returns
    -------
    tuple[Tensor, Tensor]
        Source (persona + history) ids ``(1, 14)`` and decoder response ids ``(1, 7)``.
    """

    source_ids = torch.randint(0, 100, (1, 14))
    decoder_ids = torch.randint(0, 100, (1, 7))
    return source_ids, decoder_ids


# ---------------------------------------------------------------------------
# Pangu-Bot (fine-tuned PanGu-alpha backbone): GPT-style decoder-only
# Transformer with a "top query layer" -- the final block replaces the usual
# self-attention query (derived from hidden states) with a learned
# position-indexed query embedding attending over the stacked blocks' K/V.
# ---------------------------------------------------------------------------


class TopQueryAttention(nn.Module):
    """Self-attention whose query comes from a position embedding, not the hidden state."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        """Build separate query/key-value projections and the output projection."""

        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query_embed: Tensor, hidden_states: Tensor) -> Tensor:
        """Return top-query attention output ``(batch, seq, d_model)``."""

        batch, seq_len, d_model = hidden_states.shape
        q = self.q_proj(query_embed)
        k, v = self.kv_proj(hidden_states).chunk(2, dim=-1)

        def split_heads(x: Tensor) -> Tensor:
            return x.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=hidden_states.device), diagonal=1
        )
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores + causal_mask
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v).transpose(1, 2).reshape(batch, seq_len, d_model)
        return self.out_proj(context)


class TopQueryLayer(nn.Module):
    """Final PanGu-alpha block: top-query attention + feed-forward, both pre-LayerNorm'd."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        """Build the top-query attention block and its feed-forward sublayer."""

        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = TopQueryAttention(d_model, n_heads)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )

    def forward(self, query_embed: Tensor, hidden_states: Tensor) -> Tensor:
        """Return the updated hidden states after top-query attention + feed-forward."""

        attn_out = self.attn(self.ln_1(query_embed), self.ln_1(hidden_states))
        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.mlp(self.ln_2(hidden_states))
        return hidden_states


class PanguBot(nn.Module):
    """GPT-style decoder LM (PanGu-alpha backbone) with a final top-query layer."""

    def __init__(self, config: GPT2Config, d_model: int, n_heads: int) -> None:
        """Build the GPT2 causal-LM backbone, the top-query embedding, and the top-query layer."""

        super().__init__()
        self.backbone = GPT2Model(config)
        self.top_query_embedding = nn.Embedding(config.n_positions, d_model)
        self.top_query_layer = TopQueryLayer(d_model, n_heads)
        self.lm_head = nn.Linear(d_model, config.vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Return next-token logits ``(batch, seq, vocab_size)`` from the top-query head."""

        hidden_states = self.backbone(input_ids=input_ids).last_hidden_state
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        query_embed = (
            self.top_query_embedding(positions).unsqueeze(0).expand(input_ids.size(0), -1, -1)
        )
        top_hidden = self.top_query_layer(query_embed, hidden_states)
        return self.lm_head(top_hidden)


def build_pangu_bot() -> nn.Module:
    """Build the compact Pangu-Bot (PanGu-alpha backbone + top-query layer) model.

    Returns
    -------
    nn.Module
        ``PanguBot`` in eval mode.
    """

    d_model = 32
    n_heads = 2
    config = GPT2Config(
        vocab_size=128,
        n_positions=32,
        n_embd=d_model,
        n_layer=2,
        n_head=n_heads,
        n_inner=64,
    )
    return PanguBot(config, d_model=d_model, n_heads=n_heads).eval()


def example_input_pangu_bot() -> Tensor:
    """Example Chinese-dialogue-style token ids for Pangu-Bot.

    Returns
    -------
    Tensor
        Input token ids ``(1, 12)``.
    """

    return torch.randint(0, 128, (1, 12))


# ---------------------------------------------------------------------------
# PAQ / RePAQ: dual-encoder dense retriever (query vs. cached QA-pair
# embeddings) plus a cross-encoder reranker over the retrieved top-k
# candidates (multiple-choice-style joint scoring).
# ---------------------------------------------------------------------------


class RePAQ(nn.Module):
    """Dense QA-pair retriever (dual encoder) + cross-encoder reranker over top-k candidates."""

    def __init__(self, bert_config: BertConfig) -> None:
        """Build the shared BERT retriever encoder and a separate cross-encoder scorer."""

        super().__init__()
        self.retriever_encoder = BertModel(bert_config)
        self.reranker_encoder = BertModel(bert_config)
        self.rerank_score = nn.Linear(bert_config.hidden_size, 1)

    def _mean_pool(self, hidden: Tensor, mask: Tensor) -> Tensor:
        """Mean-pool token representations over non-padding positions."""

        mask = mask.unsqueeze(-1).float()
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

    def embed_query(self, query_ids: Tensor, query_mask: Tensor) -> Tensor:
        """Embed a query into the shared retrieval space."""

        hidden = self.retriever_encoder(
            input_ids=query_ids, attention_mask=query_mask
        ).last_hidden_state
        return self._mean_pool(hidden, query_mask)

    def embed_candidates(self, candidate_ids: Tensor, candidate_mask: Tensor) -> Tensor:
        """Embed a batch of candidate QA pairs into the shared retrieval space."""

        batch, n_cand, clen = candidate_ids.shape
        flat_ids = candidate_ids.view(batch * n_cand, clen)
        flat_mask = candidate_mask.view(batch * n_cand, clen)
        hidden = self.retriever_encoder(
            input_ids=flat_ids, attention_mask=flat_mask
        ).last_hidden_state
        pooled = self._mean_pool(hidden, flat_mask)
        return pooled.view(batch, n_cand, -1)

    def forward(
        self,
        query_ids: Tensor,
        query_mask: Tensor,
        candidate_ids: Tensor,
        candidate_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return (retrieval_scores, rerank_scores) over the candidate QA pairs.

        ``retrieval_scores`` is a dot-product dense-retrieval score per candidate
        (dual-encoder stage). ``rerank_scores`` is a joint cross-encoder score per
        candidate, computed by concatenating the query with each candidate and
        classifying the pooled representation (multiple-choice-style reranking).
        """

        query_emb = self.embed_query(query_ids, query_mask)
        candidate_emb = self.embed_candidates(candidate_ids, candidate_mask)
        retrieval_scores = torch.einsum("bd,bnd->bn", query_emb, candidate_emb)

        batch, n_cand, clen = candidate_ids.shape
        qlen = query_ids.size(1)
        query_rep = query_ids.unsqueeze(1).expand(-1, n_cand, -1)
        query_rep_mask = query_mask.unsqueeze(1).expand(-1, n_cand, -1)
        joint_ids = torch.cat([query_rep, candidate_ids], dim=-1).view(batch * n_cand, qlen + clen)
        joint_mask = torch.cat([query_rep_mask, candidate_mask], dim=-1).view(
            batch * n_cand, qlen + clen
        )
        joint_hidden = self.reranker_encoder(
            input_ids=joint_ids, attention_mask=joint_mask
        ).last_hidden_state
        pooled = joint_hidden[:, 0, :]
        rerank_scores = self.rerank_score(pooled).view(batch, n_cand)

        return retrieval_scores, rerank_scores


def build_paq() -> nn.Module:
    """Build the compact PAQ / RePAQ (dual-encoder retriever + cross-encoder reranker) model.

    Returns
    -------
    nn.Module
        ``RePAQ`` in eval mode.
    """

    bert_config = BertConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=32,
    )
    return RePAQ(bert_config).eval()


def example_input_paq() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Example query tokens and a small batch of candidate QA-pair tokens for PAQ/RePAQ.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        Query ids ``(1, 8)``, query mask ``(1, 8)``, candidate ids ``(1, 4, 10)``,
        candidate mask ``(1, 4, 10)``.
    """

    query_ids = torch.randint(0, 128, (1, 8))
    query_mask = torch.ones(1, 8, dtype=torch.long)
    candidate_ids = torch.randint(0, 128, (1, 4, 10))
    candidate_mask = torch.ones(1, 4, 10, dtype=torch.long)
    return query_ids, query_mask, candidate_ids, candidate_mask


MENAGERIE_ENTRIES = [
    (
        "MTTOD (Multi-Task TOD with Span Prediction)",
        "build_mttod",
        "example_input_mttod",
        "2021",
        "NLP",
    ),
    ("NCI (Neural Corpus Indexer)", "build_nci", "example_input_nci", "2022", "NLP"),
    (
        "PAML (Personalized Adaptive Meta-Learning for Dialogue)",
        "build_paml",
        "example_input_paml",
        "2019",
        "NLP",
    ),
    (
        "Pangu-Bot (Chinese Dialogue LM)",
        "build_pangu_bot",
        "example_input_pangu_bot",
        "2022",
        "NLP",
    ),
    (
        "PAQ (Probably Asked Questions)",
        "build_paq",
        "example_input_paq",
        "2021",
        "NLP",
    ),
]
