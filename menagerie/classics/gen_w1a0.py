"""Wave w1a0: five NLP dialogue / retrieval architecture classics.

Sources checked (repo trees + primary source files fetched via GitHub API,
paper metadata from arXiv ids in the build queue):

- AGIF -- LooperXX/AGIF (``models/module.py``), Qin et al., EMNLP 2020 Findings,
  arXiv:2004.10087. Adaptive Graph-Interactive Framework for joint multi-intent
  detection and slot filling: a BiLSTM+self-attention sentence encoder, a
  multi-label intent classifier, and a per-token slot LSTM decoder that at every
  decoding step runs a small Graph Attention Network (GAT) over a graph built
  from {current token, predicted-intent embeddings} -- the intent-slot
  interaction graph is the distinctive mechanism.
- Atlas -- facebookresearch/atlas (``src/fid.py``, ``src/retrievers.py``),
  Izacard et al., JMLR 2023, arXiv:2208.03299. A Contriever-style mean-pooled
  BERT dense retriever selects top-k passages; a Fusion-in-Decoder (FiD) T5
  encodes each retrieved passage independently (each concatenated with the
  query) and then concatenates all per-passage encoder states into one long
  sequence that the decoder cross-attends over jointly -- the "fuse in
  decoder, not in encoder" trick is the distinctive mechanism.
- AuGPT -- ufal/augpt (``model.py``), Kulhanek et al., ACL NLP4ConvAI 2021,
  arXiv:2102.05126. A GPT-2 task-oriented-dialogue model with a second
  auxiliary head (``consistency_head``) that gathers the hidden state at a
  designated token position via ``.gather`` and predicts a binary
  belief-state/response consistency label alongside the standard LM head --
  the auxiliary consistency-classification head trained jointly with the LM
  loss is the distinctive mechanism (data-augmentation / back-translation is a
  training-time technique, not part of the network graph, and is not modeled).
- BERT Ranker -- nyu-dl/dl4marco-bert (``modeling.py``), Nogueira & Cho 2019,
  arXiv:1901.04085. The canonical BERT cross-encoder reranker: encode
  ``[CLS] query [SEP] passage [SEP]`` jointly through BERT and classify
  relevance from the pooled ``[CLS]`` vector with a single linear layer.
- BERT-SLU (ToD-BERT) -- jasonwu0731/ToD-BERT
  (``models/dual_encoder_ranking.py``), Wu et al., EMNLP 2020,
  arXiv:2004.06871. A dual-encoder response-selection model: one shared BERT
  encoder embeds the dialogue context and the candidate response separately
  (pooled ``[CLS]``), and the model scores every context against every
  response in the batch via a dot-product similarity matrix trained with
  in-batch-negative contrastive cross-entropy against the diagonal -- the
  in-batch contrastive dual-encoder objective is the distinctive mechanism
  ToD-BERT is pretrained with (the row's alias for cand_00011).

All models are compact, randomly initialized, CPU, forward-only reimplementations
built from scratch in base-env torch/torch.nn -- no cloning, no pip installs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoConfig, AutoModel, BertConfig, BertModel, GPT2Config, GPT2Model

# ---------------------------------------------------------------------------
# AGIF: adaptive graph-interactive framework for multi-intent SLU
# ---------------------------------------------------------------------------


class AGIFGraphAttentionLayer(nn.Module):
    """Single dense Graph Attention Network (GAT) layer over a token+intent graph."""

    def __init__(self, in_features: int, out_features: int, alpha: float = 0.2) -> None:
        """Initialize the shared linear projection and attention vector."""

        super().__init__()
        self.out_features = out_features
        self.proj = nn.Linear(in_features, out_features, bias=False)
        self.attn_vec = nn.Parameter(torch.zeros(2 * out_features, 1))
        nn.init.xavier_uniform_(self.attn_vec, gain=1.414)
        self.leaky_relu = nn.LeakyReLU(alpha)

    def forward(self, nodes: Tensor, adj: Tensor) -> Tensor:
        """Return ELU-activated GAT node updates given node feats and a dense adjacency."""

        h = self.proj(nodes)
        n_nodes = h.size(1)
        h_i = h.unsqueeze(2).expand(-1, -1, n_nodes, -1)
        h_j = h.unsqueeze(1).expand(-1, n_nodes, -1, -1)
        pair = torch.cat([h_i, h_j], dim=-1)
        e = self.leaky_relu(torch.matmul(pair, self.attn_vec).squeeze(-1))
        e = e.masked_fill(adj == 0, float("-inf"))
        attn = F.softmax(e, dim=-1)
        return F.elu(torch.matmul(attn, h))


class AGIF(nn.Module):
    """Compact AGIF: BiLSTM+self-attn encoder, multi-intent head, per-step adaptive graph decoder.

    AGIF's namesake "adaptive graph-interactive framework" mechanism is a
    PER-TOKEN-STEP adaptive interaction graph: slot filling is decoded
    token-by-token (autoregressively), and at EACH step t the interaction
    graph between token t's current slot-representation and the (fixed)
    intent-embedding set is REBUILT from token t's live hidden state --
    edge weights differ per step because they are a function of that step's
    state, not a single static graph shared/reused across the whole
    sequence. Each step's per-step GAT-style aggregation output also feeds
    into the next token's slot LSTM cell state, matching AGIF's iterative
    label-aware decoding.
    """

    def __init__(
        self,
        vocab_size: int = 96,
        embed_dim: int = 32,
        hidden_dim: int = 32,
        n_intent: int = 6,
        n_slot: int = 12,
        intent_embed_dim: int = 16,
    ) -> None:
        """Build the embedding, encoder, intent head, intent embeddings, and per-step GAT decoder."""

        super().__init__()
        self.n_intent = n_intent
        self.intent_embed_dim = intent_embed_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.encoder_lstm = nn.LSTM(
            embed_dim, hidden_dim // 2, batch_first=True, bidirectional=True
        )
        self.self_attn_q = nn.Linear(embed_dim, hidden_dim // 2)
        self.self_attn_k = nn.Linear(embed_dim, hidden_dim // 2)
        self.self_attn_v = nn.Linear(embed_dim, hidden_dim // 2)
        enc_dim = hidden_dim + hidden_dim // 2
        self.intent_head = nn.Linear(enc_dim, n_intent)
        self.intent_embedding = nn.Parameter(torch.randn(n_intent, intent_embed_dim))
        self.slot_pre = nn.Linear(enc_dim, intent_embed_dim)
        # Per-step slot decoder LSTM cell: its hidden state carries the
        # previous step's graph-aggregated, label-aware slot representation
        # into the next step, per AGIF's iterative decoding.
        self.slot_decoder_cell = nn.LSTMCell(intent_embed_dim, intent_embed_dim)
        self.gat = AGIFGraphAttentionLayer(intent_embed_dim, intent_embed_dim)
        self.slot_head = nn.Linear(intent_embed_dim, n_slot)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        """Return (intent_logits, per-token slot_logits) via the per-step adaptive graph decoder."""

        embedded = self.embed(tokens)
        lstm_out, _ = self.encoder_lstm(embedded)
        q, k, v = self.self_attn_q(embedded), self.self_attn_k(embedded), self.self_attn_v(embedded)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
        attn_out = torch.matmul(F.softmax(scores, dim=-1), v)
        hiddens = torch.cat([lstm_out, attn_out], dim=-1)

        sent_vec = hiddens.mean(dim=1)
        intent_logits = self.intent_head(sent_vec)
        intent_weights = torch.sigmoid(intent_logits)

        batch, seq_len, _ = hiddens.shape
        # Fixed intent-embedding set (n_intent nodes); the token-intent graph
        # below is rebuilt against this set from scratch at every step.
        intent_nodes = intent_weights.unsqueeze(-1) * self.intent_embedding.unsqueeze(0)
        slot_pre = self.slot_pre(hiddens)  # (batch, seq_len, intent_embed_dim)

        cell_h = hiddens.new_zeros(batch, self.intent_embed_dim)
        cell_c = hiddens.new_zeros(batch, self.intent_embed_dim)
        step_logits = []
        for t in range(seq_len):
            # The slot decoder's per-step input is the token's own encoding
            # PLUS the previous step's label-aware graph output (cell_h) --
            # an iterative, autoregressive slot-decoding step.
            cell_h, cell_c = self.slot_decoder_cell(slot_pre[:, t], (cell_h, cell_c))

            # Rebuild the token-intent interaction graph FROM SCRATCH at this
            # step: a fresh 1-token+n_intent-node graph whose node features
            # and (via the GAT's own attention) edge weights are functions of
            # THIS step's live cell_h -- not a graph shared across steps.
            step_node = cell_h.unsqueeze(1)  # (batch, 1, intent_embed_dim)
            graph_nodes_t = torch.cat([step_node, intent_nodes], dim=1)
            n_total = 1 + self.n_intent
            adj_t = (
                torch.eye(n_total, device=tokens.device).unsqueeze(0).expand(batch, -1, -1).clone()
            )
            adj_t[:, 0, 1:] = 1.0
            adj_t[:, 1:, 0] = 1.0

            gat_out_t = self.gat(graph_nodes_t, adj_t)
            token_repr_t = gat_out_t[:, 0]
            # Feed this step's graph-aggregated representation back as the
            # next step's decoder hidden state -- AGIF's iterative
            # label-aware mechanism.
            cell_h = token_repr_t
            step_logits.append(self.slot_head(token_repr_t))

        slot_logits = torch.stack(step_logits, dim=1)
        return intent_logits, slot_logits


def build_agif() -> nn.Module:
    """Build the compact AGIF joint multi-intent + slot-filling model."""

    return AGIF().eval()


def example_input_agif() -> Tensor:
    """Return a batch of token-id sequences, fixed length for graph-shape traceability."""

    return torch.randint(0, 96, (2, 10), dtype=torch.long)


# ---------------------------------------------------------------------------
# Atlas: Contriever retriever + Fusion-in-Decoder T5 reader, jointly trained
# ---------------------------------------------------------------------------


class ContrieverRetriever(nn.Module):
    """Mean-pooled BERT dense retriever (Contriever) scoring passages against a query."""

    def __init__(self, bert: BertModel) -> None:
        """Store the shared BERT encoder used for both queries and passages."""

        super().__init__()
        self.encoder = bert

    def embed(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Mean-pool BERT's last hidden state over the (non-padding) sequence."""

        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        return (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

    def forward(
        self, query_ids: Tensor, query_mask: Tensor, passage_ids: Tensor, passage_mask: Tensor
    ) -> Tensor:
        """Return dense retrieval scores ``(batch, n_passages)`` for one query each."""

        batch, n_passages, plen = passage_ids.shape
        query_emb = self.embed(query_ids, query_mask)
        flat_ids = passage_ids.view(batch * n_passages, plen)
        flat_mask = passage_mask.view(batch * n_passages, plen)
        passage_emb = self.embed(flat_ids, flat_mask).view(batch, n_passages, -1)
        return torch.einsum("bd,bnd->bn", query_emb, passage_emb)


class AtlasFiD(nn.Module):
    """Fusion-in-Decoder: encode passages independently, decode jointly over all of them."""

    def __init__(
        self, retriever: ContrieverRetriever, fid_encoder: nn.Module, fid_decoder: nn.Module
    ) -> None:
        """Store the retriever plus a shared T5 encoder/decoder pair used for FiD."""

        super().__init__()
        self.retriever = retriever
        self.fid_encoder = fid_encoder
        self.fid_decoder = fid_decoder

    def forward(
        self,
        query_ids: Tensor,
        query_mask: Tensor,
        passage_ids: Tensor,
        passage_mask: Tensor,
        decoder_input_ids: Tensor,
    ) -> Tensor:
        """Retrieve-then-fuse-in-decoder: return decoder logits fused over all passages."""

        retrieval_scores = self.retriever(query_ids, query_mask, passage_ids, passage_mask)

        batch, n_passages, plen = passage_ids.shape
        flat_ids = passage_ids.view(batch * n_passages, plen)
        flat_mask = passage_mask.view(batch * n_passages, plen)
        enc_out = self.fid_encoder(input_ids=flat_ids, attention_mask=flat_mask).last_hidden_state
        # Fusion-in-Decoder: concatenate every passage's encoder states into one long
        # sequence the decoder cross-attends over jointly (not fused in the encoder).
        fused_hidden = enc_out.view(batch, n_passages * plen, -1)
        fused_mask = flat_mask.view(batch, n_passages * plen)

        # Retrieval scores modulate the fused encoder states (Atlas-style read-with-scores).
        gate = torch.sigmoid(retrieval_scores).view(batch, n_passages, 1).expand(-1, -1, plen)
        fused_hidden = fused_hidden * gate.reshape(batch, n_passages * plen, 1)

        dec_out = self.fid_decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=fused_hidden,
            encoder_attention_mask=fused_mask,
        ).last_hidden_state
        return dec_out


def build_atlas() -> nn.Module:
    """Build the compact Atlas (Contriever retriever + FiD T5 reader) model."""

    bert_cfg = BertConfig(
        vocab_size=256,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=32,
    )
    retriever = ContrieverRetriever(BertModel(bert_cfg))

    t5_cfg = AutoConfig.for_model(
        "t5",
        vocab_size=256,
        d_model=32,
        d_ff=64,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=2,
        d_kv=16,
    )
    fid_encoder = AutoModel.from_config(t5_cfg).encoder
    fid_decoder = AutoModel.from_config(t5_cfg).decoder
    return AtlasFiD(retriever, fid_encoder, fid_decoder).eval()


def example_input_atlas() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return (query_ids, query_mask, passage_ids, passage_mask, decoder_input_ids)."""

    query_ids = torch.randint(0, 256, (1, 8), dtype=torch.long)
    query_mask = torch.ones(1, 8, dtype=torch.long)
    passage_ids = torch.randint(0, 256, (1, 3, 12), dtype=torch.long)
    passage_mask = torch.ones(1, 3, 12, dtype=torch.long)
    decoder_input_ids = torch.randint(0, 256, (1, 6), dtype=torch.long)
    return query_ids, query_mask, passage_ids, passage_mask, decoder_input_ids


# ---------------------------------------------------------------------------
# AuGPT: GPT-2 task-oriented dialogue model with an auxiliary consistency head
# ---------------------------------------------------------------------------


class AuGPT(nn.Module):
    """GPT-2 LM head plus a gathered-position auxiliary consistency classifier."""

    def __init__(self, gpt2: GPT2Model, vocab_size: int, hidden: int) -> None:
        """Store the GPT-2 backbone and build the LM + consistency heads."""

        super().__init__()
        self.transformer = gpt2
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)
        self.consistency_head = nn.Linear(hidden, 1)

    def forward(self, input_ids: Tensor, consistency_token_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Return (lm_logits, consistency_logits) gathered at a designated token position."""

        hidden_states = self.transformer(input_ids=input_ids).last_hidden_state
        lm_logits = self.lm_head(hidden_states)

        # Gather the hidden state at a per-example designated token position (e.g. <|eob|>)
        # and predict a binary belief/response consistency label from it -- the auxiliary
        # task head that distinguishes AuGPT from a plain GPT-2 TOD baseline.
        gather_idx = consistency_token_ids.unsqueeze(-1).unsqueeze(-1)
        gather_idx = gather_idx.expand(-1, -1, hidden_states.size(-1))
        aux_features = hidden_states.gather(1, gather_idx).squeeze(1)
        consistency_logits = self.consistency_head(aux_features).squeeze(-1)
        return lm_logits, consistency_logits


def build_augpt() -> nn.Module:
    """Build the compact AuGPT (GPT-2 + auxiliary consistency head) model."""

    vocab_size, hidden = 256, 32
    cfg = GPT2Config(
        vocab_size=vocab_size,
        n_positions=32,
        n_embd=hidden,
        n_layer=2,
        n_head=2,
        n_inner=64,
    )
    return AuGPT(GPT2Model(cfg), vocab_size, hidden).eval()


def example_input_augpt() -> tuple[Tensor, Tensor]:
    """Return (input_ids, consistency_token_ids) -- the latter indexes one position per row."""

    input_ids = torch.randint(0, 256, (2, 10), dtype=torch.long)
    consistency_token_ids = torch.full((2,), 9, dtype=torch.long)
    return input_ids, consistency_token_ids


# ---------------------------------------------------------------------------
# BERT Ranker: BERT [CLS] cross-encoder for query-passage relevance
# ---------------------------------------------------------------------------


class BERTRanker(nn.Module):
    """BERT cross-encoder: jointly encode query+passage, classify from pooled [CLS]."""

    def __init__(self, bert: BertModel, hidden: int) -> None:
        """Store the BERT encoder and a binary relevance classification head."""

        super().__init__()
        self.bert = bert
        self.classifier = nn.Linear(hidden, 2)

    def forward(self, input_ids: Tensor, token_type_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Return relevant/non-relevant logits from the jointly-encoded [CLS] token."""

        out = self.bert(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )
        cls_repr = out.last_hidden_state[:, 0]
        return self.classifier(cls_repr)


def build_bert_ranker() -> nn.Module:
    """Build the compact BERT [CLS] cross-encoder MS-MARCO reranker."""

    hidden = 32
    cfg = BertConfig(
        vocab_size=256,
        hidden_size=hidden,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=32,
        type_vocab_size=2,
    )
    return BERTRanker(BertModel(cfg), hidden).eval()


def example_input_bert_ranker() -> tuple[Tensor, Tensor, Tensor]:
    """Return (input_ids, token_type_ids, attention_mask) for a joint query+passage pair."""

    input_ids = torch.randint(0, 256, (2, 20), dtype=torch.long)
    token_type_ids = torch.cat(
        [torch.zeros(2, 8, dtype=torch.long), torch.ones(2, 12, dtype=torch.long)], dim=1
    )
    attention_mask = torch.ones(2, 20, dtype=torch.long)
    return input_ids, token_type_ids, attention_mask


# ---------------------------------------------------------------------------
# BERT-SLU (ToD-BERT): dual-encoder in-batch-negative contrastive response ranker
# ---------------------------------------------------------------------------


class ToDBERTDualEncoder(nn.Module):
    """Shared BERT dual encoder scoring context-vs-response via in-batch dot products."""

    def __init__(self, bert: BertModel) -> None:
        """Store the single shared BERT encoder used for both contexts and responses."""

        super().__init__()
        self.utterance_encoder = bert

    def _pool(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Return the pooled [CLS] representation for a batch of token sequences."""

        return self.utterance_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0]

    def forward(
        self, context_ids: Tensor, context_mask: Tensor, response_ids: Tensor, response_mask: Tensor
    ) -> Tensor:
        """Return the ``(batch, batch)`` context-vs-response similarity matrix."""

        context_repr = self._pool(context_ids, context_mask)
        response_repr = self._pool(response_ids, response_mask)
        # In-batch-negative contrastive scoring: every context is scored against every
        # response in the batch; training targets the diagonal -- the dual-encoder
        # response-selection pretraining objective that defines ToD-BERT / BERT-SLU.
        return torch.matmul(context_repr, response_repr.transpose(0, 1))


def build_bert_slu() -> nn.Module:
    """Build the compact ToD-BERT dual-encoder in-batch contrastive response ranker."""

    hidden = 32
    cfg = BertConfig(
        vocab_size=256,
        hidden_size=hidden,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=32,
    )
    return ToDBERTDualEncoder(BertModel(cfg)).eval()


def example_input_bert_slu() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return (context_ids, context_mask, response_ids, response_mask) for a small batch."""

    batch = 4
    context_ids = torch.randint(0, 256, (batch, 14), dtype=torch.long)
    context_mask = torch.ones(batch, 14, dtype=torch.long)
    response_ids = torch.randint(0, 256, (batch, 10), dtype=torch.long)
    response_mask = torch.ones(batch, 10, dtype=torch.long)
    return context_ids, context_mask, response_ids, response_mask


MENAGERIE_ENTRIES = [
    ("AGIF", "build_agif", "example_input_agif", "2020", "NLP"),
    ("Atlas", "build_atlas", "example_input_atlas", "2023", "NLP"),
    ("AuGPT", "build_augpt", "example_input_augpt", "2021", "NLP"),
    ("BERT Ranker", "build_bert_ranker", "example_input_bert_ranker", "2019", "NLP"),
    ("BERT-SLU (Aghajanyan et al.)", "build_bert_slu", "example_input_bert_slu", "2020", "NLP"),
]
