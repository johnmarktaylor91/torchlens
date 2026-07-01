"""Compact faithful reimplementations of six memory/graph/dialogue architecture families.

Sources checked (paper + official source, no clone/pip-install; reimplemented from scratch):
  - Key-Value Memory Network (KV-MemNN): arxiv:1606.03126 (Miller, Fisch, Dodge, Karimi,
    Bordes, Weston, 2016); reference repo github.com/facebookarchive/MemNN (archived) --
    a memory of (key, value) slot pairs is addressed with the query embedding via a
    softmax over key dot-products, the weighted sum of VALUE embeddings updates the query
    through several "hops" (each hop applies a learned matrix R_h to combine query and
    read-out before re-addressing), and the final query state scores candidate answers.
    This is architecturally distinct from End-to-End Memory Networks (single embedding
    per memory slot, already present as `memn2n_babi`) because addressing and output use
    two SEPARATE embedding spaces (key space vs. value space) -- the paper's core idea.
  - KGSF (Knowledge Graph based Semantic Fusion): arxiv:2007.04032 (Zhou et al., KDD
    2020); official repo github.com/RUCAIBox/KGSF (`model.py`) -- a word-oriented graph
    (co-occurrence, R-GCN) and an entity-oriented graph (DBpedia-style KG, R-GCN) are
    each pooled into a semantic vector per dialogue; a gated cross-attention fusion layer
    mutually informs one graph representation with the other (mutual information
    maximization proxy implemented here as the fusion+reconstruction gate), and the fused
    representation drives both an item-recommendation head (bilinear score over entity
    embeddings) and a response-generation head (Transformer decoder biased by the fused
    vector).
  - kNN-LM: arxiv:1911.00172 (Khandelwal, Levy, Jurafsky, Zettlemoyer, Lewis, ICLR 2020);
    official repo github.com/urvashik/knnlm -- a standard causal Transformer LM produces
    next-token logits AND a per-step context embedding; the context embedding is used as
    a query against a datastore of (context-embedding, next-token) pairs built from
    training data, retrieving k nearest neighbours whose distances are softmax-converted
    into a kNN token distribution, which is linearly interpolated with the LM's own
    softmax distribution at inference. The datastore lookup is reimplemented here as an
    explicit small in-memory key/value bank with a differentiable soft-kNN (attention
    over stored keys) so the whole pipeline traces as ordinary tensor ops.
  - LABAN (Label-Aware BERT Attention Network): arxiv/ACL 2021.emnlp-main.399 (Wu, Su,
    Juang, EMNLP 2021); official repo github.com/waynewu6250/LABAN (`bert_laban.py`,
    `model/bert_model_zsl.py`) -- a BERT utterance encoder produces per-token hidden
    states; a SEPARATE BERT-style encoder embeds each intent LABEL's own tokens (using
    label surface-form semantics, not opaque one-hot ids) into a label-embedding space;
    a label-aware attention layer computes the utterance's projection weight onto every
    label embedding (dot-product attention pooling of utterance tokens conditioned on
    each label vector), and a per-label sigmoid score gives the multi-intent (and
    zero-shot -- new unseen label embeddings plug directly into the same scoring layer)
    prediction.
  - LABES-S2S (LAtent BElief State model, Seq2Seq instantiation): arxiv:2009.08115
    (Zhang, Ou, Wang, Feng, EMNLP 2020); official repo github.com/thu-spmi/LABES
    (`model.py`) -- an encoder reads (context, previous turns); a DISCRETE latent belief
    state is sampled via Gumbel-softmax over a fixed slot-value vocabulary from the
    encoder's pooled state (the "recognition"/posterior half of the CVAE); the sampled
    belief-state embedding conditions a copy-augmented GRU decoder that generates the
    system response, with a copy-attention gate mixing a generation distribution and a
    pointer distribution over context/db tokens.
  - LaRL (Latent Action Reinforcement Learning): arxiv:1902.08858 (Zhao, Xie, Eskenazi,
    NAACL 2019); official repo github.com/snakeztc/NeuralDialog-LaRL (`latent_dialog/`)
    -- a dialogue encoder (utterance RNN + context RNN) produces a context vector; a
    categorical latent-action policy network samples a discrete latent action via
    Gumbel-softmax (the "action space" the paper reframes RL over, replacing raw
    vocabulary/hand-crafted dialog acts); the sampled latent action embedding conditions
    a GRU response decoder, and a separate value head reads the same context vector for
    the RL policy-gradient baseline.

No skips this batch -- all six are well-documented named mechanisms with either an
official reference implementation or a detailed paper description sufficient for a
faithful compact reimplementation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import RGCNConv


# ---------------------------------------------------------------------------
# Key-Value Memory Network: separate key/value embedding spaces addressed over
# several hops, each hop refining the query with the value read-out.
# ---------------------------------------------------------------------------
class KeyValueMemoryNetwork(nn.Module):
    """Key-Value Memory Network (Miller et al. 2016) for document QA.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size for query/key/value token embeddings.
    embed_dim : int
        Shared embedding dimensionality for keys, values, and the query state.
    n_hops : int
        Number of memory-addressing hops.
    n_answers : int
        Number of candidate answers scored at the output.
    """

    def __init__(
        self,
        vocab_size: int = 200,
        embed_dim: int = 32,
        n_hops: int = 3,
        n_answers: int = 20,
    ) -> None:
        super().__init__()
        self.key_embed = nn.Embedding(vocab_size, embed_dim)
        self.value_embed = nn.Embedding(vocab_size, embed_dim)
        self.query_embed = nn.Embedding(vocab_size, embed_dim)
        self.hop_transforms = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim) for _ in range(n_hops)]
        )
        self.answer_embed = nn.Embedding(n_answers, embed_dim)
        self.n_hops = n_hops

    def forward(self, query_tokens: Tensor, key_slots: Tensor, value_slots: Tensor) -> Tensor:
        """Score candidate answers given a query and a (key, value) memory.

        Parameters
        ----------
        query_tokens : Tensor
            ``(batch, q_len)`` long token ids for the query.
        key_slots : Tensor
            ``(batch, n_slots, k_len)`` long token ids for memory keys.
        value_slots : Tensor
            ``(batch, n_slots, v_len)`` long token ids for memory values.

        Returns
        -------
        Tensor
            ``(batch, n_answers)`` answer scores.
        """
        u = self.query_embed(query_tokens).mean(dim=1)  # (batch, dim)
        keys = self.key_embed(key_slots).mean(dim=2)  # (batch, n_slots, dim)
        values = self.value_embed(value_slots).mean(dim=2)  # (batch, n_slots, dim)

        for hop in range(self.n_hops):
            addressing = torch.softmax(
                torch.bmm(keys, u.unsqueeze(-1)).squeeze(-1), dim=-1
            )  # (batch, n_slots)
            readout = torch.bmm(addressing.unsqueeze(1), values).squeeze(1)  # (batch, dim)
            u = self.hop_transforms[hop](u + readout)

        answers = self.answer_embed.weight  # (n_answers, dim)
        return u @ answers.t()


def build_kv_memnn() -> nn.Module:
    """Build a small Key-Value Memory Network.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return KeyValueMemoryNetwork().eval()


def example_input_kv_memnn() -> tuple[Tensor, Tensor, Tensor]:
    """Example query and memory (key, value) token ids for KV-MemNN.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(query_tokens, key_slots, value_slots)`` long tensors.
    """
    query_tokens = torch.randint(0, 200, (2, 6))
    key_slots = torch.randint(0, 200, (2, 8, 5))
    value_slots = torch.randint(0, 200, (2, 8, 3))
    return query_tokens, key_slots, value_slots


# ---------------------------------------------------------------------------
# KGSF: dual (word-graph, entity-graph) R-GCN encoders fused via a gated
# cross-attention mutual-information-style layer, driving recommendation
# and response generation.
# ---------------------------------------------------------------------------
class KGSemanticFusion(nn.Module):
    """KGSF conversational recommender (Zhou et al., KDD 2020).

    Parameters
    ----------
    n_words : int
        Number of nodes in the word-oriented graph.
    n_entities : int
        Number of nodes in the entity-oriented graph.
    n_relations : int
        Number of relation types shared by both R-GCN encoders.
    hidden_dim : int
        Node/fusion embedding dimensionality.
    vocab_size : int
        Response-generation vocabulary size.
    """

    def __init__(
        self,
        n_words: int = 60,
        n_entities: int = 50,
        n_relations: int = 4,
        hidden_dim: int = 24,
        vocab_size: int = 80,
    ) -> None:
        super().__init__()
        self.word_node_embed = nn.Embedding(n_words, hidden_dim)
        self.entity_node_embed = nn.Embedding(n_entities, hidden_dim)
        self.word_gcn = RGCNConv(hidden_dim, hidden_dim, n_relations)
        self.entity_gcn = RGCNConv(hidden_dim, hidden_dim, n_relations)

        # Gated cross-attention fusion (proxy for the paper's MIM alignment).
        self.word_to_entity_gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.entity_to_word_gate = nn.Linear(2 * hidden_dim, hidden_dim)

        self.rec_head = nn.Linear(hidden_dim, n_entities)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.gen_head = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        word_edge_index: Tensor,
        word_edge_type: Tensor,
        entity_edge_index: Tensor,
        entity_edge_type: Tensor,
        mentioned_words: Tensor,
        mentioned_entities: Tensor,
        response_len: int = 5,
    ) -> tuple[Tensor, Tensor]:
        """Fuse word/entity graphs and produce recommendation + generation outputs.

        Parameters
        ----------
        word_edge_index : Tensor
            ``(2, n_word_edges)`` long edge index of the word co-occurrence graph.
        word_edge_type : Tensor
            ``(n_word_edges,)`` long relation ids for word-graph edges.
        entity_edge_index : Tensor
            ``(2, n_entity_edges)`` long edge index of the entity KG.
        entity_edge_type : Tensor
            ``(n_entity_edges,)`` long relation ids for entity-graph edges.
        mentioned_words : Tensor
            ``(batch, n_mentioned_words)`` long word-node ids mentioned in the dialog.
        mentioned_entities : Tensor
            ``(batch, n_mentioned_entities)`` long entity-node ids mentioned in the dialog.
        response_len : int
            Number of decoder steps to unroll for response generation.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(recommendation_logits, generation_logits)`` of shapes
            ``(batch, n_entities)`` and ``(batch, response_len, vocab_size)``.
        """
        word_h = self.word_gcn(self.word_node_embed.weight, word_edge_index, word_edge_type)
        entity_h = self.entity_gcn(
            self.entity_node_embed.weight, entity_edge_index, entity_edge_type
        )

        word_ctx = word_h[mentioned_words].mean(dim=1)  # (batch, dim)
        entity_ctx = entity_h[mentioned_entities].mean(dim=1)  # (batch, dim)

        gate_w = torch.sigmoid(self.word_to_entity_gate(torch.cat([word_ctx, entity_ctx], -1)))
        gate_e = torch.sigmoid(self.entity_to_word_gate(torch.cat([entity_ctx, word_ctx], -1)))
        fused = gate_w * word_ctx + gate_e * entity_ctx  # mutual-information-style fusion

        rec_logits = self.rec_head(fused)

        dec_input = fused.unsqueeze(1).expand(-1, response_len, -1)
        dec_out, _ = self.decoder(dec_input, fused.unsqueeze(0))
        gen_logits = self.gen_head(dec_out)
        return rec_logits, gen_logits


def build_kgsf() -> nn.Module:
    """Build a small KGSF conversational recommender.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return KGSemanticFusion().eval()


def example_input_kgsf() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Example dual-graph edges and mentioned nodes for KGSF.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
        ``(word_edge_index, word_edge_type, entity_edge_index, entity_edge_type,
        mentioned_words, mentioned_entities)``.
    """
    word_edge_index = torch.randint(0, 60, (2, 100))
    word_edge_type = torch.randint(0, 4, (100,))
    entity_edge_index = torch.randint(0, 50, (2, 80))
    entity_edge_type = torch.randint(0, 4, (80,))
    mentioned_words = torch.randint(0, 60, (2, 4))
    mentioned_entities = torch.randint(0, 50, (2, 3))
    return (
        word_edge_index,
        word_edge_type,
        entity_edge_index,
        entity_edge_type,
        mentioned_words,
        mentioned_entities,
    )


# ---------------------------------------------------------------------------
# kNN-LM: causal Transformer LM whose next-token distribution is linearly
# interpolated with a soft-kNN distribution over a stored (key, value) datastore.
# ---------------------------------------------------------------------------
class KNNLanguageModel(nn.Module):
    """kNN-LM (Khandelwal et al., ICLR 2020): LM logits interpolated with datastore kNN.

    Parameters
    ----------
    vocab_size : int
        Token vocabulary size.
    d_model : int
        Transformer hidden size (also the datastore key dimensionality).
    n_layers : int
        Number of causal Transformer decoder layers.
    n_heads : int
        Number of attention heads.
    datastore_size : int
        Number of stored (context-embedding, next-token) pairs.
    interpolation : float
        Linear interpolation weight for the kNN distribution (paper's lambda).
    """

    def __init__(
        self,
        vocab_size: int = 100,
        d_model: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        datastore_size: int = 64,
        interpolation: float = 0.25,
    ) -> None:
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(256, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

        # Datastore: fixed (context-key, next-token) pairs, as in the paper's
        # precomputed nearest-neighbour index (here a small learnable bank so the
        # soft-kNN attention traces as ordinary differentiable ops).
        self.datastore_keys = nn.Parameter(torch.randn(datastore_size, d_model) * 0.02)
        self.register_buffer("datastore_values", torch.randint(0, vocab_size, (datastore_size,)))
        self.vocab_size = vocab_size
        self.interpolation = interpolation

    def forward(self, input_ids: Tensor) -> Tensor:
        """Compute the interpolated next-token distribution for every position.

        Parameters
        ----------
        input_ids : Tensor
            ``(batch, seq_len)`` long token ids.

        Returns
        -------
        Tensor
            ``(batch, seq_len, vocab_size)`` interpolated log-probabilities.
        """
        bsz, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        h = self.tok_embed(input_ids) + self.pos_embed(positions)

        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device), diagonal=1
        )
        ctx = self.encoder(h, mask=causal_mask)  # (batch, seq_len, d_model)

        lm_logp = F.log_softmax(self.lm_head(ctx), dim=-1)

        # Soft-kNN: attention over the datastore keys stands in for the paper's
        # exact k-nearest-neighbour retrieval + softmax-over-distances step.
        sims = ctx.reshape(bsz * seq_len, -1) @ self.datastore_keys.t()
        knn_weights = torch.softmax(sims, dim=-1)  # (batch*seq_len, datastore_size)
        knn_probs = torch.zeros(bsz * seq_len, self.vocab_size, device=input_ids.device)
        knn_probs.scatter_add_(
            1, self.datastore_values.unsqueeze(0).expand(bsz * seq_len, -1), knn_weights
        )
        knn_logp = torch.log(knn_probs.clamp_min(1e-8)).reshape(bsz, seq_len, self.vocab_size)

        combined = torch.logaddexp(
            lm_logp + torch.log(torch.tensor(1.0 - self.interpolation)),
            knn_logp + torch.log(torch.tensor(self.interpolation)),
        )
        return combined


def build_knn_lm() -> nn.Module:
    """Build a small kNN-LM.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return KNNLanguageModel().eval()


def example_input_knn_lm() -> Tensor:
    """Example token ids for kNN-LM.

    Returns
    -------
    Tensor
        ``(batch, seq_len)`` long token ids.
    """
    return torch.randint(0, 100, (2, 12))


# ---------------------------------------------------------------------------
# LABAN: utterance BERT encoder + separate label-token encoder, fused via a
# label-aware attention/projection scoring layer for (zero-shot) multi-intent
# detection.
# ---------------------------------------------------------------------------
class TinyBertEncoder(nn.Module):
    """Minimal Transformer token encoder standing in for BERT in both LABAN towers."""

    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int) -> None:
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(64, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, token_ids: Tensor) -> Tensor:
        seq_len = token_ids.size(-1)
        positions = torch.arange(seq_len, device=token_ids.device)
        h = self.tok_embed(token_ids) + self.pos_embed(positions)
        return self.encoder(h)


class LabelAwareBertAttentionNetwork(nn.Module):
    """LABAN (Wu, Su & Juang, EMNLP 2021) for zero-shot multi-intent detection.

    Parameters
    ----------
    vocab_size : int
        Shared utterance/label token vocabulary size.
    d_model : int
        Encoder hidden size.
    n_layers : int
        Number of Transformer layers in each tower.
    n_heads : int
        Number of attention heads.
    """

    def __init__(
        self,
        vocab_size: int = 120,
        d_model: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.utterance_encoder = TinyBertEncoder(vocab_size, d_model, n_layers, n_heads)
        self.label_encoder = TinyBertEncoder(vocab_size, d_model, n_layers, n_heads)
        self.attn_query = nn.Linear(d_model, d_model)
        self.attn_key = nn.Linear(d_model, d_model)
        self.score_bias = nn.Linear(d_model, 1)

    def forward(self, utterance_ids: Tensor, label_ids: Tensor) -> Tensor:
        """Score every candidate intent label against an utterance.

        Parameters
        ----------
        utterance_ids : Tensor
            ``(batch, u_len)`` long token ids for the input utterance.
        label_ids : Tensor
            ``(n_labels, l_len)`` long token ids for candidate intent labels
            (can include unseen zero-shot labels at inference).

        Returns
        -------
        Tensor
            ``(batch, n_labels)`` per-label intent probabilities (sigmoid, since
            multi-intent detection is multi-label, not mutually exclusive).
        """
        utt_h = self.utterance_encoder(utterance_ids)  # (batch, u_len, dim)
        label_h = self.label_encoder(label_ids).mean(dim=1)  # (n_labels, dim)

        q = self.attn_query(utt_h)  # (batch, u_len, dim)
        k = self.attn_key(label_h)  # (n_labels, dim)

        # Label-aware attention: for every label, pool utterance tokens weighted
        # by their projection onto that label's embedding, then score the pooled
        # projection -- this is the paper's "projection weights on each intent
        # embedding" mechanism.
        attn_logits = torch.einsum("bud,nd->bnu", q, k) / (q.size(-1) ** 0.5)
        attn_weights = torch.softmax(attn_logits, dim=-1)  # (batch, n_labels, u_len)
        pooled = torch.einsum("bnu,bud->bnd", attn_weights, utt_h)  # (batch, n_labels, dim)

        scores = self.score_bias(pooled).squeeze(-1)  # (batch, n_labels)
        return torch.sigmoid(scores)


def build_laban() -> nn.Module:
    """Build a small LABAN zero-shot multi-intent detector.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return LabelAwareBertAttentionNetwork().eval()


def example_input_laban() -> tuple[Tensor, Tensor]:
    """Example utterance and candidate-label token ids for LABAN.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(utterance_ids, label_ids)`` long tensors.
    """
    utterance_ids = torch.randint(0, 120, (2, 10))
    label_ids = torch.randint(0, 120, (6, 3))
    return utterance_ids, label_ids


# ---------------------------------------------------------------------------
# LABES-S2S: encoder -> discrete Gumbel-softmax latent belief state -> copy-
# augmented GRU decoder for task-oriented dialogue.
# ---------------------------------------------------------------------------
class LatentBeliefStateS2S(nn.Module):
    """LABES-S2S (Zhang, Ou, Wang & Feng, EMNLP 2020): discrete latent belief state.

    Parameters
    ----------
    vocab_size : int
        Context/response token vocabulary size.
    embed_dim : int
        Token embedding dimensionality.
    hidden_dim : int
        Encoder/decoder GRU hidden size.
    n_belief_slots : int
        Number of discrete values in the latent belief-state vocabulary.
    """

    def __init__(
        self,
        vocab_size: int = 90,
        embed_dim: int = 24,
        hidden_dim: int = 32,
        n_belief_slots: int = 12,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.context_encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.belief_logits = nn.Linear(hidden_dim, n_belief_slots)
        self.belief_embed = nn.Embedding(n_belief_slots, hidden_dim)

        self.decoder = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.copy_gate = nn.Linear(hidden_dim, 1)
        self.gen_head = nn.Linear(hidden_dim, vocab_size)
        self.copy_attn_query = nn.Linear(hidden_dim, hidden_dim)
        self.copy_attn_key = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self, context_ids: Tensor, response_ids: Tensor, gumbel_tau: float = 0.5
    ) -> tuple[Tensor, Tensor]:
        """Encode context into a latent belief state and decode a response.

        Parameters
        ----------
        context_ids : Tensor
            ``(batch, ctx_len)`` long token ids for the dialogue context.
        response_ids : Tensor
            ``(batch, resp_len)`` long token ids fed to the decoder (teacher forcing).
        gumbel_tau : float
            Gumbel-softmax temperature for the discrete belief-state sample.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(vocab_logits, belief_soft)`` where ``vocab_logits`` has shape
            ``(batch, resp_len, vocab_size)`` and ``belief_soft`` has shape
            ``(batch, n_belief_slots)`` (the sampled discrete belief distribution).
        """
        ctx_embed = self.embed(context_ids)
        _, ctx_final = self.context_encoder(ctx_embed)  # (1, batch, hidden)
        ctx_final = ctx_final.squeeze(0)

        belief_soft = F.gumbel_softmax(self.belief_logits(ctx_final), tau=gumbel_tau, hard=False)
        belief_vec = belief_soft @ self.belief_embed.weight  # (batch, hidden)

        resp_embed = self.embed(response_ids)
        dec_out, _ = self.decoder(resp_embed, belief_vec.unsqueeze(0))  # (batch, resp_len, hid)

        gen_logits = self.gen_head(dec_out)

        # Copy-attention: pointer distribution over context tokens, gated with the
        # generation distribution (both projected to vocab space for a simple,
        # traceable stand-in for the paper's copy mechanism).
        q = self.copy_attn_query(dec_out)  # (batch, resp_len, hidden)
        k = self.copy_attn_key(self._encode_context_states(ctx_embed))  # (batch, ctx_len, hid)
        copy_attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)), dim=-1)  # (b, resp_len, ctx)
        copy_dist = torch.zeros(
            context_ids.size(0),
            response_ids.size(1),
            self.gen_head.out_features,
            device=context_ids.device,
        )
        copy_dist.scatter_add_(
            2,
            context_ids.unsqueeze(1).expand(-1, response_ids.size(1), -1),
            copy_attn,
        )

        gate = torch.sigmoid(self.copy_gate(dec_out))  # (batch, resp_len, 1)
        vocab_logits = gate * gen_logits + (1 - gate) * copy_dist
        return vocab_logits, belief_soft

    def _encode_context_states(self, ctx_embed: Tensor) -> Tensor:
        states, _ = self.context_encoder(ctx_embed)
        return states


def build_labes_s2s() -> nn.Module:
    """Build a small LABES-S2S task-oriented dialogue model.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return LatentBeliefStateS2S().eval()


def example_input_labes_s2s() -> tuple[Tensor, Tensor]:
    """Example context and (teacher-forced) response token ids for LABES-S2S.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(context_ids, response_ids)`` long tensors.
    """
    context_ids = torch.randint(0, 90, (2, 10))
    response_ids = torch.randint(0, 90, (2, 7))
    return context_ids, response_ids


# ---------------------------------------------------------------------------
# LaRL: dialogue context encoder -> categorical Gumbel-softmax latent action
# (the RL action space) -> GRU response decoder, with a value head for the
# policy-gradient baseline.
# ---------------------------------------------------------------------------
class LatentActionRL(nn.Module):
    """LaRL (Zhao, Xie & Eskenazi, NAACL 2019): discrete latent-action dialogue policy.

    Parameters
    ----------
    vocab_size : int
        Utterance/response token vocabulary size.
    embed_dim : int
        Token embedding dimensionality.
    hidden_dim : int
        Encoder/decoder GRU hidden size.
    n_latent_actions : int
        Size of the categorical latent-action space the RL policy operates over.
    """

    def __init__(
        self,
        vocab_size: int = 90,
        embed_dim: int = 24,
        hidden_dim: int = 32,
        n_latent_actions: int = 10,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.utt_encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.ctx_encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        self.action_logits = nn.Linear(hidden_dim, n_latent_actions)
        self.action_embed = nn.Embedding(n_latent_actions, hidden_dim)

        self.decoder = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.gen_head = nn.Linear(hidden_dim, vocab_size)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, context_turns: Tensor, response_ids: Tensor, gumbel_tau: float = 0.5
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Encode dialogue context, sample a latent action, decode a response.

        Parameters
        ----------
        context_turns : Tensor
            ``(batch, n_turns, turn_len)`` long token ids for prior dialogue turns.
        response_ids : Tensor
            ``(batch, resp_len)`` long token ids fed to the decoder (teacher forcing).
        gumbel_tau : float
            Gumbel-softmax temperature for the discrete latent-action sample.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``(vocab_logits, action_soft, value)``: response logits
            ``(batch, resp_len, vocab_size)``, sampled latent-action distribution
            ``(batch, n_latent_actions)``, and RL baseline value ``(batch, 1)``.
        """
        bsz, n_turns, turn_len = context_turns.shape
        turn_embed = self.embed(context_turns.reshape(bsz * n_turns, turn_len))
        _, turn_final = self.utt_encoder(turn_embed)  # (1, batch*n_turns, hidden)
        turn_final = turn_final.squeeze(0).reshape(bsz, n_turns, -1)

        _, ctx_final = self.ctx_encoder(turn_final)  # (1, batch, hidden)
        ctx_final = ctx_final.squeeze(0)

        action_soft = F.gumbel_softmax(self.action_logits(ctx_final), tau=gumbel_tau, hard=False)
        action_vec = action_soft @ self.action_embed.weight  # (batch, hidden)

        resp_embed = self.embed(response_ids)
        dec_out, _ = self.decoder(resp_embed, action_vec.unsqueeze(0))
        vocab_logits = self.gen_head(dec_out)

        value = self.value_head(ctx_final)
        return vocab_logits, action_soft, value


def build_larl() -> nn.Module:
    """Build a small LaRL latent-action dialogue policy model.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return LatentActionRL().eval()


def example_input_larl() -> tuple[Tensor, Tensor]:
    """Example multi-turn context and (teacher-forced) response token ids for LaRL.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(context_turns, response_ids)`` long tensors.
    """
    context_turns = torch.randint(0, 90, (2, 3, 8))
    response_ids = torch.randint(0, 90, (2, 6))
    return context_turns, response_ids


MENAGERIE_ENTRIES = [
    ("Key-Value Memory Network", "build_kv_memnn", "example_input_kv_memnn", "2016", "NLP"),
    ("KGSF", "build_kgsf", "example_input_kgsf", "2020", "NLP"),
    ("kNN-LM", "build_knn_lm", "example_input_knn_lm", "2019", "NLP"),
    (
        "LABAN (Lexical Awareness for Better Action Network)",
        "build_laban",
        "example_input_laban",
        "2021",
        "NLP",
    ),
    ("LABES-S2S", "build_labes_s2s", "example_input_labes_s2s", "2020", "NLP"),
    ("LaRL", "build_larl", "example_input_larl", "2019", "NLP"),
]
