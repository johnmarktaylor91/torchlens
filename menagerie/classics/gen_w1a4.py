"""Compact faithful reimplementations of five dialogue-state-tracking / QA / ERC families.

Sources checked (paper + official source, no clone/pip-install; reimplemented from scratch):
  - DSTBERT (BERT-DST): arxiv:1907.03040 (Chao & Lane, INTERSPEECH 2019); official repo
    github.com/guanlinchao/bert-dst (`main.py::create_model`) -- a BERT encoder shared
    across all slots; per slot, the pooled [CLS] output feeds a class-prediction MLP head
    (none / dontcare / span-pointable / true-or-false) while the full token sequence output
    feeds a token-level start/end span-extraction MLP head (auxiliary task), jointly trained
    with a weighted sum of class cross-entropy and span cross-entropy.
  - DSTC / Neural Belief Tracker (NBT): arxiv:1606.03777 (Mrksic et al., ACL 2017); official
    repo github.com/nmrksic/neural-belief-tracker (`code/models.py::model_definition`,
    `define_CNN_model`) -- an n-gram CNN (filter widths 1/2/3) turns the utterance's word
    vectors into a single representation r; each ontology value's semantic vector is
    sigmoid-gated into a candidate vector c; r is combined with every candidate via
    elementwise multiplication then a joint hidden layer collapses each interaction to one
    pre-softmax score per value. Two auxiliary subnetworks add signal: a system-request
    branch (does a dot-product between the slot vector and the system's offered-slot vector
    gate the utterance representation) and a system-confirmation branch (slot+value dot
    products gate the utterance representation), both summed additively into the final
    per-value logits before softmax over the value list ("semantic delexicalization").
  - DSTQA: arxiv:1911.06192 (Zhou & Small, NeurIPS 2019 Workshop on Conversational AI);
    official repo github.com/alexa/dstqa (`dstqa/dstqa.py`) -- multi-domain DST as dynamic
    graph-attention QA: a bidirectional dialog<->domain-slot-value (DSV) attention
    (`bi_att`) fuses dialog token representations with DSV embeddings both ways; a
    slot-level dot-product classifier scores value embeddings against the attended dialog
    representation; and a slot-graph update (`ds_graph_embeddings`) folds each domain-slot's
    *previous-turn predicted value* embedding back into that slot's node embedding before
    the next turn, i.e. a dynamically-evolving knowledge graph over turns, gated by a
    learned sigmoid gamma that blends the graph-attended and plain dialog representations.
  - Dynamic Memory Network (DMN), Kumar et al. 2016: arxiv:1506.07285; official PyTorch port
    github.com/jhyuklee/dmn-pytorch (`model.py::DMN`) -- an input-fact GRU and a
    question GRU produce sentence/question representations; the episodic memory module
    computes a soft attention gate per fact from the feature interaction
    `[fact*question, fact*memory, |fact-question|, |fact-memory|]` through a small MLP
    (`g1`/`g2`), then runs an attention-modulated GRU cell over facts where the gate
    interpolates between the new GRU state and the previous state
    (`h = g*GRUCell(fact,h) + (1-g)*h`); the resulting episode vector updates the memory via
    a second GRU cell, repeated for `num_hop` reasoning hops; an RNN-based answer module
    recurrently decodes an output sequence from the final memory and question vector.
  - EmotionIC (Emotional Inertia and Contagion), Li et al. 2023: arxiv:2303.11117;
    official repo github.com/lijfrank/EmotionIC (`model.py::EmotionIC`,
    `model_attention.py::ConvPosMultiHeadAttn`, `model_gru.py::ConvGRUCell`) -- two
    parallel branches over an utterance sequence with per-utterance speaker ids: (1) a
    "global" IM-MHA Transformer branch (`TransformerEncoder`, attn_type='convpos') that
    splits every attention head's score into a same-speaker stream and an other-speaker
    stream via a speaker-equality mask, each stream additionally biased by a relative
    positional term, then merges the two streams additively before softmax (this is the
    "inertia" vs "contagion" split at the attention level); (2) a "local"/individual
    DiaGRU branch (`ConvGRU`/`ConvGRUCell`) whose custom GRU cell takes *two* hidden
    states per step -- the same-speaker running memory `hx` and the immediately-previous
    (any-speaker) memory `hy` -- with reset/select gates over each, so every utterance is
    updated against both its own speaker's history and the general conversational flow.
    The two branch outputs are LayerNorm'd, concatenated with the (LayerNorm'd) input
    embedding, projected, and classified per utterance.

Skipped: cand_00042 "Dynamic Memory Network Plus (DMN+)" (arxiv:1603.01417,
github.com/dandelin/Dynamic-memory-networks-plus-Pytorch) -- already in the catalog. It is
the exact same paper/repo as cand_00035, built earlier under the identical canonical name
"DMN+ (Dynamic Memory Network Plus)" in `menagerie/classics/gen_w1a3.py`
(`build_dmn_plus`/`example_input_dmn_plus`); the build_queue.tsv row for cand_00042 itself
flags POTENTIAL_DEDUP with cand_00035. See MENAGERIE_ENTRIES below for the 5 built here.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# DSTBERT (BERT-DST): shared BERT encoder + per-slot dual heads -- a class
# head over the pooled [CLS] vector and a token-level start/end span head
# over the full sequence output (auxiliary span-extraction task).
# ---------------------------------------------------------------------------
class TinyBertEncoder(nn.Module):
    """Minimal BERT-style encoder: token+position+segment embeddings + Transformer stack."""

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        max_len: int = 48,
    ) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.seg_embed = nn.Embedding(2, d_model)
        self.ln = nn.LayerNorm(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.pooler = nn.Linear(d_model, d_model)

    def forward(self, input_ids: Tensor, segment_ids: Tensor) -> tuple[Tensor, Tensor]:
        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        embed = (
            self.token_embed(input_ids) + self.pos_embed(positions) + self.seg_embed(segment_ids)
        )
        sequence_output = self.encoder(self.ln(embed))
        pooled_output = torch.tanh(self.pooler(sequence_output[:, 0]))
        return sequence_output, pooled_output


class DSTBERT(nn.Module):
    """Compact BERT-DST: shared BERT + per-slot class head and start/end span head."""

    def __init__(
        self, vocab_size: int = 256, d_model: int = 32, n_slots: int = 3, n_class_labels: int = 4
    ) -> None:
        """Build the shared encoder plus one class head and one span head per slot."""

        super().__init__()
        self.n_slots = n_slots
        self.bert = TinyBertEncoder(vocab_size=vocab_size, d_model=d_model)
        self.class_heads = nn.ModuleList(
            [nn.Linear(d_model, n_class_labels) for _ in range(n_slots)]
        )
        self.span_heads = nn.ModuleList([nn.Linear(d_model, 2) for _ in range(n_slots)])

    def forward(self, input_ids: Tensor, segment_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Return (per-slot class logits [n_slots,B,C], per-slot start/end logits [n_slots,B,L,2])."""

        sequence_output, pooled_output = self.bert(input_ids, segment_ids)
        class_logits = torch.stack([head(pooled_output) for head in self.class_heads], dim=0)
        span_logits = torch.stack([head(sequence_output) for head in self.span_heads], dim=0)
        return class_logits, span_logits


def build_dstbert() -> nn.Module:
    """Build the compact DSTBERT dialogue-state-tracker."""

    return DSTBERT().eval()


def example_input_dstbert() -> tuple[Tensor, Tensor]:
    """Return (token ids [B,L], segment ids [B,L]) for a query+context pair."""

    input_ids = torch.randint(0, 256, (2, 24), dtype=torch.long)
    segment_ids = torch.cat(
        [torch.zeros(2, 12, dtype=torch.long), torch.ones(2, 12, dtype=torch.long)], dim=1
    )
    return input_ids, segment_ids


# ---------------------------------------------------------------------------
# DSTC / Neural Belief Tracker (NBT): n-gram CNN utterance encoder, sigmoid-
# gated candidate value vectors, elementwise interaction -> joint hidden ->
# per-value score, plus additive system-request and confirmation subnets.
# ---------------------------------------------------------------------------
class NgramCNNEncoder(nn.Module):
    """N-gram (width 1/2/3) max-over-time CNN utterance encoder, summed across widths."""

    def __init__(self, vector_dim: int = 32, num_filters: int = 24) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d(vector_dim, num_filters, kernel_size=k, padding=k // 2) for k in (1, 2, 3)]
        )

    def forward(self, word_vectors: Tensor) -> Tensor:
        x = word_vectors.transpose(1, 2)
        pooled = [torch.tanh(conv(x)).amax(dim=-1) for conv in self.convs]
        return torch.stack(pooled, dim=0).sum(dim=0)


class NeuralBeliefTracker(nn.Module):
    """Compact NBT: CNN utterance rep, gated value candidates, req/confirm subnets."""

    def __init__(
        self, vector_dim: int = 32, num_filters: int = 24, n_values: int = 5, hidden_dim: int = 20
    ) -> None:
        """Build the CNN encoder, candidate gate, joint scorer, and the two auxiliary subnets."""

        super().__init__()
        self.n_values = n_values
        self.utterance_cnn = NgramCNNEncoder(vector_dim, num_filters)
        self.candidate_gate = nn.Linear(vector_dim, num_filters)
        self.joint_hidden = nn.Linear(num_filters, hidden_dim)
        self.joint_presoftmax = nn.Linear(hidden_dim, 1)
        self.sysreq_hidden = nn.Linear(num_filters, hidden_dim)
        self.sysreq_presoftmax = nn.Linear(hidden_dim, 1)
        self.confirm_hidden = nn.Linear(num_filters, hidden_dim)
        self.confirm_presoftmax = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        word_vectors: Tensor,
        value_vectors: Tensor,
        system_req_slot_vec: Tensor,
        slot_vec: Tensor,
    ) -> Tensor:
        """Return per-value pre-softmax scores [B, n_values] combining all three signals."""

        utter_rep = self.utterance_cnn(word_vectors)
        candidates = torch.sigmoid(self.candidate_gate(value_vectors))

        interaction = utter_rep.unsqueeze(1) * candidates.unsqueeze(0)
        joint_hidden = torch.sigmoid(self.joint_hidden(interaction))
        joint_logits = self.joint_presoftmax(joint_hidden).squeeze(-1)

        gate = (slot_vec * system_req_slot_vec).mean(dim=-1, keepdim=True)
        decision = gate * utter_rep
        sysreq_hidden = torch.sigmoid(self.sysreq_hidden(decision))
        sysreq_logits = self.sysreq_presoftmax(sysreq_hidden).expand(-1, self.n_values)

        confirm_hidden = torch.sigmoid(self.confirm_hidden(decision))
        confirm_logits = self.confirm_presoftmax(confirm_hidden).expand(-1, self.n_values)

        return joint_logits + sysreq_logits + confirm_logits


def build_neural_belief_tracker() -> nn.Module:
    """Build the compact Neural Belief Tracker (NBT)."""

    return NeuralBeliefTracker().eval()


def example_input_neural_belief_tracker() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return (word vectors [B,L,D], value vectors [V,D], system-req slot vec, slot vec)."""

    word_vectors = torch.randn(2, 16, 32)
    value_vectors = torch.randn(5, 32)
    system_req_slot_vec = torch.randn(2, 32)
    slot_vec = torch.randn(2, 32)
    return word_vectors, value_vectors, system_req_slot_vec, slot_vec


# ---------------------------------------------------------------------------
# DSTQA: bidirectional dialog<->domain-slot-value attention, dot-product
# value classifier, and a turn-to-turn dynamic knowledge-graph update that
# folds the previous turn's predicted value back into the slot embedding.
# ---------------------------------------------------------------------------
class BiAttention(nn.Module):
    """Bidirectional attention fusing dialog tokens with domain-slot-value (DSV) embeddings."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Bilinear(d_model, d_model, 1)

    def forward(self, dialog: Tensor, dsv: Tensor) -> tuple[Tensor, Tensor]:
        batch, seq_len, d_model = dialog.shape
        num_values = dsv.shape[1]
        dialog_exp = dialog.unsqueeze(2).expand(batch, seq_len, num_values, d_model)
        dsv_exp = dsv.unsqueeze(1).expand(batch, seq_len, num_values, d_model)
        similarity = self.proj(dialog_exp.reshape(-1, d_model), dsv_exp.reshape(-1, d_model))
        similarity = similarity.view(batch, seq_len, num_values)

        dialog_to_dsv_attn = torch.softmax(similarity, dim=-1)
        dialog_ctx = torch.bmm(dialog_to_dsv_attn, dsv)
        new_dialog = dialog + dialog_ctx

        dsv_to_dialog_attn = torch.softmax(similarity.transpose(1, 2), dim=-1)
        dsv_ctx = torch.bmm(dsv_to_dialog_attn, dialog)
        new_dsv = dsv + dsv_ctx
        return new_dialog, new_dsv


class DSTQA(nn.Module):
    """Compact DSTQA: bi-attention DST with a turn-to-turn dynamic slot-value graph."""

    def __init__(self, d_model: int = 24, n_values: int = 5, n_slots: int = 2) -> None:
        """Build the phrase encoder, bi-attention, dot-product classifier, and graph gate."""

        super().__init__()
        self.n_slots = n_slots
        self.n_values = n_values
        self.phrase_rnn = nn.GRU(d_model, d_model, batch_first=True, bidirectional=True)
        self.phrase_proj = nn.Linear(2 * d_model, d_model)
        self.bi_att = BiAttention(d_model)
        self.ds_dialog_attn = nn.Linear(d_model, d_model)
        self.class_head = nn.Linear(d_model, d_model)
        self.graph_self_attn = nn.Bilinear(d_model, d_model, 1)
        self.graph_gamma = nn.Linear(d_model, 1)

    def forward(
        self,
        dialog_embed: Tensor,
        slot_embed: Tensor,
        value_embed: Tensor,
        prev_value_embed: Tensor,
    ) -> Tensor:
        """Return per-slot per-value logits [n_slots, B, n_values] for the current turn."""

        batch = dialog_embed.shape[0]
        dialog_repr, _ = self.phrase_rnn(dialog_embed)
        dialog_repr = self.phrase_proj(dialog_repr)

        graph_slot_embed = slot_embed + prev_value_embed.mean(dim=0)

        all_logits = []
        for slot_idx in range(self.n_slots):
            dsv = slot_embed[slot_idx].unsqueeze(0) + value_embed
            dsv = dsv.unsqueeze(0).expand(batch, -1, -1)
            fused_dialog, fused_dsv = self.bi_att(dialog_repr, dsv)

            ds_dialog_att = torch.softmax(self.ds_dialog_attn(fused_dialog).sum(dim=-1), dim=-1)
            ds_dialog_repr = torch.bmm(ds_dialog_att.unsqueeze(1), fused_dialog).squeeze(1)

            graph_query = graph_slot_embed[slot_idx].unsqueeze(0).expand(batch, -1)
            self_score = self.graph_self_attn(ds_dialog_repr, graph_query)
            gamma = torch.sigmoid(self.graph_gamma(ds_dialog_repr + graph_query))
            ds_dialog_repr = (1 - gamma) * ds_dialog_repr + gamma * (
                torch.sigmoid(self_score) * graph_query
            )

            w = self.class_head(ds_dialog_repr)
            logits = torch.bmm(w.unsqueeze(1), fused_dsv.transpose(1, 2)).squeeze(1)
            all_logits.append(logits)
        return torch.stack(all_logits, dim=0)


def build_dstqa() -> nn.Module:
    """Build the compact DSTQA dynamic-graph dialogue-state tracker."""

    return DSTQA().eval()


def example_input_dstqa() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return (dialog embeds [B,L,D], slot embeds [S,D], value embeds [V,D], prev-value embeds [V,D])."""

    dialog_embed = torch.randn(2, 20, 24)
    slot_embed = torch.randn(2, 24)
    value_embed = torch.randn(5, 24)
    prev_value_embed = torch.randn(5, 24)
    return dialog_embed, slot_embed, value_embed, prev_value_embed


# ---------------------------------------------------------------------------
# Dynamic Memory Network (DMN, Kumar et al. 2016): input/question GRUs, a
# gated attention-GRU episodic memory (independent sigmoid gates from a
# 4-way feature interaction), multi-hop memory GRU, RNN answer decoder.
# ---------------------------------------------------------------------------
class DMNEpisodicMemory(nn.Module):
    """Gated attention-GRU episode module: per-fact sigmoid gate blends GRU update vs. hold."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.gate_hidden = nn.Linear(4 * hidden_dim, hidden_dim)
        self.gate_out = nn.Linear(hidden_dim, 1)
        self.attn_cell = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, facts: Tensor, question: Tensor, memory: Tensor) -> Tensor:
        batch, n_facts, hidden_dim = facts.shape
        q_exp = question.unsqueeze(1).expand(batch, n_facts, hidden_dim)
        m_exp = memory.unsqueeze(1).expand(batch, n_facts, hidden_dim)
        z = torch.cat(
            [facts * q_exp, facts * m_exp, (facts - q_exp).abs(), (facts - m_exp).abs()], dim=-1
        )
        gates = torch.sigmoid(self.gate_out(torch.tanh(self.gate_hidden(z)))).squeeze(-1)

        h = torch.zeros(batch, hidden_dim, device=facts.device, dtype=facts.dtype)
        for t in range(n_facts):
            g = gates[:, t].unsqueeze(-1)
            h = g * self.attn_cell(facts[:, t], h) + (1 - g) * h
        return h


class DynamicMemoryNetwork(nn.Module):
    """Compact original DMN: input/question GRUs, gated episodic memory, RNN answer decoder."""

    def __init__(
        self, vocab_size: int = 64, hidden_dim: int = 24, num_hop: int = 3, answer_len: int = 3
    ) -> None:
        """Build the input/question encoders, episodic memory, memory-update cell, answer RNN."""

        super().__init__()
        self.num_hop = num_hop
        self.answer_len = answer_len
        self.vocab_size = vocab_size
        self.word_embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.input_rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.question_rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.episode = DMNEpisodicMemory(hidden_dim)
        self.memory_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.answer_cell = nn.GRUCell(hidden_dim + vocab_size, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, facts_tokens: Tensor, question_tokens: Tensor) -> Tensor:
        """Return answer logits [B, answer_len, vocab_size]."""

        batch, n_facts, fact_len = facts_tokens.shape
        fact_embed = self.word_embed(facts_tokens.reshape(batch * n_facts, fact_len))
        fact_gru_out, _ = self.input_rnn(fact_embed)
        facts = fact_gru_out[:, -1].reshape(batch, n_facts, -1)

        q_embed = self.word_embed(question_tokens)
        q_gru_out, _ = self.question_rnn(q_embed)
        question = q_gru_out[:, -1]

        memory = question
        for _ in range(self.num_hop):
            episode = self.episode(facts, question, memory)
            memory = self.memory_cell(episode, memory)

        y = torch.softmax(self.out(memory), dim=-1)
        a_hidden = memory
        outputs = []
        for _ in range(self.answer_len):
            a_hidden = self.answer_cell(torch.cat([y, question], dim=-1), a_hidden)
            z = self.out(a_hidden)
            y = torch.softmax(z, dim=-1)
            outputs.append(z)
        return torch.stack(outputs, dim=1)


def build_dmn() -> nn.Module:
    """Build the compact original Dynamic Memory Network (DMN)."""

    return DynamicMemoryNetwork().eval()


def example_input_dmn() -> tuple[Tensor, Tensor]:
    """Return (fact token ids [B, n_facts, fact_len], question token ids [B, q_len])."""

    facts_tokens = torch.randint(1, 64, (2, 4, 6), dtype=torch.long)
    question_tokens = torch.randint(1, 64, (2, 5), dtype=torch.long)
    return facts_tokens, question_tokens


# ---------------------------------------------------------------------------
# EmotionIC (Li et al. 2023): a global IM-MHA branch that splits every
# attention head into same-speaker and other-speaker streams merged
# additively before softmax ("inertia" vs "contagion"), fused with a local
# DiaGRU branch whose cell gates over both a same-speaker and a
# most-recent-any-speaker hidden state, then concat -> project -> classify.
# ---------------------------------------------------------------------------
class InertiaContagionAttention(nn.Module):
    """IM-MHA: same-speaker / other-speaker attention streams merged before softmax."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 5 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor, speaker_ids: Tensor) -> Tensor:
        batch, seq_len, d_model = x.shape
        q, k_self, k_other, k_pos, v = self.qkv(x).chunk(5, dim=-1)

        def split_heads(t: Tensor) -> Tensor:
            return t.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q, k_self, k_other, v = (
            split_heads(q),
            split_heads(k_self),
            split_heads(k_other),
            split_heads(v),
        )

        same_speaker = (speaker_ids.unsqueeze(-1) == speaker_ids.unsqueeze(-2)).unsqueeze(1)
        attn_self = torch.matmul(q, k_self.transpose(-1, -2))
        attn_other = torch.matmul(q, k_other.transpose(-1, -2))
        attn_self = attn_self.masked_fill(~same_speaker, 0.0)
        attn_other = attn_other.masked_fill(same_speaker, 0.0)
        attn = (attn_self + attn_other) / math.sqrt(self.head_dim)

        weights = torch.softmax(attn, dim=-1)
        context = torch.matmul(weights, v).transpose(1, 2).reshape(batch, seq_len, d_model)
        return self.out_proj(context)


class DiaGRUCell(nn.Module):
    """Dual-state GRU cell: gates over same-speaker memory hx and most-recent memory hy."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w_ir = nn.Linear(input_dim, hidden_dim)
        self.w_is = nn.Linear(input_dim, hidden_dim)
        self.w_iz = nn.Linear(input_dim, hidden_dim)
        self.w_il = nn.Linear(input_dim, hidden_dim)
        self.w_hr = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_hs = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_hrz = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_hsz = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_hn = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_hm = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: Tensor, hx: Tensor, hy: Tensor) -> Tensor:
        r = torch.sigmoid(self.w_ir(x) + self.w_hr(hy))
        s = torch.sigmoid(self.w_is(x) + self.w_hs(hx))
        z = torch.sigmoid(self.w_iz(x) + self.w_hrz(hy) + self.w_hsz(hx))
        n = torch.tanh(self.w_il(x) + r * self.w_hn(hy) + s * self.w_hm(hx))
        return (1 - z) * n + z * hx


class DiaGRU(nn.Module):
    """Sequential DiaGRU branch: per-utterance same-speaker memory + running last state."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = DiaGRUCell(input_dim, hidden_dim)

    def forward(self, x: Tensor, speaker_ids: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape
        max_speakers = int(speaker_ids.max().item()) + 1
        speaker_memory = x.new_zeros(batch, max_speakers, self.hidden_dim)
        prev_state = x.new_zeros(batch, self.hidden_dim)
        outputs = []
        for t in range(seq_len):
            spk = speaker_ids[:, t]
            hx = speaker_memory[torch.arange(batch, device=x.device), spk]
            h_new = self.cell(x[:, t], hx, prev_state)
            speaker_memory = speaker_memory.clone()
            speaker_memory[torch.arange(batch, device=x.device), spk] = h_new
            prev_state = h_new
            outputs.append(h_new)
        return torch.stack(outputs, dim=1)


class EmotionIC(nn.Module):
    """Compact EmotionIC: global IM-MHA branch + local DiaGRU branch, fused and classified."""

    def __init__(
        self, d_model: int = 32, hidden_dim: int = 24, n_heads: int = 4, n_classes: int = 6
    ) -> None:
        """Build the global attention branch, local DiaGRU branch, and the fusion classifier."""

        super().__init__()
        self.global_attn = InertiaContagionAttention(d_model, n_heads)
        self.local_gru = DiaGRU(d_model, hidden_dim)
        self.ln_global = nn.LayerNorm(d_model)
        self.ln_local = nn.LayerNorm(hidden_dim)
        self.ln_origin = nn.LayerNorm(d_model)
        self.fuse = nn.Linear(hidden_dim + 2 * d_model, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, utterance_embed: Tensor, speaker_ids: Tensor) -> Tensor:
        """Return per-utterance log-probabilities [B, L, n_classes]."""

        global_hidden = self.global_attn(utterance_embed, speaker_ids)
        local_hidden = self.local_gru(utterance_embed, speaker_ids)
        origin = self.ln_origin(utterance_embed)
        fused = self.fuse(
            torch.cat([self.ln_global(global_hidden), self.ln_local(local_hidden), origin], dim=-1)
        )
        logits = self.classifier(fused)
        return F.log_softmax(logits, dim=-1)


def build_emotion_ic() -> nn.Module:
    """Build the compact EmotionIC emotion-recognition-in-conversation model."""

    return EmotionIC().eval()


def example_input_emotion_ic() -> tuple[Tensor, Tensor]:
    """Return (utterance embeddings [B, L, D], per-utterance speaker ids [B, L])."""

    utterance_embed = torch.randn(2, 6, 32)
    speaker_ids = torch.tensor([[0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0]], dtype=torch.long)
    return utterance_embed, speaker_ids


MENAGERIE_ENTRIES = [
    (
        "DSTBERT (DST via BERT + Auxiliary Tasks)",
        "build_dstbert",
        "example_input_dstbert",
        "2019",
        "NLP",
    ),
    (
        "DSTC (Neural Belief Tracker / NBT)",
        "build_neural_belief_tracker",
        "example_input_neural_belief_tracker",
        "2017",
        "NLP",
    ),
    ("DSTQA", "build_dstqa", "example_input_dstqa", "2019", "NLP"),
    ("Dynamic Memory Network (DMN)", "build_dmn", "example_input_dmn", "2016", "NLP"),
    (
        "EmotionIC (Emotional Inertia and Contagion Dialogue)",
        "build_emotion_ic",
        "example_input_emotion_ic",
        "2023",
        "NLP",
    ),
]
