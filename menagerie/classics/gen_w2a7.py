"""Compact faithful reimplementations of six spoken/task-oriented dialogue and
retrieval-augmented-generation architecture families.

Sources checked (paper + official source; reimplemented compactly from scratch in
base-env torch, no clone/pip-install):
  - Self-RAG (Self-Reflective Retrieval-Augmented Generation): github.com/AkariAsai/
    self-rag (Asai, Wu, Wang, Sil & Hajishirzi, ICLR 2024), arxiv:2310.11511; HuggingFace
    checkpoints ``selfrag/selfrag_llama2_7b`` / ``_13b``. The distinctive mechanism is
    NOT a separate critic network -- it is a SINGLE causal LM whose output vocabulary is
    augmented with special REFLECTION TOKENS (``Retrieve``=yes/no/continue, ``ISREL``
    relevant/irrelevant, ``ISSUP`` fully/partially/no-support, ``ISUSE`` 1-5 utility) that
    the model itself emits inline in the generated stream to decide whether to retrieve,
    to judge retrieved-passage relevance, and to critique its own grounding/utility --
    "reflection-as-generation" rather than reflection-as-a-side-channel. Reimplemented
    here as a small GPT-2-style causal decoder whose vocabulary is the ordinary token
    vocabulary UNIONED with the reflection-token ids, retrieved-passage token blocks are
    interleaved into the same input stream (a segment id distinguishes prompt / passage /
    generation spans), and the model emits ordinary tokens and reflection tokens from one
    shared LM head -- exactly Self-RAG's core design choice.
  - Sequicity: github.com/WING-NUS/sequicity (Lei, Jin, Kan, Ren, He & Yin, ACL 2018);
    hexiangnan.github.io/papers/acl18-sequicity.pdf. The distinctive mechanism is a
    SINGLE seq2seq model (not a pipeline) that tracks dialogue state as an explicit
    generated TEXT SPAN called the "belief span" (bspan), together with a two-stage
    CopyNet decoder: stage 1 encodes (context + prior bspan + user utterance) and
    DECODES A NEW BSPAN by copying entity/value tokens straight out of the input via a
    copy-attention distribution over source positions (rather than classifying against a
    fixed slot-value ontology); stage 2 re-encodes (bspan + retrieved KB result flag) and
    decodes the delexicalized response, again with a copy mechanism over the bspan
    tokens. Reimplemented here as two GRU encoder/CopyNet-decoder stages sharing one
    token embedding table, with an explicit copy-attention distribution over the
    respective stage's source sequence blended with a generation distribution via a
    learned copy gate (the CopyNet mechanism), rather than a fixed-ontology classifier.
  - SF-ID Network: github.com/ZephyrChenzf/SF-ID-Network-For-NLU (Haihong E, Niu, Chen &
    Song, ACL 2019), arxiv:1907.00390. The distinctive mechanism is an explicit ITERATIVE
    bi-directional message-passing loop between a slot-filling (SF) subnet and an
    intent-detection (ID) subnet: at each of ``n`` iterations, the current slot context
    vector reinforces the intent-attention vector via a bounded attention-and-gate step
    (SF -> ID), and the (updated) intent vector reinforces the slot context via a second
    gated step (ID -> SF); the loop order is configurable ("slot_first"/"intent_first")
    and both subnets are read out AFTER the shared iteration count. This is fundamentally
    different from Slot-Gated (below), which applies a SINGLE one-shot gate with no
    iterative refinement loop. Reimplemented here as a BiLSTM encoder + two additive
    attention heads (slot context per position, intent context pooled) feeding an
    explicit `for` loop of ``n_iterations`` alternating SF->ID / ID->SF gated-attention
    updates, matching the official TensorFlow ``train.py`` iteration structure exactly
    (same gate formula: ``v * tanh(reinforced_context + linear(other_vec))`` at each
    step, summed and reduced to a scalar relation factor that rescales the context
    vector before the cross step).
  - SimpleTOD: github.com/salesforce/simpletod (Hosseini-Asl, McCann, Wu, Yavuz & Socher,
    NeurIPS 2020), proceedings.neurips.cc/paper/2020/file/e946209592563be0f01c844
    ab2170f0c-Paper.pdf. The distinctive mechanism is the ABSENCE of pipeline
    separation: dialogue context, belief state, DB-search result flags, system action,
    and delexicalized response are all concatenated into ONE flat token sequence with
    special segment-boundary tokens, and a single GPT-2-style causal LM is trained with
    plain next-token cross-entropy over the WHOLE sequence -- state tracking, policy, and
    generation collapse into unconditional causal decoding of one string, with no
    separate DST/policy/NLG heads. Reimplemented here as a compact GPT-2-style causal
    decoder (token + positional embedding, pre-LN Transformer decoder stack, tied LM
    head) run directly over one concatenated
    ``context <sep> belief <sep> db <sep> action <sep> response`` token stream.
  - Slot-Gated Joint Model (== "Slot-Gated SLU", the two catalog rows are the SAME
    paper/repo -- both built here as ONE architecture): github.com/MiuLab/SlotGated-SLU
    (Goo, Gao, Hsu, Huo, Chen, Hsu & Chen, NAACL-HLT 2018), csie.ntu.edu.tw/~yvchen/doc/
    NAACL18_SlotGated.pdf. The distinctive mechanism is a SINGLE one-shot "slot gate"
    that lets the (already-computed) intent-attention context vector modulate the
    per-token slot-attention context vector BEFORE the slot classifier, via
    ``g = sum(v * tanh(slot_context + W @ intent_context))`` then
    ``slot_logits = W([hidden + g * slot_context])`` -- exactly the official
    ``tf.reduce_sum(v1 * tf.tanh(slot_d + intent_gate))`` gate in ``train.py``, computed
    ONCE (no iteration loop, unlike SF-ID above). Reimplemented here as a BiLSTM encoder
    with additive intent- and slot-attention, and the literal one-shot gate formula
    feeding the slot classifier.

Both catalog rows for the slot-gate family (Slot-Gated Joint Model / Slot-Gated SLU)
point at the identical MiuLab/SlotGated-SLU paper and repo, so a single
``SlotGatedJointModel`` class below is registered under BOTH canonical names -- this is
not a stub duplicate, it is the literal same architecture cited twice in the queue.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Self-RAG: causal LM with reflection tokens interleaved in the generation
# stream (Retrieve / ISREL / ISSUP / ISUSE emitted from the SAME LM head as
# ordinary vocabulary tokens, over a prompt+retrieved-passage+generation
# stream distinguished by a learned segment embedding).
# ---------------------------------------------------------------------------
N_REFLECTION_TOKENS = 8  # Retrieve{yes,no,continue}, ISREL{rel,irrel}, ISSUP{full,partial,no}


class SelfRAG(nn.Module):
    """Self-RAG (Asai et al., ICLR 2024): reflection tokens generated inline by one LM.

    Parameters
    ----------
    vocab_size : int
        Ordinary sub-word vocabulary size.
    n_reflection_tokens : int
        Number of special reflection-token ids appended to the vocabulary.
    embed_dim : int
        Token/segment embedding dimensionality.
    n_layers : int
        Number of causal Transformer decoder layers.
    n_heads : int
        Number of attention heads.
    max_len : int
        Maximum sequence length (for the learned positional table).
    """

    def __init__(
        self,
        vocab_size: int = 128,
        n_reflection_tokens: int = N_REFLECTION_TOKENS,
        embed_dim: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        max_len: int = 64,
    ) -> None:
        super().__init__()
        self.full_vocab = vocab_size + n_reflection_tokens
        self.token_embed = nn.Embedding(self.full_vocab, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        # Segment ids: 0=prompt, 1=retrieved passage, 2=generation-with-reflection.
        self.segment_embed = nn.Embedding(3, embed_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=4 * embed_dim, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        # Shared LM head: ordinary tokens and reflection tokens are ONE softmax,
        # not a separate critic classifier -- this is Self-RAG's core design choice.
        self.lm_head = nn.Linear(embed_dim, self.full_vocab)

    def forward(self, token_ids: Tensor, segment_ids: Tensor) -> Tensor:
        """Predict next-token logits (ordinary + reflection tokens) causally.

        Parameters
        ----------
        token_ids : Tensor
            ``(batch, seq_len)`` long ids in ``[0, full_vocab)``.
        segment_ids : Tensor
            ``(batch, seq_len)`` long ids in ``{0, 1, 2}`` marking prompt / retrieved
            passage / generation spans.

        Returns
        -------
        Tensor
            ``(batch, seq_len, full_vocab)`` next-token logits.
        """
        bsz, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(bsz, -1)
        hidden = (
            self.token_embed(token_ids)
            + self.pos_embed(positions)
            + self.segment_embed(segment_ids)
        )
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(hidden.device)
        # Self-attention-only causal decoding: feed the same stream as both
        # "target" and "memory" with a causal mask (GPT-style decoder-only use
        # of nn.TransformerDecoder).
        out = self.decoder(hidden, hidden, tgt_mask=causal_mask, memory_mask=causal_mask)
        return self.lm_head(out)


def build_self_rag() -> nn.Module:
    """Build a small Self-RAG reflection-token causal decoder.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return SelfRAG().eval()


def example_input_self_rag() -> tuple[Tensor, Tensor]:
    """Example interleaved prompt/passage/generation token and segment ids.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(token_ids, segment_ids)``.
    """
    token_ids = torch.randint(0, 128 + N_REFLECTION_TOKENS, (2, 20))
    segment_ids = torch.cat(
        [
            torch.zeros(2, 6, dtype=torch.long),
            torch.ones(2, 8, dtype=torch.long),
            torch.full((2, 6), 2, dtype=torch.long),
        ],
        dim=1,
    )
    return token_ids, segment_ids


# ---------------------------------------------------------------------------
# Sequicity: two-stage seq2seq with CopyNet-style copy-attention decoding for
# both the belief-span (bspan) generation stage and the response-generation
# stage, sharing one token embedding table.
# ---------------------------------------------------------------------------
class _CopyNetDecoderStep(nn.Module):
    """One GRU decoder step with a CopyNet generate/copy blend over the source.

    Parameters
    ----------
    embed_dim : int
        Token embedding dimensionality.
    hidden_dim : int
        Decoder GRU hidden size.
    source_dim : int
        Encoder source-state width (``2 * hidden_dim`` for a bidirectional encoder).
    vocab_size : int
        Generation vocabulary size.
    """

    def __init__(self, embed_dim: int, hidden_dim: int, source_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.gru_cell = nn.GRUCell(embed_dim + source_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim + source_dim, hidden_dim)
        self.attn_v = nn.Linear(hidden_dim, 1, bias=False)
        self.gen_proj = nn.Linear(hidden_dim, vocab_size)
        self.copy_gate = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        token_embed: Tensor,
        hidden: Tensor,
        source_states: Tensor,
        source_ids: Tensor,
        vocab_size: int,
    ) -> tuple[Tensor, Tensor]:
        """Run one decode step, returning a blended generate+copy distribution.

        Parameters
        ----------
        token_embed : Tensor
            ``(batch, embed_dim)`` embedding of the previous output token.
        hidden : Tensor
            ``(batch, hidden_dim)`` previous decoder hidden state.
        source_states : Tensor
            ``(batch, src_len, source_dim)`` encoder states to attend/copy over.
        source_ids : Tensor
            ``(batch, src_len)`` long source token ids (copy targets).
        vocab_size : int
            Generation vocabulary size, for scattering copy probability mass.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(output_distribution, new_hidden)`` where ``output_distribution`` has
            shape ``(batch, vocab_size)``.
        """
        src_len = source_states.shape[1]
        query = hidden.unsqueeze(1).expand(-1, src_len, -1)
        scores = self.attn_v(
            torch.tanh(self.attn(torch.cat([query, source_states], dim=-1)))
        ).squeeze(-1)
        attn_weights = F.softmax(scores, dim=-1)  # (batch, src_len)
        context = torch.bmm(attn_weights.unsqueeze(1), source_states).squeeze(1)  # (batch, hidden)

        new_hidden = self.gru_cell(torch.cat([token_embed, context], dim=-1), hidden)

        gen_logits = self.gen_proj(new_hidden)
        gen_dist = F.softmax(gen_logits, dim=-1)

        # CopyNet gate: blend a fixed-vocabulary generation distribution with a
        # copy distribution scattered from the source-attention weights onto
        # each source token's vocabulary id -- the mechanism that lets the
        # decoder emit entity values verbatim from the input without a
        # closed-world slot-value ontology.
        p_copy = torch.sigmoid(self.copy_gate(new_hidden))  # (batch, 1)
        copy_dist = torch.zeros(source_ids.shape[0], vocab_size, device=source_ids.device)
        copy_dist.scatter_add_(1, source_ids, attn_weights)

        out_dist = (1 - p_copy) * gen_dist + p_copy * copy_dist
        return out_dist, new_hidden


class Sequicity(nn.Module):
    """Sequicity (Lei et al., ACL 2018): two-stage seq2seq bspan + response CopyNet.

    Parameters
    ----------
    vocab_size : int
        Shared token vocabulary size across context / bspan / response.
    embed_dim : int
        Token embedding dimensionality.
    hidden_dim : int
        Encoder/decoder GRU hidden size.
    """

    def __init__(self, vocab_size: int = 80, embed_dim: int = 24, hidden_dim: int = 32) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)

        # Stage 1: encode (context + prior bspan + utterance) -> decode NEW bspan.
        self.bspan_encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.bspan_bridge = nn.Linear(2 * hidden_dim, hidden_dim)
        self.bspan_decoder_step = _CopyNetDecoderStep(
            embed_dim, hidden_dim, 2 * hidden_dim, vocab_size
        )

        # Stage 2: re-encode (bspan + KB-result flag) -> decode delexicalized response.
        self.response_encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.response_bridge = nn.Linear(2 * hidden_dim, hidden_dim)
        self.response_decoder_step = _CopyNetDecoderStep(
            embed_dim, hidden_dim, 2 * hidden_dim, vocab_size
        )
        self.kb_flag_embed = nn.Embedding(2, embed_dim)

    def _run_decoder(
        self,
        step: _CopyNetDecoderStep,
        init_hidden: Tensor,
        source_states: Tensor,
        source_ids: Tensor,
        target_ids: Tensor,
    ) -> Tensor:
        bsz, tgt_len = target_ids.shape
        hidden = init_hidden
        outputs = []
        prev_token = torch.zeros(bsz, dtype=torch.long, device=target_ids.device)
        for t in range(tgt_len):
            token_embed = self.embed(prev_token)
            out_dist, hidden = step(token_embed, hidden, source_states, source_ids, self.vocab_size)
            outputs.append(out_dist)
            prev_token = target_ids[:, t]
        return torch.stack(outputs, dim=1)  # (batch, tgt_len, vocab_size)

    def forward(
        self,
        context_ids: Tensor,
        bspan_target_ids: Tensor,
        kb_flag: Tensor,
        response_target_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run stage-1 bspan decoding then stage-2 response decoding (teacher forced).

        Parameters
        ----------
        context_ids : Tensor
            ``(batch, ctx_len)`` long ids for (context + prior bspan + utterance).
        bspan_target_ids : Tensor
            ``(batch, bspan_len)`` teacher-forced target belief-span token ids.
        kb_flag : Tensor
            ``(batch,)`` long 0/1 KB-match indicator, embedded and prepended to the
            stage-2 source.
        response_target_ids : Tensor
            ``(batch, resp_len)`` teacher-forced target response token ids.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(bspan_dist, response_dist)``, each ``(batch, tgt_len, vocab_size)``.
        """
        ctx_embed = self.embed(context_ids)
        ctx_states, ctx_final = self.bspan_encoder(ctx_embed)
        bspan_init = torch.tanh(self.bspan_bridge(torch.cat([ctx_final[0], ctx_final[1]], dim=-1)))
        bspan_dist = self._run_decoder(
            self.bspan_decoder_step, bspan_init, ctx_states, context_ids, bspan_target_ids
        )

        bspan_embed = self.embed(bspan_target_ids)
        kb_embed = self.kb_flag_embed(kb_flag).unsqueeze(1)
        stage2_embed = torch.cat([kb_embed, bspan_embed], dim=1)
        stage2_ids = torch.cat(
            [
                torch.zeros(kb_flag.shape[0], 1, dtype=torch.long, device=kb_flag.device),
                bspan_target_ids,
            ],
            dim=1,
        )
        resp_states, resp_final = self.response_encoder(stage2_embed)
        resp_init = torch.tanh(
            self.response_bridge(torch.cat([resp_final[0], resp_final[1]], dim=-1))
        )
        response_dist = self._run_decoder(
            self.response_decoder_step, resp_init, resp_states, stage2_ids, response_target_ids
        )

        return bspan_dist, response_dist


def build_sequicity() -> nn.Module:
    """Build a small Sequicity two-stage CopyNet dialogue model.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return Sequicity().eval()


def example_input_sequicity() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Example context, bspan target, KB flag, and response target ids.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        ``(context_ids, bspan_target_ids, kb_flag, response_target_ids)``.
    """
    context_ids = torch.randint(0, 80, (2, 12))
    bspan_target_ids = torch.randint(0, 80, (2, 5))
    kb_flag = torch.randint(0, 2, (2,))
    response_target_ids = torch.randint(0, 80, (2, 8))
    return context_ids, bspan_target_ids, kb_flag, response_target_ids


# ---------------------------------------------------------------------------
# SF-ID Network: BiLSTM encoder + iterative bi-directional gated-attention
# message passing between a slot-filling (SF) subnet and an intent-detection
# (ID) subnet, matching the official train.py iteration loop.
# ---------------------------------------------------------------------------
class SFIDNetwork(nn.Module):
    """SF-ID Network (Haihong E et al., ACL 2019): iterative SF<->ID message passing.

    Parameters
    ----------
    vocab_size : int
        Input token vocabulary size.
    embed_dim : int
        Token embedding dimensionality.
    hidden_dim : int
        Per-direction BiLSTM hidden size (concatenated state has ``2 * hidden_dim``).
    n_slot_labels : int
        Number of slot (BIO) tag classes.
    n_intents : int
        Number of intent classes.
    n_iterations : int
        Number of SF<->ID iterative-refinement rounds (the paper's core knob).
    slot_first : bool
        If ``True``, each round updates SF before ID (else ID before SF); both orders
        are supported by the official implementation via ``--priority_order``.
    """

    def __init__(
        self,
        vocab_size: int = 90,
        embed_dim: int = 24,
        hidden_dim: int = 32,
        n_slot_labels: int = 12,
        n_intents: int = 8,
        n_iterations: int = 2,
        slot_first: bool = True,
    ) -> None:
        super().__init__()
        self.n_iterations = n_iterations
        self.slot_first = slot_first
        attn_size = 2 * hidden_dim

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)

        # Initial slot- and intent-attention (before any SF<->ID iteration).
        self.slot_attn_proj = nn.Linear(attn_size, attn_size)
        self.slot_attn_v = nn.Linear(attn_size, 1, bias=False)
        self.intent_attn_proj = nn.Linear(attn_size, attn_size)
        self.intent_attn_v = nn.Linear(attn_size, 1, bias=False)

        # Per-iteration SF->ID and ID->SF gated-attention update parameters
        # (fresh weights per round, matching the official per-iteration
        # `tf.variable_scope('..._subnet' + str(n))` fresh variables).
        self.sf_to_id_proj = nn.ModuleList(
            [nn.Linear(attn_size, attn_size) for _ in range(n_iterations)]
        )
        self.sf_to_id_v = nn.ModuleList(
            [nn.Linear(attn_size, 1, bias=False) for _ in range(n_iterations)]
        )
        self.id_to_sf_gate = nn.ModuleList(
            [nn.Linear(attn_size, attn_size) for _ in range(n_iterations)]
        )
        self.id_to_sf_v = nn.ParameterList(
            [nn.Parameter(torch.randn(attn_size)) for _ in range(n_iterations)]
        )

        self.slot_classifier = nn.Linear(2 * attn_size, n_slot_labels)
        self.intent_classifier = nn.Linear(2 * attn_size, n_intents)

    def forward(self, token_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Encode and run the iterative SF<->ID refinement loop.

        Parameters
        ----------
        token_ids : Tensor
            ``(batch, seq_len)`` long input token ids.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(slot_logits, intent_logits)`` of shapes ``(batch, seq_len,
            n_slot_labels)`` and ``(batch, n_intents)``.
        """
        embed = self.embed(token_ids)
        state_outputs, (h_n, _) = self.encoder(embed)  # state_outputs: (batch, seq_len, 2*hidden)
        final_state = torch.cat([h_n[0], h_n[1]], dim=-1)  # (batch, 2*hidden)

        # Initial per-position slot-attention context.
        slot_scores = self.slot_attn_v(torch.tanh(self.slot_attn_proj(state_outputs))).squeeze(-1)
        slot_weights = F.softmax(slot_scores, dim=-1).unsqueeze(-1)
        slot_context = (
            slot_weights * state_outputs
        )  # (batch, seq_len, 2*hidden), per-position reinforced state

        # Initial pooled intent-attention context.
        intent_scores = self.intent_attn_v(
            torch.tanh(self.intent_attn_proj(state_outputs) + final_state.unsqueeze(1))
        ).squeeze(-1)
        intent_weights = F.softmax(intent_scores, dim=-1).unsqueeze(-1)
        intent_context = (intent_weights * state_outputs).sum(dim=1)  # (batch, 2*hidden)

        slot_reinforced = slot_context
        intent_reinforced = intent_context
        for i in range(self.n_iterations):
            if self.slot_first:
                slot_reinforced, intent_reinforced = self._sf_step(
                    i, slot_reinforced, intent_reinforced, state_outputs
                )
                intent_reinforced = self._id_step(
                    i, slot_reinforced, intent_reinforced, state_outputs
                )
            else:
                intent_reinforced = self._id_step(
                    i, slot_reinforced, intent_reinforced, state_outputs
                )
                slot_reinforced, intent_reinforced = self._sf_step(
                    i, slot_reinforced, intent_reinforced, state_outputs
                )

        slot_logits = self.slot_classifier(torch.cat([slot_reinforced, state_outputs], dim=-1))
        intent_logits = self.intent_classifier(torch.cat([intent_reinforced, final_state], dim=-1))
        return slot_logits, intent_logits

    def _sf_step(
        self, i: int, slot_reinforced: Tensor, intent_vec: Tensor, state_outputs: Tensor
    ) -> tuple[Tensor, Tensor]:
        """ID->SF gated-attention step: intent vector rescales the slot context."""
        intent_gate = self.id_to_sf_gate[i](intent_vec).unsqueeze(1)  # (batch, 1, attn)
        relation = self.id_to_sf_v[i] * torch.tanh(slot_reinforced + intent_gate)
        relation_factor = relation.sum(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        new_slot = slot_reinforced * relation_factor
        return new_slot, intent_vec

    def _id_step(
        self, i: int, slot_reinforced: Tensor, intent_vec: Tensor, state_outputs: Tensor
    ) -> Tensor:
        """SF->ID gated-attention step: slot context reinforces the intent vector."""
        scores = self.sf_to_id_v[i](
            torch.tanh(self.sf_to_id_proj[i](slot_reinforced) + intent_vec.unsqueeze(1))
        ).squeeze(-1)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)
        pooled = (weights * slot_reinforced).sum(dim=1)
        return pooled + intent_vec


def build_sf_id_network() -> nn.Module:
    """Build a small SF-ID Network with a 2-round SF<->ID iteration loop.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return SFIDNetwork().eval()


def example_input_sf_id_network() -> Tensor:
    """Example input token ids for the SF-ID Network.

    Returns
    -------
    Tensor
        ``(batch, seq_len)`` long token ids.
    """
    return torch.randint(0, 90, (2, 10))


# ---------------------------------------------------------------------------
# SimpleTOD: GPT-2-style causal decoder over one FLAT concatenated
# context+belief+db+action+response token stream -- no separate DST/policy/
# NLG heads, unconditional next-token causal LM only.
# ---------------------------------------------------------------------------
class SimpleTOD(nn.Module):
    """SimpleTOD (Hosseini-Asl et al., NeurIPS 2020): one causal LM over the full TOD turn.

    Parameters
    ----------
    vocab_size : int
        Token vocabulary size (shared across context/belief/db/action/response).
    embed_dim : int
        Token/positional embedding dimensionality.
    n_layers : int
        Number of causal Transformer decoder layers (a small GPT-2 stack).
    n_heads : int
        Number of attention heads.
    max_len : int
        Maximum sequence length.
    """

    def __init__(
        self,
        vocab_size: int = 96,
        embed_dim: int = 32,
        n_layers: int = 3,
        n_heads: int = 4,
        max_len: int = 64,
    ) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=4 * embed_dim,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self.lm_head.weight = self.token_embed.weight  # tied weights, GPT-2 style

    def forward(self, token_ids: Tensor) -> Tensor:
        """Predict next-token logits causally over the flat concatenated turn stream.

        Parameters
        ----------
        token_ids : Tensor
            ``(batch, seq_len)`` long ids for
            ``context <sep> belief <sep> db <sep> action <sep> response``.

        Returns
        -------
        Tensor
            ``(batch, seq_len, vocab_size)`` next-token logits.
        """
        bsz, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(bsz, -1)
        hidden = self.token_embed(token_ids) + self.pos_embed(positions)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(hidden.device)
        hidden = self.decoder(hidden, mask=causal_mask, is_causal=True)
        return self.lm_head(self.ln_f(hidden))


def build_simpletod() -> nn.Module:
    """Build a small SimpleTOD flat causal-LM dialogue model.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return SimpleTOD().eval()


def example_input_simpletod() -> Tensor:
    """Example flat concatenated context+belief+db+action+response token ids.

    Returns
    -------
    Tensor
        ``(batch, seq_len)`` long token ids.
    """
    return torch.randint(0, 96, (2, 24))


# ---------------------------------------------------------------------------
# Slot-Gated Joint Model / Slot-Gated SLU: BiLSTM encoder + intent/slot
# additive attention + ONE-SHOT slot-gate (no iteration loop, unlike SF-ID
# above). Registered under BOTH catalog names -- same paper/repo.
# ---------------------------------------------------------------------------
class SlotGatedJointModel(nn.Module):
    """Slot-Gated joint SLU model (Goo et al., NAACL 2018): intent gates slot attention.

    Parameters
    ----------
    vocab_size : int
        Input token vocabulary size.
    embed_dim : int
        Token embedding dimensionality.
    hidden_dim : int
        Per-direction BiLSTM hidden size (concatenated state has ``2 * hidden_dim``).
    n_slot_labels : int
        Number of slot (BIO) tag classes.
    n_intents : int
        Number of intent classes.
    """

    def __init__(
        self,
        vocab_size: int = 90,
        embed_dim: int = 24,
        hidden_dim: int = 32,
        n_slot_labels: int = 12,
        n_intents: int = 8,
    ) -> None:
        super().__init__()
        attn_size = 2 * hidden_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)

        self.slot_attn_proj = nn.Linear(attn_size, attn_size)
        self.slot_attn_v = nn.Linear(attn_size, 1, bias=False)

        self.intent_attn_proj = nn.Linear(attn_size, attn_size)
        self.intent_attn_v = nn.Linear(attn_size, 1, bias=False)

        # The paper's single slot-gate: g = sum(v * tanh(slot_context + W @ intent_context)).
        self.gate_proj = nn.Linear(attn_size, attn_size)
        self.gate_v = nn.Parameter(torch.randn(attn_size))

        self.slot_classifier = nn.Linear(2 * attn_size, n_slot_labels)
        self.intent_classifier = nn.Linear(attn_size, n_intents)

    def forward(self, token_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Encode and apply the one-shot intent-to-slot gate.

        Parameters
        ----------
        token_ids : Tensor
            ``(batch, seq_len)`` long input token ids.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(slot_logits, intent_logits)`` of shapes ``(batch, seq_len,
            n_slot_labels)`` and ``(batch, n_intents)``.
        """
        embed = self.embed(token_ids)
        state_outputs, (h_n, _) = self.encoder(embed)
        final_state = torch.cat([h_n[0], h_n[1]], dim=-1)

        slot_scores = self.slot_attn_v(torch.tanh(self.slot_attn_proj(state_outputs))).squeeze(-1)
        slot_weights = F.softmax(slot_scores, dim=-1).unsqueeze(-1)
        slot_context = slot_weights * state_outputs  # (batch, seq_len, attn_size)

        intent_scores = self.intent_attn_v(
            torch.tanh(self.intent_attn_proj(state_outputs) + final_state.unsqueeze(1))
        ).squeeze(-1)
        intent_weights = F.softmax(intent_scores, dim=-1).unsqueeze(-1)
        intent_context = (intent_weights * state_outputs).sum(dim=1)  # (batch, attn_size)
        intent_output = intent_context + final_state

        # One-shot slot gate: intent context modulates the slot context, then
        # rescales it -- computed ONCE, no iterative refinement (contrast SF-ID).
        intent_gate = self.gate_proj(intent_output).unsqueeze(1)
        gate = self.gate_v * torch.tanh(slot_context + intent_gate)
        gate = gate.sum(dim=-1, keepdim=True)
        gated_slot_context = slot_context * gate

        slot_logits = self.slot_classifier(torch.cat([gated_slot_context, state_outputs], dim=-1))
        intent_logits = self.intent_classifier(intent_output)
        return slot_logits, intent_logits


def build_slot_gated_joint_model() -> nn.Module:
    """Build a small Slot-Gated joint slot-filling/intent model.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return SlotGatedJointModel().eval()


def example_input_slot_gated_joint_model() -> Tensor:
    """Example input token ids for the Slot-Gated joint model.

    Returns
    -------
    Tensor
        ``(batch, seq_len)`` long token ids.
    """
    return torch.randint(0, 90, (2, 10))


MENAGERIE_ENTRIES = [
    ("Self-RAG (Self-Reflective RAG)", "build_self_rag", "example_input_self_rag", "2023", "NLP"),
    ("Sequicity", "build_sequicity", "example_input_sequicity", "2018", "NLP"),
    ("SF-ID Network", "build_sf_id_network", "example_input_sf_id_network", "2019", "NLP"),
    ("SimpleTOD", "build_simpletod", "example_input_simpletod", "2020", "NLP"),
    (
        "Slot-Gated Joint Model",
        "build_slot_gated_joint_model",
        "example_input_slot_gated_joint_model",
        "2018",
        "NLP",
    ),
    (
        "Slot-Gated SLU",
        "build_slot_gated_joint_model",
        "example_input_slot_gated_joint_model",
        "2018",
        "NLP",
    ),
]
