"""Compact faithful reimplementations of five dialogue/NLU architecture families.

Sources checked (paper + official source; reimplemented compactly from scratch, no
clone/pip-install except the already-installed ``transformers`` library used for the
BERT/GPT-2 backbones):
  - SlotRefine: github.com/moore3930/SlotRefine (Wu, Ding, Lu & Xie, EMNLP 2020),
    aclanthology.org/2020.emnlp-main.152; official repo is TensorFlow/THUMT-based
    (``models.py`` ``NatSLU.create_model``). Its distinctive mechanism is a
    NON-AUTOREGRESSIVE joint intent+slot Transformer ENCODER (no decoder, no
    autoregressive label generation): token embeddings are summed with the PREVIOUS
    slot-TAG embeddings (self-conditioning on the model's own last-round predictions,
    the way BERT-style masked-LM unmasking is self-conditioned) and a prepended [CLS]
    token before one shared Transformer-encoder stack; the [CLS] output goes through an
    intent-classification head and the remaining token outputs go through a per-token
    slot-classification head that is additionally conditioned on the pooled intent
    state. Reimplemented here as a two-pass loop: pass 1 predicts slot tags from a
    "no tag" placeholder embedding; pass 2 re-embeds those predicted tags and reruns
    the SAME encoder (single shared weights, matching the paper's *iterative refinement*
    idea of coordinating uncoordinated slot-chunk boundaries across an update).
  - SOLOIST: github.com/pengbaolin/soloist (Peng, Li, Li, Shayandeh, Liden & Gao, TACL
    2021), arxiv:2005.05298; official repo (``soloist/soloist_train.py``) fine-tunes
    ``transformers.GPT2DoubleHeadsModel`` -- a single GPT-2 causal decoder subsuming
    belief-state tracking, DB/KB grounding, and response generation into ONE flat
    (history, belief, KB, response) token stream, with a SECOND "multiple-choice" head
    reading the hidden state at a designated ``mc_token_id`` position (the end-of-belief
    marker) to score whether the sampled (belief, response) pair is a real
    machine-taught positive or a negative/mismatched contrastive example -- this
    contrastive matching head trained jointly with the LM loss is the distinctive
    "grounded + machine-teaching" mechanism (vs. a plain GPT-2 dialogue LM). Built
    directly from the installed ``transformers`` ``GPT2DoubleHeadsModel`` with a tiny
    config, exactly mirroring the official training call's ``(lm_labels, mc_labels,
    mc_token_ids)`` contract.
  - SOM-DST: github.com/clovaai/som-dst (Kim, Yang, Kim & Lee, ACL 2020),
    arxiv:1911.03906; official repo (``model.py``, classes ``SomDST``/``Encoder``/
    ``Decoder``). Distinctive mechanism is "Selectively Overwriting Memory": a BERT
    encoder produces per-slot state-operation logits (delete/update/dontcare/carryover)
    at the ``[SLOT]`` token positions gathered from the dialogue-plus-previous-state
    input; only the (small) subset of slots the encoder predicts as ``update`` are
    routed to a GRU pointer-generator DECODER that copies from the dialogue history OR
    generates from the vocabulary (soft-gated by ``p_gen``) to produce the new slot
    value -- the rest of the belief state is CARRIED OVER unchanged rather than
    regenerated from scratch every turn (the "selective overwriting" that gives the
    paper its name and its efficiency win over full-state DST). Reimplemented compactly
    with the same encoder state-operation classifier + gated copy-or-generate GRU
    decoder, restricted to a fixed (trace-static) number of update slots and decode
    steps so the control flow is trace-friendly.
  - Stack-Propagation (a.k.a. Stack-Propagation SLU): github.com/LeePleased/
    StackPropagation-SLU (Qin, Che, Li, Ni & Liu, EMNLP 2019), aclanthology.org/
    D19-1214; official repo (``utils/module.py``, ``ModelManager``/``LSTMEncoder``/
    ``LSTMDecoder``). Distinctive mechanism is TOKEN-LEVEL INTENT DETECTION feeding
    slot filling via explicit "stack propagation": a shared BiLSTM+self-attention
    encoder produces per-token features; an intent LSTM decoder emits a PER-TOKEN
    intent distribution; each per-token intent distribution (soft, not hard-argmax, in
    the paper's ``differentiable=True`` regime) is then concatenated as EXTRA INPUT
    into a second LSTM decoder that fills slots -- i.e. slot filling is directly
    conditioned, token by token, on the intent decoder's own output, rather than the
    two tasks being independent multi-task heads over a shared encoder. Reimplemented
    with the same encoder->intent-decoder->(concat)->slot-decoder stack-propagation
    data flow. NOTE: cand_00135 "Stack-Propagation SLU" is a verified duplicate of this
    same model/repo (queue notes: "both entries refer to same model") and is skipped
    below rather than re-emitted under a second label with no architectural delta.
  - STAR for DST (Slot Self-Attentive Dialogue State Tracking): github.com/smartyfh/
    DST-STAR (Ye, Manotumruksa, Zhang, Li & Yilmaz, WWW 2021), arxiv:2101.09374;
    official repo (``models/ModelBERT.py``, classes ``BeliefTracker``/``Decoder``/
    ``SlotSelfAttention``). Distinctive mechanism is a two-stage SLOT SELF-ATTENTION:
    (1) a slot-token attention where each slot's embedding attends over the BERT-encoded
    dialogue context to pull out slot-specific features (concatenated with the raw slot
    embedding, then MLP-projected); (2) a STACKED multi-head self-attention over the
    SLOT AXIS itself (one "token" per slot, attending slot-to-slot) that lets the model
    learn inter-slot correlations the paper argues prior per-slot-independent DST
    trackers miss (e.g. "hotel-area" correlating with "restaurant-area"); the resulting
    contextualized slot vectors are matched against candidate slot-value embeddings by
    cosine distance for classification. Reimplemented with the same BERT encoder ->
    slot-utterance cross-attention -> MLP -> stacked slot self-attention -> distance-
    based value-matching pipeline.

No further skips this batch beyond the verified Stack-Propagation/Stack-Propagation SLU
duplicate documented above -- all five built models are well-documented named mechanisms
with an official reference implementation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import BertConfig, BertModel, GPT2Config, GPT2DoubleHeadsModel


# ---------------------------------------------------------------------------
# SlotRefine: non-autoregressive joint intent+slot Transformer encoder with a
# two-pass ITERATIVE-REFINEMENT loop -- pass 2 re-embeds pass 1's predicted
# slot tags and reruns the SAME shared encoder to coordinate slot-chunk labels.
# ---------------------------------------------------------------------------
class SlotRefine(nn.Module):
    """SlotRefine (Wu et al., EMNLP 2020): non-autoregressive joint intent/slot NLU.

    Parameters
    ----------
    vocab_size : int
        Input word-token vocabulary size.
    slot_size : int
        Number of slot-tag classes (including a "no tag yet" placeholder id 0).
    intent_size : int
        Number of intent classes.
    hidden_size : int
        Shared Transformer hidden size.
    n_layers : int
        Number of Transformer encoder layers.
    n_heads : int
        Number of self-attention heads.
    """

    def __init__(
        self,
        vocab_size: int = 120,
        slot_size: int = 20,
        intent_size: int = 12,
        hidden_size: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, hidden_size)
        # Slot-TAG embedding: encodes the model's OWN previous-round slot-tag
        # prediction back into the input stream (self-conditioning), the
        # mechanism that lets a second pass refine uncoordinated tag chunks.
        self.tag_embedding = nn.Embedding(slot_size, hidden_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=n_heads, dim_feedforward=4 * hidden_size, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.intent_head = nn.Linear(hidden_size, intent_size)
        self.slot_head = nn.Linear(hidden_size + intent_size, slot_size)

    def _encode_pass(self, input_ids: Tensor, tag_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Run one non-autoregressive encoder pass and return (intent, slot) logits."""
        tok = self.word_embedding(input_ids)
        tag = self.tag_embedding(tag_ids)
        seq = tok + tag
        cls = self.cls_token.expand(seq.size(0), -1, -1)
        seq = torch.cat([cls, seq], dim=1)
        hidden = self.encoder(seq)
        intent_state, slot_state = hidden[:, 0], hidden[:, 1:]
        intent_logits = self.intent_head(intent_state)
        intent_ctx = intent_logits.unsqueeze(1).expand(-1, slot_state.size(1), -1)
        slot_logits = self.slot_head(torch.cat([slot_state, intent_ctx], dim=-1))
        return intent_logits, slot_logits

    def forward(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Run SlotRefine's two-pass non-autoregressive refinement.

        Parameters
        ----------
        input_ids : Tensor
            ``(batch, seq_len)`` long word-token ids.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(intent_logits, slot_logits)`` of shapes ``(batch, intent_size)`` and
            ``(batch, seq_len, slot_size)`` after the second (refinement) pass.
        """
        bsz, seq_len = input_ids.shape
        # Pass 1: no prior tag information (placeholder id 0 everywhere).
        placeholder_tags = torch.zeros(bsz, seq_len, dtype=torch.long, device=input_ids.device)
        _, slot_logits_1 = self._encode_pass(input_ids, placeholder_tags)
        pass1_tags = slot_logits_1.argmax(dim=-1)
        # Pass 2: refine by self-conditioning on pass 1's own slot predictions,
        # coordinating chunk boundaries across the full non-autoregressive tag set.
        intent_logits_2, slot_logits_2 = self._encode_pass(input_ids, pass1_tags)
        return intent_logits_2, slot_logits_2


def build_slotrefine() -> nn.Module:
    """Build a small SlotRefine non-autoregressive joint NLU model.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return SlotRefine().eval()


def example_input_slotrefine() -> Tensor:
    """Example word-token ids for SlotRefine.

    Returns
    -------
    Tensor
        ``(batch, seq_len)`` long token ids.
    """
    return torch.randint(0, 120, (2, 10))


# ---------------------------------------------------------------------------
# SOLOIST: GPT-2 causal decoder over one flat (history, belief, KB, response)
# stream with a SECOND multiple-choice head (GPT2DoubleHeadsModel) scoring
# belief/response consistency at a designated end-of-belief token position.
# ---------------------------------------------------------------------------
class Soloist(nn.Module):
    """SOLOIST (Peng et al., TACL 2021): single GPT-2 stack for grounded TOD.

    Parameters
    ----------
    vocab_size : int
        Shared token vocabulary size.
    n_embd : int
        GPT-2 hidden size.
    n_layer : int
        Number of GPT-2 decoder blocks.
    n_head : int
        Number of attention heads.
    n_positions : int
        Maximum sequence length.
    """

    def __init__(
        self,
        vocab_size: int = 150,
        n_embd: int = 32,
        n_layer: int = 2,
        n_head: int = 2,
        n_positions: int = 64,
    ) -> None:
        super().__init__()
        cfg = GPT2Config(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            num_labels=1,
        )
        # GPT2DoubleHeadsModel already IS the SOLOIST parameterization: one
        # shared causal stack, one LM head over the flat (history, belief, KB,
        # response) stream, and one multiple-choice ("mc") contrastive head
        # reading the hidden state at `mc_token_ids` -- the machine-teaching
        # signal that scores real vs. negative-sampled (belief, response) pairs.
        self.gpt2_double_head = GPT2DoubleHeadsModel(cfg)

    def forward(self, input_ids: Tensor, mc_token_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Compute LM logits and belief/response matching logits for SOLOIST.

        Parameters
        ----------
        input_ids : Tensor
            ``(batch, num_choices, seq_len)`` long token ids: the flat
            (history, belief, KB, response) stream, with ``num_choices``
            candidate continuations (1 real + negatives) per example.
        mc_token_ids : Tensor
            ``(batch, num_choices)`` long positions of the end-of-belief marker
            token whose hidden state feeds the matching head.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(lm_logits, mc_logits)``: next-token logits
            ``(batch, num_choices, seq_len, vocab_size)`` and matching-head
            logits ``(batch, num_choices)``.
        """
        out = self.gpt2_double_head(input_ids=input_ids, mc_token_ids=mc_token_ids)
        return out.logits, out.mc_logits


def build_soloist() -> nn.Module:
    """Build a small SOLOIST grounded-dialogue model.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return Soloist().eval()


def example_input_soloist() -> tuple[Tensor, Tensor]:
    """Example multi-choice token/marker-position ids for SOLOIST.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(input_ids, mc_token_ids)``.
    """
    batch, num_choices, seq_len = 2, 2, 16
    input_ids = torch.randint(0, 150, (batch, num_choices, seq_len))
    mc_token_ids = torch.full((batch, num_choices), seq_len - 1, dtype=torch.long)
    return input_ids, mc_token_ids


# ---------------------------------------------------------------------------
# SOM-DST: BERT encoder predicts per-slot state OPERATIONS (delete/update/
# dontcare/carryover); only slots operated as "update" are routed to a GRU
# pointer-generator decoder that copies from context or generates from vocab.
# ---------------------------------------------------------------------------
class SomDst(nn.Module):
    """SOM-DST (Kim et al., ACL 2020): selectively-overwriting-memory DST.

    Parameters
    ----------
    vocab_size : int
        Shared dialogue-token / value vocabulary size.
    hidden_size : int
        BERT encoder hidden size (also the GRU decoder hidden size).
    n_layers : int
        Number of BERT encoder layers.
    n_heads : int
        Number of BERT attention heads.
    n_ops : int
        Number of state-operation classes (delete/update/dontcare/carryover = 4).
    n_slots : int
        Number of tracked dialogue-state slots (one ``[SLOT]`` marker each).
    max_update : int
        Fixed (trace-static) number of slot-update decode "lanes" the GRU
        decoder fills in parallel, matching the paper's per-turn update budget.
    max_len : int
        Fixed number of value-token decode steps per updated slot.
    """

    def __init__(
        self,
        vocab_size: int = 100,
        hidden_size: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        n_ops: int = 4,
        n_slots: int = 5,
        max_update: int = 3,
        max_len: int = 4,
    ) -> None:
        super().__init__()
        bert_cfg = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            intermediate_size=4 * hidden_size,
            max_position_embeddings=64,
        )
        self.bert = BertModel(bert_cfg)
        self.op_head = nn.Linear(hidden_size, n_ops)
        self.n_slots = n_slots
        self.max_update = max_update
        self.max_len = max_len
        self.hidden_size = hidden_size

        self.gru = nn.GRUCell(hidden_size, hidden_size)
        # Gated copy-or-generate pointer-generator head: p_gen blends a
        # vocabulary-generation distribution with a copy-from-dialogue-history
        # distribution, the mechanism that lets SOM-DST fill slot VALUES by
        # either pointing back at context tokens or generating novel ones.
        self.gen_proj = nn.Linear(hidden_size, vocab_size)
        self.p_gen = nn.Linear(hidden_size * 3, 1)

    def forward(self, input_ids: Tensor, slot_positions: Tensor) -> tuple[Tensor, Tensor]:
        """Run SOM-DST: predict per-slot operations, then decode updated values.

        Parameters
        ----------
        input_ids : Tensor
            ``(batch, seq_len)`` long dialogue-plus-previous-state token ids.
        slot_positions : Tensor
            ``(batch, n_slots)`` long positions of each ``[SLOT]`` marker token
            in ``input_ids``.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(op_logits, gen_logits)``: per-slot operation logits
            ``(batch, n_slots, n_ops)`` and per-(update-lane, step) generation
            distributions ``(batch, max_update, max_len, vocab_size)`` for the
            fixed decode budget.
        """
        sequence_output = self.bert(input_ids).last_hidden_state
        pos = slot_positions.unsqueeze(-1).expand(-1, -1, sequence_output.size(-1))
        slot_states = torch.gather(sequence_output, 1, pos)  # (batch, n_slots, hidden)
        op_logits = self.op_head(slot_states)  # (batch, n_slots, n_ops)

        # Fixed-budget decode "lanes": the first `max_update` slot states seed
        # the GRU pointer-generator, mirroring the paper's SELECTIVE decoding
        # of only the slots the operation classifier marks for an update
        # (kept trace-static here instead of the official data-dependent
        # gather/mask, which is not compatible with static graph capture).
        lane_seed = slot_states[:, : self.max_update]
        bsz = input_ids.size(0)
        hidden = lane_seed.reshape(bsz * self.max_update, self.hidden_size)
        w = hidden
        gen_steps = []
        for _ in range(self.max_len):
            hidden = self.gru(w, hidden)
            attn = torch.bmm(
                sequence_output.repeat_interleave(self.max_update, dim=0),
                hidden.unsqueeze(-1),
            ).squeeze(-1)
            attn = F.softmax(attn, dim=-1)
            context = torch.bmm(
                attn.unsqueeze(1), sequence_output.repeat_interleave(self.max_update, dim=0)
            )
            context = context.squeeze(1)
            gate = torch.sigmoid(self.p_gen(torch.cat([w, hidden, context], dim=-1)))
            vocab_dist = F.softmax(self.gen_proj(hidden), dim=-1)
            copy_dist = torch.zeros_like(vocab_dist).scatter_add_(
                1,
                input_ids.repeat_interleave(self.max_update, dim=0),
                attn,
            )
            final_dist = gate * vocab_dist + (1 - gate) * copy_dist
            gen_steps.append(final_dist)
            w = hidden
        gen_logits = torch.stack(gen_steps, dim=1).reshape(bsz, self.max_update, self.max_len, -1)
        return op_logits, gen_logits


def build_som_dst() -> nn.Module:
    """Build a small SOM-DST selective-overwriting DST model.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return SomDst().eval()


def example_input_som_dst() -> tuple[Tensor, Tensor]:
    """Example dialogue-state token ids and slot-marker positions for SOM-DST.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(input_ids, slot_positions)``.
    """
    input_ids = torch.randint(0, 100, (2, 24))
    slot_positions = torch.tensor([[1, 5, 9, 13, 17], [1, 5, 9, 13, 17]], dtype=torch.long)
    return input_ids, slot_positions


# ---------------------------------------------------------------------------
# Stack-Propagation: shared BiLSTM+self-attention encoder -> intent LSTM
# decoder -> its per-token intent distribution is fed FORWARD as extra input
# into a second LSTM decoder that fills slots (token-level stack propagation).
# ---------------------------------------------------------------------------
class StackPropagation(nn.Module):
    """Stack-Propagation SLU (Qin et al., EMNLP 2019): intent-to-slot stack propagation.

    Parameters
    ----------
    vocab_size : int
        Input word-token vocabulary size.
    embed_dim : int
        Word embedding dimensionality.
    encoder_hidden : int
        BiLSTM encoder hidden size (per direction).
    attn_hidden : int
        Self-attention hidden/output dimensionality.
    intent_size : int
        Number of (per-token) intent classes.
    slot_size : int
        Number of slot-tag classes.
    decoder_hidden : int
        Hidden size of the intent and slot LSTM decoders.
    """

    def __init__(
        self,
        vocab_size: int = 110,
        embed_dim: int = 24,
        encoder_hidden: int = 32,
        attn_hidden: int = 24,
        intent_size: int = 8,
        slot_size: int = 18,
        decoder_hidden: int = 32,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.bilstm = nn.LSTM(embed_dim, encoder_hidden, batch_first=True, bidirectional=True)
        self.self_attn_qkv = nn.Linear(embed_dim, 3 * attn_hidden)
        self.self_attn_out = nn.Linear(attn_hidden, attn_hidden)

        encoded_dim = 2 * encoder_hidden + attn_hidden
        self.intent_decoder = nn.LSTM(encoded_dim, decoder_hidden, batch_first=True)
        self.intent_out = nn.Linear(decoder_hidden, intent_size)

        # Slot decoder input includes the encoded features AND the intent
        # decoder's own per-token output distribution -- the "stack
        # propagation" of token-level intent detection directly into slot
        # filling, rather than two independent multi-task heads.
        self.slot_decoder = nn.LSTM(encoded_dim + intent_size, decoder_hidden, batch_first=True)
        self.slot_out = nn.Linear(decoder_hidden, slot_size)

    def forward(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Run Stack-Propagation SLU: encode, decode intent, then propagate into slots.

        Parameters
        ----------
        input_ids : Tensor
            ``(batch, seq_len)`` long word-token ids.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(slot_logits, intent_logits)`` of shapes ``(batch, seq_len, slot_size)``
            and ``(batch, seq_len, intent_size)`` (per-token intent, as in the paper).
        """
        embedded = self.embedding(input_ids)
        lstm_hiddens, _ = self.bilstm(embedded)

        qkv = self.self_attn_qkv(embedded)
        q, k, v = qkv.chunk(3, dim=-1)
        scores = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / (q.size(-1) ** 0.5), dim=-1)
        attn_hiddens = self.self_attn_out(torch.bmm(scores, v))

        encoded = torch.cat([lstm_hiddens, attn_hiddens], dim=-1)

        intent_hiddens, _ = self.intent_decoder(encoded)
        intent_logits = self.intent_out(intent_hiddens)

        # Feed the intent decoder's own (soft) per-token output distribution
        # forward as extra slot-decoder input: the stack-propagation link.
        slot_input = torch.cat([encoded, intent_logits], dim=-1)
        slot_hiddens, _ = self.slot_decoder(slot_input)
        slot_logits = self.slot_out(slot_hiddens)

        return slot_logits, intent_logits


def build_stack_propagation() -> nn.Module:
    """Build a small Stack-Propagation SLU model.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return StackPropagation().eval()


def example_input_stack_propagation() -> Tensor:
    """Example word-token ids for Stack-Propagation SLU.

    Returns
    -------
    Tensor
        ``(batch, seq_len)`` long token ids.
    """
    return torch.randint(0, 110, (2, 11))


# ---------------------------------------------------------------------------
# STAR for DST: BERT encoder -> per-slot utterance cross-attention -> MLP ->
# STACKED slot self-attention (attention across the SLOT axis, learning
# inter-slot correlations) -> cosine-distance slot-value matching.
# ---------------------------------------------------------------------------
class StarDst(nn.Module):
    """STAR (Ye et al., WWW 2021): slot self-attentive dialogue state tracking.

    Parameters
    ----------
    vocab_size : int
        Dialogue-context token vocabulary size.
    hidden_size : int
        BERT encoder hidden size (also the slot-representation width).
    n_bert_layers : int
        Number of BERT encoder layers.
    n_slots : int
        Number of tracked dialogue-state slots.
    n_values_per_slot : int
        Number of candidate value embeddings scored per slot.
    n_self_attn_layers : int
        Number of stacked slot-self-attention layers (the paper's core
        inter-slot-correlation mechanism).
    n_heads : int
        Number of attention heads (both utterance- and slot-self-attention).
    """

    def __init__(
        self,
        vocab_size: int = 90,
        hidden_size: int = 32,
        n_bert_layers: int = 2,
        n_slots: int = 6,
        n_values_per_slot: int = 5,
        n_self_attn_layers: int = 2,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        bert_cfg = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=n_bert_layers,
            num_attention_heads=n_heads,
            intermediate_size=4 * hidden_size,
            max_position_embeddings=64,
        )
        self.bert = BertModel(bert_cfg)
        self.slot_embedding = nn.Parameter(torch.randn(n_slots, hidden_size) * 0.02)
        self.value_embedding = nn.Parameter(
            torch.randn(n_slots, n_values_per_slot, hidden_size) * 0.02
        )

        # Slot-token attention: each slot embedding queries the BERT-encoded
        # dialogue context for slot-specific evidence.
        self.slot_utter_attn = nn.MultiheadAttention(hidden_size, n_heads, batch_first=True)
        self.slot_mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Stacked SLOT self-attention -- one "token" per slot, attention runs
        # across the slot axis so slots can inform each other (e.g.
        # "hotel-area" correlating with "restaurant-area"), the paper's
        # distinctive departure from per-slot-independent DST trackers.
        slot_self_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=n_heads, dim_feedforward=hidden_size, batch_first=True
        )
        self.slot_self_attn = nn.TransformerEncoder(slot_self_layer, num_layers=n_self_attn_layers)

        self.pred_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size)
        )

    def forward(self, input_ids: Tensor) -> Tensor:
        """Run STAR: encode context, attend per slot, self-attend across slots, match values.

        Parameters
        ----------
        input_ids : Tensor
            ``(batch, seq_len)`` long dialogue-context token ids.

        Returns
        -------
        Tensor
            ``(batch, n_slots, n_values_per_slot)`` cosine-similarity scores of
            each slot's contextualized representation against its candidate
            value embeddings.
        """
        bsz = input_ids.size(0)
        sequence_output = self.bert(input_ids).last_hidden_state

        slot_query = self.slot_embedding.unsqueeze(0).expand(bsz, -1, -1)
        slot_utter_emb, _ = self.slot_utter_attn(slot_query, sequence_output, sequence_output)

        slot_utter_concat = torch.cat([slot_query, slot_utter_emb], dim=-1)
        slot_features = self.slot_mlp(slot_utter_concat)  # (batch, n_slots, hidden)

        # Stacked self-attention across the slot axis -- the mechanism that
        # sets STAR apart from prior per-slot-independent trackers.
        slot_hidden = self.slot_self_attn(slot_features)
        slot_repr = self.pred_head(slot_hidden)  # (batch, n_slots, hidden)

        value_emb = self.value_embedding.unsqueeze(0).expand(
            bsz, -1, -1, -1
        )  # (batch, n_slots, n_val, hidden)
        scores = F.cosine_similarity(slot_repr.unsqueeze(2), value_emb, dim=-1)
        return scores


def build_star_dst() -> nn.Module:
    """Build a small STAR slot-self-attentive DST model.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return StarDst().eval()


def example_input_star_dst() -> Tensor:
    """Example dialogue-context token ids for STAR-DST.

    Returns
    -------
    Tensor
        ``(batch, seq_len)`` long token ids.
    """
    return torch.randint(0, 90, (2, 20))


MENAGERIE_ENTRIES = [
    ("SlotRefine", "build_slotrefine", "example_input_slotrefine", "2020", "NLP"),
    ("SOLOIST", "build_soloist", "example_input_soloist", "2021", "NLP"),
    ("SOM-DST", "build_som_dst", "example_input_som_dst", "2020", "NLP"),
    (
        "Stack-Propagation",
        "build_stack_propagation",
        "example_input_stack_propagation",
        "2019",
        "NLP",
    ),
    ("STAR for DST", "build_star_dst", "example_input_star_dst", "2021", "NLP"),
]
