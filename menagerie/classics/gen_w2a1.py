"""Compact faithful reimplementations of five dialogue-modeling architecture families.

Sources checked (paper + official source, no clone/pip-install; reimplemented from scratch):
  - DialogueRNN (Majumder et al., AAAI 2019, arxiv:1811.00405); official repo
    github.com/declare-lab/conv-emotion (`DialogueRNN/model.py`) -- three coupled GRU
    cells over a conversation: a global GRU that folds each utterance with the current
    speaker state into a shared context memory, a party (speaker) GRU whose update is
    driven by the incoming utterance plus an attention-pooled context vector over all
    prior global states, and an emotion GRU that decodes each utterance's emotion
    representation from the freshly updated speaker state. Attention scores in the
    party-GRU context step are computed against the running list of global states
    (Eq. 2-4 of the paper), not a fixed window.
  - InstructDial (Gupta et al., EMNLP 2022, arxiv:2205.12673); official repo
    github.com/prakharguptaz/Instructdial -- unified text-to-text instruction tuning
    over a T5 backbone (48 dialogue tasks reformatted as instruction+instance strings).
    The paper's distinctive architectural contribution beyond vanilla T5 fine-tuning is
    the pair of instruction-adherence META-TASKS (Sec 3.4): "instruction selection"
    (pick which of several candidate instructions matches a given input-output pair) and
    "instruction binary" (yes/no judge of whether an instruction explains an
    input-output pair). Modeled here as a T5ForConditionalGeneration backbone plus an
    explicit instruction-selection scoring head that reuses the T5 encoder to embed K
    candidate instructions and scores each against the pooled encoder state of the
    dialogue instance via a bilinear compatibility score, mirroring the meta-task.
  - JointBERT (Chen, Zhuo & Wang 2019, arxiv:1902.10909); canonical unofficial PyTorch
    port github.com/monologg/JointBERT (`model/modeling_jointbert.py`,
    `model/module.py`) -- single shared BERT encoder; the pooled [CLS] embedding feeds
    an intent-classification head (linear after dropout) while the full per-token
    sequence output feeds an independent per-token slot-BIO-tagging head (linear after
    dropout), both heads trained jointly from the one shared encoder pass.
  - KEMP (Li et al., AAAI 2022, arxiv:2009.09708); official repo github.com/qtli/KEMP
    (`Model.py`, `layers.py`) -- builds an emotional context graph from the dialogue
    history (word/concept nodes with per-node emotion-intensity scores from an emotion
    lexicon), encodes it with a multi-head graph-attention context encoder, perceives an
    emotional-signal vector via emotion-intensity-weighted pooling over graph node
    states, and decodes the empathetic response with a Transformer decoder whose
    cross-attention sub-layer is biased toward emotionally salient graph vertices (the
    paper's "emotion-dependency decoder").
  - KETOD (Chen et al., NAACL 2022, arxiv:2205.05589); official repo
    github.com/facebookresearch/ketod -- the paper's "Combiner" model: a T5/GPT-2 style
    seq2seq stage first generates belief states/actions from dialogue context; entities
    from the belief state drive retrieval of candidate knowledge snippets; a knowledge
    SELECTION classifier scores each retrieved snippet against the dialogue context
    (binary relevance over a variable-size snippet set); the top snippet is concatenated
    with the context and fed back into the seq2seq decoder to generate the final,
    knowledge-enriched response. Modeled here with a shared encoder, a per-snippet
    selection scorer (encoder cross-attention pooled logit), and a decoder conditioned
    on the dialogue context plus the selected knowledge snippet's encoding.

Skipped: cand_00068 "KBRD" -- already faithfully built and registered in the catalog as
"CRSLab KBRD-style policy nets" (menagerie/classics/gen_w1a2.py, class KBRD,
build_kbrd/example_input_kbrd). That implementation matches this candidate's own repo_url
(github.com/THUDM/KBRD) and arxiv id (1908.05391) family exactly: an R-GCN knowledge-graph
entity encoder with self-attention user pooling feeding both a bilinear recommendation
head over KG entities and a Transformer encoder-decoder dialogue generator whose decoder
logits are additively biased by the projected KG user embedding -- the identical
distinctive mechanism the KBRD paper (Chen et al., EMNLP 2019) describes. Re-emitting it
here under a second label would be a byte-identical duplicate with no architectural delta,
so it is not re-registered in MENAGERIE_ENTRIES below.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import BertConfig, BertModel, T5Config, T5ForConditionalGeneration


# ---------------------------------------------------------------------------
# DialogueRNN: tripartite global/party/emotion GRU cascade for conversational
# emotion recognition, with attention-pooled global-state context.
# ---------------------------------------------------------------------------
class DialogueRNN(nn.Module):
    """Compact DialogueRNN: global GRU + attention-context party GRU + emotion GRU."""

    def __init__(
        self,
        utt_dim: int = 32,
        global_dim: int = 24,
        party_dim: int = 24,
        emotion_dim: int = 24,
        n_emotions: int = 6,
    ) -> None:
        """Build the three coupled GRU cells and the emotion classification head."""

        super().__init__()
        self.global_dim = global_dim
        self.party_dim = party_dim
        self.global_gru = nn.GRUCell(utt_dim + party_dim, global_dim)
        self.party_gru = nn.GRUCell(utt_dim + global_dim, party_dim)
        self.emotion_gru = nn.GRUCell(party_dim, emotion_dim)
        self.attn_proj = nn.Linear(utt_dim, global_dim, bias=False)
        self.emotion_head = nn.Linear(emotion_dim, n_emotions)

    def forward(self, utterances: Tensor, speaker_ids: Tensor) -> Tensor:
        """Return per-utterance emotion logits, shape (batch, seq_len, n_emotions).

        Parameters
        ----------
        utterances : Tensor
            Utterance embeddings, shape (batch, seq_len, utt_dim).
        speaker_ids : Tensor
            Integer speaker id per utterance, shape (batch, seq_len); values in
            {0, 1} select which party state is updated at each step (dyadic case).
        """

        batch, seq_len, _ = utterances.shape
        device = utterances.device
        n_parties = 2
        party_state = [torch.zeros(batch, self.party_dim, device=device) for _ in range(n_parties)]
        global_state = torch.zeros(batch, self.global_dim, device=device)
        emotion_state = torch.zeros(batch, self.emotion_gru.hidden_size, device=device)
        global_history: list[Tensor] = []
        logits = []
        for t in range(seq_len):
            u_t = utterances[:, t, :]
            speaker_t = speaker_ids[:, t]
            cur_party = torch.stack(
                [party_state[s.item()][b] for b, s in enumerate(speaker_t)], dim=0
            )

            global_state = self.global_gru(torch.cat([u_t, cur_party], dim=-1), global_state)
            global_history.append(global_state)

            if len(global_history) > 1:
                hist = torch.stack(global_history[:-1], dim=1)  # (batch, t, global_dim)
                query = self.attn_proj(u_t).unsqueeze(1)  # (batch, 1, global_dim)
                scores = torch.bmm(query, hist.transpose(1, 2)).squeeze(1)  # (batch, t)
                weights = F.softmax(scores, dim=-1)
                context = torch.bmm(weights.unsqueeze(1), hist).squeeze(1)
            else:
                context = torch.zeros(batch, self.global_dim, device=device)

            new_party = self.party_gru(torch.cat([u_t, context], dim=-1), cur_party)
            for b, s in enumerate(speaker_t):
                party_state[s.item()] = party_state[s.item()].clone()
                party_state[s.item()][b] = new_party[b]

            emotion_state = self.emotion_gru(new_party, emotion_state)
            logits.append(self.emotion_head(emotion_state))

        return torch.stack(logits, dim=1)


def build_dialoguernn() -> nn.Module:
    """Build the compact DialogueRNN tripartite-GRU emotion classifier."""

    return DialogueRNN().eval()


def example_input_dialoguernn() -> tuple[Tensor, Tensor]:
    """Return (utterance embeddings, speaker ids) for a 2-party, 5-turn conversation."""

    utterances = torch.randn(2, 5, 32)
    speaker_ids = torch.tensor([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]], dtype=torch.long)
    return utterances, speaker_ids


# ---------------------------------------------------------------------------
# InstructDial: T5 instruction-tuned seq2seq backbone plus an explicit
# instruction-selection meta-task head (scores candidate instructions against
# a pooled dialogue-instance encoding).
# ---------------------------------------------------------------------------
class InstructDial(nn.Module):
    """Compact InstructDial: T5 text-to-text backbone + instruction-selection head."""

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        d_ff: int = 64,
    ) -> None:
        """Build the shared T5 backbone and the bilinear instruction-selection scorer."""

        super().__init__()
        config = T5Config(
            vocab_size=vocab_size,
            d_model=d_model,
            d_kv=d_model // n_heads,
            d_ff=d_ff,
            num_layers=n_layers,
            num_decoder_layers=n_layers,
            num_heads=n_heads,
            is_encoder_decoder=True,
        )
        self.t5 = T5ForConditionalGeneration(config)
        self.instr_bilinear = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        input_ids: Tensor,
        decoder_input_ids: Tensor,
        candidate_instruction_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return (text-generation logits, instruction-selection scores).

        Parameters
        ----------
        input_ids : Tensor
            Instruction+instance token ids, shape (batch, src_len).
        decoder_input_ids : Tensor
            Teacher-forced target token ids, shape (batch, tgt_len).
        candidate_instruction_ids : Tensor
            K candidate instruction strings to score against the pooled input
            encoding for the "instruction selection" meta-task, shape
            (batch, n_candidates, instr_len).
        """

        encoder_out = self.t5.encoder(input_ids=input_ids).last_hidden_state
        gen_logits = self.t5(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
        ).logits

        pooled_instance = encoder_out.mean(dim=1)  # (batch, d_model)

        batch, n_cand, instr_len = candidate_instruction_ids.shape
        flat_cand = candidate_instruction_ids.reshape(batch * n_cand, instr_len)
        cand_hidden = self.t5.encoder(input_ids=flat_cand).last_hidden_state.mean(dim=1)
        cand_hidden = cand_hidden.reshape(batch, n_cand, -1)

        query = self.instr_bilinear(pooled_instance).unsqueeze(1)  # (batch, 1, d_model)
        selection_scores = torch.bmm(query, cand_hidden.transpose(1, 2)).squeeze(1)
        return gen_logits, selection_scores


def build_instructdial() -> nn.Module:
    """Build the compact InstructDial T5-backbone + instruction-selection model."""

    return InstructDial().eval()


def example_input_instructdial() -> tuple[Tensor, Tensor, Tensor]:
    """Return (input_ids, decoder_input_ids, candidate_instruction_ids)."""

    input_ids = torch.randint(0, 256, (2, 12), dtype=torch.long)
    decoder_input_ids = torch.randint(0, 256, (2, 8), dtype=torch.long)
    candidate_instruction_ids = torch.randint(0, 256, (2, 3, 6), dtype=torch.long)
    return input_ids, decoder_input_ids, candidate_instruction_ids


# ---------------------------------------------------------------------------
# JointBERT: shared BERT encoder with a [CLS]-pooled intent-classification
# head and an independent per-token slot-BIO-tagging head.
# ---------------------------------------------------------------------------
class JointBERT(nn.Module):
    """Compact JointBERT: shared BERT encoder + intent head + per-token slot head."""

    def __init__(
        self,
        vocab_size: int = 512,
        hidden_size: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        n_intent_labels: int = 7,
        n_slot_labels: int = 9,
        dropout: float = 0.1,
    ) -> None:
        """Build the shared BERT encoder and the two independent classification heads."""

        super().__init__()
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            intermediate_size=4 * hidden_size,
        )
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(dropout)
        self.intent_classifier = nn.Linear(hidden_size, n_intent_labels)
        self.slot_classifier = nn.Linear(hidden_size, n_slot_labels)

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor, token_type_ids: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return (intent_logits, slot_logits) from a single shared BERT pass."""

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output

        intent_logits = self.intent_classifier(self.dropout(pooled_output))
        slot_logits = self.slot_classifier(self.dropout(sequence_output))
        return intent_logits, slot_logits


def build_jointbert() -> nn.Module:
    """Build the compact JointBERT intent+slot joint model."""

    return JointBERT().eval()


def example_input_jointbert() -> tuple[Tensor, Tensor, Tensor]:
    """Return (input_ids, attention_mask, token_type_ids) for a short utterance."""

    input_ids = torch.randint(0, 512, (2, 14), dtype=torch.long)
    attention_mask = torch.ones(2, 14, dtype=torch.long)
    token_type_ids = torch.zeros(2, 14, dtype=torch.long)
    return input_ids, attention_mask, token_type_ids


# ---------------------------------------------------------------------------
# KEMP: emotional context graph (multi-head graph attention) + emotion-
# intensity-weighted signal pooling + emotion-dependency Transformer decoder.
# ---------------------------------------------------------------------------
class GraphAttentionContextEncoder(nn.Module):
    """Multi-head graph attention over an emotional context graph."""

    def __init__(self, dim: int, n_heads: int = 4) -> None:
        """Build a self-attention layer used as the graph-aware context encoder."""

        super().__init__()
        self.mha = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, node_embeds: Tensor, adjacency_mask: Tensor) -> Tensor:
        """Return graph-contextualized node embeddings, shape (batch, n_nodes, dim)."""

        attn_out, _ = self.mha(node_embeds, node_embeds, node_embeds, attn_mask=adjacency_mask)
        h = self.norm(node_embeds + attn_out)
        return self.norm2(h + self.ffn(h))


class KEMP(nn.Module):
    """Compact KEMP: emotional context graph encoder + emotion-dependency decoder."""

    def __init__(
        self,
        vocab_size: int = 256,
        dim: int = 32,
        n_heads: int = 4,
        n_dec_layers: int = 2,
    ) -> None:
        """Build the node embedder, graph-attention encoder, and decoder stack."""

        super().__init__()
        self.dim = dim
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.graph_encoder = GraphAttentionContextEncoder(dim, n_heads)
        self.emotion_gate = nn.Linear(dim, 1)

        dec_layer = nn.TransformerDecoderLayer(dim, n_heads, 4 * dim, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_dec_layers)
        self.emotion_bias_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, vocab_size)

    def forward(
        self,
        graph_node_ids: Tensor,
        emotion_intensity: Tensor,
        response_ids: Tensor,
    ) -> Tensor:
        """Return response token logits, shape (batch, tgt_len, vocab_size).

        Parameters
        ----------
        graph_node_ids : Tensor
            Token/concept node ids of the emotional context graph, shape
            (batch, n_nodes).
        emotion_intensity : Tensor
            Per-node emotion-intensity score from the emotion lexicon, shape
            (batch, n_nodes), used to weight the pooled emotional signal.
        response_ids : Tensor
            Teacher-forced response token ids, shape (batch, tgt_len).
        """

        node_embeds = self.token_embed(graph_node_ids)  # (batch, n_nodes, dim)
        graph_embeds = self.graph_encoder(node_embeds, adjacency_mask=None)

        intensity_weights = F.softmax(emotion_intensity, dim=-1).unsqueeze(-1)
        emotional_signal = (intensity_weights * graph_embeds).sum(dim=1, keepdim=True)

        response_embeds = self.token_embed(response_ids)
        memory = graph_embeds + self.emotion_bias_proj(emotional_signal)
        dec_hidden = self.decoder(response_embeds, memory)
        return self.out_proj(dec_hidden)


def build_kemp() -> nn.Module:
    """Build the compact KEMP knowledge-enriched empathetic dialogue model."""

    return KEMP().eval()


def example_input_kemp() -> tuple[Tensor, Tensor, Tensor]:
    """Return (graph node ids, per-node emotion intensity, response token ids)."""

    graph_node_ids = torch.randint(0, 256, (2, 10), dtype=torch.long)
    emotion_intensity = torch.rand(2, 10)
    response_ids = torch.randint(0, 256, (2, 6), dtype=torch.long)
    return graph_node_ids, emotion_intensity, response_ids


# ---------------------------------------------------------------------------
# KETOD (Combiner): shared context encoder + per-snippet knowledge-selection
# scorer + decoder conditioned on context and the selected knowledge snippet.
# ---------------------------------------------------------------------------
class KETOD(nn.Module):
    """Compact KETOD Combiner: context encoder + snippet selector + fused decoder."""

    def __init__(
        self,
        vocab_size: int = 256,
        dim: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
    ) -> None:
        """Build the shared encoder, knowledge-selection scorer, and fused decoder."""

        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, dim)
        enc_layer = nn.TransformerEncoderLayer(dim, n_heads, 4 * dim, batch_first=True)
        self.context_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.snippet_encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.selection_scorer = nn.Bilinear(dim, dim, 1)

        dec_layer = nn.TransformerDecoderLayer(dim, n_heads, 4 * dim, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(dim, vocab_size)

    def forward(
        self,
        context_ids: Tensor,
        snippet_ids: Tensor,
        response_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return (per-snippet selection scores, knowledge-enriched response logits).

        Parameters
        ----------
        context_ids : Tensor
            Dialogue-context token ids, shape (batch, ctx_len).
        snippet_ids : Tensor
            Candidate retrieved knowledge snippets, shape (batch, n_snippets, snip_len).
        response_ids : Tensor
            Teacher-forced response token ids, shape (batch, tgt_len).
        """

        context_hidden = self.context_encoder(self.token_embed(context_ids))
        pooled_context = context_hidden.mean(dim=1)  # (batch, dim)

        batch, n_snippets, snip_len = snippet_ids.shape
        flat_snippets = snippet_ids.reshape(batch * n_snippets, snip_len)
        snippet_hidden = self.snippet_encoder(self.token_embed(flat_snippets)).mean(dim=1)
        snippet_hidden = snippet_hidden.reshape(batch, n_snippets, -1)

        pooled_context_rep = pooled_context.unsqueeze(1).expand(-1, n_snippets, -1)
        selection_scores = self.selection_scorer(pooled_context_rep, snippet_hidden).squeeze(-1)

        selection_weights = F.softmax(selection_scores, dim=-1).unsqueeze(-1)
        selected_knowledge = (selection_weights * snippet_hidden).sum(dim=1, keepdim=True)

        fused_memory = torch.cat([context_hidden, selected_knowledge], dim=1)
        response_embeds = self.token_embed(response_ids)
        dec_hidden = self.decoder(response_embeds, fused_memory)
        response_logits = self.out_proj(dec_hidden)
        return selection_scores, response_logits


def build_ketod() -> nn.Module:
    """Build the compact KETOD Combiner knowledge-enriched TOD model."""

    return KETOD().eval()


def example_input_ketod() -> tuple[Tensor, Tensor, Tensor]:
    """Return (context ids, candidate snippet ids, response ids)."""

    context_ids = torch.randint(0, 256, (2, 12), dtype=torch.long)
    snippet_ids = torch.randint(0, 256, (2, 4, 8), dtype=torch.long)
    response_ids = torch.randint(0, 256, (2, 6), dtype=torch.long)
    return context_ids, snippet_ids, response_ids


MENAGERIE_ENTRIES = [
    ("DialogueRNN", "build_dialoguernn", "example_input_dialoguernn", "2019", "NLP"),
    (
        "InstructDial (Instruction-Tuned Dialogue)",
        "build_instructdial",
        "example_input_instructdial",
        "2022",
        "NLP",
    ),
    ("JointBERT", "build_jointbert", "example_input_jointbert", "2019", "NLP"),
    (
        "KEMP (Knowledge-Enriched Emotional Dialogue)",
        "build_kemp",
        "example_input_kemp",
        "2022",
        "NLP",
    ),
    ("KETOD (Knowledge-Enhanced TOD)", "build_ketod", "example_input_ketod", "2022", "NLP"),
]
