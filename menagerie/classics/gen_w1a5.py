"""Compact faithful reimplementations for build_queue rows 31-36 (W1A5).

Sources checked (repo/API browsed via ``gh api``/raw GitHub fetch, no clone or
pip-install; base env only):
  - EmpathyEar / CEM: https://github.com/Sahandfer/CEM
    (``src/models/CEM/model.py``) + arxiv:2109.05739 (Sabour et al., AAAI'22,
    "CEM: Commonsense-aware Empathetic Response Generation"). Distinctive
    mechanism: the dialogue context is encoded once, then five COMET
    commonsense-relation sequences (``x_intent``, ``x_need``, ``x_want``,
    ``x_effect`` -> a shared "cognition" encoder, ``x_react`` -> a separate
    "emotion" encoder) are each encoded and their CLS ([:, 0]) tokens fused
    back against the context via dedicated ``emo_ref_encoder`` /
    ``cog_ref_encoder`` cross networks. The four cognitive-relation CLS
    fusions are concatenated and passed through a **sigmoid self-gate**
    (``cog_contrib = sigmoid(cog_ref_ctx); cog_ref_ctx = cog_contrib *
    cog_ref_ctx``) before an ``MLP`` projection; the emotion fusion feeds a
    linear emotion-classification head. Both pathways condition a shared
    Transformer decoder (cognitive+emotional empathy pathways, per the repo
    README and paper Fig. 2).
  - EmpDG: https://github.com/qtli/EmpDG (``Model/Empdg_G.py``,
    ``Model/EmpDG_D.py``) + arxiv:1911.08698 (Li et al., COLING 2020).
    Distinctive mechanism: **two parallel encoders** over two different
    token streams -- a ``Semantic_Encoder`` over the raw dialogue context and
    an ``Emotion_Encoder`` over an emotion-focused-word context (the "coarse"
    global signal). Their CLS tokens are concatenated and passed through
    ``identify_new`` to predict a fine-grained emotion label (the
    "multi-resolution" -- coarse context + fine label -- emotion signal).
    The predicted emotion logits are embedded and used as the decoder's
    start-of-sequence vector, and the decoder cross-attends over the
    concatenation of both encoders' outputs. The official repo also trains
    an interactive discriminator (``EmpDG_D.py``, an LSTM + 2-gram conv
    "Semantic_Discriminator") that scores generated vs. gold responses
    against the fused semantic+emotion context and feeds a loss back into
    the generator (the "interactive adversarial training" of the title).
    Reimplemented here as a single traceable module containing both the
    generator's dual-encoder/fused-decode path and a compact CPU-safe
    discriminator head (LSTM + 1-D conv n-gram pooling gated by the fused
    context), returning ``(lm_logits, discriminator_score)``.
  - FiD (Fusion-in-Decoder): https://github.com/facebookresearch/FiD
    (``src/model.py``) + arxiv:2007.01282 (Izacard & Grave, ICLR 2021).
    Distinctive mechanism: ``EncoderWrapper.forward`` reshapes
    ``(batch, n_passages, passage_len)`` into ``(batch * n_passages,
    passage_len)``, runs the *shared* T5 encoder independently per passage
    (no cross-passage attention in the encoder), then reshapes the encoded
    states back to ``(batch, n_passages * passage_len, d_model)`` so a
    single T5 decoder cross-attends **jointly** over the concatenation of
    all passages ("fusion in decoder"). Reimplemented natively with
    ``transformers.T5Config`` + ``T5Stack`` at tiny dims, matching the exact
    reshape-encode-concat-decode sequence of the official ``model.py``.
  - FoCus: https://github.com/pkchat-focus/FoCus
    (``classification_modules.py``, class ``GPT2PK_ctxt``) +
    arxiv:2112.08619 (Jang et al., AAAI 2022). Distinctive mechanism: a
    **two-stage grounding** pipeline before generation. (1) A shared GPT-2
    encoder produces an EOS-pooled context representation and, per knowledge
    paragraph, an EOS-pooled representation combined via a small
    ``attn1``/``attn2`` bottleneck into an "in-context" state; (2) each of
    the persona-sentence and knowledge-paragraph *candidates* is scored by
    concatenating the in-context state with the candidate's own EOS
    representation and passing it through a ``ConcatSummary`` linear head
    (persona: multi-label sigmoid grounding; knowledge: single-label softmax
    grounding). The top-scored persona sentences and the top-1 knowledge
    paragraph are then spliced into the GPT-2 input for language-model
    generation. Reimplemented compactly and vectorized (batched candidate
    scoring instead of the reference's per-example Python loops) with a
    small GPT-2 backbone from ``transformers.GPT2Config``.

Skipped (see MENAGERIE_ENTRIES omission + build-queue notes):
  - End-to-End Memory Networks (MemN2N), cand_00047: already present in the
    catalog as ``memn2n_babi`` / ``MemN2N`` in
    ``menagerie/classics/memn2n_match_lstm.py`` (identical paper,
    arXiv:1503.08895, identical multi-hop-attention mechanism). Skipped as
    already_in_catalog.
  - FLARE (Forward-Looking Active Retrieval), cand_00049: the official
    ``jzbjyb/FLARE`` repo (``src/openai_api.py``, ``src/retriever.py``,
    ``src/bing.py``) contains no ``nn.Module`` at all -- FLARE is an
    inference-time prompting/orchestration algorithm (generate a provisional
    sentence with a black-box OpenAI completion API, detect low-confidence
    tokens, trigger a Bing-search retrieval + regeneration) layered on top
    of an external API-served LM, not a distinct trainable architecture.
    There is no faithful traceable ``nn.Module`` to build without inventing
    a generic stub, so it is skipped as not_a_traceable_architecture.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model, T5Config
from transformers.models.t5.modeling_t5 import T5Stack


# ---------------------------------------------------------------------------
# EmpathyEar / CEM: commonsense-aware cognitive + affective empathy pathways
# ---------------------------------------------------------------------------


class _TinyTransformerEncoder(nn.Module):
    """Compact pre-norm Transformer encoder stack (stand-in for CEM's ``Encoder``)."""

    def __init__(self, dim: int, n_layers: int = 2, n_heads: int = 2) -> None:
        """Initialize the stacked ``TransformerEncoderLayer`` blocks.

        Parameters
        ----------
        dim:
            Hidden feature width.
        n_layers:
            Number of encoder layers.
        n_heads:
            Attention head count.
        """

        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 2, batch_first=True
        )
        self.net = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: Tensor) -> Tensor:
        """Encode a token-embedding sequence.

        Parameters
        ----------
        x:
            Embedded sequence of shape ``(batch, seq_len, dim)``.

        Returns
        -------
        Tensor
            Contextualized sequence of the same shape.
        """

        return self.net(x)


class CEMEmpathyModel(nn.Module):
    """CEM: cognitive (COMET x_intent/x_need/x_want/x_effect) + affective (x_react) empathy."""

    def __init__(
        self,
        vocab_size: int = 80,
        dim: int = 24,
        n_emotions: int = 8,
        n_layers: int = 2,
    ) -> None:
        """Initialize the shared embedding, context/commonsense encoders, and decoder.

        Parameters
        ----------
        vocab_size:
            Shared token vocabulary size.
        dim:
            Hidden feature width.
        n_emotions:
            Number of coarse emotion classes for the ``emo_lin`` head.
        n_layers:
            Layer count for every encoder/decoder stack.
        """

        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.context_encoder = _TinyTransformerEncoder(dim, n_layers)
        self.cog_encoder = _TinyTransformerEncoder(dim, n_layers)
        self.emo_encoder = _TinyTransformerEncoder(dim, n_layers)
        # Reference fusion networks consume [context ; broadcast CLS] (2*dim).
        self.emo_ref_encoder = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.cog_ref_encoder = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.emo_lin = nn.Linear(dim, n_emotions, bias=False)
        # Four cognitive relations (x_intent, x_need, x_want, x_effect) concatenated.
        self.cog_lin = nn.Sequential(nn.Linear(4 * dim, dim), nn.GELU(), nn.Linear(dim, dim))
        dec_layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=2, dim_feedforward=dim * 2, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(
        self,
        context: Tensor,
        x_intent: Tensor,
        x_need: Tensor,
        x_want: Tensor,
        x_effect: Tensor,
        x_react: Tensor,
        decoder_input_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Fuse the dialogue context with five COMET commonsense relations, then decode.

        Parameters
        ----------
        context:
            Dialogue-context token ids, shape ``(batch, ctx_len)``.
        x_intent, x_need, x_want, x_effect:
            Cognitive COMET-relation token ids feeding the shared
            "cognition" pathway, each shape ``(batch, rel_len)``.
        x_react:
            Affective COMET-relation token ids feeding the "emotion"
            pathway, shape ``(batch, rel_len)``.
        decoder_input_ids:
            Response-decoder token ids, shape ``(batch, tgt_len)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Next-token logits ``(batch, tgt_len, vocab_size)`` and emotion
            classification logits ``(batch, n_emotions)``.
        """

        enc_out = self.context_encoder(self.embed(context))
        ctx_len = enc_out.shape[1]

        cog_rel_cls = []
        for rel in (x_intent, x_need, x_want, x_effect):
            rel_out = self.cog_encoder(self.embed(rel))
            cog_rel_cls.append(rel_out[:, 0:1, :])
        emo_out = self.emo_encoder(self.embed(x_react))
        emo_cls = emo_out.mean(dim=1, keepdim=True)

        # Emotion pathway: fuse context with the affective (x_react) CLS token.
        emo_concat = torch.cat([enc_out, emo_cls.expand(-1, ctx_len, -1)], dim=-1)
        emo_ref_ctx = self.emo_ref_encoder(emo_concat)
        emo_logits = self.emo_lin(emo_ref_ctx[:, 0])

        # Cognitive pathway: fuse context with each of the four cognitive CLS
        # tokens, concatenate, and apply a sigmoid self-gate before projection.
        cog_fused = []
        for cls in cog_rel_cls:
            cog_concat = torch.cat([enc_out, cls.expand(-1, ctx_len, -1)], dim=-1)
            cog_fused.append(self.cog_ref_encoder(cog_concat)[:, 0])
        cog_ref_ctx = torch.cat(cog_fused, dim=-1)
        cog_contrib = torch.sigmoid(cog_ref_ctx)
        cog_ref_ctx = cog_contrib * cog_ref_ctx
        cog_ref_ctx = self.cog_lin(cog_ref_ctx)

        # Combine cognitive + affective empathy pathways as decoder memory.
        memory = torch.cat([enc_out, (cog_ref_ctx + emo_ref_ctx[:, 0]).unsqueeze(1)], dim=1)
        dec_emb = self.embed(decoder_input_ids)
        dec_out = self.decoder(dec_emb, memory)
        return self.lm_head(dec_out), emo_logits


def build_cem_empathy() -> nn.Module:
    """Build a small CEM (EmpathyEar) commonsense-empathy dialogue model.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """

    return CEMEmpathyModel().eval()


def example_input_cem_empathy() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Example context, four COMET relation streams, x_react, and decoder inputs.

    Returns
    -------
    tuple[Tensor, ...]
        ``(context, x_intent, x_need, x_want, x_effect, x_react,
        decoder_input_ids)`` long tensors.
    """

    ctx = torch.randint(0, 80, (2, 10))
    x_intent = torch.randint(0, 80, (2, 4))
    x_need = torch.randint(0, 80, (2, 4))
    x_want = torch.randint(0, 80, (2, 4))
    x_effect = torch.randint(0, 80, (2, 4))
    x_react = torch.randint(0, 80, (2, 4))
    dec = torch.randint(0, 80, (2, 6))
    return ctx, x_intent, x_need, x_want, x_effect, x_react, dec


# ---------------------------------------------------------------------------
# EmpDG: multi-resolution emotion (coarse context + fine label) + adversarial
# interactive discriminator
# ---------------------------------------------------------------------------


class _DualStreamEncoder(nn.Module):
    """One shared-architecture Transformer encoder tower used for both streams."""

    def __init__(self, dim: int, n_layers: int = 2, n_heads: int = 2) -> None:
        """Initialize the encoder stack.

        Parameters
        ----------
        dim:
            Hidden feature width.
        n_layers:
            Number of encoder layers.
        n_heads:
            Attention head count.
        """

        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 2, batch_first=True
        )
        self.net = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: Tensor) -> Tensor:
        """Encode a token-embedding sequence.

        Parameters
        ----------
        x:
            Embedded sequence, shape ``(batch, seq_len, dim)``.

        Returns
        -------
        Tensor
            Contextualized sequence, same shape.
        """

        return self.net(x)


class EmpDGModel(nn.Module):
    """EmpDG generator + interactive semantic discriminator (COLING 2020)."""

    def __init__(
        self,
        vocab_size: int = 72,
        dim: int = 20,
        n_emotions: int = 8,
        n_layers: int = 2,
    ) -> None:
        """Initialize the dual encoders, emotion-fused decoder, and discriminator head.

        Parameters
        ----------
        vocab_size:
            Shared token vocabulary size.
        dim:
            Hidden feature width.
        n_emotions:
            Number of fine-grained emotion classes.
        n_layers:
            Layer count used throughout.
        """

        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.semantic_und = _DualStreamEncoder(dim, n_layers)
        self.emotion_pec = _DualStreamEncoder(dim, n_layers)
        # Multi-resolution fusion: concat semantic + emotion CLS -> fine label.
        self.identify_new = nn.Linear(2 * dim, n_emotions, bias=False)
        self.emotion_embedding = nn.Linear(n_emotions, dim)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=2, dim_feedforward=dim * 2, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Interactive semantic discriminator: LSTM feedback + 2-gram conv
        # classifier gated by the fused semantic+emotion context (stand-in
        # for EmpDG_D.py's Semantic_Discriminator, made CPU/trace-safe).
        self.disc_lstm = nn.LSTM(dim, dim, batch_first=True)
        self.disc_conv = nn.Conv1d(dim, dim, kernel_size=2)
        self.disc_dense = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU())
        self.disc_out = nn.Linear(dim // 2, 1)

    def forward(
        self, context: Tensor, emotion_context: Tensor, decoder_input_ids: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Encode semantic + emotion streams, predict emotion, decode, then discriminate.

        Parameters
        ----------
        context:
            Raw dialogue-context token ids, shape ``(batch, ctx_len)``.
        emotion_context:
            Emotion-word-focused context token ids (the coarse global
            signal), shape ``(batch, emo_len)``.
        decoder_input_ids:
            Response-decoder token ids, shape ``(batch, tgt_len)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Next-token logits ``(batch, tgt_len, vocab_size)``, fine-grained
            emotion logits ``(batch, n_emotions)``, and a scalar
            interactive-discriminator realness score ``(batch, 1)``.
        """

        sem_out = self.semantic_und(self.embed(context))
        emo_out = self.emotion_pec(self.embed(emotion_context))

        # Multi-resolution emotion perception: fuse both streams' CLS tokens.
        emotion_logit = self.identify_new(torch.cat([emo_out[:, 0], sem_out[:, 0]], dim=-1))

        # Combine two contexts for cross-attention memory (src_emb in the reference).
        fused_ctx = torch.cat([sem_out, emo_out], dim=1)

        sos_emb = self.emotion_embedding(emotion_logit).unsqueeze(1)
        dec_emb = torch.cat([sos_emb, self.embed(decoder_input_ids)[:, :-1]], dim=1)
        dec_out = self.decoder(dec_emb, fused_ctx)
        lm_logits = self.lm_head(dec_out)

        # Interactive adversarial discriminator: LSTM feedback over the
        # decoded sequence, gated 2-gram conv pooling against the fused
        # semantic+emotion context (mean-pooled), scored to a single logit.
        lstm_out, _ = self.disc_lstm(dec_out)
        conv_in = lstm_out.transpose(1, 2)
        conv_out = F.relu(self.disc_conv(conv_in))
        pooled = F.max_pool1d(conv_out, conv_out.size(-1)).squeeze(-1)
        gated = F.relu(pooled + fused_ctx.mean(dim=1))
        disc_score = self.disc_out(self.disc_dense(gated))
        return lm_logits, emotion_logit, disc_score


def build_empdg() -> nn.Module:
    """Build a small EmpDG generator + interactive discriminator model.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """

    return EmpDGModel().eval()


def example_input_empdg() -> tuple[Tensor, Tensor, Tensor]:
    """Example raw-context, emotion-context, and decoder token ids for EmpDG.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(context, emotion_context, decoder_input_ids)`` long tensors.
    """

    context = torch.randint(0, 72, (2, 12))
    emotion_context = torch.randint(0, 72, (2, 6))
    decoder_input_ids = torch.randint(0, 72, (2, 7))
    return context, emotion_context, decoder_input_ids


# ---------------------------------------------------------------------------
# FiD (Fusion-in-Decoder): per-passage independent T5 encoding, joint decode
# ---------------------------------------------------------------------------


class FusionInDecoderT5(nn.Module):
    """Fusion-in-Decoder: encode N passages independently, decode jointly over all."""

    def __init__(
        self,
        vocab_size: int = 96,
        d_model: int = 24,
        n_layer: int = 2,
        n_head: int = 4,
    ) -> None:
        """Store a shared T5 encoder/decoder pair used for FiD-style fusion.

        Parameters
        ----------
        vocab_size:
            Shared token vocabulary size.
        d_model:
            Transformer hidden width.
        n_layer:
            Encoder and decoder layer count.
        n_head:
            Attention head count.
        """

        super().__init__()
        shared_cfg = T5Config(
            vocab_size=vocab_size,
            d_model=d_model,
            d_ff=d_model * 4,
            num_layers=n_layer,
            num_decoder_layers=n_layer,
            num_heads=n_head,
            d_kv=d_model // n_head,
            is_gated_act=False,
            dense_act_fn="relu",
            use_cache=False,
        )
        self.shared_embed = nn.Embedding(vocab_size, d_model)
        dec_cfg = T5Config(
            **{**shared_cfg.to_dict(), "is_decoder": True, "is_encoder_decoder": False}
        )
        self.encoder = T5Stack(shared_cfg, embed_tokens=self.shared_embed)
        self.decoder = T5Stack(dec_cfg, embed_tokens=self.shared_embed)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, passage_input_ids: Tensor, decoder_input_ids: Tensor) -> Tensor:
        """Encode each retrieved passage independently, then fuse in the decoder.

        Parameters
        ----------
        passage_input_ids:
            Token ids of shape ``(batch, n_passages, passage_len)`` --
            e.g. up to 100 retrieved passages in the original paper.
        decoder_input_ids:
            Decoder token ids of shape ``(batch, tgt_len)``.

        Returns
        -------
        Tensor
            Next-token logits of shape ``(batch, tgt_len, vocab_size)``.
        """

        bsz, n_passages, passage_len = passage_input_ids.shape
        # EncoderWrapper.forward: (B, N, L) -> (B*N, L), independent encode.
        flat = passage_input_ids.reshape(bsz * n_passages, passage_len)
        encoded = self.encoder(input_ids=flat).last_hidden_state
        # Reshape back to (B, N*L, D): every passage's states are now visible
        # to the decoder's cross-attention as one long fused memory.
        fused = encoded.reshape(bsz, n_passages * passage_len, -1)
        dec_out = self.decoder(
            input_ids=decoder_input_ids, encoder_hidden_states=fused
        ).last_hidden_state
        return self.lm_head(dec_out)


def build_fid() -> nn.Module:
    """Build a small Fusion-in-Decoder (FiD) T5 reader model.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """

    return FusionInDecoderT5().eval()


def example_input_fid() -> tuple[Tensor, Tensor]:
    """Example multi-passage and decoder token ids for FiD.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(passage_input_ids, decoder_input_ids)`` long tensors; 4 passages
        of length 8, matching the paper's "scales to ~100 passages" fusion
        mechanism at catalog-friendly dims.
    """

    passages = torch.randint(0, 96, (1, 4, 8))
    decoder_input_ids = torch.randint(0, 96, (1, 5))
    return passages, decoder_input_ids


# ---------------------------------------------------------------------------
# FoCus: two-stage persona + knowledge grounding heads before GPT-2 generation
# ---------------------------------------------------------------------------


class FoCusModel(nn.Module):
    """FoCus: persona + knowledge grounding classification, then GPT-2 generation."""

    def __init__(
        self,
        vocab_size: int = 100,
        n_positions: int = 48,
        n_embd: int = 24,
        n_layer: int = 2,
        n_head: int = 2,
        n_persona_can: int = 3,
        n_knowledge_can: int = 3,
    ) -> None:
        """Initialize the shared GPT-2 backbone and the grounding-classification heads.

        Parameters
        ----------
        vocab_size:
            Shared token vocabulary size.
        n_positions:
            Maximum GPT-2 context length.
        n_embd:
            Transformer hidden width.
        n_layer:
            GPT-2 layer count.
        n_head:
            GPT-2 attention head count.
        n_persona_can:
            Number of persona-sentence grounding candidates.
        n_knowledge_can:
            Number of knowledge-paragraph grounding candidates.
        """

        super().__init__()
        self.n_persona_can = n_persona_can
        self.n_knowledge_can = n_knowledge_can
        cfg = GPT2Config(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
        )
        self.transformer = GPT2Model(cfg)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # In-context bottleneck: attn1/attn2 compress each knowledge
        # paragraph's EOS representation before concatenation with the
        # dialogue-context EOS representation (GPT2PK_ctxt.attn1/attn2).
        self.attn1 = nn.Linear(n_embd, 5)
        self.attn2 = nn.Linear(5, n_embd)

        # Two-stage grounding heads (ConcatSummary over [in-context ; candidate]).
        self.persona_summary = nn.Linear(2 * n_embd, 1)
        self.knowledge_summary = nn.Linear(2 * n_embd, 1)

    def forward(
        self,
        dialogue_ids: Tensor,
        knowledge_context_ids: Tensor,
        persona_candidate_ids: Tensor,
        knowledge_candidate_ids: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Score persona/knowledge grounding candidates, then generate conditioned on them.

        Parameters
        ----------
        dialogue_ids:
            Dialogue-context token ids, shape ``(batch, dlg_len)``.
        knowledge_context_ids:
            Full knowledge-paragraph token ids used for the in-context
            bottleneck, shape ``(batch, n_knowledge_can, know_len)``.
        persona_candidate_ids:
            Persona-sentence candidate token ids, shape
            ``(batch, n_persona_can, persona_len)``.
        knowledge_candidate_ids:
            Knowledge-paragraph candidate token ids, shape
            ``(batch, n_knowledge_can, know_len)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Persona grounding logits ``(batch, n_persona_can)``, knowledge
            grounding logits ``(batch, n_knowledge_can)``, and LM logits
            ``(batch, dlg_len, vocab_size)``.
        """

        bsz = dialogue_ids.size(0)
        dlg_hidden = self.transformer(input_ids=dialogue_ids).last_hidden_state
        dlg_eos = dlg_hidden[:, -1, :]  # (batch, n_embd)

        know_len = knowledge_context_ids.size(-1)
        know_hidden = self.transformer(
            input_ids=knowledge_context_ids.reshape(bsz * self.n_knowledge_can, know_len)
        ).last_hidden_state
        know_eos = know_hidden[:, -1, :].reshape(bsz, self.n_knowledge_can, -1)
        know_bottleneck = self.attn2(self.attn1(know_eos))  # (batch, n_knowledge_can, n_embd)
        in_ctxt = dlg_eos.unsqueeze(1) + know_bottleneck.mean(
            dim=1, keepdim=True
        )  # (batch, 1, n_embd)

        persona_len = persona_candidate_ids.size(-1)
        persona_hidden = self.transformer(
            input_ids=persona_candidate_ids.reshape(bsz * self.n_persona_can, persona_len)
        ).last_hidden_state
        persona_eos = persona_hidden[:, -1, :].reshape(bsz, self.n_persona_can, -1)
        persona_rep = torch.cat([in_ctxt.expand(-1, self.n_persona_can, -1), persona_eos], dim=-1)
        persona_logits = self.persona_summary(persona_rep).squeeze(-1)

        know_len2 = knowledge_candidate_ids.size(-1)
        know_can_hidden = self.transformer(
            input_ids=knowledge_candidate_ids.reshape(bsz * self.n_knowledge_can, know_len2)
        ).last_hidden_state
        know_can_eos = know_can_hidden[:, -1, :].reshape(bsz, self.n_knowledge_can, -1)
        know_rep = torch.cat([in_ctxt.expand(-1, self.n_knowledge_can, -1), know_can_eos], dim=-1)
        knowledge_logits = self.knowledge_summary(know_rep).squeeze(-1)

        # Grounding-conditioned generation: weight the top persona/knowledge
        # candidates (soft selection, differentiable stand-in for the
        # reference's hard top-k splice) and add as a bias to the dialogue
        # hidden states before the LM head.
        persona_weights = torch.sigmoid(persona_logits).unsqueeze(-1)
        knowledge_weights = F.softmax(knowledge_logits, dim=-1).unsqueeze(-1)
        grounding_bias = (persona_weights * persona_eos).mean(dim=1) + (
            knowledge_weights * know_can_eos
        ).sum(dim=1)
        lm_logits = self.lm_head(dlg_hidden + grounding_bias.unsqueeze(1))

        return persona_logits, knowledge_logits, lm_logits


def build_focus() -> nn.Module:
    """Build a small FoCus persona+knowledge-grounded GPT-2 dialogue model.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """

    return FoCusModel().eval()


def example_input_focus() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Example dialogue, knowledge-context, and grounding-candidate token ids for FoCus.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        ``(dialogue_ids, knowledge_context_ids, persona_candidate_ids,
        knowledge_candidate_ids)`` long tensors.
    """

    dialogue_ids = torch.randint(0, 100, (1, 14))
    knowledge_context_ids = torch.randint(0, 100, (1, 3, 8))
    persona_candidate_ids = torch.randint(0, 100, (1, 3, 6))
    knowledge_candidate_ids = torch.randint(0, 100, (1, 3, 8))
    return dialogue_ids, knowledge_context_ids, persona_candidate_ids, knowledge_candidate_ids


MENAGERIE_ENTRIES = [
    ("EmpathyEar", "build_cem_empathy", "example_input_cem_empathy", "2022", "NLP"),
    (
        "EmpDG (Empathetic Dialogue Generation with Multi-resolution Emotion)",
        "build_empdg",
        "example_input_empdg",
        "2020",
        "NLP",
    ),
    ("FiD (Fusion-in-Decoder)", "build_fid", "example_input_fid", "2021", "NLP"),
    (
        "FoCus (Persona + Knowledge Grounded Dialogue)",
        "build_focus",
        "example_input_focus",
        "2022",
        "NLP",
    ),
]
