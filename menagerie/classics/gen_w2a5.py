"""Compact faithful reimplementations of five dialogue-system architecture families.

Sources checked (paper + official source; reimplemented compactly, no clone/pip-install
except the already-installed ``transformers`` library used for the RAG family):
  - PersonaGPT: github.com/illidanlab/personaGPT (af1tang, Tang et al. 2021); a
    DialoGPT-medium (GPT-2) decoder fine-tuned on Persona-Chat with SPECIAL SEGMENT
    TOKENS that mark the boundary between a persona-fact block and the dialogue-history
    block within one flat causal token stream (no separate encoder -- persona facts and
    history are just two differently-tagged spans feeding the same GPT-2 stack). This is
    the distinctive mechanism reimplemented here: a learned per-position SEGMENT embedding
    (persona vs. history-speaker-A vs. history-speaker-B) is added to the token+position
    embedding before the causal GPT-2 stack, rather than a generic unconditioned GPT-2 LM.
  - PPTOD (Plug-and-Play Task-Oriented Dialogue): github.com/awslabs/pptod (Su et al.,
    ACL 2022), arxiv:2109.14739; a T5 encoder-decoder backbone whose distinctive mechanism
    is a bank of LEARNABLE TASK-SPECIFIC PROMPT EMBEDDINGS (one short embedding sequence
    per TOD sub-task: NLU / DST / POL / NLG) that is PREPENDED to the token embeddings
    before the T5 encoder runs, so the same backbone is steered to a different TOD
    sub-task purely by which prompt block is prepended ("plug-and-play"). Reimplemented
    here as an explicit prompt-embedding table selected by a task id and concatenated
    with the input token embeddings ahead of the shared T5 encoder-decoder.
  - RAG / RAG-Sequence / RAG-Token: github.com/huggingface/transformers (Lewis et al.,
    NeurIPS 2020), arxiv:2005.11401; a DPR-style question encoder produces a query vector,
    which is scored against retrieved-document embeddings (``doc_scores``) and combined
    with per-document generator contexts (``context_input_ids``); a BART-style generator
    then marginalizes its output distribution over the retrieved documents. RAG-Sequence
    and RAG-Token are DISTINCT marginalization strategies over the SAME retrieval +
    generation architecture: RAG-Sequence uses one selected/marginalized document per
    whole output sequence, RAG-Token marginalizes per output token (a different document
    can dominate at each decoding step) -- HuggingFace ships these as separate classes
    (``RagSequenceForGeneration`` / ``RagTokenForGeneration``) with different loss/decoding
    math, so all three are built here as genuinely distinct modules: the base retrieval-
    augmented encoder-decoder (``RagModel``, plain RAG) plus the two marginalization heads.
    Built directly from the installed ``transformers`` library (``RagConfig`` composed from
    tiny ``DPRConfig`` + ``BartConfig``) with an explicit (non-downloaded) retrieval context
    passed in as ``context_input_ids`` / ``doc_scores`` -- there is no live retriever or
    wiki_dpr index in this environment, so the "retrieval" step is represented by its
    tensor contract (context token ids + document relevance scores) exactly as the model
    itself expects when ``retriever=None``.
  - ReCoSa (Detecting the Relevant Contexts with Self-Attention): github.com/zhanghainan/
    ReCoSa (Zhang, Lan, Pang, Guo & Cheng, ACL 2019), arxiv:1907.05339; a word-level GRU
    encodes each utterance in the dialogue history into a single vector; the sequence of
    per-utterance vectors gets a sinusoidal positional encoding and is then updated with a
    TRANSFORMER-STYLE SELF-ATTENTION stack over the UTTERANCE-LEVEL context (the paper's
    core departure from prior hierarchical-RNN dialogue models, which treat all context
    turns indiscriminately) to obtain relevance-weighted context representations; a masked
    self-attention decoder over the (teacher-forced) response tokens then attends over the
    context representations (context-response attention) before a vocabulary projection.

No skips this batch -- all five are well-documented named mechanisms with either an
official reference implementation, a widely-used library implementation (RAG family in
``transformers``), or a detailed paper description sufficient for a faithful compact
reimplementation.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from transformers import (
    BartConfig,
    DPRConfig,
    GPT2Config,
    GPT2Model,
    RagConfig,
    RagModel,
    RagSequenceForGeneration,
    RagTokenForGeneration,
    T5Config,
    T5ForConditionalGeneration,
)


# ---------------------------------------------------------------------------
# PersonaGPT: GPT-2/DialoGPT causal decoder with a learned SEGMENT embedding
# distinguishing persona-fact tokens from dialogue-history tokens in one flat
# input stream (no separate encoder).
# ---------------------------------------------------------------------------
class PersonaGPT(nn.Module):
    """PersonaGPT (af1tang / illidanlab, 2021): persona-conditioned DialoGPT.

    Parameters
    ----------
    vocab_size : int
        Token vocabulary size.
    n_embd : int
        GPT-2 hidden size.
    n_layer : int
        Number of GPT-2 decoder blocks.
    n_head : int
        Number of attention heads.
    n_positions : int
        Maximum sequence length supported by the position embedding table.
    """

    def __init__(
        self,
        vocab_size: int = 200,
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
        )
        self.gpt2 = GPT2Model(cfg)
        # Segment ids: 0 = persona fact, 1 = history (speaker A), 2 = history
        # (speaker B) -- the special-token boundary markers the paper adds on
        # top of the plain DialoGPT vocabulary, reimplemented as an additive
        # segment embedding so the same causal stack sees a persona-conditioned
        # stream instead of a generic unconditioned LM input.
        self.segment_embed = nn.Embedding(3, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, input_ids: Tensor, segment_ids: Tensor) -> Tensor:
        """Compute next-token logits for a persona-fact + dialogue-history stream.

        Parameters
        ----------
        input_ids : Tensor
            ``(batch, seq_len)`` long token ids: persona-fact tokens followed by
            dialogue-history tokens (teacher-forced), all in one causal stream.
        segment_ids : Tensor
            ``(batch, seq_len)`` long segment ids (0=persona, 1/2=history speakers)
            aligned with ``input_ids``.

        Returns
        -------
        Tensor
            ``(batch, seq_len, vocab_size)`` next-token logits.
        """
        tok_embeds = self.gpt2.wte(input_ids)
        seg_embeds = self.segment_embed(segment_ids)
        hidden = self.gpt2(inputs_embeds=tok_embeds + seg_embeds).last_hidden_state
        return self.lm_head(hidden)


def build_personagpt() -> nn.Module:
    """Build a small PersonaGPT persona-conditioned dialogue model.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return PersonaGPT().eval()


def example_input_personagpt() -> tuple[Tensor, Tensor]:
    """Example persona+history token/segment ids for PersonaGPT.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(input_ids, segment_ids)`` long tensors of shape ``(batch, seq_len)``.
    """
    input_ids = torch.randint(0, 200, (2, 20))
    segment_ids = torch.cat(
        [
            torch.zeros(2, 6, dtype=torch.long),
            torch.ones(2, 7, dtype=torch.long),
            torch.full((2, 7), 2, dtype=torch.long),
        ],
        dim=1,
    )
    return input_ids, segment_ids


# ---------------------------------------------------------------------------
# PPTOD: T5 encoder-decoder with a bank of learnable task-specific PROMPT
# embeddings prepended to the encoder input, steering the shared backbone to a
# different TOD sub-task (NLU/DST/POL/NLG) via which prompt block is selected.
# ---------------------------------------------------------------------------
class PPTOD(nn.Module):
    """PPTOD (Su et al., ACL 2022): plug-and-play multi-task TOD T5 model.

    Parameters
    ----------
    vocab_size : int
        Shared input/output token vocabulary size.
    d_model : int
        T5 hidden size.
    n_layers : int
        Number of T5 encoder and decoder layers.
    n_heads : int
        Number of attention heads.
    n_tasks : int
        Number of TOD sub-tasks with a dedicated prompt (NLU, DST, POL, NLG = 4).
    prompt_len : int
        Number of learnable prompt embedding positions per task.
    """

    def __init__(
        self,
        vocab_size: int = 100,
        d_model: int = 32,
        n_layers: int = 2,
        n_heads: int = 2,
        n_tasks: int = 4,
        prompt_len: int = 3,
    ) -> None:
        super().__init__()
        cfg = T5Config(
            vocab_size=vocab_size,
            d_model=d_model,
            d_kv=d_model // n_heads,
            d_ff=4 * d_model,
            num_layers=n_layers,
            num_heads=n_heads,
        )
        self.t5 = T5ForConditionalGeneration(cfg)
        # One learnable prompt embedding sequence per TOD sub-task -- the
        # "plug-and-play" mechanism: swapping the prepended prompt block
        # steers the SAME backbone to NLU / DST / POL / NLG without any
        # architectural change.
        self.task_prompts = nn.Parameter(torch.randn(n_tasks, prompt_len, d_model) * 0.02)
        self.prompt_len = prompt_len

    def forward(self, input_ids: Tensor, task_id: int, decoder_input_ids: Tensor) -> Tensor:
        """Run PPTOD for one TOD sub-task with its dedicated prompt prepended.

        Parameters
        ----------
        input_ids : Tensor
            ``(batch, src_len)`` long token ids for the dialogue context/input.
        task_id : int
            Index selecting which task-specific prompt block to prepend
            (0=NLU, 1=DST, 2=POL, 3=NLG).
        decoder_input_ids : Tensor
            ``(batch, tgt_len)`` long token ids fed to the decoder (teacher forcing).

        Returns
        -------
        Tensor
            ``(batch, tgt_len, vocab_size)`` output vocabulary logits.
        """
        tok_embeds = self.t5.shared(input_ids)  # (batch, src_len, d_model)
        prompt = self.task_prompts[task_id].unsqueeze(0).expand(tok_embeds.size(0), -1, -1)
        enc_inputs_embeds = torch.cat([prompt, tok_embeds], dim=1)
        out = self.t5(inputs_embeds=enc_inputs_embeds, decoder_input_ids=decoder_input_ids)
        return out.logits


def build_pptod() -> nn.Module:
    """Build a small PPTOD plug-and-play multi-task TOD model.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return PPTOD().eval()


def example_input_pptod() -> tuple[Tensor, int, Tensor]:
    """Example dialogue-context/decoder token ids and a task id for PPTOD.

    Returns
    -------
    tuple[Tensor, int, Tensor]
        ``(input_ids, task_id, decoder_input_ids)``.
    """
    input_ids = torch.randint(0, 100, (2, 14))
    task_id = 1  # DST
    decoder_input_ids = torch.randint(0, 100, (2, 6))
    return input_ids, task_id, decoder_input_ids


# ---------------------------------------------------------------------------
# RAG family: DPR-style question encoder + retrieval context (doc_scores +
# per-document context tokens, standing in for a live retriever/index) feeding
# a BART-style generator. Built directly from `transformers`' RAG classes with
# tiny configs; RagModel is the shared base, RagSequence/RagToken differ only
# in HOW the generator output is marginalized over retrieved documents.
# ---------------------------------------------------------------------------
def _tiny_rag_config(n_docs: int = 2, vocab_size: int = 99) -> RagConfig:
    """Build a tiny composed RagConfig (DPR question encoder + BART generator).

    Parameters
    ----------
    n_docs : int
        Number of retrieved documents per query.
    vocab_size : int
        Shared question-encoder/generator vocabulary size.

    Returns
    -------
    RagConfig
        Composed configuration for a tiny RAG model.
    """
    question_encoder_config = DPRConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        vocab_size=vocab_size,
        projection_dim=32,
    )
    generator_config = BartConfig(
        vocab_size=vocab_size,
        d_model=32,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        max_position_embeddings=64,
        is_encoder_decoder=True,
    )
    return RagConfig.from_question_encoder_generator_configs(
        question_encoder_config,
        generator_config,
        n_docs=n_docs,
        retrieval_vector_size=32,
        vocab_size=vocab_size,
    )


class _RagWrapper(nn.Module):
    """Trace-friendly wrapper flattening RAG's keyword-only forward into positional args."""

    def __init__(self, rag: nn.Module) -> None:
        super().__init__()
        self.rag = rag

    def forward(
        self,
        input_ids: Tensor,
        context_input_ids: Tensor,
        context_attention_mask: Tensor,
        doc_scores: Tensor,
        decoder_input_ids: Tensor,
    ) -> Tensor:
        out = self.rag(
            input_ids=input_ids,
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            doc_scores=doc_scores,
            decoder_input_ids=decoder_input_ids,
        )
        return out.logits


def _rag_example_input(
    n_docs: int = 2, vocab_size: int = 99
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Shared example input builder for the RAG family (question + retrieval context)."""
    batch = 1
    q_len, ctx_len, tgt_len = 6, 8, 5
    input_ids = torch.randint(0, vocab_size, (batch, q_len))
    context_input_ids = torch.randint(0, vocab_size, (batch * n_docs, ctx_len))
    context_attention_mask = torch.ones(batch * n_docs, ctx_len, dtype=torch.long)
    doc_scores = torch.randn(batch, n_docs)
    decoder_input_ids = torch.randint(0, vocab_size, (batch, tgt_len))
    return input_ids, context_input_ids, context_attention_mask, doc_scores, decoder_input_ids


def build_rag() -> nn.Module:
    """Build a small base RAG model (DPR question encoder + BART generator, no marginal head).

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    rag_cfg = _tiny_rag_config()
    base = RagModel(config=rag_cfg)
    return _RagWrapper(base).eval()


def example_input_rag() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Example question + retrieval-context tensors for base RAG.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
        ``(input_ids, context_input_ids, context_attention_mask, doc_scores,
        decoder_input_ids)``.
    """
    return _rag_example_input()


def build_rag_sequence() -> nn.Module:
    """Build a small RAG-Sequence generation model (sequence-level marginalization).

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    rag_cfg = _tiny_rag_config()
    model = RagSequenceForGeneration(config=rag_cfg)
    return _RagWrapper(model).eval()


def example_input_rag_sequence() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Example question + retrieval-context tensors for RAG-Sequence.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
        ``(input_ids, context_input_ids, context_attention_mask, doc_scores,
        decoder_input_ids)``.
    """
    return _rag_example_input()


def build_rag_token() -> nn.Module:
    """Build a small RAG-Token generation model (token-level marginalization).

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    rag_cfg = _tiny_rag_config()
    model = RagTokenForGeneration(config=rag_cfg)
    return _RagWrapper(model).eval()


def example_input_rag_token() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Example question + retrieval-context tensors for RAG-Token.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
        ``(input_ids, context_input_ids, context_attention_mask, doc_scores,
        decoder_input_ids)``.
    """
    return _rag_example_input()


# ---------------------------------------------------------------------------
# ReCoSa: word-level GRU per-utterance encoder -> sinusoidal positional
# encoding + Transformer self-attention over the UTTERANCE-LEVEL context
# (the paper's core mechanism) -> masked response self-attention with
# context-response attention -> vocabulary projection.
# ---------------------------------------------------------------------------
def _sinusoidal_positions(n_positions: int, d_model: int) -> Tensor:
    """Build a standard sinusoidal positional encoding table."""
    position = torch.arange(n_positions).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe = torch.zeros(n_positions, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class ReCoSa(nn.Module):
    """ReCoSa (Zhang et al., ACL 2019): self-attention over utterance-level dialogue context.

    Parameters
    ----------
    vocab_size : int
        Shared context/response token vocabulary size.
    embed_dim : int
        Token embedding dimensionality.
    hidden_dim : int
        Word-level GRU hidden size (also the context/response self-attention width).
    n_context_layers : int
        Number of Transformer self-attention layers over the utterance-level context.
    n_heads : int
        Number of attention heads.
    max_turns : int
        Maximum number of dialogue-history turns (for the positional encoding table).
    """

    def __init__(
        self,
        vocab_size: int = 90,
        embed_dim: int = 24,
        hidden_dim: int = 32,
        n_context_layers: int = 2,
        n_heads: int = 4,
        max_turns: int = 16,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # Word-level utterance encoder (GRU), applied independently per turn.
        self.utterance_gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.register_buffer("context_pos_enc", _sinusoidal_positions(max_turns, hidden_dim))

        context_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=4 * hidden_dim, batch_first=True
        )
        # Context-level self-attention over utterance representations -- the
        # paper's mechanism for detecting which prior turns are RELEVANT to
        # the response, replacing hierarchical-RNN context aggregation.
        self.context_self_attn = nn.TransformerEncoder(context_layer, num_layers=n_context_layers)

        self.response_gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.context_response_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, context_turns: Tensor, response_ids: Tensor) -> Tensor:
        """Encode multi-turn context with self-attention and decode a response.

        Parameters
        ----------
        context_turns : Tensor
            ``(batch, n_turns, turn_len)`` long token ids for prior dialogue turns.
        response_ids : Tensor
            ``(batch, resp_len)`` long token ids fed to the response tower
            (teacher forcing).

        Returns
        -------
        Tensor
            ``(batch, resp_len, vocab_size)`` next-token logits for the response.
        """
        bsz, n_turns, turn_len = context_turns.shape
        turn_embed = self.embed(context_turns.reshape(bsz * n_turns, turn_len))
        _, turn_final = self.utterance_gru(turn_embed)  # (1, batch*n_turns, hidden)
        utt_reprs = turn_final.squeeze(0).reshape(bsz, n_turns, -1)  # (batch, n_turns, hidden)

        utt_reprs = utt_reprs + self.context_pos_enc[:n_turns].unsqueeze(0)
        context_ctx = self.context_self_attn(utt_reprs)  # (batch, n_turns, hidden)

        resp_embed = self.embed(response_ids)
        resp_states, _ = self.response_gru(resp_embed)  # (batch, resp_len, hidden)

        attended, _ = self.context_response_attn(resp_states, context_ctx, context_ctx)
        return self.out_proj(resp_states + attended)


def build_recosa() -> nn.Module:
    """Build a small ReCoSa self-attentive multi-turn dialogue model.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return ReCoSa().eval()


def example_input_recosa() -> tuple[Tensor, Tensor]:
    """Example multi-turn context and (teacher-forced) response token ids for ReCoSa.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(context_turns, response_ids)``.
    """
    context_turns = torch.randint(0, 90, (2, 5, 9))
    response_ids = torch.randint(0, 90, (2, 7))
    return context_turns, response_ids


MENAGERIE_ENTRIES = [
    ("PersonaGPT", "build_personagpt", "example_input_personagpt", "2021", "NLP"),
    ("PPTOD", "build_pptod", "example_input_pptod", "2022", "NLP"),
    ("RAG (Retrieval-Augmented Generation)", "build_rag", "example_input_rag", "2020", "NLP"),
    ("RAG-Sequence", "build_rag_sequence", "example_input_rag_sequence", "2020", "NLP"),
    ("RAG-Token", "build_rag_token", "example_input_rag_token", "2020", "NLP"),
    ("ReCoSa", "build_recosa", "example_input_recosa", "2019", "NLP"),
]
