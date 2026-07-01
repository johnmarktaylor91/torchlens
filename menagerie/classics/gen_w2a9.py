"""Compact faithful reimplementations of four tool-augmented-LLM and dialogue-state
architecture families.

Sources checked (paper + official/community source; reimplemented compactly from
scratch in base-env torch, no clone/pip-install):
  - Toolformer: Schick, Dwivedi-Yu, Dessi, Raileanu, Lomeli, Zettlemoyer, Cancedda &
    Scialom, "Toolformer: Language Models Can Teach Themselves to Use Tools"
    (arxiv:2302.04761); community reference lucidrains/toolformer-pytorch (Meta released
    no official code). The distinctive mechanism is NOT a separate tool-selection
    network -- it is a SINGLE causal LM that self-supervises API-call insertion inline
    in its own generated text: the model samples candidate API-call spans
    (``[tool_name(args) -> result]``) at candidate positions, and a scalar FITNESS/
    filtering score (the drop in next-token cross-entropy loss the model achieves when
    the call's *result* is spliced into the context, versus no-call/no-result
    baselines) decides whether that call is kept as self-supervised training data. Once
    filtered, ordinary LM fine-tuning teaches the model to emit ``[`` at the right
    position and the whole call as ordinary tokens. Reimplemented here as one causal
    decoder LM over a vocabulary unioned with API special tokens (``[``, ``->``, ``]``)
    plus a fitness-scoring head that reproduces the filtering computation directly: it
    consumes the LM's per-position hidden state under three parallel continuations
    (no-call, call-with-result, call-without-result) and outputs the weighted loss
    reduction that governs whether a call is retained -- the paper's actual filtering
    statistic, not a generic classifier.
  - ToolkenGPT: Hao, Liu, Wang, Hu & Wang, "ToolkenGPT: Augmenting Frozen Language
    Models with Massive Tools via Tool Embeddings" (NeurIPS 2023 oral, arxiv:2305.11554);
    official repo Ber666/ToolkenGPT. The distinctive mechanism is representing each tool
    as a new vocabulary token ("toolken") whose embedding is APPENDED to the (frozen)
    LM's output head, rather than fine-tuning the LM body or using a text-based tool
    description in-context: the model runs in "reasoning mode" generating ordinary
    tokens with the toolken logits scored alongside the normal vocabulary at every step
    (so tool selection is literally next-token prediction over an augmented head); once
    a toolken is the argmax, the model switches to "tool mode" and fills in the
    call's numeric/text arguments using a few in-context demonstrations, then the
    executed result is spliced back into the token stream and reasoning-mode resumes.
    Reimplemented here as a small frozen-simulated causal decoder body plus a lightweight
    trainable toolken embedding table concatenated onto the frozen LM head's weight
    matrix at the vocabulary dimension (exactly the augmented-head design), with a
    reasoning/tool-mode gate driven by whether a toolken id was the previous step's
    argmax.
  - ToolLLM / ToolLLaMA: Qin, Liang, Ye, Zhu, Yan, Lu, Lin, Cong, Tang, Qian, Zhu, Xie,
    Zhou, Gerstein, Li, Liu & Sun, "ToolLLM: Facilitating Large Language Models to
    Master 16000+ Real-world APIs" (ICLR 2024 spotlight, arxiv:2307.16789); official
    repo OpenBMB/ToolBench. ToolLLaMA is the LLaMA checkpoint fine-tuned by the ToolLLM
    methodology on the ToolBench instruction dataset -- SAME repo, SAME paper, SAME
    trainable architecture (the DFSDT search procedure at inference time is a decoding
    algorithm over an ordinary function-calling LLM, not a second trainable network), so
    only ONE classics entry is built here (registered under both names would be a
    literal duplicate). The distinctive mechanism captured is the FUNCTION-CALLING
    causal LM: a standard LLaMA-style decoder whose vocabulary is augmented with
    structured "Thought / API Name / API Input / Observation" role-segment tokens and a
    dedicated API-name classification head reading the current hidden state, so a single
    autoregressive decoder emits interleaved natural-language reasoning tokens and
    structured API-call tokens from one shared trunk -- the design DFSDT then searches
    over at decode time. Reimplemented here as a small LLaMA-style (RMSNorm + rotary
    attention + SwiGLU) causal decoder with role-segment embeddings for the
    thought/API-name/API-input/observation spans and a joint LM-head / API-selection
    head over the final hidden states.
  - TRADE (TRAnsferable Dialogue statE generator): Wu, Madotto, Hosseini-Asl, Xiong,
    Zhou & Fung, "Transferable Multi-Domain State Generator for Task-Oriented Dialogue
    Systems" (ACL 2019, arxiv:1905.08743); official repo jasonwu0731/trade-dst. The
    distinctive mechanism is a SINGLE bidirectional-GRU dialogue encoder shared across
    ALL (domain, slot) pairs, feeding two components per pair rather than a per-slot
    classifier over a closed ontology: (1) a SLOT GATE that predicts one of
    {ptr, dontcare, none} for that (domain, slot) from a learned domain-slot embedding
    attending over the dialogue encoding, and (2) a per-slot RNN STATE GENERATOR that
    decodes the slot's value token-by-token with a soft-gated COPY mechanism blending a
    fixed-vocabulary generation distribution with a copy distribution over dialogue
    history token positions (via encoder-attention weights scattered onto their source
    vocabulary ids) -- letting the model copy out-of-vocabulary entity values it has
    never seen, which is what gives TRADE its zero-shot cross-domain transfer property.
    Reimplemented here with a shared bidirectional-GRU dialogue encoder, a learned
    (domain, slot) embedding table feeding both a 3-way slot gate and a GRU decoder with
    the paper's soft copy-generate blend, run independently per (domain, slot) pair from
    one shared encoder pass.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Toolformer
# ---------------------------------------------------------------------------


class _CausalSelfAttention(nn.Module):
    """Standard causal multi-head self-attention block.

    Parameters
    ----------
    dim : int
        Model width.
    n_heads : int
        Number of attention heads.
    """

    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply causal self-attention.

        Parameters
        ----------
        x : Tensor
            ``(batch, seq_len, dim)`` input.

        Returns
        -------
        Tensor
            ``(batch, seq_len, dim)`` attended output.
        """
        b, t, c = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(torch.full((t, t), float("-inf"), device=x.device), diagonal=1)
        scores = scores + causal_mask
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, t, c)
        return self.proj(out)


class _DecoderBlock(nn.Module):
    """Pre-norm transformer decoder block (attention + MLP).

    Parameters
    ----------
    dim : int
        Model width.
    n_heads : int
        Number of attention heads.
    """

    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = _CausalSelfAttention(dim, n_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x: Tensor) -> Tensor:
        """Apply one decoder block.

        Parameters
        ----------
        x : Tensor
            ``(batch, seq_len, dim)`` input.

        Returns
        -------
        Tensor
            ``(batch, seq_len, dim)`` block output.
        """
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Toolformer(nn.Module):
    """Toolformer (Schick et al., 2023): causal LM + inline API-call fitness scorer.

    Parameters
    ----------
    vocab_size : int
        Ordinary token vocabulary size (API special tokens are additional ids
        appended at the top of this range by the caller).
    dim : int
        Model width.
    n_layers : int
        Number of decoder blocks.
    n_heads : int
        Number of attention heads.
    max_len : int
        Maximum sequence length (for learned position embeddings).
    """

    def __init__(
        self,
        vocab_size: int = 96,
        dim: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        max_len: int = 32,
    ) -> None:
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_len, dim)
        self.blocks = nn.ModuleList([_DecoderBlock(dim, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Fitness-filtering head: reads the shared hidden state under three parallel
        # continuations (no-call / call+result / call-without-result) and produces the
        # scalar loss-reduction statistic that decides whether an API call is kept as
        # self-supervised training data (Toolformer's core filtering mechanism).
        self.fitness_proj = nn.Linear(dim, 3)

    def _encode(self, token_ids: Tensor) -> Tensor:
        b, t = token_ids.shape
        pos = torch.arange(t, device=token_ids.device).unsqueeze(0).expand(b, -1)
        x = self.tok_embed(token_ids) + self.pos_embed(pos)
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    def forward(
        self, token_ids: Tensor, result_ids: Tensor, no_result_ids: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Run the causal LM and compute the API-call fitness-filtering score.

        Parameters
        ----------
        token_ids : Tensor
            ``(batch, seq_len)`` long token ids of the plain (no-call) continuation.
        result_ids : Tensor
            ``(batch, seq_len)`` long token ids with the API call's result spliced in.
        no_result_ids : Tensor
            ``(batch, seq_len)`` long token ids with the call present but no result.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(lm_logits, fitness_scores)`` -- ``lm_logits`` is
            ``(batch, seq_len, vocab_size)`` next-token logits over the plain
            continuation, and ``fitness_scores`` is ``(batch, 3)`` per-continuation
            loss-reduction statistics (no-call, call+result, call-no-result) whose
            pairwise differences form the paper's filtering criterion.
        """
        hidden = self._encode(token_ids)
        lm_logits = self.lm_head(hidden)

        pooled_plain = hidden[:, -1, :]
        pooled_result = self._encode(result_ids)[:, -1, :]
        pooled_no_result = self._encode(no_result_ids)[:, -1, :]
        fitness_scores = torch.stack(
            [
                self.fitness_proj(pooled_plain)[:, 0],
                self.fitness_proj(pooled_result)[:, 1],
                self.fitness_proj(pooled_no_result)[:, 2],
            ],
            dim=-1,
        )
        return lm_logits, fitness_scores


def build_toolformer() -> nn.Module:
    """Build a small Toolformer causal LM with an API-call fitness-scoring head.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return Toolformer().eval()


def example_input_toolformer() -> tuple[Tensor, Tensor, Tensor]:
    """Example plain / with-result / without-result token id sequences.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(token_ids, result_ids, no_result_ids)``, each ``(2, 16)`` long tensors.
    """
    token_ids = torch.randint(0, 96, (2, 16))
    result_ids = torch.randint(0, 96, (2, 16))
    no_result_ids = torch.randint(0, 96, (2, 16))
    return token_ids, result_ids, no_result_ids


# ---------------------------------------------------------------------------
# ToolkenGPT
# ---------------------------------------------------------------------------


class ToolkenGPT(nn.Module):
    """ToolkenGPT (Hao et al., NeurIPS 2023): frozen-LM body + augmented toolken head.

    Parameters
    ----------
    vocab_size : int
        Ordinary (frozen) vocabulary size.
    n_tools : int
        Number of tool ("toolken") entries appended to the LM head.
    dim : int
        Model width.
    n_layers : int
        Number of decoder blocks in the (simulated-frozen) LM body.
    n_heads : int
        Number of attention heads.
    max_len : int
        Maximum sequence length.
    """

    def __init__(
        self,
        vocab_size: int = 64,
        n_tools: int = 6,
        dim: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        max_len: int = 24,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.n_tools = n_tools

        # Simulated frozen LM body (frozen in ToolkenGPT training; left trainable here
        # since this is an architecture catalog, not a checkpoint -- the FROZEN-BODY
        # design constraint is expressed by the lm_head weights below being untouched
        # by the toolken-specific parameters).
        self.tok_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_len, dim)
        self.blocks = nn.ModuleList([_DecoderBlock(dim, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Toolken embeddings: a small, separately-trained embedding table whose rows
        # are APPENDED to the frozen LM head's weight matrix at the vocabulary
        # dimension, so tool selection is literally next-token prediction over an
        # augmented head (ToolkenGPT's core design choice, vs. e.g. a separate router).
        self.toolken_embed = nn.Embedding(n_tools, dim)

        # Tool-mode argument-filling head: once a toolken is selected, a small head
        # over the same hidden state fills in the call's argument value.
        self.arg_head = nn.Linear(dim, vocab_size)

    def forward(self, token_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Run reasoning-mode decoding with the vocabulary augmented by toolkens.

        Parameters
        ----------
        token_ids : Tensor
            ``(batch, seq_len)`` long ordinary-vocabulary token ids.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(augmented_logits, arg_logits)`` where ``augmented_logits`` is
            ``(batch, seq_len, vocab_size + n_tools)`` -- the ordinary LM-head logits
            concatenated with the toolken logits at every position (reasoning mode) --
            and ``arg_logits`` is ``(batch, seq_len, vocab_size)``, the tool-mode
            argument-filling distribution computed from the same hidden states.
        """
        b, t = token_ids.shape
        pos = torch.arange(t, device=token_ids.device).unsqueeze(0).expand(b, -1)
        x = self.tok_embed(token_ids) + self.pos_embed(pos)
        for block in self.blocks:
            x = block(x)
        hidden = self.ln_f(x)

        word_logits = self.lm_head(hidden)  # (b, t, vocab_size)
        toolken_weight = self.toolken_embed.weight  # (n_tools, dim)
        tool_logits = hidden @ toolken_weight.t()  # (b, t, n_tools)
        augmented_logits = torch.cat([word_logits, tool_logits], dim=-1)

        arg_logits = self.arg_head(hidden)
        return augmented_logits, arg_logits


def build_toolkengpt() -> nn.Module:
    """Build a small ToolkenGPT model with an augmented toolken LM head.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return ToolkenGPT().eval()


def example_input_toolkengpt() -> Tensor:
    """Example ordinary-vocabulary token id sequence.

    Returns
    -------
    Tensor
        ``(2, 12)`` long token ids.
    """
    return torch.randint(0, 64, (2, 12))


# ---------------------------------------------------------------------------
# ToolLLM / ToolLLaMA
# ---------------------------------------------------------------------------


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class _RMSNorm(nn.Module):
    """Root-mean-square layer normalization (LLaMA-style, no mean-subtraction).

    Parameters
    ----------
    dim : int
        Feature dimensionality.
    eps : float
        Numerical-stability epsilon.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Apply RMSNorm.

        Parameters
        ----------
        x : Tensor
            Input of any shape with the last dim equal to ``dim``.

        Returns
        -------
        Tensor
            Normalized-and-scaled output, same shape as ``x``.
        """
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm * self.weight


class _RotaryAttention(nn.Module):
    """Causal self-attention with rotary position embeddings (LLaMA-style).

    Parameters
    ----------
    dim : int
        Model width.
    n_heads : int
        Number of attention heads.
    """

    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rope(self, x: Tensor, t: int) -> Tensor:
        pos = torch.arange(t, device=x.device).float()
        freqs = torch.outer(pos, self.inv_freq)  # type: ignore[arg-type]
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        return x * cos + _rotate_half(x) * sin

    def forward(self, x: Tensor) -> Tensor:
        """Apply rotary causal self-attention.

        Parameters
        ----------
        x : Tensor
            ``(batch, seq_len, dim)`` input.

        Returns
        -------
        Tensor
            ``(batch, seq_len, dim)`` attended output.
        """
        b, t, c = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        q = self._rope(q, t)
        k = self._rope(k, t)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(torch.full((t, t), float("-inf"), device=x.device), diagonal=1)
        attn = F.softmax(scores + causal_mask, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, t, c)
        return self.o_proj(out)


class _SwiGLU(nn.Module):
    """SwiGLU gated MLP (LLaMA-style feed-forward block).

    Parameters
    ----------
    dim : int
        Model width.
    hidden_dim : int
        Gated hidden width.
    """

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the SwiGLU feed-forward transform.

        Parameters
        ----------
        x : Tensor
            ``(batch, seq_len, dim)`` input.

        Returns
        -------
        Tensor
            ``(batch, seq_len, dim)`` output.
        """
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _LlamaBlock(nn.Module):
    """LLaMA-style decoder block: RMSNorm + rotary attention + RMSNorm + SwiGLU.

    Parameters
    ----------
    dim : int
        Model width.
    n_heads : int
        Number of attention heads.
    """

    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        self.attn_norm = _RMSNorm(dim)
        self.attn = _RotaryAttention(dim, n_heads)
        self.mlp_norm = _RMSNorm(dim)
        self.mlp = _SwiGLU(dim, 4 * dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply one LLaMA-style decoder block.

        Parameters
        ----------
        x : Tensor
            ``(batch, seq_len, dim)`` input.

        Returns
        -------
        Tensor
            ``(batch, seq_len, dim)`` block output.
        """
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class ToolLLM(nn.Module):
    """ToolLLM / ToolLLaMA (Qin et al., ICLR 2024): function-calling LLaMA decoder.

    Parameters
    ----------
    vocab_size : int
        Token vocabulary size.
    n_roles : int
        Number of structured role segments (Thought / API Name / API Input /
        Observation).
    n_apis : int
        Number of distinct APIs in the (small, illustrative) tool registry.
    dim : int
        Model width.
    n_layers : int
        Number of decoder blocks.
    n_heads : int
        Number of attention heads.
    """

    def __init__(
        self,
        vocab_size: int = 64,
        n_roles: int = 4,
        n_apis: int = 8,
        dim: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, dim)
        self.role_embed = nn.Embedding(n_roles, dim)
        self.blocks = nn.ModuleList([_LlamaBlock(dim, n_heads) for _ in range(n_layers)])
        self.norm_f = _RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Dedicated API-name selection head reading the same trunk hidden state, so a
        # single decoder emits interleaved reasoning tokens and structured API calls.
        self.api_head = nn.Linear(dim, n_apis)

    def forward(self, token_ids: Tensor, role_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Run the function-calling decoder over interleaved role-tagged tokens.

        Parameters
        ----------
        token_ids : Tensor
            ``(batch, seq_len)`` long token ids.
        role_ids : Tensor
            ``(batch, seq_len)`` long role-segment ids (Thought/API-name/API-input/
            Observation) tagging each token position.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(lm_logits, api_logits)``: ``lm_logits`` is
            ``(batch, seq_len, vocab_size)`` next-token logits and ``api_logits`` is
            ``(batch, seq_len, n_apis)`` API-selection logits from the same trunk.
        """
        x = self.tok_embed(token_ids) + self.role_embed(role_ids)
        for block in self.blocks:
            x = block(x)
        hidden = self.norm_f(x)
        return self.lm_head(hidden), self.api_head(hidden)


def build_toolllm() -> nn.Module:
    """Build a small ToolLLM/ToolLLaMA function-calling LLaMA-style decoder.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return ToolLLM().eval()


def example_input_toolllm() -> tuple[Tensor, Tensor]:
    """Example token ids and role-segment ids.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(token_ids, role_ids)``, each ``(2, 16)`` long tensors (``role_ids`` in
        ``[0, 4)``).
    """
    token_ids = torch.randint(0, 64, (2, 16))
    role_ids = torch.randint(0, 4, (2, 16))
    return token_ids, role_ids


# ---------------------------------------------------------------------------
# TRADE
# ---------------------------------------------------------------------------


class TRADE(nn.Module):
    """TRADE (Wu et al., ACL 2019): shared encoder + per-slot gate & copy generator.

    Parameters
    ----------
    vocab_size : int
        Shared dialogue-history / value token vocabulary size.
    n_domain_slots : int
        Number of (domain, slot) pairs tracked.
    embed_dim : int
        Token embedding dimensionality.
    hidden_dim : int
        Encoder/decoder GRU hidden size.
    """

    def __init__(
        self,
        vocab_size: int = 60,
        n_domain_slots: int = 5,
        embed_dim: int = 20,
        hidden_dim: int = 28,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.n_domain_slots = n_domain_slots
        self.hidden_dim = hidden_dim

        self.embed = nn.Embedding(vocab_size, embed_dim)
        # Single bidirectional-GRU dialogue encoder SHARED across all (domain, slot)
        # pairs -- TRADE's transferability comes from not having a per-slot encoder.
        self.encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.encoder_bridge = nn.Linear(2 * hidden_dim, hidden_dim)

        # Learned per-(domain, slot) embedding, used both to drive the 3-way slot gate
        # and to seed the value-decoder's first input.
        self.domain_slot_embed = nn.Embedding(n_domain_slots, embed_dim)

        # 3-way slot gate: {ptr, dontcare, none} for each (domain, slot) pair.
        self.gate_attn = nn.Linear(embed_dim + hidden_dim, 2 * hidden_dim)
        self.gate_proj = nn.Linear(2 * hidden_dim, 3)

        # Per-slot GRU value generator with a soft copy/generate blend.
        self.decoder_cell = nn.GRUCell(embed_dim, hidden_dim)
        self.decoder_query = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.decoder_attn = nn.Linear(2 * hidden_dim, hidden_dim)
        self.gen_proj = nn.Linear(hidden_dim, vocab_size)
        self.copy_gate = nn.Linear(hidden_dim, 1)

    def _decode_slot(
        self, slot_seed: Tensor, encoder_states: Tensor, source_ids: Tensor, target_ids: Tensor
    ) -> Tensor:
        bsz, tgt_len = target_ids.shape
        hidden = slot_seed
        prev_token = torch.zeros(bsz, dtype=torch.long, device=target_ids.device)
        outputs = []
        for t in range(tgt_len):
            token_embed = self.embed(prev_token)
            hidden = self.decoder_cell(token_embed, hidden)

            query = self.decoder_query(hidden)  # (b, 2*hidden_dim)
            scores = torch.bmm(encoder_states, query.unsqueeze(-1)).squeeze(-1)  # (b, src_len)
            attn_weights = F.softmax(scores, dim=-1)
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_states).squeeze(
                1
            )  # (b, 2*hidden_dim)

            gen_dist = F.softmax(self.gen_proj(hidden + self.decoder_attn(context)), dim=-1)
            p_copy = torch.sigmoid(self.copy_gate(hidden))
            copy_dist = torch.zeros(bsz, self.vocab_size, device=target_ids.device)
            copy_dist.scatter_add_(1, source_ids, attn_weights)

            outputs.append((1 - p_copy) * gen_dist + p_copy * copy_dist)
            prev_token = target_ids[:, t]
        return torch.stack(outputs, dim=1)  # (b, tgt_len, vocab_size)

    def forward(self, dialogue_ids: Tensor, value_target_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Encode the dialogue once and decode a gate + value per (domain, slot) pair.

        Parameters
        ----------
        dialogue_ids : Tensor
            ``(batch, dial_len)`` long dialogue-history token ids.
        value_target_ids : Tensor
            ``(batch, n_domain_slots, value_len)`` teacher-forced target value token
            ids per (domain, slot) pair.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(gate_logits, value_dists)``: ``gate_logits`` is
            ``(batch, n_domain_slots, 3)`` and ``value_dists`` is
            ``(batch, n_domain_slots, value_len, vocab_size)``.
        """
        embedded = self.embed(dialogue_ids)
        raw_states, final = self.encoder(embedded)
        # (batch, dial_len, 2*hidden) bidirectional states used for attention/copy.
        encoder_states = raw_states
        init_hidden = torch.tanh(self.encoder_bridge(torch.cat([final[0], final[1]], dim=-1)))

        bsz = dialogue_ids.shape[0]
        ds_ids = torch.arange(self.n_domain_slots, device=dialogue_ids.device)
        ds_embed = self.domain_slot_embed(ds_ids)  # (n_domain_slots, embed_dim)

        gate_logits_list = []
        value_dists_list = []
        for s in range(self.n_domain_slots):
            slot_vec = ds_embed[s].unsqueeze(0).expand(bsz, -1)  # (b, embed_dim)

            gate_scores = torch.bmm(
                encoder_states,
                self.gate_attn(torch.cat([slot_vec, init_hidden], dim=-1)).unsqueeze(-1),
            ).squeeze(-1)
            gate_weights = F.softmax(gate_scores, dim=-1)
            gate_context = torch.bmm(gate_weights.unsqueeze(1), encoder_states).squeeze(1)
            gate_logits_list.append(self.gate_proj(gate_context))

            value_dists_list.append(
                self._decode_slot(
                    init_hidden, encoder_states, dialogue_ids, value_target_ids[:, s, :]
                )
            )

        gate_logits = torch.stack(gate_logits_list, dim=1)
        value_dists = torch.stack(value_dists_list, dim=1)
        return gate_logits, value_dists


def build_trade() -> nn.Module:
    """Build a small TRADE dialogue-state-tracking model.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return TRADE().eval()


def example_input_trade() -> tuple[Tensor, Tensor]:
    """Example dialogue-history ids and per-slot value target ids.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(dialogue_ids, value_target_ids)``: ``dialogue_ids`` is ``(2, 14)`` and
        ``value_target_ids`` is ``(2, 5, 4)`` long tensors.
    """
    dialogue_ids = torch.randint(0, 60, (2, 14))
    value_target_ids = torch.randint(0, 60, (2, 5, 4))
    return dialogue_ids, value_target_ids


MENAGERIE_ENTRIES = [
    ("Toolformer", "build_toolformer", "example_input_toolformer", "2023", "NLP"),
    ("ToolkenGPT", "build_toolkengpt", "example_input_toolkengpt", "2023", "NLP"),
    ("ToolLLaMA", "build_toolllm", "example_input_toolllm", "2023", "NLP"),
    ("TRADE", "build_trade", "example_input_trade", "2019", "NLP"),
]
