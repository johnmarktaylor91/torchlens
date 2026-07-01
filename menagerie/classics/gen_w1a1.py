"""Compact faithful reimplementations for build_queue rows 7-12 (W1A1).

Sources checked (repo browsed via ``gh api``, no clone/pip-install):
  - Capsule-NLU: https://github.com/czhang99/Capsule-NLU (``module.py``,
    ``create_model.py``) + arxiv:1812.09471 (Zhang et al., ACL'19). Dynamic
    routing-by-agreement capsule network: WordCapsule -> SlotCapsule (routing)
    -> IntentCapsule (routing over slot capsules), with an explicit
    "re-routing" pass that feeds intent-capsule information back into the
    slot-capsule routing logits.
  - CDial-GPT: https://github.com/thu-coai/CDial-GPT + arxiv:2008.03946
    (Wang et al., 2020). Standard GPT-2 causal LM pretrained on Chinese
    dialogue (LCCC); reimplemented via ``transformers.GPT2LMHeadModel`` from
    a tiny ``GPT2Config`` as instructed for library-config candidates.
  - ChatPLUG: https://github.com/X-PLUG/ChatPLUG + arxiv:2304.07849 (Tian
    et al., 2023). Internet/persona-augmented open-domain dialogue system
    built on a Fusion-in-Decoder (FiD) architecture: each of N retrieved
    knowledge/persona passages is independently run through a shared T5-style
    encoder, the per-passage encoded states are concatenated along the
    sequence axis, and a single decoder cross-attends jointly over the fused
    representation (Izacard & Grave FiD mechanism, per ``ChatPLUG.md``).
  - Co-Interactive Transformer (DCA-Net): https://github.com/kangbrilliant/
    DCA-Net (``model/joint_model_trans.py``) + arxiv:2010.03880 (Qin et al.,
    ICASSP 2021). BiLSTM encoder, label-embedding attention seed, then
    stacked co-interactive blocks where intent/slot streams cross-attend via
    dedicated query/key/value projections so each stream's context is built
    from the *other* stream's values, followed by a windowed
    (t-1, t, t+1) fusion feed-forward that writes back into both streams.
  - ConveRT: https://github.com/davidalami/ConveRT + arxiv:1911.03688
    (Henderson et al., PolyAI). Weight-tied dual-encoder Transformer sentence
    encoder: subword + positional embeddings, a SINGLE-HEADED self-attention
    transformer stack shared between the context and reply towers (one
    encoder module/parameters used for both sides), and a final
    self-attentive pooling layer that softmax-weights token states into a
    single sentence embedding, projected to a response-selection space by a
    small per-tower head (paper Fig. 1; original PolyAI TF checkpoints
    referenced from the ``conversational_sentence_encoder`` wrapper,
    reimplemented natively here in torch).
  - CoS-E: https://github.com/salesforce/cos-e (``code/generation``) +
    arxiv:1906.02761 (Rajani et al., ACL'19). A causal LM fine-tuned to
    generate free-form commonsense explanations conditioned on a
    question+answer-choices prompt; reimplemented as a tiny
    ``transformers.GPT2LMHeadModel`` matching the repo's generator setup.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel, T5Config
from transformers.models.t5.modeling_t5 import T5Stack


# ---------------------------------------------------------------------------
# Capsule-NLU: dynamic-routing capsule network for joint intent + slot filling
# ---------------------------------------------------------------------------


def _squash(caps: Tensor) -> Tensor:
    """Apply the capsule "squash" norm nonlinearity.

    Parameters
    ----------
    caps:
        Capsule pose tensor of shape ``(..., atoms)``.

    Returns
    -------
    Tensor
        Squashed capsules with the same shape, norm in ``[0, 1)``.
    """

    norm = caps.norm(dim=-1, keepdim=True)
    norm_sq = norm * norm
    return (caps / (norm + 1e-8)) * (norm_sq / (1.0 + norm_sq))


class CapsuleLayer(nn.Module):
    """Fully-connected capsule layer with dynamic routing-by-agreement."""

    def __init__(
        self,
        input_dim: int,
        input_atoms: int,
        output_dim: int,
        output_atoms: int,
        num_routing: int = 3,
    ) -> None:
        """Initialize the transform weights and routing bias.

        Parameters
        ----------
        input_dim:
            Number of input capsules.
        input_atoms:
            Pose-vector width of each input capsule.
        output_dim:
            Number of output capsules.
        output_atoms:
            Pose-vector width of each output capsule.
        num_routing:
            Number of routing-by-agreement iterations.
        """

        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_atoms = output_atoms
        self.num_routing = num_routing
        self.weight = nn.Parameter(
            torch.randn(1, input_dim, output_dim, input_atoms, output_atoms) * 0.1
        )
        self.bias = nn.Parameter(torch.zeros(output_dim, output_atoms))

    def forward(self, x: Tensor, extra_logits: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Route input capsules to output capsules.

        Parameters
        ----------
        x:
            Input capsules of shape ``(batch, input_dim, input_atoms)``.
        extra_logits:
            Optional additive routing-logit bias of shape
            ``(batch, input_dim, output_dim)``, used for the intent-guided
            "re-routing" pass over slot capsules.

        Returns
        -------
        tuple[Tensor, Tensor]
            Output capsule activations ``(batch, output_dim, output_atoms)``
            and the final routing logits ``(batch, input_dim, output_dim)``.
        """

        batch = x.size(0)
        votes = torch.einsum("bic,ioca->bioa", x, self.weight.squeeze(0))
        # votes: (batch, input_dim, output_dim, output_atoms)
        logits = x.new_zeros(batch, self.input_dim, self.output_dim)
        if extra_logits is not None:
            logits = logits + extra_logits
        activation = votes.new_zeros(batch, self.output_dim, self.output_atoms)
        for _ in range(self.num_routing):
            route = F.softmax(logits, dim=2)
            preact = (route.unsqueeze(-1) * votes).sum(dim=1) + self.bias
            activation = _squash(preact)
            agreement = torch.einsum("bioa,boa->bio", votes, activation)
            logits = logits + agreement
        return activation, logits


class CapsuleNLU(nn.Module):
    """Capsule-NLU: joint intent detection + slot filling via capsule routing."""

    def __init__(
        self,
        vocab_size: int = 64,
        embed_dim: int = 16,
        hidden_dim: int = 16,
        num_slots: int = 8,
        num_intents: int = 6,
    ) -> None:
        """Initialize embedding, BiLSTM word encoder, and capsule stack.

        Parameters
        ----------
        vocab_size:
            Token vocabulary size.
        embed_dim:
            Word-embedding width.
        hidden_dim:
            Per-direction BiLSTM hidden width (word-capsule atom width).
        num_slots:
            Number of slot-label capsules.
        num_intents:
            Number of intent-label capsules.
        """

        super().__init__()
        atoms = hidden_dim * 2
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.slot_caps = CapsuleLayer(1, atoms, num_slots, atoms, num_routing=2)
        self.intent_caps = CapsuleLayer(num_slots, atoms, num_intents, atoms, num_routing=2)
        self.rerouting_proj = nn.Linear(num_intents, num_slots)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        """Run word capsules through slot- and intent-capsule routing.

        Parameters
        ----------
        tokens:
            Token ids of shape ``(batch, seq_len)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Slot logits ``(batch, seq_len, num_slots)`` and intent logits
            ``(batch, num_intents)``.
        """

        word_caps, _ = self.rnn(self.embed(tokens))
        batch, seq_len, atoms = word_caps.shape
        flat_words = word_caps.reshape(batch * seq_len, 1, atoms)
        slot_acts, _ = self.slot_caps(flat_words)
        slot_acts = slot_acts.reshape(batch, seq_len, -1, atoms)
        slot_logits = slot_acts.norm(dim=-1)

        pooled_slots = slot_acts.mean(dim=1)
        intent_acts, intent_logits_raw = self.intent_caps(pooled_slots)
        intent_logits = intent_acts.norm(dim=-1)

        # Re-routing: feed intent-capsule agreement back as extra routing
        # bias for a second slot-capsule pass, matching create_model.py's
        # ``re_routing=True`` path.
        reroute_bias = self.rerouting_proj(intent_logits).unsqueeze(1).expand(-1, seq_len, -1)
        reroute_bias = reroute_bias.reshape(batch * seq_len, 1, -1)
        slot_acts2, _ = self.slot_caps(flat_words, extra_logits=reroute_bias)
        slot_logits2 = slot_acts2.reshape(batch, seq_len, -1, atoms).norm(dim=-1)
        return slot_logits2, intent_logits


def build_capsule_nlu() -> nn.Module:
    """Build a small Capsule-NLU joint intent/slot model.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """

    return CapsuleNLU().eval()


def example_capsule_nlu() -> Tensor:
    """Example token-id sequence for Capsule-NLU.

    Returns
    -------
    Tensor
        Long tensor of shape ``(2, 10)``.
    """

    return torch.randint(0, 64, (2, 10))


# ---------------------------------------------------------------------------
# CDial-GPT: GPT-2 causal LM pretrained for Chinese open-domain dialogue
# ---------------------------------------------------------------------------


def build_cdial_gpt() -> nn.Module:
    """Build a tiny CDial-GPT-style GPT-2 causal LM.

    Returns
    -------
    nn.Module
        ``GPT2LMHeadModel`` in eval mode.
    """

    cfg = GPT2Config(
        vocab_size=128,
        n_positions=32,
        n_embd=32,
        n_layer=2,
        n_head=2,
        n_inner=64,
        type_vocab_size=2,
    )
    return GPT2LMHeadModel(cfg).eval()


def example_cdial_gpt() -> Tensor:
    """Example dialogue token sequence for CDial-GPT.

    Returns
    -------
    Tensor
        Long tensor of shape ``(1, 16)``.
    """

    return torch.randint(0, 128, (1, 16))


# ---------------------------------------------------------------------------
# ChatPLUG: Fusion-in-Decoder open-domain dialogue system
# ---------------------------------------------------------------------------


class ChatPLUGFiD(nn.Module):
    """Fusion-in-Decoder dialogue model (persona/knowledge fusion)."""

    def __init__(
        self,
        vocab_size: int = 96,
        d_model: int = 24,
        n_layer: int = 2,
        n_head: int = 4,
        n_passages: int = 3,
        passage_len: int = 6,
    ) -> None:
        """Initialize the shared T5 encoder/decoder and fusion bookkeeping.

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
        n_passages:
            Number of independently-encoded knowledge/persona passages that
            get fused in the decoder (the FiD mechanism).
        passage_len:
            Token length of each passage.
        """

        super().__init__()
        self.n_passages = n_passages
        self.passage_len = passage_len
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
        enc_cfg = shared_cfg
        dec_cfg = T5Config(
            **{**shared_cfg.to_dict(), "is_decoder": True, "is_encoder_decoder": False}
        )
        self.encoder = T5Stack(enc_cfg, embed_tokens=self.shared_embed)
        self.decoder = T5Stack(dec_cfg, embed_tokens=self.shared_embed)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, passages: Tensor, decoder_input_ids: Tensor) -> Tensor:
        """Encode each passage independently, fuse, then decode.

        Parameters
        ----------
        passages:
            Token ids of shape ``(batch, n_passages, passage_len)`` — e.g.
            retrieved knowledge snippets and persona sentences.
        decoder_input_ids:
            Decoder token ids of shape ``(batch, tgt_len)``.

        Returns
        -------
        Tensor
            Next-token logits of shape ``(batch, tgt_len, vocab_size)``.
        """

        batch, n_passages, passage_len = passages.shape
        flat_passages = passages.reshape(batch * n_passages, passage_len)
        encoded = self.encoder(input_ids=flat_passages).last_hidden_state
        # Fusion-in-Decoder: concatenate all per-passage encodings along the
        # sequence axis before the decoder cross-attends over the union.
        fused = encoded.reshape(batch, n_passages * passage_len, -1)
        dec_out = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=fused,
        ).last_hidden_state
        return self.lm_head(dec_out)


def build_chatplug_fid() -> nn.Module:
    """Build a small ChatPLUG-style Fusion-in-Decoder model.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """

    return ChatPLUGFiD().eval()


def example_chatplug_fid() -> tuple[Tensor, Tensor]:
    """Example fused-passage and decoder inputs for ChatPLUG.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(passages, decoder_input_ids)`` long tensors.
    """

    passages = torch.randint(0, 96, (1, 3, 6))
    decoder_input_ids = torch.randint(0, 96, (1, 5))
    return passages, decoder_input_ids


# ---------------------------------------------------------------------------
# Co-Interactive Transformer (DCA-Net): joint slot filling + intent detection
# ---------------------------------------------------------------------------


class CoInteractiveAttention(nn.Module):
    """Cross-stream self-attention: each stream's query attends the other's keys/values."""

    def __init__(self, hidden_size: int, num_heads: int = 4) -> None:
        """Initialize per-stream query/key/value projections.

        Parameters
        ----------
        hidden_size:
            Shared feature width of the intent and slot streams.
        num_heads:
            Attention head count.
        """

        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_intent = nn.Linear(hidden_size, hidden_size)
        self.k_intent = nn.Linear(hidden_size, hidden_size)
        self.v_intent = nn.Linear(hidden_size, hidden_size)
        self.q_slot = nn.Linear(hidden_size, hidden_size)
        self.k_slot = nn.Linear(hidden_size, hidden_size)
        self.v_slot = nn.Linear(hidden_size, hidden_size)

    def _split_heads(self, x: Tensor) -> Tensor:
        """Reshape ``(batch, seq, hidden)`` into ``(batch, heads, seq, head_dim)``."""

        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(self, h_intent: Tensor, h_slot: Tensor) -> tuple[Tensor, Tensor]:
        """Cross-attend the intent and slot streams.

        Parameters
        ----------
        h_intent:
            Intent-stream features ``(batch, seq_len, hidden)``.
        h_slot:
            Slot-stream features ``(batch, seq_len, hidden)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Updated ``(h_intent, h_slot)`` context tensors.
        """

        q_i = self._split_heads(self.q_intent(h_intent))
        k_s = self._split_heads(self.k_intent(h_slot))
        v_s = self._split_heads(self.v_intent(h_slot))
        q_s = self._split_heads(self.q_slot(h_slot))
        k_i = self._split_heads(self.k_slot(h_intent))
        v_i = self._split_heads(self.v_slot(h_intent))

        scale = math.sqrt(self.head_dim)
        attn_i = F.softmax(torch.matmul(q_i, k_s.transpose(-1, -2)) / scale, dim=-1)
        ctx_i = torch.matmul(attn_i, v_s)
        attn_s = F.softmax(torch.matmul(q_s, k_i.transpose(-1, -2)) / scale, dim=-1)
        ctx_s = torch.matmul(attn_s, v_i)

        batch, _, seq_len, _ = ctx_i.shape
        ctx_i = ctx_i.permute(0, 2, 1, 3).reshape(batch, seq_len, -1)
        ctx_s = ctx_s.permute(0, 2, 1, 3).reshape(batch, seq_len, -1)
        return ctx_i, ctx_s


class CoInteractiveBlock(nn.Module):
    """One co-interactive layer: cross-attention + windowed fusion feed-forward."""

    def __init__(self, hidden_size: int) -> None:
        """Initialize the cross-attention and fusion sublayers.

        Parameters
        ----------
        hidden_size:
            Shared feature width of the intent and slot streams.
        """

        super().__init__()
        self.attn = CoInteractiveAttention(hidden_size)
        self.norm_i = nn.LayerNorm(hidden_size)
        self.norm_s = nn.LayerNorm(hidden_size)
        self.fuse = nn.Linear(hidden_size * 6, hidden_size)
        self.fuse_norm_i = nn.LayerNorm(hidden_size)
        self.fuse_norm_s = nn.LayerNorm(hidden_size)

    def forward(self, h_intent: Tensor, h_slot: Tensor) -> tuple[Tensor, Tensor]:
        """Apply co-interactive cross-attention then windowed fusion.

        Parameters
        ----------
        h_intent:
            Intent-stream features ``(batch, seq_len, hidden)``.
        h_slot:
            Slot-stream features ``(batch, seq_len, hidden)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Updated ``(h_intent, h_slot)`` tensors after fusion.
        """

        ctx_i, ctx_s = self.attn(h_intent, h_slot)
        h_intent = self.norm_i(ctx_i + h_intent)
        h_slot = self.norm_s(ctx_s + h_slot)

        cat = torch.cat([h_intent, h_slot], dim=-1)
        pad = cat.new_zeros(cat.size(0), 1, cat.size(2))
        left = torch.cat([pad, cat[:, :-1, :]], dim=1)
        right = torch.cat([cat[:, 1:, :], pad], dim=1)
        windowed = torch.cat([cat, left, right], dim=-1)
        fused = F.relu(self.fuse(windowed))
        h_intent = self.fuse_norm_i(fused + h_intent)
        h_slot = self.fuse_norm_s(fused + h_slot)
        return h_intent, h_slot


class CoInteractiveTransformer(nn.Module):
    """Co-Interactive Transformer (DCA-Net) for joint intent + slot filling."""

    def __init__(
        self,
        vocab_size: int = 64,
        embed_dim: int = 16,
        hidden_dim: int = 16,
        num_slots: int = 8,
        num_intents: int = 6,
        n_blocks: int = 2,
    ) -> None:
        """Initialize the BiLSTM encoder, label attention, and co-interactive stack.

        Parameters
        ----------
        vocab_size:
            Token vocabulary size.
        embed_dim:
            Word-embedding width.
        hidden_dim:
            Per-direction BiLSTM hidden width.
        num_slots:
            Number of slot labels.
        num_intents:
            Number of intent labels.
        n_blocks:
            Number of stacked co-interactive blocks.
        """

        super().__init__()
        hidden = hidden_dim * 2
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.intent_fc = nn.Linear(hidden, num_intents)
        self.slot_fc = nn.Linear(hidden, num_slots)
        self.blocks = nn.ModuleList([CoInteractiveBlock(hidden) for _ in range(n_blocks)])

    def _label_attend(self, h_intent: Tensor, h_slot: Tensor) -> tuple[Tensor, Tensor]:
        """Softmax-attend hidden states over the intent/slot label embeddings."""

        intent_probs = F.softmax(h_intent @ self.intent_fc.weight.t(), dim=-1)
        slot_probs = F.softmax(h_slot @ self.slot_fc.weight.t(), dim=-1)
        intent_res = intent_probs @ self.intent_fc.weight
        slot_res = slot_probs @ self.slot_fc.weight
        return intent_res, slot_res

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        """Run the BiLSTM encoder then stacked co-interactive blocks.

        Parameters
        ----------
        tokens:
            Token ids of shape ``(batch, seq_len)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Intent logits ``(batch, num_intents)`` and slot logits
            ``(batch, seq_len, num_slots)``.
        """

        h, _ = self.rnn(self.embed(tokens))
        h_intent, h_slot = self._label_attend(h, h)
        h_intent = h_intent + h
        h_slot = h_slot + h
        for block in self.blocks:
            h_intent, h_slot = block(h_intent, h_slot)
            label_i, label_s = self._label_attend(h_intent, h_slot)
            h_intent = h_intent + label_i
            h_slot = h_slot + label_s

        pooled_intent = F.max_pool1d(h_intent.transpose(1, 2), h_intent.size(1)).squeeze(2)
        intent_logits = self.intent_fc(pooled_intent)
        slot_logits = self.slot_fc(h_slot)
        return intent_logits, slot_logits


def build_co_interactive_transformer() -> nn.Module:
    """Build a small Co-Interactive Transformer (DCA-Net) model.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """

    return CoInteractiveTransformer().eval()


def example_co_interactive_transformer() -> Tensor:
    """Example token-id sequence for the Co-Interactive Transformer.

    Returns
    -------
    Tensor
        Long tensor of shape ``(2, 10)``.
    """

    return torch.randint(0, 64, (2, 10))


# ---------------------------------------------------------------------------
# ConveRT: weight-tied dual-encoder Transformer with SINGLE-HEADED self-
# attention and self-attentive pooling (Henderson et al., PolyAI, 2019).
# ---------------------------------------------------------------------------


class SingleHeadSelfAttention(nn.Module):
    """Single-headed scaled dot-product self-attention (the paper's stated design)."""

    def __init__(self, dim: int) -> None:
        """Initialize the single-head query/key/value and output projections.

        Parameters
        ----------
        dim:
            Feature width. ConveRT's transformer blocks use ONE attention head
            (not multi-head) per Henderson et al.
        """

        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.scale = math.sqrt(dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply single-headed self-attention.

        Parameters
        ----------
        x:
            Sequence tensor of shape ``(batch, seq_len, dim)``.

        Returns
        -------
        Tensor
            Attention output of shape ``(batch, seq_len, dim)``.
        """

        q, k, v = self.q(x), self.k(x), self.v(x)
        attn = F.softmax(torch.matmul(q, k.transpose(-1, -2)) / self.scale, dim=-1)
        ctx = torch.matmul(attn, v)
        return self.out(ctx)


class ConveRTBlock(nn.Module):
    """One single-headed-attention transformer block with a feed-forward sublayer."""

    def __init__(self, dim: int) -> None:
        """Initialize attention and feed-forward sublayers.

        Parameters
        ----------
        dim:
            Feature width.
        """

        super().__init__()
        self.attn = SingleHeadSelfAttention(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply one single-headed-attention + feed-forward block.

        Parameters
        ----------
        x:
            Sequence tensor of shape ``(batch, seq_len, dim)``.

        Returns
        -------
        Tensor
            Updated sequence tensor.
        """

        x = self.norm1(x + self.attn(x))
        return self.norm2(x + self.ff(x))


class SelfAttentivePool(nn.Module):
    """Final self-attentive pooling layer collapsing a sequence to one vector."""

    def __init__(self, dim: int) -> None:
        """Initialize the pooling score projection.

        Parameters
        ----------
        dim:
            Input (and output) sequence feature width -- pooling alone does
            not change dimensionality; the per-tower projection heads (see
            ``ConveRTDualEncoder``) handle the final response-selection space.
        """

        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Softmax-pool the sequence into a single sentence embedding.

        Parameters
        ----------
        x:
            Sequence tensor of shape ``(batch, seq_len, dim)``.

        Returns
        -------
        Tensor
            Pooled embedding of shape ``(batch, dim)``.
        """

        weights = F.softmax(self.score(x), dim=1)
        return (weights * x).sum(dim=1)


class ConveRTEncoder(nn.Module):
    """ConveRT's SHARED sentence-encoder trunk (subword embed + single-head transformer stack).

    Per Henderson et al., the context and reply towers of the dual encoder are
    WEIGHT-TIED: they literally share this one encoder module/parameters
    (shared subword embedding + shared transformer stack). Only the final
    per-tower projection heads differ (see ``ConveRTDualEncoder``), so this
    class intentionally stops at the pooled sentence embedding and does not
    own a final response-selection projection.
    """

    def __init__(
        self,
        vocab_size: int = 128,
        ngram_buckets: int = 64,
        dim: int = 24,
        n_layers: int = 3,
        max_len: int = 32,
    ) -> None:
        """Initialize subword + n-gram-hash embeddings, transformer stack, and pooling.

        Parameters
        ----------
        vocab_size:
            Subword vocabulary size.
        ngram_buckets:
            Hash-bucket count for the character n-gram embedding table.
        dim:
            Transformer hidden width.
        n_layers:
            Number of single-headed-attention transformer blocks.
        max_len:
            Maximum sequence length (for positional embeddings).
        """

        super().__init__()
        self.subword_embed = nn.Embedding(vocab_size, dim)
        self.ngram_embed = nn.Embedding(ngram_buckets, dim)
        self.pos_embed = nn.Embedding(max_len, dim)
        self.blocks = nn.ModuleList([ConveRTBlock(dim) for _ in range(n_layers)])
        self.pool = SelfAttentivePool(dim)

    def forward(self, subword_ids: Tensor, ngram_ids: Tensor) -> Tensor:
        """Encode a batch of token sequences into pooled sentence embeddings.

        Parameters
        ----------
        subword_ids:
            Subword token ids of shape ``(batch, seq_len)``.
        ngram_ids:
            Hashed character n-gram ids of shape ``(batch, seq_len)``.

        Returns
        -------
        Tensor
            Pooled sentence embeddings of shape ``(batch, dim)``.
        """

        seq_len = subword_ids.size(1)
        positions = torch.arange(seq_len, device=subword_ids.device).unsqueeze(0)
        x = (
            self.subword_embed(subword_ids)
            + self.ngram_embed(ngram_ids)
            + self.pos_embed(positions)
        )
        for block in self.blocks:
            x = block(x)
        return self.pool(x)


class ConveRTDualEncoder(nn.Module):
    """ConveRT context/response dual-encoder for response selection.

    The context and reply towers share ONE ``ConveRTEncoder`` instance (true
    weight tying -- both towers reference the exact same parameters), and
    only the final small linear projection heads differ per side, matching
    the paper's "shared subword embedding + shared transformer stack, with
    only the final projection heads differing between context and reply
    sides" design.
    """

    def __init__(self, dim: int = 24, out_dim: int = 16) -> None:
        """Initialize the single shared encoder trunk and two per-tower projection heads."""

        super().__init__()
        # ONE shared encoder instance used by both towers below -- this is
        # the weight-tying itself, not just architecturally-identical twins.
        self.shared_encoder = ConveRTEncoder(dim=dim)
        self.context_head = nn.Linear(dim, out_dim)
        self.reply_head = nn.Linear(dim, out_dim)

    def forward(
        self,
        context_subwords: Tensor,
        context_ngrams: Tensor,
        response_subwords: Tensor,
        response_ngrams: Tensor,
    ) -> Tensor:
        """Encode context and response through the SHARED trunk, returning a similarity matrix.

        Parameters
        ----------
        context_subwords:
            Context subword ids ``(batch, seq_len)``.
        context_ngrams:
            Context n-gram-hash ids ``(batch, seq_len)``.
        response_subwords:
            Response subword ids ``(batch, seq_len)``.
        response_ngrams:
            Response n-gram-hash ids ``(batch, seq_len)``.

        Returns
        -------
        Tensor
            Dot-product similarity matrix of shape ``(batch, batch)`` -- the
            actual ConveRT response-selection training objective.
        """

        ctx_pooled = self.shared_encoder(context_subwords, context_ngrams)
        resp_pooled = self.shared_encoder(response_subwords, response_ngrams)
        ctx = self.context_head(ctx_pooled)
        resp = self.reply_head(resp_pooled)
        return ctx @ resp.t()


def build_convert() -> nn.Module:
    """Build a small ConveRT dual-encoder model.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """

    return ConveRTDualEncoder().eval()


def example_convert() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Example context/response subword and n-gram id tensors for ConveRT.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        ``(context_subwords, context_ngrams, response_subwords, response_ngrams)``.
    """

    ctx_sw = torch.randint(0, 128, (4, 12))
    ctx_ng = torch.randint(0, 64, (4, 12))
    resp_sw = torch.randint(0, 128, (4, 12))
    resp_ng = torch.randint(0, 64, (4, 12))
    return ctx_sw, ctx_ng, resp_sw, resp_ng


# ---------------------------------------------------------------------------
# CoS-E: causal LM generator fine-tuned for commonsense-explanation generation
# ---------------------------------------------------------------------------


def build_cos_e() -> nn.Module:
    """Build a tiny CoS-E-style rationale-generation GPT-2 LM.

    Returns
    -------
    nn.Module
        ``GPT2LMHeadModel`` in eval mode.
    """

    cfg = GPT2Config(vocab_size=112, n_positions=32, n_embd=32, n_layer=2, n_head=2, n_inner=64)
    return GPT2LMHeadModel(cfg).eval()


def example_cos_e() -> Tensor:
    """Example question+choices+rationale-prefix token sequence for CoS-E.

    Returns
    -------
    Tensor
        Long tensor of shape ``(1, 20)``.
    """

    return torch.randint(0, 112, (1, 20))


MENAGERIE_ENTRIES = [
    ("Capsule-NLU", "build_capsule_nlu", "example_capsule_nlu", "2019", "NLP"),
    ("CDial-GPT (Chinese Dialogue GPT)", "build_cdial_gpt", "example_cdial_gpt", "2020", "NLP"),
    ("ChatPLUG (Chinese Dialogue)", "build_chatplug_fid", "example_chatplug_fid", "2023", "NLP"),
    (
        "Co-Interactive Transformer",
        "build_co_interactive_transformer",
        "example_co_interactive_transformer",
        "2021",
        "NLP",
    ),
    (
        "ConveRT (Conversational Representation from Transformer)",
        "build_convert",
        "example_convert",
        "2019",
        "NLP",
    ),
    ("CoS-E", "build_cos_e", "example_cos_e", "2019", "NLP"),
]
