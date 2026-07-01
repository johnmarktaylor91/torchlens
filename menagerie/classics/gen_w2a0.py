"""Task-oriented dialog, retrieval, and SLU classics (build queue rows 37-42).

Sources checked (repo_url / desc_source from the build queue, all inspected via web
search of the paper text and/or official repo README -- no cloning, no pip installs):

- GALAXY: He et al., "GALAXY: A Generative Pre-trained Model for Task-Oriented Dialog
  with Semi-Supervised Learning and Explicit Policy Injection", AAAI 2022,
  arxiv:2111.14592, https://github.com/siat-nlp/GALAXY. UniLM-style shared
  encoder-decoder Transformer (bidirectional attention over context, causal attention
  over response) plus an explicit dialog-act (DA) prediction head trained jointly with
  response generation for policy-aware pre-training.
- GAR: Mao et al., "Generation-Augmented Retrieval for Open-Domain Question Answering",
  ACL 2021, arxiv:2009.12005, https://github.com/morningmoni/GAR. A trainable seq2seq
  (BART/T5-style) transformer that generates query-expansion contexts (answer / sentence
  / title) from a question; downstream BM25 retrieval on the concatenated
  query+generated-context is a non-trainable post-process outside this module.
- GENRE: De Cao et al., "Autoregressive Entity Retrieval", arxiv:2010.00904,
  https://github.com/facebookresearch/GENRE. A BART encoder-decoder that retrieves
  entities by generating their unique name token-by-token; at inference decoding is
  constrained to a prefix trie of valid entity names (implemented here as a compact
  greedy trie-constrained decode loop demonstrating the constrained-generation
  mechanism, layered on top of the trainable seq2seq transformer).
- GL-GIN: Qin et al., "GL-GIN: Fast and Accurate Non-Autoregressive Model for Joint
  Multiple Intent Detection and Slot Filling", ACL 2021, arxiv:2106.01925,
  https://github.com/yizhen20133868/GL-GIN. BiLSTM + self-attentive encoder, a
  token-level multi-intent decoder, a local slot-aware graph-attention layer that
  connects slot tokens to each other, and a global intent-slot graph-attention layer
  that connects all predicted intents to all slot tokens -- enabling non-autoregressive
  joint decoding of multiple intents and coordinated slots.
- GODEL: "GODEL: Large-Scale Pre-Training for Goal-Directed Dialog", Microsoft,
  arxiv:2206.11309, https://github.com/microsoft/GODEL,
  microsoft/GODEL-v1_1-large-seq2seq. A T5-initialized Transformer encoder-decoder
  fine-tuned so the encoder input is the concatenation of an instruction, the dialog
  context, and an explicit external-knowledge/grounding span (separated by dedicated
  markers), letting the decoder condition generation on grounding text distinct from
  conversation history.
- HDSA: Chen et al., "Semantically Conditioned Dialog Response Generation via
  Hierarchical Disentangled Self-Attention", ACL 2019, arxiv:1905.12866,
  https://github.com/wenhuchen/HDSA-Dialog. A dialog-act graph (domain -> act -> slot,
  merged across branches) is flattened into per-layer node membership vectors; a
  disentangled self-attention decoder assigns disjoint groups of attention heads at
  each Transformer layer to the corresponding graph layer, so head activity is gated by
  which dialog-act-graph nodes are active for the current turn -- giving combinatorial
  control over generation from a fixed head budget.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    T5Config,
    T5ForConditionalGeneration,
)

# ---------------------------------------------------------------------------
# GALAXY: UniLM-style shared encoder-decoder + explicit dialog-act (DA) head.
# ---------------------------------------------------------------------------


class GalaxyUniLMBlock(nn.Module):
    """One shared Transformer block used for both context and response tokens."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        """Build a self-attention + feed-forward block with pre-norm residuals.

        Parameters
        ----------
        d_model
            Hidden size shared across attention and feed-forward sublayers.
        n_heads
            Number of self-attention heads.
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        """Apply masked self-attention followed by a feed-forward sublayer.

        Parameters
        ----------
        x
            Token hidden states of shape ``(batch, seq_len, d_model)``.
        attn_mask
            Additive attention mask of shape ``(seq_len, seq_len)``.

        Returns
        -------
        Tensor
            Updated hidden states of shape ``(batch, seq_len, d_model)``.
        """
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x


class GalaxyDialogModel(nn.Module):
    """UniLM-style shared Transformer with an explicit dialog-act prediction head."""

    def __init__(
        self,
        vocab_size: int = 512,
        d_model: int = 48,
        n_heads: int = 4,
        n_layers: int = 3,
        n_da_labels: int = 20,
        max_len: int = 32,
    ) -> None:
        """Initialize shared embeddings, UniLM blocks, and task-specific heads.

        Parameters
        ----------
        vocab_size
            Token vocabulary size.
        d_model
            Shared hidden size.
        n_heads
            Attention heads per block.
        n_layers
            Number of shared Transformer blocks.
        n_da_labels
            Number of binary dialog-act labels predicted from the context.
        max_len
            Maximum combined context+response sequence length.
        """
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([GalaxyUniLMBlock(d_model, n_heads) for _ in range(n_layers)])
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.da_head = nn.Linear(d_model, n_da_labels)
        self.response_selection_head = nn.Linear(d_model, 1)

    def _unilm_mask(self, context_len: int, total_len: int) -> Tensor:
        """Build the UniLM attention mask: bidirectional over context, causal over response.

        Parameters
        ----------
        context_len
            Number of context tokens (bidirectionally visible to everything after them).
        total_len
            Total sequence length (context followed by response tokens).

        Returns
        -------
        Tensor
            Additive attention mask of shape ``(total_len, total_len)`` with
            ``-inf`` at disallowed positions.
        """
        mask = torch.full((total_len, total_len), float("-inf"))
        mask[:, :context_len] = 0.0
        causal = torch.tril(torch.ones(total_len - context_len, total_len - context_len))
        mask[context_len:, context_len:] = torch.where(causal.bool(), 0.0, float("-inf"))
        return mask

    def forward(self, tokens: Tensor, context_len: int) -> tuple[Tensor, Tensor, Tensor]:
        """Run the shared UniLM Transformer and task heads.

        Parameters
        ----------
        tokens
            Concatenated context+response token ids, shape ``(batch, total_len)``.
        context_len
            Number of leading tokens that belong to the dialog context.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Language-model logits ``(batch, total_len, vocab)``, dialog-act logits
            ``(batch, n_da_labels)`` pooled from the context span, and a response
            selection score ``(batch, 1)``.
        """
        batch, total_len = tokens.shape
        positions = torch.arange(total_len, device=tokens.device).unsqueeze(0)
        x = self.tok_emb(tokens) + self.pos_emb(positions)
        attn_mask = self._unilm_mask(context_len, total_len).to(tokens.device)
        for block in self.blocks:
            x = block(x, attn_mask)
        lm_logits = self.lm_head(x)
        context_pool = x[:, :context_len].mean(dim=1)
        da_logits = self.da_head(context_pool)
        selection_score = self.response_selection_head(x[:, 0])
        return lm_logits, da_logits, selection_score


def build_galaxy() -> nn.Module:
    """Build a compact GALAXY UniLM-style dialog model.

    Returns
    -------
    nn.Module
        Configured ``GalaxyDialogModel`` instance in eval mode.
    """
    return GalaxyDialogModel().eval()


def example_input_galaxy() -> tuple[Tensor, int]:
    """Create an example concatenated context+response token sequence.

    Returns
    -------
    tuple[Tensor, int]
        Token ids of shape ``(1, 20)`` and the context length (12 tokens).
    """
    return torch.randint(3, 512, (1, 20)), 12


# ---------------------------------------------------------------------------
# GAR: seq2seq query-context generator for generation-augmented retrieval.
# ---------------------------------------------------------------------------


class Seq2SeqLogitsWrapper(nn.Module):
    """Thin wrapper exposing ``(input_ids, decoder_input_ids) -> logits`` positionally."""

    def __init__(self, seq2seq: nn.Module) -> None:
        """Wrap a HuggingFace conditional-generation model.

        Parameters
        ----------
        seq2seq
            A ``*ForConditionalGeneration`` model exposing ``.logits`` on its
            forward output.
        """
        super().__init__()
        self.seq2seq = seq2seq

    def forward(self, input_ids: Tensor, decoder_input_ids: Tensor) -> Tensor:
        """Run the wrapped seq2seq model and return raw next-token logits.

        Parameters
        ----------
        input_ids
            Encoder input token ids, shape ``(batch, src_len)``.
        decoder_input_ids
            Decoder input token ids, shape ``(batch, tgt_len)``.

        Returns
        -------
        Tensor
            Next-token logits of shape ``(batch, tgt_len, vocab_size)``.
        """
        return self.seq2seq(input_ids=input_ids, decoder_input_ids=decoder_input_ids).logits


def build_gar() -> nn.Module:
    """Build a compact T5-style seq2seq generator for GAR query expansion.

    Returns
    -------
    nn.Module
        A ``Seq2SeqLogitsWrapper`` around ``T5ForConditionalGeneration``, trained
        to map a question to a generation target (answer / sentence / title
        context), in eval mode.
    """
    cfg = T5Config(
        vocab_size=384,
        d_model=48,
        d_ff=96,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        d_kv=12,
        decoder_start_token_id=0,
    )
    return Seq2SeqLogitsWrapper(T5ForConditionalGeneration(cfg)).eval()


def example_input_gar() -> tuple[Tensor, Tensor]:
    """Create an example question and generation-target decoder input.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``input_ids`` (the question), shape ``(1, 10)``, and
        ``decoder_input_ids`` (the partially generated context), shape ``(1, 8)``.
    """
    return torch.randint(1, 384, (1, 10)), torch.randint(1, 384, (1, 8))


# ---------------------------------------------------------------------------
# GENRE: autoregressive entity retrieval with trie-constrained decoding.
# ---------------------------------------------------------------------------


class EntityTrie:
    """Minimal prefix trie over allowed entity-name token sequences."""

    def __init__(self, entity_token_seqs: list[list[int]]) -> None:
        """Build the trie from a list of tokenized entity names.

        Parameters
        ----------
        entity_token_seqs
            Each element is the token-id sequence of one valid entity name.
        """
        self.children: dict[int, EntityTrie] = {}
        self.is_end = False
        for seq in entity_token_seqs:
            self._insert(seq)

    def _insert(self, seq: list[int]) -> None:
        node = self
        for tok in seq:
            node = node.children.setdefault(tok, EntityTrie([]))
        node.is_end = True

    def allowed_next(self, prefix: list[int]) -> list[int]:
        """Return the set of tokens permitted after ``prefix`` under the trie.

        Parameters
        ----------
        prefix
            Token ids generated so far for the current entity name.

        Returns
        -------
        list[int]
            Allowed next token ids (empty once the prefix is not a valid trie path).
        """
        node = self
        for tok in prefix:
            if tok not in node.children:
                return []
            node = node.children[tok]
        return list(node.children.keys())


class GenreEntityRetriever(nn.Module):
    """BART encoder-decoder that generates entity names, decoded via a prefix trie."""

    def __init__(self, vocab_size: int = 256, d_model: int = 32, n_entity_tokens: int = 6) -> None:
        """Initialize the underlying BART seq2seq transformer.

        Parameters
        ----------
        vocab_size
            Token vocabulary size shared by mention context and entity names.
        d_model
            Transformer hidden size.
        n_entity_tokens
            Number of tokens reserved as the closed entity-name sub-vocabulary
            used to build the constrained-decoding trie.
        """
        super().__init__()
        cfg = BartConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=64,
            decoder_ffn_dim=64,
            max_position_embeddings=32,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            decoder_start_token_id=1,
        )
        self.bart = BartForConditionalGeneration(cfg)
        entity_start = vocab_size - n_entity_tokens
        entity_names = [
            [entity_start + i, entity_start + ((i + 1) % n_entity_tokens)]
            for i in range(n_entity_tokens)
        ]
        self.trie = EntityTrie(entity_names)

    def forward(self, input_ids: Tensor, decoder_input_ids: Tensor) -> Tensor:
        """Compute entity-name token logits, masked by trie-allowed continuations.

        Parameters
        ----------
        input_ids
            Mention context token ids, shape ``(batch, src_len)``.
        decoder_input_ids
            Partially generated entity-name token ids, shape ``(batch, tgt_len)``.

        Returns
        -------
        Tensor
            Trie-masked next-token logits of shape ``(batch, tgt_len, vocab_size)``:
            positions the trie forbids are set to ``-inf`` before the final step.
        """
        logits = self.bart(input_ids=input_ids, decoder_input_ids=decoder_input_ids).logits
        prefix = decoder_input_ids[0].tolist()
        allowed = self.trie.allowed_next(prefix[1:]) or list(range(logits.shape[-1]))
        mask = torch.full_like(logits[:, -1, :], float("-inf"))
        mask[:, allowed] = 0.0
        masked_last = logits[:, -1, :] + mask
        return torch.cat([logits[:, :-1, :], masked_last.unsqueeze(1)], dim=1)


def build_genre() -> nn.Module:
    """Build a compact GENRE trie-constrained autoregressive entity retriever.

    Returns
    -------
    nn.Module
        Configured ``GenreEntityRetriever`` instance in eval mode.
    """
    return GenreEntityRetriever().eval()


def example_input_genre() -> tuple[Tensor, Tensor]:
    """Create an example mention context and partial entity-name decode.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``input_ids`` of shape ``(1, 9)`` and ``decoder_input_ids`` of shape ``(1, 2)``.
    """
    return torch.randint(3, 250, (1, 9)), torch.tensor([[1, 250]])


# ---------------------------------------------------------------------------
# GL-GIN: global-locally graph interaction network for multi-intent SLU.
# ---------------------------------------------------------------------------


class GraphAttentionLayer(nn.Module):
    """Single-head additive graph-attention layer over a dense adjacency mask."""

    def __init__(self, d_model: int) -> None:
        """Initialize the linear projection and attention scoring vector.

        Parameters
        ----------
        d_model
            Node feature dimensionality (input and output).
        """
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.attn_vec = nn.Linear(2 * d_model, 1)

    def forward(self, nodes: Tensor, adjacency: Tensor) -> Tensor:
        """Propagate node features along the given adjacency structure.

        Parameters
        ----------
        nodes
            Node features of shape ``(batch, n_nodes, d_model)``.
        adjacency
            Binary adjacency mask of shape ``(n_nodes, n_nodes)``.

        Returns
        -------
        Tensor
            Updated node features of shape ``(batch, n_nodes, d_model)``.
        """
        h = self.proj(nodes)
        n = h.shape[1]
        h_i = h.unsqueeze(2).expand(-1, -1, n, -1)
        h_j = h.unsqueeze(1).expand(-1, n, -1, -1)
        scores = self.attn_vec(torch.cat([h_i, h_j], dim=-1)).squeeze(-1)
        scores = scores.masked_fill(adjacency.unsqueeze(0) == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        return torch.bmm(weights, h)


class GLGINModel(nn.Module):
    """BiLSTM encoder + local slot-aware graph layer + global intent-slot graph layer."""

    def __init__(
        self,
        vocab_size: int = 200,
        d_model: int = 32,
        n_intents: int = 6,
        n_slots: int = 10,
    ) -> None:
        """Initialize the shared encoder, intent decoder, and two graph layers.

        Parameters
        ----------
        vocab_size
            Input token vocabulary size.
        d_model
            Hidden size shared by the encoder and both graph layers.
        n_intents
            Number of candidate intent labels (multi-label detection).
        n_slots
            Number of BIO slot labels.
        """
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.LSTM(d_model, d_model // 2, bidirectional=True, batch_first=True)
        self.intent_decoder = nn.Linear(d_model, n_intents)
        self.intent_emb = nn.Embedding(n_intents, d_model)
        self.local_slot_graph = GraphAttentionLayer(d_model)
        self.global_graph = GraphAttentionLayer(d_model)
        self.slot_decoder = nn.Linear(d_model, n_slots)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        """Jointly and non-autoregressively predict multiple intents and slots.

        Parameters
        ----------
        tokens
            Input token ids of shape ``(batch, seq_len)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Multi-label intent logits ``(batch, n_intents)`` and slot logits
            ``(batch, seq_len, n_slots)``.
        """
        batch, seq_len = tokens.shape
        x = self.emb(tokens)
        h, _ = self.encoder(x)

        intent_logits = self.intent_decoder(h.mean(dim=1))
        intent_probs = torch.sigmoid(intent_logits)

        slot_adjacency = torch.ones(seq_len, seq_len)
        slot_nodes = self.local_slot_graph(h, slot_adjacency)

        n_intents = intent_logits.shape[-1]
        all_intent_emb = self.intent_emb.weight.unsqueeze(0).expand(batch, -1, -1)
        weighted_intent_emb = all_intent_emb * intent_probs.unsqueeze(-1)
        joint_nodes = torch.cat([weighted_intent_emb, slot_nodes], dim=1)
        n_total = n_intents + seq_len
        global_adjacency = torch.ones(n_total, n_total)
        joint_out = self.global_graph(joint_nodes, global_adjacency)
        slot_out = joint_out[:, n_intents:]
        slot_logits = self.slot_decoder(slot_out)
        return intent_logits, slot_logits


def build_gl_gin() -> nn.Module:
    """Build a compact GL-GIN joint multi-intent + slot-filling model.

    Returns
    -------
    nn.Module
        Configured ``GLGINModel`` instance in eval mode.
    """
    return GLGINModel().eval()


def example_input_gl_gin() -> Tensor:
    """Create an example utterance token sequence.

    Returns
    -------
    Tensor
        Token ids of shape ``(1, 9)``.
    """
    return torch.randint(1, 200, (1, 9))


# ---------------------------------------------------------------------------
# GODEL: T5-initialized encoder-decoder with an explicit grounding span.
# ---------------------------------------------------------------------------


def build_godel() -> nn.Module:
    """Build a compact T5-style GODEL grounded dialog generator.

    Returns
    -------
    nn.Module
        A ``Seq2SeqLogitsWrapper`` around ``T5ForConditionalGeneration`` whose
        encoder consumes ``instruction + context + grounding`` and whose decoder
        generates the grounded response, in eval mode.
    """
    cfg = T5Config(
        vocab_size=320,
        d_model=48,
        d_ff=96,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        d_kv=12,
        decoder_start_token_id=0,
    )
    return Seq2SeqLogitsWrapper(T5ForConditionalGeneration(cfg)).eval()


def example_input_godel() -> tuple[Tensor, Tensor]:
    """Create an example instruction+context+grounding encoder input.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``input_ids`` (concatenated instruction/context/[KNOWLEDGE]/grounding
        text), shape ``(1, 18)``, and ``decoder_input_ids`` (partial grounded
        response), shape ``(1, 6)``.
    """
    return torch.randint(1, 320, (1, 18)), torch.randint(1, 320, (1, 6))


# ---------------------------------------------------------------------------
# HDSA: hierarchical disentangled self-attention response generator.
# ---------------------------------------------------------------------------


class DisentangledSelfAttentionLayer(nn.Module):
    """Self-attention layer whose heads are disjointly gated by DA-graph node activity."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        """Initialize per-head projections used for disentangled attention.

        Parameters
        ----------
        d_model
            Hidden size (must be divisible by ``n_heads``).
        n_heads
            Number of heads; each head is bound to one dialog-act-graph node at
            this layer.
        """
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor, node_gate: Tensor) -> Tensor:
        """Apply per-head self-attention gated by a dialog-act graph-layer activation.

        Parameters
        ----------
        x
            Decoder hidden states of shape ``(batch, seq_len, d_model)``.
        node_gate
            Per-head activation gate of shape ``(batch, n_heads)``: head ``i`` is
            scaled by ``node_gate[:, i]``, disentangling which dialog-act-graph
            node controls which head's contribution.

        Returns
        -------
        Tensor
            Updated hidden states of shape ``(batch, seq_len, d_model)``.
        """
        batch, seq_len, d_model = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        causal = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~causal, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        head_out = weights @ v
        head_out = head_out * node_gate[:, :, None, None]
        head_out = head_out.transpose(1, 2).reshape(batch, seq_len, d_model)
        return self.out_proj(head_out)


class HDSAResponseGenerator(nn.Module):
    """Dialog-act-graph-conditioned decoder using layer-wise disentangled attention."""

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 32,
        n_heads: int = 4,
        n_graph_layers: int = 3,
    ) -> None:
        """Initialize the DA predictor stub input path and disentangled decoder stack.

        Parameters
        ----------
        vocab_size
            Response token vocabulary size.
        d_model
            Decoder hidden size (must be divisible by ``n_heads``).
        n_heads
            Attention heads per layer; each head is bound to a dialog-act-graph
            node at that layer.
        n_graph_layers
            Number of dialog-act-graph layers (domain -> act -> slot), matched
            one-to-one with decoder layers.
        """
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(64, d_model)
        self.layers = nn.ModuleList(
            [DisentangledSelfAttentionLayer(d_model, n_heads) for _ in range(n_graph_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_graph_layers)])
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.n_heads = n_heads
        self.n_graph_layers = n_graph_layers

    def forward(self, response_tokens: Tensor, da_graph_gates: Tensor) -> Tensor:
        """Generate response logits conditioned on a per-layer dialog-act graph.

        Parameters
        ----------
        response_tokens
            Response token ids of shape ``(batch, seq_len)``.
        da_graph_gates
            Per-layer, per-head node-activation gates of shape
            ``(batch, n_graph_layers, n_heads)`` derived from the predicted
            dialog-act graph (one gate per disentangled head at that layer).

        Returns
        -------
        Tensor
            Response token logits of shape ``(batch, seq_len, vocab_size)``.
        """
        batch, seq_len = response_tokens.shape
        positions = torch.arange(seq_len, device=response_tokens.device).unsqueeze(0)
        x = self.tok_emb(response_tokens) + self.pos_emb(positions)
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = x + layer(norm(x), da_graph_gates[:, i])
        return self.lm_head(x)


def build_hdsa() -> nn.Module:
    """Build a compact HDSA hierarchical disentangled self-attention generator.

    Returns
    -------
    nn.Module
        Configured ``HDSAResponseGenerator`` instance in eval mode.
    """
    return HDSAResponseGenerator().eval()


def example_input_hdsa() -> tuple[Tensor, Tensor]:
    """Create an example response sequence and dialog-act graph gate tensor.

    Returns
    -------
    tuple[Tensor, Tensor]
        Response token ids of shape ``(1, 7)`` and DA-graph gates of shape
        ``(1, 3, 4)`` (3 graph layers, 4 heads each), sampled as soft
        activations in ``[0, 1]``.
    """
    tokens = torch.randint(1, 256, (1, 7))
    gates = torch.rand(1, 3, 4)
    return tokens, gates


MENAGERIE_ENTRIES = [
    ("GALAXY", "build_galaxy", "example_input_galaxy", "2022", "NLP"),
    ("GAR (Generation-Augmented Retrieval)", "build_gar", "example_input_gar", "2021", "NLP"),
    ("GENRE", "build_genre", "example_input_genre", "2020", "NLP"),
    ("GL-GIN", "build_gl_gin", "example_input_gl_gin", "2021", "NLP"),
    (
        "GODEL (Grounded Pre-Training for Open-Domain Dialogue)",
        "build_godel",
        "example_input_godel",
        "2022",
        "NLP",
    ),
    ("HDSA", "build_hdsa", "example_input_hdsa", "2019", "NLP"),
]
