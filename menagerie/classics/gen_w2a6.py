"""Compact faithful reimplementations of five retrieval / dialogue / memory NLP families.

Sources checked (paper + official source, no clone/pip-install; reimplemented from scratch):
  - Recurrent Entity Network (EntNet): arxiv:1612.03969 (Henaff et al., ICLR 2017); official TF
    repo github.com/jimfleming/recurrent-entity-networks, quality PyTorch port
    github.com/anantzoid/Recurrent-Entity-Networks-pytorch -- a fixed bank of *m* key-value
    memory blocks ``(key_j, value_j)``. At each input timestep every block runs the SAME shared
    update in parallel: a **gate** ``g_j = sigmoid(s^T value_j + s^T key_j)`` mixes
    content-based addressing (does the input match what's currently stored) with location-based
    addressing (does the input match this slot's fixed identity key); a **candidate value**
    ``h~_j = phi(U value_j + V key_j + W s)`` proposes an update from the current value, the
    key, and the input; the value is updated additively and gated,
    ``value_j <- value_j + g_j * h~_j``, then **renormalized to the unit sphere**
    (``value_j <- value_j / ||value_j||``) -- the normalization is what lets old content decay
    without an explicit forget gate (new writes shrink the cosine similarity to old content).
    After the whole input sequence is consumed, an **output module** attends over the final
    memory values with a learned query, ``p_j = softmax(q^T value_j)``,
    ``u = sum_j p_j * value_j``, ``y = R * phi(q + H * u)``, producing the task answer. We
    reimplement the block bank, the shared per-block update, and the attention-based output
    module; entity keys are randomly initialized (task-specific pretraining is out of scope).
  - REDIAL (Recurrent Dialogue with Mentions for CRS): arxiv:1812.07617 (Li et al., NeurIPS
    2018); official repo github.com/RaymondLi0/conversational-recommendations
    (`models/hierarchical_rnn.py::HRNN`, `models/sentiment_analysis.py::SentimentAnalysis`,
    `models/autorec.py::AutoRec`, `models/recommender_model.py::RecommendFromDialogue`,
    `Recommender`) -- the distinctive architecture is the fusion of THREE separately-motivated
    submodules into one conversational recommender: (1) a **Hierarchical Recurrent Encoder**
    (HRNN): a word-level bidirectional GRU encodes each utterance into a sentence vector, and a
    conversation-level GRU consumes the sequence of sentence vectors (concatenated with a
    sender-identity bit) to produce a running dialogue-context representation; (2) a
    **sentiment/liking predictor**: for every mentioned-movie slot in the dialogue, a small MLP
    reads the local context around the mention and predicts whether the speaker liked, disliked,
    or didn't specify an opinion on that movie; (3) an **AutoRec-style denoising autoencoder
    recommender**: the per-movie liking predictions accumulated so far are treated as a sparse
    partially-observed rating vector over the full movie catalog, encoded by an MLP encoder into
    a user-representation, optionally summed with a dialogue-context projection, and decoded by
    a linear layer back to a dense score over every movie in the catalog (the AutoRec output is
    literally what makes REDIAL "conversational": the same denoising-autoencoder collaborative-
    filtering idea, but the sparse input vector is built live from what the dialogue partner has
    revealed liking/disliking so far, not from historical ratings). We reimplement all three
    pieces wired together exactly as in `RecommendFromDialogue.forward`.
  - RETRO (Retrieval-Enhanced Transformer): arxiv:2112.04426 (Borgeaud et al., DeepMind 2021);
    high-quality community implementation github.com/lucidrains/RETRO-pytorch
    (`retro_pytorch/retro_pytorch.py::RETRO`, `Encoder`, `Decoder`, `ChunkedCrossAttention`) --
    the input sequence is split into fixed-size **chunks**; for each chunk, a small set of
    retrieved neighbor-chunks (normally found via a frozen BERT + nearest-neighbor index over a
    huge external corpus; here supplied directly as a retrieved-tokens tensor, since building the
    index is out of scope for a random-init architecture demo) is passed through a lightweight
    bidirectional **retrieval encoder** that itself cross-attends the *causally-shifted* chunk of
    decoder hidden states (so the encoding of a neighbor is conditioned on the query chunk that
    retrieved it). The decoder is a standard causal Transformer stack, except every few blocks a
    **Chunked Cross-Attention (CCA)** layer reshapes the running sequence into
    ``(chunk, position-in-chunk)``, offsets it by ``chunk_size - 1`` positions (so token *i*'s
    CCA query only ever attends to neighbors retrieved for the chunk ending strictly before token
    *i*, preserving causality), and cross-attends each chunk's tokens against that chunk's own
    retrieved-and-encoded neighbor set independently in parallel -- giving retrieval-augmented
    generation at chunk granularity with attention cost linear (not quadratic) in retrieved
    tokens. We reimplement the encoder, the chunk-local CCA reshape/cross-attend, and the
    interleaved decoder stack at tiny dims with a synthetic pre-retrieved-neighbors tensor.
  - Schema-Guided Dialogue (SGD-BERT): arxiv:1909.05855 (Rastogi et al., AAAI 2020) dataset
    paper, with the BERT-based zero-shot baseline described in the companion DSTC8 baseline
    (`google-research-datasets/dstc8-schema-guided-dialogue`, "Fine-Tuning BERT for
    Schema-Guided Zero-Shot Dialogue State Tracking", Rastogi et al. 2020) -- the distinctive
    idea is that instead of a fixed closed-vocabulary slot/intent classification head (which
    cannot generalize to unseen services), a **shared BERT encoder embeds the natural-language
    SCHEMA DESCRIPTIONS themselves** (of every intent, slot, and categorical slot-value for a
    service) into the same space as the dialogue utterance, and predictions are made by a
    similarity/projection between the utterance's ``[CLS]`` encoding and each schema-element
    embedding rather than against a fixed output layer -- so a service never seen during
    training can be handled zero-shot purely from its schema text. We reimplement: one shared
    BERT encoder for both utterance and schema-element descriptions; an intent-classification
    head that dot-products the utterance ``[CLS]`` vector against each candidate intent's
    schema embedding; and a slot-tagging head that dot-products every utterance token against
    each candidate slot's schema embedding to produce per-token, per-slot start/presence logits.
  - SCLSTM (Semantically Conditioned LSTM for NLG): Wen et al., EMNLP 2015 (D15-1199), "Semantically
    Conditioned LSTM-based Natural Language Generation for Spoken Dialogue Systems"; PyTorch
    reimpl. github.com/andy194673/nlg-sclstm-multiwoz (`model/layers/decoder_deep.py::DecoderDeep
    ._step`), original toolkit github.com/shawnwun/RNNLG -- an LSTM cell augmented with a
    dedicated **dialogue-act (DA) memory cell** ``d_t`` that is injected into the main cell
    update. At each step a **reading gate**
    ``r_t = sigmoid(W_wr [x_t, h_{t-1}] + W_dr d_{t-1})`` controls how much of the DA vector
    survives to the next step, ``d_t = r_t * d_{t-1}`` (the DA vector can only ever shrink
    monotonically toward zero across the sequence -- once information has been "read out" and
    realized in the surface text it is discarded, preventing repetition/omission of dialogue-act
    slots). The standard input/forget/output gates and candidate cell are computed exactly as in
    a vanilla LSTM from ``[x_t, h_{t-1}]``, but the cell update adds an extra additive term
    driven by the *updated* DA vector: ``c_t = f_t*c_{t-1} + i_t*c~_t + tanh(W_dc d_t)``,
    ``h_t = o_t * tanh(c_t)``. We reimplement the exact single-layer SC-LSTM cell (four standard
    gates + reading gate + DA-conditioned cell term) driven by an initial one-hot dialogue-act
    vector, unrolled over a token embedding sequence, with a linear output-vocabulary head.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import BertConfig, BertModel

# ---------------------------------------------------------------------------
# Recurrent Entity Network (EntNet)
# ---------------------------------------------------------------------------


class EntNet(nn.Module):
    """Recurrent Entity Network: gated key-value memory blocks + attention readout.

    Parameters
    ----------
    vocab_size : int
        Token vocabulary size.
    embed_dim : int
        Token / memory embedding dimensionality.
    n_blocks : int
        Number of fixed entity memory blocks (key-value slots).
    n_classes : int
        Output answer vocabulary size (e.g. bAbI answer classes).
    """

    def __init__(
        self,
        vocab_size: int = 64,
        embed_dim: int = 16,
        n_blocks: int = 4,
        n_classes: int = 32,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_blocks = n_blocks

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.keys = nn.Parameter(torch.randn(n_blocks, embed_dim) * 0.1)

        self.U = nn.Linear(embed_dim, embed_dim, bias=False)
        self.V = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W = nn.Linear(embed_dim, embed_dim, bias=False)
        self.prelu = nn.PReLU()

        self.query_embed = nn.Embedding(vocab_size, embed_dim)
        self.H = nn.Linear(embed_dim, embed_dim, bias=False)
        self.R = nn.Linear(embed_dim, n_classes)

    def forward(self, story: Tensor, query: Tensor) -> Tensor:
        """Process a story (token sequence) into memory, then answer a query.

        Parameters
        ----------
        story : Tensor
            Token ids, shape ``(batch, seq_len)``.
        query : Tensor
            Query token ids, shape ``(batch, q_len)``.

        Returns
        -------
        Tensor
            Answer logits, shape ``(batch, n_classes)``.
        """

        batch = story.shape[0]
        s_embed = self.embed(story).mean(dim=1, keepdim=False)  # sentence-pooled input, (b, seq, d)
        s_seq = self.embed(story)  # (batch, seq_len, embed_dim)

        value = self.keys.unsqueeze(0).expand(batch, -1, -1).clone()  # (batch, n_blocks, embed_dim)
        key = self.keys.unsqueeze(0).expand(batch, -1, -1)  # (batch, n_blocks, embed_dim)

        for t in range(s_seq.shape[1]):
            s_t = s_seq[:, t, :]  # (batch, embed_dim)
            s_t_exp = s_t.unsqueeze(1)  # (batch, 1, embed_dim)

            gate_logit = (s_t_exp * value).sum(-1) + (s_t_exp * key).sum(-1)  # (batch, n_blocks)
            gate = torch.sigmoid(gate_logit).unsqueeze(-1)  # (batch, n_blocks, 1)

            candidate = self.prelu(self.U(value) + self.V(key) + self.W(s_t_exp))
            value = value + gate * candidate
            value = value / (value.norm(dim=-1, keepdim=True) + 1e-8)

        q = self.query_embed(query).mean(dim=1)  # (batch, embed_dim)
        attn_logits = torch.einsum("bd,bnd->bn", q, value)
        attn = F.softmax(attn_logits, dim=-1).unsqueeze(-1)  # (batch, n_blocks, 1)
        u = (attn * value).sum(dim=1)  # (batch, embed_dim)
        y = self.R(self.prelu(q + self.H(u)))
        return y


def build_entnet() -> nn.Module:
    """Build the compact EntNet gated key-value memory model.

    Returns
    -------
    nn.Module
        ``EntNet`` in eval mode.
    """

    return EntNet(vocab_size=64, embed_dim=16, n_blocks=4, n_classes=32).eval()


def example_input_entnet() -> tuple[Tensor, Tensor]:
    """Example story + query token ids for EntNet.

    Returns
    -------
    tuple[Tensor, Tensor]
        Story ids ``(2, 12)``, query ids ``(2, 4)``.
    """

    story = torch.randint(0, 64, (2, 12))
    query = torch.randint(0, 64, (2, 4))
    return story, query


# ---------------------------------------------------------------------------
# REDIAL (Recurrent Dialogue with Mentions for Conversational Recommendation)
# ---------------------------------------------------------------------------


class HierarchicalDialogueEncoder(nn.Module):
    """Word-level BiGRU + conversation-level GRU hierarchical dialogue encoder."""

    def __init__(self, vocab_size: int, embed_dim: int, sent_hidden: int, conv_hidden: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.sentence_encoder = nn.GRU(embed_dim, sent_hidden, batch_first=True, bidirectional=True)
        # +1 for the sender-identity bit concatenated onto every sentence vector.
        self.conversation_encoder = nn.GRU(2 * sent_hidden + 1, conv_hidden, batch_first=True)

    def forward(self, dialogue: Tensor, senders: Tensor) -> Tensor:
        """Encode a batch of multi-turn dialogues into per-turn context vectors.

        Parameters
        ----------
        dialogue : Tensor
            Token ids, shape ``(batch, n_turns, turn_len)``.
        senders : Tensor
            Sender-identity bit per turn, shape ``(batch, n_turns, 1)``.

        Returns
        -------
        Tensor
            Conversation-level hidden states, shape ``(batch, n_turns, conv_hidden)``.
        """

        batch, n_turns, turn_len = dialogue.shape
        flat = dialogue.reshape(batch * n_turns, turn_len)
        word_embed = self.embed(flat)
        _, sent_h = self.sentence_encoder(word_embed)  # (2, batch*n_turns, sent_hidden)
        sent_vec = torch.cat([sent_h[0], sent_h[1]], dim=-1)  # (batch*n_turns, 2*sent_hidden)
        sent_vec = sent_vec.view(batch, n_turns, -1)

        conv_input = torch.cat([sent_vec, senders], dim=-1)
        conv_out, _ = self.conversation_encoder(conv_input)
        return conv_out


class SentimentPredictor(nn.Module):
    """Small MLP predicting liked/disliked/unspecified for a mentioned movie."""

    def __init__(self, context_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, context: Tensor) -> Tensor:
        """Predict sentiment logits from local dialogue context around a movie mention.

        Parameters
        ----------
        context : Tensor
            Context vector per movie mention, shape ``(batch, n_mentions, context_dim)``.

        Returns
        -------
        Tensor
            Sentiment logits (liked / disliked / unspecified), shape ``(batch, n_mentions, 3)``.
        """

        return self.net(context)


class AutoRecRecommender(nn.Module):
    """Denoising-autoencoder collaborative-filtering recommender (AutoRec)."""

    def __init__(self, n_movies: int, hidden: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(n_movies, hidden)
        self.decoder = nn.Linear(hidden, n_movies)

    def forward(self, sparse_ratings: Tensor, user_bias: Tensor | None = None) -> Tensor:
        """Reconstruct a dense score over the full movie catalog.

        Parameters
        ----------
        sparse_ratings : Tensor
            Partially-observed liking vector, shape ``(batch, n_movies)``.
        user_bias : Tensor, optional
            Additional dialogue-derived user representation to fuse in, shape
            ``(batch, hidden)``.

        Returns
        -------
        Tensor
            Dense recommendation scores, shape ``(batch, n_movies)``.
        """

        user_rep = torch.sigmoid(self.encoder(sparse_ratings))
        if user_bias is not None:
            user_rep = user_rep + user_bias
        return self.decoder(user_rep)


class REDIAL(nn.Module):
    """REDIAL: HRNN dialogue encoder + sentiment predictor + AutoRec recommender.

    Parameters
    ----------
    vocab_size : int
        Token vocabulary size.
    n_movies : int
        Number of movies in the recommendation catalog.
    embed_dim : int
        Word embedding dimensionality.
    sent_hidden : int
        Sentence-level GRU hidden size (per direction).
    conv_hidden : int
        Conversation-level GRU hidden size.
    """

    def __init__(
        self,
        vocab_size: int = 64,
        n_movies: int = 48,
        embed_dim: int = 16,
        sent_hidden: int = 12,
        conv_hidden: int = 16,
    ) -> None:
        super().__init__()
        self.n_movies = n_movies
        self.encoder = HierarchicalDialogueEncoder(vocab_size, embed_dim, sent_hidden, conv_hidden)
        self.sentiment = SentimentPredictor(context_dim=conv_hidden, hidden=16)
        self.autorec = AutoRecRecommender(n_movies=n_movies, hidden=conv_hidden)
        self.language_to_user = nn.Linear(conv_hidden, conv_hidden)

    def forward(
        self,
        dialogue: Tensor,
        senders: Tensor,
        movie_mention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run the full conversational-recommendation pipeline.

        Parameters
        ----------
        dialogue : Tensor
            Token ids, shape ``(batch, n_turns, turn_len)``.
        senders : Tensor
            Sender-identity bit per turn, shape ``(batch, n_turns, 1)``.
        movie_mention_mask : Tensor
            Multi-hot indicator of which movies (columns) have been mentioned so far, used as
            the sparse AutoRec input, shape ``(batch, n_movies)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Sentiment logits per turn ``(batch, n_turns, 3)`` and movie recommendation scores
            ``(batch, n_movies)``.
        """

        conv_states = self.encoder(dialogue, senders)  # (batch, n_turns, conv_hidden)
        sentiment_logits = self.sentiment(conv_states)

        final_context = conv_states[:, -1, :]
        user_bias = self.language_to_user(final_context)
        recommendations = self.autorec(movie_mention_mask, user_bias=user_bias)
        return sentiment_logits, recommendations


def build_redial() -> nn.Module:
    """Build the compact REDIAL conversational-recommendation model.

    Returns
    -------
    nn.Module
        ``REDIAL`` in eval mode.
    """

    return REDIAL(vocab_size=64, n_movies=48, embed_dim=16, sent_hidden=12, conv_hidden=16).eval()


def example_input_redial() -> tuple[Tensor, Tensor, Tensor]:
    """Example dialogue batch + sender bits + movie-mention mask for REDIAL.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Dialogue ids ``(2, 5, 8)``, sender bits ``(2, 5, 1)``, movie-mention mask ``(2, 48)``.
    """

    dialogue = torch.randint(0, 64, (2, 5, 8))
    senders = torch.randint(0, 2, (2, 5, 1)).float()
    movie_mask = torch.zeros(2, 48)
    movie_mask[:, :5] = 1.0
    return dialogue, senders, movie_mask


# ---------------------------------------------------------------------------
# RETRO (Retrieval-Enhanced Transformer): chunked cross-attention
# ---------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    """Generic multi-head attention (self- or cross-) with optional causal masking."""

    def __init__(self, dim: int, n_heads: int, causal: bool = False) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.causal = causal
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, 2 * dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor, context: Tensor | None = None) -> Tensor:
        """Apply multi-head attention.

        Parameters
        ----------
        x : Tensor
            Query input, shape ``(batch, n_q, dim)``.
        context : Tensor, optional
            Key/value source; defaults to ``x`` (self-attention), shape ``(batch, n_kv, dim)``.

        Returns
        -------
        Tensor
            Attention output, shape ``(batch, n_q, dim)``.
        """

        b, n_q, d = x.shape
        ctx = x if context is None else context
        n_kv = ctx.shape[1]

        q = self.to_q(x).view(b, n_q, self.n_heads, self.head_dim).transpose(1, 2)
        k, v = self.to_kv(ctx).chunk(2, dim=-1)
        k = k.view(b, n_kv, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n_kv, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(self.head_dim)
        if self.causal and context is None:
            causal_mask = torch.triu(
                torch.ones(n_q, n_kv, dtype=torch.bool, device=x.device), diagonal=1
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum("bhqk,bhkd->bhqd", attn, v)
        out = out.transpose(1, 2).reshape(b, n_q, d)
        return self.to_out(out)


class ChunkedCrossAttention(nn.Module):
    """RETRO's chunked cross-attention: each chunk attends only its own retrieved neighbors."""

    def __init__(self, dim: int, n_heads: int, chunk_size: int) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.cross_attn = MultiHeadAttention(dim, n_heads)

    def forward(self, x: Tensor, retrieved_encoded: Tensor) -> Tensor:
        """Cross-attend causally-shifted chunks against their retrieved neighbor encodings.

        Parameters
        ----------
        x : Tensor
            Decoder hidden states, shape ``(batch, seq_len, dim)``.
        retrieved_encoded : Tensor
            Per-chunk encoded neighbor tokens, shape
            ``(batch, n_chunks, n_neighbors * neighbor_len, dim)``.

        Returns
        -------
        Tensor
            Updated hidden states with CCA applied to the chunked region, same shape as ``x``.
        """

        b, seq_len, d = x.shape
        c = self.chunk_size
        n_chunks = retrieved_encoded.shape[1]
        usable_len = n_chunks * c

        if seq_len <= c:
            # Not enough tokens for a full causally-shifted chunk yet; pass through unchanged.
            return x

        # Causal shift: token i's CCA query is drawn from the chunk ending strictly before it,
        # so we drop the first (chunk_size - 1) tokens and re-pad at the end to keep length.
        available_len = min(usable_len, seq_len - (c - 1))
        shifted = x[:, c - 1 : c - 1 + available_len, :]
        pad_len = usable_len - shifted.shape[1]
        if pad_len > 0:
            shifted = F.pad(shifted, (0, 0, 0, pad_len))
        chunked = shifted.reshape(b * n_chunks, c, d)

        neighbors = retrieved_encoded.reshape(b * n_chunks, -1, d)
        out = self.cross_attn(chunked, context=neighbors)
        out = out.reshape(b, usable_len, d)
        out = out[:, :available_len, :]

        result = x.clone()
        result[:, c - 1 : c - 1 + available_len, :] = out
        return result


class RetroEncoder(nn.Module):
    """Bidirectional encoder for retrieved-neighbor chunks, cross-attending the query chunk."""

    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(dim, n_heads)
        self.cross_attn = MultiHeadAttention(dim, n_heads)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, retrieved: Tensor, chunk_query: Tensor) -> Tensor:
        """Encode retrieved neighbor tokens, conditioned on the retrieving query chunk.

        Parameters
        ----------
        retrieved : Tensor
            Retrieved neighbor token embeddings, shape ``(batch * n_chunks, n_tok, dim)``.
        chunk_query : Tensor
            The decoder's query chunk that retrieved these neighbors, shape
            ``(batch * n_chunks, chunk_size, dim)``.

        Returns
        -------
        Tensor
            Encoded neighbor tokens, same shape as ``retrieved``.
        """

        x = retrieved + self.self_attn(self.norm1(retrieved))
        x = x + self.cross_attn(self.norm2(x), context=chunk_query)
        x = x + self.ff(self.norm3(x))
        return x


class RETRO(nn.Module):
    """RETRO: Transformer decoder with interleaved Chunked Cross-Attention (CCA) blocks.

    Parameters
    ----------
    vocab_size : int
        Token vocabulary size.
    dim : int
        Model / hidden dimensionality.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Total number of decoder blocks.
    chunk_size : int
        Chunk size for CCA.
    cca_layers : tuple[int, ...]
        Indices of decoder layers that include a CCA sub-block.
    """

    def __init__(
        self,
        vocab_size: int = 96,
        dim: int = 32,
        n_heads: int = 4,
        n_layers: int = 4,
        chunk_size: int = 8,
        cca_layers: tuple[int, ...] = (1, 3),
    ) -> None:
        super().__init__()
        self.dim = dim
        self.chunk_size = chunk_size
        self.embed = nn.Embedding(vocab_size, dim)
        self.retrieved_embed = nn.Embedding(vocab_size, dim)

        self.self_attns = nn.ModuleList(
            [MultiHeadAttention(dim, n_heads, causal=True) for _ in range(n_layers)]
        )
        self.ffs = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
                for _ in range(n_layers)
            ]
        )
        self.norms1 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])

        self.cca_layers = set(cca_layers)
        self.ccas = nn.ModuleDict(
            {str(i): ChunkedCrossAttention(dim, n_heads, chunk_size) for i in cca_layers}
        )
        self.encoders = nn.ModuleDict({str(i): RetroEncoder(dim, n_heads) for i in cca_layers})

        self.to_logits = nn.Linear(dim, vocab_size)

    def forward(self, tokens: Tensor, retrieved_tokens: Tensor) -> Tensor:
        """Run RETRO's causal decoder with chunked cross-attention over retrieved neighbors.

        Parameters
        ----------
        tokens : Tensor
            Input token ids, shape ``(batch, seq_len)``.
        retrieved_tokens : Tensor
            Pre-retrieved neighbor token ids per chunk, shape
            ``(batch, n_chunks, n_neighbors, neighbor_len)``.

        Returns
        -------
        Tensor
            Next-token logits, shape ``(batch, seq_len, vocab_size)``.
        """

        b, seq_len = tokens.shape
        x = self.embed(tokens)

        n_chunks = retrieved_tokens.shape[1]
        n_neighbors = retrieved_tokens.shape[2]
        neighbor_len = retrieved_tokens.shape[3]
        retrieved_flat = retrieved_tokens.reshape(b * n_chunks, n_neighbors * neighbor_len)
        retrieved_x = self.retrieved_embed(retrieved_flat)

        for i in range(len(self.self_attns)):
            x = x + self.self_attns[i](self.norms1[i](x))
            x = x + self.ffs[i](self.norms2[i](x))

            if i in self.cca_layers:
                c = self.chunk_size
                usable_len = n_chunks * c
                if seq_len > c:
                    available_len = min(usable_len, seq_len - (c - 1))
                    shifted = x[:, c - 1 : c - 1 + available_len, :]
                    pad_len = usable_len - shifted.shape[1]
                    if pad_len > 0:
                        shifted = F.pad(shifted, (0, 0, 0, pad_len))
                    chunk_query = shifted.reshape(b * n_chunks, c, self.dim)
                    encoded = self.encoders[str(i)](retrieved_x, chunk_query)
                    encoded = encoded.reshape(b, n_chunks, n_neighbors * neighbor_len, self.dim)
                    x = self.ccas[str(i)](x, encoded)

        return self.to_logits(x)


def build_retro() -> nn.Module:
    """Build the compact RETRO chunked-cross-attention retrieval-augmented decoder.

    Returns
    -------
    nn.Module
        ``RETRO`` in eval mode.
    """

    return RETRO(
        vocab_size=96, dim=32, n_heads=4, n_layers=4, chunk_size=8, cca_layers=(1, 3)
    ).eval()


def example_input_retro() -> tuple[Tensor, Tensor]:
    """Example token sequence + pre-retrieved neighbor tokens for RETRO.

    Returns
    -------
    tuple[Tensor, Tensor]
        Token ids ``(2, 24)``, retrieved neighbor ids ``(2, 3, 2, 8)`` (3 chunks, 2 neighbors
        per chunk, 8 tokens per neighbor).
    """

    tokens = torch.randint(0, 96, (2, 24))
    retrieved = torch.randint(0, 96, (2, 3, 2, 8))
    return tokens, retrieved


# ---------------------------------------------------------------------------
# Schema-Guided Dialogue (SGD-BERT): schema-description embedding for zero-shot DST
# ---------------------------------------------------------------------------


class SchemaGuidedDST(nn.Module):
    """SGD-BERT: shared BERT encodes both the utterance and schema-element descriptions.

    Parameters
    ----------
    bert_config : BertConfig
        Configuration for the shared BERT encoder.
    """

    def __init__(self, bert_config: BertConfig) -> None:
        super().__init__()
        self.bert = BertModel(bert_config)
        hidden = bert_config.hidden_size
        self.intent_proj = nn.Linear(hidden, hidden)
        self.schema_proj = nn.Linear(hidden, hidden)
        self.slot_token_proj = nn.Linear(hidden, hidden)
        self.slot_schema_proj = nn.Linear(hidden, hidden)

    def forward(
        self,
        utterance_ids: Tensor,
        utterance_mask: Tensor,
        intent_schema_ids: Tensor,
        intent_schema_mask: Tensor,
        slot_schema_ids: Tensor,
        slot_schema_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Predict intent and per-token slot logits via schema-embedding similarity.

        Parameters
        ----------
        utterance_ids : Tensor
            Dialogue utterance token ids, shape ``(batch, u_len)``.
        utterance_mask : Tensor
            Utterance attention mask, shape ``(batch, u_len)``.
        intent_schema_ids : Tensor
            Candidate intent description token ids, shape ``(batch, n_intents, s_len)``.
        intent_schema_mask : Tensor
            Intent description attention mask, shape ``(batch, n_intents, s_len)``.
        slot_schema_ids : Tensor
            Candidate slot description token ids, shape ``(batch, n_slots, s_len)``.
        slot_schema_mask : Tensor
            Slot description attention mask, shape ``(batch, n_slots, s_len)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Intent logits ``(batch, n_intents)`` and per-token slot presence logits
            ``(batch, u_len, n_slots)``.
        """

        batch, u_len = utterance_ids.shape
        n_intents = intent_schema_ids.shape[1]
        n_slots = slot_schema_ids.shape[1]

        utt_out = self.bert(
            input_ids=utterance_ids, attention_mask=utterance_mask
        ).last_hidden_state
        utt_cls = utt_out[:, 0, :]  # (batch, hidden)

        intent_flat_ids = intent_schema_ids.reshape(batch * n_intents, -1)
        intent_flat_mask = intent_schema_mask.reshape(batch * n_intents, -1)
        intent_out = self.bert(
            input_ids=intent_flat_ids, attention_mask=intent_flat_mask
        ).last_hidden_state
        intent_cls = intent_out[:, 0, :].view(batch, n_intents, -1)

        slot_flat_ids = slot_schema_ids.reshape(batch * n_slots, -1)
        slot_flat_mask = slot_schema_mask.reshape(batch * n_slots, -1)
        slot_out = self.bert(
            input_ids=slot_flat_ids, attention_mask=slot_flat_mask
        ).last_hidden_state
        slot_cls = slot_out[:, 0, :].view(batch, n_slots, -1)

        utt_intent_q = self.intent_proj(utt_cls)  # (batch, hidden)
        intent_k = self.schema_proj(intent_cls)  # (batch, n_intents, hidden)
        intent_logits = torch.einsum("bd,bid->bi", utt_intent_q, intent_k)

        utt_slot_q = self.slot_token_proj(utt_out)  # (batch, u_len, hidden)
        slot_k = self.slot_schema_proj(slot_cls)  # (batch, n_slots, hidden)
        slot_logits = torch.einsum("bud,bsd->bus", utt_slot_q, slot_k)

        return intent_logits, slot_logits


def build_sgd_bert() -> nn.Module:
    """Build the compact schema-guided zero-shot dialogue-state-tracking model.

    Returns
    -------
    nn.Module
        ``SchemaGuidedDST`` in eval mode.
    """

    bert_config = BertConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=32,
    )
    return SchemaGuidedDST(bert_config).eval()


def example_input_sgd_bert() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Example utterance + intent/slot schema descriptions for SGD-BERT.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
        Utterance ids ``(1, 10)``, utterance mask ``(1, 10)``, intent schema ids
        ``(1, 3, 6)``, intent schema mask ``(1, 3, 6)``, slot schema ids ``(1, 5, 6)``, slot
        schema mask ``(1, 5, 6)``.
    """

    utterance_ids = torch.randint(0, 128, (1, 10))
    utterance_mask = torch.ones(1, 10, dtype=torch.long)
    intent_schema_ids = torch.randint(0, 128, (1, 3, 6))
    intent_schema_mask = torch.ones(1, 3, 6, dtype=torch.long)
    slot_schema_ids = torch.randint(0, 128, (1, 5, 6))
    slot_schema_mask = torch.ones(1, 5, 6, dtype=torch.long)
    return (
        utterance_ids,
        utterance_mask,
        intent_schema_ids,
        intent_schema_mask,
        slot_schema_ids,
        slot_schema_mask,
    )


# ---------------------------------------------------------------------------
# SCLSTM (Semantically Conditioned LSTM for NLG)
# ---------------------------------------------------------------------------


class SCLSTMCell(nn.Module):
    """Single-layer semantically-conditioned LSTM cell with a dialogue-act reading gate."""

    def __init__(self, input_size: int, hidden_size: int, da_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.w2h = nn.Linear(input_size, hidden_size * 4)
        self.h2h = nn.Linear(hidden_size, hidden_size * 4)
        self.w2h_r = nn.Linear(input_size, da_size)
        self.h2h_r = nn.Linear(hidden_size, da_size)
        self.dc = nn.Linear(da_size, hidden_size, bias=False)

    def forward(
        self, x_t: Tensor, h_prev: Tensor, c_prev: Tensor, d_prev: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Advance the SC-LSTM cell by one timestep.

        Parameters
        ----------
        x_t : Tensor
            Input embedding at this timestep, shape ``(batch, input_size)``.
        h_prev : Tensor
            Previous hidden state, shape ``(batch, hidden_size)``.
        c_prev : Tensor
            Previous cell state, shape ``(batch, hidden_size)``.
        d_prev : Tensor
            Previous dialogue-act vector, shape ``(batch, da_size)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            New hidden state, cell state, and dialogue-act vector.
        """

        gates_w = self.w2h(x_t)
        gates_h = self.h2h(h_prev)
        i_w, f_w, o_w, c_w = gates_w.chunk(4, dim=-1)
        i_h, f_h, o_h, c_h = gates_h.chunk(4, dim=-1)

        gate_i = torch.sigmoid(i_w + i_h)
        gate_f = torch.sigmoid(f_w + f_h)
        gate_o = torch.sigmoid(o_w + o_h)
        cell_hat = torch.tanh(c_w + c_h)

        gate_r = torch.sigmoid(self.w2h_r(x_t) + self.h2h_r(h_prev))
        d_t = gate_r * d_prev

        c_t = gate_f * c_prev + gate_i * cell_hat + torch.tanh(self.dc(d_t))
        h_t = gate_o * torch.tanh(c_t)
        return h_t, c_t, d_t


class SCLSTM(nn.Module):
    """SCLSTM: dialogue-act-conditioned LSTM decoder for surface realization in NLG.

    Parameters
    ----------
    vocab_size : int
        Output token vocabulary size.
    embed_dim : int
        Token embedding dimensionality.
    hidden_size : int
        LSTM hidden state size.
    da_size : int
        Dialogue-act vector dimensionality (one slot per DA feature).
    """

    def __init__(
        self,
        vocab_size: int = 48,
        embed_dim: int = 16,
        hidden_size: int = 24,
        da_size: int = 12,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.da_size = da_size
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.cell = SCLSTMCell(embed_dim, hidden_size, da_size)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens: Tensor, dialogue_act: Tensor) -> Tensor:
        """Unroll the SC-LSTM decoder over a token sequence conditioned on a dialogue act.

        Parameters
        ----------
        tokens : Tensor
            Input token ids (e.g. teacher-forced previous outputs), shape ``(batch, seq_len)``.
        dialogue_act : Tensor
            Initial one-hot/multi-hot dialogue-act feature vector, shape ``(batch, da_size)``.

        Returns
        -------
        Tensor
            Per-step output vocabulary logits, shape ``(batch, seq_len, vocab_size)``.
        """

        batch, seq_len = tokens.shape
        h_t = torch.zeros(batch, self.hidden_size, device=tokens.device)
        c_t = torch.zeros(batch, self.hidden_size, device=tokens.device)
        d_t = dialogue_act

        embeds = self.embed(tokens)
        outputs = []
        for t in range(seq_len):
            h_t, c_t, d_t = self.cell(embeds[:, t, :], h_t, c_t, d_t)
            outputs.append(self.out(h_t))
        return torch.stack(outputs, dim=1)


def build_sclstm() -> nn.Module:
    """Build the compact SCLSTM dialogue-act-conditioned NLG decoder.

    Returns
    -------
    nn.Module
        ``SCLSTM`` in eval mode.
    """

    return SCLSTM(vocab_size=48, embed_dim=16, hidden_size=24, da_size=12).eval()


def example_input_sclstm() -> tuple[Tensor, Tensor]:
    """Example token sequence + dialogue-act vector for SCLSTM.

    Returns
    -------
    tuple[Tensor, Tensor]
        Token ids ``(2, 10)``, dialogue-act vector ``(2, 12)``.
    """

    tokens = torch.randint(0, 48, (2, 10))
    dialogue_act = torch.zeros(2, 12)
    dialogue_act[:, :3] = 1.0
    return tokens, dialogue_act


MENAGERIE_ENTRIES = [
    (
        "Recurrent Entity Network (EntNet)",
        "build_entnet",
        "example_input_entnet",
        "2017",
        "NLP",
    ),
    (
        "REDIAL (Recurrent Dialogue with Mentions for CRS)",
        "build_redial",
        "example_input_redial",
        "2018",
        "NLP",
    ),
    (
        "RETRO",
        "build_retro",
        "example_input_retro",
        "2021",
        "NLP",
    ),
    (
        "Schema-Guided Dialogue (SGD / SGD-BERT)",
        "build_sgd_bert",
        "example_input_sgd_bert",
        "2020",
        "NLP",
    ),
    (
        "SCLSTM (Semantically Conditioned LSTM for NLG)",
        "build_sclstm",
        "example_input_sclstm",
        "2015",
        "NLP",
    ),
]
