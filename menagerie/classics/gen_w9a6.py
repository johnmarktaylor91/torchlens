"""Compact faithful reimplementations for build_queue rows 37-42 (W9A6).

Sources checked (repo/paper browsed via ``gh api`` / web search, no clone/pip-install):
  - R-SMILES: Zhong, Song, Han, Wang, Sun, "Root-aligned SMILES: a tight
    representation for chemical reaction prediction", Chemical Science 2022,
    arXiv:2203.11444. Official repo github.com/otori-bird/retrosynthesis
    (OpenNMT-py based encoder-decoder Transformer over root-aligned SMILES
    tokens). Distinctive mechanism: R-SMILES is not a new network -- it is a
    standard seq2seq Transformer whose *input representation* is the novel
    contribution. Product and reactant SMILES are re-rooted at the same
    atom before tokenization, producing near-identical token order between
    source and target and shrinking the effective edit distance the decoder
    must learn (vs. arbitrary canonical SMILES). Reproduced here as a
    standard pre-norm Transformer encoder-decoder over a small SMILES-token
    vocabulary, with an explicit ``root_align`` helper that rotates a
    tokenized ring/chain sequence to start at a shared anchor token -- the
    input-representation step that is R-SMILES's namesake mechanism --
    applied to both source and target streams before the seq2seq pass.
  - ReactionT5: Sagawa & Kojima, "ReactionT5: a large-scale pre-trained
    model towards application of limited reaction data", arXiv:2311.06708.
    Official repo github.com/sagawatatsuya/ReactionT5 (HuggingFace
    ``T5ForConditionalGeneration`` fine-tuned on Open Reaction Database).
    Distinctive mechanism: a single pretrained T5 encoder-decoder is reused
    across three reaction tasks (forward-product generation, retrosynthesis,
    yield regression) by prefixing the input SMILES/reagent string with a
    task tag and, for yield prediction, pooling the encoder's final hidden
    state through a small regression head instead of decoding text.
    Reproduced here via ``transformers.T5Config`` + ``T5ForConditionalGeneration``
    at tiny dims (matching the base env's installed transformers library, as
    instructed for config-of-installed-library models) with an added linear
    yield-regression head on the pooled encoder output, and a task-prefix
    tokenization convention mirroring the three-task multitask setup.
  - REINVENT (REINVENT4): Loeffler, He, Tibo, Janet, Voronov, Mervin,
    Engkvist, "Reinvent 4: Modern AI-driven generative molecule design",
    J. Cheminformatics 2024, arXiv:2304.00702. Official repo
    github.com/MolecularAI/REINVENT4,
    ``reinvent/models/reinvent/models/rnn.py`` (``RNN``). Distinctive
    mechanism: the classic/default REINVENT prior is an autoregressive
    SMILES *character-level* generator -- an embedding layer feeding a
    multi-layer GRU (or LSTM) stack, with a linear projection back to
    vocabulary logits at every step -- later fine-tuned via REINFORCE-style
    reinforcement learning against a scoring function (the RL loop itself
    is training-time, not part of the traced network). Reproduced here as a
    3-layer GRU character-RNN over a SMILES-token vocabulary with embedding
    -> GRU stack -> output-vocabulary linear head, matching
    ``reinvent/models/reinvent/models/rnn.py::RNN.forward`` exactly in
    structure.
  - ResGen: Zhang & Liu, "ResGen is a pocket-aware 3D molecular generation
    model based on parallel multiscale modelling", Nature Machine
    Intelligence 2023, arXiv:2211.07658. Official repo
    github.com/OdinZhang/ResGen. Distinctive mechanism: rather than
    autoregressing over individual atoms one at a time (slow, as in prior
    work such as Pocket2Mol), ResGen performs *hierarchical* autoregression:
    a coarse global autoregressive step predicts the next residue/fragment
    "focal" position in one shot from a multiscale (atom + residue level)
    equivariant encoding of the protein pocket + partial ligand, and a
    second, *parallel* atom-level sub-step then places every atom of that
    fragment simultaneously (not one-by-one) conditioned on the focal
    embedding, giving an ~8x wall-clock speedup over strictly sequential
    atom-by-atom baselines. Reproduced here with a compact multiscale
    (atom-level + residue-level) graph encoder, a global-step focal-position
    scorer over residues, and a parallel per-atom-slot head that predicts
    all atoms of the next fragment (element type + 3D offset) in one
    non-autoregressive pass conditioned on the focal embedding.
  - Retroformer: Wan, Remon, Coley, "Retroformer: Pushing the Limits of
    End-to-end Retrosynthesis Transformer", ICML 2022, arXiv:2201.12475.
    Official repo github.com/yuewan2/Retroformer,
    ``retroformer/models/module.py`` (``MultiHeadedAttention``) +
    ``retroformer/models/encoder.py``. Distinctive mechanism: Retroformer's
    encoder attention head is split in half -- one half runs ordinary dense
    "global" self-attention over the full token sequence, the other half
    runs "local" attention restricted to bonded-atom pairs (a molecular
    graph mask) and is *gated by an edge-feature embedding* projected from
    each bond's features; the local branch also emits an *updated* edge
    feature (concatenate the two endpoint node features, MLP back to
    edge-dim) that is layer-normed and fed to the next encoder layer,
    giving a genuine node<->edge co-update across layers, not just
    graph-masked attention. Reproduced here with a from-scratch
    ``LocalGlobalAttention`` module implementing exactly this split
    (global heads = dense softmax attention; local heads = bond-mask-gated
    attention keyed by a learned bond-pair adjacency, edge-feature
    projection multiplied into the local keys) plus the edge-feature update
    MLP, stacked into a small Transformer encoder.
  - RetroPrime: Wang, Xu, Wang, Wu, Guo, Pei, Lai, "RetroPrime: A diverse,
    plausible and Transformer-based method for single-step retrosynthesis
    predictions", Chemical Engineering Journal 2021, arXiv:2105.08321.
    Official repo github.com/wangxr0526/RetroPrime (OpenNMT-py based).
    Distinctive mechanism: RetroPrime is explicitly *two* independently
    trained sequence Transformers chained in a synthon-then-reactant
    pipeline mirroring expert retrosynthetic strategy -- stage 1
    ("synthon-generation Transformer") takes the tokenized product SMILES
    and predicts the disconnected synthon SMILES (bond broken, no leaving
    groups attached yet); stage 2 ("reactant-generation Transformer") takes
    those synthon tokens and predicts the final reactant SMILES by
    attaching leaving groups. Reproduced here as two chained encoder-decoder
    Transformers (shared tiny config, separate weights) wired
    product-tokens -> stage1 -> synthon-tokens -> stage2 -> reactant-tokens,
    with the intermediate synthon representation exposed as a genuine
    intermediate tensor (argmax-embedded, not merely conceptual) so the
    two-stage decomposition is visible in the traced graph.

All models use random initialization and small dims; this module is an
architecture catalog, not a trained-weights zoo.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# cand_01260: R-SMILES -- root-aligned-SMILES seq2seq Transformer
# ---------------------------------------------------------------------------


def root_align(tokens: Tensor, anchor_pos: Tensor) -> Tensor:
    """Rotate each token sequence in a batch to start at its anchor position.

    Mirrors the R-SMILES input-representation trick of re-rooting SMILES at
    a shared anchor atom before tokenization, so that source and target
    streams share a similar token order.

    Parameters
    ----------
    tokens : Tensor
        Integer token ids, shape ``(batch, seq_len)``.
    anchor_pos : Tensor
        Per-example rotation offset (the "root" position), shape ``(batch,)``.

    Returns
    -------
    Tensor
        Rotated token ids, same shape as ``tokens``.
    """

    batch, seq_len = tokens.shape
    idx = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
    idx = (idx + anchor_pos.unsqueeze(1)) % seq_len
    return torch.gather(tokens, 1, idx)


class RSmilesTransformer(nn.Module):
    """Root-aligned-SMILES seq2seq Transformer (R-SMILES).

    A standard Transformer encoder-decoder whose source/target token streams
    are first root-aligned (rotated to a shared anchor atom position),
    mirroring the tight product/reactant token correspondence that is
    R-SMILES's distinctive contribution.
    """

    def __init__(
        self, vocab_size: int = 64, d_model: int = 32, nhead: int = 4, n_layers: int = 2
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(vocab_size, d_model)
        self.tgt_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(128, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=4 * d_model,
            batch_first=True,
        )
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(
        self, src_tokens: Tensor, tgt_tokens: Tensor, src_anchor: Tensor, tgt_anchor: Tensor
    ) -> Tensor:
        """Root-align both streams, then run seq2seq translation.

        Parameters
        ----------
        src_tokens : Tensor
            Product-SMILES token ids, shape ``(batch, src_len)``.
        tgt_tokens : Tensor
            Reactant-SMILES token ids, shape ``(batch, tgt_len)``.
        src_anchor : Tensor
            Per-example root-atom offset for the source stream, shape ``(batch,)``.
        tgt_anchor : Tensor
            Per-example root-atom offset for the target stream, shape ``(batch,)``.

        Returns
        -------
        Tensor
            Per-position vocabulary logits, shape ``(batch, tgt_len, vocab_size)``.
        """

        src_tokens = root_align(src_tokens, src_anchor)
        tgt_tokens = root_align(tgt_tokens, tgt_anchor)

        src_len = src_tokens.shape[1]
        tgt_len = tgt_tokens.shape[1]
        src_pos = torch.arange(src_len, device=src_tokens.device)
        tgt_pos = torch.arange(tgt_len, device=tgt_tokens.device)

        src = self.src_embed(src_tokens) + self.pos_embed(src_pos)
        tgt = self.tgt_embed(tgt_tokens) + self.pos_embed(tgt_pos)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(tgt.device)
        hidden = self.transformer(src, tgt, tgt_mask=causal_mask)
        return self.out_proj(hidden)


def build_r_smiles() -> nn.Module:
    """Build a compact R-SMILES root-aligned Transformer.

    Returns
    -------
    nn.Module
        ``RSmilesTransformer`` in eval mode.
    """

    return RSmilesTransformer().eval()


def example_input_r_smiles() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Create example input for :func:`build_r_smiles`.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        ``(src_tokens, tgt_tokens, src_anchor, tgt_anchor)``.
    """

    torch.manual_seed(0)
    src_tokens = torch.randint(0, 64, (2, 20))
    tgt_tokens = torch.randint(0, 64, (2, 16))
    src_anchor = torch.randint(0, 20, (2,))
    tgt_anchor = torch.randint(0, 16, (2,))
    return src_tokens, tgt_tokens, src_anchor, tgt_anchor


# ---------------------------------------------------------------------------
# cand_01261: ReactionT5 -- multitask T5 over chemical reactions
# ---------------------------------------------------------------------------


class ReactionT5(nn.Module):
    """Multitask T5 for forward/retro reaction prediction and yield regression.

    Wraps a tiny ``transformers.T5ForConditionalGeneration`` (matching
    ReactionT5's pretrained-T5 backbone) with an added linear yield-regression
    head pooling the encoder's final hidden state, mirroring ReactionT5's
    reuse of one pretrained encoder-decoder across forward-product,
    retrosynthesis, and yield-prediction tasks.
    """

    def __init__(
        self, vocab_size: int = 96, d_model: int = 32, n_layers: int = 2, n_heads: int = 4
    ) -> None:
        super().__init__()
        from transformers import T5Config, T5ForConditionalGeneration

        config = T5Config(
            vocab_size=vocab_size,
            d_model=d_model,
            d_ff=4 * d_model,
            num_layers=n_layers,
            num_decoder_layers=n_layers,
            num_heads=n_heads,
            d_kv=d_model // n_heads,
            decoder_start_token_id=0,
            pad_token_id=0,
        )
        self.t5 = T5ForConditionalGeneration(config)
        self.yield_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )

    def forward(self, input_ids: Tensor, decoder_input_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Run the shared T5 encoder-decoder and the yield-regression head.

        Parameters
        ----------
        input_ids : Tensor
            Task-prefixed source token ids, shape ``(batch, src_len)``.
        decoder_input_ids : Tensor
            Decoder input token ids, shape ``(batch, tgt_len)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(seq_logits, yield_pred)``: per-position vocabulary logits
            ``(batch, tgt_len, vocab_size)`` from the decoder, and a scalar
            yield prediction ``(batch, 1)`` pooled from the encoder.
        """

        encoder_outputs = self.t5.encoder(input_ids=input_ids)
        decoder_outputs = self.t5.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
        )
        seq_logits = self.t5.lm_head(decoder_outputs.last_hidden_state)
        pooled = encoder_outputs.last_hidden_state.mean(dim=1)
        yield_pred = self.yield_head(pooled)
        return seq_logits, yield_pred


def build_reactiont5() -> nn.Module:
    """Build a compact multitask ReactionT5.

    Returns
    -------
    nn.Module
        ``ReactionT5`` in eval mode.
    """

    return ReactionT5().eval()


def example_input_reactiont5() -> tuple[Tensor, Tensor]:
    """Create example input for :func:`build_reactiont5`.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(input_ids, decoder_input_ids)``.
    """

    torch.manual_seed(1)
    input_ids = torch.randint(1, 96, (2, 18))
    decoder_input_ids = torch.randint(1, 96, (2, 12))
    return input_ids, decoder_input_ids


# ---------------------------------------------------------------------------
# cand_01262: REINVENT -- SMILES character-level GRU prior
# ---------------------------------------------------------------------------


class ReinventRNN(nn.Module):
    """REINVENT's classic SMILES-RNN prior: embedding -> GRU stack -> vocab head.

    Mirrors ``reinvent/models/reinvent/models/rnn.py::RNN``: an autoregressive
    character-level SMILES generator later fine-tuned with reinforcement
    learning (the RL loop is a training-time procedure, not part of the
    traced network).
    """

    def __init__(
        self,
        voc_size: int = 48,
        embedding_size: int = 32,
        layer_size: int = 64,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.layer_size = layer_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(voc_size, embedding_size)
        self.rnn = nn.GRU(embedding_size, layer_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(layer_size, voc_size)

    def forward(self, input_vector: Tensor) -> Tensor:
        """Autoregress vocabulary logits over a SMILES token sequence.

        Parameters
        ----------
        input_vector : Tensor
            Integer token ids, shape ``(batch, seq_len)``.

        Returns
        -------
        Tensor
            Per-position vocabulary logits, shape ``(batch, seq_len, voc_size)``.
        """

        batch_size, seq_len = input_vector.shape
        hidden = torch.zeros(
            self.num_layers, batch_size, self.layer_size, device=input_vector.device
        )
        embedded = self.embedding(input_vector)
        output, _ = self.rnn(embedded, hidden)
        return self.linear(output.reshape(-1, self.layer_size)).view(batch_size, seq_len, -1)


def build_reinvent() -> nn.Module:
    """Build a compact REINVENT SMILES-RNN prior.

    Returns
    -------
    nn.Module
        ``ReinventRNN`` in eval mode.
    """

    return ReinventRNN().eval()


def example_input_reinvent() -> Tensor:
    """Create example input for :func:`build_reinvent`.

    Returns
    -------
    Tensor
        Token ids, shape ``(4, 24)``.
    """

    torch.manual_seed(2)
    return torch.randint(0, 48, (4, 24))


# ---------------------------------------------------------------------------
# cand_01263: ResGen -- hierarchical multiscale 3D pocket-aware ligand generator
# ---------------------------------------------------------------------------


class ResGen(nn.Module):
    """Hierarchical parallel-multiscale 3D ligand generator (ResGen).

    Encodes the pocket at both atom and residue granularity, picks the next
    fragment's "focal" residue with a single global-autoregressive step, then
    places every atom of that fragment in one *parallel* (non-autoregressive)
    per-slot pass conditioned on the focal embedding -- the hierarchical
    global/parallel-local split that gives ResGen its ~8x speedup over
    strictly sequential atom-by-atom baselines.
    """

    def __init__(
        self,
        atom_feat_dim: int = 16,
        residue_feat_dim: int = 20,
        hidden_dim: int = 32,
        n_atom_types: int = 10,
        max_atoms_per_fragment: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_atoms_per_fragment = max_atoms_per_fragment

        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_feat_dim + 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.residue_encoder = nn.Sequential(
            nn.Linear(residue_feat_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Multiscale fusion: pool atom-level features into each residue's context.
        self.multiscale_fuse = nn.Linear(2 * hidden_dim, hidden_dim)

        # Global autoregressive step: score each residue as the next focal point.
        self.focal_scorer = nn.Linear(hidden_dim, 1)

        # Parallel atom-level sub-step: every atom slot of the next fragment
        # predicted simultaneously, conditioned on the focal embedding.
        self.slot_query = nn.Parameter(torch.randn(max_atoms_per_fragment, hidden_dim))
        self.slot_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.element_head = nn.Linear(hidden_dim, n_atom_types)
        self.offset_head = nn.Linear(hidden_dim, 3)

    def forward(
        self,
        atom_feat: Tensor,
        atom_pos: Tensor,
        residue_feat: Tensor,
        residue_pos: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run the hierarchical global-focal + parallel-atom generation step.

        Parameters
        ----------
        atom_feat : Tensor
            Ligand/pocket atom features, shape ``(batch, n_atoms, atom_feat_dim)``.
        atom_pos : Tensor
            Atom 3D coordinates, shape ``(batch, n_atoms, 3)``.
        residue_feat : Tensor
            Pocket residue features, shape ``(batch, n_residues, residue_feat_dim)``.
        residue_pos : Tensor
            Residue centroid 3D coordinates, shape ``(batch, n_residues, 3)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``(focal_scores, element_logits, offsets)``: per-residue focal
            scores ``(batch, n_residues)``, per-slot element-type logits
            ``(batch, max_atoms_per_fragment, n_atom_types)``, and per-slot
            3D offsets ``(batch, max_atoms_per_fragment, 3)``.
        """

        atom_h = self.atom_encoder(torch.cat([atom_feat, atom_pos], dim=-1))
        residue_h = self.residue_encoder(torch.cat([residue_feat, residue_pos], dim=-1))

        # Multiscale fusion: pool atom-level context into the residue stream.
        atom_pooled = atom_h.mean(dim=1, keepdim=True).expand(-1, residue_h.shape[1], -1)
        residue_h = self.multiscale_fuse(torch.cat([residue_h, atom_pooled], dim=-1))

        # Global autoregressive step (single-shot focal-residue scoring).
        focal_scores = self.focal_scorer(residue_h).squeeze(-1)
        focal_weights = torch.softmax(focal_scores, dim=-1).unsqueeze(-1)
        focal_embed = (residue_h * focal_weights).sum(dim=1, keepdim=True)

        # Parallel atom-level sub-step: all fragment atoms placed at once.
        batch = atom_feat.shape[0]
        queries = self.slot_query.unsqueeze(0).expand(batch, -1, -1)
        slot_context, _ = self.slot_attn(
            queries,
            focal_embed.expand(-1, self.max_atoms_per_fragment, -1),
            focal_embed.expand(-1, self.max_atoms_per_fragment, -1),
        )
        element_logits = self.element_head(slot_context)
        offsets = self.offset_head(slot_context)

        return focal_scores, element_logits, offsets


def build_resgen() -> nn.Module:
    """Build a compact ResGen hierarchical multiscale ligand generator.

    Returns
    -------
    nn.Module
        ``ResGen`` in eval mode.
    """

    return ResGen().eval()


def example_input_resgen() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Create example input for :func:`build_resgen`.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        ``(atom_feat, atom_pos, residue_feat, residue_pos)``.
    """

    torch.manual_seed(3)
    atom_feat = torch.randn(2, 12, 16)
    atom_pos = torch.randn(2, 12, 3)
    residue_feat = torch.randn(2, 9, 20)
    residue_pos = torch.randn(2, 9, 3)
    return atom_feat, atom_pos, residue_feat, residue_pos


# ---------------------------------------------------------------------------
# cand_01264: Retroformer -- local-global attention retrosynthesis transformer
# ---------------------------------------------------------------------------


class LocalGlobalAttention(nn.Module):
    """Split-head local/global attention with edge-feature gating and update.

    Half the attention heads run dense "global" self-attention over the full
    token sequence; the other half run "local" attention restricted (and
    gated) to a bonded-atom-pair adjacency, keyed by a projected edge
    feature. The local branch also produces an *updated* edge feature from
    the two endpoint node outputs, mirroring
    ``retroformer/models/module.py::MultiHeadedAttention``.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert n_heads % 2 == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.edge_project = nn.Linear(d_model, d_model // 2)
        self.edge_update = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )

    def _shape(self, x: Tensor, batch: int, seq_len: int) -> Tensor:
        return x.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)

    def forward(self, x: Tensor, bond_adj: Tensor, edge_feature: Tensor) -> tuple[Tensor, Tensor]:
        """Run one local-global attention layer with an edge-feature co-update.

        Parameters
        ----------
        x : Tensor
            Token hidden states, shape ``(batch, seq_len, d_model)``.
        bond_adj : Tensor
            Binary bonded-pair adjacency (1 = local attention allowed),
            shape ``(batch, seq_len, seq_len)``.
        edge_feature : Tensor
            Per-pair edge features, shape ``(batch, seq_len, seq_len, d_model)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(output, edge_feature_updated)``: attended hidden states
            ``(batch, seq_len, d_model)`` and the updated edge feature of
            the same shape as the input edge feature.
        """

        batch, seq_len, _ = x.shape
        half = self.n_heads // 2

        q = self._shape(self.q_proj(x), batch, seq_len)
        k = self._shape(self.k_proj(x), batch, seq_len)
        v = self._shape(self.v_proj(x), batch, seq_len)

        q_global, q_local = q[:, :half], q[:, half:]
        k_global, k_local = k[:, :half], k[:, half:]
        v_global, v_local = v[:, :half], v[:, half:]

        # Global heads: ordinary dense self-attention.
        scale = math.sqrt(self.d_head)
        score_global = torch.matmul(q_global, k_global.transpose(-2, -1)) / scale
        attn_global = torch.softmax(score_global, dim=-1)
        context_global = torch.matmul(attn_global, v_global)

        # Local heads: bond-mask-gated attention keyed by an edge-feature
        # projection multiplied into the local keys.
        edge_proj = self.edge_project(edge_feature)  # (batch, seq, seq, half*d_head)
        edge_proj = edge_proj.view(batch, seq_len, seq_len, half, self.d_head).permute(
            0, 3, 1, 2, 4
        )
        k_local_gated = k_local.unsqueeze(3) * edge_proj  # (batch, half, seq, seq, d_head)
        score_local = (q_local.unsqueeze(3) * k_local_gated).sum(
            -1
        ) / scale  # (batch, half, seq_q, seq_k)
        mask = bond_adj.unsqueeze(1).expand(-1, half, -1, -1)
        score_local = score_local.masked_fill(mask == 0, -1e9)
        attn_local = torch.softmax(score_local, dim=-1)
        context_local = torch.matmul(attn_local, v_local)

        context = torch.cat([context_global, context_local], dim=1)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        output = self.out_proj(context)

        # Edge co-update: concat the two endpoint node outputs, MLP back to edge-dim.
        node_i = output.unsqueeze(2).expand(-1, -1, seq_len, -1)
        node_j = output.unsqueeze(1).expand(-1, seq_len, -1, -1)
        edge_feature_updated = self.edge_update(torch.cat([node_i, node_j], dim=-1))

        return output, edge_feature_updated


class RetroformerEncoderLayer(nn.Module):
    """One Retroformer encoder layer: local-global attention + feedforward."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.attn = LocalGlobalAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm_edge = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, bond_adj: Tensor, edge_feature: Tensor) -> tuple[Tensor, Tensor]:
        """Apply local-global attention then a feedforward block, pre-norm style."""

        normed = self.norm1(x)
        attn_out, edge_feature_updated = self.attn(normed, bond_adj, edge_feature)
        x = x + attn_out
        edge_feature = self.norm_edge(edge_feature + edge_feature_updated)
        x = x + self.ff(self.norm2(x))
        return x, edge_feature


class Retroformer(nn.Module):
    """Retroformer: local-global attention encoder + reaction-center head + decoder.

    A Transformer encoder whose attention is split into dense global heads
    and bond-graph-gated local heads with an explicit edge-feature co-update
    across layers, followed by a reaction-center scorer and a standard
    autoregressive decoder over reactant tokens.
    """

    def __init__(
        self, vocab_size: int = 64, d_model: int = 32, n_heads: int = 4, n_layers: int = 2
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(64, d_model)
        self.bond_embed = nn.Linear(7, d_model)

        self.layers = nn.ModuleList(
            [RetroformerEncoderLayer(d_model, n_heads, 4 * d_model) for _ in range(n_layers)]
        )

        self.reaction_center_head = nn.Linear(d_model, 1)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.tgt_embed = nn.Embedding(vocab_size, d_model)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(
        self, src_tokens: Tensor, bond_feat: Tensor, tgt_tokens: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Encode a product molecule with local-global attention, then decode reactants.

        Parameters
        ----------
        src_tokens : Tensor
            Product-SMILES token ids, shape ``(batch, src_len)``.
        bond_feat : Tensor
            Raw per-pair bond features, shape ``(batch, src_len, src_len, 7)``.
        tgt_tokens : Tensor
            Reactant-SMILES token ids, shape ``(batch, tgt_len)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(reaction_center_scores, seq_logits)``: per-atom reaction-center
            scores ``(batch, src_len)`` and per-position decoder vocabulary
            logits ``(batch, tgt_len, vocab_size)``.
        """

        batch, src_len = src_tokens.shape
        bond_adj = (bond_feat.abs().sum(-1) > 0).float()
        edge_feature = self.bond_embed(bond_feat)

        pos = torch.arange(src_len, device=src_tokens.device)
        x = self.token_embed(src_tokens) + self.pos_embed(pos)

        for layer in self.layers:
            x, edge_feature = layer(x, bond_adj, edge_feature)

        reaction_center_scores = self.reaction_center_head(x).squeeze(-1)

        tgt_len = tgt_tokens.shape[1]
        tgt_pos = torch.arange(tgt_len, device=tgt_tokens.device)
        tgt = self.tgt_embed(tgt_tokens) + self.pos_embed(tgt_pos)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(tgt.device)
        decoded = self.decoder(tgt, x, tgt_mask=causal_mask)
        seq_logits = self.out_proj(decoded)

        return reaction_center_scores, seq_logits


def build_retroformer() -> nn.Module:
    """Build a compact Retroformer local-global-attention retrosynthesis model.

    Returns
    -------
    nn.Module
        ``Retroformer`` in eval mode.
    """

    return Retroformer().eval()


def example_input_retroformer() -> tuple[Tensor, Tensor, Tensor]:
    """Create example input for :func:`build_retroformer`.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(src_tokens, bond_feat, tgt_tokens)``.
    """

    torch.manual_seed(4)
    src_len = 10
    src_tokens = torch.randint(0, 64, (2, src_len))
    raw_bond = torch.rand(2, src_len, src_len, 7)
    bond_feat = 0.5 * (raw_bond + raw_bond.transpose(1, 2))
    tgt_tokens = torch.randint(0, 64, (2, 8))
    return src_tokens, bond_feat, tgt_tokens


# ---------------------------------------------------------------------------
# cand_01265: RetroPrime -- two-stage synthon-then-reactant Transformer
# ---------------------------------------------------------------------------


class SmallSeq2SeqTransformer(nn.Module):
    """A small shared-shape Transformer encoder-decoder used by both RetroPrime stages."""

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(vocab_size, d_model)
        self.tgt_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(64, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=4 * d_model,
            batch_first=True,
        )
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, src_tokens: Tensor, tgt_tokens: Tensor) -> tuple[Tensor, Tensor]:
        """Encode ``src_tokens`` and decode ``tgt_tokens``, returning logits and encoder memory."""

        src_len = src_tokens.shape[1]
        tgt_len = tgt_tokens.shape[1]
        src_pos = torch.arange(src_len, device=src_tokens.device)
        tgt_pos = torch.arange(tgt_len, device=tgt_tokens.device)

        src = self.src_embed(src_tokens) + self.pos_embed(src_pos)
        tgt = self.tgt_embed(tgt_tokens) + self.pos_embed(tgt_pos)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(tgt.device)
        memory = self.transformer.encoder(src)
        decoded = self.transformer.decoder(tgt, memory, tgt_mask=causal_mask)
        logits = self.out_proj(decoded)
        return logits, memory


class RetroPrime(nn.Module):
    """RetroPrime: chained synthon-generation then reactant-generation Transformers.

    Two independently-weighted Transformers wired in a pipeline mirroring
    expert retrosynthetic strategy: stage 1 predicts disconnected synthons
    from the product; the synthon prediction is discretized (argmax) and
    re-embedded as genuine intermediate tokens; stage 2 predicts the final
    reactants (with leaving groups attached) from those synthon tokens.
    """

    def __init__(
        self, vocab_size: int = 64, d_model: int = 32, n_heads: int = 4, n_layers: int = 2
    ) -> None:
        super().__init__()
        self.synthon_stage = SmallSeq2SeqTransformer(vocab_size, d_model, n_heads, n_layers)
        self.reactant_stage = SmallSeq2SeqTransformer(vocab_size, d_model, n_heads, n_layers)

    def forward(
        self,
        product_tokens: Tensor,
        synthon_decoder_tokens: Tensor,
        reactant_decoder_tokens: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run the two-stage synthon-then-reactant pipeline.

        Parameters
        ----------
        product_tokens : Tensor
            Tokenized product SMILES, shape ``(batch, prod_len)``.
        synthon_decoder_tokens : Tensor
            Teacher-forced synthon decoder input tokens, shape ``(batch, synthon_len)``.
        reactant_decoder_tokens : Tensor
            Teacher-forced reactant decoder input tokens, shape ``(batch, reactant_len)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(synthon_logits, reactant_logits)``: stage-1 synthon vocabulary
            logits ``(batch, synthon_len, vocab_size)`` and stage-2 reactant
            vocabulary logits ``(batch, reactant_len, vocab_size)``.
        """

        synthon_logits, _ = self.synthon_stage(product_tokens, synthon_decoder_tokens)

        # Discretize the predicted synthon tokens into a genuine intermediate
        # representation feeding stage 2, mirroring the decomposed-synthon
        # hand-off between RetroPrime's two independently trained models.
        synthon_tokens = synthon_logits.argmax(dim=-1).detach()

        reactant_logits, _ = self.reactant_stage(synthon_tokens, reactant_decoder_tokens)
        return synthon_logits, reactant_logits


def build_retroprime() -> nn.Module:
    """Build a compact two-stage RetroPrime model.

    Returns
    -------
    nn.Module
        ``RetroPrime`` in eval mode.
    """

    return RetroPrime().eval()


def example_input_retroprime() -> tuple[Tensor, Tensor, Tensor]:
    """Create example input for :func:`build_retroprime`.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(product_tokens, synthon_decoder_tokens, reactant_decoder_tokens)``.
    """

    torch.manual_seed(5)
    product_tokens = torch.randint(0, 64, (2, 14))
    synthon_decoder_tokens = torch.randint(0, 64, (2, 10))
    reactant_decoder_tokens = torch.randint(0, 64, (2, 12))
    return product_tokens, synthon_decoder_tokens, reactant_decoder_tokens


MENAGERIE_ENTRIES = [
    ("R-SMILES", "build_r_smiles", "example_input_r_smiles", "2022", "GEN"),
    ("ReactionT5", "build_reactiont5", "example_input_reactiont5", "2023", "GEN"),
    ("REINVENT", "build_reinvent", "example_input_reinvent", "2024", "GEN"),
    ("ResGen", "build_resgen", "example_input_resgen", "2023", "BIO"),
    ("Retroformer", "build_retroformer", "example_input_retroformer", "2022", "GEN"),
    ("RetroPrime", "build_retroprime", "example_input_retroprime", "2021", "GEN"),
]
