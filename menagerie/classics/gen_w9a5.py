"""Compact faithful reimplementations for build_queue rows 31-36 (W9A5).

Sources checked (repo browsed via ``gh api`` / web, no clone/pip-install):

  * ORGAN: Guimaraes, Sanchez-Lengeling, Outeiral, Farias, Aspuru-Guzik,
    "Objective-Reinforced Generative Adversarial Networks (ORGAN) for
    Sequence Generation Models", arXiv:1705.10843 (2017). Official repo
    github.com/gablg1/ORGAN, ``organ/generator.py`` (``Generator``) and
    ``organ/discriminator.py`` (``Discriminator``). Distinctive mechanism:
    a SeqGAN-style character-level LSTM generator that autoregressively
    emits one-hot SMILES-token logits, trained adversarially against a
    CNN-over-character-embeddings discriminator built from parallel
    multi-filter-width 1-D convolutions ("multi_filter_sizes") + max-pool
    + a highway network (``discriminator.py::highway``, a sigmoid
    transform-gate mix of a nonlinear branch and the identity branch)
    feeding a final linear classifier -- with REINFORCE-derived rollout
    rewards (Monte-Carlo rollout policy, out of scope for a single traced
    forward pass) combining the discriminator score with hand-crafted
    objective rewards (e.g. drug-likeness) at training time. Reimplemented
    here as one LSTM-generator forward step (embedding -> LSTMCell ->
    vocab logits, unrolled a fixed number of steps) plus the CNN+highway
    discriminator scoring a fixed-length one-hot/embedded sequence.
  * Parrot: Wang, Hsieh, Yin, Wang, Li, Deng, Jiang, Wu, Du, Chen, Li, Liu,
    Wang, Luo, Hou, Yao, "Generic Interpretable Reaction Condition
    Predictions with Open Reaction Condition Datasets and Unsupervised
    Learning of Reaction Center", Research 2023, arXiv:2202.01602. Official
    repo github.com/wangxr0526/Parrot, ``models/parrot_model.py``
    (``ParrotConditionModel``, subclassing ``BertForSequenceClassification``).
    Distinctive mechanism: a BERT encoder reads the reaction SMILES (via
    ``self.bert``), and a Transformer *decoder* (``models/model_layer.py::
    TransformerDecoder``) autoregressively cross-attends from
    reaction-condition-label token embeddings (catalyst/solvent/reagent
    vocabulary, ``self.tgt_tok_emb`` + sinusoidal ``PositionalEncoding``)
    to the BERT memory, emitting per-step condition-label logits via a
    final linear ``generator`` head -- i.e. reaction condition prediction
    is cast as encoder-decoder *sequence* generation over a small
    condition-label vocabulary, not single-label classification.
    Reimplemented compactly with a small ``transformers`` ``BertModel``
    encoder + a ``nn.TransformerDecoder`` reading learned condition-label
    embeddings, matching the encoder/decoder split exactly.
  * PGCGM: Zhao (Yong Zhao) et al., "Physics Guided Generative Adversarial
    Networks for Generation of Crystal Materials with High Symmetry
    Constraints", arXiv:2201.11932 (2022). Official repo
    github.com/MilesZhao/PGCGM, ``model.py`` (``Generator``,
    ``Discriminator``). Distinctive mechanism: a conditional GAN generator
    that fuses three inputs -- a Gaussian noise vector, a per-element
    one-hot/feature tensor (``ele_block``, 1-D convs over the element
    axis), and a fixed space-group symmetry-operation matrix
    (``sp_block``, 2-D convs over the (n_symm_ops x 4 x 4) affine matrices
    for the crystal's space group) -- via concatenated embeddings that
    are then upsampled with a 2-D ``ConvTranspose2d`` stack into
    fractional atomic coordinates, and separately regressed (through
    a small MLP) into lattice-length parameters; a matching discriminator
    scores (crystal-coordinate-tensor, symmetry-matrix) pairs with 2-D
    convolutions. Physics is injected structurally by conditioning
    generation on the space-group's exact symmetry operators rather than
    letting the network discover them. Reimplemented with the same
    three-branch generator (noise/element/space-group fusion ->
    deconv coordinates + MLP lattice lengths) and matching discriminator.
  * Pocket2Mol: Peng, Luo, Guan, Xie, Peng, Ma, "Pocket2Mol: Efficient
    Molecular Sampling Based on 3D Protein Pockets", ICML 2022,
    arXiv:2205.07249. Official repo github.com/pengxingang/Pocket2Mol,
    ``models/maskfill.py`` (``MaskFillModelVN``) with
    ``models/invariant.py`` (``GVLinear``, ``GVPerceptronVN``,
    ``MessageModule``) and ``models/position.py``
    (``PositionPredictor``). Distinctive mechanism: an E(3)-equivariant
    "vector-neuron" (geometric-vector-perceptron-style) message-passing
    encoder that carries *paired* scalar and 3-D-vector per-atom features
    (``GVLinear`` mixes scalar/vector channels while keeping the vector
    channel equivariant via norm-based scalar gating, never breaking
    rotation equivariance with a raw linear map on vector coordinates)
    over a composed protein-pocket + partial-ligand point cloud;
    downstream heads operate directly on these paired features: a
    frontier classifier picks which existing atom to grow from, a
    mixture-density-network position predictor
    (``PositionPredictor.mu_net/logsigma_net/pi_net``) predicts a 3-D
    Gaussian-mixture over the next atom's *relative* position (so the
    predicted position is equivariant by construction), and an
    element/bond classifier types the new atom and its bonds to existing
    context atoms -- one step of this autoregressive atom-by-atom
    "frontier -> position -> element/bond" pipeline is what is traced
    here (the full generation loop is a Python-level sampling wrapper).
  * PocketFlow: Zhang, Liu, et al., "Generalized Protein Pocket Generation
    with Prior-Informed Flow Matching", NeurIPS 2024, arXiv:2409.19520.
    Repo github.com/zaixizhang/PocketFlow is a placeholder (README only,
    "code will be released together [with a forthcoming journal
    submission]" -- no source available) as of this check, so this
    reimplementation is built directly from the paper's architecture
    description (arXiv HTML, Section on model design): an IPA
    (Invariant-Point-Attention, AF2-style)-based encoder modified from
    FrameDiff combined with sequence-level Transformer layers encodes a
    composed protein-residue + ligand-atom point cloud at flow-matching
    time ``t``; separate small MLP heads on the shared residue/atom
    embeddings predict (a) Calpha coordinate velocities, (b) SO(3)
    backbone-orientation velocities (via a skew-symmetric generator, the
    tangent space of SO(3)), (c) torsion-angle velocities on the
    hypertorus, (d) residue-type logits, and (e) protein-ligand
    interaction-type logits -- flow matching regresses straight-line
    (Euclidean for coordinates/probabilities, geodesic for SO(3)/torus)
    paths from a noise prior at t=0 to data at t=1. Reimplemented here as
    a compact IPA-style attention block (query/key/value gated by pairwise
    invariant distance features, matching AF2 IPA's invariance mechanism)
    stacked with a Transformer-encoder sequence layer, feeding the five
    described prediction heads for one point-in-time (t, x_t) forward
    pass.
  * PSVAE (Principal Subgraph VAE): Kong, Huang, Liu, Yin, Cui, Ma, Zhang,
    "Molecule Generation by Principal Subgraph Mining and Assembling",
    NeurIPS 2022, arXiv:2106.15098. Official repo
    github.com/THUNLP-MT/PS-VAE, ``src/modules/encoder.py`` (``Encoder``,
    a ``GINEConv``-based molecular-graph encoder) and
    ``src/modules/vae_piece_decoder.py`` (``VAEPieceDecoder``).
    Distinctive mechanism: molecules are first tokenized (offline, a
    BPE-like frequency-driven merge algorithm, not a neural network) into
    a vocabulary of "principal subgraphs" (frequent, chemically valid
    multi-atom fragments) rather than single atoms; a GNN encoder over
    the atom-level graph is pooled into a VAE latent
    (``VAEPieceDecoder.rsample``, standard reparameterization trick); the
    decoder then reconstructs the molecule as a *sequence of principal
    -subgraph pieces* via a latent-conditioned GRU
    (``latent_to_rnn_hidden`` seeds the GRU state, autoregressively
    emitting piece-vocabulary logits, ``to_vocab``) interleaved with an
    edge-type classifier (``edge_predictor``, an MLP over
    concatenated (src-node, dst-node, latent) triples) that links
    fragment-attachment atoms together. Reimplemented here as the GINE
    graph encoder + VAE bottleneck + one GRU piece-decoding step + the
    edge-type classifier over a fixed small atom/piece graph.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# ORGAN
# ---------------------------------------------------------------------------


class _Highway(nn.Module):
    """Highway network layer (Srivastava et al. 2015), as used by ORGAN's CNN
    discriminator (``organ/discriminator.py::highway``).
    """

    def __init__(self, dim: int, bias: float = -2.0) -> None:
        super().__init__()
        self.transform = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.gate.bias.data.fill_(bias)

    def forward(self, x: Tensor) -> Tensor:
        """Mix a nonlinear transform branch with the identity via a sigmoid gate."""

        g = torch.sigmoid(self.gate(x))
        t = F.relu(self.transform(x))
        return g * t + (1.0 - g) * x


class OrganGAN(nn.Module):
    """SeqGAN-style ORGAN generator + CNN/highway discriminator.

    Reproduces ``organ/generator.py::Generator`` (character-LSTM emitting
    one-hot SMILES-token logits over a fixed sequence length) and
    ``organ/discriminator.py::Discriminator`` (multi-filter-width Conv1d
    bank + max-pool + highway network + linear classifier) in a single
    module: the generator unrolls autoregressively from a start token,
    then the discriminator scores the resulting soft (embedded) sequence.
    """

    def __init__(
        self,
        vocab_size: int = 24,
        emb_dim: int = 16,
        hidden_dim: int = 32,
        seq_len: int = 12,
        filter_sizes: tuple[int, ...] = (2, 3, 4),
        num_filters: int = 8,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        # generator
        self.g_embedding = nn.Embedding(vocab_size, emb_dim)
        self.g_cell = nn.LSTMCell(emb_dim, hidden_dim)
        self.g_out = nn.Linear(hidden_dim, vocab_size)
        # discriminator
        self.d_embedding = nn.Embedding(vocab_size, emb_dim)
        self.d_convs = nn.ModuleList([nn.Conv1d(emb_dim, num_filters, k) for k in filter_sizes])
        d_feat_dim = num_filters * len(filter_sizes)
        self.d_highway = _Highway(d_feat_dim)
        self.d_out = nn.Linear(d_feat_dim, 1)

    def generate(self, start_token: Tensor) -> Tensor:
        """Autoregressively unroll the LSTM generator into token logits.

        Parameters
        ----------
        start_token : Tensor
            Shape ``(batch,)`` integer start-token ids.

        Returns
        -------
        Tensor
            Shape ``(batch, seq_len, vocab_size)`` per-step token logits.
        """

        batch = start_token.shape[0]
        h = start_token.new_zeros(batch, self.g_cell.hidden_size, dtype=torch.float32)
        c = torch.zeros_like(h)
        x = self.g_embedding(start_token)
        logits_steps = []
        for _ in range(self.seq_len):
            h, c = self.g_cell(x, (h, c))
            step_logits = self.g_out(h)
            logits_steps.append(step_logits)
            next_token = step_logits.argmax(dim=-1)
            x = self.g_embedding(next_token)
        return torch.stack(logits_steps, dim=1)

    def discriminate(self, token_ids: Tensor) -> Tensor:
        """Score a fixed-length token sequence with the CNN+highway discriminator.

        Parameters
        ----------
        token_ids : Tensor
            Shape ``(batch, seq_len)`` integer token ids.

        Returns
        -------
        Tensor
            Shape ``(batch, 1)`` real-vs-fake logit.
        """

        emb = self.d_embedding(token_ids).transpose(1, 2)  # (batch, emb_dim, seq_len)
        pooled = [F.relu(conv(emb)).amax(dim=-1) for conv in self.d_convs]
        feat = torch.cat(pooled, dim=-1)
        feat = self.d_highway(feat)
        return self.d_out(feat)

    def forward(self, start_token: Tensor) -> Tensor:
        """Generate a sequence, then discriminate it (full ORGAN adversarial pipeline)."""

        logits = self.generate(start_token)
        gen_tokens = logits.argmax(dim=-1)
        return self.discriminate(gen_tokens)


def build_organ() -> OrganGAN:
    """Build a compact :class:`OrganGAN`."""

    return OrganGAN().eval()


def example_input_organ() -> Tensor:
    """Create example input for :func:`build_organ`.

    Returns
    -------
    Tensor
        Shape ``(4,)`` integer start-token ids.
    """

    torch.manual_seed(0)
    return torch.zeros(4, dtype=torch.long)


# ---------------------------------------------------------------------------
# Parrot
# ---------------------------------------------------------------------------


class _SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (matches
    ``models/model_layer.py::PositionalEncoding`` in the Parrot repo).
    """

    def __init__(self, dim: int, max_len: int = 64) -> None:
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encodings up to ``x``'s sequence length."""

        return x + self.pe[: x.shape[1]].unsqueeze(0)


class ParrotConditionModel(nn.Module):
    """BERT-encoder + Transformer-decoder reaction-condition predictor.

    Reproduces the encoder/decoder split of ``models/parrot_model.py::
    ParrotConditionModel``: a BERT encoder reads the reaction SMILES
    tokens, and a Transformer decoder autoregressively cross-attends from
    reaction-condition-label embeddings to the BERT memory, emitting
    per-step condition-label logits via a linear ``generator`` head.
    """

    def __init__(
        self,
        vocab_size: int = 64,
        cond_vocab_size: int = 16,
        d_model: int = 32,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 64,
    ) -> None:
        super().__init__()
        from transformers import BertConfig, BertModel

        bert_cfg = BertConfig(
            vocab_size=vocab_size,
            hidden_size=d_model,
            num_hidden_layers=num_encoder_layers,
            num_attention_heads=nhead,
            intermediate_size=dim_feedforward,
            max_position_embeddings=64,
        )
        self.bert = BertModel(bert_cfg, add_pooling_layer=False)
        self.tgt_tok_emb = nn.Embedding(cond_vocab_size, d_model)
        self.positional_encoding = _SinusoidalPositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.generator = nn.Linear(d_model, cond_vocab_size)

    def forward(self, input_ids: Tensor, label_input: Tensor) -> Tensor:
        """Encode the reaction SMILES, decode reaction-condition-label logits.

        Parameters
        ----------
        input_ids : Tensor
            Shape ``(batch, src_len)`` reaction-SMILES token ids.
        label_input : Tensor
            Shape ``(batch, tgt_len)`` reaction-condition-label ids
            (teacher-forced decoder input).

        Returns
        -------
        Tensor
            Shape ``(batch, tgt_len, cond_vocab_size)`` per-step condition
            -label logits.
        """

        memory = self.bert(input_ids=input_ids).last_hidden_state
        tgt = self.positional_encoding(self.tgt_tok_emb(label_input))
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1])
        out = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        return self.generator(out)


def build_parrot() -> ParrotConditionModel:
    """Build a compact :class:`ParrotConditionModel`."""

    return ParrotConditionModel().eval()


def example_input_parrot() -> tuple[Tensor, Tensor]:
    """Create example input for :func:`build_parrot`.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``input_ids`` (shape ``(2, 20)``) and ``label_input`` (shape
        ``(2, 6)``, teacher-forced condition-label sequence).
    """

    torch.manual_seed(1)
    return torch.randint(0, 64, (2, 20)), torch.randint(0, 16, (2, 6))


# ---------------------------------------------------------------------------
# PGCGM
# ---------------------------------------------------------------------------


class PGCGMGenerator(nn.Module):
    """Space-group-conditioned crystal-structure GAN generator.

    Reproduces ``model.py::Generator``: noise, per-element features, and
    a fixed space-group symmetry-operation matrix are embedded by three
    parallel branches and fused; a transposed-conv stack decodes
    fractional atomic coordinates while a small MLP regresses lattice
    lengths, matching the two-headed output of the reference.
    """

    def __init__(self, ele_vec_dim: int = 23, noise_dim: int = 32) -> None:
        super().__init__()
        self.sp_block = nn.Sequential(
            nn.Conv2d(6, 16, 2, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
        )
        self.ele_block = nn.Sequential(
            nn.Conv1d(ele_vec_dim, 16, 2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, 2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Flatten(),
            # element axis 4 -> Conv1d(k=2) -> 3 -> Conv1d(k=2) -> 2, so flattened is 32*2=64
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.noise_block = nn.Sequential(
            nn.Linear(noise_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        # sp_block output: 32 channels over a (5,5) input -> (3,3) spatial map -> 288
        self.sp_proj = nn.Sequential(nn.Linear(288, 64), nn.ReLU())
        self.coords_block = nn.Sequential(
            nn.ConvTranspose2d(128, 64, (2, 2), (1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, (2, 2), (1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, (1, 1), (1, 1)),
            nn.Tanh(),
        )
        self.length_block = nn.Sequential(
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Tanh(),
        )

    def forward(self, sp_inputs: Tensor, ele_inputs: Tensor, z: Tensor) -> tuple[Tensor, Tensor]:
        """Fuse noise/element/symmetry branches into coordinates + lattice lengths.

        Parameters
        ----------
        sp_inputs : Tensor
            Shape ``(batch, 6, 5, 5)`` space-group symmetry-operation
            matrices (stacked affine-transform channels).
        ele_inputs : Tensor
            Shape ``(batch, 23, 4)`` per-element one-hot/feature vectors.
        z : Tensor
            Shape ``(batch, noise_dim)`` latent noise.

        Returns
        -------
        tuple[Tensor, Tensor]
            Fractional atomic coordinates ``(batch, 3, H, W)`` and lattice
            lengths ``(batch, 3)``.
        """

        sp_embedding = self.sp_proj(self.sp_block(sp_inputs))
        ele_embedding = self.ele_block(ele_inputs)
        z_embedding = self.noise_block(z)

        x1 = torch.cat((z_embedding, ele_embedding), dim=1)
        x2 = torch.cat((z_embedding, sp_embedding), dim=1)

        coords = self.coords_block(x1.view(-1, 128, 1, 1))
        length = self.length_block(x2)
        return coords, length


def build_pgcgm() -> PGCGMGenerator:
    """Build a compact :class:`PGCGMGenerator`."""

    return PGCGMGenerator().eval()


def example_input_pgcgm() -> tuple[Tensor, Tensor, Tensor]:
    """Create example input for :func:`build_pgcgm`.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Space-group symmetry matrices ``(2, 6, 5, 5)``, element features
        ``(2, 23, 4)``, and noise ``(2, 32)``.
    """

    torch.manual_seed(2)
    return torch.randn(2, 6, 5, 5), torch.randn(2, 23, 4), torch.randn(2, 32)


# ---------------------------------------------------------------------------
# Pocket2Mol
# ---------------------------------------------------------------------------


class _GVLinear(nn.Module):
    """Geometric-vector linear layer (equivariant scalar<->vector mixing).

    Reproduces ``models/invariant.py::GVLinear``: the vector channel is
    only ever scaled or linearly recombined across its *channel* axis
    (never mixed across the 3-D coordinate axis with a learned bias),
    which preserves E(3) equivariance; the vector norm feeds into the
    scalar branch, and the scalar branch gates the vector branch.
    """

    def __init__(self, in_sca: int, in_vec: int, out_sca: int, out_vec: int) -> None:
        super().__init__()
        hid = max(in_vec, out_vec)
        self.lin_vector = nn.Linear(in_vec, hid, bias=False)
        self.lin_vector2 = nn.Linear(hid, out_vec, bias=False)
        self.lin_scalar = nn.Linear(in_sca + hid, out_sca, bias=False)
        self.gate = nn.Linear(out_sca, out_vec)

    def forward(self, sca: Tensor, vec: Tensor) -> tuple[Tensor, Tensor]:
        """Mix scalar features ``sca`` (..., in_sca) and vector features
        ``vec`` (..., in_vec, 3) into new scalar/vector features."""

        vec_inter = torch.einsum("...ic,oi->...oc", vec, self.lin_vector.weight)
        vec_norm = torch.norm(vec_inter, p=2, dim=-1)
        sca_cat = torch.cat([vec_norm, sca], dim=-1)
        out_sca = self.lin_scalar(sca_cat)
        out_vec = torch.einsum("...ic,oi->...oc", vec_inter, self.lin_vector2.weight)
        gating = torch.sigmoid(self.gate(out_sca)).unsqueeze(-1)
        out_vec = out_vec * gating
        return out_sca, out_vec


class Pocket2MolStep(nn.Module):
    """One autoregressive atom-growth step of Pocket2Mol's E(3)-equivariant SBDD model.

    Reproduces the ``models/maskfill.py::MaskFillModelVN`` pipeline for a
    single step: vector-neuron message passing over a composed
    protein+ligand point cloud (``models/invariant.py``), a frontier
    classifier, a mixture-density-network relative-position predictor
    (``models/position.py::PositionPredictor``), and an element
    classifier for the newly placed atom.
    """

    def __init__(
        self,
        atom_feat_dim: int = 8,
        hid_sca: int = 16,
        hid_vec: int = 4,
        n_component: int = 3,
        num_elements: int = 6,
    ) -> None:
        super().__init__()
        self.atom_emb_sca = nn.Linear(atom_feat_dim, hid_sca)
        self.atom_emb_vec = nn.Linear(1, hid_vec, bias=False)
        self.message = _GVLinear(hid_sca, hid_vec, hid_sca, hid_vec)
        self.update = _GVLinear(hid_sca, hid_vec, hid_sca, hid_vec)
        self.frontier_pred = nn.Linear(hid_sca, 1)
        self.mu_net = _GVLinear(hid_sca, hid_vec, n_component, n_component)
        self.logsigma_net = _GVLinear(hid_sca, hid_vec, n_component, n_component)
        self.pi_net = _GVLinear(hid_sca, hid_vec, n_component, 1)
        self.element_head = nn.Linear(hid_sca, num_elements)

    def forward(
        self, atom_feat: Tensor, atom_pos: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Run one frontier -> position -> element step over a composed point cloud.

        Parameters
        ----------
        atom_feat : Tensor
            Shape ``(batch, n_atoms, atom_feat_dim)`` per-atom scalar
            features (protein + partial ligand, composed).
        atom_pos : Tensor
            Shape ``(batch, n_atoms, 3)`` per-atom 3-D coordinates.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
            Frontier logits ``(batch, n_atoms)``, mixture means
            ``(batch, n_atoms, n_component, 3)``, mixture sigmas (same
            shape), mixture weights ``(batch, n_atoms, n_component)``,
            and next-atom element logits ``(batch, n_atoms, num_elements)``.
        """

        h_sca = self.atom_emb_sca(atom_feat)
        # atom_pos: (batch, n, 3) -> (batch, n, 1, 3) single input vector channel,
        # then linearly expand the channel axis to hid_vec channels (equivariant:
        # only the channel axis is mixed, the 3-D coordinate axis is untouched)
        h_vec = self.atom_emb_vec(atom_pos.unsqueeze(-2).transpose(-2, -1)).transpose(
            -2, -1
        )  # (..., hid_vec, 3)
        # dense mean-field message: every atom attends to the point-cloud centroid
        centroid_sca = h_sca.mean(dim=1, keepdim=True).expand_as(h_sca)
        centroid_vec = h_vec.mean(dim=1, keepdim=True).expand_as(h_vec)
        msg_sca, msg_vec = self.message(centroid_sca, centroid_vec)
        h_sca = h_sca + msg_sca
        h_vec = h_vec + msg_vec
        h_sca, h_vec = self.update(h_sca, h_vec)

        frontier_logits = self.frontier_pred(h_sca).squeeze(-1)
        relative_mu = self.mu_net(h_sca, h_vec)[1]
        logsigma = self.logsigma_net(h_sca, h_vec)[1]
        sigma = torch.exp(logsigma)
        pi = F.softmax(self.pi_net(h_sca, h_vec)[0], dim=-1)
        element_logits = self.element_head(h_sca)
        return frontier_logits, relative_mu, sigma, pi, element_logits


def build_pocket2mol() -> Pocket2MolStep:
    """Build a compact :class:`Pocket2MolStep`."""

    return Pocket2MolStep().eval()


def example_input_pocket2mol() -> tuple[Tensor, Tensor]:
    """Create example input for :func:`build_pocket2mol`.

    Returns
    -------
    tuple[Tensor, Tensor]
        Composed protein+ligand atom features ``(2, 10, 8)`` and
        positions ``(2, 10, 3)``.
    """

    torch.manual_seed(3)
    return torch.randn(2, 10, 8), torch.randn(2, 10, 3)


# ---------------------------------------------------------------------------
# PocketFlow
# ---------------------------------------------------------------------------


class _InvariantPointAttention(nn.Module):
    """Compact AF2-style invariant point attention block.

    Reproduces the paper-described "IPA from AF2" encoder component of
    PocketFlow: attention logits are gated by pairwise squared-distance
    features between projected 3-D query/key points, which keeps the
    attention pattern invariant to any shared rigid-body transform of
    the input coordinates -- the paper's stated basis for its FrameDiff
    -style structural encoder.
    """

    def __init__(self, dim: int, n_heads: int = 4, n_points: int = 4) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_points = n_points
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.q_points = nn.Linear(dim, n_heads * n_points * 3)
        self.k_points = nn.Linear(dim, n_heads * n_points * 3)
        self.dist_gate = nn.Parameter(torch.ones(n_heads))
        self.out = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pos: Tensor) -> Tensor:
        """Attend over ``x`` (features) gated by ``pos`` (3-D coordinates).

        Parameters
        ----------
        x : Tensor
            Shape ``(batch, n, dim)`` per-residue/atom features.
        pos : Tensor
            Shape ``(batch, n, 3)`` per-residue/atom coordinates.
        """

        batch, n, dim = x.shape
        h = self.n_heads
        q = self.q(x).view(batch, n, h, dim // h)
        k = self.k(x).view(batch, n, h, dim // h)
        v = self.v(x).view(batch, n, h, dim // h)
        content_logits = torch.einsum("bihd,bjhd->bhij", q, k) / math.sqrt(dim // h)

        qp = self.q_points(x).view(batch, n, h, self.n_points, 3)
        kp = self.k_points(x).view(batch, n, h, self.n_points, 3)
        qp = qp + pos.view(batch, n, 1, 1, 3)
        kp = kp + pos.view(batch, n, 1, 1, 3)
        sq_dist = (qp.unsqueeze(2) - kp.unsqueeze(1)).pow(2).sum(dim=(-1, -2))  # (b, i, j, h)
        point_logits = -0.5 * sq_dist.permute(0, 3, 1, 2) * self.dist_gate.view(1, h, 1, 1)

        attn = torch.softmax(content_logits + point_logits, dim=-1)
        out = torch.einsum("bhij,bjhd->bihd", attn, v).reshape(batch, n, dim)
        return self.out(out)


class PocketFlowStep(nn.Module):
    """One prior-informed flow-matching denoising step for pocket generation.

    Reimplemented from the arXiv description (no released source):
    an IPA-style structural block plus a Transformer sequence layer
    jointly encode a composed protein-residue + ligand-atom point cloud
    at flow time ``t``; five small MLP heads predict Calpha-coordinate
    velocities, SO(3) backbone-orientation velocities (a 3-vector
    generator of the tangent skew-symmetric matrix), torsion-angle
    velocities, residue-type logits, and protein-ligand interaction-type
    logits, matching the paper's stated five-head decomposition.
    """

    def __init__(
        self,
        dim: int = 32,
        n_heads: int = 4,
        num_residue_types: int = 20,
        num_interaction_types: int = 5,
        num_torsions: int = 4,
    ) -> None:
        super().__init__()
        self.node_embed = nn.Linear(dim + 1, dim)  # +1 for flow-matching time t
        self.ipa = _InvariantPointAttention(dim, n_heads=n_heads)
        self.ipa_norm = nn.LayerNorm(dim)
        seq_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 2, batch_first=True
        )
        self.seq_encoder = nn.TransformerEncoder(seq_layer, num_layers=1)

        self.coord_head = nn.Linear(dim, 3)
        self.orientation_head = nn.Linear(dim, 3)  # so(3) tangent generator
        self.torsion_head = nn.Linear(dim, num_torsions)
        self.residue_type_head = nn.Linear(dim, num_residue_types)
        self.interaction_head = nn.Linear(dim, num_interaction_types)

    def forward(
        self, feat: Tensor, pos: Tensor, t: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Predict the five flow-matching velocity/logit heads at time ``t``.

        Parameters
        ----------
        feat : Tensor
            Shape ``(batch, n, dim)`` composed residue/atom features.
        pos : Tensor
            Shape ``(batch, n, 3)`` composed Calpha/atom coordinates.
        t : Tensor
            Shape ``(batch, n, 1)`` per-node flow-matching time in [0, 1].

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
            Coordinate velocities ``(batch, n, 3)``, orientation
            velocities ``(batch, n, 3)``, torsion velocities
            ``(batch, n, num_torsions)``, residue-type logits, and
            interaction-type logits.
        """

        x = self.node_embed(torch.cat([feat, t], dim=-1))
        x = self.ipa_norm(x + self.ipa(x, pos))
        x = self.seq_encoder(x)

        coord_v = self.coord_head(x)
        orient_v = self.orientation_head(x)
        torsion_v = self.torsion_head(x)
        residue_logits = self.residue_type_head(x)
        interaction_logits = self.interaction_head(x)
        return coord_v, orient_v, torsion_v, residue_logits, interaction_logits


def build_pocketflow() -> PocketFlowStep:
    """Build a compact :class:`PocketFlowStep`."""

    return PocketFlowStep().eval()


def example_input_pocketflow() -> tuple[Tensor, Tensor, Tensor]:
    """Create example input for :func:`build_pocketflow`.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Composed features ``(2, 12, 32)``, coordinates ``(2, 12, 3)``,
        and per-node flow time ``(2, 12, 1)``.
    """

    torch.manual_seed(4)
    feat = torch.randn(2, 12, 32)
    pos = torch.randn(2, 12, 3)
    t = torch.rand(2, 12, 1)
    return feat, pos, t


# ---------------------------------------------------------------------------
# PSVAE (Principal Subgraph VAE)
# ---------------------------------------------------------------------------


class _GINELayer(nn.Module):
    """One GINE-style (edge-feature-augmented GIN) message-passing layer.

    Reproduces the aggregation used by ``src/modules/encoder.py::Encoder``
    (built on ``torch_geometric.nn.GINEConv``): messages are node
    features plus edge features, summed over neighbors via a dense
    adjacency (fixed small graphs here, in place of the reference's
    sparse ``edge_index``), then passed through an MLP update -- the
    GIN sum-aggregation that gives the encoder its expressive power on
    molecular graphs.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor, edge_feat: Tensor, adj: Tensor) -> Tensor:
        """Aggregate ``x + edge_feat`` messages over dense adjacency ``adj``.

        Parameters
        ----------
        x : Tensor
            Shape ``(batch, n, dim)`` node features.
        edge_feat : Tensor
            Shape ``(batch, n, n, dim)`` edge features.
        adj : Tensor
            Shape ``(batch, n, n)`` dense (0/1) adjacency.
        """

        msg = x.unsqueeze(1) + edge_feat  # (batch, n, n, dim), broadcast src node onto every dst
        msg = F.relu(msg) * adj.unsqueeze(-1)
        agg = msg.sum(dim=2)
        return self.mlp((1 + self.eps) * x + agg)


class PSVAEModel(nn.Module):
    """Principal-subgraph VAE: GINE graph encoder + latent-conditioned piece decoder.

    Reproduces ``src/modules/encoder.py::Encoder`` (GINE message passing
    + sum-pool graph embedding) feeding ``src/modules/vae_piece_decoder.py
    ::VAEPieceDecoder`` (diagonal-Gaussian reparameterization, a
    latent-seeded GRU that autoregressively predicts principal-subgraph
    -piece ids, and an edge-type classifier over node-embedding pairs).
    """

    def __init__(
        self,
        atom_types: int = 8,
        edge_types: int = 4,
        hidden_dim: int = 24,
        latent_dim: int = 16,
        piece_vocab: int = 32,
        n_conv_layers: int = 3,
    ) -> None:
        super().__init__()
        self.atom_embed = nn.Embedding(atom_types, hidden_dim)
        self.edge_embed = nn.Embedding(edge_types, hidden_dim)
        self.gine_layers = nn.ModuleList([_GINELayer(hidden_dim) for _ in range(n_conv_layers)])
        self.to_latent_params = nn.Linear(hidden_dim, 2 * latent_dim)

        self.piece_embedding = nn.Embedding(piece_vocab, hidden_dim)
        self.latent_to_rnn_hidden = nn.Linear(latent_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.to_vocab = nn.Linear(hidden_dim, piece_vocab)
        edge_mlp_in = 2 * hidden_dim + latent_dim
        self.edge_predictor = nn.Sequential(
            nn.Linear(edge_mlp_in, edge_mlp_in // 2),
            nn.ReLU(),
            nn.Linear(edge_mlp_in // 2, edge_types),
        )

    def encode(self, atom_ids: Tensor, edge_ids: Tensor, adj: Tensor) -> tuple[Tensor, Tensor]:
        """Run GINE message passing + sum-pool to a fixed-size graph embedding."""

        x = self.atom_embed(atom_ids)
        edge_feat = self.edge_embed(edge_ids)
        for layer in self.gine_layers:
            x = layer(x, edge_feat, adj)
        return x.sum(dim=1), x

    def forward(
        self, atom_ids: Tensor, edge_ids: Tensor, adj: Tensor, piece_prev: Tensor, eps: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Encode a molecular graph, sample a latent, decode one piece step + edge logits.

        Parameters
        ----------
        atom_ids : Tensor
            Shape ``(batch, n_atoms)`` integer atom-type ids.
        edge_ids : Tensor
            Shape ``(batch, n_atoms, n_atoms)`` integer edge-type ids.
        adj : Tensor
            Shape ``(batch, n_atoms, n_atoms)`` dense adjacency mask.
        piece_prev : Tensor
            Shape ``(batch,)`` previous principal-subgraph-piece id fed
            into the GRU decoder for this step.
        eps : Tensor
            Shape ``(batch, latent_dim)`` standard-normal reparameterization
            noise.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Latent code ``(batch, latent_dim)``, next-piece logits
            ``(batch, piece_vocab)``, and pairwise edge-type logits
            ``(batch, n_atoms, n_atoms, edge_types)``.
        """

        pooled, node_x = self.encode(atom_ids, edge_ids, adj)
        mean, log_var = self.to_latent_params(pooled).chunk(2, dim=-1)
        log_var = -torch.abs(log_var)
        z = mean + torch.exp(0.5 * log_var) * eps

        hidden = self.latent_to_rnn_hidden(z)
        piece_in = self.piece_embedding(piece_prev)
        hidden = self.rnn(piece_in, hidden)
        piece_logits = self.to_vocab(hidden)

        n = node_x.shape[1]
        src = node_x.unsqueeze(2).expand(-1, -1, n, -1)
        dst = node_x.unsqueeze(1).expand(-1, n, -1, -1)
        z_expand = z.view(z.shape[0], 1, 1, -1).expand(-1, n, n, -1)
        edge_logits = self.edge_predictor(torch.cat([src, dst, z_expand], dim=-1))
        return z, piece_logits, edge_logits


def build_psvae() -> PSVAEModel:
    """Build a compact :class:`PSVAEModel`."""

    return PSVAEModel().eval()


def example_input_psvae() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Create example input for :func:`build_psvae`.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
        Atom ids ``(2, 9)``, edge-type ids ``(2, 9, 9)``, dense adjacency
        ``(2, 9, 9)``, previous-piece ids ``(2,)``, and latent noise
        ``(2, 16)``.
    """

    torch.manual_seed(5)
    batch, n_atoms = 2, 9
    atom_ids = torch.randint(0, 8, (batch, n_atoms))
    edge_ids = torch.randint(0, 4, (batch, n_atoms, n_atoms))
    adj = (torch.rand(batch, n_atoms, n_atoms) > 0.5).float()
    piece_prev = torch.randint(0, 32, (batch,))
    eps = torch.randn(batch, 16)
    return atom_ids, edge_ids, adj, piece_prev, eps


MENAGERIE_ENTRIES = [
    ("ORGAN", "build_organ", "example_input_organ", "2017", "BIO"),
    ("Parrot", "build_parrot", "example_input_parrot", "2023", "BIO"),
    (
        "PGCGM (Periodic Graph Crystal Generative Model)",
        "build_pgcgm",
        "example_input_pgcgm",
        "2022",
        "BIO",
    ),
    ("Pocket2Mol", "build_pocket2mol", "example_input_pocket2mol", "2022", "BIO"),
    ("PocketFlow", "build_pocketflow", "example_input_pocketflow", "2024", "BIO"),
    ("PSVAE (Principal Subgraph VAE)", "build_psvae", "example_input_psvae", "2022", "BIO"),
]
