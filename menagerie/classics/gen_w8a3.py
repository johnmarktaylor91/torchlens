"""Faithful, compact TorchLens menagerie classics for build-queue batch w8a3.

Sources checked (repo READMEs / source files via `gh api`, and arXiv abstracts via
WebFetch) for each of the six candidates in rows 19-24 of
``.research/menagerie-redesign/build_queue.tsv``:

  - Stormer (weather forecasting transformer): Nguyen et al., "Scaling transformer
    neural networks for skillful and reliable medium-range weather forecasting",
    arXiv:2312.03876 (NeurIPS 2024). Repo https://github.com/tung-nd/stormer,
    ``stormer/models/hub/stormer.py`` + ``weather_embedding.py`` fetched directly.
    Stormer's central novelty is a near-vanilla ViT/DiT backbone with two
    weather-specific pieces: (1) a per-variable ``WeatherEmbedding`` that patch-embeds
    each atmospheric variable through its OWN ``PatchEmbed`` layer, adds a learned
    per-variable channel embedding, then cross-attends a single learned query token
    over the variable axis to aggregate all variables per spatial location into one
    token sequence (a ClimaX-style variable-aggregation head); and (2) DiT-style
    adaLN-Zero conditioning of every transformer block (and the final head) on an
    embedding of the forecast lead time / interval, so one model can be queried at
    arbitrary lead times. This reproduces Stormer's namesake
    per-variable-tokenize -> cross-attend -> adaLN-lead-time-conditioned-transformer
    design, its central contribution over a plain ViT that concatenates variables as
    channels with no lead-time conditioning.
  - Synthon Completion Network (GraphRetro): Somnath, Bunne, Coley, Krause, Barzilay,
    "Learning Graph Models for Retrosynthesis Prediction", arXiv:2006.07038
    (NeurIPS 2021). Repo https://github.com/vsomnath/graphretro,
    ``seq_graph_retro/models/lg_edits/lg_ind_embed.py`` (``LGIndEmbed``) fetched
    directly. The synthon-completion stage's central novelty is: given synthon
    fragments produced by an upstream edit-prediction stage, run a shared
    message-passing graph encoder over BOTH the full product graph and the
    disconnected synthon-fragment graph, then for each synthon fragment
    autoregressively classify which "leaving group" (from a precomputed vocabulary of
    common leaving-group subgraphs) should be attached to complete it into a full
    reactant -- conditioned on the fragment's own graph embedding, the pooled product
    context vector, and (via a leaving-group embedding table) the previously attached
    leaving group at the prior autoregressive step. This reproduces GraphRetro's
    namesake shared-encoder + per-synthon autoregressive leaving-group classification,
    its central contribution over generating reactants atom-by-atom or template-based.
  - TF-Net turbulence (Tensor Basis Neural Network, TBNN): Sandia's
    ``sandialabs/tbnn`` implements Ling, Kurzawski, Templeton, "Reynolds averaged
    turbulence modelling using deep neural networks with embedded invariance", JFM
    2016, and is the physics-informed model referenced by arXiv:2311.14576 (repo
    ``tbnn/core.py`` fetched directly; the ``TensorLayer``/``TBNN`` classes there are
    the ground truth for TBNN's central mechanism, since arXiv:2311.14576 is a later
    application paper using the same architecture). TBNN's central novelty is
    embedding Galilean/rotational invariance directly into the network: a plain MLP
    consumes 5 scalar invariants of the local mean strain-rate and rotation-rate
    tensors and outputs coefficients for a fixed integrity (Ling) basis of 10
    independent 3x3 tensors built from those same strain/rotation tensors; the
    predicted Reynolds-stress anisotropy tensor is then a coefficient-weighted linear
    combination (contraction) of that tensor basis, NOT a direct regression target,
    which guarantees the output transforms correctly under rotation/reflection of the
    coordinate frame. This reproduces TBNN's namesake scalar-MLP-predicts-basis-
    coefficients + tensor-basis-contraction design, its central contribution over a
    black-box regressor that ignores frame invariance.
  - TinNet (theory-infused neural network): Wang, Xin, et al. papers behind
    https://github.com/hlxin/tinnet (repo's ``tinnet/band_center.py``/
    ``adsorption_energy.py`` fetched directly, confirming the physical target
    quantities). TinNet's central novelty is NOT predicting adsorption energy as a
    single black-box regression target; instead a crystal-graph convolutional
    encoder (CGCNN-style atom/bond message passing over the local adsorption-site
    neighborhood) predicts the physically meaningful parameters of an analytical
    Newns-Anderson d-band hybridization model -- the d-band center, d-band width, and
    d-band filling of the adsorption site -- and those three learned physical
    parameters are then passed through a fixed closed-form Newns-Anderson-style
    energy expression (coupling-matrix-element-weighted hybridization + Coulomb
    repulsion terms) to produce the final adsorption energy, making every
    intermediate quantity physically interpretable. This reproduces TinNet's
    namesake GNN-predicts-physical-descriptors + analytic-physics-formula design, its
    central contribution over an end-to-end black-box GNN regressor.
  - TrackML GNN (HEPTrkX): Ju et al. et al., "Graph Neural Networks for Particle
    Track Reconstruction" (arXiv:1810.06111), repo
    https://github.com/HEPTrkX/heptrkx-gnn-tracking,
    ``models/gnn.py`` (``EdgeNetwork``/``NodeNetwork``/``GNNSegmentClassifier``)
    fetched directly and used verbatim as the source of truth. The central novelty is
    an interaction-network-style iterative edge/node message-passing classifier over
    a bipartite hit graph encoded via two dense incidence matrices ``Ri``/``Ro``
    (receiving/outgoing endpoint selection per candidate segment): an
    ``EdgeNetwork`` scores every candidate hit-pair segment from its two endpoint hit
    features via a small MLP + sigmoid, and a ``NodeNetwork`` re-embeds every hit by
    aggregating its edge-weighted incoming/outgoing neighbor features, iterated for
    several message-passing rounds with an input-feature skip connection at every
    round, ending in one final edge-score pass that classifies every candidate
    segment as real-track vs. fake. This reproduces HEPTrkX's namesake
    incidence-matrix-driven alternating edge/node network for segment classification,
    its central contribution over per-hit or per-track classifiers that ignore graph
    structure.
  - TransPolymer: Xu, Chen, Vlachos, "TransPolymer: a Transformer-based language
    model for polymer property predictions", arXiv:2209.01307 (npj Comput. Mater.
    2023). Repo https://github.com/ChangwenXu98/TransPolymer,
    ``Downstream.py`` (``DownstreamRegression``) fetched directly, confirming the
    architecture is literally a HuggingFace ``RobertaModel`` backbone plus a small
    2-layer MLP ``Regressor`` head applied to the backbone's pooled sequence output.
    TransPolymer's central novelty is NOT a custom attention mechanism -- it is a
    chemically-aware P-SMILES tokenizer (multi-character polymer-SMILES tokens plus
    a special ``[*]`` polymer end-group / repeat-point token) feeding a standard
    RoBERTa masked-language-model backbone that is pretrained on ~5M polymer P-SMILES
    strings, then fine-tuned with a lightweight regression head for downstream
    polymer property prediction. This reproduces TransPolymer's namesake
    P-SMILES-tokenized-RoBERTa + MLP-regression-head design (built here via
    ``transformers.RobertaConfig``/``RobertaModel`` at tiny dims, since the repo
    itself uses the HuggingFace RoBERTa implementation directly rather than a custom
    attention stack), its central contribution over generic SMILES tokenizers that
    do not represent the polymer repeat-unit / end-group structure.

All six modules below use small random-init dimensions (this is an architecture
catalog, not a trained-weights zoo).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Stormer: per-variable tokenize -> cross-attend variable-aggregation ->
# adaLN-Zero lead-time-conditioned transformer (DiT-style) backbone.
# ---------------------------------------------------------------------------


class _StormerWeatherEmbedding(nn.Module):
    """Per-variable patch embedding + learned-query cross-attention aggregation."""

    def __init__(
        self,
        n_variables: int,
        img_size: tuple[int, int],
        patch_size: int,
        embed_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        # One conv-based patch embedder per variable (variable tokenization).
        self.token_embeds = nn.ModuleList(
            [
                nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
                for _ in range(n_variables)
            ]
        )
        self.channel_embed = nn.Parameter(torch.zeros(1, n_variables, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))

        # Variable aggregation: single learned query + one cross-attention layer.
        self.channel_query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.channel_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed (B, V, H, W) weather fields into (B, L, D) aggregated tokens."""
        b, v, _, _ = x.shape
        embeds = [self.token_embeds[i](x[:, i : i + 1]) for i in range(v)]
        # Each embed is (B, D, h, w) -> flatten to (B, L, D).
        embeds = [e.flatten(2).transpose(1, 2) for e in embeds]
        tokens = torch.stack(embeds, dim=1)  # (B, V, L, D)
        tokens = tokens + self.channel_embed.unsqueeze(2)

        _, _, length, dim = tokens.shape
        tokens = tokens.permute(0, 2, 1, 3).reshape(b * length, v, dim)
        query = self.channel_query.expand(b * length, -1, -1)
        agg, _ = self.channel_agg(query, tokens, tokens)  # (B*L, 1, D)
        agg = agg.reshape(b, length, dim)
        return agg + self.pos_embed


class _StormerAdaLNBlock(nn.Module):
    """Standard pre-LN transformer block with adaLN-Zero conditioning."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply attention + MLP sublayers, each modulated by the lead-time cond."""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            cond
        ).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(h, h, h)
        x = x + gate_msa.unsqueeze(1) * attn_out
        h = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        return x


class Stormer(nn.Module):
    """Simple ViT/DiT weather forecaster with per-variable tokens + lead-time adaLN.

    Reproduces Stormer's per-variable ``WeatherEmbedding`` (separate patch embedder
    per atmospheric variable, learned-query cross-attention aggregation across
    variables) feeding a stack of adaLN-Zero transformer blocks conditioned on a
    scalar forecast-lead-time embedding, per Nguyen et al. arXiv:2312.03876.
    """

    def __init__(
        self,
        n_variables: int = 4,
        img_size: tuple[int, int] = (16, 16),
        patch_size: int = 4,
        hidden_size: int = 32,
        depth: int = 2,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.n_variables = n_variables
        self.img_size = img_size
        self.patch_size = patch_size

        self.embedding = _StormerWeatherEmbedding(
            n_variables, img_size, patch_size, hidden_size, num_heads
        )
        self.embed_norm = nn.LayerNorm(hidden_size)
        self.t_embedder = nn.Linear(1, hidden_size)
        self.blocks = nn.ModuleList(
            [_StormerAdaLNBlock(hidden_size, num_heads) for _ in range(depth)]
        )
        self.final_norm_mod = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.head = nn.Linear(hidden_size, patch_size * patch_size * n_variables)

    def forward(self, x: torch.Tensor, lead_time: torch.Tensor) -> torch.Tensor:
        """Forecast all variables from an input field at the given lead time(s)."""
        tokens = self.embed_norm(self.embedding(x))
        cond = self.t_embedder(lead_time.unsqueeze(-1))
        for block in self.blocks:
            tokens = block(tokens, cond)
        shift, scale = self.final_norm_mod(cond).chunk(2, dim=-1)
        tokens = tokens * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        out = self.head(tokens)  # (B, L, p*p*V)

        b = x.shape[0]
        h = self.img_size[0] // self.patch_size
        w = self.img_size[1] // self.patch_size
        p = self.patch_size
        out = out.reshape(b, h, w, p, p, self.n_variables)
        out = torch.einsum("bhwpqv->bvhpwq", out)
        return out.reshape(b, self.n_variables, h * p, w * p)


def build_stormer_weather() -> nn.Module:
    """Build a tiny Stormer weather-forecasting transformer."""
    return Stormer(
        n_variables=4, img_size=(16, 16), patch_size=4, hidden_size=32, depth=2, num_heads=4
    ).eval()


def example_input_stormer_weather() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a random weather field batch and lead-time tensor for Stormer."""
    x = torch.randn(2, 4, 16, 16)
    lead_time = torch.tensor([6.0, 24.0])
    return x, lead_time


# ---------------------------------------------------------------------------
# Synthon Completion Network (GraphRetro): shared graph encoder over product +
# synthon fragments -> autoregressive per-fragment leaving-group classification.
# ---------------------------------------------------------------------------


class _SynthonMPNLayer(nn.Module):
    """One WLN-style bonded message-passing round over an atom graph."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.msg = nn.Linear(dim, dim)
        self.update = nn.GRUCell(dim, dim)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Aggregate bonded-neighbor messages and GRU-update every atom embedding."""
        b, n, d = h.shape
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        msg = torch.bmm(adj, self.msg(h)) / deg
        h_flat = h.reshape(b * n, d)
        msg_flat = msg.reshape(b * n, d)
        return self.update(msg_flat, h_flat).reshape(b, n, d)


class _SynthonGraphEncoder(nn.Module):
    """Shared message-passing encoder producing per-atom + pooled graph vectors."""

    def __init__(self, atom_dim: int, hidden: int, depth: int) -> None:
        super().__init__()
        self.atom_embed = nn.Linear(atom_dim, hidden)
        self.layers = nn.ModuleList([_SynthonMPNLayer(hidden) for _ in range(depth)])

    def forward(
        self, atom_feats: torch.Tensor, adj: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (per-atom embeddings, mean-pooled graph embedding)."""
        h = self.atom_embed(atom_feats)
        for layer in self.layers:
            h = layer(h, adj)
        pooled = h.mean(dim=1)
        return h, pooled


class SynthonCompletionNetwork(nn.Module):
    """GraphRetro's synthon-completion stage: shared encoder + AR leaving-group head.

    Encodes the full product graph and a disconnected multi-fragment synthon graph
    with one shared message-passing encoder, then autoregressively classifies a
    leaving group (from a fixed vocabulary) for each synthon fragment, conditioned on
    the fragment's own pooled embedding, the pooled product-context embedding, and an
    embedding of the previously predicted leaving group.
    """

    def __init__(
        self,
        atom_dim: int = 12,
        hidden: int = 24,
        depth: int = 2,
        n_leaving_groups: int = 10,
        n_fragments: int = 2,
    ) -> None:
        super().__init__()
        self.n_fragments = n_fragments
        self.n_leaving_groups = n_leaving_groups
        self.encoder = _SynthonGraphEncoder(atom_dim, hidden, depth)
        self.lg_embedding = nn.Linear(n_leaving_groups, hidden)
        self.lg_score = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_leaving_groups),
        )
        self.register_buffer("lg_eye", torch.eye(n_leaving_groups))
        self.bos = nn.Parameter(torch.zeros(1, n_leaving_groups))

    def forward(
        self,
        prod_atom_feats: torch.Tensor,
        prod_adj: torch.Tensor,
        frag_atom_feats: torch.Tensor,
        frag_adj: torch.Tensor,
        frag_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score leaving-group vocabulary logits for every synthon fragment.

        Parameters
        ----------
        prod_atom_feats : torch.Tensor
            (B, N_p, atom_dim) product-graph atom features.
        prod_adj : torch.Tensor
            (B, N_p, N_p) product bonded-adjacency.
        frag_atom_feats : torch.Tensor
            (B, F, N_f, atom_dim) per-fragment atom features (F synthon fragments).
        frag_adj : torch.Tensor
            (B, F, N_f, N_f) per-fragment bonded-adjacency.
        frag_mask : torch.Tensor
            (B, F, N_f) atom-validity mask per fragment, used for masked mean pool.
        """
        _, prod_pooled = self.encoder(prod_atom_feats, prod_adj)

        b, f, n_f, _ = frag_atom_feats.shape
        flat_feats = frag_atom_feats.reshape(b * f, n_f, -1)
        flat_adj = frag_adj.reshape(b * f, n_f, n_f)
        frag_h, _ = self.encoder(flat_feats, flat_adj)
        mask = frag_mask.reshape(b * f, n_f, 1)
        frag_pooled = (frag_h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        frag_pooled = frag_pooled.reshape(b, f, -1)

        prev_lg = self.bos.expand(b, -1)
        logits_steps = []
        for idx in range(f):
            prev_emb = self.lg_embedding(prev_lg)
            step_in = torch.cat([frag_pooled[:, idx], prod_pooled, prev_emb], dim=-1)
            logits = self.lg_score(step_in)
            logits_steps.append(logits)
            chosen = torch.argmax(logits, dim=-1)
            prev_lg = self.lg_eye[chosen]
        return torch.stack(logits_steps, dim=1)


def build_synthon_completion_network() -> nn.Module:
    """Build a tiny GraphRetro synthon-completion network."""
    return SynthonCompletionNetwork(
        atom_dim=12, hidden=24, depth=2, n_leaving_groups=10, n_fragments=2
    ).eval()


def example_input_synthon_completion_network() -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Return random product-graph and synthon-fragment tensors."""
    b, n_p, f, n_f, atom_dim = 2, 8, 2, 5, 12
    prod_atom_feats = torch.randn(b, n_p, atom_dim)
    prod_adj = (torch.rand(b, n_p, n_p) > 0.6).float()
    prod_adj = prod_adj * (1 - torch.eye(n_p))
    frag_atom_feats = torch.randn(b, f, n_f, atom_dim)
    frag_adj = (torch.rand(b, f, n_f, n_f) > 0.6).float()
    eye_f = torch.eye(n_f).unsqueeze(0).unsqueeze(0)
    frag_adj = frag_adj * (1 - eye_f)
    frag_mask = torch.ones(b, f, n_f)
    return prod_atom_feats, prod_adj, frag_atom_feats, frag_adj, frag_mask


# ---------------------------------------------------------------------------
# TF-Net turbulence (Tensor Basis Neural Network): scalar-invariant MLP predicts
# tensor-basis coefficients -> contracted against a fixed Ling integrity basis.
# ---------------------------------------------------------------------------


class TensorBasisNeuralNetwork(nn.Module):
    """TBNN: rotation-invariant Reynolds-stress anisotropy tensor predictor.

    A plain MLP maps scalar invariants of the local strain/rotation tensors to
    coefficients over a fixed tensor basis; the anisotropy tensor prediction is the
    coefficient-weighted contraction of that basis, per Ling et al. (JFM 2016), the
    architecture underlying the Sandia ``tbnn`` package and TF-Net-style turbulence
    closures such as arXiv:2311.14576.
    """

    def __init__(
        self, n_scalar_invariants: int = 5, n_tensor_basis: int = 10, hidden: int = 16
    ) -> None:
        super().__init__()
        self.n_tensor_basis = n_tensor_basis
        self.mlp = nn.Sequential(
            nn.Linear(n_scalar_invariants, hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden, n_tensor_basis),
        )

    def forward(self, scalar_invariants: torch.Tensor, tensor_basis: torch.Tensor) -> torch.Tensor:
        """Predict the flattened 3x3 anisotropy tensor at every spatial point.

        Parameters
        ----------
        scalar_invariants : torch.Tensor
            (N, n_scalar_invariants) invariants of the strain/rotation tensors.
        tensor_basis : torch.Tensor
            (N, n_tensor_basis, 9) flattened 3x3 integrity-basis tensors built from
            the same strain/rotation tensors (precomputed, not learned).
        """
        coeffs = self.mlp(scalar_invariants)  # (N, n_tensor_basis)
        # Batched contraction over the tensor-basis dimension (TensorLayer).
        aniso = torch.einsum("nb,nbi->ni", coeffs, tensor_basis)
        return aniso


def build_tfnet_turbulence() -> nn.Module:
    """Build a tiny Tensor Basis Neural Network for turbulence closure."""
    return TensorBasisNeuralNetwork(n_scalar_invariants=5, n_tensor_basis=10, hidden=16).eval()


def example_input_tfnet_turbulence() -> tuple[torch.Tensor, torch.Tensor]:
    """Return random scalar invariants and a random flattened tensor basis."""
    n = 8
    scalar_invariants = torch.randn(n, 5)
    tensor_basis = torch.randn(n, 10, 9)
    return scalar_invariants, tensor_basis


# ---------------------------------------------------------------------------
# TinNet: crystal-graph encoder predicts physical d-band parameters, contracted
# through a closed-form Newns-Anderson hybridization energy expression.
# ---------------------------------------------------------------------------


class _TinNetConvLayer(nn.Module):
    """One CGCNN-style gated atom/bond message-passing convolution."""

    def __init__(self, atom_dim: int, bond_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(2 * atom_dim + bond_dim, 2 * atom_dim)
        self.bn = nn.BatchNorm1d(2 * atom_dim)

    def forward(
        self, atom_h: torch.Tensor, bond_h: torch.Tensor, nbr_idx: torch.Tensor
    ) -> torch.Tensor:
        """Gate-and-sum neighbor messages into every atom embedding."""
        b, n, m, _ = bond_h.shape
        nbr_atom_h = torch.gather(
            atom_h.unsqueeze(2).expand(-1, -1, m, -1),
            1,
            nbr_idx.unsqueeze(-1).expand(-1, -1, -1, atom_h.shape[-1]),
        )
        self_h = atom_h.unsqueeze(2).expand(-1, -1, m, -1)
        total = torch.cat([self_h, nbr_atom_h, bond_h], dim=-1)
        gated = self.gate(total)
        gated = self.bn(gated.reshape(b * n * m, -1)).reshape(b, n, m, -1)
        filt, core = gated.chunk(2, dim=-1)
        msg = torch.sigmoid(filt) * F.softplus(core)
        return F.softplus(atom_h + msg.sum(dim=2))


class TinNet(nn.Module):
    """Theory-infused NN: CGCNN encoder -> d-band params -> analytic energy.

    Predicts the physically meaningful d-band center, width, and filling of an
    adsorption site from a crystal-graph convolutional encoding of its local
    neighborhood, then evaluates a fixed closed-form Newns-Anderson-style
    hybridization energy expression from those three learned physical quantities
    plus a learned coupling-matrix-element magnitude, per the TinNet family of
    models (https://github.com/hlxin/tinnet).
    """

    def __init__(
        self, atom_dim: int = 16, bond_dim: int = 8, hidden: int = 24, depth: int = 2
    ) -> None:
        super().__init__()
        self.atom_embed = nn.Linear(atom_dim, hidden)
        self.convs = nn.ModuleList([_TinNetConvLayer(hidden, bond_dim) for _ in range(depth)])
        # Physical-descriptor heads: d-band center, width, filling, coupling |V|.
        self.phys_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Softplus(), nn.Linear(hidden, 4)
        )

    def forward(
        self, atom_feats: torch.Tensor, bond_feats: torch.Tensor, nbr_idx: torch.Tensor
    ) -> torch.Tensor:
        """Predict adsorption energy for the site graph (atom 0 is the site).

        Parameters
        ----------
        atom_feats : torch.Tensor
            (B, N, atom_dim) local-neighborhood atom features.
        bond_feats : torch.Tensor
            (B, N, M, bond_dim) per-neighbor bond features.
        nbr_idx : torch.Tensor
            (B, N, M) integer neighbor-atom indices for each atom's M neighbors.
        """
        h = self.atom_embed(atom_feats)
        for conv in self.convs:
            h = conv(h, bond_feats, nbr_idx)
        site_h = h[:, 0]
        params = self.phys_head(site_h)
        d_center, d_width, d_filling, coupling = params.unbind(dim=-1)
        d_width = d_width + 1e-2
        coupling = coupling + 1e-2

        # Closed-form Newns-Anderson-style hybridization + repulsion energy.
        hybridization = (
            -2.0 * coupling.pow(2) / d_width * torch.exp(-d_center.pow(2) / (2 * d_width.pow(2)))
        )
        repulsion = d_filling * coupling.pow(2) / d_width
        return hybridization + repulsion


def build_tinnet() -> nn.Module:
    """Build a tiny TinNet theory-infused adsorption-energy predictor."""
    return TinNet(atom_dim=16, bond_dim=8, hidden=24, depth=2).eval()


def example_input_tinnet() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return random adsorption-site neighborhood graph tensors for TinNet."""
    b, n, m, atom_dim, bond_dim = 2, 6, 4, 16, 8
    atom_feats = torch.randn(b, n, atom_dim)
    bond_feats = torch.randn(b, n, m, bond_dim)
    nbr_idx = torch.randint(0, n, (b, n, m))
    return atom_feats, bond_feats, nbr_idx


# ---------------------------------------------------------------------------
# TrackML GNN (HEPTrkX): incidence-matrix edge/node message-passing segment
# classifier over a bipartite detector-hit graph.
# ---------------------------------------------------------------------------


class _TrackEdgeNetwork(nn.Module):
    """Score every candidate hit-pair segment from its two endpoint features."""

    def __init__(self, input_dim: int, hidden_dim: int = 8) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, r_in: torch.Tensor, r_out: torch.Tensor) -> torch.Tensor:
        """Compute an edge-existence probability for every candidate segment."""
        b_out = torch.bmm(r_out.transpose(1, 2), x)
        b_in = torch.bmm(r_in.transpose(1, 2), x)
        edge_feats = torch.cat([b_out, b_in], dim=2)
        return self.network(edge_feats).squeeze(-1)


class _TrackNodeNetwork(nn.Module):
    """Re-embed every hit from its edge-weighted incoming/outgoing neighbors."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim * 3, output_dim),
            nn.Tanh(),
            nn.Linear(output_dim, output_dim),
            nn.Tanh(),
        )

    def forward(
        self, x: torch.Tensor, edge_w: torch.Tensor, r_in: torch.Tensor, r_out: torch.Tensor
    ) -> torch.Tensor:
        """Aggregate edge-weighted neighbor messages into every hit's new features."""
        b_out = torch.bmm(r_out.transpose(1, 2), x)
        b_in = torch.bmm(r_in.transpose(1, 2), x)
        r_w_out = r_out * edge_w[:, None]
        r_w_in = r_in * edge_w[:, None]
        m_in = torch.bmm(r_w_in, b_out)
        m_out = torch.bmm(r_w_out, b_in)
        combined = torch.cat([m_in, m_out, x], dim=2)
        return self.network(combined)


class TrackMLGNN(nn.Module):
    """HEPTrkX segment-classification GNN over a bipartite detector-hit graph.

    Iteratively alternates an ``EdgeNetwork`` (scores every candidate hit-pair
    segment) and a ``NodeNetwork`` (re-embeds every hit from edge-weighted neighbor
    aggregation via dense incidence matrices ``Ri``/``Ro``), with an input-feature
    skip connection each round, ending in one final edge-scoring pass, per
    Ju et al., "Graph Neural Networks for Particle Track Reconstruction"
    (arXiv:1810.06111), reproduced directly from
    ``HEPTrkX/heptrkx-gnn-tracking/models/gnn.py``.
    """

    def __init__(self, input_dim: int = 3, hidden_dim: int = 8, n_iters: int = 3) -> None:
        super().__init__()
        self.n_iters = n_iters
        self.input_network = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.edge_network = _TrackEdgeNetwork(input_dim + hidden_dim, hidden_dim)
        self.node_network = _TrackNodeNetwork(input_dim + hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, r_in: torch.Tensor, r_out: torch.Tensor) -> torch.Tensor:
        """Classify every candidate segment as real-track vs. fake.

        Parameters
        ----------
        x : torch.Tensor
            (B, N_hits, input_dim) per-hit spatial features.
        r_in : torch.Tensor
            (B, N_hits, N_edges) incoming-endpoint incidence matrix.
        r_out : torch.Tensor
            (B, N_hits, N_edges) outgoing-endpoint incidence matrix.
        """
        h = self.input_network(x)
        h = torch.cat([h, x], dim=-1)
        for _ in range(self.n_iters):
            edge_w = self.edge_network(h, r_in, r_out)
            h = self.node_network(h, edge_w, r_in, r_out)
            h = torch.cat([h, x], dim=-1)
        return self.edge_network(h, r_in, r_out)


def build_trackml_gnn() -> nn.Module:
    """Build a tiny HEPTrkX-style TrackML GNN segment classifier."""
    return TrackMLGNN(input_dim=3, hidden_dim=8, n_iters=3).eval()


def example_input_trackml_gnn() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a random bipartite hit graph (features + incidence matrices)."""
    b, n_hits, n_edges, input_dim = 2, 10, 15, 3
    x = torch.randn(b, n_hits, input_dim)
    r_in = torch.zeros(b, n_hits, n_edges)
    r_out = torch.zeros(b, n_hits, n_edges)
    hit_pairs = torch.randint(0, n_hits, (b, n_edges, 2))
    for bi in range(b):
        for ei in range(n_edges):
            src, dst = hit_pairs[bi, ei]
            r_out[bi, src, ei] = 1.0
            r_in[bi, dst, ei] = 1.0
    return x, r_in, r_out


# ---------------------------------------------------------------------------
# TransPolymer: chemically-aware P-SMILES tokenized sequence -> RoBERTa backbone
# -> lightweight MLP regression head.
# ---------------------------------------------------------------------------


class TransPolymer(nn.Module):
    """RoBERTa-backbone polymer-property regressor over tokenized P-SMILES.

    Wraps a tiny ``transformers.RobertaModel`` (standing in for TransPolymer's
    chemically-aware P-SMILES-tokenized pretrained RoBERTa encoder) with the
    2-layer MLP ``Regressor`` head from ``Downstream.py``'s
    ``DownstreamRegression``, applied to the backbone's pooled ``[CLS]``-token
    output, per Xu, Chen, Vlachos, arXiv:2209.01307.
    """

    def __init__(
        self,
        vocab_size: int = 64,
        hidden_size: int = 32,
        num_layers: int = 2,
        num_heads: int = 4,
        max_position_embeddings: int = 64,
    ) -> None:
        super().__init__()
        from transformers import RobertaConfig, RobertaModel

        config = RobertaConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=max_position_embeddings + 2,
            pad_token_id=1,
        )
        self.backbone = RobertaModel(config, add_pooling_layer=True)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Predict a scalar polymer property from a P-SMILES token sequence."""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        return self.regressor(pooled)


def build_transpolymer() -> nn.Module:
    """Build a tiny TransPolymer RoBERTa-backbone polymer-property regressor."""
    return TransPolymer(
        vocab_size=64, hidden_size=32, num_layers=2, num_heads=4, max_position_embeddings=64
    ).eval()


def example_input_transpolymer() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a random tokenized P-SMILES batch for TransPolymer."""
    input_ids = torch.randint(2, 64, (2, 24))
    attention_mask = torch.ones(2, 24, dtype=torch.long)
    return input_ids, attention_mask


MENAGERIE_ENTRIES = [
    ("Stormer weather", "build_stormer_weather", "example_input_stormer_weather", "2023", "SCI"),
    (
        "Synthon Completion Network",
        "build_synthon_completion_network",
        "example_input_synthon_completion_network",
        "2021",
        "GRAPH",
    ),
    (
        "TF-Net turbulence",
        "build_tfnet_turbulence",
        "example_input_tfnet_turbulence",
        "2023",
        "SCI",
    ),
    ("TinNet", "build_tinnet", "example_input_tinnet", "2021", "SCI"),
    ("TrackML GNN", "build_trackml_gnn", "example_input_trackml_gnn", "2018", "GRAPH"),
    ("TransPolymer", "build_transpolymer", "example_input_transpolymer", "2022", "NLP"),
]
