"""Compact faithful classics for six genomics/single-cell architectures.

Sources checked (repo code inspected via GitHub API, base env only, no clone
or pip install):
  - Saluki: https://github.com/vagarwal87/saluki_paper (analysis/figures only;
    model config at https://github.com/calico/basenji
    ``manuscripts/saluki/params.json``). Agarwal & Kelley, "The genetic and
    biochemical determinants of mRNA degradation rates in mammals", Genome
    Biology 2022. Hybrid conv+recurrent regressor: one-hot mRNA sequence
    (4 bases) concatenated with per-position coding-frame and splice-site
    indicator channels, a tower of dilated/pooled 1D conv blocks with
    residual adds, final GRU over the position axis, two linear heads
    (human/mouse half-life).
  - SAMS-VAE: https://github.com/insitro/sams-vae (sams_vae/models/sams_vae/
    model.py ``SAMSVAEModel``, guides/mean_field_normal_guide.py
    ``SAMSVAEMeanFieldNormalGuide``). Bereket & Karaletsos, "Modelling
    Cellular Perturbations with the Sparse Additive Mechanism Shift
    Variational Autoencoder", NeurIPS 2023. Perturbation VAE whose latent is
    a sum of a basal (perturbation-free) code and a *sparse* dosage-weighted
    combination of per-treatment embeddings ``E`` gated by a learned
    Bernoulli ``mask`` (here relaxed via a Gumbel-sigmoid straight-through
    estimator to keep the mask differentiable and traceable), decoded back
    to expression space by an MLP.
  - SATURN: https://github.com/snap-stanford/SATURN (model/saturn_model.py,
    ``SATURNPretrainModel``). Rosen et al., "Toward universal cell embeddings:
    integrating single-cell RNA-sequencing datasets across species", Nature
    Methods 2024. Cross-species integration via a shared *macrogene* space: a
    learned non-negative gene-to-macrogene weight matrix (``p_weights``,
    seeded from cross-species protein-embedding similarity, kept positive via
    ``.exp()``) projects each species' expression vector onto a common set of
    macrogene "centroids"; a VAE then encodes/decodes in macrogene space and
    the same weight matrix (transposed) projects back out to the querying
    species' gene set, with a per-species dropout/rate decoder head.
  - SAUCIE: https://github.com/KrishnaswamyLab/SAUCIE (model.py, class
    ``SAUCIE``). Amodio et al., "Exploring single-cell data with deep
    multitasking neural networks", Nature Methods 2019. Original is
    TensorFlow 1.x; reimplemented in torch. Symmetric fully-connected
    autoencoder (3 LeakyReLU encoder layers -> 2D visualization bottleneck ->
    3 LeakyReLU/ReLU decoder layers) whose penultimate encoder layer
    (``layer_c``) is exposed separately because in the original it carries
    an L1 "ID-regularization" sparsity penalty used downstream for
    binarization-based clustering; we surface that activation as a named
    output alongside the 2D embedding and the reconstruction.
  - scArches (trVAE base model): https://github.com/theislab/scarches
    (scarches/models/trvae/modules.py, ``CondLayers``/``Encoder``/
    ``Decoder``). Lotfollahi et al., "Mapping single-cell data to reference
    atlases by transfer learning", Nature Biotechnology 2022. Conditional VAE
    whose first encoder layer and first decoder layer are ``CondLayers``:
    the condition one-hot is routed through a *separate* linear
    (``cond_L``, no bias) summed with the expression linear (``expr_L``),
    so a new condition can later be added by only training new ``cond_L``
    columns (architecture-surgery / reference-mapping design) while
    ``expr_L`` stays frozen. Distinct from the already-catalogued scPoli
    (which adds prototype classification on top of this same CondLayers
    backbone) -- this classic is the plain trVAE/scArches conditional VAE.
  - scCello: https://github.com/DeepGraphLearning/scCello
    (sccello/src/model_prototype_contrastive.py,
    ``PrototypeContrastiveModel``/``PrototypeContrastiveForMaskedLM``). Yuan
    (Wenlong Yuan) et al. / Fang, Ji et al., "scCello: Cell-ontology guided
    transcriptome foundation model", NeurIPS 2024 Spotlight. BERT-style
    encoder (Geneformer/scBERT-family rank-value gene-token input, MLM head)
    whose distinctive addition is cell-ontology-guided *prototype
    contrastive learning*: a learned per-cell-type centroid embedding table
    (``learned_centroid``) that CLS embeddings are L2-normalized and
    temperature-scaled against, with a cell-ontology similarity graph
    reweighting the prototype/cell contrast during training. We expose the
    encoder forward pass plus the CLS-to-prototype similarity computation
    (the traceable inference-time mechanism); the ontology-graph loss
    reweighting itself is a training-time loss term, not part of the forward
    computation graph.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Saluki: dilated conv tower + GRU regressor for mRNA half-life prediction.
# Input channels = 4 one-hot bases + coding-frame indicator + splice-site
# indicator (6 channels total, per basenji/manuscripts/saluki/params.json:
# filters=64, kernel_size=5, num_layers=6, rnn_type=gru, heads=2).
# ---------------------------------------------------------------------------


class SalukiConvBlock(nn.Module):
    """One dilated-conv + batchnorm + relu + max-pool block with residual add."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=5,
            dilation=dilation,
            padding=(5 - 1) * dilation // 2,
        )
        self.bn = nn.BatchNorm1d(channels)
        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn(self.conv(x)))
        h = h + x
        return self.pool(h)


class SalukiRegressor(nn.Module):
    """Dilated-conv tower + GRU regressor for mRNA degradation rate."""

    def __init__(self, in_channels: int = 6, filters: int = 32, num_layers: int = 4) -> None:
        super().__init__()
        self.stem = nn.Conv1d(in_channels, filters, kernel_size=5, padding=2)
        self.stem_bn = nn.BatchNorm1d(filters)
        self.blocks = nn.ModuleList(
            [SalukiConvBlock(filters, dilation=2**i) for i in range(num_layers)]
        )
        self.gru = nn.GRU(filters, filters, batch_first=True)
        self.head_human = nn.Linear(filters, 1)
        self.head_mouse = nn.Linear(filters, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.stem_bn(self.stem(x)))
        for block in self.blocks:
            h = block(h)
        h = h.transpose(1, 2)
        _, h_last = self.gru(h)
        h_last = h_last.squeeze(0)
        return self.head_human(h_last), self.head_mouse(h_last)


def build_saluki() -> nn.Module:
    """Build a compact Saluki mRNA half-life regressor."""

    return SalukiRegressor().eval()


def example_input_saluki() -> torch.Tensor:
    """Return a one-hot(4) + frame + splice encoded mRNA sequence batch."""

    return torch.rand(2, 6, 256)


# ---------------------------------------------------------------------------
# SAMS-VAE: sparse additive mechanism shift VAE. Latent z = z_basal +
# D @ (E * mask), where E is a per-treatment embedding matrix and mask is a
# learned (relaxed-Bernoulli) sparsity gate, D are per-sample dosages.
# ---------------------------------------------------------------------------


class SamsVaeEncoder(nn.Module):
    """Maps observed expression to a basal latent code (mean-field guide)."""

    def __init__(self, n_phenos: int, n_latent: int, n_hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_phenos, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.LeakyReLU(),
        )
        self.loc = nn.Linear(n_hidden, n_latent)
        self.log_scale = nn.Linear(n_hidden, n_latent)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.loc(h), self.log_scale(h)


class SamsVaeModel(nn.Module):
    """Sparse additive mechanism shift VAE over perturbation dosages."""

    def __init__(
        self,
        n_phenos: int = 48,
        n_treatments: int = 6,
        n_latent: int = 16,
        n_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.n_latent = n_latent
        self.encoder = SamsVaeEncoder(n_phenos, n_latent, n_hidden)
        self.q_e_loc = nn.Parameter(torch.randn(n_treatments, n_latent) * 0.1)
        self.q_e_log_scale = nn.Parameter(torch.zeros(n_treatments, n_latent))
        self.q_mask_logits = nn.Parameter(torch.zeros(n_treatments, n_latent))
        self.decoder = nn.Sequential(
            nn.Linear(n_latent, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_phenos),
        )

    def forward(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        z_loc, z_log_scale = self.encoder(x)
        z_basal = z_loc + torch.randn_like(z_loc) * z_log_scale.exp()

        e_sample = self.q_e_loc + torch.randn_like(self.q_e_loc) * self.q_e_log_scale.exp()
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(self.q_mask_logits).clamp_min(1e-8)))
        mask_sample = torch.sigmoid(self.q_mask_logits + gumbel_noise)

        z = z_basal + torch.matmul(d, e_sample * mask_sample)
        return self.decoder(z)


def build_sams_vae() -> nn.Module:
    """Build a compact SAMS-VAE perturbation model."""

    return SamsVaeModel().eval()


def example_input_sams_vae() -> List[torch.Tensor]:
    """Return expression observations and treatment dosages for SAMS-VAE."""

    return [torch.rand(4, 48), torch.rand(4, 6)]


# ---------------------------------------------------------------------------
# SATURN: cross-species integration through a shared "macrogene" space. A
# learned non-negative gene->macrogene weight matrix projects expression
# into a common centroid space; a VAE round-trips through that space and the
# transposed weight matrix maps back to the querying species' gene set.
# ---------------------------------------------------------------------------


def saturn_full_block(in_features: int, out_features: int, p_drop: float = 0.1) -> nn.Sequential:
    """Linear + LayerNorm + ReLU + Dropout block used throughout SATURN."""

    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=True),
        nn.LayerNorm(out_features),
        nn.ReLU(),
        nn.Dropout(p=p_drop),
    )


class SaturnPretrainModel(nn.Module):
    """Cross-species macrogene VAE (SATURN pretraining architecture)."""

    def __init__(
        self,
        num_genes: int = 40,
        num_macrogenes: int = 24,
        hidden_dim: int = 32,
        embed_dim: int = 8,
        num_species: int = 2,
    ) -> None:
        super().__init__()
        self.num_genes = num_genes
        self.num_macrogenes = num_macrogenes
        self.num_species = num_species

        self.p_weights = nn.Parameter(torch.randn(num_macrogenes, num_genes) * 0.1)
        self.cl_layer_norm = nn.LayerNorm(num_macrogenes)

        self.encoder = nn.Sequential(saturn_full_block(num_macrogenes, hidden_dim))
        self.fc_mu = nn.Linear(hidden_dim, embed_dim)
        self.fc_var = nn.Linear(hidden_dim, embed_dim)

        self.px_decoder = saturn_full_block(embed_dim + num_species, hidden_dim)
        self.cl_scale_decoder = saturn_full_block(hidden_dim, num_macrogenes)
        self.px_dropout_decoder = nn.Linear(hidden_dim, num_genes)

    def forward(
        self, expr: torch.Tensor, species_onehot: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expr = torch.log1p(expr)
        macrogene = F.linear(expr, self.p_weights.exp())
        macrogene = F.relu(self.cl_layer_norm(macrogene))

        h = self.encoder(macrogene)
        mu, log_var = self.fc_mu(h), self.fc_var(h)
        z = mu + torch.randn_like(mu) * (0.5 * log_var).exp()

        decoded = self.px_decoder(torch.cat([z, species_onehot], dim=-1))
        cl_scale = self.cl_scale_decoder(decoded)
        gene_logits = F.linear(cl_scale, self.p_weights.exp().t())
        px_scale = F.softmax(gene_logits, dim=-1)
        px_drop = self.px_dropout_decoder(decoded)
        return px_scale, px_drop, z


def build_saturn() -> nn.Module:
    """Build a compact SATURN cross-species macrogene VAE."""

    return SaturnPretrainModel().eval()


def example_input_saturn() -> List[torch.Tensor]:
    """Return an expression vector and a one-hot species indicator."""

    return [torch.rand(4, 40), F.one_hot(torch.zeros(4, dtype=torch.long), 2).float()]


# ---------------------------------------------------------------------------
# SAUCIE: symmetric fully-connected autoencoder with a 2D visualization
# bottleneck; penultimate encoder layer exposed separately (in the original
# it carries the ID-regularization sparsity penalty for clustering).
# ---------------------------------------------------------------------------


class SaucieAutoencoder(nn.Module):
    """Symmetric LeakyReLU autoencoder with a 2D embedding bottleneck."""

    def __init__(
        self, input_dim: int = 64, layers: Tuple[int, int, int, int] = (32, 16, 8, 2)
    ) -> None:
        super().__init__()
        h1, h2, h3, emb = layers
        self.enc0 = nn.Linear(input_dim, h1)
        self.enc1 = nn.Linear(h1, h2)
        self.enc2 = nn.Linear(h2, h3)
        self.embed = nn.Linear(h3, emb)

        self.dec0 = nn.Linear(emb, h3)
        self.dec1 = nn.Linear(h3, h2)
        self.dec2 = nn.Linear(h2, h1)
        self.recon = nn.Linear(h1, input_dim)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h1 = self.act(self.enc0(x))
        h2 = self.act(self.enc1(h1))
        layer_c = self.act(self.enc2(h2))
        embedding = self.embed(layer_c)

        d0 = self.act(self.dec0(embedding))
        d1 = self.act(self.dec1(d0))
        d2 = self.act(self.dec2(d1))
        reconstruction = self.recon(d2)
        return embedding, layer_c, reconstruction


def build_saucie() -> nn.Module:
    """Build a compact SAUCIE autoencoder."""

    return SaucieAutoencoder().eval()


def example_input_saucie() -> torch.Tensor:
    """Return a batch of single-cell expression vectors for SAUCIE."""

    return torch.rand(8, 64)


# ---------------------------------------------------------------------------
# scArches (trVAE base model): conditional VAE with CondLayers -- the first
# encoder/decoder linear splits into an expression path and a separate
# (bias-free) condition path that are summed, enabling later addition of
# new conditions without disturbing the expression weights.
# ---------------------------------------------------------------------------


class CondLayer(nn.Module):
    """Linear layer with a separate additive one-hot-condition projection."""

    def __init__(self, n_in: int, n_out: int, n_cond: int, bias: bool = True) -> None:
        super().__init__()
        self.expr_l = nn.Linear(n_in, n_out, bias=bias)
        self.cond_l = nn.Linear(n_cond, n_out, bias=False)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.expr_l(x) + self.cond_l(cond)


class TrvaeEncoder(nn.Module):
    """scArches/trVAE conditional encoder: CondLayers input layer + MLP."""

    def __init__(self, n_genes: int, n_cond: int, hidden: int, latent: int) -> None:
        super().__init__()
        self.cond_in = CondLayer(n_genes, hidden, n_cond)
        self.bn = nn.BatchNorm1d(hidden)
        self.hidden = nn.Sequential(nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU())
        self.mean_encoder = nn.Linear(hidden, latent)
        self.log_var_encoder = nn.Linear(hidden, latent)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.bn(self.cond_in(x, cond)))
        h = self.hidden(h)
        return self.mean_encoder(h), self.log_var_encoder(h)


class TrvaeDecoder(nn.Module):
    """scArches/trVAE conditional decoder: CondLayers first layer + MLP."""

    def __init__(self, n_genes: int, n_cond: int, hidden: int, latent: int) -> None:
        super().__init__()
        self.cond_in = CondLayer(latent, hidden, n_cond, bias=False)
        self.bn = nn.BatchNorm1d(hidden)
        self.recon = nn.Sequential(nn.Linear(hidden, n_genes), nn.ReLU())

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn(self.cond_in(z, cond)))
        return self.recon(h)


class TrvaeModel(nn.Module):
    """scArches base conditional VAE (trVAE) for reference-mapping."""

    def __init__(
        self,
        n_genes: int = 48,
        n_cond: int = 4,
        hidden: int = 32,
        latent: int = 10,
    ) -> None:
        super().__init__()
        self.encoder = TrvaeEncoder(n_genes, n_cond, hidden, latent)
        self.decoder = TrvaeDecoder(n_genes, n_cond, hidden, latent)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_var = self.encoder(x, cond)
        z = mean + torch.randn_like(mean) * (0.5 * log_var).exp()
        recon = self.decoder(z, cond)
        return recon, mean


def build_scarches() -> nn.Module:
    """Build a compact scArches (trVAE) conditional VAE."""

    return TrvaeModel().eval()


def example_input_scarches() -> List[torch.Tensor]:
    """Return an expression batch and a one-hot batch/condition label."""

    return [torch.rand(6, 48), F.one_hot(torch.randint(0, 4, (6,)), 4).float()]


# ---------------------------------------------------------------------------
# scCello: BERT-style transcriptome encoder over rank-ordered gene tokens
# with a cell-ontology-guided prototype-contrastive head -- CLS embeddings
# are L2-normalized and compared to a learned per-cell-type centroid table.
# ---------------------------------------------------------------------------


class ScCelloEncoderLayer(nn.Module):
    """One pre-norm transformer encoder layer (BERT-style)."""

    def __init__(self, hidden: int, heads: int, ff: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, ff), nn.GELU(), nn.Linear(ff, hidden))
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class ScCelloModel(nn.Module):
    """scCello: gene-token BERT encoder + cell-ontology prototype contrast head."""

    def __init__(
        self,
        vocab_size: int = 256,
        hidden: int = 32,
        heads: int = 4,
        ff: int = 64,
        n_layers: int = 2,
        max_len: int = 48,
        n_prototypes: int = 12,
        tau: float = 0.2,
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_len, hidden)
        self.embed_norm = nn.LayerNorm(hidden)
        self.layers = nn.ModuleList(
            [ScCelloEncoderLayer(hidden, heads, ff) for _ in range(n_layers)]
        )
        self.learned_centroid = nn.Embedding(n_prototypes, hidden)
        self.tau = tau

    def forward(self, gene_rank_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = gene_rank_ids.shape[1]
        positions = torch.arange(seq_len, device=gene_rank_ids.device).unsqueeze(0)
        h = self.embed_norm(
            self.word_embeddings(gene_rank_ids) + self.position_embeddings(positions)
        )
        for layer in self.layers:
            h = layer(h)

        cls = h[:, 0, :]
        cell_embed = F.normalize(cls, p=2, dim=-1) / (self.tau**0.5)
        prototypes = F.normalize(self.learned_centroid.weight, p=2, dim=-1) / (self.tau**0.5)
        similarity = torch.matmul(cell_embed, prototypes.t())
        return cell_embed, similarity


def build_sccello() -> nn.Module:
    """Build a compact scCello cell-ontology prototype-contrastive encoder."""

    return ScCelloModel().eval()


def example_input_sccello() -> torch.Tensor:
    """Return a batch of rank-ordered gene-token id sequences for scCello."""

    return torch.randint(1, 256, (4, 48))


MENAGERIE_ENTRIES = [
    ("Saluki", "build_saluki", "example_input_saluki", "2022", "BIO"),
    ("SAMS-VAE", "build_sams_vae", "example_input_sams_vae", "2023", "BIO"),
    ("SATURN", "build_saturn", "example_input_saturn", "2024", "BIO"),
    ("SAUCIE", "build_saucie", "example_input_saucie", "2019", "BIO"),
    ("scArches", "build_scarches", "example_input_scarches", "2022", "BIO"),
    ("scCello", "build_sccello", "example_input_sccello", "2024", "BIO"),
]
