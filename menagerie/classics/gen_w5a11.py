"""Compact faithful classics for six genomics/single-cell architectures.

Sources checked (repo code inspected via GitHub API, base env only, no clone
or pip install):
  - SATORI: https://github.com/fahadahaf/satori (satori/models.py,
    ``AttentionNet``). Ullah & Ben-Hur, "A self-attention model for inferring
    cooperativity between regulatory features", Nucleic Acids Research 2021.
  - scGAE: https://github.com/ZixiangLuo1161/scGAE (layers.py, scgae.py,
    ``SCGAE``). Luo et al., bioRxiv 2021 ("scGAE: a Graph Autoencoder-based
    single-cell RNA-seq analysis method"). Original is TensorFlow/Spektral;
    reimplemented with ``torch_geometric.nn.GATConv`` for the encoder.
  - scGPT-perturbation: https://github.com/bowang-lab/scGPT
    (scgpt/model/generation_model.py, ``TransformerGenerator``). Cui et al.,
    "scGPT: toward building a foundation model for single-cell multi-omics
    using generative AI", Nature Methods 2024. Distinct from the base scGPT
    masked-reconstruction classic already in the catalog: this variant adds a
    perturbation-flag embedding stream and predicts post-perturbation
    expression via an affine decoder head.
  - scNODE: https://github.com/rsinghlab/scNODE (model/dynamic_model.py,
    model/layer.py, model/diff_solver.py, ``scNODE``/``LinearVAENet``/``ODE``).
    Zhang & Singh, "scNODE: generative model for temporal single cell
    transcriptomic data prediction", Bioinformatics 2024. Original solves the
    latent ODE with ``torchdiffeq``; reimplemented with a fixed-step RK4
    integrator (base env has no ``torchdiffeq``) over the same VAE-encode ->
    latent-drift-integrate -> decode pipeline.
  - scPoli: https://github.com/theislab/scarches
    (scarches/models/scpoli/scpoli.py, ``scpoli``/``Encoder``/``Decoder``/
    ``CondLayers``). De Donno et al., "Population-level integration of
    single-cell datasets enables multi-scale analysis across samples", Nature
    Methods 2023. Conditional-VAE with condition embeddings injected via a
    split linear (``CondLayers``) plus Euclidean-distance prototype
    classification in latent space.
  - SemiBin(2): https://github.com/BigDataBiology/SemiBin
    (SemiBin/semi_supervised_model.py, ``Semi_encoding_multiple``). Pan et al.,
    "A deep siamese neural network improves metagenome-assembled genomes in
    microbiome datasets across different environments", Bioinformatics 2023
    Supplement (SemiBin2). Siamese shared-weight autoencoder over paired
    contig feature vectors, trained with a must-link/cannot-link contrastive
    margin loss; here we expose the shared forward path (embed both contigs,
    reconstruct both) as the traceable module.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


# ---------------------------------------------------------------------------
# SATORI: CNN motif scanner + explicit multi-head self-attention (per-head
# Q/K/V linear projections, manual scaled dot-product attention) over one-hot
# encoded genomic sequence, classification head on the pooled attention output.
# ---------------------------------------------------------------------------


class SatoriAttentionNet(nn.Module):
    """CNN + multi-head self-attention regulatory-interaction model.

    Faithful to ``AttentionNet`` in SATORI: a 1D convolutional motif scanner
    over one-hot DNA sequence, followed by an explicit multi-head
    self-attention block built from per-head ``nn.Linear`` Q/K/V projections
    and manual scaled dot-product attention (not ``nn.MultiheadAttention``),
    then a linear classification head.
    """

    def __init__(
        self,
        num_channels: int = 4,
        num_filters: int = 32,
        filter_size: int = 9,
        num_heads: int = 4,
        head_size: int = 16,
        multihead_size: int = 48,
        num_classes: int = 2,
    ) -> None:
        """Initialize the CNN motif scanner and multi-head attention block.

        Parameters
        ----------
        num_channels:
            One-hot input channels (4 for DNA nucleotides).
        num_filters:
            Number of 1D convolutional motif filters.
        filter_size:
            Convolutional kernel width.
        num_heads:
            Number of self-attention heads.
        head_size:
            Per-head Q/K/V projection width.
        multihead_size:
            Width after concatenating and mixing all heads.
        num_classes:
            Output classification width.
        """

        super().__init__()
        self.num_heads = num_heads
        self.conv = nn.Sequential(
            nn.Conv1d(num_channels, num_filters, filter_size, padding=filter_size // 2, bias=False),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(0.2)
        self.query = nn.ModuleList([nn.Linear(num_filters, head_size) for _ in range(num_heads)])
        self.key = nn.ModuleList([nn.Linear(num_filters, head_size) for _ in range(num_heads)])
        self.value = nn.ModuleList([nn.Linear(num_filters, head_size) for _ in range(num_heads)])
        self.head_relu = nn.ModuleList([nn.ReLU() for _ in range(num_heads)])
        self.multihead_linear = nn.Linear(head_size * num_heads, multihead_size)
        self.multihead_relu = nn.ReLU()
        self.classifier = nn.Linear(multihead_size, num_classes)

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Return scaled dot-product attention output for one head."""

        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k**0.5)
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, v)

    def forward(self, one_hot_seq: torch.Tensor) -> torch.Tensor:
        """Score a batch of one-hot DNA sequences for regulatory interactions.

        Parameters
        ----------
        one_hot_seq:
            Tensor shaped ``(batch, channels, length)``.

        Returns
        -------
        torch.Tensor
            Class logits shaped ``(batch, num_classes)``.
        """

        feat = self.conv(one_hot_seq)
        feat = self.dropout(feat).permute(0, 2, 1)
        heads = []
        for i in range(self.num_heads):
            q, k, v = self.query[i](feat), self.key[i](feat), self.value[i](feat)
            attn_out = self.head_relu[i](self._attention(q, k, v))
            heads.append(attn_out)
        combined = torch.cat(heads, dim=2)
        mixed = self.multihead_relu(self.multihead_linear(combined))
        pooled = mixed.mean(dim=1)
        return self.classifier(pooled)


def build_satori() -> nn.Module:
    """Build a compact SATORI attention-genomics model."""

    return SatoriAttentionNet().eval()


def example_input_satori() -> torch.Tensor:
    """Return a batch of one-hot DNA sequences for SATORI."""

    return torch.eye(4)[torch.randint(0, 4, (2, 200))].permute(0, 2, 1).contiguous()


# ---------------------------------------------------------------------------
# scGAE: graph-attention encoder over a cell-cell similarity graph, decoded by
# a bilinear adjacency reconstruction head and an MLP expression-reconstruction
# head (dual-decoder graph autoencoder).
# ---------------------------------------------------------------------------


class BilinearAdjDecoder(nn.Module):
    """Bilinear adjacency-reconstruction decoder (faithful to ``Bilinear``)."""

    def __init__(self, latent_dim: int, adj_dim: int) -> None:
        """Initialize the projection and bilinear kernel.

        Parameters
        ----------
        latent_dim:
            Encoder output width.
        adj_dim:
            Width of the bilinear projection space.
        """

        super().__init__()
        self.proj = nn.Linear(latent_dim, adj_dim, bias=False)
        self.kernel = nn.Parameter(torch.randn(adj_dim, adj_dim) * 0.1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct a sigmoid adjacency matrix from latent codes.

        Parameters
        ----------
        z:
            Latent node embeddings shaped ``(num_nodes, latent_dim)``.

        Returns
        -------
        torch.Tensor
            Reconstructed adjacency shaped ``(num_nodes, num_nodes)``.
        """

        h = self.proj(z)
        h = h @ self.kernel
        return torch.sigmoid(h @ h.t())


class ScGAE(nn.Module):
    """Graph-attention autoencoder for scRNA-seq embedding and clustering.

    Faithful to ``SCGAE`` (``layer_enc="GAT"``): a two-layer graph-attention
    encoder over the cell-cell kNN similarity graph produces a latent
    embedding, which is decoded by (1) a bilinear adjacency-reconstruction
    head and (2) an MLP expression-reconstruction head.
    """

    def __init__(
        self,
        in_dim: int = 32,
        hidden_dim: int = 24,
        latent_dim: int = 8,
        adj_dim: int = 12,
        dec_dims: Tuple[int, int, int] = (16, 32, 48),
    ) -> None:
        """Initialize the GAT encoder and dual decoders.

        Parameters
        ----------
        in_dim:
            Gene-expression feature width per cell.
        hidden_dim:
            Hidden GAT layer width.
        latent_dim:
            Latent embedding width.
        adj_dim:
            Bilinear decoder projection width.
        dec_dims:
            Hidden widths of the expression-reconstruction MLP.
        """

        super().__init__()
        self.drop_in = nn.Dropout(0.2)
        self.gat1 = GATConv(in_dim, hidden_dim, heads=1)
        self.gat2 = GATConv(hidden_dim, latent_dim, heads=1)
        self.decoder_a = BilinearAdjDecoder(latent_dim, adj_dim)
        self.decoder_x = nn.Sequential(
            nn.Linear(latent_dim, dec_dims[0]),
            nn.ReLU(),
            nn.Linear(dec_dims[0], dec_dims[1]),
            nn.ReLU(),
            nn.Linear(dec_dims[1], dec_dims[2]),
            nn.ReLU(),
            nn.Linear(dec_dims[2], in_dim),
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode cells and reconstruct adjacency and expression.

        Parameters
        ----------
        x:
            Cell-by-gene expression features shaped ``(num_cells, in_dim)``.
        edge_index:
            Cell-cell similarity graph edges shaped ``(2, num_edges)``.

        Returns
        -------
        tuple of torch.Tensor
            ``(latent, reconstructed_adjacency, reconstructed_expression)``.
        """

        h = self.drop_in(x)
        h = F.relu(self.gat1(h, edge_index))
        z = self.gat2(h, edge_index)
        adj_rec = self.decoder_a(z)
        x_rec = self.decoder_x(z)
        return z, adj_rec, x_rec


def build_scgae() -> nn.Module:
    """Build a compact scGAE graph autoencoder."""

    return ScGAE().eval()


def example_input_scgae() -> Tuple[torch.Tensor, torch.Tensor]:
    """Return cell features and a kNN-style similarity graph for scGAE."""

    num_cells = 20
    x = torch.randn(num_cells, 32)
    src = torch.randint(0, num_cells, (60,))
    dst = torch.randint(0, num_cells, (60,))
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    return x, edge_index


# ---------------------------------------------------------------------------
# scGPT-perturbation: TransformerGenerator variant of scGPT that adds a
# perturbation-flag embedding stream (on top of gene + value embeddings) and
# predicts post-perturbation expression with an affine decoder head.
# ---------------------------------------------------------------------------


class ScGPTPerturbationGeneEncoder(nn.Module):
    """Gene-identity embedding lookup."""

    def __init__(self, num_genes: int, d_model: int) -> None:
        """Initialize the gene embedding table."""

        super().__init__()
        self.embedding = nn.Embedding(num_genes, d_model, padding_idx=0)

    def forward(self, gene_ids: torch.Tensor) -> torch.Tensor:
        """Embed gene identity tokens."""

        return self.embedding(gene_ids)


class ScGPTPerturbationValueEncoder(nn.Module):
    """Continuous expression-value encoder (MLP)."""

    def __init__(self, d_model: int) -> None:
        """Initialize the value-encoding MLP."""

        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, d_model), nn.ReLU(), nn.Linear(d_model, d_model))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Encode continuous expression values shaped ``(batch, genes)``."""

        return self.net(values.unsqueeze(-1))


class AffineExprDecoder(nn.Module):
    """Affine expression-prediction decoder (faithful to ``AffineExprDecoder``).

    Predicts a per-gene scale and shift from the transformer output and
    applies them to a learned base value, matching scGPT's affine
    perturbation-response decoding head.
    """

    def __init__(self, d_model: int) -> None:
        """Initialize the scale/shift projection heads."""

        super().__init__()
        self.fc = nn.Sequential(nn.Linear(d_model, d_model), nn.LeakyReLU())
        self.scale = nn.Linear(d_model, 1)
        self.shift = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict post-perturbation expression from transformer output."""

        h = self.fc(x)
        scale = F.softplus(self.scale(h)).squeeze(-1)
        shift = self.shift(h).squeeze(-1)
        return scale + shift


class ScGPTPerturbation(nn.Module):
    """scGPT perturbation-response transformer (``TransformerGenerator``).

    Faithful to the perturbation fine-tuning head in scGPT: gene identity
    embeddings, continuous expression-value embeddings, and a
    perturbation-flag embedding (padding/control/perturbed) are summed into a
    single token stream, passed through a standard transformer encoder, and
    decoded per-gene with an affine (scale, shift) head that predicts
    post-perturbation expression.
    """

    def __init__(
        self,
        num_genes: int = 48,
        d_model: int = 32,
        nhead: int = 4,
        d_hid: int = 64,
        nlayers: int = 2,
    ) -> None:
        """Initialize gene/value/perturbation embeddings, transformer, decoder."""

        super().__init__()
        self.gene_encoder = ScGPTPerturbationGeneEncoder(num_genes, d_model)
        self.value_encoder = ScGPTPerturbationValueEncoder(d_model)
        self.pert_encoder = nn.Embedding(3, d_model, padding_idx=2)
        layer = nn.TransformerEncoderLayer(d_model, nhead, d_hid, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, nlayers)
        self.decoder = AffineExprDecoder(d_model)

    def forward(
        self, gene_ids: torch.Tensor, values: torch.Tensor, pert_flags: torch.Tensor
    ) -> torch.Tensor:
        """Predict post-perturbation expression values.

        Parameters
        ----------
        gene_ids:
            Gene identity tokens shaped ``(batch, genes)``.
        values:
            Pre-perturbation expression values shaped ``(batch, genes)``.
        pert_flags:
            Perturbation flags (0=pad, 1=control, 2=perturbed) shaped
            ``(batch, genes)``.

        Returns
        -------
        torch.Tensor
            Predicted post-perturbation expression shaped ``(batch, genes)``.
        """

        total = (
            self.gene_encoder(gene_ids) + self.value_encoder(values) + self.pert_encoder(pert_flags)
        )
        encoded = self.transformer(total)
        return self.decoder(encoded)


def build_scgpt_perturbation() -> nn.Module:
    """Build a compact scGPT-perturbation transformer."""

    return ScGPTPerturbation().eval()


def example_input_scgpt_perturbation() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return gene tokens, values, and perturbation flags for scGPT-perturbation."""

    gene_ids = torch.randint(0, 48, (2, 24))
    values = torch.randn(2, 24)
    pert_flags = torch.randint(0, 2, (2, 24))
    return gene_ids, values, pert_flags


# ---------------------------------------------------------------------------
# scNODE: VAE encoder maps the first-timepoint expression matrix to a latent
# Gaussian, a neural-ODE drift network integrates the latent sample forward to
# the requested timepoints (fixed-step RK4, no torchdiffeq dependency), and a
# decoder maps each latent timepoint back to gene space.
# ---------------------------------------------------------------------------


class ScNodeVAEEncoder(nn.Module):
    """Linear VAE encoder (faithful to ``LinearVAENet``)."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        """Initialize the trunk and mean/std heads."""

        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.std_layer = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return latent mean and (non-negative) std."""

        h = self.trunk(x)
        mu = self.mu_layer(h)
        std = torch.abs(self.std_layer(h)) + 1e-4
        return mu, std


class ScNodeDriftNet(nn.Module):
    """Latent-space drift network (faithful to ``LinearNet`` used as ODE RHS)."""

    def __init__(self, latent_dim: int, hidden_dim: int) -> None:
        """Initialize the drift MLP."""

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return the instantaneous drift at latent state ``z``."""

        return self.net(z)


class ScNODE(nn.Module):
    """VAE + neural-ODE model for temporal scRNA-seq trajectory prediction.

    Faithful to ``scNODE``: the first-timepoint expression matrix is encoded
    to a latent Gaussian (reparameterized sample), a drift network integrates
    the latent sample forward across the requested timepoints via a
    fixed-step RK4 ODE solver (the source uses ``torchdiffeq.odeint``; base
    env has no ``torchdiffeq`` so a compact RK4 stands in for the same
    latent-ODE mechanism), and a decoder maps every latent timepoint back to
    gene-expression space.
    """

    def __init__(
        self,
        input_dim: int = 40,
        hidden_dim: int = 24,
        latent_dim: int = 8,
        rk4_steps: int = 4,
    ) -> None:
        """Initialize the VAE encoder, ODE drift net, and observation decoder."""

        super().__init__()
        self.latent_dim = latent_dim
        self.rk4_steps = rk4_steps
        self.encoder = ScNodeVAEEncoder(input_dim, hidden_dim, latent_dim)
        self.drift = ScNodeDriftNet(latent_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def _rk4_integrate(self, z0: torch.Tensor, tps: torch.Tensor) -> torch.Tensor:
        """Integrate the latent drift forward over ``tps`` with fixed-step RK4.

        Parameters
        ----------
        z0:
            Initial latent state shaped ``(batch, latent_dim)``.
        tps:
            Increasing timepoints shaped ``(num_tps,)``.

        Returns
        -------
        torch.Tensor
            Latent trajectory shaped ``(batch, num_tps, latent_dim)``.
        """

        states = [z0]
        z = z0
        for i in range(tps.shape[0] - 1):
            dt = (tps[i + 1] - tps[i]) / self.rk4_steps
            for _ in range(self.rk4_steps):
                k1 = self.drift(z)
                k2 = self.drift(z + 0.5 * dt * k1)
                k3 = self.drift(z + 0.5 * dt * k2)
                k4 = self.drift(z + dt * k3)
                z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            states.append(z)
        return torch.stack(states, dim=1)

    def forward(self, first_tp_data: torch.Tensor, tps: torch.Tensor) -> torch.Tensor:
        """Predict expression trajectories from the first-timepoint data.

        Parameters
        ----------
        first_tp_data:
            Expression matrix at the first timepoint, shaped
            ``(batch, genes)``.
        tps:
            Timepoints to predict, shaped ``(num_tps,)``.

        Returns
        -------
        torch.Tensor
            Predicted expression trajectory shaped
            ``(batch, num_tps, genes)``.
        """

        mu, std = self.encoder(first_tp_data)
        eps = torch.randn_like(std)
        z0 = mu + eps * std
        latent_seq = self._rk4_integrate(z0, tps)
        return self.decoder(latent_seq)


def build_scnode() -> nn.Module:
    """Build a compact scNODE VAE + latent-ODE model."""

    return ScNODE().eval()


def example_input_scnode() -> Tuple[torch.Tensor, torch.Tensor]:
    """Return first-timepoint expression data and target timepoints for scNODE."""

    first_tp_data = torch.rand(6, 40)
    tps = torch.tensor([0.0, 1.0, 2.0, 3.0])
    return first_tp_data, tps


# ---------------------------------------------------------------------------
# scPoli: conditional VAE whose first encoder/decoder layer splits into an
# expression branch and a condition-embedding branch summed together
# (``CondLayers``), plus Euclidean-distance prototype classification of the
# latent embedding against per-cell-type prototype vectors.
# ---------------------------------------------------------------------------


class ScPoliCondLayer(nn.Module):
    """Condition-conditioned linear layer (faithful to ``CondLayers``)."""

    def __init__(self, in_dim: int, out_dim: int, cond_dim: int) -> None:
        """Initialize the expression and condition-embedding projections."""

        super().__init__()
        self.expr = nn.Linear(in_dim, out_dim, bias=True)
        self.cond = nn.Linear(cond_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        """Sum the expression and condition-embedding projections."""

        return self.expr(x) + self.cond(cond_embed)


class ScPoli(nn.Module):
    """Conditional VAE with condition embeddings and prototype classification.

    Faithful to ``scpoli``: a batch/condition embedding table is looked up per
    cell and injected into the first encoder and decoder layers via
    ``CondLayers`` (expression branch + condition branch summed). The encoder
    produces a latent Gaussian; cell-type prediction is done by Euclidean
    distance from the reparameterized latent sample to learned per-cell-type
    prototype vectors (nearest-prototype classification), matching scPoli's
    "population-level" prototype-based cell typing.
    """

    def __init__(
        self,
        input_dim: int = 30,
        hidden_dim: int = 24,
        latent_dim: int = 10,
        num_conditions: int = 5,
        cond_dim: int = 6,
        num_cell_types: int = 4,
    ) -> None:
        """Initialize condition embeddings, encoder/decoder, and prototypes."""

        super().__init__()
        self.condition_embedding = nn.Embedding(num_conditions, cond_dim)
        self.enc_l0 = ScPoliCondLayer(input_dim, hidden_dim, cond_dim)
        self.enc_act = nn.ReLU()
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.log_var_layer = nn.Linear(hidden_dim, latent_dim)

        self.dec_l0 = ScPoliCondLayer(latent_dim, hidden_dim, cond_dim)
        self.dec_act = nn.ReLU()
        self.recon = nn.Linear(hidden_dim, input_dim)

        self.prototypes = nn.Parameter(torch.randn(num_cell_types, latent_dim) * 0.1)

    def forward(
        self, x: torch.Tensor, condition: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode/reconstruct expression and classify by nearest prototype.

        Parameters
        ----------
        x:
            Cell-by-gene expression matrix shaped ``(batch, input_dim)``.
        condition:
            Integer condition/batch id per cell shaped ``(batch,)``.

        Returns
        -------
        tuple of torch.Tensor
            ``(reconstruction, latent_sample, prototype_distances)``.
        """

        cond_embed = self.condition_embedding(condition)
        h = self.enc_act(self.enc_l0(x, cond_embed))
        mu = self.mu_layer(h)
        log_var = self.log_var_layer(h)
        std = torch.sqrt(torch.exp(log_var) + 1e-4)
        z = mu + std * torch.randn_like(std)

        dec_h = self.dec_act(self.dec_l0(z, cond_embed))
        recon = self.recon(dec_h)

        dists = torch.cdist(z, self.prototypes)
        return recon, z, dists


def build_scpoli() -> nn.Module:
    """Build a compact scPoli conditional VAE with prototype classification."""

    return ScPoli().eval()


def example_input_scpoli() -> Tuple[torch.Tensor, torch.Tensor]:
    """Return expression data and condition ids for scPoli."""

    x = torch.rand(8, 30)
    condition = torch.randint(0, 5, (8,))
    return x, condition


# ---------------------------------------------------------------------------
# SemiBin(2): siamese shared-weight autoencoder over paired contig feature
# vectors (k-mer + coverage composition), used with a must-link/cannot-link
# contrastive margin loss for semi-supervised metagenomic contig binning.
# ---------------------------------------------------------------------------


class SemiBinSiameseAutoencoder(nn.Module):
    """Siamese autoencoder over paired contig features (``Semi_encoding_multiple``).

    Faithful to SemiBin2's semi-supervised model: a single shared encoder and
    a single shared decoder are applied independently to two contig feature
    vectors (a "pair"), producing two embeddings and two reconstructions.
    Training combines a must-link/cannot-link contrastive margin loss on the
    embedding pair with a reconstruction loss on both decoded outputs; only
    the shared forward path is exposed here as the traceable module.
    """

    def __init__(self, num_features: int = 136, embed_dim: int = 20) -> None:
        """Initialize the shared encoder and decoder MLPs.

        Parameters
        ----------
        num_features:
            Width of the contig feature vector (k-mer + coverage composition).
        embed_dim:
            Embedding bottleneck width.
        """

        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, embed_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_features),
            nn.Sigmoid(),
        )

    def forward(
        self, contig_a: torch.Tensor, contig_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed and reconstruct a pair of contig feature vectors.

        Parameters
        ----------
        contig_a:
            First contig's feature vector shaped ``(batch, num_features)``.
        contig_b:
            Second contig's (must-link/cannot-link partner) feature vector
            shaped ``(batch, num_features)``.

        Returns
        -------
        tuple of torch.Tensor
            ``(embedding_a, embedding_b, recon_a, recon_b)``.
        """

        embed_a = self.encoder(contig_a)
        embed_b = self.encoder(contig_b)
        recon_a = self.decoder(embed_a)
        recon_b = self.decoder(embed_b)
        return embed_a, embed_b, recon_a, recon_b


def build_semibin() -> nn.Module:
    """Build a compact SemiBin siamese contig autoencoder."""

    return SemiBinSiameseAutoencoder().eval()


def example_input_semibin() -> List[torch.Tensor]:
    """Return a pair of contig feature vectors for SemiBin."""

    return [torch.rand(4, 136), torch.rand(4, 136)]


MENAGERIE_ENTRIES = [
    ("SATORI (attention genomics)", "build_satori", "example_input_satori", "2021", "BIO"),
    ("scGAE", "build_scgae", "example_input_scgae", "2021", "BIO"),
    (
        "scGPT-perturbation",
        "build_scgpt_perturbation",
        "example_input_scgpt_perturbation",
        "2024",
        "BIO",
    ),
    ("scNODE", "build_scnode", "example_input_scnode", "2024", "BIO"),
    ("scPoli", "build_scpoli", "example_input_scpoli", "2023", "BIO"),
    ("SemiBin", "build_semibin", "example_input_semibin", "2023", "BIO"),
]
