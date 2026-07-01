"""Wave 6 batch 1 menagerie classics: genomics/bioinformatics family.

Sources checked (repo code inspected via GitHub API, base env only, no clone or
pip install):
  - GeNet: https://github.com/mrojascarulla/GeNet (``code/network.py``,
    class ``GENEt``). Rojas-Carulla et al., "GeNet: Deep Representations for
    Metagenomics" (arXiv:1901.11015). Distinctive mechanism: raw DNA reads are
    one-hot + trainable-embedding + positional-embedding encoded, then run
    through a *hierarchical residual CNN*: a strided "conv_project" layer
    projects the sequence into the filter space, followed by a stack of
    pre-activation (BN->ReLU->conv) ResNet blocks that double filter width
    every two blocks (average-pool downsampling on width change,
    1x1-conv-projected skip connections), global-average-pooled and dense-
    projected into an encoder state. The paper's key idea -- reused here --
    is that this single encoder state feeds a *cascaded per-taxonomic-level*
    softmax head stack (superkingdom -> phylum -> ... -> species), where each
    level's logits are added to a ReLU-projected copy of the previous level's
    logits (``connect_softmax``), letting coarse-rank predictions inform
    fine-rank predictions through the taxonomy tree rather than training
    independent per-level classifiers.
  - DeepTE: https://github.com/LiLabAtVT/DeepTE (``DeepTE.py``,
    ``scripts/DeepTE_one_hot_rep_kmer.py`` for the k=7 k-mer count-vector
    front end; the shipped repo is inference-only over pretrained ``.h5``
    files, so the hidden-layer architecture is taken from the paper). Yan
    et al., "DeepTE: a computational method for de novo classification of
    transposons with convolutional neural network", Bioinformatics 2020
    (36:4269-4275, doi:10.1093/bioinformatics/btaa519). Distinctive
    mechanism: a DNA sequence is turned into a fixed-length k-mer
    *frequency-count* vector (all 4**7 = 16384 7-mers, not a positional
    embedding), reshaped into a length-16384 single-channel signal and passed
    through three Conv1d(kernel=3)+MaxPool1d(pool=2) hidden layers, then a
    dense classification head. Real DeepTE trains eight separate models
    (order-level plus per-order superfamily-level) sharing this architecture;
    reimplemented here as a single order-level instance of that hidden-layer
    stack, which is the paper's distinctive contribution (a CNN over k-mer
    count histograms, not raw bases).
  - DeepVelo: https://github.com/gersteinlab/DeepVelo (``code/vae.py``,
    functions ``create_encoder``/``create_decoder``; the released
    ``code/Figure2.ipynb`` analysis notebook shows how the trained
    autoencoder is used as the right-hand side of a ``scipy.integrate.
    solve_ivp`` neural-ODE call, ``raw_ae(t, x) = autoencoder(x) *
    scaling_factor``). Chen et al., "DeepVelo: deep learning extends RNA
    velocity to multi-lineage systems with cell-specific kinetic rates"
    (bioRxiv 2022.02.15.480564; Genome Biology 2024 revision). Distinctive
    mechanism: a *denoising* variational autoencoder maps a cell's (noise-
    augmented) gene-expression vector through a funnel of Dense+ReLU layers
    (64->64->64->16, activity-L1-regularized) down to a VAE bottleneck
    (reparameterized ``z = mu + exp(0.5*logvar) * eps``), then a symmetric
    defunnel (16->64->64->gene_dim) decodes a *velocity* vector rather than a
    reconstruction of the input -- i.e. the AE forward pass IS the learned
    ODE vector field ``dx/dt = f_theta(x)`` that is integrated forward in
    time to simulate cell-state trajectories. Reimplemented here as the
    trainable VAE-as-ODE-vector-field module (the ``solve_ivp`` integration
    loop itself is inference-time analysis code, not a trainable component).
  - DeepVirFinder: https://github.com/jessieren/DeepVirFinder
    (``training.py``, the ``get_output``/hidden_layers block building the
    Keras ``Model``). Ren et al., "Identifying viruses from metagenomic data
    using deep learning", Quantitative Biology 2020. Distinctive mechanism: a
    *weight-shared Siamese* Conv1d network is applied independently to the
    one-hot forward-strand and one-hot reverse-complement-strand encodings of
    a variable-length DNA contig (4-channel one-hot, uniform-1/4 for
    ambiguous bases): each strand branch is
    Conv1d(kernel=filter_len)->GlobalMaxPool->Dropout->Dense(relu)->
    Dropout->Dense(1, sigmoid), and the two strand-branch sigmoid outputs are
    *averaged* to give strand-orientation-invariant virus/host probability.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# 1. GeNet: hierarchical residual CNN + taxonomy-cascaded softmax heads.
# ---------------------------------------------------------------------------


class _GeNetResBlock(nn.Module):
    """Pre-activation 1D residual block with optional width-change + downsample."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        """Initialize a pre-activation residual block.

        Parameters
        ----------
        in_channels:
            Input channel width.
        out_channels:
            Output channel width (may differ from ``in_channels``).
        kernel_size:
            Convolution kernel size along the sequence axis.
        """

        super().__init__()
        self.change_width = in_channels != out_channels
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding="same")
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding="same")
        self.project = (
            nn.Conv1d(in_channels, out_channels, 1) if self.change_width else nn.Identity()
        )
        self.downsample = nn.AvgPool1d(2) if self.change_width else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Apply the residual block.

        Parameters
        ----------
        x:
            Input of shape ``(batch, in_channels, length)``.

        Returns
        -------
        Tensor
            Output of shape ``(batch, out_channels, length')``.
        """

        if self.change_width:
            x = self.downsample(x)
        h = F.relu(self.bn1(x))
        h = self.conv1(h)
        h = F.relu(self.bn2(h))
        h = self.conv2(h)
        return self.project(x) + h


class GeNet(nn.Module):
    """GeNet: hierarchical residual CNN with a taxonomy-cascaded softmax head stack."""

    def __init__(
        self,
        vocab_size: int = 5,
        seq_length: int = 64,
        num_filters: int = 16,
        num_resnet_blocks: int = 2,
        fully_connected: int = 32,
        num_units_per_level: tuple[int, ...] = (3, 6, 10),
    ) -> None:
        """Initialize GeNet.

        Parameters
        ----------
        vocab_size:
            Number of DNA symbol classes (A/C/G/T/N).
        seq_length:
            Length of the input read (base-pair positions).
        num_filters:
            Base convolutional filter width; doubles every two resnet blocks.
        num_resnet_blocks:
            Number of filter-doubling resnet-block pairs.
        fully_connected:
            Width of the dense encoder-state projection.
        num_units_per_level:
            Number of taxonomic classes at each cascaded rank (coarse->fine).
        """

        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)
        self.pos_embedding = nn.Embedding(seq_length, vocab_size)
        self.conv_project = nn.Conv1d(vocab_size, num_filters, 3, padding="same")

        blocks = [_GeNetResBlock(num_filters, num_filters, 3) for _ in range(2)]
        width = num_filters
        for _ in range(num_resnet_blocks):
            blocks.append(_GeNetResBlock(width, width * 2, 3))
            blocks.append(_GeNetResBlock(width * 2, width * 2, 3))
            width *= 2
        self.resnet = nn.Sequential(*blocks)
        self.final_bn = nn.BatchNorm1d(width)
        self.encoder_proj = nn.Linear(width, fully_connected)

        self.level_heads = nn.ModuleList(
            [nn.Linear(fully_connected, n) for n in num_units_per_level]
        )
        self.cascade_proj = nn.ModuleList(
            [
                nn.Linear(num_units_per_level[i - 1], num_units_per_level[i])
                for i in range(1, len(num_units_per_level))
            ]
        )

    def forward(self, x: Tensor) -> list[Tensor]:
        """Classify a batch of tokenized DNA reads at every cascaded taxonomic rank.

        Parameters
        ----------
        x:
            Integer token tensor of shape ``(batch, seq_length)``.

        Returns
        -------
        list[Tensor]
            One logits tensor per taxonomic rank, coarse to fine.
        """

        one_hot = F.one_hot(x, num_classes=self.embedding.num_embeddings).float()
        positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        h = one_hot + self.embedding(x) + self.pos_embedding(positions)
        h = h.transpose(1, 2)

        h = self.conv_project(h)
        h = self.resnet(h)
        h = F.relu(self.final_bn(h))
        pooled = h.mean(dim=-1)
        encoder_state = F.relu(self.encoder_proj(pooled))

        logits = [self.level_heads[0](encoder_state)]
        for level_idx in range(1, len(self.level_heads)):
            base = self.level_heads[level_idx](encoder_state)
            cascade = F.relu(self.cascade_proj[level_idx - 1](logits[-1]))
            logits.append(base + cascade)
        return logits


def build_genet() -> nn.Module:
    """Build a compact random-init GeNet hierarchical taxonomic classifier."""

    return GeNet(
        vocab_size=5,
        seq_length=64,
        num_filters=16,
        num_resnet_blocks=2,
        fully_connected=32,
        num_units_per_level=(3, 6, 10),
    ).eval()


def example_input_genet() -> Tensor:
    """Return a batch of tokenized DNA reads for GeNet."""

    return torch.randint(0, 5, (2, 64))


# ---------------------------------------------------------------------------
# 2. DeepTE: CNN over k-mer frequency-count vectors.
# ---------------------------------------------------------------------------


class DeepTE(nn.Module):
    """DeepTE: 3-layer Conv1d + max-pool stack over a k-mer count histogram."""

    def __init__(self, kmer_vocab: int = 4**7, num_classes: int = 7, channels: int = 16) -> None:
        """Initialize DeepTE.

        Parameters
        ----------
        kmer_vocab:
            Length of the k-mer count-frequency input vector (``4**k``).
        num_classes:
            Number of transposable-element order classes.
        channels:
            Convolutional channel width shared across the three hidden layers.
        """

        super().__init__()
        self.kmer_vocab = kmer_vocab
        self.hidden = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        reduced_len = kmer_vocab // 8
        self.classifier = nn.Linear(channels * reduced_len, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Classify a batch of k-mer count vectors into TE order classes.

        Parameters
        ----------
        x:
            K-mer frequency-count vector of shape ``(batch, kmer_vocab)``.

        Returns
        -------
        Tensor
            Class logits of shape ``(batch, num_classes)``.
        """

        h = self.hidden(x.unsqueeze(1))
        return self.classifier(h.flatten(1))


def build_deepte() -> nn.Module:
    """Build a compact random-init DeepTE order-level classifier.

    Uses a reduced ``k=6`` (4**6 = 4096) k-mer vocabulary so the traced
    module stays small; the real DeepTE default is ``k=7`` (16384-dim), a
    pure hyperparameter change of the identical hidden-layer stack.
    """

    return DeepTE(kmer_vocab=4**6, num_classes=7, channels=16).eval()


def example_input_deepte() -> Tensor:
    """Return a batch of k-mer frequency-count vectors for DeepTE."""

    return torch.rand(2, 4**6)


# ---------------------------------------------------------------------------
# 3. DeepVelo: denoising VAE used as a neural-ODE velocity vector field.
# ---------------------------------------------------------------------------


class DeepVeloVAE(nn.Module):
    """DeepVelo: denoising VAE whose decoder output is a transcriptomic velocity field."""

    def __init__(self, gene_dim: int = 64, latent_dim: int = 16) -> None:
        """Initialize the DeepVelo denoising VAE velocity-field network.

        Parameters
        ----------
        gene_dim:
            Number of (velocity) genes in the expression vector.
        latent_dim:
            VAE bottleneck dimensionality.
        """

        super().__init__()
        self.encoder_hidden = nn.Sequential(
            nn.Linear(gene_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(latent_dim, latent_dim)
        self.logvar_head = nn.Linear(latent_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, gene_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Map a (noisy) gene-expression state to a predicted RNA-velocity vector.

        Parameters
        ----------
        x:
            Gene-expression vector of shape ``(batch, gene_dim)``; this acts
            as the state ``x(t)`` in the implied neural ODE ``dx/dt =
            f_theta(x)`` when this module is called from an external solver.

        Returns
        -------
        Tensor
            Predicted velocity vector of shape ``(batch, gene_dim)``.
        """

        h = self.encoder_hidden(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return self.decoder(z)


def build_deepvelo() -> nn.Module:
    """Build a compact random-init DeepVelo denoising VAE velocity field."""

    return DeepVeloVAE(gene_dim=64, latent_dim=16).eval()


def example_input_deepvelo() -> Tensor:
    """Return a batch of gene-expression vectors for DeepVelo."""

    return torch.randn(4, 64)


# ---------------------------------------------------------------------------
# 4. DeepVirFinder: weight-shared Siamese Conv1d over fwd/revcomp DNA strands.
# ---------------------------------------------------------------------------


class DeepVirFinder(nn.Module):
    """DeepVirFinder: Siamese Conv1d branch shared across fwd and revcomp strands."""

    def __init__(
        self, channel_num: int = 4, num_filters: int = 32, filter_len: int = 10, dense: int = 32
    ) -> None:
        """Initialize DeepVirFinder.

        Parameters
        ----------
        channel_num:
            One-hot nucleotide channel count (A/C/G/T).
        num_filters:
            Number of Conv1d filters in the shared branch.
        filter_len:
            Conv1d kernel size (motif length) in the shared branch.
        dense:
            Width of the shared branch's hidden dense layer.
        """

        super().__init__()
        self.conv = nn.Conv1d(channel_num, num_filters, filter_len)
        self.pool_dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(num_filters, dense)
        self.dense_dropout = nn.Dropout(0.1)
        self.out = nn.Linear(dense, 1)

    def _branch(self, x: Tensor) -> Tensor:
        """Run one weight-shared strand branch.

        Parameters
        ----------
        x:
            One-hot strand encoding of shape ``(batch, channel_num, length)``.

        Returns
        -------
        Tensor
            Branch sigmoid probability of shape ``(batch, 1)``.
        """

        h = F.relu(self.conv(x))
        h = h.amax(dim=-1)
        h = self.pool_dropout(h)
        h = F.relu(self.dense(h))
        h = self.dense_dropout(h)
        return torch.sigmoid(self.out(h))

    def forward(self, forward_strand: Tensor, reverse_strand: Tensor) -> Tensor:
        """Average the shared-branch prediction over forward and reverse-complement strands.

        Parameters
        ----------
        forward_strand:
            One-hot forward-strand encoding, shape ``(batch, channel_num, length)``.
        reverse_strand:
            One-hot reverse-complement-strand encoding, same shape.

        Returns
        -------
        Tensor
            Strand-orientation-invariant virus probability, shape ``(batch, 1)``.
        """

        forward_prob = self._branch(forward_strand)
        reverse_prob = self._branch(reverse_strand)
        return (forward_prob + reverse_prob) / 2


def build_deepvirfinder() -> nn.Module:
    """Build a compact random-init DeepVirFinder siamese virus-contig classifier."""

    return DeepVirFinder(channel_num=4, num_filters=32, filter_len=10, dense=32).eval()


def example_input_deepvirfinder() -> tuple[Tensor, Tensor]:
    """Return (forward_strand, reverse_strand) one-hot DNA encodings for DeepVirFinder."""

    forward_strand = torch.rand(2, 4, 300)
    reverse_strand = torch.rand(2, 4, 300)
    return forward_strand, reverse_strand


MENAGERIE_ENTRIES = [
    ("GeNet", "build_genet", "example_input_genet", "2019", "BIO"),
    ("DeepTE", "build_deepte", "example_input_deepte", "2020", "BIO"),
    ("DeepVelo", "build_deepvelo", "example_input_deepvelo", "2024", "BIO"),
    ("DeepVirFinder", "build_deepvirfinder", "example_input_deepvirfinder", "2020", "BIO"),
]
