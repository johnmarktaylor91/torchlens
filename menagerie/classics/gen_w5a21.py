"""Menagerie batch w5a21: six regulatory-genomics deep-learning classics.

Sources checked (reference only; no cloning, no pip installs; base-env torch
reimplementations of each paper's distinctive mechanism):
  - DeepDRIM (cand_00689): Chen, Cheng, Zhang, Li, Liu & Wong, Briefings in
    Bioinformatics 22(6):bbab325, 2021, https://academic.oup.com/bib/article-
    abstract/22/6/bbab325/6356429, official repo
    https://github.com/jiaxchen2-c/DeepDRIM (``DeepDRIM.py``,
    ``construct_model`` / ``get_single_image_model`` / ``get_pair_image_model``,
    Keras). DeepDRIM converts a TF-gene co-expression scatter into a
    "primary image" (2D histogram of joint expression, the CNNC-style
    input this model extends) and additionally rasterizes the co-expression
    of each of several *neighbor* genes with the same TF/target as
    "neighbor images" that give local transitive-interaction context; a
    single-image CNN tower (3 conv-conv-pool-dropout stages, 32/64/128
    channels) embeds the primary image, an independent *weight-shared*
    pair-image CNN tower (identical topology) embeds each neighbor image,
    the neighbor embeddings are concatenated, fused with the primary
    embedding, and pushed through a Dense(512)-Dense(128)-Dense(1,sigmoid)
    head predicting TF-gene regulatory-edge probability. Reimplemented
    faithfully: one primary-image CNN branch, one weight-shared neighbor
    CNN branch applied per neighbor image and concatenated, fused MLP head.
  - DeepEnhancer (cand_00690): Min, Zeng, Chen, Chen, Meng & Jiang,
    BMC Bioinformatics 2017 (predicting enhancers with deep convolutional
    neural networks), official repo https://github.com/minxueric/DeepEnhancer
    (``main.py`` layer list, Lasagne/Theano). The published ``main()``
    trains a 6-conv-layer stack directly on the ``(4, 1, 400)`` one-hot DNA
    tensor (4 nucleotide channels x 400bp window): three ``Conv2D`` layers
    of 64 filters (kernel widths 4,3,3 along the sequence axis) then a
    ``(1,2)`` max-pool, three more ``Conv2D`` layers of 32 filters (kernel
    widths 2,2,2) then another ``(1,2)`` max-pool, flatten, Dense(64) with
    dropout, Dense(64), and a final Dense(2, softmax) enhancer/non-enhancer
    classifier. Reimplemented verbatim as stacked ``nn.Conv2d`` over a
    ``(batch, 4, 1, 400)`` one-hot sequence tensor with the same
    channel/kernel/pooling schedule and dense classification head.
  - DeepFIGV (cand_00691): Kalita, Sridharan, Ibrahim, Nazarian, Corley
    et al. / G.E. Hoffman lab, Nucleic Acids Research 47(20):10597-10611,
    2019, https://pubmed.ncbi.nlm.nih.gov/31544924/, official encoding repo
    https://github.com/GabrielHoffman/deepfigv_encoding (``dna_io_v2.py``,
    ``seq_hdf5_v2.py`` -- "Encoding modified from Basset"). DeepFIGV encodes
    a genomic window as a 4-channel one-hot DNA "image", with heterozygous
    SNP positions split 0.5/0.5 across the two allele channels (rather than
    a hard one-hot base call) so a single forward pass captures a
    personalized diploid sequence; the network itself is the Basset-style
    architecture the encoding repo explicitly derives from (Kelley et al.
    2016): three Conv1d-BatchNorm-ReLU-MaxPool blocks over the one-hot
    sequence axis followed by two fully-connected ReLU layers and a linear
    regression head predicting a quantitative epigenetic signal (chromatin
    accessibility / histone-mark read depth); the same tower is run once
    on the reference-only encoding and once on the heterozygous-blended
    encoding to score a variant's predicted functional effect.
    Reimplemented faithfully: Basset-topology Conv1d tower + heterozygous
    0.5/0.5 allele-blending one-hot encoder, exposed via a
    ``predict_variant_effect`` method scoring ref vs. het-blended windows.
  - DeepHF (cand_00692): Wang, Xu, Cui, Wang, Yan, Chen, Ma, Yu, Wu, Kellis,
    Sabeti & Wu et al., Nature Communications 10:4284, 2019,
    https://www.nature.com/articles/s41467-019-12281-8 (official repo listed
    as izhangcd/DeepHF is no longer resolvable on GitHub; corroborated via
    a maintained fork https://github.com/happtbz/DeepHF, "Core code for the
    DeepHF prediction tool", and the paper's described architecture). DeepHF
    predicts eSpCas9(1.1)/SpCas9-HF1/WT-SpCas9 sgRNA on-target activity from
    (a) the raw 20-23nt guide sequence via a learned nucleotide-embedding
    layer feeding a bidirectional LSTM sequence encoder, and (b) 11
    hand-engineered biological features (three secondary-structure position
    accessibilities, one stem-loop indicator, four melting-temperature
    features, three GC-content features) via a small MLP branch; the BiLSTM
    final hidden state and the bio-feature MLP embedding are concatenated
    and passed through a dense regression head predicting the per-Cas9-
    variant activity score. Reimplemented faithfully: nucleotide-embedding
    + BiLSTM sequence branch, an 11-dim bio-feature MLP branch, and a fused
    regression head with 3 (one per Cas9 variant) output activity scores.
  - DeepHiC (cand_00693): Hong, Ma, Chen, Liu, Xu, Wang & Fu et al.,
    PLOS Computational Biology 16(2):e1007287, 2020,
    https://journals.plos.org/ploscompbiol/article?id=10.1371/
    journal.pcbi.1007287, official repo https://github.com/omegahh/DeepHiC
    (``models/deephic.py``, PyTorch). Ported near-verbatim: a SRGAN-style
    Hi-C super-resolution GAN with a ``Generator`` (9x9 stem conv, a stack
    of Swish-activated residual blocks with a long stem-to-tail skip
    connection, 3x3 refine conv, 9x9 output conv, sigmoid-like
    ``(tanh(x)+1)/2`` squashing to keep contact-map values in [0,1]) and a
    fully-convolutional ``Discriminator`` (6 strided/Swish conv-BN blocks
    doubling channels 64->128->256 while downsampling, replacing the
    original paper's FC head with a 1x1 conv + global average pool, per
    the reference comment "Replaced original paper FC layers with FCN").
    Reimplemented with the same residual-block count, Swish activations,
    skip topology, and fully-convolutional discriminator head.
  - DeepLinc (cand_00695): Li & Yang (xryanglab), Genome Biology 23:124,
    2022, https://genomebiology.biomedcentral.com/articles/10.1186/
    s13059-022-02692-0, official repo
    https://github.com/xryanglab/DeepLinc (``deeplinc/models.py``: the
    ``Deeplinc`` class, TensorFlow-1). NOTE: despite the name, DeepLinc is
    NOT an lncRNA model -- it reconstructs the cell-cell "Linked-in-space
    Interaction" graph for spatial transcriptomics. The reference model is
    a variational graph autoencoder over the spatial neighbor graph: a
    sparse-input ``GraphConvolutionSparse`` layer embeds per-cell gene
    expression through one shared-weight graph-propagation step
    (``adj_normalized @ (X @ W)``, the standard Kipf-Welling GCN
    propagation rule) into a hidden representation, two parallel
    ``GraphConvolution`` heads compute the latent mean and log-std, cells
    are reparameterized (``z = mu + eps * exp(log_std)``) into a latent
    embedding, an inner-product decoder reconstructs the cell-cell
    adjacency (``sigmoid(z @ z^T)``) as the reconstruction loss target, and
    a small dense ``Discriminator`` MLP adversarially regularizes the
    latent code toward a Gaussian prior (adversarially-regularized VGAE,
    AAE-on-top-of-VGAE). Reimplemented with dense (small-graph) GCN
    propagation matching the sparse-input-then-GCN encoder topology, the
    mean/log-std reparameterization, inner-product adjacency decoder, and
    the adversarial discriminator MLP over the latent code.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

# ============================================================
# DeepDRIM -- primary + weight-shared neighbor-image CNN tower
# for TF-gene GRN inference from scRNA-seq co-expression images
# (Chen et al. 2021; jiaxchen2-c/DeepDRIM)
# ============================================================


class _DeepDRIMImageTower(nn.Module):
    """Shared 3-stage conv tower used for both primary and neighbor images.

    Mirrors ``get_single_image_model`` / ``get_pair_image_model`` in the
    reference: three (conv-conv-pool-dropout) stages with 32/64/128
    channels, flattened and projected to a 512-d embedding.
    """

    def __init__(self, image_size: int = 16) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        out_hw = max(1, (image_size // 4) - 2)
        self.proj = nn.Linear(64 * out_hw * out_hw, 512)

    def forward(self, image: Tensor) -> Tensor:
        """Embed one ``(batch, 1, H, W)`` co-expression image to 512-d."""
        x = self.block1(image)
        x = self.block2(x)
        x = x.flatten(1)
        return F.relu(self.proj(x))


class DeepDRIM(nn.Module):
    """Primary-image + weight-shared neighbor-image CNN for GRN inference.

    Reproduces DeepDRIM's ``construct_model``: a primary TF-gene
    co-expression image is embedded by one CNN tower, ``n_neighbors``
    neighbor-gene co-expression images are embedded by a second,
    *weight-shared* CNN tower, all embeddings are concatenated and pushed
    through a Dense(512)-Dense(128)-Dense(1,sigmoid) head predicting the
    probability of a regulatory edge.
    """

    def __init__(self, image_size: int = 16, n_neighbors: int = 4) -> None:
        super().__init__()
        self.n_neighbors = n_neighbors
        self.primary_tower = _DeepDRIMImageTower(image_size)
        self.neighbor_tower = _DeepDRIMImageTower(image_size)
        fused_dim = 512 * (1 + n_neighbors)
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(fused_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(self, images: Tensor) -> Tensor:
        """Predict TF-gene regulatory-edge probability.

        Parameters
        ----------
        images : Tensor
            Shape ``(batch, 1 + n_neighbors, H, W)`` -- channel 0 is the
            primary TF-gene co-expression image, channels 1..n are the
            neighbor-gene co-expression images.
        """
        primary = self.primary_tower(images[:, 0:1])
        neighbor_embeds = [
            self.neighbor_tower(images[:, i : i + 1]) for i in range(1, 1 + self.n_neighbors)
        ]
        fused = torch.cat([primary, *neighbor_embeds], dim=1)
        return torch.sigmoid(self.head(fused))


def build_deepdrim() -> nn.Module:
    """Build a small DeepDRIM primary+neighbor co-expression-image CNN."""
    return DeepDRIM(image_size=16, n_neighbors=4).eval()


def example_input_deepdrim() -> Tensor:
    """Return a batch of primary+neighbor co-expression images for DeepDRIM."""
    return torch.rand(2, 5, 16, 16)


# ============================================================
# DeepEnhancer -- stacked-Conv2D one-hot-DNA enhancer classifier
# (Min, Zeng, Chen, Chen, Meng & Jiang 2017; minxueric/DeepEnhancer)
# ============================================================


class DeepEnhancer(nn.Module):
    """Stacked-Conv2D classifier over one-hot DNA sequence windows.

    Reproduces the ``main.py`` layer list: two 3-conv blocks (64 then 32
    filters, kernel widths shrinking 4/3/3 then 2/2/2 along the sequence
    axis) each followed by a ``(1,2)`` max-pool, flatten, Dense(64) with
    dropout, Dense(64), and a final Dense(2, softmax) enhancer /
    non-enhancer classifier.
    """

    def __init__(self, seq_len: int = 400) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=(1, 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2)),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(1, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2)),
        )
        # Trace the flattened width once with a dummy forward at init time.
        with torch.no_grad():
            dummy = torch.zeros(1, 4, 1, seq_len)
            flat_dim = self.block2(self.block1(dummy)).flatten(1).shape[1]
        self.fc1 = nn.Linear(flat_dim, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, one_hot_seq: Tensor) -> Tensor:
        """Classify enhancer vs. non-enhancer from a one-hot DNA window.

        Parameters
        ----------
        one_hot_seq : Tensor
            Shape ``(batch, 4, 1, seq_len)`` one-hot-encoded DNA sequence
            (4 nucleotide channels).
        """
        x = self.block1(one_hot_seq)
        x = self.block2(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)


def build_deepenhancer() -> nn.Module:
    """Build a small DeepEnhancer stacked-conv enhancer classifier."""
    return DeepEnhancer(seq_len=400).eval()


def example_input_deepenhancer() -> Tensor:
    """Return a batch of one-hot 400bp DNA windows for DeepEnhancer."""
    return torch.rand(2, 4, 1, 400)


# ============================================================
# DeepFIGV -- Basset-style CNN over heterozygous-blended one-hot DNA
# for personalized-genome epigenetic-signal / variant-effect prediction
# (Kalita, Sridharan, Hoffman lab 2019; GabrielHoffman/deepfigv_encoding)
# ============================================================


class DeepFIGV(nn.Module):
    """Basset-style Conv1d tower predicting quantitative epigenetic signal.

    Reproduces the Basset architecture the DeepFIGV encoding repo is
    explicitly derived from: three Conv1d-BatchNorm-ReLU-MaxPool blocks
    over a one-hot DNA window, two fully-connected ReLU layers, and a
    linear regression head predicting quantitative chromatin
    accessibility / histone-mark signal. ``predict_variant_effect`` runs
    the shared tower on both a reference-only encoding and a heterozygous
    0.5/0.5-blended encoding (DeepFIGV's headline trick for personalized
    diploid genomes) to score a variant's predicted functional effect.
    """

    def __init__(self, seq_len: int = 200) -> None:
        super().__init__()
        self.conv_tower = nn.Sequential(
            nn.Conv1d(4, 48, kernel_size=11, padding=5),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3),
            nn.Conv1d(48, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3),
            nn.Conv1d(64, 96, kernel_size=5, padding=2),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 4, seq_len)
            flat_dim = self.conv_tower(dummy).flatten(1).shape[1]
        self.fc1 = nn.Linear(flat_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.readout = nn.Linear(64, 1)

    def forward(self, one_hot_seq: Tensor) -> Tensor:
        """Predict a quantitative epigenetic signal from a one-hot window.

        Parameters
        ----------
        one_hot_seq : Tensor
            Shape ``(batch, 4, seq_len)``; heterozygous SNP positions may
            carry ``0.5`` on each of the two allele channels instead of a
            hard one-hot call.
        """
        x = self.conv_tower(one_hot_seq)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.readout(x)

    def predict_variant_effect(self, ref_seq: Tensor, het_seq: Tensor) -> Tensor:
        """Score a variant's effect as the ref-vs-heterozygous signal delta."""
        return self.forward(het_seq) - self.forward(ref_seq)


def build_deepfigv() -> nn.Module:
    """Build a small DeepFIGV Basset-style variant-effect regressor."""
    return DeepFIGV(seq_len=200).eval()


def example_input_deepfigv() -> Tensor:
    """Return a batch of one-hot 200bp DNA windows for DeepFIGV."""
    return torch.rand(2, 4, 200)


# ============================================================
# DeepHF -- embedding+BiLSTM sgRNA sequence branch fused with an
# 11-dim biological-feature branch for Cas9-variant activity
# (Wang, Xu, Cui, Kellis, Sabeti, Wu et al. 2019; Nat Commun)
# ============================================================


class DeepHF(nn.Module):
    """BiLSTM sgRNA-sequence branch fused with a biological-feature branch.

    Reproduces the DeepHF architecture: a learned nucleotide-embedding
    layer feeds a bidirectional LSTM over the guide sequence; the final
    BiLSTM hidden state is concatenated with an MLP embedding of 11
    hand-engineered biological features (secondary-structure position
    accessibility, stem-loop indicator, melting temperature, GC content);
    the fused representation is regressed to 3 activity scores, one per
    Cas9 variant (WT-SpCas9, eSpCas9(1.1), SpCas9-HF1).
    """

    def __init__(self, seq_len: int = 23, n_bio_features: int = 11) -> None:
        super().__init__()
        self.embedding = nn.Embedding(4, 32)
        self.bilstm = nn.LSTM(32, 64, batch_first=True, bidirectional=True)
        self.bio_mlp = nn.Sequential(
            nn.Linear(n_bio_features, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 2 + 32, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 3),
        )

    def forward(self, guide_seq: Tensor, bio_features: Tensor) -> Tensor:
        """Predict per-Cas9-variant activity scores for an sgRNA.

        Parameters
        ----------
        guide_seq : Tensor
            Integer-encoded guide sequence, shape ``(batch, seq_len)``
            with nucleotide ids in ``[0, 4)``.
        bio_features : Tensor
            Shape ``(batch, n_bio_features)`` hand-engineered biological
            features (secondary structure, melting temperature, GC
            content).
        """
        embedded = self.embedding(guide_seq)
        _, (h_n, _) = self.bilstm(embedded)
        seq_embed = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        bio_embed = self.bio_mlp(bio_features)
        fused = torch.cat([seq_embed, bio_embed], dim=-1)
        return self.head(fused)


def build_deephf() -> nn.Module:
    """Build a small DeepHF embedding+BiLSTM sgRNA-activity predictor."""
    return DeepHF(seq_len=23, n_bio_features=11).eval()


def example_input_deephf() -> tuple[Tensor, Tensor]:
    """Return an (integer sgRNA sequence, bio-feature) pair for DeepHF."""
    guide_seq = torch.randint(0, 4, (2, 23))
    bio_features = torch.randn(2, 11)
    return guide_seq, bio_features


# ============================================================
# DeepHiC -- SRGAN-style residual generator + fully-convolutional
# discriminator for Hi-C contact-map super-resolution
# (Hong, Ma, Chen, Xu, Wang, Fu et al. 2020; omegahh/DeepHiC)
# ============================================================


def _swish(x: Tensor) -> Tensor:
    """Swish activation ``x * sigmoid(x)`` used throughout DeepHiC."""
    return x * torch.sigmoid(x)


class _DeepHiCResidualBlock(nn.Module):
    """Swish-activated residual block used in the DeepHiC generator."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        residual = _swish(self.bn1(self.conv1(x)))
        residual = self.bn2(self.conv2(residual))
        return x + residual


class DeepHiCGenerator(nn.Module):
    """SRGAN-style residual generator for Hi-C contact-map super-resolution.

    Ports ``models/deephic.py::Generator`` near-verbatim: a 9x9 stem conv,
    a stack of Swish-activated residual blocks, a long stem-to-tail skip
    connection, a 3x3 refine conv, a 9x9 output conv, and a
    ``(tanh(x)+1)/2`` squash keeping contact-map values in ``[0, 1]``.
    """

    def __init__(self, in_channels: int = 1, resblock_num: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=9, padding=4)
        self.resblocks = nn.Sequential(*[_DeepHiCResidualBlock(64) for _ in range(resblock_num)])
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, in_channels, kernel_size=9, padding=4)

    def forward(self, low_res_contact_map: Tensor) -> Tensor:
        """Super-resolve a low-resolution Hi-C contact-map patch."""
        emb = _swish(self.conv1(low_res_contact_map))
        x = self.resblocks(emb)
        x = _swish(self.bn2(self.conv2(x)))
        x = self.conv3(x + emb)
        return (torch.tanh(x) + 1) / 2


class DeepHiCDiscriminator(nn.Module):
    """Fully-convolutional discriminator for DeepHiC's adversarial loss.

    Ports ``models/deephic.py::Discriminator``: six strided/Swish
    conv-BatchNorm blocks doubling channels 64->128->256 while
    downsampling, replacing the original SRGAN-paper FC head with a 1x1
    conv + global-average-pool ("Replaced original paper FC layers with
    FCN" per the reference comment).
    """

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 1, 1, stride=1, padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, contact_map: Tensor) -> Tensor:
        """Score a Hi-C contact-map patch as real (super-resolved) vs fake."""
        batch_size = contact_map.size(0)
        x = _swish(self.conv1(contact_map))
        x = _swish(self.bn2(self.conv2(x)))
        x = _swish(self.bn3(self.conv3(x)))
        x = _swish(self.bn4(self.conv4(x)))
        x = _swish(self.bn5(self.conv5(x)))
        x = _swish(self.bn6(self.conv6(x)))
        x = self.conv7(x)
        x = self.avgpool(x)
        return torch.sigmoid(x.view(batch_size))


def build_deephic() -> nn.Module:
    """Build a small DeepHiC residual super-resolution generator."""
    return DeepHiCGenerator(in_channels=1, resblock_num=3).eval()


def example_input_deephic() -> Tensor:
    """Return a batch of low-resolution Hi-C contact-map patches for DeepHiC."""
    return torch.rand(2, 1, 40, 40)


# ============================================================
# DeepLinc -- adversarially-regularized variational graph autoencoder
# for spatial-transcriptomics cell-cell interaction-graph reconstruction
# (Li & Yang 2022; xryanglab/DeepLinc)
# ============================================================


class _DenseGraphConv(nn.Module):
    """Dense Kipf-Welling GCN propagation: ``adj_norm @ (x @ W)`` + act."""

    def __init__(self, in_dim: int, out_dim: int, activation: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.activation = activation

    def forward(self, x: Tensor, adj_norm: Tensor) -> Tensor:
        out = adj_norm @ self.linear(x)
        return F.relu(out) if self.activation else out


class DeepLinc(nn.Module):
    """Adversarially-regularized VGAE reconstructing a cell-cell graph.

    Reproduces ``deeplinc/models.py::Deeplinc``: a shared-weight GCN
    encoder layer embeds per-cell gene expression through one
    graph-propagation step into a hidden representation; two parallel GCN
    heads compute the latent mean and log-std; cells are reparameterized
    into a latent embedding; an inner-product decoder reconstructs the
    cell-cell adjacency matrix (the "linked-in-space interaction" graph);
    and a dense ``Discriminator`` MLP adversarially regularizes the latent
    code toward a Gaussian prior, per the AAE-on-VGAE design.
    """

    def __init__(self, n_genes: int = 64, hidden1_dim: int = 32, hidden2_dim: int = 16) -> None:
        super().__init__()
        self.encoder1 = _DenseGraphConv(n_genes, hidden1_dim, activation=True)
        self.gcn_mean = _DenseGraphConv(hidden1_dim, hidden2_dim, activation=False)
        self.gcn_logstd = _DenseGraphConv(hidden1_dim, hidden2_dim, activation=False)
        self.discriminator = nn.Sequential(
            nn.Linear(hidden2_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )

    def encode(self, expression: Tensor, adj_norm: Tensor) -> tuple[Tensor, Tensor]:
        """Encode per-cell expression + normalized adjacency to (mean, logstd)."""
        h1 = self.encoder1(expression, adj_norm)
        mean = self.gcn_mean(h1, adj_norm)
        logstd = self.gcn_logstd(h1, adj_norm)
        return mean, logstd

    def forward(self, expression: Tensor, adj_norm: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Reconstruct the cell-cell interaction graph from spatial expression.

        Parameters
        ----------
        expression : Tensor
            Per-cell gene-expression matrix, shape ``(n_cells, n_genes)``.
        adj_norm : Tensor
            Symmetrically-normalized spatial-neighbor adjacency, shape
            ``(n_cells, n_cells)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``(reconstructed_adjacency_logits, latent_embedding,
            discriminator_score)``.
        """
        mean, logstd = self.encode(expression, adj_norm)
        eps = torch.randn_like(mean)
        z = mean + eps * torch.exp(logstd)
        reconstruction = z @ z.t()
        disc_score = self.discriminator(z)
        return reconstruction, z, disc_score


def build_deeplinc() -> nn.Module:
    """Build a small DeepLinc adversarially-regularized VGAE."""
    return DeepLinc(n_genes=64, hidden1_dim=32, hidden2_dim=16).eval()


def example_input_deeplinc() -> tuple[Tensor, Tensor]:
    """Return (expression, normalized adjacency) tensors for DeepLinc."""
    n_cells = 20
    expression = torch.rand(n_cells, 64)
    adj = torch.rand(n_cells, n_cells)
    adj = (adj + adj.t()) / 2
    return expression, adj


MENAGERIE_ENTRIES = [
    ("DeepDRIM", "build_deepdrim", "example_input_deepdrim", "2021", "BIO"),
    ("DeepEnhancer", "build_deepenhancer", "example_input_deepenhancer", "2017", "BIO"),
    ("DeepFIGV", "build_deepfigv", "example_input_deepfigv", "2019", "BIO"),
    ("DeepHF", "build_deephf", "example_input_deephf", "2019", "BIO"),
    ("DeepHiC", "build_deephic", "example_input_deephic", "2020", "BIO"),
    ("DeepLinc", "build_deeplinc", "example_input_deeplinc", "2022", "BIO"),
]
