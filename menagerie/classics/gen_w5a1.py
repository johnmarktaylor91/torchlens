"""Wave w5a1: six computational-biology architecture classics.

Sources checked (repo trees + primary source files fetched via GitHub/GitLab
API, paper metadata from the build queue):

- DeepSignalP (SignalP 6.0) -- fteufel/signalp-6.0
  (``src/signalp6/models/bert_crf.py``), Teufel et al., Nature Biotechnology
  2022. A protein language-model backbone (a BERT-family transformer encoder
  over the amino-acid sequence, here a small from-scratch encoder standing in
  for the pretrained ProtBERT) whose per-position hidden states are linearly
  projected to per-residue tag emissions and decoded by a linear-chain
  Conditional Random Field (CRF) into a signal-peptide segmentation, while a
  pooled sequence representation feeds a second head that predicts the global
  signal-peptide type label -- the joint LM-encoder-into-CRF-decoder with an
  auxiliary global classification head is the distinctive mechanism.
- DeepSol -- sameerkhurana10/DSOL_rv0.2 (``scripts/dsol/Models_dsol1.py``),
  Khurana et al., Bioinformatics 2018. An amino-acid embedding followed by
  three sequential Conv1D blocks (each block runs one or more parallel
  Conv1D kernels of different widths over the same input, max-pools each,
  and concatenates the pooled branches before the next block) and a small
  dense head with a sigmoid solubility score -- the per-block multi-kernel
  parallel-convolution-then-concatenate motif is the distinctive mechanism.
- DeepSTARR -- bernardo-de-almeida/DeepSTARR
  (``DeepSTARR/DeepSTARR_training.ipynb``, function ``DeepSTARR``), de
  Almeida et al., Nature Genetics 2022. A 4-layer Conv1D+BatchNorm+ReLU+
  MaxPool tower over one-hot DNA sequence, followed by two dense+BN+ReLU+
  dropout layers into a shared bottleneck, and finally two parallel linear
  regression heads (developmental and housekeeping enhancer activity) --
  the shared-trunk-with-two-regression-heads multi-task design is the
  distinctive mechanism.
- DeepTCR -- sidhomj/DeepTCR (``DeepTCR/functions/Layers.py``, functions
  ``Convolutional_Features`` / ``Conv_Model``), Sidhom et al., Nature
  Communications 2021. Separate small 1D-convolutional towers (here modeled
  as 1D CNNs with global max-pooling, mirroring the paper's per-position
  max-over-length reduction) independently encode the alpha-chain and
  beta-chain CDR3 amino-acid sequences; their pooled features are
  concatenated with learned V/D/J gene-usage embeddings into one joint
  repertoire-level feature vector fed to a classifier head -- the
  dual-chain-conv-plus-gene-embedding fusion is the distinctive mechanism.
- DeepTrio (google/deepvariant, ``docs/deeptrio-details.md`` /
  ``docs/deepvariant-small-model-details.md``), Google, trio-aware extension
  of DeepVariant. A shared CNN classifies pileup images per individual, but
  DeepTrio stacks the child's pileup image together with both parents'
  pileup images along the channel axis and runs one joint CNN trunk before
  three separate per-individual genotype heads, letting the network exploit
  Mendelian-inheritance constraints across the trio jointly rather than
  calling each sample independently -- the channel-stacked trio-joint CNN
  with per-individual output heads is the distinctive mechanism.
- DeLTA -- gitlab.com/dunloplab/delta (``delta/model.py``, functions
  ``unet_seg`` / ``unet_track``), Lugagne et al. / O'Connor et al., PLOS
  Computational Biology 2020 / 2022. Two sequential U-Nets: a 1-channel
  segmentation U-Net produces a per-pixel cell mask from a single
  mother-machine microscopy frame, and a second 4-channel tracking U-Net
  takes the current frame concatenated with a seed mask of one cell of
  interest plus the previous frame and that cell's previous mask, and
  predicts the same cell's (and any daughter's) mask in the current frame --
  the two-U-Net segment-then-track pipeline, with the tracking net
  conditioned on the previous frame/mask via channel concatenation, is the
  distinctive mechanism.

All models are compact, randomly initialized, CPU, forward-only reimplementations
built from scratch in base-env torch/torch.nn -- no cloning, no pip installs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# DeepSignalP (SignalP 6.0): protein-LM encoder -> CRF tagging + global head
# ---------------------------------------------------------------------------


class _TinyProteinEncoder(nn.Module):
    """Small from-scratch transformer encoder standing in for ProtBERT."""

    def __init__(
        self, vocab_size: int = 25, d_model: int = 32, n_layers: int = 2, n_heads: int = 4
    ) -> None:
        """Build a token+positional embedding followed by a few encoder layers."""

        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(512, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Return per-position hidden states, shape (batch, seq_len, d_model)."""

        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = self.token_embed(input_ids) + self.pos_embed(positions).unsqueeze(0)
        return self.encoder(hidden)


class DeepSignalPCRFTagger(nn.Module):
    """Protein-LM encoder with a per-residue CRF-emission head + a global type head."""

    def __init__(
        self,
        vocab_size: int = 25,
        d_model: int = 32,
        num_tags: int = 8,
        num_global_labels: int = 6,
    ) -> None:
        """Set up the LM encoder, per-position CRF emission head, and pooled global head."""

        super().__init__()
        self.encoder = _TinyProteinEncoder(vocab_size, d_model)
        self.to_emissions = nn.Linear(d_model, num_tags)
        # Learned pairwise tag-transition matrix, the CRF's core parameter.
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags) * 0.1)
        self.global_head = nn.Linear(d_model, num_global_labels)

    def forward(self, input_ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return (per-position emissions, transition matrix, global type logits)."""

        hidden = self.encoder(input_ids)
        emissions = self.to_emissions(hidden)
        pooled = hidden.mean(dim=1)
        global_logits = self.global_head(pooled)
        return emissions, self.transitions, global_logits


def build_deepsignalp() -> nn.Module:
    """Build a small DeepSignalP (SignalP 6.0)-style CRF tagger."""

    return DeepSignalPCRFTagger(vocab_size=25, d_model=32, num_tags=8, num_global_labels=6).eval()


def example_input_deepsignalp() -> Tensor:
    """Return a batch of tokenized short protein sequences."""

    return torch.randint(0, 25, (2, 40), dtype=torch.long)


# ---------------------------------------------------------------------------
# DeepSol: embedding -> multi-kernel Conv1D blocks -> dense solubility head
# ---------------------------------------------------------------------------


class DeepSolConvBlock(nn.Module):
    """One DeepSol block: parallel multi-kernel Conv1D branches, pooled and concatenated."""

    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: tuple[int, ...]) -> None:
        """Build one Conv1D branch per kernel size in ``kernel_sizes``."""

        super().__init__()
        self.branches = nn.ModuleList(
            nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run every branch, ReLU + max-pool(2) each, and concatenate along channels."""

        outs = [F.max_pool1d(F.relu(branch(x)), kernel_size=2) for branch in self.branches]
        return torch.cat(outs, dim=1)


class DeepSol(nn.Module):
    """DeepSol: AA embedding, stacked multi-kernel conv blocks, dense sigmoid head."""

    def __init__(self, vocab_size: int = 21, embed_dim: int = 16, num_classes: int = 1) -> None:
        """Build the embedding, three conv blocks, and the final dense classifier."""

        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.block1 = DeepSolConvBlock(embed_dim, 32, kernel_sizes=(3, 5))
        self.block2 = DeepSolConvBlock(64, 32, kernel_sizes=(3,))
        self.block3 = DeepSolConvBlock(32, 16, kernel_sizes=(3,))
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(16, 32)
        self.dropout = nn.Dropout(0.0)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, tokens: Tensor) -> Tensor:
        """Return solubility logits for a batch of tokenized sequences."""

        x = self.embedding(tokens).transpose(1, 2)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc2(x))


def build_deepsol() -> nn.Module:
    """Build a small DeepSol solubility-prediction CNN."""

    return DeepSol(vocab_size=21, embed_dim=16, num_classes=1).eval()


def example_input_deepsol() -> Tensor:
    """Return a batch of tokenized protein sequences of fixed max length."""

    return torch.randint(0, 21, (2, 64), dtype=torch.long)


# ---------------------------------------------------------------------------
# DeepSTARR: 4-block Conv1D tower + shared dense trunk + two regression heads
# ---------------------------------------------------------------------------


class DeepSTARR(nn.Module):
    """DeepSTARR: DNA-sequence CNN with two parallel enhancer-activity heads."""

    def __init__(self, seq_len: int = 249, num_filters: tuple[int, ...] = (32, 16, 16, 16)) -> None:
        """Build the 4-layer conv tower, the dense trunk, and the Dev/Hk output heads."""

        super().__init__()
        kernel_sizes = (7, 3, 5, 3)
        conv_layers = []
        in_channels = 4
        for out_channels, kernel_size in zip(num_filters, kernel_sizes, strict=True):
            conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                )
            )
            in_channels = out_channels
        self.conv_tower = nn.Sequential(*conv_layers)
        flat_len = seq_len // (2 ** len(num_filters))
        flat_dim = in_channels * flat_len
        self.dense_trunk = nn.Sequential(
            nn.Linear(flat_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.0),
        )
        self.dev_head = nn.Linear(64, 1)
        self.hk_head = nn.Linear(64, 1)

    def forward(self, one_hot_seq: Tensor) -> tuple[Tensor, Tensor]:
        """Return (developmental activity, housekeeping activity) regression outputs."""

        x = self.conv_tower(one_hot_seq)
        x = x.flatten(1)
        bottleneck = self.dense_trunk(x)
        return self.dev_head(bottleneck), self.hk_head(bottleneck)


def build_deepstarr() -> nn.Module:
    """Build a small DeepSTARR enhancer-activity CNN."""

    return DeepSTARR(seq_len=249, num_filters=(32, 16, 16, 16)).eval()


def example_input_deepstarr() -> Tensor:
    """Return a batch of one-hot-encoded 249bp DNA sequences, shape (batch, 4, 249)."""

    return F.one_hot(torch.randint(0, 4, (2, 249)), num_classes=4).permute(0, 2, 1).float()


# ---------------------------------------------------------------------------
# DeepTCR: dual-chain CDR3 conv towers + gene-usage embeddings, fused
# ---------------------------------------------------------------------------


class DeepTCRChainTower(nn.Module):
    """Small 1D-conv tower over one CDR3 chain with a global max-pool reduction."""

    def __init__(self, vocab_size: int = 21, embed_dim: int = 16, out_dim: int = 32) -> None:
        """Build the AA embedding and the two-layer conv tower for one chain."""

        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, out_dim, kernel_size=3, padding=1)

    def forward(self, tokens: Tensor) -> Tensor:
        """Return a pooled per-sequence feature vector, shape (batch, out_dim)."""

        x = self.embedding(tokens).transpose(1, 2)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        return x.amax(dim=-1)


class DeepTCR(nn.Module):
    """DeepTCR: dual alpha/beta CDR3 conv towers fused with V/J gene embeddings."""

    def __init__(
        self,
        vocab_size: int = 21,
        n_v_genes: int = 50,
        n_j_genes: int = 13,
        gene_embed_dim: int = 8,
        num_classes: int = 2,
    ) -> None:
        """Build the two chain towers, the gene-usage embeddings, and the classifier."""

        super().__init__()
        self.alpha_tower = DeepTCRChainTower(vocab_size, out_dim=32)
        self.beta_tower = DeepTCRChainTower(vocab_size, out_dim=32)
        self.v_beta_embed = nn.Embedding(n_v_genes, gene_embed_dim)
        self.j_beta_embed = nn.Embedding(n_j_genes, gene_embed_dim)
        fused_dim = 32 + 32 + gene_embed_dim + gene_embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 32), nn.ReLU(), nn.Linear(32, num_classes)
        )

    def forward(
        self, alpha_seq: Tensor, beta_seq: Tensor, v_beta: Tensor, j_beta: Tensor
    ) -> Tensor:
        """Return classification logits for a batch of paired alpha/beta TCR repertoire entries."""

        alpha_feat = self.alpha_tower(alpha_seq)
        beta_feat = self.beta_tower(beta_seq)
        gene_feat = torch.cat([self.v_beta_embed(v_beta), self.j_beta_embed(j_beta)], dim=-1)
        fused = torch.cat([alpha_feat, beta_feat, gene_feat], dim=-1)
        return self.classifier(fused)


def build_deeptcr() -> nn.Module:
    """Build a small DeepTCR dual-chain repertoire classifier."""

    return DeepTCR(
        vocab_size=21, n_v_genes=50, n_j_genes=13, gene_embed_dim=8, num_classes=2
    ).eval()


def example_input_deeptcr() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return (alpha_seq, beta_seq, v_beta_gene, j_beta_gene) for a small batch."""

    batch = 3
    alpha_seq = torch.randint(0, 21, (batch, 20), dtype=torch.long)
    beta_seq = torch.randint(0, 21, (batch, 20), dtype=torch.long)
    v_beta = torch.randint(0, 50, (batch,), dtype=torch.long)
    j_beta = torch.randint(0, 13, (batch,), dtype=torch.long)
    return alpha_seq, beta_seq, v_beta, j_beta


# ---------------------------------------------------------------------------
# DeepTrio: channel-stacked trio pileup CNN with per-individual output heads
# ---------------------------------------------------------------------------


class DeepTrio(nn.Module):
    """DeepTrio: joint CNN trunk over channel-stacked child/parent pileup images."""

    def __init__(self, channels_per_individual: int = 3, num_genotype_classes: int = 3) -> None:
        """Build the shared conv trunk and the three per-individual genotype heads."""

        super().__init__()
        total_channels = channels_per_individual * 3  # child + mother + father
        self.trunk = nn.Sequential(
            nn.Conv2d(total_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.child_head = nn.Linear(64, num_genotype_classes)
        self.mother_head = nn.Linear(64, num_genotype_classes)
        self.father_head = nn.Linear(64, num_genotype_classes)

    def forward(
        self, child: Tensor, mother: Tensor, father: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return per-individual genotype logits given the trio's stacked pileup images."""

        stacked = torch.cat([child, mother, father], dim=1)
        features = self.pool(self.trunk(stacked)).flatten(1)
        return self.child_head(features), self.mother_head(features), self.father_head(features)


def build_deeptrio() -> nn.Module:
    """Build a small DeepTrio channel-stacked trio pileup-image classifier."""

    return DeepTrio(channels_per_individual=3, num_genotype_classes=3).eval()


def example_input_deeptrio() -> tuple[Tensor, Tensor, Tensor]:
    """Return (child, mother, father) pileup-image tensors, each (batch, 3, 32, 32)."""

    batch = 2
    shape = (batch, 3, 32, 32)
    return torch.rand(*shape), torch.rand(*shape), torch.rand(*shape)


# ---------------------------------------------------------------------------
# DeLTA: two sequential U-Nets -- segmentation, then previous-mask-conditioned tracking
# ---------------------------------------------------------------------------


class _UNetBlock(nn.Module):
    """One contracting/expanding level of a small recursive U-Net."""

    def __init__(self, in_channels: int, filters: list[int]) -> None:
        """Build the current level's convs and, if any levels remain, the recursive core."""

        super().__init__()
        out_channels = filters[0]
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.is_bottom = len(filters) == 1
        if not self.is_bottom:
            self.down = nn.MaxPool2d(2)
            self.inner = _UNetBlock(out_channels, filters[1:])
            self.up = nn.Upsample(scale_factor=2, mode="nearest")
            self.conv3 = nn.Conv2d(filters[1], out_channels, kernel_size=3, padding=1)
            self.conv4 = nn.Conv2d(2 * out_channels, out_channels, kernel_size=3, padding=1)
            self.conv5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """Run the contracting conv pair, recurse if not at the bottom, then expand and merge."""

        skip = F.relu(self.conv2(F.relu(self.conv1(x))))
        if self.is_bottom:
            return skip
        down = self.down(skip)
        inner_out = self.inner(down)
        up = F.relu(self.conv3(self.up(inner_out)))
        merged = torch.cat([skip, up], dim=1)
        return F.relu(self.conv5(F.relu(self.conv4(merged))))


class DeLTAUNet(nn.Module):
    """A single small U-Net with a 1x1 output convolution (used for both seg and track)."""

    def __init__(self, in_channels: int, filters: tuple[int, ...] = (16, 32, 64)) -> None:
        """Build the recursive U-Net body and the final 1x1 mask-logit convolution."""

        super().__init__()
        self.body = _UNetBlock(in_channels, list(filters))
        self.out_conv = nn.Conv2d(filters[0], 1, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Return per-pixel mask logits, shape (batch, 1, H, W)."""

        return self.out_conv(self.body(x))


class DeLTAPipeline(nn.Module):
    """DeLTA: segmentation U-Net followed by a previous-mask-conditioned tracking U-Net."""

    def __init__(self) -> None:
        """Build the 1-channel segmentation U-Net and the 4-channel tracking U-Net."""

        super().__init__()
        self.seg_unet = DeLTAUNet(in_channels=1, filters=(16, 32, 64))
        self.track_unet = DeLTAUNet(in_channels=4, filters=(16, 32, 64))

    def forward(
        self, frame: Tensor, prev_frame: Tensor, prev_mask: Tensor, seed_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return (segmentation logits, tracked-cell logits) for the current frame."""

        seg_logits = self.seg_unet(frame)
        track_input = torch.cat([frame, seed_mask, prev_frame, prev_mask], dim=1)
        track_logits = self.track_unet(track_input)
        return seg_logits, track_logits


def build_delta() -> nn.Module:
    """Build a small DeLTA two-U-Net segmentation+tracking pipeline."""

    return DeLTAPipeline().eval()


def example_input_delta() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return (frame, prev_frame, prev_mask, seed_mask), each (batch, 1, 64, 32)."""

    batch = 1
    shape = (batch, 1, 64, 32)
    return torch.rand(*shape), torch.rand(*shape), torch.rand(*shape), torch.rand(*shape)


MENAGERIE_ENTRIES = [
    ("DeepSignalP", "build_deepsignalp", "example_input_deepsignalp", "2022", "BIO"),
    ("DeepSol", "build_deepsol", "example_input_deepsol", "2018", "BIO"),
    ("DeepSTARR", "build_deepstarr", "example_input_deepstarr", "2022", "BIO"),
    ("DeepTCR", "build_deeptcr", "example_input_deeptcr", "2021", "BIO"),
    (
        "DeepTrio",
        "build_deeptrio",
        "example_input_deeptrio",
        "2020",
        "BIO",
    ),
    ("DeLTA", "build_delta", "example_input_delta", "2020", "BIO"),
]
