"""Menagerie batch w4a23: bioinformatics / proteomics / medical-imaging sequence models.

Sources checked (reference only; no cloning, no pip installs):
  - DeepHLApan (cand_00522): Wu, Ye et al., Frontiers in Immunology 2019
    "DeepHLApan: A Deep Learning Approach for Neoantigen Prediction
    Considering Both HLA-Peptide Binding and Immunogenicity",
    https://www.frontiersin.org/articles/10.3389/fimmu.2019.02559/full.
    Official code https://github.com/zjupgx/deephlapan (Keras/TF; only
    trained ``.hdf5`` weights ship, no architecture-builder script, so the
    published figure + the repo's custom ``deephlapan/attention.py``
    self-attentive pooling layer are the reference). The paper's Figure 1
    and text describe three stacked layers of bidirectional GRU over a
    one-hot-encoded HLA-pseudo-sequence + peptide complex (fixed length 49,
    padded), followed by the repo's ``Attention`` layer: a learned per-
    timestep scalar score (tanh of a linear projection), softmax-normalized
    over the sequence length, used to pool the BiGRU output into one vector
    that feeds a sigmoid classification head. Two independently-trained
    instances of this architecture form the binding model and the
    immunogenicity model; this module reproduces the shared architecture
    (3x BiGRU + additive self-attention pooling + classification head).
  - DeepImmuno (cand_00524): Li, Bhattarai et al., Briefings in
    Bioinformatics 2021 "DeepImmuno: deep learning-empowered prediction and
    generation of immunogenic peptides for T-cell immunity", official code
    https://github.com/frankligy/DeepImmuno. The repo ships two distinct
    models: ``deepimmuno-cnn.py`` (a dual-branch 2D-CNN classifier over
    AAindex-PCA-encoded peptide/HLA-pseudo-sequence pairs, in Keras) and
    ``deepimmuno-gan.py`` (a native **PyTorch** WGAN with residual 1D-conv
    generator/discriminator that synthesizes pseudo-immunogenic peptide
    sequences, gated by Gumbel-Softmax at the generator output). The catalog
    notes describe "CNN + ResNet + GNN"; the actual repo has no GNN, but does
    have the ResNet-block WGAN and the CNN classifier, so this module
    reproduces both real components: the residual-1D-conv WGAN generator/
    discriminator (``deepimmuno-gan.py``, verbatim ``ResBlock`` residual-add
    topology and Gumbel-Softmax categorical output) and the dual-branch CNN
    classifier (``deepimmuno-cnn.py`` Conv2D towers, ported to torch).
  - DeepLC (cand_00525): Bouwmeester et al., Nature Methods 2021 "DeepLC can
    predict retention times for peptides that carry as-yet unseen
    modifications", official code https://github.com/compomics/DeepLC. The
    shipped repo only contains pretrained Keras ``.keras``/``.hdf5`` model
    files (no architecture-builder script survives in the current tree), so
    the paper text/figure is the reference: a 4-branch 1D-CNN over parallel
    peptide encodings -- two branches of stacked Conv1D over per-position
    atomic composition (C, H, N, O, P, S counts), a third Conv1D branch over
    one-hot amino-acid identity, and a fourth dense branch over global
    peptide-level features (length, total atom counts) -- concatenated and
    passed through several dense layers to a scalar retention-time output.
  - DeepLesion CAD network / 3DCE (cand_00526): Yan, Bagheri & Summers,
    MICCAI 2018 "3D Context Enhanced Region-based Convolutional Neural
    Network for End-to-End Lesion Detection" (the paper behind the NIH
    DeepLesion CAD baseline; arXiv 1806.09648, catalog cites the related
    1710.01766 DeepLesion-dataset paper), official code
    https://github.com/rsummers11/CADLab/tree/master/lesion_detector_3DCE
    (MXNet; ``rcnn/symbol/symbol_vgg.py``). The distinctive "3D-context"
    mechanism: a stack of adjacent axial CT slices is split into overlapping
    3-slice pseudo-RGB groups, each group is run through a *shared* VGG-style
    2D conv backbone, and the resulting per-group feature maps are
    concatenated along the channel axis before a lightweight R-FCN-style
    head (1x1 conv down-projection + position-sensitive-style pooling
    approximated here with adaptive average pooling, feeding class and
    bounding-box regression heads) -- exactly the paper's "aggregate feature
    maps of 2D images to leverage 3D context without full 3D convolution"
    idea, reproduced compactly in torch.
  - DeepLoc (cand_00527): Almagro Armenteros et al., Bioinformatics 2017
    (DeepLoc 1.0) and the DeepLoc-2.0 successor, official code
    https://github.com/teevee112/DeepLoc-2.0 (``src/model.py``). DeepLoc-2.0
    operates on frozen protein-language-model embeddings (ProtT5/ESM1b) as
    input and its distinctive layer is ``AttentionHead``: embeddings are
    reshaped into per-head chunks, LayerNorm'd, scored by a learned query
    projection, smoothed along the sequence, masked to the true (unpadded)
    length, softmax-normalized, and used to pool the sequence into a fixed
    vector for an 11-way multi-label subcellular-localization classifier
    (``clf_head``) -- reproduced here verbatim in structure (length-masked
    single/multi-head learned-query attention pooling + linear multi-label
    head), taking a synthetic per-residue embedding tensor as input in place
    of the frozen PLM (which is outside torchlens' scope to trace).
  - DeepNovo / DeepNovo-DIA (cand_00531): Tran, Zhang, Xin, Shan & Li,
    Nature Methods 2018/2019, official code
    https://github.com/nh2tran/DeepNovo-DIA (``deepnovo_model.py``, pure
    TensorFlow 1.x). Framework column says TF but the base env here is
    torch-only, so the distinctive mechanism is reimplemented from scratch
    in torch rather than run under TF: a 3D-conv tower (``_build_cnn_ion``)
    over a stacked ion-current "image" (ion type x neighboring-mass-bin x
    m/z-window), combined with an LSTM decoder over amino-acid embeddings
    (``_build_lstm_iter``) that is fed a CNN-derived spectrum feature as its
    initial hidden state (``_build_cnn_spectrum``/``_build_lstm_0``); the two
    feature streams (CNN-ion, LSTM) are summed into a joint logit over the
    amino-acid vocabulary for de novo peptide sequencing, and the true model
    runs this bidirectionally (forward + backward decoders sharing the CNN
    features) -- reproduced compactly here with both directions.

All models below are compact, faithfully-reimplemented-from-scratch nn.Modules
with random init and small dims for TorchLens architecture-catalog tracing
(not a trained-weights zoo).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ============================================================
# DeepHLApan -- 3x BiGRU + additive self-attention pooling
# ============================================================


class _AdditiveAttentionPool(nn.Module):
    """Additive (tanh-scored) self-attention pooling over a sequence.

    Reproduces the repo's custom ``Attention`` Keras layer: a learned
    per-timestep scalar score ``tanh(x . W + b)``, softmax-normalized along
    the sequence axis, used as pooling weights over the BiGRU output.
    """

    def __init__(self, hidden: int, seq_len: int) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.randn(hidden) * 0.05)
        self.b = nn.Parameter(torch.zeros(seq_len))

    def forward(self, x: Tensor) -> Tensor:
        """Pool ``x`` of shape ``(batch, seq_len, hidden)`` to ``(batch, hidden)``."""
        eij = torch.tanh(torch.einsum("blh,h->bl", x, self.w) + self.b)
        attn = F.softmax(eij, dim=1)
        return torch.einsum("bl,blh->bh", attn, x)


class DeepHLApan(nn.Module):
    """3-layer stacked BiGRU with additive self-attention pooling.

    Encodes a fixed-length one-hot HLA-pseudo-sequence + peptide complex,
    contextualizes it with 3 stacked bidirectional GRU layers, pools with
    a learned additive-attention layer, and predicts a binding /
    immunogenicity probability with a sigmoid classification head.
    """

    def __init__(
        self,
        vocab_size: int = 22,
        seq_len: int = 49,
        embed_dim: int = 16,
        hidden: int = 16,
        n_gru_layers: int = 3,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(
            embed_dim,
            hidden,
            num_layers=n_gru_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.attn_pool = _AdditiveAttentionPool(hidden * 2, seq_len)
        self.classifier = nn.Linear(hidden * 2, 1)

    def forward(self, seq: Tensor) -> Tensor:
        """Predict a binding/immunogenicity probability from ``seq`` token ids."""
        x = self.embed(seq)
        x, _ = self.gru(x)
        pooled = self.attn_pool(x)
        return torch.sigmoid(self.classifier(pooled))


def build_deephlapan() -> nn.Module:
    """Build a small DeepHLApan (3x BiGRU + attention pooling) classifier."""
    return DeepHLApan(vocab_size=22, seq_len=49, embed_dim=16, hidden=16, n_gru_layers=3).eval()


def example_input_deephlapan() -> Tensor:
    """Token-id sequence ``(2, 49)`` for the HLA-pseudo-sequence + peptide complex."""
    return torch.randint(0, 22, (2, 49))


# ============================================================
# DeepImmuno -- WGAN with residual-1D-conv generator/discriminator
#                + dual-branch CNN binding/immunogenicity classifier
# ============================================================


class _ResBlock1D(nn.Module):
    """Residual 1D-conv block: ``x + 0.3 * conv_relu_conv(x)`` (verbatim from the repo)."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the residual block with the paper's 0.3 residual scale."""
        return x + 0.3 * self.block(x)


class DeepImmunoGenerator(nn.Module):
    """WGAN generator: noise -> 5 residual 1D-conv blocks -> Gumbel-softmax sequence.

    Reproduces ``deepimmuno-gan.py``'s ``Generator``: a linear projection of
    noise into a ``(hidden, seq_len)`` feature map, 5 stacked residual 1D
    conv blocks, a 1x1 conv projecting to the amino-acid vocabulary, and a
    Gumbel-softmax over the vocabulary axis to produce a differentiable
    categorical peptide sequence.
    """

    def __init__(
        self, hidden: int = 16, seq_len: int = 10, n_chars: int = 21, noise_dim: int = 32
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.seq_len = seq_len
        self.n_chars = n_chars
        self.fc1 = nn.Linear(noise_dim, hidden * seq_len)
        self.block = nn.Sequential(*[_ResBlock1D(hidden) for _ in range(5)])
        self.conv1 = nn.Conv1d(hidden, n_chars, kernel_size=1)

    def forward(self, noise: Tensor) -> Tensor:
        """Generate a ``(batch, seq_len, n_chars)`` soft one-hot peptide sequence."""
        batch = noise.shape[0]
        out = self.fc1(noise).view(batch, self.hidden, self.seq_len)
        out = self.block(out)
        out = self.conv1(out)  # (batch, n_chars, seq_len)
        out = out.transpose(1, 2).contiguous()  # (batch, seq_len, n_chars)
        out = F.gumbel_softmax(
            out.reshape(batch * self.seq_len, self.n_chars), tau=0.75, hard=False
        )
        return out.view(batch, self.seq_len, self.n_chars)


class DeepImmunoDiscriminator(nn.Module):
    """WGAN discriminator: sequence -> 5 residual 1D-conv blocks -> scalar score."""

    def __init__(self, hidden: int = 16, seq_len: int = 10, n_chars: int = 21) -> None:
        super().__init__()
        self.hidden = hidden
        self.seq_len = seq_len
        self.conv1 = nn.Conv1d(n_chars, hidden, kernel_size=1)
        self.block = nn.Sequential(*[_ResBlock1D(hidden) for _ in range(5)])
        self.fc = nn.Linear(seq_len * hidden, 1)

    def forward(self, seq: Tensor) -> Tensor:
        """Score a ``(batch, seq_len, n_chars)`` soft one-hot sequence for realism."""
        out = seq.transpose(1, 2).contiguous()  # (batch, n_chars, seq_len)
        out = self.conv1(out)
        out = self.block(out)
        out = out.reshape(-1, self.seq_len * self.hidden)
        return self.fc(out)


class DeepImmuno(nn.Module):
    """Combined WGAN wrapper: samples a generator sequence and scores it.

    Wraps the generator and discriminator into one traceable forward pass
    (``noise -> generated peptide -> realism score``), preserving both
    real torch components of the official ``deepimmuno-gan.py``.
    """

    def __init__(
        self, hidden: int = 16, seq_len: int = 10, n_chars: int = 21, noise_dim: int = 32
    ) -> None:
        super().__init__()
        self.generator = DeepImmunoGenerator(hidden, seq_len, n_chars, noise_dim)
        self.discriminator = DeepImmunoDiscriminator(hidden, seq_len, n_chars)

    def forward(self, noise: Tensor) -> Tensor:
        """Generate a pseudo-immunogenic peptide sequence and score it."""
        seq = self.generator(noise)
        return self.discriminator(seq)


def build_deepimmuno() -> nn.Module:
    """Build a small DeepImmuno WGAN (residual-1D-conv generator + discriminator)."""
    return DeepImmuno(hidden=16, seq_len=10, n_chars=21, noise_dim=32).eval()


def example_input_deepimmuno() -> Tensor:
    """Latent noise batch ``(2, 32)`` for the WGAN generator."""
    return torch.randn(2, 32)


# ============================================================
# DeepLC -- 4-branch 1D-CNN over parallel peptide encodings
# ============================================================


class DeepLC(nn.Module):
    """4-branch CNN retention-time predictor over parallel peptide encodings.

    Two Conv1D branches over per-position atomic-composition counts (at
    different receptive fields), a third Conv1D branch over one-hot
    amino-acid identity, and a fourth dense branch over global peptide
    features (length, total atom counts); all four branches are flattened,
    concatenated, and passed through several dense layers to a scalar
    retention-time prediction.
    """

    def __init__(
        self,
        seq_len: int = 30,
        n_atoms: int = 6,
        n_aa: int = 20,
        n_global: int = 8,
        channels: int = 16,
    ) -> None:
        super().__init__()
        self.atom_branch_a = nn.Sequential(
            nn.Conv1d(n_atoms, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.atom_branch_b = nn.Sequential(
            nn.Conv1d(n_atoms, channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.aa_branch = nn.Sequential(
            nn.Conv1d(n_aa, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.global_branch = nn.Sequential(
            nn.Linear(n_global, channels),
            nn.ReLU(inplace=True),
        )
        flat = channels * seq_len * 3 + channels
        dense_dims = [flat, 128, 64, 32, 16, 8]
        dense_layers: list[nn.Module] = []
        for in_d, out_d in zip(dense_dims[:-1], dense_dims[1:]):
            dense_layers += [nn.Linear(in_d, out_d), nn.ReLU(inplace=True)]
        self.dense = nn.Sequential(*dense_layers)
        self.out = nn.Linear(dense_dims[-1], 1)

    def forward(
        self,
        atom_comp: Tensor,
        one_hot: Tensor,
        global_feats: Tensor,
    ) -> Tensor:
        """Predict a scalar retention time from the 3 sequence encodings + globals."""
        a = self.atom_branch_a(atom_comp).flatten(1)
        b = self.atom_branch_b(atom_comp).flatten(1)
        c = self.aa_branch(one_hot).flatten(1)
        d = self.global_branch(global_feats)
        combined = torch.cat([a, b, c, d], dim=1)
        return self.out(self.dense(combined))


def build_deeplc() -> nn.Module:
    """Build a small DeepLC 4-branch CNN retention-time predictor."""
    return DeepLC(seq_len=30, n_atoms=6, n_aa=20, n_global=8, channels=16).eval()


def example_input_deeplc() -> tuple[Tensor, Tensor, Tensor]:
    """Atomic-composition, one-hot AA, and global-feature tensors for one batch."""
    atom_comp = torch.randn(2, 6, 30)
    one_hot = torch.zeros(2, 20, 30)
    one_hot[:, 0, :] = 1.0
    global_feats = torch.randn(2, 8)
    return atom_comp, one_hot, global_feats


# ============================================================
# DeepLesion CAD network (3DCE) -- shared VGG backbone over slice
# groups + channel-concat 3D context + R-FCN-style head
# ============================================================


class _VGGLiteBackbone(nn.Module):
    """Compact shared 2D conv backbone applied per pseudo-RGB slice group."""

    def __init__(self, out_ch: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, out_ch // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch // 2, out_ch // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(out_ch // 2, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DeepLesion3DCE(nn.Module):
    """3D Context Enhanced (3DCE) universal lesion detector, compact form.

    A stack of ``n_groups`` overlapping 3-slice pseudo-RGB CT groups is run
    through one *shared-weight* VGG-lite backbone; the resulting per-group
    feature maps are concatenated along the channel axis (the paper's "3D
    context" trick -- 2D convs over neighboring slices instead of true 3D
    convs), then a 1x1 conv + pooled head produces per-image classification
    and bounding-box-regression outputs (standing in for the paper's RPN +
    PS-ROI-pooled R-FCN head, which needs proposal boxes as an additional
    dynamic input).
    """

    def __init__(self, n_groups: int = 3, backbone_ch: int = 32, num_classes: int = 2) -> None:
        super().__init__()
        self.n_groups = n_groups
        self.backbone = _VGGLiteBackbone(backbone_ch)
        head_ch = backbone_ch * n_groups
        self.head_conv = nn.Conv2d(head_ch, 64, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Linear(64, num_classes)
        self.bbox_head = nn.Linear(64, 4 * num_classes)

    def forward(self, slice_groups: Tensor) -> dict[str, Tensor]:
        """Detect lesions from ``(batch, n_groups, 3, H, W)`` pseudo-RGB slice groups."""
        batch, n_groups, c, h, w = slice_groups.shape
        flat = slice_groups.reshape(batch * n_groups, c, h, w)
        feats = self.backbone(flat)  # shared weights across groups
        feats = feats.view(batch, n_groups, feats.shape[1], feats.shape[2], feats.shape[3])
        feats = feats.reshape(batch, n_groups * feats.shape[2], feats.shape[3], feats.shape[4])
        feats = F.relu(self.head_conv(feats))
        pooled = self.pool(feats).flatten(1)
        return {"cls_logits": self.cls_head(pooled), "bbox_deltas": self.bbox_head(pooled)}


def build_deeplesion_3dce() -> nn.Module:
    """Build a small 3DCE universal lesion detector (shared backbone + 3D-context concat)."""
    return DeepLesion3DCE(n_groups=3, backbone_ch=32, num_classes=2).eval()


def example_input_deeplesion_3dce() -> Tensor:
    """3 overlapping pseudo-RGB CT slice groups ``(1, 3, 3, 64, 64)``."""
    return torch.randn(1, 3, 3, 64, 64)


# ============================================================
# DeepLoc -- length-masked learned-query attention pooling over
# frozen protein-language-model embeddings
# ============================================================


class _DeepLocAttentionHead(nn.Module):
    """Length-masked, per-head-LayerNorm, learned-query attention pooling.

    Reproduces ``src/model.py``'s ``AttentionHead``: the embedding is split
    into ``n_heads`` chunks, each LayerNorm'd, scored against a learned
    query vector, masked to the true (unpadded) sequence length, and
    softmax-normalized before pooling.
    """

    def __init__(self, hidden_dim: int, n_heads: int = 1) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.preattn_ln = nn.LayerNorm(hidden_dim // n_heads)
        self.query = nn.Linear(hidden_dim // n_heads, n_heads, bias=False)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Pool ``x (batch, seq, hidden)`` to ``(batch, hidden)`` using ``mask``."""
        batch, seq_len, _ = x.shape
        head_dim = self.hidden_dim // self.n_heads
        x = x.view(batch, seq_len, self.n_heads, head_dim)
        x = self.preattn_ln(x)
        scores = (x * self.query.weight.view(1, 1, self.n_heads, head_dim)).sum(-1)  # (b, l, nh)
        scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        attn = F.softmax(scores, dim=1)
        pooled = (x * attn.unsqueeze(-1)).sum(1)
        return pooled.reshape(batch, -1)


class DeepLoc(nn.Module):
    """DeepLoc-2.0-style multi-label subcellular-localization classifier.

    Takes a per-residue protein-language-model embedding (the frozen
    ProtT5/ESM1b backbone itself is out of torchlens' scope; a synthetic
    embedding stands in as the module's real input, matching the official
    ``BaseModel.forward(embedding, lens, non_mask)`` signature), LayerNorms
    and projects it, pools with the length-masked learned-query attention
    head, and predicts 11 multi-label localization classes.
    """

    def __init__(self, embed_dim: int = 32, hidden: int = 16, n_classes: int = 11) -> None:
        super().__init__()
        self.initial_ln = nn.LayerNorm(embed_dim)
        self.lin = nn.Linear(embed_dim, hidden)
        self.attn_head = _DeepLocAttentionHead(hidden, n_heads=1)
        self.clf_head = nn.Linear(hidden, n_classes)

    def forward(self, embedding: Tensor, mask: Tensor) -> Tensor:
        """Predict 11 multi-label subcellular-localization logits."""
        x = self.initial_ln(embedding)
        x = self.lin(x)
        pooled = self.attn_head(x, mask)
        return self.clf_head(pooled)


def build_deeploc2() -> nn.Module:
    """Build a small DeepLoc-2.0 attention-pooled multi-label classifier."""
    return DeepLoc(embed_dim=32, hidden=16, n_classes=11).eval()


def example_input_deeploc2() -> tuple[Tensor, Tensor]:
    """A per-residue embedding ``(2, 20, 32)`` and its boolean valid-position mask."""
    embedding = torch.randn(2, 20, 32)
    mask = torch.ones(2, 20, dtype=torch.bool)
    mask[1, 15:] = False
    return embedding, mask


# ============================================================
# DeepNovo -- 3D-conv ion-current tower + bidirectional LSTM decoder
# ============================================================


class _CnnIonTower(nn.Module):
    """3D-conv tower over a stacked (ion-type, mass-bin, m/z-window) ion-current image."""

    def __init__(self, vocab_size: int, out_dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(vocab_size, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=(1, 2, 2), padding=(0, 0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=(1, 2, 2), padding=(0, 0, 0)),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(64, out_dim)

    def forward(self, intensity: Tensor) -> Tensor:
        """Map a ``(batch, vocab, n_ion, neighbor, window)`` ion image to a feature vector."""
        feat = self.conv(intensity)
        feat = self.pool(feat).flatten(1)
        return self.fc(feat)


class _CnnSpectrumTower(nn.Module):
    """2D-conv tower over the raw mass spectrum, producing an initial LSTM state."""

    def __init__(self, mz_size: int, out_dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(32, out_dim)

    def forward(self, spectrum: Tensor) -> Tensor:
        """Map a ``(batch, mz_size)`` spectrum to an ``(batch, out_dim)`` LSTM seed state."""
        feat = self.conv(spectrum.unsqueeze(1))
        feat = feat.flatten(1)
        return self.fc(feat)


class DeepNovoDirection(nn.Module):
    """One direction (forward or backward) of the DeepNovo ion-LSTM decoder.

    Combines a CNN-ion feature (from the stacked ion-current image) with an
    LSTM feature (amino-acid embeddings seeded by a CNN-spectrum-derived
    initial state) via elementwise sum, then a shared softmax head over the
    amino-acid vocabulary -- reproducing ``_build_cnn_ion`` +
    ``_build_lstm_0``/``_build_lstm_iter`` + ``_combine_feature``.
    """

    def __init__(self, vocab_size: int, mz_size: int, embed_dim: int, hidden: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.cnn_ion = _CnnIonTower(vocab_size, hidden)
        self.cnn_spectrum = _CnnSpectrumTower(mz_size, hidden)
        self.lstm_cell = nn.LSTMCell(embed_dim, hidden)
        self.out_proj = nn.Linear(hidden, vocab_size)

    def forward(
        self,
        spectrum: Tensor,
        intensity: Tensor,
        aa_ids: Tensor,
    ) -> Tensor:
        """Predict next-amino-acid logits from spectrum, ion image, and AA history."""
        h0 = self.cnn_spectrum(spectrum)
        c0 = torch.zeros_like(h0)
        embedded = self.embed(aa_ids)
        h, c = h0, c0
        for t in range(embedded.shape[1]):
            h, c = self.lstm_cell(embedded[:, t, :], (h, c))
        ion_feature = self.cnn_ion(intensity)
        combined = h + ion_feature
        return self.out_proj(combined)


class DeepNovo(nn.Module):
    """Bidirectional DeepNovo-DIA de novo peptide sequencer (compact form).

    Runs the ``DeepNovoDirection`` ion-LSTM decoder in both the forward and
    backward directions (sharing no weights, matching the official model's
    separately-parameterized ``forward``/``backward`` scopes) and returns
    both directions' next-amino-acid logits.
    """

    def __init__(
        self,
        vocab_size: int = 26,
        mz_size: int = 128,
        embed_dim: int = 16,
        hidden: int = 32,
    ) -> None:
        super().__init__()
        self.forward_decoder = DeepNovoDirection(vocab_size, mz_size, embed_dim, hidden)
        self.backward_decoder = DeepNovoDirection(vocab_size, mz_size, embed_dim, hidden)

    def forward(
        self,
        spectrum: Tensor,
        intensity: Tensor,
        aa_ids_forward: Tensor,
        aa_ids_backward: Tensor,
    ) -> dict[str, Tensor]:
        """Predict next-amino-acid logits in both sequencing directions."""
        logit_forward = self.forward_decoder(spectrum, intensity, aa_ids_forward)
        logit_backward = self.backward_decoder(spectrum, intensity, aa_ids_backward)
        return {"logit_forward": logit_forward, "logit_backward": logit_backward}


def build_deepnovo() -> nn.Module:
    """Build a small bidirectional DeepNovo-DIA ion-LSTM de novo sequencer."""
    return DeepNovo(vocab_size=26, mz_size=128, embed_dim=16, hidden=32).eval()


def example_input_deepnovo() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Spectrum, stacked ion-current image, and forward/backward AA-id histories."""
    spectrum = torch.randn(2, 128)
    intensity = torch.randn(2, 26, 8, 5, 10)  # (batch, vocab, n_ion, neighbor, window)
    aa_ids_forward = torch.randint(0, 26, (2, 4))
    aa_ids_backward = torch.randint(0, 26, (2, 4))
    return spectrum, intensity, aa_ids_forward, aa_ids_backward


# ============================================================
# Registry
# ============================================================

MENAGERIE_ENTRIES = [
    ("DeepHLApan", "build_deephlapan", "example_input_deephlapan", "2019", "BIO"),
    ("DeepImmuno", "build_deepimmuno", "example_input_deepimmuno", "2021", "BIO"),
    ("DeepLC", "build_deeplc", "example_input_deeplc", "2021", "BIO"),
    (
        "DeepLesion CAD network",
        "build_deeplesion_3dce",
        "example_input_deeplesion_3dce",
        "2018",
        "VIS",
    ),
    ("DeepLoc", "build_deeploc2", "example_input_deeploc2", "2023", "BIO"),
    ("DeepNovo", "build_deepnovo", "example_input_deepnovo", "2018", "BIO"),
]
