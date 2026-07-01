"""Menagerie batch w6a8: computational-genomics deep-learning classics for
promoter/enhancer regulatory-activity prediction, plasmid-vs-chromosome
metagenomic contig classification, adversarial cross-domain single-cell
atlas integration, RNA-structure-aware protein-binding-site prediction, and
ProbSparse-attention transformer-based scATAC-seq chromatin-accessibility
prediction.

Sources checked (reference only; no cloning, no pip installs):
  - PARM (cand_00754): de Boer lab / van Steensel lab, "PARM: Promoter
    Activity Regulatory Model", Nature 2026 (van Steensel et al.); official
    repo vansteensellab/PARM, ``PARM/PARM_utils_load_model.py``, class
    ``ResNet_Attentionpool``. The defining mechanism is a **1D dilated-free
    residual CNN over one-hot DNA with soft AttentionPool downsampling**:
    a conv stem followed by a stack of residual ``ConvBlock`` units
    (``BatchNorm1d -> sigmoid-GELU -> Conv1d``, each wrapped in a residual
    skip), where every downsampling step is NOT a hard max/avg pool but an
    ``AttentionPool`` layer (Enformer-style) that reshapes the sequence axis
    into fixed-size pooling windows, computes a per-window softmax attention
    map over positions via a ``Conv2d`` logit head, and takes the attention-
    weighted sum within each window -- i.e. "residual 1D-CNN tower with
    learned-softmax attention pooling instead of max pooling at every stage"
    is PARM's namesake regulatory-activity mechanism. Reimplemented with the
    same stem -> residual-block -> AttentionPool tower topology (5 blocks,
    reduced filter width) and a final max-pool-over-length + linear head.
  - PlasFlow (cand_00756): Krawczyk, Lipinski & Dziembowski, Nucleic Acids
    Research 2018; official repo smaegol/plasflow,
    ``scripts/PlasFlow_train.py``. The original tool trains a
    ``tf.contrib.learn.DNNClassifier`` (a plain multi-layer perceptron) on
    TF-IDF-transformed k-mer (5-mer/6-mer/7-mer composition) frequency
    vectors extracted per contig; the paper's own selected architecture is a
    2-hidden-layer sigmoid MLP (30/20 hidden units) over the k-mer feature
    vector. There is no architectural novelty beyond the classifier itself
    -- the distinctive element is the **k-mer-frequency-vector-to-MLP
    genome-signature classification** pipeline. Reimplemented as the same
    2-hidden-layer sigmoid MLP over a k-mer composition feature vector
    (reduced k-mer vocabulary size) with a final softmax over
    plasmid/chromosome/unclassified classes.
  - Portal (cand_00757): Zhao, Cai, Sun, Hu, Zeng & Yang, Nature
    Computational Science 2022; official repo YangLabHKUST/Portal,
    ``portal/networks.py`` (classes ``encoder``/``generator``/
    ``discriminator``), ``portal/model.py``. The defining mechanism is
    **adversarial domain-translation for atlas-level single-cell
    integration**: two domain-specific linear encoders (one per batch/
    dataset, domain A and domain B) map each domain's PCA-reduced expression
    into a SHARED latent space; a single SHARED generator decodes latents
    from either domain back into the other domain's data space (cross-domain
    translation, not domain-specific decoders); and a shared discriminator
    is trained adversarially to distinguish real vs. generator-translated
    samples in data space, driving the two domains' latents/translations
    into a common manifold -- "two domain encoders + one shared cross-domain
    generator + one shared adversarial discriminator" is Portal's namesake
    integration mechanism (as opposed to a single joint autoencoder or
    domain-specific decoders). Reimplemented with the same two-encoder /
    shared-generator / shared-discriminator topology at reduced input and
    latent widths, returning encoder-A and encoder-B latents plus the
    generator's cross-domain reconstructions and the discriminator's realism
    scores for both real and translated samples.
  - PrismNet (cand_00758): Sun, Xu, Wang, et al., Cell Research 2021;
    official repo kuixu/PrismNet, ``prismnet/model/PrismNet.py`` (class
    ``PrismNet``), ``prismnet/model/se.py`` (``SEBlock``),
    ``prismnet/model/resnet.py`` (``ResidualBlock1D``/``ResidualBlock2D`` --
    exact PyTorch source, used near-verbatim at reduced channel width). The
    defining mechanism is **RNA-structure-aware SE-gated residual CNN**:
    a stacked one-hot-sequence + icSHAPE-structure-profile "image" (sequence
    positions x [4 base channels + 1 structure-reactivity channel]) is
    embedded by a single 2D conv, gated by a **squeeze-and-excitation block**
    whose per-channel sigmoid attention is multiplied elementwise into the
    conv features BEFORE (not after) a 2D bottleneck residual block
    (``x * se_attention`` feeding ``ResidualBlock2D``), average-pooled across
    the feature-channel axis down to a 1D sequence representation, passed
    through a 1D bottleneck residual block, global-average-pooled, and
    mapped to a single RBP-binding-intensity score by a linear head -- i.e.
    "SE-gate-then-residual-2D over sequence+structure, collapse the feature
    axis, residual-1D, global pool, linear" is PrismNet's namesake in-vivo
    structure-integration mechanism. Reimplemented with the same SE-gate ->
    ResidualBlock2D -> feature-axis avgpool -> ResidualBlock1D -> global-pool
    -> linear topology at reduced base channel width.
  - PROTRAIT (cand_00759): Zhang lab, "PROTRAIT: a ProbSparse-Attention
    Transformer for Integrated scATAC-seq Analysis", International Journal
    of Molecular Sciences 2023; official repo ZhangLab312/PROTRAIT,
    ``public/model.py`` (class ``Protrait`` and its ``Encoder``/
    ``EncoderLayer``/``SelfAttention``/``ProbAttention``/``BottleneckLayer``/
    ``Prediction`` components -- adapted from the Informer long-sequence
    forecasting transformer). The defining mechanism is a **ProbSparse
    self-attention transformer over one-hot DNA sequence** for scATAC-seq
    chromatin-accessibility prediction: a base-pair conv embedding plus
    sinusoidal positional embedding feeds a stack of transformer encoder
    layers, each using ``ProbAttention`` -- a sparse variant of self-
    attention that RANDOMLY SUBSAMPLES a small set of keys per query to
    estimate each query's max-vs-mean attention "sparsity score", SELECTS
    only the top-scoring queries to compute full attention against all keys,
    and fills every other query's output with the uniform (mean-pooled)
    value -- interleaved with 1D conv-based downsampling/pooling layers
    between encoder stages, and a final bottleneck conv + linear head
    predicting a per-cell-type accessibility-probability vector via sigmoid
    -- "query-sparsifying ProbSparse attention transformer encoder with
    interleaved conv pooling, ending in a multi-cell-type sigmoid
    accessibility head" is PROTRAIT's namesake mechanism. Reimplemented with
    the same base-pair embedding + positional embedding, ProbSparse
    self-attention encoder layers interleaved with conv pooling layers, and
    bottleneck + sigmoid prediction head, at reduced model width / depth /
    sequence length.

Not built:
  - PEPPER (cand_00755): kishwarshafin/pepper's GRU-based variant-candidate
    network is the SAME namesake pileup-image-to-variant stacked-(Bi)LSTM/
    GRU transducer already faithfully captured as
    "PEPPER-Margin-DeepVariant" in ``menagerie/classics/gen_w5a8.py``
    (``TransducerGRU`` -> 2-layer stacked BiLSTM encoder/decoder, flattened
    across the candidate window, 5-layer SELU MLP variant-type head).
    Skipped as already_in_catalog.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ============================================================
# PARM -- residual 1D-CNN tower with Enformer-style AttentionPool
# downsampling over one-hot DNA sequence (vansteensellab/PARM)
# ============================================================


class _GeLUSigmoid(nn.Module):
    """Sigmoid-approximated GELU used in PARM's ``ConvBlock`` (matches the
    reference repo's custom ``GELU`` class: ``sigmoid(1.702 * x) * x``)."""

    def forward(self, x: Tensor) -> Tensor:
        """Apply the sigmoid-approximated GELU nonlinearity."""
        return torch.sigmoid(1.702 * x) * x


class _ParmConvBlock(nn.Module):
    """BatchNorm1d -> sigmoid-GELU -> Conv1d, PARM's basic conv unit."""

    def __init__(self, dim_in: int, dim_out: int, kernel_size: int = 1) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(dim_in)
        self.act = _GeLUSigmoid()
        self.conv = nn.Conv1d(dim_in, dim_out, kernel_size, padding=kernel_size // 2)

    def forward(self, x: Tensor) -> Tensor:
        """Apply norm -> activation -> conv."""
        return self.conv(self.act(self.norm(x)))


class _ParmResidual(nn.Module):
    """Wrap a same-shape submodule with an additive residual skip."""

    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        """Return ``fn(x) + x``."""
        return self.fn(x) + x


class _ParmAttentionPool(nn.Module):
    """Enformer-style learned-softmax attention pooling along the sequence
    axis (PARM's ``AttentionPool``): reshape into fixed windows, compute a
    per-window softmax attention map via a Conv2d logit head, and take the
    attention-weighted sum within each window instead of a hard max/avg."""

    def __init__(self, dim: int, pool_size: int = 2) -> None:
        super().__init__()
        self.pool_size = pool_size
        self.to_attn_logits = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Attention-pool ``x`` (shape ``[b, d, n]``) along its last axis."""
        b, d, n = x.shape
        remainder = n % self.pool_size
        if remainder > 0:
            pad = self.pool_size - remainder
            x = F.pad(x, (0, pad), value=0.0)
            n = x.shape[-1]
        windows = x.view(b, d, n // self.pool_size, self.pool_size)
        logits = self.to_attn_logits(windows)
        attn = logits.softmax(dim=-1)
        return (windows * attn).sum(dim=-1)


class ParmResNetAttentionPool(nn.Module):
    """Compact PARM-style residual 1D-CNN tower with AttentionPool
    downsampling for promoter/enhancer regulatory-activity prediction from
    one-hot DNA sequence."""

    def __init__(
        self,
        vocab: int = 4,
        filter_size: int = 16,
        n_blocks: int = 5,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(vocab, filter_size, 7, padding=3),
            _ParmResidual(_ParmConvBlock(filter_size, filter_size)),
            _ParmAttentionPool(filter_size, pool_size=2),
        )

        conv_layers = []
        prev_filter_size = filter_size
        for block in range(n_blocks):
            out_filter_size = int(filter_size * 0.6) if block > 2 else filter_size
            conv_layers.append(
                nn.Sequential(
                    _ParmConvBlock(prev_filter_size, out_filter_size, kernel_size=kernel_size),
                    _ParmResidual(_ParmConvBlock(out_filter_size, out_filter_size, kernel_size=1)),
                    _ParmAttentionPool(out_filter_size, pool_size=2),
                )
            )
            prev_filter_size = out_filter_size
        self.conv_tower = nn.Sequential(*conv_layers)
        self.linear = nn.Linear(prev_filter_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """Predict a single Poisson-rate promoter-activity score per sequence."""
        out = self.stem(x)
        out = self.conv_tower(out)
        out = torch.max(out, dim=-1).values
        out = self.linear(out)
        return self.relu(out)


def build_parm() -> nn.Module:
    """Build a compact PARM-style AttentionPool residual CNN.

    Returns
    -------
    nn.Module
        ``ParmResNetAttentionPool`` in eval mode.
    """
    model = ParmResNetAttentionPool()
    model.eval()
    return model


def example_input_parm() -> Tensor:
    """One-hot DNA sequence batch, shape ``[batch, 4, length]``."""
    torch.manual_seed(0)
    idx = torch.randint(0, 4, (2, 256))
    return F.one_hot(idx, num_classes=4).permute(0, 2, 1).float()


# ============================================================
# PlasFlow -- k-mer-composition MLP genome-signature classifier
# (smaegol/plasflow)
# ============================================================


class PlasFlowMLP(nn.Module):
    """Compact PlasFlow-style 2-hidden-layer sigmoid MLP over a k-mer
    composition feature vector, classifying a metagenomic contig as
    plasmid / chromosome / unclassified from its k-mer genome signature."""

    def __init__(self, n_kmer_features: int = 64, n_classes: int = 3) -> None:
        super().__init__()
        self.hidden1 = nn.Linear(n_kmer_features, 30)
        self.hidden2 = nn.Linear(30, 20)
        self.output = nn.Linear(20, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Classify a TF-IDF k-mer-frequency feature vector."""
        h = torch.sigmoid(self.hidden1(x))
        h = torch.sigmoid(self.hidden2(h))
        return self.output(h)


def build_plasflow() -> nn.Module:
    """Build a compact PlasFlow-style k-mer MLP classifier.

    Returns
    -------
    nn.Module
        ``PlasFlowMLP`` in eval mode.
    """
    model = PlasFlowMLP()
    model.eval()
    return model


def example_input_plasflow() -> Tensor:
    """TF-IDF-normalized k-mer frequency vectors, shape ``[batch, 64]``."""
    torch.manual_seed(0)
    counts = torch.rand(4, 64)
    return counts / counts.sum(dim=-1, keepdim=True)


# ============================================================
# Portal -- two domain encoders + shared cross-domain generator
# + shared adversarial discriminator (YangLabHKUST/Portal)
# ============================================================


class _PortalEncoder(nn.Module):
    """Linear encoder mapping a domain's PCA-reduced expression into the
    shared latent space (Portal's ``encoder``)."""

    def __init__(self, n_input: int, n_latent: int, n_hidden: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_latent)

    def forward(self, x: Tensor) -> Tensor:
        """Encode ``x`` into the shared latent space."""
        h = F.relu(self.fc1(x))
        return self.fc2(h)


class _PortalGenerator(nn.Module):
    """Shared generator decoding a latent (from either domain) back into
    data space (Portal's ``generator``); shared across A->B and B->A."""

    def __init__(self, n_input: int, n_latent: int, n_hidden: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_latent, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_input)

    def forward(self, z: Tensor) -> Tensor:
        """Decode a latent code into data space."""
        h = F.relu(self.fc1(z))
        return self.fc2(h)


class _PortalDiscriminator(nn.Module):
    """Shared adversarial discriminator scoring real-vs-translated samples
    in data space (Portal's ``discriminator``); output clamped to
    ``[-50, 50]`` as in the reference."""

    def __init__(self, n_input: int, n_hidden: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Score ``x`` for realism, clamped to ``[-50, 50]``."""
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return torch.clamp(self.fc3(h), min=-50.0, max=50.0)


class PortalIntegration(nn.Module):
    """Compact Portal-style adversarial domain-translation network for
    atlas-level single-cell integration: domain-A and domain-B encoders map
    into a shared latent space, a single shared generator cross-decodes
    each domain's latent into the OTHER domain's data space, and a shared
    discriminator scores real vs. generator-translated samples."""

    def __init__(self, n_input: int = 30, n_latent: int = 8) -> None:
        super().__init__()
        self.encoder_a = _PortalEncoder(n_input, n_latent)
        self.encoder_b = _PortalEncoder(n_input, n_latent)
        self.generator = _PortalGenerator(n_input, n_latent)
        self.discriminator = _PortalDiscriminator(n_input)

    def forward(self, x_a: Tensor, x_b: Tensor) -> tuple[Tensor, ...]:
        """Encode both domains, cross-translate via the shared generator,
        and score real/translated samples with the shared discriminator."""
        z_a = self.encoder_a(x_a)
        z_b = self.encoder_b(x_b)

        translated_a_to_b = self.generator(z_a)
        translated_b_to_a = self.generator(z_b)

        score_real_a = self.discriminator(x_a)
        score_real_b = self.discriminator(x_b)
        score_fake_b = self.discriminator(translated_a_to_b)
        score_fake_a = self.discriminator(translated_b_to_a)

        return (
            z_a,
            z_b,
            translated_a_to_b,
            translated_b_to_a,
            score_real_a,
            score_real_b,
            score_fake_a,
            score_fake_b,
        )


def build_portal() -> nn.Module:
    """Build a compact Portal-style adversarial domain-translation network.

    Returns
    -------
    nn.Module
        ``PortalIntegration`` in eval mode.
    """
    model = PortalIntegration()
    model.eval()
    return model


def example_input_portal() -> tuple[Tensor, Tensor]:
    """Paired PCA-reduced expression batches for domain A and domain B,
    each shape ``[batch, 30]``."""
    torch.manual_seed(0)
    return torch.randn(6, 30), torch.randn(6, 30)


# ============================================================
# PrismNet -- SE-gated residual 2D-CNN over sequence+structure,
# collapsed to a residual 1D-CNN head (kuixu/PrismNet)
# ============================================================


class _PrismSEBlock(nn.Module):
    """Squeeze-and-excitation channel-gating block (PrismNet's ``SEBlock``):
    global-average-pool, then a bottleneck-MLP sigmoid gate per channel."""

    def __init__(self, channel: int, reduction: int = 2) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Return the per-channel sigmoid gate, broadcastable over ``x``."""
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y


class _PrismResidualBlock2D(nn.Module):
    """Bottleneck residual block over the 2D sequence x feature map
    (PrismNet's ``ResidualBlock2D``): 1x1 reduce -> kxk expand -> 1x1
    expand, summed with a 1x1-conv-projected residual skip."""

    def __init__(self, planes: int, kernel_size: tuple[int, int], padding: tuple[int, int]) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.b1 = nn.BatchNorm2d(planes)
        self.c2 = nn.Conv2d(
            planes, planes * 2, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.b2 = nn.BatchNorm2d(planes * 2)
        self.c3 = nn.Conv2d(planes * 2, planes * 4, kernel_size=1, bias=False)
        self.b3 = nn.BatchNorm2d(planes * 4)
        self.downsample = nn.Sequential(
            nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 4),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the bottleneck 2D residual transform."""
        identity = self.downsample(x)
        out = self.relu(self.b1(self.c1(x)))
        out = self.relu(self.b2(self.c2(out)))
        out = self.b3(self.c3(out))
        return self.relu(out + identity)


class _PrismResidualBlock1D(nn.Module):
    """Bottleneck residual block over the collapsed 1D sequence axis
    (PrismNet's ``ResidualBlock1D``)."""

    def __init__(self, planes: int) -> None:
        super().__init__()
        self.c1 = nn.Conv1d(planes, planes, kernel_size=1, bias=False)
        self.b1 = nn.BatchNorm1d(planes)
        self.c2 = nn.Conv1d(planes, planes * 2, kernel_size=11, padding=5, bias=False)
        self.b2 = nn.BatchNorm1d(planes * 2)
        self.c3 = nn.Conv1d(planes * 2, planes * 8, kernel_size=1, bias=False)
        self.b3 = nn.BatchNorm1d(planes * 8)
        self.downsample = nn.Sequential(
            nn.Conv1d(planes, planes * 8, kernel_size=1, bias=False),
            nn.BatchNorm1d(planes * 8),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the bottleneck 1D residual transform."""
        identity = self.downsample(x)
        out = self.relu(self.b1(self.c1(x)))
        out = self.relu(self.b2(self.c2(out)))
        out = self.b3(self.c3(out))
        return self.relu(out + identity)


class PrismNetClassics(nn.Module):
    """Compact PrismNet-style SE-gated residual CNN integrating in-vivo RNA
    structure (icSHAPE) with sequence for RBP-binding-intensity prediction.
    The SE block's sigmoid channel-attention gates the conv features
    ELEMENTWISE (``x * se_attention``) before the 2D residual block; the
    feature (base+structure) axis is then average-pooled away and a second
    residual block operates on the resulting 1D sequence representation."""

    def __init__(self, n_features: int = 5, base_channel: int = 8) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, base_channel, kernel_size=(11, 5), padding=(5, 2)),
            nn.BatchNorm2d(base_channel),
            nn.ReLU(inplace=True),
        )
        self.se = _PrismSEBlock(base_channel)
        self.res2d = _PrismResidualBlock2D(base_channel, kernel_size=(11, 5), padding=(5, 2))
        self.res1d = _PrismResidualBlock1D(base_channel * 4)
        self.avgpool = nn.AvgPool2d((1, n_features))
        self.gpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_channel * 4 * 8, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Predict a single RBP-binding-intensity score per sequence
        window, given a ``[batch, 1, length, n_features]`` sequence+
        structure "image" input."""
        x = self.conv(x)
        z = self.se(x)
        x = self.res2d(x * z)
        x = self.avgpool(x)
        x = x.view(x.shape[0], x.shape[1], x.shape[2])
        x = self.res1d(x)
        x = self.gpool(x)
        x = x.view(x.shape[0], x.shape[1])
        return self.fc(x)


def build_prismnet() -> nn.Module:
    """Build a compact PrismNet-style SE-gated structure-aware residual CNN.

    Returns
    -------
    nn.Module
        ``PrismNetClassics`` in eval mode.
    """
    model = PrismNetClassics()
    model.eval()
    return model


def example_input_prismnet() -> Tensor:
    """Sequence+icSHAPE-structure "image", shape ``[batch, 1, length, 5]``
    (4 one-hot base channels + 1 structure-reactivity channel)."""
    torch.manual_seed(0)
    return torch.randn(2, 1, 64, 5)


# ============================================================
# PROTRAIT -- ProbSparse-attention transformer over one-hot DNA
# for scATAC-seq accessibility prediction (ZhangLab312/PROTRAIT)
# ============================================================


class _ProtraitBasePairEmbedding(nn.Module):
    """Conv1d + GELU embedding of one-hot DNA sequence into ``d_model``
    channels (PROTRAIT's ``BasePairEmbedding``, short-sequence branch)."""

    def __init__(self, c_in: int, d_model: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(c_in, d_model, kernel_size=3, padding=1)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        """Embed ``[batch, c_in, length]`` into ``[batch, length, d_model]``."""
        return self.act(self.conv(x)).permute(0, 2, 1)


class _ProtraitPositionalEmbedding(nn.Module):
    """Fixed sinusoidal positional embedding added to the sequence
    embedding (PROTRAIT's ``PositionalEmbedding``)."""

    def __init__(self, d_model: int, seq_len: int) -> None:
        super().__init__()
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self) -> Tensor:
        """Return the fixed positional-embedding table, shape ``[1, L, D]``."""
        return self.pe


class _ProtraitProbAttention(nn.Module):
    """ProbSparse self-attention (PROTRAIT/Informer's ``ProbAttention``):
    randomly subsamples a small set of keys per query to estimate each
    query's max-vs-mean "sparsity score", selects only the top-scoring
    queries to compute FULL attention against all keys, and fills every
    other query's output with the uniform (mean-pooled) value -- avoiding
    full O(L^2) attention while preserving the highest-information queries."""

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Apply ProbSparse attention; inputs are ``[b, h, l, d]``."""
        b, h, l_query, dim = query.shape
        _, _, l_key, _ = key.shape

        down_sample_k = max(1, l_key // 4)
        n_query = max(1, l_query // 4)

        expanded_key = key.unsqueeze(-3).expand(b, h, l_query, l_key, dim)
        index_key = torch.randint(high=l_key, size=(l_query, down_sample_k), device=query.device)
        sampled_key = expanded_key[
            :, :, torch.arange(l_query, device=query.device).unsqueeze(1), index_key, :
        ]

        sampled_scores = torch.matmul(query.unsqueeze(-2), sampled_key.transpose(-2, -1)).squeeze(
            -2
        )
        sparsity_score = sampled_scores.max(-1).values - sampled_scores.sum(-1) / l_key
        top_query_idx = sparsity_score.topk(k=n_query, sorted=False).indices

        b_idx = torch.arange(b, device=query.device)[:, None, None]
        h_idx = torch.arange(h, device=query.device)[None, :, None]
        top_query = query[b_idx, h_idx, top_query_idx, :]

        scale = 1.0 / math.sqrt(dim)
        top_scores = torch.matmul(top_query, key.transpose(-2, -1)) * scale
        attn = torch.softmax(top_scores, dim=-1)
        top_context = torch.matmul(attn, value)

        context = value.mean(dim=-2, keepdim=True).expand(b, h, l_query, dim).clone()
        context[b_idx, h_idx, top_query_idx, :] = top_context.type_as(context)
        return context


class _ProtraitSelfAttention(nn.Module):
    """Multi-head wrapper around ``_ProtraitProbAttention`` (PROTRAIT's
    ``SelfAttention``)."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.heads = n_heads
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.prob_attention = _ProtraitProbAttention()

    def forward(self, x: Tensor) -> Tensor:
        """Apply multi-head ProbSparse self-attention to ``x``."""
        b, seq_len, _ = x.shape
        q = self.query_linear(x).view(b, seq_len, self.heads, -1).transpose(2, 1)
        k = self.key_linear(x).view(b, seq_len, self.heads, -1).transpose(2, 1)
        v = self.value_linear(x).view(b, seq_len, self.heads, -1).transpose(2, 1)
        out = self.prob_attention(q, k, v)
        return out.transpose(2, 1).contiguous().view(b, seq_len, -1)


class _ProtraitEncoderLayer(nn.Module):
    """ProbSparse self-attention + Conv1d feed-forward transformer encoder
    layer (PROTRAIT's ``EncoderLayer``)."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        d_ff = d_model * 2
        self.self_attention = _ProtraitSelfAttention(d_model, n_heads)
        self.conv_ffn_1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv_ffn_2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.act = nn.GELU()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Apply ProbSparse self-attention, then a conv feed-forward block,
        each with a residual skip and post-LayerNorm."""
        x = self.norm1(x + self.self_attention(x))
        h = x.transpose(-1, 1)
        h = self.act(self.conv_ffn_1(h))
        h = self.act(self.conv_ffn_2(h))
        h = h.transpose(-1, 1)
        return self.norm2(x + h)


class _ProtraitPoolingLayer(nn.Module):
    """Conv1d + BatchNorm1d + ELU + MaxPool1d downsampling between encoder
    stages (PROTRAIT's ``PoolingLayer``)."""

    def __init__(self, dim: int, pool_degree: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=5, padding=2)
        self.norm = nn.BatchNorm1d(dim)
        self.act = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=pool_degree)

    def forward(self, x: Tensor) -> Tensor:
        """Downsample ``[batch, length, dim]`` along the length axis."""
        h = x.permute(0, 2, 1)
        h = self.act(self.norm(self.conv(h)))
        h = self.pool(h)
        return h.permute(0, 2, 1)


class ProtraitAccessibility(nn.Module):
    """Compact PROTRAIT-style ProbSparse-attention transformer for
    scATAC-seq chromatin-accessibility prediction: a conv+positional
    embedding feeds a stack of ProbSparse-attention encoder layers
    interleaved with conv-pooling downsampling layers, followed by a
    bottleneck conv+linear head predicting a per-cell-type accessibility
    probability vector via sigmoid."""

    def __init__(
        self,
        seq_len: int = 96,
        c_in: int = 4,
        d_model: int = 32,
        n_heads: int = 4,
        n_stages: int = 3,
        n_cells: int = 8,
    ) -> None:
        super().__init__()
        self.embedding = _ProtraitBasePairEmbedding(c_in, d_model)
        self.positional_embedding = _ProtraitPositionalEmbedding(d_model, seq_len)

        stages = []
        length = seq_len
        for _ in range(n_stages):
            stages.append(_ProtraitEncoderLayer(d_model, n_heads))
            stages.append(_ProtraitPoolingLayer(d_model, pool_degree=2))
            length = length // 2
        self.stages = nn.ModuleList(stages)

        self.bottleneck_conv = nn.Conv1d(d_model, d_model // 2, kernel_size=1)
        self.bottleneck_norm = nn.BatchNorm1d(d_model // 2)
        self.bottleneck_act = nn.ELU()
        self.bottleneck_pool = nn.MaxPool1d(kernel_size=2)
        flat_features = (d_model // 2) * (length // 2)
        self.bottleneck_linear = nn.Linear(flat_features, 32)
        self.bottleneck_ln = nn.LayerNorm(32)
        self.bottleneck_act2 = nn.ELU()

        self.prediction = nn.Linear(32, n_cells)

    def forward(self, x: Tensor) -> Tensor:
        """Predict a per-cell-type accessibility-probability vector from a
        one-hot DNA sequence, shape ``[batch, c_in, seq_len]``."""
        h = self.embedding(x) + self.positional_embedding()
        for stage in self.stages:
            h = stage(h)

        h = h.permute(0, 2, 1)
        h = self.bottleneck_act(self.bottleneck_norm(self.bottleneck_conv(h)))
        h = self.bottleneck_pool(h)
        h = h.flatten(start_dim=1)
        h = self.bottleneck_act2(self.bottleneck_ln(self.bottleneck_linear(h)))

        return torch.sigmoid(self.prediction(h))


def build_protrait() -> nn.Module:
    """Build a compact PROTRAIT-style ProbSparse-attention transformer.

    Returns
    -------
    nn.Module
        ``ProtraitAccessibility`` in eval mode.
    """
    model = ProtraitAccessibility()
    model.eval()
    return model


def example_input_protrait() -> Tensor:
    """One-hot DNA sequence batch, shape ``[batch, 4, 96]``."""
    torch.manual_seed(0)
    idx = torch.randint(0, 4, (2, 96))
    return F.one_hot(idx, num_classes=4).permute(0, 2, 1).float()


MENAGERIE_ENTRIES = [
    ("PARM", "build_parm", "example_input_parm", "2026", "BIO"),
    ("PlasFlow", "build_plasflow", "example_input_plasflow", "2018", "BIO"),
    ("Portal", "build_portal", "example_input_portal", "2022", "BIO"),
    ("PrismNet", "build_prismnet", "example_input_prismnet", "2021", "BIO"),
    ("PROTRAIT", "build_protrait", "example_input_protrait", "2023", "BIO"),
]
