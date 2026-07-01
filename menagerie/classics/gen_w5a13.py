"""Compact faithful classics for six bioinformatics/EEG architectures.

Sources checked (repo code inspected via GitHub API, base env only, no clone
or pip install):
  - TaxoNN: https://github.com/divya031090/taxoNN_OTU (``NN_Cirr.py``,
    ``ensembling_Cirr.py``). Sharma, Xu, Sharma & Datta, Bioinformatics 2020,
    "TaxoNN: ensemble of neural networks on stratified microbiome data for
    disease prediction". Original code is a raw TensorFlow-1 graph (per-
    phylum ``conv_layer`` in ``NN_Cirr.py``, one 1D-conv-over-OTU-abundance
    CNN branch trained separately per bacterial phylum, one Keras ``.h5``
    file each) plus a Keras stacking ensemble (``ensembling_Cirr.py``,
    ``define_stacked_model``: freeze all per-phylum CNN branches, concatenate
    their softmax outputs, and fit a small dense meta-classifier on top).
    The distinctive architecture-level idea -- independent of the raw
    TF1 graph plumbing -- is this two-stage "stratify-by-phylum, ensemble
    the branches" topology: reproduced here as ``TaxoNN`` with one
    ``PhylumCNNBranch`` (conv1d -> conv1d -> dense, matching ``conv_layer``'s
    two-conv-then-dense stack) per phylum, whose per-branch logits are
    concatenated and passed through a small dense meta-model (matching
    ``define_stacked_model``'s ``Dense(10) -> Dense(n_classes)`` head).
  - TcellMatch: https://github.com/theislab/tcellmatch
    (``tcellmatch/models/models_ffn.py`` ``ModelBiRnn``/``ModelSa``,
    ``tcellmatch/models/layers/layer_aa_embedding.py``
    ``LayerAaEmbedding``, ``tcellmatch/models/layers/layer_attention.py``
    ``LayerMultiheadSelfAttention``). Fischer, Yang, Walzthoeni, Vazquez-
    Garcia, Chevrier & Theis, bioRxiv 2020.11.28.403634, "TcellMatch: a deep
    learning framework for predicting T cell receptor -- antigen binding".
    Original is TensorFlow/Keras. The distinctive mechanism is: one-hot
    amino-acid sequences of the TCR (CDR3) and peptide/epitope are
    concatenated along the sequence axis, projected through a shared 1x1-
    conv amino-acid embedding, and passed through a multi-headed
    self-attention stack (``LayerMultiheadSelfAttention``: separate Q/K/V
    dense projections, split into heads, residual connection, final dense)
    that lets TCR positions attend jointly over the peptide positions in one
    sequence before a dense binding-affinity head. Reimplemented faithfully
    in ``TorchTcellMatch`` with the same concat-then-self-attend-then-residual
    topology using ``nn.MultiheadAttention`` as the traced self-attention op.
  - TCNet-Fusion: https://github.com/Altaheri/EEG-ATCNet (``models.py``,
    ``TCNet_Fusion``/``TCN_block``/``EEGNet``), which vendors the original
    reference implementation credited to Ingolfsson et al. 2020
    (https://github.com/iis-eth-zurich/eeg-tcnet) and cites Musallam,
    AlFassam, Muhammad, Amin, Alsulaiman, Abdul, Altaheri, Bencherif &
    Algabri, Biomedical Signal Processing and Control 2021,
    "Electroencephalography-based motor imagery classification using
    temporal convolutional network fusion" (arXiv:2006.00622 companion
    EEG-TCNet paper). Distinctive mechanism: an EEGNet-style depthwise +
    separable spatial-temporal Conv2D front end over raw multichannel EEG,
    whose last-timestep feature map feeds a dilated-causal Temporal
    Convolutional Network (TCN) residual block stack; the "Fusion" part is
    that the EEGNet block-2 features, the TCN output, and the flattened
    EEGNet features are all concatenated (not just the TCN output, unlike
    plain EEGTCNet) before the final dense classifier. Reimplemented as
    ``TCNetFusion`` reproducing the EEGNet depthwise/separable conv front
    end, the dilated causal TCN residual stack, and the three-way
    concat-fusion head exactly as in the reference ``TCNet_Fusion()``.
  - TCR-BERT: https://github.com/wukevin/tcr-bert
    (``tcr/models/transformer_custom.py``, ``TwoPartBertClassifier``,
    ``TwoPartClassLogitsHead``). Wu, Guo, Fang & Ma, bioRxiv
    2021.11.18.469186, "TCR-BERT: learning the grammar of T-cell receptors
    for flexible antigen-binding analyses"; HuggingFace checkpoint
    ``wukevin/tcr-bert-mlm-only``. Distinctive mechanism: a standard BERT
    encoder (here instantiated compactly via ``transformers.BertModel``)
    pretrained on TCR CDR3 amino-acid sequences with an amino-acid-level
    vocabulary, finetuned with two encoder passes -- one over the TCR-alpha
    chain, one over the TCR-beta chain (``separate_encoders=True`` in the
    reference) -- whose pooled ``[CLS]`` embeddings are each projected
    through a small per-chain fully-connected layer and concatenated before
    a final classification head (``TwoPartClassLogitsHead``). Reimplemented
    as ``TcrBertTwoPart`` reproducing this two-encoder, two-part-head
    topology with a tiny ``BertModel`` (random init, no HF Hub download).
  - TITAN: https://github.com/mahmoodlab/TITAN (model weights gated on
    HuggingFace; architecture source vendored verbatim by downstream users,
    e.g. https://github.com/DIAGNijmegen/unicorn_baseline
    ``src/unicorn_baseline/vision/pathology/titan/vision_transformer.py``,
    ``VisionTransformer``/``Attention``/``get_alibi``). Ding, Vaidya, Zhang
    et al., Nature Medicine 2025 (also arXiv:2411.19666), "A multimodal
    whole-slide foundation model for pathology". Distinctive mechanism:
    TITAN's slide encoder ingests a *grid* of pre-extracted per-patch
    feature vectors (positioned by their 2D coordinates on the whole-slide
    image, not raw pixels), embeds them with an MLP patch-embed (not a conv
    patchify stem, since inputs are already features), and runs a ViT
    encoder whose self-attention uses a 2D-Euclidean-distance ALiBi
    positional bias (``get_alibi``: per-head linear-slope penalty on
    pairwise grid distance, added to the attention logits in place of
    learned/absolute position embeddings) with a prepended CLS token pooled
    at the output. Reimplemented as ``TitanSlideEncoder``, reproducing the
    feature-grid MLP patch-embed, 2D-distance ALiBi attention bias (computed
    fresh per forward call for traceability), and CLS-pooled ViT block
    stack. The CONCH v1.5 patch encoder and the paired text encoder /
    captioning decoder are separate models outside this candidate's scope
    (patch features are the traced module's *input*, matching the released
    public-weights TITAN, whose captioning decoder was withheld).
  - TranceptEVE: https://github.com/OATML-Markslab/Tranception
    (``tranception/model_pytorch.py``, ``SpatialDepthWiseConvolution``,
    ``TranceptionBlockAttention``, ``TranceptionLMHeadModel``). Notin,
    Dias, Frazer, Marchena-Hurtado, Gomez, Marks & Gal, ICML 2022,
    "Tranception: Protein Fitness Prediction with Autoregressive
    Transformers and Inference-time Retrieval"; TranceptEVE extension
    (Notin et al., bioRxiv 2022.12.07.519495, "TranceptEVE: Combining
    Family-specific and Family-agnostic Models of Protein Sequences for
    Improved Fitness Prediction") fuses Tranception's autoregressive
    protein language model with retrieval from an EVE family-specific
    MSA variational autoencoder (log-prior blending, see
    ``TranceptionLMHeadModel.forward``'s ``retrieval_aggregation_mode``
    path, and the EVE model referenced therein,
    https://github.com/OATML-Markslab/EVE). Distinctive mechanism (1):
    Tranception's causal self-attention splits attention heads into four
    equal groups and applies grouped depthwise ``Conv1d`` (kernel sizes
    1, 3, 5, 7) independently to the Q/K/V projections of each group before
    the causal dot-product attention (``SpatialDepthWiseConvolution``,
    ``TranceptionBlockAttention.forward``'s ``attention_mode=="tranception"``
    branch) -- a multi-scale local/global mixture-of-convolutions built
    directly into self-attention. Distinctive mechanism (2): EVE is a
    Bayesian VAE over MSA columns (one-hot amino-acid matrix in, per-column
    categorical reconstruction out) whose ELBO log-probability acts as a
    family-specific "retrieval" prior that TranceptEVE log-linearly blends
    with the autoregressive LM's per-token log-probabilities. Reimplemented
    as ``TranceptionBlock`` (GPT-2-style causal transformer block with the
    grouped multi-scale depthwise-conv attention, traced end-to-end as an
    autoregressive protein LM) and ``EveMsaVae`` (compact MSA-VAE with a
    diagonal-Gaussian encoder and per-position categorical decoder over
    amino-acid one-hot columns), composed in ``TranceptEVE`` which blends
    the LM's log-softmax with the VAE's reconstruction log-probability
    exactly as ``retrieval_aggregation_mode=="aggregate_substitution"``
    does (convex combination of the two per-position log-distributions).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import BertConfig, BertModel


# --------------------------------------------------------------------------
# TaxoNN
# --------------------------------------------------------------------------


class PhylumCNNBranch(nn.Module):
    """One per-phylum 1D-CNN branch (matches ``conv_layer`` x2 + dense)."""

    def __init__(self, n_otus: int, n_classes: int, conv1: int = 8, conv2: int = 16) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, conv1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(conv1, conv2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        pooled_len = n_otus // 4
        self.fc = nn.Linear(conv2 * max(pooled_len, 1), 32)
        self.out = nn.Linear(32, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Run one phylum's OTU-abundance vector through the branch.

        Parameters
        ----------
        x : Tensor
            Shape ``(batch, n_otus)`` relative-abundance features for OTUs
            belonging to this phylum.

        Returns
        -------
        Tensor
            Class logits of shape ``(batch, n_classes)``.
        """
        h = x.unsqueeze(1)
        h = self.pool(F.relu(self.conv1(h)))
        h = self.pool(F.relu(self.conv2(h)))
        h = h.flatten(1)
        h = F.relu(self.fc(h))
        return self.out(h)


class TaxoNN(nn.Module):
    """Ensemble of per-phylum CNN branches with a dense stacking meta-model."""

    def __init__(
        self, otus_per_phylum: tuple[int, ...] = (40, 35, 30, 25, 20), n_classes: int = 2
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList([PhylumCNNBranch(n, n_classes) for n in otus_per_phylum])
        self.meta_hidden = nn.Linear(n_classes * len(otus_per_phylum), 10)
        self.meta_out = nn.Linear(10, n_classes)

    def forward(self, phyla: tuple[Tensor, ...]) -> Tensor:
        """Stratify by phylum, run each CNN branch, then stack.

        Parameters
        ----------
        phyla : tuple[Tensor, ...]
            One OTU-abundance tensor of shape ``(batch, n_otus_p)`` per
            phylum, matching ``otus_per_phylum``.

        Returns
        -------
        Tensor
            Final disease-status class logits of shape ``(batch, n_classes)``.
        """
        branch_logits = [branch(p) for branch, p in zip(self.branches, phyla, strict=True)]
        merged = torch.cat(branch_logits, dim=-1)
        hidden = F.relu(self.meta_hidden(merged))
        return self.meta_out(hidden)


def build_taxonn() -> nn.Module:
    """Build a small TaxoNN ensemble (5 phyla, binary disease status)."""
    return TaxoNN().eval()


def example_input_taxonn() -> tuple[Tensor, ...]:
    """Example per-phylum OTU-abundance batch for :func:`build_taxonn`."""
    otus_per_phylum = (40, 35, 30, 25, 20)
    return tuple(torch.rand(4, n) for n in otus_per_phylum)


# --------------------------------------------------------------------------
# TcellMatch
# --------------------------------------------------------------------------


class TorchTcellMatch(nn.Module):
    """Self-attention over concatenated TCR CDR3 + peptide amino-acid sequence."""

    def __init__(
        self,
        n_aa: int = 21,
        tcr_len: int = 20,
        pep_len: int = 12,
        embed_dim: int = 32,
        n_heads: int = 4,
        n_covariates: int = 4,
        labels_dim: int = 1,
    ) -> None:
        super().__init__()
        self.tcr_len = tcr_len
        self.pep_len = pep_len
        # 1x1-conv amino acid embedding (matches LayerAaEmbedding).
        self.aa_embed = nn.Conv1d(n_aa, embed_dim, kernel_size=1)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.final_dense = nn.Linear(embed_dim, embed_dim)
        seq_len = tcr_len + pep_len
        self.covariate_proj = nn.Linear(n_covariates, embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * seq_len + embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, labels_dim),
        )

    def forward(self, tcr_onehot: Tensor, pep_onehot: Tensor, covariates: Tensor) -> Tensor:
        """Predict TCR-peptide binding strength.

        Parameters
        ----------
        tcr_onehot : Tensor
            One-hot TCR CDR3 sequence, shape ``(batch, tcr_len, n_aa)``.
        pep_onehot : Tensor
            One-hot peptide/epitope sequence, shape ``(batch, pep_len, n_aa)``.
        covariates : Tensor
            Cell/assay covariate features, shape ``(batch, n_covariates)``.

        Returns
        -------
        Tensor
            Predicted binding-strength logits, shape ``(batch, labels_dim)``.
        """
        seq = torch.cat([tcr_onehot, pep_onehot], dim=1)  # (batch, tcr_len+pep_len, n_aa)
        x = 2.0 * (seq.transpose(1, 2) - 0.5)  # centering, matches reference preprocessing
        x = self.aa_embed(x).transpose(1, 2)  # (batch, seq_len, embed_dim)

        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.attn_norm(x + F.relu(self.final_dense(attn_out)))  # residual self-attention block

        flat = x.flatten(1)
        cov = self.covariate_proj(covariates)
        merged = torch.cat([flat, cov], dim=-1)
        return self.head(merged)


def build_tcellmatch() -> nn.Module:
    """Build a small TcellMatch self-attention binding-affinity model."""
    return TorchTcellMatch().eval()


def example_input_tcellmatch() -> tuple[Tensor, Tensor, Tensor]:
    """Example (TCR one-hot, peptide one-hot, covariates) for :func:`build_tcellmatch`."""
    n_aa, tcr_len, pep_len = 21, 20, 12
    tcr = F.one_hot(torch.randint(0, n_aa, (4, tcr_len)), n_aa).float()
    pep = F.one_hot(torch.randint(0, n_aa, (4, pep_len)), n_aa).float()
    covariates = torch.rand(4, 4)
    return tcr, pep, covariates


# --------------------------------------------------------------------------
# TCNet-Fusion
# --------------------------------------------------------------------------


class EEGNetFrontEnd(nn.Module):
    """EEGNet depthwise + separable spatiotemporal front end."""

    def __init__(
        self, chans: int, f1: int = 8, depth: int = 2, kern_length: int = 32, dropout: float = 0.3
    ) -> None:
        super().__init__()
        f2 = f1 * depth
        self.temporal_conv = nn.Conv2d(
            1, f1, (1, kern_length), padding=(0, kern_length // 2), bias=False
        )
        self.temporal_bn = nn.BatchNorm2d(f1)
        self.depthwise_conv = nn.Conv2d(f1, f1 * depth, (chans, 1), groups=f1, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(f1 * depth)
        self.pool1 = nn.AvgPool2d((1, 8))
        self.dropout1 = nn.Dropout(dropout)
        self.separable_conv = nn.Conv2d(f2, f2, (1, 16), padding=(0, 8), groups=f2, bias=False)
        self.separable_point = nn.Conv2d(f2, f2, 1, bias=False)
        self.separable_bn = nn.BatchNorm2d(f2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Run the EEGNet stem.

        Parameters
        ----------
        x : Tensor
            Raw EEG, shape ``(batch, 1, chans, samples)``.

        Returns
        -------
        Tensor
            Feature map of shape ``(batch, f2, 1, samples // 64)``.
        """
        h = self.temporal_bn(self.temporal_conv(x))
        h = F.elu(self.depthwise_bn(self.depthwise_conv(h)))
        h = self.dropout1(self.pool1(h))
        h = self.separable_point(self.separable_conv(h))
        h = F.elu(self.separable_bn(h))
        return self.dropout2(self.pool2(h))


class TcnResidualBlock(nn.Module):
    """One dilated-causal TCN residual block (matches ``TCN_block`` iteration)."""

    def __init__(
        self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float
    ) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation, padding=pad)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.pad = pad

    def _causal_trim(self, x: Tensor) -> Tensor:
        return x[:, :, : -self.pad] if self.pad > 0 else x

    def forward(self, x: Tensor) -> Tensor:
        """Apply one causal dilated-conv residual block.

        Parameters
        ----------
        x : Tensor
            Shape ``(batch, in_ch, length)``.

        Returns
        -------
        Tensor
            Shape ``(batch, out_ch, length)``.
        """
        h = self._causal_trim(self.conv1(x))
        h = self.dropout(F.elu(self.bn1(h)))
        h = self._causal_trim(self.conv2(h))
        h = self.dropout(F.elu(self.bn2(h)))
        residual = x if self.downsample is None else self.downsample(x)
        return F.elu(h + residual)


class TCNetFusion(nn.Module):
    """EEGNet front end + dilated-causal TCN, three-way fusion classifier head."""

    def __init__(
        self,
        n_classes: int = 4,
        chans: int = 22,
        samples: int = 256,
        f1: int = 8,
        depth: int = 2,
        tcn_filters: int = 12,
        tcn_layers: int = 2,
        kernel_size: int = 4,
    ) -> None:
        super().__init__()
        self.eegnet = EEGNetFrontEnd(chans, f1=f1, depth=depth)
        f2 = f1 * depth
        tcn_len = samples // 64
        blocks = []
        in_ch = f2
        for i in range(tcn_layers):
            blocks.append(
                TcnResidualBlock(in_ch, tcn_filters, kernel_size, dilation=2**i, dropout=0.3)
            )
            in_ch = tcn_filters
        self.tcn = nn.Sequential(*blocks)
        fusion_dim = f2 * tcn_len + tcn_filters * tcn_len + f2 * tcn_len
        self.classifier = nn.Linear(fusion_dim, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Classify a raw multichannel EEG trial.

        Parameters
        ----------
        x : Tensor
            Raw EEG, shape ``(batch, 1, chans, samples)``.

        Returns
        -------
        Tensor
            Class logits of shape ``(batch, n_classes)``.
        """
        eeg_feat = self.eegnet(x).squeeze(2)  # (batch, f2, tcn_len)
        tcn_out = self.tcn(eeg_feat)  # (batch, tcn_filters, tcn_len)
        fused = torch.cat([eeg_feat, tcn_out, eeg_feat], dim=1).flatten(1)
        return self.classifier(fused)


def build_tcnet_fusion() -> nn.Module:
    """Build a small TCNet-Fusion motor-imagery EEG classifier."""
    return TCNetFusion().eval()


def example_input_tcnet_fusion() -> Tensor:
    """Example raw EEG batch for :func:`build_tcnet_fusion`."""
    return torch.randn(4, 1, 22, 256)


# --------------------------------------------------------------------------
# TCR-BERT
# --------------------------------------------------------------------------


class TwoPartClassLogitsHead(nn.Module):
    """Per-chain FC projections concatenated into a final classifier."""

    def __init__(
        self, a_dim: int, b_dim: int, n_out: int, hidden: int = 32, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc_a = nn.Linear(a_dim, hidden)
        self.fc_b = nn.Linear(b_dim, hidden)
        self.final_fc = nn.Linear(hidden * 2, n_out)

    def forward(self, a_enc: Tensor, b_enc: Tensor) -> Tensor:
        """Combine pooled alpha- and beta-chain encodings into class logits."""
        a = F.relu(self.fc_a(self.dropout(a_enc)))
        b = F.relu(self.fc_b(self.dropout(b_enc)))
        return self.final_fc(torch.cat([a, b], dim=-1))


class TcrBertTwoPart(nn.Module):
    """Two-encoder BERT (TCR-alpha, TCR-beta) with a two-part classifier head."""

    def __init__(
        self,
        vocab_size: int = 25,
        hidden_size: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        n_classes: int = 2,
    ) -> None:
        super().__init__()
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=32,
        )
        self.encoder_a = BertModel(config)
        self.encoder_b = BertModel(config)
        self.head = TwoPartClassLogitsHead(hidden_size, hidden_size, n_classes)

    def forward(self, tra_ids: Tensor, trb_ids: Tensor) -> Tensor:
        """Predict a class from paired TCR-alpha/beta CDR3 token sequences.

        Parameters
        ----------
        tra_ids : Tensor
            TCR-alpha amino-acid token ids, shape ``(batch, seq_len)``.
        trb_ids : Tensor
            TCR-beta amino-acid token ids, shape ``(batch, seq_len)``.

        Returns
        -------
        Tensor
            Class logits of shape ``(batch, n_classes)``.
        """
        a_pooled = self.encoder_a(input_ids=tra_ids).pooler_output
        b_pooled = self.encoder_b(input_ids=trb_ids).pooler_output
        return self.head(a_pooled, b_pooled)


def build_tcr_bert() -> nn.Module:
    """Build a tiny two-chain TCR-BERT classifier (random init, no HF download)."""
    return TcrBertTwoPart().eval()


def example_input_tcr_bert() -> tuple[Tensor, Tensor]:
    """Example (TCR-alpha ids, TCR-beta ids) for :func:`build_tcr_bert`."""
    tra_ids = torch.randint(0, 25, (4, 16))
    trb_ids = torch.randint(0, 25, (4, 16))
    return tra_ids, trb_ids


# --------------------------------------------------------------------------
# TITAN
# --------------------------------------------------------------------------


def _alibi_slopes(n_heads: int) -> Tensor:
    """Compute the standard power-of-two ALiBi per-head slopes."""

    def _slopes_pow2(n: int) -> list[float]:
        start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3)))
        return [start * (start**i) for i in range(n)]

    if math.log2(n_heads).is_integer():
        return torch.tensor(_slopes_pow2(n_heads))
    nearest = 2 ** math.floor(math.log2(n_heads))
    base = _slopes_pow2(nearest)
    extra = _slopes_pow2(2 * nearest)[0::2][: n_heads - nearest]
    return torch.tensor(base + extra)


class AlibiAttention(nn.Module):
    """Self-attention with a 2D-Euclidean-distance ALiBi bias on the grid."""

    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.register_buffer("slopes", _alibi_slopes(n_heads).view(n_heads, 1, 1), persistent=False)

    def forward(self, x: Tensor, alibi_bias: Tensor) -> Tensor:
        """Apply multi-head self-attention with an additive ALiBi bias.

        Parameters
        ----------
        x : Tensor
            Token sequence (CLS + patch grid), shape ``(batch, n, dim)``.
        alibi_bias : Tensor
            Additive attention bias, shape ``(n_heads, n, n)``.

        Returns
        -------
        Tensor
            Shape ``(batch, n, dim)``.
        """
        batch, n, dim = x.shape
        qkv = self.qkv(x).reshape(batch, n, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = torch.matmul(q, k.transpose(-1, -2)) * (self.head_dim**-0.5)
        attn = attn + alibi_bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(batch, n, dim)
        return self.proj(out)


class AlibiBlock(nn.Module):
    """Pre-norm transformer block using :class:`AlibiAttention`."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = AlibiAttention(dim, n_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x: Tensor, alibi_bias: Tensor) -> Tensor:
        """Apply one ALiBi self-attention block with residual MLP."""
        x = x + self.attn(self.norm1(x), alibi_bias)
        x = x + self.mlp(self.norm2(x))
        return x


class TitanSlideEncoder(nn.Module):
    """MLP-patch-embed ViT with 2D-distance ALiBi bias over a patch-feature grid."""

    def __init__(
        self, patch_feat_dim: int = 64, embed_dim: int = 96, depth: int = 2, n_heads: int = 4
    ) -> None:
        super().__init__()
        self.patch_embed = nn.Sequential(nn.Linear(patch_feat_dim, embed_dim), nn.GELU())
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([AlibiBlock(embed_dim, n_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.n_heads = n_heads

    def _alibi_bias(self, coords: Tensor) -> Tensor:
        """Build the 2D-distance ALiBi additive bias, prefixed for the CLS token."""
        diffs = coords.unsqueeze(1) - coords.unsqueeze(0)
        dists = torch.sqrt((diffs.float() ** 2).sum(-1) + 1e-12)
        slopes = _alibi_slopes(self.n_heads).to(coords.device).view(self.n_heads, 1, 1)
        bias_patches = -dists.unsqueeze(0) * slopes  # (n_heads, n_patches, n_patches)
        n = coords.shape[0] + 1
        full = torch.zeros(self.n_heads, n, n, device=coords.device, dtype=bias_patches.dtype)
        full[:, 1:, 1:] = bias_patches
        return full

    def forward(self, patch_features: Tensor, patch_coords: Tensor) -> Tensor:
        """Encode a whole-slide-image patch-feature grid into a slide embedding.

        Parameters
        ----------
        patch_features : Tensor
            Pre-extracted per-patch features, shape ``(batch, n_patches, patch_feat_dim)``.
        patch_coords : Tensor
            Integer grid coordinates per patch, shape ``(n_patches, 2)`` (shared
            across the batch, matching the reference's single-slide grid layout).

        Returns
        -------
        Tensor
            CLS-pooled slide embedding, shape ``(batch, embed_dim)``.
        """
        batch = patch_features.shape[0]
        tokens = self.patch_embed(patch_features)
        cls = self.cls_token.expand(batch, -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        alibi_bias = self._alibi_bias(patch_coords)
        for block in self.blocks:
            x = block(x, alibi_bias)
        x = self.norm(x)
        return x[:, 0]


def build_titan() -> nn.Module:
    """Build a small TITAN-style ALiBi slide encoder over patch features."""
    return TitanSlideEncoder().eval()


def example_input_titan() -> tuple[Tensor, Tensor]:
    """Example (patch features, patch grid coordinates) for :func:`build_titan`."""
    n_patches = 36
    patch_features = torch.randn(2, n_patches, 64)
    grid = torch.stack(
        torch.meshgrid(torch.arange(6), torch.arange(6), indexing="ij"), dim=-1
    ).reshape(-1, 2)
    return patch_features, grid


# --------------------------------------------------------------------------
# TranceptEVE
# --------------------------------------------------------------------------


class SpatialDepthWiseConvolution(nn.Module):
    """Causal grouped depthwise Conv1d applied along the sequence axis per head."""

    def __init__(self, head_dim: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            head_dim, head_dim, kernel_size, padding=kernel_size - 1, groups=head_dim
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the depthwise conv to a group of attention heads.

        Parameters
        ----------
        x : Tensor
            Shape ``(batch, heads, seq_len, head_dim)``.

        Returns
        -------
        Tensor
            Shape ``(batch, heads, seq_len, head_dim)``.
        """
        batch, heads, seq_len, head_dim = x.shape
        x = x.permute(0, 1, 3, 2).reshape(batch * heads, head_dim, seq_len)
        x = self.conv(x)
        if self.kernel_size > 1:
            x = x[:, :, : -(self.kernel_size - 1)]
        x = x.view(batch, heads, head_dim, seq_len).permute(0, 1, 3, 2)
        return x


class TranceptionAttention(nn.Module):
    """Causal self-attention with multi-scale depthwise-conv Q/K/V head groups."""

    causal_mask: Tensor

    def __init__(self, dim: int, n_heads: int, max_len: int) -> None:
        super().__init__()
        assert n_heads % 4 == 0, "Tranception requires num_heads to be a multiple of 4."
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.heads_per_group = n_heads // 4
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.q_conv = nn.ModuleDict(
            {str(i): SpatialDepthWiseConvolution(self.head_dim, k) for i, k in enumerate([3, 5, 7])}
        )
        self.k_conv = nn.ModuleDict(
            {str(i): SpatialDepthWiseConvolution(self.head_dim, k) for i, k in enumerate([3, 5, 7])}
        )
        self.v_conv = nn.ModuleDict(
            {str(i): SpatialDepthWiseConvolution(self.head_dim, k) for i, k in enumerate([3, 5, 7])}
        )
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_len, max_len, dtype=torch.bool)),
            persistent=False,
        )

    def _apply_multiscale(self, t: Tensor, conv_dict: nn.ModuleDict) -> Tensor:
        g = self.heads_per_group
        parts = [t[:, :g]]
        for i in range(3):
            parts.append(conv_dict[str(i)](t[:, (i + 1) * g : (i + 2) * g]))
        return torch.cat(parts, dim=1)

    def forward(self, x: Tensor) -> Tensor:
        """Apply grouped multi-scale causal self-attention.

        Parameters
        ----------
        x : Tensor
            Shape ``(batch, seq_len, dim)``.

        Returns
        -------
        Tensor
            Shape ``(batch, seq_len, dim)``.
        """
        batch, seq_len, dim = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch, seq_len, 3, self.n_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q = self._apply_multiscale(q, self.q_conv)
        k = self._apply_multiscale(k, self.k_conv)
        v = self._apply_multiscale(v, self.v_conv)

        attn = torch.matmul(q, k.transpose(-1, -2)) * (self.head_dim**-0.5)
        mask = self.causal_mask[:seq_len, :seq_len]
        attn = attn.masked_fill(~mask, float("-inf"))
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(batch, seq_len, dim)
        return self.proj(out)


class TranceptionBlock(nn.Module):
    """GPT-2-style causal transformer block using :class:`TranceptionAttention`."""

    def __init__(self, dim: int, n_heads: int, max_len: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TranceptionAttention(dim, n_heads, max_len)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x: Tensor) -> Tensor:
        """Apply one causal Tranception block with pre-norm residuals."""
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class EveMsaVae(nn.Module):
    """Compact MSA variational autoencoder (EVE): per-column categorical VAE."""

    def __init__(
        self, msa_len: int, n_aa: int = 21, latent_dim: int = 16, hidden: int = 64
    ) -> None:
        super().__init__()
        self.msa_len = msa_len
        self.n_aa = n_aa
        in_dim = msa_len * n_aa
        self.enc = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(), nn.Linear(hidden, in_dim)
        )

    def forward(self, msa_onehot: Tensor) -> Tensor:
        """Encode an MSA one-hot column matrix and return per-position log-probs.

        Parameters
        ----------
        msa_onehot : Tensor
            One-hot amino-acid MSA columns for one sequence, shape
            ``(batch, msa_len, n_aa)``.

        Returns
        -------
        Tensor
            Per-position log-probability of each amino acid under the VAE
            reconstruction, shape ``(batch, msa_len, n_aa)``.
        """
        batch = msa_onehot.shape[0]
        flat = msa_onehot.flatten(1)
        h = self.enc(flat)
        mu = self.mu(h)
        logvar = self.logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        recon = self.dec(z).view(batch, self.msa_len, self.n_aa)
        return F.log_softmax(recon, dim=-1)


class TranceptEVE(nn.Module):
    """Tranception autoregressive protein LM fused with an EVE MSA-VAE retrieval prior."""

    def __init__(
        self,
        vocab_size: int = 25,
        dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        max_len: int = 64,
        retrieval_weight: float = 0.6,
    ) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_len, dim)
        self.blocks = nn.ModuleList(
            [TranceptionBlock(dim, n_heads, max_len) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size)
        self.eve = EveMsaVae(msa_len=max_len, n_aa=vocab_size)
        self.retrieval_weight = retrieval_weight

    def forward(self, token_ids: Tensor, msa_onehot: Tensor) -> Tensor:
        """Blend the autoregressive LM's log-probs with the EVE VAE's prior.

        Parameters
        ----------
        token_ids : Tensor
            Amino-acid token ids of the query protein sequence, shape
            ``(batch, seq_len)``.
        msa_onehot : Tensor
            One-hot family MSA columns aligned to the same positions, shape
            ``(batch, seq_len, vocab_size)``.

        Returns
        -------
        Tensor
            Fused per-position log-probabilities over the amino-acid
            vocabulary, shape ``(batch, seq_len, vocab_size)``.
        """
        batch, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch, -1)
        x = self.token_embed(token_ids) + self.pos_embed(positions)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        lm_log_probs = F.log_softmax(self.lm_head(x), dim=-1)

        eve_log_probs = self.eve(msa_onehot)

        # Convex-combination log-prior blending, matching
        # ``aggregate_substitution`` in TranceptionLMHeadModel.forward.
        w = self.retrieval_weight
        fused = torch.logaddexp(
            math.log(1 - w) + lm_log_probs,
            math.log(w) + eve_log_probs,
        )
        return fused - torch.logsumexp(fused, dim=-1, keepdim=True)


def build_tranception_eve() -> nn.Module:
    """Build a small TranceptEVE (Tranception LM + EVE MSA-VAE fusion) model."""
    return TranceptEVE().eval()


def example_input_tranception_eve() -> tuple[Tensor, Tensor]:
    """Example (query token ids, aligned MSA one-hot) for :func:`build_tranception_eve`."""
    vocab_size, seq_len = 25, 64
    token_ids = torch.randint(0, vocab_size, (2, seq_len))
    msa_onehot = F.one_hot(torch.randint(0, vocab_size, (2, seq_len)), vocab_size).float()
    return token_ids, msa_onehot


MENAGERIE_ENTRIES = [
    ("TaxoNN", "build_taxonn", "example_input_taxonn", "2020", "BIO"),
    ("TcellMatch", "build_tcellmatch", "example_input_tcellmatch", "2020", "BIO"),
    ("TCNet-Fusion", "build_tcnet_fusion", "example_input_tcnet_fusion", "2021", "BIO"),
    ("TCR-BERT", "build_tcr_bert", "example_input_tcr_bert", "2021", "BIO"),
    ("TITAN", "build_titan", "example_input_titan", "2025", "VIS"),
    ("TranceptEVE", "build_tranception_eve", "example_input_tranception_eve", "2022", "BIO"),
]
