"""Biomedical / bioinformatics classics (batch w4a21).

Sources checked (repo/paper architecture; no clone, no pip install --
reimplemented from scratch in base-env torch):

- DAGAN (De-Aliasing Generative Adversarial Network): Yang, Yu, Zhang, Fu, Xu,
  Slabaugh, Ye, Liu, Firmin, Keegan, Guo & Yang, IEEE TMI 2018,
  arXiv:1801.07198. Official TensorLayer repo
  https://github.com/tensorlayer/DAGAN ; community PyTorch reimplementation
  https://github.com/kaaarho/DAGAN_PyTorch (``model.py`` -- ``UNet`` +
  ``Discriminator``), used here as the reference for tracing (the official
  repo is TF1/TensorLayer). DAGAN removes MRI undersampling artifacts with a
  conditional-GAN generator: a U-Net (8 downsampling conv layers / 8
  transposed-conv upsampling layers with encoder-decoder skip
  concatenations, matching the official ``UNet.forward``) that predicts a
  *residual* correction added back onto the input aliased image and clamped
  to the valid intensity range (the ``is_refine`` branch: ``output =
  clamp(output + input, -1, 1)``) -- refinement-via-residual is DAGAN's
  namesake "de-aliasing" mechanism, distinct from a plain image-to-image
  U-Net that predicts pixels directly. The discriminator is a strided-conv
  stack ending in one residual bottleneck block (``res8``: 1x1 reduce -> 3x3
  -> 3x3 expand, added back to its input) before the real/fake sigmoid head.
  Reimplemented here with a compact U-Net (4 down/4 up stages, real skip
  concatenations, real residual-refinement output) and a compact
  discriminator with the same conv-stack + residual-bottleneck + sigmoid
  head topology.

- Deep Cox Mixtures (DCM): Nagpal, Yadlowsky, Rostamzadeh & Heller, MLHC 2021,
  arXiv:2101.06536. The paper's own repo
  https://github.com/chiragnagpal/deep_cox_mixtures ships only the paper PDF;
  the maintained official implementation lives in the same group's
  ``auton-survival`` package,
  https://github.com/autonlab/auton-survival/blob/master/auton_survival/models/dcm/dcm_torch.py
  (class ``DeepCoxMixturesTorch``). DCM assumes the population is a latent
  mixture of ``k`` Cox proportional-hazards subgroups: a shared MLP
  "embedding" trunk feeds two linear heads from the *same* representation --
  a ``gate`` head producing per-subgroup mixture-assignment log-probabilities
  via log-softmax, and an ``expert`` head producing per-subgroup log
  hazard-ratios (clamped to ``[-gamma, gamma]``, or optionally
  ``gamma * tanh(...)``) -- so each individual gets ``k`` competing
  proportional-hazards models plus a soft gate over them, rather than one
  single global Cox model. Reimplemented here exactly as
  ``DeepCoxMixturesTorch.forward``: shared embedding trunk, linear gate with
  log-softmax, linear expert with clamped log hazard ratios, both returned.

- DeepAC (DeePaC): Bartoszewicz, Seidel, Rentzsch & Renard, Bioinformatics
  2020, https://doi.org/10.1093/bioinformatics/btz541. Official repo
  https://github.com/JakubBartoszewicz/DeePaC (``deepac/nn_train.py`` --
  class ``RCNet``, methods ``_build_rc_model`` / ``_add_rc_conv1d`` /
  ``_add_rc_merge_dense``). DeePaC classifies pathogenic vs. non-pathogenic
  short reads directly from one-hot DNA using a *reverse-complement (RC)
  weight-sharing* CNN: the same convolution kernel is applied both to the
  forward one-hot sequence and to its reverse-complement (sequence-order
  reversed AND channel-order reversed, since complementing a base is a fixed
  permutation of the ACGT channel axis) -- exactly the official
  ``_add_rc_conv1d``, which reverses the input on both the length and
  channel axes, applies one shared ``Conv1d``, reverses the RC branch's
  output back, and concatenates the two feature maps -- so a single set of
  learned motif filters is guaranteed to fire identically regardless of
  which strand the read came from. After global pooling, a shared ``Dense``
  layer is applied to the forward and (length-reversed) RC halves and the
  two scores are merged additively (``_add_rc_merge_dense``, official
  default ``merge_function=add``), before a final classification head.
  Reimplemented here as an explicit ``RCConv1d`` module (weight-shared conv
  applied to both orientations via ``flip`` on length+channel axes) followed
  by global max pooling, a shared RC-merge dense layer, and a sigmoid
  classification head.

- DeepAffinity: Karimi, Wu, Wang & Shen, Bioinformatics 2019,
  https://academic.oup.com/bioinformatics/article/35/14/2329/5232939.
  Official repo https://github.com/Shen-Lab/DeepAffinity
  (``Joint_models/joint_attention/joint_warm_start/joint-Model.py``). Predicts
  compound-protein binding affinity from SMILES (compound) and SPS (protein
  secondary-structure) token sequences using two independent stacked-GRU
  encoders (drug GRU, protein GRU), followed by a *pairwise co-attention*: a
  bilinear compatibility score ``V @ W`` between every protein position and
  every drug position produces an attention matrix (softmax over the
  flattened pairwise scores, the official ``alphas_pair``), which then
  re-weights a second bilinear projection of both streams to build one
  pooled joint-interaction vector (the official ``Attn`` accumulation loop),
  fed through a small 1-D conv + pooling head to the final affinity score.
  Reimplemented here with the same two independent GRU encoders, an explicit
  bilinear pairwise-attention matrix (softmax-normalized over both sequence
  axes) used to pool a joint representation, and a matching conv+pool
  regression head -- the pairwise bilinear co-attention over two independent
  sequence encoders is DeepAffinity's distinguishing mechanism versus a
  plain single-sequence RNN/CNN affinity regressor.

- DeepBCR: Liu lab (DFCI) official repo
  https://github.com/liulab-dfci/DeepBCR (redirects to the source-of-record
  Bitbucket mirror, ``https://bitbucket.org/liulab/deepbcr``,
  ``src/deep_bcr.py`` -- classes ``EncodingLayerModel`` / ``GeneSwitchModel``
  / ``GeneSwitchModelFast`` / ``DeepBCR``). Classifies cancer type / estimates
  survival from a repertoire (many B-cell-receptor CDR3 k-mer sequences per
  patient, each tagged with per-isotype/constant-gene usage counts) using a
  distinctive "gene switch" motif-scanning architecture: (1) a learned
  per-amino-acid encoding matrix embeds every k-mer's residues (the official
  ``weights0`` gather), (2) a per-k-mer linear "motif" layer + ReLU scores
  each k-mer against ``num_motifs`` learned motif filters, (3) that
  per-k-mer motif-activation vector is *outer-producted with the k-mer's
  isotype/gene-usage count vector* (``scores x counts``, i.e. gating each
  motif score by which constant-region genes co-occur with it) before (4) a
  *max-pool over all k-mers in the repertoire* selects, per motif and per
  gene, the single most-activating k-mer in that patient's whole repertoire
  (the official ``max_pooling`` reduction over the k-mer axis) -- this
  repertoire-level max-pool over gene-gated motif scores is what lets the
  model use "class-switch" (isotype) patterns as a signal, not just CDR3
  sequence content. A final motif-layer (weighted sum over genes + ReLU) and
  linear output head produce the classification/hazard logits. Reimplemented
  here with the same amino-acid embedding gather, per-k-mer motif linear
  layer, gene-count gating via an outer product, and repertoire-level
  max-pool over k-mers, exactly mirroring ``GeneSwitchModel.hidden_layers``.

- DeepBGC: Hannigan, Prihoda, Palicka, Soukup, Klempir, Rampula, Durcak,
  Wurst, Kotowski, Chang, Wang, Piizzi, Temesi, Hazuda, Woelk & Bitton,
  Nucleic Acids Research 2019, https://doi.org/10.1093/nar/gkz654. Official
  Merck repo https://github.com/Merck/deepbgc (pip-installable as
  ``deepbgc``; ``deepbgc/models/rnn.py`` -- class ``KerasRNN._build_model``).
  Detects biosynthetic gene clusters (BGCs) directly from a genome's ordered
  sequence of protein-domain (Pfam) embeddings using a *stacked bidirectional
  LSTM with a per-timestep (TimeDistributed) sigmoid classification head*:
  every protein in the input ORF sequence gets its own BGC-membership
  probability, produced from a bidirectional context window over neighboring
  proteins rather than a single sequence-level label -- this per-residue
  (per-protein) dense sequence-tagging output, built on pre-trained
  ``pfam2vec`` domain embeddings, is DeepBGC's namesake mechanism.
  Reimplemented here as a bidirectional LSTM (optionally stacked) with a
  shared per-timestep linear+sigmoid head applied at every position of the
  input domain-embedding sequence.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# DAGAN
# ---------------------------------------------------------------------------


def _dagan_conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Build a strided conv + BatchNorm + LeakyReLU downsampling block."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


def _dagan_deconv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Build a transposed-conv + BatchNorm + ReLU upsampling block."""
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class DAGANUNet(nn.Module):
    """Compact DAGAN U-Net generator with residual "de-aliasing" refinement."""

    def __init__(self, base: int = 16) -> None:
        """Initialize a 4-stage encoder/decoder U-Net with skip connections.

        Parameters
        ----------
        base : int, default=16
            Base channel width of the first encoder stage.
        """
        super().__init__()
        self.down1 = _dagan_conv_block(1, base)
        self.down2 = _dagan_conv_block(base, base * 2)
        self.down3 = _dagan_conv_block(base * 2, base * 4)
        self.down4 = _dagan_conv_block(base * 4, base * 4)

        self.up4 = _dagan_deconv_block(base * 4, base * 4)
        self.up3 = _dagan_deconv_block(base * 8, base * 2)
        self.up2 = _dagan_deconv_block(base * 4, base)
        self.up1 = _dagan_deconv_block(base * 2, base)

        self.out = nn.Sequential(
            nn.Conv2d(base, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Encode, decode with skip concatenations, then residual-refine.

        Parameters
        ----------
        x : Tensor
            Aliased (undersampled-reconstruction) MRI image, shape
            ``(batch, 1, H, W)``.

        Returns
        -------
        Tensor
            De-aliased image, computed as ``clamp(residual + x, -1, 1)``.
        """
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        u4 = self.up4(d4)
        u3 = self.up3(torch.cat([d3, u4], dim=1))
        u2 = self.up2(torch.cat([d2, u3], dim=1))
        u1 = self.up1(torch.cat([d1, u2], dim=1))

        residual = self.out(u1)
        return torch.clamp(residual + x, min=-1.0, max=1.0)


class DAGANDiscriminator(nn.Module):
    """Compact DAGAN discriminator: strided convs + one residual bottleneck."""

    def __init__(self, base: int = 16) -> None:
        """Initialize the strided-conv stack and the residual bottleneck.

        Parameters
        ----------
        base : int, default=16
            Base channel width of the first conv stage.
        """
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(1, base, 5, stride=2, padding=2), nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(base, base * 2, 5, stride=2, padding=2),
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, 5, stride=2, padding=2),
            nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res_block = nn.Sequential(
            nn.Conv2d(base * 4, base, 1),
            nn.BatchNorm2d(base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base * 4, 3, padding=1),
            nn.BatchNorm2d(base * 4),
        )
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.out = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(base * 4, 1), nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run the strided conv stack, add the residual bottleneck, classify.

        Parameters
        ----------
        x : Tensor
            Real or generated image, shape ``(batch, 1, H, W)``.

        Returns
        -------
        Tensor
            Real/fake probability, shape ``(batch, 1)``.
        """
        h = self.conv2(self.conv1(self.conv0(x)))
        res = self.res_block(h)
        h = self.lrelu(res + h)
        return self.out(h)


def build_dagan() -> nn.Module:
    """Build a compact DAGAN U-Net de-aliasing generator.

    Returns
    -------
    nn.Module
        Random-initialized ``DAGANUNet`` in eval mode.
    """
    return DAGANUNet().eval()


def example_input_dagan() -> Tensor:
    """Create an example aliased MRI image.

    Returns
    -------
    Tensor
        Random image tensor of shape ``(1, 1, 64, 64)``.
    """
    return torch.randn(1, 1, 64, 64)


# ---------------------------------------------------------------------------
# Deep Cox Mixtures
# ---------------------------------------------------------------------------


class DeepCoxMixtures(nn.Module):
    """Compact Deep Cox Mixtures: shared MLP trunk + gate head + expert head."""

    def __init__(
        self, input_dim: int = 24, k: int = 3, hidden: int = 32, gamma: float = 5.0
    ) -> None:
        """Initialize the shared embedding trunk and the gate/expert heads.

        Parameters
        ----------
        input_dim : int, default=24
            Number of input covariates.
        k : int, default=3
            Number of latent Cox-mixture subgroups.
        hidden : int, default=32
            Hidden width of the shared embedding trunk.
        gamma : float, default=5.0
            Clamp bound for the per-subgroup log hazard ratios.
        """
        super().__init__()
        self.k = k
        self.gamma = gamma
        self.embedding = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU6())
        self.gate = nn.Linear(hidden, k, bias=False)
        self.expert = nn.Linear(hidden, k, bias=False)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Embed covariates and produce mixture-gate and expert hazard logits.

        Parameters
        ----------
        x : Tensor
            Covariate matrix of shape ``(batch, input_dim)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(log_gate_prob, log_hazard_ratios)``, each ``(batch, k)``.
        """
        h = self.embedding(x)
        log_hazard_ratios = torch.clamp(self.expert(h), min=-self.gamma, max=self.gamma)
        log_gate_prob = F.log_softmax(self.gate(h), dim=1)
        return log_gate_prob, log_hazard_ratios


def build_deep_cox_mixtures() -> nn.Module:
    """Build a compact Deep Cox Mixtures survival model.

    Returns
    -------
    nn.Module
        Random-initialized ``DeepCoxMixtures`` in eval mode.
    """
    return DeepCoxMixtures().eval()


def example_input_deep_cox_mixtures() -> Tensor:
    """Create example patient covariates.

    Returns
    -------
    Tensor
        Random covariate tensor of shape ``(8, 24)``.
    """
    return torch.randn(8, 24)


# ---------------------------------------------------------------------------
# DeepAC (DeePaC)
# ---------------------------------------------------------------------------


class RCConv1d(nn.Module):
    """Reverse-complement weight-shared 1-D convolution (DeePaC's RC-layer).

    Applies one shared ``Conv1d`` to the forward one-hot DNA sequence and to
    its reverse complement (sequence-length reversed and ACGT-channel
    reversed), then reverses the RC branch's output back and concatenates it
    with the forward branch -- so the same learned motif filters see both
    strands and produce a strand-symmetric feature map.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 12) -> None:
        """Initialize the single shared convolution kernel.

        Parameters
        ----------
        in_channels : int, default=4
            Number of DNA channels (A, C, G, T).
        out_channels : int
            Number of learned motif filters.
        kernel_size : int, default=12
            Convolution kernel width.
        """
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)

    def forward(self, x: Tensor) -> Tensor:
        """Convolve the forward and reverse-complement strands with shared weights.

        Parameters
        ----------
        x : Tensor
            One-hot DNA sequence, shape ``(batch, 4, length)``.

        Returns
        -------
        Tensor
            Concatenated forward/RC feature maps, shape
            ``(batch, 2 * out_channels, length')``.
        """
        x_rc = torch.flip(x, dims=(1, 2))
        fwd = self.conv(x)
        rc = torch.flip(self.conv(x_rc), dims=(2,))
        return torch.cat([fwd, rc], dim=1)


class DeePaC(nn.Module):
    """Compact DeePaC: RC-conv motif scanner + RC-merged dense classifier."""

    def __init__(self, n_filters: int = 16, kernel_size: int = 12, dense_units: int = 24) -> None:
        """Initialize the RC-conv layer and the shared RC-merge dense head.

        Parameters
        ----------
        n_filters : int, default=16
            Number of motif filters per strand.
        kernel_size : int, default=12
            Convolution kernel width.
        dense_units : int, default=24
            Width of the shared post-pooling dense layer.
        """
        super().__init__()
        self.rc_conv = RCConv1d(4, n_filters, kernel_size)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.shared_dense = nn.Linear(n_filters, dense_units)
        self.classifier = nn.Linear(dense_units, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Scan both strands, RC-merge-pool, and classify pathogenicity.

        Parameters
        ----------
        x : Tensor
            One-hot DNA read, shape ``(batch, 4, length)``.

        Returns
        -------
        Tensor
            Pathogenicity probability, shape ``(batch, 1)``.
        """
        feats = self.rc_conv(x)
        pooled = self.pool(feats).squeeze(-1)
        fwd, rc = pooled.chunk(2, dim=1)
        merged = self.shared_dense(fwd) + self.shared_dense(rc)
        merged = F.relu(merged)
        return torch.sigmoid(self.classifier(merged))


def build_deepac() -> nn.Module:
    """Build a compact DeePaC reverse-complement CNN pathogen classifier.

    Returns
    -------
    nn.Module
        Random-initialized ``DeePaC`` in eval mode.
    """
    return DeePaC().eval()


def example_input_deepac() -> Tensor:
    """Create an example one-hot DNA read.

    Returns
    -------
    Tensor
        Random one-hot tensor of shape ``(2, 4, 100)``.
    """
    idx = torch.randint(0, 4, (2, 100))
    return F.one_hot(idx, num_classes=4).permute(0, 2, 1).float()


# ---------------------------------------------------------------------------
# DeepAffinity
# ---------------------------------------------------------------------------


class DeepAffinity(nn.Module):
    """Compact DeepAffinity: dual GRU encoders + pairwise bilinear co-attention."""

    def __init__(
        self,
        vocab_compound: int = 68,
        vocab_protein: int = 76,
        gru_drug: int = 16,
        gru_prot: int = 24,
        attn_dim: int = 20,
    ) -> None:
        """Initialize the drug/protein GRU encoders and the co-attention heads.

        Parameters
        ----------
        vocab_compound : int, default=68
            SMILES token vocabulary size.
        vocab_protein : int, default=76
            Protein SPS token vocabulary size.
        gru_drug : int, default=16
            Hidden size of the compound GRU encoder.
        gru_prot : int, default=24
            Hidden size of the protein GRU encoder.
        attn_dim : int, default=20
            Dimensionality of the second bilinear attention projection.
        """
        super().__init__()
        self.drug_embed = nn.Embedding(vocab_compound, gru_drug)
        self.drug_gru = nn.GRU(gru_drug, gru_drug, num_layers=2, batch_first=True)
        self.prot_embed = nn.Embedding(vocab_protein, gru_prot)
        self.prot_gru = nn.GRU(gru_prot, gru_prot, num_layers=2, batch_first=True)

        self.attn_w = nn.Parameter(torch.randn(gru_prot, gru_drug) * 0.1)
        self.u_drug = nn.Linear(gru_drug, attn_dim, bias=False)
        self.u_prot = nn.Linear(gru_prot, attn_dim, bias=False)
        self.attn_bias = nn.Parameter(torch.zeros(attn_dim))

        self.conv = nn.Conv1d(1, 8, 4, stride=2)
        self.pool = nn.AdaptiveMaxPool1d(4)
        self.head = nn.Linear(8 * 4, 1)

    def forward(self, compound: Tensor, protein: Tensor) -> Tensor:
        """Encode both sequences, compute pairwise co-attention, and regress.

        Parameters
        ----------
        compound : Tensor
            SMILES token ids, shape ``(batch, comp_len)``.
        protein : Tensor
            Protein SPS token ids, shape ``(batch, prot_len)``.

        Returns
        -------
        Tensor
            Predicted binding-affinity score, shape ``(batch, 1)``.
        """
        drug_h, _ = self.drug_gru(self.drug_embed(compound))
        prot_h, _ = self.prot_gru(self.prot_embed(protein))

        # Bilinear pairwise compatibility: (batch, prot_len, drug_len)
        scores = torch.einsum("bpi,ij,bdj->bpd", prot_h, self.attn_w, drug_h)
        scores = torch.tanh(scores)
        batch = scores.shape[0]
        alphas = F.softmax(scores.reshape(batch, -1), dim=-1).reshape(scores.shape)

        proj_drug = self.u_drug(drug_h)  # (batch, drug_len, attn_dim)
        proj_prot = self.u_prot(prot_h)  # (batch, prot_len, attn_dim)
        # For each drug position, combine with every protein position weighted by alphas.
        combined = proj_drug.unsqueeze(1) + proj_prot.unsqueeze(2) + self.attn_bias
        joint = (alphas.unsqueeze(-1) * combined).sum(dim=1)  # (batch, drug_len, attn_dim)

        joint = joint.mean(dim=-1, keepdim=True).transpose(1, 2)  # (batch, 1, drug_len)
        conv_out = F.leaky_relu(self.conv(joint))
        pooled = self.pool(conv_out).flatten(1)
        return self.head(pooled)


def build_deepaffinity() -> nn.Module:
    """Build a compact DeepAffinity compound-protein affinity regressor.

    Returns
    -------
    nn.Module
        Random-initialized ``DeepAffinity`` in eval mode.
    """
    return DeepAffinity().eval()


def example_input_deepaffinity() -> tuple[Tensor, Tensor]:
    """Create example SMILES and protein SPS token sequences.

    Returns
    -------
    tuple[Tensor, Tensor]
        Compound token ids ``(2, 12)`` and protein token ids ``(2, 16)``.
    """
    compound = torch.randint(0, 68, (2, 12))
    protein = torch.randint(0, 76, (2, 16))
    return compound, protein


# ---------------------------------------------------------------------------
# DeepBCR
# ---------------------------------------------------------------------------


class DeepBCR(nn.Module):
    """Compact DeepBCR: amino-acid embedding + gene-gated max-pool motif scan.

    Reimplements the official ``GeneSwitchModel.hidden_layers`` "gene switch"
    architecture: an amino-acid embedding gather, a per-k-mer motif linear
    layer, gating each k-mer's motif scores by an outer product with its
    isotype/gene-usage counts, a max-pool over all k-mers in the repertoire
    (per motif, per gene), and a final motif-layer + linear output head.
    """

    def __init__(
        self,
        n_amino_acids: int = 21,
        encode_size: int = 8,
        kmer_size: int = 4,
        num_motifs: int = 12,
        num_genes: int = 5,
        num_labels: int = 2,
    ) -> None:
        """Initialize the amino-acid encoding and the gene-switch motif layers.

        Parameters
        ----------
        n_amino_acids : int, default=21
            Amino-acid alphabet size (20 standard + padding/unknown).
        encode_size : int, default=8
            Dimensionality of the learned amino-acid encoding.
        kmer_size : int, default=4
            Length of each CDR3 k-mer.
        num_motifs : int, default=12
            Number of learned motif filters.
        num_genes : int, default=5
            Number of constant-region (isotype) genes tracked per k-mer.
        num_labels : int, default=2
            Number of output classes (or 1 for regression).
        """
        super().__init__()
        self.kmer_size = kmer_size
        self.aa_encode = nn.Embedding(n_amino_acids, encode_size)
        self.kmer_layer = nn.Linear(kmer_size * encode_size, num_motifs)
        self.gene_weight = nn.Parameter(torch.randn(num_motifs, num_genes) * 0.1)
        self.gene_bias = nn.Parameter(torch.zeros(num_motifs))
        self.output_layer = nn.Linear(num_motifs, num_labels)

    def forward(self, kmers: Tensor, gene_counts: Tensor) -> Tensor:
        """Score every k-mer's motifs, gate by gene usage, and max-pool.

        Parameters
        ----------
        kmers : Tensor
            Integer amino-acid ids per k-mer, shape
            ``(batch, max_kmer, kmer_size)``.
        gene_counts : Tensor
            Per-k-mer isotype/constant-gene usage counts, shape
            ``(batch, max_kmer, num_genes)``.

        Returns
        -------
        Tensor
            Classification/hazard logits, shape ``(batch, num_labels)``.
        """
        batch, max_kmer, _ = kmers.shape
        encoded = self.aa_encode(kmers).reshape(batch, max_kmer, -1)
        motif_scores = F.relu(self.kmer_layer(encoded))  # (batch, max_kmer, num_motifs)

        gene_sign = torch.sign(gene_counts)  # (batch, max_kmer, num_genes)
        gated = motif_scores.unsqueeze(-1) * gene_sign.unsqueeze(
            -2
        )  # (batch, max_kmer, num_motifs, num_genes)

        pooled, _ = gated.max(dim=1)  # (batch, num_motifs, num_genes)
        motif_repr = (pooled * self.gene_weight).sum(dim=-1) + self.gene_bias
        motif_repr = F.relu(motif_repr)
        return self.output_layer(motif_repr)


def build_deepbcr() -> nn.Module:
    """Build a compact DeepBCR repertoire classifier.

    Returns
    -------
    nn.Module
        Random-initialized ``DeepBCR`` in eval mode.
    """
    return DeepBCR().eval()


def example_input_deepbcr() -> tuple[Tensor, Tensor]:
    """Create example BCR k-mer ids and per-k-mer gene usage counts.

    Returns
    -------
    tuple[Tensor, Tensor]
        K-mer amino-acid ids ``(2, 10, 4)`` and gene counts ``(2, 10, 5)``.
    """
    kmers = torch.randint(0, 21, (2, 10, 4))
    gene_counts = torch.randint(0, 3, (2, 10, 5)).float()
    return kmers, gene_counts


# ---------------------------------------------------------------------------
# DeepBGC
# ---------------------------------------------------------------------------


class DeepBGC(nn.Module):
    """Compact DeepBGC: (stacked) BiLSTM + per-timestep sigmoid BGC head."""

    def __init__(self, input_size: int = 32, hidden_size: int = 24, num_layers: int = 2) -> None:
        """Initialize the stacked bidirectional LSTM and the tagging head.

        Parameters
        ----------
        input_size : int, default=32
            Dimensionality of each protein-domain (pfam2vec) embedding.
        hidden_size : int, default=24
            Hidden size of each LSTM direction.
        num_layers : int, default=2
            Number of stacked bidirectional LSTM layers.
        """
        super().__init__()
        self.blstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Linear(2 * hidden_size, 1)

    def forward(self, domain_embeddings: Tensor) -> Tensor:
        """Tag every protein domain in the ORF sequence with a BGC probability.

        Parameters
        ----------
        domain_embeddings : Tensor
            Ordered pfam2vec protein-domain embeddings, shape
            ``(batch, n_proteins, input_size)``.

        Returns
        -------
        Tensor
            Per-timestep BGC-membership probability, shape
            ``(batch, n_proteins, 1)``.
        """
        h, _ = self.blstm(domain_embeddings)
        return torch.sigmoid(self.head(h))


def build_deepbgc() -> nn.Module:
    """Build a compact DeepBGC biosynthetic-gene-cluster tagger.

    Returns
    -------
    nn.Module
        Random-initialized ``DeepBGC`` in eval mode.
    """
    return DeepBGC().eval()


def example_input_deepbgc() -> Tensor:
    """Create an example ordered sequence of protein-domain embeddings.

    Returns
    -------
    Tensor
        Random pfam2vec-style embedding sequence of shape ``(1, 30, 32)``.
    """
    return torch.randn(1, 30, 32)


MENAGERIE_ENTRIES = [
    ("DAGAN", "build_dagan", "example_input_dagan", "2018", "VIS"),
    (
        "Deep Cox Mixtures",
        "build_deep_cox_mixtures",
        "example_input_deep_cox_mixtures",
        "2021",
        "BIO",
    ),
    ("DeepAC", "build_deepac", "example_input_deepac", "2020", "BIO"),
    ("DeepAffinity", "build_deepaffinity", "example_input_deepaffinity", "2019", "BIO"),
    ("DeepBCR", "build_deepbcr", "example_input_deepbcr", "2021", "BIO"),
    ("DeepBGC", "build_deepbgc", "example_input_deepbgc", "2019", "BIO"),
]
