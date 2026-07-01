"""Menagerie batch w6a7: computational-genomics/bioinformatics deep-learning
classics for multi-omics late-integration drug-response prediction, deep-CNN
long-read SNP/indel calling from pileup images, deep-CNN somatic-mutation
calling from tumor/normal alignment images, optimal-transport spatial
transcriptomics reconstruction, cross-modal genomic foundation-model
generation, and bimodal context-attention drug-sensitivity prediction.

Sources checked (reference only; no cloning, no pip installs):
  - MOLI (cand_00746): Sharifi-Noghabi, Zolotareva, Collins & Ester,
    Bioinformatics 2019, https://github.com/hosseinshn/MOLI
    (``EGFR/EGFRv8.ipynb``, classes ``AEE``/``AEM``/``AEC``/``Classifier``).
    The defining mechanism is **multi-omics late integration**: three
    separate per-omic encoders -- ``AEE`` for gene-expression, ``AEM`` for
    somatic-mutation, ``AEC`` for copy-number-alteration -- each a
    ``Linear -> BatchNorm1d -> ReLU -> Dropout`` block projecting its own
    input modality into its own small latent space (``h_dim1``/``h_dim2``/
    ``h_dim3``); the three per-omic latents are concatenated
    (``torch.cat((ZEX, ZMX, ZCX), 1)``) and L2-normalized
    (``F.normalize(ZTX, p=2, dim=0)``) into a single fused representation,
    which a shared ``Classifier`` (``Linear -> Dropout -> Sigmoid``) maps to
    a drug-response probability -- "one small encoder per omic modality,
    late-fused by concat + L2-normalize, one shared classifier head" is
    MOLI's namesake late-integration architecture (as opposed to early
    concatenation of raw omics or a single joint encoder). Reimplemented
    with the same three-encoder / concat-normalize / classifier topology at
    reduced per-omic feature counts and latent widths.
  - NanoCaller (cand_00748): Ahsan, Liu, Fang & Wang, Genome Biology 2021,
    https://github.com/WGLab/NanoCaller (``nanocaller_src/model_architect.py``,
    class ``SNP_model``, ported here from the reference's TensorFlow/Keras
    ``tf.keras.Model`` subclass to an equivalent ``nn.Module``). The
    defining mechanism: a **multi-branch inception-style 2D-CNN** over a
    long-read pileup image runs three parallel convolutions with different
    receptive-field shapes over the same input (row-wise ``[1,5]``,
    column-wise ``[5,1]``, and square ``[5,5]`` kernels, matching
    ``conv1_1``/``conv1_2``/``conv1_3``) and concatenates their feature maps
    channel-wise (``merge_conv_0``) before two further strided convolutions
    and a dense trunk; the trunk feeds **four parallel per-allele heads**
    (A/G/T/C), each of which additionally concatenates in that allele's
    reference-base indicator (``tf.concat([fa, A_ref], 1)``) before a
    softmax, and the four allele-head outputs are themselves concatenated
    back in to a final genotype (GT) head -- i.e. "multi-kernel-shape
    inception stem -> shared trunk -> four ref-conditioned allele heads
    feeding a genotype head" is NanoCaller's namesake SNP-calling mechanism.
    Reimplemented with the same three-shape inception stem, shared conv/
    dense trunk, four ref-conditioned per-allele softmax heads, and a
    genotype head conditioned on the trunk plus all four allele outputs, at
    reduced channel widths / pileup dimensions.
  - NeuSomatic (cand_00749): Sahraeian, Liu, Lau, et al., Nature
    Communications 2019, https://github.com/bioinform/neusomatic
    (``neusomatic/python/network.py``, classes ``NSBlock``/
    ``NeuSomaticNet`` -- an exact PyTorch source, used near-verbatim at
    reduced width). The defining mechanism: a stem conv+pool over a
    tumor/normal alignment "image" feeds a stack of **residual dilated 2D
    conv blocks** (``NSBlock``: two dilated convs with increasing dilation
    rate across the stack, summed residually with the block's input, then
    max-pooled only along the read/column axis -- preserving the reference-
    position axis while reducing the read-pileup axis), and the pooled
    features feed **three parallel task heads** off one shared dense trunk:
    mutation-type classification (SNP/insertion/deletion/none, ``fc2``),
    VAF regression (``fc3``), and mutation-length-category classification
    (``fc4``) -- one shared residual-dilated-CNN trunk, three simultaneous
    prediction heads is NeuSomatic's namesake multi-task somatic-calling
    architecture. Reimplemented as a near-direct port of ``NSBlock``/
    ``NeuSomaticNet`` at reduced channel width / pileup dimensions (the
    four-block dilation/stride schedule and residual-then-pool pattern are
    preserved exactly).
  - novoSpaRc (cand_00750): Nitzan, Karaiskos, Friedman & Rajewsky, Nature
    Protocols 2021 (algorithm: Moriel, Nitzan, et al., Nature Communications
    2021), https://github.com/rajewsky-lab/novosparc (``novosparc/
    reconstruction/_GWadjusted.py``, function
    ``gromov_wasserstein_adjusted_norm``). The defining mechanism is
    **optimal-transport spatial reconstruction**: given a source-space
    (expression-similarity) cost matrix ``C1``, a target-space (physical-
    location-similarity) cost matrix ``C2``, and (if markers are available)
    a source-target linear cost matrix, the algorithm iteratively (a)
    computes the square-loss Gromov-Wasserstein tensor
    ``tensor_square_loss_adjusted`` = ``-C1 @ T @ C2^T`` (shifted to be
    non-negative) for the current coupling ``T``, (b) blends it with the
    linear marker-based cost via ``alpha_linear``, and (c) re-solves the
    entropic-regularized optimal-transport coupling with a **Sinkhorn
    fixed-point iteration** (row/column marginal-rescaling of
    ``exp(-cost/epsilon)``) -- repeating until the coupling converges. This
    fixed-point double-loop (outer GW-tensor update, inner Sinkhorn) *is*
    novoSpaRc's namesake contribution over plain marker-based mapping.
    Reimplemented as a single traceable ``nn.Module`` that unrolls both
    loops for a *fixed* number of iterations (a traceable analogue of the
    reference's ``while err > tol`` loop): a learnable scalar
    ``alpha_linear`` blend weight (the only free parameter, matching the
    reference's single tunable hyperparameter of the same name) and a fixed
    number of outer GW-tensor-update / inner-Sinkhorn iterations, taking
    expression-similarity and location-similarity cost matrices (plus an
    optional marker-based linear cost) and returning the final transport
    coupling used to project single-cell expression onto spatial locations.
  - Omni-DNA (cand_00752): Zhou, Zehui, et al. (Zehui127), NeurIPS 2025
    poster / preprint arXiv:2502.03499, https://github.com/Zehui127/Omni-DNA
    (``src/FT_CLS_Head/omni_dna.py`` loads an ``AutoModelForCausalLM``
    OLMo-family decoder; ``src/utils.py``'s ``extend_model_tokenizer`` /
    ``compute_added_vocabs`` shows the SFT recipe: the tokenizer vocabulary
    is *extended* with new task-specific tokens -- text words for
    DNA-to-text, or VQ-VAE image-codebook indices for DNA-to-image -- and
    the model's token-embedding table is resized to match, so DNA tokens
    and cross-modal output tokens share one embedding space and one
    autoregressive decoder). The defining mechanism: a **single shared
    decoder-only (GPT/OLMo-style) transformer with an extended, unified
    vocabulary** spanning both DNA nucleotide tokens and task-output tokens
    (classification labels / text tokens / discrete image-codebook tokens),
    so that DNA understanding, generation, classification, and cross-modal
    generation are all just next-token prediction over one merged
    vocabulary and one backbone -- as opposed to separate task-specific
    heads or encoder-decoder architectures. Reimplemented as ``OmniDNA``: a
    compact pre-norm causal-decoder transformer (rotary-free learned
    positional embedding, multi-head causal self-attention, GEGLU MLP,
    matching the OLMo-family block used by the reference) over a unified
    vocabulary partitioned into a DNA-token range and an extended
    task-token range (covering both a classification-style label token and
    an image-codebook-style discrete token, mirroring the reference's two
    demonstrated SFT extensions), at reduced depth/width/context length.
  - PaccMann (cand_00753): Manica, Oskooei, Born, et al., Molecular
    Pharmaceutics 2019, https://github.com/PaccMann/paccmann_predictor
    (``paccmann_predictor/models/bimodal_mca.py`` class ``BimodalMCA``, and
    ``paccmann_predictor/utils/layers.py`` class ``ContextAttentionLayer``).
    The defining mechanism is **bimodal multiscale context attention**: a
    SMILES ("ligand") token sequence and a gene-expression/protein
    ("receptor") sequence are each embedded and passed through their own
    stack of multiscale 1D convolutions (several parallel kernel widths per
    branch, matching ``ligand_filters``/``receptor_filters``); at *every*
    scale (including the raw embedding), a ``ContextAttentionLayer``
    computes attention over one branch's sequence *using the other
    branch's* sequence as context (project both to a shared attention
    space, ``tanh(ref_proj + context_proj)``, score, softmax, weight-and-
    sum) -- i.e. the ligand attends to the receptor and the receptor
    attends to the ligand, at every convolutional scale, symmetrically. All
    attention-pooled multiscale encodings from both branches are
    concatenated and fed through a dense trunk to a sigmoid drug-sensitivity
    (IC50) prediction. Reimplemented with the same dual-branch multiscale-
    conv + bidirectional cross-context-attention-at-every-scale + dense
    sigmoid-head topology at reduced vocabulary/sequence-length/filter
    counts.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ============================================================
# MOLI -- three per-omic encoders, concat + L2-normalize late
# fusion, shared classifier head (hosseinshn/MOLI)
# ============================================================


class _MOLIEncoder(nn.Module):
    """One per-omic encoder: Linear -> BatchNorm1d -> ReLU -> Dropout."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MOLI(nn.Module):
    """Multi-Omics Late Integration drug-response predictor.

    Ports ``EGFR/EGFRv8.ipynb``'s ``AEE``/``AEM``/``AEC``/``Classifier``:
    three independent per-omic encoders (expression, mutation, copy-number
    alteration) project their modality into a small latent space; the three
    latents are concatenated and L2-normalized, and a shared classifier
    predicts drug-response probability from the fused representation.
    """

    def __init__(
        self,
        expr_dim: int = 40,
        mut_dim: int = 40,
        cna_dim: int = 40,
        h_expr: int = 16,
        h_mut: int = 8,
        h_cna: int = 8,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.enc_expr = _MOLIEncoder(expr_dim, h_expr, dropout)
        self.enc_mut = _MOLIEncoder(mut_dim, h_mut, dropout)
        self.enc_cna = _MOLIEncoder(cna_dim, h_cna, dropout)
        fused_dim = h_expr + h_mut + h_cna
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 1),
            nn.Dropout(dropout),
            nn.Sigmoid(),
        )

    def forward(self, expr: Tensor, mut: Tensor, cna: Tensor) -> Tensor:
        """Predict drug-response probability from three omics modalities.

        Parameters
        ----------
        expr : Tensor
            Shape ``(batch, expr_dim)`` gene-expression features.
        mut : Tensor
            Shape ``(batch, mut_dim)`` somatic-mutation features.
        cna : Tensor
            Shape ``(batch, cna_dim)`` copy-number-alteration features.
        """
        z_expr = self.enc_expr(expr)
        z_mut = self.enc_mut(mut)
        z_cna = self.enc_cna(cna)
        fused = torch.cat((z_expr, z_mut, z_cna), dim=1)
        fused = F.normalize(fused, p=2, dim=0)
        return self.classifier(fused)


def build_moli() -> nn.Module:
    """Build a small MOLI three-encoder late-integration model."""
    return MOLI(expr_dim=40, mut_dim=40, cna_dim=40, h_expr=16, h_mut=8, h_cna=8).eval()


def example_input_moli() -> tuple[Tensor, Tensor, Tensor]:
    """Return (expression, mutation, CNA) feature batches for MOLI."""
    batch = 6
    return (
        torch.randn(batch, 40),
        torch.randn(batch, 40),
        torch.randn(batch, 40),
    )


# ============================================================
# NanoCaller -- multi-kernel-shape inception stem + shared trunk
# + four ref-conditioned per-allele heads + genotype head
# (WGLab/NanoCaller)
# ============================================================


class NanoCallerSNP(nn.Module):
    """Multi-branch inception CNN over pileup images for SNP calling.

    Ports ``nanocaller_src/model_architect.py``'s ``SNP_model`` (TensorFlow
    original, reimplemented in ``torch.nn``): three parallel convolutions
    with row-wise / column-wise / square kernels scan the same pileup
    image and are concatenated channel-wise; two strided convolutions and a
    dense trunk feed four parallel per-allele (A/G/T/C) softmax heads, each
    conditioned on that allele's one-hot reference indicator; the four
    allele outputs are concatenated back in to a genotype (GT) head.
    """

    def __init__(self, height: int = 15, width: int = 15, base_ch: int = 8) -> None:
        super().__init__()
        self.conv1_1 = nn.Conv2d(1, base_ch, kernel_size=(1, 5), padding=(0, 2))
        self.conv1_2 = nn.Conv2d(1, base_ch, kernel_size=(5, 1), padding=(2, 0))
        self.conv1_3 = nn.Conv2d(1, base_ch, kernel_size=(5, 5), padding=(2, 2))
        merged_ch = base_ch * 3
        self.conv2 = nn.Conv2d(merged_ch, base_ch * 4, kernel_size=(2, 3), stride=(1, 2))
        self.conv3 = nn.Conv2d(base_ch * 4, base_ch * 8, kernel_size=(2, 3), stride=(1, 2))
        self.act = nn.SELU()
        self.dropout = nn.Dropout(0.5)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, height, width)
            flat_dim = self._conv_trunk(dummy).flatten(1).shape[1]

        self.fc1 = nn.Linear(flat_dim, 48)
        self.fa = nn.Linear(48, 16)
        self.allele_A = nn.Linear(16 + 1, 2)
        self.allele_G = nn.Linear(16 + 1, 2)
        self.allele_T = nn.Linear(16 + 1, 2)
        self.allele_C = nn.Linear(16 + 1, 2)
        self.fc2 = nn.Linear(48, 16)
        self.fc3 = nn.Linear(16 + 8, 8)
        self.gt_head = nn.Linear(8, 2)

    def _conv_trunk(self, x: Tensor) -> Tensor:
        c1_1 = self.act(self.conv1_1(x))
        c1_2 = self.act(self.conv1_2(x))
        c1_3 = self.act(self.conv1_3(x))
        merged = torch.cat([c1_1, c1_2, c1_3], dim=1)
        c2 = self.act(self.conv2(merged))
        return self.act(self.conv3(c2))

    def forward(
        self,
        pileup: Tensor,
        ref_a: Tensor,
        ref_g: Tensor,
        ref_t: Tensor,
        ref_c: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Predict per-allele probabilities and a genotype from a pileup image.

        Parameters
        ----------
        pileup : Tensor
            Shape ``(batch, 1, height, width)`` one-channel pileup image.
        ref_a, ref_g, ref_t, ref_c : Tensor
            Shape ``(batch, 1)`` reference-base indicator for each allele.
        """
        x = self._conv_trunk(pileup)
        flat = x.flatten(1)
        fc1 = self.act(self.fc1(flat))
        fc1 = self.dropout(fc1)
        fa = self.act(self.fa(fc1))

        out_a = F.softmax(self.allele_A(torch.cat([fa, ref_a], dim=1)), dim=1)
        out_g = F.softmax(self.allele_G(torch.cat([fa, ref_g], dim=1)), dim=1)
        out_t = F.softmax(self.allele_T(torch.cat([fa, ref_t], dim=1)), dim=1)
        out_c = F.softmax(self.allele_C(torch.cat([fa, ref_c], dim=1)), dim=1)

        fc2 = self.act(self.fc2(fc1))
        fc3 = self.act(self.fc3(torch.cat([fc2, out_a, out_g, out_t, out_c], dim=1)))
        out_gt = F.softmax(self.gt_head(fc3), dim=1)
        return out_a, out_g, out_t, out_c, out_gt


def build_nanocaller() -> nn.Module:
    """Build a small NanoCaller SNP-calling inception-CNN model."""
    return NanoCallerSNP(height=15, width=15, base_ch=8).eval()


def example_input_nanocaller() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return (pileup image, per-allele reference indicators) for NanoCaller."""
    batch = 4
    pileup = torch.randn(batch, 1, 15, 15)
    ref_a = (torch.rand(batch, 1) > 0.75).float()
    ref_g = (torch.rand(batch, 1) > 0.75).float()
    ref_t = (torch.rand(batch, 1) > 0.75).float()
    ref_c = (torch.rand(batch, 1) > 0.75).float()
    return pileup, ref_a, ref_g, ref_t, ref_c


# ============================================================
# NeuSomatic -- residual dilated-conv blocks + three parallel
# task heads over tumor/normal alignment images
# (bioinform/neusomatic)
# ============================================================


class NSBlock(nn.Module):
    """Residual block: two dilated convs, summed with input, then pooled.

    Ports ``neusomatic/python/network.py``'s ``NSBlock`` exactly: two
    ``Conv2d`` layers at (possibly different) dilation rates with matching
    'same'-style padding, the second's output added residually to the
    block's input, then max-pooled only along the last (read/column) axis.
    """

    def __init__(
        self,
        dim: int,
        ks_1: int = 3,
        ks_2: int = 3,
        dl_1: int = 1,
        dl_2: int = 1,
        mp_ks: int = 3,
        mp_st: int = 1,
    ) -> None:
        super().__init__()
        self.conv_r1 = nn.Conv2d(
            dim, dim, kernel_size=ks_1, dilation=dl_1, padding=(dl_1 * (ks_1 - 1)) // 2
        )
        self.bn_r1 = nn.BatchNorm2d(dim)
        self.conv_r2 = nn.Conv2d(
            dim, dim, kernel_size=ks_2, dilation=dl_2, padding=(dl_2 * (ks_2 - 1)) // 2
        )
        self.bn_r2 = nn.BatchNorm2d(dim)
        self.pool_r2 = nn.MaxPool2d((1, mp_ks), padding=(0, (mp_ks - 1) // 2), stride=(1, mp_st))

    def forward(self, x: Tensor) -> Tensor:
        y1 = F.relu(self.bn_r1(self.conv_r1(x)))
        y2 = self.bn_r2(self.conv_r2(y1))
        y3 = x + y2
        return self.pool_r2(y3)


class NeuSomaticNet(nn.Module):
    """Residual dilated-CNN trunk with three parallel somatic-calling heads.

    Ports ``neusomatic/python/network.py``'s ``NeuSomaticNet``: a stem
    conv+pool feeds four stacked ``NSBlock`` residual dilated-conv blocks
    (increasing dilation, matching the reference's ``nsblocks`` schedule);
    the flattened pooled features feed a shared dense layer, then three
    parallel heads predict mutation type, VAF, and mutation-length
    category.
    """

    def __init__(
        self, num_channels: int = 4, dim: int = 16, height: int = 5, width: int = 32
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, dim, kernel_size=(1, 3), padding=(0, 1), stride=1)
        self.bn1 = nn.BatchNorm2d(dim)
        self.pool1 = nn.MaxPool2d((1, 3), padding=(0, 1), stride=(1, 1))
        nsblocks = [
            [3, 5, 1, 1, 3, 1],
            [3, 5, 1, 1, 3, 2],
            [3, 5, 2, 1, 3, 2],
            [3, 5, 4, 2, 3, 2],
        ]
        res_layers = [NSBlock(dim, *cfg) for cfg in nsblocks]
        self.res_layers = nn.Sequential(*res_layers)

        with torch.no_grad():
            dummy = torch.zeros(1, num_channels, height, width)
            pooled = self.pool1(F.relu(self.bn1(self.conv1(dummy))))
            trunk_out = self.res_layers(pooled)
            self.fc_dim = trunk_out.flatten(1).shape[1]

        self.fc1 = nn.Linear(self.fc_dim, 48)
        self.fc2 = nn.Linear(48, 4)
        self.fc3 = nn.Linear(48, 1)
        self.fc4 = nn.Linear(48, 4)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict (mutation type, VAF, length category) from an alignment image.

        Parameters
        ----------
        x : Tensor
            Shape ``(batch, num_channels, height, width)`` tumor/normal
            alignment "image" (reference position x aligned-read channels).
        """
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.res_layers(x)
        x2 = x.flatten(1)
        x3 = F.relu(self.fc1(x2))
        mutation_type = self.fc2(x3)
        vaf = self.fc3(x3)
        length_category = self.fc4(x3)
        return mutation_type, vaf, length_category


def build_neusomatic() -> nn.Module:
    """Build a small NeuSomatic residual dilated-CNN multi-task model."""
    return NeuSomaticNet(num_channels=4, dim=16, height=5, width=32).eval()


def example_input_neusomatic() -> Tensor:
    """Return a batch of tumor/normal alignment images for NeuSomatic."""
    return torch.randn(3, 4, 5, 32)


# ============================================================
# novoSpaRc -- unrolled fused Gromov-Wasserstein-adjusted spatial
# reconstruction (outer GW-tensor update + inner Sinkhorn)
# (rajewsky-lab/novosparc)
# ============================================================


class NovoSpaRcOT(nn.Module):
    """Fixed-iteration fused Gromov-Wasserstein spatial-reconstruction solver.

    Ports ``novosparc/reconstruction/_GWadjusted.py``'s
    ``gromov_wasserstein_adjusted_norm`` / ``tensor_square_loss_adjusted``:
    given an expression-similarity cost matrix, a location-similarity cost
    matrix, and an optional marker-based linear cost, iteratively (1)
    computes the square-loss GW tensor ``-C1 @ T @ C2^T`` for the current
    coupling, (2) blends it with the linear marker cost via a learnable
    ``alpha_linear`` weight, and (3) re-solves the entropic OT coupling
    with a Sinkhorn fixed-point iteration -- unrolled here for a fixed
    number of outer/inner steps so the whole solve is one traceable
    ``nn.Module`` (the reference's ``while err > tol`` loop is not
    traceable as-is).
    """

    def __init__(
        self,
        n_cells: int = 12,
        n_locations: int = 10,
        epsilon: float = 0.05,
        outer_iters: int = 3,
        sinkhorn_iters: int = 20,
    ) -> None:
        super().__init__()
        self.n_cells = n_cells
        self.n_locations = n_locations
        self.epsilon = epsilon
        self.outer_iters = outer_iters
        self.sinkhorn_iters = sinkhorn_iters
        self.alpha_linear = nn.Parameter(torch.tensor(0.2))

    @staticmethod
    def _tensor_square_loss(c1: Tensor, c2: Tensor, coupling: Tensor) -> Tensor:
        tens = -torch.matmul(torch.matmul(c1, coupling), c2.transpose(-1, -2))
        tens = tens - tens.amin(dim=(-2, -1), keepdim=True)
        return tens

    def _sinkhorn(self, p: Tensor, q: Tensor, cost: Tensor) -> Tensor:
        kernel = torch.exp(-cost / self.epsilon)
        u = torch.ones_like(p)
        for _ in range(self.sinkhorn_iters):
            v = q / (torch.matmul(kernel.transpose(-1, -2), u) + 1e-12)
            u = p / (torch.matmul(kernel, v) + 1e-12)
        return u.unsqueeze(-1) * kernel * v.unsqueeze(-2)

    def forward(
        self,
        cost_expression: Tensor,
        cost_locations: Tensor,
        linear_cost: Tensor,
        p_expression: Tensor,
        p_locations: Tensor,
    ) -> Tensor:
        """Solve for the fused GW-adjusted optimal-transport coupling.

        Parameters
        ----------
        cost_expression : Tensor
            Shape ``(n_cells, n_cells)`` source-space (expression-
            similarity) cost matrix.
        cost_locations : Tensor
            Shape ``(n_locations, n_locations)`` target-space (physical-
            location-similarity) cost matrix.
        linear_cost : Tensor
            Shape ``(n_cells, n_locations)`` marker-gene-based linear cost
            between cells and locations.
        p_expression : Tensor
            Shape ``(n_cells,)`` source marginal distribution.
        p_locations : Tensor
            Shape ``(n_locations,)`` target marginal distribution.
        """
        alpha = torch.sigmoid(self.alpha_linear)
        linear_norm = linear_cost / (linear_cost.amax() + 1e-12)
        coupling = torch.outer(p_expression, p_locations)
        for _ in range(self.outer_iters):
            gw_tensor = self._tensor_square_loss(cost_expression, cost_locations, coupling)
            blended = (1.0 - alpha) * gw_tensor + alpha * linear_norm
            coupling = self._sinkhorn(p_expression, p_locations, blended)
        return coupling


def build_novosparc() -> nn.Module:
    """Build a small novoSpaRc fused-GW optimal-transport reconstruction model."""
    return NovoSpaRcOT(n_cells=12, n_locations=10, outer_iters=3, sinkhorn_iters=20).eval()


def example_input_novosparc() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return (expression cost, location cost, linear cost, marginals) for novoSpaRc."""
    n_cells, n_locations = 12, 10
    expr = torch.randn(n_cells, 6)
    cost_expression = torch.cdist(expr, expr)
    loc = torch.rand(n_locations, 2)
    cost_locations = torch.cdist(loc, loc)
    linear_cost = torch.rand(n_cells, n_locations)
    p_expression = torch.full((n_cells,), 1.0 / n_cells)
    p_locations = torch.full((n_locations,), 1.0 / n_locations)
    return cost_expression, cost_locations, linear_cost, p_expression, p_locations


# ============================================================
# Omni-DNA -- decoder-only transformer over a unified DNA +
# extended cross-modal-task vocabulary (Zehui127/Omni-DNA)
# ============================================================


class _OmniDNABlock(nn.Module):
    """Pre-norm causal self-attention + GEGLU MLP transformer block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.gate_proj = nn.Linear(d_model, d_ff)
        self.up_proj = nn.Linear(d_model, d_ff)
        self.down_proj = nn.Linear(d_ff, d_model)

    def forward(self, x: Tensor, causal_mask: Tensor) -> Tensor:
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=causal_mask, need_weights=False)
        x = x + attn_out
        normed = self.ln2(x)
        gated = F.gelu(self.gate_proj(normed)) * self.up_proj(normed)
        return x + self.down_proj(gated)


class OmniDNA(nn.Module):
    """Decoder-only transformer over a unified DNA + cross-modal-task vocabulary.

    Ports the Omni-DNA design described in ``src/FT_CLS_Head/omni_dna.py``
    (which loads an OLMo-family ``AutoModelForCausalLM``) and
    ``src/utils.py``'s ``extend_model_tokenizer``: a single GPT/OLMo-style
    causal decoder is trained over one merged vocabulary spanning DNA
    nucleotide tokens *and* extended task-output tokens (label tokens for
    classification, discrete image-codebook tokens for DNA-to-image, text
    tokens for DNA-to-text), so every task is next-token prediction with
    the same shared backbone and embedding table.
    """

    def __init__(
        self,
        dna_vocab_size: int = 8,
        task_vocab_size: int = 24,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 128,
        max_len: int = 64,
    ) -> None:
        super().__init__()
        vocab_size = dna_vocab_size + task_vocab_size
        self.dna_vocab_size = dna_vocab_size
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [_OmniDNABlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids: Tensor) -> Tensor:
        """Predict next-token logits over the unified DNA + task vocabulary.

        Parameters
        ----------
        token_ids : Tensor
            Shape ``(batch, seq_len)`` integer token ids drawn from the
            unified vocabulary (DNA-token range followed by extended
            task-token range within one sequence, e.g. DNA context followed
            by a classification label or cross-modal output tokens).
        """
        seq_len = token_ids.shape[1]
        x = self.token_embed(token_ids) + self.pos_embed[:, :seq_len, :]
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=token_ids.device), diagonal=1
        )
        for block in self.blocks:
            x = block(x, causal_mask)
        x = self.ln_f(x)
        return self.lm_head(x)


def build_omni_dna() -> nn.Module:
    """Build a small Omni-DNA unified-vocabulary decoder-only transformer."""
    return OmniDNA(
        dna_vocab_size=8,
        task_vocab_size=24,
        d_model=64,
        n_heads=4,
        n_layers=3,
        d_ff=128,
        max_len=64,
    ).eval()


def example_input_omni_dna() -> Tensor:
    """Return a batch of unified-vocabulary token ids for Omni-DNA."""
    return torch.randint(0, 8 + 24, (2, 40))


# ============================================================
# PaccMann -- bimodal multiscale conv + symmetric cross-context
# attention at every scale (PaccMann/paccmann_predictor)
# ============================================================


class _ContextAttentionLayer(nn.Module):
    """Attention over one branch's sequence using the other branch as context.

    Ports ``paccmann_predictor/utils/layers.py``'s ``ContextAttentionLayer``:
    project the reference sequence and the context sequence into a shared
    attention space, combine additively through a tanh nonlinearity, score
    with a linear projection, softmax over the reference sequence length,
    and weight-and-sum the reference sequence by the resulting attention
    weights.
    """

    def __init__(
        self,
        ref_hidden: int,
        ref_len: int,
        context_hidden: int,
        context_len: int,
        attention_size: int = 16,
    ) -> None:
        super().__init__()
        self.reference_projection = nn.Linear(ref_hidden, attention_size)
        self.context_projection = nn.Linear(context_hidden, attention_size)
        if context_len > 1:
            self.context_hidden_projection = nn.Linear(context_len, ref_len)
        else:
            self.context_hidden_projection = None
        self.alpha_projection = nn.Linear(attention_size, 1, bias=False)

    def forward(self, reference: Tensor, context: Tensor) -> Tensor:
        """Cross-attend ``reference`` using ``context``, pooled over sequence.

        Parameters
        ----------
        reference : Tensor
            Shape ``(batch, ref_len, ref_hidden)``.
        context : Tensor
            Shape ``(batch, context_len, context_hidden)``.
        """
        ref_attn = self.reference_projection(reference)
        ctx_attn = self.context_projection(context)
        if self.context_hidden_projection is not None:
            ctx_attn = self.context_hidden_projection(ctx_attn.transpose(1, 2)).transpose(1, 2)
        alphas = F.softmax(
            self.alpha_projection(torch.tanh(ref_attn + ctx_attn)).squeeze(-1), dim=1
        )
        return torch.sum(reference * alphas.unsqueeze(-1), dim=1)


class _MultiscaleConvBranch(nn.Module):
    """One embed -> multiscale-1D-conv branch (several parallel kernel widths)."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        filters: tuple[int, ...],
        kernel_sizes: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(embed_dim, n_filters, kernel_size=k, padding=k // 2)
                for n_filters, k in zip(filters, kernel_sizes)
            ]
        )

    def forward(self, tokens: Tensor) -> list[Tensor]:
        """Embed tokens and run each parallel-scale conv, returning all scales.

        Parameters
        ----------
        tokens : Tensor
            Shape ``(batch, seq_len)`` integer token ids.

        Returns
        -------
        list[Tensor]
            ``[embedded]`` followed by one ``(batch, seq_len, n_filters)``
            tensor per convolutional scale (embedded sequence is scale 0,
            matching the reference's ``[embedded_x] + [conv scales]``).
        """
        embedded = self.embedding(tokens)
        scales = [embedded]
        conv_in = embedded.transpose(1, 2)
        for conv in self.convs:
            scales.append(F.relu(conv(conv_in)).transpose(1, 2)[:, : embedded.shape[1], :])
        return scales


class PaccMannBimodalMCA(nn.Module):
    """Bimodal multiscale convolutional attentive drug-sensitivity encoder.

    Ports ``paccmann_predictor/models/bimodal_mca.py``'s ``BimodalMCA``: a
    ligand (SMILES) branch and a receptor (gene-expression/protein) branch
    each run multiscale 1D convolutions over their own embedded token
    sequence; at every scale a ``ContextAttentionLayer`` lets the ligand
    attend to the receptor and, symmetrically, the receptor attend to the
    ligand; all attention-pooled encodings are concatenated and a dense
    trunk with a sigmoid head predicts IC50 drug-sensitivity.
    """

    def __init__(
        self,
        ligand_vocab: int = 32,
        receptor_vocab: int = 24,
        ligand_len: int = 20,
        receptor_len: int = 16,
        embed_dim: int = 8,
        filters: tuple[int, ...] = (12, 12),
        kernel_sizes: tuple[int, ...] = (3, 5),
        attention_size: int = 16,
        dense_hidden: int = 20,
    ) -> None:
        super().__init__()
        self.ligand_branch = _MultiscaleConvBranch(ligand_vocab, embed_dim, filters, kernel_sizes)
        self.receptor_branch = _MultiscaleConvBranch(
            receptor_vocab, embed_dim, filters, kernel_sizes
        )

        ligand_scale_dims = [embed_dim] + list(filters)
        receptor_scale_dims = [embed_dim] + list(filters)

        self.ligand_attn = nn.ModuleList(
            [
                _ContextAttentionLayer(
                    ligand_scale_dims[i],
                    ligand_len,
                    receptor_scale_dims[i],
                    receptor_len,
                    attention_size,
                )
                for i in range(len(ligand_scale_dims))
            ]
        )
        self.receptor_attn = nn.ModuleList(
            [
                _ContextAttentionLayer(
                    receptor_scale_dims[i],
                    receptor_len,
                    ligand_scale_dims[i],
                    ligand_len,
                    attention_size,
                )
                for i in range(len(receptor_scale_dims))
            ]
        )

        fused_dim = sum(ligand_scale_dims) + sum(receptor_scale_dims)
        self.batch_norm = nn.BatchNorm1d(fused_dim)
        self.dense = nn.Sequential(nn.Linear(fused_dim, dense_hidden), nn.ReLU())
        self.final = nn.Sequential(nn.Linear(dense_hidden, 1), nn.Sigmoid())

    def forward(self, ligand: Tensor, receptor: Tensor) -> Tensor:
        """Predict IC50 drug-sensitivity from a ligand/receptor token pair.

        Parameters
        ----------
        ligand : Tensor
            Shape ``(batch, ligand_len)`` SMILES token ids.
        receptor : Tensor
            Shape ``(batch, receptor_len)`` gene-expression/protein token ids.
        """
        ligand_scales = self.ligand_branch(ligand)
        receptor_scales = self.receptor_branch(receptor)

        ligand_pooled = [
            attn(ref, ctx)
            for attn, ref, ctx in zip(self.ligand_attn, ligand_scales, receptor_scales)
        ]
        receptor_pooled = [
            attn(ref, ctx)
            for attn, ref, ctx in zip(self.receptor_attn, receptor_scales, ligand_scales)
        ]

        fused = torch.cat(ligand_pooled + receptor_pooled, dim=1)
        fused = self.batch_norm(fused)
        hidden = self.dense(fused)
        return self.final(hidden)


def build_paccmann() -> nn.Module:
    """Build a small PaccMann bimodal multiscale-attention model."""
    return PaccMannBimodalMCA(
        ligand_vocab=32,
        receptor_vocab=24,
        ligand_len=20,
        receptor_len=16,
        embed_dim=8,
        filters=(12, 12),
        kernel_sizes=(3, 5),
        attention_size=16,
        dense_hidden=20,
    ).eval()


def example_input_paccmann() -> tuple[Tensor, Tensor]:
    """Return (ligand SMILES tokens, receptor expression tokens) for PaccMann."""
    batch = 4
    ligand = torch.randint(0, 32, (batch, 20))
    receptor = torch.randint(0, 24, (batch, 16))
    return ligand, receptor


MENAGERIE_ENTRIES = [
    ("MOLI", "build_moli", "example_input_moli", "2019", "BIO"),
    ("NanoCaller", "build_nanocaller", "example_input_nanocaller", "2021", "BIO"),
    ("NeuSomatic", "build_neusomatic", "example_input_neusomatic", "2019", "BIO"),
    ("novoSpaRc", "build_novosparc", "example_input_novosparc", "2021", "BIO"),
    ("Omni-DNA", "build_omni_dna", "example_input_omni_dna", "2025", "BIO"),
    ("PaccMann", "build_paccmann", "example_input_paccmann", "2019", "BIO"),
]
