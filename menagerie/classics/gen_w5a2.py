"""Six faithful, compact reimplementations of named bioinformatics/medical-imaging models.

Sources checked (repo code and/or paper, no clone/pip-install; base-env torch reimplementation
of the distinctive mechanism):
  - DESC:       https://github.com/eleozzr/desc (desc/models/SAE.py, desc/models/network.py)
                Li et al., Nature Communications 2020. https://www.nature.com/articles/s41467-020-16822-0
  - DNCON2:     https://github.com/multicom-toolbox/DNCON2 (scripts/libcnnpredict.py,
                model-config-n-weights/model-arch.config). Adhikari et al., Bioinformatics 2018.
                https://pmc.ncbi.nlm.nih.gov/articles/PMC5925776/
  - DrugCell:   https://github.com/idekerlab/DrugCell (code/drugcell_NN.py).
                Kuenzi et al., Cancer Cell 2020 / Nature methods coverage.
                https://www.nature.com/articles/s41586-020-2804-6
  - DuDoRNet:   https://github.com/bbbbbbzhou/DuDoRNet (networks/networks.py,
                models/du_recurrent_model.py, models/utils.py). Zhou & Zaharchuk, CVPR 2020.
                arXiv:2001.03799
  - ERGO:       https://github.com/IdoSpringer/ERGO (ERGO_models.py: DoubleLSTMClassifier).
                Springer et al., Frontiers in Immunology 2020.
  - EternaFold: https://github.com/eternagame/EternaFold (src/Contrafold.cpp,
                src/InferenceEngine.*pp; README base-pairing-probability / partition-function
                usage). Wayment-Steele et al., Nature Methods 2022.
                https://www.nature.com/articles/s41592-022-01605-0
                The official tool is a C++ log-linear CRF (CONTRAfold-SE lineage) scored over
                hand-engineered RNA structural features and decoded with a McCaskill/Nussinov-
                style partition-function dynamic program -- not a neural network. The distinctive
                mechanism reimplemented here is that CRF-over-DP structure: learnable per-feature
                score tables (pairing propensity + stacking + loop-length terms, the trainable
                "parameters" file in the real tool) combined with a differentiable, soft
                (log-sum-exp) Nussinov-style DP that produces a base-pair probability matrix, plus
                the auxiliary multitask heads (structure-probing-evidence and binding-affinity
                regression) that give EternaFold its "multitask learning" identity over vanilla
                CONTRAfold.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# DESC -- Deep Embedding for Single-cell Clustering
# ---------------------------------------------------------------------------


class ClusteringLayer(nn.Module):
    """Student's t-distribution soft-assignment layer (DEC-style), as used by DESC."""

    def __init__(self, n_clusters: int, feature_dim: int, alpha: float = 1.0) -> None:
        """Initialize learnable cluster centers.

        Parameters
        ----------
        n_clusters:
            Number of clusters.
        feature_dim:
            Dimensionality of the embedded feature space.
        alpha:
            Degrees of freedom of the Student's t-distribution.
        """
        super().__init__()
        self.alpha = alpha
        self.clusters = nn.Parameter(torch.randn(n_clusters, feature_dim) * 0.1)

    def forward(self, z: Tensor) -> Tensor:
        """Compute soft cluster assignments q_ij via Student's t-kernel.

        Parameters
        ----------
        z:
            Embedded features, shape ``(N, feature_dim)``.

        Returns
        -------
        Tensor
            Soft assignment probabilities, shape ``(N, n_clusters)``.
        """
        dist_sq = torch.sum((z.unsqueeze(1) - self.clusters.unsqueeze(0)) ** 2, dim=2)
        q = 1.0 / (1.0 + dist_sq / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        return q / q.sum(dim=1, keepdim=True)


class DESC(nn.Module):
    """Stacked-autoencoder embedding + Student's-t clustering layer for scRNA-seq.

    Reimplements the DESC pipeline: a symmetric dense autoencoder (``dims``) maps
    normalized gene-expression vectors to a low-dimensional embedding; the embedding
    both reconstructs the input (denoising autoencoder objective) and feeds a
    :class:`ClusteringLayer` that produces soft cluster assignments, iteratively
    self-trained against a sharpened target distribution (KL self-training, not
    reproduced here since inference only needs a forward pass).
    """

    def __init__(self, dims: list[int] | None = None, n_clusters: int = 8) -> None:
        """Build the stacked-autoencoder + clustering head.

        Parameters
        ----------
        dims:
            Layer sizes ``[input_dim, hidden1, ..., embedding_dim]``.
        n_clusters:
            Number of target clusters for the DEC-style clustering layer.
        """
        super().__init__()
        if dims is None:
            dims = [64, 32, 10]
        enc_layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            enc_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                enc_layers.append(nn.ReLU())
            else:
                enc_layers.append(nn.Tanh())  # actincenter="tanh" in the original SAE
        self.encoder = nn.Sequential(*enc_layers)

        rev_dims = list(reversed(dims))
        dec_layers: list[nn.Module] = []
        for i in range(len(rev_dims) - 1):
            dec_layers.append(nn.Linear(rev_dims[i], rev_dims[i + 1]))
            if i < len(rev_dims) - 2:
                dec_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*dec_layers)

        self.clustering = ClusteringLayer(n_clusters, dims[-1])

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode, reconstruct, and softly cluster a batch of expression vectors.

        Parameters
        ----------
        x:
            Normalized expression matrix, shape ``(N, input_dim)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(reconstruction, cluster_assignment)``.
        """
        z = self.encoder(x)
        recon = self.decoder(z)
        q = self.clustering(z)
        return recon, q


def build_desc() -> nn.Module:
    """Construct a small DESC model.

    Returns
    -------
    nn.Module
        DESC in eval mode.
    """
    return DESC(dims=[64, 32, 10], n_clusters=6).eval()


def example_input_desc() -> Tensor:
    """Example input for DESC: a batch of normalized expression vectors.

    Returns
    -------
    Tensor
        Shape ``(16, 64)``.
    """
    return torch.randn(16, 64)


# ---------------------------------------------------------------------------
# DNCON2 -- two-stage residual/plain CNN ensemble for protein contact prediction
# ---------------------------------------------------------------------------


class DNCON2Stage1Net(nn.Module):
    """A single stage-1 2D-CNN that predicts residue-residue contacts at one distance threshold.

    Mirrors ``build_model_for_this_input_shape`` in ``libcnnpredict.py``: a stack of
    same-padded 2D convolutions (each optionally followed by BatchNorm) over an
    ``L x L x F`` stack of pairwise coevolution/sequence features (each 0D/1D feature
    is broadcast/outer-summed into an ``L x L`` map before stacking), ending in a
    single-channel contact-probability map.
    """

    def __init__(self, in_channels: int = 6, width: int = 12, n_layers: int = 4) -> None:
        """Build a stage-1 same-padded conv stack.

        Parameters
        ----------
        in_channels:
            Number of stacked pairwise feature channels (0D/1D features broadcast to
            2D, plus native 2D features such as CCMpred/PSICOV coupling maps).
        width:
            Number of channels in the hidden conv layers.
        n_layers:
            Number of convolutional layers.
        """
        super().__init__()
        layers: list[nn.Module] = []
        c_in = in_channels
        for i in range(n_layers):
            layers.append(nn.Conv2d(c_in, width, kernel_size=5, padding=2))
            layers.append(nn.BatchNorm2d(width))
            layers.append(nn.ReLU(inplace=True))
            c_in = width
        self.body = nn.Sequential(*layers)
        self.out = nn.Conv2d(width, 1, kernel_size=5, padding=2)

    def forward(self, x: Tensor) -> Tensor:
        """Predict a per-residue-pair contact logit map.

        Parameters
        ----------
        x:
            Stacked pairwise feature maps, shape ``(B, in_channels, L, L)``.

        Returns
        -------
        Tensor
            Contact logits, shape ``(B, 1, L, L)``.
        """
        return self.out(self.body(x))


class DNCON2(nn.Module):
    """Two-level ensemble CNN for protein residue-residue contact prediction.

    Stage 1 runs several :class:`DNCON2Stage1Net` instances (in the real tool, one per
    distance threshold, e.g. 10A/60A/75A/80A/85A) over the same pairwise feature stack;
    their contact-probability maps are stacked as new 2D "features" and fed, together
    with the original raw features, to a stage-2 CNN that produces the final ensembled
    contact map. This two-stage feature-stacking ensemble is DNCON2's signature
    mechanism (vs. single-CNN predecessors).
    """

    def __init__(self, in_channels: int = 6, n_stage1: int = 3, width: int = 12) -> None:
        """Build the two-stage ensemble.

        Parameters
        ----------
        in_channels:
            Number of raw pairwise feature channels.
        n_stage1:
            Number of stage-1 threshold-specific sub-networks in the ensemble.
        width:
            Hidden channel width shared by stage-1 and stage-2 nets.
        """
        super().__init__()
        self.stage1 = nn.ModuleList(
            [DNCON2Stage1Net(in_channels, width, n_layers=3) for _ in range(n_stage1)]
        )
        self.stage2 = DNCON2Stage1Net(in_channels + n_stage1, width, n_layers=3)

    def forward(self, x: Tensor) -> Tensor:
        """Run stage-1 ensemble then stage-2 ensembling CNN.

        Parameters
        ----------
        x:
            Stacked pairwise feature maps, shape ``(B, in_channels, L, L)``.

        Returns
        -------
        Tensor
            Final contact-probability map (post-sigmoid), shape ``(B, 1, L, L)``.
        """
        stage1_maps = [torch.sigmoid(net(x)) for net in self.stage1]
        stacked = torch.cat([x, *stage1_maps], dim=1)
        logits = self.stage2(stacked)
        return torch.sigmoid(logits)


def build_dncon2() -> nn.Module:
    """Construct a small DNCON2 two-stage ensemble.

    Returns
    -------
    nn.Module
        DNCON2 in eval mode.
    """
    return DNCON2(in_channels=6, n_stage1=3, width=8).eval()


def example_input_dncon2() -> Tensor:
    """Example input for DNCON2: a stack of pairwise feature maps for a short protein.

    Returns
    -------
    Tensor
        Shape ``(1, 6, 24, 24)`` (``L=24`` residues, 6 pairwise feature channels).
    """
    return torch.randn(1, 6, 24, 24)


# ---------------------------------------------------------------------------
# DrugCell -- Visible Neural Network over a Gene Ontology hierarchy
# ---------------------------------------------------------------------------


class DrugCell(nn.Module):
    """Visible Neural Network (VNN) wired directly onto a Gene Ontology sub-DAG.

    Reimplements the defining mechanism of ``drugcell_nn`` in ``drugcell_NN.py``: instead
    of an opaque MLP, the genotype branch is built by literally instantiating one hidden
    "term" unit per GO term, wired bottom-up so each term's linear layer consumes the
    concatenated outputs of its child terms plus any genes directly annotated to it. This
    fixed small hierarchy (3 levels: genes -> leaf terms -> root term) is a compact stand-in
    for the thousands-of-terms real ontology, preserving the "visible"/hierarchical wiring.
    The drug branch is a plain MLP; genotype and drug embeddings are concatenated into a
    final response-prediction head.
    """

    def __init__(
        self,
        n_genes: int = 20,
        n_leaf_terms: int = 4,
        genes_per_leaf: int = 5,
        term_hidden: int = 6,
        n_drug_features: int = 16,
        drug_hidden: int = 8,
    ) -> None:
        """Build a 3-level GO-DAG visible neural network plus a drug MLP branch.

        Parameters
        ----------
        n_genes:
            Total number of genes (mutation-indicator input dimension).
        n_leaf_terms:
            Number of leaf GO terms, each directly annotated with a disjoint gene subset.
        genes_per_leaf:
            Number of genes directly annotated to each leaf term.
        term_hidden:
            Hidden width used for every term (leaf and root) unit.
        n_drug_features:
            Dimensionality of the drug fingerprint/descriptor input.
        drug_hidden:
            Hidden width of the drug MLP branch.
        """
        super().__init__()
        self.n_genes = n_genes
        self.n_leaf_terms = n_leaf_terms
        self.genes_per_leaf = genes_per_leaf
        self.n_drug_features = n_drug_features

        # Direct-gene layers: one per leaf GO term, each reading the FULL gene vector
        # but effectively focused via its own learned weights (mirrors
        # `contruct_direct_gene_layer`, which also takes the full gene_dim as input).
        self.direct_gene_layers = nn.ModuleList(
            [nn.Linear(n_genes, genes_per_leaf) for _ in range(n_leaf_terms)]
        )
        self.leaf_term_layers = nn.ModuleList(
            [nn.Linear(genes_per_leaf, term_hidden) for _ in range(n_leaf_terms)]
        )
        self.leaf_bn = nn.ModuleList([nn.BatchNorm1d(term_hidden) for _ in range(n_leaf_terms)])

        # Root term: consumes the concatenation of all leaf-term outputs (mirrors
        # `construct_NN_graph`'s bottom-up wiring where a parent term's input size is
        # the sum of its children's term_dim).
        self.root_layer = nn.Linear(n_leaf_terms * term_hidden, term_hidden)
        self.root_bn = nn.BatchNorm1d(term_hidden)

        # Drug branch (mirrors `construct_NN_drug`).
        self.drug_layer1 = nn.Linear(n_drug_features, drug_hidden)
        self.drug_bn1 = nn.BatchNorm1d(drug_hidden)

        # Final fusion layer (mirrors the `final_linear_layer` / aux heads).
        self.final_linear = nn.Linear(term_hidden + drug_hidden, term_hidden)
        self.final_bn = nn.BatchNorm1d(term_hidden)
        self.final_out = nn.Linear(term_hidden, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Predict a scalar drug-response score from genotype + drug features.

        Parameters
        ----------
        x:
            Concatenated ``[gene_mutation_vector || drug_features]``, shape
            ``(B, n_genes + n_drug_features)``.

        Returns
        -------
        Tensor
            Predicted response score, shape ``(B, 1)``.
        """
        gene_input = x.narrow(1, 0, self.n_genes)
        drug_input = x.narrow(1, self.n_genes, self.n_drug_features)

        leaf_outs = []
        for direct_layer, term_layer, bn in zip(
            self.direct_gene_layers, self.leaf_term_layers, self.leaf_bn
        ):
            gene_feat = torch.tanh(direct_layer(gene_input))
            leaf_outs.append(torch.tanh(bn(term_layer(gene_feat))))
        genotype_embed = torch.tanh(self.root_bn(self.root_layer(torch.cat(leaf_outs, dim=1))))

        drug_embed = torch.tanh(self.drug_bn1(self.drug_layer1(drug_input)))

        fused = torch.cat([genotype_embed, drug_embed], dim=1)
        hidden = torch.tanh(self.final_bn(self.final_linear(fused)))
        return self.final_out(hidden)


def build_drugcell() -> nn.Module:
    """Construct a small DrugCell VNN.

    Returns
    -------
    nn.Module
        DrugCell in eval mode.
    """
    return DrugCell(
        n_genes=20,
        n_leaf_terms=4,
        genes_per_leaf=5,
        term_hidden=6,
        n_drug_features=16,
        drug_hidden=8,
    ).eval()


def example_input_drugcell() -> Tensor:
    """Example input for DrugCell: concatenated gene-mutation + drug-feature vector.

    Returns
    -------
    Tensor
        Shape ``(4, 36)`` (``20`` genes + ``16`` drug features, batch of 4).
    """
    return torch.randn(4, 36)


# ---------------------------------------------------------------------------
# DuDoRNet -- dual-domain recurrent network for MRI reconstruction
# ---------------------------------------------------------------------------


class SELayer1d(nn.Module):
    """Squeeze-and-Excitation channel gate for 2D feature maps."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        """Build the squeeze-excite MLP.

        Parameters
        ----------
        channels:
            Number of input/output channels.
        reduction:
            Channel-reduction ratio for the bottleneck MLP.
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 1), channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Rescale channels by a learned global-context gate.

        Parameters
        ----------
        x:
            Feature map, shape ``(B, C, H, W)``.

        Returns
        -------
        Tensor
            Channel-gated feature map, same shape as ``x``.
        """
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SEDilatedResidualDenseBlock(nn.Module):
    """Squeeze-Excite Dilated Residual Dense Block (SEDRDB), DuDoRNet's core primitive."""

    def __init__(self, g0: int, growth: int, n_conv: int, dilate_set: tuple[int, ...]) -> None:
        """Build a dilated, densely-connected conv stack with an SE gate.

        Parameters
        ----------
        g0:
            Input/output ("growth-rate-0") channel width.
        growth:
            Per-conv-layer growth-rate channel width.
        n_conv:
            Number of densely-connected dilated conv layers inside the block.
        dilate_set:
            Dilation rate used by each conv layer (length ``n_conv``).
        """
        super().__init__()
        self.convs = nn.ModuleList()
        c_in = g0
        for i in range(n_conv):
            dilation = dilate_set[i]
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(c_in, growth, 3, padding=dilation, dilation=dilation),
                    nn.ReLU(inplace=True),
                )
            )
            c_in += growth
        self.local_fusion = nn.Conv2d(c_in, g0, kernel_size=1)
        self.se = SELayer1d(g0, reduction=4)

    def forward(self, x: Tensor) -> Tensor:
        """Densely stack dilated conv features, fuse, gate, and residual-add.

        Parameters
        ----------
        x:
            Feature map, shape ``(B, g0, H, W)``.

        Returns
        -------
        Tensor
            Residual-refined feature map, same shape as ``x``.
        """
        feats = x
        for conv in self.convs:
            out = conv(feats)
            feats = torch.cat([feats, out], dim=1)
        fused = self.local_fusion(feats)
        gated = self.se(fused)
        return gated + x


class DRDN(nn.Module):
    """Dilated Residual Dense Network: DuDoRNet's per-domain image-restoration generator."""

    def __init__(
        self, n_channels: int = 2, g0: int = 16, n_blocks: int = 2, growth: int = 8
    ) -> None:
        """Build the shallow-feature-extraction + RDB-cascade + global-fusion generator.

        Parameters
        ----------
        n_channels:
            Number of input/output channels (2 for real+imaginary MRI data).
        g0:
            Base channel width used throughout the network.
        n_blocks:
            Number of cascaded :class:`SEDilatedResidualDenseBlock` blocks.
        growth:
            Growth-rate width inside each block.
        """
        super().__init__()
        self.sfe1 = nn.Conv2d(n_channels, g0, 3, padding=1)
        self.sfe2 = nn.Conv2d(g0, g0, 3, padding=1)
        dilate_set = (1, 2, 4, 4)
        self.blocks = nn.ModuleList(
            [
                SEDilatedResidualDenseBlock(g0, growth, n_conv=3, dilate_set=dilate_set)
                for _ in range(n_blocks)
            ]
        )
        self.global_fusion = nn.Sequential(
            nn.Conv2d(n_blocks * g0, g0, kernel_size=1),
            nn.Conv2d(g0, g0, 3, padding=1),
        )
        self.up = nn.Sequential(
            nn.Conv2d(g0, growth, 3, padding=1),
            nn.Conv2d(growth, n_channels, 3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Restore a real+imaginary image-domain or k-space-domain slice.

        Parameters
        ----------
        x:
            Input slice, shape ``(B, n_channels, H, W)``.

        Returns
        -------
        Tensor
            Restored slice, same shape as ``x``.
        """
        f1 = self.sfe1(x)
        feats = self.sfe2(f1)
        block_outs = []
        for block in self.blocks:
            feats = block(feats)
            block_outs.append(feats)
        fused = self.global_fusion(torch.cat(block_outs, dim=1))
        fused = fused + f1
        return self.up(fused)


class DuDoRNet(nn.Module):
    """Dual-domain (k-space + image) recurrent MRI reconstruction network.

    Reimplements the ``RecurrentModel`` alternation from ``du_recurrent_model.py``: two
    :class:`DRDN` generators (one operating in k-space, one in image space) are applied
    across ``n_recurrent`` unrolled steps; after each domain's generator, a soft
    data-consistency step re-injects the originally sampled k-space values at their known
    locations (approximated here in image space via a masked FFT/IFFT blend, since torch's
    ``fft`` ops are traceable and functionally equivalent to the reference's
    ``DataConsistencyInKspace_I/K`` layers).
    """

    def __init__(self, n_recurrent: int = 2, g0: int = 12) -> None:
        """Build the recurrent dual-domain reconstruction cascade.

        Parameters
        ----------
        n_recurrent:
            Number of unrolled image<->k-space recurrent refinement steps.
        g0:
            Base channel width shared by both domain generators.
        """
        super().__init__()
        self.n_recurrent = n_recurrent
        self.net_g_i = DRDN(n_channels=2, g0=g0, n_blocks=2, growth=8)
        self.net_g_k = DRDN(n_channels=2, g0=g0, n_blocks=2, growth=8)

    @staticmethod
    def _data_consistency(x: Tensor, k0: Tensor, mask: Tensor) -> Tensor:
        """Soft-blend predicted k-space with originally sampled k-space at known locations.

        Parameters
        ----------
        x:
            Image-domain estimate, shape ``(B, 2, H, W)`` (real, imag channels).
        k0:
            Originally sampled k-space values, same shape as ``x``.
        mask:
            Binary sampling mask, broadcastable to ``x``.

        Returns
        -------
        Tensor
            Data-consistent image-domain reconstruction, same shape as ``x``.
        """
        x_complex = torch.complex(x[:, 0], x[:, 1])
        k = torch.fft.fft2(x_complex)
        k0_complex = torch.complex(k0[:, 0], k0[:, 1])
        mask_c = mask[:, 0]
        k_dc = mask_c * k0_complex + (1 - mask_c) * k
        x_dc = torch.fft.ifft2(k_dc)
        return torch.stack([x_dc.real, x_dc.imag], dim=1)

    def forward(self, image_sub: Tensor, kspace_sub: Tensor, mask: Tensor) -> Tensor:
        """Run the recurrent image/k-space refinement cascade.

        Parameters
        ----------
        image_sub:
            Zero-filled image-domain reconstruction, shape ``(B, 2, H, W)``.
        kspace_sub:
            Originally sampled (sub-sampled) k-space, shape ``(B, 2, H, W)``.
        mask:
            Sampling mask, shape ``(B, 1, H, W)`` (broadcast over real/imag).

        Returns
        -------
        Tensor
            Final reconstructed image, shape ``(B, 2, H, W)``.
        """
        x = image_sub
        for _ in range(self.n_recurrent):
            x_i = self.net_g_i(x) + x
            x_i = self._data_consistency(x_i, kspace_sub, mask)

            x_k = self.net_g_k(x_i) + x_i
            x_k = self._data_consistency(x_k, kspace_sub, mask)
            x = x_k
        return x


def build_dudornet() -> nn.Module:
    """Construct a small DuDoRNet.

    Returns
    -------
    nn.Module
        DuDoRNet in eval mode.
    """
    return DuDoRNet(n_recurrent=2, g0=8).eval()


def example_input_dudornet() -> tuple[Tensor, Tensor, Tensor]:
    """Example input for DuDoRNet: zero-filled image, sub-sampled k-space, and mask.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(image_sub, kspace_sub, mask)`` each shaped ``(1, 2 or 1, 32, 32)``.
    """
    image_sub = torch.randn(1, 2, 32, 32)
    kspace_sub = torch.randn(1, 2, 32, 32)
    mask = torch.randint(0, 2, (1, 1, 32, 32)).float()
    return image_sub, kspace_sub, mask


# ---------------------------------------------------------------------------
# ERGO -- TCR-epitope binding predictor (dual-LSTM classifier)
# ---------------------------------------------------------------------------


class ERGO(nn.Module):
    """Dual-LSTM TCR-epitope binding classifier (DoubleLSTMClassifier from ERGO_models.py).

    Reimplements ERGO's default ``lstm`` model type: separate amino-acid embedding + 2-layer
    LSTM encoders process the T-cell-receptor (TCR) beta-chain and the peptide/epitope
    sequence independently, producing two fixed-size sequence embeddings (last hidden
    states) that are concatenated and passed through a small MLP to predict binding
    probability. Padding-aware packing is skipped here (fixed-length sequences, no
    padding) since ``pack_padded_sequence`` control flow is not traceable; a plain LSTM
    over pre-padded sequences captures the same encode-concatenate-classify mechanism.
    """

    def __init__(
        self,
        vocab_size: int = 21,
        embedding_dim: int = 10,
        lstm_dim: int = 16,
        dropout: float = 0.1,
    ) -> None:
        """Build the dual amino-acid-sequence LSTM encoders and MLP classifier head.

        Parameters
        ----------
        vocab_size:
            Amino-acid vocabulary size (20 residues + padding token).
        embedding_dim:
            Amino-acid embedding dimensionality.
        lstm_dim:
            Hidden size of each LSTM encoder.
        dropout:
            Dropout probability inside the LSTM stacks and the MLP head.
        """
        super().__init__()
        self.tcr_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pep_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.tcr_lstm = nn.LSTM(
            embedding_dim, lstm_dim, num_layers=2, batch_first=True, dropout=dropout
        )
        self.pep_lstm = nn.LSTM(
            embedding_dim, lstm_dim, num_layers=2, batch_first=True, dropout=dropout
        )
        self.hidden_layer = nn.Linear(lstm_dim * 2, lstm_dim)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.output_layer = nn.Linear(lstm_dim, 1)

    def forward(self, tcr: Tensor, pep: Tensor) -> Tensor:
        """Predict TCR-peptide binding probability.

        Parameters
        ----------
        tcr:
            TCR beta-chain amino-acid index sequence, shape ``(B, tcr_len)``.
        pep:
            Peptide/epitope amino-acid index sequence, shape ``(B, pep_len)``.

        Returns
        -------
        Tensor
            Binding probability, shape ``(B, 1)``.
        """
        tcr_embed = self.tcr_embedding(tcr)
        _, (tcr_h, _) = self.tcr_lstm(tcr_embed)
        tcr_last = tcr_h[-1]

        pep_embed = self.pep_embedding(pep)
        _, (pep_h, _) = self.pep_lstm(pep_embed)
        pep_last = pep_h[-1]

        fused = torch.cat([tcr_last, pep_last], dim=1)
        hidden = self.dropout(self.relu(self.hidden_layer(fused)))
        return torch.sigmoid(self.output_layer(hidden))


def build_ergo() -> nn.Module:
    """Construct a small ERGO dual-LSTM classifier.

    Returns
    -------
    nn.Module
        ERGO in eval mode.
    """
    return ERGO(vocab_size=21, embedding_dim=10, lstm_dim=16, dropout=0.0).eval()


def example_input_ergo() -> tuple[Tensor, Tensor]:
    """Example input for ERGO: fixed-length TCR and peptide amino-acid index sequences.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(tcr, pep)`` index tensors of shape ``(4, 20)`` and ``(4, 10)``.
    """
    tcr = torch.randint(1, 21, (4, 20))
    pep = torch.randint(1, 21, (4, 10))
    return tcr, pep


# ---------------------------------------------------------------------------
# EternaFold -- multitask CRF + differentiable Nussinov-style DP for RNA structure
# ---------------------------------------------------------------------------


class EternaFold(nn.Module):
    """Multitask CRF-over-DP RNA secondary-structure scorer (EternaFold/CONTRAfold-SE lineage).

    The official tool is a log-linear CRF: hand-engineered structural features (base-pair
    identity, helix stacking, loop-length terms) are scored by trained feature weights, and
    a McCaskill/Nussinov-style partition-function dynamic program turns those scores into
    base-pairing probabilities. This reimplements that CRF-over-DP structure directly:

      1. A learnable per-(base-pair-type) score table plus a small pairwise-context MLP
         produce a symmetric pairwise compatibility score matrix from one-hot sequence
         features -- the "trained parameters" role in the real tool.
      2. A **soft, differentiable Nussinov recursion** (log-sum-exp over the classic
         ``DP[i,j] = max(DP[i+1,j], max_k DP[i+1,k-1] + pair(i,k) + DP[k+1,j])`` split,
         unrolled for a fixed number of diagonal sweeps since real dynamic-programming
         control flow is not traceable) integrates local pair scores into a global
         base-pairing-probability matrix, mirroring the real partition-function decoder.
      3. Two small multitask heads regress structure-probing-evidence likelihood and
         ligand/protein binding affinity from the pooled sequence encoding, mirroring
         EternaFold's defining "multitask learning" extension over vanilla CONTRAfold.
    """

    def __init__(self, vocab_size: int = 4, hidden: int = 16, seq_len: int = 20) -> None:
        """Build the pairwise CRF scorer, soft-DP integrator, and multitask heads.

        Parameters
        ----------
        vocab_size:
            Nucleotide vocabulary size (A, C, G, U).
        hidden:
            Hidden width of the per-nucleotide context encoder.
        seq_len:
            Fixed sequence length used to size the diagonal-sweep DP unroll.
        """
        super().__init__()
        self.seq_len = seq_len
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.context = nn.LSTM(hidden, hidden, batch_first=True, bidirectional=True)
        # Pairwise compatibility scorer: learned bilinear form over context features,
        # standing in for CONTRAfold's trained base-pair + stacking feature weights.
        self.pair_score = nn.Bilinear(2 * hidden, 2 * hidden, 1)
        # Loop-length / local penalty MLP (stand-in for hairpin/bulge/multiloop terms).
        self.loop_penalty = nn.Sequential(
            nn.Linear(1, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1)
        )

        self.evidence_head = nn.Linear(2 * hidden, 1)
        self.affinity_head = nn.Linear(2 * hidden, 1)

    def forward(self, seq: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Score all base pairs and produce multitask predictions.

        Parameters
        ----------
        seq:
            Nucleotide index sequence, shape ``(B, seq_len)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``(base_pair_probs, evidence_likelihood, binding_affinity)`` where
            ``base_pair_probs`` has shape ``(B, seq_len, seq_len)``.
        """
        embed = self.embedding(seq)
        ctx, _ = self.context(embed)  # (B, L, 2*hidden)
        batch, length, feat = ctx.shape

        ctx_i = ctx.unsqueeze(2).expand(batch, length, length, feat)
        ctx_j = ctx.unsqueeze(1).expand(batch, length, length, feat)
        pair_logits = self.pair_score(ctx_i.reshape(-1, feat), ctx_j.reshape(-1, feat))
        pair_logits = pair_logits.view(batch, length, length)

        idx = torch.arange(length, device=seq.device, dtype=ctx.dtype)
        loop_len = (idx.view(1, -1) - idx.view(-1, 1)).abs().unsqueeze(-1)
        loop_pen = self.loop_penalty(loop_len.reshape(-1, 1)).view(length, length)
        pair_logits = pair_logits + loop_pen.unsqueeze(0)

        # Enforce symmetry and mask the diagonal/near-diagonal (no sharp turns), as the
        # real DP forbids pairs closer than a minimum hairpin-loop distance.
        pair_logits = 0.5 * (pair_logits + pair_logits.transpose(1, 2))
        min_loop = 3
        forbid = (idx.view(1, -1) - idx.view(-1, 1)).abs() <= min_loop
        pair_logits = pair_logits.masked_fill(forbid.unsqueeze(0), -1e4)

        # Soft Nussinov-style DP: a fixed number of diagonal sweeps accumulating
        # log-sum-exp evidence for "i and j are both paired, possibly through an
        # intervening pair (i+1, k)". This is an unrolled (traceable) relaxation of the
        # real O(L^3) dynamic program.
        dp = pair_logits
        for _ in range(3):
            shifted = F.pad(dp[:, 1:, :-1], (0, 1, 0, 1), value=-1e4)
            dp = torch.logsumexp(torch.stack([dp, shifted + pair_logits], dim=0), dim=0)
        base_pair_probs = torch.sigmoid(dp)

        pooled = ctx.mean(dim=1)
        evidence = torch.sigmoid(self.evidence_head(pooled))
        affinity = self.affinity_head(pooled)
        return base_pair_probs, evidence, affinity


def build_eternafold() -> nn.Module:
    """Construct a small EternaFold multitask CRF-DP model.

    Returns
    -------
    nn.Module
        EternaFold in eval mode.
    """
    return EternaFold(vocab_size=4, hidden=12, seq_len=20).eval()


def example_input_eternafold() -> Tensor:
    """Example input for EternaFold: a nucleotide index sequence.

    Returns
    -------
    Tensor
        Shape ``(2, 20)`` (batch of 2 RNA sequences of length 20, A/C/G/U indices).
    """
    return torch.randint(0, 4, (2, 20))


MENAGERIE_ENTRIES = [
    ("DESC", "build_desc", "example_input_desc", "2020", "BIO"),
    ("DNCON2", "build_dncon2", "example_input_dncon2", "2018", "BIO"),
    ("DrugCell", "build_drugcell", "example_input_drugcell", "2020", "BIO"),
    ("DuDoRNet", "build_dudornet", "example_input_dudornet", "2020", "VIS"),
    ("ERGO", "build_ergo", "example_input_ergo", "2020", "BIO"),
    ("EternaFold", "build_eternafold", "example_input_eternafold", "2022", "BIO"),
]
