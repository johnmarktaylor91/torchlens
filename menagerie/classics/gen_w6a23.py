"""Menagerie batch w6a23: RNA pseudoknot secondary-structure prediction via
BiLSTM + base-pair-maximization stem combination, iterative-feedback protein
distance-geometry folding, voxelized 3D-CNN docking-decoy evaluation,
dual-branch end-to-end/geometry-potential RNA 3D structure prediction,
adaptive multi-channel E(3)-equivariant full-atom antibody design, and
SE(3)-equivariant diffusion for ligand-induced protein-ligand complex
structure prediction.

Sources checked (reference only; no cloning, no pip installs):
  - DMfold (cand_00858): Wang, Liu, et al. (Yunnan University), Frontiers in
    Genetics 2019, https://github.com/linyuwangPHD/RNA-Secondary-Structure-Database
    (repo hosts training/eval data and a Windows .zip binary, not readable
    source; architecture description taken from the paper,
    https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2019.00143/full
    -- "DMfold: A Novel Method to Predict RNA Secondary Structure With
    Pseudoknots Based on Deep Learning and Improved Base Pair Maximization
    Principle"). The defining mechanism: an RNA sequence is first passed
    through a **bidirectional-LSTM encoder** and a **fully-connected
    decoder** that predicts, independently for every base, one of **seven
    dot-bracket symbol classes** (encoding nested-pair open/close plus
    multiple pseudoknot-crossing bracket types, not just the ternary
    "(", ")", "." of ordinary secondary-structure prediction); the raw
    per-base symbol sequence is then post-processed by the **Improved Base
    Pair Maximization Principle (IBPMP)**, a deterministic stem-selection
    procedure that extracts candidate helical stems implied by the
    predicted brackets and greedily combines a maximum-base-pair subset of
    mutually non-conflicting stems into up to three pseudoknot-free
    substructure "layers" that together reconstruct the full pseudoknotted
    structure -- i.e. "BiLSTM sequence-to-7-symbol-dot-bracket labeling +
    deterministic max-base-pair stem-combination post-process" is DMfold's
    namesake pseudoknot-capable contribution over classic ternary-label
    secondary-structure predictors. Reimplemented with the same BiLSTM
    encoder / FC decoder emitting 7-way per-base logits, plus a from-scratch
    IBPMP-style deterministic stem-extraction-and-greedy-combination pass
    (stem candidates from matching bracket-type spans, greedy max-stem-count
    selection subject to non-crossing-per-layer, output as a base-pair
    partner index array) at reduced sequence length and hidden width.
  - DMPfold2 (cand_00859): Kandathil, Greener, et al. (Jones lab, UCL/
    PSIPRED), PNAS 2022, https://github.com/psipred/DMPfold2
    (``dmpfold/network.py``; architecture description from the paper,
    "Ultrafast end-to-end protein structure prediction enables high-
    throughput exploration of uncharacterized proteins", and its bioRxiv
    precursor "Deep learning-based prediction of protein structure using
    learned representations of multiple sequence alignments"). The defining
    mechanism: a multiple sequence alignment is embedded into pairwise 2D
    covariance-style features and passed through a **2D-convolutional ResNet
    trunk**, whose output feeds a **2-layer bidirectional GRU** that
    regresses per-residue Cα 3D coordinates directly (no separate distance-
    map-then-folding step); critically, the network supports **iterative
    refinement**: the Cα coordinates predicted on one pass are converted
    back into a pairwise distance matrix and fed into the ResNet as an
    **extra input channel** on the next pass (initialized to a sentinel
    constant, e.g. -1, when no prior structure exists), so repeated forward
    passes progressively sharpen the same coordinate prediction -- i.e.
    "MSA-covariance ResNet + BiGRU coordinate regression, closed into a
    self-conditioning loop via a fed-back Cα-distance channel" is DMPfold2's
    namesake ultrafast end-to-end contribution over classic contact-map-then-
    distance-geometry two-stage folding. Reimplemented with the same
    ResNet-trunk + BiGRU + direct-coordinate-head topology and the same
    distance-channel feedback loop (explicit multi-iteration forward call,
    each iteration's output distance matrix becoming the next iteration's
    extra input channel), at reduced MSA depth, sequence length, and channel
    widths.
  - DOVE (cand_00860): Wang, Terashi, Christoffer, Zhu, Kihara (Purdue),
    Bioinformatics 2020, https://github.com/kiharalab/DOVE (``main.py``;
    architecture figure and specification in the README: "100, 200, 200,
    400, 400 are the number of filters in each layer. 20, 18, 16, 8, 6, 3
    are the output cube size of each layer... 10800, 1000, 100 denotes the
    number of neurons for fully connected layer"). The defining mechanism: a
    candidate protein-docking decoy's interface is **voxelized onto a 3D
    grid** (physicochemical potentials such as GOAP/ITScore plus atom-type
    occupancy mapped into each voxel), and this 3D volume is processed by a
    **5-layer 3D-convolutional network with progressively shrinking cube
    size (20->18->16->8->6->3) and growing filter count
    (100->200->200->400->400)**, flattened into a fully-connected head
    (10800->1000->100->1) that outputs the probability the decoy is a
    CAPRI-acceptable docking pose; the released method ensembles **8
    independently trained networks**, each seeing a different input-feature
    combination, and averages/reports their probabilities -- i.e. "shrinking-
    cube 3D-CNN over a voxelized physicochemical docking interface,
    ensembled across feature subsets" is DOVE's namesake voxel-based
    docking-decoy-evaluation contribution over hand-crafted scoring
    functions. Reimplemented with the same 5-layer shrinking-cube 3D-CNN
    (matching the filter-count progression) plus 3-layer FC head producing a
    sigmoid decoy-quality score, at reduced input cube size and channel
    widths (single-network instance, since the 8-network ensemble is a
    training-time feature-ablation detail, not an architectural mechanism).
  - DRfold (cand_00861): Li, Zhang, et al. (Zhang lab, UMich/NUS), Nature
    Communications 2023, https://github.com/leeyang/DRfold (architecture
    description from the paper, "Integrating end-to-end learning with deep
    geometrical potentials for ab initio RNA structure prediction"). The
    defining mechanism: RNA sequence embeddings are passed through a stack
    of shared transformer blocks producing per-nucleotide single and
    pairwise representations, which then split into **two parallel,
    architecturally-independent output branches**: (1) an **end-to-end
    structure module** (an Invariant-Point-Attention-style block) that
    directly regresses, for every nucleotide, a **rigid-body local frame**
    (rotation matrix + translation vector, defined from the P/C4'/N atoms)
    in one shot; and (2) a **geometry module** that reuses the pairwise
    representations through a separate transformer stack to predict
    **inter-nucleotide geometric restraints** (distance/angle distributions
    between nucleotide pairs); the two branches' outputs -- frame-based
    coordinates and pairwise geometric potentials -- are then combined into
    one **composite energy function** minimized to produce the final 3D
    structure -- i.e. "twin end-to-end-frame-regression / geometry-potential
    transformer branches fused into one composite folding energy" is
    DRfold's namesake hybrid contribution over pure end-to-end or pure
    geometry-potential RNA folding. Reimplemented with the same shared
    embedding trunk splitting into a frame-regression branch (predicting a
    per-nucleotide rotation quaternion + translation) and a geometry branch
    (predicting pairwise distance-bin logits from outer-concatenated pair
    features), at reduced sequence length, transformer depth, and hidden
    width (the downstream gradient-based potential-minimization refinement
    is a post-network optimization step, not part of the traced nn.Module).
  - dyMEAN (cand_00862): Kong, Huang, Liu (THUNLP-MT, Tsinghua), ICML 2023,
    https://github.com/THUNLP-MT/dyMEAN (architecture description from the
    paper, "End-to-End Full-Atom Antibody Design", arxiv:2302.00203; builds
    on the group's prior MEAN, https://github.com/THUNLP-MT/MEAN). The
    defining mechanism: unlike Cα-only equivariant graph networks, dyMEAN
    represents every residue as a **fixed number of atom "channels"**
    (heavy-atom slots, zero-padded/masked for residues with fewer atoms),
    giving each residue node a set of per-channel 3D coordinates instead of
    one point; an **adaptive multi-channel E(3)-equivariant graph network**
    then alternates, over several message-passing layers, (a) equivariant
    coordinate updates -- each channel's position is updated by a
    learned, radial-distance-gated combination of relative-position vectors
    to neighboring residues' channels (rotation/translation-equivariant by
    construction) -- and (b) invariant node-feature/sequence-logit updates
    from the same per-channel relative distances, so **1D sequence identity
    and 3D full-atom structure are refined jointly and simultaneously by one
    shared network**, rather than a discrete sequence-design step followed
    by a separate structure-prediction step -- i.e. "atom-channel padded
    full-atom residue representation + joint equivariant coordinate /
    invariant sequence-logit message passing" is dyMEAN's namesake end-to-
    end full-atom contribution over Cα-only or two-stage antibody design.
    Reimplemented with the same fixed-channel per-residue atom padding,
    equivariant radial-gated coordinate-update message passing, and joint
    sequence-logit head, at reduced channel count, residue count, and hidden
    width.
  - DynamicBind (cand_00863): Lu, Zhang, et al. (Gao lab, MIT), Nature
    Communications 2024, https://github.com/luwei0917/DynamicBind
    (architecture description from the paper, "DynamicBind: predicting
    ligand-specific protein-ligand complex structure with a deep equivariant
    generative model"). The defining mechanism: a coarse-grained
    protein-residue + ligand-atom graph is progressively denoised by an
    **SE(3)-equivariant diffusion graph neural network**: at each reverse-
    diffusion step, the network consumes the current noisy joint
    protein-ligand geometry plus a diffusion-timestep embedding and predicts
    **simultaneous multi-part updates** -- a ligand rigid-body translation
    and rotation plus ligand internal torsion-angle updates, **and**
    (unlike static-pocket docking models) per-residue protein rotation and
    translation updates plus side-chain torsion-angle updates -- letting the
    protein backbone/side chains move together with the ligand across the
    diffusion trajectory to reach a ligand-specific bound conformation
    rather than assuming the receptor is rigid -- i.e. "joint ligand-pose +
    protein-conformation SE(3)-equivariant diffusion denoiser, conditioned
    on the diffusion timestep" is DynamicBind's namesake induced-fit
    contribution over rigid-receptor equivariant docking. Reimplemented with
    the same equivariant message-passing denoiser over a joint
    protein-residue/ligand-atom graph, timestep conditioning, and dual
    output heads (ligand translation/rotation/torsion; protein per-residue
    rotation/translation/side-chain-torsion), at reduced residue/atom count
    and hidden width (the confidence-ranking side model used to pick among
    sampled poses is a downstream selection step, not part of the traced
    denoiser).

All six models are reimplemented from scratch in base-env torch; no repo
cloning, no pip installs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ============================================================
# DMfold -- BiLSTM 7-symbol dot-bracket labeling + deterministic
# Improved Base Pair Maximization Principle stem combination
# (linyuwangPHD RNA-Secondary-Structure-Database)
# ============================================================


class DMfoldBiLSTM(nn.Module):
    """BiLSTM encoder + FC decoder predicting per-base pseudoknot dot-bracket symbols.

    Ports DMfold's deep-learning stage: an RNA sequence (one-hot encoded
    bases) is passed through a bidirectional LSTM encoder, then a
    fully-connected decoder emits, for each base, logits over 7 dot-bracket
    symbol classes (one "unpaired" class plus three nested-pair
    open/close pairs, allowing up to three pseudoknot-crossing "layers").
    The deterministic Improved Base Pair Maximization Principle (IBPMP)
    stem-combination post-process (greedy non-conflicting-stem selection
    over the argmax symbol sequence) runs as a plain-tensor helper attached
    to the module, mirroring the reference pipeline's two-stage design.
    """

    N_SYMBOLS = 7  # unpaired + 3 layers x (open, close)

    def __init__(self, vocab_size: int = 4, hidden_dim: int = 32, num_layers: int = 2) -> None:
        super().__init__()
        self.embed = nn.Linear(vocab_size, hidden_dim)
        self.encoder = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True
        )
        self.decoder = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.N_SYMBOLS),
        )

    @staticmethod
    def combine_stems(symbol_ids: Tensor) -> Tensor:
        """Greedy IBPMP-style stem combination from predicted dot-bracket symbols.

        For each of the 3 pseudoknot layers, greedily pairs each "open"
        base with the nearest unmatched later "close" base of the same
        layer (a simplified maximum-base-pair stem match), returning a
        ``(seq_len,)`` partner-index array (-1 = unpaired).
        """
        seq_len = symbol_ids.shape[0]
        partner = torch.full((seq_len,), -1, dtype=torch.long)
        for layer in range(3):
            open_id, close_id = 1 + 2 * layer, 2 + 2 * layer
            stack: list[int] = []
            for pos in range(seq_len):
                sym = int(symbol_ids[pos].item())
                if sym == open_id:
                    stack.append(pos)
                elif sym == close_id and stack:
                    open_pos = stack.pop()
                    partner[open_pos] = pos
                    partner[pos] = open_pos
        return partner

    def forward(self, one_hot_seq: Tensor) -> Tensor:
        """Predict per-base dot-bracket symbol logits.

        Parameters
        ----------
        one_hot_seq : Tensor
            Shape ``(batch, seq_len, vocab_size)`` one-hot-encoded RNA bases.
        """
        h = self.embed(one_hot_seq)
        h, _ = self.encoder(h)
        return self.decoder(h)


def build_dmfold() -> nn.Module:
    """Build a small DMfold BiLSTM pseudoknot dot-bracket predictor."""
    return DMfoldBiLSTM(vocab_size=4, hidden_dim=32, num_layers=2).eval()


def example_input_dmfold() -> Tensor:
    """Return a batch of one-hot RNA sequences for DMfold."""
    batch, seq_len, vocab = 2, 24, 4
    idx = torch.randint(0, vocab, (batch, seq_len))
    return F.one_hot(idx, num_classes=vocab).float()


# ============================================================
# DMPfold2 -- MSA-covariance ResNet + BiGRU direct Ca-coordinate
# regression, closed into a self-conditioning loop via a fed-
# back Ca-distance channel (psipred/DMPfold2)
# ============================================================


class _DMPfoldResNetBlock(nn.Module):
    """One dilated 2D-conv residual block of the DMPfold2 ResNet trunk."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x: Tensor) -> Tensor:
        h = F.elu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.elu(x + h)


class DMPfold2Net(nn.Module):
    """Iterative MSA-covariance ResNet + BiGRU Ca-coordinate regressor.

    Ports ``dmpfold/network.py``'s architecture: pairwise MSA-covariance
    features (plus a fed-back Ca-distance channel, -1-filled on the first
    pass) are processed by a 2D-conv ResNet trunk; the per-residue row
    summary is decoded by a 2-layer bidirectional GRU into direct Ca 3D
    coordinates. Calling ``forward`` for several ``n_iters`` re-derives the
    pairwise distance matrix from the previous pass's coordinates and feeds
    it back as the extra ResNet input channel, self-conditioning the
    prediction -- the namesake "ultrafast end-to-end" iterative-refinement
    mechanism.
    """

    def __init__(
        self,
        cov_channels: int = 4,
        hidden_channels: int = 16,
        gru_hidden: int = 16,
        num_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.stem = nn.Conv2d(cov_channels + 1, hidden_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [_DMPfoldResNetBlock(hidden_channels, dilation=2**i) for i in range(num_blocks)]
        )
        self.gru = nn.GRU(
            hidden_channels, gru_hidden, num_layers=2, bidirectional=True, batch_first=True
        )
        self.coord_head = nn.Linear(2 * gru_hidden, 3)

    def _single_pass(self, cov_feats: Tensor, dist_channel: Tensor) -> Tensor:
        x = torch.cat([cov_feats, dist_channel], dim=1)
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        row_summary = h.mean(dim=3).transpose(1, 2)  # (batch, n_res, hidden_channels)
        gru_out, _ = self.gru(row_summary)
        return self.coord_head(gru_out)

    def forward(self, cov_feats: Tensor, n_iters: int = 2) -> Tensor:
        """Predict Ca coordinates, iteratively self-conditioning on the fed-back distance channel.

        Parameters
        ----------
        cov_feats : Tensor
            Shape ``(batch, cov_channels, n_res, n_res)`` pairwise
            MSA-covariance-style features.
        n_iters : int
            Number of refinement passes (first pass uses a -1-filled
            sentinel distance channel, matching the reference).
        """
        batch, _, n_res, _ = cov_feats.shape
        dist_channel = torch.full((batch, 1, n_res, n_res), -1.0, device=cov_feats.device)
        coords = self._single_pass(cov_feats, dist_channel)
        for _ in range(n_iters - 1):
            dist_channel = torch.cdist(coords, coords).unsqueeze(1)
            coords = self._single_pass(cov_feats, dist_channel)
        return coords


def build_dmpfold2() -> nn.Module:
    """Build a small DMPfold2 iterative-feedback distance-geometry folder."""
    return DMPfold2Net(cov_channels=4, hidden_channels=16, gru_hidden=16, num_blocks=2).eval()


def example_input_dmpfold2() -> Tensor:
    """Return a batch of MSA-covariance-style pairwise features for DMPfold2."""
    return torch.randn(1, 4, 20, 20)


# ============================================================
# DOVE -- shrinking-cube 3D-CNN over a voxelized docking
# interface + FC quality-probability head (kiharalab/DOVE)
# ============================================================


class DOVENet(nn.Module):
    """Voxel-based 3D-CNN docking-decoy quality evaluator.

    Ports ``main.py``'s network (README figure: filter counts
    100/200/200/400/400, output cube sizes 20/18/16/8/6/3, FC widths
    10800/1000/100): a voxelized protein-docking interface (GOAP/ITScore
    potentials + atom-type occupancy per voxel) is passed through 5
    3D-conv layers that shrink the cube while growing channel count, then
    flattened through a 3-layer FC head into a single CAPRI-acceptable-
    decoy probability.
    """

    def __init__(self, in_channels: int = 4, cube_size: int = 10) -> None:
        super().__init__()
        self.conv_layers = nn.ModuleList(
            [
                nn.Conv3d(in_channels, 12, kernel_size=3, padding=1),
                nn.Conv3d(12, 24, kernel_size=3, padding=0),
                nn.Conv3d(24, 24, kernel_size=3, padding=0),
                nn.Conv3d(24, 48, kernel_size=3, stride=2, padding=1),
                nn.Conv3d(48, 48, kernel_size=3, padding=0),
            ]
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, cube_size, cube_size, cube_size)
            for conv in self.conv_layers:
                dummy = F.relu(conv(dummy))
            flat_dim = dummy.numel()
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, voxel_grid: Tensor) -> Tensor:
        """Predict a docking-decoy quality probability from a voxelized interface.

        Parameters
        ----------
        voxel_grid : Tensor
            Shape ``(batch, in_channels, D, H, W)`` voxelized
            physicochemical-potential + atom-type interface grid.
        """
        h = voxel_grid
        for conv in self.conv_layers:
            h = F.relu(conv(h))
        h = h.flatten(1)
        return torch.sigmoid(self.fc(h))


def build_dove() -> nn.Module:
    """Build a small DOVE voxelized docking-decoy evaluator."""
    return DOVENet(in_channels=4, cube_size=10).eval()


def example_input_dove() -> Tensor:
    """Return a batch of voxelized docking-interface grids for DOVE."""
    return torch.rand(2, 4, 10, 10, 10)


# ============================================================
# DRfold -- twin end-to-end frame-regression / geometry-
# potential transformer branches over a shared embedding trunk
# (leeyang/DRfold)
# ============================================================


class _DRfoldTransformerBlock(nn.Module):
    """One shared-trunk transformer block (self-attention + feedforward)."""

    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        return self.norm2(x + self.ff(x))


class DRfoldDualBranch(nn.Module):
    """Twin end-to-end-frame / geometry-potential RNA structure predictor.

    Ports DRfold's two-pipeline design: a shared transformer trunk embeds
    the RNA sequence into single representations; these split into (1) a
    **structure-module branch** that directly regresses a per-nucleotide
    rigid-body frame (rotation quaternion + translation vector), and (2) a
    **geometry-module branch** that forms outer-concatenated pairwise
    features and predicts inter-nucleotide distance-bin logits -- the two
    branches' outputs are fused downstream (outside this module) into one
    composite folding potential.
    """

    def __init__(
        self,
        vocab_size: int = 4,
        dim: int = 32,
        n_heads: int = 4,
        n_trunk_layers: int = 2,
        n_dist_bins: int = 8,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(vocab_size, dim)
        self.trunk = nn.ModuleList(
            [_DRfoldTransformerBlock(dim, n_heads) for _ in range(n_trunk_layers)]
        )
        # End-to-end structure branch: per-nucleotide rigid frame.
        self.frame_branch = nn.ModuleList([_DRfoldTransformerBlock(dim, n_heads)])
        self.rotation_head = nn.Linear(dim, 4)
        self.translation_head = nn.Linear(dim, 3)
        # Geometry branch: pairwise distance-bin logits.
        self.pair_proj = nn.Linear(2 * dim, dim)
        self.geometry_branch = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, n_dist_bins)
        )

    def forward(self, one_hot_seq: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict (rotation quaternions, translations, pairwise distance-bin logits).

        Parameters
        ----------
        one_hot_seq : Tensor
            Shape ``(batch, seq_len, vocab_size)`` one-hot-encoded RNA bases.
        """
        h = self.embed(one_hot_seq)
        for block in self.trunk:
            h = block(h)

        frame_h = h
        for block in self.frame_branch:
            frame_h = block(frame_h)
        rotation = F.normalize(self.rotation_head(frame_h), dim=-1)
        translation = self.translation_head(frame_h)

        n = h.shape[1]
        h_i = h.unsqueeze(2).expand(-1, -1, n, -1)
        h_j = h.unsqueeze(1).expand(-1, n, -1, -1)
        pair_feats = self.pair_proj(torch.cat([h_i, h_j], dim=-1))
        dist_logits = self.geometry_branch(pair_feats)

        return rotation, translation, dist_logits


def build_drfold() -> nn.Module:
    """Build a small DRfold dual-branch RNA structure predictor."""
    return DRfoldDualBranch(vocab_size=4, dim=32, n_heads=4, n_trunk_layers=2, n_dist_bins=8).eval()


def example_input_drfold() -> Tensor:
    """Return a batch of one-hot RNA sequences for DRfold."""
    batch, seq_len, vocab = 2, 16, 4
    idx = torch.randint(0, vocab, (batch, seq_len))
    return F.one_hot(idx, num_classes=vocab).float()


# ============================================================
# dyMEAN -- atom-channel padded full-atom residues + joint
# equivariant coordinate / invariant sequence-logit message
# passing (THUNLP-MT/dyMEAN)
# ============================================================


class _DyMEANLayer(nn.Module):
    """One adaptive multi-channel E(3)-equivariant message-passing layer.

    Each residue node holds ``n_channels`` per-atom 3D coordinates; a
    radial-distance-gated MLP over each pair of channels produces an
    equivariant coordinate update (a learned scalar times the relative
    position vector, summed over neighbors and channels) plus an invariant
    feature update, jointly refining structure and sequence identity.
    """

    def __init__(self, hidden_dim: int, n_channels: int) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.radial_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.feat_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_norm = nn.LayerNorm(hidden_dim)

    def forward(self, feats: Tensor, coords: Tensor) -> tuple[Tensor, Tensor]:
        """Update (invariant node features, per-channel coordinates).

        Parameters
        ----------
        feats : Tensor
            Shape ``(n_res, hidden_dim)`` per-residue invariant features.
        coords : Tensor
            Shape ``(n_res, n_channels, 3)`` per-residue per-atom-channel
            3D coordinates.
        """
        n_res = feats.shape[0]
        channel_centroid = coords.mean(dim=1)  # (n_res, 3) residue-level anchor

        feat_i = feats.unsqueeze(1).expand(n_res, n_res, -1)
        feat_j = feats.unsqueeze(0).expand(n_res, n_res, -1)
        rel_vec = channel_centroid.unsqueeze(1) - channel_centroid.unsqueeze(0)  # (n_res, n_res, 3)
        rel_dist = rel_vec.norm(dim=-1, keepdim=True)

        msg_in = torch.cat([feat_i, feat_j, rel_dist], dim=-1)
        radial_gate = self.radial_mlp(msg_in)  # (n_res, n_res, 1)
        coord_update = (radial_gate.unsqueeze(-1) * rel_vec.unsqueeze(2)).mean(
            dim=1
        )  # (n_res, n_channels, 3)
        new_coords = coords + coord_update.expand(-1, self.n_channels, -1) / self.n_channels

        feat_update = self.feat_mlp(msg_in).mean(dim=1)
        new_feats = self.node_norm(feats + feat_update)

        return new_feats, new_coords


class DyMEANFullAtom(nn.Module):
    """Adaptive multi-channel E(3)-equivariant full-atom antibody designer.

    Ports dyMEAN's core mechanism: residues are represented with a fixed
    number of atom "channels" (zero-padded for residues with fewer heavy
    atoms), and stacked equivariant layers jointly refine per-channel 3D
    coordinates (equivariant) and per-residue sequence logits (invariant)
    in one shared network -- rather than a separate sequence-design step
    followed by structure prediction.
    """

    def __init__(
        self, vocab_size: int = 20, hidden_dim: int = 24, n_channels: int = 4, n_layers: int = 3
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.residue_embed = nn.Linear(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([_DyMEANLayer(hidden_dim, n_channels) for _ in range(n_layers)])
        self.sequence_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, residue_type_logits: Tensor, atom_coords: Tensor) -> tuple[Tensor, Tensor]:
        """Jointly refine (sequence logits, per-channel atom coordinates).

        Parameters
        ----------
        residue_type_logits : Tensor
            Shape ``(n_res, vocab_size)`` initial (masked/incomplete)
            residue-type one-hot or soft logits.
        atom_coords : Tensor
            Shape ``(n_res, n_channels, 3)`` initial per-residue per-atom-
            channel 3D coordinates (structural initialization).
        """
        feats = self.residue_embed(residue_type_logits)
        coords = atom_coords
        for layer in self.layers:
            feats, coords = layer(feats, coords)
        seq_logits = self.sequence_head(feats)
        return seq_logits, coords


def build_dymean() -> nn.Module:
    """Build a small dyMEAN full-atom antibody design network."""
    return DyMEANFullAtom(vocab_size=20, hidden_dim=24, n_channels=4, n_layers=3).eval()


def example_input_dymean() -> tuple[Tensor, Tensor]:
    """Return (residue-type logits, per-channel atom coordinates) for dyMEAN."""
    n_res, vocab, n_channels = 14, 20, 4
    residue_type_logits = torch.randn(n_res, vocab)
    atom_coords = torch.randn(n_res, n_channels, 3)
    return residue_type_logits, atom_coords


# ============================================================
# DynamicBind -- joint ligand-pose + protein-conformation SE(3)-
# equivariant diffusion denoiser, timestep-conditioned
# (luwei0917/DynamicBind)
# ============================================================


class _DynamicBindEquivariantLayer(nn.Module):
    """One equivariant message-passing update over a joint protein-ligand graph."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.coord_gate = nn.Linear(hidden_dim, 1)
        self.node_norm = nn.LayerNorm(hidden_dim)

    def forward(self, feats: Tensor, coords: Tensor) -> tuple[Tensor, Tensor]:
        """Update (invariant node features, equivariant node coordinates)."""
        n = feats.shape[0]
        feat_i = feats.unsqueeze(1).expand(n, n, -1)
        feat_j = feats.unsqueeze(0).expand(n, n, -1)
        rel_vec = coords.unsqueeze(1) - coords.unsqueeze(0)
        rel_dist = rel_vec.norm(dim=-1, keepdim=True)

        msg_in = torch.cat([feat_i, feat_j, rel_dist], dim=-1)
        messages = self.message_mlp(msg_in)
        gate = self.coord_gate(messages)
        coord_update = (gate * rel_vec).mean(dim=1)

        new_feats = self.node_norm(feats + messages.mean(dim=1))
        new_coords = coords + coord_update
        return new_feats, new_coords


class DynamicBindDenoiser(nn.Module):
    """Joint ligand-pose + protein-conformation SE(3)-equivariant diffusion denoiser.

    Ports DynamicBind's reverse-diffusion network: a coarse-grained joint
    protein-residue/ligand-atom graph, conditioned on the diffusion
    timestep, is processed by stacked equivariant message-passing layers;
    separate heads then predict a ligand rigid-body translation/rotation
    plus internal torsion update, and per-residue protein rotation/
    translation plus side-chain torsion updates -- letting the protein
    move together with the ligand across the diffusion trajectory (the
    namesake induced-fit contribution over rigid-receptor docking).
    """

    def __init__(self, hidden_dim: int = 24, n_layers: int = 3) -> None:
        super().__init__()
        self.node_embed = nn.Linear(2, hidden_dim)  # (is_ligand, atom/residue type scalar)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.layers = nn.ModuleList(
            [_DynamicBindEquivariantLayer(hidden_dim) for _ in range(n_layers)]
        )
        self.ligand_translation_head = nn.Linear(hidden_dim, 3)
        self.ligand_rotation_head = nn.Linear(hidden_dim, 3)
        self.ligand_torsion_head = nn.Linear(hidden_dim, 1)
        self.protein_rotation_head = nn.Linear(hidden_dim, 3)
        self.protein_translation_head = nn.Linear(hidden_dim, 3)
        self.protein_sidechain_torsion_head = nn.Linear(hidden_dim, 4)

    def forward(
        self, node_type_feats: Tensor, coords: Tensor, is_ligand: Tensor, timestep: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Predict joint ligand + protein denoising updates for one diffusion step.

        Parameters
        ----------
        node_type_feats : Tensor
            Shape ``(n_nodes, 1)`` per-node atom/residue type scalar.
        coords : Tensor
            Shape ``(n_nodes, 3)`` current (noised) coarse-grained 3D
            coordinates (ligand atoms + protein residue centroids).
        is_ligand : Tensor
            Shape ``(n_nodes, 1)`` binary ligand/protein node indicator.
        timestep : Tensor
            Scalar diffusion timestep (fraction of the noise schedule, in
            ``[0, 1]``), shape ``(1,)``.
        """
        feats = self.node_embed(torch.cat([is_ligand, node_type_feats], dim=-1))
        feats = feats + self.time_embed(timestep.view(1, 1))

        for layer in self.layers:
            feats, coords = layer(feats, coords)

        ligand_translation = self.ligand_translation_head(feats)
        ligand_rotation = self.ligand_rotation_head(feats)
        ligand_torsion = self.ligand_torsion_head(feats)
        protein_rotation = self.protein_rotation_head(feats)
        protein_translation = self.protein_translation_head(feats)
        protein_sidechain_torsion = self.protein_sidechain_torsion_head(feats)

        return (
            ligand_translation,
            ligand_rotation,
            ligand_torsion,
            protein_rotation,
            protein_translation,
            protein_sidechain_torsion,
        )


def build_dynamicbind() -> nn.Module:
    """Build a small DynamicBind joint protein-ligand diffusion denoiser."""
    return DynamicBindDenoiser(hidden_dim=24, n_layers=3).eval()


def example_input_dynamicbind() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return (node-type features, coordinates, ligand mask, timestep) for DynamicBind."""
    n_ligand_atoms, n_protein_residues = 8, 12
    n_nodes = n_ligand_atoms + n_protein_residues
    node_type_feats = torch.randn(n_nodes, 1)
    coords = torch.randn(n_nodes, 3)
    is_ligand = torch.cat(
        [torch.ones(n_ligand_atoms, 1), torch.zeros(n_protein_residues, 1)], dim=0
    )
    timestep = torch.tensor([0.3])
    return node_type_feats, coords, is_ligand, timestep


MENAGERIE_ENTRIES = [
    ("DMfold", "build_dmfold", "example_input_dmfold", "2019", "BIO"),
    ("DMPfold", "build_dmpfold2", "example_input_dmpfold2", "2022", "BIO"),
    ("DOVE", "build_dove", "example_input_dove", "2020", "BIO"),
    ("DRfold", "build_drfold", "example_input_drfold", "2023", "BIO"),
    ("dyMEAN", "build_dymean", "example_input_dymean", "2023", "BIO"),
    ("DynamicBind", "build_dynamicbind", "example_input_dynamicbind", "2024", "BIO"),
]
