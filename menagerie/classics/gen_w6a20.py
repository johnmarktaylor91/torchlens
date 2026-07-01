"""Menagerie batch w6a20: cryo-EM structure-quality / cryo-EM & antibody /
protein-distance / ligand-docking / cryo-EM-map-enhancement architectures
from structural biology.

Sources checked (reference only; no cloning, no pip installs):
  - DAQ (cand_00835): Terashi, Wang, et al. (Kihara lab, Purdue), Nature
    Methods 2022, https://github.com/kiharalab/DAQ (``models/resnet.py``,
    classes ``ResNet_custom``/``Bottleneck``, function ``resnetN``). The
    defining mechanism: a per-residue 3D density sub-cube (cropped around
    each residue's backbone position from the cryo-EM map) is passed
    through a **3D ResNeXt** backbone -- ``Bottleneck`` blocks use
    **grouped ("cardinality") 3x3x3 convolutions** rather than plain dense
    convolutions, giving wide multi-path feature aggregation per residue
    -- and the pooled backbone feature is fed to **three parallel linear
    heads** predicting (1) a 20-way amino-acid-identity distribution, (2)
    a 6-way secondary-structure-type distribution, and (3) a 3-way local
    fit-quality/confidence distribution, all from the *same* shared 3D-CNN
    trunk -- i.e. "shared 3D-ResNeXt trunk + three independent per-residue
    classification heads" is DAQ's namesake residue-quality-assignment
    contribution over single-task local-density classifiers. Reimplemented
    with the same grouped-convolution ``Bottleneck`` stack and three
    parallel linear output heads, at reduced cube size, channel width, and
    block depth.
  - DeepAb (cand_00836): Ruffolo, Sulam, Gray (Johns Hopkins), Patterns
    2022, https://github.com/RosettaCommons/DeepAb (``deepab/models/
    AbResNet/AbResNet.py``, class ``AbResNet``; ``deepab/resnets/
    CrissCrossResNet2D.py``, classes ``CrissCrossAttention``/
    ``RCCAModule``; ``deepab/layers``, ``OuterConcatenation2D``). The
    defining mechanism: a one-hot antibody heavy+light-chain sequence is
    encoded by a **1D ResNet**, then **outer-concatenated** into a dense
    pairwise (residue x residue) 2D feature map, which is refined by a
    **dilated 2D ResNet** (cyclically increasing dilation rate) and
    finally passed through a **recurrent criss-cross attention** module
    (``RCCAModule``: two sequential applications of axis-restricted
    row/column self-attention approximate full 2D self-attention in O(N)
    rather than O(N^2)) before six parallel binned-output heads predict
    inter-residue distance/dihedral/planar-angle distributions -- i.e.
    "1D sequence ResNet -> outer-concat pairwise map -> dilated 2D ResNet
    -> twice-applied criss-cross self-attention -> six geometric output
    heads" is DeepAb's namesake antibody-structure-prediction contribution
    over template-based antibody modeling. Reimplemented with the same
    1D-ResNet encoder, outer-concatenation pairwise-map construction,
    dilated 2D residual trunk, twice-applied criss-cross attention module,
    and six parallel output heads (with symmetrization on the symmetric
    distance/omega channels, matching ``out + out.transpose(2, 3)`` in the
    reference), at drastically reduced sequence length, channel width, and
    block depth. The reference's frozen bidirectional-LSTM PSSM side-input
    is out of scope (a separately pretrained language model, not part of
    AbResNet's own trainable mechanism); omitted, with the 1D-ResNet
    encoding alone driving the pairwise map.
  - DeepAccNet (cand_00837): Hiranuma, Park, et al. (Baker lab, UW),
    Nature Communications 2021, https://github.com/hiranumn/DeepAccNet
    (``deepAccNet/model.py``, class ``DeepAccNet``; ``deepAccNet/
    resnet.py``, class ``ResNet``). The defining mechanism: sparse
    per-residue atomic coordinates are voxelized into a dense **3D
    occupancy grid** (``scatter_nd``-style scatter of one-hot atom types
    into a ``24^3`` cube per residue) and passed through a small **3D CNN
    stem**; the resulting per-residue 3D feature is flattened, concatenated
    with hand-crafted one-body features, and **tiled/broadcast into a 2D
    (residue x residue) pairwise tensor** (``tile`` along two different
    axes) together with two-body pairwise features, which feeds a shared
    **instance-normalized 2D ResNet trunk**; two separate downstream
    ResNet arms then predict (a) a per-residue-pair **distance-error
    histogram ("estogram")** and (b) a pairwise **confidence mask**, both
    symmetrized (``out + out.transpose``), and a final **closed-form lDDT
    estimate is computed analytically from the estogram+mask** (weighted
    sum of near-diagonal estogram bins, not a learned head) -- i.e. "3D
    voxel-CNN per-residue embedding, tiled into a 2D pairwise tensor,
    instance-norm ResNet trunk with dual estogram/mask arms, and an
    analytic lDDT readout from the estogram" is DeepAccNet's namesake
    per-residue-accuracy-estimation contribution over global-only model
    quality scores. Reimplemented with the same voxel-CNN stem, tile-based
    1-body-to-2-body broadcast, shared 2D ResNet trunk, dual estogram/mask
    output arms with symmetrization, and the analytic lDDT calculation
    ported verbatim from ``calculate_LDDT``, at drastically reduced voxel
    resolution, residue count, and channel width.
  - DeepDist (cand_00838): Wu, Guo, et al. (Cheng lab, U. Missouri /
    MULTICOM), BMC Bioinformatics 2021,
    https://github.com/multicom-toolbox/deepdist (``lib/
    Model_construct.py``, functions ``DeepDistRes_with_paras_2D``,
    ``_dilated_residual_block``, ``dilated_bottleneck_rc``, ``MaxoutAct``;
    Keras/TF source -- reimplemented as a torch port). The defining
    mechanism: a 2D pairwise co-evolution feature map is instance-
    normalized and passed through a **maxout-activation stem**
    (``MaxoutAct``: several parallel 1x1 convs, then an elementwise max
    across them) before a deep stack of **dilated residual "RC" bottleneck
    blocks** -- each block factorizes its 3x3 conv into **parallel 3x3,
    7x1, and 1x7 convolutions concatenated together** (an inception-style
    multi-receptive-field mix), applies a **cyclically repeating dilation
    schedule** (``[1, 2, 4, 8, 1]`` repeated), and gates its output with a
    **squeeze-excite channel-attention block** -- and the trunk terminates
    in **two parallel heads**: a multi-class softmax distance-bin
    classifier and a real-valued regression head for the same
    inter-residue distance -- i.e. "maxout stem + cyclically-dilated
    multi-kernel (3x3/7x1/1x7) squeeze-excite residual trunk + dual
    classification/regression distance heads" is DeepDist's namesake
    real-value-and-binned inter-residue-distance contribution over
    binned-only contact predictors. Reimplemented with the same maxout
    stem, cyclic-dilation multi-kernel-concat squeeze-excite residual
    block, and dual classification/regression output heads, at reduced
    channel width, block depth, and 2D map size.
  - DeepDock (cand_00839): Mendez-Lucio, Ahmad, et al. (OptiMaL-PSE Lab),
    Nature Machine Intelligence 2021,
    https://github.com/OptiMaL-PSE-Lab/DeepDock (``deepdock/models.py``,
    classes ``TargetNet``/``LigandNet``/``EdgeModel``/``NodeModel``/
    ``ResBlock``/``DeepDock``). The defining mechanism: separate **graph
    neural network encoders** for the ligand and the protein-target
    surface (each a stack of ``torch_geometric.nn.MetaLayer`` edge/node
    message-passing residual blocks) produce per-atom/per-point node
    embeddings; every ligand-atom / target-point pair's embeddings are
    then **broadcast-concatenated into a dense all-pairs interaction
    tensor**, passed through a small MLP, and finally consumed by a
    **mixture-density-network (MDN) head** that outputs, per atom-point
    pair, the parameters (``pi``, ``sigma``, ``mu``) of a Gaussian mixture
    over the predicted inter-atomic distance -- i.e. binding pose
    likelihood is modeled as a *learned probability distribution over
    distances* rather than a single point estimate -- "dual-GNN encoders
    -> broadcast all-pairs interaction tensor -> Gaussian-mixture
    distance-likelihood head" is DeepDock's namesake statistical-potential
    docking contribution over deterministic pose-scoring networks.
    Reimplemented with the same dual MetaLayer-based GNN encoders (edge
    model + node model + residual projection blocks), broadcast all-pairs
    tensor construction, and MDN (pi/sigma/mu) output head, at reduced
    node count, hidden width, and residual-block depth.
  - DeepEMhancer (cand_00840): Sanchez-Garcia, Gomez-Blanco, et al.
    (Carazo lab, CNB-CSIC), Communications Biology 2021,
    https://github.com/rsanchezgarc/deepEMhancer (Keras/TF source --
    reimplemented as a torch port; ``deepEMhancer/config.py`` confirms a
    fixed ``64^3`` sliding-window input cube trained end-to-end for
    map-to-map regression). The defining mechanism: a cryo-EM density map
    is processed **chunk-by-chunk as ``64^3`` sub-cubes** (sliding window
    with 1/4-cube-size stride, stitched back together with overlap
    averaging at inference) by a **3D U-Net** (encoder/decoder with
    skip connections at each resolution) trained to directly regress a
    **sharpened-and-masked output density**, replacing the classical
    B-factor-sharpening + manual-masking cryo-EM postprocessing pipeline
    with a single learned **volume-to-volume 3D U-Net** -- i.e. "3D U-Net
    trained for direct density-map-to-sharpened-density-map regression on
    fixed-size sub-cubes" is DeepEMhancer's namesake learned-post-
    processing contribution over classical B-factor sharpening.
    Reimplemented as a compact 3D U-Net (encoder/decoder with skip
    connections, matching the reference's map-to-map regression target)
    operating on one sub-cube directly (the sliding-window
    chunking/stitching is inference-time tiling logic outside the
    trainable network itself), at drastically reduced cube size and
    channel width.

All six models are reimplemented from scratch in base-env torch (DeepDist
and DeepEMhancer are torch ports of Keras/TF reference source); no repo
cloning, no pip installs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import MetaLayer

# ============================================================
# DAQ -- shared 3D-ResNeXt trunk (grouped/cardinality convs) +
# three parallel per-residue classification heads
# (kiharalab/DAQ)
# ============================================================


class _DaqBottleneck(nn.Module):
    """Grouped-convolution ("cardinality") 3D residual bottleneck.

    Ports ``models/resnet.py``'s ``Bottleneck``: the 3x3x3 conv uses
    ``groups=cardinality`` (a ResNeXt-style grouped convolution) instead
    of a plain dense convolution.
    """

    expansion = 2

    def __init__(
        self,
        inplanes: int,
        planes: int,
        cardinality: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        mid_planes = cardinality * (planes // 8)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class DaqResNeXt(nn.Module):
    """Shared 3D-ResNeXt trunk with three parallel per-residue heads.

    Ports ``ResNet_custom``: a per-residue voxel sub-cube of the cryo-EM
    density map is processed by a grouped-convolution 3D ResNeXt trunk;
    the pooled trunk feature feeds three independent linear heads
    (amino-acid identity, secondary-structure type, local fit quality).
    """

    def __init__(
        self,
        cardinality: int = 4,
        base_width: int = 16,
        num_classes: tuple[int, int, int] = (20, 6, 3),
    ) -> None:
        super().__init__()
        self.inplanes = base_width
        self.conv1 = nn.Conv3d(1, base_width, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(base_width)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(base_width * 2, 1, cardinality)
        self.layer2 = self._make_layer(base_width * 4, 1, cardinality, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        out_width = base_width * 4 * _DaqBottleneck.expansion
        self.fc1 = nn.Linear(out_width, num_classes[0])
        self.fc2 = nn.Linear(out_width, num_classes[1])
        self.fc3 = nn.Linear(out_width, num_classes[2])

    def _make_layer(
        self, planes: int, blocks: int, cardinality: int, stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        out_planes = planes * _DaqBottleneck.expansion
        if stride != 1 or self.inplanes != out_planes:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_planes),
            )
        layers = [_DaqBottleneck(self.inplanes, planes, cardinality, stride, downsample)]
        self.inplanes = out_planes
        for _ in range(1, blocks):
            layers.append(_DaqBottleneck(self.inplanes, planes, cardinality))
        return nn.Sequential(*layers)

    def forward(self, voxel_cube: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict (amino-acid, secondary-structure, quality) logits.

        Parameters
        ----------
        voxel_cube : Tensor
            Shape ``(batch, 1, D, H, W)`` per-residue cryo-EM density
            sub-cube.
        """
        x = self.maxpool(self.relu(self.bn1(self.conv1(voxel_cube))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x).flatten(1)
        return self.fc1(x), self.fc2(x), self.fc3(x)


def build_daq() -> nn.Module:
    """Build a small DAQ 3D-ResNeXt residue-quality scorer."""
    return DaqResNeXt(cardinality=4, base_width=16, num_classes=(20, 6, 3)).eval()


def example_input_daq() -> Tensor:
    """Return a batch of per-residue density sub-cubes for DAQ."""
    return torch.randn(2, 1, 16, 16, 16)


# ============================================================
# DeepAb -- 1D-ResNet sequence encoder + outer-concat pairwise
# map + dilated 2D ResNet + recurrent criss-cross attention +
# six geometric output heads (RosettaCommons/DeepAb)
# ============================================================


class _ResBlock1D(nn.Module):
    """Two Conv1d+BatchNorm1d+ReLU layers with a residual skip."""

    def __init__(self, channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + x)


class _DilatedResBlock2D(nn.Module):
    """Two Conv2d+BatchNorm2d+ReLU layers with a given dilation, residual skip."""

    def __init__(self, channels: int, dilation: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = dilation * (kernel_size // 2)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + x)


class _CrissCrossAttention(nn.Module):
    """Axis-restricted row/column self-attention (ports ``CrissCrossAttention``).

    Each position attends only to positions sharing its row or column
    (O(N) rather than the O(N^2) of full 2D self-attention).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query = nn.Conv2d(dim, max(dim // 8, 1), kernel_size=1)
        self.key = nn.Conv2d(dim, max(dim // 8, 1), kernel_size=1)
        self.value = nn.Conv2d(dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        """Apply one criss-cross attention update.

        Parameters
        ----------
        x : Tensor
            Shape ``(batch, dim, H, W)`` feature map.
        """
        b, _, h, w = x.shape
        q, k, v = self.query(x), self.key(x), self.value(x)

        # Row attention: each (h, w) attends over all w' in its row.
        q_row = q.permute(0, 2, 3, 1).reshape(b * h, w, -1)
        k_row = k.permute(0, 2, 1, 3).reshape(b * h, -1, w)
        v_row = v.permute(0, 2, 1, 3).reshape(b * h, -1, w)
        attn_row = torch.softmax(torch.bmm(q_row, k_row), dim=-1)
        out_row = (
            torch.bmm(v_row, attn_row.transpose(1, 2)).reshape(b, h, -1, w).permute(0, 2, 1, 3)
        )

        # Column attention: each (h, w) attends over all h' in its column.
        q_col = q.permute(0, 3, 2, 1).reshape(b * w, h, -1)
        k_col = k.permute(0, 3, 1, 2).reshape(b * w, -1, h)
        v_col = v.permute(0, 3, 1, 2).reshape(b * w, -1, h)
        attn_col = torch.softmax(torch.bmm(q_col, k_col), dim=-1)
        out_col = (
            torch.bmm(v_col, attn_col.transpose(1, 2)).reshape(b, w, -1, h).permute(0, 2, 3, 1)
        )

        return self.gamma * (out_row + out_col) + x


class _RCCAModule(nn.Module):
    """Recurrent (twice-applied) criss-cross attention block (ports ``RCCAModule``)."""

    def __init__(self, in_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        inter = max(in_channels // 4, 4)
        pad = kernel_size // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, inter, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(inter),
            nn.ReLU(inplace=True),
        )
        self.cca = _CrissCrossAttention(inter)
        self.conv2 = nn.Sequential(
            nn.Conv2d(inter, in_channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        for _ in range(2):
            out = self.cca(out)
        return self.conv2(out)


class AbResNet(nn.Module):
    """1D-ResNet + outer-concat pairwise map + dilated 2D ResNet + RCCA head.

    Ports ``deepab/models/AbResNet/AbResNet.py``'s ``AbResNet``: a one-hot
    antibody sequence is encoded by a 1D ResNet, outer-concatenated into a
    2D pairwise map, refined by a cyclically-dilated 2D ResNet trunk, then
    a recurrent criss-cross attention module, before six parallel binned
    output heads predict inter-residue geometry (three symmetrized).
    """

    def __init__(
        self,
        in_planes: int = 21,
        hidden1d: int = 32,
        hidden2d: int = 32,
        num_blocks1d: int = 2,
        num_blocks2d: int = 4,
        num_out_bins: int = 8,
    ) -> None:
        super().__init__()
        self.stem1d = nn.Sequential(
            nn.Conv1d(in_planes, hidden1d, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden1d),
            nn.ReLU(inplace=True),
        )
        self.resnet1d = nn.Sequential(*[_ResBlock1D(hidden1d) for _ in range(num_blocks1d)])

        pairwise_channels = 2 * hidden1d
        self.pair_proj = nn.Sequential(
            nn.Conv2d(pairwise_channels, hidden2d, kernel_size=1),
            nn.BatchNorm2d(hidden2d),
            nn.ReLU(inplace=True),
        )
        dilations = [1, 2, 4, 1]
        self.resnet2d = nn.ModuleList(
            [
                _DilatedResBlock2D(hidden2d, dilations[i % len(dilations)])
                for i in range(num_blocks2d)
            ]
        )
        self.rcca = _RCCAModule(hidden2d)

        self.output_names = ["ca_dist", "cb_dist", "no_dist", "omega", "theta", "phi"]
        self.heads = nn.ModuleDict(
            {
                name: nn.Conv2d(hidden2d, num_out_bins, kernel_size=3, padding=1)
                for name in self.output_names
            }
        )

    def forward(self, one_hot_seq: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Predict six binned inter-residue geometric distributions.

        Parameters
        ----------
        one_hot_seq : Tensor
            Shape ``(batch, in_planes, L)`` one-hot-encoded antibody
            heavy+light sequence.
        """
        h = self.resnet1d(self.stem1d(one_hot_seq))

        n = h.shape[-1]
        left = h.unsqueeze(3).expand(-1, -1, -1, n)
        right = h.unsqueeze(2).expand(-1, -1, n, -1)
        pair = torch.cat([left, right], dim=1)

        out = self.pair_proj(pair)
        for block in self.resnet2d:
            out = block(out)
        out = self.rcca(out)

        outputs = {name: head(out) for name, head in self.heads.items()}
        for sym_name in ("ca_dist", "cb_dist", "omega"):
            outputs[sym_name] = outputs[sym_name] + outputs[sym_name].transpose(2, 3)

        return tuple(outputs[name] for name in self.output_names)


def build_deepab() -> nn.Module:
    """Build a small DeepAb antibody structure-prediction model."""
    return AbResNet(
        in_planes=21, hidden1d=32, hidden2d=32, num_blocks1d=2, num_blocks2d=4, num_out_bins=8
    ).eval()


def example_input_deepab() -> Tensor:
    """Return a batch of one-hot antibody sequences for DeepAb."""
    return torch.rand(1, 21, 20)


# ============================================================
# DeepAccNet -- 3D voxel-CNN per-residue embedding tiled into a
# 2D pairwise tensor, instance-norm ResNet trunk with dual
# estogram/mask arms, analytic lDDT readout (hiranumn/DeepAccNet)
# ============================================================


class _DeepAccResBlock(nn.Module):
    """Instance-normalized 2D residual block (ports ``resnet.py``'s ``ResBlock``)."""

    def __init__(self, channels: int, use_inorm: bool = True) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(channels, affine=True) if use_inorm else nn.Identity()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels, affine=True) if use_inorm else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        out = F.elu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return F.elu(out + x)


class DeepAccNet(nn.Module):
    """Voxel-CNN + tiled 2D pairwise ResNet trunk with estogram/mask arms.

    Ports ``deepAccNet/model.py``'s ``DeepAccNet``: a dense per-residue 3D
    occupancy sub-grid is embedded by a small 3D CNN stem, flattened and
    tiled/broadcast into a 2D residue-pair tensor, refined by a shared
    instance-normalized 2D ResNet trunk, then split into an "estogram"
    (distance-error histogram) arm and a confidence-mask arm, both
    symmetrized; a final lDDT score is computed analytically from the two.
    """

    def __init__(
        self, num_restype: int = 8, num_channel: int = 24, num_chunks: int = 2, num_bins: int = 15
    ) -> None:
        super().__init__()
        self.retype = nn.Conv3d(num_restype, num_restype, kernel_size=1, bias=False)
        self.conv3d_1 = nn.Conv3d(num_restype, num_restype, kernel_size=3, padding=0)
        self.conv3d_2 = nn.Conv3d(num_restype, 12, kernel_size=3, padding=0)
        self.pool3d = nn.AvgPool3d(kernel_size=2, stride=2)

        self.node_proj = nn.Conv1d(12, num_channel // 2, kernel_size=1)
        self.pair_proj = nn.Conv2d(num_channel, num_channel, kernel_size=1)
        self.inorm = nn.InstanceNorm2d(num_channel, affine=True)

        self.base_resnet = nn.Sequential(
            *[_DeepAccResBlock(num_channel, use_inorm=True) for _ in range(num_chunks)]
        )
        self.error_resnet = _DeepAccResBlock(num_channel, use_inorm=False)
        self.conv2d_error = nn.Conv2d(num_channel, num_bins, kernel_size=1)
        self.mask_resnet = _DeepAccResBlock(num_channel, use_inorm=False)
        self.conv2d_mask = nn.Conv2d(num_channel, 1, kernel_size=1)
        self.num_bins = num_bins

    def forward(self, voxel_grid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict (estogram, confidence mask, analytic lDDT) for a residue set.

        Parameters
        ----------
        voxel_grid : Tensor
            Shape ``(n_res, num_restype, D, D, D)`` per-residue one-hot
            atomic-occupancy sub-grid.
        """
        n_res = voxel_grid.shape[0]
        x = F.elu(self.retype(voxel_grid))
        x = F.elu(self.conv3d_1(x))
        x = F.elu(self.conv3d_2(x))
        x = self.pool3d(x)

        node = x.flatten(2).mean(dim=2, keepdim=True)  # (n_res, 12, 1)
        node = F.elu(self.node_proj(node)).squeeze(-1)  # (n_res, C/2)

        left = node.unsqueeze(1).expand(n_res, n_res, -1)
        right = node.unsqueeze(0).expand(n_res, n_res, -1)
        pair = (
            torch.cat([left, right], dim=-1).permute(2, 0, 1).unsqueeze(0)
        )  # (1, C, n_res, n_res)

        out = F.elu(self.inorm(self.pair_proj(pair)))
        out = F.elu(self.base_resnet(out))

        error = F.elu(self.error_resnet(out))
        estogram_logits = self.conv2d_error(error)
        estogram_logits = (estogram_logits + estogram_logits.permute(0, 1, 3, 2)) / 2
        estogram = F.softmax(estogram_logits, dim=1)[0]

        mask_feat = F.elu(self.mask_resnet(out))
        mask_logits = self.conv2d_mask(mask_feat)[:, 0, :, :]
        mask_logits = (mask_logits + mask_logits.permute(0, 2, 1)) / 2
        mask = torch.sigmoid(mask_logits)[0]

        lddt = self._calculate_lddt(estogram, mask)
        return estogram, mask, lddt

    def _calculate_lddt(self, estogram: Tensor, mask: Tensor, center: int | None = None) -> Tensor:
        """Analytic lDDT readout from the estogram + mask (ports ``calculate_LDDT``)."""
        center = self.num_bins // 2 if center is None else center
        device = estogram.device
        n = mask.shape[-1]
        eye = torch.eye(n, device=device)
        masked_mask = mask * (torch.ones((n, n), device=device) - eye)
        masked = estogram * masked_mask

        span = min(center, self.num_bins - 1 - center)
        p0 = masked[center].sum(dim=0)
        p1 = (
            (masked[center - 1] + masked[center + 1]).sum(dim=0)
            if span >= 1
            else torch.zeros_like(p0)
        )
        p2 = (
            (masked[center - 2] + masked[center + 2]).sum(dim=0)
            if span >= 2
            else torch.zeros_like(p0)
        )
        p3 = (
            (masked[center - 3] + masked[center + 3]).sum(dim=0)
            if span >= 3
            else torch.zeros_like(p0)
        )
        p4 = masked_mask.sum(dim=0)
        return 0.25 * (4.0 * p0 + 3.0 * p1 + 2.0 * p2 + p3) / (p4 + 1e-8)


def build_deepaccnet() -> nn.Module:
    """Build a small DeepAccNet per-residue accuracy estimator."""
    return DeepAccNet(num_restype=8, num_channel=24, num_chunks=2, num_bins=15).eval()


def example_input_deepaccnet() -> Tensor:
    """Return a per-residue voxel occupancy grid for DeepAccNet."""
    return torch.rand(6, 8, 10, 10, 10)


# ============================================================
# DeepDist -- maxout stem + cyclically-dilated multi-kernel
# squeeze-excite residual trunk + dual classification/
# regression distance heads (multicom-toolbox/deepdist)
# ============================================================


class _MaxoutAct(nn.Module):
    """Parallel 1x1 convs reduced by an elementwise max (ports ``MaxoutAct``)."""

    def __init__(self, in_channels: int, out_channels: int, num_pieces: int = 4) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, kernel_size=1) for _ in range(num_pieces)]
        )

    def forward(self, x: Tensor) -> Tensor:
        pieces = torch.stack([F.elu(conv(x)) for conv in self.convs], dim=0)
        return pieces.max(dim=0).values


class _SqueezeExcite2D(nn.Module):
    """Channel squeeze-excite gate."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: Tensor) -> Tensor:
        pooled = x.mean(dim=(2, 3))
        gate = torch.sigmoid(self.fc2(F.relu(self.fc1(pooled))))
        return x * gate.unsqueeze(-1).unsqueeze(-1)


class _DilatedBottleneckRC(nn.Module):
    """Cyclically-dilated multi-kernel (3x3/7x1/1x7 concat) SE residual block.

    Ports ``dilated_bottleneck_rc``: a 1x1 projection is followed by a 3x3
    conv, whose output feeds parallel 7x1 and 1x7 convs; all three are
    concatenated, projected back to the residual width, squeeze-excite
    gated, and added to the block input.
    """

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        third = channels // 3
        self.norm_in = nn.InstanceNorm2d(channels, affine=True)
        self.proj_in = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm_mid = nn.InstanceNorm2d(channels, affine=True)
        self.conv_3x3 = nn.Conv2d(
            channels, third, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.conv_7x1 = nn.Conv2d(third, third, kernel_size=(7, 1), padding=(3, 0))
        self.conv_1x7 = nn.Conv2d(third, third, kernel_size=(1, 7), padding=(0, 3))
        self.norm_cat = nn.InstanceNorm2d(third * 3, affine=True)
        self.proj_out = nn.Conv2d(third * 3, channels, kernel_size=1)
        self.se = _SqueezeExcite2D(channels)

    def forward(self, x: Tensor) -> Tensor:
        h = F.relu(self.proj_in(F.relu(self.norm_in(x))))
        c33 = self.conv_3x3(F.relu(self.norm_mid(h)))
        c71 = self.conv_7x1(c33)
        c17 = self.conv_1x7(c33)
        cat = torch.cat([c33, c71, c17], dim=1)
        residual = self.proj_out(F.relu(self.norm_cat(cat)))
        residual = self.se(residual)
        return x + residual


class DeepDistNet(nn.Module):
    """Maxout-stem, cyclically-dilated SE residual trunk, dual distance heads.

    Ports ``DeepDistRes_with_paras_2D``: an instance-normalized 2D
    co-evolution feature map is projected by a maxout stem, refined by a
    stack of cyclically-dilated multi-kernel squeeze-excite residual
    blocks, and read out by two parallel heads (a multi-class distance-bin
    classifier and a real-valued distance regressor).
    """

    def __init__(
        self, feature_2d: int = 20, channels: int = 32, num_blocks: int = 6, num_bins: int = 25
    ) -> None:
        super().__init__()
        self.in_norm = nn.InstanceNorm2d(feature_2d, affine=True)
        self.stem_conv = nn.Conv2d(feature_2d, channels, kernel_size=1)
        self.maxout = _MaxoutAct(channels, channels, num_pieces=4)
        dilation_cycle = [1, 2, 4, 8, 1]
        self.blocks = nn.ModuleList(
            [
                _DilatedBottleneckRC(channels, dilation_cycle[i % len(dilation_cycle)])
                for i in range(num_blocks)
            ]
        )
        self.final_norm = nn.InstanceNorm2d(channels, affine=True)
        self.class_head = nn.Conv2d(channels, num_bins, kernel_size=3, padding=1)
        self.regress_head = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, pairwise_features: Tensor) -> tuple[Tensor, Tensor]:
        """Predict (distance-bin logits, real-valued distance) from a pairwise map.

        Parameters
        ----------
        pairwise_features : Tensor
            Shape ``(batch, feature_2d, L, L)`` pairwise co-evolution
            feature map.
        """
        x = self.stem_conv(self.in_norm(pairwise_features))
        x = self.maxout(x)
        for block in self.blocks:
            x = block(x)
        x = F.relu(self.final_norm(x))
        class_logits = self.class_head(x)
        real_dist = F.relu(self.regress_head(x))
        return class_logits, real_dist


def build_deepdist() -> nn.Module:
    """Build a small DeepDist dual-head inter-residue distance predictor."""
    return DeepDistNet(feature_2d=20, channels=32, num_blocks=6, num_bins=25).eval()


def example_input_deepdist() -> Tensor:
    """Return a pairwise co-evolution feature map for DeepDist."""
    return torch.randn(1, 20, 24, 24)


# ============================================================
# DeepDock -- dual MetaLayer-GNN encoders + broadcast all-pairs
# interaction tensor + mixture-density-network distance head
# (OptiMaL-PSE-Lab/DeepDock)
# ============================================================


class _DeepDockEdgeModel(nn.Module):
    """Edge-update MLP (ports ``EdgeModel``)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels * 3, channels), nn.BatchNorm1d(channels), nn.ELU()
        )

    def forward(
        self, src: Tensor, dest: Tensor, edge_attr: Tensor, u: Tensor | None, batch: Tensor
    ) -> Tensor:
        return self.mlp(torch.cat([src, dest, edge_attr], dim=1))


class _DeepDockNodeModel(nn.Module):
    """Node-update MLP with mean-scatter aggregation (ports ``NodeModel``)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(channels * 2, channels), nn.BatchNorm1d(channels), nn.ELU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(channels * 2, channels), nn.BatchNorm1d(channels), nn.ELU()
        )

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, u: Tensor | None, batch: Tensor
    ) -> Tensor:
        row, col = edge_index
        out = self.mlp1(torch.cat([x[row], edge_attr], dim=1))
        agg = torch.zeros(x.size(0), out.size(1), device=x.device, dtype=out.dtype)
        agg = agg.index_add(0, col, out) / (
            torch.bincount(col, minlength=x.size(0)).clamp(min=1).unsqueeze(-1).to(out.dtype)
        )
        return self.mlp2(torch.cat([x, agg], dim=1))


class _DeepDockResBlock(nn.Module):
    """Bottleneck-projected MetaLayer residual block (ports ``ResBlock``)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        mid = channels // 4
        self.down_node = nn.Linear(channels, mid)
        self.down_edge = nn.Linear(channels, mid)
        self.bn1_node = nn.BatchNorm1d(mid)
        self.bn1_edge = nn.BatchNorm1d(mid)
        self.conv = MetaLayer(
            edge_model=_DeepDockEdgeModel(mid), node_model=_DeepDockNodeModel(mid)
        )
        self.up_node = nn.Linear(mid, channels)
        self.up_edge = nn.Linear(mid, channels)
        self.bn2_node = nn.BatchNorm1d(channels)
        self.bn2_edge = nn.BatchNorm1d(channels)

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, batch: Tensor
    ) -> tuple[Tensor, Tensor]:
        h_node = F.elu(self.bn1_node(self.down_node(x)))
        h_edge = F.elu(self.bn1_edge(self.down_edge(edge_attr)))
        h_node, h_edge, _ = self.conv(h_node, edge_index, h_edge, None, batch)
        h_node = self.bn2_node(self.up_node(h_node))
        h_edge = self.bn2_edge(self.up_edge(h_edge))
        return F.elu(h_node + x), F.elu(h_edge + edge_attr)


class _DeepDockGNN(nn.Module):
    """Node/edge encoder + stacked MetaLayer + residual blocks (ports ``TargetNet``/``LigandNet``)."""

    def __init__(
        self, in_channels: int, edge_features: int, hidden_dim: int = 32, residual_layers: int = 2
    ) -> None:
        super().__init__()
        self.node_encoder = nn.Linear(in_channels, hidden_dim)
        self.edge_encoder = nn.Linear(edge_features, hidden_dim)
        self.conv1 = MetaLayer(
            edge_model=_DeepDockEdgeModel(hidden_dim), node_model=_DeepDockNodeModel(hidden_dim)
        )
        self.res_blocks = nn.ModuleList(
            [_DeepDockResBlock(hidden_dim) for _ in range(residual_layers)]
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, batch: Tensor) -> Tensor:
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        x, edge_attr, _ = self.conv1(x, edge_index, edge_attr, None, batch)
        for block in self.res_blocks:
            x, edge_attr = block(x, edge_index, edge_attr, batch)
        return x


class DeepDock(nn.Module):
    """Dual-GNN docking model with a mixture-density-network distance head.

    Ports ``deepdock/models.py``'s ``DeepDock``: ligand and target node
    embeddings (each from an independent MetaLayer-based GNN) are
    broadcast into an all-pairs ligand-atom x target-point interaction
    tensor, passed through an MLP, and read out by a Gaussian-mixture
    (pi, sigma, mu) head over the predicted inter-atomic distance.
    """

    def __init__(
        self,
        ligand_in: int = 12,
        target_in: int = 10,
        edge_dim: int = 4,
        hidden_dim: int = 32,
        n_gaussians: int = 6,
    ) -> None:
        super().__init__()
        self.ligand_model = _DeepDockGNN(ligand_in, edge_dim, hidden_dim, residual_layers=2)
        self.target_model = _DeepDockGNN(target_in, edge_dim, hidden_dim, residual_layers=2)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ELU()
        )
        self.z_pi = nn.Linear(hidden_dim, n_gaussians)
        self.z_sigma = nn.Linear(hidden_dim, n_gaussians)
        self.z_mu = nn.Linear(hidden_dim, n_gaussians)

    def forward(
        self,
        ligand_x: Tensor,
        ligand_edge_index: Tensor,
        ligand_edge_attr: Tensor,
        ligand_pos: Tensor,
        target_x: Tensor,
        target_edge_index: Tensor,
        target_edge_attr: Tensor,
        target_pos: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Predict Gaussian-mixture distance parameters for one ligand-target pair.

        Parameters
        ----------
        ligand_x, target_x : Tensor
            Node feature matrices, shape ``(N_l, ligand_in)`` / ``(N_t,
            target_in)``.
        ligand_edge_index, target_edge_index : Tensor
            Shape ``(2, E)`` edge index tensors (single-graph, unbatched).
        ligand_edge_attr, target_edge_attr : Tensor
            Shape ``(E, edge_dim)`` edge feature tensors.
        ligand_pos, target_pos : Tensor
            Shape ``(N_l, 3)`` / ``(N_t, 3)`` 3D coordinates.
        """
        batch_l = torch.zeros(ligand_x.size(0), dtype=torch.long, device=ligand_x.device)
        batch_t = torch.zeros(target_x.size(0), dtype=torch.long, device=target_x.device)
        h_l = self.ligand_model(ligand_x, ligand_edge_index, ligand_edge_attr, batch_l)
        h_t = self.target_model(target_x, target_edge_index, target_edge_attr, batch_t)

        n_l, n_t = h_l.size(0), h_t.size(0)
        left = h_l.unsqueeze(1).expand(n_l, n_t, -1)
        right = h_t.unsqueeze(0).expand(n_l, n_t, -1)
        interaction = torch.cat([left, right], dim=-1).reshape(n_l * n_t, -1)

        c = self.mlp(interaction)
        pi = F.softmax(self.z_pi(c), dim=-1)
        sigma = F.elu(self.z_sigma(c)) + 1.1
        mu = F.elu(self.z_mu(c)) + 1

        dist = torch.cdist(ligand_pos, target_pos).reshape(-1, 1)
        return pi, sigma, mu, dist


def build_deepdock() -> nn.Module:
    """Build a small DeepDock dual-GNN mixture-density docking model."""
    return DeepDock(ligand_in=12, target_in=10, edge_dim=4, hidden_dim=32, n_gaussians=6).eval()


def example_input_deepdock() -> tuple[
    Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor
]:
    """Return a small (ligand graph, target graph) pair for DeepDock."""
    n_l, n_t = 8, 10

    def _rand_edges(n: int) -> Tensor:
        src = torch.randint(0, n, (n * 2,))
        dst = torch.randint(0, n, (n * 2,))
        return torch.stack([src, dst], dim=0)

    ligand_x = torch.randn(n_l, 12)
    ligand_edge_index = _rand_edges(n_l)
    ligand_edge_attr = torch.randn(ligand_edge_index.size(1), 4)
    ligand_pos = torch.randn(n_l, 3)

    target_x = torch.randn(n_t, 10)
    target_edge_index = _rand_edges(n_t)
    target_edge_attr = torch.randn(target_edge_index.size(1), 4)
    target_pos = torch.randn(n_t, 3)

    return (
        ligand_x,
        ligand_edge_index,
        ligand_edge_attr,
        ligand_pos,
        target_x,
        target_edge_index,
        target_edge_attr,
        target_pos,
    )


# ============================================================
# DeepEMhancer -- 3D U-Net trained for direct sharpened-density
# map-to-map regression on fixed-size sub-cubes
# (rsanchezgarc/deepEMhancer)
# ============================================================


class _UNet3DBlock(nn.Module):
    """Two Conv3d+BatchNorm3d+ReLU layers."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class DeepEMhancerUNet(nn.Module):
    """Compact 3D U-Net for cryo-EM density-map sharpening/masking regression.

    Ports the map-to-map regression architecture described in
    ``deepEMhancer``'s ``config.py`` (fixed sub-cube inference, 3D
    encoder-decoder with skip connections): an encoder/decoder 3D U-Net
    directly regresses a sharpened-and-masked output density from a raw
    input density sub-cube, replacing classical B-factor sharpening.
    """

    def __init__(self, in_channels: int = 1, base_width: int = 8) -> None:
        super().__init__()
        self.enc1 = _UNet3DBlock(in_channels, base_width)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = _UNet3DBlock(base_width, base_width * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.bottleneck = _UNet3DBlock(base_width * 2, base_width * 4)

        self.up2 = nn.Upsample(scale_factor=2, mode="trilinear")
        self.dec2 = _UNet3DBlock(base_width * 4 + base_width * 2, base_width * 2)
        self.up1 = nn.Upsample(scale_factor=2, mode="trilinear")
        self.dec1 = _UNet3DBlock(base_width * 2 + base_width, base_width)

        self.out_conv = nn.Conv3d(base_width, 1, kernel_size=1)

    def forward(self, density_cube: Tensor) -> Tensor:
        """Regress a sharpened+masked density cube from a raw density cube.

        Parameters
        ----------
        density_cube : Tensor
            Shape ``(batch, 1, D, H, W)`` raw cryo-EM density sub-cube.
        """
        e1 = self.enc1(density_cube)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return F.relu(self.out_conv(d1))


def build_deepemhancer() -> nn.Module:
    """Build a small DeepEMhancer 3D U-Net map-enhancement model."""
    return DeepEMhancerUNet(in_channels=1, base_width=8).eval()


def example_input_deepemhancer() -> Tensor:
    """Return a raw cryo-EM density sub-cube for DeepEMhancer."""
    return torch.rand(1, 1, 16, 16, 16)


MENAGERIE_ENTRIES = [
    ("DAQ", "build_daq", "example_input_daq", "2022", "BIO"),
    ("DeepAb", "build_deepab", "example_input_deepab", "2022", "BIO"),
    ("DeepAccNet", "build_deepaccnet", "example_input_deepaccnet", "2021", "BIO"),
    ("DeepDist", "build_deepdist", "example_input_deepdist", "2021", "BIO"),
    ("DeepDock", "build_deepdock", "example_input_deepdock", "2021", "BIO"),
    ("DeepEMhancer", "build_deepemhancer", "example_input_deepemhancer", "2021", "BIO"),
]
