# FAITHFUL PORT of hiranumn/DeepAccNet @ master (original framework: TensorFlow 1.x)
# https://github.com/hiranumn/DeepAccNet/blob/master/pyErrorPred/model.py
# https://github.com/hiranumn/DeepAccNet/blob/master/pyErrorPred/resnet.py
# https://github.com/hiranumn/DeepAccNet/blob/master/pyErrorPred/layers.py
#
# NOTE: the repo's own README advertises "Python-PyTorch implementation", but
# the actual shipped model code (pyErrorPred/model.py, resnet.py, layers.py)
# is TensorFlow-1.x graph-mode (tf.placeholder / tf.Session / tf.layers.* /
# tf.contrib.layers.instance_norm). It cannot run in a TF1-less base env, so
# this is a faithful architectural transcription into torch, not a vendor.
#
# DeepAccNet predicts per-residue/per-residue-pair model accuracy (lDDT)
# from a protein structure. Architecture (Model.build in model.py):
#   1) 3D conv "retyper" branch consuming a scattered/one-hot 3D atom-grid
#      per residue: 1x1x1 conv -> conv3d(k=3,valid)+dropout+elu ->
#      conv3d(k=4,valid)+elu -> conv3d(k=4,valid)+elu -> avgpool3d(4,4)
#   2) Flatten + concat with 1D per-residue features -> project to
#      channel//2 via 1D conv + elu
#   3) Broadcast projected 1D features to a pairwise (res x res) tensor,
#      concat with the raw 2D pairwise ("tbt") features, project to
#      `channel` via 1x1 2D conv, instance norm, elu
#   4) A dilated-ResNet2D "trunk" (AlphaFold-style): `num_chunks` groups of
#      4 full-pre-activation residual blocks cycling dilation (1,2,4,8),
#      each block: elu->1x1 conv down -> elu-> kxk dilated conv -> elu ->
#      1x1 conv up, added back to its input.
#   5) Two separate 1-chunk resnet "heads" off the trunk output (error head
#      and mask head), each followed by a 1x1 conv to the output channel
#      count, symmetrized ((X + X^T)/2), and squashed (softmax / sigmoid).
#
# Names/kwargs (channel, num_chunks, dilation_cycle=[1,2,4,8], the
# full-preactivation block structure, the two-head split, the symmetrization
# of both output heads) are preserved from the real TF graph. TF's
# channels-last conv layout is mapped to torch's channels-first layout;
# `tf.contrib.layers.instance_norm` (no learnable affine by default) maps to
# `nn.InstanceNorm{2d,3d}(affine=False)`. Training-only machinery (the TF
# Session/placeholder/optimizer plumbing, tf.Saver checkpoints, the label
# cross-entropy/MSE loss) is intentionally dropped since this staging module
# is a forward-pass-only capture target; the inference math (retyper -> 3D
# conv stack -> 2D dilated resnet trunk -> symmetrized error/mask heads ->
# LDDT integral) is preserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock2D(nn.Module):
    """Full pre-activation residual block (resnet_block in layers TF code):
    [in]-> (in)norm? -> elu -> 1x1 conv (down) -> (in)norm? -> elu ->
    kxk dilated conv (down) -> (in)norm? -> elu -> 1x1 conv (up)."""

    def __init__(self, channel, dilation=1, kernel_size=3, require_in=False):
        super().__init__()
        assert channel % 2 == 0, "Even number channels are required."
        down_channel = channel // 2
        self.require_in = require_in

        self.in1 = nn.InstanceNorm2d(channel, affine=False) if require_in else nn.Identity()
        self.conv1 = nn.Conv2d(channel, down_channel, kernel_size=1)

        self.in2 = nn.InstanceNorm2d(down_channel, affine=False) if require_in else nn.Identity()
        padding = dilation * (kernel_size - 1) // 2
        self.conv2 = nn.Conv2d(
            down_channel, down_channel, kernel_size=kernel_size, dilation=dilation, padding=padding
        )

        self.in3 = nn.InstanceNorm2d(down_channel, affine=False) if require_in else nn.Identity()
        self.conv3 = nn.Conv2d(down_channel, channel, kernel_size=1)

    def forward(self, x):
        h = self.in1(x)
        h = F.elu(h)
        h = self.conv1(h)

        h = self.in2(h)
        h = F.elu(h)
        h = self.conv2(h)

        h = self.in3(h)
        h = F.elu(h)
        h = self.conv3(h)
        return h


class ResNetTrunk2D(nn.Module):
    """build_resnet in resnet.py: first 1x1 projection, then `num_chunks`
    groups of 4 residual blocks with cycling dilation (1,2,4,8), optionally
    followed by 2 extra non-dilated residual blocks."""

    def __init__(
        self,
        in_channel,
        channel,
        num_chunks,
        require_in=False,
        no_last_dilation=False,
        dilation_cycle=(1, 2, 4, 8),
    ):
        super().__init__()
        self.first_projection = nn.Conv2d(in_channel, channel, kernel_size=1)

        blocks = []
        for _ in range(num_chunks):
            for dr in dilation_cycle:
                blocks.append(ResNetBlock2D(channel, dilation=dr, require_in=require_in))
        self.blocks = nn.ModuleList(blocks)

        self.no_last_dilation = no_last_dilation
        if no_last_dilation:
            self.extra_blocks = nn.ModuleList(
                [ResNetBlock2D(channel, dilation=1, require_in=require_in) for _ in range(2)]
            )
        else:
            self.extra_blocks = nn.ModuleList([])

    def forward(self, x):
        x = self.first_projection(x)
        for block in self.blocks:
            x = x + block(x)
        for block in self.extra_blocks:
            x = x + block(x)
        return x


class DeepAccNet(nn.Module):
    """Faithful port of pyErrorPred.model.Model.build (inference path only).

    Real constructor defaults (kept identical): num_chunks=8, channel=256,
    no_last_dilation=True, partial_instance_norm=True, self_attention=False,
    nretype=20. The staging build_deepaccnet() below shrinks `channel` and
    `num_chunks` for a fast trace while keeping every architectural stage.
    """

    def __init__(
        self,
        obt_size,
        tbt_size,
        nretype=20,
        num_chunks=8,
        channel=256,
        no_last_dilation=True,
        partial_instance_norm=True,
    ):
        super().__init__()
        self.obt_size = obt_size
        self.tbt_size = tbt_size
        self.nretype = nretype
        self.channel = channel

        # --- 3D conv "retyper" branch ---
        self.retype = nn.Conv3d(nretype, 20, kernel_size=1, bias=False)
        self.conv3d_1 = nn.Conv3d(20, 20, kernel_size=3, padding=0, bias=True)
        self.dropout3d = nn.Dropout(p=0.15)
        self.conv3d_2 = nn.Conv3d(20, 30, kernel_size=4, padding=0, bias=True)
        self.conv3d_3 = nn.Conv3d(30, 10, kernel_size=4, padding=0, bias=True)
        self.avgpool3d = nn.AvgPool3d(kernel_size=4, stride=4)

        # --- 1D projection branch ---
        # After the 3D stack + avgpool the grid is downsampled from a 24^3
        # atom cube through two valid convs (3, then 4, then 4) and a 4x
        # avgpool; concatenated with the raw 1D obt features.
        grid_out_side = (((24 - 2) - 3) - 3) // 4  # -> 3 for the default 24-cube
        flat_3d_dim = 10 * grid_out_side * grid_out_side * grid_out_side
        self.proj1d = nn.Conv1d(flat_3d_dim + obt_size, channel // 2, kernel_size=1)

        # --- pairwise ("tbt") fusion ---
        self.fuse2d = nn.Conv2d(tbt_size + channel, channel, kernel_size=1)
        self.fuse_in = nn.InstanceNorm2d(channel, affine=False)

        # --- dilated resnet trunk ---
        self.trunk = ResNetTrunk2D(
            in_channel=channel,
            channel=channel,
            num_chunks=num_chunks,
            require_in=partial_instance_norm,
            no_last_dilation=False,
        )

        # --- error (estogram) head ---
        self.error_head = ResNetTrunk2D(
            in_channel=channel,
            channel=channel,
            num_chunks=1,
            require_in=False,
            no_last_dilation=no_last_dilation,
        )
        self.error_out = nn.Conv2d(channel, 15, kernel_size=1)

        # --- mask head ---
        self.mask_head = ResNetTrunk2D(
            in_channel=channel,
            channel=channel,
            num_chunks=1,
            require_in=False,
            no_last_dilation=no_last_dilation,
        )
        self.mask_out = nn.Conv2d(channel, 1, kernel_size=1)

    def forward(self, grid3d, obt, tbt):
        """
        grid3d: [nres, nretype, 24, 24, 24] one-hot-ish atom grid per residue
        obt:    [nres, obt_size] per-residue 1D features
        tbt:    [nres, nres, tbt_size] pairwise 2D features
        """
        nres = grid3d.shape[0]

        # 3D conv branch (treat residues as the conv3d batch dimension)
        h = self.retype(grid3d)
        h = self.conv3d_1(h)
        h = self.dropout3d(h)
        h = F.elu(h)
        h = self.conv3d_2(h)
        h = F.elu(h)
        h = self.conv3d_3(h)
        h = F.elu(h)
        h = self.avgpool3d(h)
        h = h.reshape(nres, -1)

        # concat with 1D features, project
        h1d = torch.cat([h, obt], dim=1)  # [nres, flat_3d_dim + obt_size]
        h1d = h1d.t().unsqueeze(0)  # [1, C, nres]
        h1d = self.proj1d(h1d)
        h1d = F.elu(h1d)
        h1d = h1d.squeeze(0).t()  # [nres, channel//2]

        # broadcast to pairwise tensor, fuse with tbt
        left = h1d.unsqueeze(1).expand(nres, nres, -1)
        right = h1d.unsqueeze(0).expand(nres, nres, -1)
        pair = torch.cat([left, right, tbt], dim=-1)  # [nres, nres, channel + tbt_size]
        pair = pair.permute(2, 0, 1).unsqueeze(0)  # [1, C, nres, nres]
        pair = self.fuse2d(pair)
        pair = self.fuse_in(pair)
        pair = F.elu(pair)

        # dilated resnet trunk
        trunk_out = self.trunk(pair)
        trunk_out = F.elu(trunk_out)

        # error head
        err = self.error_head(trunk_out)
        err = F.elu(err)
        logits_error = self.error_out(err)
        logits_error = (logits_error + logits_error.permute(0, 1, 3, 2)) / 2
        estogram_predicted = torch.softmax(logits_error, dim=1)[0]  # [15, nres, nres]

        # mask head
        msk = self.mask_head(trunk_out)
        msk = F.elu(msk)
        logits_mask = self.mask_out(msk)[:, 0, :, :]
        logits_mask = (logits_mask + logits_mask.permute(0, 2, 1)) / 2
        mask_predicted = torch.sigmoid(logits_mask)[0]  # [nres, nres]

        lddt_predicted = self.calculate_lddt(estogram_predicted, mask_predicted)

        return estogram_predicted, mask_predicted, lddt_predicted

    @staticmethod
    def calculate_lddt(estogram, mask, center=7):
        """Model.calculate_LDDT: integrates the estogram distribution mass
        within the four "lDDT bins" around the diagonal center bin."""
        n = mask.shape[0]
        mask = mask * (torch.ones_like(mask) - torch.eye(n, device=mask.device, dtype=mask.dtype))
        masked = estogram.permute(1, 2, 0) * mask.unsqueeze(-1)

        p0 = masked[:, :, center].sum(dim=-1)
        p1 = (masked[:, :, center - 1] + masked[:, :, center + 1]).sum(dim=-1)
        p2 = (masked[:, :, center - 2] + masked[:, :, center + 2]).sum(dim=-1)
        p3 = (masked[:, :, center - 3] + masked[:, :, center + 3]).sum(dim=-1)
        p4 = mask.sum(dim=-1)

        return 0.25 * (4.0 * p0 + 3.0 * p1 + 2.0 * p2 + p3) / p4


# ---------------------------------------------------------------------------
# Menagerie staging entries
# ---------------------------------------------------------------------------


def build_deepaccnet():
    torch.manual_seed(0)
    model = DeepAccNet(
        obt_size=70,
        tbt_size=33,
        nretype=20,
        num_chunks=1,
        channel=32,
        no_last_dilation=True,
        partial_instance_norm=True,
    )
    model.eval()
    return model


def example_input_deepaccnet():
    torch.manual_seed(0)
    nres = 6
    grid3d = torch.rand(nres, 20, 24, 24, 24)
    obt = torch.randn(nres, 70)
    tbt = torch.randn(nres, nres, 33)
    return (grid3d, obt, tbt)


MENAGERIE_ZOO = "ported-pytorch"

MENAGERIE_ENTRIES = [
    ("DeepAccNet", "build_deepaccnet", "example_input_deepaccnet", 2021, "ported-pytorch"),
]
