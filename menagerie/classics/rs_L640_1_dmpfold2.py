# SOURCE: vendored from psipred/DMPfold2 @ master
#
# DMPfold2 (D.T. Jones, 2020/2021) predicts protein 3D structure end-to-end from a
# multiple sequence alignment: a bidirectional-GRU sequence encoder builds an outer-product
# pairwise map, which is fused with DCA covariance features and refined by a deep dilated
# ResNet (Maxout2d + spatial/channel squeeze-excite blocks) into a distance map; the distance
# map is converted to 3D coordinates via classical MDS and a coordinate-refinement GRU, with
# `nloops` recycling iterations that feed the predicted distance map back into the ResNet.
# The class below (`GRUResNet`, plus its helper modules `Maxout2d`/`CSE`/`SSE`/`SCSE`/
# `ResNet_Block`) is copied from dmpfold/network.py, unmodified except for one Torch-API
# compatibility fix: `torch.symeig` (removed in torch>=1.13) is replaced with the documented
# modern equivalent `torch.linalg.eigh` (same eigendecomposition math, PyTorch's own migration
# guide names this the drop-in replacement) so the real 2020-era code runs unmodified on
# current torch. `refine_coords`/`calpha_to_main_chain` (post-forward geometry helpers, not
# nn.Module state) and the `checkpoint_sequential` gradient-checkpointing branch (an
# autograd-only fast path irrelevant to a no-grad trace) are kept as in the original.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential
from math import sqrt, asin, cos, pi, sin

MENAGERIE_ZOO = "vendored-pytorch"

NUM_CHANNELS = 442


class Maxout2d(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size, kernel_size=1, dilation=1, block=0):
        super(Maxout2d, self).__init__()
        self.in_channels, self.out_channels, self.pool_size = in_channels, out_channels, pool_size
        self.lin = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels * pool_size,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=dilation * (kernel_size - 1) // 2,
        )
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        if block > 0:
            nn.init.xavier_uniform_(self.lin.weight, gain=1.0 / sqrt(block))
        else:
            nn.init.xavier_uniform_(self.lin.weight, gain=1.0)

    def forward(self, inputs):
        x = self.lin(inputs)

        N, C, H, W = x.size()

        x = x.view(N, C // self.pool_size, self.pool_size, H, W)
        x = x.max(dim=2)[0]
        x = self.norm(x)

        return x


class CSE(nn.Module):
    def __init__(self, width, reduction=16):
        super(CSE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(width, width // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(width // reduction, width, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        N, C, _, _ = x.size()
        y = self.avg_pool(x).view(N, C)
        y = self.fc(y).view(N, C, 1, 1)

        return x * y.expand_as(x)


class SSE(nn.Module):
    def __init__(self, in_ch):
        super(SSE, self).__init__()

        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1, stride=1)

    def forward(self, x):
        y = self.conv(x)
        y = torch.sigmoid(y)

        return x * y


class SCSE(nn.Module):
    def __init__(self, in_ch, r):
        super(SCSE, self).__init__()

        self.cSE = CSE(in_ch, r)
        self.sSE = SSE(in_ch)

    def forward(self, x):
        cSE = self.cSE(x)
        sSE = self.sSE(x)

        return cSE + sSE


# ResNet Module
class ResNet_Block(nn.Module):
    def __init__(self, width, fsize, dilv, nblock):
        super(ResNet_Block, self).__init__()
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout2d(p=0.2)
        self.layer1 = Maxout2d(
            in_channels=width,
            out_channels=width,
            pool_size=4,
            kernel_size=fsize,
            dilation=dilv,
            block=nblock,
        )
        self.scSE = SCSE(width, 16)

    def forward(self, x):
        residual = x
        out = self.dropout1(x)
        out = self.dropout2(out)
        out = self.layer1(out)
        out = self.scSE(out)
        out = out + residual

        return out


def refine_coords(coords, n_steps):
    vdw_dist, cov_dist = 3.0, 3.78
    k_vdw, k_cov = 100.0, 100.0
    min_speed = 0.001

    for i in range(n_steps):
        n_res = coords.size(0)
        accels = coords * 0

        # Steric clashing
        crep = coords.unsqueeze(0).expand(n_res, -1, -1)
        diffs = crep - crep.transpose(0, 1)
        dists = diffs.norm(dim=2).clamp(min=0.01, max=10.0)
        norm_diffs = diffs / dists.unsqueeze(2)
        violate = (dists < vdw_dist).to(torch.float) * (vdw_dist - dists)
        forces = k_vdw * violate
        pair_accels = forces.unsqueeze(2) * norm_diffs
        accels += pair_accels.sum(dim=0)

        # Adjacent C-alphas
        diffs = coords[1:] - coords[:-1]
        dists = diffs.norm(dim=1).clamp(min=0.1)
        norm_diffs = diffs / dists.unsqueeze(1)
        violate = (dists - cov_dist).clamp(max=3.0)
        forces = k_cov * violate
        accels_cov = forces.unsqueeze(1) * norm_diffs
        accels[:-1] += accels_cov
        accels[1:] -= accels_cov

        coords = coords + accels.clamp(min=-100.0, max=100.0) * min_speed

    return coords


# Calculate main chain and Cbeta atom from Calphas, based on code by W. Taylor using the Levitt method
def calpha_to_main_chain(coords_ca):
    # Place dummy extra Cα atoms on end to get the required vectors
    vec_can2_can1 = coords_ca[:, :1, :] - coords_ca[:, 1:2, :]
    vec_can2_can3 = coords_ca[:, 2:3, :] - coords_ca[:, 1:2, :]
    vec_cac2_cac1 = coords_ca[:, -1:, :] - coords_ca[:, -2:-1, :]
    vec_cac2_cac3 = coords_ca[:, -3:-2, :] - coords_ca[:, -2:-1, :]
    coord_ca_nterm = coords_ca[:, :1, :] + 3.82 * F.normalize(
        torch.cross(vec_can2_can1, vec_can2_can3), dim=2
    )
    coord_ca_cterm = coords_ca[:, -1:, :] + 3.82 * F.normalize(
        torch.cross(vec_cac2_cac1, vec_cac2_cac3), dim=2
    )
    coords_ca_ext = torch.cat((coord_ca_nterm, coords_ca, coord_ca_cterm), dim=1)

    vec_ca_can = coords_ca_ext[:, :-2, :] - coords_ca_ext[:, 1:-1, :]
    vec_ca_cac = coords_ca_ext[:, 2:, :] - coords_ca_ext[:, 1:-1, :]
    mid_ca_can = (coords_ca_ext[:, 1:, :] + coords_ca_ext[:, :-1, :]) / 2
    cross_vcan_vcac = F.normalize(torch.cross(vec_ca_can, vec_ca_cac, dim=2), dim=2)
    coords_n = mid_ca_can[:, :-1, :] - vec_ca_can / 8 + cross_vcan_vcac / 4
    # coords_h = mid_ca_can[:, :-1, :] + cross_vcan_vcac * 1.0
    coords_c_shift = mid_ca_can[:, :-1, :] + vec_ca_can / 8 - cross_vcan_vcac / 2
    coords_o_shift = mid_ca_can[:, :-1, :] - cross_vcan_vcac * 1.8
    # Add the final C and O
    coords_c_cterm = (
        mid_ca_can[:, -1:, :] - vec_ca_cac[:, -1:, :] / 8 + cross_vcan_vcac[:, -1:, :] / 2
    )
    coords_o_cterm = mid_ca_can[:, -1:, :] + cross_vcan_vcac[:, -1:, :] * 2.0
    coords_c = torch.cat((coords_c_shift[:, 1:, :], coords_c_cterm), dim=1)
    coords_o = torch.cat((coords_o_shift[:, 1:, :], coords_o_cterm), dim=1)

    vec_n_ca = coords_ca - coords_n
    vec_c_ca = coords_ca - coords_c
    cross_vn_vc = torch.cross(vec_n_ca, vec_c_ca, dim=2)
    vec_ca_cb = vec_n_ca + vec_c_ca
    ang = pi / 2 - asin(1 / sqrt(3))
    sx = (1.5 * cos(ang) / vec_ca_cb.norm(dim=2)).unsqueeze(2)
    sy = (1.5 * sin(ang) / cross_vn_vc.norm(dim=2)).unsqueeze(2)
    coords_cb = coords_ca + sx * vec_ca_cb + sy * cross_vn_vc

    out = torch.cat(
        (
            coords_n.unsqueeze(2),  # coords_h.unsqueeze(2),
            coords_ca.unsqueeze(2),
            coords_c.unsqueeze(2),
            coords_o.unsqueeze(2),
            coords_cb.unsqueeze(2),
        ),
        dim=2,
    )
    return out.view(coords_ca.size(0), 5 * coords_ca.size(1), 3)


# RNNResNet Module
class GRUResNet(nn.Module):
    def __init__(self, width, cwidth):
        super(GRUResNet, self).__init__()

        self.width = width
        self.cwidth = cwidth

        self.embed = nn.Embedding.from_pretrained(torch.eye(22), freeze=True)
        self.vgru = nn.GRU(22, width, batch_first=False, num_layers=2, bidirectional=False)
        self.hgru = nn.GRU(
            width, width // 2, batch_first=False, num_layers=2, dropout=0.1, bidirectional=True
        )

        layers = []

        layer = Maxout2d(in_channels=NUM_CHANNELS + width + 1, out_channels=cwidth, pool_size=3)

        layers.append(layer)

        nblock = 1

        for rep in range(16):
            for fsize, dilv in [(5, 1)]:
                if fsize > 0:
                    layer = ResNet_Block(cwidth, fsize, dilv, nblock)
                    layers.append(layer)
                    nblock += 1

        layer = nn.Conv2d(in_channels=cwidth, out_channels=2, kernel_size=1)

        layers.append(layer)

        self.resnet = nn.Sequential(*layers)

        self.coord_gru = nn.GRU(
            width + 8, width // 2, batch_first=True, num_layers=3, dropout=0.1, bidirectional=True
        )

        self.coord_fc = nn.Linear(width, 3, bias=False)

    def forward(self, x, x2, nloops=5, refine_steps=0):
        nres = x.size()[1]

        x = self.embed(x)
        x = self.vgru(x)[0]
        x = self.hgru(x[-1, :, :].unsqueeze(1))[0]
        mat1d = x.permute(1, 2, 0)
        x = mat1d.unsqueeze(2) * mat1d.unsqueeze(3)

        resinp = torch.cat((x, x2), dim=1)

        if x.requires_grad:
            # now call the checkpoint API and get the output
            x = checkpoint_sequential(self.resnet, 4, resinp)
        else:
            x = self.resnet(resinp)

        dm = x[:, 0, :, :]
        conf = x[:, 1, :, :].mean(dim=2)
        best_conf = conf

        # Ensure symmetric matrix
        dm = (dm + dm.transpose(1, 2)) / 2
        # Force values to be non-negative
        dm = torch.abs(dm)
        # See https://math.stackexchange.com/questions/156161/finding-the-coordinates-of-points-from-distance-matrix
        M = 0.5 * (
            dm[:, 0:1, :].expand(-1, nres, -1) ** 2
            + dm[:, :, 0:1].expand(-1, -1, nres) ** 2
            - dm**2
        )
        # torch.symeig removed in torch>=1.13; torch.linalg.eigh is the documented
        # drop-in replacement (same eigendecomposition of a symmetric matrix).
        w, v = torch.linalg.eigh(M.float())
        w = torch.clamp(F.relu(w, inplace=False), min=1e-8)
        w = torch.diag_embed(w.sqrt())
        mds_coords = torch.matmul(v, w)[:, :, -8:]
        coordembed = torch.cat((mat1d.permute(0, 2, 1), mds_coords), dim=2)

        gru_out = self.coord_gru(coordembed)[0]

        ca_coords = self.coord_fc(gru_out)

        if refine_steps > 0:
            ca_coords = refine_coords(ca_coords.squeeze(0), refine_steps).unsqueeze(0)

        best_coords = ca_coords

        # print(best_conf.sum().item())

        for i in range(nloops):
            # print(ca_coords.size())

            # Add small amount of noise to seed coordinates
            # ca_coords = ca_coords + 0.5 * torch.randn_like(ca_coords)
            # if refine_steps > 0:
            #    ca_coords = refine_coords(ca_coords.squeeze(0), refine_steps).unsqueeze(0)
            dmap = torch.clamp(
                (ca_coords - ca_coords.transpose(0, 1)).pow(2).sum(dim=2), min=1e-8
            ).sqrt()
            resinp = torch.cat((resinp[:, :-1], dmap.unsqueeze(0).unsqueeze(0)), dim=1)

            if resinp.requires_grad:
                # now call the checkpoint API and get the output
                x = checkpoint_sequential(self.resnet, 4, resinp)
            else:
                x = self.resnet(resinp)

            dm = x[:, 0, :, :]
            conf = x[:, 1, :, :].mean(dim=2)

            # print(conf.sum().item())

            # Ensure symmetric matrix
            dm = (dm + dm.transpose(1, 2)) / 2
            # Force values to be non-negative
            dm = torch.abs(dm)
            # See https://math.stackexchange.com/questions/156161/finding-the-coordinates-of-points-from-distance-matrix
            M = 0.5 * (
                dm[:, 0:1, :].expand(-1, nres, -1) ** 2
                + dm[:, :, 0:1].expand(-1, -1, nres) ** 2
                - dm**2
            )
            w, v = torch.linalg.eigh(M.float())
            w = torch.clamp(F.relu(w, inplace=False), min=1e-8)
            w = torch.diag_embed(w.sqrt())
            mds_coords = torch.matmul(v, w)[:, :, -8:]
            coordembed = torch.cat((mat1d.permute(0, 2, 1), mds_coords), dim=2)

            gru_out = self.coord_gru(coordembed)[0]

            ca_coords = self.coord_fc(gru_out)

            if conf.mean() > best_conf.mean():
                # if refine_steps > 0:
                #    ca_coords = refine_coords(ca_coords.squeeze(0), refine_steps).unsqueeze(0)
                best_conf = conf
                best_coords = ca_coords

        if refine_steps > 0:
            best_coords = refine_coords(best_coords.squeeze(0), refine_steps).unsqueeze(0)

        coords_out = calpha_to_main_chain(best_coords)
        conf_out = torch.sigmoid(best_conf)

        return coords_out, conf_out


def build_dmpfold2():
    # Real caller (predict.py aln_to_coords) constructs GRUResNet(512, 128). Width/cwidth
    # shrunk for a tiny random-init trace; NUM_CHANNELS (DCA feature depth, module-level
    # constant used by the real construction of `resnet`'s first Maxout2d) is left as-is
    # since it is architectural, not a size knob a caller controls.
    return GRUResNet(width=8, cwidth=4)


def example_input_dmpfold2():
    torch.manual_seed(0)
    # length must be >= 8: the classical-MDS step slices the eigendecomposition of an
    # [nres, nres] Gram matrix down to its top-8 components (`mds_coords = ...[:, :, -8:]`),
    # which requires at least 8 residues for that slice to actually width to 8 columns.
    length = 8
    # x: [nseqs, length] amino-acid index MSA (real caller builds this via
    # aa_trans-translated alignment rows fed through nn.Embedding.from_pretrained(eye(22))).
    x = torch.randint(0, 22, (2, length))
    # x2: [1, NUM_CHANNELS+1, length, length] DCA covariance features concatenated with the
    # (recycled) distance map channel, matching `resinp = cat((x, x2), dim=1)` in forward().
    x2 = torch.randn(1, NUM_CHANNELS + 1, length, length)
    return (x, x2, 2, 0)


MENAGERIE_ENTRIES = [
    ("DMPfold2", "build_dmpfold2", "example_input_dmpfold2", 2020, MENAGERIE_ZOO),
]
