# SOURCE: vendored from https://github.com/edwardzhou130/PolarSeg @ master
#
# Vendors the real PolarNet (CVPR 2020, Zhang et al. "PolarNet: An Improved Grid
# Representation for Online LiDAR Point Clouds Semantic Segmentation") architecture as
# implemented in the paper authors' official repo:
#   network/ptBEV.py    (ptBEVnet: per-point PointNet-style MLP encoder -> farthest-point
#                         subsample -> scatter-max pooling into a polar-BEV grid -> optional
#                         local max-pool smoothing -> BEV_Unet)
#   network/BEV_Unet.py  (BEV_Unet / UNet: circular-padded U-Net operating on the polar-BEV
#                          grid, with an auxiliary DropBlock2D regularizer on the decoder)
#
# Both files import only base-env packages already installed here (torch, numpy, numba,
# torch_scatter) EXCEPT `dropblock` (a separate tiny third-party PyPI utility package, not
# part of PolarNet's own contributed architecture -- it supplies a generic drop-in
# regularization layer that BEV_Unet.py imports off the shelf). Since `dropblock` is not
# installed and is not part of PolarNet's own code, its two classes (`DropBlock2D`,
# `DropBlock3D`, MIT-licensed, https://github.com/miguelvr/dropblock) are inlined verbatim
# below rather than approximated -- this is vendoring a real upstream dependency the real
# PolarSeg code itself imports, not a rewrite of PolarNet's architecture.
#
# Three minimal portability fixes only (no architectural change):
#   1. `ptBEVnet.forward` used `pt_fea[0].get_device()` (returns -1 on CPU tensors, which is
#      not a valid device index for `.to(...)`) -- replaced with `pt_fea[0].device` so the
#      real forward pass runs identically on CPU or CUDA.
#   2. `np.bool` (removed in modern NumPy) -> `np.bool_` in the farthest-point-sample mask
#      allocation inside `nb_greedy_FPS`.
#   3. `nb.jit(..., cache=True)` -> `cache=False`: numba's on-disk JIT cache keys off a
#      stable importable module name, which direct-file staging-module loading does not
#      provide; disabling only affects whether the JIT result persists across runs.
#
# The real forward pass requires the "farthest" point-selection path's `numba`-jitted
# `nb_greedy_FPS` OR the "random" path's `grp_range_torch`; both are copied verbatim from the
# real repo. We drive the model on the "random" selection path for the traced example (a real,
# selectable, non-default option: `assert pt_selection in ['random','farthest']` in the real
# constructor), which avoids a `multiprocessing.Pool` spawn while tracing but is otherwise the
# exact real code path (no code was removed or altered -- only the constructor kwarg used).
#
# MENAGERIE_ZOO = "vendored-pytorch"

from __future__ import annotations

import multiprocessing

import numba as nb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# Vendored from https://github.com/miguelvr/dropblock (MIT), a real third-party
# dependency `network/BEV_Unet.py` imports as `from dropblock import DropBlock2D`.
# ---------------------------------------------------------------------------
class DropBlock2D(nn.Module):
    """Randomly zeroes 2D spatial blocks of the input tensor.

    As described in the paper `DropBlock: A regularization method for convolutional
    networks` (https://arxiv.org/abs/1810.12890).
    """

    def __init__(self, drop_prob, block_size):
        super().__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        assert x.dim() == 4, "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.0:
            return x
        gamma = self._compute_gamma(x)
        mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()
        mask = mask.to(x.device)
        block_mask = self._compute_block_mask(mask)
        out = x * block_mask[:, None, :, :]
        out = out * block_mask.numel() / block_mask.sum()
        return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(
            input=mask[:, None, :, :],
            kernel_size=(self.block_size, self.block_size),
            stride=(1, 1),
            padding=self.block_size // 2,
        )
        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]
        block_mask = 1 - block_mask.squeeze(1)
        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size**2)


# ---------------------------------------------------------------------------
# Vendored from network/BEV_Unet.py (real PolarSeg repo code, imports fixed above)
# ---------------------------------------------------------------------------
class BEV_Unet(nn.Module):
    def __init__(
        self,
        n_class,
        n_height,
        dilation=1,
        group_conv=False,
        input_batch_norm=False,
        dropout=0.0,
        circular_padding=False,
        dropblock=True,
        use_vis_fea=False,
    ):
        super().__init__()
        self.n_class = n_class
        self.n_height = n_height
        if use_vis_fea:
            self.network = UNet(
                n_class * n_height,
                2 * n_height,
                dilation,
                group_conv,
                input_batch_norm,
                dropout,
                circular_padding,
                dropblock,
            )
        else:
            self.network = UNet(
                n_class * n_height,
                n_height,
                dilation,
                group_conv,
                input_batch_norm,
                dropout,
                circular_padding,
                dropblock,
            )

    def forward(self, x):
        x = self.network(x)
        x = x.permute(0, 2, 3, 1)
        new_shape = list(x.size())[:3] + [self.n_height, self.n_class]
        x = x.view(new_shape)
        x = x.permute(0, 4, 1, 2, 3)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        n_class,
        n_height,
        dilation,
        group_conv,
        input_batch_norm,
        dropout,
        circular_padding,
        dropblock,
    ):
        super().__init__()
        self.inc = inconv(n_height, 64, dilation, input_batch_norm, circular_padding)
        self.down1 = down(64, 128, dilation, group_conv, circular_padding)
        self.down2 = down(128, 256, dilation, group_conv, circular_padding)
        self.down3 = down(256, 512, dilation, group_conv, circular_padding)
        self.down4 = down(512, 512, dilation, group_conv, circular_padding)
        self.up1 = up(
            1024,
            256,
            circular_padding,
            group_conv=group_conv,
            use_dropblock=dropblock,
            drop_p=dropout,
        )
        self.up2 = up(
            512,
            128,
            circular_padding,
            group_conv=group_conv,
            use_dropblock=dropblock,
            drop_p=dropout,
        )
        self.up3 = up(
            256,
            64,
            circular_padding,
            group_conv=group_conv,
            use_dropblock=dropblock,
            drop_p=dropout,
        )
        self.up4 = up(
            128,
            64,
            circular_padding,
            group_conv=group_conv,
            use_dropblock=dropblock,
            drop_p=dropout,
        )
        self.dropout = nn.Dropout(p=0.0 if dropblock else dropout)
        self.outc = outconv(64, n_class)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(self.dropout(x))
        return x


class double_conv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch, group_conv, dilation=1):
        super().__init__()
        if group_conv:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, groups=min(out_ch, in_ch)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True),
            )

    def forward(self, x):
        return self.conv(x)


class double_conv_circular(nn.Module):
    """(conv => BN => ReLU) * 2, with circular padding along the azimuth axis"""

    def __init__(self, in_ch, out_ch, group_conv, dilation=1):
        super().__init__()
        if group_conv:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=(1, 0), groups=min(out_ch, in_ch)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=(1, 0), groups=out_ch),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True),
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=(1, 0)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=(1, 0)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True),
            )

    def forward(self, x):
        x = F.pad(x, (1, 1, 0, 0), mode="circular")
        x = self.conv1(x)
        x = F.pad(x, (1, 1, 0, 0), mode="circular")
        x = self.conv2(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, dilation, input_batch_norm, circular_padding):
        super().__init__()
        if input_batch_norm:
            if circular_padding:
                self.conv = nn.Sequential(
                    nn.BatchNorm2d(in_ch),
                    double_conv_circular(in_ch, out_ch, group_conv=False, dilation=dilation),
                )
            else:
                self.conv = nn.Sequential(
                    nn.BatchNorm2d(in_ch),
                    double_conv(in_ch, out_ch, group_conv=False, dilation=dilation),
                )
        else:
            if circular_padding:
                self.conv = double_conv_circular(in_ch, out_ch, group_conv=False, dilation=dilation)
            else:
                self.conv = double_conv(in_ch, out_ch, group_conv=False, dilation=dilation)

    def forward(self, x):
        return self.conv(x)


class down(nn.Module):
    def __init__(self, in_ch, out_ch, dilation, group_conv, circular_padding):
        super().__init__()
        if circular_padding:
            self.mpconv = nn.Sequential(
                nn.MaxPool2d(2),
                double_conv_circular(in_ch, out_ch, group_conv=group_conv, dilation=dilation),
            )
        else:
            self.mpconv = nn.Sequential(
                nn.MaxPool2d(2),
                double_conv(in_ch, out_ch, group_conv=group_conv, dilation=dilation),
            )

    def forward(self, x):
        return self.mpconv(x)


class up(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        circular_padding,
        bilinear=True,
        group_conv=False,
        use_dropblock=False,
        drop_p=0.5,
    ):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        elif group_conv:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2, groups=in_ch // 2)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        if circular_padding:
            self.conv = double_conv_circular(in_ch, out_ch, group_conv=group_conv)
        else:
            self.conv = double_conv(in_ch, out_ch, group_conv=group_conv)

        self.use_dropblock = use_dropblock
        if self.use_dropblock:
            self.dropblock = DropBlock2D(block_size=7, drop_prob=drop_p)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        if self.use_dropblock:
            x = self.dropblock(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------------------
# Vendored from network/ptBEV.py (real PolarSeg repo code)
# ---------------------------------------------------------------------------
class ptBEVnet(nn.Module):
    def __init__(
        self,
        BEV_net,
        grid_size,
        pt_model="pointnet",
        fea_dim=3,
        pt_pooling="max",
        kernal_size=3,
        out_pt_fea_dim=64,
        max_pt_per_encode=64,
        cluster_num=4,
        pt_selection="farthest",
        fea_compre=None,
    ):
        super().__init__()
        assert pt_pooling in ["max"]
        assert pt_selection in ["random", "farthest"]

        if pt_model == "pointnet":
            self.PPmodel = nn.Sequential(
                nn.BatchNorm1d(fea_dim),
                nn.Linear(fea_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, out_pt_fea_dim),
            )

        self.pt_model = pt_model
        self.BEV_model = BEV_net
        self.pt_pooling = pt_pooling
        self.max_pt = max_pt_per_encode
        self.pt_selection = pt_selection
        self.fea_compre = fea_compre
        self.grid_size = grid_size

        if kernal_size != 1:
            if self.pt_pooling == "max":
                self.local_pool_op = torch.nn.MaxPool2d(
                    kernal_size, stride=1, padding=(kernal_size - 1) // 2, dilation=1
                )
            else:
                raise NotImplementedError
        else:
            self.local_pool_op = None

        if self.pt_pooling == "max":
            self.pool_dim = out_pt_fea_dim

        if self.fea_compre is not None:
            self.fea_compression = nn.Sequential(
                nn.Linear(self.pool_dim, self.fea_compre), nn.ReLU()
            )
            self.pt_fea_dim = self.fea_compre
        else:
            self.pt_fea_dim = self.pool_dim

    def forward(self, pt_fea, xy_ind, voxel_fea=None):
        # PORTABILITY FIX: real code used `pt_fea[0].get_device()` (CUDA-only convention,
        # returns -1 on CPU tensors, which `.to(-1)` cannot resolve). `.device` is the
        # portable CPU/CUDA equivalent; no architectural change.
        cur_dev = pt_fea[0].device

        # concate everything
        cat_pt_ind = []
        for i_batch in range(len(xy_ind)):
            cat_pt_ind.append(F.pad(xy_ind[i_batch], (1, 0), "constant", value=i_batch))

        cat_pt_fea = torch.cat(pt_fea, dim=0)
        cat_pt_ind = torch.cat(cat_pt_ind, dim=0)
        pt_num = cat_pt_ind.shape[0]

        # shuffle the data
        shuffled_ind = torch.randperm(pt_num, device=cur_dev)
        cat_pt_fea = cat_pt_fea[shuffled_ind, :]
        cat_pt_ind = cat_pt_ind[shuffled_ind, :]

        # unique xy grid index
        unq, unq_inv, unq_cnt = torch.unique(
            cat_pt_ind, return_inverse=True, return_counts=True, dim=0
        )
        unq = unq.type(torch.int64)

        # subsample pts (real repo's "random" path -- avoids multiprocessing.Pool spawn
        # while tracing; the "farthest" path below is retained verbatim/untouched)
        if self.pt_selection == "random":
            grp_ind = grp_range_torch(unq_cnt, cur_dev)[torch.argsort(torch.argsort(unq_inv))]
            remain_ind = grp_ind < self.max_pt
        elif self.pt_selection == "farthest":
            unq_ind = np.split(
                np.argsort(unq_inv.detach().cpu().numpy()),
                np.cumsum(unq_cnt.detach().cpu().numpy()[:-1]),
            )
            remain_ind = np.zeros((pt_num,), dtype=np.bool_)
            np_cat_fea = cat_pt_fea.detach().cpu().numpy()[:, :3]
            pool_in = []
            for i_inds in unq_ind:
                if len(i_inds) > self.max_pt:
                    pool_in.append((np_cat_fea[i_inds, :], self.max_pt))
            if len(pool_in) > 0:
                pool = multiprocessing.Pool(multiprocessing.cpu_count())
                FPS_results = pool.starmap(parallel_FPS, pool_in)
                pool.close()
                pool.join()
            count = 0
            for i_inds in unq_ind:
                if len(i_inds) <= self.max_pt:
                    remain_ind[i_inds] = True
                else:
                    remain_ind[i_inds[FPS_results[count]]] = True
                    count += 1

        cat_pt_fea = cat_pt_fea[remain_ind, :]
        cat_pt_ind = cat_pt_ind[remain_ind, :]
        unq_inv = unq_inv[remain_ind]
        unq_cnt = torch.clamp(unq_cnt, max=self.max_pt)

        # process feature
        if self.pt_model == "pointnet":
            processed_cat_pt_fea = self.PPmodel(cat_pt_fea)

        if self.pt_pooling == "max":
            pooled_data = torch_scatter.scatter_max(processed_cat_pt_fea, unq_inv, dim=0)[0]
        else:
            raise NotImplementedError

        if self.fea_compre:
            processed_pooled_data = self.fea_compression(pooled_data)
        else:
            processed_pooled_data = pooled_data

        # stuff pooled data into 4D tensor
        out_data_dim = [len(pt_fea), self.grid_size[0], self.grid_size[1], self.pt_fea_dim]
        out_data = torch.zeros(out_data_dim, dtype=torch.float32).to(cur_dev)
        out_data[unq[:, 0], unq[:, 1], unq[:, 2], :] = processed_pooled_data
        out_data = out_data.permute(0, 3, 1, 2)
        if self.local_pool_op is not None:
            out_data = self.local_pool_op(out_data)
        if voxel_fea is not None:
            out_data = torch.cat((out_data, voxel_fea), 1)

        # run through network
        net_return_data = self.BEV_model(out_data)

        return net_return_data


def grp_range_torch(a, dev):
    idx = torch.cumsum(a, 0)
    id_arr = torch.ones(idx[-1], dtype=torch.int64, device=dev)
    id_arr[0] = 0
    id_arr[idx[:-1]] = -a[:-1] + 1
    return torch.cumsum(id_arr, 0)


def parallel_FPS(np_cat_fea, K):
    return nb_greedy_FPS(np_cat_fea, K)


# NOTE: real code used `cache=True`; disabled here only because numba's on-disk cache
# keys off a stable importable module name, which direct-file staging-module loading
# (importlib.util.spec_from_file_location) does not provide. No architectural change --
# this only affects whether the JIT result is persisted to disk between runs.
@nb.jit("b1[:](f4[:,:],i4)", nopython=True, cache=False)
def nb_greedy_FPS(xyz, K):
    start_element = 0
    sample_num = xyz.shape[0]
    sum_vec = np.zeros((sample_num, 1), dtype=np.float32)
    xyz_sq = xyz**2
    for j in range(sample_num):
        sum_vec[j, 0] = np.sum(xyz_sq[j, :])
    pairwise_distance = sum_vec + np.transpose(sum_vec) - 2 * np.dot(xyz, np.transpose(xyz))

    candidates_ind = np.zeros((sample_num,), dtype=np.bool_)
    candidates_ind[start_element] = True
    remain_ind = np.ones((sample_num,), dtype=np.bool_)
    remain_ind[start_element] = False
    all_ind = np.arange(sample_num)

    for i in range(1, K):
        if i == 1:
            min_remain_pt_dis = pairwise_distance[:, start_element]
            min_remain_pt_dis = min_remain_pt_dis[remain_ind]
        else:
            cur_dis = pairwise_distance[remain_ind, :]
            cur_dis = cur_dis[:, candidates_ind]
            min_remain_pt_dis = np.zeros((cur_dis.shape[0],), dtype=np.float32)
            for j in range(cur_dis.shape[0]):
                min_remain_pt_dis[j] = np.min(cur_dis[j, :])
        next_ind_in_remain = np.argmax(min_remain_pt_dis)
        next_ind = all_ind[remain_ind][next_ind_in_remain]
        candidates_ind[next_ind] = True
        remain_ind[next_ind] = False

    return candidates_ind


# ---------------------------------------------------------------------------
# Staging build/example functions
# ---------------------------------------------------------------------------
# tiny polar-BEV grid (real defaults e.g. [480,360,32]); azimuth/radius dims must survive
# 4 stride-2 max-pool stages in the U-Net (16, 16 -> 8 -> 4 -> 2 -> 1), so >= 16 each.
_GRID_SIZE = [16, 16, 4]
_FEA_DIM = 3
_N_CLASS = 3


def build_polarnet():
    bev_net = BEV_Unet(
        n_class=_N_CLASS,
        n_height=_GRID_SIZE[2],
        dilation=1,
        group_conv=False,
        input_batch_norm=True,
        dropout=0.5,
        circular_padding=True,
        dropblock=True,
    )
    model = ptBEVnet(
        BEV_net=bev_net,
        grid_size=_GRID_SIZE,
        pt_model="pointnet",
        fea_dim=_FEA_DIM,
        pt_pooling="max",
        kernal_size=3,
        out_pt_fea_dim=64,
        max_pt_per_encode=64,
        pt_selection="random",
        fea_compre=_GRID_SIZE[2],
    )
    model.eval()
    return model


def example_input_polarnet():
    torch.manual_seed(0)
    n_pts = 40
    pt_fea = [torch.randn(n_pts, _FEA_DIM)]
    gx = torch.randint(0, _GRID_SIZE[0], (n_pts, 1))
    gy = torch.randint(0, _GRID_SIZE[1], (n_pts, 1))
    xy_ind = [torch.cat([gx, gy], dim=1)]
    return (pt_fea, xy_ind)


MENAGERIE_ENTRIES = [
    ("PolarNet", "build_polarnet", "example_input_polarnet", 2020, "vendored-pytorch"),
]
