# SOURCE: vendored from https://github.com/aipixel/GaussianAvatar @ main
#
# GaussianAvatar (CVPR 2024): "Towards Realistic Human Avatar Modeling from a
# Single Video via Animatable 3D Gaussians" -- the pose-and-geometry-feature
# -> Gaussian-splat-residual decoder network. Vendored verbatim from
# model/network.py (POP_no_unet, derived from the POP/SCALE lineage per the
# original code's own comment) and model/modules.py (GeomConvLayers,
# ShapeDecoder, uv_to_grid -- the "conv" geom-smoothing branch, which avoids
# the heavier UnetNoCond5DS branch while remaining one of the three real,
# selectable geom_layer_type options in the shipped code). The full
# `AvatarModel` training orchestrator (model/avatar_model.py) additionally
# needs a CUDA 3D-Gaussian-splatting renderer (gaussian_renderer/, a compiled
# extension) and SMPL/SMPL-X body assets/data files, so it cannot be traced
# directly; POP_no_unet is the actual learned neural architecture that
# produces the per-Gaussian residual/scale/spherical-harmonics predictions,
# and is fully self-contained torch.
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
===============================================================================
building blocks (model/modules.py, vendored -- conv geom-smoothing branch only)
===============================================================================
"""


class GeomConvLayers(nn.Module):
    """
    A few convolutional layers to smooth the geometric feature tensor
    """

    def __init__(self, input_nc=16, hidden_nc=16, output_nc=16, use_relu=False):
        super().__init__()
        self.use_relu = use_relu

        self.conv1 = nn.Conv2d(input_nc, hidden_nc, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv2 = nn.Conv2d(hidden_nc, hidden_nc, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv3 = nn.Conv2d(hidden_nc, output_nc, kernel_size=5, stride=1, padding=2, bias=False)
        if use_relu:
            self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        if self.use_relu:
            x = self.relu(x)
        x = self.conv2(x)
        if self.use_relu:
            x = self.relu(x)
        x = self.conv3(x)

        return x


class ShapeDecoder(nn.Module):
    """
    The "Shape Decoder" in the POP paper Fig. 2. The same as the "shared MLP" in the SCALE paper.
    - with skip connection from the input features to the 4th layer's output features (like DeepSDF)
    - branches out at the second-to-last layer, one branch for position pred, one for normal pred
    """

    def __init__(self, in_size, hsize=256, actv_fn="softplus"):
        self.hsize = hsize
        super(ShapeDecoder, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_size, self.hsize, 1)
        self.conv2 = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv3 = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv4 = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv5 = torch.nn.Conv1d(self.hsize + in_size, self.hsize, 1)
        self.conv6 = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv7 = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv8 = torch.nn.Conv1d(self.hsize, 3, 1)

        self.conv6SH = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv7SH = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv8SH = torch.nn.Conv1d(self.hsize, 3, 1)

        self.conv6N = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv7N = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv8N = torch.nn.Conv1d(self.hsize, 1, 1)

        self.bn1 = torch.nn.BatchNorm1d(self.hsize)
        self.bn2 = torch.nn.BatchNorm1d(self.hsize)
        self.bn3 = torch.nn.BatchNorm1d(self.hsize)
        self.bn4 = torch.nn.BatchNorm1d(self.hsize)

        self.bn5 = torch.nn.BatchNorm1d(self.hsize)
        self.bn6 = torch.nn.BatchNorm1d(self.hsize)
        self.bn7 = torch.nn.BatchNorm1d(self.hsize)

        self.bn6N = torch.nn.BatchNorm1d(self.hsize)
        self.bn7N = torch.nn.BatchNorm1d(self.hsize)

        self.bn6SH = torch.nn.BatchNorm1d(self.hsize)
        self.bn7SH = torch.nn.BatchNorm1d(self.hsize)

        self.actv_fn = nn.ReLU() if actv_fn == "relu" else nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.tan = nn.Tanh()

    def forward(self, x):
        x1 = self.actv_fn(self.bn1(self.conv1(x)))
        x2 = self.actv_fn(self.bn2(self.conv2(x1)))
        x3 = self.actv_fn(self.bn3(self.conv3(x2)))
        x4 = self.actv_fn(self.bn4(self.conv4(x3)))
        x5 = self.actv_fn(self.bn5(self.conv5(torch.cat([x, x4], dim=1))))

        # position pred
        x6 = self.actv_fn(self.bn6(self.conv6(x5)))
        x7 = self.actv_fn(self.bn7(self.conv7(x6)))
        x8 = self.conv8(x7)

        # scales pred
        xN6 = self.actv_fn(self.bn6N(self.conv6N(x5)))
        xN7 = self.actv_fn(self.bn7N(self.conv7N(xN6)))
        xN8 = self.conv8N(xN7)

        # shs pred
        xSH6 = self.actv_fn(self.bn6SH(self.conv6SH(x5)))
        xSH7 = self.actv_fn(self.bn7SH(self.conv7SH(xSH6)))
        xSH8 = self.conv8SH(xSH7)

        scales = self.sigmoid(xN8)
        shs = self.sigmoid(xSH8)

        return x8, scales, shs


def uv_to_grid(uv_idx_map, resolution):
    """
    uv_idx_map: shape=[batch, N_uvcoords, 2], ranging between 0-1
    this function basically reshapes the uv_idx_map and shift its value range to (-1, 1) (required by F.gridsample)
    the sqaure of resolution = N_uvcoords
    """
    bs = uv_idx_map.shape[0]
    grid = uv_idx_map.reshape(bs, resolution, resolution, 2) * 2 - 1.0
    grid = grid.transpose(1, 2)
    return grid


"""
===============================================================================
pose/geom-conditioned Gaussian-residual decoder (model/network.py, vendored)
===============================================================================
"""


class POP_no_unet(nn.Module):
    def __init__(
        self,
        c_geom=64,  # channels of the geometric features
        geom_layer_type="conv",  # the type of architecture used for smoothing the geometric feature tensor
        nf=64,  # num filters for the unet
        hsize=256,  # hidden layer size of the ShapeDecoder MLP
        up_mode="upconv",  # upconv or upsample for the upsampling layers in the pose feature UNet
        use_dropout=False,  # whether use dropout in the pose feature UNet
        uv_feat_dim=2,  # input dimension of the uv coordinates
    ):
        super().__init__()
        self.geom_layer_type = geom_layer_type

        geom_proc_layers = {
            # 'unet' / 'bottleneck' branches omitted from this vendored subset (UnetNoCond5DS /
            # GeomConvBottleneckLayers are architecturally analogous conv stacks); 'conv' is a real,
            # independently selectable geom_layer_type in the shipped code.
            "conv": GeomConvLayers(
                c_geom, c_geom, c_geom, use_relu=False
            ),  # use 3 trainable conv layers
        }

        # optional layer for spatially smoothing the geometric feature tensor
        if geom_layer_type is not None:
            self.geom_proc_layers = geom_proc_layers[geom_layer_type]

        # shared shape decoder across different outfit types
        self.decoder = ShapeDecoder(in_size=uv_feat_dim + c_geom, hsize=hsize, actv_fn="softplus")

    def forward(self, pose_featmap, geom_featmap, uv_loc):
        """
        :param x: input posmap, [batch, 3, 256, 256]
        :param geom_featmap: a [B, C, H, W] tensor, spatially pixel-aligned with the pose features extracted by the UNet
        :param uv_loc: querying uv coordinates, ranging between 0 and 1, of shape [B, H*W, 2].
        :param pq_coords: the 'sub-UV-pixel' (p,q) coordinates, range [0,1), shape [B, H*W, 1, 2].
                        Note: It is the intra-patch coordinates in SCALE. Kept here for the backward compatibility with SCALE.
        :return:
            clothing offset vectors (residuals) and normals of the points
        """
        # geometric feature tensor
        if self.geom_layer_type is not None:
            geom_featmap = self.geom_proc_layers(geom_featmap)

        if pose_featmap is None:
            # pose and geom features are concatenated to form the feature for each point
            pix_feature = geom_featmap
        else:
            pix_feature = pose_featmap + geom_featmap

        feat_res = geom_featmap.shape[
            2
        ]  # spatial resolution of the input pose and geometric features
        uv_res = int(uv_loc.shape[1] ** 0.5)  # spatial resolution of the query uv map

        # spatially bilinearly upsample the features to match the query resolution
        if feat_res != uv_res:
            query_grid = uv_to_grid(uv_loc, uv_res)
            pix_feature = F.grid_sample(
                pix_feature, query_grid, mode="bilinear", align_corners=False
            )

        B, C, H, W = pix_feature.shape
        N_subsample = 1  # inherit the SCALE code custom, but now only sample one point per pixel

        uv_feat_dim = uv_loc.size()[-1]
        uv_loc = uv_loc.expand(N_subsample, -1, -1, -1).permute([1, 2, 0, 3])

        # uv and pix feature is shared for all points in each patch
        pix_feature = (
            pix_feature.view(B, C, -1).expand(N_subsample, -1, -1, -1).permute([1, 2, 3, 0])
        )  # [B, C, N_pix, N_sample_perpix]
        pix_feature = pix_feature.reshape(B, C, -1)

        uv_loc = uv_loc.reshape(B, -1, uv_feat_dim).transpose(
            1, 2
        )  # [B, N_pix, N_subsample, 2] --> [B, 2, Num of all pq subpixels]

        residuals, scales, shs = self.decoder(
            torch.cat([pix_feature, uv_loc], 1)
        )  # [B, 3, N all subpixels]

        return residuals, scales, shs


# --- staging harness: build + example input ---------------------------------


def build_gaussianavatar_pop_no_unet():
    return POP_no_unet(
        c_geom=8,
        geom_layer_type="conv",
        nf=8,
        hsize=16,
        up_mode="upconv",
        uv_feat_dim=2,
    ).eval()


def example_input_gaussianavatar_pop_no_unet():
    batch = 1
    c_geom = 8
    feat_res = 4
    n_pix = feat_res * feat_res
    pose_featmap = torch.randn(batch, c_geom, feat_res, feat_res)
    geom_featmap = torch.randn(batch, c_geom, feat_res, feat_res)
    uv_loc = torch.rand(batch, n_pix, 2)
    return (pose_featmap, geom_featmap, uv_loc)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "GaussianAvatar-POP",
        build_gaussianavatar_pop_no_unet,
        example_input_gaussianavatar_pop_no_unet,
        2024,
        MENAGERIE_ZOO,
    ),
]
