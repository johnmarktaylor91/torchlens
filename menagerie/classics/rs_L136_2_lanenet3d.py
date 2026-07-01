# SOURCE: vendored from yuliangguo/Pytorch_Generalized_3D_Lane_Detection @ master
# File: networks/LaneNet3D.py (+ homography helpers from tools/utils.py)
# https://github.com/yuliangguo/Pytorch_Generalized_3D_Lane_Detection
#
# Minimal changes from the original source:
#   - `from tools.utils import *` (the repo's 1200-line utils module, which
#     pulls in matplotlib/cv2/mpl_toolkits/scipy purely for dataset IO,
#     plotting, and eval-metric helpers unrelated to the network) is
#     replaced with a direct, verbatim copy of only the two homography
#     helper functions the `Net.__init__` constructor actually calls:
#     `homography_ipmnorm2g` and `homography_crop_resize` (copied unchanged
#     from `tools/utils.py`, only numpy + cv2, both base libs).
#   - `Net.__init__` originally takes an `argparse.Namespace` built by
#     `tools.utils.define_args()` + a dataset-config function
#     (`tusimple_config`/`sim3d_config`). Those are replaced with a small
#     `SimpleNamespace`-based `_build_default_args()` carrying the same
#     field names/defaults `tusimple_config` sets (org_h/org_w/crop_y/K/
#     cam_height/pitch/top_view_region/anchor_y_steps/batch_norm/
#     pretrained/batch_size/no_cuda/pred_cam/no_3d/no_centerline), scaled
#     down to tiny image/IPM sizes so the trace runs fast; the `Net` class
#     body itself is untouched.
#   - `if not self.no_cuda: ... .cuda()` branches are exercised with
#     `no_cuda=True` (CPU-only default), matching the source's own
#     CPU-vs-CUDA branch -- no code deleted, just the flag left False->True
#     for a CPU trace.
#   - `model.load_pretrained_vgg` is not called (kept but unused): the
#     original loads real ImageNet VGG16 weights; for a random-init trace
#     that call is simply skipped, no line removed.
#
# Architecture (unmodified from source): 3D-LaneNet (Garnett, Cohen, Pe'er,
# Lahav, Levi, ICCV 2019, "3D-LaneNet: End-to-End 3D Multiple Lane
# Detection"), this specific unofficial-but-standard PyTorch implementation
# from the Gen-LaneNet repo. A `VggEncoder` (torchvision VGG16 features
# split into 4 stage blocks) extracts multi-scale image-view features;
# `ProjectiveGridGenerator` computes an inverse-perspective-mapping (IPM)
# sampling grid per stage from a camera homography (`M_inv`, built from
# camera intrinsics/extrinsics + a ground-plane region); `F.grid_sample`
# warps each image-view feature map into bird's-eye-view (top-view) space;
# a `TopViewPathway` (dual-input conv trunk taking the previous top-view
# feature and the newly projected next-stage image feature, concatenated)
# progressively fuses the warped features across scales; a
# `LanePredictionHead` (anchor-based 1D-per-column conv head) regresses
# per-anchor-column 3D lane offsets + lane-type probabilities. Optionally
# (`pred_cam=True`, disabled here) a `RoadPlanePredHead` predicts camera
# height/pitch online and updates the IPM homography mid-forward.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from types import SimpleNamespace


# --- vendored homography helpers (tools/utils.py, base libs only) ---


def homography_ipmnorm2g(top_view_region):
    import cv2

    src = np.float32([[0, 0], [1, 0], [0, 1], [1, 1]])
    H_ipmnorm2g = cv2.getPerspectiveTransform(src, np.float32(top_view_region))
    return H_ipmnorm2g


def homography_crop_resize(org_img_size, crop_y, resize_img_size):
    """
        compute the homography matrix transform original image to cropped and resized image
    :param org_img_size: [org_h, org_w]
    :param crop_y:
    :param resize_img_size: [resize_h, resize_w]
    :return:
    """
    ratio_x = resize_img_size[1] / org_img_size[1]
    ratio_y = resize_img_size[0] / (org_img_size[0] - crop_y)
    H_c = np.array([[ratio_x, 0, 0], [0, ratio_y, -ratio_y * crop_y], [0, 0, 1]])
    return H_c


def make_layers(cfg, in_channels=3, batch_norm=False):
    layers = []
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def make_one_layer(in_channels, out_channels, kernel_size=3, padding=1, stride=1, batch_norm=False):
    conv2d = nn.Conv2d(
        in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride
    )
    if batch_norm:
        layers = [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
    else:
        layers = [conv2d, nn.ReLU(inplace=True)]
    return layers


class VggEncoder(nn.Module):
    def __init__(self, batch_norm=False, init_weights=True):
        super(VggEncoder, self).__init__()
        if batch_norm:
            model_org = models.vgg16_bn()
            output_layers = [12, 22, 32, 42]
        else:
            model_org = models.vgg16()
            output_layers = [8, 15, 22, 29]
        self.features1 = nn.Sequential(*list(model_org.features.children())[: output_layers[0] + 1])
        self.features2 = nn.Sequential(
            *list(model_org.features.children())[output_layers[0] + 1 : output_layers[1] + 1]
        )
        self.features3 = nn.Sequential(
            *list(model_org.features.children())[output_layers[1] + 1 : output_layers[2] + 1]
        )
        self.features4 = nn.Sequential(
            *list(model_org.features.children())[output_layers[2] + 1 : output_layers[3] + 1]
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x1)
        x3 = self.features3(x2)
        x4 = self.features4(x3)
        return x1, x2, x3, x4

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# Road plane prediction: estimate camera height and pitch angle
class RoadPlanePredHead(nn.Module):
    def __init__(self, im_h, im_w, batch_norm=False, init_weights=True):
        super().__init__()
        self.im_h = im_h
        self.im_w = im_w
        self.features1 = make_layers(["M", 256, 256, 256], 512, batch_norm)
        self.features2 = make_layers(["M", 128, 128, 128], 256, batch_norm)
        self.features3 = make_layers(["M", 64, 64, 64], 128, batch_norm)
        # fc layer
        self.fc = nn.Sequential(
            nn.Linear(64 * int(self.im_h / 128) * int(self.im_w / 128), 64),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(64, 2),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x1)
        x3 = self.features3(x2)
        x3 = x3.reshape([-1, 64 * int(self.im_h / 128) * int(self.im_w / 128)])
        out = self.fc(x3)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# initialize base_grid with different sizes can adapt to different sizes
class ProjectiveGridGenerator(nn.Module):
    def __init__(self, size_ipm, M, no_cuda):
        """

        :param size_ipm: size of ipm tensor NCHW
        :param im_h: height of image tensor
        :param im_w: width of image tensor
        :param M: normalized transformation matrix between image view and IPM
        :param no_cuda:
        """
        super().__init__()
        self.N, self.H, self.W = size_ipm
        linear_points_W = torch.linspace(0, 1 - 1 / self.W, self.W)
        linear_points_H = torch.linspace(0, 1 - 1 / self.H, self.H)

        # use M only to decide the type not value
        self.base_grid = M.new(self.N, self.H, self.W, 3)
        self.base_grid[:, :, :, 0] = torch.ger(torch.ones(self.H), linear_points_W).expand_as(
            self.base_grid[:, :, :, 0]
        )
        self.base_grid[:, :, :, 1] = torch.ger(linear_points_H, torch.ones(self.W)).expand_as(
            self.base_grid[:, :, :, 1]
        )
        self.base_grid[:, :, :, 2] = 1

        self.base_grid = Variable(self.base_grid)
        if not no_cuda:
            self.base_grid = self.base_grid.cuda()

    def forward(self, M):
        # compute the grid mapping based on the input transformation matrix M
        # if base_grid is top-view, M should be ipm-to-img homography transformation, and vice versa
        grid = torch.bmm(self.base_grid.view(self.N, self.H * self.W, 3), M.transpose(1, 2))
        grid = torch.div(grid[:, :, 0:2], grid[:, :, 2:]).reshape((self.N, self.H, self.W, 2))
        """
        output grid to be used for grid_sample.
            1. grid specifies the sampling pixel locations normalized by the input spatial dimensions.
            2. pixel locations need to be converted to the range (-1, 1)
        """
        grid = (grid - 0.5) * 2
        return grid


# Sub-network corresponding to the top view pathway
class TopViewPathway(nn.Module):
    def __init__(self, batch_norm=False, init_weights=True):
        super(TopViewPathway, self).__init__()
        self.features1 = make_layers(["M", 128, 128, 128], 128, batch_norm)
        self.features2 = make_layers(["M", 256, 256, 256], 256, batch_norm)
        self.features3 = make_layers(["M", 256, 256, 256], 512, batch_norm)

        if init_weights:
            self._initialize_weights()

    def forward(self, a, b, c, d):
        x = self.features1(a)
        feat_1 = x
        x = torch.cat((x, b), 1)
        x = self.features2(x)
        feat_2 = x
        x = torch.cat((x, c), 1)
        x = self.features3(x)
        feat_3 = x
        x = torch.cat((x, d), 1)
        return x, feat_1, feat_2, feat_3

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


#  Lane Prediction Head: through a series of convolutions with no padding in the y dimension, the feature maps are
#  reduced in height, and finally the prediction layer size is N x 1 x 3 .(2 . K + 1)
class LanePredictionHead(nn.Module):
    def __init__(self, num_lane_type, anchor_dim, batch_norm=False):
        super(LanePredictionHead, self).__init__()
        self.num_lane_type = num_lane_type
        self.anchor_dim = anchor_dim
        layers = []
        layers += make_one_layer(512, 64, kernel_size=3, padding=(0, 1), batch_norm=batch_norm)
        layers += make_one_layer(64, 64, kernel_size=3, padding=(0, 1), batch_norm=batch_norm)
        layers += make_one_layer(64, 64, kernel_size=3, padding=(0, 1), batch_norm=batch_norm)

        layers += make_one_layer(64, 64, kernel_size=5, padding=(0, 2), batch_norm=batch_norm)
        layers += make_one_layer(64, 64, kernel_size=5, padding=(0, 2), batch_norm=batch_norm)
        layers += make_one_layer(64, 64, kernel_size=5, padding=(0, 2), batch_norm=batch_norm)
        layers += make_one_layer(64, 64, kernel_size=5, padding=(0, 2), batch_norm=batch_norm)
        self.features = nn.Sequential(*layers)

        # x suppose to be N X 64 X 4 X ipm_w/8, need to be reshaped to N X 256 X ipm_w/8 X 1
        dim_rt_layers = []
        dim_rt_layers += make_one_layer(
            256, 128, kernel_size=(5, 1), padding=(2, 0), batch_norm=batch_norm
        )
        dim_rt_layers += [
            nn.Conv2d(128, self.num_lane_type * self.anchor_dim, kernel_size=(5, 1), padding=(2, 0))
        ]
        self.dim_rt = nn.Sequential(*dim_rt_layers)

    def forward(self, x):
        x = self.features(x)
        # x suppose to be N X 64 X 4 X ipm_w/8, reshape to N X 256 X ipm_w/8 X 1
        sizes = x.shape
        x = x.reshape(sizes[0], sizes[1] * sizes[2], sizes[3], 1)
        x = self.dim_rt(x)
        x = x.squeeze(-1).transpose(1, 2)
        # apply sigmoid to the probability terms to make it in (0, 1)
        for i in range(self.num_lane_type):
            x[:, :, (i + 1) * self.anchor_dim - 1] = torch.sigmoid(
                x[:, :, (i + 1) * self.anchor_dim - 1]
            )
        return x


# The 3D-lanenet composed of image encode, top view pathway, and lane predication head
class Net(nn.Module):
    def __init__(self, args, debug=False):
        super().__init__()

        self.no_cuda = args.no_cuda
        self.debug = debug
        self.pred_cam = args.pred_cam
        self.batch_size = args.batch_size
        if args.no_centerline:
            self.num_lane_type = 1
        else:
            self.num_lane_type = 3

        if args.no_3d:
            self.anchor_dim = args.num_y_steps + 1
        else:
            self.anchor_dim = 2 * args.num_y_steps + 1

        # define required transformation matrices
        # define homographic transformation between image and ipm
        org_img_size = np.array([args.org_h, args.org_w])
        resize_img_size = np.array([args.resize_h, args.resize_w])
        cam_pitch = np.pi / 180 * args.pitch

        self.cam_height = (
            torch.tensor(args.cam_height)
            .unsqueeze_(0)
            .expand([self.batch_size, 1])
            .type(torch.FloatTensor)
        )
        self.cam_pitch = (
            torch.tensor(cam_pitch)
            .unsqueeze_(0)
            .expand([self.batch_size, 1])
            .type(torch.FloatTensor)
        )
        self.cam_height_default = (
            torch.tensor(args.cam_height)
            .unsqueeze_(0)
            .expand(self.batch_size)
            .type(torch.FloatTensor)
        )
        self.cam_pitch_default = (
            torch.tensor(cam_pitch).unsqueeze_(0).expand(self.batch_size).type(torch.FloatTensor)
        )

        # image scale matrix
        self.S_im = torch.from_numpy(
            np.array([[args.resize_w, 0, 0], [0, args.resize_h, 0], [0, 0, 1]], dtype=np.float32)
        )
        self.S_im_inv = torch.from_numpy(
            np.array(
                [
                    [1 / np.float32(args.resize_w), 0, 0],
                    [0, 1 / np.float32(args.resize_h), 0],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
        )
        self.S_im_inv_batch = (
            self.S_im_inv.unsqueeze_(0).expand([self.batch_size, 3, 3]).type(torch.FloatTensor)
        )

        # image transform matrix
        H_c = homography_crop_resize(org_img_size, args.crop_y, resize_img_size)
        self.H_c = (
            torch.from_numpy(H_c)
            .unsqueeze_(0)
            .expand([self.batch_size, 3, 3])
            .type(torch.FloatTensor)
        )

        # camera intrinsic matrix
        self.K = (
            torch.from_numpy(args.K)
            .unsqueeze_(0)
            .expand([self.batch_size, 3, 3])
            .type(torch.FloatTensor)
        )

        # homograph ground to camera
        H_g2cam = np.array(
            [[1, 0, 0], [0, np.sin(-cam_pitch), args.cam_height], [0, np.cos(-cam_pitch), 0]]
        )
        self.H_g2cam = (
            torch.from_numpy(H_g2cam)
            .unsqueeze_(0)
            .expand([self.batch_size, 3, 3])
            .type(torch.FloatTensor)
        )

        # transform from ipm normalized coordinates to ground coordinates
        H_ipmnorm2g = homography_ipmnorm2g(args.top_view_region)
        self.H_ipmnorm2g = (
            torch.from_numpy(H_ipmnorm2g)
            .unsqueeze_(0)
            .expand([self.batch_size, 3, 3])
            .type(torch.FloatTensor)
        )

        # compute the transformation from ipm norm coords to image norm coords
        M_ipm2im = torch.bmm(self.H_g2cam, self.H_ipmnorm2g)
        M_ipm2im = torch.bmm(self.K, M_ipm2im)
        M_ipm2im = torch.bmm(self.H_c, M_ipm2im)
        M_ipm2im = torch.bmm(self.S_im_inv_batch, M_ipm2im)
        M_ipm2im = torch.div(
            M_ipm2im,
            M_ipm2im[:, 2, 2].reshape([self.batch_size, 1, 1]).expand([self.batch_size, 3, 3]),
        )
        self.M_inv = M_ipm2im

        if not self.no_cuda:
            self.M_inv = self.M_inv.cuda()
            self.S_im = self.S_im.cuda()
            self.S_im_inv = self.S_im_inv.cuda()
            self.S_im_inv_batch = self.S_im_inv_batch.cuda()
            self.H_c = self.H_c.cuda()
            self.K = self.K.cuda()
            self.H_g2cam = self.H_g2cam.cuda()
            self.H_ipmnorm2g = self.H_ipmnorm2g.cuda()
            self.cam_height_default = self.cam_height_default.cuda()
            self.cam_pitch_default = self.cam_pitch_default.cuda()

        # Define network
        self.im_encoder = VggEncoder(batch_norm=args.batch_norm)

        if self.pred_cam:
            self.road_plane_pred_head = RoadPlanePredHead(
                args.resize_h, args.resize_w, batch_norm=False
            )

        # the grid considers both src and dst grid normalized
        size_top1 = torch.Size([self.batch_size, args.ipm_h, args.ipm_w])
        self.project_layer1 = ProjectiveGridGenerator(size_top1, self.M_inv, args.no_cuda)
        size_top2 = torch.Size([self.batch_size, int(args.ipm_h / 2), int(args.ipm_w / 2)])
        self.project_layer2 = ProjectiveGridGenerator(size_top2, self.M_inv, args.no_cuda)
        size_top3 = torch.Size([self.batch_size, int(args.ipm_h / 4), int(args.ipm_w / 4)])
        self.project_layer3 = ProjectiveGridGenerator(size_top3, self.M_inv, args.no_cuda)
        size_top4 = torch.Size([self.batch_size, int(args.ipm_h / 8), int(args.ipm_w / 8)])
        self.project_layer4 = ProjectiveGridGenerator(size_top4, self.M_inv, args.no_cuda)

        self.dim_rt1 = nn.Sequential(
            *make_one_layer(256, 128, kernel_size=1, padding=0, batch_norm=args.batch_norm)
        )
        self.dim_rt2 = nn.Sequential(
            *make_one_layer(512, 256, kernel_size=1, padding=0, batch_norm=args.batch_norm)
        )
        self.dim_rt3 = nn.Sequential(
            *make_one_layer(512, 256, kernel_size=1, padding=0, batch_norm=args.batch_norm)
        )

        self.top_pathway = TopViewPathway(args.batch_norm)
        self.lane_out = LanePredictionHead(self.num_lane_type, self.anchor_dim, args.batch_norm)

    def forward(self, input):
        # compute image features from multiple layers
        x1, x2, x3, x4 = self.im_encoder(input)

        if self.pred_cam:
            pred_cam = self.road_plane_pred_head(x4)
            cam_height = self.cam_height_default + pred_cam[:, 0]
            cam_pitch = self.cam_pitch_default + pred_cam[:, 1]
            # compute projection matrix based on predicted camera height and pitch
            with torch.no_grad():
                self.H_g2cam[:, 1, 1] = torch.sin(-cam_pitch)
                self.H_g2cam[:, 2, 1] = torch.cos(-cam_pitch)
                self.H_g2cam[:, 1, 2] = cam_height
                M_ipm2im = torch.bmm(self.H_g2cam, self.H_ipmnorm2g)
                M_ipm2im = torch.bmm(self.K, M_ipm2im)
                M_ipm2im = torch.bmm(self.H_c, M_ipm2im)
                M_ipm2im = torch.bmm(self.S_im_inv_batch, M_ipm2im)
                M_ipm2im = torch.div(
                    M_ipm2im,
                    M_ipm2im[:, 2, 2]
                    .reshape([self.batch_size, 1, 1])
                    .expand([self.batch_size, 3, 3]),
                )
                self.M_inv = M_ipm2im
        else:
            cam_height = self.cam_height
            cam_pitch = self.cam_pitch

        # spatial transfer image features to IPM features
        grid1 = self.project_layer1(self.M_inv)
        grid2 = self.project_layer2(self.M_inv)
        grid3 = self.project_layer3(self.M_inv)
        grid4 = self.project_layer4(self.M_inv)

        x1_proj = F.grid_sample(x1, grid1, align_corners=False)
        x2_proj = F.grid_sample(x2, grid2, align_corners=False)
        x2_proj_out = x2_proj
        x2_proj = self.dim_rt1(x2_proj)
        x3_proj = F.grid_sample(x3, grid3, align_corners=False)
        x3_proj_out = x3_proj
        x3_proj = self.dim_rt2(x3_proj)
        x4_proj = F.grid_sample(x4, grid4, align_corners=False)
        x4_proj_out = x4_proj
        x4_proj = self.dim_rt3(x4_proj)

        # process features from top view
        x, top_2, top_3, top_4 = self.top_pathway(x1_proj, x2_proj, x3_proj, x4_proj)

        # convert top-view features to anchor output
        out = self.lane_out(x)

        if self.debug:
            return (
                out,
                cam_height,
                cam_pitch,
                x1,
                x2,
                x3,
                x4,
                x1_proj,
                x2_proj_out,
                x3_proj_out,
                x4_proj_out,
                top_2,
                top_3,
                top_4,
            )

        return out, cam_height, cam_pitch


def _build_default_args():
    """Tiny stand-in for tools.utils.define_args() + tusimple_config(args):
    same field names/semantics. `resize_h`/`resize_w` are shrunk from the
    paper's 360x480 to a smaller-but-VGG16-valid 128x128 for a fast trace;
    `ipm_h`/`ipm_w` are kept at the source's own tusimple_config defaults
    (208x128) because `LanePredictionHead`'s 7 no-y-padding convs need
    ipm_h/8 - 22 == 4, which only the paper's own IPM height satisfies at
    this padding/kernel schedule -- shrinking it collapses the feature map
    to <=0 height before the head's stride-1 valid convs finish."""
    args = SimpleNamespace()
    args.org_h = 128
    args.org_w = 128
    args.crop_y = 0
    args.no_centerline = True
    args.no_3d = True
    args.fix_cam = True
    args.pred_cam = False
    args.K = np.array([[100, 0, 64], [0, 100, 64], [0, 0, 1]], dtype=np.float32)
    args.cam_height = 1.6
    args.pitch = 9
    args.top_view_region = np.array([[-10, 103], [10, 103], [-10, 3], [10, 3]])
    args.anchor_y_steps = np.array([5, 10, 15, 20, 30, 40, 50, 60, 80, 100])
    args.num_y_steps = len(args.anchor_y_steps)
    args.pretrained = False
    args.batch_norm = True
    args.resize_h = 128
    args.resize_w = 128
    args.ipm_h = 208
    args.ipm_w = 128
    args.batch_size = 1
    args.no_cuda = True
    return args


def build_lanenet3d():
    args = _build_default_args()
    return Net(args)


def example_input_lanenet3d():
    args = _build_default_args()
    return torch.randn(args.batch_size, 3, args.resize_h, args.resize_w)


MENAGERIE_ZOO = "vendored-pytorch"

MENAGERIE_ENTRIES = [
    (
        "3D-LaneNet (end-to-end 3D multiple lane detection)",
        "build_lanenet3d",
        "example_input_lanenet3d",
        2019,
        MENAGERIE_ZOO,
    ),
]
