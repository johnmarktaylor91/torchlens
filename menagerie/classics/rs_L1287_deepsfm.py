# SOURCE: vendored from weixk2015/DeepSFM @ 393ad8f39c5a305f3b351af497dc510697ee931e
# models/submodule.py + models/PSNet.py + inverse_warp.py (real classes/functions,
# unmodified except: (1) added an explicit `import math` -- the upstream file uses
# `math.sqrt` in PSNet.__init__ but never imports `math` itself, relying on transitive
# leakage from a wildcard import chain that doesn't reliably re-export it; (2) the
# module-level global `pixel_coords` cache in inverse_warp.py is kept as a plain
# module attribute. Architecture, control flow, and all `.cuda()` placements are
# preserved verbatim -- this repo requires a CUDA device to construct (buffers built
# with Variable(...).cuda() at __init__ time), so this staging module needs a GPU.
"""DeepSFM (ECCV 2020): PSNet -- structure-from-motion depth estimation via a
plane-sweep cost volume + 3D-CNN cost-volume regularization, refined by deep
bundle adjustment (this module covers the PSNet depth branch)."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# --------------------------------------------------------------------------
# inverse_warp.py (vendored verbatim)
# --------------------------------------------------------------------------

pixel_coords = None


def set_id_grid(depth):
    global pixel_coords
    b, h, w = depth.size()
    i_range = Variable(torch.arange(0, h).view(1, h, 1).expand(1, h, w)).type_as(depth)  # [1, H, W]
    j_range = Variable(torch.arange(0, w).view(1, 1, w).expand(1, h, w)).type_as(depth)  # [1, H, W]
    ones = Variable(torch.ones(1, h, w)).type_as(depth)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]


def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i, size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert all(condition), "wrong size for {}, expected {}, got  {}".format(
        input_name, "x".join(expected), list(input.size())
    )


def pixel2cam(depth, intrinsics_inv):
    global pixel_coords
    """Transform coordinates in the pixel frame to the camera frame."""
    b, h, w = depth.size()
    if (pixel_coords is None) or pixel_coords.size(2) < h:
        set_id_grid(depth)
    current_pixel_coords = (
        pixel_coords[:, :, :h, :w].expand(b, 3, h, w).contiguous().view(b, 3, -1).cuda()
    )  # [B, 3, H*W]
    cam_coords = intrinsics_inv.bmm(current_pixel_coords).view(b, 3, h, w)
    return cam_coords * depth.unsqueeze(1)


def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode, rounded=False):
    """Transform coordinates in the camera frame to the pixel frame."""
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.view(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot.bmm(cam_coords_flat)
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)
    if rounded:
        X_norm = torch.round(2 * (X / Z)) / (w - 1) - 1
        Y_norm = torch.round(2 * (Y / Z)) / (h - 1) - 1
    else:
        X_norm = 2 * (X / Z) / (w - 1) - 1
        Y_norm = 2 * (Y / Z) / (h - 1) - 1

    if padding_mode == "zeros":
        X_mask = ((X_norm > 1) + (X_norm < -1)).detach()
        X_norm[X_mask] = 2
        Y_mask = ((Y_norm > 1) + (Y_norm < -1)).detach()
        Y_norm[Y_mask] = 2

    pixel_coords_out = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    return pixel_coords_out.view(b, h, w, 2)


def cam2depth(cam_coords, proj_c2p_rot, proj_c2p_tr):
    """Transform coordinates in the camera frame to the pixel frame (depth only)."""
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.view(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot.bmm(cam_coords_flat)
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    z = pcoords[:, 2, :].contiguous()
    return z.view(b, h, w)


def depth_warp(fdepth, depth, pose, intrinsics, intrinsics_inv, padding_mode="zeros"):
    """Warp a target depth to the source image plane."""
    check_sizes(depth, "depth", "BHW")
    check_sizes(pose, "pose", "B34")
    check_sizes(intrinsics, "intrinsics", "B33")
    check_sizes(intrinsics_inv, "intrinsics", "B33")
    assert intrinsics_inv.size() == intrinsics.size()

    batch_size, feat_height, feat_width = depth.size()

    cam_coords = pixel2cam(depth, intrinsics_inv)
    pose_mat = pose
    pose_mat = pose_mat.cuda()

    proj_cam_to_src_pixel = intrinsics.bmm(pose_mat)  # [B, 3, 4]
    src_pixel_coords = cam2pixel(
        cam_coords,
        proj_cam_to_src_pixel[:, :, :3],
        proj_cam_to_src_pixel[:, :, -1:],
        padding_mode,
        rounded=True,
    )  # [B,H,W,2]
    projected_depth = cam2depth(cam_coords, pose_mat[:, :, :3], pose_mat[:, :, -1:])
    fdepth_expand = fdepth.unsqueeze(1)
    fdepth_expand = torch.nn.functional.upsample(
        fdepth_expand, [feat_height, feat_width], mode="bilinear"
    )

    warped_depth = torch.nn.functional.grid_sample(
        fdepth_expand, src_pixel_coords, mode="nearest", padding_mode=padding_mode
    )
    warped_depth = warped_depth.view(batch_size, feat_height, feat_width)
    projected_depth = projected_depth.clamp(min=1e-3, max=float(torch.max(warped_depth) + 10))
    return projected_depth, warped_depth


def inverse_warp(feat, depth, pose, intrinsics, intrinsics_inv, padding_mode="zeros"):
    """Inverse warp a source feature map to the target image plane."""
    check_sizes(depth, "depth", "BHW")
    check_sizes(pose, "pose", "B34")
    check_sizes(intrinsics, "intrinsics", "B33")
    check_sizes(intrinsics_inv, "intrinsics", "B33")

    assert intrinsics_inv.size() == intrinsics.size()

    batch_size, _, feat_height, feat_width = feat.size()

    cam_coords = pixel2cam(depth, intrinsics_inv)

    pose_mat = pose
    pose_mat = pose_mat.cuda()

    proj_cam_to_src_pixel = intrinsics.bmm(pose_mat)  # [B, 3, 4]

    src_pixel_coords = cam2pixel(
        cam_coords,
        proj_cam_to_src_pixel[:, :, :3],
        proj_cam_to_src_pixel[:, :, -1:],
        padding_mode,
        rounded=True,
    )  # [B,H,W,2]

    projected_feat = torch.nn.functional.grid_sample(
        feat, src_pixel_coords, mode="nearest", padding_mode=padding_mode
    )

    return projected_feat


# --------------------------------------------------------------------------
# models/submodule.py (vendored verbatim)
# --------------------------------------------------------------------------


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation if dilation > 1 else pad,
            dilation=dilation,
            bias=False,
        ),
        nn.BatchNorm2d(out_planes),
    )


def convbn_3d_o(in_planes, out_planes, kernel_size, stride, pad):
    return nn.Sequential(
        nn.Conv3d(
            in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False
        ),
        nn.BatchNorm3d(out_planes),
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(
            convbn(inplanes, planes, 3, stride, pad, dilation), nn.ReLU(inplace=True)
        )

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = Variable(
            torch.Tensor(np.reshape(np.array(range(maxdisp)), [1, maxdisp, 1, 1])).cuda(),
            requires_grad=False,
        )

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        out = torch.sum(x * disp, 1)
        return out


class feature_extraction(nn.Module):
    def __init__(self, pool=False):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(
            convbn(3, 32, 3, 2, 1, 1),
            nn.ReLU(inplace=True),
            convbn(32, 32, 3, 1, 1, 1),
            nn.ReLU(inplace=True),
            convbn(32, 32, 3, 1, 1, 1),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

        self.branch1 = nn.Sequential(
            nn.AvgPool2d((32, 32), stride=(32, 32)),
            convbn(128, 32, 1, 1, 0, 1),
            nn.ReLU(inplace=True),
        )

        self.branch2 = nn.Sequential(
            nn.AvgPool2d((16, 16), stride=(16, 16)),
            convbn(128, 32, 1, 1, 0, 1),
            nn.ReLU(inplace=True),
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d((8, 8), stride=(8, 8)), convbn(128, 32, 1, 1, 0, 1), nn.ReLU(inplace=True)
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d((4, 4), stride=(4, 4)), convbn(128, 32, 1, 1, 0, 1), nn.ReLU(inplace=True)
        )
        if pool:
            self.lastconv = nn.Sequential(
                convbn(320, 128, 3, 1, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=False),
                nn.AvgPool2d((2, 2), stride=(2, 2)),
            )
        else:
            self.lastconv = nn.Sequential(
                convbn(320, 128, 3, 1, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=False),
            )

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.firstconv(x)
        output = self.layer1(output)
        output_raw = self.layer2(output)
        output = self.layer3(output_raw)
        output_skip = self.layer4(output)

        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.upsample(
            output_branch1, (output_skip.size()[2], output_skip.size()[3]), mode="bilinear"
        )

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.upsample(
            output_branch2, (output_skip.size()[2], output_skip.size()[3]), mode="bilinear"
        )

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.upsample(
            output_branch3, (output_skip.size()[2], output_skip.size()[3]), mode="bilinear"
        )

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.upsample(
            output_branch4, (output_skip.size()[2], output_skip.size()[3]), mode="bilinear"
        )

        output_feature = torch.cat(
            (
                output_raw,
                output_skip,
                output_branch4,
                output_branch3,
                output_branch2,
                output_branch1,
            ),
            1,
        )
        output_feature = self.lastconv(output_feature)

        return output_feature


# --------------------------------------------------------------------------
# models/PSNet.py (vendored verbatim)
# --------------------------------------------------------------------------


def convtext(in_planes, out_planes, kernel_size=3, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=((kernel_size - 1) * dilation) // 2,
            bias=False,
        ),
        nn.LeakyReLU(0.1, inplace=True),
    )


class PSNet(nn.Module):
    def __init__(self, nlabel, mindepth, add_geo_cost=False, depth_augment=False):
        super(PSNet, self).__init__()
        self.nlabel = nlabel
        self.mindepth = mindepth
        self.add_geo = add_geo_cost

        self.depth_augment = depth_augment
        self.feature_extraction = feature_extraction()

        self.convs = nn.Sequential(
            convtext(33, 128, 3, 1, 1),
            convtext(128, 128, 3, 1, 2),
            convtext(128, 128, 3, 1, 4),
            convtext(128, 96, 3, 1, 8),
            convtext(96, 64, 3, 1, 16),
            convtext(64, 32, 3, 1, 1),
            convtext(32, 1, 3, 1, 1),
        )
        if add_geo_cost:
            self.n_dres0 = nn.Sequential(
                convbn_3d_o(66, 32, 3, 1, 1),
                nn.ReLU(inplace=True),
                convbn_3d_o(32, 32, 3, 1, 1),
                nn.ReLU(inplace=True),
            )
        else:
            self.dres0 = nn.Sequential(
                convbn_3d_o(64, 32, 3, 1, 1),
                nn.ReLU(inplace=True),
                convbn_3d_o(32, 32, 3, 1, 1),
                nn.ReLU(inplace=True),
            )

        self.dres1 = nn.Sequential(
            convbn_3d_o(32, 32, 3, 1, 1), nn.ReLU(inplace=True), convbn_3d_o(32, 32, 3, 1, 1)
        )

        self.dres2 = nn.Sequential(
            convbn_3d_o(32, 32, 3, 1, 1), nn.ReLU(inplace=True), convbn_3d_o(32, 32, 3, 1, 1)
        )

        self.dres3 = nn.Sequential(
            convbn_3d_o(32, 32, 3, 1, 1), nn.ReLU(inplace=True), convbn_3d_o(32, 32, 3, 1, 1)
        )

        self.dres4 = nn.Sequential(
            convbn_3d_o(32, 32, 3, 1, 1), nn.ReLU(inplace=True), convbn_3d_o(32, 32, 3, 1, 1)
        )

        self.classify = nn.Sequential(
            convbn_3d_o(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(
        self, ref, targets, pose, intrinsics, intrinsics_inv, targets_depth=None, mindepth=0.5
    ):
        intrinsics4 = intrinsics.clone()
        intrinsics_inv4 = intrinsics_inv.clone()
        intrinsics4[:, :2, :] = intrinsics4[:, :2, :] / 4
        intrinsics_inv4[:, :2, :2] = intrinsics_inv4[:, :2, :2] * 4

        refimg_fea = self.feature_extraction(ref)

        disp2depth = (
            Variable(torch.ones(refimg_fea.size(0), refimg_fea.size(2), refimg_fea.size(3))).cuda()
            * self.mindepth
            * self.nlabel
        )
        for j, target in enumerate(targets):
            if self.add_geo:
                cost = Variable(
                    torch.FloatTensor(
                        refimg_fea.size()[0],
                        refimg_fea.size()[1] * 2 + 2,
                        self.nlabel,
                        refimg_fea.size()[2],
                        refimg_fea.size()[3],
                    ).zero_()
                ).cuda()
            else:
                cost = Variable(
                    torch.FloatTensor(
                        refimg_fea.size()[0],
                        refimg_fea.size()[1] * 2,
                        self.nlabel,
                        refimg_fea.size()[2],
                        refimg_fea.size()[3],
                    ).zero_()
                ).cuda()
            targetimg_fea = self.feature_extraction(target)
            if self.depth_augment:
                noise = (
                    Variable(
                        torch.from_numpy(
                            np.random.normal(loc=0.0, scale=mindepth / 10, size=(1, 240, 320))
                        )
                    )
                    .float()
                    .cuda()
                )
            else:
                noise = 0
            for i in range(self.nlabel):
                depth = torch.div(disp2depth, i + 1e-16)
                targetimg_fea_t = inverse_warp(
                    targetimg_fea, depth, pose[:, j], intrinsics4, intrinsics_inv4
                )
                if self.add_geo:
                    assert targets_depth is not None

                    projected_depth, warped_depth = depth_warp(
                        targets_depth[j] + noise, depth, pose[:, j], intrinsics4, intrinsics_inv4
                    )
                    cost[:, -2, i, :, :] = projected_depth
                    cost[:, -1, i, :, :] = warped_depth
                cost[:, : refimg_fea.size()[1], i, :, :] = refimg_fea
                cost[:, refimg_fea.size()[1] : refimg_fea.size()[1] * 2, i, :, :] = targetimg_fea_t

            cost = cost.contiguous()
            if self.add_geo:
                cost0 = self.n_dres0(cost)
            else:
                cost0 = self.dres0(cost)
            cost0 = self.dres1(cost0) + cost0
            cost0 = self.dres2(cost0) + cost0
            cost0 = self.dres3(cost0) + cost0
            cost0 = self.dres4(cost0) + cost0
            cost0 = self.classify(cost0)

            if j == 0:
                costs = cost0
            else:
                costs = costs + cost0

        costs = costs / len(targets)

        costss = Variable(
            torch.FloatTensor(
                refimg_fea.size()[0], 1, self.nlabel, refimg_fea.size()[2], refimg_fea.size()[3]
            ).zero_()
        ).cuda()
        for i in range(self.nlabel):
            costt = costs[:, :, i, :, :]
            costss[:, :, i, :, :] = self.convs(torch.cat([refimg_fea, costt], 1)) + costt

        costs = F.upsample(costs, [self.nlabel, ref.size()[2], ref.size()[3]], mode="trilinear")
        costs = torch.squeeze(costs, 1)
        pred0 = F.softmax(costs, dim=1)
        pred0 = disparityregression(self.nlabel)(pred0)
        depth0 = self.mindepth * self.nlabel / (pred0.unsqueeze(1) + 1e-16)

        costss = F.upsample(costss, [self.nlabel, ref.size()[2], ref.size()[3]], mode="trilinear")
        costss = torch.squeeze(costss, 1)

        pred = F.softmax(costss, dim=1)
        pred = disparityregression(self.nlabel)(pred)
        depth = self.mindepth * self.nlabel / (pred.unsqueeze(1) + 1e-16)

        if self.training:
            return depth0, depth
        else:
            return depth


def build_deepsfm_psnet():
    # Small nlabel keeps the plane-sweep cost volume + 3D-CNN tractable to trace.
    model = PSNet(nlabel=8, mindepth=0.5, add_geo_cost=False, depth_augment=False)
    model.eval()  # eval branch returns a single `depth` tensor (simpler traced output)
    return model.cuda()


def example_input_deepsfm_psnet():
    # Tiny 2-source-view plane-sweep setup; real repo default image size is 256x320,
    # shrunk here to keep the O(nlabel) feature_extraction + 3D-conv loop fast. Must
    # stay >=128x128: feature_extraction downsamples by 4x (firstconv+layer2, both
    # stride 2) then branch1 does AvgPool2d(kernel_size=32) on that feature map.
    h, w = 128, 128
    ref = torch.randn(1, 3, h, w).cuda()
    targets = [torch.randn(1, 3, h, w).cuda()]
    pose = (
        torch.eye(3, 4).unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 1).cuda()
    )  # [B, num_targets, 3, 4]
    intrinsics = torch.eye(3).unsqueeze(0).cuda()
    intrinsics[:, 0, 0] = w
    intrinsics[:, 1, 1] = h
    intrinsics[:, 0, 2] = w / 2
    intrinsics[:, 1, 2] = h / 2
    intrinsics_inv = torch.inverse(intrinsics)
    return (ref, targets, pose, intrinsics, intrinsics_inv)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "DeepSFM_PSNet",
        "build_deepsfm_psnet",
        "example_input_deepsfm_psnet",
        2020,
        "vendored-pytorch",
    ),
]
