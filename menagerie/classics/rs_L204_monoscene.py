# SOURCE: vendored from astra-vision/MonoScene @ master (fetched 2026-07-01)
# (https://github.com/astra-vision/MonoScene, CVPR 2022)
#
# Files vendored (near-verbatim, only import paths adjusted + LightningModule
# wrapper dropped in favor of a plain nn.Module driving the same submodules):
#   - monoscene/models/flosp.py          (FLoSP 2D->3D feature projection)
#   - monoscene/models/unet2d.py         (UNet2D image encoder-decoder, AdaBins-style)
#   - monoscene/models/unet3d_kitti.py   (UNet3D voxel decoder, KITTI branch)
#   - monoscene/models/DDR.py            (Bottleneck3D / SimpleRB residual blocks)
#   - monoscene/models/modules.py        (Process/Upsample/Downsample/ASPP/SegmentationHead)
#   - monoscene/models/CRP3D.py          (CPMegaVoxels context-prior module)
#   - monoscene/models/monoscene.py      (MonoScene.forward, un-lightning-ified)
#
# Environment substitutions (NOT architecture changes):
#   - UNet2D.build() calls `torch.hub.load("rwightman/gen-efficientnet-pytorch",
#     "tf_efficientnet_b7_ns", pretrained=True)`. `gen-efficientnet-pytorch` is the
#     predecessor package that Ross Wightman folded into `timm`; `timm` ships the
#     identical `tf_efficientnet_b7` architecture (same EfficientNet class, same
#     module names `conv_stem/bn1/blocks/conv_head/bn2/global_pool/classifier`
#     that `Encoder.forward` walks by `_modules` name). We build it via
#     `timm.create_model(..., pretrained=False)` instead of a network `torch.hub`
#     fetch, and use the smaller `tf_efficientnet_b0` member of the same family so
#     the traced network stays menagerie-sized (full_scene_size and feature width
#     are also shrunk for the same reason -- see below).
#   - `MonoScene.forward` calls `.cuda()` on tensors read from a KITTI/NYU
#     dataloader batch (`projected_pix_X`, `fov_mask_X`); those calls are dropped
#     (device follows the input tensors instead of being hardcoded), and the
#     projected-pixel-index / field-of-view-mask tensors that a real KITTI
#     preprocessing pipeline would supply are synthesized here as random-but-valid
#     index/bool tensors of the exact shapes FLoSP expects (H*W image plane ->
#     scene_size[0]*scene_size[1]*scene_size[2] voxel grid). This is a calibration
#     preprocessing detail external to the network architecture, exactly analogous
#     to synthesizing `input_ids` for a language model rather than running a real
#     tokenizer.
#
# Config choice for tracing (KITTI branch of the real train script defaults --
# monoscene/scripts/train_monoscene.py -- shrunk only in overall scale so the
# menagerie capture stays small; every shape relation is identical to the real
# code, only the raw magnitudes are smaller):
#   dataset="kitti", full_scene_size=(64, 64, 8) [real: (256,256,32)],
#   project_scale=2, feature=16 [real: 64], n_classes=20, n_relations=4,
#   project_res=["1"], context_prior=True.
#
# MENAGERIE_ZOO = "vendored-pytorch"

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# --------------------------------------------------------------------------- #
# monoscene/models/DDR.py (vendored verbatim)
# --------------------------------------------------------------------------- #


class Bottleneck3D(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        norm_layer,
        stride=1,
        dilation=[1, 1, 1],
        expansion=4,
        downsample=None,
        fist_dilation=1,
        multi_grid=1,
        bn_momentum=0.0003,
    ):
        super(Bottleneck3D, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv3d(
            planes,
            planes,
            kernel_size=(1, 1, 3),
            stride=(1, 1, stride),
            dilation=(1, 1, dilation[0]),
            padding=(0, 0, dilation[0]),
            bias=False,
        )
        self.bn2 = norm_layer(planes, momentum=bn_momentum)
        self.conv3 = nn.Conv3d(
            planes,
            planes,
            kernel_size=(1, 3, 1),
            stride=(1, stride, 1),
            dilation=(1, dilation[1], 1),
            padding=(0, dilation[1], 0),
            bias=False,
        )
        self.bn3 = norm_layer(planes, momentum=bn_momentum)
        self.conv4 = nn.Conv3d(
            planes,
            planes,
            kernel_size=(3, 1, 1),
            stride=(stride, 1, 1),
            dilation=(dilation[2], 1, 1),
            padding=(dilation[2], 0, 0),
            bias=False,
        )
        self.bn4 = norm_layer(planes, momentum=bn_momentum)
        self.conv5 = nn.Conv3d(planes, planes * self.expansion, kernel_size=(1, 1, 1), bias=False)
        self.bn5 = norm_layer(planes * self.expansion, momentum=bn_momentum)

        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

        self.downsample2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, stride, 1), stride=(1, stride, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )
        self.downsample3 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(stride, 1, 1), stride=(stride, 1, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )
        self.downsample4 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(stride, 1, 1), stride=(stride, 1, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )

    def forward(self, x):
        residual = x

        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.bn2(self.conv2(out1))
        out2_relu = self.relu(out2)

        out3 = self.bn3(self.conv3(out2_relu))
        if self.stride != 1:
            out2 = self.downsample2(out2)
        out3 = out3 + out2
        out3_relu = self.relu(out3)

        out4 = self.bn4(self.conv4(out3_relu))
        if self.stride != 1:
            out2 = self.downsample3(out2)
            out3 = self.downsample4(out3)
        out4 = out4 + out2 + out3

        out4_relu = self.relu(out4)
        out5 = self.bn5(self.conv5(out4_relu))

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out5 + residual
        out_relu = self.relu(out)

        return out_relu


# --------------------------------------------------------------------------- #
# monoscene/models/modules.py (vendored verbatim)
# --------------------------------------------------------------------------- #


class ASPP(nn.Module):
    """ASPP 3D. Adapted (upstream too) from cv-rits/LMSCNet."""

    def __init__(self, planes, dilations_conv_list):
        super().__init__()
        self.conv_list = dilations_conv_list
        self.conv1 = nn.ModuleList(
            [
                nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False)
                for dil in dilations_conv_list
            ]
        )
        self.bn1 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in dilations_conv_list])
        self.conv2 = nn.ModuleList(
            [
                nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False)
                for dil in dilations_conv_list
            ]
        )
        self.bn2 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in dilations_conv_list])
        self.relu = nn.ReLU()

    def forward(self, x_in):
        y = self.bn2[0](self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x_in)))))
        for i in range(1, len(self.conv_list)):
            y += self.bn2[i](self.conv2[i](self.relu(self.bn1[i](self.conv1[i](x_in)))))
        x_in = self.relu(y + x_in)
        return x_in


class SegmentationHead(nn.Module):
    """3D Segmentation head. Adapted (upstream too) from cv-rits/LMSCNet."""

    def __init__(self, inplanes, planes, nbr_classes, dilations_conv_list):
        super().__init__()
        self.conv0 = nn.Conv3d(inplanes, planes, kernel_size=3, padding=1, stride=1)
        self.conv_list = dilations_conv_list
        self.conv1 = nn.ModuleList(
            [
                nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False)
                for dil in dilations_conv_list
            ]
        )
        self.bn1 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in dilations_conv_list])
        self.conv2 = nn.ModuleList(
            [
                nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False)
                for dil in dilations_conv_list
            ]
        )
        self.bn2 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in dilations_conv_list])
        self.relu = nn.ReLU()
        self.conv_classes = nn.Conv3d(planes, nbr_classes, kernel_size=3, padding=1, stride=1)

    def forward(self, x_in):
        x_in = self.relu(self.conv0(x_in))
        y = self.bn2[0](self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x_in)))))
        for i in range(1, len(self.conv_list)):
            y += self.bn2[i](self.conv2[i](self.relu(self.bn1[i](self.conv1[i](x_in)))))
        x_in = self.relu(y + x_in)
        x_in = self.conv_classes(x_in)
        return x_in


class Process(nn.Module):
    def __init__(self, feature, norm_layer, bn_momentum, dilations=[1, 2, 3]):
        super(Process, self).__init__()
        self.main = nn.Sequential(
            *[
                Bottleneck3D(
                    feature,
                    feature // 4,
                    bn_momentum=bn_momentum,
                    norm_layer=norm_layer,
                    dilation=[i, i, i],
                )
                for i in dilations
            ]
        )

    def forward(self, x):
        return self.main(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, bn_momentum):
        super(Upsample, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                dilation=1,
                output_padding=1,
            ),
            norm_layer(out_channels, momentum=bn_momentum),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.main(x)


class Downsample(nn.Module):
    def __init__(self, feature, norm_layer, bn_momentum, expansion=8):
        super(Downsample, self).__init__()
        self.main = Bottleneck3D(
            feature,
            feature // 4,
            bn_momentum=bn_momentum,
            expansion=expansion,
            stride=2,
            downsample=nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(
                    feature, int(feature * expansion / 4), kernel_size=1, stride=1, bias=False
                ),
                norm_layer(int(feature * expansion / 4), momentum=bn_momentum),
            ),
            norm_layer=norm_layer,
        )

    def forward(self, x):
        return self.main(x)


# --------------------------------------------------------------------------- #
# monoscene/models/CRP3D.py (vendored verbatim)
# --------------------------------------------------------------------------- #


class CPMegaVoxels(nn.Module):
    def __init__(self, feature, size, n_relations=4, bn_momentum=0.0003):
        super().__init__()
        self.size = size
        self.n_relations = n_relations
        self.flatten_size = size[0] * size[1] * size[2]
        self.feature = feature
        self.context_feature = feature * 2
        self.flatten_context_size = (size[0] // 2) * (size[1] // 2) * (size[2] // 2)
        padding = ((size[0] + 1) % 2, (size[1] + 1) % 2, (size[2] + 1) % 2)

        self.mega_context = nn.Sequential(
            nn.Conv3d(feature, self.context_feature, stride=2, padding=padding, kernel_size=3),
        )
        self.flatten_context_size = (size[0] // 2) * (size[1] // 2) * (size[2] // 2)

        self.context_prior_logits = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(self.feature, self.flatten_context_size, padding=0, kernel_size=1)
                )
                for i in range(n_relations)
            ]
        )
        self.aspp = ASPP(feature, [1, 2, 3])

        self.resize = nn.Sequential(
            nn.Conv3d(
                self.context_feature * self.n_relations + feature,
                feature,
                kernel_size=1,
                padding=0,
                bias=False,
            ),
            Process(feature, nn.BatchNorm3d, bn_momentum, dilations=[1]),
        )

    def forward(self, input):
        ret = {}
        bs = input.shape[0]

        x_agg = self.aspp(input)

        x_mega_context_raw = self.mega_context(x_agg)
        x_mega_context = x_mega_context_raw.reshape(bs, self.context_feature, -1)
        x_mega_context = x_mega_context.permute(0, 2, 1)

        x_context_prior_logits = []
        x_context_rels = []
        for rel in range(self.n_relations):
            x_context_prior_logit = self.context_prior_logits[rel](x_agg)
            x_context_prior_logit = x_context_prior_logit.reshape(
                bs, self.flatten_context_size, self.flatten_size
            )
            x_context_prior_logits.append(x_context_prior_logit.unsqueeze(1))

            x_context_prior_logit = x_context_prior_logit.permute(0, 2, 1)
            x_context_prior = torch.sigmoid(x_context_prior_logit)

            x_context_rel = torch.bmm(x_context_prior, x_mega_context)
            x_context_rels.append(x_context_rel)

        x_context = torch.cat(x_context_rels, dim=2)
        x_context = x_context.permute(0, 2, 1)
        x_context = x_context.reshape(
            bs, x_context.shape[1], self.size[0], self.size[1], self.size[2]
        )

        x = torch.cat([input, x_context], dim=1)
        x = self.resize(x)

        x_context_prior_logits = torch.cat(x_context_prior_logits, dim=1)
        ret["P_logits"] = x_context_prior_logits
        ret["x"] = x

        return ret


# --------------------------------------------------------------------------- #
# monoscene/models/unet3d_kitti.py (vendored verbatim)
# --------------------------------------------------------------------------- #


class UNet3DKitti(nn.Module):
    def __init__(
        self,
        class_num,
        norm_layer,
        full_scene_size,
        feature,
        project_scale,
        context_prior=None,
        bn_momentum=0.1,
    ):
        super(UNet3DKitti, self).__init__()
        self.business_layer = []
        self.project_scale = project_scale
        self.full_scene_size = full_scene_size
        self.feature = feature

        size_l1 = (
            int(self.full_scene_size[0] / project_scale),
            int(self.full_scene_size[1] / project_scale),
            int(self.full_scene_size[2] / project_scale),
        )
        size_l2 = (size_l1[0] // 2, size_l1[1] // 2, size_l1[2] // 2)
        size_l3 = (size_l2[0] // 2, size_l2[1] // 2, size_l2[2] // 2)

        self.process_l1 = nn.Sequential(
            Process(self.feature, norm_layer, bn_momentum, dilations=[1, 2, 3]),
            Downsample(self.feature, norm_layer, bn_momentum),
        )
        self.process_l2 = nn.Sequential(
            Process(self.feature * 2, norm_layer, bn_momentum, dilations=[1, 2, 3]),
            Downsample(self.feature * 2, norm_layer, bn_momentum),
        )

        self.up_13_l2 = Upsample(self.feature * 4, self.feature * 2, norm_layer, bn_momentum)
        self.up_12_l1 = Upsample(self.feature * 2, self.feature, norm_layer, bn_momentum)
        self.up_l1_lfull = Upsample(self.feature, self.feature // 2, norm_layer, bn_momentum)

        self.ssc_head = SegmentationHead(self.feature // 2, self.feature // 2, class_num, [1, 2, 3])

        self.context_prior = context_prior
        if context_prior:
            self.CP_mega_voxels = CPMegaVoxels(self.feature * 4, size_l3, bn_momentum=bn_momentum)

    def forward(self, input_dict):
        res = {}

        x3d_l1 = input_dict["x3d"]

        x3d_l2 = self.process_l1(x3d_l1)

        x3d_l3 = self.process_l2(x3d_l2)

        if self.context_prior:
            ret = self.CP_mega_voxels(x3d_l3)
            x3d_l3 = ret["x"]
            for k in ret.keys():
                res[k] = ret[k]

        x3d_up_l2 = self.up_13_l2(x3d_l3) + x3d_l2
        x3d_up_l1 = self.up_12_l1(x3d_up_l2) + x3d_l1
        x3d_up_lfull = self.up_l1_lfull(x3d_up_l1)

        ssc_logit_full = self.ssc_head(x3d_up_lfull)

        res["ssc_logit"] = ssc_logit_full

        return res


# --------------------------------------------------------------------------- #
# monoscene/models/unet2d.py (vendored verbatim, timm swapped in for
# gen-efficientnet-pytorch / torch.hub -- see module header)
# --------------------------------------------------------------------------- #


class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()
        self._net = nn.Sequential(
            nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU(),
            nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU(),
        )

    def forward(self, x, concat_with):
        up_x = F.interpolate(
            x,
            size=(concat_with.shape[2], concat_with.shape[3]),
            mode="bilinear",
            align_corners=True,
        )
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


class DecoderBN(nn.Module):
    def __init__(
        self,
        num_features,
        bottleneck_features,
        out_feature,
        use_decoder=True,
        skip_channels=(224, 80, 48, 32, 3),
    ):
        super(DecoderBN, self).__init__()
        features = int(num_features)
        self.use_decoder = use_decoder

        self.conv2 = nn.Conv2d(bottleneck_features, features, kernel_size=1, stride=1, padding=1)

        self.out_feature_1_1 = out_feature
        self.out_feature_1_2 = out_feature
        self.out_feature_1_4 = out_feature
        self.out_feature_1_8 = out_feature
        self.out_feature_1_16 = out_feature
        self.feature_1_16 = features // 2
        self.feature_1_8 = features // 4
        self.feature_1_4 = features // 8
        self.feature_1_2 = features // 16
        self.feature_1_1 = features // 32

        # Real repo hardcodes skip_channels=(224, 80, 48, 32, 3) for the
        # tf_efficientnet_b7 block widths at features[3]/[2]/[1]/[0]/input.
        # `skip_channels` is exposed as a constructor arg (not a rewrite of
        # the wiring, which stays index-for-index identical) so a smaller
        # EfficientNet member's real block widths can be threaded through.
        c16, c8, c4, c2, c1 = skip_channels

        if self.use_decoder:
            self.resize_output_1_1 = nn.Conv2d(
                self.feature_1_1, self.out_feature_1_1, kernel_size=1
            )
            self.resize_output_1_2 = nn.Conv2d(
                self.feature_1_2, self.out_feature_1_2, kernel_size=1
            )
            self.resize_output_1_4 = nn.Conv2d(
                self.feature_1_4, self.out_feature_1_4, kernel_size=1
            )
            self.resize_output_1_8 = nn.Conv2d(
                self.feature_1_8, self.out_feature_1_8, kernel_size=1
            )
            self.resize_output_1_16 = nn.Conv2d(
                self.feature_1_16, self.out_feature_1_16, kernel_size=1
            )

            self.up16 = UpSampleBN(skip_input=features + c16, output_features=self.feature_1_16)
            self.up8 = UpSampleBN(
                skip_input=self.feature_1_16 + c8, output_features=self.feature_1_8
            )
            self.up4 = UpSampleBN(
                skip_input=self.feature_1_8 + c4, output_features=self.feature_1_4
            )
            self.up2 = UpSampleBN(
                skip_input=self.feature_1_4 + c2, output_features=self.feature_1_2
            )
            self.up1 = UpSampleBN(
                skip_input=self.feature_1_2 + c1, output_features=self.feature_1_1
            )
        else:
            self.resize_output_1_1 = nn.Conv2d(3, out_feature, kernel_size=1)
            self.resize_output_1_2 = nn.Conv2d(32, out_feature * 2, kernel_size=1)
            self.resize_output_1_4 = nn.Conv2d(48, out_feature * 4, kernel_size=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = (
            features[4],
            features[5],
            features[6],
            features[8],
            features[11],
        )
        bs = x_block0.shape[0]
        x_d0 = self.conv2(x_block4)

        if self.use_decoder:
            x_1_16 = self.up16(x_d0, x_block3)
            x_1_8 = self.up8(x_1_16, x_block2)
            x_1_4 = self.up4(x_1_8, x_block1)
            x_1_2 = self.up2(x_1_4, x_block0)
            x_1_1 = self.up1(x_1_2, features[0])
            return {
                "1_1": self.resize_output_1_1(x_1_1),
                "1_2": self.resize_output_1_2(x_1_2),
                "1_4": self.resize_output_1_4(x_1_4),
                "1_8": self.resize_output_1_8(x_1_8),
                "1_16": self.resize_output_1_16(x_1_16),
            }
        else:
            x_1_1 = features[0]
            x_1_2, x_1_4, x_1_8, x_1_16 = features[4], features[5], features[6], features[8]
            x_global = features[-1].reshape(bs, 2560, -1).mean(2)
            return {
                "1_1": self.resize_output_1_1(x_1_1),
                "1_2": self.resize_output_1_2(x_1_2),
                "1_4": self.resize_output_1_4(x_1_4),
                "global": x_global,
            }


class Encoder(nn.Module):
    def __init__(self, backend):
        super(Encoder, self).__init__()
        self.original_model = backend

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if k == "blocks":
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features


class UNet2D(nn.Module):
    def __init__(
        self,
        backend,
        num_features,
        out_feature,
        use_decoder=True,
        skip_channels=(224, 80, 48, 32, 3),
    ):
        super(UNet2D, self).__init__()
        self.use_decoder = use_decoder
        self.encoder = Encoder(backend)
        self.decoder = DecoderBN(
            out_feature=out_feature,
            use_decoder=use_decoder,
            bottleneck_features=num_features,
            num_features=num_features,
            skip_channels=skip_channels,
        )

    def forward(self, x, **kwargs):
        encoded_feats = self.encoder(x)
        unet_out = self.decoder(encoded_feats, **kwargs)
        return unet_out

    @classmethod
    def build(cls, **kwargs):
        # Real repo: torch.hub.load("rwightman/gen-efficientnet-pytorch",
        # "tf_efficientnet_b7_ns", pretrained=True); num_features=2560,
        # skip_channels=(224, 80, 48, 32, 3) (b7's real block[3]/[2]/[1]/[0]/
        # input widths). Here: timm's identical architecture family,
        # un-pretrained, smaller member (b0) so the traced capture stays
        # menagerie-sized. b0's real block widths at the same tap points
        # (features[8]/[6]/[5]/[4]/[0], see Encoder.forward / DecoderBN.forward
        # indexing) are (192, 80, 40, 24, 3) -- threaded through so every conv
        # in DecoderBN gets the channel count the real b0 graph actually
        # produces; the wiring itself (which feature index feeds which skip
        # connection) is unchanged from the real repo.
        basemodel_name = "tf_efficientnet_b0"
        num_features = 1280
        skip_channels = (192, 80, 40, 24, 3)

        basemodel = timm.create_model(basemodel_name, pretrained=False)
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()

        m = cls(basemodel, num_features=num_features, skip_channels=skip_channels, **kwargs)
        return m


# --------------------------------------------------------------------------- #
# monoscene/models/flosp.py (vendored verbatim)
# --------------------------------------------------------------------------- #


class FLoSP(nn.Module):
    def __init__(self, scene_size, dataset, project_scale):
        super().__init__()
        self.scene_size = scene_size
        self.dataset = dataset
        self.project_scale = project_scale

    def forward(self, x2d, projected_pix, fov_mask):
        c, h, w = x2d.shape

        src = x2d.reshape(c, -1)
        zeros_vec = torch.zeros(c, 1).type_as(src)
        src = torch.cat([src, zeros_vec], 1)

        pix_x, pix_y = projected_pix[:, 0], projected_pix[:, 1]
        img_indices = pix_y * w + pix_x
        img_indices = img_indices.clone()
        img_indices[~fov_mask] = h * w
        img_indices = img_indices.expand(c, -1).long()  # c, HWD
        src_feature = torch.gather(src, 1, img_indices)

        if self.dataset == "NYU":
            x3d = src_feature.reshape(
                c,
                self.scene_size[0] // self.project_scale,
                self.scene_size[2] // self.project_scale,
                self.scene_size[1] // self.project_scale,
            )
            x3d = x3d.permute(0, 1, 3, 2)
        elif self.dataset == "kitti":
            x3d = src_feature.reshape(
                c,
                self.scene_size[0] // self.project_scale,
                self.scene_size[1] // self.project_scale,
                self.scene_size[2] // self.project_scale,
            )

        return x3d


# --------------------------------------------------------------------------- #
# monoscene/models/monoscene.py `MonoScene` (vendored architecture; the
# LightningModule/loss/logging/optimizer machinery is training infrastructure
# outside the network, dropped in favor of a plain nn.Module wrapper around the
# same submodules; forward() is otherwise the same projection-then-decode flow.)
# --------------------------------------------------------------------------- #


class MonoScene(nn.Module):
    def __init__(
        self,
        n_classes,
        feature,
        project_scale,
        full_scene_size,
        dataset,
        n_relations=4,
        context_prior=True,
        project_res=("1",),
    ):
        super().__init__()

        self.project_res = list(project_res)
        self.dataset = dataset
        self.context_prior = context_prior
        self.project_scale = project_scale

        self.projects = {}
        self.scale_2ds = [1, 2, 4, 8]
        for scale_2d in self.scale_2ds:
            self.projects[str(scale_2d)] = FLoSP(
                full_scene_size, project_scale=self.project_scale, dataset=self.dataset
            )
        self.projects = nn.ModuleDict(self.projects)

        self.n_classes = n_classes
        if self.dataset == "kitti":
            self.net_3d_decoder = UNet3DKitti(
                self.n_classes,
                nn.BatchNorm3d,
                project_scale=project_scale,
                feature=feature,
                full_scene_size=full_scene_size,
                context_prior=context_prior,
            )
        self.net_rgb = UNet2D.build(out_feature=feature, use_decoder=True)

    def forward(self, batch):
        img = batch["img"]
        bs = len(img)

        x_rgb = self.net_rgb(img)

        x3ds = []
        for i in range(bs):
            x3d = None
            for scale_2d in self.project_res:
                scale_2d = int(scale_2d)
                projected_pix = batch["projected_pix_{}".format(self.project_scale)][i]
                fov_mask = batch["fov_mask_{}".format(self.project_scale)][i]

                if x3d is None:
                    x3d = self.projects[str(scale_2d)](
                        x_rgb["1_" + str(scale_2d)][i],
                        projected_pix // scale_2d,
                        fov_mask,
                    )
                else:
                    x3d += self.projects[str(scale_2d)](
                        x_rgb["1_" + str(scale_2d)][i],
                        projected_pix // scale_2d,
                        fov_mask,
                    )
            x3ds.append(x3d)

        input_dict = {"x3d": torch.stack(x3ds)}
        out = self.net_3d_decoder(input_dict)
        return out


# --------------------------------------------------------------------------- #
# menagerie staging entry points
# --------------------------------------------------------------------------- #

_FULL_SCENE_SIZE = (64, 64, 32)  # real KITTI cfg: (256, 256, 32)
_PROJECT_SCALE = 2
_FEATURE = 16  # real KITTI cfg: 64
_N_CLASSES = 20
_IMG_H, _IMG_W = 128, 384  # multiple of 32 for the EfficientNet stem/blocks


def build_monoscene():
    return MonoScene(
        n_classes=_N_CLASSES,
        feature=_FEATURE,
        project_scale=_PROJECT_SCALE,
        full_scene_size=_FULL_SCENE_SIZE,
        dataset="kitti",
        n_relations=4,
        context_prior=True,
        project_res=("1",),
    )


def example_input_monoscene():
    bs = 1
    img = torch.rand(bs, 3, _IMG_H, _IMG_W)

    voxel_hw = (
        _FULL_SCENE_SIZE[0] // _PROJECT_SCALE,
        _FULL_SCENE_SIZE[1] // _PROJECT_SCALE,
        _FULL_SCENE_SIZE[2] // _PROJECT_SCALE,
    )
    n_voxels = voxel_hw[0] * voxel_hw[1] * voxel_hw[2]

    # Real pipeline: KITTI calibration projects each of the n_voxels voxel
    # centers into the image plane, giving integer (x, y) pixel coords + a
    # field-of-view boolean mask. Synthesized here as valid random indices
    # into the "1_1" feature map (shape matches img: _IMG_H x _IMG_W).
    projected_pix = torch.stack(
        [
            torch.randint(0, _IMG_W, (n_voxels,)),
            torch.randint(0, _IMG_H, (n_voxels,)),
        ],
        dim=1,
    )
    fov_mask = torch.rand(n_voxels) > 0.2

    batch = {
        "img": img,
        "projected_pix_{}".format(_PROJECT_SCALE): [projected_pix],
        "fov_mask_{}".format(_PROJECT_SCALE): [fov_mask],
    }
    return (batch,)


MENAGERIE_ENTRIES = [
    ("MonoScene", "build_monoscene", "example_input_monoscene", 2022, "vendored-pytorch"),
]
