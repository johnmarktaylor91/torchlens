# SOURCE: vendored from kiharalab/CryoREAD @ 5c49c2f6083d9a70ac47014a64daaed01a9f1328
# https://raw.githubusercontent.com/kiharalab/CryoREAD/main/model/Cascade_Unet.py
# https://raw.githubusercontent.com/kiharalab/CryoREAD/main/model/Unet_Layers.py
# https://raw.githubusercontent.com/kiharalab/CryoREAD/main/model/init_weights.py
#
# Terashi, Kagaya, Kihara. "CryoREAD: de novo structure modeling for nucleic acids in
# cryo-EM maps using deep learning" (Nature Methods, 2023). `Cascade_Unet` (constructed in
# `main.py` for RNA/DNA backbone-atom segmentation of cryo-EM density maps) chains two 3D
# UNet-family CNNs: `Small_UNet_3Plus_DeepSup` is a small UNet3+ (full-scale skip
# aggregation with deep supervision, `hd2`/`hd1` decoder stages fusing all encoder levels
# via Conv3d+BatchNorm3d+ReLU) that both produces its own multi-scale deep-sup outputs AND
# passes its 3 skip-connection feature maps (`hidden_input`) into `Base_Unet`, a second
# UNet3+-style network that concatenates those hidden features at every encoder stage
# before its own full-scale decoder. Both submodules share `unetConv3d`/`init_weights`
# from `Unet_Layers.py`. All three files are copied verbatim; no architecture lines were
# changed (only trilinear `Upsample` calls, which warn on older torch about missing
# `align_corners`, are left exactly as in the real code).

import torch
import torch.nn as nn
from torch.nn import init


# --- init_weights.py (verbatim) ---


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find("Linear") != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find("BatchNorm") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("Linear") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("BatchNorm") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find("Linear") != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find("BatchNorm") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type="normal"):
    if init_type == "normal":
        net.apply(weights_init_normal)
    elif init_type == "xavier":
        net.apply(weights_init_xavier)
    elif init_type == "kaiming":
        net.apply(weights_init_kaiming)
    elif init_type == "orthogonal":
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError("initialization method [%s] is not implemented" % init_type)


# --- Unet_Layers.py (verbatim) ---


class unetConv3d(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv3d, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(
                    nn.Conv3d(in_size, out_size, ks, s, p),
                    nn.BatchNorm3d(out_size),
                    nn.ReLU(inplace=True),
                )
                setattr(self, "conv%d" % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(
                    nn.Conv3d(in_size, out_size, ks, s, p),
                    nn.ReLU(inplace=True),
                )
                setattr(self, "conv%d" % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type="kaiming")

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, "conv%d" % i)
            x = conv(x)
        return x


# --- Cascade_Unet.py (verbatim) ---


class Small_UNet_3Plus_DeepSup(nn.Module):
    def __init__(
        self, in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True
    ):
        super(Small_UNet_3Plus_DeepSup, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        # small unet
        filters = [64, 128, 256]
        ## -------------Encoder--------------
        self.conv1 = unetConv3d(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2)

        self.conv2 = unetConv3d(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2)

        self.conv3 = unetConv3d(filters[1], filters[2], self.is_batchnorm)

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 3
        self.UpChannels = self.CatChannels * self.CatBlocks

        # stage 2d
        self.h1_PT_hd2 = nn.MaxPool3d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv3d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm3d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        self.h2_Cat_hd2_conv = nn.Conv3d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm3d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode="trilinear")  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv3d(filters[2], self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        self.conv2d_1 = nn.Conv3d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm3d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        # stage 1
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv3d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm3d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode="trilinear")  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv3d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode="trilinear")  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv3d(filters[2], self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv3d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm3d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)
        # final process

        self.upscore3 = nn.Upsample(scale_factor=4, mode="trilinear")
        self.upscore2 = nn.Upsample(scale_factor=2, mode="trilinear")

        # DeepSup
        self.outconv1 = nn.Conv3d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv2 = nn.Conv3d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv3 = nn.Conv3d(filters[2], n_classes, 3, padding=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type="kaiming")
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type="kaiming")

    def forward(self, inputs):
        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->320*320*64

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128

        h3 = self.maxpool2(h2)
        hd3 = self.conv3(h3)  # h3->80*80*256

        ## -------------Decoder-------------
        # stage 2:
        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(
            self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3)))
        )
        hd2 = self.relu2d_1(
            self.bn2d_1(self.conv2d_1(torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2), 1)))
        )  # hd4->40*40*UpChannels

        # stage 1:
        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(
            self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2)))
        )
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(
            self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3)))
        )
        hd1 = self.relu1d_1(
            self.bn1d_1(self.conv1d_1(torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1), 1)))
        )  # hd1->320*320*UpChannels

        d3 = self.outconv3(hd3)
        d3 = self.upscore3(d3)  # 64->256

        d2 = self.outconv2(hd2)
        d2 = self.upscore2(d2)  # 128->256

        d1 = self.outconv1(hd1)  # 256

        return [d1, d2, d3], [hd1, hd2, hd3]


class Base_Unet(nn.Module):
    def __init__(
        self, in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True
    ):
        super(Base_Unet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        # small unet
        filters = [64, 128, 256]
        self.CatChannels = filters[0]
        self.CatBlocks = 3
        self.UpChannels = self.CatChannels * self.CatBlocks
        ## -------------Encoder--------------
        self.conv1 = unetConv3d(self.in_channels + self.UpChannels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2)

        self.conv2 = unetConv3d(filters[0] + self.UpChannels, filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2)

        self.conv3 = unetConv3d(filters[1] + filters[2], filters[2], self.is_batchnorm)

        ## -------------Decoder--------------
        # stage 2d
        self.h1_PT_hd2 = nn.MaxPool3d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv3d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm3d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        self.h2_Cat_hd2_conv = nn.Conv3d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm3d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode="trilinear")  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv3d(filters[2], self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        self.conv2d_1 = nn.Conv3d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm3d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        # stage 1
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv3d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm3d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode="trilinear")  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv3d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode="trilinear")  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv3d(filters[2], self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv3d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm3d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)
        # final process

        self.upscore3 = nn.Upsample(scale_factor=4, mode="trilinear")
        self.upscore2 = nn.Upsample(scale_factor=2, mode="trilinear")

        # DeepSup
        self.outconv1 = nn.Conv3d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv2 = nn.Conv3d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv3 = nn.Conv3d(filters[2], n_classes, 3, padding=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type="kaiming")
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type="kaiming")

    def forward(self, inputs, hidden_inputs):
        ## -------------Encoder-------------
        inputs1 = torch.cat([inputs, hidden_inputs[0]], dim=1)
        h1 = self.conv1(inputs1)  # h1->320*320*64

        h2 = self.maxpool1(h1)

        h2_input = torch.cat([h2, hidden_inputs[1]], dim=1)
        h2 = self.conv2(h2_input)  # h2->160*160*128

        h3 = self.maxpool2(h2)

        h3_input = torch.cat([h3, hidden_inputs[2]], dim=1)
        hd3 = self.conv3(h3_input)  # h3->80*80*256

        ## -------------Decoder-------------
        # stage 2:
        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(
            self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3)))
        )
        hd2 = self.relu2d_1(
            self.bn2d_1(self.conv2d_1(torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2), 1)))
        )  # hd4->40*40*UpChannels

        # stage 1:
        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(
            self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2)))
        )
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(
            self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3)))
        )
        hd1 = self.relu1d_1(
            self.bn1d_1(self.conv1d_1(torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1), 1)))
        )  # hd1->320*320*UpChannels

        d3 = self.outconv3(hd3)
        d3 = self.upscore3(d3)  # 64->256

        d2 = self.outconv2(hd2)
        d2 = self.upscore2(d2)  # 128->256

        d1 = self.outconv1(hd1)  # 256

        return [d1, d2, d3]  # channel size: up_channels,up_channels, filters[2]


class Cascade_Unet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        n_classes1=4,
        n_classes2=4,
        feature_scale=4,
        is_deconv=True,
        is_batchnorm=True,
    ):
        super(Cascade_Unet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        # channel sizes of different levels
        filters = [64, 128, 256]  # noqa: F841 (dead in the real Cascade_Unet.__init__ too; kept verbatim)
        self.chain_net = Small_UNet_3Plus_DeepSup(
            in_channels=in_channels,
            n_classes=n_classes1,
            feature_scale=feature_scale,
            is_deconv=is_deconv,
            is_batchnorm=is_batchnorm,
        )
        self.base_net = Base_Unet(
            in_channels=in_channels,
            n_classes=n_classes2,
            feature_scale=feature_scale,
            is_deconv=is_deconv,
            is_batchnorm=is_batchnorm,
        )

    def forward(self, inputs):
        chain_output, hidden_input = self.chain_net(inputs)
        base_output = self.base_net(inputs, hidden_input)
        return chain_output, base_output


# --- staging harness (tiny sizes; not part of the real repo) ---


def build_cryoread_cascade_unet() -> nn.Module:
    # in_channels=1 mirrors the real single-channel cryo-EM density map input
    # (main.py constructs Cascade_Unet(in_channels=1, ...)); n_classes1/n_classes2 left at
    # the real 4-class (background + phosphate/sugar/base for CryoREAD's RNA/DNA backbone
    # labeling) defaults. Only the input voxel cube is shrunk (real inference chunks the
    # full map into 64**3 boxes) for a fast trace -- two maxpool3d(2) stages need a size
    # divisible by 4, so 16**3 is used.
    model = Cascade_Unet(in_channels=1, n_classes1=4, n_classes2=4)
    model.eval()
    return model


def example_input_cryoread_cascade_unet():
    # (batch, 1, D, H, W) single-channel voxelized cryo-EM density cube, exactly what
    # Cascade_Unet.forward(inputs) consumes.
    return (torch.randn(1, 1, 16, 16, 16),)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "CryoREAD",
        "build_cryoread_cascade_unet",
        "example_input_cryoread_cascade_unet",
        2023,
        "vendored",
    ),
]
