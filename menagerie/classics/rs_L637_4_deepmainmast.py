# SOURCE: vendored from kiharalab/DeepMainMast @ main
# Files combined: server_bin/emap2secplus/model/init_weights.py,
# server_bin/emap2secplus/model/Unet_Layers.py (unetConv3d only -- unetUp/unetUp_origin
# are unused by Small_UNet_3Plus_DeepSup's forward pass and are dropped),
# server_bin/emap2secplus/model/Small_Unet_3Plus_DeepSup.py
# Only minimal changes: merged the three source files into one module (relative package
# imports `from model.X import Y` flattened to direct references), and build_deepmainmast()
# reduces `n_classes` to a smaller atom-type count for a tiny random-init trace. The real
# caller (server_bin/emap2secplus/predict/unet_detect_protein.py) constructs the model
# with in_channels=1 (single-channel cryo-EM density map voxel grid), n_classes=7 for
# backbone-atom detection ('BG','N','CA','C','O','CB','Others') -- this is the CNN that
# guides DeepMainMast's main-chain tracing / C-alpha detection from cryo-EM density.
# Architecture (UNet3+ full-scale skip connections with deep supervision) is untouched.
import torch
import torch.nn as nn
from torch.nn import init

MENAGERIE_ZOO = "vendored-pytorch"


# --- server_bin/emap2secplus/model/init_weights.py ---
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("Linear") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("BatchNorm") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type="normal"):
    if init_type == "kaiming":
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError("initialization method [%s] is not implemented" % init_type)


# --- server_bin/emap2secplus/model/Unet_Layers.py (unetConv3d) ---
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

        for m in self.children():
            init_weights(m, init_type="kaiming")

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, "conv%d" % i)
            x = conv(x)
        return x


# --- server_bin/emap2secplus/model/Small_Unet_3Plus_DeepSup.py ---
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
        )

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
        )

        d3 = self.outconv3(hd3)
        d3 = self.upscore3(d3)  # 64->256

        d2 = self.outconv2(hd2)
        d2 = self.upscore2(d2)  # 128->256

        d1 = self.outconv1(hd1)  # 256

        return [d1, d2, d3]


def build_deepmainmast():
    # Real caller (unet_detect_protein.py) uses in_channels=1, n_classes=7 (atom-type
    # detection: BG/N/CA/C/O/CB/Others). n_classes kept as-is; only the input voxel-grid
    # size is shrunk for a tiny random-init trace.
    return Small_UNet_3Plus_DeepSup(
        in_channels=1, n_classes=7, feature_scale=4, is_deconv=True, is_batchnorm=True
    )


def example_input_deepmainmast():
    torch.manual_seed(0)
    # [batch, in_channels=1, D, H, W] cryo-EM density-map sub-voxel grid; D/H/W must be
    # divisible by 4 (two 2x maxpools in the encoder) -- 8 is the smallest such size.
    return (torch.randn(1, 1, 8, 8, 8),)


MENAGERIE_ENTRIES = [
    ("DeepMainMast", build_deepmainmast, example_input_deepmainmast, 2024, MENAGERIE_ZOO),
]
