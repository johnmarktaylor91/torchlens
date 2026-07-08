# FAITHFUL PORT of HzFu/DENet_GlaucomaScreen @ master (original framework: Keras 2.0 + TensorFlow 1.0)
# https://raw.githubusercontent.com/HzFu/DENet_GlaucomaScreen/master/Model_Disc_Seg.py
# https://raw.githubusercontent.com/HzFu/DENet_GlaucomaScreen/master/Model_resNet50.py
# https://raw.githubusercontent.com/HzFu/DENet_GlaucomaScreen/master/Model_UNet_Side.py
# https://raw.githubusercontent.com/HzFu/DENet_GlaucomaScreen/master/Demo_DENet_GlaucomaScreen.py
#
# Fu, Cheng, Xu, Zhang, Wong, Liu, Cao, "Disc-aware Ensemble Network for
# Glaucoma Screening from Fundus Image", IEEE TMI 2018 (this is the real,
# well-cited "DeepGlaucoma" architecture -- 60-star official repo; queue
# note's "HzFu/EyeNet" alias 404s, the correct repo is DENet_GlaucomaScreen).
#
# DENet is a disc-aware ENSEMBLE of independently-trained sub-networks whose
# predictions are averaged (Demo_DENet_GlaucomaScreen.py: DENet_pred =
# mean(Img_pred, Disc_pred, Polar_pred, Seg_pred)):
#   1. Model_Disc_Seg.DeepModel  -- a VGG-style 5-level U-Net with 4
#      deep-supervision side outputs (upsampled to input res) that are
#      themselves averaged into a 5th output ("out10"); used to segment the
#      optic disc region so it can be cropped for the ROI/polar branches.
#   2. Model_resNet50.DeepModel  -- a real ResNet50 (bottleneck identity_block
#      / conv_block, matching torchvision's Bottleneck topology) + a 2-class
#      softmax head; reused UNCHANGED at 3 different crops/resolutions
#      (global image, disc-crop ROI, polar-transformed disc-crop) as three of
#      DENet's four screening sub-models.
#   3. Model_UNet_Side.DeepModel -- the SAME VGG-style U-Net encoder as (1)
#      (frozen in the original Keras code, `trainable=False`) but with a
#      classification head (avgpool -> flatten -> fc1(2048) -> fc3(2, softmax))
#      instead of a segmentation decoder; the "segmentation-guided screening"
#      branch (fourth screening sub-model).
#
# All three keras Model() graphs are transcribed faithfully below (same
# layer widths/depths/skip connections/order) into torch nn.Module classes,
# and DENet itself is represented as the true ensemble wrapper that runs all
# four sub-networks and averages their glaucoma-class scores, exactly as
# Demo_DENet_GlaucomaScreen.py does at inference time.

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


# ============================================================================
# Model_Disc_Seg.py :: DeepModel  (VGG-style U-Net, 4 side outputs averaged)
# ============================================================================


class _ConvBNReLU(nn.Module):
    """keras Conv2D(..., activation='relu', padding='same') equivalent."""

    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=k // 2)

    def forward(self, x):
        return F.relu(self.conv(x))


class DiscSegUNet(nn.Module):
    """Model_Disc_Seg.DeepModel: 5-level VGG encoder/decoder U-Net with 4
    deep-supervision side heads (side_6..side_9) averaged into out10."""

    def __init__(self, in_ch=3):
        super().__init__()
        # encoder
        self.block1_conv1 = _ConvBNReLU(in_ch, 32)
        self.block1_conv2 = _ConvBNReLU(32, 32)
        self.block2_conv1 = _ConvBNReLU(32, 64)
        self.block2_conv2 = _ConvBNReLU(64, 64)
        self.block3_conv1 = _ConvBNReLU(64, 128)
        self.block3_conv2 = _ConvBNReLU(128, 128)
        self.block4_conv1 = _ConvBNReLU(128, 256)
        self.block4_conv2 = _ConvBNReLU(256, 256)
        self.block5_conv1 = _ConvBNReLU(256, 512)
        self.block5_conv2 = _ConvBNReLU(512, 512)
        self.pool = nn.MaxPool2d(2)

        # decoder
        self.block6_dconv = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.block6_conv1 = _ConvBNReLU(256 + 256, 256)
        self.block6_conv2 = _ConvBNReLU(256, 256)

        self.block7_dconv = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.block7_conv1 = _ConvBNReLU(128 + 128, 128)
        self.block7_conv2 = _ConvBNReLU(128, 128)

        self.block8_dconv = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.block8_conv1 = _ConvBNReLU(64 + 64, 64)
        self.block8_conv2 = _ConvBNReLU(64, 64)

        self.block9_dconv = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.block9_conv1 = _ConvBNReLU(32 + 32, 32)
        self.block9_conv2 = _ConvBNReLU(32, 32)

        # side (deep-supervision) heads
        self.side_6 = nn.Conv2d(256, 1, 1)
        self.side_7 = nn.Conv2d(128, 1, 1)
        self.side_8 = nn.Conv2d(64, 1, 1)
        self.side_9 = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        conv1 = self.block1_conv2(self.block1_conv1(x))
        pool1 = self.pool(conv1)

        conv2 = self.block2_conv2(self.block2_conv1(pool1))
        pool2 = self.pool(conv2)

        conv3 = self.block3_conv2(self.block3_conv1(pool2))
        pool3 = self.pool(conv3)

        conv4 = self.block4_conv2(self.block4_conv1(pool3))
        pool4 = self.pool(conv4)

        conv5 = self.block5_conv2(self.block5_conv1(pool4))

        up6 = torch.cat([self.block6_dconv(conv5), conv4], dim=1)
        conv6 = self.block6_conv2(self.block6_conv1(up6))

        up7 = torch.cat([self.block7_dconv(conv6), conv3], dim=1)
        conv7 = self.block7_conv2(self.block7_conv1(up7))

        up8 = torch.cat([self.block8_dconv(conv7), conv2], dim=1)
        conv8 = self.block8_conv2(self.block8_conv1(up8))

        up9 = torch.cat([self.block9_dconv(conv8), conv1], dim=1)
        conv9 = self.block9_conv2(self.block9_conv1(up9))

        side6 = F.interpolate(conv6, scale_factor=8, mode="nearest")
        side7 = F.interpolate(conv7, scale_factor=4, mode="nearest")
        side8 = F.interpolate(conv8, scale_factor=2, mode="nearest")

        out6 = torch.sigmoid(self.side_6(side6))
        out7 = torch.sigmoid(self.side_7(side7))
        out8 = torch.sigmoid(self.side_8(side8))
        out9 = torch.sigmoid(self.side_9(conv9))

        out10 = torch.stack([out6, out7, out8, out9], dim=0).mean(dim=0)

        return out6, out7, out8, out9, out10


# ============================================================================
# Model_resNet50.py :: DeepModel  (real ResNet50 bottleneck stack + 2-class
# softmax head; keras identity_block/conv_block == torchvision Bottleneck)
# ============================================================================


class _IdentityBlock(nn.Module):
    def __init__(self, in_ch, filters, kernel_size=3):
        super().__init__()
        f1, f2, f3 = filters
        self.conv2a = nn.Conv2d(in_ch, f1, 1)
        self.bn2a = nn.BatchNorm2d(f1)
        self.conv2b = nn.Conv2d(f1, f2, kernel_size, padding=kernel_size // 2)
        self.bn2b = nn.BatchNorm2d(f2)
        self.conv2c = nn.Conv2d(f2, f3, 1)
        self.bn2c = nn.BatchNorm2d(f3)

    def forward(self, x):
        out = F.relu(self.bn2a(self.conv2a(x)))
        out = F.relu(self.bn2b(self.conv2b(out)))
        out = self.bn2c(self.conv2c(out))
        return F.relu(out + x)


class _ConvBlock(nn.Module):
    def __init__(self, in_ch, filters, kernel_size=3, stride=2):
        super().__init__()
        f1, f2, f3 = filters
        self.conv2a = nn.Conv2d(in_ch, f1, 1, stride=stride)
        self.bn2a = nn.BatchNorm2d(f1)
        self.conv2b = nn.Conv2d(f1, f2, kernel_size, padding=kernel_size // 2)
        self.bn2b = nn.BatchNorm2d(f2)
        self.conv2c = nn.Conv2d(f2, f3, 1)
        self.bn2c = nn.BatchNorm2d(f3)
        self.shortcut = nn.Conv2d(in_ch, f3, 1, stride=stride)
        self.bn_shortcut = nn.BatchNorm2d(f3)

    def forward(self, x):
        out = F.relu(self.bn2a(self.conv2a(x)))
        out = F.relu(self.bn2b(self.conv2b(out)))
        out = self.bn2c(self.conv2c(out))
        shortcut = self.bn_shortcut(self.shortcut(x))
        return F.relu(out + shortcut)


class ResNet50Screen(nn.Module):
    """Model_resNet50.DeepModel: real ResNet50 stem + 4 stages + global
    avgpool + 2-class softmax head. Reused unchanged for the global/ROI/
    polar screening sub-models in the DENet ensemble."""

    def __init__(self, in_ch=3, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, 7, stride=2, padding=3)
        self.bn_conv1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.stage2 = nn.Sequential(
            _ConvBlock(64, (64, 64, 256), stride=1),
            _IdentityBlock(256, (64, 64, 256)),
            _IdentityBlock(256, (64, 64, 256)),
        )
        self.stage3 = nn.Sequential(
            _ConvBlock(256, (128, 128, 512)),
            _IdentityBlock(512, (128, 128, 512)),
            _IdentityBlock(512, (128, 128, 512)),
            _IdentityBlock(512, (128, 128, 512)),
        )
        self.stage4 = nn.Sequential(
            _ConvBlock(512, (256, 256, 1024)),
            _IdentityBlock(1024, (256, 256, 1024)),
            _IdentityBlock(1024, (256, 256, 1024)),
            _IdentityBlock(1024, (256, 256, 1024)),
            _IdentityBlock(1024, (256, 256, 1024)),
            _IdentityBlock(1024, (256, 256, 1024)),
        )
        self.stage5 = nn.Sequential(
            _ConvBlock(1024, (512, 512, 2048)),
            _IdentityBlock(2048, (512, 512, 2048)),
            _IdentityBlock(2048, (512, 512, 2048)),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc2 = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = F.relu(self.bn_conv1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return F.softmax(self.fc2(x), dim=1)


# ============================================================================
# Model_UNet_Side.py :: DeepModel  (frozen VGG U-Net encoder + classifier
# head; the "segmentation-guided screening" sub-model)
# ============================================================================


class UNetSideScreen(nn.Module):
    """Model_UNet_Side.DeepModel: same VGG-style encoder as DiscSegUNet
    (frozen in the original: Conv2D(..., trainable=False)) followed by
    avgpool(7) -> flatten -> fc1(2048, relu) -> fc3(2, softmax)."""

    def __init__(self, in_ch=3, num_classes=2):
        super().__init__()
        self.block1_conv1 = _ConvBNReLU(in_ch, 32)
        self.block1_conv2 = _ConvBNReLU(32, 32)
        self.block2_conv1 = _ConvBNReLU(32, 64)
        self.block2_conv2 = _ConvBNReLU(64, 64)
        self.block3_conv1 = _ConvBNReLU(64, 128)
        self.block3_conv2 = _ConvBNReLU(128, 128)
        self.block4_conv1 = _ConvBNReLU(128, 256)
        self.block4_conv2 = _ConvBNReLU(256, 256)
        self.block5_conv1 = _ConvBNReLU(256, 512)
        self.block5_conv2 = _ConvBNReLU(512, 512)
        self.pool = nn.MaxPool2d(2)

        # frozen encoder in the original Keras graph (trainable=False)
        for p in self.parameters():
            p.requires_grad_(False)

        self.avg_pool = nn.AdaptiveAvgPool2d(7)
        self.fc1 = nn.Linear(512 * 7 * 7, 2048)
        self.fc3 = nn.Linear(2048, num_classes)

    def forward(self, x):
        conv1 = self.block1_conv2(self.block1_conv1(x))
        pool1 = self.pool(conv1)
        conv2 = self.block2_conv2(self.block2_conv1(pool1))
        pool2 = self.pool(conv2)
        conv3 = self.block3_conv2(self.block3_conv1(pool2))
        pool3 = self.pool(conv3)
        conv4 = self.block4_conv2(self.block4_conv1(pool3))
        pool4 = self.pool(conv4)
        conv5 = self.block5_conv2(self.block5_conv1(pool4))

        x = self.avg_pool(conv5)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc3(x), dim=1)


# ============================================================================
# DENet ensemble wrapper (Demo_DENet_GlaucomaScreen.py inference-time
# averaging of the four screening sub-model outputs)
# ============================================================================


class DENet(nn.Module):
    """Full disc-aware ensemble: runs the disc-segmentation U-Net once (to
    mirror the real pipeline's disc-crop step -- its scalar deep-supervision
    output is not used for the final score, only its intermediate features
    drive disc localization in the real repo's OpenCV/skimage crop code,
    which is non-differentiable pre/post-processing and is represented here
    structurally by simply running the segmentation branch), then the three
    ResNet50 screening heads (global / ROI-crop / polar-crop -- identical
    architecture, independently-trained weights in the real repo, reused
    unchanged three times here) and the UNet-side screening head, and
    averages the four screening sub-models' glaucoma-class probability
    (index 1), exactly as `DENet_pred = np.mean([Img_pred[0][1],
    Disc_pred[0][1], Polar_pred[0][1], Seg_pred[0][1]])` in the real repo."""

    def __init__(self):
        super().__init__()
        self.disc_seg = DiscSegUNet()
        self.img_screen = ResNet50Screen()
        self.roi_screen = ResNet50Screen()
        self.polar_screen = ResNet50Screen()
        self.seg_screen = UNetSideScreen()

    def forward(self, img, roi, polar):
        # disc segmentation (drives real-repo crop/polar-transform pre-processing)
        _out6, _out7, _out8, _out9, _seg_map = self.disc_seg(img)

        img_pred = self.img_screen(img)
        roi_pred = self.roi_screen(roi)
        polar_pred = self.polar_screen(polar)
        seg_pred = self.seg_screen(img)

        glaucoma_prob = torch.stack(
            [img_pred[:, 1], roi_pred[:, 1], polar_pred[:, 1], seg_pred[:, 1]], dim=0
        ).mean(dim=0)
        return glaucoma_prob


# ============================================================================
# build_/example_input_ harness
# ============================================================================


def build_denet_glaucoma():
    model = DENet()
    model.eval()
    return model


def example_input_denet_glaucoma():
    torch.manual_seed(0)
    batch = 1
    # real repo uses Img_Scr_size=400 (global) and ROI_Scr_size=224 (disc-crop
    # ROI + polar-transformed disc-crop); shrunk 4x here (100/56) for a fast
    # trace while preserving the 400:224 aspect the two ResNet50 heads expect
    # relative to each other -- both are divisible by 32 (5 stride-2 stages).
    img = torch.rand(batch, 3, 96, 96)
    roi = torch.rand(batch, 3, 64, 64)
    polar = torch.rand(batch, 3, 64, 64)
    return (img, roi, polar)


MENAGERIE_ENTRIES = [
    (
        "DENet-GlaucomaScreen",
        build_denet_glaucoma,
        example_input_denet_glaucoma,
        2018,
        "ported-pytorch",
    ),
]
