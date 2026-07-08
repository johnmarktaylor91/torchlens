# SOURCE: vendored from sunshineatnoon/LinearStyleTransfer @ 188e8e5b2ad9dbb00bed6317eb1280ecde48c657
# https://raw.githubusercontent.com/sunshineatnoon/LinearStyleTransfer/188e8e5b2ad9dbb00bed6317eb1280ecde48c657/libs/models.py (encoder4/decoder4 only)
# https://raw.githubusercontent.com/sunshineatnoon/LinearStyleTransfer/188e8e5b2ad9dbb00bed6317eb1280ecde48c657/libs/Matrix.py
# https://raw.githubusercontent.com/sunshineatnoon/LinearStyleTransfer/188e8e5b2ad9dbb00bed6317eb1280ecde48c657/TestArtistic.py (forward wiring only)
#
# Li et al. 2019 (CVPR) "Learning Linear Transformations for Fast Arbitrary Style
# Transfer" -- a VGG19-derived `encoder4` (multi-scale feature dict up to relu4_1)
# feeding `MulLayer`: two small conv-nets (`CNN`, one per content/style branch) that
# each regress a compact matrixSize x matrixSize matrix from the mean-centered
# feature map's second-moment (self-outer-product `bmm(out, out^T)`), which are then
# multiplied together (`transmatrix = sMatrix @ cMatrix`) into a single learned LINEAR
# transform applied to a 1x1-conv-compressed content feature
# (`transfeature = transmatrix @ compress_content`), then re-expanded and added back
# to the style mean before a mirrored-VGG `decoder4` renders the stylized RGB image.
# This closed-form-whitening-replacement "learn the transform matrix directly" head
# (`CNN`/`MulLayer`) is the architectural contribution over AdaIN/WCT baselines.
#
# `encoder4`, `decoder4`, `CNN`, and `MulLayer` are the models exactly as defined
# upstream (unchanged). No architectural changes were made; only mechanical fixes for
# self-containment:
#   - `libs/models.py` upstream defines TWO classes both named `decoder4` (a leftover
#     duplicate definition in the source file -- the second silently shadows the
#     first at module-import time, i.e. only the second `decoder4` is ever
#     reachable/used by any script that `from libs.models import decoder4`); both
#     definitions here are byte-identical, so keeping only one occurrence changes
#     nothing about which class ends up bound to the name `decoder4`.
#   - `encoder4.forward`'s optional style-matrix injection branches
#     (`if(matrix31 is not None): ...`) are exercised with their default `None` args,
#     matching real inference usage in `TestArtistic.py` (`vgg(styleV)` / `vgg(contentV)`
#     are called with only the image argument).
#   - `LinearStyleTransferNet.forward` below reproduces `TestArtistic.py`'s real
#     inference wiring verbatim: `cF = vgg(content); sF = vgg(style);
#     feature, transmatrix = matrix(cF['r41'], sF['r41']); transfer = dec(feature)`
#     (the `layer == 'r41'` branch, the paper's default configuration --
#     `--decoder_dir` defaults to `models/dec_r41.pth`). This wrapper is new only in
#     the sense that the original glues these three real modules together via a
#     free-standing test script rather than an `nn.Module`; the call sequence and
#     every tensor op inside it are copied unmodified from that script.

import torch
import torch.nn as nn


class encoder4(nn.Module):
    def __init__(self):
        super(encoder4, self).__init__()
        # vgg
        # 224 x 224
        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
        self.reflecPad1 = nn.ReflectionPad2d((1, 1, 1, 1))
        # 226 x 226

        self.conv2 = nn.Conv2d(3, 64, 3, 1, 0)
        self.relu2 = nn.ReLU(inplace=True)
        # 224 x 224

        self.reflecPad3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu3 = nn.ReLU(inplace=True)
        # 224 x 224

        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 112 x 112

        self.reflecPad4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 0)
        self.relu4 = nn.ReLU(inplace=True)
        # 112 x 112

        self.reflecPad5 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu5 = nn.ReLU(inplace=True)
        # 112 x 112

        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 56 x 56

        self.reflecPad6 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv6 = nn.Conv2d(128, 256, 3, 1, 0)
        self.relu6 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad7 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv7 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu7 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad8 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv8 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu8 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad9 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv9 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu9 = nn.ReLU(inplace=True)
        # 56 x 56

        self.maxPool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 28 x 28

        self.reflecPad10 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv10 = nn.Conv2d(256, 512, 3, 1, 0)
        self.relu10 = nn.ReLU(inplace=True)
        # 28 x 28

    def forward(self, x, sF=None, matrix11=None, matrix21=None, matrix31=None):
        output = {}
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        output["r11"] = self.relu2(out)
        out = self.reflecPad7(output["r11"])

        out = self.conv3(out)
        output["r12"] = self.relu3(out)

        output["p1"] = self.maxPool(output["r12"])
        out = self.reflecPad4(output["p1"])
        out = self.conv4(out)
        output["r21"] = self.relu4(out)
        out = self.reflecPad7(output["r21"])

        out = self.conv5(out)
        output["r22"] = self.relu5(out)

        output["p2"] = self.maxPool2(output["r22"])
        out = self.reflecPad6(output["p2"])
        out = self.conv6(out)
        output["r31"] = self.relu6(out)
        if matrix31 is not None:
            feature3, transmatrix3 = matrix31(output["r31"], sF["r31"])
            out = self.reflecPad7(feature3)
        else:
            out = self.reflecPad7(output["r31"])
        out = self.conv7(out)
        output["r32"] = self.relu7(out)

        out = self.reflecPad8(output["r32"])
        out = self.conv8(out)
        output["r33"] = self.relu8(out)

        out = self.reflecPad9(output["r33"])
        out = self.conv9(out)
        output["r34"] = self.relu9(out)

        output["p3"] = self.maxPool3(output["r34"])
        out = self.reflecPad10(output["p3"])
        out = self.conv10(out)
        output["r41"] = self.relu10(out)

        return output


class decoder4(nn.Module):
    def __init__(self):
        super(decoder4, self).__init__()
        # decoder
        self.reflecPad11 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv11 = nn.Conv2d(512, 256, 3, 1, 0)
        self.relu11 = nn.ReLU(inplace=True)
        # 28 x 28

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        # 56 x 56

        self.reflecPad12 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv12 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu12 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad13 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv13 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu13 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad14 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv14 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu14 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad15 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv15 = nn.Conv2d(256, 128, 3, 1, 0)
        self.relu15 = nn.ReLU(inplace=True)
        # 56 x 56

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        # 112 x 112

        self.reflecPad16 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv16 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu16 = nn.ReLU(inplace=True)
        # 112 x 112

        self.reflecPad17 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv17 = nn.Conv2d(128, 64, 3, 1, 0)
        self.relu17 = nn.ReLU(inplace=True)
        # 112 x 112

        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
        # 224 x 224

        self.reflecPad18 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv18 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu18 = nn.ReLU(inplace=True)
        # 224 x 224

        self.reflecPad19 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv19 = nn.Conv2d(64, 3, 3, 1, 0)

    def forward(self, x):
        # decoder
        out = self.reflecPad11(x)
        out = self.conv11(out)
        out = self.relu11(out)
        out = self.unpool(out)
        out = self.reflecPad12(out)
        out = self.conv12(out)

        out = self.relu12(out)
        out = self.reflecPad13(out)
        out = self.conv13(out)
        out = self.relu13(out)
        out = self.reflecPad14(out)
        out = self.conv14(out)
        out = self.relu14(out)
        out = self.reflecPad15(out)
        out = self.conv15(out)
        out = self.relu15(out)
        out = self.unpool2(out)
        out = self.reflecPad16(out)
        out = self.conv16(out)
        out = self.relu16(out)
        out = self.reflecPad17(out)
        out = self.conv17(out)
        out = self.relu17(out)
        out = self.unpool3(out)
        out = self.reflecPad18(out)
        out = self.conv18(out)
        out = self.relu18(out)
        out = self.reflecPad19(out)
        out = self.conv19(out)
        return out


class CNN(nn.Module):
    def __init__(self, layer, matrixSize=32):
        super(CNN, self).__init__()
        if layer == "r31":
            # 256x64x64
            self.convs = nn.Sequential(
                nn.Conv2d(256, 128, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, matrixSize, 3, 1, 1),
            )
        elif layer == "r41":
            # 512x32x32
            self.convs = nn.Sequential(
                nn.Conv2d(512, 256, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, matrixSize, 3, 1, 1),
            )

        # 32x8x8
        self.fc = nn.Linear(matrixSize * matrixSize, matrixSize * matrixSize)
        # self.fc = nn.Linear(32*64,256*256)

    def forward(self, x):
        out = self.convs(x)
        # 32x8x8
        b, c, h, w = out.size()
        out = out.view(b, c, -1)
        # 32x64
        out = torch.bmm(out, out.transpose(1, 2)).div(h * w)
        # 32x32
        out = out.view(out.size(0), -1)
        return self.fc(out)


class MulLayer(nn.Module):
    def __init__(self, layer, matrixSize=32):
        super(MulLayer, self).__init__()
        self.snet = CNN(layer, matrixSize)
        self.cnet = CNN(layer, matrixSize)
        self.matrixSize = matrixSize

        if layer == "r41":
            self.compress = nn.Conv2d(512, matrixSize, 1, 1, 0)
            self.unzip = nn.Conv2d(matrixSize, 512, 1, 1, 0)
        elif layer == "r31":
            self.compress = nn.Conv2d(256, matrixSize, 1, 1, 0)
            self.unzip = nn.Conv2d(matrixSize, 256, 1, 1, 0)
        self.transmatrix = None

    def forward(self, cF, sF, trans=True):
        cFBK = cF.clone()  # noqa: F841
        cb, cc, ch, cw = cF.size()
        cFF = cF.view(cb, cc, -1)
        cMean = torch.mean(cFF, dim=2, keepdim=True)
        cMean = cMean.unsqueeze(3)
        cMean = cMean.expand_as(cF)
        cF = cF - cMean

        sb, sc, sh, sw = sF.size()
        sFF = sF.view(sb, sc, -1)
        sMean = torch.mean(sFF, dim=2, keepdim=True)
        sMean = sMean.unsqueeze(3)
        sMeanC = sMean.expand_as(cF)
        sMeanS = sMean.expand_as(sF)
        sF = sF - sMeanS

        compress_content = self.compress(cF)
        b, c, h, w = compress_content.size()
        compress_content = compress_content.view(b, c, -1)

        if trans:
            cMatrix = self.cnet(cF)
            sMatrix = self.snet(sF)

            sMatrix = sMatrix.view(sMatrix.size(0), self.matrixSize, self.matrixSize)
            cMatrix = cMatrix.view(cMatrix.size(0), self.matrixSize, self.matrixSize)
            transmatrix = torch.bmm(sMatrix, cMatrix)
            transfeature = torch.bmm(transmatrix, compress_content).view(b, c, h, w)
            out = self.unzip(transfeature.view(b, c, h, w))
            out = out + sMeanC
            return out, transmatrix
        else:
            out = self.unzip(compress_content.view(b, c, h, w))
            out = out + cMean
            return out


class LinearStyleTransferNet(nn.Module):
    """Composite inference module wiring encoder4 -> MulLayer('r41') -> decoder4,
    exactly reproducing TestArtistic.py's real forward pass (opt.layer == 'r41'
    branch, the paper's default configuration)."""

    def __init__(self):
        super().__init__()
        self.vgg = encoder4()
        self.matrix = MulLayer("r41")
        self.dec = decoder4()

    def forward(self, content, style):
        cF = self.vgg(content)
        sF = self.vgg(style)
        feature, transmatrix = self.matrix(cF["r41"], sF["r41"])
        transfer = self.dec(feature)
        return transfer


def build_linearstyletransfer():
    model = LinearStyleTransferNet()
    model.eval()
    return model


def example_input_linearstyletransfer():
    content = torch.randn(1, 3, 64, 64)
    style = torch.randn(1, 3, 64, 64)
    return (content, style)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "LinearStyleTransfer",
        "build_linearstyletransfer",
        "example_input_linearstyletransfer",
        2019,
        "vendored",
    ),
]
