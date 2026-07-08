# SOURCE: vendored from ohtlab/headpose-fsanet-pytorch @ master
# https://raw.githubusercontent.com/ohtlab/headpose-fsanet-pytorch/master/src/model.py
#
# Yang, Chen, Lin, Chuang 2019 (CVPR) "FSA-Net: Learning Fine-Grained Structure Aggregation
# for Head Pose Estimation from a Single Image". The official repo (shamangary/FSA-Net) ships
# Keras/TensorFlow only; ohtlab/headpose-fsanet-pytorch ("FSANet Model, Implemented by Omar
# Hassan, August 2020") is a complete PyTorch reimplementation of the same architecture --
# copied verbatim here: the two-stream (avg-pool "SSR" stream + max-pool "var" stream)
# multi-stage separable-conv backbone (MultiStreamMultiStage), fine-grained structure mapping
# with per-stage scoring/attention (FineGrainedStructureMapping/ScoringFunction/VarianceC),
# a capsule-routing aggregation layer (CapsuleLayer1d, dynamic routing), and the Soft
# Stagewise Regression (SSR) head (ExtractSSRParams/SSRLayer) that regresses yaw/pitch/roll.
"""FSA-Net: two-stream separable-conv backbone + capsule aggregation + SSR head for head pose."""

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# Separable Conv2d
class SepConv2d(nn.Module):
    def __init__(self, nin, nout, ksize=3):
        super(SepConv2d, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=ksize, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


# Conv2d with Activation Layer
class Conv2dAct(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, activation="relu"):
        super(Conv2dAct, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, ksize)
        if activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "tanh":
            self.act = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)

        return x


class SepConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation="relu", ksize=3):
        super(SepConvBlock, self).__init__()

        self.conv = SepConv2d(in_channels, out_channels, ksize)
        self.bn = nn.BatchNorm2d(out_channels)

        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "tanh":
            self.act = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class MultiStreamMultiStage(nn.Module):
    def __init__(self, in_channels):
        super(MultiStreamMultiStage, self).__init__()

        # Stream 0 Layers #
        self.avgpool = nn.AvgPool2d(2)
        self.s0_conv0 = SepConvBlock(in_channels, 16, "relu")

        self.s0_conv1_0 = SepConvBlock(16, 32, "relu")
        self.s0_conv1_1 = SepConvBlock(32, 32, "relu")
        self.s0_conv1_out = Conv2dAct(32, 64, 1, "relu")

        self.s0_conv2_0 = SepConvBlock(32, 64, "relu")
        self.s0_conv2_1 = SepConvBlock(64, 64, "relu")
        self.s0_conv2_out = Conv2dAct(64, 64, 1, "relu")

        self.s0_conv3_0 = SepConvBlock(64, 128, "relu")
        self.s0_conv3_1 = SepConvBlock(128, 128, "relu")
        self.s0_conv3_out = Conv2dAct(128, 64, 1, "relu")

        # Stream 1 Layers #
        self.maxpool = nn.MaxPool2d(2)
        self.s1_conv0 = SepConvBlock(in_channels, 16, "relu")

        self.s1_conv1_0 = SepConvBlock(16, 32, "tanh")
        self.s1_conv1_1 = SepConvBlock(32, 32, "tanh")
        self.s1_conv1_out = Conv2dAct(32, 64, 1, "tanh")

        self.s1_conv2_0 = SepConvBlock(32, 64, "tanh")
        self.s1_conv2_1 = SepConvBlock(64, 64, "tanh")
        self.s1_conv2_out = Conv2dAct(64, 64, 1, "tanh")

        self.s1_conv3_0 = SepConvBlock(64, 128, "tanh")
        self.s1_conv3_1 = SepConvBlock(128, 128, "tanh")
        self.s1_conv3_out = Conv2dAct(128, 64, 1, "tanh")

    def forward(self, x):
        # Stage 0 #
        s0_x = self.s0_conv0(x)
        s0_x = self.avgpool(s0_x)

        s1_x = self.s1_conv0(x)
        s1_x = self.maxpool(s1_x)

        # Stage 1 #
        s0_x = self.s0_conv1_0(s0_x)
        s0_x = self.s0_conv1_1(s0_x)
        s0_x = self.avgpool(s0_x)
        s0_stage1_out = self.s0_conv1_out(s0_x)

        s1_x = self.s1_conv1_0(s1_x)
        s1_x = self.s1_conv1_1(s1_x)
        s1_x = self.maxpool(s1_x)
        s1_stage1_out = self.s1_conv1_out(s1_x)

        stage1_out = torch.mul(s0_stage1_out, s1_stage1_out)
        # To make output size into (8x8x64), we will do avgpool here
        stage1_out = self.avgpool(stage1_out)

        # Stage 2 #
        s0_x = self.s0_conv2_0(s0_x)
        s0_x = self.s0_conv2_1(s0_x)
        s0_x = self.avgpool(s0_x)
        s0_stage2_out = self.s0_conv2_out(s0_x)

        s1_x = self.s1_conv2_0(s1_x)
        s1_x = self.s1_conv2_1(s1_x)
        s1_x = self.maxpool(s1_x)
        s1_stage2_out = self.s1_conv2_out(s1_x)

        stage2_out = torch.mul(s0_stage2_out, s1_stage2_out)

        # Stage 3 #
        s0_x = self.s0_conv3_0(s0_x)
        s0_x = self.s0_conv3_1(s0_x)
        s0_stage3_out = self.s0_conv3_out(s0_x)

        s1_x = self.s1_conv3_0(s1_x)
        s1_x = self.s1_conv3_1(s1_x)
        s1_stage3_out = self.s1_conv3_out(s1_x)

        stage3_out = torch.mul(s0_stage3_out, s1_stage3_out)

        return [stage3_out, stage2_out, stage1_out]


# Channel-Wise Variance
class VarianceC(nn.Module):
    def __init__(self):
        super(VarianceC, self).__init__()

    def forward(self, x):
        mean_x = torch.mean(x, dim=1, keepdim=True)
        sub_x = x.sub(mean_x)
        x = torch.mean(torch.mul(sub_x, sub_x), dim=1, keepdim=True)

        return x


class ScoringFunction(nn.Module):
    def __init__(self, in_channels, var=False):
        super(ScoringFunction, self).__init__()
        if var:
            self.reduce_channel = VarianceC()
        else:
            self.reduce_channel = Conv2dAct(in_channels, 1, 1, "sigmoid")

    def forward(self, x):
        x = self.reduce_channel(x)
        x = x.view(x.size(0), -1)

        return x


class FineGrainedStructureMapping(nn.Module):
    def __init__(self, in_channels, num_primcaps, mdim, var=False):
        super(FineGrainedStructureMapping, self).__init__()

        self.n = 8 * 8 * 3
        self.n_new = int(num_primcaps / 3)  # this is n' in paper
        self.m = mdim

        self.attention_maps = ScoringFunction(in_channels, var)

        self.fm = nn.Linear(
            self.n // 3, self.n * self.m
        )  # this is used for calculating Mk in paper

        self.fc = nn.Linear(self.n, self.n_new * self.m)  # this is used for calculating C in paper

    # input is list of stage outputs in batches
    def forward(self, x):
        U1, U2, U3 = x
        # Attention Maps (Ak)
        A1 = self.attention_maps(U1)
        A2 = self.attention_maps(U2)
        A3 = self.attention_maps(U3)

        # Attention Maps Concatenation
        A = torch.cat((A1, A2, A3), dim=1)

        # C Matrix
        C = torch.sigmoid(self.fc(A))
        C = C.view(C.size(0), self.n_new, self.m)

        # Mk Matrices
        M1 = torch.sigmoid(self.fm(A1))
        M1 = M1.view(M1.size(0), self.m, self.n)

        M2 = torch.sigmoid(self.fm(A2))
        M2 = M2.view(M2.size(0), self.m, self.n)

        M3 = torch.sigmoid(self.fm(A3))
        M3 = M3.view(M3.size(0), self.m, self.n)

        # Sk Matrices, Sk = matmul(C,Mk)
        S1 = torch.matmul(C, M1)
        S2 = torch.matmul(C, M2)
        S3 = torch.matmul(C, M3)

        # Concatenating Feature Maps, U = [U1,U2,U3]
        _, ch, uh, uw = U1.size()
        U1 = U1.view(-1, uh * uw, ch)
        U2 = U2.view(-1, uh * uw, ch)
        U3 = U3.view(-1, uh * uw, ch)

        U = torch.cat((U1, U2, U3), dim=1)

        # Ubar_k Matrices, Ubar_k = Sk*U
        Ubar_1 = torch.matmul(S1, U)
        Ubar_2 = torch.matmul(S2, U)
        Ubar_3 = torch.matmul(S3, U)

        # Normalizing Ubar_k (L1_Norm)
        norm_S1 = torch.sum(S1, dim=-1, keepdim=True) + 1e-8
        norm_S2 = torch.sum(S2, dim=-1, keepdim=True) + 1e-8
        norm_S3 = torch.sum(S3, dim=-1, keepdim=True) + 1e-8

        Ubar_1 = Ubar_1 / norm_S1
        Ubar_2 = Ubar_2 / norm_S2
        Ubar_3 = Ubar_3 / norm_S3

        # Concatenate Ubar_k along dim=1 which is self.n_new
        Ubar = torch.cat((Ubar_1, Ubar_2, Ubar_3), dim=1)

        return Ubar


# 1d CapsuleLayer similar to nn.Linear (which outputs scalar neurons),
# here, we output vectored neurons
class CapsuleLayer1d(nn.Module):
    def __init__(
        self, num_in_capsule, in_capsule_dim, num_out_capsule, out_capsule_dim, routings=3
    ):
        super(CapsuleLayer1d, self).__init__()
        self.routings = routings
        weight_tensor = torch.empty(
            num_out_capsule, num_in_capsule, out_capsule_dim, in_capsule_dim
        )

        init_weight = torch.nn.init.xavier_uniform_(weight_tensor)
        self.affine_w = nn.Parameter(init_weight)

    def squash(self, s, dim=-1):
        norm = torch.sum(s**2, dim=dim, keepdim=True)
        return norm / (1 + norm) * s / (torch.sqrt(norm) + 1e-8)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.unsqueeze(3)
        u_hat = torch.matmul(x, torch.transpose(self.affine_w, 2, 3))
        u_hat = u_hat.squeeze(3)

        [num_out_capsule, num_in_capsule, out_capsule_dim, in_capsule_dim] = self.affine_w.shape

        b = torch.zeros(u_hat.shape[0], num_out_capsule, num_in_capsule).to(u_hat.device)

        outputs = None
        for i in range(self.routings):
            c = F.softmax(b, dim=1)
            c = c.unsqueeze(2)

            outputs = torch.matmul(c, u_hat)
            outputs = self.squash(outputs)

            if i < self.routings - 1:
                b = b + torch.matmul(u_hat, torch.transpose(outputs, 2, 3)).squeeze(3)
                b = b

        outputs = outputs.squeeze(2)
        return outputs


class ExtractAggregatedFeatures(nn.Module):
    def __init__(self, num_capsule):
        super(ExtractAggregatedFeatures, self).__init__()
        self.num_capsule = num_capsule

    def forward(self, x):
        batch_size = x.shape[0]
        bin_size = self.num_capsule // 3

        feat_s1 = x[:, :bin_size, :]
        feat_s1 = feat_s1.view(batch_size, -1)  # reshape to 1d

        feat_s2 = x[:, bin_size : 2 * bin_size, :]
        feat_s2 = feat_s2.view(batch_size, -1)

        feat_s3 = x[:, 2 * bin_size : self.num_capsule, :]
        feat_s3 = feat_s3.view(batch_size, -1)

        return [feat_s1, feat_s2, feat_s3]


class ExtractSSRParams(nn.Module):
    def __init__(self, bins, classes):
        # our classes are: pitch, roll, yaw
        # our bins per stage are: 3
        super(ExtractSSRParams, self).__init__()
        self.bins = bins
        self.classes = classes

        self.shift_fc = nn.Linear(4, classes)  # used to shift bins

        self.scale_fc = nn.Linear(4, classes)  # used to scale bins

        # every class will have its own probability distrubtion of bins
        # hence total predictions = bins*classes
        self.pred_fc = nn.Linear(8, bins * classes)  # classes probability distrubtion of bins

    # x is batches of feature vector of shape: [batches,16]
    def forward(self, x):
        shift_param = torch.tanh(self.shift_fc(x[:, :4]))
        scale_param = torch.tanh(self.scale_fc(x[:, 4:8]))
        pred_param = F.relu(self.pred_fc(x[:, 8:]))
        pred_param = pred_param.view(pred_param.size(0), self.classes, self.bins)

        return [pred_param, shift_param, scale_param]


class SSRLayer(nn.Module):
    def __init__(self, bins):
        # this ssr layer implements MD 3-stage SSR
        super(SSRLayer, self).__init__()
        self.bins_per_stage = bins

    # x is list of ssr params for each stage
    def forward(self, x):
        s1_params, s2_params, s3_params = x

        a = b = c = 0

        bins = self.bins_per_stage

        doffset = bins // 2

        V = 99  # max bin width

        # Stage 1 loop over all bins
        for i in range(bins):
            a = a + (i - doffset + s1_params[1]) * s1_params[0][:, :, i]
        a = a / (bins * (1 + s1_params[2]))

        # Stage 2 loop over all bins
        for i in range(bins):
            b = b + (i - doffset + s2_params[1]) * s2_params[0][:, :, i]
        b = b / (bins * (1 + s1_params[2])) / (bins * (1 + s2_params[2]))

        # Stage 3 loop over all bins
        for i in range(bins):
            c = c + (i - doffset + s3_params[1]) * s3_params[0][:, :, i]
        c = (
            c
            / (bins * (1 + s1_params[2]))
            / (bins * (1 + s2_params[2]))
            / (bins * (1 + s3_params[2]))
        )

        pred = (a + b + c) * V

        return pred


class FSANet(nn.Module):
    def __init__(self, var=False):
        super(FSANet, self).__init__()
        num_primcaps = 7 * 3
        primcaps_dim = 64
        num_out_capsule = 3
        out_capsule_dim = 16
        routings = 2
        mdim = 5

        self.msms = MultiStreamMultiStage(3)  # channels: rgb
        self.fgsm = FineGrainedStructureMapping(
            64, num_primcaps, mdim, var
        )  # channels: feature maps
        self.caps_layer = CapsuleLayer1d(
            num_primcaps, primcaps_dim, num_out_capsule, out_capsule_dim, routings
        )
        self.eaf = ExtractAggregatedFeatures(num_out_capsule)
        self.esp_s1 = ExtractSSRParams(3, 3)
        self.esp_s2 = ExtractSSRParams(3, 3)
        self.esp_s3 = ExtractSSRParams(3, 3)
        self.ssr = SSRLayer(3)

    # x is batch of input rgb images
    def forward(self, x):
        # Input Shape: [batch,3,64,64]
        x = self.msms(x)

        x = self.fgsm(x)

        x = self.caps_layer(x)

        x = self.eaf(x)

        ssr_s1 = self.esp_s1(x[0])
        ssr_s2 = self.esp_s2(x[1])
        ssr_s3 = self.esp_s3(x[2])

        x = self.ssr([ssr_s1, ssr_s2, ssr_s3])

        return x


def build_fsanet():
    return FSANet(var=True)


def example_input_fsanet():
    torch.manual_seed(0)
    return (torch.randn(1, 3, 64, 64),)


MENAGERIE_ENTRIES = [
    ("FSA-Net", "build_fsanet", "example_input_fsanet", 2019, "vendored"),
]
