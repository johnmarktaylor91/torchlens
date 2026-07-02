# FAITHFUL PORT of https://github.com/ThienHuynhThe/MCNet @ main
# (mcnet_commlett.m :: the `lgraph = layerGraph(); ... addLayers/connectLayers`
# network definition, original framework: MATLAB Deep Learning Toolbox)
#
# ThienHuynhThe/MCNet is the real, runnable-in-MATLAB code behind "MCNet: An
# Efficient CNN Architecture for Robust Automatic Modulation Classification"
# (Huynh-The et al., IEEE Communications Letters 2020) -- a capsule-network-
# style / Inception-ResNet-hybrid CNN for automatic modulation classification
# (AMC) of raw I/Q radio signals from the DeepSig RadioML2018 dataset. The
# repository ships only a MATLAB `.m` script that programmatically builds a
# `layerGraph` (no PyTorch/TensorFlow path exists anywhere in the repo, and
# MATLAB + its Deep Learning Toolbox cannot run in this environment), so this
# is a faithful PORT (rung 3): every `addLayers`/`connectLayers` call in
# `mcnet_commlett.m` is transcribed 1:1 below, layer-for-layer and
# connection-for-connection, from the actual MATLAB graph definition.
#
# Architecture (transcribed from mcnet_commlett.m):
#   input [2 x 1024 x 1] (MATLAB HWC; IQ signal reshaped as a 2 x 1024 "image")
#     -> conv1(64, 3x7, stride 1x2, pad 1x3) -> relu -> pool1(1x3, stride 1x2, pad 0x1)
#     -> "pre" stem: two parallel branches (pre_conv_a 3x1 + avgpool, pre_conv_b 1x3)
#        concatenated (pre_concat), which feeds both a 1x1 "jump" projection
#        (jump_conv1x1 -> relu -> jump_poolingA) and a "post_pooling" branch
#        that begins M-block A.
#     -> six M-blocks (A-F), each an Inception-style 3-branch mixer
#        (1x1 -> {1x3+avgpool, 1x1, 3x1} -> depthConcat) with a parallel
#        max-pooled "jump" shortcut added elementwise (`add_mixX`), except
#        block F which feeds directly into the final concat (no residual add).
#     -> concat_all(add_mixE output, mixF output) -> global avgpool(2x8)
#        -> fc(24) -> dropout(0.5) -> softmax -> classification.
# Every conv/pool kernel size, stride, padding, and channel count below is
# copied verbatim from the MATLAB source; `Padding=[top bottom left right]`
# in MATLAB HW order is symmetric on every axis actually used here except the
# two `jump_poolingC`/`jump_poolingE` max-pools (`Padding=[1 0 0 0]`, i.e.
# pad only the top of the height axis by 1) which are ported with an
# explicit `nn.ConstantPad2d` (using -inf so the pad does not win any max)
# ahead of a zero-padding `nn.MaxPool2d`.
#
# Not ported: MATLAB's `classificationLayer` (a training-time cross-entropy
# loss wrapper with no forward-inference behavior); the port stops at
# `softmax`, matching how a MATLAB `DAGNetwork`'s `predict()` also excludes
# the loss layer from its output.

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


def _conv_bn_relu(in_ch, out_ch, kernel, stride=(1, 1), padding=(0, 0)):
    """conv2dLayer + reluLayer, matching MATLAB's `convolution2dLayer(...) -> reluLayer`."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding),
        nn.ReLU(inplace=True),
    )


def _asym_maxpool(kernel, stride, pad_top):
    """maxPooling2dLayer with MATLAB Padding=[pad_top 0 0 0] (top-only H pad).

    MATLAB pads with -Inf for max pooling, so a genuine max-pool never
    "sees" the padded border as a real value. Ported with an explicit
    -inf ConstantPad2d (left,right,top,bottom) followed by a MaxPool2d
    with padding=0, which reproduces that semantic exactly.
    """
    return nn.Sequential(
        nn.ConstantPad2d((0, 0, pad_top, 0), float("-inf")),
        nn.MaxPool2d(kernel_size=kernel, stride=stride, padding=0),
    )


class _PreStem(nn.Module):
    """pre_conv_a / pre_conv_b branches -> pre_concat (MATLAB lines 13-25)."""

    def __init__(self, in_ch):
        super().__init__()
        self.branch_a = nn.Sequential(
            _conv_bn_relu(in_ch, 32, kernel=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.AvgPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
        )
        self.branch_b = _conv_bn_relu(in_ch, 32, kernel=(1, 3), stride=(1, 2), padding=(0, 1))

    def forward(self, x):
        # connectLayers order: pre_concat/in1 <- pre_relu_b, pre_concat/in2 <- pre_pool_a
        a = self.branch_a(x)
        b = self.branch_b(x)
        return torch.cat([b, a], dim=1)


class _MBlockADResidual(nn.Module):
    """Shared topology for M-blocks A-D: 1x1 stem -> 3-way mixer -> depthConcat,
    with an elementwise residual add against a parallel "jump" path.

    Blocks A and C are the strided (downsampling) variant: conv_b gains an
    extra stride-2 avgpool stage (`mblockX_pool_b`) and conv_c/conv_d use
    stride 1x2, matching a strided `jump_pool` on the residual branch.
    Blocks B and D are the non-strided (identity-jump) variant: all convs
    use stride 1 and conv_b has no trailing pool, matching an identity
    residual (jump_pool=None).

    connectLayers order into `mixX` (in1/in2/in3) and `add_mixX` (in1=mix,
    in2=jump) is preserved per block, since MATLAB depthConcatenationLayer
    concatenates its named inputs in `in1,in2,in3,...` order (channel dim).
    """

    def __init__(self, in_ch, c_a, c_b, c_c, c_d, mix_order, strided, jump_pool=None):
        super().__init__()
        stride = (1, 2) if strided else (1, 1)
        self.conv_a = _conv_bn_relu(in_ch, c_a, kernel=(1, 1))
        if strided:
            self.conv_b = nn.Sequential(
                _conv_bn_relu(c_a, c_b, kernel=(3, 1), padding=(1, 0)),
                nn.AvgPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            )
        else:
            self.conv_b = _conv_bn_relu(c_a, c_b, kernel=(3, 1), padding=(1, 0))
        self.conv_c = _conv_bn_relu(c_a, c_c, kernel=(1, 3), stride=stride, padding=(0, 1))
        self.conv_d = _conv_bn_relu(c_a, c_d, kernel=(1, 1), stride=stride)
        self.mix_order = mix_order  # tuple of 'b','c','d' naming the concat order
        self.jump_pool = jump_pool  # None -> jump is identity (already correct shape)

    def forward(self, x, jump_source):
        a = self.conv_a(x)
        outs = {"b": self.conv_b(a), "c": self.conv_c(a), "d": self.conv_d(a)}
        mix = torch.cat([outs[k] for k in self.mix_order], dim=1)
        jump = self.jump_pool(jump_source) if self.jump_pool is not None else jump_source
        return mix + jump


class _MBlockE(nn.Module):
    """M-block E (MATLAB lines 146-171): same 1x1 stem, but conv_d has no
    stride-2 companion pool for 'b' -- conv_b itself carries the maxpool
    (mblockE_pool_b uses maxPooling2dLayer, not averagePooling2dLayer, per
    the source), and the jump path is `jump_poolingE` (max-pool of
    add_mixD, with the asymmetric top-pad quirk)."""

    def __init__(self, in_ch, c_a=32, c_b=48, c_c=48, c_d=32):
        super().__init__()
        self.conv_a = _conv_bn_relu(in_ch, c_a, kernel=(1, 1))
        self.conv_d = _conv_bn_relu(c_a, c_d, kernel=(1, 1), stride=(1, 2))
        self.conv_b = nn.Sequential(
            _conv_bn_relu(c_a, c_b, kernel=(3, 1), padding=(1, 0)),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
        )
        self.conv_c = _conv_bn_relu(c_a, c_c, kernel=(1, 3), stride=(1, 2), padding=(0, 1))
        self.jump_pool = _asym_maxpool(kernel=(2, 2), stride=(1, 2), pad_top=1)

    def forward(self, x, jump_source):
        a = self.conv_a(x)
        c_out, pool_out, d_out = self.conv_c(a), self.conv_b(a), self.conv_d(a)
        # mixE/in1<-mblockE_relu_c, in2<-mblockE_pool_b, in3<-mblockE_relu_d
        mix = torch.cat([c_out, pool_out, d_out], dim=1)
        jump = self.jump_pool(jump_source)
        return mix + jump


class _MBlockF(nn.Module):
    """M-block F (MATLAB lines 173-194): terminal mixer feeding concat_all
    directly -- no residual add (there is no `add_mixF` in the source)."""

    def __init__(self, in_ch, c_a=32, c_b=96, c_c=96, c_d=64):
        super().__init__()
        self.conv_a = _conv_bn_relu(in_ch, c_a, kernel=(1, 1))
        self.conv_d = _conv_bn_relu(c_a, c_d, kernel=(1, 1))
        self.conv_c = _conv_bn_relu(c_a, c_c, kernel=(1, 3), padding=(0, 1))
        self.conv_b = _conv_bn_relu(c_a, c_b, kernel=(3, 1), padding=(1, 0))

    def forward(self, x):
        a = self.conv_a(x)
        # mixF/in1<-mblockF_relu_c, in2<-mblockF_relu_b, in3<-mblockF_relu_d
        return torch.cat([self.conv_c(a), self.conv_b(a), self.conv_d(a)], dim=1)


class MCNet(nn.Module):
    """
    Port of the MATLAB `layerGraph` built by `mcnet_commlett.m`
    (Huynh-The et al., "MCNet: An Efficient CNN Architecture for Robust
    Automatic Modulation Classification", IEEE Commun. Lett. 2020).

    Input is a `[N, 1, 2, L]` tensor (N batch, 1 channel, 2 = I/Q rows,
    L = signal length; MATLAB's `imageInputLayer([2 1024 1])` is HWC with
    H=2, W=1024, C=1, i.e. torch NCHW `[N, 1, 2, 1024]`).
    """

    def __init__(self, in_len: int = 1024, num_classes: int = 24):
        super().__init__()
        self.stem = nn.Sequential(
            _conv_bn_relu(1, 64, kernel=(3, 7), stride=(1, 2), padding=(1, 3)),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
        )
        self.pre = _PreStem(64)  # -> 64 channels (32 + 32)

        # jump_conv1x1 -> relu -> jump_poolingA (parallel to post_pooling/mblockA)
        self.jump_conv1x1 = _conv_bn_relu(64, 128, kernel=(1, 1), stride=(1, 2))
        self.jump_poolingA = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.post_pooling = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))

        # M-block A: in=64 (post_pooling of pre_concat, 64ch), strided;
        # jump is the externally pre-pooled jumpA (jump_conv1x1->jump_poolingA)
        self.blockA = _MBlockADResidual(
            in_ch=64,
            c_a=32,
            c_b=48,
            c_c=48,
            c_d=32,
            mix_order=("c", "b", "d"),
            strided=True,
            jump_pool=None,
        )  # mixA out = 48+48+32=128, matches jump 128ch -> add_mixA=128ch

        # M-block B: in=128 (add_mixA), non-strided; jump=add_mixA itself (identity)
        self.blockB = _MBlockADResidual(
            in_ch=128,
            c_a=32,
            c_b=48,
            c_c=48,
            c_d=32,
            mix_order=("b", "c", "d"),
            strided=False,
            jump_pool=None,
        )  # mixB=48+48+32=128 -> add_mixB=128ch

        # M-block C: in=128 (add_mixB), strided; jump=jump_poolingC(add_mixB)
        # (asym top-pad maxpool, applied inside the block to the raw add_mixB input)
        # mixC/in1<-mblockC_pool_b(b), in2<-mblockC_relu_c(c), in3<-mblockC_relu_d(d)
        self.blockC = _MBlockADResidual(
            in_ch=128,
            c_a=32,
            c_b=48,
            c_c=48,
            c_d=32,
            mix_order=("b", "c", "d"),
            strided=True,
            jump_pool=_asym_maxpool(kernel=(2, 2), stride=(1, 2), pad_top=1),
        )  # mixC=48+48+32=128 -> add_mixC=128ch (jump_poolingC downsamples W by 2)

        # M-block D: in=128 (add_mixC), non-strided; jump=add_mixC itself (identity)
        self.blockD = _MBlockADResidual(
            in_ch=128,
            c_a=32,
            c_b=48,
            c_c=48,
            c_d=32,
            mix_order=("c", "b", "d"),
            strided=False,
            jump_pool=None,
        )  # mixD=48+48+32=128 -> add_mixD=128ch

        # M-block E: in=128 (add_mixD), jump=jump_poolingE(add_mixD) (asym top-pad maxpool)
        self.blockE = _MBlockE(in_ch=128, c_a=32, c_b=48, c_c=48, c_d=32)
        # mixE=48(c)+48(pool_b)+32(d)=128 -> add_mixE=128ch

        # M-block F: in=128 (add_mixE), terminal -- feeds concat_all directly
        self.blockF = _MBlockF(in_ch=128, c_a=32, c_b=96, c_c=96, c_d=64)
        # mixF=96(c)+96(b)+64(d)=256ch

        # concat_all/in1<-mixF(256ch), in2<-add_mixE(128ch) => 384ch
        self.global_pool = nn.AvgPool2d(kernel_size=(2, 8))
        self.fc = nn.Linear(384, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.stem(x)
        x = self.pre(x)

        jumpA = self.jump_poolingA(self.jump_conv1x1(x))
        post = self.post_pooling(x)
        add_mixA = self.blockA(post, jumpA)

        add_mixB = self.blockB(add_mixA, add_mixA)
        add_mixC = self.blockC(add_mixB, add_mixB)
        add_mixD = self.blockD(add_mixC, add_mixC)
        add_mixE = self.blockE(add_mixD, add_mixD)
        mixF = self.blockF(add_mixE)

        out = torch.cat([mixF, add_mixE], dim=1)
        out = self.global_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        out = self.dropout(out)
        out = F.softmax(out, dim=-1)
        return out


# ---------------------------------------------------------------------------
# Tiny build/example for TorchLens tracing. The MATLAB source hardcodes
# `imageInputLayer([2 1024 1])` (RadioML2018 IQ frames of length 1024) and
# `fullyConnectedLayer(24)` (24 modulation classes in RadioML2018). We keep
# those real architectural constants (they define the graph's channel/class
# counts) and use a random init for weights, matching the menagerie's
# tiny-random-init convention.
# ---------------------------------------------------------------------------
def build_mcnet():
    torch.manual_seed(0)
    model = MCNet(in_len=1024, num_classes=24)
    model.eval()
    return model


def example_input_mcnet():
    torch.manual_seed(0)
    return torch.randn(2, 1, 2, 1024)


MENAGERIE_ENTRIES = [
    ("MCNet", "build_mcnet", "example_input_mcnet", 2020, MENAGERIE_ZOO),
]
