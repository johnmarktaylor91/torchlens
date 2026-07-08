# FAITHFUL PORT of YuemingJin/SV-RCNet @ 8df7e10f7cf26b8ef1eb05b2c73d1f4020393a49
# (original framework: Caffe -- SV-RCNet is a Caffe-source fork; the model architecture lives
# in Caffe .prototxt files, not Python)
# https://raw.githubusercontent.com/YuemingJin/SV-RCNet/master/surgicalVideo/models/SV-RCNet/SVRCNet-workflow-deploy.prototxt
#
# "SV-RCNet: Workflow Recognition From Surgical Videos Using Recurrent Convolutional Network"
# (Jin et al., IEEE TMI 2018). The repo is a full fork of the Caffe C++ framework (build system,
# vendored third-party sources, custom LSTM/Reshape/Scale caffe layers) with the actual SV-RCNet
# architecture expressed only as Caffe `.prototxt` layer graphs under
# surgicalVideo/models/SV-RCNet/ -- there is no standalone Python model file, and building this
# specific historical Caffe fork (BN/Scale-split BatchNorm, custom LSTM layer) is not reasonably
# reproducible in this base-lib torch environment. The prototxt is nonetheless an exact,
# unambiguous layer-by-layer architecture spec, so it is transcribed FAITHFULLY here rather than
# summarized from the paper text.
#
# Per `SVRCNet-workflow-deploy.prototxt`: a `conv1`(7x7/2)+BN+ReLU+`pool1`(3x3/2 max) stem
# followed by four residual stages (`res2*`..`res5*`, each stage's first block strided/
# 1x1-projected, non-first blocks Eltwise-summed identity) at [64,64,256] / [128,128,512] /
# [256,256,1024] / [512,512,2048] bottleneck widths with counts [3,4,6,3] -- i.e. byte-for-byte
# the standard ResNet-50 topology (every Convolution/BatchNorm+Scale/ReLU/Eltwise/Pooling layer
# name and shape in the prototxt matches torchvision's `resnet50` block-for-block; Caffe's BN+
# Scale layer pair is the split equivalent of torch's single affine BatchNorm2d). `pool5`
# (7x7 avg) output is reshaped to a (time_step, batch, 2048) sequence and fed through a
# `lstm1` (Caffe LSTM, num_output=512) -> `lstm1-drop` (Dropout 0.5) -> `fc8`
# (InnerProduct, 512->7, applied per-timestep via axis=2) -> `probs` (Softmax, axis=2) head,
# predicting one of 7 surgical workflow phases per frame.
import torch
from torch import nn
from torchvision.models import resnet50

MENAGERIE_ZOO = "ported-pytorch"

_N_PHASES = 7  # fc8: InnerProduct num_output=7 (surgical workflow phases)
_LSTM_HIDDEN = 512  # lstm1: recurrent_param num_output=512


class SVRCNet(nn.Module):
    """ResNet-50 (res2a..res5c, matching the prototxt block-for-block) + LSTM workflow head."""

    def __init__(self, lstm_hidden=_LSTM_HIDDEN, n_phases=_N_PHASES, dropout=0.5):
        super().__init__()
        backbone = resnet50(weights=None)
        # conv1 + bn_conv1/scale_conv1 + relu + pool1 + res2a..res5c (pool5 = backbone.avgpool)
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,
        )
        self.feat_dim = backbone.fc.in_features  # 2048, matches "reshape-data" dim: 2048

        # lstm1: Caffe LSTM(num_output=512) over the pool5 feature sequence
        self.lstm1 = nn.LSTM(input_size=self.feat_dim, hidden_size=lstm_hidden, batch_first=False)
        # lstm1-drop: Dropout(dropout_ratio=0.5)
        self.lstm1_drop = nn.Dropout(p=dropout)
        # fc8: InnerProduct(512 -> 7, axis=2 -> applied per-timestep)
        self.fc8 = nn.Linear(lstm_hidden, n_phases)
        # probs: Softmax(axis=2 -> over the phase dimension)
        self.probs = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: (time_step, batch, 3, 224, 224), matching the prototxt's "reshape-data" convention
        # (fc6-reshape: dim 3 x 10 x 2048 -> here generalized to (time_step, batch, feat_dim))
        time_step, batch, c, h, w = x.shape
        feats = self.backbone(x.reshape(time_step * batch, c, h, w))
        feats = feats.reshape(time_step, batch, self.feat_dim)

        lstm_out, _ = self.lstm1(feats)
        lstm_out = self.lstm1_drop(lstm_out)
        logits = self.fc8(lstm_out)
        return self.probs(logits)


def build_svrcnet():
    torch.manual_seed(0)
    return SVRCNet(lstm_hidden=16, n_phases=_N_PHASES, dropout=0.0)


def example_input_svrcnet():
    torch.manual_seed(0)
    time_step, batch = 3, 2
    return (torch.randn(time_step, batch, 3, 32, 32),)


MENAGERIE_ENTRIES = [
    ("SV-RCNet", "build_svrcnet", "example_input_svrcnet", 2018, "ported"),
]
