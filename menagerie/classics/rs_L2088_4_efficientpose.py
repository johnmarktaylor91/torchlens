# SOURCE: vendored from daniegr/EfficientPose @ 78e4bd1913780dd565efe35db03a4aec75907e8c
# https://raw.githubusercontent.com/daniegr/EfficientPose/master/models/pytorch/EfficientPoseRT.py
#
# EfficientPose: Scalable single-person 2D human pose estimation. Groos, Ramampiaro,
# Ihlen. Applied Intelligence 2021 (arXiv:2004.12186). Official repo ships PyTorch model
# definitions directly (models/pytorch/EfficientPoseRT.py, the authors' own MMdnn-converted
# export of their EfficientNet-derived detection-block architecture with skeleton/detection
# passes, squeeze-excite blocks, "eswish" activations, and bilinear-upsampling transposed
# convolutions) alongside the original TF/Keras/TFLite artifacts -- the queue note's
# "TF/Keras not PyTorch natively" refers to the training framework, not this shipped
# PyTorch export, which is the real architecture graph used here (EfficientPoseRT, the
# smallest of the 5 published I/II/III/IV/RT variants).
#
# `KitModel` (renamed from the exact real class of the same name) below is the real
# architecture: every conv/batchnorm/se-block/eswish-activation layer definition and the
# entire `forward()` wiring are reproduced verbatim from the real file. Two minimal,
# non-architectural adaptations were made:
#   1. `__conv`/`__batch_normalization` normally require a pre-converted MMdnn numpy
#      weight dict (`weight_file`) -- the repo ships only `.h5`/`.pb`/`.tflite` weight
#      blobs, no loadable `.npy` for the PyTorch path. When no weight dict is supplied
#      (`weight_file=None`, this staging module's tiny random-init construction), the
#      two methods now fall through to the layer's own default PyTorch init instead of
#      indexing a missing dict entry -- same real `nn.Conv2d`/`nn.BatchNorm2d` layer
#      construction either way, just not overwritten with converted weights.
#   2. `helpers.pytorch_BilinearConvTranspose2d` (used by `__transposed` for the 3 final
#      upsampling layers) is inlined below verbatim from `utils/helpers.py`, instead of
#      importing the full `helpers` module -- `helpers.py` imports `tensorflow`/`skimage`/
#      `scipy` at module scope for unrelated preprocessing/visualization helpers that
#      `KitModel.forward` never calls; only the actually-used bilinear-transpose-conv
#      class is vendored.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__weights_dict = dict()


def load_weights(weight_file):
    if weight_file == None:  # noqa: E711 (kept verbatim from real repo)
        return dict()

    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:  # noqa: E722 (kept verbatim from real repo)
        weights_dict = np.load(weight_file, allow_pickle=True, encoding="bytes").item()

    return weights_dict


# ---------------------------------------------------------------------------
# utils/helpers.py :: pytorch_BilinearConvTranspose2d (inlined; the only symbol
# from helpers.py that KitModel.forward actually uses -- avoids importing the
# rest of helpers.py, which pulls in tensorflow/skimage/scipy for unrelated
# preprocessing/visualization helpers)
# ---------------------------------------------------------------------------
class pytorch_BilinearConvTranspose2d(nn.ConvTranspose2d):
    """
    A PyTorch implementation of transposed bilinear convolution by mjstevens777
    (https://gist.github.com/mjstevens777/9d6771c45f444843f9e3dce6a401b183)
    """

    def __init__(self, channels, kernel_size, stride, groups=1):
        """Set up the layer.
        Parameters
        ----------
        channels: int
            The number of input and output channels
        stride: int or tuple
            The amount of upsampling to do
        groups: int
            Set to 1 for a standard convolution. Set equal to channels to
            make sure there is no cross-talk between channels.
        """
        if isinstance(stride, int):
            stride = (stride, stride)

        assert groups in (1, channels), "Must use no grouping, " + "or one group per channel"

        padding = (stride[0] - 1, stride[1] - 1)
        super().__init__(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
        )

    def reset_parameters(self):
        """Reset the weight and bias."""
        nn.init.constant_(self.bias, 0)
        nn.init.constant_(self.weight, 0)
        bilinear_kernel = self.bilinear_kernel(self.kernel_size[0])
        for i in range(self.in_channels):
            if self.groups == 1:
                j = i
            else:
                j = 0
            self.weight.data[i, j] = bilinear_kernel

    @staticmethod
    def bilinear_kernel(kernel_size):
        """Generate a bilinear upsampling kernel."""
        bilinear_kernel = np.zeros([kernel_size, kernel_size])
        scale_factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = scale_factor - 1
        else:
            center = scale_factor - 0.5
        for x in range(kernel_size):
            for y in range(kernel_size):
                bilinear_kernel[x, y] = (1 - abs(x - center) / scale_factor) * (
                    1 - abs(y - center) / scale_factor
                )

        return torch.Tensor(bilinear_kernel)


class KitModel(nn.Module):
    def __init__(self, weight_file):
        super(KitModel, self).__init__()
        global __weights_dict
        __weights_dict = load_weights(weight_file)

        self.stem_conv_res1_convolution = self.__conv(
            2,
            name="stem_conv_res1/convolution",
            in_channels=3,
            out_channels=32,
            kernel_size=(3, 3),
            stride=(2, 2),
            groups=1,
            bias=None,
        )
        self.stem_bn_res1_FusedBatchNorm_1 = self.__batch_normalization(
            2,
            "stem_bn_res1/FusedBatchNorm_1",
            num_features=32,
            eps=0.0010000000474974513,
            momentum=0.0,
        )
        self.block1a_dwconv_res1_depthwise = self.__conv(
            2,
            name="block1a_dwconv_res1/depthwise",
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=32,
            bias=None,
        )
        self.block1a_bn_res1_FusedBatchNorm_1 = self.__batch_normalization(
            2,
            "block1a_bn_res1/FusedBatchNorm_1",
            num_features=32,
            eps=0.0010000000474974513,
            momentum=0.0,
        )
        self.block1a_se_reduce_res1_convolution = self.__conv(
            2,
            name="block1a_se_reduce_res1/convolution",
            in_channels=32,
            out_channels=8,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.block1a_se_expand_res1_convolution = self.__conv(
            2,
            name="block1a_se_expand_res1/convolution",
            in_channels=8,
            out_channels=32,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.block1a_project_conv_res1_convolution = self.__conv(
            2,
            name="block1a_project_conv_res1/convolution",
            in_channels=32,
            out_channels=16,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.block1a_project_bn_res1_FusedBatchNorm_1 = self.__batch_normalization(
            2,
            "block1a_project_bn_res1/FusedBatchNorm_1",
            num_features=16,
            eps=0.0010000000474974513,
            momentum=0.0,
        )
        self.block2a_expand_conv_res1_convolution = self.__conv(
            2,
            name="block2a_expand_conv_res1/convolution",
            in_channels=16,
            out_channels=96,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.block2a_expand_bn_res1_FusedBatchNorm_1 = self.__batch_normalization(
            2,
            "block2a_expand_bn_res1/FusedBatchNorm_1",
            num_features=96,
            eps=0.0010000000474974513,
            momentum=0.0,
        )
        self.block2a_dwconv_res1_depthwise = self.__conv(
            2,
            name="block2a_dwconv_res1/depthwise",
            in_channels=96,
            out_channels=96,
            kernel_size=(3, 3),
            stride=(2, 2),
            groups=96,
            bias=None,
        )
        self.block2a_bn_res1_FusedBatchNorm_1 = self.__batch_normalization(
            2,
            "block2a_bn_res1/FusedBatchNorm_1",
            num_features=96,
            eps=0.0010000000474974513,
            momentum=0.0,
        )
        self.block2a_se_reduce_res1_convolution = self.__conv(
            2,
            name="block2a_se_reduce_res1/convolution",
            in_channels=96,
            out_channels=4,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.block2a_se_expand_res1_convolution = self.__conv(
            2,
            name="block2a_se_expand_res1/convolution",
            in_channels=4,
            out_channels=96,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.block2a_project_conv_res1_convolution = self.__conv(
            2,
            name="block2a_project_conv_res1/convolution",
            in_channels=96,
            out_channels=24,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.block2a_project_bn_res1_FusedBatchNorm_1 = self.__batch_normalization(
            2,
            "block2a_project_bn_res1/FusedBatchNorm_1",
            num_features=24,
            eps=0.0010000000474974513,
            momentum=0.0,
        )
        self.block2b_expand_conv_res1_convolution = self.__conv(
            2,
            name="block2b_expand_conv_res1/convolution",
            in_channels=24,
            out_channels=144,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.block2b_expand_bn_res1_FusedBatchNorm_1 = self.__batch_normalization(
            2,
            "block2b_expand_bn_res1/FusedBatchNorm_1",
            num_features=144,
            eps=0.0010000000474974513,
            momentum=0.0,
        )
        self.block2b_dwconv_res1_depthwise = self.__conv(
            2,
            name="block2b_dwconv_res1/depthwise",
            in_channels=144,
            out_channels=144,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=144,
            bias=None,
        )
        self.block2b_bn_res1_FusedBatchNorm_1 = self.__batch_normalization(
            2,
            "block2b_bn_res1/FusedBatchNorm_1",
            num_features=144,
            eps=0.0010000000474974513,
            momentum=0.0,
        )
        self.block2b_se_reduce_res1_convolution = self.__conv(
            2,
            name="block2b_se_reduce_res1/convolution",
            in_channels=144,
            out_channels=6,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.block2b_se_expand_res1_convolution = self.__conv(
            2,
            name="block2b_se_expand_res1/convolution",
            in_channels=6,
            out_channels=144,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.block2b_project_conv_res1_convolution = self.__conv(
            2,
            name="block2b_project_conv_res1/convolution",
            in_channels=144,
            out_channels=24,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.block2b_project_bn_res1_FusedBatchNorm_1 = self.__batch_normalization(
            2,
            "block2b_project_bn_res1/FusedBatchNorm_1",
            num_features=24,
            eps=0.0010000000474974513,
            momentum=0.0,
        )
        self.block3a_expand_conv_res1_convolution = self.__conv(
            2,
            name="block3a_expand_conv_res1/convolution",
            in_channels=24,
            out_channels=144,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.block3a_expand_bn_res1_FusedBatchNorm_1 = self.__batch_normalization(
            2,
            "block3a_expand_bn_res1/FusedBatchNorm_1",
            num_features=144,
            eps=0.0010000000474974513,
            momentum=0.0,
        )
        self.block3a_dwconv_res1_depthwise = self.__conv(
            2,
            name="block3a_dwconv_res1/depthwise",
            in_channels=144,
            out_channels=144,
            kernel_size=(5, 5),
            stride=(2, 2),
            groups=144,
            bias=None,
        )
        self.block3a_bn_res1_FusedBatchNorm_1 = self.__batch_normalization(
            2,
            "block3a_bn_res1/FusedBatchNorm_1",
            num_features=144,
            eps=0.0010000000474974513,
            momentum=0.0,
        )
        self.block3a_se_reduce_res1_convolution = self.__conv(
            2,
            name="block3a_se_reduce_res1/convolution",
            in_channels=144,
            out_channels=6,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.block3a_se_expand_res1_convolution = self.__conv(
            2,
            name="block3a_se_expand_res1/convolution",
            in_channels=6,
            out_channels=144,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.block3a_project_conv_res1_convolution = self.__conv(
            2,
            name="block3a_project_conv_res1/convolution",
            in_channels=144,
            out_channels=40,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.block3a_project_bn_res1_FusedBatchNorm_1 = self.__batch_normalization(
            2,
            "block3a_project_bn_res1/FusedBatchNorm_1",
            num_features=40,
            eps=0.0010000000474974513,
            momentum=0.0,
        )
        self.block3b_expand_conv_res1_convolution = self.__conv(
            2,
            name="block3b_expand_conv_res1/convolution",
            in_channels=40,
            out_channels=240,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.block3b_expand_bn_res1_FusedBatchNorm_1 = self.__batch_normalization(
            2,
            "block3b_expand_bn_res1/FusedBatchNorm_1",
            num_features=240,
            eps=0.0010000000474974513,
            momentum=0.0,
        )
        self.block3b_dwconv_res1_depthwise = self.__conv(
            2,
            name="block3b_dwconv_res1/depthwise",
            in_channels=240,
            out_channels=240,
            kernel_size=(5, 5),
            stride=(1, 1),
            groups=240,
            bias=None,
        )
        self.block3b_bn_res1_FusedBatchNorm_1 = self.__batch_normalization(
            2,
            "block3b_bn_res1/FusedBatchNorm_1",
            num_features=240,
            eps=0.0010000000474974513,
            momentum=0.0,
        )
        self.block3b_se_reduce_res1_convolution = self.__conv(
            2,
            name="block3b_se_reduce_res1/convolution",
            in_channels=240,
            out_channels=10,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.block3b_se_expand_res1_convolution = self.__conv(
            2,
            name="block3b_se_expand_res1/convolution",
            in_channels=10,
            out_channels=240,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.block3b_project_conv_res1_convolution = self.__conv(
            2,
            name="block3b_project_conv_res1/convolution",
            in_channels=240,
            out_channels=40,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.block3b_project_bn_res1_FusedBatchNorm_1 = self.__batch_normalization(
            2,
            "block3b_project_bn_res1/FusedBatchNorm_1",
            num_features=40,
            eps=0.0010000000474974513,
            momentum=0.0,
        )
        self.pass1_block1_mbconv1_skeleton_conv1_convolution = self.__conv(
            2,
            name="pass1_block1_mbconv1_skeleton_conv1/convolution",
            in_channels=40,
            out_channels=240,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.pass1_block1_mbconv1_skeleton_conv1_bn_FusedBatchNorm_1 = self.__batch_normalization(
            2,
            "pass1_block1_mbconv1_skeleton_conv1_bn/FusedBatchNorm_1",
            num_features=240,
            eps=0.0010000000474974513,
            momentum=0.0,
        )
        self.pass1_block1_mbconv1_skeleton_dconv1_depthwise = self.__conv(
            2,
            name="pass1_block1_mbconv1_skeleton_dconv1/depthwise",
            in_channels=240,
            out_channels=240,
            kernel_size=(5, 5),
            stride=(1, 1),
            groups=240,
            bias=None,
        )
        self.pass1_block1_mbconv1_skeleton_dconv1_bn_FusedBatchNorm_1 = self.__batch_normalization(
            2,
            "pass1_block1_mbconv1_skeleton_dconv1_bn/FusedBatchNorm_1",
            num_features=240,
            eps=0.0010000000474974513,
            momentum=0.0,
        )
        self.pass1_block1_mbconv1_skeleton_se_se_squeeze_conv_convolution = self.__conv(
            2,
            name="pass1_block1_mbconv1_skeleton_se_se_squeeze_conv/convolution",
            in_channels=240,
            out_channels=10,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.pass1_block1_mbconv1_skeleton_se_se_excite_conv_convolution = self.__conv(
            2,
            name="pass1_block1_mbconv1_skeleton_se_se_excite_conv/convolution",
            in_channels=10,
            out_channels=240,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.pass1_block1_mbconv1_skeleton_conv2_convolution = self.__conv(
            2,
            name="pass1_block1_mbconv1_skeleton_conv2/convolution",
            in_channels=240,
            out_channels=40,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.pass1_block1_mbconv1_skeleton_conv2_bn_FusedBatchNorm_1 = self.__batch_normalization(
            2,
            "pass1_block1_mbconv1_skeleton_conv2_bn/FusedBatchNorm_1",
            num_features=40,
            eps=0.0010000000474974513,
            momentum=0.0,
        )
        self.pass1_block1_mbconv2_skeleton_conv1_convolution = self.__conv(
            2,
            name="pass1_block1_mbconv2_skeleton_conv1/convolution",
            in_channels=40,
            out_channels=240,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.pass1_block1_mbconv2_skeleton_conv1_bn_FusedBatchNorm_1 = self.__batch_normalization(
            2,
            "pass1_block1_mbconv2_skeleton_conv1_bn/FusedBatchNorm_1",
            num_features=240,
            eps=0.0010000000474974513,
            momentum=0.0,
        )
        self.pass1_block1_mbconv2_skeleton_dconv1_depthwise = self.__conv(
            2,
            name="pass1_block1_mbconv2_skeleton_dconv1/depthwise",
            in_channels=240,
            out_channels=240,
            kernel_size=(5, 5),
            stride=(1, 1),
            groups=240,
            bias=None,
        )
        self.pass1_block1_mbconv2_skeleton_dconv1_bn_FusedBatchNorm_1 = self.__batch_normalization(
            2,
            "pass1_block1_mbconv2_skeleton_dconv1_bn/FusedBatchNorm_1",
            num_features=240,
            eps=0.0010000000474974513,
            momentum=0.0,
        )
        self.pass1_block1_mbconv2_skeleton_se_se_squeeze_conv_convolution = self.__conv(
            2,
            name="pass1_block1_mbconv2_skeleton_se_se_squeeze_conv/convolution",
            in_channels=240,
            out_channels=10,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.pass1_block1_mbconv2_skeleton_se_se_excite_conv_convolution = self.__conv(
            2,
            name="pass1_block1_mbconv2_skeleton_se_se_excite_conv/convolution",
            in_channels=10,
            out_channels=240,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.pass1_block1_mbconv2_skeleton_conv2_convolution = self.__conv(
            2,
            name="pass1_block1_mbconv2_skeleton_conv2/convolution",
            in_channels=240,
            out_channels=40,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.pass1_block1_mbconv2_skeleton_conv2_bn_FusedBatchNorm_1 = self.__batch_normalization(
            2,
            "pass1_block1_mbconv2_skeleton_conv2_bn/FusedBatchNorm_1",
            num_features=40,
            eps=0.0010000000474974513,
            momentum=0.0,
        )
        self.pass1_block1_mbconv3_skeleton_conv1_convolution = self.__conv(
            2,
            name="pass1_block1_mbconv3_skeleton_conv1/convolution",
            in_channels=80,
            out_channels=240,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.pass1_block1_mbconv3_skeleton_conv1_bn_FusedBatchNorm_1 = self.__batch_normalization(
            2,
            "pass1_block1_mbconv3_skeleton_conv1_bn/FusedBatchNorm_1",
            num_features=240,
            eps=0.0010000000474974513,
            momentum=0.0,
        )
        self.pass1_block1_mbconv3_skeleton_dconv1_depthwise = self.__conv(
            2,
            name="pass1_block1_mbconv3_skeleton_dconv1/depthwise",
            in_channels=240,
            out_channels=240,
            kernel_size=(5, 5),
            stride=(1, 1),
            groups=240,
            bias=None,
        )
        self.pass1_block1_mbconv3_skeleton_dconv1_bn_FusedBatchNorm_1 = self.__batch_normalization(
            2,
            "pass1_block1_mbconv3_skeleton_dconv1_bn/FusedBatchNorm_1",
            num_features=240,
            eps=0.0010000000474974513,
            momentum=0.0,
        )
        self.pass1_block1_mbconv3_skeleton_se_se_squeeze_conv_convolution = self.__conv(
            2,
            name="pass1_block1_mbconv3_skeleton_se_se_squeeze_conv/convolution",
            in_channels=240,
            out_channels=10,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.pass1_block1_mbconv3_skeleton_se_se_excite_conv_convolution = self.__conv(
            2,
            name="pass1_block1_mbconv3_skeleton_se_se_excite_conv/convolution",
            in_channels=10,
            out_channels=240,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.pass1_block1_mbconv3_skeleton_conv2_convolution = self.__conv(
            2,
            name="pass1_block1_mbconv3_skeleton_conv2/convolution",
            in_channels=240,
            out_channels=40,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.pass1_block1_mbconv3_skeleton_conv2_bn_FusedBatchNorm_1 = self.__batch_normalization(
            2,
            "pass1_block1_mbconv3_skeleton_conv2_bn/FusedBatchNorm_1",
            num_features=40,
            eps=0.0010000000474974513,
            momentum=0.0,
        )
        self.pass2_block1_mbconv1_detection1_conv1_convolution = self.__conv(
            2,
            name="pass2_block1_mbconv1_detection1_conv1/convolution",
            in_channels=160,
            out_channels=240,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.pass2_block1_mbconv1_detection1_conv1_bn_FusedBatchNorm_1 = self.__batch_normalization(
            2,
            "pass2_block1_mbconv1_detection1_conv1_bn/FusedBatchNorm_1",
            num_features=240,
            eps=0.0010000000474974513,
            momentum=0.0,
        )
        self.pass2_block1_mbconv1_detection1_dconv1_depthwise = self.__conv(
            2,
            name="pass2_block1_mbconv1_detection1_dconv1/depthwise",
            in_channels=240,
            out_channels=240,
            kernel_size=(5, 5),
            stride=(1, 1),
            groups=240,
            bias=None,
        )
        self.pass2_block1_mbconv1_detection1_dconv1_bn_FusedBatchNorm_1 = (
            self.__batch_normalization(
                2,
                "pass2_block1_mbconv1_detection1_dconv1_bn/FusedBatchNorm_1",
                num_features=240,
                eps=0.0010000000474974513,
                momentum=0.0,
            )
        )
        self.pass2_block1_mbconv1_detection1_se_se_squeeze_conv_convolution = self.__conv(
            2,
            name="pass2_block1_mbconv1_detection1_se_se_squeeze_conv/convolution",
            in_channels=240,
            out_channels=10,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.pass2_block1_mbconv1_detection1_se_se_excite_conv_convolution = self.__conv(
            2,
            name="pass2_block1_mbconv1_detection1_se_se_excite_conv/convolution",
            in_channels=10,
            out_channels=240,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.pass2_block1_mbconv1_detection1_conv2_convolution = self.__conv(
            2,
            name="pass2_block1_mbconv1_detection1_conv2/convolution",
            in_channels=240,
            out_channels=40,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.pass2_block1_mbconv1_detection1_conv2_bn_FusedBatchNorm_1 = self.__batch_normalization(
            2,
            "pass2_block1_mbconv1_detection1_conv2_bn/FusedBatchNorm_1",
            num_features=40,
            eps=0.0010000000474974513,
            momentum=0.0,
        )
        self.pass2_block1_mbconv2_detection1_conv1_convolution = self.__conv(
            2,
            name="pass2_block1_mbconv2_detection1_conv1/convolution",
            in_channels=40,
            out_channels=240,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.pass2_block1_mbconv2_detection1_conv1_bn_FusedBatchNorm_1 = self.__batch_normalization(
            2,
            "pass2_block1_mbconv2_detection1_conv1_bn/FusedBatchNorm_1",
            num_features=240,
            eps=0.0010000000474974513,
            momentum=0.0,
        )
        self.pass2_block1_mbconv2_detection1_dconv1_depthwise = self.__conv(
            2,
            name="pass2_block1_mbconv2_detection1_dconv1/depthwise",
            in_channels=240,
            out_channels=240,
            kernel_size=(5, 5),
            stride=(1, 1),
            groups=240,
            bias=None,
        )
        self.pass2_block1_mbconv2_detection1_dconv1_bn_FusedBatchNorm_1 = (
            self.__batch_normalization(
                2,
                "pass2_block1_mbconv2_detection1_dconv1_bn/FusedBatchNorm_1",
                num_features=240,
                eps=0.0010000000474974513,
                momentum=0.0,
            )
        )
        self.pass2_block1_mbconv2_detection1_se_se_squeeze_conv_convolution = self.__conv(
            2,
            name="pass2_block1_mbconv2_detection1_se_se_squeeze_conv/convolution",
            in_channels=240,
            out_channels=10,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.pass2_block1_mbconv2_detection1_se_se_excite_conv_convolution = self.__conv(
            2,
            name="pass2_block1_mbconv2_detection1_se_se_excite_conv/convolution",
            in_channels=10,
            out_channels=240,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.pass2_block1_mbconv2_detection1_conv2_convolution = self.__conv(
            2,
            name="pass2_block1_mbconv2_detection1_conv2/convolution",
            in_channels=240,
            out_channels=40,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.pass2_block1_mbconv2_detection1_conv2_bn_FusedBatchNorm_1 = self.__batch_normalization(
            2,
            "pass2_block1_mbconv2_detection1_conv2_bn/FusedBatchNorm_1",
            num_features=40,
            eps=0.0010000000474974513,
            momentum=0.0,
        )
        self.pass2_block1_mbconv3_detection1_conv1_convolution = self.__conv(
            2,
            name="pass2_block1_mbconv3_detection1_conv1/convolution",
            in_channels=80,
            out_channels=240,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.pass2_block1_mbconv3_detection1_conv1_bn_FusedBatchNorm_1 = self.__batch_normalization(
            2,
            "pass2_block1_mbconv3_detection1_conv1_bn/FusedBatchNorm_1",
            num_features=240,
            eps=0.0010000000474974513,
            momentum=0.0,
        )
        self.pass2_block1_mbconv3_detection1_dconv1_depthwise = self.__conv(
            2,
            name="pass2_block1_mbconv3_detection1_dconv1/depthwise",
            in_channels=240,
            out_channels=240,
            kernel_size=(5, 5),
            stride=(1, 1),
            groups=240,
            bias=None,
        )
        self.pass2_block1_mbconv3_detection1_dconv1_bn_FusedBatchNorm_1 = (
            self.__batch_normalization(
                2,
                "pass2_block1_mbconv3_detection1_dconv1_bn/FusedBatchNorm_1",
                num_features=240,
                eps=0.0010000000474974513,
                momentum=0.0,
            )
        )
        self.pass2_block1_mbconv3_detection1_se_se_squeeze_conv_convolution = self.__conv(
            2,
            name="pass2_block1_mbconv3_detection1_se_se_squeeze_conv/convolution",
            in_channels=240,
            out_channels=10,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.pass2_block1_mbconv3_detection1_se_se_excite_conv_convolution = self.__conv(
            2,
            name="pass2_block1_mbconv3_detection1_se_se_excite_conv/convolution",
            in_channels=10,
            out_channels=240,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.pass2_block1_mbconv3_detection1_conv2_convolution = self.__conv(
            2,
            name="pass2_block1_mbconv3_detection1_conv2/convolution",
            in_channels=240,
            out_channels=40,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.pass2_block1_mbconv3_detection1_conv2_bn_FusedBatchNorm_1 = self.__batch_normalization(
            2,
            "pass2_block1_mbconv3_detection1_conv2_bn/FusedBatchNorm_1",
            num_features=40,
            eps=0.0010000000474974513,
            momentum=0.0,
        )
        self.pass3_block1_mbconv1_detection2_conv1_convolution = self.__conv(
            2,
            name="pass3_block1_mbconv1_detection2_conv1/convolution",
            in_channels=160,
            out_channels=240,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.pass3_block1_mbconv1_detection2_conv1_bn_FusedBatchNorm_1 = self.__batch_normalization(
            2,
            "pass3_block1_mbconv1_detection2_conv1_bn/FusedBatchNorm_1",
            num_features=240,
            eps=0.0010000000474974513,
            momentum=0.0,
        )
        self.pass3_block1_mbconv1_detection2_dconv1_depthwise = self.__conv(
            2,
            name="pass3_block1_mbconv1_detection2_dconv1/depthwise",
            in_channels=240,
            out_channels=240,
            kernel_size=(5, 5),
            stride=(1, 1),
            groups=240,
            bias=None,
        )
        self.pass3_block1_mbconv1_detection2_dconv1_bn_FusedBatchNorm_1 = (
            self.__batch_normalization(
                2,
                "pass3_block1_mbconv1_detection2_dconv1_bn/FusedBatchNorm_1",
                num_features=240,
                eps=0.0010000000474974513,
                momentum=0.0,
            )
        )
        self.pass3_block1_mbconv1_detection2_se_se_squeeze_conv_convolution = self.__conv(
            2,
            name="pass3_block1_mbconv1_detection2_se_se_squeeze_conv/convolution",
            in_channels=240,
            out_channels=10,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.pass3_block1_mbconv1_detection2_se_se_excite_conv_convolution = self.__conv(
            2,
            name="pass3_block1_mbconv1_detection2_se_se_excite_conv/convolution",
            in_channels=10,
            out_channels=240,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.pass3_block1_mbconv1_detection2_conv2_convolution = self.__conv(
            2,
            name="pass3_block1_mbconv1_detection2_conv2/convolution",
            in_channels=240,
            out_channels=40,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.pass3_block1_mbconv1_detection2_conv2_bn_FusedBatchNorm_1 = self.__batch_normalization(
            2,
            "pass3_block1_mbconv1_detection2_conv2_bn/FusedBatchNorm_1",
            num_features=40,
            eps=0.0010000000474974513,
            momentum=0.0,
        )
        self.pass3_block1_mbconv2_detection2_conv1_convolution = self.__conv(
            2,
            name="pass3_block1_mbconv2_detection2_conv1/convolution",
            in_channels=40,
            out_channels=240,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.pass3_block1_mbconv2_detection2_conv1_bn_FusedBatchNorm_1 = self.__batch_normalization(
            2,
            "pass3_block1_mbconv2_detection2_conv1_bn/FusedBatchNorm_1",
            num_features=240,
            eps=0.0010000000474974513,
            momentum=0.0,
        )
        self.pass3_block1_mbconv2_detection2_dconv1_depthwise = self.__conv(
            2,
            name="pass3_block1_mbconv2_detection2_dconv1/depthwise",
            in_channels=240,
            out_channels=240,
            kernel_size=(5, 5),
            stride=(1, 1),
            groups=240,
            bias=None,
        )
        self.pass3_block1_mbconv2_detection2_dconv1_bn_FusedBatchNorm_1 = (
            self.__batch_normalization(
                2,
                "pass3_block1_mbconv2_detection2_dconv1_bn/FusedBatchNorm_1",
                num_features=240,
                eps=0.0010000000474974513,
                momentum=0.0,
            )
        )
        self.pass3_block1_mbconv2_detection2_se_se_squeeze_conv_convolution = self.__conv(
            2,
            name="pass3_block1_mbconv2_detection2_se_se_squeeze_conv/convolution",
            in_channels=240,
            out_channels=10,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.pass3_block1_mbconv2_detection2_se_se_excite_conv_convolution = self.__conv(
            2,
            name="pass3_block1_mbconv2_detection2_se_se_excite_conv/convolution",
            in_channels=10,
            out_channels=240,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.pass3_block1_mbconv2_detection2_conv2_convolution = self.__conv(
            2,
            name="pass3_block1_mbconv2_detection2_conv2/convolution",
            in_channels=240,
            out_channels=40,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.pass3_block1_mbconv2_detection2_conv2_bn_FusedBatchNorm_1 = self.__batch_normalization(
            2,
            "pass3_block1_mbconv2_detection2_conv2_bn/FusedBatchNorm_1",
            num_features=40,
            eps=0.0010000000474974513,
            momentum=0.0,
        )
        self.pass3_block1_mbconv3_detection2_conv1_convolution = self.__conv(
            2,
            name="pass3_block1_mbconv3_detection2_conv1/convolution",
            in_channels=80,
            out_channels=240,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.pass3_block1_mbconv3_detection2_conv1_bn_FusedBatchNorm_1 = self.__batch_normalization(
            2,
            "pass3_block1_mbconv3_detection2_conv1_bn/FusedBatchNorm_1",
            num_features=240,
            eps=0.0010000000474974513,
            momentum=0.0,
        )
        self.pass3_block1_mbconv3_detection2_dconv1_depthwise = self.__conv(
            2,
            name="pass3_block1_mbconv3_detection2_dconv1/depthwise",
            in_channels=240,
            out_channels=240,
            kernel_size=(5, 5),
            stride=(1, 1),
            groups=240,
            bias=None,
        )
        self.pass3_block1_mbconv3_detection2_dconv1_bn_FusedBatchNorm_1 = (
            self.__batch_normalization(
                2,
                "pass3_block1_mbconv3_detection2_dconv1_bn/FusedBatchNorm_1",
                num_features=240,
                eps=0.0010000000474974513,
                momentum=0.0,
            )
        )
        self.pass3_block1_mbconv3_detection2_se_se_squeeze_conv_convolution = self.__conv(
            2,
            name="pass3_block1_mbconv3_detection2_se_se_squeeze_conv/convolution",
            in_channels=240,
            out_channels=10,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.pass3_block1_mbconv3_detection2_se_se_excite_conv_convolution = self.__conv(
            2,
            name="pass3_block1_mbconv3_detection2_se_se_excite_conv/convolution",
            in_channels=10,
            out_channels=240,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.pass3_block1_mbconv3_detection2_conv2_convolution = self.__conv(
            2,
            name="pass3_block1_mbconv3_detection2_conv2/convolution",
            in_channels=240,
            out_channels=40,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.pass3_block1_mbconv3_detection2_conv2_bn_FusedBatchNorm_1 = self.__batch_normalization(
            2,
            "pass3_block1_mbconv3_detection2_conv2_bn/FusedBatchNorm_1",
            num_features=40,
            eps=0.0010000000474974513,
            momentum=0.0,
        )
        self.pass3_detection2_confs_convolution = self.__conv(
            2,
            name="pass3_detection2_confs/convolution",
            in_channels=120,
            out_channels=16,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )

    def forward(self, x):
        self.pass1_block1_mbconv1_skeleton_conv1_eswish_mul_x = torch.autograd.Variable(
            torch.Tensor([1.25]), requires_grad=False
        )
        self.pass1_block1_mbconv1_skeleton_dconv1_eswish_mul_x = torch.autograd.Variable(
            torch.Tensor([1.25]), requires_grad=False
        )
        self.pass1_block1_mbconv1_skeleton_se_se_squeeze_eswish_mul_x = torch.autograd.Variable(
            torch.Tensor([1.25]), requires_grad=False
        )
        self.pass1_block1_mbconv2_skeleton_conv1_eswish_mul_x = torch.autograd.Variable(
            torch.Tensor([1.25]), requires_grad=False
        )
        self.pass1_block1_mbconv2_skeleton_dconv1_eswish_mul_x = torch.autograd.Variable(
            torch.Tensor([1.25]), requires_grad=False
        )
        self.pass1_block1_mbconv2_skeleton_se_se_squeeze_eswish_mul_x = torch.autograd.Variable(
            torch.Tensor([1.25]), requires_grad=False
        )
        self.pass1_block1_mbconv3_skeleton_conv1_eswish_mul_x = torch.autograd.Variable(
            torch.Tensor([1.25]), requires_grad=False
        )
        self.pass1_block1_mbconv3_skeleton_dconv1_eswish_mul_x = torch.autograd.Variable(
            torch.Tensor([1.25]), requires_grad=False
        )
        self.pass1_block1_mbconv3_skeleton_se_se_squeeze_eswish_mul_x = torch.autograd.Variable(
            torch.Tensor([1.25]), requires_grad=False
        )
        self.pass2_block1_mbconv1_detection1_conv1_eswish_mul_x = torch.autograd.Variable(
            torch.Tensor([1.25]), requires_grad=False
        )
        self.pass2_block1_mbconv1_detection1_dconv1_eswish_mul_x = torch.autograd.Variable(
            torch.Tensor([1.25]), requires_grad=False
        )
        self.pass2_block1_mbconv1_detection1_se_se_squeeze_eswish_mul_x = torch.autograd.Variable(
            torch.Tensor([1.25]), requires_grad=False
        )
        self.pass2_block1_mbconv2_detection1_conv1_eswish_mul_x = torch.autograd.Variable(
            torch.Tensor([1.25]), requires_grad=False
        )
        self.pass2_block1_mbconv2_detection1_dconv1_eswish_mul_x = torch.autograd.Variable(
            torch.Tensor([1.25]), requires_grad=False
        )
        self.pass2_block1_mbconv2_detection1_se_se_squeeze_eswish_mul_x = torch.autograd.Variable(
            torch.Tensor([1.25]), requires_grad=False
        )
        self.pass2_block1_mbconv3_detection1_conv1_eswish_mul_x = torch.autograd.Variable(
            torch.Tensor([1.25]), requires_grad=False
        )
        self.pass2_block1_mbconv3_detection1_dconv1_eswish_mul_x = torch.autograd.Variable(
            torch.Tensor([1.25]), requires_grad=False
        )
        self.pass2_block1_mbconv3_detection1_se_se_squeeze_eswish_mul_x = torch.autograd.Variable(
            torch.Tensor([1.25]), requires_grad=False
        )
        self.pass3_block1_mbconv1_detection2_conv1_eswish_mul_x = torch.autograd.Variable(
            torch.Tensor([1.25]), requires_grad=False
        )
        self.pass3_block1_mbconv1_detection2_dconv1_eswish_mul_x = torch.autograd.Variable(
            torch.Tensor([1.25]), requires_grad=False
        )
        self.pass3_block1_mbconv1_detection2_se_se_squeeze_eswish_mul_x = torch.autograd.Variable(
            torch.Tensor([1.25]), requires_grad=False
        )
        self.pass3_block1_mbconv2_detection2_conv1_eswish_mul_x = torch.autograd.Variable(
            torch.Tensor([1.25]), requires_grad=False
        )
        self.pass3_block1_mbconv2_detection2_dconv1_eswish_mul_x = torch.autograd.Variable(
            torch.Tensor([1.25]), requires_grad=False
        )
        self.pass3_block1_mbconv2_detection2_se_se_squeeze_eswish_mul_x = torch.autograd.Variable(
            torch.Tensor([1.25]), requires_grad=False
        )
        self.pass3_block1_mbconv3_detection2_conv1_eswish_mul_x = torch.autograd.Variable(
            torch.Tensor([1.25]), requires_grad=False
        )
        self.pass3_block1_mbconv3_detection2_dconv1_eswish_mul_x = torch.autograd.Variable(
            torch.Tensor([1.25]), requires_grad=False
        )
        self.pass3_block1_mbconv3_detection2_se_se_squeeze_eswish_mul_x = torch.autograd.Variable(
            torch.Tensor([1.25]), requires_grad=False
        )
        stem_conv_res1_convolution_pad = F.pad(x, (0, 1, 0, 1))
        stem_conv_res1_convolution = self.stem_conv_res1_convolution(stem_conv_res1_convolution_pad)
        stem_bn_res1_FusedBatchNorm_1 = self.stem_bn_res1_FusedBatchNorm_1(
            stem_conv_res1_convolution
        )
        stem_activation_res1_Sigmoid = F.sigmoid(stem_bn_res1_FusedBatchNorm_1)
        stem_activation_res1_mul = stem_bn_res1_FusedBatchNorm_1 * stem_activation_res1_Sigmoid
        block1a_dwconv_res1_depthwise_pad = F.pad(stem_activation_res1_mul, (1, 1, 1, 1))
        block1a_dwconv_res1_depthwise = self.block1a_dwconv_res1_depthwise(
            block1a_dwconv_res1_depthwise_pad
        )
        block1a_bn_res1_FusedBatchNorm_1 = self.block1a_bn_res1_FusedBatchNorm_1(
            block1a_dwconv_res1_depthwise
        )
        block1a_activation_res1_Sigmoid = F.sigmoid(block1a_bn_res1_FusedBatchNorm_1)
        block1a_activation_res1_mul = (
            block1a_bn_res1_FusedBatchNorm_1 * block1a_activation_res1_Sigmoid
        )
        block1a_se_squeeze_res1_Mean = torch.mean(block1a_activation_res1_mul, 3, False)
        block1a_se_squeeze_res1_Mean = torch.mean(block1a_se_squeeze_res1_Mean, 2, False)
        block1a_se_reshape_res1_Shape = torch.Tensor(list(block1a_se_squeeze_res1_Mean.size()))
        block1a_se_reshape_res1_Reshape = torch.reshape(
            input=block1a_se_squeeze_res1_Mean, shape=(-1, 32, 1, 1)
        )  # (-1,1,1,32))
        block1a_se_reshape_res1_strided_slice = block1a_se_reshape_res1_Shape[0:1]  # noqa: F841 (kept verbatim from real repo)
        block1a_se_reduce_res1_convolution = self.block1a_se_reduce_res1_convolution(
            block1a_se_reshape_res1_Reshape
        )
        block1a_se_reduce_swish_res1_Sigmoid = F.sigmoid(block1a_se_reduce_res1_convolution)
        block1a_se_reduce_swish_res1_mul = (
            block1a_se_reduce_res1_convolution * block1a_se_reduce_swish_res1_Sigmoid
        )
        block1a_se_expand_res1_convolution = self.block1a_se_expand_res1_convolution(
            block1a_se_reduce_swish_res1_mul
        )
        block1a_se_expand_res1_Sigmoid = F.sigmoid(block1a_se_expand_res1_convolution)
        block1a_se_excite_res1_mul = block1a_activation_res1_mul * block1a_se_expand_res1_Sigmoid
        block1a_project_conv_res1_convolution = self.block1a_project_conv_res1_convolution(
            block1a_se_excite_res1_mul
        )
        block1a_project_bn_res1_FusedBatchNorm_1 = self.block1a_project_bn_res1_FusedBatchNorm_1(
            block1a_project_conv_res1_convolution
        )
        block2a_expand_conv_res1_convolution = self.block2a_expand_conv_res1_convolution(
            block1a_project_bn_res1_FusedBatchNorm_1
        )
        block2a_expand_bn_res1_FusedBatchNorm_1 = self.block2a_expand_bn_res1_FusedBatchNorm_1(
            block2a_expand_conv_res1_convolution
        )
        block2a_expand_activation_res1_Sigmoid = F.sigmoid(block2a_expand_bn_res1_FusedBatchNorm_1)
        block2a_expand_activation_res1_mul = (
            block2a_expand_bn_res1_FusedBatchNorm_1 * block2a_expand_activation_res1_Sigmoid
        )
        block2a_dwconv_res1_depthwise_pad = F.pad(block2a_expand_activation_res1_mul, (0, 1, 0, 1))
        block2a_dwconv_res1_depthwise = self.block2a_dwconv_res1_depthwise(
            block2a_dwconv_res1_depthwise_pad
        )
        block2a_bn_res1_FusedBatchNorm_1 = self.block2a_bn_res1_FusedBatchNorm_1(
            block2a_dwconv_res1_depthwise
        )
        block2a_activation_res1_Sigmoid = F.sigmoid(block2a_bn_res1_FusedBatchNorm_1)
        block2a_activation_res1_mul = (
            block2a_bn_res1_FusedBatchNorm_1 * block2a_activation_res1_Sigmoid
        )
        block2a_se_squeeze_res1_Mean = torch.mean(block2a_activation_res1_mul, 3, False)
        block2a_se_squeeze_res1_Mean = torch.mean(block2a_se_squeeze_res1_Mean, 2, False)
        block2a_se_reshape_res1_Shape = torch.Tensor(list(block2a_se_squeeze_res1_Mean.size()))
        block2a_se_reshape_res1_Reshape = torch.reshape(
            input=block2a_se_squeeze_res1_Mean, shape=(-1, 96, 1, 1)
        )  # (-1,1,1,96))
        block2a_se_reshape_res1_strided_slice = block2a_se_reshape_res1_Shape[0:1]  # noqa: F841 (kept verbatim from real repo)
        block2a_se_reduce_res1_convolution = self.block2a_se_reduce_res1_convolution(
            block2a_se_reshape_res1_Reshape
        )
        block2a_se_reduce_swish_res1_Sigmoid = F.sigmoid(block2a_se_reduce_res1_convolution)
        block2a_se_reduce_swish_res1_mul = (
            block2a_se_reduce_res1_convolution * block2a_se_reduce_swish_res1_Sigmoid
        )
        block2a_se_expand_res1_convolution = self.block2a_se_expand_res1_convolution(
            block2a_se_reduce_swish_res1_mul
        )
        block2a_se_expand_res1_Sigmoid = F.sigmoid(block2a_se_expand_res1_convolution)
        block2a_se_excite_res1_mul = block2a_activation_res1_mul * block2a_se_expand_res1_Sigmoid
        block2a_project_conv_res1_convolution = self.block2a_project_conv_res1_convolution(
            block2a_se_excite_res1_mul
        )
        block2a_project_bn_res1_FusedBatchNorm_1 = self.block2a_project_bn_res1_FusedBatchNorm_1(
            block2a_project_conv_res1_convolution
        )
        block2b_expand_conv_res1_convolution = self.block2b_expand_conv_res1_convolution(
            block2a_project_bn_res1_FusedBatchNorm_1
        )
        block2b_expand_bn_res1_FusedBatchNorm_1 = self.block2b_expand_bn_res1_FusedBatchNorm_1(
            block2b_expand_conv_res1_convolution
        )
        block2b_expand_activation_res1_Sigmoid = F.sigmoid(block2b_expand_bn_res1_FusedBatchNorm_1)
        block2b_expand_activation_res1_mul = (
            block2b_expand_bn_res1_FusedBatchNorm_1 * block2b_expand_activation_res1_Sigmoid
        )
        block2b_dwconv_res1_depthwise_pad = F.pad(block2b_expand_activation_res1_mul, (1, 1, 1, 1))
        block2b_dwconv_res1_depthwise = self.block2b_dwconv_res1_depthwise(
            block2b_dwconv_res1_depthwise_pad
        )
        block2b_bn_res1_FusedBatchNorm_1 = self.block2b_bn_res1_FusedBatchNorm_1(
            block2b_dwconv_res1_depthwise
        )
        block2b_activation_res1_Sigmoid = F.sigmoid(block2b_bn_res1_FusedBatchNorm_1)
        block2b_activation_res1_mul = (
            block2b_bn_res1_FusedBatchNorm_1 * block2b_activation_res1_Sigmoid
        )
        block2b_se_squeeze_res1_Mean = torch.mean(block2b_activation_res1_mul, 3, False)
        block2b_se_squeeze_res1_Mean = torch.mean(block2b_se_squeeze_res1_Mean, 2, False)
        block2b_se_reshape_res1_Shape = torch.Tensor(list(block2b_se_squeeze_res1_Mean.size()))
        block2b_se_reshape_res1_Reshape = torch.reshape(
            input=block2b_se_squeeze_res1_Mean, shape=(-1, 144, 1, 1)
        )  # (-1,1,1,144))
        block2b_se_reshape_res1_strided_slice = block2b_se_reshape_res1_Shape[0:1]  # noqa: F841 (kept verbatim from real repo)
        block2b_se_reduce_res1_convolution = self.block2b_se_reduce_res1_convolution(
            block2b_se_reshape_res1_Reshape
        )
        block2b_se_reduce_swish_res1_Sigmoid = F.sigmoid(block2b_se_reduce_res1_convolution)
        block2b_se_reduce_swish_res1_mul = (
            block2b_se_reduce_res1_convolution * block2b_se_reduce_swish_res1_Sigmoid
        )
        block2b_se_expand_res1_convolution = self.block2b_se_expand_res1_convolution(
            block2b_se_reduce_swish_res1_mul
        )
        block2b_se_expand_res1_Sigmoid = F.sigmoid(block2b_se_expand_res1_convolution)
        block2b_se_excite_res1_mul = block2b_activation_res1_mul * block2b_se_expand_res1_Sigmoid
        block2b_project_conv_res1_convolution = self.block2b_project_conv_res1_convolution(
            block2b_se_excite_res1_mul
        )
        block2b_project_bn_res1_FusedBatchNorm_1 = self.block2b_project_bn_res1_FusedBatchNorm_1(
            block2b_project_conv_res1_convolution
        )
        block2b_add_res1_add = (
            block2b_project_bn_res1_FusedBatchNorm_1 + block2a_project_bn_res1_FusedBatchNorm_1
        )
        block3a_expand_conv_res1_convolution = self.block3a_expand_conv_res1_convolution(
            block2b_add_res1_add
        )
        block3a_expand_bn_res1_FusedBatchNorm_1 = self.block3a_expand_bn_res1_FusedBatchNorm_1(
            block3a_expand_conv_res1_convolution
        )
        block3a_expand_activation_res1_Sigmoid = F.sigmoid(block3a_expand_bn_res1_FusedBatchNorm_1)
        block3a_expand_activation_res1_mul = (
            block3a_expand_bn_res1_FusedBatchNorm_1 * block3a_expand_activation_res1_Sigmoid
        )
        block3a_dwconv_res1_depthwise_pad = F.pad(block3a_expand_activation_res1_mul, (1, 2, 1, 2))
        block3a_dwconv_res1_depthwise = self.block3a_dwconv_res1_depthwise(
            block3a_dwconv_res1_depthwise_pad
        )
        block3a_bn_res1_FusedBatchNorm_1 = self.block3a_bn_res1_FusedBatchNorm_1(
            block3a_dwconv_res1_depthwise
        )
        block3a_activation_res1_Sigmoid = F.sigmoid(block3a_bn_res1_FusedBatchNorm_1)
        block3a_activation_res1_mul = (
            block3a_bn_res1_FusedBatchNorm_1 * block3a_activation_res1_Sigmoid
        )
        block3a_se_squeeze_res1_Mean = torch.mean(block3a_activation_res1_mul, 3, False)
        block3a_se_squeeze_res1_Mean = torch.mean(block3a_se_squeeze_res1_Mean, 2, False)
        block3a_se_reshape_res1_Shape = torch.Tensor(list(block3a_se_squeeze_res1_Mean.size()))
        block3a_se_reshape_res1_Reshape = torch.reshape(
            input=block3a_se_squeeze_res1_Mean, shape=(-1, 144, 1, 1)
        )  # (-1,1,1,144))
        block3a_se_reshape_res1_strided_slice = block3a_se_reshape_res1_Shape[0:1]  # noqa: F841 (kept verbatim from real repo)
        block3a_se_reduce_res1_convolution = self.block3a_se_reduce_res1_convolution(
            block3a_se_reshape_res1_Reshape
        )
        block3a_se_reduce_swish_res1_Sigmoid = F.sigmoid(block3a_se_reduce_res1_convolution)
        block3a_se_reduce_swish_res1_mul = (
            block3a_se_reduce_res1_convolution * block3a_se_reduce_swish_res1_Sigmoid
        )
        block3a_se_expand_res1_convolution = self.block3a_se_expand_res1_convolution(
            block3a_se_reduce_swish_res1_mul
        )
        block3a_se_expand_res1_Sigmoid = F.sigmoid(block3a_se_expand_res1_convolution)
        block3a_se_excite_res1_mul = block3a_activation_res1_mul * block3a_se_expand_res1_Sigmoid
        block3a_project_conv_res1_convolution = self.block3a_project_conv_res1_convolution(
            block3a_se_excite_res1_mul
        )
        block3a_project_bn_res1_FusedBatchNorm_1 = self.block3a_project_bn_res1_FusedBatchNorm_1(
            block3a_project_conv_res1_convolution
        )
        block3b_expand_conv_res1_convolution = self.block3b_expand_conv_res1_convolution(
            block3a_project_bn_res1_FusedBatchNorm_1
        )
        block3b_expand_bn_res1_FusedBatchNorm_1 = self.block3b_expand_bn_res1_FusedBatchNorm_1(
            block3b_expand_conv_res1_convolution
        )
        block3b_expand_activation_res1_Sigmoid = F.sigmoid(block3b_expand_bn_res1_FusedBatchNorm_1)
        block3b_expand_activation_res1_mul = (
            block3b_expand_bn_res1_FusedBatchNorm_1 * block3b_expand_activation_res1_Sigmoid
        )
        block3b_dwconv_res1_depthwise_pad = F.pad(block3b_expand_activation_res1_mul, (2, 2, 2, 2))
        block3b_dwconv_res1_depthwise = self.block3b_dwconv_res1_depthwise(
            block3b_dwconv_res1_depthwise_pad
        )
        block3b_bn_res1_FusedBatchNorm_1 = self.block3b_bn_res1_FusedBatchNorm_1(
            block3b_dwconv_res1_depthwise
        )
        block3b_activation_res1_Sigmoid = F.sigmoid(block3b_bn_res1_FusedBatchNorm_1)
        block3b_activation_res1_mul = (
            block3b_bn_res1_FusedBatchNorm_1 * block3b_activation_res1_Sigmoid
        )
        block3b_se_squeeze_res1_Mean = torch.mean(block3b_activation_res1_mul, 3, False)
        block3b_se_squeeze_res1_Mean = torch.mean(block3b_se_squeeze_res1_Mean, 2, False)
        block3b_se_reshape_res1_Shape = torch.Tensor(list(block3b_se_squeeze_res1_Mean.size()))
        block3b_se_reshape_res1_Reshape = torch.reshape(
            input=block3b_se_squeeze_res1_Mean, shape=(-1, 240, 1, 1)
        )  # (-1,1,1,240))
        block3b_se_reshape_res1_strided_slice = block3b_se_reshape_res1_Shape[0:1]  # noqa: F841 (kept verbatim from real repo)
        block3b_se_reduce_res1_convolution = self.block3b_se_reduce_res1_convolution(
            block3b_se_reshape_res1_Reshape
        )
        block3b_se_reduce_swish_res1_Sigmoid = F.sigmoid(block3b_se_reduce_res1_convolution)
        block3b_se_reduce_swish_res1_mul = (
            block3b_se_reduce_res1_convolution * block3b_se_reduce_swish_res1_Sigmoid
        )
        block3b_se_expand_res1_convolution = self.block3b_se_expand_res1_convolution(
            block3b_se_reduce_swish_res1_mul
        )
        block3b_se_expand_res1_Sigmoid = F.sigmoid(block3b_se_expand_res1_convolution)
        block3b_se_excite_res1_mul = block3b_activation_res1_mul * block3b_se_expand_res1_Sigmoid
        block3b_project_conv_res1_convolution = self.block3b_project_conv_res1_convolution(
            block3b_se_excite_res1_mul
        )
        block3b_project_bn_res1_FusedBatchNorm_1 = self.block3b_project_bn_res1_FusedBatchNorm_1(
            block3b_project_conv_res1_convolution
        )
        block3b_add_res1_add = (
            block3b_project_bn_res1_FusedBatchNorm_1 + block3a_project_bn_res1_FusedBatchNorm_1
        )
        pass1_block1_mbconv1_skeleton_conv1_convolution = (
            self.pass1_block1_mbconv1_skeleton_conv1_convolution(block3b_add_res1_add)
        )
        pass1_block1_mbconv1_skeleton_conv1_bn_FusedBatchNorm_1 = (
            self.pass1_block1_mbconv1_skeleton_conv1_bn_FusedBatchNorm_1(
                pass1_block1_mbconv1_skeleton_conv1_convolution
            )
        )
        pass1_block1_mbconv1_skeleton_conv1_eswish_mul = (
            self.pass1_block1_mbconv1_skeleton_conv1_eswish_mul_x
            * pass1_block1_mbconv1_skeleton_conv1_bn_FusedBatchNorm_1
        )
        pass1_block1_mbconv1_skeleton_conv1_eswish_Sigmoid = F.sigmoid(
            pass1_block1_mbconv1_skeleton_conv1_bn_FusedBatchNorm_1
        )
        pass1_block1_mbconv1_skeleton_conv1_eswish_mul_1 = (
            pass1_block1_mbconv1_skeleton_conv1_eswish_mul
            * pass1_block1_mbconv1_skeleton_conv1_eswish_Sigmoid
        )
        pass1_block1_mbconv1_skeleton_dconv1_depthwise_pad = F.pad(
            pass1_block1_mbconv1_skeleton_conv1_eswish_mul_1, (2, 2, 2, 2)
        )
        pass1_block1_mbconv1_skeleton_dconv1_depthwise = (
            self.pass1_block1_mbconv1_skeleton_dconv1_depthwise(
                pass1_block1_mbconv1_skeleton_dconv1_depthwise_pad
            )
        )
        pass1_block1_mbconv1_skeleton_dconv1_bn_FusedBatchNorm_1 = (
            self.pass1_block1_mbconv1_skeleton_dconv1_bn_FusedBatchNorm_1(
                pass1_block1_mbconv1_skeleton_dconv1_depthwise
            )
        )
        pass1_block1_mbconv1_skeleton_dconv1_eswish_mul = (
            self.pass1_block1_mbconv1_skeleton_dconv1_eswish_mul_x
            * pass1_block1_mbconv1_skeleton_dconv1_bn_FusedBatchNorm_1
        )
        pass1_block1_mbconv1_skeleton_dconv1_eswish_Sigmoid = F.sigmoid(
            pass1_block1_mbconv1_skeleton_dconv1_bn_FusedBatchNorm_1
        )
        pass1_block1_mbconv1_skeleton_dconv1_eswish_mul_1 = (
            pass1_block1_mbconv1_skeleton_dconv1_eswish_mul
            * pass1_block1_mbconv1_skeleton_dconv1_eswish_Sigmoid
        )
        pass1_block1_mbconv1_skeleton_se_se_squeeze_lambda_Mean = torch.mean(
            pass1_block1_mbconv1_skeleton_dconv1_eswish_mul_1, 3, True
        )
        pass1_block1_mbconv1_skeleton_se_se_squeeze_lambda_Mean = torch.mean(
            pass1_block1_mbconv1_skeleton_se_se_squeeze_lambda_Mean, 2, True
        )
        pass1_block1_mbconv1_skeleton_se_se_squeeze_conv_convolution = (
            self.pass1_block1_mbconv1_skeleton_se_se_squeeze_conv_convolution(
                pass1_block1_mbconv1_skeleton_se_se_squeeze_lambda_Mean
            )
        )
        pass1_block1_mbconv1_skeleton_se_se_squeeze_eswish_mul = (
            self.pass1_block1_mbconv1_skeleton_se_se_squeeze_eswish_mul_x
            * pass1_block1_mbconv1_skeleton_se_se_squeeze_conv_convolution
        )
        pass1_block1_mbconv1_skeleton_se_se_squeeze_eswish_Sigmoid = F.sigmoid(
            pass1_block1_mbconv1_skeleton_se_se_squeeze_conv_convolution
        )
        pass1_block1_mbconv1_skeleton_se_se_squeeze_eswish_mul_1 = (
            pass1_block1_mbconv1_skeleton_se_se_squeeze_eswish_mul
            * pass1_block1_mbconv1_skeleton_se_se_squeeze_eswish_Sigmoid
        )
        pass1_block1_mbconv1_skeleton_se_se_excite_conv_convolution = (
            self.pass1_block1_mbconv1_skeleton_se_se_excite_conv_convolution(
                pass1_block1_mbconv1_skeleton_se_se_squeeze_eswish_mul_1
            )
        )
        pass1_block1_mbconv1_skeleton_se_se_excite_sigmoid_Sigmoid = F.sigmoid(
            pass1_block1_mbconv1_skeleton_se_se_excite_conv_convolution
        )
        pass1_block1_mbconv1_skeleton_se_se_multiply_mul = (
            pass1_block1_mbconv1_skeleton_se_se_excite_sigmoid_Sigmoid
            * pass1_block1_mbconv1_skeleton_dconv1_eswish_mul_1
        )
        pass1_block1_mbconv1_skeleton_conv2_convolution = (
            self.pass1_block1_mbconv1_skeleton_conv2_convolution(
                pass1_block1_mbconv1_skeleton_se_se_multiply_mul
            )
        )
        pass1_block1_mbconv1_skeleton_conv2_bn_FusedBatchNorm_1 = (
            self.pass1_block1_mbconv1_skeleton_conv2_bn_FusedBatchNorm_1(
                pass1_block1_mbconv1_skeleton_conv2_convolution
            )
        )
        pass1_block1_mbconv2_skeleton_conv1_convolution = (
            self.pass1_block1_mbconv2_skeleton_conv1_convolution(
                pass1_block1_mbconv1_skeleton_conv2_bn_FusedBatchNorm_1
            )
        )
        pass1_block1_mbconv2_skeleton_conv1_bn_FusedBatchNorm_1 = (
            self.pass1_block1_mbconv2_skeleton_conv1_bn_FusedBatchNorm_1(
                pass1_block1_mbconv2_skeleton_conv1_convolution
            )
        )
        pass1_block1_mbconv2_skeleton_conv1_eswish_mul = (
            self.pass1_block1_mbconv2_skeleton_conv1_eswish_mul_x
            * pass1_block1_mbconv2_skeleton_conv1_bn_FusedBatchNorm_1
        )
        pass1_block1_mbconv2_skeleton_conv1_eswish_Sigmoid = F.sigmoid(
            pass1_block1_mbconv2_skeleton_conv1_bn_FusedBatchNorm_1
        )
        pass1_block1_mbconv2_skeleton_conv1_eswish_mul_1 = (
            pass1_block1_mbconv2_skeleton_conv1_eswish_mul
            * pass1_block1_mbconv2_skeleton_conv1_eswish_Sigmoid
        )
        pass1_block1_mbconv2_skeleton_dconv1_depthwise_pad = F.pad(
            pass1_block1_mbconv2_skeleton_conv1_eswish_mul_1, (2, 2, 2, 2)
        )
        pass1_block1_mbconv2_skeleton_dconv1_depthwise = (
            self.pass1_block1_mbconv2_skeleton_dconv1_depthwise(
                pass1_block1_mbconv2_skeleton_dconv1_depthwise_pad
            )
        )
        pass1_block1_mbconv2_skeleton_dconv1_bn_FusedBatchNorm_1 = (
            self.pass1_block1_mbconv2_skeleton_dconv1_bn_FusedBatchNorm_1(
                pass1_block1_mbconv2_skeleton_dconv1_depthwise
            )
        )
        pass1_block1_mbconv2_skeleton_dconv1_eswish_mul = (
            self.pass1_block1_mbconv2_skeleton_dconv1_eswish_mul_x
            * pass1_block1_mbconv2_skeleton_dconv1_bn_FusedBatchNorm_1
        )
        pass1_block1_mbconv2_skeleton_dconv1_eswish_Sigmoid = F.sigmoid(
            pass1_block1_mbconv2_skeleton_dconv1_bn_FusedBatchNorm_1
        )
        pass1_block1_mbconv2_skeleton_dconv1_eswish_mul_1 = (
            pass1_block1_mbconv2_skeleton_dconv1_eswish_mul
            * pass1_block1_mbconv2_skeleton_dconv1_eswish_Sigmoid
        )
        pass1_block1_mbconv2_skeleton_se_se_squeeze_lambda_Mean = torch.mean(
            pass1_block1_mbconv2_skeleton_dconv1_eswish_mul_1, 3, True
        )
        pass1_block1_mbconv2_skeleton_se_se_squeeze_lambda_Mean = torch.mean(
            pass1_block1_mbconv2_skeleton_se_se_squeeze_lambda_Mean, 2, True
        )
        pass1_block1_mbconv2_skeleton_se_se_squeeze_conv_convolution = (
            self.pass1_block1_mbconv2_skeleton_se_se_squeeze_conv_convolution(
                pass1_block1_mbconv2_skeleton_se_se_squeeze_lambda_Mean
            )
        )
        pass1_block1_mbconv2_skeleton_se_se_squeeze_eswish_mul = (
            self.pass1_block1_mbconv2_skeleton_se_se_squeeze_eswish_mul_x
            * pass1_block1_mbconv2_skeleton_se_se_squeeze_conv_convolution
        )
        pass1_block1_mbconv2_skeleton_se_se_squeeze_eswish_Sigmoid = F.sigmoid(
            pass1_block1_mbconv2_skeleton_se_se_squeeze_conv_convolution
        )
        pass1_block1_mbconv2_skeleton_se_se_squeeze_eswish_mul_1 = (
            pass1_block1_mbconv2_skeleton_se_se_squeeze_eswish_mul
            * pass1_block1_mbconv2_skeleton_se_se_squeeze_eswish_Sigmoid
        )
        pass1_block1_mbconv2_skeleton_se_se_excite_conv_convolution = (
            self.pass1_block1_mbconv2_skeleton_se_se_excite_conv_convolution(
                pass1_block1_mbconv2_skeleton_se_se_squeeze_eswish_mul_1
            )
        )
        pass1_block1_mbconv2_skeleton_se_se_excite_sigmoid_Sigmoid = F.sigmoid(
            pass1_block1_mbconv2_skeleton_se_se_excite_conv_convolution
        )
        pass1_block1_mbconv2_skeleton_se_se_multiply_mul = (
            pass1_block1_mbconv2_skeleton_se_se_excite_sigmoid_Sigmoid
            * pass1_block1_mbconv2_skeleton_dconv1_eswish_mul_1
        )
        pass1_block1_mbconv2_skeleton_conv2_convolution = (
            self.pass1_block1_mbconv2_skeleton_conv2_convolution(
                pass1_block1_mbconv2_skeleton_se_se_multiply_mul
            )
        )
        pass1_block1_mbconv2_skeleton_conv2_bn_FusedBatchNorm_1 = (
            self.pass1_block1_mbconv2_skeleton_conv2_bn_FusedBatchNorm_1(
                pass1_block1_mbconv2_skeleton_conv2_convolution
            )
        )
        pass1_block1_mbconv2_skeleton_dense_concat = torch.cat(
            (
                pass1_block1_mbconv2_skeleton_conv2_bn_FusedBatchNorm_1,
                pass1_block1_mbconv1_skeleton_conv2_bn_FusedBatchNorm_1,
            ),
            1,
        )
        pass1_block1_mbconv3_skeleton_conv1_convolution = (
            self.pass1_block1_mbconv3_skeleton_conv1_convolution(
                pass1_block1_mbconv2_skeleton_dense_concat
            )
        )
        pass1_block1_mbconv3_skeleton_conv1_bn_FusedBatchNorm_1 = (
            self.pass1_block1_mbconv3_skeleton_conv1_bn_FusedBatchNorm_1(
                pass1_block1_mbconv3_skeleton_conv1_convolution
            )
        )
        pass1_block1_mbconv3_skeleton_conv1_eswish_mul = (
            self.pass1_block1_mbconv3_skeleton_conv1_eswish_mul_x
            * pass1_block1_mbconv3_skeleton_conv1_bn_FusedBatchNorm_1
        )
        pass1_block1_mbconv3_skeleton_conv1_eswish_Sigmoid = F.sigmoid(
            pass1_block1_mbconv3_skeleton_conv1_bn_FusedBatchNorm_1
        )
        pass1_block1_mbconv3_skeleton_conv1_eswish_mul_1 = (
            pass1_block1_mbconv3_skeleton_conv1_eswish_mul
            * pass1_block1_mbconv3_skeleton_conv1_eswish_Sigmoid
        )
        pass1_block1_mbconv3_skeleton_dconv1_depthwise_pad = F.pad(
            pass1_block1_mbconv3_skeleton_conv1_eswish_mul_1, (2, 2, 2, 2)
        )
        pass1_block1_mbconv3_skeleton_dconv1_depthwise = (
            self.pass1_block1_mbconv3_skeleton_dconv1_depthwise(
                pass1_block1_mbconv3_skeleton_dconv1_depthwise_pad
            )
        )
        pass1_block1_mbconv3_skeleton_dconv1_bn_FusedBatchNorm_1 = (
            self.pass1_block1_mbconv3_skeleton_dconv1_bn_FusedBatchNorm_1(
                pass1_block1_mbconv3_skeleton_dconv1_depthwise
            )
        )
        pass1_block1_mbconv3_skeleton_dconv1_eswish_mul = (
            self.pass1_block1_mbconv3_skeleton_dconv1_eswish_mul_x
            * pass1_block1_mbconv3_skeleton_dconv1_bn_FusedBatchNorm_1
        )
        pass1_block1_mbconv3_skeleton_dconv1_eswish_Sigmoid = F.sigmoid(
            pass1_block1_mbconv3_skeleton_dconv1_bn_FusedBatchNorm_1
        )
        pass1_block1_mbconv3_skeleton_dconv1_eswish_mul_1 = (
            pass1_block1_mbconv3_skeleton_dconv1_eswish_mul
            * pass1_block1_mbconv3_skeleton_dconv1_eswish_Sigmoid
        )
        pass1_block1_mbconv3_skeleton_se_se_squeeze_lambda_Mean = torch.mean(
            pass1_block1_mbconv3_skeleton_dconv1_eswish_mul_1, 3, True
        )
        pass1_block1_mbconv3_skeleton_se_se_squeeze_lambda_Mean = torch.mean(
            pass1_block1_mbconv3_skeleton_se_se_squeeze_lambda_Mean, 2, True
        )
        pass1_block1_mbconv3_skeleton_se_se_squeeze_conv_convolution = (
            self.pass1_block1_mbconv3_skeleton_se_se_squeeze_conv_convolution(
                pass1_block1_mbconv3_skeleton_se_se_squeeze_lambda_Mean
            )
        )
        pass1_block1_mbconv3_skeleton_se_se_squeeze_eswish_mul = (
            self.pass1_block1_mbconv3_skeleton_se_se_squeeze_eswish_mul_x
            * pass1_block1_mbconv3_skeleton_se_se_squeeze_conv_convolution
        )
        pass1_block1_mbconv3_skeleton_se_se_squeeze_eswish_Sigmoid = F.sigmoid(
            pass1_block1_mbconv3_skeleton_se_se_squeeze_conv_convolution
        )
        pass1_block1_mbconv3_skeleton_se_se_squeeze_eswish_mul_1 = (
            pass1_block1_mbconv3_skeleton_se_se_squeeze_eswish_mul
            * pass1_block1_mbconv3_skeleton_se_se_squeeze_eswish_Sigmoid
        )
        pass1_block1_mbconv3_skeleton_se_se_excite_conv_convolution = (
            self.pass1_block1_mbconv3_skeleton_se_se_excite_conv_convolution(
                pass1_block1_mbconv3_skeleton_se_se_squeeze_eswish_mul_1
            )
        )
        pass1_block1_mbconv3_skeleton_se_se_excite_sigmoid_Sigmoid = F.sigmoid(
            pass1_block1_mbconv3_skeleton_se_se_excite_conv_convolution
        )
        pass1_block1_mbconv3_skeleton_se_se_multiply_mul = (
            pass1_block1_mbconv3_skeleton_se_se_excite_sigmoid_Sigmoid
            * pass1_block1_mbconv3_skeleton_dconv1_eswish_mul_1
        )
        pass1_block1_mbconv3_skeleton_conv2_convolution = (
            self.pass1_block1_mbconv3_skeleton_conv2_convolution(
                pass1_block1_mbconv3_skeleton_se_se_multiply_mul
            )
        )
        pass1_block1_mbconv3_skeleton_conv2_bn_FusedBatchNorm_1 = (
            self.pass1_block1_mbconv3_skeleton_conv2_bn_FusedBatchNorm_1(
                pass1_block1_mbconv3_skeleton_conv2_convolution
            )
        )
        pass1_block1_mbconv3_skeleton_dense_concat = torch.cat(
            (
                pass1_block1_mbconv3_skeleton_conv2_bn_FusedBatchNorm_1,
                pass1_block1_mbconv2_skeleton_dense_concat,
            ),
            1,
        )
        concatenate_1_concat = torch.cat(
            (pass1_block1_mbconv3_skeleton_dense_concat, block3b_add_res1_add), 1
        )
        pass2_block1_mbconv1_detection1_conv1_convolution = (
            self.pass2_block1_mbconv1_detection1_conv1_convolution(concatenate_1_concat)
        )
        pass2_block1_mbconv1_detection1_conv1_bn_FusedBatchNorm_1 = (
            self.pass2_block1_mbconv1_detection1_conv1_bn_FusedBatchNorm_1(
                pass2_block1_mbconv1_detection1_conv1_convolution
            )
        )
        pass2_block1_mbconv1_detection1_conv1_eswish_mul = (
            self.pass2_block1_mbconv1_detection1_conv1_eswish_mul_x
            * pass2_block1_mbconv1_detection1_conv1_bn_FusedBatchNorm_1
        )
        pass2_block1_mbconv1_detection1_conv1_eswish_Sigmoid = F.sigmoid(
            pass2_block1_mbconv1_detection1_conv1_bn_FusedBatchNorm_1
        )
        pass2_block1_mbconv1_detection1_conv1_eswish_mul_1 = (
            pass2_block1_mbconv1_detection1_conv1_eswish_mul
            * pass2_block1_mbconv1_detection1_conv1_eswish_Sigmoid
        )
        pass2_block1_mbconv1_detection1_dconv1_depthwise_pad = F.pad(
            pass2_block1_mbconv1_detection1_conv1_eswish_mul_1, (2, 2, 2, 2)
        )
        pass2_block1_mbconv1_detection1_dconv1_depthwise = (
            self.pass2_block1_mbconv1_detection1_dconv1_depthwise(
                pass2_block1_mbconv1_detection1_dconv1_depthwise_pad
            )
        )
        pass2_block1_mbconv1_detection1_dconv1_bn_FusedBatchNorm_1 = (
            self.pass2_block1_mbconv1_detection1_dconv1_bn_FusedBatchNorm_1(
                pass2_block1_mbconv1_detection1_dconv1_depthwise
            )
        )
        pass2_block1_mbconv1_detection1_dconv1_eswish_mul = (
            self.pass2_block1_mbconv1_detection1_dconv1_eswish_mul_x
            * pass2_block1_mbconv1_detection1_dconv1_bn_FusedBatchNorm_1
        )
        pass2_block1_mbconv1_detection1_dconv1_eswish_Sigmoid = F.sigmoid(
            pass2_block1_mbconv1_detection1_dconv1_bn_FusedBatchNorm_1
        )
        pass2_block1_mbconv1_detection1_dconv1_eswish_mul_1 = (
            pass2_block1_mbconv1_detection1_dconv1_eswish_mul
            * pass2_block1_mbconv1_detection1_dconv1_eswish_Sigmoid
        )
        pass2_block1_mbconv1_detection1_se_se_squeeze_lambda_Mean = torch.mean(
            pass2_block1_mbconv1_detection1_dconv1_eswish_mul_1, 3, True
        )
        pass2_block1_mbconv1_detection1_se_se_squeeze_lambda_Mean = torch.mean(
            pass2_block1_mbconv1_detection1_se_se_squeeze_lambda_Mean, 2, True
        )
        pass2_block1_mbconv1_detection1_se_se_squeeze_conv_convolution = (
            self.pass2_block1_mbconv1_detection1_se_se_squeeze_conv_convolution(
                pass2_block1_mbconv1_detection1_se_se_squeeze_lambda_Mean
            )
        )
        pass2_block1_mbconv1_detection1_se_se_squeeze_eswish_mul = (
            self.pass2_block1_mbconv1_detection1_se_se_squeeze_eswish_mul_x
            * pass2_block1_mbconv1_detection1_se_se_squeeze_conv_convolution
        )
        pass2_block1_mbconv1_detection1_se_se_squeeze_eswish_Sigmoid = F.sigmoid(
            pass2_block1_mbconv1_detection1_se_se_squeeze_conv_convolution
        )
        pass2_block1_mbconv1_detection1_se_se_squeeze_eswish_mul_1 = (
            pass2_block1_mbconv1_detection1_se_se_squeeze_eswish_mul
            * pass2_block1_mbconv1_detection1_se_se_squeeze_eswish_Sigmoid
        )
        pass2_block1_mbconv1_detection1_se_se_excite_conv_convolution = (
            self.pass2_block1_mbconv1_detection1_se_se_excite_conv_convolution(
                pass2_block1_mbconv1_detection1_se_se_squeeze_eswish_mul_1
            )
        )
        pass2_block1_mbconv1_detection1_se_se_excite_sigmoid_Sigmoid = F.sigmoid(
            pass2_block1_mbconv1_detection1_se_se_excite_conv_convolution
        )
        pass2_block1_mbconv1_detection1_se_se_multiply_mul = (
            pass2_block1_mbconv1_detection1_se_se_excite_sigmoid_Sigmoid
            * pass2_block1_mbconv1_detection1_dconv1_eswish_mul_1
        )
        pass2_block1_mbconv1_detection1_conv2_convolution = (
            self.pass2_block1_mbconv1_detection1_conv2_convolution(
                pass2_block1_mbconv1_detection1_se_se_multiply_mul
            )
        )
        pass2_block1_mbconv1_detection1_conv2_bn_FusedBatchNorm_1 = (
            self.pass2_block1_mbconv1_detection1_conv2_bn_FusedBatchNorm_1(
                pass2_block1_mbconv1_detection1_conv2_convolution
            )
        )
        pass2_block1_mbconv2_detection1_conv1_convolution = (
            self.pass2_block1_mbconv2_detection1_conv1_convolution(
                pass2_block1_mbconv1_detection1_conv2_bn_FusedBatchNorm_1
            )
        )
        pass2_block1_mbconv2_detection1_conv1_bn_FusedBatchNorm_1 = (
            self.pass2_block1_mbconv2_detection1_conv1_bn_FusedBatchNorm_1(
                pass2_block1_mbconv2_detection1_conv1_convolution
            )
        )
        pass2_block1_mbconv2_detection1_conv1_eswish_mul = (
            self.pass2_block1_mbconv2_detection1_conv1_eswish_mul_x
            * pass2_block1_mbconv2_detection1_conv1_bn_FusedBatchNorm_1
        )
        pass2_block1_mbconv2_detection1_conv1_eswish_Sigmoid = F.sigmoid(
            pass2_block1_mbconv2_detection1_conv1_bn_FusedBatchNorm_1
        )
        pass2_block1_mbconv2_detection1_conv1_eswish_mul_1 = (
            pass2_block1_mbconv2_detection1_conv1_eswish_mul
            * pass2_block1_mbconv2_detection1_conv1_eswish_Sigmoid
        )
        pass2_block1_mbconv2_detection1_dconv1_depthwise_pad = F.pad(
            pass2_block1_mbconv2_detection1_conv1_eswish_mul_1, (2, 2, 2, 2)
        )
        pass2_block1_mbconv2_detection1_dconv1_depthwise = (
            self.pass2_block1_mbconv2_detection1_dconv1_depthwise(
                pass2_block1_mbconv2_detection1_dconv1_depthwise_pad
            )
        )
        pass2_block1_mbconv2_detection1_dconv1_bn_FusedBatchNorm_1 = (
            self.pass2_block1_mbconv2_detection1_dconv1_bn_FusedBatchNorm_1(
                pass2_block1_mbconv2_detection1_dconv1_depthwise
            )
        )
        pass2_block1_mbconv2_detection1_dconv1_eswish_mul = (
            self.pass2_block1_mbconv2_detection1_dconv1_eswish_mul_x
            * pass2_block1_mbconv2_detection1_dconv1_bn_FusedBatchNorm_1
        )
        pass2_block1_mbconv2_detection1_dconv1_eswish_Sigmoid = F.sigmoid(
            pass2_block1_mbconv2_detection1_dconv1_bn_FusedBatchNorm_1
        )
        pass2_block1_mbconv2_detection1_dconv1_eswish_mul_1 = (
            pass2_block1_mbconv2_detection1_dconv1_eswish_mul
            * pass2_block1_mbconv2_detection1_dconv1_eswish_Sigmoid
        )
        pass2_block1_mbconv2_detection1_se_se_squeeze_lambda_Mean = torch.mean(
            pass2_block1_mbconv2_detection1_dconv1_eswish_mul_1, 3, True
        )
        pass2_block1_mbconv2_detection1_se_se_squeeze_lambda_Mean = torch.mean(
            pass2_block1_mbconv2_detection1_se_se_squeeze_lambda_Mean, 2, True
        )
        pass2_block1_mbconv2_detection1_se_se_squeeze_conv_convolution = (
            self.pass2_block1_mbconv2_detection1_se_se_squeeze_conv_convolution(
                pass2_block1_mbconv2_detection1_se_se_squeeze_lambda_Mean
            )
        )
        pass2_block1_mbconv2_detection1_se_se_squeeze_eswish_mul = (
            self.pass2_block1_mbconv2_detection1_se_se_squeeze_eswish_mul_x
            * pass2_block1_mbconv2_detection1_se_se_squeeze_conv_convolution
        )
        pass2_block1_mbconv2_detection1_se_se_squeeze_eswish_Sigmoid = F.sigmoid(
            pass2_block1_mbconv2_detection1_se_se_squeeze_conv_convolution
        )
        pass2_block1_mbconv2_detection1_se_se_squeeze_eswish_mul_1 = (
            pass2_block1_mbconv2_detection1_se_se_squeeze_eswish_mul
            * pass2_block1_mbconv2_detection1_se_se_squeeze_eswish_Sigmoid
        )
        pass2_block1_mbconv2_detection1_se_se_excite_conv_convolution = (
            self.pass2_block1_mbconv2_detection1_se_se_excite_conv_convolution(
                pass2_block1_mbconv2_detection1_se_se_squeeze_eswish_mul_1
            )
        )
        pass2_block1_mbconv2_detection1_se_se_excite_sigmoid_Sigmoid = F.sigmoid(
            pass2_block1_mbconv2_detection1_se_se_excite_conv_convolution
        )
        pass2_block1_mbconv2_detection1_se_se_multiply_mul = (
            pass2_block1_mbconv2_detection1_se_se_excite_sigmoid_Sigmoid
            * pass2_block1_mbconv2_detection1_dconv1_eswish_mul_1
        )
        pass2_block1_mbconv2_detection1_conv2_convolution = (
            self.pass2_block1_mbconv2_detection1_conv2_convolution(
                pass2_block1_mbconv2_detection1_se_se_multiply_mul
            )
        )
        pass2_block1_mbconv2_detection1_conv2_bn_FusedBatchNorm_1 = (
            self.pass2_block1_mbconv2_detection1_conv2_bn_FusedBatchNorm_1(
                pass2_block1_mbconv2_detection1_conv2_convolution
            )
        )
        pass2_block1_mbconv2_detection1_dense_concat = torch.cat(
            (
                pass2_block1_mbconv2_detection1_conv2_bn_FusedBatchNorm_1,
                pass2_block1_mbconv1_detection1_conv2_bn_FusedBatchNorm_1,
            ),
            1,
        )
        pass2_block1_mbconv3_detection1_conv1_convolution = (
            self.pass2_block1_mbconv3_detection1_conv1_convolution(
                pass2_block1_mbconv2_detection1_dense_concat
            )
        )
        pass2_block1_mbconv3_detection1_conv1_bn_FusedBatchNorm_1 = (
            self.pass2_block1_mbconv3_detection1_conv1_bn_FusedBatchNorm_1(
                pass2_block1_mbconv3_detection1_conv1_convolution
            )
        )
        pass2_block1_mbconv3_detection1_conv1_eswish_mul = (
            self.pass2_block1_mbconv3_detection1_conv1_eswish_mul_x
            * pass2_block1_mbconv3_detection1_conv1_bn_FusedBatchNorm_1
        )
        pass2_block1_mbconv3_detection1_conv1_eswish_Sigmoid = F.sigmoid(
            pass2_block1_mbconv3_detection1_conv1_bn_FusedBatchNorm_1
        )
        pass2_block1_mbconv3_detection1_conv1_eswish_mul_1 = (
            pass2_block1_mbconv3_detection1_conv1_eswish_mul
            * pass2_block1_mbconv3_detection1_conv1_eswish_Sigmoid
        )
        pass2_block1_mbconv3_detection1_dconv1_depthwise_pad = F.pad(
            pass2_block1_mbconv3_detection1_conv1_eswish_mul_1, (2, 2, 2, 2)
        )
        pass2_block1_mbconv3_detection1_dconv1_depthwise = (
            self.pass2_block1_mbconv3_detection1_dconv1_depthwise(
                pass2_block1_mbconv3_detection1_dconv1_depthwise_pad
            )
        )
        pass2_block1_mbconv3_detection1_dconv1_bn_FusedBatchNorm_1 = (
            self.pass2_block1_mbconv3_detection1_dconv1_bn_FusedBatchNorm_1(
                pass2_block1_mbconv3_detection1_dconv1_depthwise
            )
        )
        pass2_block1_mbconv3_detection1_dconv1_eswish_mul = (
            self.pass2_block1_mbconv3_detection1_dconv1_eswish_mul_x
            * pass2_block1_mbconv3_detection1_dconv1_bn_FusedBatchNorm_1
        )
        pass2_block1_mbconv3_detection1_dconv1_eswish_Sigmoid = F.sigmoid(
            pass2_block1_mbconv3_detection1_dconv1_bn_FusedBatchNorm_1
        )
        pass2_block1_mbconv3_detection1_dconv1_eswish_mul_1 = (
            pass2_block1_mbconv3_detection1_dconv1_eswish_mul
            * pass2_block1_mbconv3_detection1_dconv1_eswish_Sigmoid
        )
        pass2_block1_mbconv3_detection1_se_se_squeeze_lambda_Mean = torch.mean(
            pass2_block1_mbconv3_detection1_dconv1_eswish_mul_1, 3, True
        )
        pass2_block1_mbconv3_detection1_se_se_squeeze_lambda_Mean = torch.mean(
            pass2_block1_mbconv3_detection1_se_se_squeeze_lambda_Mean, 2, True
        )
        pass2_block1_mbconv3_detection1_se_se_squeeze_conv_convolution = (
            self.pass2_block1_mbconv3_detection1_se_se_squeeze_conv_convolution(
                pass2_block1_mbconv3_detection1_se_se_squeeze_lambda_Mean
            )
        )
        pass2_block1_mbconv3_detection1_se_se_squeeze_eswish_mul = (
            self.pass2_block1_mbconv3_detection1_se_se_squeeze_eswish_mul_x
            * pass2_block1_mbconv3_detection1_se_se_squeeze_conv_convolution
        )
        pass2_block1_mbconv3_detection1_se_se_squeeze_eswish_Sigmoid = F.sigmoid(
            pass2_block1_mbconv3_detection1_se_se_squeeze_conv_convolution
        )
        pass2_block1_mbconv3_detection1_se_se_squeeze_eswish_mul_1 = (
            pass2_block1_mbconv3_detection1_se_se_squeeze_eswish_mul
            * pass2_block1_mbconv3_detection1_se_se_squeeze_eswish_Sigmoid
        )
        pass2_block1_mbconv3_detection1_se_se_excite_conv_convolution = (
            self.pass2_block1_mbconv3_detection1_se_se_excite_conv_convolution(
                pass2_block1_mbconv3_detection1_se_se_squeeze_eswish_mul_1
            )
        )
        pass2_block1_mbconv3_detection1_se_se_excite_sigmoid_Sigmoid = F.sigmoid(
            pass2_block1_mbconv3_detection1_se_se_excite_conv_convolution
        )
        pass2_block1_mbconv3_detection1_se_se_multiply_mul = (
            pass2_block1_mbconv3_detection1_se_se_excite_sigmoid_Sigmoid
            * pass2_block1_mbconv3_detection1_dconv1_eswish_mul_1
        )
        pass2_block1_mbconv3_detection1_conv2_convolution = (
            self.pass2_block1_mbconv3_detection1_conv2_convolution(
                pass2_block1_mbconv3_detection1_se_se_multiply_mul
            )
        )
        pass2_block1_mbconv3_detection1_conv2_bn_FusedBatchNorm_1 = (
            self.pass2_block1_mbconv3_detection1_conv2_bn_FusedBatchNorm_1(
                pass2_block1_mbconv3_detection1_conv2_convolution
            )
        )
        pass2_block1_mbconv3_detection1_dense_concat = torch.cat(
            (
                pass2_block1_mbconv3_detection1_conv2_bn_FusedBatchNorm_1,
                pass2_block1_mbconv2_detection1_dense_concat,
            ),
            1,
        )
        concatenate_2_concat = torch.cat(
            (pass2_block1_mbconv3_detection1_dense_concat, block3b_add_res1_add), 1
        )
        pass3_block1_mbconv1_detection2_conv1_convolution = (
            self.pass3_block1_mbconv1_detection2_conv1_convolution(concatenate_2_concat)
        )
        pass3_block1_mbconv1_detection2_conv1_bn_FusedBatchNorm_1 = (
            self.pass3_block1_mbconv1_detection2_conv1_bn_FusedBatchNorm_1(
                pass3_block1_mbconv1_detection2_conv1_convolution
            )
        )
        pass3_block1_mbconv1_detection2_conv1_eswish_mul = (
            self.pass3_block1_mbconv1_detection2_conv1_eswish_mul_x
            * pass3_block1_mbconv1_detection2_conv1_bn_FusedBatchNorm_1
        )
        pass3_block1_mbconv1_detection2_conv1_eswish_Sigmoid = F.sigmoid(
            pass3_block1_mbconv1_detection2_conv1_bn_FusedBatchNorm_1
        )
        pass3_block1_mbconv1_detection2_conv1_eswish_mul_1 = (
            pass3_block1_mbconv1_detection2_conv1_eswish_mul
            * pass3_block1_mbconv1_detection2_conv1_eswish_Sigmoid
        )
        pass3_block1_mbconv1_detection2_dconv1_depthwise_pad = F.pad(
            pass3_block1_mbconv1_detection2_conv1_eswish_mul_1, (2, 2, 2, 2)
        )
        pass3_block1_mbconv1_detection2_dconv1_depthwise = (
            self.pass3_block1_mbconv1_detection2_dconv1_depthwise(
                pass3_block1_mbconv1_detection2_dconv1_depthwise_pad
            )
        )
        pass3_block1_mbconv1_detection2_dconv1_bn_FusedBatchNorm_1 = (
            self.pass3_block1_mbconv1_detection2_dconv1_bn_FusedBatchNorm_1(
                pass3_block1_mbconv1_detection2_dconv1_depthwise
            )
        )
        pass3_block1_mbconv1_detection2_dconv1_eswish_mul = (
            self.pass3_block1_mbconv1_detection2_dconv1_eswish_mul_x
            * pass3_block1_mbconv1_detection2_dconv1_bn_FusedBatchNorm_1
        )
        pass3_block1_mbconv1_detection2_dconv1_eswish_Sigmoid = F.sigmoid(
            pass3_block1_mbconv1_detection2_dconv1_bn_FusedBatchNorm_1
        )
        pass3_block1_mbconv1_detection2_dconv1_eswish_mul_1 = (
            pass3_block1_mbconv1_detection2_dconv1_eswish_mul
            * pass3_block1_mbconv1_detection2_dconv1_eswish_Sigmoid
        )
        pass3_block1_mbconv1_detection2_se_se_squeeze_lambda_Mean = torch.mean(
            pass3_block1_mbconv1_detection2_dconv1_eswish_mul_1, 3, True
        )
        pass3_block1_mbconv1_detection2_se_se_squeeze_lambda_Mean = torch.mean(
            pass3_block1_mbconv1_detection2_se_se_squeeze_lambda_Mean, 2, True
        )
        pass3_block1_mbconv1_detection2_se_se_squeeze_conv_convolution = (
            self.pass3_block1_mbconv1_detection2_se_se_squeeze_conv_convolution(
                pass3_block1_mbconv1_detection2_se_se_squeeze_lambda_Mean
            )
        )
        pass3_block1_mbconv1_detection2_se_se_squeeze_eswish_mul = (
            self.pass3_block1_mbconv1_detection2_se_se_squeeze_eswish_mul_x
            * pass3_block1_mbconv1_detection2_se_se_squeeze_conv_convolution
        )
        pass3_block1_mbconv1_detection2_se_se_squeeze_eswish_Sigmoid = F.sigmoid(
            pass3_block1_mbconv1_detection2_se_se_squeeze_conv_convolution
        )
        pass3_block1_mbconv1_detection2_se_se_squeeze_eswish_mul_1 = (
            pass3_block1_mbconv1_detection2_se_se_squeeze_eswish_mul
            * pass3_block1_mbconv1_detection2_se_se_squeeze_eswish_Sigmoid
        )
        pass3_block1_mbconv1_detection2_se_se_excite_conv_convolution = (
            self.pass3_block1_mbconv1_detection2_se_se_excite_conv_convolution(
                pass3_block1_mbconv1_detection2_se_se_squeeze_eswish_mul_1
            )
        )
        pass3_block1_mbconv1_detection2_se_se_excite_sigmoid_Sigmoid = F.sigmoid(
            pass3_block1_mbconv1_detection2_se_se_excite_conv_convolution
        )
        pass3_block1_mbconv1_detection2_se_se_multiply_mul = (
            pass3_block1_mbconv1_detection2_se_se_excite_sigmoid_Sigmoid
            * pass3_block1_mbconv1_detection2_dconv1_eswish_mul_1
        )
        pass3_block1_mbconv1_detection2_conv2_convolution = (
            self.pass3_block1_mbconv1_detection2_conv2_convolution(
                pass3_block1_mbconv1_detection2_se_se_multiply_mul
            )
        )
        pass3_block1_mbconv1_detection2_conv2_bn_FusedBatchNorm_1 = (
            self.pass3_block1_mbconv1_detection2_conv2_bn_FusedBatchNorm_1(
                pass3_block1_mbconv1_detection2_conv2_convolution
            )
        )
        pass3_block1_mbconv2_detection2_conv1_convolution = (
            self.pass3_block1_mbconv2_detection2_conv1_convolution(
                pass3_block1_mbconv1_detection2_conv2_bn_FusedBatchNorm_1
            )
        )
        pass3_block1_mbconv2_detection2_conv1_bn_FusedBatchNorm_1 = (
            self.pass3_block1_mbconv2_detection2_conv1_bn_FusedBatchNorm_1(
                pass3_block1_mbconv2_detection2_conv1_convolution
            )
        )
        pass3_block1_mbconv2_detection2_conv1_eswish_mul = (
            self.pass3_block1_mbconv2_detection2_conv1_eswish_mul_x
            * pass3_block1_mbconv2_detection2_conv1_bn_FusedBatchNorm_1
        )
        pass3_block1_mbconv2_detection2_conv1_eswish_Sigmoid = F.sigmoid(
            pass3_block1_mbconv2_detection2_conv1_bn_FusedBatchNorm_1
        )
        pass3_block1_mbconv2_detection2_conv1_eswish_mul_1 = (
            pass3_block1_mbconv2_detection2_conv1_eswish_mul
            * pass3_block1_mbconv2_detection2_conv1_eswish_Sigmoid
        )
        pass3_block1_mbconv2_detection2_dconv1_depthwise_pad = F.pad(
            pass3_block1_mbconv2_detection2_conv1_eswish_mul_1, (2, 2, 2, 2)
        )
        pass3_block1_mbconv2_detection2_dconv1_depthwise = (
            self.pass3_block1_mbconv2_detection2_dconv1_depthwise(
                pass3_block1_mbconv2_detection2_dconv1_depthwise_pad
            )
        )
        pass3_block1_mbconv2_detection2_dconv1_bn_FusedBatchNorm_1 = (
            self.pass3_block1_mbconv2_detection2_dconv1_bn_FusedBatchNorm_1(
                pass3_block1_mbconv2_detection2_dconv1_depthwise
            )
        )
        pass3_block1_mbconv2_detection2_dconv1_eswish_mul = (
            self.pass3_block1_mbconv2_detection2_dconv1_eswish_mul_x
            * pass3_block1_mbconv2_detection2_dconv1_bn_FusedBatchNorm_1
        )
        pass3_block1_mbconv2_detection2_dconv1_eswish_Sigmoid = F.sigmoid(
            pass3_block1_mbconv2_detection2_dconv1_bn_FusedBatchNorm_1
        )
        pass3_block1_mbconv2_detection2_dconv1_eswish_mul_1 = (
            pass3_block1_mbconv2_detection2_dconv1_eswish_mul
            * pass3_block1_mbconv2_detection2_dconv1_eswish_Sigmoid
        )
        pass3_block1_mbconv2_detection2_se_se_squeeze_lambda_Mean = torch.mean(
            pass3_block1_mbconv2_detection2_dconv1_eswish_mul_1, 3, True
        )
        pass3_block1_mbconv2_detection2_se_se_squeeze_lambda_Mean = torch.mean(
            pass3_block1_mbconv2_detection2_se_se_squeeze_lambda_Mean, 2, True
        )
        pass3_block1_mbconv2_detection2_se_se_squeeze_conv_convolution = (
            self.pass3_block1_mbconv2_detection2_se_se_squeeze_conv_convolution(
                pass3_block1_mbconv2_detection2_se_se_squeeze_lambda_Mean
            )
        )
        pass3_block1_mbconv2_detection2_se_se_squeeze_eswish_mul = (
            self.pass3_block1_mbconv2_detection2_se_se_squeeze_eswish_mul_x
            * pass3_block1_mbconv2_detection2_se_se_squeeze_conv_convolution
        )
        pass3_block1_mbconv2_detection2_se_se_squeeze_eswish_Sigmoid = F.sigmoid(
            pass3_block1_mbconv2_detection2_se_se_squeeze_conv_convolution
        )
        pass3_block1_mbconv2_detection2_se_se_squeeze_eswish_mul_1 = (
            pass3_block1_mbconv2_detection2_se_se_squeeze_eswish_mul
            * pass3_block1_mbconv2_detection2_se_se_squeeze_eswish_Sigmoid
        )
        pass3_block1_mbconv2_detection2_se_se_excite_conv_convolution = (
            self.pass3_block1_mbconv2_detection2_se_se_excite_conv_convolution(
                pass3_block1_mbconv2_detection2_se_se_squeeze_eswish_mul_1
            )
        )
        pass3_block1_mbconv2_detection2_se_se_excite_sigmoid_Sigmoid = F.sigmoid(
            pass3_block1_mbconv2_detection2_se_se_excite_conv_convolution
        )
        pass3_block1_mbconv2_detection2_se_se_multiply_mul = (
            pass3_block1_mbconv2_detection2_se_se_excite_sigmoid_Sigmoid
            * pass3_block1_mbconv2_detection2_dconv1_eswish_mul_1
        )
        pass3_block1_mbconv2_detection2_conv2_convolution = (
            self.pass3_block1_mbconv2_detection2_conv2_convolution(
                pass3_block1_mbconv2_detection2_se_se_multiply_mul
            )
        )
        pass3_block1_mbconv2_detection2_conv2_bn_FusedBatchNorm_1 = (
            self.pass3_block1_mbconv2_detection2_conv2_bn_FusedBatchNorm_1(
                pass3_block1_mbconv2_detection2_conv2_convolution
            )
        )
        pass3_block1_mbconv2_detection2_dense_concat = torch.cat(
            (
                pass3_block1_mbconv2_detection2_conv2_bn_FusedBatchNorm_1,
                pass3_block1_mbconv1_detection2_conv2_bn_FusedBatchNorm_1,
            ),
            1,
        )
        pass3_block1_mbconv3_detection2_conv1_convolution = (
            self.pass3_block1_mbconv3_detection2_conv1_convolution(
                pass3_block1_mbconv2_detection2_dense_concat
            )
        )
        pass3_block1_mbconv3_detection2_conv1_bn_FusedBatchNorm_1 = (
            self.pass3_block1_mbconv3_detection2_conv1_bn_FusedBatchNorm_1(
                pass3_block1_mbconv3_detection2_conv1_convolution
            )
        )
        pass3_block1_mbconv3_detection2_conv1_eswish_mul = (
            self.pass3_block1_mbconv3_detection2_conv1_eswish_mul_x
            * pass3_block1_mbconv3_detection2_conv1_bn_FusedBatchNorm_1
        )
        pass3_block1_mbconv3_detection2_conv1_eswish_Sigmoid = F.sigmoid(
            pass3_block1_mbconv3_detection2_conv1_bn_FusedBatchNorm_1
        )
        pass3_block1_mbconv3_detection2_conv1_eswish_mul_1 = (
            pass3_block1_mbconv3_detection2_conv1_eswish_mul
            * pass3_block1_mbconv3_detection2_conv1_eswish_Sigmoid
        )
        pass3_block1_mbconv3_detection2_dconv1_depthwise_pad = F.pad(
            pass3_block1_mbconv3_detection2_conv1_eswish_mul_1, (2, 2, 2, 2)
        )
        pass3_block1_mbconv3_detection2_dconv1_depthwise = (
            self.pass3_block1_mbconv3_detection2_dconv1_depthwise(
                pass3_block1_mbconv3_detection2_dconv1_depthwise_pad
            )
        )
        pass3_block1_mbconv3_detection2_dconv1_bn_FusedBatchNorm_1 = (
            self.pass3_block1_mbconv3_detection2_dconv1_bn_FusedBatchNorm_1(
                pass3_block1_mbconv3_detection2_dconv1_depthwise
            )
        )
        pass3_block1_mbconv3_detection2_dconv1_eswish_mul = (
            self.pass3_block1_mbconv3_detection2_dconv1_eswish_mul_x
            * pass3_block1_mbconv3_detection2_dconv1_bn_FusedBatchNorm_1
        )
        pass3_block1_mbconv3_detection2_dconv1_eswish_Sigmoid = F.sigmoid(
            pass3_block1_mbconv3_detection2_dconv1_bn_FusedBatchNorm_1
        )
        pass3_block1_mbconv3_detection2_dconv1_eswish_mul_1 = (
            pass3_block1_mbconv3_detection2_dconv1_eswish_mul
            * pass3_block1_mbconv3_detection2_dconv1_eswish_Sigmoid
        )
        pass3_block1_mbconv3_detection2_se_se_squeeze_lambda_Mean = torch.mean(
            pass3_block1_mbconv3_detection2_dconv1_eswish_mul_1, 3, True
        )
        pass3_block1_mbconv3_detection2_se_se_squeeze_lambda_Mean = torch.mean(
            pass3_block1_mbconv3_detection2_se_se_squeeze_lambda_Mean, 2, True
        )
        pass3_block1_mbconv3_detection2_se_se_squeeze_conv_convolution = (
            self.pass3_block1_mbconv3_detection2_se_se_squeeze_conv_convolution(
                pass3_block1_mbconv3_detection2_se_se_squeeze_lambda_Mean
            )
        )
        pass3_block1_mbconv3_detection2_se_se_squeeze_eswish_mul = (
            self.pass3_block1_mbconv3_detection2_se_se_squeeze_eswish_mul_x
            * pass3_block1_mbconv3_detection2_se_se_squeeze_conv_convolution
        )
        pass3_block1_mbconv3_detection2_se_se_squeeze_eswish_Sigmoid = F.sigmoid(
            pass3_block1_mbconv3_detection2_se_se_squeeze_conv_convolution
        )
        pass3_block1_mbconv3_detection2_se_se_squeeze_eswish_mul_1 = (
            pass3_block1_mbconv3_detection2_se_se_squeeze_eswish_mul
            * pass3_block1_mbconv3_detection2_se_se_squeeze_eswish_Sigmoid
        )
        pass3_block1_mbconv3_detection2_se_se_excite_conv_convolution = (
            self.pass3_block1_mbconv3_detection2_se_se_excite_conv_convolution(
                pass3_block1_mbconv3_detection2_se_se_squeeze_eswish_mul_1
            )
        )
        pass3_block1_mbconv3_detection2_se_se_excite_sigmoid_Sigmoid = F.sigmoid(
            pass3_block1_mbconv3_detection2_se_se_excite_conv_convolution
        )
        pass3_block1_mbconv3_detection2_se_se_multiply_mul = (
            pass3_block1_mbconv3_detection2_se_se_excite_sigmoid_Sigmoid
            * pass3_block1_mbconv3_detection2_dconv1_eswish_mul_1
        )
        pass3_block1_mbconv3_detection2_conv2_convolution = (
            self.pass3_block1_mbconv3_detection2_conv2_convolution(
                pass3_block1_mbconv3_detection2_se_se_multiply_mul
            )
        )
        pass3_block1_mbconv3_detection2_conv2_bn_FusedBatchNorm_1 = (
            self.pass3_block1_mbconv3_detection2_conv2_bn_FusedBatchNorm_1(
                pass3_block1_mbconv3_detection2_conv2_convolution
            )
        )
        pass3_block1_mbconv3_detection2_dense_concat = torch.cat(
            (
                pass3_block1_mbconv3_detection2_conv2_bn_FusedBatchNorm_1,
                pass3_block1_mbconv2_detection2_dense_concat,
            ),
            1,
        )
        pass3_detection2_confs_convolution = self.pass3_detection2_confs_convolution(
            pass3_block1_mbconv3_detection2_dense_concat
        )
        transposed_convolution_1 = self.__transposed(channels=16, kernel_size=4, stride=2)(
            pass3_detection2_confs_convolution
        )
        transposed_convolution_2 = self.__transposed(channels=16, kernel_size=4, stride=2)(
            transposed_convolution_1
        )
        transposed_convolution_3 = self.__transposed(channels=16, kernel_size=4, stride=2)(
            transposed_convolution_2
        )

        return transposed_convolution_3

    @staticmethod
    def __conv(dim, name, **kwargs):
        if dim == 1:
            layer = nn.Conv1d(**kwargs)  # noqa: E701 (kept verbatim from real repo)
        elif dim == 2:
            layer = nn.Conv2d(**kwargs)  # noqa: E701 (kept verbatim from real repo)
        elif dim == 3:
            layer = nn.Conv3d(**kwargs)  # noqa: E701 (kept verbatim from real repo)
        else:
            raise NotImplementedError()  # noqa: E701 (kept verbatim from real repo)

        # menagerie staging: the real repo only ships MMdnn-converted numpy weight
        # dicts (no shipped .npy); when no weight_file is loaded, keep the layer's
        # own random init (same real Conv*d construction/shape either way).
        if name in __weights_dict:
            layer.state_dict()["weight"].copy_(torch.from_numpy(__weights_dict[name]["weights"]))
            if "bias" in __weights_dict[name]:
                layer.state_dict()["bias"].copy_(torch.from_numpy(__weights_dict[name]["bias"]))
        return layer

    @staticmethod
    def __batch_normalization(dim, name, **kwargs):
        if dim == 0 or dim == 1:
            layer = nn.BatchNorm1d(**kwargs)  # noqa: E701 (kept verbatim from real repo)
        elif dim == 2:
            layer = nn.BatchNorm2d(**kwargs)  # noqa: E701 (kept verbatim from real repo)
        elif dim == 3:
            layer = nn.BatchNorm3d(**kwargs)  # noqa: E701 (kept verbatim from real repo)
        else:
            raise NotImplementedError()  # noqa: E701 (kept verbatim from real repo)

        # menagerie staging: fall back to the layer's own default init (running
        # stats zero/one, weight one, bias zero) when no MMdnn weight dict is loaded.
        if name in __weights_dict:
            if "scale" in __weights_dict[name]:
                layer.state_dict()["weight"].copy_(torch.from_numpy(__weights_dict[name]["scale"]))
            else:
                layer.weight.data.fill_(1)

            if "bias" in __weights_dict[name]:
                layer.state_dict()["bias"].copy_(torch.from_numpy(__weights_dict[name]["bias"]))
            else:
                layer.bias.data.fill_(0)

            layer.state_dict()["running_mean"].copy_(torch.from_numpy(__weights_dict[name]["mean"]))
            layer.state_dict()["running_var"].copy_(torch.from_numpy(__weights_dict[name]["var"]))
        return layer

    @staticmethod
    def __transposed(channels, kernel_size, stride):
        return pytorch_BilinearConvTranspose2d(
            channels=channels, kernel_size=kernel_size, stride=stride
        )


MENAGERIE_ZOO = "vendored-pytorch"

_INPUT_SIZE = 128


def build_efficientposert():
    torch.manual_seed(0)
    model = KitModel(weight_file=None)
    model.eval()
    return model


def example_input_efficientposert():
    torch.manual_seed(0)
    return torch.randn(1, 3, _INPUT_SIZE, _INPUT_SIZE)


MENAGERIE_ENTRIES = [
    (
        "EfficientPose-RT",
        "build_efficientposert",
        "example_input_efficientposert",
        2021,
        MENAGERIE_ZOO,
    ),
]
