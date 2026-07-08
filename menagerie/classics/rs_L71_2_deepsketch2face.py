# FAITHFUL REIMPLEMENTATION from Han, Gao & Yu, "DeepSketch2Face: A Deep
# Learning Based Sketching System for 3D Face and Caricature Modeling",
# ACM Trans. Graph. 36(4), Article 126 (SIGGRAPH 2017), arXiv:1706.02042
# (no public code: the official repo, github.com/irsisyphus/deepsketch2face
# a.k.a. github.com/liuf1989/deepsketch2face, ships only a Windows demo
# binary + README; the Caffe C++/Qt source referenced by the README was
# never released).
#
# Architecture, transcribed from Section 4.3 "Network Architecture" and
# Figure 7 of the paper:
#   - Pixel-level input: a 256x256 binary sketch image fed through an
#     AlexNet-style convolutional stack (conv1 11x11/96ch -> conv2
#     5x5/256ch -> conv3 3x3/384ch -> conv4 3x3/384ch -> conv5 3x3/256ch,
#     "set up the same way as in AlexNet (Krizhevsky et al. 2012)",
#     flattened to a 4096-d feature vector via the AlexNet FC6 layer).
#   - Shape-level input: a 66-d vector of 2D bilinear-encoded sample
#     points along the silhouette/feature lines, passed through one new
#     FC layer of 512 neurons.
#   - The 4096-d pixel feature and 512-d shape feature are concatenated
#     and fed into two independent branches of fully connected layers
#     (each FC layer in both branches has 1024 neurons):
#       * u-branch (identity, 50 coefficients): 3 new FC(1024) layers,
#         then a linear output layer of size 50.
#       * v-branch (expression, 16 coefficients): 1 new FC(1024) layer,
#         then a linear output layer of size 16.
#   - The u,v outputs are the coefficients of a bilinear face
#     representation (Eq. 1); the downstream vertex-loss / mesh
#     reconstruction (V = C x2 u^T x3 v^T against a precomputed core
#     tensor C) is a fixed linear algebra step over per-dataset tensors,
#     not a learned network layer, so it is outside the regression
#     network reproduced here (the u/v coefficient outputs are the
#     network's actual output, matching Figure 7's "Bilinear Output"
#     box which is drawn as a downstream mesh-reconstruction sink, not
#     an additional trainable layer).

import torch
import torch.nn as nn


class DeepSketch2FaceNet(nn.Module):
    """Deep regression network mapping a 2D face sketch (+ sampled
    silhouette/feature-line points) to bilinear face-model coefficients.

    Faithful to Figure 7 / Section 4.3 of the DeepSketch2Face paper:
    an AlexNet-style conv stack over the 256x256 binary sketch, fused
    with a shape-level FC branch over 66 sampled 2D points, feeding two
    independent FC branches that predict the identity (u, dim 50) and
    expression (v, dim 16) bilinear coefficients.
    """

    def __init__(self, u_dim: int = 50, v_dim: int = 16, shape_dim: int = 66):
        super().__init__()

        # AlexNet-style convolutional feature extractor over the
        # 256x256 binary pixel-level sketch input (single channel).
        self.conv_features = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # AlexNet FC6-equivalent: flatten -> 4096-d pixel-level feature.
        self.pixel_fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
        )

        # Shape-level input branch: 66-d sampled silhouette/feature-line
        # coefficients -> new FC layer of 512 neurons.
        self.shape_fc = nn.Sequential(
            nn.Linear(shape_dim, 512),
            nn.ReLU(inplace=True),
        )

        fused_dim = 4096 + 512

        # Identity branch (u vector, 50 coefficients): three new
        # FC(1024) layers followed by a linear output layer.
        self.u_branch = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
        )
        self.u_out = nn.Linear(1024, u_dim)

        # Expression branch (v vector, 16 coefficients): one new
        # FC(1024) layer followed by a linear output layer.
        self.v_branch = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.ReLU(inplace=True),
        )
        self.v_out = nn.Linear(1024, v_dim)

    def forward(self, sketch: torch.Tensor, shape_points: torch.Tensor):
        pixel_feat = self.conv_features(sketch)
        pixel_feat = self.avgpool(pixel_feat)
        pixel_feat = torch.flatten(pixel_feat, 1)
        pixel_feat = self.pixel_fc(pixel_feat)

        shape_feat = self.shape_fc(shape_points)

        fused = torch.cat([pixel_feat, shape_feat], dim=1)

        u = self.u_out(self.u_branch(fused))
        v = self.v_out(self.v_branch(fused))
        return u, v


def build_deepsketch2face():
    return DeepSketch2FaceNet()


def example_input_deepsketch2face():
    sketch = torch.randn(1, 1, 256, 256)
    shape_points = torch.randn(1, 66)
    return (sketch, shape_points)


MENAGERIE_ZOO = "reimpl-pytorch"
MENAGERIE_ENTRIES = [
    (
        "DeepSketch2Face",
        "build_deepsketch2face",
        "example_input_deepsketch2face",
        2017,
        "reimpl",
    ),
]
