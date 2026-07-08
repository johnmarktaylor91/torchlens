# FAITHFUL PORT of KrishnaswamyLab/SAUCIE @ 5ab7976fc8d19a3823b6005f51736dc70f2017fb
# (original framework: TensorFlow 1.x, raw tf.layers/tf.placeholder graph API)
# https://github.com/KrishnaswamyLab/SAUCIE/blob/master/model.py
#
# SAUCIE (Amodio et al. 2019, "Exploring single-cell data with deep
# multitasking neural networks") is a symmetric MLP autoencoder for
# single-cell data with optional batch-correction (MMD), ID-regularization,
# and intracluster-distance regularization terms baked into the loss (not
# the forward architecture). This file transcribes the *inference* forward
# architecture faithfully from `SAUCIE._build_layers`'s default (no
# lambda_b/lambda_c) branch in the real repo's model.py:
#   encoder0: Linear -> LeakyReLU(0.2)         (layers[0]=512)
#   encoder1: Linear -> Sigmoid                (layers[1]=256)
#   encoder2: Linear -> LeakyReLU(0.2)         (layers[2]=128)
#   embedding: Linear -> Identity              (layers[3]=2)   <- 'embeddings' tensor
#   decoder0: Linear -> LeakyReLU(0.2)         (layers[2]=128)
#   decoder1: Linear -> LeakyReLU(0.2)         (layers[1]=256)
#   decoder2: Linear -> LeakyReLU(0.2)         (layers[0]=512) <- 'layer_c' tensor
#   recon:    Linear -> Identity               (input_dim)     <- 'output' tensor
# The original repo cannot run in this base env (TensorFlow 1.x
# `tf.placeholder`/`tf.layers`/`tf.Session` graph-mode API, incompatible with
# the installed TF2/no-TF environment), so this is a faithful architectural
# port rather than a vendor-as-is. Only the training-loop / loss / MMD /
# clustering machinery (which lives outside `_build_layers`) is omitted,
# since it is not part of the forward network graph.

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class SAUCIE(nn.Module):
    """Faithful port of SAUCIE's default (no batch-correction, no ID-reg)
    forward architecture: a symmetric MLP autoencoder with a 2D bottleneck
    ('embeddings') and a pre-output 'layer_c' activation tap, matching the
    real repo's `_build_layers` else-branch layer-for-layer."""

    def __init__(self, input_dim, layers=(512, 256, 128, 2), leak=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.layers_cfg = layers

        self.encoder0 = nn.Linear(input_dim, layers[0])
        self.encoder1 = nn.Linear(layers[0], layers[1])
        self.encoder2 = nn.Linear(layers[1], layers[2])
        self.embedding = nn.Linear(layers[2], layers[3])

        self.decoder0 = nn.Linear(layers[3], layers[2])
        self.decoder1 = nn.Linear(layers[2], layers[1])
        self.decoder2 = nn.Linear(layers[1], layers[0])
        self.recon = nn.Linear(layers[0], input_dim)

        self.lrelu = nn.LeakyReLU(negative_slope=leak)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h1 = self.lrelu(self.encoder0(x))
        h2 = self.sigmoid(self.encoder1(h1))
        h3 = self.lrelu(self.encoder2(h2))

        embedded = self.embedding(h3)  # 'embeddings' tensor, identity activation

        h5 = self.lrelu(self.decoder0(embedded))
        h6 = self.lrelu(self.decoder1(h5))
        h7 = self.lrelu(self.decoder2(h6))  # 'layer_c' tensor

        reconstructed = self.recon(h7)  # 'output' tensor, identity activation
        return reconstructed


def build_saucie():
    return SAUCIE(input_dim=64, layers=(64, 32, 16, 2))


def example_input_saucie():
    return (torch.randn(8, 64),)


MENAGERIE_ENTRIES = [
    ("SAUCIE", build_saucie, example_input_saucie, 2019, "ported-pytorch"),
]
