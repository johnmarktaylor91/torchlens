# FAITHFUL PORT of cma2015/DeepGS @ master (original framework: R + MXNet)
# https://raw.githubusercontent.com/cma2015/DeepGS/master/R/DeepGS.R
#
# Ma et al. 2018 (Molecular Plant) "A Deep Convolutional Neural Network
# Approach for Predicting Phenotype from Genotype" -- DeepGS is an R package
# (`train_deepGSModel` in `R/DeepGS.R`) that constructs its CNN entirely via
# the `mxnet` R bindings (`mx.symbol.*`); there is no PyTorch/Python source
# anywhere in the repo (companion `FNNforMatlab/` is a separate MATLAB
# script, also not portable). R + mxnet cannot reasonably be installed in
# this base torch environment, so the architecture is faithfully
# TRANSCRIBED from the real `mx.symbol.*` call graph in `train_deepGSModel`
# into an equivalent torch `nn.Module`, using the exact layer sizes from the
# function's own `@examples` block (the paper's wheat-genotype config):
#   conv_kernel = "1*18", conv_stride = "1*1", conv_num_filter = 8,
#   pool_act_type = "relu", pool_type = "max", pool_kernel = "1*4",
#   pool_stride = "1*4", fullayer_num_hidden = c(32, 1),
#   fullayer_act_type = "sigmoid", drop_float = c(0.2, 0.1, 0.05)
#
# `train_deepGSModel`'s symbol graph (data -> Convolution -> Activation
# -> Pooling -> [per conv layer] -> Dropout -> Flatten -> FullyConnected
# -> Activation -> Dropout -> [per FC layer, last FC has no activation]
# -> LinearRegressionOutput) is reproduced 1:1 below as `DeepGSNet`:
# one Conv2d(1, 8, kernel=(1,18), stride=(1,1)) + ReLU + MaxPool2d(kernel=
# (1,4), stride=(1,4)) + Dropout(0.2), then Flatten -> Linear(->32) +
# Sigmoid + Dropout(0.1) -> Linear(32->1) + Dropout(0.05) (mxnet's
# `LinearRegressionOutput` is a training-time loss layer with an
# identity forward pass at inference, so it contributes no forward-pass
# computation and is omitted). Markers are encoded as a "1 x M" image
# (per `markerImage = paste0("1*", ncol(trainMat))` in the real usage
# example), i.e. NCHW input of shape (N, 1, 1, M).

import torch
import torch.nn as nn


class DeepGSNet(nn.Module):
    """Faithful port of the mx.symbol.* graph built by train_deepGSModel()
    for the paper's wheat genomic-selection config (see header)."""

    def __init__(
        self,
        n_markers=200,
        conv_num_filter=8,
        conv_kernel=18,
        pool_kernel=4,
        fc_hidden=32,
        drop_conv=0.2,
        drop_fc1=0.1,
        drop_fc2=0.05,
    ):
        super().__init__()
        self.conv = nn.Conv2d(1, conv_num_filter, kernel_size=(1, conv_kernel), stride=(1, 1))
        self.conv_act = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(1, pool_kernel), stride=(1, pool_kernel))
        self.drop_initial = nn.Dropout(p=drop_conv)

        conv_out_w = (n_markers - conv_kernel) // 1 + 1
        pool_out_w = (conv_out_w - pool_kernel) // pool_kernel + 1
        flat_dim = conv_num_filter * 1 * pool_out_w

        self.fc1 = nn.Linear(flat_dim, fc_hidden)
        self.fc1_act = nn.Sigmoid()
        self.drop1 = nn.Dropout(p=drop_fc1)

        self.fc2 = nn.Linear(fc_hidden, 1)
        self.drop2 = nn.Dropout(p=drop_fc2)
        # mx.symbol.LinearRegressionOutput is a training-time loss layer
        # (identity at inference) -- no module needed for forward pass.

    def forward(self, x):  # x: [N, 1, 1, M] marker "image"
        x = self.conv(x)
        x = self.conv_act(x)
        x = self.pool(x)
        x = self.drop_initial(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.fc1_act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def build_deepgs():
    return DeepGSNet(n_markers=200)


def example_input_deepgs():
    return torch.randn(2, 1, 1, 200)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("DeepGS", "build_deepgs", "example_input_deepgs", 2018, "ported"),
]
