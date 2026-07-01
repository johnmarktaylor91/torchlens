# FAITHFUL PORT of divya031090/taxoNN_OTU @ master (original framework: TensorFlow 1.x
# graph-mode for the per-phylum CNNs (`NN_Cirr.py`/`NN_Sim.py`/`NN_T2D.py`, using
# `tf.placeholder`/`tf.InteractiveSession`/`tf.train.AdadeltaOptimizer` -- legacy TF1
# API, not runnable under the installed TF2/Keras or torch stack) + Keras for the
# stacking ensemble head (`ensembling_Cirr.py`))
#
# TaxoNN (Sharma et al. 2020) classifies host disease status (e.g. Cirrhosis vs
# control) from gut-microbiome OTU (operational taxonomic unit) abundance profiles. Its
# architecture is a PHYLUM-STRATIFIED ENSEMBLE OF 1-D CNNS: OTU features are grouped by
# their phylum-level taxonomy, each phylum group gets its own small 1D-CNN classifier
# (`conv_layer` in NN_Cirr.py: Conv2d-as-1D [1 x k] kernel -> ReLU -> 1D max-pool,
# stacked twice, then a dense hidden layer and a softmax readout -- see the real
# `conv_layer`/graph-construction code in NN_Cirr.py), and the `n_members` per-phylum
# CNNs' final hidden (or output) representations are concatenated and passed through a
# small stacking MLP (`define_stacked_model` in ensembling_Cirr.py: concatenate ->
# Dense(10, relu) -> Dense(n_classes, softmax)) to produce the final disease-status
# prediction. This module transcribes that exact two-level architecture (per-phylum
# CNN member + concatenation-based ensemble head) faithfully into torch, since the
# original graph-mode TF1 + Keras H5-model-file code cannot run under the installed
# base env (TF2/Keras 3 dropped the `tf.placeholder`/graph-mode API entirely, and the
# ensembling stage literally reloads independently-trained `.h5` files from disk, which
# has no meaningful "random init" analogue).
#
# `PhylumCNN` mirrors NN_Cirr.py's `conv_layer` calls: input reshaped to
# [B, 1, n_otu_features, 1] (`x_shape = tf.reshape(x, [-1,1,feature,1])`), passed
# through two `conv_layer` stages (Conv[1,k]-stride1-SAME -> ReLU -> MaxPool[1,k]),
# then flattened and passed through the dense hidden layer + softmax readout
# (`h_fc1`/`y_conv` in NN_Cirr.py). `TaxoNNEnsemble` mirrors ensembling_Cirr.py's
# `define_stacked_model`: run every phylum member CNN on the same OTU input tensor
# (`X = [inputX for _ in range(len(model.input))]` -- every member consumes the full
# input in the real code, since phylum-specific slicing happens upstream in the OTU
# data split, not inside `define_stacked_model`), concatenate their softmax outputs,
# then Dense(10, relu) -> Dense(n_classes, softmax).

import torch
import torch.nn as nn


class PhylumCNN(nn.Module):
    """One phylum-specific member CNN, matching NN_Cirr.py's `conv_layer`-based graph:
    two [1 x k] Conv+ReLU+MaxPool stages over the length-`n_features` OTU vector
    (reshaped to a [1, n_features] "image" exactly as `x_shape = tf.reshape(x,
    [-1,1,feature,1])` does), then a dense hidden layer + softmax classification head
    (`h_fc1`/`y_conv`)."""

    def __init__(
        self,
        n_features,
        n_classes=2,
        conv1_channels=32,
        conv2_channels=64,
        dense_dim=1024,
        kernel_size=1,
        pool_size=1,
    ):
        super().__init__()
        self.n_features = n_features
        # Conv2d with a [1, k] kernel over a [1, n_features] "image" (matches
        # `conv2d(x, W, stride)` using strides=[1,1,stride_feat,1], padding='SAME').
        self.conv1 = nn.Conv2d(
            1,
            conv1_channels,
            kernel_size=(1, kernel_size),
            stride=(1, 1),
            padding=(0, kernel_size // 2),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(1, pool_size), stride=(1, pool_size))
        self.conv2 = nn.Conv2d(
            conv1_channels,
            conv2_channels,
            kernel_size=(1, kernel_size),
            stride=(1, 1),
            padding=(0, kernel_size // 2),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(1, pool_size), stride=(1, pool_size))
        self.relu = nn.ReLU()

        # Infer flattened dim exactly as the real code does by probing shapes (the real
        # NN_Cirr.py prints h_pool2's shape and hardcodes the resulting dim; here it is
        # computed directly instead of hardcoded, with identical conv/pool math).
        with torch.no_grad():
            probe = torch.zeros(1, 1, 1, n_features)
            probe = self.pool1(self.relu(self.conv1(probe)))
            probe = self.pool2(self.relu(self.conv2(probe)))
            flat_dim = probe.numel()

        self.fc1 = nn.Linear(flat_dim, dense_dim)
        self.fc2 = nn.Linear(dense_dim, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x: [B, n_features] OTU abundance vector -> reshape to [B, 1, 1, n_features]
        # (matches `tf.reshape(x, [-1,1,feature,1])`).
        x = x.view(x.shape[0], 1, 1, self.n_features)
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)
        h = torch.relu(self.fc1(x))
        out = self.softmax(self.fc2(h))
        return out


class TaxoNNEnsemble(nn.Module):
    """Stacking ensemble over `n_members` phylum CNNs, matching ensembling_Cirr.py's
    `define_stacked_model`: run every member on the (full) OTU input, concatenate their
    softmax outputs, then Dense(10, relu) -> Dense(n_classes, softmax)."""

    def __init__(self, n_features, n_members=4, n_classes=2, member_n_classes=2):
        super().__init__()
        self.members = nn.ModuleList(
            [PhylumCNN(n_features, n_classes=member_n_classes) for _ in range(n_members)]
        )
        self.stack_hidden = nn.Linear(n_members * member_n_classes, 10)
        self.stack_out = nn.Linear(10, n_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        member_outputs = [member(x) for member in self.members]
        merged = torch.cat(member_outputs, dim=1)
        h = self.relu(self.stack_hidden(merged))
        out = self.softmax(self.stack_out(h))
        return out


# --- staging harness (tiny sizes; not part of the real repo) ---


def build_taxonn_ensemble():
    # Real usage: n_features = 185 OTU features (Cirrhosis dataset, see NN_Cirr.py's
    # `feature = 185`), n_members = 4 phylum-stratified CNNs (see ensembling_Cirr.py's
    # `n_members = 4`), n_classes = 2 (Positive/Negative disease status). Kept at the
    # real repo's exact sizes since they are already small.
    return TaxoNNEnsemble(n_features=185, n_members=4, n_classes=2, member_n_classes=2)


def example_input_taxonn_ensemble():
    return (torch.rand(6, 185),)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("TaxoNN", "build_taxonn_ensemble", "example_input_taxonn_ensemble", 2020, "ported"),
]
