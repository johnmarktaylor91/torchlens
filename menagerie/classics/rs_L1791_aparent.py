# FAITHFUL PORT of johli/aparent @ master (original framework: TensorFlow 1.13 / Keras 2.2)
# https://raw.githubusercontent.com/johli/aparent/master/aparent/model/aparent_model_plasmid_large.py
#
# Bogard, Linder, Rosenberg, Seelig 2019 (Cell) "A Deep Neural Network for Predicting and
# Engineering Alternative Polyadenylation" -- APARENT (APA Regression Net), the
# "plasmid_large" architecture (`load_aparent_model` in `aparent_model_plasmid_large.py`,
# the final/largest published variant). The real repo is TF1.13/Keras-2.2 (functional-API
# `Model`) and additionally depends on the non-base `isolearn` package for its custom
# data-generator loss wiring -- it cannot run in the base torch env, so the architecture is
# TRANSCRIBED (not guessed) directly from the real Keras layer graph into PyTorch:
#
#   seq_input: (batch, 4, 205)     one-hot 3'UTR sequence (205nt x {A,C,G,T}), channel-first
#   lib_input: (batch, 13)         one-hot library/context indicator
#   distal_pas_input: (batch, 1)   distance-to-distal-PAS feature
#
#   layer_1      = Conv2d(4->96, kernel=(4,8), valid)          [Keras Conv2D(96,(8,4))]
#   layer_1_pool = MaxPool2d(kernel=(1,2))                     [Keras MaxPooling2D((2,1))]
#   layer_2      = Conv2d(96->128, kernel=(1,6), valid)        [Keras Conv2D(128,(6,1))]
#   flatten -> concat(distal_pas_input) -> Linear(->512) -> ReLU -> Dropout(0.2)
#            -> Linear(512->256) -> ReLU -> Dropout(0.2)                 [shared_model]
#   concat(lib_input) -> two heads:
#     cut head:  Linear(->206) -> Softmax   (cleavage-site distribution, 206 positions)
#     iso head:  Linear(->1)   -> Sigmoid   (distal isoform usage)
#
# Keras's `Conv2D(filters, (kh, kw))` on an `(H, W, C)`-last input applies the kernel over
# (H=sequence position, W=one-hot base); the PyTorch port keeps sequence-position on dim 2
# and the one-hot base axis (4) as the second spatial dim of a single `Conv2d` (channels=1
# in, matching Keras's `seq_input_shape=(205,4,1)`), so `layer_1`'s kernel is (8,4) exactly
# as in Keras (kernel spans all 4 bases + 8 positions), and `layer_1_pool`/`layer_2` mirror
# the Keras `(2,1)`/`(6,1)` shapes with axes correspondingly swapped. All layer widths,
# activations, dropout rates, and the two-head softmax/sigmoid outputs match the real
# `aparent_model_plasmid_large.py` `plasmid_model` (the loss-model / SeqProp / isolearn
# training machinery is out of scope -- this ports the trainable network only).

import torch
import torch.nn as nn


class APARENT(nn.Module):
    def __init__(self, seq_len=205, lib_size=13):
        super().__init__()
        self.seq_len = seq_len
        self.lib_size = lib_size

        # Keras Conv2D(96, (8, 4), padding='valid') on (seq_len, 4, 1) NHWC input
        # -> PyTorch Conv2d(1, 96, kernel=(8, 4)) on (batch, 1, seq_len, 4) NCHW input.
        self.layer_1 = nn.Conv2d(1, 96, kernel_size=(8, 4))
        self.relu_1 = nn.ReLU()
        # Keras MaxPooling2D(pool_size=(2, 1)) -> pools over the sequence-position axis.
        self.layer_1_pool = nn.MaxPool2d(kernel_size=(2, 1))
        # Keras Conv2D(128, (6, 1), padding='valid')
        self.layer_2 = nn.Conv2d(96, 128, kernel_size=(6, 1))
        self.relu_2 = nn.ReLU()

        pooled_len = (seq_len - 8 + 1) // 2  # after layer_1 (valid) + (2,1) maxpool
        conv2_len = pooled_len - 6 + 1  # after layer_2 (valid), width axis is now 1
        flat_features = conv2_len * 128

        self.layer_dense = nn.Linear(flat_features + 1, 512)  # + distal_pas_input
        self.relu_dense = nn.ReLU()
        self.layer_drop = nn.Dropout(0.2)
        self.layer_dense2 = nn.Linear(512, 256)
        self.relu_dense2 = nn.ReLU()
        self.layer_drop2 = nn.Dropout(0.2)

        self.out_cut = nn.Linear(256 + lib_size, 206)
        self.softmax_cut = nn.Softmax(dim=-1)
        self.out_iso = nn.Linear(256 + lib_size, 1)
        self.sigmoid_iso = nn.Sigmoid()

    def forward(self, seq_input, lib_input, distal_pas_input):
        # seq_input: (batch, 1, seq_len, 4)
        x = self.layer_1(seq_input)
        x = self.relu_1(x)
        x = self.layer_1_pool(x)
        x = self.layer_2(x)
        x = self.relu_2(x)

        x = x.flatten(1)
        x = torch.cat([x, distal_pas_input], dim=1)
        x = self.layer_dense(x)
        x = self.relu_dense(x)
        x = self.layer_drop(x)
        x = self.layer_dense2(x)
        x = self.relu_dense2(x)
        shared = self.layer_drop2(x)

        fused = torch.cat([shared, lib_input], dim=1)
        out_cut = self.softmax_cut(self.out_cut(fused))
        out_iso = self.sigmoid_iso(self.out_iso(fused))
        return out_iso, out_cut


def build_aparent():
    return APARENT(seq_len=205, lib_size=13)


def example_input_aparent():
    seq_input = torch.randn(1, 1, 205, 4)
    lib_input = torch.zeros(1, 13)
    lib_input[0, 0] = 1.0
    distal_pas_input = torch.zeros(1, 1)
    return (seq_input, lib_input, distal_pas_input)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("APARENT", "build_aparent", "example_input_aparent", 2019, "ported"),
]
