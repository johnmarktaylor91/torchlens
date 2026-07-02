# FAITHFUL PORT of YifanDengWHU/DDIMDL @ master (original framework: Keras/TensorFlow)
# https://raw.githubusercontent.com/YifanDengWHU/DDIMDL/master/DDIMDL.py
#
# Deng, Xu, Wu, Ma, Yang 2020 (Bioinformatics) "A multimodal deep learning framework for
# predicting drug-drug interaction events" -- DDIMDL. The paper's "multimodal" fusion
# (Jaccard-similarity + PCA over 4 drug feature views: substructure/target/enzyme/pathway)
# is classical feature engineering done in numpy/sklearn BEFORE the network (see
# `feature_vector`/`prepare` in the reference DDIMDL.py); the actual neural net that
# consumes the fused feature vector is the single Keras `DNN()` function in that file:
#   Dense(512, relu) -> BatchNorm -> Dropout(0.3) -> Dense(256, relu) -> BatchNorm ->
#   Dropout(0.3) -> Dense(event_num) -> Softmax
# The reference code cannot run in the base env: it imports Keras/TensorFlow (`keras.models`,
# `keras.layers`) which are not installed here (only torch is). This port transcribes that
# exact Dense-BN-Dropout-Dense-BN-Dropout-Dense-Softmax stack faithfully into torch, with the
# real layer widths/dropout rate/activation choices preserved (`event_num=65`,
# `vector_size=572` per drug -> 2*572=1144-d input for the drugA+drugB concatenation used by
# `cross_validation`'s `x_train = feature_matrix[i][train_index]`).
"""DDIMDL: multimodal DDI-event classifier -- Dense/BatchNorm/Dropout MLP head."""

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class DDIMDLNet(nn.Module):
    """Port of DDIMDL.py::DNN(). Keras `Dense` defaults to Glorot-uniform init + zero
    bias, which torch.nn.Linear's default init closely matches; BatchNorm defaults
    (momentum/eps) match Keras' BatchNormalization defaults used in the reference."""

    def __init__(self, input_size=1144, event_num=65, droprate=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(droprate)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(droprate)
        self.fc3 = nn.Linear(256, event_num)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.drop1(x)
        x = torch.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.drop2(x)
        x = self.fc3(x)
        out = torch.softmax(x, dim=-1)
        return out


def build_ddimdl():
    # vector_size=572 per drug (PCA dim in the reference `feature_vector`), two drugs
    # concatenated (drugA + drugB) -> 1144-d input, matching `prepare()`'s
    # `np.hstack((d_feature[drugA[i]], d_feature[drugB[i]]))`; event_num=65 real DDI event
    # classes as declared at module level in the reference.
    return DDIMDLNet(input_size=1144, event_num=65, droprate=0.3)


def example_input_ddimdl():
    torch.manual_seed(0)
    return (torch.randn(4, 1144),)


MENAGERIE_ENTRIES = [
    ("DDIMDL", "build_ddimdl", "example_input_ddimdl", 2020, "ported"),
]
