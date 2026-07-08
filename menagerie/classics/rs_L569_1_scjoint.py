# SOURCE: vendored from SydneyBioX/scJoint @ main
# https://github.com/SydneyBioX/scJoint/blob/main/util/model_regress.py
#
# scJoint (Lin, Wu, Sun, Yang, Yang & Cao 2022, Nature Biotechnology,
# "scJoint integrates atlas-scale single-cell RNA-seq and ATAC-seq data
# through transfer learning") jointly embeds scRNA-seq and scATAC-seq (or
# CITE-seq / ASAP-seq) profiles into a shared low-dimensional space via a
# tiny shared linear encoder (Net_encoder) feeding a linear cell-type
# classification head (Net_cell); training uses a 3-stage transfer-learning
# procedure (not part of the architecture) with cosine-similarity-based
# "center loss" for cross-modality alignment. Vendored verbatim (import
# path only) from the real model file; no architecture change.

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


class Net_encoder(nn.Module):
    def __init__(self, input_size):
        super(Net_encoder, self).__init__()
        self.input_size = input_size
        self.k = 64
        self.f = 64

        self.encoder = nn.Sequential(nn.Linear(self.input_size, 64))

    def forward(self, data):
        data = data.float().view(-1, self.input_size)
        embedding = self.encoder(data)

        return embedding


class Net_cell(nn.Module):
    def __init__(self, num_of_class):
        super(Net_cell, self).__init__()
        self.cell = nn.Sequential(nn.Linear(64, num_of_class))

    def forward(self, embedding):
        cell_prediction = self.cell(embedding)

        return cell_prediction


class ScJoint(nn.Module):
    """Staging wrapper chaining the real Net_encoder -> Net_cell pair into a
    single traceable forward pass, matching how scJoint's training loop
    calls model_encoder(data) then model_cell(embedding) end-to-end."""

    def __init__(self, input_size, num_of_class):
        super().__init__()
        self.encoder = Net_encoder(input_size)
        self.cell = Net_cell(num_of_class)

    def forward(self, data):
        embedding = self.encoder(data)
        cell_prediction = self.cell(embedding)
        return embedding, cell_prediction


def build_scjoint():
    # Real config.py default (10x DB config): input_size=15463 genes,
    # number_of_class=11. Shrunk to a tiny gene count for menagerie tracing;
    # architecture (Linear(input_size,64) -> Linear(64,num_of_class)) unchanged.
    return ScJoint(input_size=256, num_of_class=11)


def example_input_scjoint():
    # (batch, input_size) gene-expression / gene-activity vector, matching
    # data.float().view(-1, self.input_size) in Net_encoder.forward.
    return (torch.randn(4, 256),)


MENAGERIE_ENTRIES = [
    ("scJoint", "build_scjoint", "example_input_scjoint", 2022, "vendored-pytorch"),
]
