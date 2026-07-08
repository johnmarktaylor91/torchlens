# SOURCE: vendored from BigDataBiology/SemiBin @ main
#
# SemiBin2's siamese-autoencoder binning networks, transcribed verbatim from
# `SemiBin/semi_supervised_model.py` (classes `Semi_encoding_multiple` and
# `Semi_encoding_single`). Both are contrastive siamese encoder/decoder MLPs used to
# embed metagenomic contig k-mer/coverage feature vectors for clustering; `_multiple`
# handles combined k-mer+multi-sample-coverage features (decoder ends in Sigmoid),
# `_single` handles k-mer-only features (decoder ends in Softmax). Only torch imports
# are used in the real class definitions -- no changes were made to layer types, sizes,
# or forward/decoder/embedding methods.

import torch
from torch import nn
from torch.nn import Linear, LeakyReLU


class Semi_encoding_multiple(torch.nn.Module):
    """
    Model for combined features
    """

    def __init__(self, num):
        super(Semi_encoding_multiple, self).__init__()
        self.encoder1 = torch.nn.Sequential(
            Linear(num, 512),
            nn.BatchNorm1d(512),
            LeakyReLU(),
            nn.Dropout(0.2),
            Linear(512, 512),
            nn.BatchNorm1d(512),
            LeakyReLU(),
            nn.Dropout(0.2),
            Linear(512, 100),
        )

        self.decoder1 = torch.nn.Sequential(
            Linear(100, 512),
            nn.BatchNorm1d(512),
            LeakyReLU(),
            nn.Dropout(0.2),
            Linear(512, 512),
            nn.BatchNorm1d(512),
            LeakyReLU(),
            nn.Dropout(0.2),
            Linear(512, num),
            nn.Sigmoid(),
        )

    def forward(self, input1, input2):
        return self.encoder1(input1), self.encoder1(input2)

    def decoder(self, input1, input2):
        return self.decoder1(input1), self.decoder1(input2)

    def embedding(self, input):
        return self.encoder1(input)


class Semi_encoding_single(torch.nn.Module):
    """
    Model for k-mer features
    """

    def __init__(self, num):
        super(Semi_encoding_single, self).__init__()
        self.encoder1 = torch.nn.Sequential(
            Linear(num, 512),
            nn.BatchNorm1d(512),
            LeakyReLU(),
            nn.Dropout(0.2),
            Linear(512, 512),
            nn.BatchNorm1d(512),
            LeakyReLU(),
            nn.Dropout(0.2),
            Linear(512, 100),
        )

        self.decoder1 = torch.nn.Sequential(
            Linear(100, 512),
            nn.BatchNorm1d(512),
            LeakyReLU(),
            nn.Dropout(0.2),
            Linear(512, 512),
            nn.BatchNorm1d(512),
            LeakyReLU(),
            nn.Dropout(0.2),
            Linear(512, num),
            nn.Softmax(dim=1),
        )

    def forward(self, input1, input2):
        return self.encoder1(input1), self.encoder1(input2)

    def decoder(self, input1, input2):
        return self.decoder1(input1), self.decoder1(input2)

    def embedding(self, input):
        return self.encoder1(input)


# --- staging harness (tiny sizes; not part of the real repo) ---


def build_semibin_multiple():
    # Real usage: `num` = combined k-mer(136) + multi-sample-coverage feature width
    # (data.shape[1], typically 138+). Model needs batch_size > 1 for BatchNorm1d.
    return Semi_encoding_multiple(num=138)


def example_input_semibin_multiple():
    return (torch.randn(4, 138), torch.randn(4, 138))


def build_semibin_single():
    # Real usage: `num` = k-mer-only feature width (136 for 4-mers).
    return Semi_encoding_single(num=136)


def example_input_semibin_single():
    return (torch.randn(4, 136), torch.randn(4, 136))


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "SemiBin2-MultiFeature",
        "build_semibin_multiple",
        "example_input_semibin_multiple",
        2023,
        "vendored",
    ),
    (
        "SemiBin2-KmerFeature",
        "build_semibin_single",
        "example_input_semibin_single",
        2023,
        "vendored",
    ),
]
