# SOURCE: vendored from ChiehLo/MetaNN @ master
# https://raw.githubusercontent.com/ChiehLo/MetaNN/master/DataSet/HMP/Code/classifier/NN_HMP_cuda.py
#
# Lo, Marculescu 2019 (BMC Bioinformatics) "MetaNN: accurate classification of host
# phenotypes from metagenomic data using neural networks" -- dense-feedforward (MLP)
# and 1D-convolutional classifiers that map metagenomic taxon-relative-abundance
# vectors (HMP dataset, input_dim=323 features) to a host-phenotype class label
# (output_dim=5 classes). `Net` (3-hidden-layer MLP with dropout, dims
# 323 -> 512 -> 256 -> 128 -> 5, matching the real script's `h1_dim/h2_dim/h3_dim`
# module-level constants) and `CNN1d` (two Conv1d+MaxPool1d blocks over the
# abundance vector treated as a length-323 1-channel signal, flattened to a linear
# classifier head, matching the real script's `net = CNN1d(n_feature=input_dim,
# out_dim=160, n_output=output_dim)` call site) are copied verbatim from the real
# `NN_HMP_cuda.py`. No architectural changes were made; only the argparse/training
# driver code (data loading, augmentation-set concatenation, optimizer loop) is
# dropped, as it is training plumbing, not part of the network definition.

import torch
import torch.nn.functional as F


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_hidden2, n_hidden3, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden2)  # hidden layer
        self.hidden3 = torch.nn.Linear(n_hidden2, n_hidden3)  # hidden layer
        self.dropout1 = torch.nn.Dropout(p=0.0)  # hidden layer
        self.out = torch.nn.Linear(n_hidden3, n_output)  # output layer

    def forward(self, x):
        x = self.hidden(x)
        x = self.dropout1(x)
        x = F.relu(x)  # activation function for hidden layer
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.hidden3(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.out(x)
        return x


class CNN1d(torch.nn.Module):
    def __init__(self, n_feature, out_dim, n_output):
        super(CNN1d, self).__init__()
        self.c1 = torch.nn.Conv1d(1, 8, 3, stride=2, padding=1)
        self.c2 = torch.nn.Conv1d(8, 8, 3, stride=2, padding=1)
        self.p1 = torch.nn.MaxPool1d(2)  # hidden layer
        self.p2 = torch.nn.MaxPool1d(2)  # hidden layer
        self.dropout1 = torch.nn.Dropout(p=0.0)  # hidden layer
        self.out = torch.nn.Linear(out_dim, n_output)  # output layer

    def forward(self, x):
        x = F.relu(self.dropout1(self.c1(x)))  # activation function for hidden layer
        x = self.p1(x)
        x = F.relu(self.c2(x))
        x = self.p2(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x


def build_metann_mlp():
    # Real dims from the script's module-level constants:
    # input_dim=323, h1_dim=512, h2_dim=256, h3_dim=128, output_dim=5.
    return Net(n_feature=323, n_hidden=512, n_hidden2=256, n_hidden3=128, n_output=5)


def example_input_metann_mlp():
    return torch.randn(4, 323)


def build_metann_cnn1d():
    # Real call site: CNN1d(n_feature=input_dim, out_dim=160, n_output=output_dim).
    return CNN1d(n_feature=323, out_dim=160, n_output=5)


def example_input_metann_cnn1d():
    # Real code reshapes the (batch, 323) abundance vector to (batch, 1, 323)
    # before feeding CNN1d (see `sample_tensor.view(sample_tensor.size(0), 1,
    # sample_tensor.size(1))` in the real training script).
    return torch.randn(4, 1, 323)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("MetaNN-MLP", "build_metann_mlp", "example_input_metann_mlp", 2019, "vendored"),
    ("MetaNN-CNN1d", "build_metann_cnn1d", "example_input_metann_cnn1d", 2019, "vendored"),
]
