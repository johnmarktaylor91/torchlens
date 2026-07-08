# FAITHFUL REIMPLEMENTATION from Mayr, Klambauer, Unterthiner & Hochreiter,
# "DeepTox: Toxicity Prediction using Deep Learning", Frontiers in Environmental
# Science 3:80 (2016) (no public code -- original Theano/Lasagne pipeline was never
# released, and www.bioinf.jku.at/research/DeepTox/ hosts only the Tox21 feature
# dataset, not model source). Section 2.2.4 "Hyperparameter settings and DNN network
# architectures" gives the exact architecture: "The networks consist of multiple layers
# of ReLUs, followed by a final layer of sigmoid output units, one for each task....DNNs
# with up to four hidden layers were tested" with "Number of Hidden Units {1024, 2048,
# 4096, 8192, 16384}" and "Dropout usage/rate {no, yes (50% Hidden Dropout, 20% Input
# Dropout)}" (Table 2). Section 2.2.3 confirms the multi-task sigmoid-cross-entropy output
# formulation (one sigmoid unit per of the 12 Tox21 toxicity-assay tasks). This file
# reimplements that fully-connected multi-task ReLU/dropout/sigmoid-output DNN faithfully
# per the paper's description: input dropout, N hidden ReLU layers each followed by
# dropout, and a final Linear+Sigmoid layer with one output per task.
"""Faithful reimplementation of the DeepTox multi-task DNN (Mayr et al. 2016)."""

import torch
import torch.nn as nn

MENAGERIE_ZOO = "reimpl-pytorch"


class DeepToxDNN(nn.Module):
    """Multi-task DNN toxicity predictor per DeepTox Section 2.2.4: stacked
    ReLU hidden layers (paper-searched widths 1024-16384, up to 4 layers) with
    50% hidden dropout / 20% input dropout, ending in a sigmoid output layer
    with one unit per toxicity-assay task (12 tasks in the Tox21 challenge).
    """

    def __init__(
        self,
        input_dim,
        hidden_units=128,
        num_hidden_layers=2,
        n_tasks=12,
        input_dropout=0.2,
        hidden_dropout=0.5,
    ):
        super().__init__()
        assert 1 <= num_hidden_layers <= 4

        layers = []
        layers.append(nn.Dropout(input_dropout))
        in_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(hidden_dropout))
            in_dim = hidden_units
        self.hidden = nn.Sequential(*layers)

        self.output_layer = nn.Linear(in_dim, n_tasks)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden(x)
        x = self.output_layer(x)
        return self.output_activation(x)


# ---------------------------------------------------------------------------
# Staging build/example helpers (tiny sizes vs. the paper's searched
# 1024-16384 widths, scaled down for fast tracing; static Tox21 feature
# vector stand-in as the single tensor input, 12-task multi-task output as
# used in the Tox21 challenge).
# ---------------------------------------------------------------------------


def build_deeptox():
    return DeepToxDNN(input_dim=100, hidden_units=64, num_hidden_layers=2, n_tasks=12)


def example_input_deeptox():
    return (torch.rand(4, 100),)


MENAGERIE_ENTRIES = [
    ("DeepTox", "build_deeptox", "example_input_deeptox", 2016, "reimpl-pytorch"),
]
