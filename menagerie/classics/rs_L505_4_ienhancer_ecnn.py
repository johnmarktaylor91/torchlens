# SOURCE: vendored from ngphubinh/enhancers @ master (external code archive linked from the
# repo README's "Download" section: https://homepages.ecs.vuw.ac.nz/~nguyenb5/bioinformatics/
# enhancers/code_enhancers.zip -- the ngphubinh/enhancers GitHub repo itself hosts only the
# README/figures; the authors' official code drop is this zip, Test/test_layer1.py, class
# EnhancerCnnModel)
#
# iEnhancer-ECNN: identifying enhancers and their strength using ensembles of convolutional
# neural networks (Nguyen et al., Bioinformatics 2019). `EnhancerCnnModel` is the real, base
# single-CNN member of the paper's ensemble (the "ECNN" ensemble trains 10 independently-seeded
# copies of this exact architecture and averages their outputs; the architecture itself is this
# one class) -- six 1-D conv blocks with BatchNorm in two 3-block stages (32 -> 32 -> 32 filters,
# then 64 -> 64 -> 64 filters), a MaxPool1d after each stage, then two FC layers (768 -> 256,
# 256 -> 1) with a final sigmoid, exactly matching the CNN architecture figure in the repo
# README. Copied verbatim aside from dropping the `self.criterion = nn.BCELoss()` training-only
# attribute and the surrounding data-loading/training/evaluation script code in the same file
# (argparse-driven training loop, k-fold CV, motif/AUC reporting -- none of which is part of the
# traced network).
import torch
import torch.nn.functional as F
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"

SAMPLE_LENGTH = 200
CONV1D_KERNEL_SIZE = 3
CONV1D_FEATURE_SIZE_BLOCK1 = 32
CONV1D_FEATURE_SIZE_BLOCK2 = 64
AVGPOOL1D_KERNEL_SIZE = 4
FULLY_CONNECTED_LAYER_SIZE = 256


class EnhancerCnnModel(nn.Module):
    def __init__(self):
        super(EnhancerCnnModel, self).__init__()
        self.c1_1 = nn.Conv1d(8, CONV1D_FEATURE_SIZE_BLOCK1, CONV1D_KERNEL_SIZE, padding=1)
        self.c1_1bn = nn.BatchNorm1d(CONV1D_FEATURE_SIZE_BLOCK1)
        self.c1_2 = nn.Conv1d(
            CONV1D_FEATURE_SIZE_BLOCK1, CONV1D_FEATURE_SIZE_BLOCK1, CONV1D_KERNEL_SIZE, padding=1
        )
        self.c1_2bn = nn.BatchNorm1d(CONV1D_FEATURE_SIZE_BLOCK1)
        self.c1_3 = nn.Conv1d(
            CONV1D_FEATURE_SIZE_BLOCK1, CONV1D_FEATURE_SIZE_BLOCK1, CONV1D_KERNEL_SIZE, padding=1
        )
        self.c1_3bn = nn.BatchNorm1d(CONV1D_FEATURE_SIZE_BLOCK1)
        self.p1 = nn.MaxPool1d(AVGPOOL1D_KERNEL_SIZE)

        self.c2_1 = nn.Conv1d(
            CONV1D_FEATURE_SIZE_BLOCK1, CONV1D_FEATURE_SIZE_BLOCK2, CONV1D_KERNEL_SIZE, padding=1
        )
        self.c2_1bn = nn.BatchNorm1d(CONV1D_FEATURE_SIZE_BLOCK2)
        self.c2_2 = nn.Conv1d(
            CONV1D_FEATURE_SIZE_BLOCK2, CONV1D_FEATURE_SIZE_BLOCK2, CONV1D_KERNEL_SIZE, padding=1
        )
        self.c2_2bn = nn.BatchNorm1d(CONV1D_FEATURE_SIZE_BLOCK2)
        self.c2_3 = nn.Conv1d(
            CONV1D_FEATURE_SIZE_BLOCK2, CONV1D_FEATURE_SIZE_BLOCK2, CONV1D_KERNEL_SIZE, padding=1
        )
        self.c2_3bn = nn.BatchNorm1d(CONV1D_FEATURE_SIZE_BLOCK2)
        self.p2 = nn.MaxPool1d(AVGPOOL1D_KERNEL_SIZE)

        self.fc = nn.Linear(768, FULLY_CONNECTED_LAYER_SIZE)
        self.out = nn.Linear(FULLY_CONNECTED_LAYER_SIZE, 1)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        output = F.relu(self.c1_1bn(self.c1_1(inputs)))
        output = F.relu(self.c1_2bn(self.c1_2(output)))
        output = F.relu(self.c1_3bn(self.c1_3(output)))
        output = self.p1(output)

        output = F.relu(self.c2_1bn(self.c2_1(output)))
        output = F.relu(self.c2_2bn(self.c2_2(output)))
        output = F.relu(self.c2_3bn(self.c2_3(output)))
        output = self.p2(output)

        output = output.view(batch_size, -1)

        output = F.relu(self.fc(output))

        output = torch.sigmoid(self.out(output))

        return output


def build_ienhancer_ecnn():
    return EnhancerCnnModel()


def example_input_ienhancer_ecnn():
    # (batch, 8, 200): 4 one-hot bases + 1-mer + 2-mer + 3-mer frequency channels, length 200.
    return torch.rand(2, 8, SAMPLE_LENGTH)


MENAGERIE_ENTRIES = [
    (
        "iEnhancer-ECNN",
        build_ienhancer_ecnn,
        example_input_ienhancer_ecnn,
        2019,
        "SOURCE_AVAILABLE",
    ),
]
