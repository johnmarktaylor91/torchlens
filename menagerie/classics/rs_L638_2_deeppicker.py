# FAITHFUL PORT of nejyeah/DeepPicker-python @ master (original framework: TensorFlow 1.x)
# Source: deepModel.py `DeepModel.__inference` / `init_model_graph_evaluate` (the cryo-EM
# particle-picking CNN classifier). The original graph-mode TF1 code (tf.get_variable +
# tf.nn.conv2d/max_pool, VALID padding, no bias in the raw kernel-var style but bias added
# via tf.nn.bias_add) is transcribed layer-for-layer into an equivalent torch nn.Module:
# 4x (conv VALID -> ReLU -> 2x2 maxpool VALID) then 2 FC layers (dropout 0.5 during
# training only, matching the original). Real hyperparameters from train.py/autoPick.py:
# model_input_size = [batch, 64, 64, 1] (single-channel 64x64 micrograph patch),
# num_class = 2 (particle / non-particle). Kernel sizes (9,5,3,2) and channel widths
# (1->8->16->32->64) and fc sizes (64*2*2 -> 128 -> num_class) are copied verbatim from
# deepModel.py's init_model_graph_evaluate. Only the framework is ported (TF1 -> torch);
# no architectural change.
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


class DeepPickerCNN(nn.Module):
    def __init__(self, num_class=2):
        super().__init__()
        # kernel1: [9, 9, 1, 8], kernel2: [5, 5, 8, 16], kernel3: [3, 3, 16, 32],
        # kernel4: [2, 2, 32, 64] -- all VALID (no) padding, matching tf.nn.conv2d.
        self.conv1 = nn.Conv2d(1, 8, kernel_size=9, padding=0)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, padding=0)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=0)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=2, padding=0)

        # 64x64 input -> conv1(9)->56 -> pool->28 -> conv2(5)->24 -> pool->12
        # -> conv3(3)->10 -> pool->5 -> conv4(2)->4 -> pool->2  =>  64*2*2 flattened dim
        dim = 64 * 2 * 2
        self.fc1 = nn.Linear(dim, 128)
        self.fc2 = nn.Linear(128, num_class)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x: [batch, 1, 64, 64]
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2, stride=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, stride=2)
        x = F.max_pool2d(F.relu(self.conv3(x)), kernel_size=2, stride=2)
        x = F.max_pool2d(F.relu(self.conv4(x)), kernel_size=2, stride=2)

        x = x.reshape(x.size(0), -1)
        if self.training:
            x = self.dropout(x)

        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits


def build_deeppicker():
    return DeepPickerCNN(num_class=2)


def example_input_deeppicker():
    torch.manual_seed(0)
    # [batch, 1, 64, 64] single-channel micrograph patch, matching
    # model_input_size = [batch, 64, 64, 1] from train.py/autoPick.py (NHWC -> NCHW).
    return (torch.randn(1, 1, 64, 64),)


MENAGERIE_ENTRIES = [
    ("DeepPicker", build_deeppicker, example_input_deeppicker, 2016, MENAGERIE_ZOO),
]
