# FAITHFUL PORT of LiLabAtVT/DeepTE @ master
# (training_example_dir/classify_TE_keras_model_predict_kmer.py) (original framework: Keras/TensorFlow)
#
# DeepTE (Yan et al., Bioinformatics 2020) classifies transposable-element (TE)
# superfamilies from k-mer count vectors reshaped into a (1, 16384, 1) "1D-as-2D"
# tensor. The real repo's training script `train_model()` builds the classifier
# with Keras `Sequential`:
#   Conv2D(100, (1,3), relu) -> MaxPool2D((1,2))
#   -> Conv2D(150, (1,3), relu) -> MaxPool2D((1,2))
#   -> Conv2D(225, (1,3), relu) -> MaxPool2D((1,2))
#   -> Dropout(0.5) -> Flatten() -> Dense(128, relu) -> Dropout(0.5)
#   -> Dense(class_num, softmax)
# (the shipped inference script `DeepTE.py` only calls `keras.models.load_model`
# on a pretrained .h5 -- the architecture itself lives in this training script).
# Ported layer-for-layer: same channel progression (100 -> 150 -> 225), same
# (1,3) kernels / (1,2) pools (the "1" dimension is a dummy singleton axis
# standing in for the k-mer feature vector reshaped to (1, 16384, 1) NHWC in the
# original), same dropout placement, same 128-unit Dense bottleneck, same final
# softmax classification head.
import torch
from torch import nn

MENAGERIE_ZOO = "ported-pytorch"


class DeepTECNN(nn.Module):
    """Faithful port of the Sequential model built in train_model()."""

    def __init__(self, class_num=4, flatten_dim=1350):
        super().__init__()
        # torch is channels-first (N,C,H,W); the real Keras model used a
        # dummy singleton "height" axis with the k-mer feature axis as
        # "width" (reshape to (1, 16384, 1)) -- here the input is laid out as
        # (N, 1, 1, L) so the (1,3) kernel slides along the k-mer axis exactly
        # as in the original, just with the singleton on H instead of W.
        self.conv1 = nn.Conv2d(1, 100, kernel_size=(1, 3))
        self.pool1 = nn.MaxPool2d((1, 2))
        self.conv2 = nn.Conv2d(100, 150, kernel_size=(1, 3))
        self.pool2 = nn.MaxPool2d((1, 2))
        self.conv3 = nn.Conv2d(150, 225, kernel_size=(1, 3))
        self.pool3 = nn.MaxPool2d((1, 2))
        self.dropout1 = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flatten_dim, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, class_num)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = torch.softmax(self.fc2(x), dim=-1)
        return x


def build_deepte():
    # 64-length k-mer axis, three (1,3) convs + (1,2) pools ->
    # 225 channels * 1 * 6 = 1350 flatten_dim.
    return DeepTECNN(class_num=4, flatten_dim=1350)


def example_input_deepte():
    # Real input is a k-mer count vector reshaped to (N, 1, 16384, 1); shrink
    # the k-mer axis for a tiny trace while keeping the (N, 1, 1, L) layout so
    # the (1,3) kernels / (1,2) pools operate over the same axis as the real
    # model.
    return torch.randn(1, 1, 1, 64)


MENAGERIE_ENTRIES = [
    ("DeepTE", build_deepte, example_input_deepte, 2020, "ported-pytorch"),
]
