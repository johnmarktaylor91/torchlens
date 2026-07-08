# FAITHFUL PORT of pmoris/ImRex @ d3975b744c030c4bc67c942c82a32b1c83857c4e (original framework: TensorFlow/Keras)
# https://raw.githubusercontent.com/pmoris/ImRex/master/src/models/model_padded.py
#
# Moris et al. 2021 (Briefings in Bioinformatics) "ImRex: interaction map recognition".
# The queue notes describe this as a "PyTorch" repo but the real `src/models/` code is
# `tensorflow.keras` (Sequential API); TorchLens only captures eager PyTorch, so this is
# a faithful transcription rather than a vendor. `ModelPadded._build_model()` (the
# flagship "padded" variant referenced by the paper's headline CNN-over-interaction-map
# architecture) is a plain Conv2D stack over a fixed-size CDR3-vs-epitope interaction
# map: two Conv2D+BatchNorm blocks (depth1_1 -> depth1_2, then depth2_1 -> depth2_2),
# each followed by MaxPool2D(2,2) (+ optional SpatialDropout2D) and BatchNorm2D, then
# either GlobalAveragePooling2D (`gap=True`) or Flatten -> Dense(32) -> Dropout
# (`gap=False`, the class default used here), and a final Dense(1, sigmoid) head.
# Every Conv2D/BatchNorm/MaxPool/Dense/Dropout layer of the real `_build_model()` is
# reproduced 1:1 in `nn.Conv2d`/`nn.BatchNorm2d`/`nn.MaxPool2d`/`nn.Linear`/`nn.Dropout2d`
# with `padding="same"` semantics preserved via explicit `padding=1` for the real
# `kernel_size=(3, 3)` convs (odd kernel, stride 1 -> same as TF "same" padding), He-normal
# initialization matched via `nn.init.kaiming_normal_`, and Keras' NHWC layout mapped to
# torch's NCHW (the interaction-map "channels" become the input's channel dim). Training
# loss/optimizer plumbing (`get_loss`/`get_optimizer`) is not part of the architecture and
# is omitted.

import torch
import torch.nn as nn


class ModelPadded(nn.Module):
    def __init__(
        self,
        width: int,
        height: int,
        channels: int,
        depth1_1: int = 128,
        depth1_2: int = 64,
        depth2_1: int = 128,
        depth2_2: int = 64,
        gap: bool = False,
        dropout_conv: float = None,
        dropout_dense: float = None,
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.channels = channels
        self.gap = gap
        self.dropout_conv = dropout_conv
        self.dropout_dense = dropout_dense

        # Module 1: Conv2D(depth1_1) -> BN -> Conv2D(depth1_2) -> MaxPool -> [Dropout] -> BN
        self.conv1_1 = nn.Conv2d(channels, depth1_1, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(depth1_1)
        self.conv1_2 = nn.Conv2d(depth1_1, depth1_2, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.drop1 = nn.Dropout2d(dropout_conv) if dropout_conv else nn.Identity()
        self.bn1_2 = nn.BatchNorm2d(depth1_2)

        # Module 2: Conv2D(depth2_1) -> BN -> Conv2D(depth2_2) -> MaxPool -> [Dropout] -> BN
        self.conv2_1 = nn.Conv2d(depth1_2, depth2_1, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(depth2_1)
        self.conv2_2 = nn.Conv2d(depth2_1, depth2_2, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.drop2 = nn.Dropout2d(dropout_conv) if dropout_conv else nn.Identity()
        self.bn2_2 = nn.BatchNorm2d(depth2_2)

        self.relu = nn.ReLU()

        if self.gap:
            self.gap_pool = nn.AdaptiveAvgPool2d(1)
            head_in = depth2_2
        else:
            # Flattened size depends on width/height after two 2x2 max-pools;
            # computed lazily via nn.LazyLinear to stay faithful to the real
            # Sequential model's shape-inference without hardcoding H'*W'*C.
            self.dense = nn.LazyLinear(32)
            self.drop_dense = nn.Dropout(dropout_dense) if dropout_dense else nn.Identity()
            head_in = 32

        self.head = nn.Linear(head_in, 1)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            # nn.LazyLinear (the flatten -> Dense(32) head when gap=False) has an
            # uninitialized weight until its first forward call; skip it here and
            # let the staging harness re-init after the LazyLinear warm-up pass.
            if isinstance(m, nn.LazyLinear) or (
                isinstance(m, (nn.Conv2d, nn.Linear))
                and isinstance(m.weight, nn.UninitializedParameter)
            ):
                continue
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1_1(x))
        x = self.bn1_1(x)
        x = self.relu(self.conv1_2(x))
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.bn1_2(x)

        x = self.relu(self.conv2_1(x))
        x = self.bn2_1(x)
        x = self.relu(self.conv2_2(x))
        x = self.pool2(x)
        x = self.drop2(x)
        x = self.bn2_2(x)

        if self.gap:
            x = self.gap_pool(x)
            x = torch.flatten(x, 1)
        else:
            x = torch.flatten(x, 1)
            x = self.relu(self.dense(x))
            x = self.drop_dense(x)

        x = self.head(x)
        return self.sigmoid(x)


# --- staging harness (tiny sizes; not part of the real repo) ---


def build_imrex():
    # width=20 (max_length_cdr3 default), height=11 (max_length_epitope default),
    # channels=5 (5 physicochemical property maps: hydrophob, isoelectric, mass,
    # hydrophil, charge -- the real scenario_padding.py CLI default feature set),
    # depth1_1/1_2/2_1/2_2 and gap=False mirror ModelPadded's real __init__ defaults.
    model = ModelPadded(width=20, height=11, channels=5)
    # Warm up the LazyLinear head with one forward pass so its in_features are
    # materialized before tracing, then eval() so BatchNorm2d uses running stats
    # (the trace below uses batch size 1, which training-mode batch stats can't
    # support for BatchNorm).
    model.eval()
    with torch.no_grad():
        model(example_input_imrex()[0])
    if not model.gap:
        nn.init.kaiming_normal_(model.dense.weight)
        nn.init.zeros_(model.dense.bias)
    return model


def example_input_imrex():
    # NCHW interaction map: batch=1, channels=5 (physicochemical properties),
    # height=11 (epitope length), width=20 (CDR3 length) -- Keras' NHWC
    # (width, height, channels) input_shape maps to torch's (channels, height, width).
    return (torch.rand(1, 5, 11, 20),)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("ImRex", "build_imrex", "example_input_imrex", 2021, "ported-pytorch"),
]
