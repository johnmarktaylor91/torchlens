# FAITHFUL PORT of https://github.com/wnlUc3m/deepcog @ 5c749938aab67bf60011cbf85019a75f895d5df8
# (original framework: Keras/TensorFlow)
#
# Ports the DeepCog mobile-traffic-forecasting architecture, defined in `DeepCog.ipynb` cell
# `make_nn_model`:
#   Sequential(
#     Conv3D(32, kernel=(3,3,3), relu, padding='same'),   # input (lookback, H, W, 1)
#     Conv3D(32, kernel=(6,6,6), relu, padding='same'),
#     Dropout(0.3),
#     Conv3D(16, kernel=(6,6,6), relu, padding='same'),
#     Dropout(0.3),
#     Flatten(),
#     Dense(64, relu),
#     Dense(32, relu),
#     Dense(num_cluster),
#   )
# Ported layer-for-layer to torch.nn (Conv3d expects channel-first NCDHW; the original Keras
# Conv3D is channel-last NDHWC over an input of shape (batch, lookback, H, W, 1) -- the "depth"
# axis of the 3D conv is the lookback/time axis, matching Zhang et al. "DeepCog: Cognitive
# Network Management in Sliced 5G Networks with Deep Learning" (INFOCOM'19).
import torch
from torch import nn


class DeepCog(nn.Module):
    def __init__(self, lookback: int = 6, num_cluster: int = 1, height: int = 8, width: int = 8):
        super().__init__()
        self.lookback = lookback
        self.num_cluster = num_cluster

        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=6, padding=3)
        self.drop1 = nn.Dropout(0.3)
        self.conv3 = nn.Conv3d(32, 16, kernel_size=6, padding=3)
        self.drop2 = nn.Dropout(0.3)
        self.relu = nn.ReLU()

        # kernel=6 with padding=3 grows each spatial dim by 1 per conv (PyTorch 'valid-shifted'
        # padding does not reproduce Keras 'same' exactly for even kernels); compute the
        # flattened feature size deterministically from the conv arithmetic rather than using a
        # lazy module, so the module traces cleanly under TorchLens.
        def _grow(n, k, p):
            return n + 2 * p - k + 1

        d, h, w = lookback, height, width
        d, h, w = _grow(d, 3, 1), _grow(h, 3, 1), _grow(w, 3, 1)
        d, h, w = _grow(d, 6, 3), _grow(h, 6, 3), _grow(w, 6, 3)
        d, h, w = _grow(d, 6, 3), _grow(h, 6, 3), _grow(w, 6, 3)
        flat_size = 16 * d * h * w

        self.fc1 = nn.Linear(flat_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_cluster)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback, H, W, 1) as in the original Keras `input_shape`; convert to
        # PyTorch's channel-first Conv3d layout (batch, channel, depth, H, W).
        x = x.permute(0, 4, 1, 2, 3)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.drop1(x)
        x = self.relu(self.conv3(x))
        x = self.drop2(x)
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# --- TorchLens menagerie staging harness (not part of the original repo) ---


def build_deepcog():
    return DeepCog(lookback=6, num_cluster=1)


def example_input_deepcog():
    torch.manual_seed(0)
    # (batch, lookback, H, W, channel=1) matching the Keras input_shape convention.
    return (torch.rand(2, 6, 8, 8, 1),)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("DeepCog", build_deepcog, example_input_deepcog, 2019, "ported-pytorch"),
]
