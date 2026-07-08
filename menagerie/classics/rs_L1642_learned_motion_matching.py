# SOURCE: vendored from orangeduck/Motion-Matching @ main
# (resources/train_decompressor.py, resources/train_stepper.py,
# resources/train_projector.py -- network classes only)
# https://github.com/orangeduck/Motion-Matching -- companion repo for "Learned Motion
# Matching" (Holden, Kanoun, Perepichka, Popovic; SIGGRAPH 2020). The paper replaces
# classic motion-matching's giant pose database + nearest-neighbor search with three
# small MLPs distilled from it: a `Compressor`/`Decompressor` autoencoder pair that
# learns a compact latent pose code from the motion-capture feature database, a
# `Projector` that maps a (possibly out-of-distribution) query feature vector onto the
# nearest on-manifold latent code (replacing the nearest-neighbor database search), and
# a `Stepper` that advances the latent code forward in time (replacing frame-by-frame
# playback from the database). The four `nn.Module` class bodies below are transcribed
# verbatim from the three official training scripts; the surrounding
# `if __name__ == '__main__':` training procedures (which pull in the repo's
# quaternion/BVH/database-loading helpers `tquat`, `txform`, `quat`, `bvh`,
# `train_common`, plus `sklearn.neighbors.BallTree` and matplotlib/tensorboard) are not
# part of the network architecture and are dropped; only base-torch imports remain.
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---- verbatim from resources/train_decompressor.py ----
class Compressor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512):
        super(Compressor, self).__init__()

        self.linear0 = nn.Linear(input_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        nbatch, nwindow = x.shape[:2]
        x = x.reshape([nbatch * nwindow, -1])
        x = F.elu(self.linear0(x))
        x = F.elu(self.linear1(x))
        x = F.elu(self.linear2(x))
        x = self.linear3(x)
        return x.reshape([nbatch, nwindow, -1])


class Decompressor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512):
        super(Decompressor, self).__init__()

        self.linear0 = nn.Linear(input_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        nbatch, nwindow = x.shape[:2]
        x = x.reshape([nbatch * nwindow, -1])
        x = F.relu(self.linear0(x))
        x = self.linear1(x)
        return x.reshape([nbatch, nwindow, -1])


# ---- verbatim from resources/train_stepper.py ----
class Stepper(nn.Module):
    def __init__(self, input_size, hidden_size=512):
        super(Stepper, self).__init__()

        self.linear0 = nn.Linear(input_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = F.relu(self.linear0(x))
        x = F.relu(self.linear1(x))
        return self.linear2(x)


# ---- verbatim from resources/train_projector.py ----
class Projector(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512):
        super(Projector, self).__init__()

        self.linear0 = nn.Linear(input_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear0(x))
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return self.linear4(x)


# ---- staging build/example helpers (tiny sizes for fast tracing) ----
def build_decompressor():
    torch.manual_seed(0)
    # output_size = nbones*(3 pos + 6 rot6d + 3 vel + 3 ang) + nextra(contacts), input = latent + features
    return Decompressor(input_size=32 + 16, output_size=8 * 15 + 2, hidden_size=64)


def example_input_decompressor():
    torch.manual_seed(0)
    nbatch, nwindow = 2, 2
    return (torch.randn(nbatch, nwindow, 32 + 16),)


def build_stepper():
    torch.manual_seed(0)
    return Stepper(input_size=32 + 16, hidden_size=64)


def example_input_stepper():
    torch.manual_seed(0)
    return (torch.randn(4, 32 + 16),)


def build_projector():
    torch.manual_seed(0)
    return Projector(input_size=16, output_size=16 + 32 + 1, hidden_size=64)


def example_input_projector():
    torch.manual_seed(0)
    return (torch.randn(4, 16),)


MENAGERIE_ENTRIES = [
    (
        "Learned Motion Matching Decompressor",
        build_decompressor,
        example_input_decompressor,
        2020,
        "vendored-pytorch",
    ),
    (
        "Learned Motion Matching Stepper",
        build_stepper,
        example_input_stepper,
        2020,
        "vendored-pytorch",
    ),
    (
        "Learned Motion Matching Projector",
        build_projector,
        example_input_projector,
        2020,
        "vendored-pytorch",
    ),
]
