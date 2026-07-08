# SOURCE: vendored from https://github.com/asztr/Neural-BRDF @ main
# (binary_to_nbrdf/pytorch_code/train_NBRDF_pytorch.py, MLP class)
#
# "Neural BRDF Representation and Importance Sampling" (Sztrajman et al.,
# CGF 2021). The paper's reference training/export code is TF/Keras
# (binary_to_nbrdf/binary_to_nbrdf.py), but the repo also ships an official
# PyTorch reimplementation of the exact same architecture used to produce
# npy weights consumed by the Mitsuba BSDF plugin (pytorch_code/
# train_NBRDF_pytorch.py). Architecture is a tiny compact MLP over the
# half/difference-vector (Rusinkiewicz) parameterization: Linear(6->21) ->
# ReLU -> Linear(21->21) -> ReLU -> Linear(21->3) -> exp -> ReLU(x-1),
# matching the "additional relu is max() op as in code in nn.h" comment
# from the original author reproducing the C++ Mitsuba BSDF exactly.
# Vendored verbatim (dataset/training-loop code dropped; only the nn.Module
# is kept).
import torch
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.fc1 = torch.nn.Linear(in_features=6, out_features=21, bias=True)
        self.fc2 = torch.nn.Linear(in_features=21, out_features=21, bias=True)
        self.fc3 = torch.nn.Linear(in_features=21, out_features=3, bias=True)

        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.zeros_(self.fc2.bias)
        torch.nn.init.zeros_(self.fc3.bias)

        self.fc1.weight = torch.nn.Parameter(
            torch.zeros((6, 21), dtype=torch.float32).uniform_(-0.05, 0.05).T, requires_grad=True
        )
        self.fc2.weight = torch.nn.Parameter(
            torch.zeros((21, 21), dtype=torch.float32).uniform_(-0.05, 0.05).T, requires_grad=True
        )
        self.fc3.weight = torch.nn.Parameter(
            torch.zeros((21, 3), dtype=torch.float32).uniform_(-0.05, 0.05).T, requires_grad=True
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(torch.exp(self.fc3(x)) - 1.0)  # additional relu is max() op as in code in nn.h
        return x


# --- staging harness: build + example input ---------------------------------


def build_neural_brdf():
    return MLP()


def example_input_neural_brdf():
    # 6-dim half/difference-vector (hx, hy, hz, dx, dy, dz) samples, matching
    # Xvars in the original training script.
    batch_size = 8
    return torch.rand(batch_size, 6)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("Neural BRDF", build_neural_brdf, example_input_neural_brdf, 2021, MENAGERIE_ZOO),
]
