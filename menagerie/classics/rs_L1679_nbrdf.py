# SOURCE: vendored from asztr/Neural-BRDF @ main (binary_to_nbrdf/pytorch_code/train_NBRDF_pytorch.py)
# https://github.com/asztr/Neural-BRDF -- "Neural BRDF Representation and Importance
# Sampling" (Sztrajman, Rainer, Ritschel, Weyrich; Computer Graphics Forum 2021). The
# real architecture is a compact 3-layer MLP (`MLP` class below) that maps a 6D
# half/difference-vector encoding of the incoming/outgoing light directions to an RGB
# BRDF value, with a custom final activation `relu(exp(x) - 1)` (mirroring the
# `nn.h`-based Mitsuba/TF reference implementation the repo's docstring references) and
# a specific uniform weight init transposed to match that reference convention. This is
# transcribed verbatim from the repo's real training script; only the data-loading /
# training-loop code (MERL binary BRDF dataset, matplotlib plotting, weight export) is
# dropped since it is not part of the model's forward graph.
import torch
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


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


# ---- staging build/example helpers ----
def build_nbrdf_mlp():
    torch.manual_seed(0)
    return MLP()


def example_input_nbrdf_mlp():
    torch.manual_seed(0)
    # Real repo's Xvars = ['hx','hy','hz','dx','dy','dz'] (Rusinkiewicz half/diff vector
    # coords), batch_size=512 in the reference training script; kept smaller for tracing.
    return (torch.randn(8, 6),)


MENAGERIE_ENTRIES = [
    ("NeuralBRDF-NBRDF-MLP", build_nbrdf_mlp, example_input_nbrdf_mlp, 2021, "vendored-pytorch"),
]
