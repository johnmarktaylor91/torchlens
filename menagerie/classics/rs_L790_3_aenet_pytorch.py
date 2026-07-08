# SOURCE: vendored from atomisticnet/aenet-PyTorch @ main (src/network.py)
# https://github.com/atomisticnet/aenet-PyTorch
"""aenet-PyTorch: GPU-supported PyTorch implementation of the aenet
Behler-Parrinello-style atomic neural network (ANN) potential, using
Chebyshev / Behler-Parrinello symmetry-function descriptors as per-element
ANN inputs.

``NetAtom`` (one independently-parameterized feed-forward ANN
``Linear -> activation -> ... -> Linear(-> 1)`` per chemical species, wired
together via ``nn.ModuleList``) is transcribed unmodified from
``src/network.py``. Only the ``forward_F`` force-training path (which needs
``torch.autograd.grad`` inside the traced forward and is a *training*-time
concern, not part of the model's inference architecture) and the loss
helpers (``get_loss_RMSE*``, pure training utilities) are dropped; the
energy-prediction ``forward`` method used for inference is kept unmodified.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


class NetAtom(nn.Module):
    """ANN for each atomic element present in the training set.

    input_size  :: Dimension of the descriptor vectors
    hidden_size :: Number of nodes in the hidden layers
    activations :: Activation functions of each layer
    functions   :: List of functions that are applied in the ANN. A series of
                    Linear + Activation + Linear + Activation + ... + Linear
    """

    def __init__(self, input_size, hidden_size, species, activations, alpha, device):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.species = species
        self.active_names = activations
        self.alpha = torch.tensor(alpha)
        self.device = device

        N_fun = [len(hidden_size[i]) + 1 for i in range(len(species))]

        self.linear = nn.Identity()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.activations = []
        for i in range(len(species)):
            aux = []
            for j in range(len(hidden_size[i])):
                if activations[i][j] == "linear":
                    aux.append(self.linear)
                if activations[i][j] == "tanh":
                    aux.append(self.tanh)
                if activations[i][j] == "sigmoid":
                    aux.append(self.sigmoid)
            self.activations.append(aux)

        self.functions = []
        for i in range(len(species)):
            function_i = OrderedDict()
            name1 = "Linear_Sp" + str(i + 1) + "_F" + str(1)
            name2 = "Active_Sp" + str(i + 1) + "_F" + str(1)

            function_i[name1] = nn.Linear(input_size[i], hidden_size[i][0])
            function_i[name2] = self.activations[i][0]
            for j in range(1, N_fun[i] - 1):
                name1 = "Linear_Sp" + str(i + 1) + "_F" + str(j + 1)
                name2 = "Active_Sp" + str(i + 1) + "_F" + str(j + 1)
                function_i[name1] = nn.Linear(hidden_size[i][j - 1], hidden_size[i][j])
                function_i[name2] = self.activations[i][j]
            name1 = "Linear_Sp" + str(i + 1) + "_F" + str(N_fun[i])
            function_i[name1] = nn.Linear(hidden_size[i][-1], 1)

            self.functions.append(nn.Sequential(function_i))
        self.functions = nn.ModuleList(self.functions)

    def forward(self, grp_descrp, logic_reduce):
        """
        [Energy training] Compute atomic energy for each atom in the current batch.
        INPUT:
            grp_descrp    :: Descriptors of the atoms of the batch, ordered by
                              element, without considering to which structure
                              belongs each
            logic_reduce  :: Auxiliar tensor to reorder the atomic contributions
                              back to each structure
        OUTPUT:
            list_E_ann    :: total ANN energies of each structure in the batch
        """
        partial_E_ann = [0 for i in range(len(self.species))]
        for iesp in range(len(self.species)):
            partial_E_ann[iesp] = self.functions[iesp](grp_descrp[iesp])

        list_E_ann = torch.zeros((len(logic_reduce[0])), device=self.device).double()
        for iesp in range(len(self.species)):
            list_E_ann = list_E_ann + torch.einsum(
                "ij,ki->k", partial_E_ann[iesp], logic_reduce[iesp]
            )

        return list_E_ann


def build_aenet_pytorch() -> NetAtom:
    """Tiny random-init NetAtom for a 2-species system (e.g. a binary alloy),
    with a small Behler-Parrinello descriptor width and one hidden layer.
    The real ``src/aenet_pytorch.py`` entry point builds the model with
    ``NetAtom(...).double()`` (energies are accumulated in float64), so the
    same cast is applied here."""
    torch.manual_seed(0)
    species = ["A", "B"]
    input_size = [8, 8]
    hidden_size = [[6], [6]]
    activations = [["tanh"], ["tanh"]]
    alpha = 0.0
    device = "cpu"
    return (
        NetAtom(
            input_size=input_size,
            hidden_size=hidden_size,
            species=species,
            activations=activations,
            alpha=alpha,
            device=device,
        )
        .double()
        .eval()
    )


def example_input_aenet_pytorch() -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """A tiny synthetic 2-structure, 2-species batch: 3 atoms of species A and
    2 atoms of species B spread across the 2 structures (matching the real
    aenet-PyTorch batch layout: descriptors grouped by species in
    ``grp_descrp``, with ``logic_reduce[iesp]`` a (n_structures, n_atoms_of_species)
    0/1 matrix mapping each species-grouped atomic energy back to its
    structure for the per-structure energy sum)."""
    torch.manual_seed(0)
    n_struct = 2
    # Species A: 3 atoms total, 2 in structure 0, 1 in structure 1.
    # Species B: 2 atoms total, 1 in structure 0, 1 in structure 1.
    grp_descrp = [
        torch.randn(3, 8, dtype=torch.float64),
        torch.randn(2, 8, dtype=torch.float64),
    ]
    logic_reduce = [
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
    ]
    assert logic_reduce[0].shape[0] == n_struct
    return (grp_descrp, logic_reduce)


MENAGERIE_ENTRIES = [
    ("aenet-PyTorch", "build_aenet_pytorch", "example_input_aenet_pytorch", 2021, MENAGERIE_ZOO),
]
