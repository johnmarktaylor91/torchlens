# SOURCE: vendored from https://github.com/semodi/neuralxc @ master
# https://github.com/semodi/neuralxc/blob/master/neuralxc/ml/network.py
#
# NeuralXC's `EnergyNetwork`: a Behler-Parinello style neural-network exchange-correlation
# (XC) functional for DFT. Per-chemical-species MLPs (`species_nets`, an `nn.ModuleDict`)
# consume per-atom local electron-density descriptors and predict a per-atom energy
# contribution, summed across atoms to give the total XC energy correction. Copied
# verbatim from the real repo (only this staging-entrypoint section added at the bottom
# plus a `_build_species_nets` helper that mirrors the exact layer-construction logic the
# real `EnergyNetwork.train()` runs on first use -- upstream only builds `species_nets`
# lazily inside `.train()`, so this helper is the same construction code lifted out to
# run without an actual training call).
#
# MENAGERIE_ZOO = "vendored-pytorch"

from __future__ import annotations

import numpy as np
import torch

MENAGERIE_ZOO = "vendored-pytorch"


# ------------------------------------------------------------------
# neuralxc/ml/network.py  (verbatim; NetworkEstimator / train_net / Dataset omitted --
# they are sklearn-estimator + training-loop plumbing around EnergyNetwork, not part of
# the traceable architecture)
# ------------------------------------------------------------------
class EnergyNetwork(torch.nn.Module):
    def __init__(self, n_nodes, n_layers, activation):
        super(EnergyNetwork, self).__init__()
        self.n_nodes = n_nodes
        self.n_layers = n_layers
        if hasattr(torch.nn, activation):
            self.activation = getattr(torch.nn, activation)()
        else:
            print("Activation unknown, defaulting to GELU")
            self.activation = torch.nn.GELU()

    def forward(self, input):
        output = 0
        for spec in input:
            output += torch.sum(self.species_nets[spec](input[spec]), dim=-2)

        return output


Energy_Network = EnergyNetwork  # Needed to unpickle old models


# ------------------------------------------------------------------
# Menagerie staging entrypoints
# ------------------------------------------------------------------
def _build_species_nets(net, feature_dims):
    """Mirrors the layer-construction block inside the real EnergyNetwork.train()
    (neuralxc/ml/network.py lines ~204-215), lifted out so the module can be built and
    traced without going through an actual sklearn-style training call. `feature_dims`
    maps species symbol -> local-descriptor dimension, exactly as the real code reads
    `X[spec].shape[-1]` off the training-data dict.
    """
    species_nets = {}
    for spec, dim in feature_dims.items():
        if net.n_layers < 1:
            species_nets[spec] = torch.nn.Linear(dim, 1)
        else:
            species_nets[spec] = torch.nn.Sequential(
                *(
                    [torch.nn.Linear(dim, net.n_nodes)]
                    + (net.n_layers - 1)
                    * [net.activation, torch.nn.Linear(net.n_nodes, net.n_nodes)]
                    + [net.activation, torch.nn.Linear(net.n_nodes, 1)]
                )
            )
    net.species_nets = torch.nn.ModuleDict(species_nets)
    return net


def build_neuralxc():
    torch.manual_seed(0)
    net = EnergyNetwork(n_nodes=16, n_layers=2, activation="GELU")
    # Two species (O, H), matching the real repo's water-system examples; descriptor dim
    # 12 is a small stand-in for the real projected-density feature length.
    _build_species_nets(net, {"O": 12, "H": 12})
    return net


def example_input_neuralxc():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    # forward(input) expects a dict: species symbol -> tensor[n_atoms_of_species, descriptor_dim]
    return (
        {
            "O": torch.rand(2, 12, generator=g),
            "H": torch.rand(4, 12, generator=g),
        },
    )


MENAGERIE_ENTRIES = [
    ("NeuralXC", build_neuralxc, example_input_neuralxc, 2020, MENAGERIE_ZOO),
]
