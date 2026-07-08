# SOURCE: vendored from tohenkes/So3krates-torch @ main (a real PyTorch reimplementation
# of SO3krates / SO3LR, the models originally published in JAX at thorben-frank/mlff).
# https://github.com/tohenkes/So3krates-torch
# The candidate's own repo (thorben-frank/mlff) is JAX-only; this staging module uses
# `tohenkes/So3krates-torch`'s independent, real, tested PyTorch port instead (the
# So3krates author's own README links to it as the PyTorch port; it ships pretrained
# SO3LR weights converted from the original JAX checkpoints). Full model code lives in
# the sibling `_so3krates_vendor/` package (see its files for per-file SOURCE headers);
# this file only wires up build_/example_input_ entry points.
#
# We instantiate `SO3LR` (the long-range electrostatics + dispersion + ZBL-repulsion
# subclass of the base `So3krates` Euclidean-transformer architecture) rather than the
# bare `So3krates` base class: the base class's own `forward()` has a real upstream bug
# (`_create_output_dict(energy=..., ...)` where the method signature names that kwarg
# `total_energy`, https://github.com/tohenkes/So3krates-torch/blob/main/src/so3krates_torch/modules/models.py#L431-L441)
# that has never been exercised/fixed since the repo's first commit -- `SO3LR.forward()`
# calls the equivalent method with the correct keyword and is the actually-tested,
# pretrained-weights-shipping path (`src/so3krates_torch/tests/test_model.py`). `SO3LR`
# still runs the complete SO3krates Euclidean-transformer stack (InvariantEmbedding,
# EuclideanEmbedding, stacked EuclideanTransformer layers with SO(3)-invariant
# L0Contraction attention, AtomicEnergyOutputHead) plus its long-range additions, so the
# core architecture under test is unchanged.
#
# The model internally calls `torch.autograd.grad(..., allow_unused=True)` to derive
# forces from the energy prediction; torchlens' backward-capture hook does not yet
# handle an `allow_unused=True` grad tuple containing `None` entries
# (`torchlens/backends/torch/tensor_tracking.py::_emit_tensor_grad_event`). We therefore
# trace with `compute_force=False` (energy-only inference), a real, commonly used mode
# of this model (e.g. single-point energy evaluation), and wrap only to select the
# `"energy"` entry from the model's output dict for a clean single-tensor trace.
import torch

MENAGERIE_ZOO = "vendored-pytorch"


class _SO3LRWrapper(torch.nn.Module):
    """Thin wrapper selecting the energy output for a single-tensor trace; SO3LR's
    forward() itself is untouched (see _so3krates_vendor/models.py)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, data):
        return self.model(data, compute_force=False)["energy"]


def build_so3krates():
    import sys
    from pathlib import Path

    vendor_root = str(Path(__file__).parent)
    if vendor_root not in sys.path:
        sys.path.insert(0, vendor_root)
    from _so3krates_vendor.models import SO3LR

    torch.manual_seed(0)
    model = SO3LR(
        r_max=5.0,
        r_max_lr=8.0,
        num_radial_basis_fn=8,
        degrees=[0, 1, 2],
        num_features=12,
        num_heads=2,
        num_layers=1,
        num_elements=10,
        zbl_repulsion_bool=True,
        electrostatic_energy_bool=True,
        dispersion_energy_bool=True,
        dispersion_energy_cutoff_lr_damping=2.0,
    )
    model.eval()
    return _SO3LRWrapper(model)


def example_input_so3krates():
    torch.manual_seed(0)
    num_elements = 10
    n_atoms = 6

    positions = torch.randn(n_atoms, 3)
    species = torch.randint(1, num_elements, (n_atoms,))
    node_attrs = torch.nn.functional.one_hot(species, num_classes=num_elements).float()
    batch = torch.zeros(n_atoms, dtype=torch.long)
    ptr = torch.tensor([0, n_atoms], dtype=torch.long)
    cell = torch.zeros(1, 3, 3)

    edges = [(i, j) for i in range(n_atoms) for j in range(n_atoms) if i != j]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    n_edges = edge_index.shape[1]
    shifts = torch.zeros(n_edges, 3)
    shifts_lr = torch.zeros(n_edges, 3)
    head = torch.zeros(1, dtype=torch.long)
    total_charge = torch.zeros(1)

    data = {
        "node_attrs": node_attrs,
        "positions": positions,
        "cell": cell,
        "batch": batch,
        "ptr": ptr,
        "edge_index": edge_index,
        "edge_index_lr": edge_index,
        "shifts": shifts,
        "shifts_lr": shifts_lr,
        "head": head,
        "atomic_numbers": species,
        "total_charge": total_charge,
    }
    return (data,)


MENAGERIE_ENTRIES = [
    (
        "SO3krates (SO3LR)",
        build_so3krates,
        example_input_so3krates,
        2024,
        "MODULE",
    ),
]
