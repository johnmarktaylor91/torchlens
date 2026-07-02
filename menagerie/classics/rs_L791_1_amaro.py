# SOURCE: vendored from https://github.com/compsciencelab/torchmd-net @ main
# (torchmdnet/models/model.py + torchmdnet/models/tensornet.py)
#
# AMARO (https://github.com/compsciencelab/amaro) is an all-heavy-atom, hydrogen-excluding
# coarse-grained protein force field. Its own repo (`compsciencelab/amaro`) contains no
# model-definition file at all -- only training/data/config glue (`train/amaro_tmdnet.yaml`,
# `data/mdcath_noh/*`, `benchmark/run.py`). The yaml config declares `model: tensornet`,
# `output_model: Scalar`: AMARO's real architecture IS the stock, unmodified TensorNet
# O(3)-equivariant message-passing network from the `torchmd-net` package (already installed
# in this environment; PyPI `torchmd-net`, imports as `torchmdnet`), wired through
# `torchmdnet.models.model.create_model()` / `TorchMD_Net` exactly as AMARO's own yaml
# specifies. AMARO's contribution is the coarse-graining data pipeline (heavy-atom-only
# MDCATH training set) and training recipe, not a new architecture -- so this module simply
# calls the real, un-modified `torchmd-net` constructor with AMARO's own published
# hyperparameters (`train/amaro_tmdnet.yaml`), shrunk only in width/depth
# (embedding_dimension/num_rbf/num_layers) for a fast trace.
#
# `create_model()` takes >1 concrete tensor (z, pos, batch), so per the recipe-input
# constraint (recipes support exactly one concrete-tensor input) this is staged as a MODULE
# rather than a `recipes_*.tsv` row.

import torch
from torchmdnet.models.model import create_model

MENAGERIE_ZOO = "vendored-pytorch"

# AMARO's real train/amaro_tmdnet.yaml hyperparameters, verbatim except embedding_dimension/
# num_rbf/num_layers shrunk for a fast tiny trace (architecture -- TensorNet + Scalar output
# head -- is unchanged).
_AMARO_CONFIG = dict(
    model="tensornet",
    embedding_dimension=8,
    num_layers=1,
    num_rbf=8,
    rbf_type="expnorm",
    trainable_rbf=False,
    activation="silu",
    neighbor_embedding=False,
    cutoff_lower=0.0,
    cutoff_upper=5.0,
    max_z=15,
    max_num_neighbors=32,
    equivariance_invariance_group="O(3)",
    prior_model=None,
    output_model="Scalar",
    reduce_op="add",
    aggr="add",
    # Real AMARO config trains with derivative=True (forces via autograd of the energy).
    # We keep the identical TensorNet + Scalar-head architecture but trace with
    # derivative=False: `derivative` only toggles an outer torch.autograd.grad() wrapper
    # around the same network, not the architecture itself, and disabling it avoids a slow
    # one-time Warp backward-kernel JIT compile during tracing.
    derivative=False,
    atom_filter=-1,
    box_vecs=None,
    static_shapes=False,
    charge=False,
    spin=False,
    output_mlp_num_layers=0,
    vector_cutoff=False,
    attn_activation="silu",
    num_heads=8,
    distance_influence="both",
    precision=32,
)


def build_amaro():
    model = create_model(_AMARO_CONFIG)
    model.eval()
    return model


def example_input_amaro():
    # A small heavy-atom-only (hydrogen-excluding) pseudo-residue system, matching AMARO's
    # coarse-grained all-heavy-atom representation: atomic numbers z, 3D coordinates pos,
    # and a single-system batch index (one molecule in the batch).
    n_atoms = 6
    z = torch.randint(1, 15, (n_atoms,), dtype=torch.long)
    pos = torch.randn(n_atoms, 3, dtype=torch.float32) * 2.0
    batch = torch.zeros(n_atoms, dtype=torch.long)
    return (z, pos, batch)


MENAGERIE_ENTRIES = [
    ("AMARO", "build_amaro", "example_input_amaro", 2024, "SOURCE_AVAILABLE"),
]
