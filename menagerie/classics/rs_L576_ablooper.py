# SOURCE: vendored from oxpig/ABlooper @ 2dcf208c76bb3ba6553debe8761a4202e1b02953
#   ABlooper/models.py
#
# ABlooper (Abanades, Georges, Bonet, Deane. Bioinformatics 2022) predicts antibody CDR
# loop backbone coordinates via an E(n)-equivariant graph neural network (EGNN, following
# Satorras et al. 2021 / the egnn-pytorch design). `EGNN` updates per-residue features and
# 3D coordinates jointly through an edge MLP (pairwise feature+distance -> message),
# a coordinate-update MLP (message -> scalar weight on the relative-coordinate vector),
# and a node MLP (feature+aggregated-message -> residual feature update). `ResEGNN` stacks
# several EGNN correction layers; `DecoyGen` runs several independent `ResEGNN` blocks
# ("decoys") over the same input to produce an ensemble of loop-coordinate predictions,
# which is exactly the model class ABlooper.py loads at import time
# (`DecoyGen().float().to(device)` / `DecoyGen(coors_norm=False)`) before restoring
# trained weights. Vendored verbatim; only the module-level docstring/comment above the
# class definitions and the trailing `default_model_*`/weight-loading lines (not part of
# the architecture) are dropped.
#
# MENAGERIE_ZOO = "vendored-pytorch"

import torch
from einops import rearrange


# Most of the code in this file is based on egnn-pytorch by lucidrains.


class Swish_(torch.nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


SiLU = torch.nn.SiLU if hasattr(torch.nn, "SiLU") else Swish_


class CoorsNorm(torch.nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.fn = torch.nn.LayerNorm(1)

    def forward(self, coors):
        norm = coors.norm(dim=-1, keepdim=True)
        normed_coors = coors / norm.clamp(min=self.eps)
        phase = self.fn(norm)
        return phase * normed_coors


# classes


class EGNN(torch.nn.Module):
    def __init__(self, dim, m_dim=32, coors_norm=True):
        super().__init__()

        self.norm = coors_norm
        edge_input_dim = (dim * 2) + 1

        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(edge_input_dim, edge_input_dim * 2),
            SiLU(),
            torch.nn.Linear(edge_input_dim * 2, m_dim),
            SiLU(),
        )

        if self.norm:
            self.coors_norm = CoorsNorm()

        self.node_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim + m_dim, dim * 2),
            SiLU(),
            torch.nn.Linear(dim * 2, dim),
        )

        self.coors_mlp = torch.nn.Sequential(
            torch.nn.Linear(m_dim, m_dim * 4), SiLU(), torch.nn.Linear(m_dim * 4, 1)
        )

    def forward(self, feats, coors):
        rel_coors = rearrange(coors, "b i d -> b i () d") - rearrange(coors, "b j d -> b () j d")
        rel_dist = (rel_coors**2).sum(dim=-1, keepdim=True)

        feats_j = rearrange(feats, "b j d -> b () j d")
        feats_i = rearrange(feats, "b i d -> b i () d")
        feats_i, feats_j = torch.broadcast_tensors(feats_i, feats_j)

        edge_input = torch.cat((feats_i, feats_j, rel_dist), dim=-1)

        m_ij = self.edge_mlp(edge_input)

        coor_weights = self.coors_mlp(m_ij)
        coor_weights = rearrange(coor_weights, "b i j () -> b i j")

        if self.norm:
            rel_coors = self.coors_norm(rel_coors)
            coors_out = torch.einsum("b i j, b i j c -> b i c", coor_weights, rel_coors) + coors
            m_i = m_ij.sum(dim=-2)
        else:
            rel_coors = rel_coors / rel_dist.clip(min=1e-8)
            coors_out = torch.einsum("b i j, b i j c -> b i c", coor_weights, rel_coors) + coors
            m_i = m_ij.mean(dim=-2)

        node_mlp_input = torch.cat((feats, m_i), dim=-1)
        node_out = self.node_mlp(node_mlp_input) + feats

        return node_out, coors_out


class ResEGNN(torch.nn.Module):
    def __init__(self, corrections=4, dims_in=41, coors_norm=True, **kwargs):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [EGNN(dim=dims_in, coors_norm=coors_norm, **kwargs) for _ in range(corrections)]
        )

    def forward(self, amino, geom):
        for layer in self.layers:
            amino, geom = layer(amino, geom)
        return geom


class DecoyGen(torch.nn.Module):
    def __init__(self, dims_in=41, decoys=5, coors_norm=True, **kwargs):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [ResEGNN(dims_in=dims_in, coors_norm=coors_norm, **kwargs) for _ in range(decoys)]
        )
        self.decoys = decoys

    def forward(self, amino, geom):
        geoms = torch.zeros((self.decoys, *geom.shape[1:]), device=geom.device)

        for i, block in enumerate(self.blocks):
            geoms[i] = block(amino, geom)

        return geoms


def build_ablooper():
    # Real constructor defaults from ABlooper.py: DecoyGen().float() (chothia model) /
    # DecoyGen(coors_norm=False) (imgt model). dims_in=41 matches the real per-residue
    # feature width used by prepare_input_loop (20 amino-acid one-hot + 6 CDR one-hot +
    # positional/anchor encodings); shrunk `corrections`/`decoys`/`m_dim` for tiny tracing.
    return DecoyGen(dims_in=41, decoys=2, coors_norm=True, corrections=2, m_dim=8)


def example_input_ablooper():
    # Real usage (ABlooper.py CDR_Predictor.predict_CDRs): amino is (batch, n_res, dims_in)
    # one-hot+context node features, geom is (batch, n_res, 3) backbone coordinates for one
    # atom-type slice (the model is called once per atom type in the real pipeline).
    batch, n_res, dims_in = 1, 12, 41
    amino = torch.rand(batch, n_res, dims_in)
    geom = torch.rand(batch, n_res, 3)
    return (amino, geom)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("ABlooper", "build_ablooper", "example_input_ablooper", 2022, MENAGERIE_ZOO),
]
