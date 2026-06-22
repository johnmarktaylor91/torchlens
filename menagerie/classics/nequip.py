"""NequIP: E(3)-equivariant interatomic potential graph network.

Batzner et al. 2022, Nature Communications, "E(3)-equivariant graph neural
networks for data-efficient and accurate interatomic potentials".

This compact random-init classic preserves NequIP's load-bearing mechanism:
atom-type scalar embeddings, edge features from radial Bessel bases with a
smooth polynomial cutoff and spherical harmonics of edge directions,
radial-MLP-weighted equivariant tensor-product convolutions, repeated
interaction blocks, and per-atom scalar energy readout.
"""

from __future__ import annotations

import math
from typing import Final

import torch
from torch import Tensor, nn

try:
    from e3nn import nn as e3nn_nn
    from e3nn import o3

    _HAS_E3NN: Final = True
except ImportError:
    e3nn_nn = None
    o3 = None
    _HAS_E3NN = False


class BesselRadialBasis(nn.Module):
    """Bessel radial basis with a polynomial cutoff envelope."""

    def __init__(self, num_basis: int = 6, cutoff: float = 4.0) -> None:
        """Initialize radial frequencies.

        Parameters
        ----------
        num_basis:
            Number of sinusoidal Bessel basis functions.
        cutoff:
            Distance cutoff radius.
        """
        super().__init__()
        self.cutoff = cutoff
        frequencies = torch.arange(1, num_basis + 1, dtype=torch.float32) * math.pi
        self.register_buffer("frequencies", frequencies)

    def forward(self, distances: Tensor) -> Tensor:
        """Evaluate cutoff-weighted radial Bessel features.

        Parameters
        ----------
        distances:
            Edge distances with shape ``(num_edges, 1)``.

        Returns
        -------
        Tensor
            Radial features with shape ``(num_edges, num_basis)``.
        """
        scaled = (distances / self.cutoff).clamp_min(0.0)
        envelope = _polynomial_cutoff(scaled)
        bessel = torch.sin(self.frequencies.view(1, -1) * scaled) / distances.clamp_min(1e-6)
        return bessel * envelope


def _polynomial_cutoff(scaled_distances: Tensor) -> Tensor:
    """Compute a smooth quintic cutoff envelope.

    Parameters
    ----------
    scaled_distances:
        Distances divided by the cutoff radius.

    Returns
    -------
    Tensor
        Envelope values equal to zero outside the cutoff.
    """
    x = scaled_distances.clamp(0.0, 1.0)
    envelope = 1.0 - 10.0 * x.pow(3) + 15.0 * x.pow(4) - 6.0 * x.pow(5)
    return torch.where(scaled_distances < 1.0, envelope, torch.zeros_like(envelope))


if _HAS_E3NN:

    class E3NNTensorProductConvolution(nn.Module):
        """NequIP-style radial-weighted equivariant tensor-product convolution."""

        def __init__(
            self,
            irreps_node: "o3.Irreps",
            irreps_edge: "o3.Irreps",
            num_radial: int,
            radial_hidden: int = 32,
        ) -> None:
            """Initialize spherical harmonics, tensor product, and radial MLP.

            Parameters
            ----------
            irreps_node:
                Node feature irreducible representations.
            irreps_edge:
                Spherical-harmonic edge irreducible representations.
            num_radial:
                Number of radial Bessel features.
            radial_hidden:
                Hidden width of the radial network that emits tensor-product weights.
            """
            super().__init__()
            self.spherical_harmonics = o3.SphericalHarmonics(
                irreps_edge, normalize=True, normalization="component"
            )
            self.tensor_product = o3.FullyConnectedTensorProduct(
                irreps_node,
                irreps_edge,
                irreps_node,
                shared_weights=False,
            )
            self.radial_mlp = e3nn_nn.FullyConnectedNet(
                [num_radial, radial_hidden, radial_hidden, self.tensor_product.weight_numel],
                torch.nn.functional.silu,
            )
            self.self_connection = o3.Linear(irreps_node, irreps_node)

        def forward(
            self,
            node_features: Tensor,
            edge_src: Tensor,
            edge_dst: Tensor,
            edge_vectors: Tensor,
            radial_features: Tensor,
        ) -> Tensor:
            """Apply one tensor-product message-passing step.

            Parameters
            ----------
            node_features:
                Node irreps features ``(num_nodes, irreps_node.dim)``.
            edge_src:
                Source node indices for directed edges.
            edge_dst:
                Destination node indices for directed edges.
            edge_vectors:
                Relative edge vectors ``positions[dst] - positions[src]``.
            radial_features:
                Cutoff-weighted Bessel radial edge features.

            Returns
            -------
            Tensor
                Updated node irreps features.
            """
            edge_sh = self.spherical_harmonics(edge_vectors)
            weights = self.radial_mlp(radial_features)
            messages = self.tensor_product(node_features[edge_src], edge_sh, weights)
            aggregated = torch.zeros_like(node_features).index_add(0, edge_dst, messages)
            return node_features + self.self_connection(aggregated)

    class NequIPInteractionBlock(nn.Module):
        """E(3)-equivariant interaction block with scalar gating."""

        def __init__(
            self,
            irreps_node: "o3.Irreps",
            irreps_edge: "o3.Irreps",
            num_radial: int,
            scalar_dim: int,
        ) -> None:
            """Initialize one compact NequIP interaction block.

            Parameters
            ----------
            irreps_node:
                Node feature irreducible representations.
            irreps_edge:
                Edge spherical-harmonic irreducible representations.
            num_radial:
                Number of radial Bessel features.
            scalar_dim:
                Number of leading scalar channels used for gating.
            """
            super().__init__()
            self.convolution = E3NNTensorProductConvolution(irreps_node, irreps_edge, num_radial)
            self.scalar_gate = nn.Sequential(
                nn.Linear(scalar_dim, scalar_dim),
                nn.SiLU(),
                nn.Linear(scalar_dim, irreps_node.dim),
            )
            self.scalar_dim = scalar_dim

        def forward(
            self,
            node_features: Tensor,
            edge_src: Tensor,
            edge_dst: Tensor,
            edge_vectors: Tensor,
            radial_features: Tensor,
        ) -> Tensor:
            """Update node features using tensor-product messages and scalar gates.

            Parameters
            ----------
            node_features:
                Node irreps features.
            edge_src:
                Source node indices.
            edge_dst:
                Destination node indices.
            edge_vectors:
                Relative edge vectors.
            radial_features:
                Radial Bessel features.

            Returns
            -------
            Tensor
                Updated node irreps features.
            """
            updated = self.convolution(
                node_features,
                edge_src,
                edge_dst,
                edge_vectors,
                radial_features,
            )
            gate = torch.sigmoid(self.scalar_gate(updated[:, : self.scalar_dim]))
            return updated * gate

    class NequIPPotential(nn.Module):
        """Compact e3nn-backed NequIP interatomic potential."""

        def __init__(
            self,
            num_atom_types: int = 16,
            scalar_dim: int = 8,
            vector_mul: int = 3,
            num_radial: int = 6,
            num_interactions: int = 2,
            cutoff: float = 4.0,
            lmax: int = 2,
        ) -> None:
            """Initialize atom embedding, equivariant blocks, and energy head.

            Parameters
            ----------
            num_atom_types:
                Size of the atom-type embedding table.
            scalar_dim:
                Number of scalar ``0e`` channels.
            vector_mul:
                Multiplicity of vector ``1o`` channels.
            num_radial:
                Number of radial Bessel features.
            num_interactions:
                Number of interaction blocks.
            cutoff:
                Interatomic cutoff radius.
            lmax:
                Maximum spherical-harmonic degree used on edges.
            """
            super().__init__()
            self.irreps_node = o3.Irreps(f"{scalar_dim}x0e + {vector_mul}x1o")
            self.irreps_edge = o3.Irreps.spherical_harmonics(lmax)
            self.scalar_dim = scalar_dim
            self.atom_embedding = nn.Embedding(num_atom_types, scalar_dim)
            self.radial_basis = BesselRadialBasis(num_basis=num_radial, cutoff=cutoff)
            self.interactions = nn.ModuleList(
                [
                    NequIPInteractionBlock(
                        self.irreps_node,
                        self.irreps_edge,
                        num_radial,
                        scalar_dim,
                    )
                    for _ in range(num_interactions)
                ]
            )
            self.energy_readout = nn.Sequential(
                nn.Linear(scalar_dim, scalar_dim),
                nn.SiLU(),
                nn.Linear(scalar_dim, 1),
            )

        def forward(self, positions: Tensor, atom_types: Tensor, edge_index: Tensor) -> Tensor:
            """Predict total molecular energy from a directed atomic graph.

            Parameters
            ----------
            positions:
                Atom coordinates with shape ``(num_atoms, 3)``.
            atom_types:
                Atom-type integer ids with shape ``(num_atoms,)``.
            edge_index:
                Directed edge indices with shape ``(2, num_edges)``.

            Returns
            -------
            Tensor
                Scalar total energy with shape ``(1,)``.
            """
            edge_src, edge_dst = edge_index
            edge_vectors = positions[edge_dst] - positions[edge_src]
            distances = edge_vectors.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            radial_features = self.radial_basis(distances)
            scalar_features = self.atom_embedding(atom_types)
            higher_order = torch.zeros(
                positions.shape[0],
                self.irreps_node.dim - self.scalar_dim,
                dtype=positions.dtype,
                device=positions.device,
            )
            node_features = torch.cat([scalar_features, higher_order], dim=-1)
            for block in self.interactions:
                node_features = block(
                    node_features,
                    edge_src,
                    edge_dst,
                    edge_vectors,
                    radial_features,
                )
            per_atom_energy = self.energy_readout(node_features[:, : self.scalar_dim])
            return per_atom_energy.sum(dim=0)

else:

    class FallbackTensorProductConvolution(nn.Module):
        """Fallback angular tensor-product message passing without e3nn."""

        def __init__(self, scalar_dim: int, vector_dim: int, num_radial: int) -> None:
            """Initialize fallback tensor-product weights.

            Parameters
            ----------
            scalar_dim:
                Number of scalar node channels.
            vector_dim:
                Number of vector node channels.
            num_radial:
                Number of radial Bessel features.
            """
            super().__init__()
            self.scalar_dim = scalar_dim
            self.vector_dim = vector_dim
            self.radial_mlp = nn.Sequential(
                nn.Linear(num_radial, 32),
                nn.SiLU(),
                nn.Linear(32, 4 * scalar_dim + 3 * vector_dim),
            )

        def forward(
            self,
            scalars: Tensor,
            vectors: Tensor,
            edge_src: Tensor,
            edge_dst: Tensor,
            edge_vectors: Tensor,
            radial_features: Tensor,
        ) -> tuple[Tensor, Tensor]:
            """Apply a radial-weighted angular message update.

            Parameters
            ----------
            scalars:
                Scalar node features.
            vectors:
                Vector node features ``(num_nodes, vector_dim, 3)``.
            edge_src:
                Source node indices.
            edge_dst:
                Destination node indices.
            edge_vectors:
                Relative edge vectors.
            radial_features:
                Cutoff-weighted Bessel radial edge features.

            Returns
            -------
            tuple[Tensor, Tensor]
                Updated scalar and vector node features.
            """
            unit = edge_vectors / edge_vectors.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            quadrupole = _fallback_l2_channels(unit)
            weights = self.radial_mlp(radial_features)
            ss, vs, hv, vv, qh, qv, gate = torch.split(
                weights,
                [
                    self.scalar_dim,
                    self.scalar_dim,
                    self.vector_dim,
                    self.vector_dim,
                    self.scalar_dim,
                    self.vector_dim,
                    self.vector_dim,
                ],
                dim=-1,
            )
            src_scalars = scalars[edge_src]
            src_vectors = vectors[edge_src]
            projected_vectors = (src_vectors * unit[:, None, :]).sum(dim=-1)
            scalar_messages = ss * src_scalars + vs * projected_vectors.mean(dim=-1, keepdim=True)
            scalar_messages = scalar_messages + qh * quadrupole.mean(dim=-1, keepdim=True)
            vector_messages = vv[:, :, None] * src_vectors
            vector_messages = vector_messages + hv[:, :, None] * unit[:, None, :]
            vector_messages = vector_messages + qv[:, :, None] * quadrupole[:, :3].unsqueeze(1)
            vector_messages = vector_messages * torch.sigmoid(gate[:, :, None])
            scalar_agg = torch.zeros_like(scalars).index_add(0, edge_dst, scalar_messages)
            vector_agg = torch.zeros_like(vectors).index_add(0, edge_dst, vector_messages)
            return scalars + scalar_agg, vectors + vector_agg

    class NequIPInteractionBlock(nn.Module):
        """Fallback interaction block retaining radial-angular tensor products."""

        def __init__(self, scalar_dim: int, vector_dim: int, num_radial: int) -> None:
            """Initialize fallback interaction block.

            Parameters
            ----------
            scalar_dim:
                Number of scalar channels.
            vector_dim:
                Number of vector channels.
            num_radial:
                Number of radial Bessel features.
            """
            super().__init__()
            self.convolution = FallbackTensorProductConvolution(scalar_dim, vector_dim, num_radial)
            self.scalar_update = nn.Sequential(
                nn.Linear(scalar_dim, scalar_dim),
                nn.SiLU(),
                nn.Linear(scalar_dim, scalar_dim),
            )

        def forward(
            self,
            scalars: Tensor,
            vectors: Tensor,
            edge_src: Tensor,
            edge_dst: Tensor,
            edge_vectors: Tensor,
            radial_features: Tensor,
        ) -> tuple[Tensor, Tensor]:
            """Update fallback scalar and vector node features.

            Parameters
            ----------
            scalars:
                Scalar node features.
            vectors:
                Vector node features.
            edge_src:
                Source node indices.
            edge_dst:
                Destination node indices.
            edge_vectors:
                Relative edge vectors.
            radial_features:
                Radial Bessel features.

            Returns
            -------
            tuple[Tensor, Tensor]
                Updated scalar and vector features.
            """
            scalars, vectors = self.convolution(
                scalars,
                vectors,
                edge_src,
                edge_dst,
                edge_vectors,
                radial_features,
            )
            return scalars + self.scalar_update(scalars), vectors

    class NequIPPotential(nn.Module):
        """Compact fallback NequIP potential used only when e3nn is unavailable."""

        def __init__(
            self,
            num_atom_types: int = 16,
            scalar_dim: int = 8,
            vector_dim: int = 3,
            num_radial: int = 6,
            num_interactions: int = 2,
            cutoff: float = 4.0,
        ) -> None:
            """Initialize fallback atom embedding, interactions, and readout.

            Parameters
            ----------
            num_atom_types:
                Size of the atom-type embedding table.
            scalar_dim:
                Number of scalar channels.
            vector_dim:
                Number of vector channels.
            num_radial:
                Number of radial Bessel features.
            num_interactions:
                Number of interaction blocks.
            cutoff:
                Interatomic cutoff radius.
            """
            super().__init__()
            self.atom_embedding = nn.Embedding(num_atom_types, scalar_dim)
            self.radial_basis = BesselRadialBasis(num_basis=num_radial, cutoff=cutoff)
            self.interactions = nn.ModuleList(
                [
                    NequIPInteractionBlock(scalar_dim, vector_dim, num_radial)
                    for _ in range(num_interactions)
                ]
            )
            self.vector_dim = vector_dim
            self.energy_readout = nn.Sequential(
                nn.Linear(scalar_dim + 1, scalar_dim),
                nn.SiLU(),
                nn.Linear(scalar_dim, 1),
            )

        def forward(self, positions: Tensor, atom_types: Tensor, edge_index: Tensor) -> Tensor:
            """Predict total molecular energy from a directed atomic graph.

            Parameters
            ----------
            positions:
                Atom coordinates with shape ``(num_atoms, 3)``.
            atom_types:
                Atom-type integer ids with shape ``(num_atoms,)``.
            edge_index:
                Directed edge indices with shape ``(2, num_edges)``.

            Returns
            -------
            Tensor
                Scalar total energy with shape ``(1,)``.
            """
            edge_src, edge_dst = edge_index
            edge_vectors = positions[edge_dst] - positions[edge_src]
            distances = edge_vectors.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            radial_features = self.radial_basis(distances)
            scalars = self.atom_embedding(atom_types)
            vectors = torch.zeros(
                positions.shape[0],
                self.vector_dim,
                3,
                dtype=positions.dtype,
                device=positions.device,
            )
            for block in self.interactions:
                scalars, vectors = block(
                    scalars,
                    vectors,
                    edge_src,
                    edge_dst,
                    edge_vectors,
                    radial_features,
                )
            vector_norm = vectors.norm(dim=-1).mean(dim=-1, keepdim=True)
            per_atom_energy = self.energy_readout(torch.cat([scalars, vector_norm], dim=-1))
            return per_atom_energy.sum(dim=0)


def _fallback_l2_channels(unit_vectors: Tensor) -> Tensor:
    """Build simple real degree-2 spherical-harmonic-like channels.

    Parameters
    ----------
    unit_vectors:
        Unit direction vectors with shape ``(num_edges, 3)``.

    Returns
    -------
    Tensor
        Five quadratic angular channels.
    """
    x, y, z = unit_vectors.unbind(dim=-1)
    return torch.stack(
        [
            x * y,
            y * z,
            0.5 * (3.0 * z.pow(2) - 1.0),
            x * z,
            0.5 * (x.pow(2) - y.pow(2)),
        ],
        dim=-1,
    )


def build() -> nn.Module:
    """Build a compact random-initialized NequIP model.

    Returns
    -------
    nn.Module
        NequIP-style interatomic potential model.
    """
    return NequIPPotential().eval()


def example_input() -> tuple[Tensor, Tensor, Tensor]:
    """Return a small directed molecule graph input.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Positions ``(5, 3)``, atom types ``(5,)``, and directed edge index
        ``(2, 14)``.
    """
    positions = torch.tensor(
        [
            [0.0000, 0.0000, 0.0000],
            [0.9584, 0.0000, 0.0000],
            [-0.2390, 0.9270, 0.0000],
            [1.7000, 0.6500, 0.2500],
            [-0.8500, -0.6500, -0.2000],
        ],
        dtype=torch.float32,
    )
    atom_types = torch.tensor([8, 1, 1, 6, 7], dtype=torch.long)
    edge_index = torch.tensor(
        [
            [0, 1, 0, 2, 1, 3, 3, 1, 2, 4, 4, 2, 0, 4],
            [1, 0, 2, 0, 3, 1, 2, 4, 3, 0, 2, 4, 4, 1],
        ],
        dtype=torch.long,
    )
    return positions, atom_types, edge_index


MENAGERIE_ENTRIES = [
    ("NequIP", "build", "example_input", "2022", "E7"),
]
