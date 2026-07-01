"""Molecular / crystal generative-model classics (batch w8a16).

Sources checked (paper + official repo source; no clone, no pip install --
reimplemented from scratch in base-env torch):

- CrystalFormer: Cao et al., 2024, arXiv:2403.15734.
  https://github.com/deepmodeling/CrystalFormer (official, JAX/Haiku)
  A unified autoregressive transformer for space-group-controlled crystal
  generation and crystal-structure prediction. Each symmetry-inequivalent
  atom is represented as a "quintuple" of sub-tokens -- Wyckoff letter (W),
  atom type (A), and fractional coordinates (X, Y, Z), each independently
  projected then interleaved into a length-``5n`` token sequence (see
  ``transformer.py``: ``h = jnp.concatenate([hW, hA, hX, hY, hZ]).reshape(5n,
  ...)``). The sequence is causally self-attended (Haiku
  ``MultiHeadAttention`` blocks, pre-LN residual) so that later sub-tokens
  condition on earlier ones. Every token is additively conditioned on the
  space-group embedding (``g_embeddings``, gathered by space-group id 1-230)
  and a composition embedding (fixed per crystal). We reimplement the
  space-group + composition conditioning, the per-atom (W, A, X, Y, Z)
  quintuple-token interleave, and the causal transformer stack compactly as
  ``CrystalFormerCore``.

- CubicGAN: Zhao et al., npj Comput. Mater. 2021, arXiv:2102.01880.
  https://github.com/MilesZhao/CubicGAN (official, TensorFlow/Keras)
  A conditional WGAN for ternary cubic crystal generation. The generator
  (``build_generator``) fuses three streams -- a space-group-id embedding, a
  3-element atomic-number embedding (via a fixed, pretrained atom-embedding
  lookup table), and latent Gaussian noise -- concatenated then reshaped to a
  ``(3, 1, hidden)`` per-atom feature grid. A stack of ``Conv2DTranspose`` /
  ``Conv2D`` layers (kernel ``(1, 2)`` then ``(1, 1)``) expands this into
  ``(3, 3)`` fractional atomic coordinates (tanh-bounded), and a separate MLP
  head regresses a single scalar lattice-length parameter from the flattened
  coordinates. The critic is a stack of ``Conv1D`` layers over the ``(3, 28)``
  atom-feature matrix followed by an MLP down to a scalar WGAN score (no
  sigmoid, gradient-penalty trained). We reimplement the fused
  space-group+element+noise generator (using ``ConvTranspose2d``/``Conv2d``
  over the 3-atom coordinate grid) and the ``Conv1d`` critic compactly as
  ``CubicGANGenerator`` / ``CubicGANCritic``.

- DecompDiff: Guan et al., ICML 2023, arXiv:2403.07902.
  https://github.com/bytedance/DecompDiff (official, PyTorch)
  A structure-based drug-design diffusion model that decomposes the ligand
  into a rigid scaffold plus flexible arms, each attached to its own
  subpocket-derived Gaussian *prior* (a learned center + std per decomposed
  component; see ``add_prior_node`` / ``prior_atom_emb`` /
  ``GaussianSmearing`` in ``decompdiff.py``). Protein atoms, ligand atoms, and
  prior nodes are embedded (each tagged with a one-hot node-type indicator),
  concatenated into one context, and jointly denoised by an SE(3)-equivariant
  message-passing "uni-transformer" refine net that maintains both scalar
  node features ``h`` and 3D coordinates. A time-embedding (sinusoidal MLP)
  conditions every step, and separate heads decode per-atom type logits and
  optionally bond logits. We reimplement the tri-source (protein/ligand/prior)
  node-indicator embedding, the E(n)-style equivariant coordinate-update graph
  message-passing block, and the sinusoidal time-conditioning compactly as
  ``DecompDiffCore``.

- DeepDTA: Ozturk et al., Bioinformatics 2018, arXiv:1801.10193.
  https://github.com/hkmztrk/DeepDTA (official, Keras/TensorFlow)
  Drug-target binding-affinity regression via two independent 1D-CNN towers:
  a SMILES-string tower and a protein-sequence tower, each a character
  embedding followed by three ``Conv1D`` layers of increasing filter width
  (``NUM_FILTERS``, ``2x``, ``3x``) then ``GlobalMaxPooling1D`` (see
  ``build_combined_categorical`` in ``run_experiments.py``). The two pooled
  feature vectors are concatenated and passed through three fully-connected
  layers (1024-1024-512, with dropout) down to a single affinity-score
  output. We reimplement the dual-tower character-CNN encoder with
  global-max-pool fusion compactly as ``DeepDTA``.

- DeLinker: Imrie et al., J. Chem. Inf. Model. 2020, arXiv:2001.04106.
  https://github.com/oxpig/DeLinker (official, TensorFlow)
  A conditional graph-VAE for fragment-linking molecule design (gated graph
  neural network, GGNN). Two fixed fragments are encoded via a GGNN into a
  per-node latent distribution (mean/logvar), conditioned on an explicit
  3D-geometry summary (the relative distance and orientation *angle* between
  the two fragments, see ``prepare_specific_graph_model``'s ``distance`` /
  ``angle`` placeholders). A GGNN decoder then autoregressively grows the
  linker graph: at each step, an attention-style "focus" node is selected,
  and a distance-embedding-biased score (``distance_repr`` looked up per
  candidate node) determines the next bond/atom to add, conditioned on the
  sampled latent plus the geometric distance-to-fragment feature. We
  reimplement the fragment-conditioned graph-VAE encoder (GRU-based gated
  graph message passing to mean/logvar), the geometry-conditioning (distance
  + angle) fusion, and the distance-biased autoregressive node-scoring
  decoder step compactly as ``DeLinkerCore``.

- DiffLinker: Igashov et al., Nature Machine Intelligence 2024 (orig.
  arXiv:2210.05274).
  https://github.com/igashov/DiffLinker (official, PyTorch + PyTorch
  Lightning)
  An E(3)-equivariant diffusion model (DDPM over 3D point clouds, EGNN
  backbone) that generates a linker connecting one or more fixed fragments,
  additionally predicting the linker size. The EGNN (``egnn.py``) alternates
  a scalar-feature message-passing update (``GCL``: edge MLP on
  concatenated node-feature pairs, then a node MLP with a residual add) with
  an *equivariant* coordinate update (``EquivariantUpdate``:
  ``coord_diff * coord_mlp(h_i, h_j, edge_attr)`` summed over neighbors, which
  preserves E(3) equivariance because the update is a scalar-weighted
  combination of the raw relative-position vectors). A binary linker/fragment
  mask keeps fragment atoms' coordinates fixed (only linker-atom coordinates
  receive gradient/noise) while conditioning every message on the full
  fragment+linker context. We reimplement the alternating scalar-message /
  equivariant-coordinate-update EGNN layer and the fragment-fixed linker mask
  compactly as ``DiffLinkerEGNN``.

Random init, tiny dims, CPU-only -- architecture catalog entries, not trained
weights.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# CrystalFormer
# ---------------------------------------------------------------------------


class CrystalFormerLayer(nn.Module):
    """Pre-norm causal self-attention + MLP block over the interleaved token sequence."""

    def __init__(self, dim: int = 32, num_heads: int = 4) -> None:
        """Initialize the causal self-attention block and the feed-forward block.

        Parameters
        ----------
        dim:
            Shared token feature width.
        num_heads:
            Number of self-attention heads.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, tokens: Tensor, causal_mask: Tensor) -> Tensor:
        """Apply causal self-attention then an MLP block, each pre-norm residual.

        Parameters
        ----------
        tokens:
            Interleaved per-atom sub-tokens, shape ``(batch, 5*n, dim)``.
        causal_mask:
            Additive causal attention mask, shape ``(5*n, 5*n)``.

        Returns
        -------
        Tensor
            Updated tokens, same shape as ``tokens``.
        """
        normed = self.norm1(tokens)
        attended, _ = self.attn(normed, normed, normed, attn_mask=causal_mask)
        tokens = tokens + attended
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens


class CrystalFormerCore(nn.Module):
    """Space-group-conditioned autoregressive transformer over interleaved (W,A,X,Y,Z) tokens."""

    def __init__(
        self,
        dim: int = 32,
        depth: int = 2,
        num_space_groups: int = 230,
        wyckoff_types: int = 12,
        atom_types: int = 16,
    ) -> None:
        """Initialize the space-group/composition embeddings, sub-token projections, and stack.

        Parameters
        ----------
        dim:
            Shared token feature width.
        depth:
            Number of causal self-attention layers.
        num_space_groups:
            Number of distinct space-group ids (1-230).
        wyckoff_types:
            Number of distinct Wyckoff-letter classes.
        atom_types:
            Number of distinct chemical-element classes.
        """
        super().__init__()
        self.dim = dim
        self.space_group_embed = nn.Embedding(num_space_groups, dim)
        self.wyckoff_embed = nn.Embedding(wyckoff_types, dim)
        self.atom_embed = nn.Embedding(atom_types, dim)
        self.w_proj = nn.Linear(dim * 2, dim)
        self.a_proj = nn.Linear(dim * 2, dim)
        self.x_proj = nn.Linear(dim * 2 + 1, dim)
        self.y_proj = nn.Linear(dim * 2 + 1, dim)
        self.z_proj = nn.Linear(dim * 2 + 1, dim)
        self.layers = nn.ModuleList([CrystalFormerLayer(dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.w_head = nn.Linear(dim, wyckoff_types)
        self.a_head = nn.Linear(dim, atom_types)
        self.xyz_head = nn.Linear(dim, 1)

    def forward(
        self,
        space_group: Tensor,
        wyckoff_idx: Tensor,
        atom_idx: Tensor,
        frac_coords: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Interleave per-atom sub-tokens and causally self-attend, conditioned on the space group.

        Parameters
        ----------
        space_group:
            Space-group id per crystal, shape ``(batch,)``.
        wyckoff_idx:
            Wyckoff-letter class per symmetry-inequivalent atom, shape
            ``(batch, n_atoms)``.
        atom_idx:
            Chemical-element class per atom, shape ``(batch, n_atoms)``.
        frac_coords:
            Fractional coordinates per atom, shape ``(batch, n_atoms, 3)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Wyckoff-letter logits ``(batch, n_atoms, wyckoff_types)``,
            atom-type logits ``(batch, n_atoms, atom_types)``, and a scalar
            coordinate readout per (x, y, z) sub-token, shape
            ``(batch, n_atoms, 3)``.
        """
        batch, n_atoms = wyckoff_idx.shape
        g_embed = self.space_group_embed(space_group)
        g_rep = g_embed.unsqueeze(1).expand(-1, n_atoms, -1)

        w_tok = self.w_proj(torch.cat([g_rep, self.wyckoff_embed(wyckoff_idx)], dim=-1))
        a_tok = self.a_proj(torch.cat([g_rep, self.atom_embed(atom_idx)], dim=-1))
        x_tok = self.x_proj(
            torch.cat([g_rep, self.wyckoff_embed(wyckoff_idx), frac_coords[..., :1]], dim=-1)
        )
        y_tok = self.y_proj(
            torch.cat([g_rep, self.wyckoff_embed(wyckoff_idx), frac_coords[..., 1:2]], dim=-1)
        )
        z_tok = self.z_proj(
            torch.cat([g_rep, self.wyckoff_embed(wyckoff_idx), frac_coords[..., 2:3]], dim=-1)
        )

        tokens = torch.stack([w_tok, a_tok, x_tok, y_tok, z_tok], dim=2).reshape(
            batch, 5 * n_atoms, self.dim
        )

        total = tokens.shape[1]
        causal_mask = torch.triu(
            torch.full((total, total), float("-inf"), device=tokens.device), diagonal=1
        )
        for layer in self.layers:
            tokens = layer(tokens, causal_mask)
        tokens = self.norm(tokens)

        tokens = tokens.reshape(batch, n_atoms, 5, self.dim)
        w_logits = self.w_head(tokens[:, :, 0])
        a_logits = self.a_head(tokens[:, :, 1])
        xyz = torch.cat(
            [
                self.xyz_head(tokens[:, :, 2]),
                self.xyz_head(tokens[:, :, 3]),
                self.xyz_head(tokens[:, :, 4]),
            ],
            dim=-1,
        )
        return w_logits, a_logits, xyz


def build_crystalformer() -> nn.Module:
    """Build a compact CrystalFormer model.

    Returns
    -------
    nn.Module
        Random-initialized ``CrystalFormerCore`` in eval mode.
    """
    return CrystalFormerCore().eval()


def example_input_crystalformer() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Create an example space group, Wyckoff/atom indices, and fractional coordinates.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        Space-group id ``(1,)``, Wyckoff index ``(1, 6)``, atom index
        ``(1, 6)``, and fractional coordinates ``(1, 6, 3)``.
    """
    space_group = torch.randint(0, 230, (1,))
    wyckoff_idx = torch.randint(0, 12, (1, 6))
    atom_idx = torch.randint(0, 16, (1, 6))
    frac_coords = torch.rand(1, 6, 3)
    return space_group, wyckoff_idx, atom_idx, frac_coords


# ---------------------------------------------------------------------------
# CubicGAN
# ---------------------------------------------------------------------------


class CubicGANGenerator(nn.Module):
    """Space-group + 3-element + noise fused generator, ConvTranspose2d coordinate decoder."""

    def __init__(
        self,
        n_elements: int = 63,
        n_space_groups: int = 123,
        latent_dim: int = 32,
        hidden: int = 32,
    ) -> None:
        """Initialize the space-group/element embeddings and the coordinate/lattice decoders.

        Parameters
        ----------
        n_elements:
            Number of distinct chemical elements in the atomic-number lookup.
        n_space_groups:
            Number of distinct cubic space-group classes.
        latent_dim:
            Latent Gaussian noise dimensionality.
        hidden:
            Shared hidden feature width.
        """
        super().__init__()
        self.space_group_embed = nn.Embedding(n_space_groups, hidden)
        self.element_embed = nn.Embedding(n_elements, hidden)
        self.noise_proj = nn.Linear(latent_dim, hidden * 2)
        self.fuse = nn.Linear(hidden + hidden * 3 + hidden * 2, hidden * 3)

        self.coord_decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden, hidden * 2, kernel_size=(1, 2)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden * 2, hidden * 2, kernel_size=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden * 2, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Tanh(),
        )
        self.lattice_head = nn.Sequential(
            nn.Linear(9, 16), nn.ReLU(inplace=True), nn.Linear(16, 1), nn.Tanh()
        )
        self.hidden = hidden

    def forward(
        self, space_group: Tensor, element_ids: Tensor, latent_noise: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Fuse conditioning streams and decode fractional coordinates + a lattice-length scalar.

        Parameters
        ----------
        space_group:
            Space-group id, shape ``(batch,)``.
        element_ids:
            Three chosen element ids per crystal, shape ``(batch, 3)``.
        latent_noise:
            Latent Gaussian noise, shape ``(batch, latent_dim)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Fractional coordinates ``(batch, 3, 3)`` and lattice-length
            scalar ``(batch, 1)``.
        """
        batch = space_group.shape[0]
        sg_feat = self.space_group_embed(space_group)
        el_feat = self.element_embed(element_ids).reshape(batch, -1)
        noise_feat = self.noise_proj(latent_noise)

        fused = self.fuse(torch.cat([sg_feat, el_feat, noise_feat], dim=-1))
        grid = fused.view(batch, self.hidden, 3, 1)
        coords = self.coord_decoder(grid).view(batch, 3, 3)

        lengths = self.lattice_head(coords.reshape(batch, -1))
        return coords, lengths


class CubicGANCritic(nn.Module):
    """WGAN critic: Conv1d stack over the per-atom coordinate/feature matrix down to a scalar score."""

    def __init__(self, feature_dim: int = 28, hidden: int = 32) -> None:
        """Initialize the Conv1d trunk and the scalar-score MLP head.

        Parameters
        ----------
        feature_dim:
            Per-atom feature-vector width (coordinates + element/lattice
            side information).
        hidden:
            Base Conv1d channel width.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(feature_dim, hidden, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden, hidden * 2, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.score = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden * 2 * 3, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, atom_features: Tensor) -> Tensor:
        """Score a batch of 3-atom crystal feature matrices with no output activation.

        Parameters
        ----------
        atom_features:
            Per-atom feature matrix, shape ``(batch, 3, feature_dim)``.

        Returns
        -------
        Tensor
            Unbounded WGAN critic score, shape ``(batch, 1)``.
        """
        x = self.conv(atom_features.transpose(1, 2))
        return self.score(x)


class CubicGAN(nn.Module):
    """CubicGAN: conditional generator + WGAN critic for ternary cubic crystal structures."""

    def __init__(self) -> None:
        """Initialize the generator and the critic."""
        super().__init__()
        self.generator = CubicGANGenerator()
        self.critic = CubicGANCritic()

    def forward(
        self, space_group: Tensor, element_ids: Tensor, latent_noise: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Generate a crystal then score its atomic-coordinate matrix with the critic.

        Parameters
        ----------
        space_group:
            Space-group id, shape ``(batch,)``.
        element_ids:
            Three chosen element ids per crystal, shape ``(batch, 3)``.
        latent_noise:
            Latent Gaussian noise, shape ``(batch, latent_dim)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Generated fractional coordinates ``(batch, 3, 3)``, lattice
            length ``(batch, 1)``, and critic score ``(batch, 1)``.
        """
        coords, lengths = self.generator(space_group, element_ids, latent_noise)
        pad = coords.new_zeros(coords.shape[0], 3, 28 - 3)
        critic_input = torch.cat([coords, pad], dim=-1)
        score = self.critic(critic_input)
        return coords, lengths, score


def build_cubicgan() -> nn.Module:
    """Build a compact CubicGAN model.

    Returns
    -------
    nn.Module
        Random-initialized ``CubicGAN`` in eval mode.
    """
    return CubicGAN().eval()


def example_input_cubicgan() -> tuple[Tensor, Tensor, Tensor]:
    """Create an example space group, element triple, and latent noise.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Space-group id ``(1,)``, element ids ``(1, 3)``, and latent noise
        ``(1, 32)``.
    """
    space_group = torch.randint(0, 123, (1,))
    element_ids = torch.randint(0, 63, (1, 3))
    latent_noise = torch.randn(1, 32)
    return space_group, element_ids, latent_noise


# ---------------------------------------------------------------------------
# DecompDiff
# ---------------------------------------------------------------------------


class DecompDiffEquivariantLayer(nn.Module):
    """SE(3)-equivariant message-passing layer: scalar node update + equivariant coordinate update."""

    def __init__(self, hidden: int = 32) -> None:
        """Initialize the edge MLP, node-update MLP, and equivariant coordinate-update MLP.

        Parameters
        ----------
        hidden:
            Shared node/edge feature width.
        """
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden * 2 + 1, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU()
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.coord_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, h: Tensor, pos: Tensor, node_type_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Update scalar node features and 3D coordinates via fully-connected equivariant message passing.

        Parameters
        ----------
        h:
            Node scalar features, shape ``(batch, n_nodes, hidden)``.
        pos:
            Node 3D coordinates, shape ``(batch, n_nodes, 3)``.
        node_type_mask:
            Per-node updatable mask (``1.0`` for ligand/prior nodes whose
            coordinates diffuse, ``0.0`` for fixed protein context nodes),
            shape ``(batch, n_nodes, 1)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Updated node features and updated coordinates, same shapes as
            inputs.
        """
        rel_pos = pos.unsqueeze(2) - pos.unsqueeze(1)
        dist2 = (rel_pos**2).sum(-1, keepdim=True)

        h_i = h.unsqueeze(2).expand(-1, -1, h.shape[1], -1)
        h_j = h.unsqueeze(1).expand(-1, h.shape[1], -1, -1)
        edge_feat = self.edge_mlp(torch.cat([h_i, h_j, dist2], dim=-1))

        agg = edge_feat.mean(dim=2)
        h_new = h + self.node_mlp(torch.cat([h, agg], dim=-1))

        coord_weight = self.coord_mlp(edge_feat)
        coord_update = (rel_pos * coord_weight).mean(dim=2)
        pos_new = pos + coord_update * node_type_mask
        return h_new, pos_new


class DecompDiffCore(nn.Module):
    """Prior-node-conditioned equivariant diffusion refine-net over protein+ligand+prior context."""

    def __init__(self, hidden: int = 32, depth: int = 2, num_atom_classes: int = 10) -> None:
        """Initialize the tri-source node embeddings, time embedding, and equivariant message-passing stack.

        Parameters
        ----------
        hidden:
            Shared node feature width.
        depth:
            Number of equivariant message-passing layers.
        num_atom_classes:
            Number of ligand atom-type classes predicted per node.
        """
        super().__init__()
        self.protein_embed = nn.Linear(8 + 1, hidden)
        self.ligand_embed = nn.Linear(num_atom_classes + 1, hidden)
        self.prior_embed = nn.Linear(1 + 1, hidden)
        self.time_embed = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.layers = nn.ModuleList([DecompDiffEquivariantLayer(hidden) for _ in range(depth)])
        self.atom_type_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, num_atom_classes)
        )

    def forward(
        self,
        protein_feat: Tensor,
        protein_pos: Tensor,
        ligand_feat: Tensor,
        ligand_pos: Tensor,
        prior_std: Tensor,
        prior_pos: Tensor,
        time_step: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compose protein/ligand/prior nodes, denoise coordinates, and predict ligand atom types.

        Parameters
        ----------
        protein_feat:
            Protein atom features, shape ``(batch, n_protein, 8)``.
        protein_pos:
            Protein atom coordinates, shape ``(batch, n_protein, 3)``.
        ligand_feat:
            One-hot ligand atom-type features, shape
            ``(batch, n_ligand, num_atom_classes)``.
        ligand_pos:
            Noised ligand atom coordinates, shape ``(batch, n_ligand, 3)``.
        prior_std:
            Per-decomposed-component subpocket prior std, shape
            ``(batch, n_prior, 1)``.
        prior_pos:
            Prior-node (arm/scaffold center) coordinates, shape
            ``(batch, n_prior, 3)``.
        time_step:
            Diffusion timestep (normalized), shape ``(batch, 1)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Predicted ligand atom-type logits ``(batch, n_ligand,
            num_atom_classes)`` and denoised ligand coordinates
            ``(batch, n_ligand, 3)``.
        """
        batch, n_protein = protein_feat.shape[:2]
        n_ligand = ligand_feat.shape[1]
        n_prior = prior_std.shape[1]
        time_feat = self.time_embed(time_step)

        h_protein = self.protein_embed(
            torch.cat([protein_feat, time_feat.new_zeros(batch, n_protein, 1)], dim=-1)
        )
        time_rep_lig = time_feat.unsqueeze(1).expand(-1, n_ligand, -1).mean(-1, keepdim=True)
        h_ligand = self.ligand_embed(torch.cat([ligand_feat, time_rep_lig], dim=-1))
        h_prior = self.prior_embed(
            torch.cat([prior_std, time_feat.new_ones(batch, n_prior, 1)], dim=-1)
        )

        h = torch.cat([h_protein, h_ligand, h_prior], dim=1)
        pos = torch.cat([protein_pos, ligand_pos, prior_pos], dim=1)
        node_type_mask = torch.cat(
            [
                h.new_zeros(batch, n_protein, 1),
                h.new_ones(batch, n_ligand, 1),
                h.new_zeros(batch, n_prior, 1),
            ],
            dim=1,
        )

        for layer in self.layers:
            h, pos = layer(h, pos, node_type_mask)

        h_ligand_out = h[:, n_protein : n_protein + n_ligand]
        pos_ligand_out = pos[:, n_protein : n_protein + n_ligand]
        atom_logits = self.atom_type_head(h_ligand_out)
        return atom_logits, pos_ligand_out


def build_decompdiff() -> nn.Module:
    """Build a compact DecompDiff model.

    Returns
    -------
    nn.Module
        Random-initialized ``DecompDiffCore`` in eval mode.
    """
    return DecompDiffCore().eval()


def example_input_decompdiff() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Create example protein, ligand, and decomposed-prior context tensors.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
        Protein features ``(1, 6, 8)``, protein positions ``(1, 6, 3)``,
        ligand one-hot features ``(1, 5, 10)``, ligand positions
        ``(1, 5, 3)``, prior std ``(1, 2, 1)``, prior positions
        ``(1, 2, 3)``, and normalized time step ``(1, 1)``.
    """
    protein_feat = torch.randn(1, 6, 8)
    protein_pos = torch.randn(1, 6, 3)
    ligand_feat = F.one_hot(torch.randint(0, 10, (1, 5)), num_classes=10).float()
    ligand_pos = torch.randn(1, 5, 3)
    prior_std = torch.rand(1, 2, 1)
    prior_pos = torch.randn(1, 2, 3)
    time_step = torch.rand(1, 1)
    return protein_feat, protein_pos, ligand_feat, ligand_pos, prior_std, prior_pos, time_step


# ---------------------------------------------------------------------------
# DeepDTA
# ---------------------------------------------------------------------------


class DeepDTATower(nn.Module):
    """Character-embedding + 3-layer widening Conv1d tower with global max pooling."""

    def __init__(
        self, vocab_size: int, embed_dim: int = 32, num_filters: int = 16, kernel_size: int = 4
    ) -> None:
        """Initialize the character embedding and the three widening Conv1d layers.

        Parameters
        ----------
        vocab_size:
            Number of distinct characters in the sequence alphabet.
        embed_dim:
            Character-embedding width.
        num_filters:
            Base number of convolution filters (widens 1x, 2x, 3x across
            the three layers).
        kernel_size:
            Convolution kernel width.
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, num_filters, kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_filters, num_filters * 2, kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_filters * 2, num_filters * 3, kernel_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, char_ids: Tensor) -> Tensor:
        """Embed a character sequence, convolve, and global-max-pool over the sequence axis.

        Parameters
        ----------
        char_ids:
            Integer character-id sequence, shape ``(batch, seq_len)``.

        Returns
        -------
        Tensor
            Pooled feature vector, shape ``(batch, num_filters * 3)``.
        """
        x = self.embed(char_ids).transpose(1, 2)
        x = self.conv(x)
        return x.amax(dim=-1)


class DeepDTA(nn.Module):
    """DeepDTA: dual SMILES/protein character-CNN towers fused into a binding-affinity regressor."""

    def __init__(
        self,
        smiles_vocab: int = 64,
        protein_vocab: int = 25,
        num_filters: int = 16,
    ) -> None:
        """Initialize the SMILES tower, the protein tower, and the fusion MLP head.

        Parameters
        ----------
        smiles_vocab:
            Number of distinct SMILES characters.
        protein_vocab:
            Number of distinct amino-acid characters.
        num_filters:
            Base Conv1d filter count (shared tower hyperparameter).
        """
        super().__init__()
        self.smiles_tower = DeepDTATower(smiles_vocab, num_filters=num_filters, kernel_size=4)
        self.protein_tower = DeepDTATower(protein_vocab, num_filters=num_filters, kernel_size=8)
        fused_dim = num_filters * 3 * 2
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, smiles_ids: Tensor, protein_ids: Tensor) -> Tensor:
        """Encode SMILES + protein sequences independently, concatenate, and regress affinity.

        Parameters
        ----------
        smiles_ids:
            Integer SMILES-character-id sequence, shape ``(batch, smi_len)``.
        protein_ids:
            Integer protein-character-id sequence, shape ``(batch, seq_len)``.

        Returns
        -------
        Tensor
            Predicted binding-affinity score, shape ``(batch, 1)``.
        """
        smiles_feat = self.smiles_tower(smiles_ids)
        protein_feat = self.protein_tower(protein_ids)
        fused = torch.cat([smiles_feat, protein_feat], dim=-1)
        return self.head(fused)


def build_deepdta() -> nn.Module:
    """Build a compact DeepDTA model.

    Returns
    -------
    nn.Module
        Random-initialized ``DeepDTA`` in eval mode.
    """
    return DeepDTA().eval()


def example_input_deepdta() -> tuple[Tensor, Tensor]:
    """Create example SMILES and protein-sequence character-id tensors.

    Returns
    -------
    tuple[Tensor, Tensor]
        SMILES character ids ``(1, 40)`` and protein character ids
        ``(1, 60)``.
    """
    smiles_ids = torch.randint(1, 64, (1, 40))
    protein_ids = torch.randint(1, 25, (1, 60))
    return smiles_ids, protein_ids


# ---------------------------------------------------------------------------
# DeLinker
# ---------------------------------------------------------------------------


class DeLinkerGGNNBlock(nn.Module):
    """Gated graph message-passing block (fully-connected edge sum + GRU node update)."""

    def __init__(self, hidden: int = 32) -> None:
        """Initialize the message MLP and the GRU cell used for the gated node update.

        Parameters
        ----------
        hidden:
            Shared node feature width.
        """
        super().__init__()
        self.message = nn.Linear(hidden, hidden)
        self.gru = nn.GRUCell(hidden, hidden)

    def forward(self, h: Tensor) -> Tensor:
        """Aggregate fully-connected messages and apply a GRU-gated node update.

        Parameters
        ----------
        h:
            Node features, shape ``(batch, n_nodes, hidden)``.

        Returns
        -------
        Tensor
            Updated node features, same shape as ``h``.
        """
        batch, n_nodes, hidden = h.shape
        messages = self.message(h)
        agg = (messages.sum(dim=1, keepdim=True) - messages) / max(n_nodes - 1, 1)
        flat_h = h.reshape(batch * n_nodes, hidden)
        flat_agg = agg.reshape(batch * n_nodes, hidden)
        return self.gru(flat_agg, flat_h).reshape(batch, n_nodes, hidden)


class DeLinkerCore(nn.Module):
    """Fragment-conditioned graph-VAE: GGNN encoder to mean/logvar, distance-biased decoder step."""

    def __init__(self, hidden: int = 32, depth: int = 2, max_distance: int = 20) -> None:
        """Initialize the node embedding, GGNN encoder stack, geometry fusion, and decoder step.

        Parameters
        ----------
        hidden:
            Shared node/latent feature width.
        depth:
            Number of GGNN message-passing rounds.
        max_distance:
            Size of the discretized fragment-distance embedding table.
        """
        super().__init__()
        self.node_embed = nn.Linear(8, hidden)
        self.encoder_layers = nn.ModuleList([DeLinkerGGNNBlock(hidden) for _ in range(depth)])
        self.geometry_fuse = nn.Sequential(nn.Linear(hidden + 2, hidden), nn.ReLU(inplace=True))
        self.mean_head = nn.Linear(hidden, hidden)
        self.logvar_head = nn.Linear(hidden, hidden)
        self.distance_embed = nn.Embedding(max_distance, hidden)
        self.focus_score = nn.Linear(hidden, 1)
        self.decoder_layers = nn.ModuleList([DeLinkerGGNNBlock(hidden) for _ in range(depth)])
        self.node_type_head = nn.Linear(hidden, 8)

    def forward(
        self,
        node_features: Tensor,
        fragment_distance: Tensor,
        fragment_angle: Tensor,
        distance_to_fragment: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Encode fragments (geometry-conditioned) to a latent, then score candidate linker nodes.

        Parameters
        ----------
        node_features:
            Per-atom fragment feature vectors, shape
            ``(batch, n_nodes, 8)``.
        fragment_distance:
            Scalar distance between the two fragments, shape ``(batch, 1)``.
        fragment_angle:
            Scalar relative orientation angle between the two fragments,
            shape ``(batch, 1)``.
        distance_to_fragment:
            Discretized per-node distance-to-fragment bucket id, used to
            bias the autoregressive focus-node score, shape
            ``(batch, n_nodes)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor]
            Latent mean ``(batch, n_nodes, hidden)``, latent logvar
            ``(batch, n_nodes, hidden)``, per-node focus/attachment logits
            ``(batch, n_nodes)``, and decoded node-type logits
            ``(batch, n_nodes, 8)``.
        """
        batch, n_nodes = node_features.shape[:2]
        h = self.node_embed(node_features)
        for layer in self.encoder_layers:
            h = layer(h)

        geometry = torch.cat([fragment_distance, fragment_angle], dim=-1)
        geometry_rep = geometry.unsqueeze(1).expand(-1, n_nodes, -1)
        h = self.geometry_fuse(torch.cat([h, geometry_rep], dim=-1))

        mean = self.mean_head(h)
        logvar = self.logvar_head(h)
        z = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)

        dist_bias = self.distance_embed(distance_to_fragment)
        focus_logits = self.focus_score(z + dist_bias).squeeze(-1)

        h_dec = z
        for layer in self.decoder_layers:
            h_dec = layer(h_dec)
        node_type_logits = self.node_type_head(h_dec)

        return mean, logvar, focus_logits, node_type_logits


def build_delinker() -> nn.Module:
    """Build a compact DeLinker model.

    Returns
    -------
    nn.Module
        Random-initialized ``DeLinkerCore`` in eval mode.
    """
    return DeLinkerCore().eval()


def example_input_delinker() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Create example fragment node features, fragment geometry, and distance buckets.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        Node features ``(1, 10, 8)``, fragment distance ``(1, 1)``, fragment
        angle ``(1, 1)``, and per-node distance-to-fragment bucket ids
        ``(1, 10)``.
    """
    node_features = torch.randn(1, 10, 8)
    fragment_distance = torch.rand(1, 1) * 10.0
    fragment_angle = torch.rand(1, 1) * 3.14159
    distance_to_fragment = torch.randint(0, 20, (1, 10))
    return node_features, fragment_distance, fragment_angle, distance_to_fragment


# ---------------------------------------------------------------------------
# DiffLinker
# ---------------------------------------------------------------------------


class DiffLinkerEGNNLayer(nn.Module):
    """EGNN layer: scalar message-passing node update + E(3)-equivariant coordinate update."""

    def __init__(self, hidden: int = 32) -> None:
        """Initialize the edge MLP, node-update MLP, and equivariant coordinate MLP.

        Parameters
        ----------
        hidden:
            Shared node/edge feature width.
        """
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden * 2 + 1, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU()
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        coord_out = nn.Linear(hidden, 1, bias=False)
        nn.init.xavier_uniform_(coord_out.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), coord_out
        )

    def forward(self, h: Tensor, coord: Tensor, linker_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Update scalar node features and equivariantly update coordinates (linker atoms only).

        Parameters
        ----------
        h:
            Node scalar features, shape ``(batch, n_nodes, hidden)``.
        coord:
            Node 3D coordinates, shape ``(batch, n_nodes, 3)``.
        linker_mask:
            Per-node mask, ``1.0`` for linker atoms whose coordinates may
            move, ``0.0`` for fixed fragment atoms, shape
            ``(batch, n_nodes, 1)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Updated node features and updated coordinates, same shapes as
            inputs.
        """
        coord_diff = coord.unsqueeze(2) - coord.unsqueeze(1)
        dist2 = (coord_diff**2).sum(-1, keepdim=True)

        h_i = h.unsqueeze(2).expand(-1, -1, h.shape[1], -1)
        h_j = h.unsqueeze(1).expand(-1, h.shape[1], -1, -1)
        edge_feat = self.edge_mlp(torch.cat([h_i, h_j, dist2], dim=-1))

        agg_message = edge_feat.mean(dim=2)
        h_new = h + self.node_mlp(torch.cat([h, agg_message], dim=-1))

        coord_weight = self.coord_mlp(edge_feat)
        coord_update = (coord_diff * coord_weight).mean(dim=2)
        coord_new = coord + coord_update * linker_mask
        return h_new, coord_new


class DiffLinkerEGNN(nn.Module):
    """DiffLinker: E(3)-equivariant EGNN denoiser generating a linker between fixed fragments."""

    def __init__(self, hidden: int = 32, depth: int = 3, num_atom_types: int = 8) -> None:
        """Initialize the atom-type/time embedding and the stacked EGNN layers.

        Parameters
        ----------
        hidden:
            Shared node feature width.
        depth:
            Number of EGNN message-passing layers.
        num_atom_types:
            Number of distinct atom-type classes.
        """
        super().__init__()
        self.atom_embed = nn.Linear(num_atom_types + 1, hidden)
        self.time_embed = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.layers = nn.ModuleList([DiffLinkerEGNNLayer(hidden) for _ in range(depth)])
        self.atom_type_head = nn.Linear(hidden, num_atom_types)
        self.size_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(
        self,
        atom_types: Tensor,
        coords: Tensor,
        linker_mask: Tensor,
        time_step: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Denoise linker-atom coordinates and predict atom types + linker size, given fixed fragments.

        Parameters
        ----------
        atom_types:
            One-hot atom-type features for the fragment+linker point cloud,
            shape ``(batch, n_atoms, num_atom_types)``.
        coords:
            3D coordinates (fragments fixed, linker noised), shape
            ``(batch, n_atoms, 3)``.
        linker_mask:
            Per-atom mask, ``1.0`` for linker atoms, ``0.0`` for fragment
            atoms, shape ``(batch, n_atoms, 1)``.
        time_step:
            Diffusion timestep (normalized), shape ``(batch, 1)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Predicted atom-type logits ``(batch, n_atoms, num_atom_types)``,
            denoised coordinates ``(batch, n_atoms, 3)``, and a pooled
            predicted linker-size scalar ``(batch, 1)``.
        """
        batch, n_atoms = atom_types.shape[:2]
        time_scalar = time_step.unsqueeze(1).expand(-1, n_atoms, -1).mean(-1, keepdim=True)
        h = self.atom_embed(torch.cat([atom_types, time_scalar], dim=-1))
        time_feat = self.time_embed(time_step)
        h = h + time_feat.unsqueeze(1)

        for layer in self.layers:
            h, coords = layer(h, coords, linker_mask)

        atom_logits = self.atom_type_head(h)
        pooled = (h * linker_mask).sum(dim=1) / linker_mask.sum(dim=1).clamp(min=1.0)
        linker_size = self.size_head(pooled)
        return atom_logits, coords, linker_size


def build_difflinker() -> nn.Module:
    """Build a compact DiffLinker model.

    Returns
    -------
    nn.Module
        Random-initialized ``DiffLinkerEGNN`` in eval mode.
    """
    return DiffLinkerEGNN().eval()


def example_input_difflinker() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Create an example fragment+linker point cloud with a linker mask and time step.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        One-hot atom types ``(1, 12, 8)``, coordinates ``(1, 12, 3)``,
        linker mask ``(1, 12, 1)``, and normalized time step ``(1, 1)``.
    """
    atom_types = F.one_hot(torch.randint(0, 8, (1, 12)), num_classes=8).float()
    coords = torch.randn(1, 12, 3)
    linker_mask = torch.zeros(1, 12, 1)
    linker_mask[:, 4:8] = 1.0
    time_step = torch.rand(1, 1)
    return atom_types, coords, linker_mask, time_step


MENAGERIE_ENTRIES = [
    ("CrystalFormer", "build_crystalformer", "example_input_crystalformer", "2024", "BIO"),
    ("CubicGAN", "build_cubicgan", "example_input_cubicgan", "2021", "BIO"),
    ("DecompDiff", "build_decompdiff", "example_input_decompdiff", "2023", "BIO"),
    ("DeepDTA", "build_deepdta", "example_input_deepdta", "2018", "BIO"),
    ("DeLinker", "build_delinker", "example_input_delinker", "2020", "BIO"),
    ("DiffLinker", "build_difflinker", "example_input_difflinker", "2024", "BIO"),
]
