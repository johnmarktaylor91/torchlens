# SOURCE: vendored from arneschneuing/DiffSBDD @ main
# https://raw.githubusercontent.com/arneschneuing/DiffSBDD/main/equivariant_diffusion/egnn_new.py
# https://raw.githubusercontent.com/arneschneuing/DiffSBDD/main/equivariant_diffusion/dynamics.py
#
# DiffSBDD -- Schneuing et al. 2023 "Structure-based Drug Design with Equivariant
# Diffusion Models" -- an SE(3)-equivariant E(n)-GNN denoising network jointly
# operating on ligand atoms and protein-pocket residues (the noise-prediction
# "dynamics" model wrapped by the diffusion process). `GCL`, `EquivariantUpdate`,
# `EquivariantBlock`, `EGNN`, `GNN`, `SinusoidsEmbeddingNew`, and the
# `coord2diff`/`coord2cross`/`unsorted_segment_sum` helpers below are copied
# verbatim from the real `equivariant_diffusion/egnn_new.py`. `EGNNDynamics`
# (the top-level dynamics network) is copied verbatim from
# `equivariant_diffusion/dynamics.py`, with one change: the real file does
# `from equivariant_diffusion.en_diffusion import EnVariationalDiffusion` purely
# to grab the single static method `EnVariationalDiffusion.remove_mean_batch`;
# that import chain pulls in the repo's root `utils.py`, which needs `rdkit` and
# `Bio.PDB` for unrelated molecule/graph-isomorphism bookkeeping used elsewhere
# in that module -- none of it is exercised by `EGNNDynamics.forward`. Rather
# than install those two packages to satisfy an unrelated transitive import, the
# 3-line `remove_mean_batch` staticmethod is inlined verbatim below (see
# `en_diffusion.py` lines ~918-923) as a free function. No architectural code was
# rewritten.
#
# Upstream license: MIT (arneschneuing/DiffSBDD).

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

MENAGERIE_ZOO = "vendored-pytorch"


# --- verbatim from equivariant_diffusion/egnn_new.py ---------------------------


class GCL(nn.Module):
    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        normalization_factor,
        aggregation_method,
        edges_in_d=0,
        nodes_att_dim=0,
        act_fn=nn.SiLU(),
        attention=False,
    ):
        super(GCL, self).__init__()
        input_edge = input_nf * 2
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def edge_model(self, source, target, edge_attr, edge_mask):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)
        mij = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij

        if edge_mask is not None:
            out = out * edge_mask
        return out, mij

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(
            edge_attr,
            row,
            num_segments=x.size(0),
            normalization_factor=self.normalization_factor,
            aggregation_method=self.aggregation_method,
        )
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = x + self.node_mlp(agg)
        return out, agg

    def forward(
        self, h, edge_index, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None
    ):
        row, col = edge_index
        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        if node_mask is not None:
            h = h * node_mask
        return h, mij


class EquivariantUpdate(nn.Module):
    def __init__(
        self,
        hidden_nf,
        normalization_factor,
        aggregation_method,
        edges_in_d=1,
        act_fn=nn.SiLU(),
        tanh=False,
        coords_range=10.0,
        reflection_equiv=True,
    ):
        super(EquivariantUpdate, self).__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        self.reflection_equiv = reflection_equiv
        input_edge = hidden_nf * 2 + edges_in_d
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf), act_fn, nn.Linear(hidden_nf, hidden_nf), act_fn, layer
        )
        self.cross_product_mlp = (
            nn.Sequential(
                nn.Linear(input_edge, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, hidden_nf),
                act_fn,
                layer,
            )
            if not self.reflection_equiv
            else None
        )
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

    def coord_model(
        self,
        h,
        coord,
        edge_index,
        coord_diff,
        coord_cross,
        edge_attr,
        edge_mask,
        update_coords_mask=None,
    ):
        row, col = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
        if self.tanh:
            trans = coord_diff * torch.tanh(self.coord_mlp(input_tensor)) * self.coords_range
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)

        if not self.reflection_equiv:
            phi_cross = self.cross_product_mlp(input_tensor)
            if self.tanh:
                phi_cross = torch.tanh(phi_cross) * self.coords_range
            trans = trans + coord_cross * phi_cross

        if edge_mask is not None:
            trans = trans * edge_mask

        agg = unsorted_segment_sum(
            trans,
            row,
            num_segments=coord.size(0),
            normalization_factor=self.normalization_factor,
            aggregation_method=self.aggregation_method,
        )

        if update_coords_mask is not None:
            agg = update_coords_mask * agg

        coord = coord + agg
        return coord

    def forward(
        self,
        h,
        coord,
        edge_index,
        coord_diff,
        coord_cross,
        edge_attr=None,
        node_mask=None,
        edge_mask=None,
        update_coords_mask=None,
    ):
        coord = self.coord_model(
            h,
            coord,
            edge_index,
            coord_diff,
            coord_cross,
            edge_attr,
            edge_mask,
            update_coords_mask=update_coords_mask,
        )
        if node_mask is not None:
            coord = coord * node_mask
        return coord


class EquivariantBlock(nn.Module):
    def __init__(
        self,
        hidden_nf,
        edge_feat_nf=2,
        device="cpu",
        act_fn=nn.SiLU(),
        n_layers=2,
        attention=True,
        norm_diff=True,
        tanh=False,
        coords_range=15,
        norm_constant=1,
        sin_embedding=None,
        normalization_factor=100,
        aggregation_method="sum",
        reflection_equiv=True,
    ):
        super(EquivariantBlock, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.reflection_equiv = reflection_equiv

        for i in range(0, n_layers):
            self.add_module(
                "gcl_%d" % i,
                GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    edges_in_d=edge_feat_nf,
                    act_fn=act_fn,
                    attention=attention,
                    normalization_factor=self.normalization_factor,
                    aggregation_method=self.aggregation_method,
                ),
            )
        self.add_module(
            "gcl_equiv",
            EquivariantUpdate(
                hidden_nf,
                edges_in_d=edge_feat_nf,
                act_fn=nn.SiLU(),
                tanh=tanh,
                coords_range=self.coords_range_layer,
                normalization_factor=self.normalization_factor,
                aggregation_method=self.aggregation_method,
                reflection_equiv=self.reflection_equiv,
            ),
        )
        self.to(self.device)

    def forward(
        self,
        h,
        x,
        edge_index,
        node_mask=None,
        edge_mask=None,
        edge_attr=None,
        update_coords_mask=None,
        batch_mask=None,
    ):
        # Edit Emiel: Remove velocity as input
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        if self.reflection_equiv:
            coord_cross = None
        else:
            coord_cross = coord2cross(x, edge_index, batch_mask, self.norm_constant)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        edge_attr = torch.cat([distances, edge_attr], dim=1)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](
                h, edge_index, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask
            )
        x = self._modules["gcl_equiv"](
            h,
            x,
            edge_index,
            coord_diff,
            coord_cross,
            edge_attr,
            node_mask,
            edge_mask,
            update_coords_mask=update_coords_mask,
        )

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h, x


class EGNN(nn.Module):
    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        device="cpu",
        act_fn=nn.SiLU(),
        n_layers=3,
        attention=False,
        norm_diff=True,
        out_node_nf=None,
        tanh=False,
        coords_range=15,
        norm_constant=1,
        inv_sublayers=2,
        sin_embedding=False,
        normalization_factor=100,
        aggregation_method="sum",
        reflection_equiv=True,
    ):
        super(EGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range / n_layers)
        self.norm_diff = norm_diff
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.reflection_equiv = reflection_equiv

        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_feat_nf = 2

        edge_feat_nf = edge_feat_nf + in_edge_nf

        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module(
                "e_block_%d" % i,
                EquivariantBlock(
                    hidden_nf,
                    edge_feat_nf=edge_feat_nf,
                    device=device,
                    act_fn=act_fn,
                    n_layers=inv_sublayers,
                    attention=attention,
                    norm_diff=norm_diff,
                    tanh=tanh,
                    coords_range=coords_range,
                    norm_constant=norm_constant,
                    sin_embedding=self.sin_embedding,
                    normalization_factor=self.normalization_factor,
                    aggregation_method=self.aggregation_method,
                    reflection_equiv=self.reflection_equiv,
                ),
            )
        self.to(self.device)

    def forward(
        self,
        h,
        x,
        edge_index,
        node_mask=None,
        edge_mask=None,
        update_coords_mask=None,
        batch_mask=None,
        edge_attr=None,
    ):
        # Edit Emiel: Remove velocity as input
        edge_feat, _ = coord2diff(x, edge_index)
        if self.sin_embedding is not None:
            edge_feat = self.sin_embedding(edge_feat)
        if edge_attr is not None:
            edge_feat = torch.cat([edge_feat, edge_attr], dim=1)
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, x = self._modules["e_block_%d" % i](
                h,
                x,
                edge_index,
                node_mask=node_mask,
                edge_mask=edge_mask,
                edge_attr=edge_feat,
                update_coords_mask=update_coords_mask,
                batch_mask=batch_mask,
            )

        # Important, the bias of the last linear might be non-zero
        h = self.embedding_out(h)
        if node_mask is not None:
            h = h * node_mask
        return h, x


class GNN(nn.Module):
    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        aggregation_method="sum",
        device="cpu",
        act_fn=nn.SiLU(),
        n_layers=4,
        attention=False,
        normalization_factor=1,
        out_node_nf=None,
    ):
        super(GNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        # Encoder
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module(
                "gcl_%d" % i,
                GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    normalization_factor=normalization_factor,
                    aggregation_method=aggregation_method,
                    edges_in_d=in_edge_nf,
                    act_fn=act_fn,
                    attention=attention,
                ),
            )
        self.to(self.device)

    def forward(self, h, edges, edge_attr=None, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](
                h, edges, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask
            )
        h = self.embedding_out(h)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h


class SinusoidsEmbeddingNew(nn.Module):
    def __init__(self, max_res=15.0, min_res=15.0 / 2000.0, div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = 2 * math.pi * div_factor ** torch.arange(self.n_frequencies) / max_res
        self.dim = len(self.frequencies) * 2

    def forward(self, x):
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()


def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff / (norm + norm_constant)
    return radial, coord_diff


def coord2cross(x, edge_index, batch_mask, norm_constant=1):
    mean = unsorted_segment_sum(
        x,
        batch_mask,
        num_segments=batch_mask.max() + 1,
        normalization_factor=None,
        aggregation_method="mean",
    )
    row, col = edge_index
    cross = torch.cross(x[row] - mean[batch_mask[row]], x[col] - mean[batch_mask[col]], dim=1)
    norm = torch.linalg.norm(cross, dim=1, keepdim=True)
    cross = cross / (norm + norm_constant)
    return cross


def unsorted_segment_sum(
    data, segment_ids, num_segments, normalization_factor, aggregation_method: str
):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
    Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == "sum":
        result = result / normalization_factor

    if aggregation_method == "mean":
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result


# --- inlined from equivariant_diffusion/en_diffusion.py (EnVariationalDiffusion.remove_mean_batch) ---
# (a static method pulled out as a free function to avoid the module's unrelated
# rdkit/Bio.PDB-dependent `import utils` transitive dependency; body is verbatim)


def remove_mean_batch(x, indices):
    mean = scatter_mean(x, indices, dim=0)
    x = x - mean[indices]
    return x


# --- verbatim from equivariant_diffusion/dynamics.py (EGNNDynamics) ------------


class EGNNDynamics(nn.Module):
    def __init__(
        self,
        atom_nf,
        residue_nf,
        n_dims,
        joint_nf=16,
        hidden_nf=64,
        device="cpu",
        act_fn=torch.nn.SiLU(),
        n_layers=4,
        attention=False,
        condition_time=True,
        tanh=False,
        mode="egnn_dynamics",
        norm_constant=0,
        inv_sublayers=2,
        sin_embedding=False,
        normalization_factor=100,
        aggregation_method="sum",
        update_pocket_coords=True,
        edge_cutoff_ligand=None,
        edge_cutoff_pocket=None,
        edge_cutoff_interaction=None,
        reflection_equivariant=True,
        edge_embedding_dim=None,
    ):
        super().__init__()
        self.mode = mode
        self.edge_cutoff_l = edge_cutoff_ligand
        self.edge_cutoff_p = edge_cutoff_pocket
        self.edge_cutoff_i = edge_cutoff_interaction
        self.edge_nf = edge_embedding_dim

        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_nf, 2 * atom_nf), act_fn, nn.Linear(2 * atom_nf, joint_nf)
        )

        self.atom_decoder = nn.Sequential(
            nn.Linear(joint_nf, 2 * atom_nf), act_fn, nn.Linear(2 * atom_nf, atom_nf)
        )

        self.residue_encoder = nn.Sequential(
            nn.Linear(residue_nf, 2 * residue_nf), act_fn, nn.Linear(2 * residue_nf, joint_nf)
        )

        self.residue_decoder = nn.Sequential(
            nn.Linear(joint_nf, 2 * residue_nf), act_fn, nn.Linear(2 * residue_nf, residue_nf)
        )

        self.edge_embedding = nn.Embedding(3, self.edge_nf) if self.edge_nf is not None else None
        self.edge_nf = 0 if self.edge_nf is None else self.edge_nf

        if condition_time:
            dynamics_node_nf = joint_nf + 1
        else:
            print("Warning: dynamics model is _not_ conditioned on time.")
            dynamics_node_nf = joint_nf

        if mode == "egnn_dynamics":
            self.egnn = EGNN(
                in_node_nf=dynamics_node_nf,
                in_edge_nf=self.edge_nf,
                hidden_nf=hidden_nf,
                device=device,
                act_fn=act_fn,
                n_layers=n_layers,
                attention=attention,
                tanh=tanh,
                norm_constant=norm_constant,
                inv_sublayers=inv_sublayers,
                sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                reflection_equiv=reflection_equivariant,
            )
            self.node_nf = dynamics_node_nf
            self.update_pocket_coords = update_pocket_coords

        elif mode == "gnn_dynamics":
            self.gnn = GNN(
                in_node_nf=dynamics_node_nf + n_dims,
                in_edge_nf=self.edge_nf,
                hidden_nf=hidden_nf,
                out_node_nf=n_dims + dynamics_node_nf,
                device=device,
                act_fn=act_fn,
                n_layers=n_layers,
                attention=attention,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
            )

        self.device = device
        self.n_dims = n_dims
        self.condition_time = condition_time

    def forward(self, xh_atoms, xh_residues, t, mask_atoms, mask_residues):
        x_atoms = xh_atoms[:, : self.n_dims].clone()
        h_atoms = xh_atoms[:, self.n_dims :].clone()

        x_residues = xh_residues[:, : self.n_dims].clone()
        h_residues = xh_residues[:, self.n_dims :].clone()

        # embed atom features and residue features in a shared space
        h_atoms = self.atom_encoder(h_atoms)
        h_residues = self.residue_encoder(h_residues)

        # combine the two node types
        x = torch.cat((x_atoms, x_residues), dim=0)
        h = torch.cat((h_atoms, h_residues), dim=0)
        mask = torch.cat([mask_atoms, mask_residues])

        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t[mask]
            h = torch.cat([h, h_time], dim=1)

        # get edges of a complete graph
        edges = self.get_edges(mask_atoms, mask_residues, x_atoms, x_residues)
        assert torch.all(mask[edges[0]] == mask[edges[1]])

        # Get edge types
        if self.edge_nf > 0:
            # 0: ligand-pocket, 1: ligand-ligand, 2: pocket-pocket
            edge_types = torch.zeros(edges.size(1), dtype=int, device=edges.device)
            edge_types[(edges[0] < len(mask_atoms)) & (edges[1] < len(mask_atoms))] = 1
            edge_types[(edges[0] >= len(mask_atoms)) & (edges[1] >= len(mask_atoms))] = 2

            # Learnable embedding
            edge_types = self.edge_embedding(edge_types)
        else:
            edge_types = None

        if self.mode == "egnn_dynamics":
            update_coords_mask = (
                None
                if self.update_pocket_coords
                else torch.cat(
                    (torch.ones_like(mask_atoms), torch.zeros_like(mask_residues))
                ).unsqueeze(1)
            )
            h_final, x_final = self.egnn(
                h,
                x,
                edges,
                update_coords_mask=update_coords_mask,
                batch_mask=mask,
                edge_attr=edge_types,
            )
            vel = x_final - x

        elif self.mode == "gnn_dynamics":
            xh = torch.cat([x, h], dim=1)
            output = self.gnn(xh, edges, node_mask=None, edge_attr=edge_types)
            vel = output[:, :3]
            h_final = output[:, 3:]

        else:
            raise Exception("Wrong mode %s" % self.mode)

        if self.condition_time:
            # Slice off last dimension which represented time.
            h_final = h_final[:, :-1]

        # decode atom and residue features
        h_final_atoms = self.atom_decoder(h_final[: len(mask_atoms)])
        h_final_residues = self.residue_decoder(h_final[len(mask_atoms) :])

        if torch.any(torch.isnan(vel)):
            if self.training:
                vel[torch.isnan(vel)] = 0.0
            else:
                raise ValueError("NaN detected in EGNN output")

        if self.update_pocket_coords:
            # in case of unconditional joint distribution, include this as in
            # the original code
            vel = remove_mean_batch(vel, mask)

        return torch.cat([vel[: len(mask_atoms)], h_final_atoms], dim=-1), torch.cat(
            [vel[len(mask_atoms) :], h_final_residues], dim=-1
        )

    def get_edges(self, batch_mask_ligand, batch_mask_pocket, x_ligand, x_pocket):
        adj_ligand = batch_mask_ligand[:, None] == batch_mask_ligand[None, :]
        adj_pocket = batch_mask_pocket[:, None] == batch_mask_pocket[None, :]
        adj_cross = batch_mask_ligand[:, None] == batch_mask_pocket[None, :]

        if self.edge_cutoff_l is not None:
            adj_ligand = adj_ligand & (torch.cdist(x_ligand, x_ligand) <= self.edge_cutoff_l)

        if self.edge_cutoff_p is not None:
            adj_pocket = adj_pocket & (torch.cdist(x_pocket, x_pocket) <= self.edge_cutoff_p)

        if self.edge_cutoff_i is not None:
            adj_cross = adj_cross & (torch.cdist(x_ligand, x_pocket) <= self.edge_cutoff_i)

        adj = torch.cat(
            (
                torch.cat((adj_ligand, adj_cross), dim=1),
                torch.cat((adj_cross.T, adj_pocket), dim=1),
            ),
            dim=0,
        )
        edges = torch.stack(torch.where(adj), dim=0)

        return edges


# --- menagerie staging wrapper --------------------------------------------------


def build_diffsbdd_egnn_dynamics():
    model = EGNNDynamics(
        atom_nf=8,
        residue_nf=20,
        n_dims=3,
        joint_nf=16,
        hidden_nf=32,
        n_layers=2,
        attention=False,
        mode="egnn_dynamics",
    )
    model.eval()
    return model


def example_input_diffsbdd_egnn_dynamics():
    n_atoms, n_residues = 6, 5
    atom_nf, residue_nf, n_dims = 8, 20, 3

    xh_atoms = torch.randn(n_atoms, n_dims + atom_nf)
    xh_residues = torch.randn(n_residues, n_dims + residue_nf)
    t = torch.tensor([0.5])
    mask_atoms = torch.zeros(n_atoms, dtype=torch.long)
    mask_residues = torch.zeros(n_residues, dtype=torch.long)

    return (xh_atoms, xh_residues, t, mask_atoms, mask_residues)


MENAGERIE_ENTRIES = [
    (
        "DiffSBDD (EGNN dynamics)",
        "build_diffsbdd_egnn_dynamics",
        "example_input_diffsbdd_egnn_dynamics",
        2023,
        "vendored-pytorch",
    ),
]
