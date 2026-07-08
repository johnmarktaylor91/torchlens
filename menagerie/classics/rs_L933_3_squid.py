# SOURCE: vendored from https://github.com/keiradams/SQUID @ main
# Files: models/models.py (ROCS_Model_Point_Cloud), models/encoder.py (Encoder_point_cloud),
# models/fragment_encoder.py (FragmentLibraryEncoder), models/EGNN.py (EGNN_MLP, EGNN_static,
# unsorted_segment_sum, unsorted_segment_mean), models/egnn_vn_point_cloud.py
# (EGNN_VN_Encoder_point_cloud), models/vnn/models/vn_layers.py (VNLinear, VNLeakyReLU,
# VNLinearLeakyReLU, VNLinearAndLeakyReLU, VNStdFeature, mean_pool), and
# models/vnn/models/utils/vn_dgcnn_util.py (knn, get_graph_feature) -- all architecture
# classes copied verbatim; only import plumbing (relative-package imports collapsed into
# this single file) was changed.
#
# SQUID (Adams & Coley 2023, "Equivariant Shape-Conditioned Generation of 3D Molecules for
# Ligand-Based Drug Design") jointly encodes (1) a 3D point cloud of a molecular shape via a
# stack of Vector-Neuron (VN) equivariant point convolutions (get_graph_feature/VN* layers,
# ported from Deng et al.'s Vector Neurons) and (2) the molecular graph via an E(n)-equivariant
# GNN (EGNN_static, Satorras et al.), pooling per-fragment "fragment library" embeddings into
# both. This vendors the ROCS_Model_Point_Cloud head (the shape-comparison scorer), which
# exercises the full encoder stack (fragment library encoder -> graph EGNN encoder -> subgraph
# EGNN encoder -> VN point-cloud shape encoder -> ROCS scorer MLP) without requiring the
# decoder's autoregressive fragment-attachment sampling loop.
#
# Real training data for this model is built by a bespoke PyG `Collater`/`PairData` pipeline
# (utils/graph_generator_datasets_and_loaders.py) driven by RDKit conformer generation, which
# is a data-preparation concern outside the architecture. For tracing we hand-build a minimal
# but shape-faithful synthetic batch: a small molecular graph (nodes/edges/3D coords), a
# "subgraph" (partial molecule) of the same schema, a per-graph point cloud (fixed points per
# cloud, reshaped batch-major exactly as encoder_point_cloud.forward expects), and a tiny
# fragment library graph batch consumed by FragmentLibraryEncoder -- all wired with the same
# index-alignment conventions used in the real forward() (x_library_fragment_index indexes into
# the fragment library's per-fragment embedding for every graph node, points_atom_index maps
# each point in the cloud to its owning graph node).

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter

EPS = 1e-6


# ============================== models/EGNN.py ==============================
class EGNN_MLP(nn.Module):
    """a simple 4-layer MLP"""

    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nout),
        )

    def forward(self, x):
        return self.net(x)


def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


class EGNN_static(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph."""

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        edges_in_d=0,
        act_fn=None,
        coords_weight=0.0,
        residual=True,
        no_3D=False,
        old_EGNN=False,
    ):
        super().__init__()
        if act_fn is None:
            act_fn = nn.LeakyReLU(0.2)
        self.no_3D = no_3D

        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        edge_coords_nf = 1

        self.residual = residual
        self.norm_diff = False

        if old_EGNN:
            self.edge_mlp = nn.Sequential(
                nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, hidden_nf),
                act_fn,
            )
            self.node_mlp = nn.Sequential(
                nn.Linear(input_nf + hidden_nf, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, output_nf),
            )
        else:
            self.edge_mlp = nn.Sequential(
                nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, output_nf),
                act_fn,
            )
            self.node_mlp = nn.Sequential(
                nn.Linear(input_nf + output_nf, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, output_nf),
            )

        if coords_weight > 0.0:
            layer = nn.Linear(hidden_nf, 1, bias=False)
            torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
            coord_mlp = [nn.Linear(hidden_nf, hidden_nf), act_fn, layer]
            self.coord_mlp = nn.Sequential(*coord_mlp)

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        trans = torch.clamp(trans, min=-100, max=100)
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        coord = coord + agg * self.coords_weight
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff / (norm)
        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        if self.no_3D:
            edge_feat = self.edge_model(h[row], h[col], radial * 0.0, edge_attr)
        else:
            edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
            if self.coords_weight > 0.0:
                coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)

        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_feat


# ================= models/vnn/models/utils/vn_dgcnn_util.py =================
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None, x_coord=None, device=torch.device("cpu")):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if x_coord is None:
            idx = knn(x, k=k)
        else:
            idx = knn(x_coord, k=k)

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3)
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 4, 1, 2).contiguous()

    return feature


# ===================== models/vnn/models/vn_layers.py =====================
class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        x_out = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)
        return x_out


class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.2):
        super().__init__()
        if share_nonlinearity:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x):
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdim=True)
        x_out = self.negative_slope * x + (1 - self.negative_slope) * (
            mask * x + (1 - mask) * (x - (dotprod / (d_norm_sq + EPS)) * d)
        )
        return x_out


class VNLinearLeakyReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dim=5,
        share_nonlinearity=False,
        negative_slope=0.2,
        use_batchnorm=False,
    ):
        super().__init__()
        self.dim = dim
        self.negative_slope = negative_slope

        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.batchnorm = VNBatchNorm(out_channels, dim=dim)

        if share_nonlinearity:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        p = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)

        if self.use_batchnorm:
            p = self.batchnorm(p)

        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (p * d).sum(2, keepdims=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdims=True)
        x_out = self.negative_slope * p + (1 - self.negative_slope) * (
            mask * p + (1 - mask) * (p - (dotprod / (d_norm_sq + EPS)) * d)
        )
        return x_out


class VNLinearAndLeakyReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dim=5,
        share_nonlinearity=False,
        use_batchnorm=False,
        negative_slope=0.2,
    ):
        super().__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope

        self.linear = VNLinear(in_channels, out_channels)
        self.leaky_relu = VNLeakyReLU(
            out_channels, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope
        )

        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.batchnorm = VNBatchNorm(out_channels, dim=dim)

    def forward(self, x):
        x = self.linear(x)
        if self.use_batchnorm:
            x = self.batchnorm(x)
        x_out = self.leaky_relu(x)
        return x_out


class VNBatchNorm(nn.Module):
    def __init__(self, num_features, dim):
        super().__init__()
        self.dim = dim
        if dim in (3, 4):
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        norm = torch.norm(x, dim=2) + EPS
        norm_bn = self.bn(norm)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        x = x / norm * norm_bn
        return x


def mean_pool(x, dim=-1, keepdim=False):
    return x.mean(dim=dim, keepdim=keepdim)


class VNStdFeature(nn.Module):
    def __init__(
        self,
        in_channels,
        dim=4,
        normalize_frame=False,
        share_nonlinearity=False,
        negative_slope=0.2,
    ):
        super().__init__()
        self.dim = dim
        self.normalize_frame = normalize_frame

        self.vn1 = VNLinearLeakyReLU(
            in_channels,
            in_channels // 2,
            dim=dim,
            share_nonlinearity=share_nonlinearity,
            negative_slope=negative_slope,
        )
        self.vn2 = VNLinearLeakyReLU(
            in_channels // 2,
            in_channels // 4,
            dim=dim,
            share_nonlinearity=share_nonlinearity,
            negative_slope=negative_slope,
        )
        if normalize_frame:
            self.vn_lin = nn.Linear(in_channels // 4, 2, bias=False)
        else:
            self.vn_lin = nn.Linear(in_channels // 4, 3, bias=False)

    def forward(self, x):
        z0 = x
        z0 = self.vn1(z0)
        z0 = self.vn2(z0)
        z0 = self.vn_lin(z0.transpose(1, -1)).transpose(1, -1)

        if self.normalize_frame:
            v1 = z0[:, 0, :]
            v1_norm = torch.sqrt((v1 * v1).sum(1, keepdims=True))
            u1 = v1 / (v1_norm + EPS)
            v2 = z0[:, 1, :]
            v2 = v2 - (v2 * u1).sum(1, keepdims=True) * u1
            v2_norm = torch.sqrt((v2 * v2).sum(1, keepdims=True))
            u2 = v2 / (v2_norm + EPS)
            u3 = torch.cross(u1, u2)
            z0 = torch.stack([u1, u2, u3], dim=1).transpose(1, 2)
        else:
            z0 = z0.transpose(1, 2)

        if self.dim == 4:
            x_std = torch.einsum("bijm,bjkm->bikm", x, z0)
        elif self.dim == 3:
            x_std = torch.einsum("bij,bjk->bik", x, z0)
        elif self.dim == 5:
            x_std = torch.einsum("bijmn,bjkmn->bikmn", x, z0)

        return x_std, z0


# ========================= models/fragment_encoder.py =========================
class FragmentLibraryEncoder(nn.Module):
    def __init__(
        self,
        input_nf=45,
        edges_in_d=5,
        output_dim=64,
        N_layers=2,
        append_noise=False,
        N_members=72,
        no_3D=False,
        old_EGNN=False,
    ):
        super().__init__()

        self.no_3D = no_3D
        self.output_dim = output_dim
        self.N_layers = N_layers
        self.append_noise = append_noise
        self.N_members = N_members

        self.EGNN_layers = nn.ModuleList(
            [
                EGNN_static(
                    input_nf=input_nf,
                    output_nf=output_dim,
                    hidden_nf=output_dim,
                    edges_in_d=edges_in_d,
                    residual=False,
                    no_3D=no_3D,
                    old_EGNN=old_EGNN,
                )
            ]
        )

        for layer in range(1, N_layers):
            self.EGNN_layers.append(
                EGNN_static(
                    input_nf=output_dim,
                    output_nf=output_dim
                    if ((layer < (N_layers - 1)) | (self.append_noise is False))
                    else output_dim // 2,
                    hidden_nf=output_dim,
                    edges_in_d=output_dim,
                    residual=True
                    if ((layer < (N_layers - 1)) | (self.append_noise is False))
                    else False,
                    no_3D=no_3D,
                    old_EGNN=old_EGNN,
                )
            )

        if self.append_noise:
            self.noise_embedding = torch.nn.Embedding(N_members, output_dim - (output_dim // 2))

    def forward(self, x, edge_index, pos, edge_attr, batch_index, device=torch.device("cpu")):
        h, _, edge_feat = self.EGNN_layers[0](
            x, edge_index, pos, edge_attr=edge_attr, node_attr=None
        )
        for EGNN_layer in self.EGNN_layers[1:]:
            h, _, edge_feat = EGNN_layer(h, edge_index, pos, edge_attr=edge_feat, node_attr=None)

        graph_features = torch_scatter.scatter_add(h, batch_index, dim=0)

        if self.append_noise:
            graph_features = torch.cat(
                [
                    graph_features,
                    self.noise_embedding(torch.arange(0, self.N_members, device=device)),
                ],
                dim=1,
            )
            h = torch.cat([h, self.noise_embedding(batch_index)], dim=1)

        return graph_features, h, batch_index


# ======================= models/egnn_vn_point_cloud.py =======================
class EGNN_VN_Encoder_point_cloud(nn.Module):
    def __init__(
        self,
        node_input_dim=(45 + 64),
        edges_in_d=5,
        num_components=64,
        EGNN_layer_dim=64,
        n_knn=10,
        conv_dims=None,
        pooling_MLP=True,
        N_EGNN_layers=3,
        variational_GNN=False,
        variational_GNN_mol=False,
        mix_node_inv_to_equi=False,
        mix_shape_to_nodes=False,
        ablate_HvarCat=False,
        old_EGNN=False,
    ):
        super().__init__()
        if conv_dims is None:
            conv_dims = [64, 64, 128, 256]

        self.N_EGNN_layers = N_EGNN_layers
        self.pooling = "mean"
        self.n_knn = n_knn
        self.num_components = num_components
        self.pooling_MLP = pooling_MLP
        self.conv_dims = conv_dims

        self.EGNN_layer_dim = EGNN_layer_dim
        self.point_invariant_mlp_hidden_dim = EGNN_layer_dim
        self.h_dim = EGNN_layer_dim
        self.pooling_MLP_dim = EGNN_layer_dim

        self.ablate_HvarCat = ablate_HvarCat
        self.mix_shape_to_nodes = mix_shape_to_nodes
        if self.mix_shape_to_nodes:
            self.std_feature_mix_shape_to_nodes = VNStdFeature(
                self.num_components * 2, dim=4, normalize_frame=False
            )

        self.EGNN_layers = nn.ModuleList(
            [
                EGNN_static(
                    input_nf=node_input_dim
                    + int(self.mix_shape_to_nodes) * self.num_components * 2 * 3,
                    output_nf=EGNN_layer_dim,
                    hidden_nf=EGNN_layer_dim + int(self.mix_shape_to_nodes) * EGNN_layer_dim,
                    edges_in_d=edges_in_d,
                    residual=False,
                    old_EGNN=old_EGNN,
                )
            ]
        )

        for layer in range(1, N_EGNN_layers):
            self.EGNN_layers.append(
                EGNN_static(
                    input_nf=EGNN_layer_dim,
                    output_nf=EGNN_layer_dim,
                    hidden_nf=EGNN_layer_dim,
                    edges_in_d=EGNN_layer_dim,
                    residual=True,
                    old_EGNN=old_EGNN,
                )
            )

        self.variational_GNN = variational_GNN
        if self.variational_GNN:
            self.variational_GNN_encoder = nn.Sequential(
                nn.Linear(EGNN_layer_dim, EGNN_layer_dim),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(EGNN_layer_dim, EGNN_layer_dim * 2),
            )

        self.variational_GNN_mol = variational_GNN_mol
        if self.variational_GNN_mol:
            self.variational_GNN_mol_encoder = nn.Sequential(
                nn.Linear(EGNN_layer_dim, EGNN_layer_dim),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(EGNN_layer_dim, EGNN_layer_dim),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(EGNN_layer_dim, EGNN_layer_dim * 2),
            )
            self.h_predictor = nn.Sequential(
                nn.Linear(self.num_components * 2 * 3 * 2 + EGNN_layer_dim, EGNN_layer_dim * 4),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(EGNN_layer_dim * 4, EGNN_layer_dim * 2),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(EGNN_layer_dim * 2, EGNN_layer_dim),
            )

        self.mix_node_inv_to_equi = mix_node_inv_to_equi
        if self.mix_node_inv_to_equi:
            self.project_h_embeddings = nn.Sequential(
                nn.Linear(EGNN_layer_dim, num_components),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(num_components, num_components * num_components // 2),
            )
            self.Equi_linear_leaky_mixing = VNLinearAndLeakyReLU(
                num_components + num_components // 2,
                num_components,
                use_batchnorm=False,
                negative_slope=0.2,
            )

        self.conv1 = VNLinearLeakyReLU(2, self.conv_dims[0] // 3)
        self.conv2 = VNLinearLeakyReLU(self.conv_dims[0] // 3 * 2, self.conv_dims[1] // 3)
        self.conv3 = VNLinearLeakyReLU(self.conv_dims[1] // 3 * 2, self.conv_dims[2] // 3)
        self.conv4 = VNLinearLeakyReLU(self.conv_dims[2] // 3 * 2, self.conv_dims[3] // 3)
        self.conv5 = VNLinearLeakyReLU(
            self.conv_dims[3] // 3
            + self.conv_dims[2] // 3
            + self.conv_dims[1] // 3
            + self.conv_dims[0] // 3,
            self.num_components,
            dim=4,
            share_nonlinearity=True,
        )

        self.pool1 = mean_pool
        self.pool2 = mean_pool
        self.pool3 = mean_pool
        self.pool4 = mean_pool

        self.std_feature = VNStdFeature(self.num_components * 2, dim=4, normalize_frame=False)

        self.point_invariant_mlp = nn.Sequential(
            nn.Linear(
                self.num_components * 2 * 3 + self.h_dim, self.point_invariant_mlp_hidden_dim * 2
            ),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.point_invariant_mlp_hidden_dim * 2, self.point_invariant_mlp_hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.point_invariant_mlp_hidden_dim, self.point_invariant_mlp_hidden_dim),
        )

        if self.pooling_MLP:
            self.mlp = nn.Sequential(
                nn.Linear(self.point_invariant_mlp_hidden_dim, self.pooling_MLP_dim),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(self.pooling_MLP_dim, self.pooling_MLP_dim),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(self.pooling_MLP_dim, self.pooling_MLP_dim),
            )

    def forward(
        self,
        h,
        edge_index,
        pos,
        edge_attr,
        batch_size,
        points,
        points_atom_index,
        select_indices=None,
        select_indices_batch=None,
        device=torch.device("cpu"),
        use_variational_GNN=False,
        variational_GNN_factor=1.0,
        interpolate_to_GNN_prior=0.0,
        h_interpolate=None,
    ):
        pos_reshaped = (pos.reshape(batch_size, -1, 3)).permute(0, 2, 1)  # noqa: F841 (unused in original repo code too)
        points_reshaped = (points.reshape(batch_size, -1, 3)).permute(0, 2, 1)
        points_atom_index_reshape = points_atom_index.reshape(batch_size, -1)

        batch_size = points_reshaped.size(0)

        x = points_reshaped.unsqueeze(1)
        x = get_graph_feature(x, k=self.n_knn, device=device)
        x = self.conv1(x)
        x1 = self.pool1(x)

        x = get_graph_feature(x1, k=self.n_knn, device=device)
        x = self.conv2(x)
        x2 = self.pool2(x)

        x = get_graph_feature(x2, k=self.n_knn, device=device)
        x = self.conv3(x)
        x3 = self.pool3(x)

        x = get_graph_feature(x3, k=self.n_knn, device=device)
        x = self.conv4(x)
        x4 = self.pool4(x)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)  # x is now shape [batch, num_components, 3, N_points_per_cloud]

        x_atom_pooled = torch_scatter.scatter_mean(
            x, points_atom_index_reshape.unsqueeze(1).unsqueeze(1)
        )

        if self.mix_shape_to_nodes:
            num_nodes = x_atom_pooled.size(-1)
            x_mean_expanded = x_atom_pooled.sum(dim=-1, keepdim=True).expand(x_atom_pooled.size())
            x_gnn_invariant, _ = self.std_feature_mix_shape_to_nodes(
                torch.cat((x_atom_pooled, x_mean_expanded), dim=1)
            )
            x_gnn_invariant_reshaped = x_gnn_invariant.reshape(batch_size, -1, num_nodes)
            x_gnn_to_cat_to_h = x_gnn_invariant_reshaped.permute(0, 2, 1).reshape(
                batch_size * num_nodes, -1
            )
            h = torch.cat((h, x_gnn_to_cat_to_h), dim=1)

        h, _, edge_feat = self.EGNN_layers[0](
            h, edge_index, pos, edge_attr=edge_attr, node_attr=None
        )
        for EGNN_layer in self.EGNN_layers[1:]:
            h, _, edge_feat = EGNN_layer(h, edge_index, pos, edge_attr=edge_feat, node_attr=None)

        if (self.variational_GNN) & (use_variational_GNN) & (self.variational_GNN_mol is False):
            h_variational = self.variational_GNN_encoder(h)
            h_mean, h_logvar = h_variational.chunk(2, dim=1)
            h_std = torch.exp(0.5 * h_logvar)
            h_eps = torch.randn_like(h_mean) * variational_GNN_factor

            if interpolate_to_GNN_prior > 1e-4:
                h_mean = torch.lerp(h_mean, torch.zeros_like(h_mean), interpolate_to_GNN_prior)
                h_std = torch.lerp(h_std, torch.ones_like(h_std), interpolate_to_GNN_prior)

            h = h_mean + h_std * h_eps
        else:
            h_mean = None
            h_std = None

        if h_interpolate is not None:
            h = h_interpolate

        h_reshaped = h.unsqueeze(0).reshape(batch_size, -1, h.shape[1]).permute(0, 2, 1)

        if self.mix_node_inv_to_equi:
            h_projected = self.project_h_embeddings(h_reshaped.permute(0, 2, 1))
            h_projected_reshaped = h_projected.reshape(
                h_projected.shape[0],
                h_projected.shape[1],
                self.num_components // 2,
                self.num_components,
            )
            x_atom_pooled_mixed = torch.einsum(
                "bijk,bikm->bijm", h_projected_reshaped, x_atom_pooled.permute(0, 3, 1, 2)
            )
            x_atom_pooled = self.Equi_linear_leaky_mixing(
                torch.cat((x_atom_pooled.permute(0, 3, 1, 2), x_atom_pooled_mixed), dim=2).permute(
                    0, 2, 3, 1
                )
            )

        num_nodes = x_atom_pooled.size(-1)

        Z_equivariant = x_atom_pooled.sum(dim=-1, keepdim=False)

        if select_indices is not None:
            Z_equivariant_select = torch_scatter.scatter_add(
                x_atom_pooled.permute(0, 3, 1, 2).reshape(-1, self.num_components, 3)[
                    select_indices
                ],
                select_indices_batch,
                dim=0,
            )

        x_mean_expanded = x_atom_pooled.sum(dim=-1, keepdim=True).expand(x_atom_pooled.size())
        x_cat_x_mean = torch.cat((x_atom_pooled, x_mean_expanded), 1)
        x_invariant, trans = self.std_feature(x_cat_x_mean)

        x_invariant = x_invariant.reshape(batch_size, -1, num_nodes)

        if (self.variational_GNN_mol) & (use_variational_GNN) & (self.variational_GNN is False):
            h_mol = h_reshaped.sum(-1)

            h_mol_mean_logvar = self.variational_GNN_mol_encoder(h_mol)
            h_mol_mean, h_mol_logvar = h_mol_mean_logvar.chunk(2, dim=1)
            h_mol_std = torch.exp(0.5 * h_mol_logvar)
            h_mol_eps = torch.randn_like(h_mol_mean) * variational_GNN_factor

            if interpolate_to_GNN_prior > 1e-4:
                h_mol_mean = torch.lerp(
                    h_mol_mean, torch.zeros_like(h_mol_mean), interpolate_to_GNN_prior
                )
                h_mol_std = torch.lerp(
                    h_mol_std, torch.ones_like(h_mol_std), interpolate_to_GNN_prior
                )

            h_mol = h_mol_mean + h_mol_std * h_mol_eps

            x_global = x_invariant.sum(-1)
            x_global_h_mol = torch.cat((x_global, h_mol), dim=1)
            x_invariant_x_global_h_mol_cat = torch.cat(
                (x_invariant, x_global_h_mol.unsqueeze(2).expand(-1, -1, x_invariant.shape[2])),
                dim=1,
            )
            h_reshaped_gnn = h_reshaped
            h_predicted_reshaped = self.h_predictor(
                x_invariant_x_global_h_mol_cat.permute(0, 2, 1)
            ).permute(0, 2, 1)
            h_reshaped = h_predicted_reshaped

            h_mean = h_mol_mean
            h_std = h_mol_std
        else:
            h_reshaped_gnn = None
            h_predicted_reshaped = None

        if self.ablate_HvarCat:
            x_invariant = torch.cat((x_invariant, torch.zeros_like(h_reshaped)), dim=1)
        else:
            x_invariant = torch.cat((x_invariant, h_reshaped), dim=1)

        x_invariant = self.point_invariant_mlp(x_invariant.permute(0, 2, 1)).permute(0, 2, 1)
        Z_invariant = x_invariant.sum(dim=-1, keepdim=False)

        if self.pooling_MLP:
            Z_invariant = self.mlp(Z_invariant)

        if select_indices is not None:
            Z_invariant_select = torch_scatter.scatter_add(
                x_invariant.permute(0, 2, 1).reshape(-1, x_invariant.shape[1])[select_indices],
                select_indices_batch,
                dim=0,
            )

            if self.pooling_MLP:
                Z_invariant_select = self.mlp(Z_invariant_select)

            return (
                x_invariant,
                Z_equivariant,
                Z_invariant,
                Z_equivariant_select,
                Z_invariant_select,
                h_mean,
                h_std,
                h_reshaped_gnn,
                h_predicted_reshaped,
                h_reshaped,
            )
        else:
            return (
                x_invariant,
                Z_equivariant,
                Z_invariant,
                None,
                None,
                h_mean,
                h_std,
                h_reshaped_gnn,
                h_predicted_reshaped,
                h_reshaped,
            )


# ============================= models/encoder.py =============================
class Encoder_point_cloud(nn.Module):
    def __init__(
        self,
        input_nf=45,
        edges_in_d=5,
        n_knn=10,
        conv_dims=None,
        num_components=64,
        fragment_library_dim=64,
        N_fragment_layers=2,
        append_noise=False,
        N_members=72,
        EGNN_layer_dim=64,
        N_EGNN_layers=3,
        pooling_MLP=True,
        shared_encoders=False,
        subtract_latent_space=False,
        variational=False,
        variational_mode="both",
        variational_GNN=False,
        variational_GNN_mol=False,
        mix_node_inv_to_equi=False,
        mix_shape_to_nodes=False,
        ablate_HvarCat=False,
        ablateEqui=False,
        old_EGNN=False,
    ):
        super().__init__()
        if conv_dims is None:
            conv_dims = [64, 64, 128, 256]

        self.input_nf = input_nf
        self.edges_in_d = edges_in_d

        self.conv_dims = conv_dims
        self.num_components = num_components
        self.fragment_library_dim = fragment_library_dim
        self.EGNN_layer_dim = EGNN_layer_dim
        self.N_EGNN_layers = N_EGNN_layers
        self.n_knn = n_knn
        self.pooling_MLP = pooling_MLP
        self.shared_encoders = shared_encoders
        self.subtract_latent_space = subtract_latent_space
        self.N_fragment_layers = N_fragment_layers
        self.N_members = N_members
        self.append_noise = append_noise
        self.variational = variational
        self.variational_mode = variational_mode
        self.variational_GNN = variational_GNN
        self.variational_GNN_mol = variational_GNN_mol
        self.mix_node_inv_to_equi = mix_node_inv_to_equi
        self.mix_shape_to_nodes = mix_shape_to_nodes
        self.ablate_HvarCat = ablate_HvarCat

        self.ablateEqui = ablateEqui

        self.fragment_encoder = FragmentLibraryEncoder(
            input_nf=input_nf,
            edges_in_d=edges_in_d,
            output_dim=fragment_library_dim,
            N_layers=N_fragment_layers,
            append_noise=append_noise,
            N_members=N_members,
            old_EGNN=old_EGNN,
        )

        self.GraphEncoder = EGNN_VN_Encoder_point_cloud(
            node_input_dim=input_nf + fragment_library_dim,
            edges_in_d=edges_in_d,
            num_components=num_components,
            EGNN_layer_dim=EGNN_layer_dim,
            n_knn=n_knn,
            conv_dims=conv_dims,
            pooling_MLP=pooling_MLP,
            N_EGNN_layers=N_EGNN_layers,
            variational_GNN=self.variational_GNN,
            variational_GNN_mol=self.variational_GNN_mol,
            mix_node_inv_to_equi=self.mix_node_inv_to_equi,
            mix_shape_to_nodes=self.mix_shape_to_nodes,
            ablate_HvarCat=self.ablate_HvarCat,
            old_EGNN=old_EGNN,
        )

        if variational:
            if (variational_mode == "both") | (variational_mode == "equi"):
                self.VariationalEncoder_equi = nn.Sequential(
                    VNLinearAndLeakyReLU(
                        num_components, num_components, use_batchnorm=False, negative_slope=0.2
                    ),
                    VNLinear(num_components, 2 * num_components),
                )
                self.VariationalEncoder_equi_T = VNStdFeature(
                    num_components, dim=3, normalize_frame=False
                )
                self.VariationEncoder_equi_linear = nn.Sequential(
                    nn.Linear(num_components * 3, num_components),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(num_components, num_components),
                )

            if (variational_mode == "both") | (variational_mode == "inv"):
                self.VariationalEncoder_inv = nn.Sequential(
                    nn.Linear(EGNN_layer_dim, EGNN_layer_dim),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(EGNN_layer_dim, 2 * EGNN_layer_dim),
                )

        if not self.shared_encoders:
            self.SubGraphEncoder = EGNN_VN_Encoder_point_cloud(
                node_input_dim=input_nf + fragment_library_dim,
                edges_in_d=edges_in_d,
                num_components=num_components,
                EGNN_layer_dim=EGNN_layer_dim,
                n_knn=n_knn,
                conv_dims=conv_dims,
                pooling_MLP=pooling_MLP,
                N_EGNN_layers=N_EGNN_layers,
                variational_GNN=False,
                variational_GNN_mol=False,
                mix_node_inv_to_equi=self.mix_node_inv_to_equi,
                mix_shape_to_nodes=self.mix_shape_to_nodes,
                ablate_HvarCat=self.ablate_HvarCat,
                old_EGNN=old_EGNN,
            )

        self.Equi_linear_leaky_1 = VNLinearAndLeakyReLU(
            num_components * 3 + int(self.subtract_latent_space) * num_components,
            num_components * 2,
            use_batchnorm=False,
            negative_slope=0.2,
        )
        self.Equi_linear_leaky_2 = VNLinearAndLeakyReLU(
            num_components * 2, num_components, use_batchnorm=False, negative_slope=0.2
        )
        self.Equi_linear_leaky_3 = VNLinearAndLeakyReLU(
            num_components, num_components, use_batchnorm=False, negative_slope=0.2
        )
        self.T_layer = VNLinearAndLeakyReLU(
            num_components, 3, use_batchnorm=False, negative_slope=0.2
        )

    def encode_fragment_library(self, fragment_batch, device=torch.device("cpu")):
        fragment_library_features, fragment_library_node_features, fragment_library_batch = (
            self.fragment_encoder(
                fragment_batch["x"],
                fragment_batch["edge_index"],
                fragment_batch["pos"],
                fragment_batch["edge_attr"],
                fragment_batch["batch"],
                device=device,
            )
        )
        return fragment_library_features, fragment_library_node_features, fragment_library_batch

    def encode(
        self,
        x,
        edge_index,
        pos,
        points,
        points_atom_index,
        edge_attr,
        batch_size,
        select_indices,
        select_indices_batch,
        shared_encoders=True,
        device=torch.device("cpu"),
        use_variational_GNN=False,
        variational_GNN_factor=1.0,
        interpolate_to_GNN_prior=0.0,
        h_interpolate=None,
    ):
        if shared_encoders:
            (
                x_inv,
                Z_equi,
                Z_inv,
                Z_equi_select,
                Z_inv_select,
                h_mean,
                h_std,
                h_reshaped_gnn,
                h_predicted_reshaped,
                h_reshaped,
            ) = self.GraphEncoder(
                x,
                edge_index,
                pos,
                edge_attr,
                batch_size,
                points,
                points_atom_index,
                select_indices=select_indices,
                select_indices_batch=select_indices_batch,
                device=device,
                use_variational_GNN=use_variational_GNN,
                variational_GNN_factor=variational_GNN_factor,
                interpolate_to_GNN_prior=interpolate_to_GNN_prior,
                h_interpolate=h_interpolate,
            )
        else:
            (
                x_inv,
                Z_equi,
                Z_inv,
                Z_equi_select,
                Z_inv_select,
                h_mean,
                h_std,
                h_reshaped_gnn,
                h_predicted_reshaped,
                h_reshaped,
            ) = self.SubGraphEncoder(
                x,
                edge_index,
                pos,
                edge_attr,
                batch_size,
                points,
                points_atom_index,
                select_indices=select_indices,
                select_indices_batch=select_indices_batch,
                device=device,
                use_variational_GNN=use_variational_GNN,
                variational_GNN_factor=variational_GNN_factor,
                interpolate_to_GNN_prior=interpolate_to_GNN_prior,
            )

        return (
            x_inv,
            Z_equi,
            Z_inv,
            Z_equi_select,
            Z_inv_select,
            h_mean,
            h_std,
            h_reshaped_gnn,
            h_predicted_reshaped,
            h_reshaped,
        )

    def mix_codes(
        self,
        batch_size,
        Z_equi,
        Z_inv,
        Z_equi_subgraph,
        Z_inv_subgraph,
        Z_equi_select,
        Z_inv_select,
    ):
        if self.ablateEqui:
            Z_equi = torch.zeros_like(Z_equi)

        if self.subtract_latent_space:
            Z_equivariant = torch.cat(
                (Z_equi, Z_equi_subgraph, Z_equi_select, Z_equi - Z_equi_subgraph), dim=1
            )
        else:
            Z_equivariant = torch.cat((Z_equi, Z_equi_subgraph, Z_equi_select), dim=1)

        if self.subtract_latent_space:
            Z_invariant = torch.cat(
                (Z_inv, Z_inv_subgraph, Z_inv_select, Z_inv - Z_inv_subgraph), dim=1
            )
        else:
            Z_invariant = torch.cat((Z_inv, Z_inv_subgraph, Z_inv_select), dim=1)

        Z_equivariant = self.Equi_linear_leaky_1(Z_equivariant)
        Z_equivariant = self.Equi_linear_leaky_2(Z_equivariant)
        Z_equivariant = self.Equi_linear_leaky_3(Z_equivariant)

        T_equivariant = self.T_layer(Z_equivariant)
        Z_T_invariant = torch.einsum("bij,bjk->bik", Z_equivariant, T_equivariant.permute(0, 2, 1))

        Z_T_invariant = Z_T_invariant.reshape(batch_size, -1)

        Z = torch.cat((Z_invariant, Z_T_invariant), dim=1)

        return Z

    def forward(
        self,
        batch_size,
        x,
        edge_index,
        edge_attr,
        pos,
        points,
        points_atom_index,
        x_library_fragment_index,
        x_subgraph,
        subgraph_edge_index,
        subgraph_edge_attr,
        subgraph_pos,
        subgraph_points,
        subgraph_points_atom_index,
        x_subgraph_library_fragment_index,
        query_indices,
        query_indices_batch,
        fragment_batch,
        device=torch.device("cpu"),
    ):
        fragment_library_features, fragment_library_node_features, fragment_library_batch = (
            self.encode_fragment_library(fragment_batch, device=device)
        )

        x = torch.cat((x, fragment_library_features[x_library_fragment_index]), dim=1)
        x_subgraph = torch.cat(
            (x_subgraph, fragment_library_features[x_subgraph_library_fragment_index]), dim=1
        )

        (
            _,
            Z_equi,
            Z_inv,
            _,
            _,
            h_mean,
            h_std,
            h_reshaped_gnn,
            h_predicted_reshaped,
            h_reshaped,
        ) = self.encode(
            x,
            edge_index,
            pos,
            points,
            points_atom_index,
            edge_attr,
            batch_size,
            select_indices=None,
            select_indices_batch=None,
            shared_encoders=True,
            device=device,
            use_variational_GNN=(self.variational_GNN) | (self.variational_GNN_mol),
            variational_GNN_factor=1.0,
            interpolate_to_GNN_prior=0.0,
        )

        if self.variational:
            if (self.variational_mode == "both") | (self.variational_mode == "equi"):
                Z_equi = self.VariationalEncoder_equi(Z_equi)
                Z_equi_mean, Z_equi_logvar = Z_equi.chunk(2, dim=1)
                Z_equi_logvar, _ = self.VariationalEncoder_equi_T(Z_equi_logvar)
                Z_equi_logvar = Z_equi_logvar.reshape(batch_size, -1)
                Z_equi_logvar = (
                    self.VariationEncoder_equi_linear(Z_equi_logvar)
                    .unsqueeze(2)
                    .expand((-1, -1, 3))
                )
                Z_equi_std = torch.exp(0.5 * Z_equi_logvar)
                Z_equi_eps = torch.randn_like(Z_equi_mean)
                Z_equi = Z_equi_mean + Z_equi_std * Z_equi_eps
            else:
                Z_equi_mean = None
                Z_equi_std = None

            if (self.variational_mode == "both") | (self.variational_mode == "inv"):
                Z_inv = self.VariationalEncoder_inv(Z_inv)
                Z_inv_mean, Z_inv_logvar = Z_inv.chunk(2, dim=1)
                Z_inv_std = torch.exp(0.5 * Z_inv_logvar)
                Z_inv_eps = torch.randn_like(Z_inv_mean)
                Z_inv = Z_inv_mean + Z_inv_std * Z_inv_eps
            else:
                Z_inv_mean = None
                Z_inv_std = None
        else:
            Z_equi_mean = None
            Z_equi_std = None
            Z_inv_mean = None
            Z_inv_std = None

        (
            x_inv_subgraph,
            Z_equi_subgraph,
            Z_inv_subgraph,
            Z_equi_select,
            Z_inv_select,
            _,
            _,
            _,
            _,
            _,
        ) = self.encode(
            x_subgraph,
            subgraph_edge_index,
            subgraph_pos,
            subgraph_points,
            subgraph_points_atom_index,
            subgraph_edge_attr,
            batch_size,
            select_indices=query_indices,
            select_indices_batch=query_indices_batch,
            shared_encoders=self.shared_encoders,
            device=device,
            use_variational_GNN=False,
            variational_GNN_factor=1.0,
            interpolate_to_GNN_prior=0.0,
        )

        if self.ablateEqui:
            Z_equi = torch.zeros_like(Z_equi)

        graph_subgraph_select_features_concat = self.mix_codes(
            batch_size, Z_equi, Z_inv, Z_equi_subgraph, Z_inv_subgraph, Z_equi_select, Z_inv_select
        )

        h_subgraph = x_inv_subgraph.permute(0, 2, 1).reshape(-1, x_inv_subgraph.shape[1])
        h_select = h_subgraph[query_indices]

        return (
            graph_subgraph_select_features_concat,
            h_subgraph,
            h_select,
            fragment_library_features,
            fragment_library_node_features,
            fragment_library_batch,
            Z_equi_mean,
            Z_equi_std,
            Z_inv_mean,
            Z_inv_std,
            h_mean,
            h_std,
            h_reshaped_gnn,
            h_predicted_reshaped,
            h_reshaped,
        )


# ============================== models/models.py ==============================
class ROCS_Model_Point_Cloud(nn.Module):
    def __init__(
        self,
        input_nf=45,
        edges_in_d=5,
        n_knn=10,
        conv_dims=None,
        num_components=64,
        fragment_library_dim=64,
        N_fragment_layers=2,
        append_noise=False,
        N_members=72,
        EGNN_layer_dim=64,
        N_EGNN_layers=3,
        output_MLP_hidden_dim=64,
        pooling_MLP=True,
        shared_encoders=False,
        subtract_latent_space=False,
        variational=False,
        variational_mode="both",
        variational_GNN=False,
        variational_GNN_mol=False,
        mix_node_inv_to_equi=False,
        mix_shape_to_nodes=False,
        ablate_HvarCat=False,
        ablateEqui=False,
        old_EGNN=False,
    ):
        super().__init__()
        if conv_dims is None:
            conv_dims = [64, 64, 128, 256]

        self.input_nf = input_nf
        self.edges_in_d = edges_in_d

        self.conv_dims = conv_dims
        self.num_components = num_components
        self.fragment_library_dim = fragment_library_dim
        self.EGNN_layer_dim = EGNN_layer_dim
        self.N_EGNN_layers = N_EGNN_layers
        self.n_knn = n_knn
        self.output_MLP_hidden_dim = output_MLP_hidden_dim
        self.pooling_MLP = pooling_MLP
        self.shared_encoders = shared_encoders
        self.subtract_latent_space = subtract_latent_space
        self.N_fragment_layers = N_fragment_layers
        self.N_members = N_members
        self.append_noise = append_noise
        self.variational = variational
        self.variational_mode = variational_mode

        if not self.subtract_latent_space:
            graph_subgraph_focal_features_concat_dim = num_components * 3 + EGNN_layer_dim * 3
        else:
            graph_subgraph_focal_features_concat_dim = num_components * 3 + EGNN_layer_dim * 4

        self.Encoder = Encoder_point_cloud(
            input_nf=input_nf,
            edges_in_d=edges_in_d,
            n_knn=n_knn,
            conv_dims=conv_dims,
            num_components=num_components,
            fragment_library_dim=fragment_library_dim,
            N_fragment_layers=N_fragment_layers,
            append_noise=append_noise,
            N_members=N_members,
            EGNN_layer_dim=EGNN_layer_dim,
            N_EGNN_layers=N_EGNN_layers,
            pooling_MLP=pooling_MLP,
            shared_encoders=shared_encoders,
            subtract_latent_space=subtract_latent_space,
            variational=variational,
            variational_mode=variational_mode,
            variational_GNN=variational_GNN,
            variational_GNN_mol=variational_GNN_mol,
            mix_node_inv_to_equi=mix_node_inv_to_equi,
            mix_shape_to_nodes=mix_shape_to_nodes,
            ablate_HvarCat=ablate_HvarCat,
            ablateEqui=ablateEqui,
            old_EGNN=old_EGNN,
        )

        self.ROCS_scorer = EGNN_MLP(
            graph_subgraph_focal_features_concat_dim, 1, output_MLP_hidden_dim
        )

    def forward(self, batch):
        (
            graph_subgraph_select_features_concat,
            h_subgraph,
            h_select,
            _,
            _,
            _,
            Z_equi_mean,
            Z_equi_std,
            Z_inv_mean,
            Z_inv_std,
            h_mean,
            h_std,
            h_reshaped_gnn,
            h_predicted_reshaped,
            h_reshaped,
        ) = self.Encoder(
            batch["batch_size"],
            batch["x"],
            batch["edge_index"],
            batch["edge_attr"],
            batch["pos"],
            batch["points"],
            batch["points_atom_index"],
            batch["x_library_fragment_index"],
            batch["x_subgraph"],
            batch["subgraph_edge_index"],
            batch["subgraph_edge_attr"],
            batch["subgraph_pos"],
            batch["subgraph_points"],
            batch["subgraph_points_atom_index"],
            batch["x_subgraph_library_fragment_index"],
            batch["query_indices"],
            batch["query_indices_batch"],
            batch["fragment_batch"],
            device=torch.device("cpu"),
        )

        scores = self.ROCS_scorer(graph_subgraph_select_features_concat)

        return scores


MENAGERIE_ZOO = "vendored-pytorch"

_INPUT_NF = 8
_EDGES_IN_D = 3
_N_KNN = 3
_CONV_DIMS = [12, 12, 24, 24]
_NUM_COMPONENTS = 8
_FRAGMENT_LIBRARY_DIM = 6
_EGNN_LAYER_DIM = 6
_N_POINTS_PER_CLOUD = 16  # must be > n_knn


def build_squid():
    return ROCS_Model_Point_Cloud(
        input_nf=_INPUT_NF,
        edges_in_d=_EDGES_IN_D,
        n_knn=_N_KNN,
        conv_dims=_CONV_DIMS,
        num_components=_NUM_COMPONENTS,
        fragment_library_dim=_FRAGMENT_LIBRARY_DIM,
        N_fragment_layers=2,
        append_noise=False,
        N_members=4,
        EGNN_layer_dim=_EGNN_LAYER_DIM,
        N_EGNN_layers=2,
        output_MLP_hidden_dim=8,
        pooling_MLP=False,
        shared_encoders=True,
        subtract_latent_space=False,
        variational=False,
        variational_GNN=False,
        variational_GNN_mol=False,
        mix_node_inv_to_equi=False,
        mix_shape_to_nodes=False,
        ablateEqui=False,
        old_EGNN=False,
    )


def _make_small_graph_batch(batch_size, n_nodes_per_graph, n_edges_per_graph, n_fragments):
    """Build a tiny fully-synthetic molecular-graph + point-cloud batch matching the
    index-alignment conventions in Encoder_point_cloud.forward / EGNN_VN_Encoder_point_cloud.forward.
    """
    total_nodes = batch_size * n_nodes_per_graph

    # graph node features / edges / 3D coords, batch-concatenated PyG-style (flat node axis)
    x = torch.randn(total_nodes, _INPUT_NF)
    pos = torch.randn(total_nodes, 3)

    edge_list = []
    for b in range(batch_size):
        offset = b * n_nodes_per_graph
        for _ in range(n_edges_per_graph):
            i, j = torch.randint(0, n_nodes_per_graph, (2,)).tolist()
            if i == j:
                j = (j + 1) % n_nodes_per_graph
            edge_list.append((offset + i, offset + j))
            edge_list.append((offset + j, offset + i))
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.randn(edge_index.shape[1], _EDGES_IN_D)

    # every node is assigned to one fragment-library member (0..n_fragments-1), batch-flat
    x_library_fragment_index = torch.randint(0, n_fragments, (total_nodes,), dtype=torch.long)

    # point cloud: fixed number of points per graph, batch-major reshape as required by
    # EGNN_VN_Encoder_point_cloud.forward (points.reshape(batch_size, -1, 3))
    points = torch.randn(batch_size * _N_POINTS_PER_CLOUD, 3)
    points_atom_index = torch.randint(
        0, n_nodes_per_graph, (batch_size, _N_POINTS_PER_CLOUD), dtype=torch.long
    ).reshape(-1)

    return x, edge_index, edge_attr, pos, points, points_atom_index, x_library_fragment_index


def example_input_squid():
    torch.manual_seed(0)
    batch_size = 2
    n_nodes_per_graph = 6
    n_edges_per_graph = 5
    n_subgraph_nodes_per_graph = 4
    n_subgraph_edges_per_graph = 3
    n_fragments = 4

    x, edge_index, edge_attr, pos, points, points_atom_index, x_library_fragment_index = (
        _make_small_graph_batch(batch_size, n_nodes_per_graph, n_edges_per_graph, n_fragments)
    )

    (
        x_subgraph,
        subgraph_edge_index,
        subgraph_edge_attr,
        subgraph_pos,
        subgraph_points,
        subgraph_points_atom_index,
        x_subgraph_library_fragment_index,
    ) = _make_small_graph_batch(
        batch_size, n_subgraph_nodes_per_graph, n_subgraph_edges_per_graph, n_fragments
    )

    # select 1 "query" (focal) node per graph in the subgraph, batch-flat indices + owning-graph index
    query_indices = torch.tensor(
        [b * n_subgraph_nodes_per_graph for b in range(batch_size)], dtype=torch.long
    )
    query_indices_batch = torch.arange(batch_size, dtype=torch.long)

    # tiny fragment library graph batch (consumed by FragmentLibraryEncoder), PyG-Batch-style:
    # n_fragments small graphs, each with a few atoms, concatenated with a `batch` index vector.
    frag_nodes_per_member = 3
    frag_total_nodes = n_fragments * frag_nodes_per_member
    frag_x = torch.randn(frag_total_nodes, _INPUT_NF)
    frag_pos = torch.randn(frag_total_nodes, 3)
    frag_edge_list = []
    for f in range(n_fragments):
        offset = f * frag_nodes_per_member
        for k in range(frag_nodes_per_member - 1):
            frag_edge_list.append((offset + k, offset + k + 1))
            frag_edge_list.append((offset + k + 1, offset + k))
    frag_edge_index = torch.tensor(frag_edge_list, dtype=torch.long).t().contiguous()
    frag_edge_attr = torch.randn(frag_edge_index.shape[1], _EDGES_IN_D)
    frag_batch_index = torch.arange(n_fragments, dtype=torch.long).repeat_interleave(
        frag_nodes_per_member
    )

    fragment_batch = {
        "x": frag_x,
        "edge_index": frag_edge_index,
        "pos": frag_pos,
        "edge_attr": frag_edge_attr,
        "batch": frag_batch_index,
    }

    batch = {
        "batch_size": batch_size,
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "pos": pos,
        "points": points,
        "points_atom_index": points_atom_index,
        "x_library_fragment_index": x_library_fragment_index,
        "x_subgraph": x_subgraph,
        "subgraph_edge_index": subgraph_edge_index,
        "subgraph_edge_attr": subgraph_edge_attr,
        "subgraph_pos": subgraph_pos,
        "subgraph_points": subgraph_points,
        "subgraph_points_atom_index": subgraph_points_atom_index,
        "x_subgraph_library_fragment_index": x_subgraph_library_fragment_index,
        "query_indices": query_indices,
        "query_indices_batch": query_indices_batch,
        "fragment_batch": fragment_batch,
    }
    return (batch,)


MENAGERIE_ENTRIES = [
    (
        "SQUID (shape-conditioned 3D molecule generator)",
        build_squid,
        example_input_squid,
        2023,
        "SOURCE_AVAILABLE",
    ),
]
