# FAITHFUL PORT of https://github.com/jertubiana/ScanNet @ main (original framework: TensorFlow/Keras)
#
# ScanNet (Tubiana, Schneidman-Duhovny & Wolfson, Nat. Methods 2022): a
# geometric deep network for protein binding-site / interface prediction
# built from spatio-chemical filters over local point-cloud neighborhoods.
#
# The original repo (network/scannet.py, network/neighborhoods.py,
# network/embeddings.py, network/attention.py, network/utils.py) is
# TensorFlow 1.x / old-style Keras (`keras.engine.base_layer.Layer`,
# `keras.backend`), which is not a base-lib-installable framework here.
# This module TRANSCRIBES the real architecture -- every layer/mechanism
# from the actual repo code -- into self-contained inference-mode PyTorch:
#
#   FrameBuilder            -> ScanNetFrameBuilder   (network/neighborhoods.py)
#   LocalNeighborhood        -> ScanNetLocalNeighborhood (network/neighborhoods.py)
#   GaussianKernel            -> ScanNetGaussianKernel (network/embeddings.py)
#   OuterProduct               -> ScanNetOuterProduct  (network/embeddings.py)
#   MultiTanh                    -> ScanNetMultiTanh    (network/embeddings.py)
#   AttentionLayer (graph attn)   -> ScanNetAttentionLayer (network/attention.py)
#   neighborhood_embedding()     -> scannet_neighborhood_embedding() (scannet.py)
#   ScanNet(...)                 -> ScanNet nn.Module (scannet.py, default
#                                    protein-protein-binding-site config:
#                                    with_atom=True, frame_aa='triplet_backbone',
#                                    coordinates_graph=['distance','ZdotZ',
#                                    'ZdotDelta','index_distance'], nrotations=1)
#
# Masking is dropped (all sequences in a batch use the same real length --
# the original masking machinery exists purely to handle ragged batches of
# variable-length proteins padded to Lmax; with no padding it is the
# identity). GMM-based Gaussian-kernel *initialization* (sklearn
# GaussianMixture fit over a sample of local coordinates, used only to seed
# training) is replaced with the repo's own random-initialization fallback
# path (`initialize_GaussianKernelRandom`, verbatim), since faithful
# initialization must run either way before any training and does not
# change the model's forward architecture. `MaxAbsPooling`
# (network/utils.py) is intentionally omitted: it is only reached when
# `nrotations > 1`, and the repo's own default (and the one used above) is
# `nrotations=1`.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


# ---------------------------------------------------------------------------
# network/embeddings.py :: initialize_GaussianKernelRandom (verbatim, numpy)
# ---------------------------------------------------------------------------


def initialize_GaussianKernelRandom(xlims, N, covariance_type):
    xlims = np.array(xlims, dtype=np.float32)
    centers = np.random.rand(xlims.shape[0], N).astype(np.float32)
    centers = centers * (xlims[:, 1] - xlims[:, 0])[:, np.newaxis] + xlims[:, 0][:, np.newaxis]

    widths = np.ones([xlims.shape[0], N], dtype=np.float32)
    widths = widths * (xlims[:, 1] - xlims[:, 0])[:, np.newaxis] / (N / 4)

    if covariance_type == "diag":
        return [centers, widths]
    else:
        sqrt_precision_matrix = np.stack(
            [np.diag(1.0 / (1e-4 + widths[:, n])).astype(np.float32) for n in range(N)], axis=-1
        )
        return [centers, sqrt_precision_matrix]


# ---------------------------------------------------------------------------
# network/neighborhoods.py :: FrameBuilder
# ---------------------------------------------------------------------------


class ScanNetFrameBuilder(nn.Module):
    """Port of neighborhoods.py::FrameBuilder. Builds a local reference frame
    (center, x/y/z axes[, dipole]) at every residue/atom from a triplet
    (or quadruplet, if dipole=True) of point indices via Schmidt
    orthogonalization."""

    def __init__(self, order="1", dipole=False):
        super().__init__()
        self.order = order
        self.dipole = dipole
        self.epsilon = 1e-6
        self.register_buffer("xaxis", torch.tensor([[1.0, 0.0, 0.0]]))
        self.register_buffer("yaxis", torch.tensor([[0.0, 1.0, 0.0]]))
        self.register_buffer("zaxis", torch.tensor([[0.0, 0.0, 1.0]]))

    def forward(self, points, triplets):
        # points: B x L x 3 ; triplets: B x L x (3 or 4) long
        triplets = torch.clamp(triplets, 0, points.shape[-2] - 1)

        # Matches tf.gather_nd(points, triplets[:,:,k:k+1], batch_dims=1):
        # for each batch b, position l: points[b, triplets[b,l,k]]
        def gather_batched(k):
            idx = triplets[:, :, k]  # B x L
            idx_exp = idx.unsqueeze(-1).expand(-1, -1, 3)  # B x L x 3
            return torch.gather(points, 1, idx_exp)  # B x L x 3

        delta_10 = gather_batched(1) - gather_batched(0)
        delta_20 = gather_batched(2) - gather_batched(0)
        if self.order in ("2", "3"):
            delta_10, delta_20 = delta_20, delta_10

        centers = gather_batched(0)
        zaxis = (delta_10 + self.epsilon * self.zaxis.view(1, 1, 3)) / (
            torch.sqrt(torch.sum(delta_10**2, dim=-1, keepdim=True)) + self.epsilon
        )

        yaxis = torch.cross(zaxis, delta_20, dim=-1)
        yaxis = (yaxis + self.epsilon * self.yaxis.view(1, 1, 3)) / (
            torch.sqrt(torch.sum(yaxis**2, dim=-1, keepdim=True)) + self.epsilon
        )

        xaxis = torch.cross(yaxis, zaxis, dim=-1)
        xaxis = (xaxis + self.epsilon * self.xaxis.view(1, 1, 3)) / (
            torch.sqrt(torch.sum(xaxis**2, dim=-1, keepdim=True)) + self.epsilon
        )

        if self.order == "3":
            xaxis, yaxis, zaxis = zaxis, xaxis, yaxis

        if self.dipole:
            dipole_vec = gather_batched(3) - gather_batched(0)
            dipole_vec = (dipole_vec + self.epsilon * self.zaxis.view(1, 1, 3)) / (
                torch.sqrt(torch.sum(dipole_vec**2, dim=-1, keepdim=True)) + self.epsilon
            )
            frames = torch.stack([centers, xaxis, yaxis, zaxis, dipole_vec], dim=-2)
        else:
            frames = torch.stack([centers, xaxis, yaxis, zaxis], dim=-2)
        return frames


def _distance(c1, c2, squared=False, ndims=3):
    d = (c1[..., 0].unsqueeze(-1) - c2[..., 0].unsqueeze(-2)) ** 2
    for n in range(1, ndims):
        d = d + (c1[..., n].unsqueeze(-1) - c2[..., n].unsqueeze(-2)) ** 2
    if not squared:
        d = torch.sqrt(d)
    return d


# ---------------------------------------------------------------------------
# network/neighborhoods.py :: LocalNeighborhood
# ---------------------------------------------------------------------------


class ScanNetLocalNeighborhood(nn.Module):
    """Port of neighborhoods.py::LocalNeighborhood (unmasked/inference form).

    Supports the coordinate subsets actually used by the default ScanNet
    config: ['euclidian'] (amino-acid & atom scale) and
    ['distance','ZdotZ','ZdotDelta','index_distance'] (graph scale), plus
    ['index_distance'] alone (atom->aa pooling bipartite graph)."""

    def __init__(
        self, Kmax=10, coordinates=("euclidian",), self_neighborhood=True, index_distance_max=None
    ):
        super().__init__()
        self.Kmax = Kmax
        self.coordinates = list(coordinates)
        self.self_neighborhood = self_neighborhood
        self.index_distance_max = index_distance_max
        self.epsilon = 1e-10
        self.big_distance = 1000.0

    def forward(self, first_frame, first_index, second_frame, second_index, attributes):
        """first_frame/second_frame: B x L x 4 x 3 (or None if not needed for this
        coordinate set); first_index/second_index: B x L x 1 (sequence
        position indices); attributes: list of B x L x d_k tensors gathered
        from the "second" set."""
        # first_index/second_index keep their trailing size-1 feature dim so
        # _distance's c[..., 0] indexing works uniformly for the ndims=1
        # (index-only, e.g. atom->aa pooling) and ndims=3 (frame-based) cases.
        first_center = first_frame[:, :, 0] if first_frame is not None else first_index.float()
        second_center = second_frame[:, :, 0] if second_frame is not None else second_index.float()
        ndims = 3 if first_frame is not None else 1

        distance_square = _distance(first_center, second_center, squared=True, ndims=ndims)

        # argsort ascending -> nearest Kmax neighbors (ties broken as in tf.argsort, stable)
        order = torch.argsort(distance_square, dim=-1)
        neighbors = order[:, :, : self.Kmax]  # B x L x Kmax

        def gather_neighbor(x):
            # x: B x L2 x d -> B x L x Kmax x d, gathered along dim=1 per (b, l)
            B, L, K = neighbors.shape
            d = x.shape[-1]
            idx = neighbors.unsqueeze(-1).expand(-1, -1, -1, d)  # B x L x K x d
            x_exp = x.unsqueeze(1).expand(-1, L, -1, -1)  # B x L x L2 x d
            return torch.gather(x_exp, 2, idx)

        neighbors_attributes = [gather_neighbor(a) for a in attributes]

        neighbor_coordinates = []

        if "euclidian" in self.coordinates:
            sec_c = gather_neighbor(second_center) - first_center.unsqueeze(-2)  # B L K 3
            axes = first_frame[:, :, 1:4]  # B L 3 3
            euclidian = torch.sum(sec_c.unsqueeze(-2) * axes.unsqueeze(-3), dim=-1)  # B L K 3
            neighbor_coordinates.append(euclidian)

        distance_neighbors = None
        if "distance" in self.coordinates:
            distance_neighbors = torch.sqrt(torch.gather(distance_square, -1, neighbors)).unsqueeze(
                -1
            )
            neighbor_coordinates.append(distance_neighbors)

        if "ZdotZ" in self.coordinates:
            first_z = first_frame[:, :, -1]
            second_z = second_frame[:, :, -1]
            zdotz = torch.sum(
                first_z.unsqueeze(-2) * gather_neighbor(second_z), dim=-1, keepdim=True
            )
            neighbor_coordinates.append(zdotz)

        if "ZdotDelta" in self.coordinates:
            first_z = first_frame[:, :, -1]
            second_z = second_frame[:, :, -1]
            delta_center = (gather_neighbor(second_center) - first_center.unsqueeze(-2)) / (
                distance_neighbors + self.epsilon
            )
            zdotdelta = torch.sum(first_z.unsqueeze(-2) * delta_center, dim=-1, keepdim=True)
            deltadotz = torch.sum(delta_center * gather_neighbor(second_z), dim=-1, keepdim=True)
            neighbor_coordinates.append(deltadotz)
            neighbor_coordinates.append(zdotdelta)

        if "index_distance" in self.coordinates:
            # first_index/second_index: B x L x 1 -> B x L x Kmax x 1, matching
            # tf.abs(expand_dims(first_index,-2) - gather_nd(second_index,neighbors)).
            idx_dist = torch.abs(
                first_index.unsqueeze(-2).float() - gather_neighbor(second_index).float()
            )
            if self.index_distance_max is not None:
                idx_dist = torch.clamp(idx_dist, 0, self.index_distance_max)
            neighbor_coordinates.append(idx_dist)

        neighbor_coordinates = torch.cat(neighbor_coordinates, dim=-1)
        return [neighbor_coordinates] + neighbors_attributes


# ---------------------------------------------------------------------------
# network/embeddings.py :: GaussianKernel
# ---------------------------------------------------------------------------


class ScanNetGaussianKernel(nn.Module):
    """Port of embeddings.py::GaussianKernel (covariance_type='full' or
    'diag'), initialized via the repo's own random fallback path."""

    def __init__(self, d, N, covariance_type="full", eps=1e-1):
        super().__init__()
        self.N = N
        self.d = d
        self.covariance_type = covariance_type
        self.eps = eps

        xlims = [[-8, 8]] * d if covariance_type in ("full", "diag") else [[-8, 8]] * d
        centers, second = initialize_GaussianKernelRandom(xlims, N, covariance_type)
        self.centers = nn.Parameter(torch.from_numpy(centers))  # d x N
        if covariance_type == "diag":
            self.widths = nn.Parameter(torch.from_numpy(second))  # d x N
        else:
            self.sqrt_precision = nn.Parameter(torch.from_numpy(second))  # d x d x N

    def forward(self, x):
        # x: ... x d
        nbatch_dim = x.dim() - 1
        if self.covariance_type == "diag":
            centers = self.centers.view(*([1] * nbatch_dim), self.d, self.N)
            widths = self.widths.view(*([1] * nbatch_dim), self.d, self.N)
            diff = (x.unsqueeze(-1) - centers) / (self.eps + widths)
            activity = torch.exp(-0.5 * torch.sum(diff**2, dim=-2))
        else:
            centers = self.centers.view(*([1] * nbatch_dim), self.d, self.N)
            intermediate = x.unsqueeze(-1) - centers  # ... x d x N
            sqrt_precision = self.sqrt_precision.unsqueeze(0)  # 1 x d x d x N
            intermediate2 = torch.sum(
                intermediate.unsqueeze(-3) * sqrt_precision, dim=-2
            )  # ... x d x N
            activity = torch.exp(-0.5 * torch.sum(intermediate2**2, dim=-2))
        return activity


# ---------------------------------------------------------------------------
# network/embeddings.py :: OuterProduct
# ---------------------------------------------------------------------------


class ScanNetOuterProduct(nn.Module):
    """Port of embeddings.py::OuterProduct (use_single1=True, use_single2=False,
    use_bias=False path, the only configuration exercised by
    neighborhood_embedding())."""

    def __init__(
        self, n1, n2, n_filters, use_single1=True, use_single2=False, use_bias=False, sum_axis=None
    ):
        super().__init__()
        self.n1 = n1
        self.n2 = n2
        self.n_filters = n_filters
        self.use_single1 = use_single1
        self.use_single2 = use_single2
        self.use_bias = use_bias
        self.sum_axis = sum_axis

        stddev = 1.0 / np.sqrt(n1 * n2)
        self.kernel12 = nn.Parameter(torch.empty(n1, n2, n_filters).normal_(0, stddev))
        if use_single1:
            self.kernel1 = nn.Parameter(torch.empty(n1, n_filters).normal_(0, 1.0 / np.sqrt(n1)))
        if use_single2:
            self.kernel2 = nn.Parameter(torch.empty(n2, n_filters).normal_(0, 1.0 / np.sqrt(n2)))
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(n_filters))

    def forward(self, first_input, second_input):
        if self.sum_axis is not None:
            outer = torch.sum(
                first_input.unsqueeze(-1) * second_input.unsqueeze(-2), dim=self.sum_axis
            )
        else:
            outer = first_input.unsqueeze(-1) * second_input.unsqueeze(-2)
        activity = torch.tensordot(outer, self.kernel12, dims=([-2, -1], [0, 1]))

        if self.use_single1:
            if self.sum_axis is not None:
                activity = activity + torch.matmul(
                    torch.sum(first_input, dim=self.sum_axis), self.kernel1
                )
            else:
                activity = activity + torch.matmul(first_input, self.kernel1)
        if self.use_single2:
            if self.sum_axis is not None:
                activity = activity + torch.matmul(
                    torch.sum(second_input, dim=self.sum_axis), self.kernel2
                )
            else:
                activity = activity + torch.matmul(second_input, self.kernel2)
        if self.use_bias:
            activity = activity + self.bias
        return activity


# ---------------------------------------------------------------------------
# network/embeddings.py :: MultiTanh
# ---------------------------------------------------------------------------


class ScanNetMultiTanh(nn.Module):
    """Port of embeddings.py::MultiTanh (used for activation='multitanh5')."""

    def __init__(self, d, ntanh=5, use_bias=True):
        super().__init__()
        self.d = d
        self.ntanh = ntanh
        self.use_bias = use_bias
        self.widths = nn.Parameter(torch.ones(d, ntanh))
        self.slopes = nn.Parameter(torch.ones(d, ntanh))
        initial_offsets = np.zeros([d, ntanh], dtype=np.float32)
        if ntanh > 1:
            initial_offsets += (np.arange(ntanh) / (ntanh - 1) * 6 - 3)[np.newaxis]
        self.offsets = nn.Parameter(torch.from_numpy(initial_offsets))
        if use_bias:
            self.biases = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        nbatch_dim = x.dim() - 1
        widths = self.widths.view(*([1] * nbatch_dim), self.d, self.ntanh)
        slopes = self.slopes.view(*([1] * nbatch_dim), self.d, self.ntanh)
        offsets = self.offsets.view(*([1] * nbatch_dim), self.d, self.ntanh)
        out = torch.sum(slopes * torch.tanh((x.unsqueeze(-1) - offsets) / (widths + 1e-4)), dim=-1)
        if self.use_bias:
            out = out + self.biases.view(*([1] * nbatch_dim), self.d)
        return out


# ---------------------------------------------------------------------------
# BatchNorm w/o masking (unmasked reduces to standard BatchNorm1d-over-last-dim)
# ---------------------------------------------------------------------------


class ScanNetMaskedBatchNorm(nn.Module):
    """Port of embeddings.py::MaskedBatchNormalization, specialized to the
    unmasked (no padding) inference case, which reduces exactly to a
    channel-last BatchNorm normalizing over all leading (batch+sequence)
    axes -- the `normalize_inference` branch of the original with a trivial
    (all-ones) mask."""

    def __init__(self, num_features, center=True, scale=True, eps=1e-3):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps=eps, affine=(center or scale))
        self.center = center
        self.scale = scale

    def forward(self, x):
        # x: B x L x C -> normalize over C using running stats (eval mode)
        shp = x.shape
        flat = x.reshape(-1, shp[-1])
        out = self.bn(flat)
        return out.reshape(shp)


# ---------------------------------------------------------------------------
# network/attention.py :: AttentionLayer
# ---------------------------------------------------------------------------


class ScanNetAttentionLayer(nn.Module):
    """Port of attention.py::AttentionLayer, supporting both configurations
    used in ScanNet: (beta=True, self_attention=True) for the final graph
    attention, and (beta=False, self_attention=False) for the atom->aa
    attention-pooling step (``beta``/``self_attn`` are then ignored, as in
    the original's branch on ``self.beta``/``self.self_attention``)."""

    def __init__(
        self, nfeatures_graph, nheads, nfeatures_output, Kmax, self_attention=True, beta=True
    ):
        super().__init__()
        self.self_attention = self_attention
        self.beta = beta
        self.epsilon = 1e-6
        self.nfeatures_graph = nfeatures_graph
        self.nheads = nheads
        self.nfeatures_output = nfeatures_output
        self.Kmax = Kmax

    def forward(self, beta, self_attn, attention_coefficients, node_outputs, graph_weights):
        B, L = attention_coefficients.shape[0], attention_coefficients.shape[1]
        attention_coefficients = attention_coefficients.reshape(
            B, L, self.Kmax, self.nfeatures_graph, self.nheads
        )
        node_outputs = node_outputs.reshape(B, L, self.Kmax, self.nfeatures_output, self.nheads)

        if self.self_attention:
            self_attn = self_attn.reshape(B, L, self.nfeatures_graph, self.nheads)
            ac_self, ac_others = torch.split(attention_coefficients, [1, self.Kmax - 1], dim=2)
            ac_self = ac_self + self_attn.unsqueeze(2)
            attention_coefficients = torch.cat([ac_self, ac_others], dim=2)

        if self.beta:
            beta = beta.reshape(B, L, self.nfeatures_graph, self.nheads)
            attention_coefficients = attention_coefficients * (beta + self.epsilon).unsqueeze(2)

        attention_coefficients = attention_coefficients - torch.amax(
            attention_coefficients, dim=(-3, -2), keepdim=True
        )
        attention_coefficients_final = torch.sum(
            graph_weights.unsqueeze(-1) * torch.exp(attention_coefficients), dim=-2
        )
        attention_coefficients_final = attention_coefficients_final / (
            torch.sum(torch.abs(attention_coefficients_final), dim=-2, keepdim=True) + self.epsilon
        )
        output_final = torch.sum(
            node_outputs * attention_coefficients_final.unsqueeze(-2), dim=2
        ).reshape(B, L, self.nfeatures_output * self.nheads)
        return output_final, attention_coefficients_final


# ---------------------------------------------------------------------------
# network/scannet.py :: neighborhood_embedding + ScanNet
# ---------------------------------------------------------------------------


def _add_nonlinearity(bn, activation, x):
    x = bn(x)
    if activation is None:
        return x
    return activation(x)


class ScanNetNeighborhoodEmbedding(nn.Module):
    """Port of scannet.py::neighborhood_embedding for the single-rotation,
    non-index-distance ('euclidian' only) case used at 'aa' and 'atom'
    scale."""

    def __init__(
        self,
        order_frame,
        dipole_frame,
        Kmax,
        Ngaussians,
        nfilters,
        nfeatures_attr,
        activation_ctor,
        covariance_type="full",
    ):
        super().__init__()
        self.frames = ScanNetFrameBuilder(order=order_frame, dipole=dipole_frame)
        self.neighborhood = ScanNetLocalNeighborhood(
            Kmax=Kmax, coordinates=("euclidian",), self_neighborhood=True
        )
        self.gaussian = ScanNetGaussianKernel(d=3, N=Ngaussians, covariance_type=covariance_type)
        self.outer = ScanNetOuterProduct(
            Ngaussians,
            nfeatures_attr,
            nfilters,
            use_single1=True,
            use_single2=False,
            use_bias=False,
            sum_axis=2,
        )
        self.bn = ScanNetMaskedBatchNorm(nfilters)
        self.activation = activation_ctor(nfilters) if activation_ctor is not None else None

    def forward(self, point_clouds, frame_indices, attributes, sequence_indices):
        frames = self.frames(point_clouds, frame_indices)
        local_coordinates, local_attributes = self.neighborhood(
            frames, sequence_indices, frames, sequence_indices, [attributes]
        )
        embedded_local_coordinates = self.gaussian(local_coordinates)
        spatiochemical = self.outer(embedded_local_coordinates, local_attributes)
        activity = _add_nonlinearity(self.bn, self.activation, spatiochemical)
        return activity, frames


class _MultiTanhActivation(nn.Module):
    def __init__(self, d, ntanh=5):
        super().__init__()
        self.mt = ScanNetMultiTanh(d, ntanh=ntanh, use_bias=True)

    def forward(self, x):
        return self.mt(x)


class ScanNet(nn.Module):
    """Port of scannet.py::ScanNet(...) at its published default
    protein-protein-binding-site configuration: with_atom=True,
    frame_aa='triplet_backbone' (order_aa='3'), activation='multitanh5',
    coordinates_graph=['distance','ZdotZ','ZdotDelta','index_distance'].

    Sized down (Lmax/K/N/nfilters) from the paper's production config for a
    fast, faithful architecture trace."""

    def __init__(
        self,
        Lmax_aa=24,
        Lmax_atom=None,
        K_aa=8,
        K_atom=8,
        K_graph=8,
        N_aa=8,
        N_atom=8,
        N_graph=8,
        nfeatures_atom=4,
        nfeatures_aa=20,
        nembedding_atom=4,
        nembedding_aa=8,
        nembedding_graph=1,
        nfilters_atom=8,
        nfilters_aa=16,
        nfilters_graph=2,
        nattentionheads_graph=1,
    ):
        super().__init__()
        self.Lmax_aa = Lmax_aa
        self.Lmax_atom = Lmax_atom or (9 * Lmax_aa)
        self.K_graph = K_graph
        self.nembedding_graph = nembedding_graph
        self.nattentionheads_graph = nattentionheads_graph
        self.nfilters_graph = nfilters_graph

        # attribute embeddings
        self.embed_aa = nn.Linear(nfeatures_aa, nembedding_aa, bias=False)
        self.embed_aa_bn = ScanNetMaskedBatchNorm(nembedding_aa)
        self.embed_aa_act = _MultiTanhActivation(nembedding_aa)

        self.embed_atom = nn.Embedding(nfeatures_atom + 1, nembedding_atom, padding_idx=0)

        # atomic-scale neighborhood embedding
        self.atom_embedding = ScanNetNeighborhoodEmbedding(
            order_frame="2",
            dipole_frame=False,
            Kmax=K_atom,
            Ngaussians=N_atom,
            nfilters=nfilters_atom,
            nfeatures_attr=nembedding_atom,
            activation_ctor=lambda d: _MultiTanhActivation(d),
        )

        # atom -> aa attention pooling
        self.pooling_attention = nn.Linear(nfilters_atom, 1, bias=False)
        self.pooling_features = nn.Linear(nfilters_atom, nfilters_atom, bias=False)
        self.pooling_neighborhood = ScanNetLocalNeighborhood(
            Kmax=14, coordinates=("index_distance",), self_neighborhood=False, index_distance_max=1
        )
        self.pooling_attention_layer = ScanNetAttentionLayer(
            nfeatures_graph=1,
            nheads=1,
            nfeatures_output=nfilters_atom,
            Kmax=14,
            self_attention=False,
            beta=False,
        )
        self.pooled_bn = ScanNetMaskedBatchNorm(nfilters_atom)
        self.pooled_act = _MultiTanhActivation(nfilters_atom)

        # amino-acid scale neighborhood embedding (input = concat(seq embed, pooled atom filters))
        d_attr_aa = nembedding_aa + nfilters_atom
        self.aa_embedding = ScanNetNeighborhoodEmbedding(
            order_frame="3",
            dipole_frame=False,
            Kmax=K_aa,
            Ngaussians=N_aa,
            nfilters=nfilters_aa,
            nfeatures_attr=d_attr_aa,
            activation_ctor=lambda d: _MultiTanhActivation(d),
        )

        # graph attention head
        self.beta_proj = nn.Linear(nfilters_aa, nembedding_graph * nattentionheads_graph, bias=True)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.ones_(self.beta_proj.bias)
        self.self_attn_proj = nn.Linear(
            nfilters_aa, nembedding_graph * nattentionheads_graph, bias=True
        )
        nn.init.zeros_(self.self_attn_proj.weight)
        nn.init.zeros_(self.self_attn_proj.bias)
        self.cross_attn_proj = nn.Linear(
            nfilters_aa, nembedding_graph * nattentionheads_graph, bias=False
        )
        nn.init.zeros_(self.cross_attn_proj.weight)
        self.node_proj = nn.Linear(nfilters_aa, nattentionheads_graph * nfilters_graph, bias=True)

        self.graph_neighborhood = ScanNetLocalNeighborhood(
            Kmax=K_graph,
            coordinates=("distance", "ZdotZ", "ZdotDelta", "index_distance"),
            self_neighborhood=True,
            index_distance_max=16,
        )
        # coordinates_graph=['distance','ZdotZ','ZdotDelta','index_distance'] contributes
        # dims 1(distance)+1(ZdotZ)+2(ZdotDelta: deltadotz & zdotdelta)+1(index_distance) = 5.
        self.graph_gaussian = ScanNetGaussianKernel(d=5, N=N_graph, covariance_type="full")
        self.edges_graph = nn.Linear(N_graph, nembedding_graph, bias=False)
        self.graph_attention = ScanNetAttentionLayer(
            nfeatures_graph=nembedding_graph,
            nheads=nattentionheads_graph,
            nfeatures_output=nfilters_graph,
            Kmax=K_graph,
            self_attention=True,
            beta=True,
        )

        self.classifier = nn.Linear(nattentionheads_graph * nfilters_graph, 2)

    def forward(
        self,
        frame_indices_aa,
        attributes_aa,
        sequence_indices_aa,
        point_clouds_aa,
        frame_indices_atom,
        attributes_atom,
        sequence_indices_atom,
        point_clouds_atom,
    ):
        embedded_attributes_aa = self.embed_aa_act(self.embed_aa_bn(self.embed_aa(attributes_aa)))

        embedded_attributes_atom = self.embed_atom(attributes_atom)
        SCAN_filters_atom, _frames_atom = self.atom_embedding(
            point_clouds_atom, frame_indices_atom, embedded_attributes_atom, sequence_indices_atom
        )

        pooling_attention = self.pooling_attention(SCAN_filters_atom)
        pooling_features = self.pooling_features(SCAN_filters_atom)

        pooling_mask, pooling_attention_local, pooling_features_local = self.pooling_neighborhood(
            None,
            sequence_indices_aa,
            None,
            sequence_indices_atom,
            [pooling_attention, pooling_features],
        )
        pooling_mask = 1 - pooling_mask  # matches Lambda(lambda x: 1 - x) in the original

        SCAN_filters_atom_aggregated_input, _attn_coeffs = self.pooling_attention_layer(
            None, None, pooling_attention_local, pooling_features_local, pooling_mask
        )
        SCAN_filters_atom_aggregated_activity = self.pooled_act(
            self.pooled_bn(SCAN_filters_atom_aggregated_input)
        )

        all_embedded_attributes_aa = torch.cat(
            [embedded_attributes_aa, SCAN_filters_atom_aggregated_activity], dim=-1
        )

        SCAN_filters_aa, frames_aa = self.aa_embedding(
            point_clouds_aa, frame_indices_aa, all_embedded_attributes_aa, sequence_indices_aa
        )

        beta = F.relu(self.beta_proj(SCAN_filters_aa))
        self_attention = self.self_attn_proj(SCAN_filters_aa)
        cross_attention = self.cross_attn_proj(SCAN_filters_aa)
        node_features = self.node_proj(SCAN_filters_aa)

        graph_weights, attention_local, node_features_local = self.graph_neighborhood(
            frames_aa,
            sequence_indices_aa,
            frames_aa,
            sequence_indices_aa,
            [cross_attention, node_features],
        )

        embedded_graph_weights = self.graph_gaussian(graph_weights)
        embedded_graph_weights = self.edges_graph(embedded_graph_weights)

        graph_attention_output, _attention_coefficients = self.graph_attention(
            beta, self_attention, attention_local, node_features_local, embedded_graph_weights
        )

        classifier_output = F.softmax(self.classifier(graph_attention_output), dim=-1)
        return classifier_output


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo).
# ---------------------------------------------------------------------------

_LMAX_AA = 20
_LMAX_ATOM = 9 * _LMAX_AA
_NFEAT_ATOM = 4


def build_scannet():
    torch.manual_seed(0)
    model = ScanNet(Lmax_aa=_LMAX_AA)
    model.eval()
    return model


def example_input_scannet():
    torch.manual_seed(0)
    B = 1
    L_aa = _LMAX_AA
    L_atom = _LMAX_ATOM

    frame_indices_aa = torch.randint(0, L_aa, (B, L_aa, 3), dtype=torch.long)
    attributes_aa = F.one_hot(torch.randint(0, 20, (B, L_aa)), num_classes=20).float()
    sequence_indices_aa = torch.arange(L_aa).view(1, L_aa, 1).expand(B, -1, -1).clone()
    point_clouds_aa = torch.randn(B, L_aa, 3)

    frame_indices_atom = torch.randint(0, L_atom, (B, L_atom, 3), dtype=torch.long)
    attributes_atom = torch.randint(1, _NFEAT_ATOM + 1, (B, L_atom), dtype=torch.long)
    sequence_indices_atom = torch.randint(0, L_aa, (B, L_atom, 1), dtype=torch.long)
    point_clouds_atom = torch.randn(B, L_atom, 3)

    return (
        frame_indices_aa,
        attributes_aa,
        sequence_indices_aa,
        point_clouds_aa,
        frame_indices_atom,
        attributes_atom,
        sequence_indices_atom,
        point_clouds_atom,
    )


MENAGERIE_ENTRIES = [
    ("ScanNet", "build_scannet", "example_input_scannet", 2022, MENAGERIE_ZOO),
]
