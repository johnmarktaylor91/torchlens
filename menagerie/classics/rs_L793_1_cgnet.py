# SOURCE: vendored from https://github.com/coarse-graining/cgnet @ master
# Files: cgnet/network/nnet.py (CGnet), cgnet/feature/geometry.py (Geometry),
#        cgnet/feature/feature.py (GeometryFeature, SchnetFeature),
#        cgnet/feature/schnet_utils.py (CGBeadEmbedding, InteractionBlock,
#        ContinuousFilterConvolution), cgnet/feature/utils.py (GaussianRBF,
#        ShiftedSoftplus, LinearLayer).
#
# CGnet (Wang, Olsson, Wehmeyer, Perez, Charron, de Fabritiis, Noe, Clementi,
# 2019, ACS Central Science, https://doi.org/10.1021/acscentsci.8b00913) is a
# feedforward neural network that learns coarse-grained molecular force fields
# from Cartesian coordinate data via force matching: a roto-translationally
# invariant GeometryFeature layer (pairwise distances/angles/dihedrals) feeds
# a small MLP that predicts a scalar potential energy, and forces are obtained
# by autograd w.r.t. the input coordinates.
#
# CGSchNet (Husic, Charron, Lemm, Wang, et al., 2020, J. Chem. Phys.; the
# SchNet-featurized successor to plain CGnet, built from the SAME cgnet repo)
# swaps the GeometryFeature for a SchnetFeature -- a CGBeadEmbedding lookup
# table followed by stacked SchNet InteractionBlocks (continuous-filter
# convolutions over a GaussianRBF-expanded distance basis) -- before the same
# CGnet energy head and force-via-autograd tail.
#
# No architecture was altered. The four dispatch-style Geometry helper
# methods (arccos/cross/norm/sum/etc.) are vendored as-is; only the 'numpy'
# branches are exercised by this file's 'torch' method usage, so the
# unreachable numpy branches are inert but left in place to match the real
# source. The real repo's priors (HarmonicLayer/RepulsionLayer/ZscoreLayer,
# used for physics-informed regularization) and its data/statistics/training
# utilities are dropped -- they are optional regularizers and data-loading
# concerns, not part of the CGnet/CGSchNet network architecture itself
# (CGnet's own __init__ makes `priors=None` the default).

import numpy as np
import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# cgnet/feature/geometry.py
# ---------------------------------------------------------------------------


class Geometry:
    """Helper class to calculate distances, angles, and dihedrals with a
    unified, vectorized framework depending on whether pytorch or numpy is
    used."""

    def __init__(self, method="torch", device=torch.device("cpu")):
        self.device = device
        if method not in ["torch", "numpy"]:
            raise RuntimeError("Allowed methods are 'torch' and 'numpy'")
        self.method = method

        if method == "torch":
            self.bool = torch.bool
            self.float32 = torch.float32
        elif self.method == "numpy":
            self.bool = np.bool_
            self.float32 = np.float32

    def check_for_nans(self, object, name=None):
        if name is None:
            name = ""
        if self.isnan(object).any():
            raise ValueError("Nan found in {}. Check your coordinates!)".format(name))

    def check_array_vs_tensor(self, object, name=None):
        if name is None:
            name = ""
        if self.method == "numpy" and type(object) is not np.ndarray:
            raise ValueError(
                "Input argument {} must be type np.ndarray for Geometry(method='numpy')".format(
                    name
                )
            )
        if self.method == "torch" and type(object) is not torch.Tensor:
            raise ValueError(
                "Input argument {} must be type torch.Tensor for Geometry(method='torch')".format(
                    name
                )
            )

    def get_distance_indices(self, n_beads, backbone_inds=[], backbone_map=None):
        pair_order = []
        adj_backbone_pairs = []
        for increment in range(1, n_beads):
            for i in range(n_beads - increment):
                pair_order.append((i, i + increment))
                if len(backbone_inds) > 0:
                    if backbone_map[i + increment] - backbone_map[i] == 1:
                        adj_backbone_pairs.append((i, i + increment))
        return pair_order, adj_backbone_pairs

    def get_vectorize_inputs(self, inds, data):
        if len(np.unique([len(feat) for feat in inds])) > 1:
            raise ValueError("All features must be the same length.")
        feat_length = len(inds[0])

        ind_list = [[feat[i] for feat in inds] for i in range(feat_length)]

        dist_list = [
            data[:, ind_list[i + 1], :] - data[:, ind_list[i], :] for i in range(feat_length - 1)
        ]

        if len(dist_list) == 1:
            dist_list = dist_list[0]

        return dist_list

    def get_distances(self, distance_inds, data, norm=True):
        self.check_array_vs_tensor(data, "data")
        distances = self.get_vectorize_inputs(distance_inds, data)
        if norm:
            distances = self.norm(distances, axis=2)
        self.check_for_nans(distances, "distances")
        return distances

    def get_angles(self, angle_inds, data, clip=True):
        self.check_array_vs_tensor(data, "data")

        base, offset = self.get_vectorize_inputs(angle_inds, data)
        base = base * -1

        angles = (
            self.sum(base * offset, axis=2) / self.norm(base, axis=2) / self.norm(offset, axis=2)
        )

        if clip:
            angles = self.arccos(self.clip(angles, lower_bound=-1.0, upper_bound=1.0))

        self.check_for_nans(angles, "angles")

        return angles

    def get_dihedrals(self, dihed_inds, data):
        self.check_array_vs_tensor(data, "data")

        angle_inds = np.concatenate(
            [[(f[i], f[i + 1], f[i + 2]) for i in range(2)] for f in dihed_inds]
        )
        base, offset = self.get_vectorize_inputs(angle_inds, data)
        offset_2 = base[:, 1:]

        cross_product_adj = self.cross(base, offset, axis=2)
        cp_base = cross_product_adj[:, :-1, :]
        cp_offset = cross_product_adj[:, 1:, :]

        plane_vector = self.cross(cp_offset, offset_2, axis=2)

        dihedral_cosines = (
            self.sum(cp_base[:, ::2] * cp_offset[:, ::2], axis=2)
            / self.norm(cp_base[:, ::2], axis=2)
            / self.norm(cp_offset[:, ::2], axis=2)
        )

        dihedral_sines = (
            self.sum(cp_base[:, ::2] * plane_vector[:, ::2], axis=2)
            / self.norm(cp_base[:, ::2], axis=2)
            / self.norm(plane_vector[:, ::2], axis=2)
        )

        self.check_for_nans(dihedral_cosines, "dihedral cosines")
        self.check_for_nans(dihedral_sines, "dihedral sines")

        return dihedral_cosines, dihedral_sines

    def get_redundant_distance_mapping(self, pair_order):
        import scipy.spatial.distance

        pairwise_dist_inds = [
            zipped_pair[1]
            for zipped_pair in sorted([z for z in zip(pair_order, np.arange(len(pair_order)))])
        ]
        map_matrix = scipy.spatial.distance.squareform(pairwise_dist_inds)
        map_matrix = map_matrix[~np.eye(map_matrix.shape[0], dtype=bool)].reshape(
            map_matrix.shape[0], -1
        )
        return map_matrix

    def _torch_eye(self, n, dtype):
        if dtype == torch.bool:
            return torch.BoolTensor(np.eye(n, dtype=bool))
        else:
            return torch.eye(n, dtype=dtype)

    def arccos(self, x):
        if self.method == "torch":
            return torch.acos(x)
        elif self.method == "numpy":
            return np.arccos(x)

    def cross(self, x, y, axis):
        if self.method == "torch":
            return torch.cross(x, y, dim=axis)
        elif self.method == "numpy":
            return np.cross(x, y, axis=axis)

    def norm(self, x, axis):
        if self.method == "torch":
            return torch.norm(x, dim=axis)
        elif self.method == "numpy":
            return np.linalg.norm(x, axis=axis)

    def sum(self, x, axis):
        if self.method == "torch":
            return torch.sum(x, dim=axis)
        elif self.method == "numpy":
            return np.sum(x, axis=axis)

    def arange(self, n):
        if self.method == "torch":
            return torch.arange(n)
        elif self.method == "numpy":
            return np.arange(n)

    def tile(self, x, shape):
        if self.method == "torch":
            return x.repeat(*shape)
        elif self.method == "numpy":
            return np.tile(x, shape)

    def eye(self, n, dtype):
        if self.method == "torch":
            return self._torch_eye(n, dtype).to(self.device)
        elif self.method == "numpy":
            return np.eye(n, dtype=dtype)

    def ones(self, shape, dtype):
        if self.method == "torch":
            return torch.ones(*shape, dtype=dtype).to(self.device)
        elif self.method == "numpy":
            return np.ones(shape, dtype=dtype)

    def to_type(self, x, dtype):
        if self.method == "torch":
            return x.type(dtype)
        elif self.method == "numpy":
            return x.astype(dtype)

    def clip(self, x, lower_bound, upper_bound, out=None):
        if self.method == "torch":
            return torch.clamp(x, min=lower_bound, max=upper_bound, out=out)
        elif self.method == "numpy":
            return np.clip(x, a_min=lower_bound, a_max=upper_bound, out=out)

    def isnan(self, x):
        if self.method == "torch":
            return torch.isnan(x)
        elif self.method == "numpy":
            return np.isnan(x)


# ---------------------------------------------------------------------------
# cgnet/feature/utils.py
# ---------------------------------------------------------------------------


class ShiftedSoftplus(nn.Module):
    """Shifted softplus (SSP) activation function."""

    def __init__(self):
        super(ShiftedSoftplus, self).__init__()

    def forward(self, input_tensor):
        return nn.functional.softplus(input_tensor) - np.log(2.0)


class _AbstractRBFLayer(nn.Module):
    def __init__(self):
        super(_AbstractRBFLayer, self).__init__()

    def __len__(self):
        raise NotImplementedError()

    def forward(self, distances):
        raise NotImplementedError()


class GaussianRBF(_AbstractRBFLayer):
    """Radial basis function (RBF) layer using Gaussian expansions."""

    def __init__(
        self,
        low_cutoff=0.0,
        high_cutoff=5.0,
        n_gaussians=50,
        variance=1.0,
        normalize_output=False,
    ):
        super(GaussianRBF, self).__init__()
        self.register_buffer("centers", torch.linspace(low_cutoff, high_cutoff, n_gaussians))
        self.variance = variance
        self.normalize_output = normalize_output

    def __len__(self):
        return len(self.centers)

    def forward(self, distances, distance_mask=None):
        dist_centered_squared = torch.pow(distances.unsqueeze(dim=3) - self.centers, 2)
        gaussian_exp = torch.exp(-(0.5 / self.variance) * dist_centered_squared)

        if self.normalize_output:
            basis_sum = torch.sum(gaussian_exp, dim=3)
            gaussian_exp = gaussian_exp / basis_sum[:, :, :, None]

        if distance_mask is not None:
            gaussian_exp = gaussian_exp * distance_mask[:, :, :, None]
        return gaussian_exp


def LinearLayer(
    d_in,
    d_out,
    bias=True,
    activation=None,
    dropout=0,
    weight_init="xavier",
    weight_init_args=None,
    weight_init_kwargs=None,
):
    """Linear layer function returning a list of torch.nn.Module instances."""
    seq = [nn.Linear(d_in, d_out, bias=bias)]
    if activation:
        if isinstance(activation, nn.Module):
            seq += [activation]
        else:
            raise TypeError("Activation {} is not a valid torch.nn.Module".format(str(activation)))
    if dropout:
        seq += [nn.Dropout(dropout)]

    with torch.no_grad():
        if weight_init == "xavier":
            torch.nn.init.xavier_uniform_(seq[0].weight)
        if weight_init == "identity":
            torch.nn.init.eye_(seq[0].weight)
        if weight_init not in ["xavier", "identity", None]:
            if isinstance(weight_init, int) or isinstance(weight_init, float):
                torch.nn.init.constant_(seq[0].weight, weight_init)
            if callable(weight_init):
                if weight_init_args is None:
                    weight_init_args = []
                if weight_init_kwargs is None:
                    weight_init_kwargs = []
                weight_init(seq[0].weight, *weight_init_args, **weight_init_kwargs)
            else:
                raise RuntimeError('Unknown weight initialization "{}"'.format(str(weight_init)))
    return seq


# ---------------------------------------------------------------------------
# cgnet/feature/schnet_utils.py
# ---------------------------------------------------------------------------


class CGBeadEmbedding(nn.Module):
    """Simple embedding class for coarse-grain beads."""

    def __init__(self, n_embeddings, embedding_dim):
        super(CGBeadEmbedding, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=n_embeddings, embedding_dim=embedding_dim, padding_idx=0
        )

    def forward(self, embedding_property):
        return self.embedding(embedding_property)


class ContinuousFilterConvolution(nn.Module):
    """Continuous-filter convolution block as described by Schuett et al. (2018)."""

    def __init__(self, n_gaussians, n_filters, activation=None, normalization_layer=None):
        super(ContinuousFilterConvolution, self).__init__()
        if activation is None:
            activation = ShiftedSoftplus()
        filter_layers = LinearLayer(n_gaussians, n_filters, bias=True, activation=activation)
        filter_layers += LinearLayer(n_filters, n_filters, bias=True)
        self.filter_generator = nn.Sequential(*filter_layers)

        if normalization_layer:
            self.normalization_layer = normalization_layer
        else:
            self.normalization_layer = None

    def forward(self, features, rbf_expansion, neighbor_list, neighbor_mask, bead_mask=None):
        conv_filter = self.filter_generator(rbf_expansion)

        n_batch, n_beads, n_neighbors = neighbor_list.size()

        neighbor_list = neighbor_list.reshape(-1, n_beads * n_neighbors, 1)
        neighbor_list = neighbor_list.expand(-1, -1, features.size(2))

        neighbor_features = torch.gather(features, 1, neighbor_list)
        neighbor_features = neighbor_features.reshape(n_batch, n_beads, n_neighbors, -1)
        conv_features = neighbor_features * conv_filter

        conv_features = conv_features * neighbor_mask[:, :, :, None]
        aggregated_features = torch.sum(conv_features, dim=2)

        if bead_mask is not None:
            aggregated_features = aggregated_features * bead_mask[:, :, None]

        if self.normalization_layer is not None:
            return self.normalization_layer(aggregated_features)
        else:
            return aggregated_features


class InteractionBlock(nn.Module):
    """SchNet interaction block as described by Schuett et al. (2018)."""

    def __init__(
        self,
        n_inputs,
        n_gaussians,
        n_filters,
        activation=None,
        normalization_layer=None,
    ):
        super(InteractionBlock, self).__init__()
        if activation is None:
            activation = ShiftedSoftplus()

        self.initial_dense = nn.Sequential(
            *LinearLayer(n_inputs, n_filters, bias=False, activation=None)
        )
        self.inital_dense = self.initial_dense

        self.cfconv = ContinuousFilterConvolution(
            n_gaussians=n_gaussians,
            n_filters=n_filters,
            activation=activation,
            normalization_layer=normalization_layer,
        )
        output_layers = LinearLayer(n_filters, n_filters, bias=True, activation=activation)
        output_layers += LinearLayer(n_filters, n_filters, bias=True, activation=None)
        self.output_dense = nn.Sequential(*output_layers)

    def forward(self, features, rbf_expansion, neighbor_list, neighbor_mask, bead_mask=None):
        init_feature_output = self.initial_dense(features)
        conv_output = self.cfconv(
            init_feature_output, rbf_expansion, neighbor_list, neighbor_mask, bead_mask=bead_mask
        )
        output_features = self.output_dense(conv_output)
        return output_features


# ---------------------------------------------------------------------------
# cgnet/feature/feature.py
# ---------------------------------------------------------------------------


class GeometryFeature(nn.Module):
    """Featurization of coarse-grained beads into pairwise distances,
    angles, and dihedrals."""

    def __init__(self, feature_tuples=None, n_beads=None, device=torch.device("cpu")):
        super(GeometryFeature, self).__init__()

        self._n_beads = n_beads
        self.device = device
        self.geometry = Geometry(method="torch", device=self.device)
        if feature_tuples != "all_backbone":
            if feature_tuples is not None:
                _temp_dict = dict(zip(feature_tuples, np.arange(len(feature_tuples))))
                if len(_temp_dict) < len(feature_tuples):
                    feature_tuples = list(_temp_dict.keys())

                self.feature_tuples = feature_tuples
                if (
                    np.min([len(feat) for feat in feature_tuples]) < 2
                    or np.max([len(feat) for feat in feature_tuples]) > 4
                ):
                    raise ValueError("Custom features must be tuples of length 2, 3, or 4.")

                self._distance_pairs = [feat for feat in feature_tuples if len(feat) == 2]
                self._angle_trips = [feat for feat in feature_tuples if len(feat) == 3]
                self._dihedral_quads = [feat for feat in feature_tuples if len(feat) == 4]
            else:
                raise RuntimeError(
                    "Either a list of feature tuples or 'all_backbone' must be specified."
                )
        else:
            if n_beads is None:
                raise RuntimeError("Must specify n_beads if feature_tuples is 'all_backone'.")
            self._distance_pairs, _ = self.geometry.get_distance_indices(n_beads)
            if n_beads > 2:
                self._angle_trips = [(i, i + 1, i + 2) for i in range(n_beads - 2)]
            else:
                self._angle_trips = []
            if n_beads > 3:
                self._dihedral_quads = [(i, i + 1, i + 2, i + 3) for i in range(n_beads - 3)]
            else:
                self._dihedral_quads = []
            self.feature_tuples = self._distance_pairs + self._angle_trips + self._dihedral_quads

    def compute_distances(self, data):
        self.distances = self.geometry.get_distances(self._distance_pairs, data, norm=True)
        self.descriptions["Distances"] = self._distance_pairs

    def compute_angles(self, data):
        self.angles = self.geometry.get_angles(self._angle_trips, data)
        self.descriptions["Angles"] = self._angle_trips

    def compute_dihedrals(self, data):
        (self.dihedral_cosines, self.dihedral_sines) = self.geometry.get_dihedrals(
            self._dihedral_quads, data
        )
        self.descriptions["Dihedral_cosines"] = self._dihedral_quads
        self.descriptions["Dihedral_sines"] = self._dihedral_quads

    def forward(self, data):
        self._coordinates = data
        self.n_beads = data.shape[1]
        if self._n_beads is not None and self.n_beads != self._n_beads:
            raise ValueError("n_beads passed to __init__ does not match n_beads in data.")
        if np.max([np.max(bead) for bead in self.feature_tuples]) > self.n_beads - 1:
            raise ValueError("Bead index in at least one feature is out of range.")

        self.descriptions = {}
        self.description_order = []
        out = torch.Tensor([]).to(self.device)

        if len(self._distance_pairs) > 0:
            self.compute_distances(data)
            out = torch.cat((out, self.distances), dim=1)
            self.description_order.append("Distances")
        else:
            self.distances = torch.Tensor([])

        if len(self._angle_trips) > 0:
            self.compute_angles(data)
            out = torch.cat((out, self.angles), dim=1)
            self.description_order.append("Angles")
        else:
            self.angles = torch.Tensor([])

        if len(self._dihedral_quads) > 0:
            self.compute_dihedrals(data)
            out = torch.cat((out, self.dihedral_cosines, self.dihedral_sines), dim=1)
            self.description_order.append("Dihedral_cosines")
            self.description_order.append("Dihedral_sines")
        else:
            self.dihedral_cosines = torch.Tensor([])
            self.dihedral_sines = torch.Tensor([])

        return out


class SchnetFeature(nn.Module):
    """Wrapper class for radial basis function layer, continuous filter
    convolution, and interaction block connecting feature inputs and
    outputs residually."""

    def __init__(
        self,
        feature_size,
        embedding_layer,
        rbf_layer,
        n_beads,
        activation=None,
        calculate_geometry=None,
        neighbor_cutoff=None,
        normalization_layer=None,
        n_interaction_blocks=1,
        share_weights=False,
        share_batchnorm_parameters=False,
        device=torch.device("cpu"),
    ):
        super(SchnetFeature, self).__init__()
        if activation is None:
            activation = ShiftedSoftplus()
        self.device = device
        self.geometry = Geometry(method="torch", device=self.device)
        self.embedding_layer = embedding_layer
        self.rbf_layer = rbf_layer
        basis_size = len(rbf_layer)

        if share_weights:
            self.interaction_blocks = nn.ModuleList(
                [
                    InteractionBlock(
                        feature_size,
                        basis_size,
                        feature_size,
                        activation=activation,
                        normalization_layer=normalization_layer,
                    )
                ]
                * n_interaction_blocks
            )
        else:
            self.interaction_blocks = nn.ModuleList(
                [
                    InteractionBlock(
                        feature_size,
                        basis_size,
                        feature_size,
                        activation=activation,
                        normalization_layer=normalization_layer,
                    )
                    for _ in range(n_interaction_blocks)
                ]
            )

        self.neighbor_cutoff = neighbor_cutoff
        self.calculate_geometry = calculate_geometry
        if self.calculate_geometry:
            pass
        else:
            self._distance_pairs, _ = self.geometry.get_distance_indices(n_beads, [], [])
            self.redundant_distance_mapping = None

    def forward(self, in_features, embedding_property):
        if self.calculate_geometry:
            n_beads = embedding_property.size()[1]
            self._distance_pairs, _ = self.geometry.get_distance_indices(n_beads, [], [])
            redundant_distance_mapping = self.geometry.get_redundant_distance_mapping(
                self._distance_pairs
            )
            distances = self.geometry.get_distances(self._distance_pairs, in_features, norm=True)
            distances = distances[:, redundant_distance_mapping]
        else:
            distances = in_features

        n_beads = embedding_property.size()[1]
        neighbors, neighbor_mask = self.geometry.get_neighbors(
            distances, cutoff=self.neighbor_cutoff
        )
        neighbor_mask = neighbor_mask.to(self.device)
        neighbors = neighbors.to(self.device)

        rbf_expansion = self.rbf_layer(distances=distances)

        features = self.embedding_layer(embedding_property)
        bead_mask = torch.clamp(embedding_property, min=0, max=1).float()

        for interaction_block in self.interaction_blocks:
            interaction_features = interaction_block(
                features=features,
                rbf_expansion=rbf_expansion,
                neighbor_list=neighbors,
                neighbor_mask=neighbor_mask,
                bead_mask=bead_mask,
            )
            features = features + interaction_features

        return features


def _geometry_get_neighbors(self, distances, cutoff=None):
    """Bound onto Geometry below -- vendored from geometry.py's
    Geometry.get_neighbors, which was omitted from the excerpt above for
    brevity but is required by SchnetFeature.forward."""
    self.check_array_vs_tensor(distances, "distances")

    n_frames, n_beads, n_neighbors = distances.shape

    neighbors = self.tile(self.arange(n_beads), (n_frames, n_beads, 1))
    neighbors = neighbors[:, ~self.eye(n_beads, dtype=self.bool)].reshape(
        n_frames, n_beads, n_neighbors
    )

    if cutoff is not None:
        neighbor_mask = distances < cutoff
        neighbor_mask = self.to_type(neighbor_mask, self.float32)
    else:
        neighbor_mask = self.ones((n_frames, n_beads, n_neighbors), dtype=self.float32)

    return neighbors, neighbor_mask


Geometry.get_neighbors = _geometry_get_neighbors


# ---------------------------------------------------------------------------
# cgnet/network/nnet.py
# ---------------------------------------------------------------------------


class CGnet(nn.Module):
    """CGnet neural network class. Predicts a coarse-grained potential of
    mean force (PMF) and its associated force field via autograd."""

    def __init__(self, arch, criterion, feature=None, priors=None):
        super(CGnet, self).__init__()
        self.arch = nn.Sequential(*arch)
        self.priors = None
        self.criterion = criterion
        self.feature = feature

    def forward(self, coordinates, embedding_property=None):
        if self.feature:
            if embedding_property is not None:
                feature_output = self.feature(coordinates, embedding_property)
            else:
                feature_output = self.feature(coordinates)
            energy = self.arch(feature_output)
        else:
            feature_output = coordinates
            energy = self.arch(feature_output)

        if len(energy.size()) == 3 and isinstance(self.feature, SchnetFeature):
            bead_mask = torch.clamp(embedding_property, min=0, max=1).float()
            masked_energy = energy * bead_mask[:, :, None]
            energy = torch.sum(masked_energy, dim=-2)

        force = torch.autograd.grad(
            -torch.sum(energy), coordinates, create_graph=True, retain_graph=True
        )
        return energy, force[0]


# ---------------------------------------------------------------------------
# build_ / example_input_
# ---------------------------------------------------------------------------


def build_cgnet():
    """Plain CGnet: GeometryFeature (distances/angles/dihedrals over 5
    backbone beads, as in the alanine-dipeptide example from the paper)
    feeding a small Tanh MLP energy head."""
    torch.manual_seed(0)
    n_beads = 5
    feature = GeometryFeature(feature_tuples="all_backbone", n_beads=n_beads)
    n_feats = len(feature.feature_tuples) + 2  # distances/angles + 2x dihedral (cos,sin)
    # GeometryFeature output size = n_distances + n_angles + 2*n_dihedrals
    n_distances = n_beads * (n_beads - 1) // 2
    n_angles = max(n_beads - 2, 0)
    n_dihedrals = max(n_beads - 3, 0)
    n_feats = n_distances + n_angles + 2 * n_dihedrals

    arch = LinearLayer(n_feats, 32, activation=nn.Tanh())
    arch += LinearLayer(32, 32, activation=nn.Tanh())
    arch += LinearLayer(32, 1, activation=None)

    model = CGnet(arch, criterion=None, feature=feature)
    model.eval()
    return model


def example_input_cgnet():
    torch.manual_seed(0)
    n_frames, n_beads = 2, 5
    coordinates = torch.randn(n_frames, n_beads, 3, requires_grad=True)
    return (coordinates,)


def build_cgschnet():
    """CGSchNet: SchnetFeature (bead embedding + GaussianRBF + stacked SchNet
    InteractionBlocks, calculate_geometry=True so raw coordinates are the
    input) feeding the same CGnet energy head."""
    torch.manual_seed(0)
    n_beads = 5
    feature_size = 16
    n_gaussians = 10

    embedding_layer = CGBeadEmbedding(n_embeddings=6, embedding_dim=feature_size)
    rbf_layer = GaussianRBF(low_cutoff=0.0, high_cutoff=5.0, n_gaussians=n_gaussians)

    feature = SchnetFeature(
        feature_size=feature_size,
        embedding_layer=embedding_layer,
        rbf_layer=rbf_layer,
        n_beads=n_beads,
        calculate_geometry=True,
        n_interaction_blocks=2,
    )

    arch = LinearLayer(feature_size, 32, activation=nn.Tanh())
    arch += LinearLayer(32, 1, activation=None)

    model = CGnet(arch, criterion=None, feature=feature)
    model.eval()
    return model


def example_input_cgschnet():
    torch.manual_seed(0)
    n_frames, n_beads = 2, 5
    coordinates = torch.randn(n_frames, n_beads, 3, requires_grad=True)
    embedding_property = torch.randint(1, 6, (n_frames, n_beads), dtype=torch.long)
    return (coordinates, embedding_property)


MENAGERIE_ENTRIES = [
    ("CGnet", "build_cgnet", "example_input_cgnet", 2019, MENAGERIE_ZOO),
    ("CGSchNet", "build_cgschnet", "example_input_cgschnet", 2020, MENAGERIE_ZOO),
]
