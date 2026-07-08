# SOURCE: vendored from https://github.com/rssrwn/semla-flow @ main
#   - semlaflow/util/functional.py  (tensor/graph utility functions used by the model)
#   - semlaflow/models/semla.py     (the SE(3)-equivariant molecular generator architecture)
#
# Both files are reproduced verbatim below, inlined into one module (imports hoisted to the
# top; `functional.py`'s own `import torch` / `from typing import Union` / scipy import are
# deduped against this file's header). The only mechanical edit to `semla.py`'s body is
# rewriting its 3 call sites from `smolF.zero_com(...)` / `smolF.edges_from_nodes(...)` to
# direct `zero_com(...)` / `edges_from_nodes(...)` calls, since `functional.py`'s functions
# are now inlined at module scope instead of imported through a `smolF` namespace alias --
# no architectural code was changed. The entry point is `SemlaGenerator`, constructed with an
# `EquiInvDynamics` core, matching the real construction in `semlaflow/train.py::build_model`
# (arch == "semla").

from typing import Union

import numpy as np
import torch
from scipy.spatial.transform import Rotation
import copy
from abc import ABC, abstractmethod

_T = torch.Tensor
TupleRot = tuple[float, float, float]


# --- begin vendored semlaflow/util/functional.py (verbatim below the import header) ---
_T = torch.Tensor
TupleRot = tuple[float, float, float]


# *************************************************************************************************
# ********************************** Tensor Util Functions ****************************************
# *************************************************************************************************


def pad_tensors(tensors: list[_T], pad_dim: int = 0) -> _T:
    """Pad a list of tensors with zeros

    All dimensions other than pad_dim must have the same shape. A single tensor is returned with the batch dimension
    first, where the batch dimension is the length of the tensors list.

    Args:
        tensors (list[torch.Tensor]): List of tensors
        pad_dim (int): Dimension on tensors to pad. All other dimensions must be the same size.

    Returns:
        torch.Tensor: Batched, padded tensor, if pad_dim is 0 then shape [B, L, *] where L is length of longest tensor.
    """

    if pad_dim != 0:
        # TODO
        raise NotImplementedError()

    padded = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
    return padded


# TODO replace with tensor version below
def one_hot_encode(indices: list[int], vocab_size: int) -> _T:
    """Create one-hot encodings from a list of indices

    Args:
        indices (list[int]): List of indices into one-hot vectors
        vocab_size (int): Length of returned vectors

    Returns:
        torch.Tensor: One-hot encoded vectors, shape [L, vocab_size] where L is length of indices list
    """

    one_hots = torch.zeros((len(indices), vocab_size), dtype=torch.int64)

    for batch_idx, vocab_idx in enumerate(indices):
        one_hots[batch_idx, vocab_idx] = 1

    return one_hots


# TODO test
def one_hot_encode_tensor(indices: _T, vocab_size: int) -> _T:
    """Create one-hot encodings from indices

    Args:
        indices (torch.Tensor): Indices into one-hot vectors, shape [*, L]
        vocab_size (int): Length of returned vectors

    Returns:
        torch.Tensor: One-hot encoded vectors, shape [*, L, vocab_size]
    """

    one_hot_shape = (*indices.shape, vocab_size)
    one_hots = torch.zeros(one_hot_shape, dtype=torch.int64, device=indices.device)
    one_hots.scatter_(-1, indices.unsqueeze(-1), 1)
    return one_hots


def pairwise_concat(t: _T) -> _T:
    """Concatenates two representations from all possible pairings in dimension 1

    Computes all possible pairs of indices into dimension 1 and concatenates whatever representation they have in
    higher dimensions. Note that all higher dimensions will be flattened. The output will have its shape for
    dimension 1 duplicated in dimension 2.

    Example:
    Input shape [100, 16, 128]
    Output shape [100, 16, 16, 256]
    """

    idx_pairs = torch.cartesian_prod(*((torch.arange(t.shape[1]),) * 2))
    output = t[:, idx_pairs].view(t.shape[0], t.shape[1], t.shape[1], -1)
    return output


def segment_sum(data, segment_ids, num_segments):
    """Computes the sum of data elements that are in each segment

    The inputs must have shapes that look like the following:
    data [batch_size, seq_length, num_features]
    segment_ids [batch_size, seq_length], must contain integers

    Then the output will have the following shape:
    output [batch_size, num_segments, num_features]
    """

    err_msg = "data and segment_ids must have the same shape in the first two dimensions"
    assert data.shape[0:2] == segment_ids.shape[0:2], err_msg

    result_shape = (data.shape[0], num_segments, data.shape[2])
    result = data.new_full(result_shape, 0)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, -1, data.shape[2])
    result.scatter_add_(1, segment_ids, data)
    return result


# *************************************************************************************************
# ******************************* Functions for handling edges ************************************
# *************************************************************************************************


def adj_from_node_mask(node_mask, self_connect=False):
    """Creates an edge mask from a given node mask assuming all nodes are fully connected excluding self-connections

    Args:
        node_mask (torch.Tensor): Node mask tensor, shape [batch_size, num_nodes], 1 for real node 0 otherwise
        self_connect (bool): Whether to include self connections in the adjacency

    Returns:
        torch.Tensor: Adjacency tensor, shape [batch_size, num_nodes, num_nodes], 1 for real edge 0 otherwise
    """

    num_nodes = node_mask.size()[1]

    # Matrix mult gives us an outer product on the node mask, which is an edge mask
    mask = node_mask.float()
    adjacency = torch.bmm(mask.unsqueeze(2), mask.unsqueeze(1))
    adjacency = adjacency.long()

    # Set diagonal connections
    node_idxs = torch.arange(num_nodes)
    self_mask = node_mask if self_connect else torch.zeros_like(node_mask)
    adjacency[:, node_idxs, node_idxs] = self_mask

    return adjacency


def _pad_edges(edges, max_edges, value=0):
    """Add fake edges to an edge tensor so that the shape matches max_edges

    Args:
        edges (torch.Tensor): Unbatched edge tensor, shape [num_edges, 2], each element is a node index for the edge
        max_edges (int): The number of edges the output tensor should have
        value (int): Padding value, default 0

    Returns:
        (torch.Tensor, torch.Tensor): Tuple of padded edge tensor and padding mask. Shapes [max_edges, 2] for edge
                tensor and [max_edges] for mask. Mask is one for pad elements, 0 otherwise.
    """

    num_edges = edges.size(0)
    mask_kwargs = {"dtype": torch.int64, "device": edges.device}

    if num_edges > max_edges:
        raise ValueError(
            "Number of edges in edge tensor to be padded cannot be greater than max_edges."
        )

    add_edges = max_edges - num_edges

    if add_edges == 0:
        pad_mask = torch.zeros(num_edges, **mask_kwargs)
        return edges, pad_mask

    pad = (0, 0, 0, add_edges)
    padded = torch.nn.functional.pad(edges, pad, mode="constant", value=value)

    zeros_mask = torch.zeros(num_edges, **mask_kwargs)
    ones_mask = torch.ones(add_edges, **mask_kwargs)
    pad_mask = torch.cat((zeros_mask, ones_mask), dim=0)

    return padded, pad_mask


# TODO change callers to use bonds_from_adj
def edges_from_adj(adj_matrix):
    """Flatten an adjacency matrix into a 1D edge representation

    Args:
        adj_matrix (torch.Tensor): Batched adjacency matrix, shape [batch_size, num_nodes, num_nodes]. It can contain
                any non-zero integer for connected nodes but must be 0 for unconnected nodes.

    Returns:
        A tuple of the edge tensor and the edge mask tensor. The edge tensor has shape [batch_size, max_num_edges, 2]
        and the mask [batch_size, max_num_edges]. The mask contains 1 for real edges, 0 otherwise.
    """

    adj_ones = torch.zeros_like(adj_matrix).int()
    adj_ones[adj_matrix != 0] = 1

    # Pad each batch element by a seperate amount so that they can all be packed into a tensor
    # It might be possible to do this in batch form without iterating, but for now this will do
    num_edges = adj_ones.sum(dim=(1, 2)).tolist()
    edge_tuples = list(adj_matrix.nonzero()[:, 1:].split(num_edges))
    padded = [_pad_edges(edges, max(num_edges), value=0) for edges in edge_tuples]

    # Unravel the padded tuples and stack them into batches
    edge_tuples_padded, pad_masks = tuple(zip(*padded))
    edges = torch.stack(edge_tuples_padded).long()
    edges = (edges[:, :, 0], edges[:, :, 1])
    edge_mask = (torch.stack(pad_masks) == 0).long()
    return edges, edge_mask


# TODO test and merge with edges_from_adj
def bonds_from_adj(adj_matrix, lower_tri=True):
    """Flatten an adjacency matrix into a 1D edge representation

    Args:
        adj_matrix (torch.Tensor): Adjacency matrix, can be batched or not, shape [batch_size, num_nodes, num_nodes].
            Each item in the matrix corrsponds to the bond type and will be placed into index 2 on dim 1 in bonds.
        lower_tri (bool): Whether to only consider bonds which sit in the lower triangular of adj_matrix.

    Returns:
        An bond list tensor, shape [batch_size, num_bonds, 3]. If an item is a padding bond index 2 on the last
            dimension will be 0.
    """

    batched = True
    if len(adj_matrix.shape) == 2:
        adj_matrix = adj_matrix.unsqueeze(0)
        batched = False

    if lower_tri:
        adj_matrix = torch.tril(adj_matrix, diagonal=-1)

    bonds = []
    for adj in list(adj_matrix):
        bond_indices = adj.nonzero()
        bond_types = adj[bond_indices[:, 0], bond_indices[:, 1]]
        bond_list = torch.cat((bond_indices, bond_types.unsqueeze(-1)), dim=-1)
        bonds.append(bond_list)

    # Bonds will be padded with 0s so the bond type will tell whether the bond is real or not
    bonds = pad_tensors(bonds, pad_dim=0)
    if not batched:
        bonds = bonds.squeeze(0)

    return bonds


def adj_from_edges(edge_indices: _T, edge_types: _T, n_nodes: int, symmetric: bool = False):
    """Create adjacency matrix from a list of edge indices and types

    If an edge pair appears multiple times with different edge types, the adj element for that edge is undefined.

    Args:
        edge_indices (torch.Tensor): Edge list tensor, shape [n_edges, 2]. Pairs of (from_idx, to_idx).
        edge_types (torch.Tensor): Edge types, shape either [n_edges] or [n_edges, edge_types].
        n_nodes (int): Number of nodes in the adjacency matrix. This must be >= to the max node index in edges.
        symmetric (bool): Whether edges are considered symmetric. If True the adjacency matrix will also be symmetric,
                otherwise only the exact node indices within edges will be used to create the adjacency.

    Returns:
        torch.Tensor: Adjacency matrix tensor, shape [n_nodes, n_nodes] or
                [n_nodes, n_nodes, edge_types] if distributions over edge types are provided.
    """

    assert len(edge_indices.shape) == 2
    assert edge_indices.shape[0] == edge_types.shape[0]
    assert edge_indices.size(1) == 2

    adj_dist = len(edge_types.shape) == 2

    edge_indices = edge_indices.long()
    edge_types = edge_types.float() if adj_dist else edge_types.long()

    if adj_dist:
        shape = (n_nodes, n_nodes, edge_types.size(-1))
        adj = torch.zeros(shape, device=edge_indices.device, dtype=torch.float)

    else:
        shape = (n_nodes, n_nodes)
        adj = torch.zeros(shape, device=edge_indices.device, dtype=torch.long)

    from_indices = edge_indices[:, 0]
    to_indices = edge_indices[:, 1]

    adj[from_indices, to_indices] = edge_types
    if symmetric:
        adj[to_indices, from_indices] = edge_types

    return adj


def edges_from_nodes(coords, k=None, node_mask=None, edge_format="adjacency"):
    """Constuct edges from node coords

    Connects a node to its k nearest nodes. If k is None then connects each node to all its neighbours. A node is
    never connected to itself.

    Args:
        coords (torch.Tensor): Node coords, shape [batch_size, num_nodes, 3]
        k (int): Number of neighbours to connect each node to, None means connect to all nodes except itself
        node_mask (torch.Tensor): Node mask, shape [batch_size, num_nodes], 1 for real nodes 0 otherwise
        edge_format (str): Edge format, should be either 'adjacency' or 'list'

    Returns:
        If format is 'adjacency' this returns an adjacency matrix, shape [batch_size, num_nodes, num_nodes] which
        contains 1 for connected nodes and 0 otherwise. Note that if a value for k is provided the adjacency matrix
        may not be symmetric and should always be used s.t. 'from nodes' are in dim 1 and 'to nodes' are in dim 2.

        If format is 'list' this returns the tuple (edges, edge mask), edges is also a two-tuple of tensors, each of
        shape [batch_size, num_edges], specifying node indices for each edge. The edge mask has shape
        [batch_size, num_edges] and contains 1 for 'real' edges and 0 otherwise.
    """

    if edge_format not in ["adjacency", "list"]:
        raise ValueError(f"Unrecognised edge format '{edge_format}'")

    adj_format = edge_format == "adjacency"
    batch_size, num_nodes, _ = coords.size()

    # If node mask is None all nodes are real
    if node_mask is None:
        node_mask = torch.ones((batch_size, num_nodes), device=coords.device, dtype=torch.int64)

    adj_matrix = adj_from_node_mask(node_mask)

    if k is not None:
        # Find k closest nodes for each node
        dists = calc_distances(coords)
        dists[adj_matrix == 0] = float("inf")
        _, best_idxs = dists.topk(k, dim=2, largest=False)

        # Adjust adj matrix to only have k connections per node
        k_adj_matrix = torch.zeros_like(adj_matrix)
        batch_idxs = torch.arange(batch_size).view(-1, 1, 1).expand(-1, num_nodes, k)
        node_idxs = torch.arange(num_nodes).view(1, -1, 1).expand(batch_size, -1, k)
        k_adj_matrix[batch_idxs, node_idxs, best_idxs] = 1

        # Ensure that there are no connections to fake nodes
        k_adj_matrix[adj_matrix == 0] = 0
        adj_matrix = k_adj_matrix

    if adj_format:
        return adj_matrix

    edges, edge_mask = edges_from_adj(adj_matrix)
    return edges, edge_mask


def gather_edge_features(pairwise_feats, adj_matrix):
    """Gather edge features for each node from pairwise features using the adjacency matrix

    All 'from nodes' (dimension 1 on the adj matrix) must have the same number of edges to 'to nodes'. Practically
    this means that the number of non-zero elements in dimension 2 of the adjacency matrix must always be the same.

    Args:
        pairwise_feats (torch.Tensor): Pairwise features tensor, shape [batch_size, num_nodes, num_nodes, num_feats]
        adj_matrix (torch.Tensor): Batched adjacency matrix, shape [batch_size, num_nodes, num_nodes]. It can contain
                any non-zero integer for connected nodes but must be 0 for unconnected nodes.

    Returns:
        torch.Tensor: Dense feature matrix, shape [batch_size, num_nodes, edges_per_node, num_feats]
    """

    # In case some of the connections don't use 1, create a 1s adjacency matrix
    adj_ones = torch.zeros_like(adj_matrix).int()
    adj_ones[adj_matrix != 0] = 1

    num_neighbours = adj_ones.sum(dim=2)
    feats_per_node = num_neighbours[0, 0].item()

    assert (num_neighbours == feats_per_node).all(), (
        "All nodes must have the same number of connections"
    )

    if len(pairwise_feats.size()) == 3:
        batch_size, num_nodes, _ = pairwise_feats.size()
        pairwise_feats = pairwise_feats.unsqueeze(3)

    elif len(pairwise_feats.size()) == 4:
        batch_size, num_nodes, _, _ = pairwise_feats.size()

    # nonzero() orders indices lexicographically with the last index changing the fastest, so we can reshape the
    # indices into a dense form with nodes along the outer axis and features along the inner
    gather_idxs = adj_ones.nonzero()[:, 2].reshape((batch_size, num_nodes, feats_per_node))
    batch_idxs = torch.arange(batch_size).view(-1, 1, 1)
    node_idxs = torch.arange(num_nodes).view(1, -1, 1)
    dense_feats = pairwise_feats[batch_idxs, node_idxs, gather_idxs, :]
    if dense_feats.size(-1) == 1:
        return dense_feats.squeeze(-1)

    return dense_feats


# *************************************************************************************************
# ********************************* Geometric Util Functions **************************************
# *************************************************************************************************


# TODO rename? Maybe also merge with inter_distances
# TODO test unbatched and coord sets inputs
def calc_distances(coords, edges=None, sqrd=False, eps=1e-6):
    """Computes distances between connected nodes

    Takes an optional edges argument. If edges is None this will calculate distances between all nodes and return the
    distances in a batched square matrix [batch_size, num_nodes, num_nodes]. If edges is provided the distances are
    returned for each edge in a batched 1D format [batch_size, num_edges].

    Args:
        coords (torch.Tensor): Coordinate tensor, shape [batch_size, num_nodes, 3]
        edges (tuple): Two-tuple of connected node indices, each tensor has shape [batch_size, num_edges]
        sqrd (bool): Whether to return the squared distances
        eps (float): Epsilon to add before taking the square root for numical stability in the gradients

    Returns:
        torch.Tensor: Distances tensor, the shape depends on whether edges is provided (see above).
    """

    # TODO add checks

    # Create fake batch dim if unbatched
    unbatched = False
    if len(coords.size()) == 2:
        coords = coords.unsqueeze(0)
        unbatched = True

    if edges is None:
        coord_diffs = coords.unsqueeze(-2) - coords.unsqueeze(-3)
        sqrd_dists = torch.sum(coord_diffs * coord_diffs, dim=-1)

    else:
        edge_is, edge_js = edges
        batch_index = torch.arange(coords.size(0)).unsqueeze(1)
        coord_diffs = coords[batch_index, edge_js, :] - coords[batch_index, edge_is, :]
        sqrd_dists = torch.sum(coord_diffs * coord_diffs, dim=2)

    sqrd_dists = sqrd_dists.squeeze(0) if unbatched else sqrd_dists

    if sqrd:
        return sqrd_dists

    return torch.sqrt(sqrd_dists + eps)


def inter_distances(coords1, coords2, sqrd=False, eps=1e-6):
    # TODO add checks and doc

    # Create fake batch dim if unbatched
    unbatched = False
    if len(coords1.size()) == 2:
        coords1 = coords1.unsqueeze(0)
        coords2 = coords2.unsqueeze(0)
        unbatched = True

    coord_diffs = coords1.unsqueeze(2) - coords2.unsqueeze(1)
    sqrd_dists = torch.sum(coord_diffs * coord_diffs, dim=3)
    sqrd_dists = sqrd_dists.squeeze(0) if unbatched else sqrd_dists

    if sqrd:
        return sqrd_dists

    return torch.sqrt(sqrd_dists + eps)


def calc_com(coords, node_mask=None):
    """Calculates the centre of mass of a pointcloud

    Args:
        coords (torch.Tensor): Coordinate tensor, shape [*, num_nodes, 3]
        node_mask (torch.Tensor): Mask for points, shape [*, num_nodes], 1 for real node, 0 otherwise

    Returns:
        torch.Tensor: CoM of pointclouds with imaginary nodes excluded, shape [*, 1, 3]
    """

    node_mask = torch.ones_like(coords[..., 0]) if node_mask is None else node_mask

    assert node_mask.shape == coords[..., 0].shape

    num_nodes = node_mask.sum(dim=-1)
    real_coords = coords * node_mask.unsqueeze(-1)
    com = real_coords.sum(dim=-2) / num_nodes.unsqueeze(-1)
    return com.unsqueeze(-2)


def zero_com(coords, node_mask=None):
    """Sets the centre of mass for a batch of pointclouds to zero for each pointcloud

    Args:
        coords (torch.Tensor): Coordinate tensor, shape [*, num_nodes, 3]
        node_mask (torch.Tensor): Mask for points, shape [*, num_nodes], 1 for real node, 0 otherwise

    Returns:
        torch.Tensor: CoM-free coordinates, where imaginary nodes are excluded from CoM calculation
    """

    com = calc_com(coords, node_mask=node_mask)
    shifted = coords - com
    return shifted


def standardise_coords(coords, node_mask=None):
    """Convert coords into a standard normal distribution

    This will first remove the centre of mass from all pointclouds in the batch, then calculate the (biased) variance
    of the shifted coords and use this to produce a standard normal distribution.

    Args:
        coords (torch.Tensor):  Coordinate tensor, shape [batch_size, num_nodes, 3]
        node_mask (torch.Tensor): Mask for points, shape [batch_size, num_nodes], 1 for real node, 0 otherwise

    Returns:
        Tuple[torch.Tensor, float]: The standardised coords and the variance of the original coords
    """

    if node_mask is None:
        node_mask = torch.ones_like(coords)[:, :, 0]

    coord_idxs = node_mask.nonzero()
    real_coords = coords[coord_idxs[:, 0], coord_idxs[:, 1], :]

    variance = torch.var(real_coords, correction=0)
    std_dev = torch.sqrt(variance)

    result = (coords / std_dev) * node_mask.unsqueeze(2)
    return result, std_dev.item()


def rotate(coords: torch.Tensor, rotation: Union[Rotation, TupleRot]):
    """Rotate coordinates for a single molecule

    Args:
        coords (torch.Tensor): Unbatched coordinate tensor, shape [num_atoms, 3]
        rotation (Union[Rotation, Tuple[float, float, float]]): Can be either a scipy Rotation object or a tuple of
                rotation values in radians, (x, y, z). These are treated as extrinsic rotations. See the scipy docs
                (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html) for info.

    Returns:
        torch.Tensor: Rotated coordinates
    """

    if not isinstance(rotation, Rotation):
        rotation = Rotation.from_euler("xyz", rotation)

    device = coords.device
    coords = coords.cpu().numpy()

    rotated = rotation.apply(coords)
    return torch.tensor(rotated, device=device)


def cartesian_to_spherical(coords):
    sqrd_dists = (coords * coords).sum(dim=-1)
    radii = torch.sqrt(sqrd_dists)
    inclination = torch.acos(coords[..., 2] / radii).unsqueeze(2)
    azimuth = torch.atan2(coords[..., 1], coords[..., 0]).unsqueeze(2)
    return torch.cat((radii.unsqueeze(2), inclination, azimuth), dim=-1)


# *************************************************************************************************
# ************************************** Util Classes *********************************************
# *************************************************************************************************


class SparseFeatures:
    def __init__(self, dense, idxs):
        assert len(dense.size()) == 3
        assert dense.size() == idxs.size()

        batch_size, num_nodes, num_feats = dense.size()

        self.bs = batch_size
        self.num_nodes = num_nodes
        self.num_feats = num_feats

        self._dense = dense
        self._idxs = idxs

    @staticmethod
    def from_sparse(sparse_feats, adj_matrix, feats_per_node):
        err_msg = "adj_matrix must have feats_per_node ones in each row"
        assert sparse_feats.size() == adj_matrix.size(), (
            "sparse_feats and adj_matrix must have the same shape"
        )
        assert adj_matrix.size()[1] == adj_matrix.size()[2], "adj_matrix must be square"
        assert (adj_matrix.sum(dim=2) == feats_per_node).all().item(), err_msg

        batch_size, num_nodes, _ = adj_matrix.size()
        feat_idxs = adj_matrix.nonzero()[:, 2].reshape((batch_size, num_nodes, feats_per_node))
        dense_feats = torch.gather(sparse_feats, 2, feat_idxs)
        return SparseFeatures(dense_feats, feat_idxs)

    @staticmethod
    def from_dense(dense_feats, idxs):
        return SparseFeatures(dense_feats, idxs)

    def to_tensor(self):
        sparse_matrix = torch.zeros(
            (self.bs, self.num_nodes, self.num_nodes), device=self._dense.device
        )
        sparse_matrix.scatter_(2, self._idxs, self._dense)
        return sparse_matrix

    def mult(self, other):
        if isinstance(other, (int, float)):
            return self.from_dense(self._dense * other, self._idxs)

        if not torch.is_tensor(other):
            raise TypeError("Object to multiply by must be an int, float or torch.Tensor")

        assert other.size() == (self.bs, self.num_nodes, self.num_nodes)

        other_dense = torch.gather(other, 2, self._idxs)
        return self.from_dense(self._dense * other_dense, self._idxs)

    def matmul(self, other):
        if not torch.is_tensor(other):
            raise TypeError("Object to multiply by must be a torch.Tensor")

        assert tuple(other.size()[:2]) == (self.bs, self.num_nodes)

        # There doesn't seem to be an efficient implementation of sparse batched matmul available atm, so just do
        # regular matmul instead. We will still get some speed benefit from having lots of zeros.
        tensor = self.to_tensor()
        return torch.bmm(tensor, other)

    def softmax(self):
        dense_softmax = torch.softmax(self._dense, dim=2)
        return self.from_dense(dense_softmax, self._idxs)

    def dropout(self, p, train=False):
        dense_dropout = torch.dropout(self._dense, p, train=train)
        return self.from_dense(dense_dropout, self._idxs)

    def add(self, other):
        """Add a matrix only at elements which are not sparse in self"""

        assert len(other.size()) == 3

        other_dense = torch.gather(other, 2, self._idxs)
        return self.from_dense(self._dense + other_dense, self._idxs)

    def sum(self, dim=None):
        if dim == 1:
            return self.to_tensor().sum(dim=1)

        return self._dense.sum(dim=dim)


# --- end vendored semlaflow/util/functional.py ---


# --- begin vendored semlaflow/models/semla.py (verbatim; smolF.* rewritten to direct calls) ---
# (semla.py's own `import copy` / `from abc import ABC, abstractmethod` / `import numpy as
# np` / `import torch` lines are already hoisted into this file's shared header above and
# omitted here to avoid duplicate imports.)


def adj_to_attn_mask(adj_matrix, pos_inf=False):
    """Assumes adj_matrix is only 0s and 1s"""

    inf = float("inf") if pos_inf else float("-inf")
    attn_mask = torch.zeros_like(adj_matrix.float())
    attn_mask[adj_matrix == 0] = inf

    # Ensure nodes with no connections (fake nodes) don't have all -inf in the attn mask
    # Otherwise we would have problems when softmaxing
    n_nodes = adj_matrix.sum(dim=-1)
    attn_mask[n_nodes == 0] = 0.0

    return attn_mask


# *************************************************************************************************
# *********************************** Helper Classes **********************************************
# *************************************************************************************************


class CoordNorm(torch.nn.Module):
    """Coordinate normalisation layer for coordinate sets with inductive bias towards molecules

    This layer allows 4 different types of coordinate normalisation (defined in the norm argument):
        1. 'none' - The coordinates are zero-centred and multiplied by learnable weights
        2. 'gvp' - Coords are zero-centred, scaled by learnable weights and each is scaled by sqrt(n_sets) / ||x_i||_2
        3. 'length' - Coords are zero-centred, multiplied by learnable weights and scaled by 1 / avg vector length

    Note that 'length' provides the same coordinate normalisation that is commonly used in current models but adapted
    to multiple coordinate sets, thereby allowing easier comparison to existing approaches.
    """

    def __init__(self, n_coord_sets, norm="length", eps=1e-6):
        super().__init__()

        norm = "none" if norm is None else norm
        if norm not in ["none", "gvp", "length"]:
            raise ValueError(f"Unknown normalisation type '{norm}'")

        self.n_coord_sets = n_coord_sets
        self.norm = norm
        self.eps = eps

        self.set_weights = torch.nn.Parameter(torch.ones((1, n_coord_sets, 1, 1)))

    def forward(self, coord_sets, node_mask):
        """Apply coordinate normlisation layer

        Args:
            coord_sets (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            node_mask (torch.Tensor): Mask for nodes, shape [batch_size, n_sets, n_nodes], 1 for real, 0 otherwise

        Returns:
            torch.Tensor: Normalised coords, shape [batch_size, n_sets, n_nodes, 3]
        """

        # Zero the CoM in case it isn't already
        coord_sets = zero_com(coord_sets, node_mask)
        coord_sets = coord_sets * node_mask.unsqueeze(-1)

        n_atoms = node_mask.sum(dim=-1, keepdim=True)
        lengths = torch.linalg.vector_norm(coord_sets, dim=-1)

        if self.norm == "length":
            scaled_lengths = lengths.sum(dim=2, keepdim=True) / n_atoms
            coord_div = scaled_lengths.unsqueeze(-1) + self.eps

        elif self.norm == "gvp":
            coord_div = (lengths.unsqueeze(-1) + self.eps) / np.sqrt(self.n_coord_sets)

        else:
            coord_div = torch.ones_like(coord_sets)

        coord_sets = (coord_sets * self.set_weights) / coord_div
        coord_sets = coord_sets * node_mask.unsqueeze(-1)
        return coord_sets

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)


class EdgeMessages(torch.nn.Module):
    def __init__(self, d_model, d_message, d_out, n_coord_sets, d_ff=None, d_edge=None, eps=1e-6):
        super().__init__()

        edge_feats = 0 if d_edge is None else d_edge
        d_ff = d_out if d_ff is None else d_ff

        extra_feats = n_coord_sets + edge_feats
        in_feats = (d_message * 2) + extra_feats

        self.n_coord_sets = n_coord_sets
        self.d_edge = d_edge
        self.eps = eps

        self.coord_norm = CoordNorm(n_coord_sets, norm="none")
        self.node_norm = torch.nn.LayerNorm(d_model)
        self.edge_norm = torch.nn.LayerNorm(d_edge) if d_edge is not None else None

        self.node_proj = torch.nn.Linear(d_model, d_message)
        self.message_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_feats, d_ff),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(d_ff, d_out),
        )

    def forward(self, coords, node_feats, node_mask, edge_feats=None):
        """Compute edge messages

        Args:
            coords (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            node_feats (torch.Tensor): Node features, shape [batch_size, n_nodes, d_model]
            node_mask (torch.Tensor): Mask for nodes, shape [batch_size, n_sets, n_nodes], 1 for real, 0 otherwise
            edge_feats (torch.Tensor): Incoming edge features, shape [batch_size, n_nodes, n_nodes, d_edge]

        Returns:
            torch.Tensor: Edge messages tensor, shape [batch_size, n_nodes, n_nodes, d_out]
        """

        batch_size, n_nodes, _ = tuple(node_feats.shape)

        if edge_feats is not None and self.d_edge is None:
            raise ValueError(
                "edge_feats was provided but the model was initialised with d_edge as None."
            )

        if edge_feats is None and self.d_edge is not None:
            raise ValueError(
                "The model was initialised with d_edge but no edge feats were provided to forward fn."
            )

        node_feats = self.node_norm(node_feats)

        coords = self.coord_norm(coords, node_mask).flatten(0, 1)
        coord_dotprods = torch.bmm(coords, coords.transpose(1, 2))
        coord_feats = coord_dotprods.unflatten(0, (-1, self.n_coord_sets)).movedim(1, -1)

        # Project to smaller dimension and create pairwise node features
        node_feats = self.node_proj(node_feats)
        node_feats_start = node_feats.unsqueeze(2).expand(batch_size, n_nodes, n_nodes, -1)
        node_feats_end = node_feats.unsqueeze(1).expand(batch_size, n_nodes, n_nodes, -1)
        node_pairs = torch.cat((node_feats_start, node_feats_end), dim=-1)

        in_edge_feats = torch.cat((node_pairs, coord_feats), dim=3)
        if edge_feats is not None:
            edge_feats = self.edge_norm(edge_feats)
            in_edge_feats = torch.cat((in_edge_feats, edge_feats), dim=-1)

        return self.message_mlp(in_edge_feats)


class NodeAttention(torch.nn.Module):
    def __init__(self, d_model, n_attn_heads, d_attn=None):
        super().__init__()

        d_attn = d_model if d_attn is None else d_attn
        d_head = d_model // n_attn_heads

        if d_attn % n_attn_heads != 0:
            raise ValueError("n_attn_heads must divide d_model (or d_attn if provided) exactly.")

        self.d_model = d_model
        self.d_attn = d_attn
        self.n_attn_heads = n_attn_heads
        self.d_head = d_head

        self.feat_norm = torch.nn.LayerNorm(d_model)
        self.in_proj = torch.nn.Linear(d_model, d_attn)
        self.out_proj = torch.nn.Linear(d_attn, d_model)

    def forward(self, node_feats, messages, adj_matrix):
        """Accumulate edge messages to each node using attention-based message passing

        Args:
            node_feats (torch.Tensor): Node feature tensor, shape [batch_size, n_nodes, d_model]
            messages (torch.Tensor): Messages tensor, shape [batch_size, n_nodes, n_nodes, d_message]
            adj_matrix (torch.Tensor): Adjacency matrix, shape [batch_size, n_nodes, n_nodes]

        Returns:
            torch.Tensor: Accumulated node features, shape [batch_size, n_nodes, d_model]
        """

        attn_mask = adj_to_attn_mask(adj_matrix)
        messages = messages + attn_mask.unsqueeze(3)
        attentions = torch.softmax(messages, dim=2)

        node_feats = self.feat_norm(node_feats)
        proj_feats = self.in_proj(node_feats)
        head_feats = proj_feats.unflatten(-1, (self.n_attn_heads, self.d_head))

        # Put n_heads into the batch dim for both the features and the attentions
        # head_feats shape [B * n_heads, n_nodes, d_head]
        # attentions shape [B * n_heads, n_nodes, n_nodes]
        head_feats = head_feats.movedim(-2, 1).flatten(0, 1)
        attentions = attentions.movedim(-1, 1).flatten(0, 1)

        attn_out = torch.bmm(attentions, head_feats)

        # Apply variance preserving updates as proposed in GNN-VPA (https://arxiv.org/abs/2403.04747)
        weights = torch.sqrt((attentions**2).sum(dim=-1))
        attn_out = attn_out * weights.unsqueeze(-1)

        attn_out = attn_out.unflatten(0, (-1, self.n_attn_heads))
        attn_out = attn_out.movedim(1, -2).flatten(2, 3)
        return self.out_proj(attn_out)


class CoordAttention(torch.nn.Module):
    def __init__(self, n_coord_sets, proj_sets=None, coord_norm="length", eps=1e-6):
        super().__init__()

        proj_sets = n_coord_sets if proj_sets is None else proj_sets

        self.eps = eps

        self.coord_norm = CoordNorm(n_coord_sets, norm=coord_norm)
        self.coord_proj = torch.nn.Linear(n_coord_sets, proj_sets, bias=False)
        self.attn_proj = torch.nn.Linear(proj_sets, n_coord_sets, bias=False)

    def forward(self, coord_sets, messages, adj_matrix, node_mask):
        """Compute an attention update for coordinate sets

        Args:
            coord_sets (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            messages (torch.Tensor): Messages tensor, shape [batch_size, n_nodes, n_nodes, proj_sets]
            adj_matrix (torch.Tensor): Adjacency matrix, shape [batch_size, n_nodes, n_nodes]
            node_mask (torch.Tensor): Mask for nodes, shape [batch_size, n_sets, n_nodes], 1 for real, 0 otherwise

        Returns:
            torch.Tensor: Updated coordinate sets, shape [batch_size, n_sets, n_nodes, 3]
        """

        coord_sets = self.coord_norm(coord_sets, node_mask)
        proj_coord_sets = self.coord_proj(coord_sets.transpose(1, -1))

        # proj_coord_sets shape [B, 3, N, P]
        # norm_dists shape [B, 1, N, N, P]
        vec_dists = proj_coord_sets.unsqueeze(3) - proj_coord_sets.unsqueeze(2)
        lengths = torch.linalg.vector_norm(vec_dists, dim=1, keepdim=True)
        norm_dists = vec_dists / (lengths + self.eps)

        attn_mask = adj_to_attn_mask(adj_matrix)
        messages = messages + attn_mask.unsqueeze(3)
        attentions = torch.softmax(messages, dim=2)

        # Dim 1 is currently 1 on dists so we need to unsqueeze attentions
        updates = norm_dists * attentions.unsqueeze(1)
        updates = updates.sum(dim=3)

        # Apply variance preserving updates as proposed in GNN-VPA (https://arxiv.org/abs/2403.04747)
        weights = torch.sqrt((attentions**2).sum(dim=2))
        updates = updates * weights.unsqueeze(1)

        # updates shape [B, 3, N, P] -> [B, S, N, 3]
        return self.attn_proj(updates).transpose(1, -1)


class LengthsMLP(torch.nn.Module):
    def __init__(self, d_model, n_coord_sets, d_ff=None):
        super().__init__()

        d_ff = d_model * 4 if d_ff is None else d_ff

        self.node_ff = torch.nn.Sequential(
            torch.nn.Linear(d_model + n_coord_sets, d_ff),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(d_ff, d_model),
        )

    def forward(self, coord_sets, node_feats):
        """Pass data through the layer

        Assumes coords and node_feats have already been normalised

        Args:
            coord_sets (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            node_feats (torch.Tensor): Node feature tensor, shape [batch_size, n_nodes, d_model]

        Returns:
            torch.Tensor: Updated node features, shape [batch_size, n_nodes, d_model]
        """

        lengths = torch.linalg.vector_norm(coord_sets, dim=-1).movedim(1, -1)
        in_feats = torch.cat((node_feats, lengths), dim=2)
        return self.node_ff(in_feats)


class EquivariantMLP(torch.nn.Module):
    def __init__(self, d_model, n_coord_sets, proj_sets=None):
        super().__init__()

        proj_sets = n_coord_sets if proj_sets is None else proj_sets

        self.node_proj = torch.nn.Sequential(
            torch.nn.Linear(d_model, proj_sets),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(proj_sets, proj_sets),
        )
        self.coord_proj = torch.nn.Linear(n_coord_sets, proj_sets, bias=False)
        self.attn_proj = torch.nn.Linear(proj_sets, n_coord_sets, bias=False)

    def forward(self, coord_sets, node_feats):
        """Pass data through the layer

        Assumes coords and node_feats have already been normalised

        Args:
            coord_sets (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            node_feats (torch.Tensor): Node feature tensor, shape [batch_size, n_nodes, d_model]

        Returns:
            torch.Tensor: Updated coord_sets, shape [batch_size, n_sets, n_nodes, 3]
        """

        # inv_feats shape [B, 1, N, P]
        # proj_sets shape [B, 3, N, P]
        inv_feats = self.node_proj(node_feats).unsqueeze(1)
        proj_sets = self.coord_proj(coord_sets.transpose(1, -1))

        # Outer product with invariant features is equivariant, then sum over original coord sets
        attentions = inv_feats.unsqueeze(-1) * proj_sets.unsqueeze(-2)
        attentions = attentions.sum(-1)

        return self.attn_proj(attentions).transpose(1, -1)


class NodeFeedForward(torch.nn.Module):
    def __init__(self, d_model, n_coord_sets, d_ff=None, proj_sets=None, coord_norm="length"):
        super().__init__()

        self.node_norm = torch.nn.LayerNorm(d_model)
        self.coord_norm = CoordNorm(n_coord_sets, norm=coord_norm)

        self.invariant_mlp = LengthsMLP(d_model, n_coord_sets, d_ff=d_ff)
        self.equivariant_mlp = EquivariantMLP(d_model, n_coord_sets, proj_sets=proj_sets)

    def forward(self, coord_sets, node_feats, node_mask):
        """Pass data through the layer

        Args:
            coord_sets (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            node_feats (torch.Tensor): Node feature tensor, shape [batch_size, n_nodes, d_model]
            node_mask (torch.Tensor): Mask for nodes, shape [batch_size, n_sets, n_nodes], 1 for real, 0 otherwise

        Returns:
            torch.Tensor, torch.Tensor: Updates to coords and node features
        """

        node_feats = self.node_norm(node_feats)
        coord_sets = self.coord_norm(coord_sets, node_mask)

        out_node_feats = self.invariant_mlp(coord_sets, node_feats)
        out_coord_sets = self.equivariant_mlp(coord_sets, node_feats)

        return out_coord_sets, out_node_feats


class BondRefine(torch.nn.Module):
    def __init__(self, d_model, d_message, d_edge, d_ff=None):
        super().__init__()

        d_ff = d_message if d_ff is None else d_ff
        in_feats = (2 * d_message) + d_edge + 2

        self.coord_norm = CoordNorm(1, norm="none")
        self.node_norm = torch.nn.LayerNorm(d_model)
        self.edge_norm = torch.nn.LayerNorm(d_edge)

        self.node_proj = torch.nn.Linear(d_model, d_message)
        self.message_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_feats, d_ff),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(d_ff, d_edge),
        )

    def forward(self, coords, node_feats, node_mask, edge_feats):
        """Refine the bond predictions with a message passing layer that only updates bonds

        Args:
            coords (torch.Tensor): Coordinate tensor without coord sets, shape [batch_size, n_nodes, 3]
            node_feats (torch.Tensor): Node feature tensor, shape [batch_size, n_nodes, d_model]
            node_mask (torch.Tensor): Mask for nodes, shape [batch_size, n_nodes], 1 for real, 0 otherwise
            edge_feats (torch.Tensor): Current edge features, shape [batch_size, n_nodes, n_nodes, d_edge]

        Returns:
            torch.Tensor: Bond predictions tensor, shape [batch_size, n_nodes, n_nodes, n_bond_types]
        """

        assert len(coords.shape) == 3

        batch_size, n_nodes, _ = tuple(node_feats.shape)

        # Calculate distances and dot products
        coords = self.coord_norm(coords.unsqueeze(1), node_mask.unsqueeze(1)).squeeze(1)
        coord_diffs = coords.unsqueeze(2) - coords.unsqueeze(1)
        dists = (coord_diffs * coord_diffs).sum(dim=-1).unsqueeze(-1)
        coord_dotprods = torch.bmm(coords, coords.transpose(1, 2)).unsqueeze(-1)

        # Project to smaller dimension and create pairwise node features
        node_feats = self.node_proj(self.node_norm(node_feats))
        node_feats_i = node_feats.unsqueeze(2).expand(batch_size, n_nodes, n_nodes, -1)
        node_feats_j = node_feats.unsqueeze(1).expand(batch_size, n_nodes, n_nodes, -1)
        node_pairs = torch.cat((node_feats_i, node_feats_j), dim=-1)

        edge_feats = self.edge_norm(edge_feats)
        in_feats = torch.cat((node_pairs, dists, coord_dotprods, edge_feats), dim=3)
        return self.message_mlp(in_feats)


# *************************************************************************************************
# ********************************** Equivariant Layers *******************************************
# *************************************************************************************************


class EquiMessagePassingLayer(torch.nn.Module):
    def __init__(
        self,
        d_model,
        d_message,
        n_coord_sets,
        n_attn_heads=None,
        d_message_hidden=None,
        d_edge_in=None,
        d_edge_out=None,
        coord_norm="length",
        eps=1e-6,
    ):
        super().__init__()

        n_attn_heads = d_message if n_attn_heads is None else n_attn_heads
        if d_model != ((d_model // n_attn_heads) * n_attn_heads):
            raise ValueError(
                f"n_attn_heads must exactly divide d_model, got {n_attn_heads} and {d_model}"
            )

        self.d_model = d_model
        self.d_message = d_message
        self.n_coord_sets = n_coord_sets
        self.n_attn_heads = n_attn_heads
        self.d_message_hidden = d_message_hidden
        self.d_edge_in = d_edge_in
        self.d_edge_out = d_edge_out
        self.d_coord_message = n_coord_sets
        self.eps = eps

        d_ff = d_model * 4
        d_attn = d_model
        d_message_out = n_attn_heads + self.d_coord_message
        d_message_out = d_message_out + d_edge_out if d_edge_out is not None else d_message_out

        if d_edge_in is not None:
            self.edge_feat_norm = torch.nn.LayerNorm(d_edge_in)

        self.node_ff = NodeFeedForward(
            d_model,
            n_coord_sets,
            d_ff=d_ff,
            proj_sets=d_message,
            coord_norm=coord_norm,
        )
        self.message_ff = EdgeMessages(
            d_model,
            d_message,
            d_message_out,
            n_coord_sets,
            d_ff=d_message_hidden,
            d_edge=d_edge_in,
            eps=eps,
        )
        self.coord_attn = CoordAttention(
            n_coord_sets, self.d_coord_message, coord_norm=coord_norm, eps=eps
        )
        self.node_attn = NodeAttention(d_model, n_attn_heads, d_attn=d_attn)

    @property
    def hparams(self):
        return {
            "d_model": self.d_model,
            "d_message": self.d_message,
            "n_coord_sets": self.n_coord_sets,
            "n_attn_heads": self.n_attn_heads,
            "d_message_hidden": self.d_message_hidden,
        }

    def forward(self, coords, node_feats, adj_matrix, node_mask, edge_feats=None):
        """Pass data through the layer

        Args:
            coords (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            node_feats (torch.Tensor): Node features, shape [batch_size, n_nodes, d_model]
            adj_matrix (torch.Tensor): Adjacency matrix, shape [batch_size, n_nodes, n_nodes]
            node_mask (torch.Tensor): Mask for nodes, shape [batch_size, n_sets, n_nodes], 1 for real, 0 otherwise
            edge_feats (torch.Tensor): Incoming edge features, shape [batch_size, n_nodes, n_nodes, d_edge_in]

        Returns:
            Either a two-tuple of the new node coordinates and the new node features, or a three-tuple of the new
            node coords, new node features and new edge features.
        """

        if edge_feats is not None and self.d_edge_in is None:
            raise ValueError(
                "edge_feats was provided but the model was initialised with d_edge_in as None."
            )

        if edge_feats is None and self.d_edge_in is not None:
            raise ValueError(
                "The model was initialised with d_edge_in but no edge feats were provided to forward."
            )

        coord_updates, node_updates = self.node_ff(coords, node_feats, node_mask)
        coords = coords + coord_updates
        node_feats = node_feats + node_updates

        messages = self.message_ff(coords, node_feats, node_mask, edge_feats=edge_feats)
        node_messages = messages[:, :, :, : self.n_attn_heads]
        coord_messages = messages[
            :, :, :, self.n_attn_heads : (self.n_attn_heads + self.d_coord_message)
        ]

        node_feats = node_feats + self.node_attn(node_feats, node_messages, adj_matrix)
        coords = coords + self.coord_attn(coords, coord_messages, adj_matrix, node_mask)

        if self.d_edge_out is not None:
            edge_out = messages[:, :, :, (self.n_attn_heads + self.d_coord_message) :]
            edge_out = edge_feats + edge_out if edge_feats is not None else edge_out
            return coords, node_feats, edge_out

        return coords, node_feats


# *************************************************************************************************
# ************************************* Dynamics Models *******************************************
# *************************************************************************************************


class EquiInvDynamics(torch.nn.Module):
    def __init__(
        self,
        d_model,
        d_message,
        n_coord_sets,
        n_layers,
        n_attn_heads=None,
        d_message_hidden=None,
        d_edge=None,
        bond_refine=True,
        self_cond=False,
        coord_norm="length",
        eps=1e-6,
    ):
        super().__init__()

        extra_layers = 2 if d_edge is not None else 0
        if extra_layers > n_layers:
            raise ValueError("n_layers is too small.")

        n_attn_heads = d_message if n_attn_heads is None else n_attn_heads
        if d_model != ((d_model // n_attn_heads) * n_attn_heads):
            raise ValueError(
                f"n_attn_heads must exactly divide d_model, got {n_attn_heads} and {d_model}"
            )

        self._hparams = {
            "d_model": d_model,
            "d_message": d_message,
            "n_coord_sets": n_coord_sets,
            "n_layers": n_layers,
            "n_attn_heads": n_attn_heads,
            "d_message_hidden": d_message_hidden,
            "d_edge": d_edge,
            "bond_refine": bond_refine,
            "self_cond": self_cond,
            "coord_norm": coord_norm,
            "eps": eps,
        }

        self.d_model = d_model
        self.n_coord_sets = n_coord_sets
        self.d_edge = d_edge
        self.bond_refine = bond_refine and d_edge is not None
        self.self_cond = self_cond

        core_layer = EquiMessagePassingLayer(
            d_model,
            d_message,
            n_coord_sets,
            n_attn_heads=n_attn_heads,
            d_message_hidden=d_message_hidden,
            coord_norm=coord_norm,
            eps=eps,
        )
        layers = self._get_clones(core_layer, n_layers - extra_layers)

        if d_edge is not None:
            # Pass d_message_hidden as None so that these layers will have the same feats as their output
            in_layer = EquiMessagePassingLayer(
                d_model,
                d_message,
                n_coord_sets,
                n_attn_heads=n_attn_heads,
                d_message_hidden=None,
                d_edge_in=d_edge,
                coord_norm=coord_norm,
                eps=eps,
            )
            out_layer = EquiMessagePassingLayer(
                d_model,
                d_message,
                n_coord_sets,
                n_attn_heads=n_attn_heads,
                d_message_hidden=None,
                d_edge_out=d_edge,
                coord_norm=coord_norm,
                eps=eps,
            )
            layers = [in_layer] + layers + [out_layer]

        self.layers = torch.nn.ModuleList(layers)

        self.final_ff_block = NodeFeedForward(d_model, n_coord_sets, coord_norm=coord_norm)
        self.coord_norm = CoordNorm(n_coord_sets, norm=coord_norm)
        self.feat_norm = torch.nn.LayerNorm(d_model)

        in_coord_sets = 2 if self_cond else 1
        self.coord_proj = torch.nn.Linear(in_coord_sets, n_coord_sets, bias=False)
        self.coord_head = torch.nn.Linear(n_coord_sets, 1, bias=False)

        if d_edge is not None:
            self.bond_norm = torch.nn.LayerNorm(d_edge)

        if self.bond_refine:
            self.refine_layer = BondRefine(d_model, d_message, d_edge)

    @property
    def hparams(self):
        return self._hparams

    def forward(
        self, coords, inv_feats, adj_matrix, atom_mask=None, edge_feats=None, cond_coords=None
    ):
        """Generate molecular coordinates and atom features

        Args:
            coords (torch.Tensor): Input coordinates, shape [batch_size, n_atoms, 3]
            inv_feats (torch.Tensor): Invariant atom features, shape [batch_size, n_atoms, d_model]
            adj_matrix (torch.Tensor): Adjacency matrix, shape [batch_size, n_atoms, n_atoms], 1 for connected
            atom_mask (torch.Tensor, Optional): Mask for fake atoms, shape [batch_size, n_atoms], 1 for real atoms
            edge_feats (torch.Tensor, Optional): In edge features, shape [batch_size, n_nodes, n_nodes, d_edge]
            cond_coords (torch.Tensor, Optional): Conditional coords, shape [batch_size, n_nodes, 3]

        Returns:
            (coords, atom feats, edge feats)
            All torch.Tensor, shapes:
                Coordinates [batch_size, n_atoms, 3],
                Atom feats [batch_size, n_atoms, d_model]
                Edge feats [batch_size, n_atoms, n_atoms, d_edge]
        """

        if edge_feats is not None and self.d_edge is None:
            raise ValueError(
                "edge_feats was provided but the model was initialised with d_edge as None."
            )

        if edge_feats is None and self.d_edge is not None:
            raise ValueError(
                "The model was initialised with d_edge but no edge feats were provided to forward."
            )

        if cond_coords is not None and not self.self_cond:
            raise ValueError(
                "cond_coords was provided but the model was initialised with self_cond as False."
            )

        if cond_coords is None and self.self_cond:
            raise ValueError(
                "The model was initialsed with self_cond but cond_coords was not provided."
            )

        # Project single coord set into a multiple learnable coord sets, while maintaining equivariance
        coords = (
            torch.stack((coords, cond_coords)) if cond_coords is not None else coords.unsqueeze(0)
        )
        coords = self.coord_proj(coords.movedim(0, -1)).movedim(-1, 1)

        atom_mask = atom_mask.unsqueeze(1).expand(-1, self.n_coord_sets, -1)
        coords = coords * atom_mask.unsqueeze(-1)

        # Update coords and node feats using the model layers
        for layer in self.layers:
            out = layer(coords, inv_feats, adj_matrix, atom_mask, edge_feats=edge_feats)
            if len(out) == 2:
                coords, inv_feats = out
                edge_feats = None

            elif len(out) == 3:
                coords, inv_feats, edge_feats = out

        # Apply a final feedforward block and project coord sets to single coord set
        coords, inv_feats = self.final_ff_block(coords, inv_feats, atom_mask)
        out_coords = self.coord_norm(coords, atom_mask)
        out_coords = self.coord_head(out_coords.transpose(1, -1))
        out_coords = out_coords.transpose(1, -1).squeeze(1)

        if self.bond_refine:
            atom_mask = atom_mask[:, 0, :]
            edge_feats = self.refine_layer(out_coords, inv_feats, atom_mask, edge_feats)

        inv_feats = self.feat_norm(inv_feats)

        if self.d_edge is None:
            return out_coords, inv_feats

        edge_feats = self.bond_norm(edge_feats)
        return out_coords, inv_feats, edge_feats

    def _get_clones(self, module, n):
        return [copy.deepcopy(module) for _ in range(n)]


# *********************************************************************************************************************
# ****************************************** Molecular Generation Models **********************************************
# *********************************************************************************************************************


class MolecularGenerator(ABC, torch.nn.Module):
    """Interface for molecular generation classes"""

    def __init__(self, **kwargs):
        super().__init__()
        self._hparams = kwargs

    @property
    def hparams(self):
        return self._hparams

    @abstractmethod
    def forward(
        self,
        coords,
        inv_feats,
        edge_feats=None,
        cond_coords=None,
        cond_atomics=None,
        cond_bonds=None,
        atom_mask=None,
    ):
        pass


class SemlaGenerator(MolecularGenerator):
    def __init__(
        self,
        d_model,
        dynamics,
        vocab_size,
        n_atom_feats,
        d_edge=None,
        n_edge_types=None,
        self_cond=False,
        size_emb=64,
        max_atoms=256,
    ):
        hparams = {
            "d_model": d_model,
            "vocab_size": vocab_size,
            "n_atom_feats": n_atom_feats,
            "d_edge": d_edge,
            "n_edge_types": n_edge_types,
            "self_cond": self_cond,
            "size_emb": size_emb,
            "max_atoms": max_atoms,
            **dynamics.hparams,
        }

        super().__init__(**hparams)

        self.self_cond = self_cond

        if d_edge is not None or n_edge_types is not None:
            if None in [d_edge, n_edge_types]:
                raise ValueError(
                    "If either d_edge or n_edge_types are given both must be provided."
                )

            edge_in_feats = n_edge_types * 2 if self_cond else n_edge_types

            self.edge_in_proj = torch.nn.Sequential(
                torch.nn.Linear(edge_in_feats, d_edge),
                torch.nn.SiLU(inplace=False),
                torch.nn.Linear(d_edge, d_edge),
            )
            self.edge_out_proj = torch.nn.Sequential(
                torch.nn.Linear(d_edge, d_edge),
                torch.nn.SiLU(inplace=False),
                torch.nn.Linear(d_edge, n_edge_types),
            )

        in_feats = n_atom_feats + vocab_size if self_cond else n_atom_feats
        in_feats = in_feats + size_emb

        self.size_emb = torch.nn.Embedding(max_atoms, size_emb)
        self.feat_proj = torch.nn.Sequential(
            torch.nn.Linear(in_feats, d_model),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(d_model, d_model),
        )

        self.dynamics = dynamics

        self.atom_classifier_head = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(d_model, vocab_size),
        )
        self.charge_classifier_head = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(d_model, 7),
        )

    def forward(
        self,
        coords,
        inv_feats,
        edge_feats=None,
        cond_coords=None,
        cond_atomics=None,
        cond_bonds=None,
        atom_mask=None,
    ):
        """Predict molecular coordinates and atom types

        Args:
            coords (torch.Tensor): Input coordinates, shape [batch_size, n_atoms, 3]
            inv_feats (torch.Tensor): Invariant atom features, shape [batch_size, n_atoms, n_feats]
            edge_feats (torch.Tensor): In edge features, shape [batch_size, n_atoms, n_atoms, n_edge_types]
            cond_coords (torch.Tensor): Conditional coords, shape [batch_size, n_atoms, 3]
            cond_atomics (torch.Tensor): Conditional atom type logits, shape [batch_size, n_atoms, n_feats]
            cond_bonds (torch.Tensor): Cond bond type logits, shape [batch_size, n_atoms, n_atoms, n_edge_types]
            atom_mask (torch.Tensor): Mask for fake atoms, shape [batch_size, n_atoms], 1 for real atoms

        Returns:
            (predicted coordinates, atom type logits, bond logits, atom charges)
            All torch.Tensor, shapes:
                Coordinates: [batch_size, n_atoms, 3]
                Type logits: [batch_size, n_atoms, vocab_size],
                Bond logits: [batch_size, n_atoms, n_atoms, n_edge_types]
                Charge logits: [batch_size, n_atoms, 7]
        """

        if cond_coords is not None and not self.self_cond:
            raise ValueError(
                "cond_coords was provided but the model was initialised with self_cond as False."
            )

        if cond_coords is None and self.self_cond:
            raise ValueError(
                "The model was initialsed with self_cond but cond_coords was not provided."
            )

        if edge_feats is None and cond_bonds is not None:
            raise ValueError("edge_feats must be provided if using bond conditioning.")

        atom_mask = torch.ones_like(coords[..., 0]) if atom_mask is None else atom_mask
        adj_matrix = edges_from_nodes(coords, node_mask=atom_mask)

        # Embed the number of atoms in a mol into a small vector and concat this to inv feats for each atom
        n_atoms = atom_mask.sum(dim=-1, keepdim=True)
        # TODO: assert that n_atoms not larger than max_atoms
        size_emb = self.size_emb(n_atoms).expand(-1, inv_feats.size(1), -1)

        inv_feats = torch.cat((inv_feats, size_emb), dim=-1)
        if cond_atomics is not None:
            inv_feats = torch.cat((inv_feats, cond_atomics), dim=-1)

        atom_feats = self.feat_proj(inv_feats)

        if edge_feats is not None:
            edge_feats = edge_feats.float()
            edge_feats = (
                torch.cat((edge_feats, cond_bonds), dim=-1)
                if cond_bonds is not None
                else edge_feats
            )
            edge_feats = self.edge_in_proj(edge_feats)

        out = self.dynamics(
            coords,
            atom_feats,
            adj_matrix,
            atom_mask=atom_mask,
            edge_feats=edge_feats,
            cond_coords=cond_coords,
        )

        pred_edges = None
        if len(out) == 2:
            pred_coords, pred_feats = out
        elif len(out) == 3:
            pred_coords, pred_feats, pred_edges = out

        pred_coords = zero_com(pred_coords, node_mask=atom_mask)
        pred_coords = pred_coords * atom_mask.unsqueeze(-1)

        type_logits = self.atom_classifier_head(pred_feats)
        charge_logits = self.charge_classifier_head(pred_feats)

        # If we are predicting edges ensure that the matrix is symmetrical
        if pred_edges is not None:
            pred_edges = pred_edges + pred_edges.transpose(1, 2)
            edge_logits = self.edge_out_proj(pred_edges)
            return pred_coords, type_logits, edge_logits, charge_logits

        return pred_coords, type_logits, charge_logits


MENAGERIE_ZOO = "vendored-pytorch"


def build_semla_generator():
    dynamics = EquiInvDynamics(
        d_model=32,
        d_message=16,
        n_coord_sets=4,
        n_layers=3,
        n_attn_heads=8,
        d_message_hidden=16,
        d_edge=8,
        bond_refine=True,
        self_cond=False,
        coord_norm="length",
    )
    model = SemlaGenerator(
        d_model=32,
        dynamics=dynamics,
        vocab_size=16,
        n_atom_feats=17,
        d_edge=8,
        n_edge_types=5,
        self_cond=False,
        size_emb=8,
        max_atoms=64,
    )
    return model


def example_input_semla_generator():
    torch.manual_seed(0)
    batch_size, n_atoms = 1, 6
    coords = torch.randn(batch_size, n_atoms, 3)
    inv_feats = torch.randn(batch_size, n_atoms, 17)
    edge_feats = torch.rand(batch_size, n_atoms, n_atoms, 5)
    # Real batches always carry an integer/bool node mask (the `.mask` used downstream by
    # `size_emb = self.size_emb(n_atoms)` requires an nn.Embedding-compatible index dtype,
    # i.e. long); an all-float `torch.ones(...)` mask -- which is what
    # `SemlaGenerator.forward`'s own `atom_mask = torch.ones_like(coords[..., 0]) if
    # atom_mask is None else atom_mask` default would produce -- hits a genuine latent dtype
    # bug in `adj_from_node_mask` (`Index put requires ... dtypes match, got Long ... Float`).
    # That default path is simply never exercised by the real training pipeline, which always
    # passes an integer/bool mask; we do the same here for the real, intended usage.
    atom_mask = torch.ones(batch_size, n_atoms, dtype=torch.long)
    return (coords, inv_feats, edge_feats, None, None, None, atom_mask)


MENAGERIE_ENTRIES = [
    (
        "SemlaFlow-Generator",
        "build_semla_generator",
        "example_input_semla_generator",
        2024,
        "SOURCE_AVAILABLE",
    ),
]
