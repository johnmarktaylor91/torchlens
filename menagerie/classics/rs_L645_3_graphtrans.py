# SOURCE: vendored from jingraham/neurips19-graph-protein-design @ master
# Files: struct2seq/struct2seq.py, struct2seq/self_attention.py, struct2seq/protein_features.py
# GraphTrans (Struct2Seq): "Generative Models for Graph-Based Protein Design" (Ingraham et al.,
# NeurIPS 2019). Graph-structured Transformer encoder-decoder for protein inverse folding:
# k-nearest-neighbor backbone-geometry featurization (ProteinFeatures) feeds a permutation-
# invariant self-attention encoder (unmasked NeighborAttention) and an autoregressive masked
# self-attention decoder (TransformerLayer) that outputs per-residue amino-acid log-probabilities.
# Vendored verbatim aside from merging three files into one module (relative imports flattened;
# matplotlib import dropped -- only used inside commented-out debug-plot code, never executed).
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""============================================================================================="""
""" struct2seq/self_attention.py """
"""============================================================================================="""


def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features


def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features


def gather_nodes_t(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


class Normalize(nn.Module):
    def __init__(self, features, epsilon=1e-6):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x, dim=-1):
        mu = x.mean(dim, keepdim=True)
        sigma = torch.sqrt(x.var(dim, keepdim=True) + self.epsilon)
        gain = self.gain
        bias = self.bias
        # Reshape
        if dim != -1:
            shape = [1] * len(mu.size())
            shape[dim] = self.gain.size()[0]
            gain = gain.view(shape)
            bias = bias.view(shape)
        return gain * (x - mu) / (sigma + self.epsilon) + bias


class TransformerLayer(nn.Module):
    def __init__(self, num_hidden, num_in, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([Normalize(num_hidden) for _ in range(2)])

        self.attention = NeighborAttention(num_hidden, num_in, num_heads)
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """Parallel computation of full transformer layer"""
        # Self-attention
        dh = self.attention(h_V, h_E, mask_attend)
        h_V = self.norm[0](h_V + self.dropout(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V

    def step(self, t, h_V, h_E, mask_V=None, mask_attend=None):
        """Sequential computation of step t of a transformer layer"""
        # Self-attention
        h_V_t = h_V[:, t, :]
        dh_t = self.attention.step(t, h_V, h_E, mask_attend)
        h_V_t = self.norm[0](h_V_t + self.dropout(dh_t))

        # Position-wise feedforward
        dh_t = self.dense(h_V_t)
        h_V_t = self.norm[1](h_V_t + self.dropout(dh_t))

        if mask_V is not None:
            mask_V_t = mask_V[:, t].unsqueeze(-1)
            h_V_t = mask_V_t * h_V_t
        return h_V_t


class MPNNLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([Normalize(num_hidden) for _ in range(2)])

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)

        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """Parallel computation of full transformer layer"""

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.W3(F.relu(self.W2(F.relu(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm[0](h_V + self.dropout(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super().__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)

    def forward(self, h_V):
        h = F.relu(self.W_in(h_V))
        h = self.W_out(h)
        return h


class NeighborAttention(nn.Module):
    def __init__(self, num_hidden, num_in, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden

        # Self-attention layers: {queries, keys, values, output}
        self.W_Q = nn.Linear(num_hidden, num_hidden, bias=False)
        self.W_K = nn.Linear(num_in, num_hidden, bias=False)
        self.W_V = nn.Linear(num_in, num_hidden, bias=False)
        self.W_O = nn.Linear(num_hidden, num_hidden, bias=False)
        return

    def _masked_softmax(self, attend_logits, mask_attend, dim=-1):
        """Numerically stable masked softmax"""
        negative_inf = np.finfo(np.float32).min
        attend_logits = torch.where(
            mask_attend > 0, attend_logits, torch.tensor(negative_inf, dtype=attend_logits.dtype)
        )
        attend = F.softmax(attend_logits, dim)
        attend = mask_attend * attend
        return attend

    def forward(self, h_V, h_E, mask_attend=None):
        """Self-attention, graph-structured O(Nk)
        Args:
            h_V:            Node features           [N_batch, N_nodes, N_hidden]
            h_E:            Neighbor features       [N_batch, N_nodes, K, N_hidden]
            mask_attend:    Mask for attention      [N_batch, N_nodes, K]
        Returns:
            h_V:            Node update
        """

        # Queries, Keys, Values
        n_batch, n_nodes, n_neighbors = h_E.shape[:3]
        n_heads = self.num_heads

        d = int(self.num_hidden / n_heads)
        Q = self.W_Q(h_V).view([n_batch, n_nodes, 1, n_heads, 1, d])
        K = self.W_K(h_E).view([n_batch, n_nodes, n_neighbors, n_heads, d, 1])
        V = self.W_V(h_E).view([n_batch, n_nodes, n_neighbors, n_heads, d])

        # Attention with scaled inner product
        attend_logits = (
            torch.matmul(Q, K).view([n_batch, n_nodes, n_neighbors, n_heads]).transpose(-2, -1)
        )
        attend_logits = attend_logits / np.sqrt(d)

        if mask_attend is not None:
            # Masked softmax
            mask = mask_attend.unsqueeze(2).expand(-1, -1, n_heads, -1)
            attend = self._masked_softmax(attend_logits, mask)
        else:
            attend = F.softmax(attend_logits, -1)

        # Attentive reduction
        h_V_update = torch.matmul(attend.unsqueeze(-2), V.transpose(2, 3))
        h_V_update = h_V_update.view([n_batch, n_nodes, self.num_hidden])
        h_V_update = self.W_O(h_V_update)
        return h_V_update

    def step(self, t, h_V, h_E, E_idx, mask_attend=None):
        """Self-attention for a specific time step t

        Args:
            h_V:            Node features           [N_batch, N_nodes, N_hidden]
            h_E:            Neighbor features       [N_batch, N_nodes, K, N_in]
            E_idx:          Neighbor indices        [N_batch, N_nodes, K]
            mask_attend:    Mask for attention      [N_batch, N_nodes, K]
        Returns:
            h_V_t:            Node update
        """
        # Dimensions
        n_batch, n_nodes, n_neighbors = h_E.shape[:3]
        n_heads = self.num_heads
        d = self.num_hidden / n_heads

        # Per time-step tensors
        h_V_t = h_V[:, t, :]
        h_E_t = h_E[:, t, :, :]
        E_idx_t = E_idx[:, t, :]

        # Single time-step
        h_V_neighbors_t = gather_nodes_t(h_V, E_idx_t)
        E_t = torch.cat([h_E_t, h_V_neighbors_t], -1)

        # Queries, Keys, Values
        Q = self.W_Q(h_V_t).view([n_batch, 1, n_heads, 1, d])
        K = self.W_K(E_t).view([n_batch, n_neighbors, n_heads, d, 1])
        V = self.W_V(E_t).view([n_batch, n_neighbors, n_heads, d])

        # Attention with scaled inner product
        attend_logits = torch.matmul(Q, K).view([n_batch, n_neighbors, n_heads]).transpose(-2, -1)
        attend_logits = attend_logits / np.sqrt(d)

        if mask_attend is not None:
            # Masked softmax
            # [N_batch, K] -=> [N_batch, N_heads, K]
            mask_t = mask_attend[:, t, :].unsqueeze(1).expand(-1, n_heads, -1)
            attend = self._masked_softmax(attend_logits, mask_t)
        else:
            attend = F.softmax(attend_logits / np.sqrt(d), -1)

        # Attentive reduction
        h_V_t_update = torch.matmul(attend.unsqueeze(-2), V.transpose(1, 2))
        return h_V_t_update


"""============================================================================================="""
""" struct2seq/protein_features.py """
"""============================================================================================="""


class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, period_range=[2, 1000]):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.period_range = period_range

    def forward(self, E_idx):
        # i-j
        N_nodes = E_idx.size(1)
        ii = torch.arange(N_nodes, dtype=torch.float32).view((1, -1, 1))
        d = (E_idx.float() - ii).unsqueeze(-1)
        # Original Transformer frequencies
        frequency = torch.exp(
            torch.arange(0, self.num_embeddings, 2, dtype=torch.float32)
            * -(np.log(10000.0) / self.num_embeddings)
        )
        angles = d * frequency.view((1, 1, 1, -1))
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E


class ProteinFeatures(nn.Module):
    def __init__(
        self,
        edge_features,
        node_features,
        num_positional_embeddings=16,
        num_rbf=16,
        top_k=30,
        features_type="full",
        augment_eps=0.0,
        dropout=0.1,
    ):
        """Extract protein features"""
        super().__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        # Feature types
        self.features_type = features_type
        self.feature_dimensions = {
            "coarse": (3, num_positional_embeddings + num_rbf + 7),
            "full": (6, num_positional_embeddings + num_rbf + 7),
            "dist": (6, num_positional_embeddings + num_rbf),
            "hbonds": (3, 2 * num_positional_embeddings),
        }

        # Positional encoding
        self.embeddings = PositionalEncodings(num_positional_embeddings)
        self.dropout = nn.Dropout(dropout)

        # Normalization and embedding
        node_in, edge_in = self.feature_dimensions[features_type]
        self.node_embedding = nn.Linear(node_in, node_features, bias=True)
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=True)
        self.norm_nodes = Normalize(node_features)
        self.norm_edges = Normalize(edge_features)

    def _dist(self, X, mask, eps=1e-6):
        """Pairwise euclidean distances"""
        # Convolutional network on NCHW
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)

        # Identify k nearest neighbors (including self)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(D_adjust, self.top_k, dim=-1, largest=False)
        mask_neighbors = gather_edges(mask_2D.unsqueeze(-1), E_idx)

        return D_neighbors, E_idx, mask_neighbors

    def _rbf(self, D):
        # Distance radial basis function
        D_min, D_max, D_count = 0.0, 20.0, self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))

        return RBF

    def _quaternions(self, R):
        """Convert a batch of 3D rotations [R] to quaternions [Q]
        R [...,3,3]
        Q [...,4]
        """
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(
            torch.abs(1 + torch.stack([Rxx - Ryy - Rzz, -Rxx + Ryy - Rzz, -Rxx - Ryy + Rzz], -1))
        )

        def _R(i, j):
            return R[:, :, :, i, j]

        signs = torch.sign(
            torch.stack([_R(2, 1) - _R(1, 2), _R(0, 2) - _R(2, 0), _R(1, 0) - _R(0, 1)], -1)
        )
        xyz = signs * magnitudes
        # The relu enforces a non-negative trace
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.0
        Q = torch.cat((xyz, w), -1)
        Q = F.normalize(Q, dim=-1)

        return Q

    def _contacts(self, D_neighbors, E_idx, mask_neighbors, cutoff=8):
        """Contacts"""
        D_neighbors = D_neighbors.unsqueeze(-1)
        neighbor_C = mask_neighbors * (D_neighbors < cutoff).type(torch.float32)
        return neighbor_C

    def _hbonds(self, X, E_idx, mask_neighbors, eps=1e-3):
        """Hydrogen bonds and contact map"""
        X_atoms = dict(zip(["N", "CA", "C", "O"], torch.unbind(X, 2)))

        # Virtual hydrogens
        X_atoms["C_prev"] = F.pad(X_atoms["C"][:, 1:, :], (0, 0, 0, 1), "constant", 0)
        X_atoms["H"] = X_atoms["N"] + F.normalize(
            F.normalize(X_atoms["N"] - X_atoms["C_prev"], dim=-1)
            + F.normalize(X_atoms["N"] - X_atoms["CA"], dim=-1),
            dim=-1,
        )

        def _distance(X_a, X_b):
            return torch.norm(X_a[:, None, :, :] - X_b[:, :, None, :], dim=-1)

        def _inv_distance(X_a, X_b):
            return 1.0 / (_distance(X_a, X_b) + eps)

        # DSSP vacuum electrostatics model
        U = (0.084 * 332) * (
            _inv_distance(X_atoms["O"], X_atoms["N"])
            + _inv_distance(X_atoms["C"], X_atoms["H"])
            - _inv_distance(X_atoms["O"], X_atoms["H"])
            - _inv_distance(X_atoms["C"], X_atoms["N"])
        )

        HB = (U < -0.5).type(torch.float32)
        neighbor_HB = mask_neighbors * gather_edges(HB.unsqueeze(-1), E_idx)
        return neighbor_HB

    def _orientations_coarse(self, X, E_idx, eps=1e-6):
        # Pair features

        # Shifted slices of unit vectors
        dX = X[:, 1:, :] - X[:, :-1, :]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:, :-2, :]
        u_1 = U[:, 1:-1, :]
        u_0 = U[:, 2:, :]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

        # Bond angle calculation
        cosA = -(u_1 * u_0).sum(-1)
        cosA = torch.clamp(cosA, -1 + eps, 1 - eps)
        A = torch.acos(cosA)
        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
        # Backbone features
        AD_features = torch.stack(
            (torch.cos(A), torch.sin(A) * torch.cos(D), torch.sin(A) * torch.sin(D)), 2
        )
        AD_features = F.pad(AD_features, (0, 0, 1, 2), "constant", 0)

        # Build relative orientations
        o_1 = F.normalize(u_2 - u_1, dim=-1)
        O = torch.stack((o_1, n_2, torch.cross(o_1, n_2)), 2)  # noqa: E741
        O = O.view(list(O.shape[:2]) + [9])  # noqa: E741
        O = F.pad(O, (0, 0, 1, 2), "constant", 0)  # noqa: E741

        O_neighbors = gather_nodes(O, E_idx)
        X_neighbors = gather_nodes(X, E_idx)

        # Re-view as rotation matrices
        O = O.view(list(O.shape[:2]) + [3, 3])  # noqa: E741
        O_neighbors = O_neighbors.view(list(O_neighbors.shape[:3]) + [3, 3])

        # Rotate into local reference frames
        dX = X_neighbors - X.unsqueeze(-2)
        dU = torch.matmul(O.unsqueeze(2), dX.unsqueeze(-1)).squeeze(-1)
        dU = F.normalize(dU, dim=-1)
        R = torch.matmul(O.unsqueeze(2).transpose(-1, -2), O_neighbors)
        Q = self._quaternions(R)

        # Orientation features
        O_features = torch.cat((dU, Q), dim=-1)

        return AD_features, O_features

    def _dihedrals(self, X, eps=1e-7):
        # First 3 coordinates are N, CA, C
        X = X[:, :, :3, :].reshape(X.shape[0], 3 * X.shape[1], 3)

        # Shifted slices of unit vectors
        dX = X[:, 1:, :] - X[:, :-1, :]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:, :-2, :]
        u_1 = U[:, 1:-1, :]
        u_0 = U[:, 2:, :]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, (1, 2), "constant", 0)
        D = D.view((D.size(0), int(D.size(1) / 3), 3))
        phi, psi, omega = torch.unbind(D, -1)

        # Lift angle representations to the circle
        D_features = torch.cat((torch.cos(D), torch.sin(D)), 2)
        return D_features

    def forward(self, X, L, mask):
        """Featurize coordinates as an attributed graph"""

        # Data augmentation
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)

        # Build k-Nearest Neighbors graph
        X_ca = X[:, :, 1, :]
        D_neighbors, E_idx, mask_neighbors = self._dist(X_ca, mask)

        # Pairwise features
        AD_features, O_features = self._orientations_coarse(X_ca, E_idx)
        RBF = self._rbf(D_neighbors)

        # Pairwise embeddings
        E_positional = self.embeddings(E_idx)

        if self.features_type == "coarse":
            # Coarse backbone features
            V = AD_features
            E = torch.cat((E_positional, RBF, O_features), -1)
        elif self.features_type == "hbonds":
            # Hydrogen bonds and contacts
            neighbor_HB = self._hbonds(X, E_idx, mask_neighbors)
            neighbor_C = self._contacts(D_neighbors, E_idx, mask_neighbors)
            # Dropout
            neighbor_C = self.dropout(neighbor_C)
            neighbor_HB = self.dropout(neighbor_HB)
            # Pack
            V = mask.unsqueeze(-1) * torch.ones_like(AD_features)
            neighbor_C = neighbor_C.expand(-1, -1, -1, int(self.num_positional_embeddings / 2))
            neighbor_HB = neighbor_HB.expand(-1, -1, -1, int(self.num_positional_embeddings / 2))
            E = torch.cat((E_positional, neighbor_C, neighbor_HB), -1)
        elif self.features_type == "full":
            # Full backbone angles
            V = self._dihedrals(X)
            E = torch.cat((E_positional, RBF, O_features), -1)
        elif self.features_type == "dist":
            # Full backbone angles
            V = self._dihedrals(X)
            E = torch.cat((E_positional, RBF), -1)

        # Embed the nodes
        V = self.node_embedding(V)
        V = self.norm_nodes(V)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)

        return V, E, E_idx


"""============================================================================================="""
""" struct2seq/struct2seq.py """
"""============================================================================================="""


class Struct2Seq(nn.Module):
    def __init__(
        self,
        num_letters,
        node_features,
        edge_features,
        hidden_dim,
        num_encoder_layers=3,
        num_decoder_layers=3,
        vocab=20,
        k_neighbors=30,
        protein_features="full",
        augment_eps=0.0,
        dropout=0.1,
        forward_attention_decoder=True,
        use_mpnn=False,
    ):
        """Graph labeling network"""
        super().__init__()

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        # Featurization layers
        self.features = ProteinFeatures(
            node_features,
            edge_features,
            top_k=k_neighbors,
            features_type=protein_features,
            augment_eps=augment_eps,
            dropout=dropout,
        )

        # Embedding layers
        self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)
        layer = TransformerLayer if not use_mpnn else MPNNLayer

        # Encoder layers
        self.encoder_layers = nn.ModuleList(
            [layer(hidden_dim, hidden_dim * 2, dropout=dropout) for _ in range(num_encoder_layers)]
        )

        # Decoder layers
        self.forward_attention_decoder = forward_attention_decoder
        self.decoder_layers = nn.ModuleList(
            [layer(hidden_dim, hidden_dim * 3, dropout=dropout) for _ in range(num_decoder_layers)]
        )
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _autoregressive_mask(self, E_idx):
        N_nodes = E_idx.size(1)
        ii = torch.arange(N_nodes)
        ii = ii.view((1, -1, 1))
        mask = E_idx - ii < 0
        mask = mask.type(torch.float32)
        return mask

    def forward(self, X, S, L, mask):
        """Graph-conditioned sequence model"""

        # Prepare node and edge embeddings
        V, E, E_idx = self.features(X, L, mask)
        h_V = self.W_v(V)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V = layer(h_V, h_EV, mask_V=mask, mask_attend=mask_attend)

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_ES_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_ESV_encoder = cat_neighbors_nodes(h_V, h_ES_encoder, E_idx)

        # Decoder uses masked self-attention
        mask_attend = self._autoregressive_mask(E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend

        if self.forward_attention_decoder:
            mask_fw = mask_1D * (1.0 - mask_attend)
            h_ESV_encoder_fw = mask_fw * h_ESV_encoder
        else:
            h_ESV_encoder_fw = 0
        for layer in self.decoder_layers:
            # Masked positions attend to encoder information, unmasked see.
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_ESV_encoder_fw
            h_V = layer(h_V, h_ESV, mask_V=mask)

        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs


MENAGERIE_ZOO = "vendored-pytorch"


def build_graphtrans():
    torch.manual_seed(0)
    return Struct2Seq(
        num_letters=20,
        node_features=32,
        edge_features=32,
        hidden_dim=32,
        num_encoder_layers=1,
        num_decoder_layers=1,
        vocab=20,
        k_neighbors=8,
        protein_features="full",
        dropout=0.0,
    )


def example_input_graphtrans():
    torch.manual_seed(0)
    batch, n_res = 1, 12
    X = torch.randn(batch, n_res, 4, 3)
    S = torch.randint(0, 20, (batch, n_res))
    L = torch.full((batch,), n_res, dtype=torch.long)
    mask = torch.ones(batch, n_res)
    return (X, S, L, mask)


MENAGERIE_ENTRIES = [
    ("GraphTrans", "build_graphtrans", "example_input_graphtrans", 2019, "SOURCE_AVAILABLE"),
]
