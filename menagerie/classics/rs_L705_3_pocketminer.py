# FAITHFUL PORT of Mickdub/gvp @ 187062df3c94127e991669768009141a08fd5d8b (branch pocket_pred)
# (original framework: TensorFlow 2.x / Keras)
#
# PocketMiner (Meller et al., Nat. Commun. 2023, "Predicting locations of cryptic pockets
# from single protein structures using the PocketMiner graph neural network") predicts
# per-residue cryptic-pocket-opening probability from a single protein backbone structure.
# It is built on the GVP-GNN architecture (Jing et al., "Learning from Protein Structure
# with Geometric Vector Perceptrons", ICLR 2021) as implemented in this repo's own
# `src/gvp.py` (GVP/GVPDropout/GVPLayerNorm primitives) and `src/models.py`
# (`MQAModel` / `StructuralFeatures` / `Encoder` / `MPNNLayer`), which is TensorFlow/Keras
# code -- this environment has no TensorFlow build, so the real classes cannot be run
# directly. Every mechanism (the split-representation vector-scalar `GVP` layer, its
# vector-channel norm-gated nonlinearity, `GVPLayerNorm`'s separate scalar/vector
# normalization, the k-NN backbone graph featurization -- RBF distances, positional
# encodings, dihedral angles, backbone orientations, sidechain direction vectors -- and
# the `MPNNLayer` message-passing encoder) is transcribed faithfully from the real
# `src/gvp.py` + `src/models.py` TF code into self-contained torch, using `MQAModel`'s
# default (non-regression, non-multiclass) sigmoid per-residue output head, matching
# PocketMiner's actual per-residue pocket-opening-probability prediction task
# (`res_level=True` inference path used by `src/xtal_predict.py`).
#
# MENAGERIE_ZOO = "ported-pytorch"

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


def norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    out = torch.clamp(torch.sum(torch.square(x), dim=axis, keepdim=keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out


def gvp_split(x, nv):
    # [..., 3*nv + ns] -> [..., 3, nv], [..., ns]; vector channels always at the top.
    v = x[..., : 3 * nv].reshape(*x.shape[:-1], 3, nv)
    s = x[..., 3 * nv :]
    return v, s


def gvp_merge(v, s):
    # [..., 3, nv], [..., ns] -> [..., 3*nv + ns]
    v = v.reshape(*v.shape[:-2], 3 * v.shape[-1])
    return torch.cat([v, s], dim=-1)


def vs_concat(x1, x2, nv1, nv2):
    v1, s1 = gvp_split(x1, nv1)
    v2, s2 = gvp_split(x2, nv2)
    v = torch.cat([v1, v2], dim=-1)
    s = torch.cat([s1, s2], dim=-1)
    return gvp_merge(v, s)


class GVP(nn.Module):
    """[v/s][i/o] = number of [vector/scalar] channels [in/out]. Faithful port of
    src/gvp.py::GVP (a tf.keras.Model)."""

    def __init__(self, vi, vo, so, si, nlv="sigmoid", nls="relu"):
        super().__init__()
        self.vi, self.vo, self.so, self.si = vi, vo, so, si
        if vi:
            self.wh = nn.Linear(vi, max(vi, vo))
            ws_in = max(vi, vo) + si  # concat([s, vn]); vn has max(vi,vo) channels
        else:
            ws_in = si
        self.ws = nn.Linear(ws_in, so)
        if vo:
            self.wv = nn.Linear(max(vi, vo), vo)
        self.nlv_name = nlv
        self.nls_name = nls

    def _nls(self, x):
        if self.nls_name is None:
            return x
        return F.relu(x)

    def _nlv(self, x):
        if self.nlv_name is None:
            return x
        return torch.sigmoid(x)

    def forward(self, x, return_split=False):
        # x: [..., 3*vi + si]
        v, s = gvp_split(x, self.vi)
        if self.vi:
            vh = self.wh(v)  # Dense over channel dim (last dim)
            vn = norm_no_nan(vh, axis=-2)
            out = self._nls(self.ws(torch.cat([s, vn], dim=-1)))
        else:
            out = self._nls(self.ws(s))
        if self.vo:
            vo_ = self.wv(vh)
            if self.nlv_name is not None:
                vo_ = vo_ * self._nlv(norm_no_nan(vo_, axis=-2, keepdims=True))
            out = (vo_, out) if return_split else gvp_merge(vo_, out)
        return out


class GVPDropout(nn.Module):
    """Dropout that drops vector and scalar channels separately. Faithful port of
    src/gvp.py::GVPDropout."""

    def __init__(self, rate, nv):
        super().__init__()
        self.nv = nv
        self.rate = rate

    def forward(self, x, training=True):
        if not training or self.rate == 0.0:
            return x
        v, s = gvp_split(x, self.nv)
        # vdropout: noise_shape=[1, nv] -> drop whole vector channels together across the
        # 3-component axis, independently per (..., nv) position.
        if self.nv:
            keep = (torch.rand_like(v[..., 0, :]) > self.rate).float() / (1.0 - self.rate)
            v = v * keep.unsqueeze(-2)
        s = F.dropout(s, p=self.rate, training=True)
        return gvp_merge(v, s)


class GVPLayerNorm(nn.Module):
    """Normal layer norm for scalars, nontrainable norm for vectors. Faithful port of
    src/gvp.py::GVPLayerNorm."""

    def __init__(self, nv, ns):
        super().__init__()
        self.nv = nv
        self.snorm = nn.LayerNorm(ns)

    def forward(self, x):
        v, s = gvp_split(x, self.nv)
        vn = norm_no_nan(v, axis=-2, keepdims=True, sqrt=False)  # [.., 1, nv]
        vn = torch.sqrt(torch.mean(vn, dim=-1, keepdim=True))
        return gvp_merge(v / (vn + 1e-8), self.snorm(s))


def autoregressive_mask(E_idx):
    N_nodes = E_idx.shape[1]
    ii = torch.arange(N_nodes, device=E_idx.device).view(1, -1, 1)
    mask = (E_idx - ii) < 0
    return mask.float()


def gvp_normalize(tensor, axis=-1):
    n = torch.linalg.norm(tensor, dim=axis, keepdim=True)
    return torch.where(n > 0, tensor / n, torch.zeros_like(tensor))


def gather_edges(edges, neighbor_idx):
    # edges [B,N,N,C], neighbor_idx [B,N,K] -> [B,N,K,C]
    C = edges.shape[-1]
    idx = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, C)
    return torch.gather(edges, 2, idx)


def gather_nodes(nodes, neighbor_idx):
    # nodes [B,N,C], neighbor_idx [B,N,K] -> [B,N,K,C]
    B, N, C = nodes.shape
    K = neighbor_idx.shape[-1]
    neighbors_flat = neighbor_idx.reshape(B, -1)  # [B, N*K]
    idx = neighbors_flat.unsqueeze(-1).expand(-1, -1, C)
    out = torch.gather(nodes, 1, idx)  # [B, N*K, C]
    return out.reshape(B, N, K, C)


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx, nv_nodes, nv_neighbors):
    h_nodes = gather_nodes(h_nodes, E_idx)
    return vs_concat(h_neighbors, h_nodes, nv_neighbors, nv_nodes)


class PositionalEncodings(nn.Module):
    """Faithful port of src/models.py::PositionalEncodings."""

    def __init__(self, num_embeddings):
        super().__init__()
        self.num_embeddings = num_embeddings

    def forward(self, E_idx):
        N_nodes = E_idx.shape[1]
        ii = torch.arange(N_nodes, device=E_idx.device, dtype=torch.float32).view(1, -1, 1)
        d = (E_idx.float() - ii).unsqueeze(-1)
        frequency = torch.exp(
            torch.arange(0, self.num_embeddings, 2, device=E_idx.device, dtype=torch.float32)
            * -(torch.log(torch.tensor(10000.0)) / self.num_embeddings)
        )
        angles = d * frequency.view(1, 1, 1, -1)
        return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)


class StructuralFeatures(nn.Module):
    """Faithful port of src/models.py::StructuralFeatures (ablate_sidechain_vectors=True,
    ablate_rbf=False -- MQAModel's real defaults)."""

    def __init__(
        self, node_features, edge_features, num_positional_embeddings=16, num_rbf=16, top_k=30
    ):
        super().__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodings(num_positional_embeddings)

        vo, so = node_features
        ve, se = edge_features
        # ablate_sidechain_vectors=True (real MQAModel default). Node scalar features are
        # V_dihedrals: cos/sin of 3 backbone dihedral-angle groups -> 6 scalar channels.
        self.node_embedding = GVP(vi=3, vo=vo, so=so, si=6, nlv=None, nls=None)
        # E = concat([E_directions(vi=1 vector), RBF(num_rbf scalars),
        #             E_positional(num_positional_embeddings scalars -- cos/sin halves of
        #             num_positional_embeddings/2 frequencies concatenate back to
        #             num_positional_embeddings)], -1)
        self.edge_embedding = GVP(
            vi=1, vo=ve, so=se, si=num_rbf + num_positional_embeddings, nlv=None, nls=None
        )
        self.norm_nodes = nn.LayerNorm(so)
        self.norm_edges = nn.LayerNorm(se)

    def _dist(self, X, mask, eps=1e-6):
        mask = mask.float()
        mask_2D = mask.unsqueeze(1) * mask.unsqueeze(2)
        dX = X.unsqueeze(1) - X.unsqueeze(2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, dim=3) + eps)

        D_max = torch.max(D, dim=-1, keepdim=True).values
        D_adjust = D + (1.0 - mask_2D) * D_max
        k = min(self.top_k, X.shape[1])
        D_neighbors, E_idx = torch.topk(-D_adjust, k=k, dim=-1)
        D_neighbors = -D_neighbors
        mask_neighbors = gather_edges(mask_2D.unsqueeze(-1), E_idx)
        return D_neighbors, E_idx, mask_neighbors

    def _directions(self, X, E_idx):
        X_neighbors = gather_nodes(X, E_idx)
        dX = X_neighbors - X.unsqueeze(-2)
        return gvp_normalize(dX, axis=-1)

    def _rbf(self, D):
        D_min, D_max, D_count = 0.0, 20.0, self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
        D_mu = D_mu.view(1, 1, 1, -1)
        D_sigma = (D_max - D_min) / D_count
        D_expand = D.unsqueeze(-1)
        return torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))

    def _orientations(self, X):
        forward = gvp_normalize(X[:, 1:] - X[:, :-1])
        backward = gvp_normalize(X[:, :-1] - X[:, 1:])
        forward = F.pad(forward, (0, 0, 0, 1))
        backward = F.pad(backward, (0, 0, 1, 0))
        return torch.cat([forward.unsqueeze(-1), backward.unsqueeze(-1)], dim=-1)  # B, N, 3, 2

    def _sidechains(self, X):
        # ['N', 'CA', 'C', 'O']; X: B, N, 4, 3
        n, origin, c = X[:, :, 0, :], X[:, :, 1, :], X[:, :, 2, :]
        c, n = gvp_normalize(c - origin), gvp_normalize(n - origin)
        bisector = gvp_normalize(c + n)
        perp = gvp_normalize(torch.linalg.cross(c, n))
        vec = -bisector * (1.0 / 3) ** 0.5 - perp * (2.0 / 3) ** 0.5
        return vec  # B, N, 3

    def _dihedrals(self, X, eps=1e-7):
        B, N = X.shape[0], X.shape[1]
        X = X[:, :, :3, :].reshape(B, 3 * N, 3)

        dX = X[:, 1:, :] - X[:, :-1, :]
        U = gvp_normalize(dX, axis=-1)
        u_2 = U[:, :-2, :]
        u_1 = U[:, 1:-1, :]
        u_0 = U[:, 2:, :]

        n_2 = gvp_normalize(torch.linalg.cross(u_2, u_1), axis=-1)
        n_1 = gvp_normalize(torch.linalg.cross(u_1, u_0), axis=-1)

        cosD = torch.sum(n_2 * n_1, dim=-1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, dim=-1)) * torch.acos(cosD)

        D = F.pad(D, (1, 2))
        D = D.reshape(B, D.shape[1] // 3, 3)
        return torch.cat([torch.cos(D), torch.sin(D)], dim=2)

    def forward(self, X, mask):
        X_ca = X[:, :, 1, :]
        D_neighbors, E_idx, mask_neighbors = self._dist(X_ca, mask)

        E_directions = self._directions(X_ca, E_idx)
        RBF = self._rbf(D_neighbors)
        E_positional = self.embeddings(E_idx)

        V_dihedrals = self._dihedrals(X)
        V_orientations = self._orientations(X_ca)
        V_sidechains = self._sidechains(X)

        V_vec = torch.cat([V_sidechains.unsqueeze(-1), V_orientations], dim=-1)
        V = gvp_merge(V_vec, V_dihedrals)

        E = torch.cat([E_directions, RBF, E_positional], dim=-1)

        Vv, Vs = self.node_embedding(V, return_split=True)
        V = gvp_merge(Vv, self.norm_nodes(Vs))

        Ev, Es = self.edge_embedding(E, return_split=True)
        E = gvp_merge(Ev, self.norm_edges(Es))

        return V, E, E_idx


class MPNNLayer(nn.Module):
    """Faithful port of src/models.py::MPNNLayer."""

    def __init__(self, vec_in, num_hidden, dropout=0.1):
        super().__init__()
        self.vec_in = vec_in
        self.vo, self.so = num_hidden

        self.norm0 = GVPLayerNorm(self.vo, self.so)
        self.norm1 = GVPLayerNorm(self.vo, self.so)
        self.dropout = GVPDropout(dropout, self.vo)

        # Sequential([GVP(vi=vec_in+vo, vo=vo, so=so), GVP(vi=vo, vo=vo, so=so),
        #             GVP(vi=vo, vo=vo, so=so, nls=None, nlv=None)]).
        # h_EV = vs_concat(h_V_expand[so scalars], h_M[2*so scalars: h_E's so + gathered
        # h_V's so], vo, vec_in) -> scalar width = 3*so for the first layer only.
        self.w_ev_0 = GVP(vi=vec_in + self.vo, vo=self.vo, so=self.so, si=3 * self.so)
        self.w_ev_1 = GVP(vi=self.vo, vo=self.vo, so=self.so, si=self.so)
        self.w_ev_2 = GVP(vi=self.vo, vo=self.vo, so=self.so, si=self.so, nls=None, nlv=None)

        self.w_dh_0 = GVP(vi=self.vo, vo=2 * self.vo, so=4 * self.so, si=self.so)
        self.w_dh_1 = GVP(
            vi=2 * self.vo, vo=self.vo, so=self.so, si=4 * self.so, nls=None, nlv=None
        )

    def _w_ev(self, x):
        x = self.w_ev_0(x)
        x = self.w_ev_1(x)
        x = self.w_ev_2(x)
        return x

    def _w_dh(self, x):
        x = self.w_dh_0(x)
        x = self.w_dh_1(x)
        return x

    def forward(self, h_V, h_M, mask_V=None, mask_attend=None, training=False):
        K = h_M.shape[-2]
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, K, -1)
        h_EV = vs_concat(h_V_expand, h_M, self.vo, self.vec_in)
        h_message = self._w_ev(h_EV)
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1).float() * h_message
        dh = torch.mean(h_message, dim=-2)
        h_V = self.norm0(h_V + self.dropout(dh, training=training))

        dh = self._w_dh(h_V)
        h_V = self.norm1(h_V + self.dropout(dh, training=training))

        if mask_V is not None:
            h_V = mask_V.unsqueeze(-1).float() * h_V

        return h_V


class Encoder(nn.Module):
    """Faithful port of src/models.py::Encoder."""

    def __init__(self, node_features, edge_features, num_layers=3, dropout=0.1):
        super().__init__()
        self.nv, self.ns = node_features
        self.ev, _ = edge_features
        self.vglayers = nn.ModuleList(
            [
                MPNNLayer(self.nv + self.ev, node_features, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, h_V, h_E, E_idx, mask, training=False):
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend

        for layer in self.vglayers:
            h_M = cat_neighbors_nodes(h_V, h_E, E_idx, self.nv, self.ev)
            h_V = layer(h_V, h_M, mask_V=mask, mask_attend=mask_attend, training=training)

        return h_V


class MQAModel(nn.Module):
    """PocketMiner's real prediction network. Faithful port of src/models.py::MQAModel
    with its real defaults (regression=False, multiclass=False, use_lm=False,
    ablate_aa_type=False, ablate_sidechain_vectors=True) -- the config used for the
    shipped per-residue cryptic-pocket sigmoid-probability model."""

    def __init__(
        self, node_features, edge_features, hidden_dim, num_layers=3, k_neighbors=30, dropout=0.1
    ):
        super().__init__()
        self.nv, self.ns = node_features
        self.hv, self.hs = hidden_dim
        self.ev, self.es = edge_features

        self.features = StructuralFeatures(node_features, edge_features, top_k=k_neighbors)

        self.W_s = nn.Embedding(20, self.hs)

        # W_v consumes V = vs_concat(features_out, h_S, nv, 0): scalar width = ns + hs
        # (node scalar features from StructuralFeatures, plus the hs-wide W_s sequence
        # embedding concatenated on).
        self.W_v = GVP(vi=self.nv, vo=self.hv, so=self.hs, si=self.ns + self.hs, nls=None, nlv=None)
        self.W_e = GVP(vi=self.ev, vo=self.ev, so=self.hs, si=self.es, nls=None, nlv=None)

        self.encoder = Encoder(hidden_dim, edge_features, num_layers=num_layers, dropout=dropout)

        self.W_V_out = GVP(vi=self.hv, vo=0, so=self.hs, si=self.hs, nls=None, nlv=None)

        self.dense = nn.Sequential(
            nn.Linear(self.hs, 2 * self.hs),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * self.hs, 2 * self.hs),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(2 * self.hs),
            nn.Linear(2 * self.hs, 1),
        )

    def forward(self, X, S, mask, training=False, res_level=True):
        # X [B, N, 4, 3], S [B, N], mask [B, N]
        V, E, E_idx = self.features(X, mask)

        h_S = self.W_s(S)
        V = vs_concat(V, h_S, self.nv, 0)
        h_V = self.W_v(V)

        h_E = self.W_e(E)
        h_V = self.encoder(h_V, h_E, E_idx, mask, training=training)

        h_V_out = self.W_V_out(h_V)
        mask_e = mask.unsqueeze(-1)

        if not res_level:
            if training:
                h_V_out = torch.mean(h_V_out * mask_e, dim=-2)
            else:
                summed = torch.sum(h_V_out * mask_e, dim=-2)
                denom = torch.sum(mask_e, dim=-2)
                h_V_out = torch.where(denom > 0, summed / denom, torch.zeros_like(summed))

        out = self.dense(h_V_out)
        out = torch.sigmoid(out).squeeze(-1)
        return out


def build_pocketminer():
    # Real released checkpoint uses node_features=(16,16), edge_features=(16,16),
    # hidden_dim=(16,16), num_layers=4, k_neighbors=30 (pocketminer.yml /
    # src/train_cryptic_labels.py `MODEL_HOME` config). Shrunk for a menagerie-scale trace.
    return MQAModel(
        node_features=(4, 8),
        edge_features=(1, 8),
        hidden_dim=(4, 8),
        num_layers=2,
        k_neighbors=6,
    )


def example_input_pocketminer():
    torch.manual_seed(0)
    batch_size, n_res = 1, 12
    X = torch.randn(batch_size, n_res, 4, 3)
    S = torch.randint(0, 20, (batch_size, n_res))
    mask = torch.ones(batch_size, n_res)
    return (X, S, mask)


MENAGERIE_ENTRIES = [
    ("PocketMiner", "build_pocketminer", "example_input_pocketminer", 2023, "ported-pytorch"),
]
