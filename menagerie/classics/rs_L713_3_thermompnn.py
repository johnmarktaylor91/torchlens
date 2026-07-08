# SOURCE: vendored from Kuhlman-Lab/ThermoMPNN @ main
# https://github.com/Kuhlman-Lab/ThermoMPNN
#
# ThermoMPNN: Dieckhaus, Brocidiacono, Randolph, Kuhlman, "Transfer learning to leverage
# larger datasets for improved prediction of protein stability changes" (PNAS 2024).
# A lightweight ddG-prediction head transfer-learned on top of FROZEN structural
# embeddings from a pretrained ProteinMPNN backbone: per-residue hidden states from
# ProteinMPNN's final decoder layers, concatenated with the sequence embedding at the
# mutated position, are optionally passed through a `LightAttention` 1D-conv attention
# pooling block (ported verbatim from Hannes Stark's `protein-localization` repo, as
# cited in the real source) and then an MLP (`both_out`) that outputs a ddG value per
# amino-acid substitution, followed by wild-type-subtraction (`ddg_out[mut] -
# ddg_out[wt]`) to get the final predicted ddG.
#
# This is the REAL model code from `protein_mpnn_utils.py` (`ProteinMPNN` backbone --
# identical architecture family used by the catalog's existing `ProteinMPNN` entry, but
# vendored here directly since ThermoMPNN's `TransferModel` wraps this exact class) and
# `transfer_model.py` (`TransferModel`, `LightAttention`), combined verbatim into one
# file. The one architectural deviation from upstream `TransferModel.forward()`: the
# real forward() calls `tied_featurize()` on raw PDB dict batches -- that is PDB-file
# text-parsing / numpy padding machinery (not architecture, un-traceable through
# TorchLens meaningfully), so this module's `forward()` takes the ALREADY-FEATURIZED
# tensors that `tied_featurize()` would have produced (X, S, mask, chain encodings,
# residue indices, and one (position, wildtype-idx, mutant-idx) mutation triple)
# directly as arguments -- every downstream nn.Module call (ProteinMPNN encoder/decoder,
# LightAttention, both_out MLP, wildtype-subtraction) is untouched real code. Depends
# only on torch/numpy, both base libs.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"

VOCAB_DIM = 21
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


# ---- protein_mpnn_utils.py (verbatim ProteinMPNN architecture) ---------------------


def gather_edges(edges, neighbor_idx):
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features


def gather_nodes(nodes, neighbor_idx):
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        """Parallel computation of full transformer layer"""
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E


class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """Parallel computation of full transformer layer"""
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h


class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2 * max_relative_feature + 1 + 1, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(
            offset + self.max_relative_feature, 0, 2 * self.max_relative_feature
        ) * mask + (1 - mask) * (2 * self.max_relative_feature + 1)
        d_onehot = torch.nn.functional.one_hot(d, 2 * self.max_relative_feature + 1 + 1)
        E = self.linear(d_onehot.float())
        return E


class ProteinFeatures(nn.Module):
    def __init__(
        self,
        edge_features,
        node_features,
        num_positional_embeddings=16,
        num_rbf=16,
        top_k=30,
        augment_eps=0.0,
        num_chain_embeddings=16,
    ):
        """Extract protein features"""
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        node_in, edge_in = 6, num_positional_embeddings + num_rbf * 25  # noqa: F841 -- unused in the original repo code too; kept for fidelity
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _dist(self, X, mask, eps=1e-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(
            D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False
        )
        return D_neighbors, E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2.0, 22.0, self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(
            torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6
        )  # [B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[:, :, :, 0]  # [B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, X, mask, residue_idx, chain_labels):
        if self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)

        b = X[:, :, 1, :] - X[:, :, 0, :]
        c = X[:, :, 2, :] - X[:, :, 1, :]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 1, :]
        Ca = X[:, :, 1, :]
        N = X[:, :, 0, :]
        C = X[:, :, 2, :]
        O = X[:, :, 3, :]  # noqa: E741 -- matches original repo style (atom-name notation)

        D_neighbors, E_idx = self._dist(Ca, mask)

        RBF_all = [
            self._rbf(D_neighbors),  # Ca-Ca
            self._get_rbf(N, N, E_idx),  # N-N
            self._get_rbf(C, C, E_idx),  # C-C
            self._get_rbf(O, O, E_idx),  # O-O
            self._get_rbf(Cb, Cb, E_idx),  # Cb-Cb
            self._get_rbf(Ca, N, E_idx),  # Ca-N
            self._get_rbf(Ca, C, E_idx),  # Ca-C
            self._get_rbf(Ca, O, E_idx),  # Ca-O
            self._get_rbf(Ca, Cb, E_idx),  # Ca-Cb
            self._get_rbf(N, C, E_idx),  # N-C
            self._get_rbf(N, O, E_idx),  # N-O
            self._get_rbf(N, Cb, E_idx),  # N-Cb
            self._get_rbf(Cb, C, E_idx),  # Cb-C
            self._get_rbf(Cb, O, E_idx),  # Cb-O
            self._get_rbf(O, C, E_idx),  # O-C
            self._get_rbf(N, Ca, E_idx),  # N-Ca
            self._get_rbf(C, Ca, E_idx),  # C-Ca
            self._get_rbf(O, Ca, E_idx),  # O-Ca
            self._get_rbf(Cb, Ca, E_idx),  # Cb-Ca
            self._get_rbf(C, N, E_idx),  # C-N
            self._get_rbf(O, N, E_idx),  # O-N
            self._get_rbf(Cb, N, E_idx),  # Cb-N
            self._get_rbf(C, Cb, E_idx),  # C-Cb
            self._get_rbf(O, Cb, E_idx),  # O-Cb
            self._get_rbf(C, O, E_idx),  # C-O
        ]
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]  # [B, L, K]

        d_chains = ((chain_labels[:, :, None] - chain_labels[:, None, :]) == 0).long()
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, E_idx


class ProteinMPNN(nn.Module):
    def __init__(
        self,
        num_letters,
        node_features,
        edge_features,
        hidden_dim,
        num_encoder_layers=3,
        num_decoder_layers=3,
        vocab=21,
        k_neighbors=64,
        augment_eps=0.05,
        dropout=0.1,
        ca_only=False,
    ):
        super(ProteinMPNN, self).__init__()

        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        self.features = ProteinFeatures(
            node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps
        )

        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)

        self.encoder_layers = nn.ModuleList(
            [
                EncLayer(hidden_dim, hidden_dim * 2, dropout=dropout)
                for _ in range(num_encoder_layers)
            ]
        )

        self.decoder_layers = nn.ModuleList(
            [
                DecLayer(hidden_dim, hidden_dim * 3, dropout=dropout)
                for _ in range(num_decoder_layers)
            ]
        )
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        X,
        S,
        mask,
        chain_M,
        residue_idx,
        chain_encoding_all,
        randn,
        use_input_decoding_order=False,
        decoding_order=None,
    ):
        """Graph-conditioned sequence model"""
        device = X.device
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        chain_M = chain_M * mask  # update chain_M to include missing regions

        if not use_input_decoding_order:
            decoding_order = torch.tensor([list(range(X.size(1)))], device=device)

        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(
            decoding_order, num_classes=mask_size
        ).float()
        order_mask_backward = torch.einsum(
            "ij, biq, bjp->bqp",
            (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
            permutation_matrix_reverse,
            permutation_matrix_reverse,
        )
        order_mask_backward = torch.ones_like(order_mask_backward)

        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1.0 - mask_attend)

        all_hidden = []
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = layer(h_V, h_ESV, mask)
            all_hidden.append(h_V)

        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return list(reversed(all_hidden)), h_S, log_probs


# ---- transfer_model.py (verbatim TransferModel + LightAttention) -------------------


class LightAttention(nn.Module):
    """Source:
    Hannes Stark et al. 2022
    https://github.com/HannesStark/protein-localization/blob/master/models/light_attention.py
    """

    def __init__(
        self,
        embeddings_dim=1024,
        output_dim=11,
        dropout=0.25,
        kernel_size=9,
        conv_dropout: float = 0.25,
    ):
        super(LightAttention, self).__init__()

        self.feature_convolution = nn.Conv1d(
            embeddings_dim, embeddings_dim, kernel_size, stride=1, padding=kernel_size // 2
        )
        self.attention_convolution = nn.Conv1d(
            embeddings_dim, embeddings_dim, kernel_size, stride=1, padding=kernel_size // 2
        )

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(conv_dropout)

    def forward(self, x: torch.Tensor, mask, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor
            mask: [batch_size, sequence_length] mask corresponding to zero padding
        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        o = self.feature_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        o = self.dropout(o)

        attention = self.attention_convolution(x)  # [batch_size, embeddings_dim, sequence_length]

        o1 = o * self.softmax(attention)
        return torch.squeeze(o1)


class TransferModel(nn.Module):
    """Real `TransferModel` architecture (transfer-learned ddG head over frozen
    ProteinMPNN embeddings). `forward()` here takes the already-featurized tensors
    that upstream `tied_featurize()` would have produced, plus a single mutation
    (position, wildtype amino-acid index, mutant amino-acid index) -- every downstream
    real nn.Module call is unmodified from the upstream `TransferModel.forward`."""

    def __init__(
        self,
        hidden_dims,
        num_final_layers=1,
        lightattn=True,
        subtract_mut=True,
        mpnn_hidden_dim=128,
        mpnn_num_encoder_layers=3,
        mpnn_num_decoder_layers=3,
        mpnn_k_neighbors=48,
        mpnn_vocab=21,
    ):
        super().__init__()
        self.hidden_dims = list(hidden_dims)
        self.subtract_mut = subtract_mut
        self.num_final_layers = num_final_layers
        self.lightattn = lightattn

        self.prot_mpnn = ProteinMPNN(
            ca_only=False,
            num_letters=21,
            node_features=mpnn_hidden_dim,
            edge_features=mpnn_hidden_dim,
            hidden_dim=mpnn_hidden_dim,
            num_encoder_layers=mpnn_num_encoder_layers,
            num_decoder_layers=mpnn_num_decoder_layers,
            vocab=mpnn_vocab,
            k_neighbors=mpnn_k_neighbors,
            augment_eps=0.0,
        )
        embed_dim = mpnn_hidden_dim
        hidden_dim = mpnn_hidden_dim

        hid_sizes = [hidden_dim * self.num_final_layers + embed_dim]
        hid_sizes += self.hidden_dims
        hid_sizes += [VOCAB_DIM]

        if self.lightattn:
            self.light_attention = LightAttention(
                embeddings_dim=hidden_dim * self.num_final_layers + embed_dim
            )

        self.both_out = nn.Sequential()
        for sz1, sz2 in zip(hid_sizes, hid_sizes[1:]):
            self.both_out.append(nn.ReLU())
            self.both_out.append(nn.Linear(sz1, sz2))

        self.ddg_out = nn.Linear(1, 1)

    def forward(
        self,
        X,
        S,
        mask,
        chain_M,
        residue_idx,
        chain_encoding_all,
        mut_position,
        mut_wildtype_idx,
        mut_mutant_idx,
    ):
        # getting ProteinMPNN structure embeddings (real backbone forward call)
        all_mpnn_hid, mpnn_embed, _ = self.prot_mpnn(
            X, S, mask, chain_M, residue_idx, chain_encoding_all, None
        )
        if self.num_final_layers > 0:
            mpnn_hid = torch.cat(all_mpnn_hid[: self.num_final_layers], -1)

        hid = mpnn_hid[0][mut_position]  # MPNN hidden embeddings at mutated position
        embed = mpnn_embed[0][mut_position]  # MPNN seq embedding at mutated position
        lin_input = torch.cat([hid, embed], -1)

        if self.lightattn:
            lin_input = torch.unsqueeze(torch.unsqueeze(lin_input, -1), 0)
            lin_input = self.light_attention(lin_input, mask)

        both_input = torch.unsqueeze(self.both_out(lin_input), -1)
        ddg_out = self.ddg_out(both_input)

        if self.subtract_mut:
            ddg = ddg_out[mut_mutant_idx][0] - ddg_out[mut_wildtype_idx][0]
        else:
            ddg = ddg_out[mut_mutant_idx][0]
        return ddg


def build_thermompnn():
    # Real released config: mpnn_hidden_dim=128, 3 ProteinMPNN encoder/decoder layers,
    # k_neighbors=48 (checkpoint 'v_48_020.pt'), num_final_layers=1, lightattn=True,
    # subtract_mut=True, hidden_dims=[64] (single-layer ddG MLP head, per
    # `local.yaml`/`config.yaml` defaults). Shrunk to mpnn_hidden_dim=8, 1 encoder/
    # 1 decoder layer, k_neighbors=6 for a menagerie-scale trace.
    return TransferModel(
        hidden_dims=[8],
        num_final_layers=1,
        lightattn=True,
        subtract_mut=True,
        mpnn_hidden_dim=8,
        mpnn_num_encoder_layers=1,
        mpnn_num_decoder_layers=1,
        mpnn_k_neighbors=6,
        mpnn_vocab=21,
    )


def example_input_thermompnn():
    torch.manual_seed(0)
    batch, n_res = 1, 12
    X = torch.randn(batch, n_res, 4, 3)
    S = torch.randint(0, 21, (batch, n_res))
    mask = torch.ones(batch, n_res)
    chain_M = torch.ones(batch, n_res)
    residue_idx = torch.arange(n_res).unsqueeze(0)
    chain_encoding_all = torch.ones(batch, n_res, dtype=torch.long)

    mut_position = 5
    mut_wildtype_idx = ALPHABET.index("A")
    mut_mutant_idx = ALPHABET.index("G")

    return (
        X,
        S,
        mask,
        chain_M,
        residue_idx,
        chain_encoding_all,
        mut_position,
        mut_wildtype_idx,
        mut_mutant_idx,
    )


MENAGERIE_ENTRIES = [
    ("ThermoMPNN", "build_thermompnn", "example_input_thermompnn", 2024, "vendored-pytorch"),
]
