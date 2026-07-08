# FAITHFUL PORT of karbalayghareh/GraphReg @ master (original framework: TensorFlow /
# tensorflow.keras)
#   train/gat_layer.py :: GraphAttention
#   train/Epi-GraphReg.py :: main()'s model-definition block (Epi-GraphReg)
#
# Epi-GraphReg (Karbalayghareh, Sahin, Leslie, "Chromatin interaction-aware gene
# regulatory modeling with graph attention networks", Genome Research 2022).
# Predicts per-bin CAGE expression along a genomic window from 1D epigenomic
# tracks (chromatin accessibility / histone marks) plus a Hi-C/HiChIP-derived
# 3D chromatin-interaction graph over the same bins. A 3-block 1D-conv stem
# downsamples the epigenomic tracks by 50x (2x then 5x then 5x max-pooling) to
# land on one feature vector per 5kb genomic bin (matching the adjacency
# matrix's node count), then a stack of custom Graph Attention layers (dense,
# batched adjacency-masked self+neighbor attention -- NOT torch_geometric's
# sparse-edge-list GATConv, this is the paper's own from-scratch GAT variant)
# propagates information along the 3D contact graph, and a final 1x1 conv head
# emits a per-bin Poisson rate (CAGE expression).
#
# This is a TensorFlow-native repo (tensorflow.keras.layers.Layer subclass with
# build()/call()/add_weight(), tf.keras.Model functional API) with no PyTorch
# port anywhere upstream, and no PyG/graph-learning package here can construct
# it (GraphAttention's dense-adjacency-masked attention mechanism is bespoke,
# not a wrapper around a standard library GAT). It is transcribed faithfully
# into self-contained base-env torch below: every weight matrix, every
# attention-score computation (self-term + neighbor-term via a learned linear
# split, LeakyReLU nonlinearity, additive adjacency mask, sigmoid + row-sum
# normalization producing a "beta_promoter" self-retention weight -- this GAT
# variant does NOT use softmax attention, per the real code) is reproduced
# exactly as in `GraphAttention.call()`. Multi-head concat/average reduction,
# dropout, and bias are preserved. The stem/head conv stack mirrors
# `Epi-GraphReg.py`'s literal layer sequence (kernel sizes, pool strides,
# activations, channel counts).

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


# ---- train/gat_layer.py :: GraphAttention (faithful port) ----


class GraphAttention(nn.Module):
    """Dense, adjacency-masked graph attention layer, per GraphAttention.call()
    in the real repo. Operates on a full (batch, num_nodes, num_nodes) dense
    adjacency (no sparse edge_index) -- this is the real code's own bespoke GAT,
    not torch_geometric's GATConv."""

    def __init__(
        self,
        in_features,
        out_features,
        attn_heads=1,
        attn_heads_reduction="concat",
        dropout_rate=0.0,
        activation="elu",
        use_bias=False,
    ):
        super().__init__()
        if attn_heads_reduction not in {"concat", "average"}:
            raise ValueError("Possible reduction methods: concat, average")

        self.F_ = out_features
        self.attn_heads = attn_heads
        self.attn_heads_reduction = attn_heads_reduction
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias

        if activation == "elu":
            self.activation = nn.ELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()

        self.output_dim = self.F_ * self.attn_heads if attn_heads_reduction == "concat" else self.F_

        self.kernel_self = nn.ParameterList()
        self.kernel_neighs = nn.ParameterList()
        self.attn_kernel_self = nn.ParameterList()
        self.attn_kernel_neighs = nn.ParameterList()
        self.biases = nn.ParameterList() if use_bias else None

        for _ in range(attn_heads):
            k_self = nn.Parameter(torch.empty(in_features, self.F_))
            k_neigh = nn.Parameter(torch.empty(in_features, self.F_))
            nn.init.xavier_uniform_(k_self)
            nn.init.xavier_uniform_(k_neigh)
            self.kernel_self.append(k_self)
            self.kernel_neighs.append(k_neigh)

            a_self = nn.Parameter(torch.empty(self.F_, 1))
            a_neigh = nn.Parameter(torch.empty(self.F_, 1))
            nn.init.xavier_uniform_(a_self)
            nn.init.xavier_uniform_(a_neigh)
            self.attn_kernel_self.append(a_self)
            self.attn_kernel_neighs.append(a_neigh)

            if use_bias:
                self.biases.append(nn.Parameter(torch.zeros(self.F_)))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, adj):
        """
        x:   (B, N, F) node features
        adj: (B, N, N) dense adjacency matrix
        """
        outputs = []
        atts = []
        for head in range(self.attn_heads):
            features_self = x @ self.kernel_self[head]  # (B, N, F')
            features_neighs = x @ self.kernel_neighs[head]  # (B, N, F')

            attn_for_self = features_self @ self.attn_kernel_self[head]  # (B, N, 1)
            attn_for_neighs = features_neighs @ self.attn_kernel_neighs[head]  # (B, N, 1)

            # a(Wh_i, Wh_j) = a_1^T Wh_i + a_2^T Wh_j, broadcast over pairs (i, j)
            att = attn_for_self + attn_for_neighs.transpose(1, 2)  # (B, N, N)
            att = self.leaky_relu(att)

            mask = -10e15 * (1.0 - adj)
            att = att + mask

            att = torch.sigmoid(att)
            att_sum = att.sum(dim=-1, keepdim=True)
            att = att / (1 + att_sum)
            beta_promoter = 1 / (1 + att_sum)

            atts.append(att)

            dropout_feat_neigh = self.dropout(features_neighs)
            dropout_feat_self = self.dropout(features_self)

            node_features = dropout_feat_self * beta_promoter + torch.bmm(att, dropout_feat_neigh)

            if self.use_bias:
                node_features = node_features + self.biases[head]

            outputs.append(node_features)

        if self.attn_heads_reduction == "concat":
            output = torch.cat(outputs, dim=-1)
        else:
            output = torch.stack(outputs, dim=0).mean(dim=0)

        output = self.activation(output)
        return output, atts


# ---- train/Epi-GraphReg.py :: model-definition block (faithful port) ----


class EpiGraphReg(nn.Module):
    """Conv1D stem (3x Conv+BN+MaxPool, downsampling 50x: 2x*5x*5x) -> stack of
    GraphAttention layers over the Hi-C/HiChIP contact graph -> 1x1 conv head
    -> per-bin Poisson-rate (CAGE) output, per Epi-GraphReg.py's functional
    Keras model definition."""

    def __init__(
        self, n_epi_features=3, n_gat_layers=2, gat_f_out=64, n_attn_heads=4, dropout_rate=0.0
    ):
        super().__init__()
        self.n_gat_layers = n_gat_layers

        # stem: (B, L, F) epigenomic tracks -> (B, N, 128) per-bin features (N = L // 50)
        self.conv1 = nn.Conv1d(n_epi_features, 128, kernel_size=25, padding=12)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(5)

        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(5)

        self.dropout = nn.Dropout(dropout_rate)

        gat_layers = []
        gat_bns = []
        in_dim = 128
        for _ in range(n_gat_layers):
            gat_layers.append(
                GraphAttention(
                    in_dim,
                    gat_f_out,
                    attn_heads=n_attn_heads,
                    attn_heads_reduction="concat",
                    dropout_rate=dropout_rate,
                    activation="elu",
                )
            )
            out_dim = gat_f_out * n_attn_heads
            gat_bns.append(nn.BatchNorm1d(out_dim))
            in_dim = out_dim
        self.gat_layers = nn.ModuleList(gat_layers)
        self.gat_bns = nn.ModuleList(gat_bns)

        self.conv_head = nn.Conv1d(in_dim, 64, kernel_size=1)
        self.bn_head = nn.BatchNorm1d(64)
        self.conv_out = nn.Conv1d(64, 1, kernel_size=1)

    def forward(self, x_epi, adj):
        """
        x_epi: (B, L, F) raw 1D epigenomic tracks (channels-last, matching the
               real Keras Input(shape=(3*T*b, F)))
        adj:   (B, N, N) dense 3D-contact adjacency over the N = L // 50 bins
        """
        x = x_epi.transpose(1, 2)  # (B, F, L) for torch Conv1d

        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)

        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)

        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.pool3(x)  # (B, 128, N)

        x = x.transpose(1, 2)  # (B, N, 128) node features for GAT

        atts = []
        for gat, bn in zip(self.gat_layers, self.gat_bns):
            x, att_ = gat(x, adj)
            x = bn(x.transpose(1, 2)).transpose(1, 2)
            atts.append(att_)

        x = self.dropout(x)
        x = x.transpose(1, 2)  # (B, C, N)
        x = F.relu(self.conv_head(x))
        x = self.bn_head(x)

        mu_cage = torch.exp(self.conv_out(x))  # 'exponential' activation, matches real Keras layer
        mu_cage = mu_cage.squeeze(1)  # (B, N)
        return mu_cage, atts


def build_graphreg_epi():
    torch.manual_seed(0)
    return EpiGraphReg(
        n_epi_features=3, n_gat_layers=2, gat_f_out=8, n_attn_heads=2, dropout_rate=0.0
    ).eval()


def example_input_graphreg_epi():
    torch.manual_seed(0)
    batch = 2
    n_bins = 6  # N: number of 5kb bins in the (tiny) genomic window
    bin_len = 50  # 50x downsampling stem (2 * 5 * 5)
    seq_len = n_bins * bin_len
    n_epi_features = 3
    x_epi = torch.randn(batch, seq_len, n_epi_features)
    # symmetric contact adjacency with self-loops, values in [0, 1]
    raw = torch.rand(batch, n_bins, n_bins)
    adj = ((raw + raw.transpose(1, 2)) / 2 > 0.5).float()
    eye = torch.eye(n_bins).unsqueeze(0).expand(batch, -1, -1)
    adj = torch.clamp(adj + eye, max=1.0)
    return (x_epi, adj)


MENAGERIE_ENTRIES = [
    ("Epi-GraphReg", build_graphreg_epi, example_input_graphreg_epi, 2022, "ported-pytorch"),
]
