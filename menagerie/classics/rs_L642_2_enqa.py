# SOURCE: vendored from https://github.com/BioinfoMachineLearning/EnQA @ b540a46022a143ab73719e9fd55152e6c1a09dec
#
# EnQA (Chen, Liu & Cheng, Nature Communications 2023) is a single-model protein
# structure quality-assessment (EMA) network: it takes per-residue 1D features, a
# pairwise 2D feature map (distance/contact/embedding tracks), 3D CA coordinates, and
# a k-NN edge list, and predicts a per-residue-pair error-distogram plus a refined
# per-residue lDDT via an E(n)-Equivariant Graph Neural Network (EGNN, Satorras et
# al. 2021) update on top of a dilated-ResNet 2D trunk. The official repo ships three
# EnQA variants (`resEGNN_with_mask`/EGNN_Full, `resEGNN_with_ne`/EGNN_esto9, and a
# separate SE(3)-Transformer variant in network/se3_model.py). We vendor the
# `resEGNN_with_mask` variant (the paper's flagship "EGNN_Full" method, and the
# default in EnQA.py's `--method` CLI) plus its EGNN/ResNet building blocks -- these
# are pure PyTorch with no extra deps. The `se3_model.py` variant needs `dgl` +
# `equivariant_attention` (not in base env) and is intentionally excluded.
#
# The classes below (ResNet, E_GCL/EGNN/EGNN_ne, resEGNN_with_mask) are copied
# unmodified from network/{resnet,EGNN,resEGNN}.py.
#
# ruff: noqa: E402 -- this file amalgamates several original vendored source files,
# each carrying its own local import block; hoisting all imports to the top would
# obscure per-section provenance.

import torch
from torch import nn
from torch.nn import functional as F

MENAGERIE_ZOO = "vendored-pytorch"

# ---- vendored from network/resnet.py ----
# Code from ResNet Architecture in DeepAccNet
# Source: https://github.com/hiranumn/DeepAccNet/blob/main/deepAccNet_noPyRosetta/resnet.py


class ResNet(torch.nn.Module):
    # Parameter initialization
    def __init__(
        self,
        num_channel,
        num_chunks,
        name,
        inorm=False,
        initial_projection=False,
        extra_blocks=False,
        dilation_cycle=[1, 2, 4, 8],
        verbose=False,
    ):
        self.num_channel = num_channel
        self.num_chunks = num_chunks
        self.name = name
        self.inorm = inorm
        self.initial_projection = initial_projection
        self.extra_blocks = extra_blocks
        self.dilation_cycle = dilation_cycle
        self.verbose = verbose

        super(ResNet, self).__init__()

        if self.initial_projection:
            self.add_module(
                "resnet_%s_init_proj" % (self.name), torch.nn.Conv2d(num_channel, num_channel, 1)
            )

        for i in range(self.num_chunks):
            for dilation_rate in self.dilation_cycle:
                if self.inorm:  # noqa: E701
                    self.add_module(
                        "resnet_%s_%i_%i_inorm_1" % (self.name, i, dilation_rate),
                        torch.nn.InstanceNorm2d(num_channel, eps=1e-06, affine=True),
                    )
                    self.add_module(
                        "resnet_%s_%i_%i_inorm_2" % (self.name, i, dilation_rate),
                        torch.nn.InstanceNorm2d(num_channel // 2, eps=1e-06, affine=True),
                    )
                    self.add_module(
                        "resnet_%s_%i_%i_inorm_3" % (self.name, i, dilation_rate),
                        torch.nn.InstanceNorm2d(num_channel // 2, eps=1e-06, affine=True),
                    )

                self.add_module(
                    "resnet_%s_%i_%i_conv2d_1" % (self.name, i, dilation_rate),
                    torch.nn.Conv2d(num_channel, num_channel // 2, 1),
                )
                self.add_module(
                    "resnet_%s_%i_%i_conv2d_2" % (self.name, i, dilation_rate),
                    torch.nn.Conv2d(
                        num_channel // 2,
                        num_channel // 2,
                        3,
                        dilation=dilation_rate,
                        padding=dilation_rate,
                    ),
                )
                self.add_module(
                    "resnet_%s_%i_%i_conv2d_3" % (self.name, i, dilation_rate),
                    torch.nn.Conv2d(num_channel // 2, num_channel, 1),
                )

        if self.extra_blocks:
            for i in range(2):
                if self.inorm:  # noqa: E701
                    self.add_module(
                        "resnet_%s_extra%i_inorm_1" % (self.name, i),
                        torch.nn.InstanceNorm2d(num_channel, eps=1e-06, affine=True),
                    )
                    self.add_module(
                        "resnet_%s_extra%i_inorm_2" % (self.name, i),
                        torch.nn.InstanceNorm2d(num_channel // 2, eps=1e-06, affine=True),
                    )
                    self.add_module(
                        "resnet_%s_extra%i_inorm_3" % (self.name, i),
                        torch.nn.InstanceNorm2d(num_channel // 2, eps=1e-06, affine=True),
                    )

                self.add_module(
                    "resnet_%s_extra%i_conv2d_1" % (self.name, i),
                    torch.nn.Conv2d(num_channel, num_channel // 2, 1),
                )
                self.add_module(
                    "resnet_%s_extra%i_conv2d_2" % (self.name, i),
                    torch.nn.Conv2d(num_channel // 2, num_channel // 2, 3, dilation=1, padding=1),
                )
                self.add_module(
                    "resnet_%s_extra%i_conv2d_3" % (self.name, i),
                    torch.nn.Conv2d(num_channel // 2, num_channel, 1),
                )

    def forward(self, x):
        if self.initial_projection:
            x = self._modules["resnet_%s_init_proj" % (self.name)](x)

        for i in range(self.num_chunks):
            for dilation_rate in self.dilation_cycle:
                _residual = x

                # Internal block
                if self.inorm:
                    x = self._modules["resnet_%s_%i_%i_inorm_1" % (self.name, i, dilation_rate)](x)  # noqa: E701
                x = F.elu(x)
                x = self._modules["resnet_%s_%i_%i_conv2d_1" % (self.name, i, dilation_rate)](x)

                if self.inorm:
                    x = self._modules["resnet_%s_%i_%i_inorm_2" % (self.name, i, dilation_rate)](x)  # noqa: E701
                x = F.elu(x)
                x = self._modules["resnet_%s_%i_%i_conv2d_2" % (self.name, i, dilation_rate)](x)

                if self.inorm:
                    x = self._modules["resnet_%s_%i_%i_inorm_3" % (self.name, i, dilation_rate)](x)  # noqa: E701
                x = F.elu(x)
                x = self._modules["resnet_%s_%i_%i_conv2d_3" % (self.name, i, dilation_rate)](x)

                x = x + _residual

        if self.extra_blocks:
            for i in range(2):
                _residual = x

                # Internal block
                if self.inorm:
                    x = self._modules["resnet_%s_extra%i_inorm_1" % (self.name, i)](x)  # noqa: E701
                x = F.elu(x)
                x = self._modules["resnet_%s_extra%i_conv2d_1" % (self.name, i)](x)

                if self.inorm:
                    x = self._modules["resnet_%s_extra%i_inorm_2" % (self.name, i)](x)  # noqa: E701
                x = F.elu(x)
                x = self._modules["resnet_%s_extra%i_conv2d_2" % (self.name, i)](x)

                if self.inorm:
                    x = self._modules["resnet_%s_extra%i_inorm_3" % (self.name, i)](x)  # noqa: E701
                x = F.elu(x)
                x = self._modules["resnet_%s_extra%i_conv2d_3" % (self.name, i)](x)

                x = x + _residual

        return x


# ---- vendored from network/EGNN.py ----
# Code from E(n) Equivariant Graph Neural Networks
# Victor Garcia Satorras, Emiel Hogeboom, Max Welling
# https://arxiv.org/abs/2102.09844
# https://github.com/vgsatorras/egnn/blob/1a2d515e069272484b4ff51930a3d5df3eeef92b/models/egnn_clean/egnn_clean.py


class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    """

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        edges_in_d=0,
        act_fn=nn.SiLU(),
        residual=True,
        attention=False,
        normalize=False,
        coords_agg="mean",
        tanh=False,
    ):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf), act_fn, nn.Linear(hidden_nf, output_nf)
        )
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)
        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
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
        if self.coords_agg == "sum":
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == "mean":
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception("Wrong coords_agg parameter" % self.coords_agg)
        coord += agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)
        if self.normalize:
            norm = torch.sqrt(radial) + self.epsilon
            coord_diff = coord_diff / norm
        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        return h, coord, edge_attr


class EGNN(nn.Module):
    def __init__(
        self,
        in_node_nf,
        hidden_nf,
        out_node_nf,
        in_edge_nf=0,
        act_fn=nn.SiLU(),
        n_layers=4,
        residual=True,
        attention=False,
        normalize=False,
        tanh=False,
    ):
        """
        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param in_edge_nf: Number of features for the edge features
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Sigma(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Sigma(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.
        """
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module(
                "gcl_%d" % i,
                E_GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    edges_in_d=in_edge_nf,
                    act_fn=act_fn,
                    residual=residual,
                    attention=attention,
                    normalize=normalize,
                    tanh=tanh,
                ),
            )

    def forward(self, h, x, edges, edge_attr):
        h = self.embedding_in(h)
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr)
        h = self.embedding_out(h)
        return h, x


class EGNN_ne(nn.Module):
    def __init__(
        self,
        in_node_nf,
        hidden_nf,
        out_node_nf,
        in_edge_nf=0,
        act_fn=nn.SiLU(),
        n_layers=1,
        residual=True,
        attention=False,
        normalize=False,
        tanh=False,
        num_k=3,
    ):
        super(EGNN_ne, self).__init__()
        self.hidden_nf = hidden_nf
        self.in_edge_nf = in_edge_nf
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.e_embedding_in = nn.Linear(self.in_edge_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        self.num_k = num_k
        for i in range(0, n_layers):
            self.add_module(
                "gcl_%d" % i,
                E_GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    edges_in_d=hidden_nf,
                    act_fn=act_fn,
                    residual=residual,
                    attention=attention,
                    normalize=normalize,
                    tanh=tanh,
                ),
            )
            self.add_module(
                "gcl_e_%d" % i,
                E_GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    edges_in_d=1,
                    act_fn=act_fn,
                    residual=residual,
                    attention=attention,
                    normalize=normalize,
                    tanh=tanh,
                ),
            )
        i += 1
        self.add_module(
            "gcl_%d" % i,
            E_GCL(
                self.hidden_nf,
                self.hidden_nf,
                self.hidden_nf,
                edges_in_d=hidden_nf,
                act_fn=act_fn,
                residual=residual,
                attention=attention,
                normalize=normalize,
                tanh=tanh,
            ),
        )

    def forward(self, h, x, edges, edge_attr):
        h = self.embedding_in(h)
        edge_attr = self.e_embedding_in(edge_attr)
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr)
            e_x = (x[edges[0]] + x[edges[1]]) / 2
            edge_dist = torch.cdist(e_x, e_x)
            edge_dist = edge_dist.fill_diagonal_(torch.max(edge_dist).item())
            e_edges = (
                torch.topk(edge_dist, self.num_k, dim=1, largest=False).indices.flatten(),
                torch.arange(0, e_x.shape[0]).unsqueeze(1).repeat(1, self.num_k).flatten(),
            )
            e_edge_attr = torch.norm(e_x[e_edges[0]] - e_x[e_edges[1]], dim=1).reshape(
                (e_edges[0].shape[0], 1)
            )
            edge_attr_e, _, _ = self._modules["gcl_e_%d" % i](
                edge_attr, e_edges, e_x, edge_attr=e_edge_attr
            )
            edge_attr = edge_attr + edge_attr_e
        i += 1
        h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr)
        h = self.embedding_out(h)
        return h, x


def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


# ---- vendored from network/resEGNN.py ----
# Part of the convolution layers and l-DDT computation are modified from DeepAccNet
# Improved protein structure refinement guided by deep learning based accuracy estimation
# Naozumi Hiranuma, Hahnbeom Park, Minkyung Baek,  View ORCID ProfileIvan Anishchanka, Justas Dauparas, David Baker
# Nature Communications doi: 10.1038/s41467-021-21511-x
# https://github.com/hiranumn/DeepAccNet


class resEGNN_with_mask(torch.nn.Module):
    def __init__(
        self,
        dim1d=23,
        dim2d=41,
        bin_num=9,
        num_chunks=5,
        num_channel=32,
        num_att_layers=6,
        name=None,
        verbose=False,
    ):
        self.dim1d = dim1d
        self.dim2d = dim2d
        self.num_chunks = num_chunks
        self.num_channel = num_channel
        self.name = name
        self.verbose = verbose
        self.bin_num = bin_num
        self.num_att_layers = num_att_layers
        super(resEGNN_with_mask, self).__init__()

        self.add_module(
            "conv1d_1", torch.nn.Conv1d(dim1d, self.num_channel // 2, 1, padding=0, bias=True)
        )
        self.add_module(
            "conv2d_1",
            torch.nn.Conv2d(
                self.num_channel + self.dim2d, self.num_channel, 1, padding=0, bias=True
            ),
        )
        self.add_module(
            "inorm_1", torch.nn.InstanceNorm2d(self.num_channel, eps=1e-06, affine=True)
        )
        self.add_module(
            "base_resnet",
            ResNet(
                num_channel,
                self.num_chunks,
                "base_resnet",
                inorm=True,
                initial_projection=True,
                extra_blocks=False,
            ),
        )

        self.add_module(
            "bin_resnet",
            ResNet(
                num_channel,
                1,
                "bin_resnet",
                inorm=False,
                initial_projection=True,
                extra_blocks=True,
            ),
        )
        self.add_module(
            "conv2d_bin", torch.nn.Conv2d(self.num_channel, self.bin_num, 1, padding=0, bias=True)
        )
        self.add_module(
            "EGNN_layers",
            EGNN(
                in_node_nf=self.num_channel // 2 + 1,
                hidden_nf=self.num_channel,
                out_node_nf=1,
                in_edge_nf=self.num_channel + 9,
                n_layers=4,
            ),
        )

    def calculate_LDDT(self, estogram, mask, center=4):
        # Get on the same device as indices
        device = estogram.device

        # Remove diagonal from calculation
        nres = mask.shape[-1]
        mask = torch.mul(
            mask, torch.ones((nres, nres), device=device) - torch.eye(nres, device=device)
        )
        masked = torch.mul(estogram.squeeze(), mask)

        p0 = (masked[center]).sum(axis=0)
        p1 = (masked[center - 1] + masked[center + 1]).sum(axis=0)
        p2 = (masked[center - 2] + masked[center + 2]).sum(axis=0)
        p3 = (masked[center - 3] + masked[center + 3]).sum(axis=0)
        p4 = mask.sum(axis=0)

        return 0.25 * (4.0 * p0 + 3.0 * p1 + 2.0 * p2 + p3) / p4

    def forward(self, f1d, f2d, pos, el, cmap):
        len_x = f1d.shape[-1]
        out_conv1d_1 = F.elu(self._modules["conv1d_1"](f1d))
        f1d_tile = torch.cat(
            (
                torch.repeat_interleave(out_conv1d_1.unsqueeze(2), repeats=len_x, dim=2),
                torch.repeat_interleave(out_conv1d_1.unsqueeze(3), repeats=len_x, dim=3),
            ),
            dim=1,
        )
        out_cat_2 = torch.cat((f1d_tile, f2d), dim=1)
        out_conv2d_1 = self._modules["conv2d_1"](out_cat_2)
        out_inorm_1 = F.elu(self._modules["inorm_1"](out_conv2d_1))

        # First ResNet
        out_base_resnet = F.elu(self._modules["base_resnet"](out_inorm_1))

        out_bin_predictor = F.elu(self._modules["bin_resnet"](out_base_resnet))
        bin_prediction = self._modules["conv2d_bin"](out_bin_predictor)
        bin_prediction = (bin_prediction + bin_prediction.permute(0, 1, 3, 2)) / 2
        estogram_prediction = F.softmax(bin_prediction, dim=1)
        lddt_prediction = self.calculate_LDDT(estogram_prediction.squeeze(), cmap.squeeze())
        edge_features = torch.cat([out_base_resnet, bin_prediction], dim=1)
        edge_features = edge_features[0, :, el[0], el[1]].transpose(0, 1)
        node_features = torch.cat(
            (out_conv1d_1.permute(0, 2, 1).squeeze(), lddt_prediction.unsqueeze(1)), dim=1
        )
        node_features, pos_new = self._modules["EGNN_layers"](node_features, pos, el, edge_features)
        lddt_prediction_final = lddt_prediction + torch.tanh(node_features.squeeze())
        return bin_prediction, pos_new, lddt_prediction_final


# ---- menagerie staging harness ----


def build_enqa():
    """Tiny-sized real EnQA (EGNN_Full / resEGNN_with_mask variant): small channel
    count and a single ResNet dilation chunk so a short protein-length input traces
    quickly while exercising the full 2D-ResNet + EGNN pipeline."""
    torch.manual_seed(0)
    model = resEGNN_with_mask(
        dim1d=23,
        dim2d=41,
        bin_num=9,
        num_chunks=1,
        num_channel=16,
        name="menagerie_enqa",
    )
    return model.eval()


def example_input_enqa():
    """(f1d, f2d, pos, el, cmap): per-residue 1D features (1,23,L), pairwise 2D
    features (1,41,L,L), CA coordinates (L,3), a fully-connected edge-index pair of
    LongTensors, and a per-residue-pair contact mask (L,L). L kept tiny (10
    residues) for a fast real trace."""
    torch.manual_seed(0)
    length = 10
    f1d = torch.rand((1, 23, length))
    f2d = torch.rand((1, 41, length, length))
    pos = torch.rand((length, 3))
    rows, cols = [], []
    for i in range(length):
        for j in range(length):
            if i != j:
                rows.append(i)
                cols.append(j)
    el = (torch.tensor(rows, dtype=torch.long), torch.tensor(cols, dtype=torch.long))
    cmap = torch.rand((length, length))
    return (f1d, f2d, pos, el, cmap)


MENAGERIE_ENTRIES = [
    (
        "EnQA",
        "build_enqa",
        "example_input_enqa",
        2023,
        "vendored-pytorch",
    ),
]
