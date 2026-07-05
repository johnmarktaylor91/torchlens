# SOURCE: vendored from Shohruh72/3DGazeNet @ 58d3fcac58efd95bd455e91551c2f7cfaf8a40b3 (nets/nn.py)
# SOURCE: vendored from MichiganCOG/A2CL-PT @ master, cloned 2026-07-05 (model.py)
# SOURCE: vendored from ispc-lab/ACM-Net @ master, cloned 2026-07-05 (model/ACMNet.py)
# SOURCE: vendored from SitaoLuan/ACM-GNN @ master, cloned 2026-07-05 (ACM-Pytorch/models)
# SOURCE: vendored from fogradio/ACPKAN @ d6e9c4aee068ac4ff651cefe8ab6b9a8543e536b

from __future__ import annotations

from math import floor, sqrt
from types import SimpleNamespace

import timm
import torch
import torch.nn.functional as F
import torch.nn.init as torch_init
from torch import nn
from torch.nn import Parameter

MENAGERIE_ZOO = "vendored-pytorch"


def weights_init(module: nn.Module) -> None:
    """Initialize linear and convolutional layers as in A2CL-PT.

    Parameters
    ----------
    module
        Module to initialize.
    """
    classname = module.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("Linear") != -1:
        torch_init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.zero_()


class GazeNet(nn.Module):
    """3DGazeNet timm-backbone regressor."""

    def __init__(self, backbone_id: str) -> None:
        """Initialize the 3DGazeNet backbone.

        Parameters
        ----------
        backbone_id
            timm model identifier.
        """
        super().__init__()
        self.backbone = timm.create_model(backbone_id, num_classes=481 * 2 * 3)
        self.loss = torch.nn.L1Loss(reduction="mean")
        self.hard_mining = False
        self.num_face = 1103
        self.num_eye = 481 * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the image backbone.

        Parameters
        ----------
        x
            Input image tensor.

        Returns
        -------
        torch.Tensor
            Predicted 3D eye vertex coordinates flattened per example.
        """
        return self.backbone(x)


class A2CLPTModel(nn.Module):
    """A2CL-PT two-stream temporal localization model."""

    def __init__(self, num_class: int, s: int, omega: float) -> None:
        """Initialize A2CL-PT.

        Parameters
        ----------
        num_class
            Number of action classes.
        s
            Temporal erase divisor.
        omega
            Attention branch scaling factor.
        """
        super().__init__()
        self.num_class = num_class
        self.s = s
        self.omega = omega

        dim = 1024
        dropout = 0.7

        self.fc_r = nn.Linear(dim, dim)
        self.fc1_r = nn.Linear(dim, dim)
        self.fc_f = nn.Linear(dim, dim)
        self.fc1_f = nn.Linear(dim, dim)
        self.classifier_r = nn.Conv1d(dim, num_class, kernel_size=1)
        self.classifier_f = nn.Conv1d(dim, num_class, kernel_size=1)
        self.classifier_ra = nn.ModuleList(
            [nn.Conv1d(dim, 1, kernel_size=1) for _ in range(num_class)]
        )
        self.classifier_fa = nn.ModuleList(
            [nn.Conv1d(dim, 1, kernel_size=1) for _ in range(num_class)]
        )

        self.dropout_r = nn.Dropout(dropout)
        self.dropout_f = nn.Dropout(dropout)

        self.apply(weights_init)

        self.mul_r = nn.Parameter(data=torch.ones(num_class))
        self.mul_f = nn.Parameter(data=torch.ones(num_class))

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """Run the temporal action localization model.

        Parameters
        ----------
        inputs
            Concatenated RGB/flow feature tensor of shape ``(batch, time, 2048)``.

        Returns
        -------
        tuple
            RGB features/logits, flow features/logits, and fused temporal CAM.
        """
        _batch, time, dim = inputs.shape
        dim //= 2
        x_r = F.relu(self.fc_r(inputs[:, :, :dim]))
        x_f = F.relu(self.fc_f(inputs[:, :, dim:]))
        x_r = F.relu(self.fc1_r(x_r)).permute(0, 2, 1)
        x_f = F.relu(self.fc1_f(x_f)).permute(0, 2, 1)

        x_r = self.dropout_r(x_r)
        x_f = self.dropout_f(x_f)

        k = max(time - floor(time / self.s), 1)
        cls_x_r = self.classifier_r(x_r).permute(0, 2, 1)
        cls_x_f = self.classifier_f(x_f).permute(0, 2, 1)
        cls_x_ra = cls_x_r.new_zeros(cls_x_r.shape)
        cls_x_fa = cls_x_f.new_zeros(cls_x_f.shape)
        cls_x_rat = cls_x_r.new_zeros(cls_x_r.shape)
        cls_x_fat = cls_x_f.new_zeros(cls_x_f.shape)

        mask_value = -100

        for i in range(self.num_class):
            mask_r = cls_x_r[:, :, i] > torch.kthvalue(cls_x_r[:, :, i], k, dim=1, keepdim=True)[0]
            x_r_erased = torch.masked_fill(x_r, mask_r.unsqueeze(1), 0)
            cls_x_ra[:, :, i] = torch.masked_fill(
                self.classifier_ra[i](x_r_erased).squeeze(1), mask_r, mask_value
            )
            cls_x_rat[:, :, i] = self.classifier_ra[i](x_r).squeeze(1)

            mask_f = cls_x_f[:, :, i] > torch.kthvalue(cls_x_f[:, :, i], k, dim=1, keepdim=True)[0]
            x_f_erased = torch.masked_fill(x_f, mask_f.unsqueeze(1), 0)
            cls_x_fa[:, :, i] = torch.masked_fill(
                self.classifier_fa[i](x_f_erased).squeeze(1), mask_f, mask_value
            )
            cls_x_fat[:, :, i] = self.classifier_fa[i](x_f).squeeze(1)

        tcam = (cls_x_r + cls_x_rat * self.omega) * self.mul_r + (
            cls_x_f + cls_x_fat * self.omega
        ) * self.mul_f

        return (
            x_r.permute(0, 2, 1),
            [cls_x_r, cls_x_ra],
            x_f.permute(0, 2, 1),
            [cls_x_f, cls_x_fa],
            tcam,
        )


class ACMNet(nn.Module):
    """Action Context Modeling Network."""

    def __init__(self, args: SimpleNamespace) -> None:
        """Initialize ACM-Net.

        Parameters
        ----------
        args
            Namespace containing the official model hyperparameters.
        """
        super().__init__()
        self.dataset = args.dataset
        self.feature_dim = args.feature_dim
        self.action_cls_num = args.action_cls_num
        self.drop_thresh = args.dropout
        self.ins_topk_seg = args.ins_topk_seg
        self.con_topk_seg = args.con_topk_seg
        self.bak_topk_seg = args.bak_topk_seg

        self.dropout = nn.Dropout(args.dropout)
        if self.dataset == "THUMOS":
            self.feature_embedding = nn.Sequential(
                nn.Conv1d(
                    in_channels=self.feature_dim,
                    out_channels=self.feature_dim,
                    kernel_size=3,
                    padding=1,
                ),
                nn.ReLU(),
            )
        else:
            self.feature_embedding = nn.Sequential(
                nn.Dropout(args.dropout),
                nn.Conv1d(
                    in_channels=self.feature_dim,
                    out_channels=self.feature_dim,
                    kernel_size=3,
                    padding=1,
                ),
                nn.ReLU(),
            )

        self.att_branch = nn.Conv1d(
            in_channels=self.feature_dim, out_channels=3, kernel_size=1, padding=0
        )
        self.snippet_cls = nn.Linear(
            in_features=self.feature_dim, out_features=(self.action_cls_num + 1)
        )

    def forward(
        self, input_features: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Run ACM-Net on video snippet features.

        Parameters
        ----------
        input_features
            Tensor of shape ``(batch, time, feature_dim)``.

        Returns
        -------
        tuple
            Classification, feature, attention, and class activation outputs.
        """
        device = input_features.device
        batch_size, temp_len = input_features.shape[0], input_features.shape[1]

        inst_topk_num = max(temp_len // self.ins_topk_seg, 1)
        cont_topk_num = max(temp_len // self.con_topk_seg, 1)
        back_topk_num = max(temp_len // self.bak_topk_seg, 1)

        input_features = input_features.permute(0, 2, 1)
        embeded_feature = self.feature_embedding(input_features)

        if self.dataset == "THUMOS":
            temp_att = self.att_branch((embeded_feature))
        else:
            temp_att = self.att_branch(self.dropout(embeded_feature))

        temp_att = temp_att.permute(0, 2, 1)
        temp_att = torch.softmax(temp_att, dim=2)

        act_inst_att = temp_att[:, :, 0].unsqueeze(2)
        act_cont_att = temp_att[:, :, 1].unsqueeze(2)
        act_back_att = temp_att[:, :, 2].unsqueeze(2)

        embeded_feature = embeded_feature.permute(0, 2, 1)
        embeded_feature_rev = embeded_feature

        select_idx = torch.ones((batch_size, temp_len, 1), device=device)
        select_idx = self.dropout(select_idx)
        embeded_feature = embeded_feature * select_idx

        act_cas = self.snippet_cls(self.dropout(embeded_feature))
        act_inst_cas = act_cas * act_inst_att
        act_cont_cas = act_cas * act_cont_att
        act_back_cas = act_cas * act_back_att

        sorted_inst_cas, _ = torch.sort(act_inst_cas, dim=1, descending=True)
        sorted_cont_cas, _ = torch.sort(act_cont_cas, dim=1, descending=True)
        sorted_back_cas, _ = torch.sort(act_back_cas, dim=1, descending=True)

        act_inst_cls = torch.mean(sorted_inst_cas[:, :inst_topk_num, :], dim=1)
        act_cont_cls = torch.mean(sorted_cont_cas[:, :cont_topk_num, :], dim=1)
        act_back_cls = torch.mean(sorted_back_cas[:, :back_topk_num, :], dim=1)
        act_inst_cls = torch.softmax(act_inst_cls, dim=1)
        act_cont_cls = torch.softmax(act_cont_cls, dim=1)
        act_back_cls = torch.softmax(act_back_cls, dim=1)

        act_inst_cas = torch.softmax(act_inst_cas, dim=2)
        act_cont_cas = torch.softmax(act_cont_cas, dim=2)
        act_back_cas = torch.softmax(act_back_cas, dim=2)

        act_cas = torch.softmax(act_cas, dim=2)

        _, sorted_act_inst_att_idx = torch.sort(act_inst_att, dim=1, descending=True)
        _, sorted_act_cont_att_idx = torch.sort(act_cont_att, dim=1, descending=True)
        _, sorted_act_back_att_idx = torch.sort(act_back_att, dim=1, descending=True)
        act_inst_feat_idx = sorted_act_inst_att_idx[:, :inst_topk_num, :].expand(
            [-1, -1, self.feature_dim]
        )
        act_cont_feat_idx = sorted_act_cont_att_idx[:, :cont_topk_num, :].expand(
            [-1, -1, self.feature_dim]
        )
        act_back_feat_idx = sorted_act_back_att_idx[:, :back_topk_num, :].expand(
            [-1, -1, self.feature_dim]
        )
        act_inst_feat = torch.mean(torch.gather(embeded_feature_rev, 1, act_inst_feat_idx), dim=1)
        act_cont_feat = torch.mean(torch.gather(embeded_feature_rev, 1, act_cont_feat_idx), dim=1)
        act_back_feat = torch.mean(torch.gather(embeded_feature_rev, 1, act_back_feat_idx), dim=1)

        return (
            act_inst_cls,
            act_cont_cls,
            act_back_cls,
            act_inst_feat,
            act_cont_feat,
            act_back_feat,
            temp_att,
            act_inst_cas,
            act_cas,
            act_cont_cas,
            act_back_cas,
        )


class GraphConvolution(nn.Module):
    """ACM-GCN adaptive low/high/MLP channel mixing layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        nnodes: int,
        model_type: str,
        output_layer: int = 0,
        variant: bool = False,
        structure_info: int = 0,
    ) -> None:
        """Initialize an ACM-GCN graph convolution layer.

        Parameters
        ----------
        in_features
            Input feature width.
        out_features
            Output feature width.
        nnodes
            Number of graph nodes.
        model_type
            Official ACM model variant string.
        output_layer
            Whether this is the output layer.
        variant
            Whether to use the variant activation order.
        structure_info
            Whether to include structural channels.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.output_layer = output_layer
        self.model_type = model_type
        self.structure_info = structure_info
        self.variant = variant
        self.att_low, self.att_high, self.att_mlp = 0, 0, 0
        self.weight_low = Parameter(torch.empty(in_features, out_features))
        self.weight_high = Parameter(torch.empty(in_features, out_features))
        self.weight_mlp = Parameter(torch.empty(in_features, out_features))
        self.att_vec_low = Parameter(torch.empty(out_features, 1))
        self.att_vec_high = Parameter(torch.empty(out_features, 1))
        self.att_vec_mlp = Parameter(torch.empty(out_features, 1))
        self.layer_norm_low = nn.LayerNorm(out_features)
        self.layer_norm_high = nn.LayerNorm(out_features)
        self.layer_norm_mlp = nn.LayerNorm(out_features)
        self.layer_norm_struc_low = nn.LayerNorm(out_features)
        self.layer_norm_struc_high = nn.LayerNorm(out_features)
        self.att_struc_low = Parameter(torch.empty(out_features, 1))
        self.struc_low = Parameter(torch.empty(nnodes, out_features))
        self.att_vec = Parameter(torch.empty(4, 4) if self.structure_info else torch.empty(3, 3))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset layer parameters."""
        stdv = 1.0 / sqrt(self.weight_mlp.size(1))
        std_att = 1.0 / sqrt(self.att_vec_mlp.size(1))
        std_att_vec = 1.0 / sqrt(self.att_vec.size(1))

        self.weight_low.data.uniform_(-stdv, stdv)
        self.weight_high.data.uniform_(-stdv, stdv)
        self.weight_mlp.data.uniform_(-stdv, stdv)
        self.struc_low.data.uniform_(-stdv, stdv)

        self.att_vec_high.data.uniform_(-std_att, std_att)
        self.att_vec_low.data.uniform_(-std_att, std_att)
        self.att_vec_mlp.data.uniform_(-std_att, std_att)
        self.att_struc_low.data.uniform_(-std_att, std_att)
        self.att_vec.data.uniform_(-std_att_vec, std_att_vec)

        self.layer_norm_low.reset_parameters()
        self.layer_norm_high.reset_parameters()
        self.layer_norm_mlp.reset_parameters()
        self.layer_norm_struc_low.reset_parameters()
        self.layer_norm_struc_high.reset_parameters()

    def attention3(
        self, output_low: torch.Tensor, output_high: torch.Tensor, output_mlp: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute three-way adaptive channel attention.

        Parameters
        ----------
        output_low
            Low-pass output.
        output_high
            High-pass output.
        output_mlp
            Identity/MLP output.

        Returns
        -------
        tuple
            Attention weights for low, high, and MLP channels.
        """
        temperature = 3
        if self.model_type in {"acmgcn+", "acmgcn++"}:
            output_low = self.layer_norm_low(output_low)
            output_high = self.layer_norm_high(output_high)
            output_mlp = self.layer_norm_mlp(output_mlp)
        logits = (
            torch.mm(
                torch.sigmoid(
                    torch.cat(
                        [
                            torch.mm(output_low, self.att_vec_low),
                            torch.mm(output_high, self.att_vec_high),
                            torch.mm(output_mlp, self.att_vec_mlp),
                        ],
                        1,
                    )
                ),
                self.att_vec,
            )
            / temperature
        )
        att = torch.softmax(logits, 1)
        return att[:, 0][:, None], att[:, 1][:, None], att[:, 2][:, None]

    def forward(
        self,
        input_features: torch.Tensor,
        adj_low: torch.Tensor,
        adj_high: torch.Tensor,
        adj_low_unnormalized: torch.Tensor,
    ) -> torch.Tensor:
        """Run adaptive graph convolution.

        Parameters
        ----------
        input_features
            Node features.
        adj_low
            Low-pass adjacency matrix.
        adj_high
            High-pass adjacency matrix.
        adj_low_unnormalized
            Unnormalized low-pass adjacency.

        Returns
        -------
        torch.Tensor
            Mixed node features.
        """
        if self.variant:
            output_low = torch.spmm(adj_low, F.relu(torch.mm(input_features, self.weight_low)))
            output_high = torch.spmm(adj_high, F.relu(torch.mm(input_features, self.weight_high)))
            output_mlp = F.relu(torch.mm(input_features, self.weight_mlp))
        else:
            output_low = F.relu(torch.spmm(adj_low, torch.mm(input_features, self.weight_low)))
            output_high = F.relu(torch.spmm(adj_high, torch.mm(input_features, self.weight_high)))
            output_mlp = F.relu(torch.mm(input_features, self.weight_mlp))

        if self.structure_info:
            output_struc_low = F.relu(torch.mm(adj_low_unnormalized, self.struc_low))
            att_low, att_high, att_mlp = self.attention3(output_low, output_high, output_mlp)
            return (
                3 * (att_low * output_low + att_high * output_high + att_mlp * output_mlp)
                + output_struc_low
            )

        att_low, att_high, att_mlp = self.attention3(output_low, output_high, output_mlp)
        return 3 * (att_low * output_low + att_high * output_high + att_mlp * output_mlp)


class ACMGCN(nn.Module):
    """Official ACM-GCN stack for heterophily graphs."""

    def __init__(
        self,
        nfeat: int,
        nhid: int,
        nclass: int,
        nnodes: int,
        dropout: float,
        model_type: str,
        structure_info: int,
        variant: bool = False,
    ) -> None:
        """Initialize ACM-GCN.

        Parameters
        ----------
        nfeat
            Input feature width.
        nhid
            Hidden width.
        nclass
            Output feature width.
        nnodes
            Number of graph nodes.
        dropout
            Dropout probability.
        model_type
            Official ACM variant.
        structure_info
            Whether to use structural channels.
        variant
            Whether to use variant activation ordering.
        """
        super().__init__()
        self.gcns = nn.ModuleList()
        self.model_type = model_type
        self.structure_info = structure_info
        self.gcns.append(
            GraphConvolution(
                nfeat, nhid, nnodes, model_type, variant=variant, structure_info=structure_info
            )
        )
        self.gcns.append(
            GraphConvolution(
                nhid,
                nclass,
                nnodes,
                model_type=model_type,
                output_layer=1,
                variant=variant,
                structure_info=structure_info,
            )
        )
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        adj_low: torch.Tensor,
        adj_high: torch.Tensor,
        adj_low_unnormalized: torch.Tensor,
    ) -> torch.Tensor:
        """Run ACM-GCN.

        Parameters
        ----------
        x
            Node feature matrix.
        adj_low
            Low-pass adjacency matrix.
        adj_high
            High-pass adjacency matrix.
        adj_low_unnormalized
            Unnormalized low-pass adjacency.

        Returns
        -------
        torch.Tensor
            Node logits.
        """
        x = F.dropout(x, self.dropout, training=self.training)
        fea1 = self.gcns[0](x, adj_low, adj_high, adj_low_unnormalized)
        fea1 = F.dropout(F.relu(fea1), self.dropout, training=self.training)
        return self.gcns[1](fea1, adj_low, adj_high, adj_low_unnormalized)


class ACMGCNTraceWrapper(nn.Module):
    """Single-argument wrapper around the real ACM-GCN multi-input forward."""

    def __init__(self) -> None:
        """Initialize the wrapped ACM-GCN model."""
        super().__init__()
        self.model = ACMGCN(
            nfeat=5,
            nhid=4,
            nclass=3,
            nnodes=6,
            dropout=0.0,
            model_type="acmgcn",
            structure_info=0,
        )

    def forward(
        self, inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Run ACM-GCN from a tuple input.

        Parameters
        ----------
        inputs
            ``(features, adj_low, adj_high, adj_low_unnormalized)``.

        Returns
        -------
        torch.Tensor
            Node logits.
        """
        x, adj_low, adj_high, adj_low_unnormalized = inputs
        return self.model(x, adj_low, adj_high, adj_low_unnormalized)


class Cheby1KANLayer(nn.Module):
    """Chebyshev Type-I KAN layer from AC-PKAN."""

    def __init__(self, input_dim: int, output_dim: int, degree: int) -> None:
        """Initialize the Chebyshev KAN layer.

        Parameters
        ----------
        input_dim
            Input width.
        output_dim
            Output width.
        degree
            Chebyshev polynomial degree.
        """
        super().__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate Chebyshev basis functions and combine coefficients.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        torch.Tensor
            Layer output.
        """
        x = torch.tanh(x)
        x = x.view((-1, self.inputdim, 1)).expand(-1, -1, self.degree + 1)
        x = x.clamp(-1 + 1e-7, 1 - 1e-7)
        x = x.acos()
        x *= self.arange
        x = x.cos()
        y = torch.einsum("bid,iod->bo", x, self.cheby_coeffs)
        return y.view(-1, self.outdim)


class WaveAct(nn.Module):
    """Learned sine/cosine activation from AC-PKAN."""

    def __init__(self) -> None:
        """Initialize wave activation weights."""
        super().__init__()
        self.w1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the learned wave activation.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        torch.Tensor
            Activated tensor.
        """
        return self.w1 * torch.sin(x) + self.w2 * torch.cos(x)


class McKAN1(nn.Module):
    """Attention-enhanced Chebyshev KAN core from AC-PKAN."""

    def __init__(
        self, input_dim: int, model_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ) -> None:
        """Initialize AC-PKAN's mcKAN1 module.

        Parameters
        ----------
        input_dim
            Concatenated ``x``/``t`` input width.
        model_dim
            Embedding width.
        hidden_dim
            Hidden KAN width.
        output_dim
            Output width.
        num_layers
            Number of hidden Chebyshev KAN layers.
        """
        super().__init__()
        self.num_layers = num_layers
        self.linear_emb = nn.Linear(input_dim, model_dim)

        self.theta_U = nn.Parameter(torch.randn(model_dim, hidden_dim))
        self.theta_V = nn.Parameter(torch.randn(model_dim, hidden_dim))
        self.b_U = nn.Parameter(torch.randn(hidden_dim))
        self.b_V = nn.Parameter(torch.randn(hidden_dim))

        self.hidden_ChebyKANLayers = nn.ModuleList(
            [Cheby1KANLayer(hidden_dim, hidden_dim, 8) for _ in range(num_layers)]
        )
        self.hidden_LNLayers = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        self.activation_U = WaveAct()
        self.activation_V = WaveAct()

    def encode_inputs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode inputs into the two AC-PKAN attention streams.

        Parameters
        ----------
        x
            Embedded input.

        Returns
        -------
        tuple
            U and V streams.
        """
        u_stream = self.activation_U(x @ self.theta_U + self.b_U)
        v_stream = self.activation_V(x @ self.theta_V + self.b_V)
        return u_stream, v_stream

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Run AC-PKAN on state and time inputs.

        Parameters
        ----------
        x
            State/features tensor.
        t
            Time or conditioning tensor.

        Returns
        -------
        torch.Tensor
            Network output.
        """
        src = torch.cat((x, t), dim=-1)
        src = self.linear_emb(src)
        u_stream, v_stream = self.encode_inputs(src)
        alpha_l = u_stream.clone()

        for layer_index in range(self.num_layers):
            alpha_l0 = self.hidden_ChebyKANLayers[layer_index](alpha_l)
            alpha_l0 = self.hidden_LNLayers[layer_index](alpha_l0)
            alpha_l = alpha_l0 + alpha_l
            alpha_l0 = (1 - alpha_l) * u_stream + alpha_l * v_stream
            alpha_l = alpha_l0 + alpha_l

        return self.output_layer(alpha_l)


class ACPKANTraceWrapper(nn.Module):
    """Single-argument wrapper around AC-PKAN's two-input forward."""

    def __init__(self) -> None:
        """Initialize the wrapped AC-PKAN model."""
        super().__init__()
        self.model = McKAN1(input_dim=3, model_dim=8, hidden_dim=8, output_dim=2, num_layers=2)

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Run AC-PKAN from a tuple input.

        Parameters
        ----------
        inputs
            State and time tensors.

        Returns
        -------
        torch.Tensor
            Network output.
        """
        x, t = inputs
        return self.model(x, t)


def build_3dgazenet() -> nn.Module:
    """Build a trace-sized 3DGazeNet model.

    Returns
    -------
    nn.Module
        3DGazeNet model.
    """
    model = GazeNet("resnet18")
    model.eval()
    return model


def example_input_3dgazenet() -> torch.Tensor:
    """Create an example 3DGazeNet input.

    Returns
    -------
    torch.Tensor
        Example image tensor.
    """
    return torch.randn(1, 3, 64, 64)


def build_a2cl_pt() -> nn.Module:
    """Build a trace-sized A2CL-PT model.

    Returns
    -------
    nn.Module
        A2CL-PT model.
    """
    model = A2CLPTModel(num_class=3, s=8, omega=0.5)
    model.eval()
    return model


def example_input_a2cl_pt() -> torch.Tensor:
    """Create an example A2CL-PT input.

    Returns
    -------
    torch.Tensor
        Example two-stream video feature tensor.
    """
    return torch.randn(1, 8, 2048)


def build_acm_net() -> nn.Module:
    """Build a trace-sized ACM-Net model.

    Returns
    -------
    nn.Module
        ACM-Net model.
    """
    args = SimpleNamespace(
        dataset="THUMOS",
        feature_dim=16,
        action_cls_num=3,
        dropout=0.0,
        ins_topk_seg=2,
        con_topk_seg=2,
        bak_topk_seg=2,
    )
    model = ACMNet(args)
    model.eval()
    return model


def example_input_acm_net() -> torch.Tensor:
    """Create an example ACM-Net input.

    Returns
    -------
    torch.Tensor
        Example video feature tensor.
    """
    return torch.randn(1, 6, 16)


def build_acm_gcn() -> nn.Module:
    """Build a trace-sized ACM-GCN model.

    Returns
    -------
    nn.Module
        ACM-GCN wrapper.
    """
    model = ACMGCNTraceWrapper()
    model.eval()
    return model


def example_input_acm_gcn() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create an example ACM-GCN input tuple.

    Returns
    -------
    tuple
        Node features and adjacency matrices.
    """
    adj = torch.eye(6)
    adj_high = torch.ones(6, 6) / 6
    return torch.randn(6, 5), adj, adj_high, adj


def build_ac_pkan() -> nn.Module:
    """Build a trace-sized AC-PKAN model.

    Returns
    -------
    nn.Module
        AC-PKAN wrapper.
    """
    model = ACPKANTraceWrapper()
    model.eval()
    return model


def example_input_ac_pkan() -> tuple[torch.Tensor, torch.Tensor]:
    """Create an example AC-PKAN input tuple.

    Returns
    -------
    tuple
        Example state and time tensors.
    """
    return torch.randn(4, 2), torch.randn(4, 1)


MENAGERIE_ENTRIES = [
    ("3DGazeNet", build_3dgazenet, example_input_3dgazenet, 2024, "3dgazenet"),
    ("A2CL-PT", build_a2cl_pt, example_input_a2cl_pt, 2021, "a2cl_pt"),
    ("ACM-Net", build_acm_net, example_input_acm_net, 2021, "acm_net"),
    ("ACM-GCN", build_acm_gcn, example_input_acm_gcn, 2022, "acm_gcn"),
    ("AC-PKAN", build_ac_pkan, example_input_ac_pkan, 2026, "ac_pkan"),
]
