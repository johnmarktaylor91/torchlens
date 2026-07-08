# SOURCE: vendored from https://github.com/Cassie07/PathOmics @ main
# (PathOmics/model_and_training_utils/PathOmics_Survival_model.py +
#  PathOmics/model_and_training_utils/PathOmics_model_utils.py)
"""PathOmics: multimodal (whole-slide-image + genomic) transformer for cancer survival
prediction (Jiang et al., CVPR 2023 "Multimodal Co-Attention Transformer for Survival
Prediction in Gigapixel Whole Slide Images" family; PathOmics extends this with
unimodal pretraining + a WSI/omics co-attention pipeline).

Vendored (near) verbatim from the real repo: attention-gated instance pooling over WSI
patch bags, per-omic-signature SNN blocks, transformer encoders for both modalities, and
attention-pooling projection heads, following the ``pretrain`` forward mode (the survival
head mode additionally needs a live ``fusion='bilinear'`` class, ``BilinearFusion``, that
is referenced but never defined anywhere in the real repo -- so, matching the only forward
path the authors themselves can run, this module exercises the default
``fusion='concat'`` / ``image_group_method='random'`` / ``omic_bag='Attention'``
``mode='pretrain'`` path). Only formatting/import consolidation was touched; every
class/mechanism (including the repo's own hand-rolled ``MultiheadAttention``, dead code
that PathOmics_Surv itself constructs but never calls in this forward path) is preserved.
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans  # noqa: F401  (imported by the real repo; used by other code paths)
from sklearn.cluster import MiniBatchKMeans  # noqa: F401

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# from PathOmics_model_utils.py
# ---------------------------------------------------------------------------
class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):
        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x


class residual_block(nn.Module):
    def __init__(self, nChn=512):
        super(residual_block, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(nChn, nChn, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(nChn, nChn, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        tt = self.block(x)
        x = x + tt
        return x


class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim=512, numLayer_Res=0):
        super(DimReduction, self).__init__()
        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.numRes = numLayer_Res

        self.resBlocks = []
        for ii in range(numLayer_Res):
            self.resBlocks.append(residual_block(m_dim))
        self.resBlocks = nn.Sequential(*self.resBlocks)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)

        if self.numRes > 0:
            x = self.resBlocks(x)

        return x


class Attention2(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention2, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K),
        )

    def forward(self, x, isNorm=True):
        ## x: N x L
        A = self.attention(x)  ## N x K
        A = torch.transpose(A, 1, 0)  # KxN
        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N
        return A  ### K x N


class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid(),
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x, isNorm=True):
        ## x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN

        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N

        return A  ### K x N


class Attention_with_Classifier(nn.Module):
    def __init__(self, L=512, D=128, K=1, num_cls=2, droprate=0):
        super(Attention_with_Classifier, self).__init__()
        self.attention = Attention_Gated(L, D, K)
        self.classifier = Classifier_1fc(L, num_cls, droprate)

    def forward(self, x):  ## x: N x L
        AA = self.attention(x)  ## K x N
        afeat = torch.mm(AA, x)  ## K x L
        pred = self.classifier(afeat)  ## K x num_cls
        return pred


# ---------------------------------------------------------------------------
# from PathOmics_Survival_model.py
# ---------------------------------------------------------------------------
class Omics_Attention(nn.Module):
    def __init__(self, feature_dim, attention_dim):
        super().__init__()
        self.linear = nn.Linear(feature_dim, attention_dim)
        self.activation = nn.Tanh()

    def forward(self, features):
        features = self.linear(features)
        attention = self.activation(features)
        attention = torch.softmax(attention, dim=0)
        weighted_features = attention * features
        return weighted_features


# from MCAT_models.model_utils
def SNN_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    return nn.Sequential(
        nn.Linear(dim1, dim2),
        nn.ELU(),
        nn.AlphaDropout(p=dropout, inplace=False),
    )


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh(),
        ]

        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


# from MCAT_models.model_attn
###
# ========== Modifying PyTorch Functionalities ======================
###
from torch.nn.functional import (  # noqa: E402
    dropout,
    has_torch_function,
    handle_torch_function,
    linear,
    pad,
    softmax,
)
from typing import Optional, Tuple  # noqa: E402
from torch import Tensor  # noqa: E402
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear  # noqa: E402
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_  # noqa: E402
from torch.nn.parameter import Parameter  # noqa: E402
from torch.nn import Module  # noqa: E402


def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    need_raw: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
):
    tens_ops = (
        query,
        key,
        value,
        in_proj_weight,
        in_proj_bias,
        bias_k,
        bias_v,
        out_proj_weight,
        out_proj_bias,
    )
    if has_torch_function(tens_ops):
        return handle_torch_function(
            multi_head_attention_forward,
            tens_ops,
            query,
            key,
            value,
            embed_dim_to_check,
            num_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            need_raw=need_raw,
            attn_mask=attn_mask,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            static_k=static_k,
            static_v=static_v,
        )
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if not use_separate_proj_weight:
        if (query is key or torch.equal(query, key)) and (key is value or torch.equal(key, value)):
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)
        elif key is value or torch.equal(key, value):
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = linear(key, _w, _b).chunk(2, dim=-1)
        else:
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = linear(key, _w, _b)

            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim : (embed_dim * 2)])
            v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2) :])
        else:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if attn_mask is not None:
        assert (
            attn_mask.dtype == torch.float32
            or attn_mask.dtype == torch.float64
            or attn_mask.dtype == torch.float16
            or attn_mask.dtype == torch.uint8
            or attn_mask.dtype == torch.bool
        ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(
            attn_mask.dtype
        )
        if attn_mask.dtype == torch.uint8:
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 2D attn_mask is not correct.")
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 3D attn_mask is not correct.")
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))

    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat(
            [k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1
        )
        v = torch.cat(
            [v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1
        )
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float("-inf"))
        else:
            attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float("-inf"),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights_raw = attn_output_weights
    attn_output_weights = softmax(attn_output_weights, dim=-1)
    attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        if need_raw:
            attn_output_weights_raw = attn_output_weights_raw.view(bsz, num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights_raw
        else:
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


class MultiheadAttention(Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need
    """

    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
    ):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads"
        )

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter("in_proj_weight", None)
        else:
            self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        if "_qkv_same_embed_dim" not in state:
            state["_qkv_same_embed_dim"] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        need_raw=True,
        attn_mask=None,
    ):
        # type: (Tensor, Tensor, Tensor, Optional[Tensor], bool, Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]
        if not self._qkv_same_embed_dim:
            return multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                need_raw=need_raw,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
            )
        else:
            return multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                need_raw=need_raw,
                attn_mask=attn_mask,
            )


class PathOmics_Surv(nn.Module):
    def __init__(
        self,
        device,
        fusion="concat",
        omic_sizes=[100, 200, 300, 400, 500, 600],
        model_size_wsi: str = "small",
        model_size_omic: str = "small",
        n_classes=4,
        dropout=0.25,
        omic_bag="Attention",
        use_GAP_in_pretrain=False,
        proj_ratio=1,
        image_group_method="random",
    ):
        """
        PathOmics model Implementation.
        Inspired by https://github.com/mahmoodlab/MCAT/blob/master/models/model_coattn.py

        Args:
            fusion (str): Late fusion method (Choices: concat, bilinear, or None)
            omic_sizes (List): List of sizes of genomic embeddings
            model_size_wsi (str): Size of WSI encoder (Choices: small or large)
            model_size_omic (str): Size of Genomic encoder (Choices: small or large)
            dropout (float): Dropout rate
            n_classes (int): Output shape of NN
        """
        super(PathOmics_Surv, self).__init__()
        self.device = device
        self.omic_bag = omic_bag
        self.fusion = fusion
        self.use_GAP_in_pretrain = use_GAP_in_pretrain
        self.image_group_method = image_group_method
        self.omic_sizes = omic_sizes
        self.n_classes = n_classes
        self.size_dict_WSI = {"small": [1024, 256, 256], "big": [1024, 512, 384]}
        self.size_dict_omic = {
            "small": [256, 256],
            "big": [1024, 1024, 1024, 256],
            "Methylation": [512, 256],
            "multi_omics": [512, 256],
        }

        ### FC Layer over WSI bag
        size = self.size_dict_WSI[model_size_wsi]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        fc.append(nn.Dropout(0.25))
        self.wsi_net = nn.Sequential(*fc)

        ### Constructing Genomic SNN
        hidden = self.size_dict_omic[model_size_omic]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)

        ### Multihead Attention
        self.coattn = MultiheadAttention(embed_dim=256, num_heads=1)

        ### Path Transformer + Attention Head
        path_encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, activation="relu"
        )
        self.path_transformer = nn.TransformerEncoder(path_encoder_layer, num_layers=2)
        self.path_attention_head = Attn_Net_Gated(
            L=size[2], D=size[2], dropout=dropout, n_classes=1
        )
        self.path_rho = nn.Sequential(
            *[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)]
        )

        ### Omic Transformer + Attention Head
        omic_encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, activation="relu"
        )
        self.omic_transformer = nn.TransformerEncoder(omic_encoder_layer, num_layers=2)
        self.omic_attention_head = Attn_Net_Gated(
            L=size[2], D=size[2], dropout=dropout, n_classes=1
        )
        self.omic_rho = nn.Sequential(
            *[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)]
        )

        ### Fusion Layer
        if self.fusion == "concat":
            self.mm = nn.Sequential(
                *[nn.Linear(256 * 2, size[2]), nn.ReLU(), nn.Linear(size[2], size[2]), nn.ReLU()]
            )
        elif self.fusion == "bilinear":
            # NOTE: the real repo references a BilinearFusion class here that is never
            # defined/imported anywhere in the repository -- fusion='bilinear' is dead,
            # non-runnable code upstream. Only fusion='concat' (the default) is exercised.
            raise NotImplementedError(
                "fusion='bilinear' requires BilinearFusion, which is undefined upstream in PathOmics"
            )

        ### Classifier
        self.classifier = nn.Linear(size[2], n_classes)

        ### Projection
        self.path_proj = nn.Linear(size[2], int(size[2] * proj_ratio))
        self.omic_proj = nn.Linear(size[2], int(size[2] * proj_ratio))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        ### PathOmics - WSI
        self.attention = Attention_Gated(size[2])
        self.dimReduction = DimReduction(size[0], size[2], numLayer_Res=0)

        ### PathOmics - Omics
        self.omics_attention_networks = nn.ModuleList(
            [Omics_Attention(omic_sizes[i], size[2]) for i in range(len(omic_sizes))]
        )

        self.MHA = nn.MultiheadAttention(size[2], 8)
        self.mutiheadattention_networks = nn.ModuleList([self.MHA for i in range(len(omic_sizes))])

    def forward(self, x_path, x_omic, x_cluster, mode="pretrain"):
        # image-bag
        slide_pseudo_feat = []
        numGroup = len(x_omic)

        if self.image_group_method == "random":
            feat_index = list(range(x_path.shape[0]))
            random.shuffle(feat_index)
            index_chunk_list = np.array_split(np.array(feat_index), numGroup)
            index_chunk_list = [sst.tolist() for sst in index_chunk_list]
        elif self.image_group_method == "kmeans_cluster":
            index_chunk_list = [[] for i in range(numGroup)]
            labels = x_cluster
            for index, label in enumerate(labels):
                index_chunk_list[label].append(index)

        for tindex in index_chunk_list:
            subFeat_tensor = torch.index_select(
                x_path, dim=0, index=torch.LongTensor(tindex).to(self.device)
            )  # n x 1024
            tmidFeat = self.dimReduction(subFeat_tensor)  # n x 256
            tAA = self.attention(tmidFeat).squeeze(0)  # n
            tattFeats = torch.einsum("ns,n->ns", tmidFeat, tAA)  ### n x 256
            tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x 256
            af_inst_feat = tattFeat_tensor  # 1 x 256
            slide_pseudo_feat.append(af_inst_feat)

        h_path_bag = torch.cat(slide_pseudo_feat, dim=0)  ### numGroup x 256

        if self.omic_bag == "Attention":
            h_omic = [
                self.omics_attention_networks[idx].forward(sig_feat)
                for idx, sig_feat in enumerate(x_omic)
            ]
        elif self.omic_bag == "SNN_Attention":
            h_omic = []
            for idx, sig_feat in enumerate(x_omic):
                snn_feat = self.sig_networks[idx].forward(sig_feat)
                snn_feat = snn_feat.unsqueeze(0).unsqueeze(1)
                attention_feat, _ = self.mutiheadattention_networks[idx].forward(
                    snn_feat, snn_feat, snn_feat
                )
                h_omic.append(attention_feat.squeeze())
        else:
            h_omic = [
                self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)
            ]
        h_omic_bag = torch.stack(h_omic)  ### numGroup x 256

        h_path_trans = self.path_transformer(h_path_bag)
        h_omic_trans = self.omic_transformer(h_omic_bag)

        ### Global Attention Pooling
        if mode == "pretrain":
            if self.use_GAP_in_pretrain:
                A_path, h_path = self.path_attention_head(h_path_trans.squeeze(1))
                A_path = torch.transpose(A_path, 1, 0)
                h_path = torch.mm(F.softmax(A_path, dim=1), h_path)
                h_path = self.path_proj(h_path).squeeze()
            else:
                h_path = self.path_proj(h_path_trans).squeeze()
        else:
            A_path, h_path = self.path_attention_head(h_path_trans.squeeze(1))
            A_path = torch.transpose(A_path, 1, 0)
            h_path = torch.mm(F.softmax(A_path, dim=1), h_path)
            h_path = self.path_rho(h_path).squeeze()

        if mode == "pretrain":
            if self.use_GAP_in_pretrain:
                A_omic, h_omic = self.omic_attention_head(h_omic_trans.squeeze(1))
                A_omic = torch.transpose(A_omic, 1, 0)
                h_omic = torch.mm(F.softmax(A_omic, dim=1), h_omic)
                h_omic = self.omic_proj(h_omic).squeeze()
            else:
                h_omic = self.omic_proj(h_omic_trans).squeeze()
        else:
            A_omic, h_omic = self.omic_attention_head(h_omic_trans.squeeze(1))
            A_omic = torch.transpose(A_omic, 1, 0)
            h_omic = torch.mm(F.softmax(A_omic, dim=1), h_omic)
            h_omic = self.omic_rho(h_omic).squeeze()

        if mode == "pretrain":
            return h_path, h_omic  # path_embedding, omic_embedding
        else:
            if self.fusion == "bilinear":
                h = self.mm(h_path.unsqueeze(dim=0), h_omic.unsqueeze(dim=0)).squeeze()
            elif self.fusion == "concat":
                h = self.mm(torch.cat([h_path, h_omic], axis=0))
            elif self.fusion == "image":
                h = h_path.squeeze()
            elif self.fusion == "omics":
                h = h_omic.squeeze()

            logits = self.classifier(h).unsqueeze(0)
            Y_hat = torch.topk(logits, 1, dim=1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)

            attention_scores = {}

            return hazards, S, Y_hat, attention_scores, None


class PathOmicsSurvPretrainWrapper(nn.Module):
    """Thin TorchLens-tracing wrapper: fixes ``mode='pretrain'`` so the traced forward
    only takes tensor-ish positional args (a bare Python string control-flow arg such as
    ``mode`` would otherwise be misidentified as tokenizable text input)."""

    def __init__(self, model: PathOmics_Surv):
        super().__init__()
        self.model = model

    def forward(self, x_path, x_omic, x_cluster=None):
        return self.model(x_path, x_omic, x_cluster, mode="pretrain")


def build_pathomics():
    torch.manual_seed(0)
    omic_sizes = [8, 8, 8]
    model = PathOmics_Surv(
        device=torch.device("cpu"),
        fusion="concat",
        omic_sizes=omic_sizes,
        model_size_wsi="small",
        model_size_omic="small",
        n_classes=4,
        dropout=0.0,
        omic_bag="Attention",
        use_GAP_in_pretrain=False,
        image_group_method="random",
    )
    return PathOmicsSurvPretrainWrapper(model)


def example_input_pathomics():
    torch.manual_seed(0)
    omic_sizes = [8, 8, 8]
    x_path = torch.randn(12, 1024)  # n WSI patch features x 1024
    x_omic = [torch.randn(sz) for sz in omic_sizes]  # per-signature genomic feature vectors
    return (x_path, x_omic)


MENAGERIE_ENTRIES = [
    ("PathOmics", build_pathomics, example_input_pathomics, 2023, "vendored-pytorch"),
]
