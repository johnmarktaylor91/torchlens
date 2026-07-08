# SOURCE: vendored from https://github.com/liuruijin17/CLGo @ 406e22848325602b0e86b2766d8dc6d1c53247aa
"""CLGo: 3D lane detection with camera-pose (extrinsics) prediction (Liu et al., AAAI 2022).

Vendored classes: `kp` (the real two-stage Pv (perspective-view) / Tv (top-view, via an
inverse-perspective-mapping / IPM grid_sample) transformer detector -- both stages'
sub-modules are constructed in `kp.__init__` exactly as in the real repo),
`ProjectiveGridGenerator`, `Transformer`/`TransformerEncoder(Layer)`/
`TransformerDecoder(Layer)` (a DETR-style encoder-decoder), and `PositionEmbeddingSine`.
Code is copied verbatim from models/py_utils/kp_pt_gtr.py, models/py_utils/transformer.py,
and models/py_utils/position_encoding.py, with only minimal glue changes:
  - The real code reads its architecture hyperparameters from a process-wide
    `config.system_configs` singleton (populated from a training-run YAML/JSON) and takes
    a `db` dataset-stats object (camera intrinsics, IPM top-view-region bounds, image crop
    geometry) built from the target dataset's on-disk label cache. Both are pure data
    containers, not architecture. `build_clgo()` constructs a `SimpleNamespace` `db`
    populated with the repo's own `db/apollosim_j.py` defaults (org_h=1080, org_w=1920,
    K, top_view_region, pitch=3, cam_height=1.55) and passes the repo's
    `config/IMG_Seq_Pv-Tv_standard.json` "standard" hyperparameters directly as
    constructor kwargs, instead of round-tripping them through the singleton/JSON loader.
  - `kp.__init__` unconditionally calls `.cuda()` on several precomputed homography
    buffers (`M_inv`, `S_im`, `K`, ...). Those `.cuda()` calls are stripped so the module
    runs on CPU; the tensors themselves and every computation that produces them are
    untouched.
  - `homography_im2ipm_norm`/`homography_ipmnorm2g`/`homography_crop_resize` (from
    models/py_utils/tools.py) are copied verbatim (only the 3 functions the model actually
    calls, out of that file's ~1200-line dataset/viz toolkit).
  - `kp._test`/`.forward` are copied verbatim (module dispatch only). `_sequential_pv_stage`
    is the released `_sequential` method's Pv-stage prefix (backbone -> DETR encoder/decoder
    -> 3D-lane regression heads), copied verbatim through its `out` dict, STOPPING before
    `_sequential`'s Tv-stage suffix. That suffix is not shape-consistent as shipped, for any
    batch size, independent of vendoring (verified against a fresh checkout of the pinned
    commit; see the inline NOTE at the truncation point for the exact 3-part shape trace:
    `Transformer.forward` returns a second DECODER stack under the misleading name
    `enc_mem`, `F.grid_sample(enc_mem, grid)` then puts `batch` where `hidden_dim` is
    expected, and the `update_projection(...)` call that would normally precede it is
    separately dead on a shape-(1, 1) tensor). The sibling `_parallel` method (used by
    `test_mode in {'Pv', 'Tv'}`, not the repo's own CLI default) has an independent,
    unrelated break: `rearrange('b c h w -> (h w) b c', p)` with a malformed/reversed
    einops call signature feeding a non-batch-aligned tensor into `self.tv_transformer`.
    Both breaks reproduce in a fresh checkout; this menagerie entry traces the
    well-formed, actually-exercised Pv-stage common to both code paths.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

MENAGERIE_ZOO = "vendored-pytorch"

BN_MOMENTUM = 0.1


# ---------------------------------------------------------------------------
# models/py_utils/tools.py (verbatim excerpt: the 3 homography helpers `kp` calls)
# ---------------------------------------------------------------------------


def homograpthy_g2im(cam_pitch, cam_height, K):
    R_g2c = np.array(
        [
            [1, 0, 0],
            [0, np.cos(np.pi / 2 + cam_pitch), -np.sin(np.pi / 2 + cam_pitch)],
            [0, np.sin(np.pi / 2 + cam_pitch), np.cos(np.pi / 2 + cam_pitch)],
        ]
    )
    H_g2im = np.matmul(K, np.concatenate([R_g2c[:, 0:2], [[0], [cam_height], [0]]], 1))
    return H_g2im


def homographic_transformation(Matrix, x, y):
    ones = np.ones((1, len(y)))
    coordinates = np.vstack((x, y, ones))
    trans = np.matmul(Matrix, coordinates)

    x_vals = trans[0, :] / trans[2, :]
    y_vals = trans[1, :] / trans[2, :]
    return x_vals, y_vals


def homography_crop_resize(org_img_size, crop_y, resize_img_size):
    ratio_x = resize_img_size[1] / org_img_size[1]
    ratio_y = resize_img_size[0] / (org_img_size[0] - crop_y)
    H_c = np.array([[ratio_x, 0, 0], [0, ratio_y, -ratio_y * crop_y], [0, 0, 1]])
    return H_c


def homography_im2ipm_norm(
    top_view_region, org_img_size, crop_y, resize_img_size, cam_pitch, cam_height, K
):
    H_g2im = homograpthy_g2im(cam_pitch, cam_height, K)
    H_c = homography_crop_resize(org_img_size, crop_y, resize_img_size)
    H_g2im = np.matmul(H_c, H_g2im)

    x_2d, y_2d = homographic_transformation(H_g2im, top_view_region[:, 0], top_view_region[:, 1])
    border_im = np.concatenate([x_2d.reshape(-1, 1), y_2d.reshape(-1, 1)], axis=1)

    border_im[:, 0] = border_im[:, 0] / resize_img_size[1]
    border_im[:, 1] = border_im[:, 1] / resize_img_size[0]
    border_im = np.float32(border_im)
    dst = np.float32([[0, 0], [1, 0], [0, 1], [1, 1]])
    import cv2

    H_im2ipm_norm = cv2.getPerspectiveTransform(border_im, dst)
    H_ipm2im_norm = cv2.getPerspectiveTransform(dst, border_im)
    return H_im2ipm_norm, H_ipm2im_norm


def homography_ipmnorm2g(top_view_region):
    import cv2

    src = np.float32([[0, 0], [1, 0], [0, 1], [1, 1]])
    H_ipmnorm2g = cv2.getPerspectiveTransform(src, np.float32(top_view_region))
    return H_ipmnorm2g


# ---------------------------------------------------------------------------
# models/py_utils/position_encoding.py (verbatim)
# ---------------------------------------------------------------------------


class PositionEmbeddingSine(nn.Module):
    """A more standard version of the position embedding, generalized to images."""

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * np.pi
        self.scale = scale

    def forward(self, x, mask):
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


def build_position_encoding(hidden_dim, type):
    N_steps = hidden_dim // 2
    if type in ("v2", "sine"):
        return PositionEmbeddingSine(N_steps, normalize=True)
    raise ValueError(f"not supported {type}")


# ---------------------------------------------------------------------------
# models/py_utils/transformer.py (verbatim: DETR-style encoder/decoder)
# ---------------------------------------------------------------------------


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, pos=None):
        output = src
        for layer in self.layers:
            output = layer(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        output = tgt
        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
    ):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )
        self.decoder_ = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)
        tgt = torch.zeros_like(query_embed)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(
            tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed
        )
        hs_ = self.decoder_(
            tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed
        )

        return hs.transpose(1, 2), hs_.transpose(1, 2)


def _get_clones(module, N):
    import copy

    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def build_transformer(
    hidden_dim,
    dropout,
    nheads,
    dim_feedforward,
    enc_layers,
    dec_layers,
    pre_norm=False,
    return_intermediate_dec=False,
):
    return Transformer(
        d_model=hidden_dim,
        dropout=dropout,
        nhead=nheads,
        dim_feedforward=dim_feedforward,
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
        normalize_before=pre_norm,
        return_intermediate_dec=return_intermediate_dec,
    )


def build_transformer_decoder(
    hidden_dim,
    dropout,
    n_heads,
    dim_feedforward,
    dec_layers,
    pre_norm=False,
    return_intermediate=False,
):
    return TransformerDecoder(
        TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            normalize_before=pre_norm,
        ),
        dec_layers,
        nn.LayerNorm(hidden_dim),
        return_intermediate=return_intermediate,
    )


# ---------------------------------------------------------------------------
# models/py_utils/kp_pt_gtr.py (verbatim `kp` model + `ProjectiveGridGenerator`)
# ---------------------------------------------------------------------------


class FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rsqrt.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        kernel_size=None,
        padding=None,
        attn_groups=None,
        embed_shape=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ProjectiveGridGenerator(nn.Module):
    def __init__(self, size_ipm, M, no_cuda=True):
        super().__init__()
        self.N, self.H, self.W = size_ipm
        linear_points_W = torch.linspace(0, 1 - 1 / self.W, self.W)
        linear_points_H = torch.linspace(0, 1 - 1 / self.H, self.H)
        self.base_grid = M.new(self.N, self.H, self.W, 3)
        self.base_grid[:, :, :, 0] = torch.ger(torch.ones(self.H), linear_points_W).expand_as(
            self.base_grid[:, :, :, 0]
        )
        self.base_grid[:, :, :, 1] = torch.ger(linear_points_H, torch.ones(self.W)).expand_as(
            self.base_grid[:, :, :, 1]
        )
        self.base_grid[:, :, :, 2] = 1
        # no_cuda=True keeps this on CPU (the real repo unconditionally moved it to
        # .cuda(); see module header).

    def forward(self, M):
        grid = torch.bmm(self.base_grid.view(self.N, self.H * self.W, 3), M.transpose(1, 2))
        grid = torch.div(grid[:, :, 0:2], grid[:, :, 2:]).reshape((self.N, self.H, self.W, 2))
        grid = (grid - 0.5) * 2
        return grid


class kp(nn.Module):
    def __init__(
        self,
        flag=False,
        test_mode=None,
        train_mode=None,
        freeze=False,
        db=None,
        block=None,
        layers=None,
        res_dims=None,
        res_strides=None,
        attn_dim=None,
        num_queries=None,
        aux_loss=None,
        pos_type=None,
        drop_out=0.1,
        num_heads=None,
        dim_feedforward=None,
        enc_layers=None,
        dec_layers=None,
        pre_norm=None,
        return_intermediate=None,
        kps_dim=None,
        mlp_layers=None,
        num_cls=None,
        norm_layer=FrozenBatchNorm2d,
    ):
        super(kp, self).__init__()
        self.flag = flag
        self.test_mode = test_mode
        self.train_mode = train_mode
        self.db = db
        self.norm_layer = norm_layer
        hidden_dim = attn_dim
        self.aux_loss = aux_loss
        self.inplanes = res_dims[0]
        # Pv-stage
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self.norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block[0], res_dims[0], layers[0], stride=res_strides[0])
        self.layer2 = self._make_layer(block[1], res_dims[1], layers[1], stride=res_strides[1])
        self.layer3 = self._make_layer(block[2], res_dims[2], layers[2], stride=res_strides[2])
        self.layer4 = self._make_layer(block[3], res_dims[3], layers[3], stride=res_strides[3])
        self.position_embedding = build_position_encoding(hidden_dim=hidden_dim, type=pos_type)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(res_dims[-1], hidden_dim, kernel_size=1)
        self.transformer = build_transformer(
            hidden_dim=hidden_dim,
            dropout=drop_out,
            nheads=num_heads,
            dim_feedforward=dim_feedforward,
            enc_layers=enc_layers,
            dec_layers=dec_layers,
            pre_norm=pre_norm,
            return_intermediate_dec=return_intermediate,
        )
        self.class_embed = nn.Linear(hidden_dim, num_cls + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, kps_dim - 2, mlp_layers)
        self.height_embed = nn.Linear(hidden_dim, 1)
        self.pitch_embed = nn.Linear(hidden_dim, 1)

        # Tv-stage
        self.inplanes = res_dims[0]
        self.tv_position_embedding = build_position_encoding(hidden_dim=hidden_dim, type=pos_type)
        self.tv_query_embed = nn.Embedding(num_queries, hidden_dim)
        self.tv_input_proj = nn.Conv2d(res_dims[-1], hidden_dim, kernel_size=1)
        self.tv_transformer = build_transformer_decoder(
            hidden_dim=hidden_dim,
            dropout=drop_out,
            n_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dec_layers=dec_layers,
            pre_norm=pre_norm,
            return_intermediate=return_intermediate,
        )
        self.tv_class_embed = nn.Linear(hidden_dim, num_cls + 1)
        self.tv_bbox_embed = MLP(hidden_dim, hidden_dim, kps_dim - 2, mlp_layers)
        self.tv_height_embed = nn.Linear(hidden_dim, 1)
        self.tv_pitch_embed = nn.Linear(hidden_dim, 1)

        # IPM
        org_img_size = np.array([self.db.org_h, self.db.org_w])
        resize_img_size = np.array([self.db.resize_h, self.db.resize_w])
        cam_pitch = np.pi / 180 * self.db.pitch

        self.cam_height = (
            torch.tensor(self.db.cam_height)
            .unsqueeze_(0)
            .expand([self.db.batch_size, 1])
            .type(torch.FloatTensor)
        )
        self.cam_pitch = (
            torch.tensor(cam_pitch)
            .unsqueeze_(0)
            .expand([self.db.batch_size, 1])
            .type(torch.FloatTensor)
        )
        self.cam_height_default = (
            torch.tensor(self.db.cam_height)
            .unsqueeze_(0)
            .expand(self.db.batch_size)
            .type(torch.FloatTensor)
        )
        self.cam_pitch_default = (
            torch.tensor(cam_pitch).unsqueeze_(0).expand(self.db.batch_size).type(torch.FloatTensor)
        )

        self.S_im = torch.from_numpy(
            np.array(
                [[self.db.resize_w, 0, 0], [0, self.db.resize_h, 0], [0, 0, 1]], dtype=np.float32
            )
        )
        self.S_im_inv = torch.from_numpy(
            np.array(
                [
                    [1 / float(self.db.resize_w), 0, 0],
                    [0, 1 / float(self.db.resize_h), 0],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
        )
        self.S_im_inv_batch = (
            self.S_im_inv.unsqueeze_(0).expand([self.db.batch_size, 3, 3]).type(torch.FloatTensor)
        )

        H_c = homography_crop_resize(org_img_size, self.db.crop_y, resize_img_size)
        self.H_c = (
            torch.from_numpy(H_c)
            .unsqueeze_(0)
            .expand([self.db.batch_size, 3, 3])
            .type(torch.FloatTensor)
        )

        self.K = (
            torch.from_numpy(self.db.K)
            .unsqueeze_(0)
            .expand([self.db.batch_size, 3, 3])
            .type(torch.FloatTensor)
        )

        H_g2cam = np.array(
            [[1, 0, 0], [0, np.sin(-cam_pitch), self.db.cam_height], [0, np.cos(-cam_pitch), 0]]
        )
        self.H_g2cam = (
            torch.from_numpy(H_g2cam)
            .unsqueeze_(0)
            .expand([self.db.batch_size, 3, 3])
            .type(torch.FloatTensor)
        )

        H_ipmnorm2g = homography_ipmnorm2g(self.db.top_view_region)
        self.H_ipmnorm2g = (
            torch.from_numpy(H_ipmnorm2g)
            .unsqueeze_(0)
            .expand([self.db.batch_size, 3, 3])
            .type(torch.FloatTensor)
        )

        M_ipm2im = torch.bmm(self.H_g2cam, self.H_ipmnorm2g)
        M_ipm2im = torch.bmm(self.K, M_ipm2im)
        M_ipm2im = torch.bmm(self.H_c, M_ipm2im)
        M_ipm2im = torch.bmm(self.S_im_inv_batch, M_ipm2im)
        M_ipm2im = torch.div(
            M_ipm2im,
            M_ipm2im[:, 2, 2]
            .reshape([self.db.batch_size, 1, 1])
            .expand([self.db.batch_size, 3, 3]),
        )
        self.M_inv = M_ipm2im
        # NOTE: the real repo unconditionally moves M_inv/S_im/... to .cuda() here.
        # Left on CPU so the module runs in a CPU-only tracing environment; every
        # tensor value and computation above is untouched.

        size_top = torch.Size([self.db.batch_size, int(7), int(4)])
        self.project_layer = ProjectiveGridGenerator(size_top, self.M_inv)

    def _sequential_pv_stage(self, *xs, **kwargs):
        """The real `_sequential` method's Pv-stage prefix (models/py_utils/kp_pt_gtr.py),
        verbatim through the Pv-stage `out` dict. See the NOTE below for exactly where
        and why this stops short of the full method (the Tv-stage suffix is
        shape-broken as released, independent of vendoring)."""
        images = xs[0]
        masks = xs[1]

        p = self.conv1(images)
        p = self.bn1(p)
        p = self.relu(p)
        p = self.maxpool(p)
        p = self.layer1(p)
        p = self.layer2(p)
        p = self.layer3(p)
        p = self.layer4(p)
        pmasks = F.interpolate(masks[:, 0, :, :][None], size=p.shape[-2:]).to(torch.bool)[0]
        pos = self.position_embedding(p, pmasks)
        hs, enc_mem = self.transformer(self.input_proj(p), pmasks, self.query_embed.weight, pos)
        output_class = self.class_embed(hs)
        output_coord = self.bbox_embed(hs)
        latent_height = self.height_embed(hs).sigmoid() + 1.0
        latent_height = torch.mean(latent_height, dim=-2, keepdim=True)
        latent_height = torch.mean(latent_height, dim=0, keepdim=True)
        latent_pitch = self.pitch_embed(hs)
        latent_pitch = torch.mean(latent_pitch, dim=-2, keepdim=True)
        latent_pitch = torch.mean(latent_pitch, dim=0, keepdim=True)
        output_coord = torch.cat(
            [
                output_coord,
                latent_height.repeat(output_coord.shape[0], 1, output_coord.shape[2], 1),
                latent_pitch.repeat(output_coord.shape[0], 1, output_coord.shape[2], 1),
            ],
            dim=-1,
        )
        out = {"pred_logits": output_class[-1], "pred_boxes": output_coord[-1]}
        # NOTE (real-repo bug, not introduced by vendoring; stops the Pv-stage/Tv-stage
        # boundary here): the released `_sequential` continues into a second,
        # IPM-warped "Tv-stage" pass (`self.update_projection(...)` ->
        # `self.project_layer(self.M_inv)` -> `F.grid_sample(enc_mem, grid)` ->
        # `self.tv_transformer(...)`), but that path is not shape-consistent as shipped,
        # independent of batch size:
        #   1. `Transformer.forward` (models/py_utils/transformer.py) returns
        #      `hs.transpose(1, 2), hs_.transpose(1, 2)` -- both are DECODER stacks, so
        #      `enc_mem` (aliasing the second one) has shape
        #      (dec_layers, batch, num_queries, hidden_dim), not the actual encoder
        #      memory. The file even has the presumably-intended line commented out
        #      immediately above: `# return hs.transpose(1, 2),
        #      memory.permute(1, 2, 0).view(bs, c, h, w)`.
        #   2. Feeding that into `F.grid_sample(enc_mem, grid)` (which requires
        #      (N, C, H, W)) makes grid_sample's output channel dim equal `batch`, not
        #      `hidden_dim`; the following `tv_position_embedding`/`tensor + pos`
        #      addition into the Tv-stage transformer then fails shape-checking for
        #      every batch_size (verified: batch_size=1 fails at grid_sample's N
        #      mismatch against dec_layers=2; batch_size=2 passes grid_sample but then
        #      fails the pos-embedding add with mismatched channel counts).
        #   3. Separately, the commented-out `self.update_projection(...)` re-projection
        #      call that would normally precede this is itself dead: its inputs
        #      (`latent_height`/`latent_pitch` after the query- and decoder-layer
        #      `torch.mean` reductions) are shape (1, 1) regardless of batch size, and
        #      `update_projection`'s per-element `.data.cpu().numpy()` call requires a
        #      scalar.
        # This is not something introduced by vendoring -- it reproduces verbatim in a
        # fresh checkout of the pinned commit. The Pv-stage above (backbone -> DETR
        # encoder/decoder -> 3D-lane regression heads, `out` above) is the complete,
        # well-formed, and genuinely exercised half of the architecture (it is also
        # exactly the sub-network reused by the `Tv`/`Pv`-only `_parallel` code path),
        # so it is what this menagerie entry traces.
        return out

    def _test(self, *xs, **kwargs):
        if self.test_mode == "PvTv":
            return self._sequential_pv_stage(*xs, **kwargs)
        else:
            raise ValueError("Not supported test_mode: {}".format(self.test_mode))

    def forward(self, *xs, **kwargs):
        if self.flag:
            raise NotImplementedError("training mode not exercised by this menagerie entry")
        return self._test(*xs, **kwargs)

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride=1,
        kernel_size=None,
        padding=None,
        attn_groups=None,
        embed_shape=None,
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                kernel_size=kernel_size,
                padding=padding,
                attn_groups=attn_groups,
                embed_shape=embed_shape,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    kernel_size=kernel_size,
                    padding=padding,
                    attn_groups=attn_groups,
                    embed_shape=embed_shape,
                )
            )
        return nn.Sequential(*layers)

    def update_projection(self, cam_height, cam_pitch):
        for i in range(cam_height.shape[0]):
            _M, M_inv = homography_im2ipm_norm(
                self.db.top_view_region,
                np.array([self.db.org_h, self.db.org_w]),
                self.db.crop_y,
                np.array([self.db.resize_h, self.db.resize_w]),
                cam_pitch[i].data.cpu().numpy(),
                cam_height[i].data.cpu().numpy(),
                self.db.K,
            )
            self.M_inv[i] = torch.from_numpy(M_inv).type(torch.FloatTensor)
        self.cam_height = cam_height
        self.cam_pitch = cam_pitch


def _rearrange_bchw_to_hwbc(p):
    """`einops.rearrange(p, 'b c h w -> (h w) b c')`, spelled out with a plain
    permute/reshape (equivalent for this fixed pattern) to avoid adding a whole-package
    `einops` import for a single call site; einops itself is an already-installed
    base-env dependency used elsewhere in the real repo."""
    b, c, h, w = p.shape
    return p.permute(2, 3, 0, 1).reshape(h * w, b, c)


# ---------------------------------------------------------------------------
# Menagerie build/example harness
# ---------------------------------------------------------------------------


def _make_db():
    """SimpleNamespace stand-in for the repo's `db/apollosim_j.py` dataset object,
    populated with that file's own default field values (org_h/org_w/K/top_view_region/
    pitch/cam_height) plus the "standard" training config's batch_size/input_size
    (config/IMG_Seq_Pv-Tv_standard.json)."""
    resize_h, resize_w = 360, 480
    return SimpleNamespace(
        org_h=1080,
        org_w=1920,
        crop_y=0,
        resize_h=resize_h,
        resize_w=resize_w,
        top_view_region=np.array([[-10, 103], [10, 103], [-10, 3], [10, 3]], dtype=np.float32),
        K=np.array([[2015.0, 0.0, 960.0], [0.0, 2015.0, 540.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        pitch=3,
        cam_height=1.55,
        batch_size=1,
    )


def build_clgo():
    db = _make_db()
    model = kp(
        flag=False,
        test_mode="PvTv",  # the repo's own CLI default (joint_test.py --test_mode)
        train_mode="sequential",
        freeze=False,
        db=db,
        block=[BasicBlock, BasicBlock, BasicBlock, BasicBlock],
        layers=[1, 2, 2, 2],
        res_dims=[16, 32, 64, 128],
        res_strides=[1, 2, 2, 2],
        attn_dim=32,
        num_queries=7,
        aux_loss=True,
        pos_type="sine",
        drop_out=0.1,
        num_heads=2,
        dim_feedforward=128,
        enc_layers=2,
        dec_layers=2,
        pre_norm=False,
        return_intermediate=True,
        kps_dim=14,
        mlp_layers=3,
        num_cls=2,
        norm_layer=FrozenBatchNorm2d,
    )
    model.eval()
    return model


def example_input_clgo():
    db = _make_db()
    images = torch.randn(db.batch_size, 3, db.resize_h, db.resize_w)
    # `masks` is cast to bool via `.to(torch.bool)` inside `_sequential` after
    # `F.interpolate`; interpolate itself requires a float/int (non-bool) input, so
    # the caller-supplied mask tensor here is float32, matching the real data
    # pipeline's padding-mask convention (0/1 float image masks).
    masks = torch.zeros(db.batch_size, 1, db.resize_h, db.resize_w, dtype=torch.float32)
    # `_sequential` only reads xs[0]/xs[1] (images/masks); heights/pitches are kept
    # here to match the real repo's demo/test call signature (`kp(*xs)` is invoked
    # with all 4 tensors from the dataloader batch), even though this test_mode
    # branch ignores the trailing two.
    heights = torch.full((db.batch_size,), db.cam_height, dtype=torch.float32)
    pitches = torch.full((db.batch_size,), np.pi / 180 * db.pitch, dtype=torch.float32)
    return (images, masks, heights, pitches)


MENAGERIE_ENTRIES = [
    ("CLGo", build_clgo, example_input_clgo, 2022, MENAGERIE_ZOO),
]
