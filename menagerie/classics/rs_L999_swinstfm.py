# SOURCE: vendored from LouisChen0104/swinstfm @ 2fd9bcf04523
# https://github.com/LouisChen0104/swinstfm
# Files combined verbatim (imports/relative-paths only adjusted to be
# self-contained in a single module): models/swin_transformer.py,
# models/fem.py, models/mfm.py, models/swinstfm.py
#
# SwinSTFM: Remote sensing spatiotemporal image fusion using Swin Transformer.
# Chen et al., IEEE TGRS 2022.
"""Vendored SwinSTFM model definition for the TorchLens menagerie."""

import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

MENAGERIE_ZOO = "vendored-pytorch"

# ---------------------------------------------------------------------------
# models/swin_transformer.py (verbatim)
# ---------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    r"""Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], (
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        )
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    r"""Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """x: B, H*W, C"""
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class Mlp(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution

            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )

            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(
                img_mask, self.window_size
            )  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
                attn_mask == 0, float(0.0)
            )
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class WindowAttention3(nn.Module):
    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)

        self.q2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.k2 = nn.Linear(dim, dim, bias=qkv_bias)

        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_diff, x_hr, x_lr, x_c, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x_diff.shape
        q = (
            self.q(x_lr)
            .reshape(B_, N, 1, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k = (
            self.k(x_hr)
            .reshape(B_, N, 1, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q2 = (
            self.q2(x_lr)
            .reshape(B_, N, 1, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k2 = (
            self.k2(x_c)
            .reshape(B_, N, 1, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        v = (
            self.v(x_diff)
            .reshape(B_, N, 1, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        q2 = q2 * self.scale
        attn2 = q2 @ k2.transpose(-2, -1)

        attn2 = attn2 + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn2 = attn2.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn2 = attn2.view(-1, self.num_heads, N, N)
            attn2 = self.softmax(attn2)
        else:
            attn2 = self.softmax(attn2)

        # avoid zero
        attn = attn * attn2 + 1e-6
        attn = attn / torch.sum(attn, dim=3, keepdim=True)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock3(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm3 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution

            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )

            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(
                img_mask, self.window_size
            )  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
                attn_mask == 0, float(0.0)
            )
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x_diff, x_lr, x_hr, x_c):
        H, W = self.input_resolution
        B, L, C = x_diff.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x_diff
        x_diff = self.norm1(x_diff)

        x_diff = x_diff.view(B, H, W, C)

        x_lr = x_lr.view(B, H, W, C)
        x_hr = x_hr.view(B, H, W, C)
        x_c = x_c.view(B, H, W, C)

        images = []
        for x in [x_diff, x_hr, x_lr, x_c]:
            # cyclic shift
            if self.shift_size > 0:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            else:
                shifted_x = x

            # partition windows
            x_windows = window_partition(
                shifted_x, self.window_size
            )  # nW*B, window_size, window_size, C
            x_windows = x_windows.view(
                -1, self.window_size * self.window_size, C
            )  # nW*B, window_size*window_size, C
            images.append(x_windows)

        # W-MSA/SW-MSA
        attn_windows = self.attn(
            images[0], images[1], images[2], images[3], mask=self.attn_mask
        )  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        x = x + shortcut

        x = x + self.mlp(self.norm3(x))

        return x


class BasicLayer3(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock3(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)

    def forward(self, x_diff, x_lr, x_hr, x_c):
        x_lr = self.norm1(x_lr)
        x_hr = self.norm2(x_hr)
        x_c = self.norm3(x_c)

        for blk in self.blocks:
            x_diff = blk(x_diff, x_lr, x_hr, x_c)
        return x_diff


# ---------------------------------------------------------------------------
# models/fem.py (verbatim)
# ---------------------------------------------------------------------------


class Down(nn.Module):
    def __init__(self, down_scale=2, in_dim=64, depths=(2, 2, 6, 2)):
        super(Down, self).__init__()
        self.inc = PatchEmbed(
            img_size=256,
            patch_size=down_scale,
            in_chans=6,
            embed_dim=in_dim,
            norm_layer=nn.LayerNorm,
        )
        self.down1 = DownBlock(
            in_channels=in_dim,
            out_channels=in_dim * 2,
            resolution=256 // down_scale,
            downsample=PatchMerging,
            cur_depth=depths[0],
        )
        self.down2 = DownBlock(
            in_channels=in_dim * 2,
            out_channels=in_dim * 4,
            resolution=128 // down_scale,
            downsample=PatchMerging,
            cur_depth=depths[1],
        )
        self.down3 = DownBlock(
            in_channels=in_dim * 4,
            out_channels=in_dim * 8,
            resolution=64 // down_scale,
            downsample=PatchMerging,
            cur_depth=depths[2],
        )
        self.down4 = DownBlock(
            in_channels=in_dim * 8,
            out_channels=in_dim * 8,
            resolution=32 // down_scale,
            downsample=PatchMerging,
            cur_depth=depths[3],
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        return x1, x2, x3, x4, x5


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, resolution, downsample, cur_depth):
        super(DownBlock, self).__init__()
        self.layer = BasicLayer(
            dim=in_channels,
            input_resolution=(resolution, resolution),
            depth=cur_depth,
            num_heads=in_channels // 32,
            window_size=8,
            mlp_ratio=1,
            qkv_bias=True,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
        )

        if downsample is not None:
            self.downsample = downsample((resolution, resolution), in_channels, out_channels)
        else:
            self.downsample = None

    def forward(self, x):
        x_o = self.layer(x)

        if self.downsample is not None:
            x_o = self.downsample(x_o)

        return x_o


# ---------------------------------------------------------------------------
# models/mfm.py (verbatim)
# ---------------------------------------------------------------------------


class FineUp(nn.Module):
    def __init__(self, in_dim=64, down_scale=2, depths=(2, 2, 6, 2)):
        super(FineUp, self).__init__()
        self.down_scale = down_scale
        self.up1 = FineUpBlock(in_dim * 8 * 2, in_dim * 4, 32 // down_scale, depths[3], 1)
        self.up2 = FineUpBlock(in_dim * 4 * 2, in_dim * 2, 64 // down_scale, depths[2], 2)
        self.up3 = FineUpBlock(in_dim * 2 * 2, in_dim, 128 // down_scale, depths[1], 4)
        self.up4 = FineUpBlock(in_dim * 2, in_dim, 256 // down_scale, depths[0], 8)

        self.outc = nn.Sequential(
            nn.Conv2d(in_dim, in_dim * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(in_dim, 6, 3, 1, 1),
            nn.Tanh(),
        )

        self.layer = BasicLayer3(
            dim=in_dim * 8,
            input_resolution=(16 // down_scale, 16 // down_scale),
            depth=1,
            num_heads=in_dim * 8 // 32,
            window_size=8,
            mlp_ratio=1,
            qkv_bias=True,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
        )

    def forward(self, diff0_features, diff1_features, fine_features, coarse_features):
        x0 = fine_features[4]
        x0 = self.layer(diff1_features[4], diff0_features[4], x0, coarse_features[4]) + x0

        x1 = self.up1(
            fine_features[3], diff0_features[3], coarse_features[3], diff1_features[3], x0
        )
        x2 = self.up2(
            fine_features[2], diff0_features[2], coarse_features[2], diff1_features[2], x1
        )
        x3 = self.up3(
            fine_features[1], diff0_features[1], coarse_features[1], diff1_features[1], x2
        )
        x4 = self.up4(
            fine_features[0], diff0_features[0], coarse_features[0], diff1_features[0], x3
        )

        B, L, C = x4.shape

        x4 = x4.transpose(1, 2).view(B, C, 256 // self.down_scale, 256 // self.down_scale)
        output_fine = self.outc(x4)

        return output_fine


class FineUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, resolution, cur_depth, gaussian_kernel_size):
        super(FineUpBlock, self).__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.up = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2 * 4, 3, 1, 1),
            nn.PixelShuffle(2),
        )

        self.layer = BasicLayer3(
            dim=in_channels // 2,
            input_resolution=(resolution, resolution),
            depth=2,
            num_heads=in_channels // 2 // 32,
            window_size=8,
            mlp_ratio=1,
            qkv_bias=True,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
        )

        self.layer2 = BasicLayer(
            dim=out_channels,
            input_resolution=(resolution, resolution),
            depth=cur_depth,
            num_heads=out_channels // 32,
            window_size=8,
            mlp_ratio=1,
            qkv_bias=True,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
        )

        self.proj1 = nn.Linear(in_channels // 2 * 3, out_channels)

    def forward(self, x_fine0, x_diff0, x_coarse0, x_diff1, x_fine1):
        """
        :param x_fine0: down-sampled fine feature
        :param x_diff0: down-sampled diff feature
        :param x_coarse0: down-sampled coarse feature
        :param x_fine1: upsampled fine feature from the previous stage
        :return:
        """
        B, L, C = x_fine1.shape

        x_f1 = x_fine1.transpose(1, 2).view(B, C, self.resolution // 2, self.resolution // 2)
        x_f1 = self.up(x_f1).flatten(2).transpose(1, 2)

        x_f0 = self.layer(x_diff1, x_diff0, x_fine0, x_coarse0) + x_fine0
        x = torch.cat([x_coarse0, x_f0, x_f1], dim=2)
        x = self.proj1(x)
        x = self.layer2(x)

        return x


# ---------------------------------------------------------------------------
# models/swinstfm.py (verbatim)
# ---------------------------------------------------------------------------


class SwinSTFM(nn.Module):
    def __init__(self):
        super(SwinSTFM, self).__init__()
        self.coarse_down = Down()
        self.fine_down = Down()
        self.fine_up = FineUp()

    def forward(self, c0, f0, c1):
        coarse_fea0 = self.coarse_down(c0)
        coarse_fea1 = self.coarse_down(c1)
        fine_fea = self.fine_down(f0)
        diff_fea1 = []
        for i in range(5):
            diff_fea1.append(coarse_fea1[i] - coarse_fea0[i])

        diff_fea0 = []
        for i in range(5):
            diff_fea0.append(fine_fea[i] - coarse_fea0[i])

        output_fine = self.fine_up(coarse_fea0, diff_fea1, fine_fea, coarse_fea1)

        return output_fine


# ---------------------------------------------------------------------------
# Menagerie staging entry points
# ---------------------------------------------------------------------------


def build_swinstfm():
    """SwinSTFM at its real fixed architecture size (256x256, 6-band inputs)."""
    return SwinSTFM()


def example_input_swinstfm():
    """SwinSTFM.forward(c0, f0, c1): coarse@t0, fine@t0, coarse@t1 -- all
    (B, 6, 256, 256) per the real repo's Down() PatchEmbed(img_size=256, in_chans=6).
    """
    c0 = torch.randn(1, 6, 256, 256)
    f0 = torch.randn(1, 6, 256, 256)
    c1 = torch.randn(1, 6, 256, 256)
    return (c0, f0, c1)


MENAGERIE_ENTRIES = [
    ("swinstfm", "build_swinstfm", "example_input_swinstfm", 2022, "vendored-pytorch"),
]
