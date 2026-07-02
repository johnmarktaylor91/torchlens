# SOURCE: vendored from guopengf/ReconFormer @ main
# Files: models/Recurrent_Transformer.py + models/RS_attention.py + data/transforms.py
# https://github.com/guopengf/ReconFormer/blob/main/models/Recurrent_Transformer.py
# https://github.com/guopengf/ReconFormer/blob/main/models/RS_attention.py
# https://github.com/guopengf/ReconFormer/blob/main/data/transforms.py
#
# ReconFormer (Guo, Darvishi, Wang & ... , "ReconFormer: Accelerated MRI Reconstruction Using
# Recurrent Transformer", IEEE TMI 2023 / arXiv:2201.09376): a recurrent Swin-style windowed
# transformer for undersampled MRI k-space reconstruction. Each "TransBlock" alternates a
# learned conv up/down-sampling path with a Recurrent Pyramid Transformer Layer (RPTL) built
# from windowed multi-scale-QK self-attention blocks whose attention maps are recurrently mixed
# across unrolled iterations (`rec_att`), followed by a k-space data-consistency layer.
#
# Vendored verbatim aside from ONE mechanical modernization: the original `data/transforms.py`
# `fft2`/`ifft2` used the pre-1.7 `torch.fft(data, 2, normalized=...)` / `torch.ifft(...)` API,
# which was removed from PyTorch years ago. They are rewritten here to the modern
# `torch.fft.fft2`/`torch.fft.ifft2` complex-tensor API with the SAME centered-shift semantics
# (`ifftshift -> fft2(norm='ortho') -> fftshift`), verified to round-trip (fft2 then ifft2)
# to float32 precision. No architectural change -- purely an API-surface migration of a helper
# function the model's `DataConsistencyInKspace` module calls.
#
# MENAGERIE_ZOO = "vendored-pytorch"

import torch
import torch.utils.checkpoint as checkpoint
from torch import nn
from torch.nn import functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# vendored from data/transforms.py (fft2/ifft2 modernized to torch.fft; see header note)
# ---------------------------------------------------------------------------


def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)


def fft2(data, normalized=True):
    """
    Apply centered 2 dimensional Fast Fourier Transform.

    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.

    Returns:
        torch.Tensor: The FFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    # modernized from removed `torch.fft(data, 2, normalized=normalized)` (pre-1.7 API)
    data_c = torch.view_as_complex(data.contiguous())
    data_c = torch.fft.fft2(data_c, dim=(-2, -1), norm="ortho" if normalized else "backward")
    data = torch.view_as_real(data_c)
    data = fftshift(data, dim=(-3, -2))
    return data


def ifft2(data, normalized=True):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.

    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.

    Returns:
        torch.Tensor: The IFFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    # modernized from removed `torch.ifft(data, 2, normalized=normalized)` (pre-1.7 API)
    data_c = torch.view_as_complex(data.contiguous())
    data_c = torch.fft.ifft2(data_c, dim=(-2, -1), norm="ortho" if normalized else "backward")
    data = torch.view_as_real(data_c)
    data = fftshift(data, dim=(-3, -2))
    return data


# ---------------------------------------------------------------------------
# vendored from models/RS_attention.py
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

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r"""Image to Patch Unembedding
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

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


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
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        scale=(1, 3, 5),
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        shift_size=0,
        rec_att=False,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.shift_size = shift_size
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        self.rec_att = rec_att
        if self.rec_att:
            self.lambda_att = torch.nn.Parameter(torch.Tensor([0.25]))

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

        self.v = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)
        qk = []
        self.per_head_dim = dim // num_heads
        self.heads_per_scale = num_heads // len(scale)
        for s in scale:
            if s == 1:
                qk.append(
                    nn.Conv2d(
                        dim,
                        self.per_head_dim * self.heads_per_scale * 2,
                        s,
                        stride=1,
                        padding=s // 2,
                    )
                )
            elif s == 3:
                out_dim = self.per_head_dim * self.heads_per_scale * 2
                qk.append(
                    nn.Sequential(
                        nn.Conv2d(dim, out_dim // 4, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(out_dim // 4, out_dim // 4, 1, 1, 0),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(out_dim // 4, out_dim, 3, 1, 1),
                    )
                )
            elif s == 5:
                out_dim = self.per_head_dim * self.heads_per_scale * 2
                qk.append(
                    nn.Sequential(
                        nn.Conv2d(dim, out_dim // 8, 5, 1, 5 // 2),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(out_dim // 8, out_dim // 8, 1, 1, 0),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(out_dim // 8, out_dim, 5, 1, 5 // 2),
                    )
                )

        self.qk = nn.ModuleList(qk)
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
        if self.rec_att:
            # input x is dict conatining input and revious att
            previous_att = x[1]
            x = x[0]
        ####
        B, H, W, C = x.shape
        # print(self.window_size)
        x = x.permute(0, 3, 1, 2).contiguous()
        q = []
        k = []
        for conv in self.qk:
            qk = (
                conv(x)
                .reshape(B, 2, self.per_head_dim * self.heads_per_scale, H, W)
                .permute(1, 0, 3, 4, 2)
                .contiguous()
            )
            q.append(qk[0])
            k.append(qk[1])

        # [B H,W, (C/6)*2]
        q = torch.cat(q, dim=-1)
        k = torch.cat(k, dim=-1)
        v = self.v(x).permute(0, 2, 3, 1).contiguous()
        # partition windows
        qkv = torch.cat([q, k, v], dim=-1)

        if self.shift_size > 0:
            shifted_qkv = torch.roll(qkv, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_qkv = qkv

        qkv_windows = window_partition(
            shifted_qkv, self.window_size[0]
        )  # nW*B, window_size, window_size, C
        qkv = qkv_windows.reshape(-1, self.window_size[0] * self.window_size[0], 3, C).permute(
            2, 0, 1, 3
        )  # nW*B, window_size*window_size, C
        ###################
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # [3, Batch,heads,tokens,chanel]
        B_, N, _ = q.shape
        q = q.reshape(B_, N, -1, self.per_head_dim).permute(0, 2, 1, 3).contiguous()
        k = k.reshape(B_, N, -1, self.per_head_dim).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(B_, N, -1, self.per_head_dim).permute(0, 2, 1, 3).contiguous()
        q = q * self.scale
        # [b,h,t,c]*[b,h,c,t]
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
            attn_before_softmax = attn
            if not self.rec_att or previous_att is None:
                attn = self.softmax(attn)
        else:
            attn_before_softmax = attn
            if not self.rec_att or previous_att is None:
                attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        if self.rec_att and previous_att is not None:
            # attn = previous_att*0.5 + attn_before_softmax*0.5
            attn = previous_att * self.lambda_att + attn_before_softmax * (1.0 - self.lambda_att)
            # print(self.lambda_att)
            attn_before_softmax = attn
            attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if self.rec_att:
            x = (x, attn_before_softmax)  # {'x':x,'p_att':attn}
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock_MS(nn.Module):
    r"""Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

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
        rec_att=False,
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

        self.norm1 = norm_layer(dim)  # layer norm
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            shift_size=shift_size,
            rec_att=rec_att,
        )
        self.rec_att = rec_att

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)  # dim*4 = 96*4
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
        )

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size  # (H,W)
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (
            slice(0, -self.window_size),  # (0,:-40),(-40:-20),(-20,None)
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),  # (0,:-40),(-40:-20),(-20,None)
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt  # (0:-40)=0 (-40,-20,0:-40)=1
                cnt += 1

        mask_windows = window_partition(
            img_mask, self.window_size
        )  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(
            2
        )  # (-1,1,wind*wind) - #(-1,1,)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )

        return attn_mask

    def forward(self, x):
        # H, W = x_size
        H, W = self.input_resolution

        if self.rec_att:
            # input x is dict conatining input and revious att
            previous_att = x[1]
            x = x[0]

        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        # x now is BHWC
        # if self.input_resolution == x_size:

        if self.rec_att:
            x = (x, previous_att)  # {'x':x,'p_att':previous_att}
        attn_windows = self.attn(x, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        # else:
        #     attn_windows = self.attn(x, mask=self.calculate_mask(x_size).to(x.device))
        if self.rec_att:
            # input x is dict conatining input and revious att
            previous_att = attn_windows[1]
            attn_windows = attn_windows[0]

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
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.rec_att:
            x = (x, previous_att)  # {'x': x, 'p_att': previous_att}
        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
        )

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

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
        downsample=None,
        use_checkpoint=False,
        rec_att=False,
        shift=False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.rec_att = rec_att

        # build blocks
        if depth == 1:
            shift_size = 0 if not shift else window_size // 2
            self.blocks = nn.ModuleList(
                [
                    SwinTransformerBlock_MS(
                        dim=dim,
                        input_resolution=input_resolution,
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=shift_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=drop_path,
                        norm_layer=norm_layer,
                        rec_att=rec_att,
                    )
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    SwinTransformerBlock_MS(
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
                        rec_att=rec_att,
                    )
                    for i in range(depth)
                ]
            )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        if self.rec_att and self.depth > 1:
            # initialize the att_list for the block with more than one SwinTransformerBlocks
            if x[1] is None:
                list_previous_att = [None for _ in range(self.depth)]
            else:
                list_previous_att = x[1]
        if self.rec_att and self.depth > 1:
            for i, blk in enumerate(self.blocks):
                x = (x[0], list_previous_att[i])
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)
                list_previous_att[i] = x[1]
            if self.downsample is not None:
                x = self.downsample(x)
            x = (x[0], list_previous_att)
        else:
            for i, blk in enumerate(self.blocks):
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)
            if self.downsample is not None:
                x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class RPTL(nn.Module):
    """Recurrent Pyramid Transformer Layer (RPTL).
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

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
        downsample=None,
        use_checkpoint=False,
        img_size=224,
        patch_size=4,
        resi_connection="1conv",
        rec_att=False,
        shift=False,
    ):
        super(RPTL, self).__init__()

        self.rec_att = rec_att
        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            rec_att=rec_att,
            shift=shift,
        )

        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == "3conv":
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None
        )

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None
        )

    def forward(self, x, x_size):
        if self.rec_att:
            # previous_att = x['p_att']
            _x = x[0]
            x = self.residual_group(x)
            previous_att = x[1]
            x = x[0]
            x = self.patch_embed(self.conv(self.patch_unembed(x, x_size))) + _x
            x = (x, previous_att)  # {'x':x,'p_att':previous_att}
            return x
        else:
            return (
                self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x), x_size))) + x
            )

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


# ---------------------------------------------------------------------------
# vendored from models/Recurrent_Transformer.py
# ---------------------------------------------------------------------------


class DataConsistencyInKspace(nn.Module):
    """Create data consistency operator

    Warning: note that FFT2 (by the default of torch.fft) is applied to the last 2 axes of the input.
    This method detects if the input tensor is 4-dim (2D data) or 5-dim (3D data)
    and applies FFT2 to the (nx, ny) axis.

    """

    def __init__(self):
        super(DataConsistencyInKspace, self).__init__()

    def forward(self, *input, **kwargs):
        return self.perform(*input)

    def data_consistency(self, k, k0, mask):
        """
        k    - input in k-space
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        """

        out = (1 - mask) * k + mask * k0
        return out

    def perform(self, x, k0, mask):
        """
        x    - input in image domain, of shape (n, 2, nx, ny[, nt])
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        """
        x = x.permute(0, 2, 3, 1)
        k0 = k0.permute(0, 2, 3, 1)
        mask = mask.permute(0, 2, 3, 1)

        k = fft2(x)

        out = self.data_consistency(k, k0, mask)
        x_res = ifft2(out)

        x_res = x_res.permute(0, 3, 1, 2)

        return x_res


class RFB(nn.Module):
    """
    ReconFormer Block
    """

    def __init__(
        self,
        img_size,
        nf,
        depth,
        num_head,
        window_size,
        mlp_ratio,
        use_checkpoint,
        resi_connection,
        down=True,
        up_scale=None,
        down_scale=None,
    ):
        super(RFB, self).__init__()

        if down:
            img_size = img_size // down_scale
        else:
            img_size = int(img_size * up_scale)
        embed_dim = nf
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=1,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=nn.LayerNorm,
        )
        num_patches = self.patch_embed.num_patches  # noqa: F841 (unused in original source; kept verbatim)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=1,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=nn.LayerNorm,
        )

        self.RPTL1 = RPTL(
            dim=embed_dim,
            input_resolution=(patches_resolution[0], patches_resolution[1]),
            depth=depth,
            num_heads=num_head,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,  # no impact on SR results
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=use_checkpoint[0],
            img_size=img_size,
            patch_size=1,
            resi_connection=resi_connection,
            rec_att=True,
        )
        self.RPTL2 = RPTL(
            dim=embed_dim,
            input_resolution=(patches_resolution[0], patches_resolution[1]),
            depth=depth,
            num_heads=num_head,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,  # no impact on SR results
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=use_checkpoint[1],
            img_size=img_size,
            patch_size=1,
            resi_connection=resi_connection,
            rec_att=True,
            shift=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, hidden, h1_att, h2_att):
        x_size = (hidden.shape[2], hidden.shape[3])
        hidden = self.patch_embed(hidden)

        hidden = (hidden, h1_att)  # {'x': hi, 'p_att': c1_att}
        h1 = self.RPTL1(hidden, x_size)
        h1_att = h1[1]
        h1 = h1[0]

        h1 = (h1, h2_att)  # {'x': ic1, 'p_att': c2_att}
        h2 = self.RPTL2(h1, x_size)
        h2_att = h2[1]
        h2 = h2[0]

        h2 = self.norm(h2)  # B L C
        h2 = self.patch_unembed(h2, x_size)

        return h2, h1_att, h2_att


class TransBlock_UC(nn.Module):
    """
    learned up&down conv
    """

    def __init__(
        self,
        in_channels=2,
        out_channels=2,
        nf=64,
        down_scale=2,
        img_size=256,
        num_head=6,
        depth=6,
        window_size=7,
        mlp_ratio=2.0,
        use_checkpoint=(False, False),
        resi_connection="1conv",
    ):
        super(TransBlock_UC, self).__init__()

        if down_scale == 2:
            kernel1, stride1 = 3, 1
            kernel2, stride2 = 4, 2
        elif down_scale == 1:
            kernel1, stride1 = 3, 1
            kernel2, stride2 = 3, 1
        else:
            exit("Error: unrecognized down_scale")

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=nf,
                kernel_size=kernel1,
                stride=stride1,
                padding=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=nf,
                out_channels=nf,
                kernel_size=kernel2,
                stride=stride2,
                padding=1,
                bias=True,
            ),
        )

        self.RFB = RFB(
            img_size,
            nf,
            depth,
            num_head,
            window_size,
            mlp_ratio,
            use_checkpoint,
            resi_connection,
            down=True,
            down_scale=down_scale,
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=nf,
                out_channels=nf,
                kernel_size=kernel2,
                stride=stride2,
                padding=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=nf,
                out_channels=out_channels,
                kernel_size=kernel1,
                stride=stride1,
                padding=1,
                bias=True,
            ),
        )

        self.activation = nn.PReLU()
        self.DC_layer = DataConsistencyInKspace()

    def forward(self, x, hidden=None, h1_att=None, h2_att=None, k0=None, mask=None):
        if hidden is None:
            hidden = self.activation(self.encoder(x))
        else:
            h2, h1_att, h2_att = self.RFB(hidden, h1_att, h2_att)
            hidden = self.activation(self.encoder(x) + h2)

        out = self.decoder(hidden)
        out = self.DC_layer(out, k0, mask)

        return out, hidden, h1_att, h2_att


class TransBlock_OC(nn.Module):
    """
    learned up&down conv
    """

    def __init__(
        self,
        in_channels=2,
        out_channels=2,
        nf=64,
        up_scale=2,
        img_size=256,
        num_head=6,
        depth=6,
        window_size=7,
        mlp_ratio=2.0,
        use_checkpoint=(False, False),
        resi_connection="1conv",
    ):
        super(TransBlock_OC, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=nf,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=up_scale),
            nn.Conv2d(
                in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1, bias=True
            ),
        )

        self.RFB = RFB(
            img_size,
            nf,
            depth,
            num_head,
            window_size,
            mlp_ratio,
            use_checkpoint,
            resi_connection,
            down=False,
            up_scale=up_scale,
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1, bias=True
            ),
            nn.Upsample(scale_factor=1 / up_scale),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=nf,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
        )

        self.activation = nn.PReLU()
        self.DC_layer = DataConsistencyInKspace()

    def forward(self, x, hidden=None, h1_att=None, h2_att=None, k0=None, mask=None):
        if hidden is None:
            hidden = self.activation(self.encoder(x))
        else:
            h2, h1_att, h2_att = self.RFB(hidden, h1_att, h2_att)
            hidden = self.activation(self.encoder(x) + h2)

        out = self.decoder(hidden)
        out = self.DC_layer(out, k0, mask)

        return out, hidden, h1_att, h2_att


class RefineModule(nn.Module):
    """
    Refine Module
    """

    def __init__(self, in_channels, nf, out_channels):
        super(RefineModule, self).__init__()

        self.rm = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=nf,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1, bias=True
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1, bias=True
            ),
            nn.Conv2d(
                in_channels=nf,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
        )

        self.DC_layer = DataConsistencyInKspace()

    def forward(self, x, k0=None, mask=None):
        return self.DC_layer(self.rm(x), k0, mask)


class ReconFormer(nn.Module):
    def __init__(
        self,
        in_channels=2,
        out_channels=2,
        num_ch=(64, 64, 64),
        down_scales=(2, 1, 1.5),
        num_iter=5,
        img_size=256,
        num_heads=(6, 6, 6),
        depths=(6, 6, 6),
        window_sizes=(8, 8, 8),
        resi_connection="1conv",
        mlp_ratio=2.0,
        use_checkpoint=(False, False, False, False, False, False),
    ):
        super(ReconFormer, self).__init__()

        self.num_iter = num_iter

        self.block1 = TransBlock_UC(
            in_channels=in_channels,
            out_channels=out_channels,
            nf=num_ch[0],
            down_scale=down_scales[0],
            num_head=num_heads[0],
            depth=depths[0],
            img_size=img_size,
            window_size=window_sizes[0],
            mlp_ratio=mlp_ratio,
            use_checkpoint=(use_checkpoint[0], use_checkpoint[1]),
            resi_connection=resi_connection,
        )

        self.block2 = TransBlock_UC(
            in_channels=in_channels,
            out_channels=out_channels,
            nf=num_ch[1],
            down_scale=down_scales[1],
            num_head=num_heads[1],
            depth=depths[1],
            img_size=img_size,
            window_size=window_sizes[1],
            mlp_ratio=mlp_ratio,
            use_checkpoint=(use_checkpoint[2], use_checkpoint[3]),
            resi_connection=resi_connection,
        )

        self.block3 = TransBlock_OC(
            in_channels=in_channels,
            out_channels=out_channels,
            nf=num_ch[2],
            up_scale=down_scales[2],
            num_head=num_heads[2],
            depth=depths[2],
            img_size=img_size,
            window_size=window_sizes[2],
            mlp_ratio=mlp_ratio,
            use_checkpoint=(use_checkpoint[4], use_checkpoint[5]),
            resi_connection=resi_connection,
        )

        self.RM = RefineModule(
            in_channels=int(out_channels * 3), nf=num_ch[2], out_channels=out_channels
        )

    def forward(self, x, k0=None, mask=None):
        outputs = []
        for i in range(self.num_iter):
            if i == 0:
                x1, h1, _, _ = self.block1(x, k0=k0, mask=mask)
                x2, h2, _, _ = self.block2(x1, k0=k0, mask=mask)
                x3, h3, _, _ = self.block3(x2, k0=k0, mask=mask)
            elif i == 1:
                x = outputs[-1]
                x1, h1, b1_c1_att, b1_c2_att = self.block1(x, hidden=h1, k0=k0, mask=mask)
                x2, h2, b2_c1_att, b2_c2_att = self.block2(x1, hidden=h2, k0=k0, mask=mask)
                x3, h3, b3_c1_att, b3_c2_att = self.block3(x2, hidden=h3, k0=k0, mask=mask)
            else:
                x = outputs[-1]
                x1, h1, b1_c1_att, b1_c2_att = self.block1(
                    x, hidden=h1, h1_att=b1_c1_att, h2_att=b1_c2_att, k0=k0, mask=mask
                )
                x2, h2, b2_c1_att, b2_c2_att = self.block2(
                    x1, hidden=h2, h1_att=b2_c1_att, h2_att=b2_c2_att, k0=k0, mask=mask
                )
                x3, h3, b3_c1_att, b3_c2_att = self.block3(
                    x2, hidden=h3, h1_att=b3_c1_att, h2_att=b3_c2_att, k0=k0, mask=mask
                )
            out = torch.cat((x1, x2, x3), dim=1)
            out = self.RM(out, k0, mask)
            outputs.append(out)

        return outputs[-1]


# ---------------------------------------------------------------------------
# menagerie staging entrypoints
# ---------------------------------------------------------------------------


def build_reconformer():
    """Tiny ReconFormer: small img_size/channel-count/depth/num_iter for fast tracing,
    same architecture shape (3 TransBlocks + RefineModule, DC layer per block) as the repo
    default. img_size=32 chosen so RFB's down_scale=2/1/1.5 patch resolutions stay integral
    (32 -> 16 -> 16 -> 24). num_ch=12/num_heads=3 is the smallest config satisfying
    WindowAttention's multi-scale QK convs, which split heads evenly across its 3 fixed
    kernel scales (1x1/3x3/5x5) and then the 5x5 branch further divides that per-scale width
    by 8 internally: num_heads must be a multiple of 3, nf (=dim) a multiple of num_heads, and
    the resulting per-scale width (per_head_dim * heads_per_scale * 2) must be >= 8."""
    return ReconFormer(
        in_channels=2,
        out_channels=2,
        num_ch=(12, 12, 12),
        down_scales=(2, 1, 1.5),
        num_iter=2,
        img_size=32,
        num_heads=(3, 3, 3),
        depths=(1, 1, 1),
        window_sizes=(4, 4, 4),
        resi_connection="1conv",
        mlp_ratio=2.0,
    )


def example_input_reconformer():
    # (x, k0, mask): x is the zero-filled image-domain input (2-channel real/imag), k0 is the
    # sampled k-space, mask is the sampling mask -- all (batch, 2, H, W) matching img_size=32,
    # as consumed by DataConsistencyInKspace inside every TransBlock.
    x = torch.randn(1, 2, 32, 32)
    k0 = torch.randn(1, 2, 32, 32)
    mask = torch.randint(0, 2, (1, 2, 32, 32)).float()
    return (x, k0, mask)


MENAGERIE_ENTRIES = [
    ("ReconFormer", "build_reconformer", "example_input_reconformer", "2023", "medical-imaging"),
]
