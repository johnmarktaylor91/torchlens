# SOURCE: vendored from https://github.com/linhanwang/Drive-JEPA @ master (2025)
# Files:
#   navsim_v1/navsim/agents/drive_jepa_perception_free/drive_jepa_model.py (DriveJEPAModel, TrajectoryHead)
#   navsim_v1/vjepa2/src/models/vision_transformer.py (VisionTransformer, vit_tiny/vit_large/... -- Meta V-JEPA2)
#   navsim_v1/vjepa2/src/models/utils/modules.py (Block, Attention, MLP, DropPath -- Meta V-JEPA2)
#   navsim_v1/vjepa2/src/models/utils/patch_embed.py (PatchEmbed, PatchEmbed3D -- Meta V-JEPA2)
#   navsim_v1/vjepa2/src/models/utils/pos_embs.py (sincos pos-embed helpers -- Meta V-JEPA2)
#   navsim_v1/vjepa2/src/utils/tensors.py (trunc_normal_ -- Meta V-JEPA2)
#   navsim_v1/vjepa2/src/masks/utils.py (apply_masks -- Meta V-JEPA2, unused by this forward path)
#   navsim_v1/navsim/common/enums.py (StateSE2Index)
"""Drive-JEPA (perception-free variant): distills a frozen V-JEPA2 video ViT encoder's frame
features into an ego trajectory via a transformer encoder-decoder + MLP trajectory head, for
NAVSIM end-to-end driving.

Vendored (near-)verbatim from the official Drive-JEPA fork of navsim. What is adapted for
staging, and why it is still a vendor (not a rewrite of architecture):
  - `DriveJEPAModel.__init__` in the real repo loads `vjepa2/configs/eval/vitl/in1k.yaml` from
    disk and calls `vit_encoder.init_module(...)`, which unconditionally `torch.load()`s a
    pretrained V-JEPA2 checkpoint file (`pretrain_pt_path`) that does not ship with the repo and
    is not obtainable offline. That is weight loading, not architecture -- the encoder is
    constructed directly here via the same `vjepa2.src.models.vision_transformer.vit_*` factory
    the real code calls, at `vit_tiny` size for a fast trace, with the model's own random
    initialization (`VisionTransformer._init_weights` / `_rescale_blocks`, run unmodified).
  - `nuplan.planning.simulation.trajectory.trajectory_sampling.TrajectorySampling` is a tiny
    nuplan-devkit dataclass (`num_poses: int`, `interval_length: float`); nuplan-devkit is a
    heavy AV-stack dependency outside the base env, so it is stood in for with an equivalent
    plain dataclass here rather than installing the whole devkit for one field access.
  - `StateSE2Index.size()` mixes a `@classmethod @property` stack that Python 3.11+ removed;
    replaced with a plain class attribute alias to the same value (3), not an architecture change.
All neural-net code (image ViT encoder, transformer fusion, trajectory head) is transcribed
exactly as in the official source.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import drop_path

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# navsim/common/enums.py::StateSE2Index (trimmed classmethod-property shim)
# ---------------------------------------------------------------------------


class StateSE2Index:
    """IntEnum-equivalent for SE(2) arrays (X, Y, HEADING); `size()` == 3."""

    X = 0
    Y = 1
    HEADING = 2

    @classmethod
    def size(cls):
        return 3


# ---------------------------------------------------------------------------
# nuplan devkit stand-in (single dataclass field access, not an architecture dependency)
# ---------------------------------------------------------------------------


@dataclass
class TrajectorySampling:
    num_poses: int
    interval_length: float = 0.5


# ---------------------------------------------------------------------------
# vjepa2/src/utils/tensors.py
# ---------------------------------------------------------------------------


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    import math

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# ---------------------------------------------------------------------------
# vjepa2/src/models/utils/pos_embs.py
# ---------------------------------------------------------------------------


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid_w, grid_h = np.meshgrid(grid_w, grid_h)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_h)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_w)
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_3d_sincos_pos_embed(embed_dim, grid_size, grid_depth, cls_token=False, uniform_power=False):
    grid_d = np.arange(grid_depth, dtype=float)
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid_h, grid_d, grid_w = np.meshgrid(grid_h, grid_d, grid_w)

    if not uniform_power:
        h_embed_dim = embed_dim // 4
        w_embed_dim = embed_dim // 4
        d_embed_dim = embed_dim // 2
    else:
        h_embed_dim = w_embed_dim = d_embed_dim = int(np.ceil(embed_dim / 6) * 2)

    emb_h = get_1d_sincos_pos_embed_from_grid(h_embed_dim, grid_h)
    emb_w = get_1d_sincos_pos_embed_from_grid(w_embed_dim, grid_w)
    emb_d = get_1d_sincos_pos_embed_from_grid(d_embed_dim, grid_d)
    pos_embed = np.concatenate([emb_d, emb_h, emb_w], axis=1)
    pos_embed = pos_embed[:, :embed_dim]
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


# ---------------------------------------------------------------------------
# vjepa2/src/models/utils/patch_embed.py
# ---------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=16, tubelet_size=2, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x, **kwargs):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# ---------------------------------------------------------------------------
# vjepa2/src/models/utils/modules.py (attention-block stack; RoPE/AC variants omitted --
# unused by vit_tiny's default use_rope=False, use_sdpa=True configuration)
# ---------------------------------------------------------------------------


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
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


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
        is_causal=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.is_causal = is_causal

    def forward(self, x, mask=None, attn_mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        if attn_mask is not None or self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=self.proj_drop_prob,
                    is_causal=self.is_causal,
                    attn_mask=attn_mask,
                )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        wide_silu=True,
        norm_layer=nn.LayerNorm,
        use_sdpa=True,
        is_causal=False,
        grid_size=16,
        use_rope=False,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            use_sdpa=use_sdpa,
            is_causal=is_causal,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
        )

    def forward(self, x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
        y = self.attn(self.norm1(x), mask=mask, attn_mask=attn_mask)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# vjepa2/src/models/vision_transformer.py (RoPE variants / apply_masks path omitted --
# unused by this model's forward call, which passes masks=None, use_rope=False)
# ---------------------------------------------------------------------------


class VisionTransformer(nn.Module):
    """Vision Transformer (video-capable, tubelet patch embedding)."""

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        out_layers=None,
        uniform_power=False,
        use_silu=False,
        wide_silu=True,
        use_sdpa=True,
        use_activation_checkpointing=False,
        use_rope=False,
        handle_nonsquare_inputs=True,
        **kwargs,
    ):
        super().__init__()
        import math

        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.out_layers = out_layers
        self.handle_nonsquare_inputs = handle_nonsquare_inputs

        if type(img_size) is int:
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1
        self.use_activation_checkpointing = use_activation_checkpointing

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        if self.is_video:
            self.patch_embed = PatchEmbed3D(
                patch_size=patch_size,
                tubelet_size=tubelet_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
            self.num_patches = (
                (num_frames // tubelet_size)
                * (img_size[0] // patch_size)
                * (img_size[1] // patch_size)
            )
        else:
            self.patch_embed = PatchEmbed(
                patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
            )
            self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        self.uniform_power = uniform_power
        self.use_rope = use_rope
        if self.use_rope:
            self.pos_embed = None
        else:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches, embed_dim), requires_grad=False
            )

        self.blocks = nn.ModuleList(
            [
                Block(
                    use_rope=use_rope,
                    grid_size=img_size[0] // patch_size,
                    grid_depth=num_frames // tubelet_size,
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_sdpa=use_sdpa,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    wide_silu=wide_silu,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        if self.pos_embed is not None:
            self._init_pos_embed(self.pos_embed.data)
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()
        self._math = math

    def _init_pos_embed(self, pos_embed):
        embed_dim = pos_embed.size(-1)
        grid_size = self.img_height // self.patch_size
        if self.is_video:
            grid_depth = self.num_frames // self.tubelet_size
            sincos = get_3d_sincos_pos_embed(
                embed_dim, grid_size, grid_depth, cls_token=False, uniform_power=self.uniform_power
            )
        else:
            sincos = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False)
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(
                self._math.sqrt(2.0 * layer_id)
                if hasattr(self, "_math")
                else (2.0 * layer_id) ** 0.5
            )

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def forward(self, x, masks=None):
        if x.ndim == 4:
            _, _, H, W = x.shape
            T = 1
        elif x.ndim == 5:
            _, _, T, H, W = x.shape
            T = T // self.tubelet_size
        H_patches = H // self.patch_size
        W_patches = W // self.patch_size
        if not self.handle_nonsquare_inputs:
            T = H_patches = W_patches = None

        pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        x = self.patch_embed(x)
        x = x + pos_embed

        outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, mask=masks, attn_mask=None, T=T, H_patches=H_patches, W_patches=W_patches)
            if self.out_layers is not None and i in self.out_layers:
                outs.append(self.norm(x))

        if self.out_layers is not None:
            return outs
        if self.norm is not None:
            x = self.norm(x)
        return x

    def interpolate_pos_encoding(self, x, pos_embed):
        _, N, dim = pos_embed.shape
        if self.is_video:
            _, _, T, H, W = x.shape
            if H == self.img_height and W == self.img_width and T == self.num_frames:
                return pos_embed
            elif H == self.img_height and W == self.img_width and T < self.num_frames:
                new_N = int(
                    (T // self.tubelet_size) * (H // self.patch_size) * (W // self.patch_size)
                )
                return pos_embed[:, :new_N, :]

            T = T // self.tubelet_size
            H = H // self.patch_size
            W = W // self.patch_size
            N_t = self.num_frames // self.tubelet_size
            N_h = self.img_height // self.patch_size
            N_w = self.img_width // self.patch_size
            assert N_h * N_w * N_t == N, "Positional embedding initialized incorrectly"
            scale_factor = (T / N_t, H / N_h, W / N_w)
            pos_embed = F.interpolate(
                pos_embed.reshape(1, N_t, N_h, N_w, dim).permute(0, 4, 1, 2, 3),
                scale_factor=scale_factor,
                mode="trilinear",
            )
            pos_embed = pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)
            return pos_embed
        else:
            _, _, H, W = x.shape
            if H == self.img_height and W == self.img_width:
                return pos_embed
            import math

            npatch = (H // self.patch_size) * (W // self.patch_size)
            scale_factor = math.sqrt(npatch / N)
            pos_embed = F.interpolate(
                pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=scale_factor,
                mode="bicubic",
            )
            pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return pos_embed


def vit_tiny(patch_size=16, **kwargs):
    from functools import partial

    return VisionTransformer(
        patch_size=patch_size,
        embed_dim=192,
        depth=4,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


VIT_EMBED_DIMS = {"vit_tiny": 192}


# ---------------------------------------------------------------------------
# navsim/agents/drive_jepa_perception_free/drive_jepa_model.py
# ---------------------------------------------------------------------------


class TrajectoryHead(nn.Module):
    """Trajectory prediction head."""

    def __init__(self, num_poses: int, d_ffn: int, d_model: int):
        super().__init__()
        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn
        self._mlp = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, StateSE2Index.size()),
        )

    def forward(self, object_queries):
        poses = self._mlp(object_queries).reshape(-1, self._num_poses, StateSE2Index.size())
        heading = poses[..., StateSE2Index.HEADING].tanh() * np.pi
        poses = torch.cat([poses[..., :2], heading.unsqueeze(-1)], dim=-1)
        return {"trajectory": poses}


class DriveJEPAModel(nn.Module):
    """Perception-free Drive-JEPA: frozen V-JEPA2 ViT encoder -> transformer fusion -> trajectory.

    `image_encoder` is constructed directly via the real `vit_tiny` factory (random init) in
    place of the official `init_module(...)` pretrained-checkpoint loader (see module header).
    """

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        image_architecture: str = "vit_tiny",
        tf_d_model: int = 64,
        tf_d_ffn: int = 128,
        tf_num_layers: int = 2,
        tf_num_head: int = 4,
        tf_dropout: float = 0.0,
        num_keyval: int = 2 * 2 + 1,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        # NOTE: the official VisionTransformer._init_pos_embed computes a single square
        # `grid_size = img_height // patch_size` for its sincos pos-embed even when img_size is
        # non-square (upstream `# TODO: update; currently assumes square input`), so a square
        # resolution is used here to exercise the identical architecture without tripping that
        # pre-existing upstream limitation.
        self.image_encoder = vit_tiny(
            img_size=(64, 64), num_frames=2, tubelet_size=2, handle_nonsquare_inputs=True
        )
        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            self.image_encoder.eval()
            for p in self.image_encoder.parameters():
                p.requires_grad = False

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.image_fc = nn.Linear(VIT_EMBED_DIMS[image_architecture], tf_d_model)
        self._status_encoding = nn.Linear(4 + 2 + 2, tf_d_model)

        num_poses = trajectory_sampling.num_poses
        self._keyval_embedding = nn.Embedding(num_keyval, tf_d_model)
        self._query_embedding = nn.Embedding(num_poses, tf_d_model)

        self._transformer = nn.Transformer(
            d_model=tf_d_model,
            nhead=tf_num_head,
            num_encoder_layers=tf_num_layers,
            num_decoder_layers=tf_num_layers,
            dim_feedforward=tf_d_ffn,
            dropout=tf_dropout,
            batch_first=True,
        )
        self._trajectory_head = TrajectoryHead(num_poses, tf_d_ffn, tf_d_model)

    def forward(self, camera_feature, status_feature):
        batch_size = camera_feature.shape[0]
        H, W = camera_feature.shape[-2:]

        camera_feature = rearrange(camera_feature, "B C S H W -> (B S) C H W")
        camera_feature = rearrange(camera_feature, "(B S) C H W -> B C S H W", S=2)

        if self.freeze_encoder:
            with torch.no_grad():
                img_feat = self.image_encoder(camera_feature)
        else:
            img_feat = self.image_encoder(camera_feature)
        img_feat = rearrange(img_feat, "B (H W) D -> B D H W", H=H // 16, W=W // 16)
        img_feat = self.avg_pool(img_feat)
        img_feat = img_feat.flatten(-2, -1).permute(0, 2, 1)
        img_feat = self.image_fc(img_feat.clone())

        status_encoding = self._status_encoding(status_feature)

        keyval = torch.cat([img_feat, status_encoding[:, None]], dim=1)
        keyval = keyval.clone() + self._keyval_embedding.weight[None, ...]

        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._transformer(src=keyval, tgt=query)
        trajectory = self._trajectory_head(query_out)
        return trajectory["trajectory"]


# ---------------------------------------------------------------------------
# Menagerie staging glue
# ---------------------------------------------------------------------------


def build_drivejepa():
    sampling = TrajectorySampling(num_poses=8, interval_length=0.5)
    return DriveJEPAModel(trajectory_sampling=sampling)


def example_input_drivejepa():
    batch = 1
    # camera_feature: (B, C=3, S=2 stacked frames, H=64, W=64) -- small square front-camera crop.
    camera_feature = torch.randn(batch, 3, 2, 64, 64)
    # status_feature: (B, 4 ego-status + 2 driving-command + 2 velocity) == 8 dims.
    status_feature = torch.randn(batch, 8)
    return (camera_feature, status_feature)


MENAGERIE_ENTRIES = [
    (
        "Drive-JEPA (perception-free)",
        "build_drivejepa",
        "example_input_drivejepa",
        2025,
        "vendored-pytorch",
    ),
]
