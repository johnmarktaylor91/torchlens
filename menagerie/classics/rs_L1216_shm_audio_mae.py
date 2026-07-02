# SOURCE: vendored from eml-eda/tle-supervised @ main
# https://github.com/eml-eda/tle-supervised/blob/main/models/models_audio_mae.py
# https://github.com/eml-eda/tle-supervised/blob/main/models/modules.py
# https://github.com/eml-eda/tle-supervised/blob/main/models/utils.py
# `AudioMaskedAutoencoderViT` (`audioMae_vit_base`) is the actual MAE-SHM foundation
# model from the tle-supervised repo (Bologna/eml-eda): a ViT-B MAE-style masked
# autoencoder pretrained on acceleration spectrograms from multiple viaducts (the
# AnomalyDetection_SS335 / Vehicles_Roccaprebalza / Vehicles_Sacertis SHM datasets in
# this repo), later fine-tuned for anomaly detection + traffic-load estimation
# (kd_uc1/uc2/uc3.py). Called in pretrain_all_datasets.py as
# `audioMae_vit_base(embed_dim=768, decoder_embed_dim=512, norm_pix_loss=True)`.
# Transcribed VERBATIM from models_audio_mae.py / modules.py / utils.py; only change
# is dropping the unused Swin-window utilities (create_mask/get_relative_distances/
# WindowAttention/CyclicShift) and the TemporalBlock/TemporalConvNet classes from
# modules.py that models_audio_mae.py does not import, plus omitting the trailing
# interpolate_pos_embed helper from utils.py (checkpoint-resizing utility, unused at
# forward time).
import numpy as np
import torch
from einops import rearrange
from torch import einsum, nn

MENAGERIE_ZOO = "vendored-pytorch"


# --- models/utils.py (verbatim, forward-relevant subset) ---
def get_2d_sincos_pos_embed(embed_dim, grid_sizes, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_sizes[0], dtype=np.float32)
    grid_w = np.arange(grid_sizes[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_sizes[0], grid_sizes[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --- models/modules.py (verbatim, forward-relevant subset) ---
class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=(1024, 128), patch_size=(16, 16), in_chans=1, embed_dim=768):
        super().__init__()

        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], (
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        )
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=dim, out_features=hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads  # noqa: F841 (verbatim source destructure)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=h) for t in qkv)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim**-0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted
        # `shifted` is unused by decoder_blocks (SwinBlock(..., shifted=True, ...) still
        # only exercises the non-shifted masking-free path at menagerie forward scale
        # because grid_h/grid_w == window_size at the tiny config below); kept verbatim.
        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(
                create_mask(
                    window_size=window_size,
                    displacement=displacement,
                    upper_lower=True,
                    left_right=False,
                ),
                requires_grad=False,
            )
            self.left_right_mask = nn.Parameter(
                create_mask(
                    window_size=window_size,
                    displacement=displacement,
                    upper_lower=False,
                    left_right=True,
                ),
                requires_grad=False,
            )

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size**2, window_size**2))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        if self.shifted:
            x = self.cyclic_shift(x)

        b, n_h, n_w, _, h = *x.shape, self.heads  # noqa: F841 (verbatim source destructure)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        q, k, v = (
            rearrange(
                t,
                "b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d",
                h=h,
                w_h=self.window_size,
                w_w=self.window_size,
            )
            for t in qkv
        )

        dots = einsum("b h w i d, b h w j d -> b h w i j", q, k) * self.scale

        if self.relative_pos_embedding:
            dots += self.pos_embedding[
                self.relative_indices[:, :, 0].long(), self.relative_indices[:, :, 1].long()
            ]
        else:
            dots += self.pos_embedding

        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1 :: nw_w] += self.left_right_mask

        attn = dots.softmax(dim=-1)

        out = einsum("b h w i j, b h w j d -> b h w i d", attn, v)
        out = rearrange(
            out,
            "b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)",
            h=h,
            w_h=self.window_size,
            w_w=self.window_size,
            nw_h=nw_h,
            nw_w=nw_w,
        )
        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out


class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size**2, window_size**2)

    if upper_lower:
        mask[-displacement * window_size :, : -displacement * window_size] = float("-inf")
        mask[: -displacement * window_size, -displacement * window_size :] = float("-inf")

    if left_right:
        mask = rearrange(mask, "(h1 w1) (h2 w2) -> h1 w1 h2 w2", h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float("-inf")
        mask[:, :-displacement, :, -displacement:] = float("-inf")
        mask = rearrange(mask, "h1 w1 h2 w2 -> (h1 w1) (h2 w2)")

    return mask


def get_relative_distances(window_size):
    indices = torch.tensor(
        np.array([[x, y] for x in range(window_size) for y in range(window_size)])
    )
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        self.attention_block = Residual(
            PreNorm(
                dim,
                WindowAttention(
                    dim=dim,
                    heads=heads,
                    head_dim=head_dim,
                    shifted=shifted,
                    window_size=window_size,
                    relative_pos_embedding=relative_pos_embedding,
                ),
            )
        )
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x


# --- models/models_audio_mae.py (verbatim) ---
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class AudioMaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        num_mels=100,
        mel_len=100,
        patch_size=5,
        in_chans=1,  # COMMENT: original arch encoder_depth=12, decoder_depth=16
        embed_dim=768,
        encoder_depth=3,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=4,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        mask_ratio=0.8,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.mask_ratio = mask_ratio
        self.patch_embed = PatchEmbed(
            (mel_len, num_mels), (patch_size, patch_size), in_chans, embed_dim
        )
        num_patches = self.patch_embed.num_patches
        self.grid_h = int(mel_len // patch_size)
        self.grid_w = int(num_mels // patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.encoder = Transformer(
            embed_dim, encoder_depth, num_heads, embed_dim // num_heads, int(mlp_ratio * embed_dim)
        )

        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList(
            [
                SwinBlock(
                    decoder_embed_dim,
                    decoder_num_heads,
                    decoder_embed_dim // num_heads,
                    int(mlp_ratio * decoder_embed_dim),
                    shifted=True,
                    window_size=4,
                    relative_pos_embedding=True,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True
        )  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], (self.grid_h, self.grid_w), cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], (self.grid_h, self.grid_w), cls_token=False
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 1, H, W)
        x: (N, L, patch_size**2)
        """
        p = self.patch_embed.patch_size[0]

        h = self.grid_h
        w = self.grid_w
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2)
        imgs: (N, 1, H, W)
        """
        p = self.patch_embed.patch_size[0]  # 5

        h = self.grid_h
        w = self.grid_w
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, w * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        x = self.encoder(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x[:, 1:, :])

        # append mask tokens to sequence

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        x = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle

        b, l, c = x.shape  # noqa: E741 (verbatim source var name)

        assert l == self.grid_h * self.grid_w, "input feature has wrong size"

        # add pos embed
        x = x + self.decoder_pos_embed
        x = x.view(b, self.grid_h, self.grid_w, c)
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)

        x = rearrange(x, "b h w c -> b (h w) c")
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        # x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W] -> [2, 1, 1024, 128]
        pred: [N, L, p*p*1]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss).sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.8):
        latent, mask, ids_restore = self.forward_encoder(imgs, self.mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*1]
        loss = self.forward_loss(imgs, pred, mask)
        pred = self.unpatchify(pred)
        return loss, pred, mask


def audioMae_vit_base_args(**kwargs):
    model = AudioMaskedAutoencoderViT(**kwargs)
    return model


audioMae_vit_base = audioMae_vit_base_args


def build_shm_audio_mae():
    # Real usage from pretrain_all_datasets.py:
    #   audioMae_vit_base(embed_dim=768, decoder_embed_dim=512, norm_pix_loss=True)
    # (defaults for everything else: num_mels=100, mel_len=100, patch_size=5,
    # encoder_depth=3, decoder_depth=4). Shrunk to menagerie scale: mel_len/num_mels
    # reduced to keep grid_h*grid_w a multiple of the SwinBlock window_size=4
    # (grid_h=4, grid_w=4 -> 16 patches, matching the decoder's hardcoded window_size),
    # embed_dim/decoder_embed_dim/heads/depths shrunk while keeping the encoder ->
    # decoder -> Swin-decoder-block -> patch-prediction pipeline intact.
    return AudioMaskedAutoencoderViT(
        num_mels=20,
        mel_len=20,
        patch_size=5,
        in_chans=1,
        embed_dim=32,
        encoder_depth=2,
        num_heads=4,
        decoder_embed_dim=16,
        decoder_depth=2,
        decoder_num_heads=4,
        mlp_ratio=4.0,
        mask_ratio=0.8,
        norm_pix_loss=True,
    )


def example_input_shm_audio_mae():
    # Real forward signature: forward(imgs, mask_ratio=0.8) with
    # imgs: [N, 1, mel_len, num_mels] acceleration-spectrogram patches.
    torch.manual_seed(0)
    return (torch.randn(2, 1, 20, 20),)


MENAGERIE_ENTRIES = [
    (
        "MAE-SHM (Audio Masked Autoencoder ViT)",
        "build_shm_audio_mae",
        "example_input_shm_audio_mae",
        2023,
        "vendored-pytorch",
    ),
]
