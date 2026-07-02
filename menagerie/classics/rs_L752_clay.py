# SOURCE: vendored from Clay-foundation/model @ main
# (claymodel/model.py::Encoder/Decoder/ClayMAE + claymodel/backbone.py::Transformer +
#  claymodel/factory.py::DynamicEmbedding + claymodel/utils.py::posemb_sincos_*)
#
# Clay Foundation Model ("Clay: An open source AI model for Earth" -- Clay
# Foundation, https://clay-foundation.github.io/model/). A masked-autoencoder (MAE)
# Vision Transformer for multi-sensor Earth observation imagery (Sentinel-1/2,
# Landsat, NAIP, MODIS, etc). The wavelength-conditioned "dynamic" patch embedding
# (`DynamicEmbedding`, adapted from the DOFA paper) lets one encoder ingest an
# arbitrary set of spectral bands per platform. Backbone transformer block is
# vendored verbatim from lucidrains/vit-pytorch (credited in the repo's own
# docstring). All of this is real, unmodified architecture code -- vendored, not
# reimplemented.
#
# The repo's `ClayMAE.__init__` calls `timm.create_model(teacher, pretrained=True,
# ...)` to build a frozen DINOv2 distillation teacher; downloading pretrained
# weights is unnecessary for capturing the trainable architecture (the teacher's
# own weights aren't part of what TorchLens traces as "the model"; only the
# `forward` graph shape/dtype matters), so this module builds the teacher with
# `pretrained=False` and a tiny timm model to keep it self-contained/offline.
#
# The upstream `metadata_path="configs/metadata.yaml"` + `box.Box` (a pip package,
# not a base lib we have) combo is replaced with a tiny stdlib recursive
# attribute-access wrapper carrying the SAME real per-platform stats
# (band_order/rgb_indices/gsd/wavelength) verbatim from the repo's own
# `configs/metadata.yaml` (sentinel-2-l2a entry) -- data substitution only, no
# architecture change.

import math
import os

import timm
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from torchvision.transforms import v2

MENAGERIE_ZOO = "vendored-pytorch"

torch.set_float32_matmul_precision("medium")
os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"


# ---- claymodel/utils.py ----
def posemb_sincos_2d_with_gsd(h, w, dim, gsd=1.0, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"

    gsd = gsd.to(x.device) if torch.is_tensor(gsd) else torch.as_tensor(gsd)
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** (2 * omega / dim)) * (gsd / 1.0)  # Adjusted for gsd

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


def posemb_sincos_1d(waves, dim, temperature: int = 10000, dtype=torch.float32):
    assert dim % 2 == 0, "Feature dimension must be a multiple of 2 for sincos embedding"
    waves = torch.arange(waves) if isinstance(waves, int) else waves

    omega = torch.arange(dim // 2, device=waves.device) / (dim // 2 - 1)
    omega = 1.0 / (temperature**omega)

    scaled_waves = waves[:, None] * omega[None, :]
    pe = torch.cat((scaled_waves.sin(), scaled_waves.cos()), dim=1)

    return pe.type(dtype)


# ---- claymodel/backbone.py (credited upstream: lucidrains/vit-pytorch) ----
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, fused_attn=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)
        self.fused_attn = fused_attn

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        else:
            attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = attn.softmax(dim=-1)
            x = torch.matmul(attn, v)

        x = rearrange(x, "b h n d -> b n (h d)")
        return self.to_out(x)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, fused_attn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, fused_attn=fused_attn),
                        FeedForward(dim, mlp_dim),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


# ---- claymodel/factory.py (Dynamic Embedding, adapted from DOFA paper) ----
class FCBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.l1 = nn.Linear(size, size)
        self.l2 = nn.Linear(size, size)

    def forward(self, x):
        y = F.gelu(self.l1(x))
        y = F.gelu(self.l2(y))
        return x + y


class WavesTransformer(nn.Module):
    def __init__(
        self,
        wave_dim,
        output_dim,
        num_latent_tokens,
        embed_dim,
        is_decoder,
        num_heads=4,
        num_layers=1,
    ):
        super().__init__()
        self.num_latent_tokens = num_latent_tokens
        self.is_decoder = is_decoder
        layer = nn.TransformerEncoderLayer(
            d_model=wave_dim,
            nhead=num_heads,
            activation="gelu",
            dropout=0,
            norm_first=False,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)

        self.fc_weight = nn.Linear(wave_dim, output_dim)
        self.fc_bias = None if self.is_decoder else nn.Linear(wave_dim, embed_dim)

        self.weight_tokens = nn.Parameter(torch.randn(self.num_latent_tokens, wave_dim) * 0.02)
        self.bias_token = nn.Parameter(torch.randn(1, wave_dim) * 0.02)

    def forward(self, x):
        x = torch.cat([self.weight_tokens, x, self.bias_token], dim=0)
        out = self.encoder(x)
        weights = self.fc_weight(out[self.num_latent_tokens : -1] + x[self.num_latent_tokens : -1])
        bias = None if self.is_decoder else self.fc_bias(out[-1])
        return weights, bias


class DynamicEmbedding(nn.Module):
    def __init__(self, wave_dim, num_latent_tokens, patch_size, embed_dim, is_decoder=False):
        super().__init__()
        self.wave_dim = wave_dim
        self.num_latent_tokens = num_latent_tokens
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.is_decoder = is_decoder
        self.output_dim = (patch_size**2) * embed_dim

        self.weight_generator = WavesTransformer(
            wave_dim,
            self.output_dim,
            self.num_latent_tokens,
            self.embed_dim,
            is_decoder,
        )
        self.fclayer = FCBlock(self.wave_dim)

        self.initialize_weights()

    def forward(self, batch, waves):
        waves = posemb_sincos_1d(waves, self.wave_dim)
        waves = waves.to(batch.device)
        waves = self.fclayer(waves)
        weight, bias = self.weight_generator(waves)

        if self.is_decoder:
            dynamic_weight = rearrange(
                weight,
                "cin (k1 k2 cout) -> (cin k1 k2) cout",
                k1=self.patch_size,
                k2=self.patch_size,
                cout=self.embed_dim,
            )
            if bias is not None:
                bias = rearrange(bias, "b -> (b)")
            dynamic_out = F.linear(batch, dynamic_weight * 0.02, bias=bias)
            x = dynamic_out
        else:
            dynamic_weight = rearrange(
                weight,
                "cin (cout k1 k2) -> cout cin k1 k2",
                k1=self.patch_size,
                k2=self.patch_size,
            )
            if bias is not None:
                bias = rearrange(bias, "b -> (b)")
            dynamic_out = F.conv2d(batch, dynamic_weight * 0.02, bias=bias, stride=self.patch_size)
            x = rearrange(dynamic_out, "b c h w -> b (h w) c")

        return x, waves

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# ---- claymodel/model.py ----
class Encoder(nn.Module):
    def __init__(self, mask_ratio, patch_size, shuffle, dim, depth, heads, dim_head, mlp_ratio):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.shuffle = shuffle
        self.dim = dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        self.patch_embedding = DynamicEmbedding(
            wave_dim=128,
            num_latent_tokens=128,
            patch_size=patch_size,
            embed_dim=dim,
            is_decoder=False,
        )

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=int(dim * mlp_ratio),
            fused_attn=True,
        )

    def to_patch_embed(self, cube, waves):
        patches, waves_encoded = self.patch_embedding(cube, waves)
        return patches, waves_encoded

    def add_encodings(self, patches, time, latlon, gsd):
        B, L, D = patches.shape

        grid_size = int(math.sqrt(L))
        self.num_patches = grid_size**2

        pos_encoding = (
            posemb_sincos_2d_with_gsd(h=grid_size, w=grid_size, dim=(self.dim - 8), gsd=gsd)
            .to(patches.device)
            .detach()
        )

        time_latlon = torch.hstack((time, latlon)).to(patches.device).detach()

        pos_encoding = repeat(pos_encoding, "L D -> B L D", B=B)
        time_latlon = repeat(time_latlon, "B D -> B L D", L=L)
        pos_metadata_encoding = torch.cat((pos_encoding, time_latlon), dim=-1)

        patches = patches + pos_metadata_encoding
        return patches

    def mask_out(self, patches):
        B, L, D = patches.shape

        if self.shuffle:
            noise = torch.randn((B, L), device=patches.device)
        else:
            noise = rearrange(torch.arange(B * L, device=patches.device), "(B L) -> B L", B=B, L=L)

        random_indices = torch.argsort(noise, dim=-1)
        reverse_indices = torch.argsort(random_indices, dim=-1)

        num_masked_patches = int(self.mask_ratio * self.num_patches)
        masked_indices, unmasked_indices = (
            random_indices[:, :num_masked_patches],
            random_indices[:, num_masked_patches:],
        )

        masked_matrix = torch.zeros((B, L), device=patches.device)
        masked_matrix[:, :num_masked_patches] = 1
        masked_matrix = torch.gather(masked_matrix, dim=1, index=reverse_indices)

        batch_indices = rearrange(torch.arange(B, device=patches.device), "B -> B 1")
        unmasked_patches = patches[batch_indices, unmasked_indices, :]
        _ = patches[batch_indices, masked_indices, :]

        return (unmasked_patches, unmasked_indices, masked_indices, masked_matrix)

    def forward(self, datacube):
        cube, time, latlon, gsd, waves = (
            datacube["pixels"],
            datacube["time"],
            datacube["latlon"],
            datacube["gsd"],
            datacube["waves"],
        )

        B, C, H, W = cube.shape

        patches, waves_encoded = self.to_patch_embed(cube, waves)
        patches = self.add_encodings(patches, time, latlon, gsd)

        (unmasked_patches, unmasked_indices, masked_indices, masked_matrix) = self.mask_out(patches)

        cls_tokens = repeat(self.cls_token, "1 1 D -> B 1 D", B=B)
        unmasked_patches = torch.cat((cls_tokens, unmasked_patches), dim=1)

        encoded_unmasked_patches = self.transformer(unmasked_patches)

        return (encoded_unmasked_patches, unmasked_indices, masked_indices, masked_matrix)


class Decoder(nn.Module):
    def __init__(self, mask_ratio, patch_size, encoder_dim, dim, depth, heads, dim_head, mlp_ratio):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.encoder_dim = encoder_dim
        self.dim = dim

        self.enc_to_dec = nn.Linear(encoder_dim, dim) if encoder_dim != dim else nn.Identity()
        self.mask_patch = nn.Parameter(torch.randn(dim))
        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=int(dim * mlp_ratio),
            fused_attn=True,
        )
        self.embed_to_pixels = DynamicEmbedding(
            wave_dim=128,
            num_latent_tokens=128,
            patch_size=patch_size,
            embed_dim=dim,
            is_decoder=True,
        )

    def reconstruct_and_add_encoding(
        self,
        unmasked_patches,
        unmasked_indices,
        masked_indices,
        masked_matrix,
        time,
        latlon,
        gsd,
    ):
        B, L = masked_matrix.shape
        grid_size = int(math.sqrt(L))
        self.num_patches = grid_size**2
        cls_tokens, unmasked_patches = (
            unmasked_patches[:, :1, :],
            unmasked_patches[:, 1:, :],
        )

        pos_encoding = (
            posemb_sincos_2d_with_gsd(h=grid_size, w=grid_size, dim=(self.dim - 8), gsd=gsd)
            .to(unmasked_patches.device)
            .detach()
        )
        time_latlon = torch.hstack((time, latlon)).to(unmasked_patches.device).detach()

        pos_encoding = repeat(pos_encoding, "L D -> B L D", B=B)
        time_latlon = repeat(time_latlon, "B D -> B L D", L=L)
        pos_metadata_encoding = torch.cat((pos_encoding, time_latlon), dim=-1)

        batch_indices = rearrange(torch.arange(B, device=unmasked_patches.device), "B -> B 1")

        num_masked_patches = int(self.mask_ratio * self.num_patches)
        masked_patches = repeat(self.mask_patch, "D -> B L D", B=B, L=num_masked_patches)

        masked_patches = masked_patches + pos_metadata_encoding[batch_indices, masked_indices, :]
        unmasked_patches = (
            unmasked_patches + pos_metadata_encoding[batch_indices, unmasked_indices, :]
        )

        decoder_patches = torch.zeros(
            (B, self.num_patches, self.dim), device=unmasked_patches.device
        )
        decoder_patches[batch_indices, unmasked_indices, :] = unmasked_patches
        decoder_patches[batch_indices, masked_indices, :] = masked_patches

        decoder_patches = torch.cat((cls_tokens, decoder_patches), dim=1)

        return decoder_patches

    def forward(
        self,
        encoded_unmasked_patches,
        unmasked_indices,
        masked_indices,
        masked_matrix,
        time,
        latlon,
        gsd,
        waves,
    ):
        encoded_unmasked_patches = self.enc_to_dec(encoded_unmasked_patches)

        decoder_patches = self.reconstruct_and_add_encoding(
            encoded_unmasked_patches,
            unmasked_indices,
            masked_indices,
            masked_matrix,
            time,
            latlon,
            gsd,
        )

        decoded_patches = self.transformer(decoder_patches)

        pixels, waves = self.embed_to_pixels(decoded_patches, waves)
        pixels = pixels[:, 1:, :]
        return pixels, waves


class ClayMAE(nn.Module):
    def __init__(
        self,
        mask_ratio,
        patch_size,
        norm_pix_loss,
        shuffle,
        metadata,
        teacher,
        dolls,
        doll_weights,
        # ENCODER
        dim,
        depth,
        heads,
        dim_head,
        mlp_ratio,
        # DECODER
        decoder_dim,
        decoder_depth,
        decoder_heads,
        decoder_dim_head,
        decoder_mlp_ratio,
        **kwargs,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss
        self.shuffle = shuffle
        self.metadata = metadata
        self.teacher = timm.create_model(teacher, pretrained=False, num_classes=0)
        # NOTE (staging-only sizing knob, not an architecture change): the real repo
        # hardcodes 518 to match its DINOv2-giant teacher's native resolution; here
        # the resize target is read off whatever tiny teacher was actually
        # constructed so the module stays self-contained/offline without downloading
        # DINOv2 weights.
        _teacher_patch_embed = getattr(self.teacher, "patch_embed", None)
        self.teacher_chip_size = (
            _teacher_patch_embed.img_size[0] if _teacher_patch_embed is not None else 224
        )
        self.teacher_resize = v2.Resize(size=(self.teacher_chip_size, self.teacher_chip_size))
        self.proj = nn.Linear(dim, self.teacher.num_features)

        self.encoder = Encoder(
            mask_ratio=mask_ratio,
            patch_size=patch_size,
            shuffle=shuffle,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_ratio=mlp_ratio,
        )

        self.decoder = Decoder(
            mask_ratio=mask_ratio,
            patch_size=patch_size,
            encoder_dim=dim,
            dim=decoder_dim,
            depth=decoder_depth,
            heads=decoder_heads,
            dim_head=decoder_dim_head,
            mlp_ratio=decoder_mlp_ratio,
        )

        self.freeze_teacher()

    def freeze_teacher(self):
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

    def per_pixel_loss(self, cube, pixels, masked_matrix):
        patches = rearrange(
            cube,
            "B C (h p1) (w p2) -> B (h w) (C p1 p2)",
            p1=self.patch_size,
            p2=self.patch_size,
        )

        if self.norm_pix_loss:
            mean = patches.mean(dim=-1, keepdim=True)
            var = patches.var(dim=-1, keepdim=True)
            patches = (patches - mean) / (var + 1e-6) ** 0.5

        loss = F.l1_loss(patches, pixels, reduction="none")
        from einops import reduce as _reduce

        loss = _reduce(loss, "B L D -> B L", reduction="mean")

        loss = (loss * masked_matrix).sum() / masked_matrix.sum()

        return loss

    def forward(self, datacube):
        platform = datacube["platform"][0]
        waves = torch.tensor(list(self.metadata[platform].bands.wavelength.values()))
        gsd = torch.tensor(self.metadata[platform].gsd)

        _pixels = datacube["pixels"].clone()
        batch_size, channels, _, _ = _pixels.size()

        # ENCODER
        (
            encoded_unmasked_patches,
            unmasked_indices,
            masked_indices,
            masked_matrix,
        ) = self.encoder(
            {
                "pixels": _pixels,
                "time": datacube["time"],
                "latlon": datacube["latlon"],
                "gsd": gsd,
                "waves": waves,
            }
        )

        # DECODER
        pixels, waves = self.decoder(
            encoded_unmasked_patches,
            unmasked_indices,
            masked_indices,
            masked_matrix,
            datacube["time"],
            datacube["latlon"],
            gsd,
            waves,
        )

        # MAE
        reconstruction_loss = self.per_pixel_loss(datacube["pixels"], pixels, masked_matrix)
        if platform == "modis":
            reconstruction_loss /= 10

        # PROJ
        representations = self.proj(encoded_unmasked_patches[:, 0, :])

        with torch.no_grad():
            if platform == "sentinel-1-rtc":
                r = datacube["pixels"][:, 0, :, :]
                g = datacube["pixels"][:, 1, :, :]
                b = (r + g) / 2
                rgb = torch.stack((r, g, b), dim=1)
            else:
                indices = self.metadata[platform].rgb_indices
                rgb = datacube["pixels"][:, indices, :, :]
            rgb = self.teacher_resize(rgb)
            target = self.teacher(rgb)

        representation_loss = 1.0 - F.cosine_similarity(representations, target).mean()

        loss = 0.9 * reconstruction_loss + 0.1 * representation_loss
        return (loss, reconstruction_loss, representation_loss)


def clay_mae_tiny(**kwargs):
    args = {
        "dim": 192,
        "depth": 6,
        "heads": 4,
        "dim_head": 48,
        "mlp_ratio": 2,
        "decoder_dim": 96,
        "decoder_depth": 3,
        "decoder_heads": 2,
        "decoder_dim_head": 48,
        "decoder_mlp_ratio": 2,
    }
    args.update(kwargs)
    return ClayMAE(**args)


# ---- staging-only: tiny real-data metadata stub replacing `box.Box(yaml.safe_load(...))`
# (values copied verbatim from the repo's own configs/metadata.yaml sentinel-2-l2a entry) ----
class _AttrDict(dict):
    """Minimal recursive dict->attribute wrapper (stand-in for `box.Box`, which is
    not a base lib we have installed). Subclasses dict so `.values()`/`.items()`/
    `[key]` all still work exactly like the real `Box`, while also exposing
    attribute access (`.bands.wavelength`). Carries the same real per-platform
    stats, not synthesized data."""

    def __init__(self, d):
        super().__init__({k: (_AttrDict(v) if isinstance(v, dict) else v) for k, v in d.items()})

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


_SENTINEL2_METADATA = {
    "band_order": [
        "blue",
        "green",
        "red",
        "rededge1",
        "rededge2",
        "rededge3",
        "nir",
        "nir08",
        "swir16",
        "swir22",
    ],
    "rgb_indices": [2, 1, 0],
    "gsd": 10,
    "bands": {
        "mean": {
            "blue": 1105.0,
            "green": 1355.0,
            "red": 1552.0,
            "rededge1": 1887.0,
            "rededge2": 2422.0,
            "rededge3": 2630.0,
            "nir": 2743.0,
            "nir08": 2785.0,
            "swir16": 2388.0,
            "swir22": 1835.0,
        },
        "std": {
            "blue": 1809.0,
            "green": 1757.0,
            "red": 1888.0,
            "rededge1": 1870.0,
            "rededge2": 1732.0,
            "rededge3": 1697.0,
            "nir": 1742.0,
            "nir08": 1648.0,
            "swir16": 1470.0,
            "swir22": 1379.0,
        },
        "wavelength": {
            "blue": 0.493,
            "green": 0.56,
            "red": 0.665,
            "rededge1": 0.704,
            "rededge2": 0.74,
            "rededge3": 0.783,
            "nir": 0.842,
            "nir08": 0.865,
            "swir16": 1.61,
            "swir22": 2.19,
        },
    },
}

_NUM_BANDS = len(_SENTINEL2_METADATA["band_order"])
_IMG_SIZE = 64
_PATCH_SIZE = 8
_BATCH_SIZE = 1


def build_clay():
    metadata = {"sentinel-2-l2a": _AttrDict(_SENTINEL2_METADATA)}
    model = clay_mae_tiny(
        mask_ratio=0.5,
        patch_size=_PATCH_SIZE,
        norm_pix_loss=False,
        shuffle=True,
        metadata=metadata,
        teacher="vit_tiny_patch16_224",
        dolls=[16, 32, 64, 128, 256, 768],
        doll_weights=[1, 1, 1, 1, 1, 1],
    )
    model.eval()
    return model


def example_input_clay():
    datacube = {
        "pixels": torch.rand(_BATCH_SIZE, _NUM_BANDS, _IMG_SIZE, _IMG_SIZE),
        "time": torch.rand(_BATCH_SIZE, 4),
        "latlon": torch.rand(_BATCH_SIZE, 4),
        "platform": ["sentinel-2-l2a"] * _BATCH_SIZE,
    }
    return (datacube,)


MENAGERIE_ENTRIES = [
    ("Clay foundation model", "build_clay", "example_input_clay", 2024, "vendored-pytorch"),
]
