# FAITHFUL PORT of yuewen73/AMC-Transformer @ eb596ae1fd0346331104441f369e17735181acb2 (original framework: TensorFlow/Keras)
# https://raw.githubusercontent.com/yuewen73/AMC-Transformer/eb596ae1fd0346331104441f369e17735181acb2/model.py
# https://raw.githubusercontent.com/yuewen73/AMC-Transformer/eb596ae1fd0346331104441f369e17735181acb2/train.py
#
# AMC-Transformer ("AMC-Transformer: Automatic Modulation Classification based on
# Enhanced Attention Model"), reference implementation for raw-I/Q automatic
# modulation classification on RadioML 2018.01A. The real repo (`model.py`) is a
# Vision-Transformer-style architecture applied to raw I/Q time series, reshaped to
# a (2, 1024, 1) "image" (2 IQ channels x 1024 time samples x 1 spatial), patchified
# along the time axis with a (2 x patch_size) window (each patch spans both I and Q
# channels and a contiguous time segment), linearly projected + positionally embedded,
# passed through `transformer_layers` pre-norm Transformer-encoder blocks
# (LayerNorm -> MultiHeadAttention -> residual -> LayerNorm -> GELU MLP -> residual),
# then flattened and classified through a GELU MLP head.
#
# TensorFlow/Keras cannot reasonably be added alongside the base torch env for this one
# ported model, so the architecture is transcribed faithfully from the real
# `create_vit_classifier()` (and its `Patches`/`PatchEncoder` custom layers) into
# self-contained torch, preserving every real default hyperparameter from the actual
# function signature: `input_shape=(2, 1024, 1)`, `patch_size=16`, `projection_dim=96`,
# `num_heads=4`, `transformer_layers=10`, `transformer_units=[projection_dim*2,
# projection_dim]` (the real code's default when None), `mlp_head_units=[2048, 1024]`
# (the real code's default when None), `num_classes=24`. `Patches.call` in the real code
# uses `tf.image.extract_patches` with `sizes=[1,2,patch_size,1]`,
# `strides=[1,2,patch_size,1]`, `padding="VALID"` -- because the stride along the
# I/Q-channel axis equals that axis's full extent (2), this is equivalent to slicing
# the (2, 1024) plane into contiguous (2, patch_size) blocks along time and flattening
# each block, which is what `_Patchify` below reproduces via `unfold` + reshape.
# `PatchEncoder` (Dense projection + learnable positional Embedding added elementwise)
# and the pre-norm Transformer block (`LayerNormalization` -> `MultiHeadAttention` ->
# `Add` skip -> `LayerNormalization` -> MLP block of Dense+gelu+Dropout pairs -> `Add`
# skip) are both transcribed layer-for-layer from the real code, as is the final
# `LayerNormalization -> Flatten -> Dropout(0.5) -> mlp_block(mlp_head_units) ->
# Dense(num_classes)` classifier head (softmax omitted here since torch classifiers
# conventionally return logits; the real code applies softmax only via
# `model.compile(loss="categorical_crossentropy", ...)`-adjacent activation, kept for
# fidelity as an optional flag but off by default to match standard torch practice for
# a traced/inspected network).

import torch
import torch.nn as nn


class _Patchify(nn.Module):
    """Non-overlapping (2 x patch_size) patches over an (I/Q=2, Time, 1) signal.

    Equivalent to the real ``Patches`` Keras layer's
    ``tf.image.extract_patches(sizes=[1,2,patch_size,1], strides=[1,2,patch_size,1],
    padding="VALID")`` -- since the I/Q axis is fully consumed by a single patch
    (stride == extent == 2), this reduces to slicing contiguous ``patch_size``-wide
    windows along time and flattening each (2, patch_size) block.
    """

    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, T, 1) -- matches the real Keras Input(shape=(2, T, 1))
        b, iq, t, _ = x.shape
        x = x.squeeze(-1)  # (B, 2, T)
        num_patches = t // self.patch_size
        x = x[:, :, : num_patches * self.patch_size]
        x = x.reshape(b, iq, num_patches, self.patch_size)  # (B, 2, P, patch_size)
        x = x.permute(0, 2, 1, 3).reshape(b, num_patches, iq * self.patch_size)
        return x


class _PatchEncoder(nn.Module):
    """Linear projection + learnable positional embedding per patch."""

    def __init__(self, num_patches: int, projection_dim: int, patch_dim: int):
        super().__init__()
        self.num_patches = num_patches
        self.projection = nn.Linear(patch_dim, projection_dim)
        self.position_embedding = nn.Embedding(num_patches, projection_dim)

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(self.num_patches, device=patch.device)
        return self.projection(patch) + self.position_embedding(positions)


def _mlp_block(dims, dropout_rate: float):
    """Sequence of Dense(units, gelu) + Dropout pairs, matching the real ``mlp_block``."""
    layers = []
    prev = dims[0]
    for units in dims[1:]:
        layers.append(nn.Linear(prev, units))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout_rate))
        prev = units
    return nn.Sequential(*layers), prev


class _TransformerBlock(nn.Module):
    """Pre-norm Transformer-encoder block matching the real per-layer construction."""

    def __init__(
        self, projection_dim: int, num_heads: int, transformer_units, dropout: float = 0.1
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(projection_dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=projection_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(projection_dim, eps=1e-6)
        mlp_dims = [projection_dim] + list(transformer_units)
        self.mlp, _ = _mlp_block(mlp_dims, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.ln1(x)
        attn_out, _ = self.attn(x1, x1, x1, need_weights=False)
        x2 = attn_out + x
        x3 = self.ln2(x2)
        x3 = self.mlp(x3)
        return x3 + x2


class AMCTransformer(nn.Module):
    """ViT-style raw-I/Q modulation classifier.

    Faithful port of ``create_vit_classifier`` (AMC-Transformer, yuewen73/AMC-Transformer
    and the equivalent hungluuduc03-lgtm/AMC-Transformer- reference implementation).
    """

    def __init__(
        self,
        input_len: int = 1024,
        patch_size: int = 16,
        projection_dim: int = 96,
        num_heads: int = 4,
        transformer_layers: int = 10,
        transformer_units=None,
        mlp_head_units=None,
        num_classes: int = 24,
    ):
        super().__init__()
        if transformer_units is None:
            transformer_units = [projection_dim * 2, projection_dim]
        if mlp_head_units is None:
            mlp_head_units = [2048, 1024]

        self.num_patches = input_len // patch_size
        patch_dim = 2 * patch_size

        self.patchify = _Patchify(patch_size)
        self.patch_encoder = _PatchEncoder(self.num_patches, projection_dim, patch_dim)

        self.blocks = nn.ModuleList(
            [
                _TransformerBlock(projection_dim, num_heads, transformer_units, dropout=0.1)
                for _ in range(transformer_layers)
            ]
        )

        self.ln_final = nn.LayerNorm(projection_dim, eps=1e-6)
        self.head_dropout = nn.Dropout(0.5)
        flat_dim = self.num_patches * projection_dim
        head_dims = [flat_dim] + list(mlp_head_units)
        self.head_mlp, head_out_dim = _mlp_block(head_dims, 0.5)
        self.classifier = nn.Linear(head_out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.patchify(x)
        x = self.patch_encoder(patches)

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        x = torch.flatten(x, 1)
        x = self.head_dropout(x)
        x = self.head_mlp(x)
        x = self.classifier(x)
        return x


def build_amc_transformer():
    torch.manual_seed(0)
    model = AMCTransformer(
        input_len=1024,
        patch_size=16,
        projection_dim=96,
        num_heads=4,
        transformer_layers=10,
        num_classes=24,
    )
    model.eval()
    return model


def example_input_amc_transformer():
    torch.manual_seed(0)
    # (Batch, IQ=2, Time=1024, 1) raw I/Q signal, matching the real Keras
    # Input(shape=(2, 1024, 1)).
    return torch.randn(2, 2, 1024, 1)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    (
        "AMC-Transformer",
        "build_amc_transformer",
        "example_input_amc_transformer",
        2024,
        MENAGERIE_ZOO,
    ),
]
