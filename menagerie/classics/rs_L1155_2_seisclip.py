# SOURCE: vendored from sixu0/SeisCLIP @ main
# https://github.com/sixu0/SeisCLIP/blob/main/Event_classification/model/model_seismic_clip.py
# https://github.com/sixu0/SeisCLIP/blob/main/Event_classification/model/ast_models.py
# SeisCLIP (Si, Chen, Peng, Ren 2023): CLIP-style contrastive pretraining of
# a seismic-waveform-spectrogram encoder (Audio Spectrogram Transformer,
# AST -- Gong et al. 2021, wrapping a timm ViT) against an event-metadata
# text/attribute encoder (`Info_embedding`, a small residual FCN stack).
# `AUDIO_CLIP` (real class name in the repo) is transcribed verbatim from
# `model_seismic_clip.py`; `ASTModel`/`PatchEmbed` transcribed verbatim from
# `ast_models.py`. Two minimal, version-drift-only fixes required to run
# under the installed timm (1.0.26) instead of the repo's pinned timm==0.4.5:
#   1. `timm.models.layers.to_2tuple`/`trunc_normal_` moved to `timm.layers`
#      in modern timm (deprecated-alias warning otherwise) -- import path
#      updated only, `to_2tuple`/`trunc_normal_` themselves untouched.
#   2. the repo's `vit_deit_*_distilled_patch16_*` timm model names were
#      renamed `deit_*_distilled_patch16_*` (no `vit_` prefix) in modern
#      timm's model registry -- same architecture, updated string only.
#   3. the `assert timm.__version__ == '0.4.5'` version pin is dropped (pure
#      compatibility guard, not architecture) and `imagenet_pretrain` is set
#      False here so construction doesn't require network access to
#      torchvision-hub weights; both are training-time/environment choices,
#      not architectural changes to `AUDIO_CLIP`/`ASTModel`.
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Optional, Tuple

import timm
from timm.layers import to_2tuple, trunc_normal_


# ---- Event_classification/model/ast_models.py (verbatim, timm import path only) ----
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ASTModel(nn.Module):
    """
    The AST model.
    :param label_dim: the label dimension, i.e., the number of total classes, it is 527 for AudioSet, 50 for ESC-50, and 35 for speechcommands v2-35
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    :param imagenet_pretrain: if use ImageNet pretrained model
    :param audioset_pretrain: if use full AudioSet and ImageNet pretrained model
    :param model_size: the model size of AST, should be in [tiny224, small224, base224, base384], base224 and base 384 are same model, but are trained differently during ImageNet pretraining.
    """

    def __init__(
        self,
        label_dim=527,
        fstride=10,
        tstride=10,
        input_fdim=128,
        input_tdim=1024,
        imagenet_pretrain=True,
        audioset_pretrain=False,
        model_size="base384",
        verbose=True,
        load_pretrain_patch=120,
    ):
        super(ASTModel, self).__init__()
        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
        if audioset_pretrain == False:  # noqa: E712
            if model_size == "tiny224":
                self.v = timm.create_model(
                    "deit_tiny_distilled_patch16_224", pretrained=imagenet_pretrain
                )
            elif model_size == "small224":
                self.v = timm.create_model(
                    "deit_small_distilled_patch16_224", pretrained=imagenet_pretrain
                )
            elif model_size == "base224":
                self.v = timm.create_model(
                    "deit_base_distilled_patch16_224", pretrained=imagenet_pretrain
                )
            elif model_size == "base384":
                self.v = timm.create_model(
                    "deit_base_distilled_patch16_384", pretrained=imagenet_pretrain
                )
            else:
                raise Exception("Model size must be one of tiny224, small224, base224, base384.")
            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches**0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.original_embedding_dim),
                nn.Linear(self.original_embedding_dim, label_dim),
            )
            self.conv1 = nn.Conv2d(
                in_channels=3, out_channels=1, kernel_size=5, stride=1, bias=False
            )

            # Modern timm binds `embed_layer: Callable = PatchEmbed` as a
            # function-default at VisionTransformer.__init__ definition time,
            # so re-assigning the module-level `timm.models.vision_transformer
            # .PatchEmbed` name (the original repo's trick against timm 0.4.5,
            # where that name WAS looked up live) no longer swaps the instance
            # timm already built. Same fix, applied at the instance instead of
            # the class: install our shape-unrestricted PatchEmbed in place of
            # the one timm constructed, carrying over its patch/embed dims.
            _orig_patch_embed = self.v.patch_embed
            self.v.patch_embed = PatchEmbed(
                img_size=_orig_patch_embed.img_size,
                patch_size=_orig_patch_embed.patch_size,
                in_chans=3,
                embed_dim=self.original_embedding_dim,
            )

            # automatcially get the intermediate shape
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches

            # the linear projection layer
            new_proj = torch.nn.Conv2d(
                1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride)
            )
            if imagenet_pretrain == True:  # noqa: E712
                new_proj.weight = torch.nn.Parameter(
                    torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1)
                )
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            # the positional embedding
            if imagenet_pretrain == True:  # noqa: E712
                new_pos_embed = (
                    self.v.pos_embed[:, 2:, :]
                    .detach()
                    .reshape(1, self.original_num_patches, self.original_embedding_dim)
                    .transpose(1, 2)
                    .reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
                )
                if t_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[
                        :,
                        :,
                        :,
                        int(self.oringal_hw / 2) - int(t_dim / 2) : int(self.oringal_hw / 2)
                        - int(t_dim / 2)
                        + t_dim,
                    ]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(
                        new_pos_embed, size=(self.oringal_hw, t_dim), mode="bilinear"
                    )
                if f_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[
                        :,
                        :,
                        int(self.oringal_hw / 2) - int(f_dim / 2) : int(self.oringal_hw / 2)
                        - int(f_dim / 2)
                        + f_dim,
                        :,
                    ]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(
                        new_pos_embed, size=(f_dim, t_dim), mode="bilinear"
                    )
                new_pos_embed = new_pos_embed.reshape(
                    1, self.original_embedding_dim, num_patches
                ).transpose(1, 2)
                self.v.pos_embed = nn.Parameter(
                    torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1)
                )
            else:
                new_pos_embed = nn.Parameter(
                    torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim)
                )
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=0.02)

        elif audioset_pretrain == True:  # noqa: E712
            raise ValueError(
                "AudioSet-pretrained checkpoint loading requires external weight files; not supported in this vendored trace-only build."
            )

    def get_shape(self, fstride, tstride, input_fdim=50, input_tdim=120):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(
            1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride)
        )
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def forward(self, x):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :return: prediction
        """
        x = self.conv1(x)  # shape = [*, width, grid, grid]

        x = x.transpose(2, 3)

        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        feature = (x[:, 0] + x[:, 1]) / 2

        return feature, x


# ---- Event_classification/model/model_seismic_clip.py (verbatim) ----
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class FullyConnectedLayer(nn.Module):
    def __init__(self, in_features, out_features, activation=None, use_batchnorm=True):
        super().__init__()
        if activation is None:
            activation = QuickGELU()

        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation
        self.use_batchnorm = use_batchnorm

        if self.use_batchnorm:
            self.batchnorm = LayerNorm(out_features)

    def forward(self, x):
        out = self.linear(x)

        if self.use_batchnorm:
            out = self.batchnorm(out)

        out = self.activation(out)

        return out


class Info_embedding(nn.Module):
    def __init__(self, width: int, hid_feature: int, layers: int, out_dim: int):
        super().__init__()
        self.width = width
        self.hid_feature = hid_feature
        self.layers = layers
        self.out_dim = out_dim

        self.FCN_input = FullyConnectedLayer(width, hid_feature)
        self.FCN = nn.Sequential(
            *[FullyConnectedLayer(hid_feature, hid_feature) for _ in range(layers)]
        )
        self.proj = nn.Parameter(torch.randn(hid_feature, out_dim))

    def forward(self, x: torch.Tensor):
        x = self.FCN_input(x)
        x = self.FCN(x)
        if self.proj is not None:
            x = x @ self.proj
        return x


class AUDIO_CLIP(nn.Module):
    def __init__(
        self,
        device_name: str,
        embed_dim: int,
        # text
        text_input: int,
        text_width: int,
        text_layers: int,
        spec_fdim: int = 50,
        spec_tdim: int = 120,
        spec_tstr: int = 10,
        spec_fstr: int = 10,
        spec_model_size: str = "base224",
        imagenet_pretrain: bool = True,
        audioset_pretrain: bool = False,
        load_pretrain_patch: int = 120,
    ):
        super().__init__()

        self.device = device_name

        self.info = Info_embedding(
            width=text_input, hid_feature=text_width, layers=text_layers, out_dim=embed_dim
        )

        self.spec = ASTModel(
            input_fdim=spec_fdim,
            input_tdim=spec_tdim,
            tstride=spec_tstr,
            fstride=spec_fstr,
            model_size=spec_model_size,
            imagenet_pretrain=imagenet_pretrain,
            audioset_pretrain=audioset_pretrain,
            load_pretrain_patch=load_pretrain_patch,
            verbose=False,
        )

        self.logit_scale_at = torch.nn.Parameter(torch.log(torch.ones([]) * 100))

    @property
    def dtype(self):
        return self.spec.v.head.weight.dtype

    def update(self, t_dim, f_dim):
        self.spec.update_position_embed(t_dim, f_dim)

    def encode_audio(self, audio):
        feature, _ = self.spec(audio.type(self.dtype))
        return feature

    def get_audio_total_feature(self, audio):
        _, total_feature = self.spec(audio.type(self.dtype))
        return total_feature

    def encode_text(self, text):
        return self.info(text.type(self.dtype))

    def forward(self, text, audio):
        text_features = self.encode_text(text)
        audio_features = self.encode_audio(audio)

        if audio is not None:
            audio_features = self.encode_audio(audio)
            audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)

        if text is not None:
            text_features = self.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        features = (audio_features, text_features)

        logit_scale_at = torch.clamp(self.logit_scale_at.exp(), min=1.0, max=100.0)

        if (audio_features is not None) and (text_features is not None):
            logits_audio_text = logit_scale_at * audio_features @ text_features.T

        loss = self.loss_fn(logits_audio_text)

        return (features), logits_audio_text, loss

    def loss_fn(self, logits_audio_text):
        if logits_audio_text is not None:
            batch_size = logits_audio_text.shape[0]
        else:
            return None

        reference = torch.arange(batch_size, dtype=torch.int64, device=self.device)

        loss = torch.tensor(0.0, dtype=self.dtype)

        num_modalities: int = 0
        scale = torch.tensor(1.0, dtype=self.dtype)

        if logits_audio_text is not None:
            loss_at = F.cross_entropy(logits_audio_text, reference) + F.cross_entropy(
                logits_audio_text.transpose(-1, -2), reference
            )
            loss = loss + loss_at
            num_modalities += 1

        for idx in range(num_modalities):
            scale = scale * (idx + 1)

        return loss / scale

    @property
    def loss_fn_name(self) -> str:
        return "Cross Entropy"


MENAGERIE_ZOO = "vendored-pytorch"

_SPEC_TDIM = 40
_SPEC_FDIM = 50


def build_seisclip():
    torch.manual_seed(0)
    # embed_dim must equal the AST/deit variant's hidden size: encode_audio()
    # returns the raw ViT feature (dim = original_embedding_dim) with no
    # projection head applied in forward()/encode_audio(), so the audio and
    # text branches only line up for a logit_scale_at @ text_features.T
    # matmul when Info_embedding's out_dim (embed_dim) matches the deit
    # variant's embedding width -- exactly how the real repo's own
    # `test.py` picks them (`spec_model_size='small224'` with
    # `embed_dim=384`, since deit-small's hidden size is 384). Here we use
    # `tiny224`, whose hidden size is 192.
    model = AUDIO_CLIP(
        device_name="cpu",
        embed_dim=192,
        text_input=8,
        text_width=16,
        text_layers=1,
        spec_fdim=_SPEC_FDIM,
        spec_tdim=_SPEC_TDIM,
        spec_tstr=10,
        spec_fstr=10,
        spec_model_size="tiny224",
        imagenet_pretrain=False,
        audioset_pretrain=False,
    )
    model.eval()
    return model


def example_input_seisclip():
    torch.manual_seed(0)
    text = torch.rand(2, 8)
    # ASTModel.conv1 is nn.Conv2d(in_channels=3, ...): the spectrogram is fed
    # as a 3-channel "image" (batch, 3, freq, time) so the AST can reuse an
    # ImageNet-pretrained ViT patch embedding, matching real usage in
    # Event_classification/test.py (`finetune_model.encode_audio(batch[2])`
    # where batch[2] is the pre-stacked 3-channel spectrogram).
    audio = torch.rand(2, 3, _SPEC_FDIM, _SPEC_TDIM)
    return (text, audio)


MENAGERIE_ENTRIES = [
    ("SeisCLIP", "build_seisclip", "example_input_seisclip", 2023, MENAGERIE_ZOO),
]
