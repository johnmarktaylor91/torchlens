# SOURCE: vendored from Meituan-AutoML/MobileVLM @ 688fdec914810485c8766da96c63d9d2ce15f750
# https://github.com/Meituan-AutoML/MobileVLM/blob/main/mobilevlm/model/mobilellama.py
# https://github.com/Meituan-AutoML/MobileVLM/blob/main/mobilevlm/model/mobilevlm.py
# https://github.com/Meituan-AutoML/MobileVLM/blob/main/mobilevlm/model/vision_encoder.py
# https://github.com/Meituan-AutoML/MobileVLM/blob/main/mobilevlm/model/vision_projector.py
#
# MobileVLM (arXiv:2312.16886 V1, arXiv:2402.03766 V2) is a mobile-scale
# vision-language model: a CLIP ViT vision tower feeds a Lightweight Downsample
# Projector (LDP / LDPv2 -- a real depthwise-separable "MobileNetV3 inverted-residual"
# token compressor, not a plain MLP) into a LLaMA-family causal LM
# (MobileLLaMA, `MobileLlamaForCausalLM` in the real repo). This module preserves the
# real architecture of every piece:
#   - `CLIPVisionTower` (vision_encoder.py): thin real wrapper around HF
#     `transformers.CLIPVisionModel`, with the real `feature_select` hidden-state-layer
#     selection logic. The upstream class calls `CLIPVisionModel.from_pretrained(...)`
#     (network + a real published checkpoint); here the *same* `CLIPVisionModel` class is
#     built from a small random `CLIPVisionConfig` instead, so the module is
#     self-contained and offline, with no architectural change.
#   - `FeatureIRLayer` / `TokenDownLayer` / `PosInjectLayer` / `LDPNetV2Projector`
#     (vision_projector.py): the real V2 lightweight downsample projector, verbatim.
#   - `MobileLlamaForCausalLM` (mobilellama.py): the real repo class is
#     `LlamaForCausalLM` + `MobileVLMMetaForCausalLM` (the multimodal mixin that
#     concatenates projected image tokens with text embeddings before the LLaMA
#     decoder stack) with zero architectural modification to `LlamaModel` itself; here
#     we build the stock HF `transformers.LlamaForCausalLM` at a tiny `LlamaConfig` and
#     reproduce the real `encode_images` + embed-and-concatenate wiring from
#     `MobileVLMMetaForCausalLM.encode_images` / `prepare_inputs_labels_for_multimodal`
#     (simplified to the common case: an <image> placeholder replaced by projected
#     image-feature tokens prepended to the text embedding sequence, matching the real
#     forward-path semantics without the training-time label/attention-mask bookkeeping
#     that is irrelevant to a forward trace).
# Needs a MODULE (not a recipe) because the real forward pass takes two tensor inputs:
# `input_ids` (text token ids) and `images` (pixel tensor).

import math

import torch
import torch.nn as nn
from transformers import CLIPVisionConfig, CLIPVisionModel, LlamaConfig, LlamaForCausalLM

MENAGERIE_ZOO = "vendored-pytorch"

IMAGE_TOKEN_INDEX = -200
IGNORE_INDEX = -100


# ========== vision_encoder.py (CLIPVisionTower, real class, offline tiny config) ==========
class CLIPVisionTower(nn.Module):
    def __init__(self, vision_config, select_layer=-1, select_feature="patch"):
        super().__init__()
        self.vision_tower = CLIPVisionModel(vision_config)
        self.select_layer = select_layer
        self.select_feature = select_feature
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    def forward(self, images):
        image_forward_outs = self.vision_tower(images, output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @property
    def hidden_size(self):
        return self.vision_tower.config.hidden_size


# ========== vision_projector.py (real LDPNetV2Projector, verbatim mechanism) ==========
class FeatureIRLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class TokenDownLayer(nn.Module):
    def __init__(self, shape) -> None:
        super().__init__()
        self.dwn = nn.Sequential(nn.AdaptiveAvgPool2d(shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        assert h * h == num_tokens
        x = x.permute(0, 2, 1).reshape(b, -1, h, h)
        x = self.dwn(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class PosInjectLayer(nn.Module):
    # https://github.com/Meituan-AutoML/Twins/blob/main/gvt.py
    def __init__(self, in_dim: int, out_dim: int, stride: int = 1) -> None:
        super().__init__()
        self.peg = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride, 1, bias=True, groups=out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        assert h * h == num_tokens
        cnn_feat = x.transpose(1, 2).view(b, c, h, h)
        x = self.peg(cnn_feat) + cnn_feat
        x = x.flatten(2).transpose(1, 2)
        return x


class LDPNetV2Projector(nn.Module):
    def __init__(self, mm_hidden_size, hidden_size, down_shape=(12, 12)):
        super().__init__()
        self.mlp = FeatureIRLayer(mm_hidden_size, hidden_size)
        self.dwn = TokenDownLayer(down_shape)
        self.peg = PosInjectLayer(hidden_size, hidden_size, stride=1)

    def forward(self, x):
        x = self.mlp(x)
        x = self.dwn(x)
        x = self.peg(x)
        return x


# ========== mobilellama.py + mobilevlm.py (real MobileLlamaForCausalLM wiring) ==========
class MobileLlamaForCausalLM(nn.Module):
    """
    Real MobileVLM multimodal wiring: CLIPVisionTower -> LDPNetV2Projector -> LLaMA LM.
    `self.language_model` is the real, unmodified `transformers.LlamaForCausalLM`
    (mobilellama.py: `MobileLlamaForCausalLM(LlamaForCausalLM, MobileVLMMetaForCausalLM)`
    subclasses it directly with no architectural change to the decoder). `encode_images`
    and `forward` reproduce the real image-token-concatenation path from
    `MobileVLMMetaForCausalLM.encode_images` / `prepare_inputs_labels_for_multimodal`.
    """

    def __init__(self, vision_config, llama_config, mm_down_shape=(2, 2)):
        super().__init__()
        self.vision_tower = CLIPVisionTower(vision_config, select_layer=-1, select_feature="patch")
        self.mm_projector = LDPNetV2Projector(
            mm_hidden_size=vision_config.hidden_size,
            hidden_size=llama_config.hidden_size,
            down_shape=mm_down_shape,
        )
        self.language_model = LlamaForCausalLM(llama_config)

    def encode_images(self, images):
        image_features = self.vision_tower(images)
        image_features = self.mm_projector(image_features)
        return image_features

    def forward(self, input_ids, images):
        # real repo behavior (prepare_inputs_labels_for_multimodal, common case): embed
        # the projected image-feature tokens and the text tokens, then concatenate them
        # into a single embedding sequence consumed by the LLaMA decoder.
        image_features = self.encode_images(images)
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        combined_embeds = torch.cat([image_features, inputs_embeds], dim=1)
        outputs = self.language_model(inputs_embeds=combined_embeds)
        return outputs.logits


def build_mobilevlm():
    vision_config = CLIPVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=32,
        patch_size=16,
    )
    llama_config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    model = MobileLlamaForCausalLM(vision_config, llama_config, mm_down_shape=(2, 2))
    return model.eval()


def example_input_mobilevlm():
    input_ids = torch.randint(0, 64, (1, 6))
    images = torch.randn(1, 3, 32, 32)
    return [input_ids, images]


MENAGERIE_ENTRIES = [
    ("MobileVLM", build_mobilevlm, example_input_mobilevlm, 2023, MENAGERIE_ZOO),
]
