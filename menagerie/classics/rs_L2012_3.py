# SOURCE: vendored from https://github.com/zhuyiche/llava-phi @ main
#
# LLaVA-Phi (Zhu, Zhu, Liu, Liu, Zeng, "LLaVA-Phi: Efficient Multi-Modal Assistant with
# Small Language Model", arXiv:2401.02330): the repo (`llava_phi/model/language_model/
# llava_phi.py`, `llava_phi/model/llava_arch.py`, `llava_phi/model/multimodal_encoder/
# clip_encoder.py`, `llava_phi/model/multimodal_projector/builder.py`) contains NO novel
# architecture code of its own -- `LlavaPhiForCausalLM` is `PhiPreTrainedModel` +
# `LLavaPhiModel(LlavaMetaModel, PhiModel)` (transformers' real, unmodified Phi-2 decoder),
# `CLIPVisionTower` is `CLIPPreTrainedModel` wrapping transformers' real, unmodified
# `CLIPVisionTransformer`, and `build_vision_projector` returns a plain `nn.Linear` (or
# small GELU-MLP) for the `mm_projector_type='linear'` default. The paper's contribution is
# the CLIP-ViT + linear-projector + Phi-2 LLaVA *recipe* (data/training-objective/usage),
# not a new module -- this is exactly the "contribution is data/objective/usage only"
# rung-1 case (like DialoGPT=GPT2), so it is staged here as a thin composition of the real
# `transformers` classes rather than a from-scratch reimplementation. It is staged as a
# MODULE (not a catalog recipe row) because the real forward pass needs two distinct
# tensor inputs (`input_ids` and `images`), which a single-tensor recipe row cannot express.
#
# Repo: https://github.com/zhuyiche/llava-phi @ main
# Files: llava_phi/model/language_model/llava_phi.py, llava_phi/model/llava_arch.py,
#        llava_phi/model/multimodal_encoder/clip_encoder.py,
#        llava_phi/model/multimodal_projector/builder.py

import torch
import torch.nn as nn
from transformers import CLIPVisionConfig, PhiConfig, PhiModel
from transformers.models.clip.modeling_clip import CLIPVisionTransformer

MENAGERIE_ZOO = "vendored-pytorch"

IMAGE_TOKEN_INDEX = -200


class CLIPVisionTower(nn.Module):
    """Faithful copy of `llava_phi/model/multimodal_encoder/clip_encoder.py`'s
    `CLIPVisionTower`, minus the `CLIPPreTrainedModel`/`post_init()` HF-checkpoint
    bookkeeping (irrelevant to the forward architecture): wraps the real
    `CLIPVisionTransformer`, selecting a hidden-state layer and dropping the CLS token
    ("patch" feature select, the repo's default) exactly as the original.
    """

    def __init__(
        self, config: CLIPVisionConfig, mm_vision_select_layer=-2, mm_vision_select_feature="patch"
    ):
        super().__init__()
        self.vision_model = CLIPVisionTransformer(config)
        self.mm_vision_select_layer = mm_vision_select_layer
        self.mm_vision_select_feature = mm_vision_select_feature

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.mm_vision_select_layer]
        if self.mm_vision_select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.mm_vision_select_feature == "cls_patch":
            pass
        else:
            raise ValueError(f"Unexpected select feature: {self.mm_vision_select_feature}")
        return image_features

    def forward(self, images):
        image_forward_outs = self.vision_model(images, output_hidden_states=True)
        return self.feature_select(image_forward_outs)


def build_vision_projector(mm_hidden_size, hidden_size, projector_type="linear"):
    """Faithful copy of `llava_phi/model/multimodal_projector/builder.py`'s
    `build_vision_projector` at the repo's default `mm_projector_type='linear'`.
    """
    if projector_type == "linear":
        return nn.Linear(mm_hidden_size, hidden_size)
    raise ValueError(f"Unknown projector type: {projector_type}")


class LlavaPhiForCausalLM(nn.Module):
    """Faithful copy of `LlavaPhiForCausalLM` / `LlavaMetaModel` / `LlavaMetaForCausalLM`'s
    multimodal composition: real `PhiModel` (transformers' unmodified Phi-2 decoder) +
    real `CLIPVisionTransformer` vision tower + linear projector, with image-patch
    features spliced into the token embedding sequence at the `IMAGE_TOKEN_INDEX`
    placeholder position (the repo's `prepare_inputs_labels_for_multimodal`, simplified to
    the single-image-token-per-sample, no-`past_key_values` case that a plain forward
    pass takes) before running the real Phi-2 decoder + `lm_head`.
    """

    def __init__(
        self, phi_config: PhiConfig, vision_config: CLIPVisionConfig, mm_hidden_size, hidden_size
    ):
        super().__init__()
        self.phi_model = PhiModel(phi_config)
        self.lm_head = nn.Linear(phi_config.hidden_size, phi_config.vocab_size, bias=True)
        self.vision_tower = CLIPVisionTower(vision_config)
        self.mm_projector = build_vision_projector(mm_hidden_size, hidden_size)

    def encode_images(self, images):
        image_features = self.vision_tower(images)
        image_features = self.mm_projector(image_features)
        return image_features

    def prepare_inputs_embeds_for_multimodal(self, input_ids, images):
        """Simplified faithful port of `LlavaMetaForCausalLM.
        prepare_inputs_labels_for_multimodal` for the plain (non-incremental, every sample
        multimodal, single image-token span per sample) forward-pass case: splice the
        projected image-patch features in at the `IMAGE_TOKEN_INDEX` placeholder position of
        each sample's token-embedding sequence, exactly as the original per-sample loop.
        """
        image_features = self.encode_images(images)  # [batch, n_patches, hidden_size]

        embed_tokens = self.phi_model.get_input_embeddings()
        new_input_embeds = []
        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_image_features = image_features[batch_idx]
            image_token_start = (cur_input_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0][0]
            pre = embed_tokens(cur_input_ids[:image_token_start])
            post = embed_tokens(cur_input_ids[image_token_start + 1 :])
            cur_new_input_embeds = torch.cat([pre, cur_image_features, post], dim=0)
            new_input_embeds.append(cur_new_input_embeds)
        return torch.stack(new_input_embeds, dim=0)

    def forward(self, input_ids, images):
        inputs_embeds = self.prepare_inputs_embeds_for_multimodal(input_ids, images)
        outputs = self.phi_model(inputs_embeds=inputs_embeds)
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        return logits


def build_llava_phi():
    # Real defaults per configuration_llava_phi.py's LlavaPhiVisionConfig / ProjectorConfig
    # (CLIP ViT hidden_size=768, projector hidden_size=2560 to match Phi-2's hidden_size)
    # and PhiConfig's own published defaults, shrunk to a tiny size for a fast trace;
    # architecture (real CLIPVisionTransformer + real PhiModel + linear projector,
    # image-token splice) unchanged.
    phi_config = PhiConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=64,
        attn_implementation="eager",
    )
    vision_config = CLIPVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        projection_dim=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=32,
        patch_size=16,
        attn_implementation="eager",
    )
    return LlavaPhiForCausalLM(phi_config, vision_config, mm_hidden_size=16, hidden_size=32)


def example_input_llava_phi():
    batch, seq_len, vocab, image_size = 2, 6, 128, 32
    input_ids = torch.randint(1, vocab, (batch, seq_len))
    # place the image-token placeholder at a fixed position in every sample
    input_ids[:, 2] = IMAGE_TOKEN_INDEX
    images = torch.randn(batch, 3, image_size, image_size)
    return (input_ids, images)


MENAGERIE_ENTRIES = [
    (
        "LLaVA-Phi (CLIP ViT + linear projector + Phi-2)",
        build_llava_phi,
        example_input_llava_phi,
        2024,
        MENAGERIE_ZOO,
    ),
]
