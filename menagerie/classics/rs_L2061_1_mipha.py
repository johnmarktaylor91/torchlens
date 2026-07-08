# SOURCE: vendored from zhuyiche/llava-phi @ main (mipha/model/mipha_arch.py,
# mipha/model/language_model/{configuration_mipha.py,mipha_phi.py}, mipha/model/multimodal_encoder/
# clip_encoder.py, mipha/model/multimodal_projector/builder.py), the Mipha architecture
# ("Mipha: A Comprehensive Overhaul of Multimodal Assistant with Small Language Models",
# Zhu, Zhu, Liu, He, Sun, Li, Fan, Ge, arXiv:2403.06199). Mipha lives in the same GitHub repo as
# LLaVA-Phi (same authors, same repo, distinct top-level `mipha/` package) but is architecturally
# its own vision-language model: it supports a swappable vision tower (CLIP / SigLIP / DINOv2,
# selected by name at construction) feeding a configurable projector (linear / mlpNx_gelu /
# identity) into a Phi causal LM, wired together through `MiphaMetaModel`/`MiphaMetaForCausalLM`
# mixins that splice projected image features into the token-embedding sequence before the LM
# decoder runs (`prepare_inputs_labels_for_multimodal`). This is a distinct multimodal-fusion
# design from LLaVA-Phi's single CLIP-only encoder + fixed MLP projector wiring (llava_phi/model/
# llava_arch.py), not the same architecture under a new name. Imports/relative paths adjusted
# minimally for standalone staging (module reorganized into one file, `mipha.constants` inlined,
# HF `PhiModel`/`PhiPreTrainedModel`/`PhiConfig` used directly instead of the repo's vendored
# phi1_5/ shim which duplicates the same upstream Phi architecture); the vision tower / projector
# / LM fusion architecture itself is untouched.

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (
    CLIPPreTrainedModel,
    CLIPVisionConfig,
    PhiConfig,
    PhiModel,
    PhiPreTrainedModel,
    PretrainedConfig,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.clip.modeling_clip import CLIPVisionTransformer

# ============================================================================
# mipha/constants.py (inlined)
# ============================================================================

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


# ============================================================================
# mipha/model/language_model/configuration_mipha.py
# ============================================================================


class MiphaVisionConfig(PretrainedConfig):
    model_type = "mipha_vision_model"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        projection_dim=512,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=32,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        mm_vision_select_feature="patch",
        mm_vision_select_layer=-2,
        vision_model_name_or_path="clip",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.mm_vision_select_feature = mm_vision_select_feature
        self.mm_vision_select_layer = mm_vision_select_layer
        self.vision_model_name_or_path = vision_model_name_or_path


class ProjectorConfig(PretrainedConfig):
    model_type = "mipha_projector"

    def __init__(self, mm_projector_type="linear", mm_hidden_size=768, hidden_size=2560, **kwargs):
        self.mm_projector_type = mm_projector_type
        self.mm_hidden_size = mm_hidden_size
        self.hidden_size = hidden_size
        super().__init__(**kwargs)


DEFAULT_VISUAL_CONFIG = {
    "vision_tower": MiphaVisionConfig().to_dict(),
    "mm_projector": ProjectorConfig().to_dict(),
}


class MiphaPhiConfig(PhiConfig):
    model_type = "mipha_phi"

    def __init__(self, vision_config=None, **kwargs):
        if vision_config is None:
            self.vision_config = DEFAULT_VISUAL_CONFIG
        else:
            self.vision_config = vision_config

        super().__init__(**kwargs)


# ============================================================================
# mipha/model/multimodal_encoder/clip_encoder.py
# ============================================================================


class CLIPVisionTower(CLIPPreTrainedModel):
    config_class = MiphaVisionConfig

    def __init__(self, config):
        super().__init__(config)

        self.vision_model = CLIPVisionTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.config.mm_vision_select_layer]
        if self.config.mm_vision_select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.config.mm_vision_select_feature == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {self.config.mm_vision_select_feature}")
        return image_features

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_model(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_model(
                images.to(device=self.device, dtype=self.dtype), output_hidden_states=True
            )
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return list(self.vision_model.parameters())[0].dtype

    @property
    def device(self):
        return list(self.vision_model.parameters())[0].device

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


# ============================================================================
# mipha/model/multimodal_projector/builder.py
# ============================================================================


def build_vision_projector(config):
    projector_type = getattr(config, "mm_projector_type", "linear")

    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    raise ValueError(f"Unknown projector type: {projector_type}")


# ============================================================================
# mipha/model/mipha_arch.py
# ============================================================================


class MiphaMetaModel:
    def __init__(self, config):
        super(MiphaMetaModel, self).__init__(config)
        vision_name = config.vision_config["vision_tower"]["vision_model_name_or_path"]
        if "clip" in vision_name:
            self.vision_tower = CLIPVisionTower(
                MiphaVisionConfig(**config.vision_config["vision_tower"])
            )
        else:
            raise ValueError("Vision model name or path should contain either 'clip' or 'siglip'")

        self.mm_projector = build_vision_projector(
            ProjectorConfig(**config.vision_config["mm_projector"])
        )

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower


class MiphaMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                half_len = cur_input_ids.shape[0] // 2
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat(
                    [cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0
                )
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                cur_new_input_embeds.append(
                    self.get_model().embed_tokens(cur_input_ids[:image_token_start])
                )
                cur_new_input_embeds.append(cur_image_features)
                if labels is not None:
                    cur_new_labels.append(cur_labels[:image_token_start])
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=labels.device,
                            dtype=labels.dtype,
                        )
                    )
                    cur_labels = cur_labels[image_token_start + 1 :]
                cur_image_idx += 1
                cur_input_ids = cur_input_ids[image_token_start + 1 :]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat(
                    (
                        cur_new_embed,
                        torch.zeros(
                            (max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                            dtype=cur_new_embed.dtype,
                            device=cur_new_embed.device,
                        ),
                    ),
                    dim=0,
                )
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat(
                        (
                            cur_new_label,
                            torch.full(
                                (max_len - cur_new_label.shape[0],),
                                IGNORE_INDEX,
                                dtype=cur_new_label.dtype,
                                device=cur_new_label.device,
                            ),
                        ),
                        dim=0,
                    )
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(
                    attention_mask, _new_labels, new_labels
                ):
                    new_attn_mask_pad_left = torch.full(
                        (cur_new_labels.shape[0] - labels.shape[1],),
                        True,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    new_attn_mask_pad_right = torch.full(
                        (cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                        False,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    cur_new_attention_mask = torch.cat(
                        (new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0
                    )
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)

        return None, attention_mask, past_key_values, new_input_embeds, new_labels


# ============================================================================
# mipha/model/language_model/mipha_phi.py
# ============================================================================


class MiphaPhiModel(MiphaMetaModel, PhiModel):
    config_class = MiphaPhiConfig

    def __init__(self, config):
        super(MiphaPhiModel, self).__init__(config)


class MiphaPhiForCausalLM(PhiPreTrainedModel, MiphaMetaForCausalLM):
    config_class = MiphaPhiConfig
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super(PhiPreTrainedModel, self).__init__(config)
        self.model = MiphaPhiModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = (
            self.prepare_inputs_labels_for_multimodal(
                input_ids, attention_mask, past_key_values, labels, images
            )
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# ============================================================================
# Menagerie staging entry points
# ============================================================================
#
# Tiny-size Mipha-Phi VLM: a small CLIP vision tower feeds a linear projector into a small Phi
# causal LM, matching the real repo's swappable-tower + swappable-projector + Phi-fusion design
# at reduced scale. Random init, eval() mode (Dropout/LayerNorm determinism for tracing).

MENAGERIE_ZOO = "vendored-pytorch"


def build_mipha():
    import torch

    torch.manual_seed(0)

    vision_config = MiphaVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=32,
        patch_size=16,
        vision_model_name_or_path="clip",
    )
    projector_config = ProjectorConfig(
        mm_projector_type="linear", mm_hidden_size=32, hidden_size=48
    )
    vision_cfg_dict = {
        "vision_tower": vision_config.to_dict(),
        "mm_projector": projector_config.to_dict(),
    }

    config = MiphaPhiConfig(
        vision_config=vision_cfg_dict,
        vocab_size=128,
        hidden_size=48,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=64,
    )
    model = MiphaPhiForCausalLM(config)
    model.eval()
    return model


def example_input_mipha():
    import torch

    torch.manual_seed(0)
    input_ids = torch.randint(0, 128, (1, 6))
    # Splice one IMAGE_TOKEN_INDEX in so the multimodal fusion path (vision tower + projector)
    # actually runs during trace, matching real usage (image placeholder token in the prompt).
    input_ids[0, 2] = IMAGE_TOKEN_INDEX
    images = torch.randn(1, 3, 32, 32)
    # Positional order matches MiphaPhiForCausalLM.forward:
    # (input_ids, attention_mask, past_key_values, inputs_embeds, labels, use_cache,
    #  output_attentions, output_hidden_states, images)
    return (input_ids, None, None, None, None, None, None, None, images)


MENAGERIE_ENTRIES = [
    ("Mipha", "build_mipha", "example_input_mipha", 2024, "vendored-pytorch"),
]
