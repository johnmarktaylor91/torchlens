# SOURCE: vendored from TinyLLaVA/TinyLLaVA_Factory @ main
#
# TinyLLaVA: A Framework of Small-scale Large Multimodal Models
# Baichuan Zhou, Ying Hu, Xi Weng, Junlong Jia, Jiawei Luo, Xien Liu, Ji Wu, Lei Huang.
# https://github.com/TinyLLaVA/TinyLLaVA_Factory  (arXiv:2402.14289)
#
# TinyLLaVA is a real fusion architecture (not a base-lib class): a frozen/tunable
# vision tower (CLIP/SigLIP/DINOv2) + a lightweight connector (MLP/linear/qformer/
# resampler) that projects vision-tower patch features into the LLM's embedding
# space, glued to a small LLM (Phi/StableLM/TinyLlama/Qwen2/Gemma/OpenELM) via
# image-token splicing. This file vendors the REAL fusion logic verbatim from:
#   - tinyllava/model/modeling_tinyllava.py
#       (TinyLlavaPreTrainedModel, TinyLlavaForConditionalGeneration: forward,
#        encode_images, prepare_inputs_labels_for_multimodal)
#   - tinyllava/model/vision_tower/base.py, vision_tower/clip.py
#       (VisionTower, CLIPVisionTower)
#   - tinyllava/model/connector/base.py, connector/mlp.py
#       (Connector, MLPConnector)
#   - tinyllava/utils/constants.py
#       (IGNORE_INDEX, IMAGE_TOKEN_INDEX)
# Only imports/module layout were touched to make this self-contained (no repo
# package, no HF-hub network calls at build time): the real TinyLlavaConfig's
# `_load_text_config`/`_load_vision_config` call `AutoConfig.from_pretrained(...)`
# to fetch remote configs by name; here we construct the identical real
# `transformers.PhiConfig` / `transformers.CLIPVisionConfig` objects locally
# (shrunk for fast tracing) and pass them straight into the real
# `PhiForCausalLM` / `CLIPVisionModel` / `MLPConnector` classes exactly as the
# real `TinyLlavaForConditionalGeneration.__init__` does (`LLMFactory(...)[0](config.text_config)`,
# `VisionTowerFactory(...)(config.vision_config)`, `ConnectorFactory(...)(config)`).
# The LLM/vision-tower/connector factory dispatch tables themselves are omitted
# (they just `if name in model_name_or_path.lower(): return <class>`); this
# module directly instantiates the real classes the "phi" / "clip" / "mlp"
# factory entries return (see tinyllava/model/llm/phi.py, vision_tower/clip.py,
# connector/mlp.py), which is the flagship TinyLLaVA recipe (Phi-2 + CLIP +
# 2-layer MLP connector) named in the paper.
# MENAGERIE_ZOO = "vendored-pytorch"

import re
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    CLIPVisionConfig,
    CLIPVisionModel,
    PhiConfig,
    PhiForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# vendored from tinyllava/model/configuration_tinyllava.py (TinyLlavaConfig),
# trimmed to the fields TinyLlavaForConditionalGeneration/vision_tower/connector
# actually read. The real class's `_load_text_config`/`_load_vision_config`
# call `AutoConfig.from_pretrained(<hub name>)`; here `text_config`/
# `vision_config` are passed in directly as real transformers config objects
# (exactly what those methods produce once fetched), so no network/hub access
# is needed to build a real, tiny, randomly-initialized instance.
# ---------------------------------------------------------------------------
class TinyLlavaConfig(PretrainedConfig):
    model_type = "tinyllava"

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        connector_type="mlp2x_gelu",
        vision_feature_layer=-2,
        vision_feature_select_strategy="patch",
        tokenizer_model_max_length=64,
        tokenizer_padding_side="right",
        use_cache=False,
        **kwargs,
    ):
        self.text_config = text_config
        self.vision_config = vision_config
        self.connector_type = connector_type
        self.vision_feature_layer = vision_feature_layer
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.tokenizer_model_max_length = tokenizer_model_max_length
        self.tokenizer_padding_side = tokenizer_padding_side
        self.use_cache = use_cache
        self.hidden_size = getattr(text_config, "hidden_size", None)
        self.vocab_size = getattr(text_config, "vocab_size", None)
        self.vision_hidden_size = getattr(vision_config, "hidden_size", None)
        super().__init__(**kwargs)


# ---------------------------------------------------------------------------
# vendored from tinyllava/utils/constants.py
# ---------------------------------------------------------------------------
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200


# ---------------------------------------------------------------------------
# vendored from tinyllava/model/connector/base.py + connector/mlp.py
# ---------------------------------------------------------------------------
ACT_TYPE = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
}


class Connector(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self._connector = None

    def forward(self, x):
        return self._connector(x)


class MLPConnector(Connector):
    def __init__(self, config):
        super().__init__()

        mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", config.connector_type)
        act_type = config.connector_type.split("_")[-1]
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.vision_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(ACT_TYPE[act_type]())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))

        self._connector = nn.Sequential(*modules)


# ---------------------------------------------------------------------------
# vendored from tinyllava/model/vision_tower/base.py + vision_tower/clip.py
# ---------------------------------------------------------------------------
class VisionTower(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._vision_tower = None
        self._image_processor = None
        self.config = cfg

    def forward(self, x, **kwargs):
        image_features = self._vision_tower(x, output_hidden_states=True)
        image_features = image_features.hidden_states[kwargs.get("vision_feature_layer", -2)]

        select_strategy = kwargs.get("vision_feature_select_strategy", "patch")
        if select_strategy == "patch":
            image_features = image_features[:, 1:]
        elif select_strategy == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {select_strategy}")

        return image_features

    @property
    def vision_tower(self):
        return self._vision_tower

    @vision_tower.setter
    def vision_tower(self, vision_tower):
        self._vision_tower = vision_tower


class CLIPVisionTower(VisionTower):
    def __init__(self, cfg):
        super().__init__(cfg)
        # real repo calls CLIPImageProcessor.from_pretrained(cfg.model_name_or_path)
        # here too (network call, preprocessing-only -- no effect on the traced
        # nn.Module graph), so it is omitted for a self-contained build.
        self._vision_tower = CLIPVisionModel(cfg)


# ---------------------------------------------------------------------------
# vendored from tinyllava/model/modeling_tinyllava.py
# ---------------------------------------------------------------------------
class TinyLlavaPreTrainedModel(PreTrainedModel):
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlavaVisionAttention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    # real repo: `@property def _supports_sdpa(self): return self.language_model._supports_sdpa`.
    # In the transformers version pinned here, `PreTrainedModel.__init__` now
    # resolves `_supports_sdpa` (via `_check_and_adjust_attn_implementation`)
    # BEFORE the subclass `__init__` has set `self.language_model`, so the real
    # property crashes with AttributeError on construction. Phi (the real
    # language_model class used below) supports SDPA, so the property's real
    # runtime answer is always True; pinned here as a plain bool to keep the
    # same effective attention-implementation resolution without depending on
    # `self.language_model` existing during `super().__init__`.
    _supports_sdpa = True


class TinyLlavaForConditionalGeneration(TinyLlavaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.language_model = PhiForCausalLM(config.text_config)
        self.vision_tower = CLIPVisionTower(config.vision_config)
        self.connector = MLPConnector(config)

        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
            )
        return self.language_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def encode_images(self, images):
        kwargs = {}
        kwargs["vision_feature_layer"] = self.config.vision_feature_layer
        kwargs["vision_feature_select_strategy"] = self.config.vision_feature_select_strategy
        images = images.to(device=self.device, dtype=self.dtype)
        image_features = self.vision_tower(images, **kwargs)
        image_features = self.connector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        image_sizes=None,
    ):
        vision_tower = self.vision_tower
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        image_features = self.encode_images(images)

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        _input_ids = input_ids
        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask)
        ]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.language_model.get_input_embeddings()(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = (
                [-1]
                + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
                + [cur_input_ids.shape[0]]
            )
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(
                    cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]]
                )
                cur_labels_noim.append(
                    cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]]
                )
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.language_model.get_input_embeddings()(
                torch.cat(cur_input_ids_noim)
            )
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device
        )
        position_ids = torch.zeros(
            (batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device
        )

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels


# ---------------------------------------------------------------------------
# staging: build a tiny flagship-recipe TinyLLaVA (Phi LLM + CLIP vision tower +
# 2-layer MLP connector, "mlp2x_gelu" -- the connector type used in the paper's
# best-performing runs) directly from the real vendored classes above, with a
# small hand-built config namespace instead of TinyLlavaConfig's HF-hub network
# fetch (config.text_config / config.vision_config are real transformers
# PhiConfig / CLIPVisionConfig objects, exactly as TinyLlavaConfig._load_text_config
# / _load_vision_config would build when given real "microsoft/phi-2" /
# "openai/clip-vit-large-patch14-336" model names).
# ---------------------------------------------------------------------------
def build_tinyllava():
    text_config = PhiConfig(
        vocab_size=200,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    vision_config = CLIPVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=32,
        patch_size=16,
        projection_dim=16,
    )

    config = TinyLlavaConfig(
        text_config=text_config,
        vision_config=vision_config,
        connector_type="mlp2x_gelu",
        vision_feature_layer=-2,
        vision_feature_select_strategy="patch",
        tokenizer_model_max_length=64,
        tokenizer_padding_side="right",
        use_cache=False,
        initializer_range=0.02,
    )

    model = TinyLlavaForConditionalGeneration(config)
    model.eval()
    return model


def example_input_tinyllava():
    # 1 image-patch token (IMAGE_TOKEN_INDEX) + 5 real text tokens; one image
    # per example (matches CLIPVisionConfig(image_size=32, patch_size=16) => 2x2=4
    # patches, so image_features has 4 rows spliced in for the 1 image token;
    # select_strategy="patch" drops CLS from CLIP's patch embeddings).
    # Returned as a positional tuple matching
    # TinyLlavaForConditionalGeneration.forward(input_ids, attention_mask,
    # position_ids, past_key_values, inputs_embeds, labels, use_cache,
    # output_attentions, output_hidden_states, images, ...) so it traces via
    # plain positional `model(*input_args)` dispatch (None for the unused
    # positional slots between input_ids and images).
    input_ids = torch.tensor([[5, 6, IMAGE_TOKEN_INDEX, 7, 8, 9]], dtype=torch.long)
    images = torch.randn(1, 3, 32, 32)
    return (input_ids, None, None, None, None, None, None, None, None, images)


MENAGERIE_ENTRIES = [
    (
        "TinyLLaVA",
        "build_tinyllava",
        "example_input_tinyllava",
        2024,
        MENAGERIE_ZOO,
    ),
]
