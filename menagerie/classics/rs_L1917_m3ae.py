# SOURCE: vendored from zhjohnchan/M3AE @ master
# (m3ae/modules/vision_encoders/clip_model.py: `LayerNorm`, `QuickGELU`,
#  `ResidualAttentionBlock`, `Transformer`, `VisualTransformer`, `CLIP`;
#  m3ae/modules/language_encoders/bert_model.py: `BertSelfAttention`,
#  `BertSelfOutput`, `BertAttention`, `BertIntermediate`, `BertOutput`,
#  `BertCrossLayer`; m3ae/modules/prediction_heads.py: `Pooler`;
#  m3ae/modules/m3ae_utils.py: `init_weights`; m3ae/modules/m3ae_module.py:
#  `M3AETransformerSS.__init__`/`infer` multimodal-fusion forward flow).
# M3AE (Multi-Modal Masked Autoencoder), Chen et al. 2022, "Multi-Modal
# Masked Autoencoders for Medical Vision-and-Language Pre-Training" (MICCAI
# 2022). The real architecture is: a from-scratch CLIP-style ViT vision
# encoder (M3AE's own `CLIP`/`VisualTransformer`, NOT the `clip` pip package
# -- self-contained `nn.MultiheadAttention` transformer blocks) + a BERT text
# encoder (M3AE uses HuggingFace `transformers.BertModel` directly, per the
# original `M3AETransformerSS.__init__`: `self.language_encoder =
# BertModel.from_pretrained(...)`; we construct the real class random-init
# instead of `from_pretrained`), fused via stacked co-attention
# `BertCrossLayer` blocks (M3AE's actual novel contribution: each layer runs
# self-attention then cross-attention onto the other modality, alternating
# text->image and image->text), pooled and concatenated into
# `multi_modal_cls_feats`.
#
# The source's `M3AETransformerSS` subclasses `pytorch_lightning.LightningModule`
# purely for the Lightning training loop (`training_step`/`configure_optimizers`/
# checkpoint loading via `self.hparams`/`save_hyperparameters`) -- none of
# that is architectural. `M3AEForStaging` below is a plain `nn.Module` with
# the exact same `__init__` model construction and the exact same `infer()`
# multimodal fusion forward body (mask_text=False, mask_image=False,
# output_attentions=False path; masking/attention-return branches and the
# pretraining/downstream task heads in `forward()` are training/task-specific
# extensions of `infer()`'s core encode-fuse-pool graph and are not part of
# the traced representation). All dims below are shrunk from the paper's
# defaults (hidden_size=768, 6 top layers, image_size=224/384) for fast
# tracing; every mechanism (co-attention alternation, modality type
# embeddings, CLIP-ViT patch embed, BERT self+cross attention) is unchanged.
"""M3AE: Multi-Modal Masked Autoencoder for medical vision-and-language.

Chen, Li, Wan, et al. 2022 (MICCAI). Vision encoder: from-scratch CLIP-style
ViT (`CLIP`/`VisualTransformer` below). Text encoder: BERT
(`transformers.BertModel`). Fusion: stacked `BertCrossLayer` co-attention
blocks alternating text->image and image->text cross-attention, producing
pooled multimodal CLS features.
"""

import math
from collections import OrderedDict
from typing import Tuple, Union

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from transformers.activations import ACT2FN

try:
    from transformers.modeling_utils import apply_chunking_to_forward
except ImportError:  # newer transformers moved this helper
    from transformers.pytorch_utils import apply_chunking_to_forward


MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# m3ae/modules/vision_encoders/clip_model.py -- M3AE's own CLIP-style vision
# encoder (self-contained; does not depend on the `clip` pip package).
# ---------------------------------------------------------------------------


class CLIPLayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (source name: `LayerNorm`)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = CLIPLayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = CLIPLayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, x_mask: torch.Tensor):
        if x_mask is not None:
            x_mask = x_mask.to(dtype=torch.bool, device=x.device)
        attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask, key_padding_mask=x_mask)[
            0
        ]

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        x = x + self.attention(self.ln_1(x), x_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers - 1)]
        )

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        for block in self.resblocks:
            x = block(x, x_mask)
        return x


class VisualTransformer(nn.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        resolution_after: int,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False
        )
        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((resolution_after // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = CLIPLayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = CLIPLayerNorm(width)

    def forward(self, x: torch.Tensor, x_mask=None):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        t = self.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([t, x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, x_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)

        return x


class CLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,
        vision_patch_size: int,
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
        resolution_after=224,
    ):
        super().__init__()

        self.context_length = context_length

        vision_heads = vision_width // 64
        self.visual = VisualTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            resolution_after=resolution_after,
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width)
        )
        self.ln_final = CLIPLayerNorm(transformer_width)

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.visual.transformer.width**-0.5) * (
            (2 * self.visual.transformer.layers) ** -0.5
        )
        attn_std = self.visual.transformer.width**-0.5
        fc_std = (2 * self.visual.transformer.width) ** -0.5
        for block in self.visual.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def forward(self, image, image_mask=None):
        return self.visual(image.type(self.dtype), image_mask)


# ---------------------------------------------------------------------------
# m3ae/modules/language_encoders/bert_model.py -- only the fusion-specific
# pieces (`BertCrossLayer` and the attention sub-modules it composes) are
# vendored; the plain BERT text encoder itself is the real
# `transformers.BertModel` (which is exactly what the source constructs via
# `BertModel.from_pretrained`).
# ---------------------------------------------------------------------------


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertCrossLayer(nn.Module):
    """M3AE's co-attention fusion block: self-attention then cross-attention
    onto the other modality's features (the paper's actual contribution)."""

    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        attention_mask=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask=None,
            output_attentions=output_attentions,
            past_key_value=None,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        cross_attention_outputs = self.crossattention(
            attention_output,
            attention_mask,
            None,
            encoder_hidden_states,
            encoder_attention_mask,
            None,
            output_attentions,
        )
        attention_output = cross_attention_outputs[0]
        outputs = outputs + cross_attention_outputs[1:]

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# ---------------------------------------------------------------------------
# m3ae/modules/prediction_heads.py -- `Pooler`.
# ---------------------------------------------------------------------------


class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# ---------------------------------------------------------------------------
# m3ae/modules/m3ae_utils.py -- `init_weights`.
# ---------------------------------------------------------------------------


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


# ---------------------------------------------------------------------------
# m3ae/modules/m3ae_module.py -- `M3AETransformerSS`, re-hosted as a plain
# nn.Module (`pl.LightningModule` -> `nn.Module`; `self.hparams.config` ->
# a plain `config` dict attribute). `__init__` model construction and
# `infer()`'s co-attention fusion body are preserved verbatim (mask_text=
# False, mask_image=False, output_attentions=False -- the plain forward-pass
# path with no pretraining-mask machinery).
# ---------------------------------------------------------------------------


class M3AEForStaging(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )

        resolution_after = config["image_size"]

        # M3AE's own CLIP-style ViT vision encoder (random init; source builds
        # it via `build_model(name, resolution_after)` which downloads
        # OpenAI-CLIP weights -- we construct the same `CLIP` class directly).
        self.vision_encoder = CLIP(
            embed_dim=config["hidden_size"],
            image_resolution=config["image_size"],
            vision_layers=config["vit_layers"],
            vision_width=config["vit_width"],
            vision_patch_size=config["patch_size"],
            context_length=config["max_text_len"],
            vocab_size=config["vocab_size"],
            transformer_width=config["vit_width"],
            transformer_heads=config["vit_width"] // 64,
            transformer_layers=config["vit_layers"],
            resolution_after=resolution_after,
        )
        self.language_encoder = BertModel(bert_config, add_pooling_layer=False)

        self.multi_modal_language_proj = nn.Linear(
            config["input_text_embed_size"], config["hidden_size"]
        )
        self.multi_modal_language_proj.apply(init_weights)
        self.multi_modal_vision_proj = nn.Linear(
            config["input_image_embed_size"], config["hidden_size"]
        )
        self.multi_modal_vision_proj.apply(init_weights)

        self.modality_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.modality_type_embeddings.apply(init_weights)

        self.multi_modal_vision_layers = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(config["num_top_layer"])]
        )
        self.multi_modal_vision_layers.apply(init_weights)
        self.multi_modal_language_layers = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(config["num_top_layer"])]
        )
        self.multi_modal_language_layers.apply(init_weights)

        self.multi_modal_vision_pooler = Pooler(config["hidden_size"])
        self.multi_modal_vision_pooler.apply(init_weights)
        self.multi_modal_language_pooler = Pooler(config["hidden_size"])
        self.multi_modal_language_pooler.apply(init_weights)

    def infer(self, img, text_ids, text_masks, image_token_type_idx=1):
        device = text_ids.device

        # == Begin: Text Encoding ==
        uni_modal_text_feats = self.language_encoder.embeddings(input_ids=text_ids)
        text_input_shape = text_masks.size()
        extended_text_masks = self.language_encoder.get_extended_attention_mask(
            text_masks, text_input_shape, device
        )
        for layer in self.language_encoder.encoder.layer:
            uni_modal_text_feats = layer(uni_modal_text_feats, extended_text_masks)[0]
        uni_modal_text_feats = self.multi_modal_language_proj(uni_modal_text_feats)
        # == End  : Text Encoding ==

        # == Begin: Image Encoding ==
        uni_modal_image_feats = self.vision_encoder(img)
        uni_modal_image_feats = self.multi_modal_vision_proj(uni_modal_image_feats)
        image_masks = torch.ones(
            (uni_modal_image_feats.size(0), uni_modal_image_feats.size(1)),
            dtype=torch.long,
            device=device,
        )
        extended_image_masks = self.language_encoder.get_extended_attention_mask(
            image_masks, image_masks.size(), device
        )
        # == End  : Image Encoding ==

        # == Begin: Assign Type Embeddings ==
        uni_modal_text_feats, uni_modal_image_feats = (
            uni_modal_text_feats + self.modality_type_embeddings(torch.zeros_like(text_masks)),
            uni_modal_image_feats
            + self.modality_type_embeddings(torch.full_like(image_masks, image_token_type_idx)),
        )
        # == End  : Assign Type Embeddings ==

        # == Begin: Multi-Modal Fusion ==
        x, y = uni_modal_text_feats, uni_modal_image_feats
        for text_layer, image_layer in zip(
            self.multi_modal_language_layers, self.multi_modal_vision_layers
        ):
            x1 = text_layer(x, y, extended_text_masks, extended_image_masks)
            y1 = image_layer(y, x, extended_image_masks, extended_text_masks)
            x, y = x1[0], y1[0]
        # == End: Multi-Modal Fusion ==

        # == Begin: Output Multi-Modal Features ==
        multi_modal_text_cls_feats = self.multi_modal_language_pooler(x)
        multi_modal_image_cls_feats = self.multi_modal_vision_pooler(y)
        multi_modal_cls_feats = torch.cat(
            [multi_modal_text_cls_feats, multi_modal_image_cls_feats], dim=-1
        )
        # == End  : Output Multi-Modal Features ==

        return multi_modal_cls_feats

    def forward(self, img, text_ids, text_masks):
        return self.infer(img, text_ids, text_masks)


# ---------------------------------------------------------------------------
# Staging build/example helpers. Dims are shrunk from the paper's defaults
# (hidden_size=768, num_top_layer=6, image_size=224/384, ViT-B/16) for fast
# tracing; every fusion/encoder mechanism above is unchanged from source.
# ---------------------------------------------------------------------------


def _tiny_config():
    return dict(
        vocab_size=128,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        mlp_ratio=4,
        max_text_len=16,
        drop_rate=0.0,
        image_size=32,
        patch_size=16,
        vit_layers=2,
        vit_width=64,
        input_text_embed_size=64,
        input_image_embed_size=64,
        num_top_layer=2,
    )


def build_m3ae():
    model = M3AEForStaging(_tiny_config())
    model.eval()
    return model


def example_input_m3ae():
    torch.manual_seed(0)
    img = torch.randn(2, 3, 32, 32)
    text_ids = torch.randint(0, 128, (2, 8))
    text_masks = torch.ones(2, 8, dtype=torch.long)
    return (img, text_ids, text_masks)


MENAGERIE_ENTRIES = [
    ("M3AE", build_m3ae, example_input_m3ae, 2022, "vendored-pytorch"),
]
