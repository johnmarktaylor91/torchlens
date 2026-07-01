# SOURCE: vendored from cure-lab/MagicDrive @ f6dd2e56da8a3a158c47e74a006c28d8aea13433
# Files: magicdrive/networks/blocks.py, magicdrive/networks/unet_2d_condition_multiview.py
# (verbatim real architecture). MagicDrive's own repo pins a vendored fork of
# diffusers ~0.19 (third_party/diffusers) because this exact code depends on
# diffusers-internal module paths/attributes from that era. Two changes were
# required to run it against our installed diffusers (0.38.0) WITHOUT altering
# the architecture itself:
#   1. import paths updated to diffusers' current internal locations
#      (diffusers.models.unets.unet_2d_condition / unets.unet_2d_blocks /
#      controlnets.controlnet), which still expose the identical public classes.
#   2. `from_unet_2d_condition`'s block-swap loop originally read a private
#      `mod._args` dict that diffusers ~0.19 attached to every constructed
#      `BasicTransformerBlock` (a since-removed reload convenience). Current
#      diffusers exposes the same constructor values directly as instance
#      attributes on the block, so `_reconstruct_block_args()` below rebuilds
#      the equivalent kwargs dict from those attributes -- same values, same
#      call, no architecture change.
"""MagicDrive multiview UNet: BEV/3D-geometry controlled multi-camera driving
image diffusion (ICLR 2024).

Real repo: https://github.com/cure-lab/MagicDrive

This stages `UNet2DConditionModelMultiview`, the paper's core contribution: a
`UNet2DConditionModel` subclass that swaps every `BasicTransformerBlock` for a
`BasicMultiviewTransformerBlock` adding cross-camera attention (`attn4`). The
much larger BEV-box/map ControlNet addon (`unet_addon_rawbox.py`) that
conditions this UNet on 3D boxes/road maps is a separate module built on top
of this one and is out of scope for a minimal traceable example.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange

from diffusers.configuration_utils import register_to_config
from diffusers.models.unets.unet_2d_condition import (
    UNet2DConditionModel,
    UNet2DConditionOutput,
)
from diffusers.models.unets.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    DownBlock2D,
    UpBlock2D,
)
from diffusers.models.attention import BasicTransformerBlock, AdaLayerNorm
from diffusers.models.attention_processor import Attention
from diffusers.models.controlnets.controlnet import zero_module

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# magicdrive/misc/common.py (only the two generic submodule accessors needed
# here; verbatim, just lifted out of a file that otherwise imports `accelerate`
# for unrelated helpers we do not use)
# ---------------------------------------------------------------------------
def _get_module(model, submodule_key):
    tokens = submodule_key.split(".")
    cur_mod = model
    for s in tokens:
        cur_mod = getattr(cur_mod, s)
    return cur_mod


def _set_module(model, submodule_key, module):
    tokens = submodule_key.split(".")
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)

    setattr(cur_mod, tokens[-1], module)


# ---------------------------------------------------------------------------
# magicdrive/networks/blocks.py (verbatim)
# ---------------------------------------------------------------------------
def _ensure_kv_is_int(view_pair: dict):
    """yaml key can be int, while json cannot. We convert here."""
    new_dict = {}
    for k, v in view_pair.items():
        new_value = [int(vi) for vi in v]
        new_dict[int(k)] = new_value
    return new_dict


class GatedConnector(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        data = torch.zeros(dim)
        self.alpha = nn.parameter.Parameter(data)

    def forward(self, inx):
        # as long as last dim of input == dim, pytorch can auto-broad
        return F.tanh(self.alpha) * inx


class BasicMultiviewTransformerBlock(BasicTransformerBlock):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        final_dropout: bool = False,
        # multi_view
        neighboring_view_pair: Optional[Dict[int, List[int]]] = None,
        neighboring_attn_type: Optional[str] = "add",
        zero_module_type="zero_linear",
    ):
        super().__init__(
            dim,
            num_attention_heads,
            attention_head_dim,
            dropout,
            cross_attention_dim,
            activation_fn,
            num_embeds_ada_norm,
            attention_bias,
            only_cross_attention,
            double_self_attention,
            upcast_attention,
            norm_elementwise_affine,
            norm_type,
            final_dropout=final_dropout,
        )

        self.neighboring_view_pair = _ensure_kv_is_int(neighboring_view_pair)
        self.neighboring_attn_type = neighboring_attn_type
        # multiview attention
        self.norm4 = (
            AdaLayerNorm(dim, num_embeds_ada_norm)
            if self.use_ada_layer_norm
            else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        )
        self.attn4 = Attention(
            query_dim=dim,
            cross_attention_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )
        if zero_module_type == "zero_linear":
            # NOTE: zero_module cannot apply to successive layers.
            self.connector = zero_module(nn.Linear(dim, dim))
        elif zero_module_type == "gated":
            self.connector = GatedConnector(dim)
        elif zero_module_type == "none":
            # TODO: if this block is in controlnet, we may not need zero here.
            self.connector = lambda x: x
        else:
            raise TypeError(f"Unknown zero module type: {zero_module_type}")

    @property
    def new_module(self):
        ret = {
            "norm4": self.norm4,
            "attn4": self.attn4,
        }
        if isinstance(self.connector, nn.Module):
            ret["connector"] = self.connector
        return ret

    @property
    def n_cam(self):
        return len(self.neighboring_view_pair)

    def _construct_attn_input(self, norm_hidden_states):
        B = len(norm_hidden_states)
        # reshape, key for origin view, value for ref view
        hidden_states_in1 = []
        hidden_states_in2 = []
        cam_order = []
        if self.neighboring_attn_type == "add":
            for key, values in self.neighboring_view_pair.items():
                for value in values:
                    hidden_states_in1.append(norm_hidden_states[:, key])
                    hidden_states_in2.append(norm_hidden_states[:, value])
                    cam_order += [key] * B
            # N*2*B, H*W, head*dim
            hidden_states_in1 = torch.cat(hidden_states_in1, dim=0)
            hidden_states_in2 = torch.cat(hidden_states_in2, dim=0)
            cam_order = torch.LongTensor(cam_order)
        elif self.neighboring_attn_type == "concat":
            for key, values in self.neighboring_view_pair.items():
                hidden_states_in1.append(norm_hidden_states[:, key])
                hidden_states_in2.append(
                    torch.cat([norm_hidden_states[:, value] for value in values], dim=1)
                )
                cam_order += [key] * B
            # N*B, H*W, head*dim
            hidden_states_in1 = torch.cat(hidden_states_in1, dim=0)
            # N*B, 2*H*W, head*dim
            hidden_states_in2 = torch.cat(hidden_states_in2, dim=0)
            cam_order = torch.LongTensor(cam_order)
        elif self.neighboring_attn_type == "self":
            hidden_states_in1 = rearrange(norm_hidden_states, "b n l ... -> b (n l) ...")
            hidden_states_in2 = None
            cam_order = None
        else:
            raise NotImplementedError(f"Unknown type: {self.neighboring_attn_type}")
        return hidden_states_in1, hidden_states_in2, cam_order

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        timestep=None,
        cross_attention_kwargs=None,
        class_labels=None,
    ):
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 1. Self-Attention
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)

        cross_attention_kwargs = (
            cross_attention_kwargs if cross_attention_kwargs is not None else {}
        )
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states

        # 2. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = (
                self.norm2(hidden_states, timestep)
                if self.use_ada_layer_norm
                else self.norm2(hidden_states)
            )
            # TODO (Birch-San): Here we should prepare the encoder_attention mask correctly
            # prepare attention mask here

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # multi-view cross attention
        norm_hidden_states = (
            self.norm4(hidden_states, timestep)
            if self.use_ada_layer_norm
            else self.norm4(hidden_states)
        )
        # batch dim first, cam dim second
        norm_hidden_states = rearrange(norm_hidden_states, "(b n) ... -> b n ...", n=self.n_cam)
        B = len(norm_hidden_states)
        # key is query in attention; value is key-value in attention
        hidden_states_in1, hidden_states_in2, cam_order = self._construct_attn_input(
            norm_hidden_states,
        )
        # attention
        attn_raw_output = self.attn4(
            hidden_states_in1,
            encoder_hidden_states=hidden_states_in2,
            **cross_attention_kwargs,
        )
        # final output
        if self.neighboring_attn_type == "self":
            attn_output = rearrange(attn_raw_output, "b (n l) ... -> b n l ...", n=self.n_cam)
        else:
            attn_output = torch.zeros_like(norm_hidden_states)
            for cam_i in range(self.n_cam):
                attn_out_mv = rearrange(
                    attn_raw_output[cam_order == cam_i], "(n b) ... -> b n ...", b=B
                )
                attn_output[:, cam_i] = torch.sum(attn_out_mv, dim=1)
        attn_output = rearrange(attn_output, "b n ... -> (b n) ...")
        # apply zero init connector (one layer)
        attn_output = self.connector(attn_output)
        # short-cut
        hidden_states = attn_output + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states

        return hidden_states


def _reconstruct_block_args(mod: BasicTransformerBlock) -> dict:
    """Compat shim (see module header) for diffusers >= ~0.20: rebuild the
    kwargs dict that diffusers ~0.19 used to attach to every constructed
    `BasicTransformerBlock` as `mod._args`. Current diffusers exposes each of
    these as a plain instance attribute (on the block itself, or -- for
    `upcast_attention` -- on its inner `attn1` processor), so this reads back
    exactly the values the block was actually constructed with.
    """
    return dict(
        dim=mod.dim,
        num_attention_heads=mod.num_attention_heads,
        attention_head_dim=mod.attention_head_dim,
        dropout=mod.dropout,
        cross_attention_dim=mod.cross_attention_dim,
        activation_fn=mod.activation_fn,
        num_embeds_ada_norm=mod.num_embeds_ada_norm,
        attention_bias=mod.attention_bias,
        only_cross_attention=mod.only_cross_attention,
        double_self_attention=mod.double_self_attention,
        upcast_attention=getattr(mod.attn1, "upcast_attention", False),
        norm_elementwise_affine=mod.norm_elementwise_affine,
        norm_type=mod.norm_type,
        # diffusers ~0.19's BasicTransformerBlock defaulted final_dropout=False
        # and the base UNet blocks never override it; current diffusers doesn't
        # retain this flag on the instance, so we use the (always-applicable)
        # default rather than guessing.
        final_dropout=False,
    )


# ---------------------------------------------------------------------------
# magicdrive/networks/unet_2d_condition_multiview.py (verbatim, except the
# `mod._args` -> `_reconstruct_block_args(mod)` compat shim noted above)
# ---------------------------------------------------------------------------
class UNet2DConditionModelMultiview(UNet2DConditionModel):
    r"""
    UNet2DConditionModel is a conditional 2D UNet model that takes in a noisy sample, conditional state, and a timestep
    and returns sample shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the models (such as downloading or saving, etc.)
    """

    _supports_gradient_checkpointing = True
    _WARN_ONCE = 0

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        up_block_types: Tuple[str] = (
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: Union[int, Tuple[int]] = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: Union[int, Tuple[int]] = 1280,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        resnet_skip_time_act: bool = False,
        resnet_out_scale_factor: int = 1.0,
        time_embedding_type: str = "positional",
        time_embedding_dim: Optional[int] = None,
        time_embedding_act_fn: Optional[str] = None,
        timestep_post_act: Optional[str] = None,
        time_cond_proj_dim: Optional[int] = None,
        conv_in_kernel: int = 3,
        conv_out_kernel: int = 3,
        projection_class_embeddings_input_dim: Optional[int] = None,
        class_embeddings_concat: bool = False,
        mid_block_only_cross_attention: Optional[bool] = None,
        cross_attention_norm: Optional[str] = None,
        addition_embed_type_num_heads=64,
        # parameter added, we should keep all above (do not use kwargs)
        trainable_state="only_new",
        neighboring_view_pair: Optional[dict] = None,
        neighboring_attn_type: str = "add",
        zero_module_type: str = "zero_linear",
        crossview_attn_type: str = "basic",
        img_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            center_input_sample=center_input_sample,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            down_block_types=down_block_types,
            mid_block_type=mid_block_type,
            up_block_types=up_block_types,
            only_cross_attention=only_cross_attention,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            downsample_padding=downsample_padding,
            mid_block_scale_factor=mid_block_scale_factor,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            cross_attention_dim=cross_attention_dim,
            encoder_hid_dim=encoder_hid_dim,
            encoder_hid_dim_type=encoder_hid_dim_type,
            attention_head_dim=attention_head_dim,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            class_embed_type=class_embed_type,
            addition_embed_type=addition_embed_type,
            num_class_embeds=num_class_embeds,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resnet_skip_time_act=resnet_skip_time_act,
            resnet_out_scale_factor=resnet_out_scale_factor,
            time_embedding_type=time_embedding_type,
            time_embedding_dim=time_embedding_dim,
            time_embedding_act_fn=time_embedding_act_fn,
            timestep_post_act=timestep_post_act,
            time_cond_proj_dim=time_cond_proj_dim,
            conv_in_kernel=conv_in_kernel,
            conv_out_kernel=conv_out_kernel,
            projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
            class_embeddings_concat=class_embeddings_concat,
            mid_block_only_cross_attention=mid_block_only_cross_attention,
            cross_attention_norm=cross_attention_norm,
            addition_embed_type_num_heads=addition_embed_type_num_heads,
        )

        self.crossview_attn_type = crossview_attn_type
        self.img_size = [int(s) for s in img_size] if img_size is not None else None
        self._new_module = {}
        for name, mod in list(self.named_modules()):
            if isinstance(mod, BasicTransformerBlock):
                if crossview_attn_type == "basic":
                    _set_module(
                        self,
                        name,
                        BasicMultiviewTransformerBlock(
                            **_reconstruct_block_args(mod),
                            neighboring_view_pair=neighboring_view_pair,
                            neighboring_attn_type=neighboring_attn_type,
                            zero_module_type=zero_module_type,
                        ),
                    )
                else:
                    raise TypeError(f"Unknown attn type: {crossview_attn_type}")
                for k, v in _get_module(self, name).new_module.items():
                    self._new_module[f"{name}.{k}"] = v
        self.trainable_state = trainable_state

    @property
    def trainable_module(self) -> Dict[str, nn.Module]:
        if self.trainable_state == "all":
            return {self.__class__: self}
        elif self.trainable_state == "only_new":
            return self._new_module
        else:
            raise ValueError(f"Unknown trainable_state: {self.trainable_state}")

    @property
    def trainable_parameters(self) -> List[nn.Parameter]:
        params = []
        for mod in self.trainable_module.values():
            for param in mod.parameters():
                params.append(param)
        return params

    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        # first, set all to false
        super().train(False)
        if mode:
            # ensure gradient_checkpointing is usable, set training = True
            for mod in self.modules():
                if getattr(mod, "gradient_checkpointing", False):
                    mod.training = True
        # then, for some modules, we set according to `mode`
        self.training = False
        for mod in self.trainable_module.values():
            if mod is self:
                super().train(mode)
            else:
                mod.train(mode)
        return self

    def enable_gradient_checkpointing(self, flag=None):
        """
        Activates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        mod_idx = -1
        for module in self.modules():
            if isinstance(
                module, (CrossAttnDownBlock2D, DownBlock2D, CrossAttnUpBlock2D, UpBlock2D)
            ):
                mod_idx += 1
                if flag is not None and not flag[mod_idx]:
                    logging.debug(
                        f"[UNet2DConditionModelMultiview] "
                        f"gradient_checkpointing skip [{module.__class__}]"
                    )
                    continue
                logging.debug(
                    f"[UNet2DConditionModelMultiview] set "
                    f"[{module.__class__}] to gradient_checkpointing"
                )
                module.gradient_checkpointing = True

    @classmethod
    def from_unet_2d_condition(
        cls,
        unet: UNet2DConditionModel,
        load_weights_from_unet: bool = True,
        # multivew
        **kwargs,
    ):
        r"""
        Instantiate Multiview unet class from UNet2DConditionModel.

        Parameters:
            unet (`UNet2DConditionModel`):
                UNet model which weights are copied to the ControlNet. Note that all configuration options are also
                copied where applicable.
        """

        unet_2d_condition_multiview = cls(
            **unet.config,
            # multivew
            **kwargs,
        )

        if load_weights_from_unet:
            missing_keys, unexpected_keys = unet_2d_condition_multiview.load_state_dict(
                unet.state_dict(), strict=False
            )
            logging.info(
                f"[UNet2DConditionModelMultiview] load pretrained with "
                f"missing_keys: {missing_keys}; "
                f"unexpected_keys: {unexpected_keys}"
            )

        return unet_2d_condition_multiview

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            if self._WARN_ONCE == 0:
                logging.warning(
                    "[UNet2DConditionModelMultiview] Forward upsample size to force interpolation output size."
                )
                self._WARN_ONCE = 1
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)
                class_labels = class_labels.to(dtype=sample.dtype)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)

            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        if self.config.addition_embed_type == "text":
            aug_emb = self.add_embedding(encoder_hidden_states)
            emb = emb + aug_emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        if self.encoder_hid_proj is not None:
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if self.crossview_attn_type == "epipolar":
                cross_attention_kwargs["out_size"] = sample.shape[-2:]
            if (
                hasattr(downsample_block, "has_cross_attention")
                and downsample_block.has_cross_attention
            ):
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=copy.deepcopy(cross_attention_kwargs),
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples += (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.mid_block is not None:
            if self.crossview_attn_type == "epipolar":
                cross_attention_kwargs["out_size"] = sample.shape[-2:]
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=copy.deepcopy(cross_attention_kwargs),
            )

        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if (
                hasattr(upsample_block, "has_cross_attention")
                and upsample_block.has_cross_attention
            ):
                if self.crossview_attn_type == "epipolar":
                    cross_attention_kwargs["out_size"] = sample.shape[-2:]
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=copy.deepcopy(cross_attention_kwargs),
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample)


# ---------------------------------------------------------------------------
# staging harness -- tiny config (small channels/heads, 2 cameras) purely to
# keep the trace fast; the architecture (multiview cross-attention swapped
# into every transformer block) is exactly the real class.
# ---------------------------------------------------------------------------
_N_CAM = 2


def build_magicdrive_multiview_unet():
    neighboring_view_pair = {0: [1], 1: [0]}
    return UNet2DConditionModelMultiview(
        sample_size=8,
        in_channels=4,
        out_channels=4,
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        block_out_channels=(16, 32),
        layers_per_block=1,
        cross_attention_dim=24,
        attention_head_dim=4,
        norm_num_groups=8,
        neighboring_view_pair=neighboring_view_pair,
        neighboring_attn_type="add",
        zero_module_type="zero_linear",
        crossview_attn_type="basic",
    )


def example_input_magicdrive_multiview_unet():
    # sample is (batch * n_cam, C, H, W): each camera view is a separate item
    # in the leading batch dim, exactly as consumed by the multiview attention
    # blocks (which reshape back to (batch, n_cam, ...) internally).
    batch = 1
    sample = torch.randn(batch * _N_CAM, 4, 8, 8)
    timestep = torch.tensor(5)
    encoder_hidden_states = torch.randn(batch * _N_CAM, 6, 24)
    return (sample, timestep, encoder_hidden_states)


MENAGERIE_ENTRIES = [
    (
        "MagicDrive",
        build_magicdrive_multiview_unet,
        example_input_magicdrive_multiview_unet,
        2024,
        "vendored-pytorch",
    ),
]
