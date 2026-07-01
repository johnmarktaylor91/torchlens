# SOURCE: vendored from PJLab-ADG/DriveArena @ main
# (WorldDreamer/projects/dreamer/networks/unet_2d_condition_multiview.py,
#  WorldDreamer/projects/dreamer/networks/blocks.py)
"""DriveArena (ICLR 2025) -- closed-loop generative driving simulation.

DriveArena is a multi-component system (WorldDreamer generative video model + TrafficManager
LimSim traffic simulator, not a NN + DrivingAgents UniAD/VAD planners, which need the heavy
mmdet3d/OpenMMLab stack and are already covered elsewhere in the menagerie). The genuinely
novel, self-contained neural architecture unique to DriveArena is WorldDreamer's
multi-view-conditioned diffusion UNet: `UNet2DConditionModelMultiview`, which subclasses
`diffusers.models.UNet2DConditionModel` and replaces every `BasicTransformerBlock` found in
the base UNet with a `BasicMultiviewTransformerBlock` (also real repo code, from the sibling
`networks/blocks.py`) that adds a 4th "cross-view" attention branch (`norm4`/`attn4`) wired
through a zero-initialized connector, letting each camera view attend to its geometric
neighbors (`neighboring_view_pair`) for multi-view-consistent generation. This is architecture
(a genuine subclass + custom attention block), not just usage -- rung 2, vendored verbatim
below with only import-path fixes for current diffusers (repo pins diffusers~=0.19 era paths
like `diffusers.models.unet_2d_condition`; those modules moved to
`diffusers.models.unets.unet_2d_condition` etc. in current diffusers, imports adjusted
accordingly, same classes) and one small compat shim (see `_btb_init_with_args` below).

Compat shim: the original `unet_2d_condition_multiview.py` recovers each base block's
constructor kwargs via `mod._args` on the base-class `BasicTransformerBlock`, populated by
the repo's own pinned/patched `third_party/diffusers` fork (their `WorldDreamer/third_party/
diffusers` vendors a modified `attention.py` that stashes `self._args = locals()`; this is
not present in stock pip diffusers at any version, including ones from the repo's era).
`_btb_init_with_args` restores that exact hook mechanically via `inspect.signature(...).bind`
on the unmodified base `__init__` -- it does not change `BasicTransformerBlock`'s own
architecture or forward computation, only recovers the kwargs the multiview conversion needs.
"""

from __future__ import annotations

import inspect
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from diffusers.models.attention import AdaLayerNorm, BasicTransformerBlock
from diffusers.models.attention_processor import Attention
from diffusers.models.unets.unet_2d_condition import (
    UNet2DConditionModel,
    UNet2DConditionOutput,
)

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# blocks.py (WorldDreamer/projects/dreamer/networks/blocks.py) -- vendored verbatim
# ---------------------------------------------------------------------------


def zero_module(module: nn.Module) -> nn.Module:
    # SOURCE: diffusers.models.controlnet.zero_module (moved to
    # diffusers.models.controlnets.controlnet in current diffusers); inlined verbatim
    # since the private old import path no longer exists.
    for p in module.parameters():
        p.detach().zero_()
    return module


def _ensure_kv_is_int(view_pair: dict) -> dict:
    """yaml key can be int, while json cannot. We convert here."""
    new_dict = {}
    for k, v in view_pair.items():
        new_value = [int(vi) for vi in v]
        new_dict[int(k)] = new_value
    return new_dict


def get_zero_module(zero_module_type: str, dim: int):
    if zero_module_type == "zero_linear":
        connector = zero_module(nn.Linear(dim, dim))
    elif zero_module_type == "gated":
        connector = GatedConnector(dim)
    elif zero_module_type == "none":

        def connector(x):
            return x

    else:
        raise TypeError(f"Unknown zero module type: {zero_module_type}")
    return connector


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
            final_dropout,
        )

        self._args = {k: v for k, v in locals().items() if k != "self" and not k.startswith("_")}
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
        self.connector = get_zero_module(zero_module_type, dim)

    @property
    def new_module(self):
        ret = {"norm4": self.norm4, "attn4": self.attn4}
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
            hidden_states_in1 = torch.cat(hidden_states_in1, dim=0)
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
        # Notice that normalization is always applied before the real computation
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
        hidden_states_in1, hidden_states_in2, cam_order = self._construct_attn_input(
            norm_hidden_states
        )
        attn_raw_output = self.attn4(
            hidden_states_in1,
            encoder_hidden_states=hidden_states_in2,
            **cross_attention_kwargs,
        )
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
        attn_output = self.connector(attn_output)
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


# ---------------------------------------------------------------------------
# _args compat shim -- see module docstring
# ---------------------------------------------------------------------------

_BTB_ARGS_KEYS = (
    "dim",
    "num_attention_heads",
    "attention_head_dim",
    "dropout",
    "cross_attention_dim",
    "activation_fn",
    "num_embeds_ada_norm",
    "attention_bias",
    "only_cross_attention",
    "double_self_attention",
    "upcast_attention",
    "norm_elementwise_affine",
    "norm_type",
    "final_dropout",
)
_orig_btb_init = BasicTransformerBlock.__init__
_orig_btb_sig = inspect.signature(_orig_btb_init)


def _btb_init_with_args(self, *args, **kwargs):
    _orig_btb_init(self, *args, **kwargs)
    bound = _orig_btb_sig.bind_partial(self, *args, **kwargs)
    bound.apply_defaults()
    self._args = {k: bound.arguments[k] for k in _BTB_ARGS_KEYS if k in bound.arguments}


if not getattr(BasicTransformerBlock, "_menagerie_args_patched", False):
    BasicTransformerBlock.__init__ = _btb_init_with_args
    BasicTransformerBlock._menagerie_args_patched = True


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
# unet_2d_condition_multiview.py -- vendored, forward loop unchanged from repo
# (removed: gradient-checkpointing / from_unet_2d_condition / trainable_module
# training-only helpers not needed for a tiny forward-pass trace)
# ---------------------------------------------------------------------------


class UNet2DConditionModelMultiview(UNet2DConditionModel):
    _supports_gradient_checkpointing = True

    def __init__(
        self,
        sample_size=None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        mid_block_type="UNetMidBlock2DCrossAttn",
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        only_cross_attention=False,
        block_out_channels=(32, 32),
        layers_per_block=1,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim=8,
        attention_head_dim=2,
        # parameters added by DriveArena for multiview
        neighboring_view_pair: Optional[dict] = None,
        neighboring_attn_type: str = "add",
        zero_module_type: str = "zero_linear",
        crossview_attn_type: str = "basic",
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
            attention_head_dim=attention_head_dim,
        )

        self.crossview_attn_type = crossview_attn_type
        self._new_module = {}
        for name, mod in list(self.named_modules()):
            if isinstance(mod, BasicTransformerBlock):
                if crossview_attn_type == "basic":
                    _set_module(
                        self,
                        name,
                        BasicMultiviewTransformerBlock(
                            **mod._args,
                            neighboring_view_pair=neighboring_view_pair,
                            neighboring_attn_type=neighboring_attn_type,
                            zero_module_type=zero_module_type,
                        ),
                    )
                else:
                    raise TypeError(f"Unknown attn type: {crossview_attn_type}")
                for k, v in _get_module(self, name).new_module.items():
                    self._new_module[f"{name}.{k}"] = v

    def forward(
        self,
        sample: torch.Tensor,
        timestep,
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs=None,
        down_block_additional_residuals=None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        default_overall_up_factor = 2**self.num_upsamplers
        forward_upsample_size = False
        upsample_size = None
        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            forward_upsample_size = True

        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

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

        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps)
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
            if (
                hasattr(downsample_block, "has_cross_attention")
                and downsample_block.has_cross_attention
            ):
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
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
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if (
                hasattr(upsample_block, "has_cross_attention")
                and upsample_block.has_cross_attention
            ):
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
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
# Menagerie staging glue
# ---------------------------------------------------------------------------

_N_CAM = 2  # real repo uses 6 (surround-view nuScenes cameras); shrunk for tracing


class DriveArenaMultiviewUNetWrapper(nn.Module):
    """Wraps UNet2DConditionModelMultiview tiny-sized for tracing, returning a single
    tensor (native forward returns a UNet2DConditionOutput dataclass with `.sample`)."""

    def __init__(self):
        super().__init__()
        self.unet = UNet2DConditionModelMultiview(
            sample_size=8,
            in_channels=4,
            out_channels=4,
            down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
            block_out_channels=(32, 32),  # real repo uses (320, 640, 1280, 1280)
            layers_per_block=1,  # real repo uses 2
            cross_attention_dim=8,  # real repo uses 768 (CLIP text/image embed dim)
            attention_head_dim=2,
            neighboring_view_pair={i: [(i + 1) % _N_CAM] for i in range(_N_CAM)},
            neighboring_attn_type="add",
            zero_module_type="zero_linear",
            crossview_attn_type="basic",
        )

    def forward(
        self, sample: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor
    ):
        return self.unet(sample, timestep, encoder_hidden_states).sample


def build_drivearena():
    return DriveArenaMultiviewUNetWrapper()


def example_input_drivearena():
    sample = torch.randn(_N_CAM, 4, 8, 8)
    timestep = torch.tensor(1.0)
    encoder_hidden_states = torch.randn(_N_CAM, 1, 8)
    return (sample, timestep, encoder_hidden_states)


MENAGERIE_ENTRIES = [
    (
        "DriveArena-WorldDreamer",
        build_drivearena,
        example_input_drivearena,
        2025,
        "vendored-pytorch",
    ),
]
