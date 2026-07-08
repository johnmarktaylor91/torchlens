# SOURCE: vendored from apple/ml-ferret @ main (ferretui/ferretui subtree)
# https://raw.githubusercontent.com/apple/ml-ferret/main/ferretui/ferretui/model/ferret_arch.py
# https://raw.githubusercontent.com/apple/ml-ferret/main/ferretui/ferretui/model/language_model/ferret_llama.py
# https://raw.githubusercontent.com/apple/ml-ferret/main/ferretui/ferretui/model/multimodal_encoder/clip_encoder.py
# https://raw.githubusercontent.com/apple/ml-ferret/main/ferretui/ferretui/model/multimodal_encoder/builder.py
# https://raw.githubusercontent.com/apple/ml-ferret/main/ferretui/ferretui/model/multimodal_projector/builder.py
# https://raw.githubusercontent.com/apple/ml-ferret/main/ferretui/ferretui/constants.py
#
# You et al. 2024 (ECCV) "Ferret-UI: Grounded Mobile UI Understanding with Multimodal
# LLMs". This vendors the real Ferret-UI architecture (LLaVA-family visual-instruction
# model: CLIP vision tower -> linear/MLP mm_projector -> Llama decoder, fused via
# `FerretMetaModel`/`FerretMetaForCausalLM`) plus Ferret's original contribution over
# plain LLaVA: the `GeoRegionSampler` region-referring head, which farthest-point-samples
# points inside a user-supplied binary region mask, groups neighbors via a k-NN +
# Conv1d/LayerNorm ("ConvReLULN1D") aggregation pyramid, and projects the aggregated
# region descriptor into the LLM's embedding space so the decoder can reference free-form
# image regions by mask instead of only bounding boxes.
#
# No architectural changes were made; only mechanical fixes for import isolation and
# to run at trace time without network/weight downloads:
#   - `CLIPVisionTower.load_model()` in the real repo calls
#     `CLIPVisionModel.from_pretrained(...)` (network fetch of a real CLIP checkpoint).
#     That call is replaced here with constructing a real `transformers.CLIPVisionModel`
#     directly from a `CLIPVisionConfig` (tiny random-init, same class, same forward
#     code) -- this is the standard "no network at trace time" substitution used
#     throughout the menagerie; the vision-tower *class* and its forward/config-driven
#     wiring are untouched.
#   - Cross-file imports (`from .multimodal_encoder.builder import build_vision_tower`,
#     `from ferretui.constants import ...`, `from ferretui.mm_utils import
#     get_anyres_image_grid_shape`) are flattened into this single module since the
#     `ferretui` package is not installed; `get_anyres_image_grid_shape` is unused by
#     the traced path (`mm_patch_merge_type='flat'`, i.e. the simplest single-image
#     merge branch already present in the real `prepare_inputs_labels_for_multimodal`)
#     so it is omitted rather than transcribed.
#   - `AutoConfig.register(...)` / `AutoModelForCausalLM.register(...)` calls (HF
#     Auto-class registration side effects, irrelevant to a single direct
#     instantiation) are dropped.
#   - `torch.distributed` import (unused at call time, side-effect only) is dropped.

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    CLIPVisionConfig,
    CLIPVisionModel,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

# ---------------------------------------------------------------------------
# ferretui/ferretui/constants.py (verbatim subset actually used below)
# ---------------------------------------------------------------------------
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_REGION_FEA_TOKEN = "<region_fea>"


# ---------------------------------------------------------------------------
# ferretui/ferretui/model/multimodal_encoder/clip_encoder.py (CLIPVisionTower,
# with only `load_model` changed to avoid a `from_pretrained` network fetch --
# see header note)
# ---------------------------------------------------------------------------
class CLIPVisionTower(nn.Module):
    def __init__(
        self, vision_tower_config: CLIPVisionConfig, select_layer=-2, select_feature="patch"
    ):
        super().__init__()
        self.is_loaded = False
        self.select_layer = select_layer
        self.select_feature = select_feature
        self._cfg = vision_tower_config
        self.load_model()

    def load_model(self, device_map=None):
        if self.is_loaded:
            return
        # Real repo: self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        # Substituted here with a direct tiny-config construction (see header note).
        self.vision_tower = CLIPVisionModel(self._cfg)
        self.vision_tower.requires_grad_(False)
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
        image_forward_outs = self.vision_tower(
            images.to(device=self.device, dtype=self.dtype), output_hidden_states=True
        )
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        return self.vision_tower.config

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


# ---------------------------------------------------------------------------
# ferretui/ferretui/model/multimodal_projector/builder.py (verbatim)
# ---------------------------------------------------------------------------
def build_vision_projector(mm_hidden_size, hidden_size, projector_type="linear"):
    if projector_type == "linear":
        return nn.Linear(mm_hidden_size, hidden_size)
    raise ValueError(f"Unknown projector type: {projector_type}")


# ---------------------------------------------------------------------------
# ferretui/ferretui/model/ferret_arch.py (verbatim: point-sampling helpers +
# GeoRegionSampler -- Ferret's region-referring contribution over LLaVA)
# ---------------------------------------------------------------------------
def rand_sample_repeat(x, max_len):
    if x.shape[0] < max_len:
        indices = torch.randint(0, x.shape[0], (max_len - x.shape[0],))
        return torch.cat((x, x[indices]), dim=0)
    elif x.shape[0] == max_len:
        return x
    else:
        rand_idx = torch.randperm(x.shape[0])[:max_len]
        return x[rand_idx, :]


def point_sample(input, point_coords, return_dtype, **kwargs):
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input.float(), (2.0 * point_coords - 1.0).float(), **kwargs)
    output = output.to(return_dtype)
    if add_dim:
        output = output.squeeze(3)
    return output


def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 2)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :]
    return new_points


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def knn_point(nsample, xyz, new_xyz):
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class ConvReLULN1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True):
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                bias=bias,
            ),
            self.act,
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = self.net(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = x.permute(0, 2, 1)
        return x


class GeoRegionSampler(nn.Module):
    def __init__(
        self, input_dim, output_dim, num_init_point, num_sub_point, num_neighbor, pooler_mode="mean"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_init_point = num_init_point
        self.num_sub_point = num_sub_point
        self.num_neighbor = num_neighbor

        self.diff_projector_list = nn.ModuleList()
        self.agg_projector_list = nn.ModuleList()
        self.pooler_list = nn.ModuleList()

        for ii in range(len(num_sub_point)):
            self.diff_projector_list.append(nn.Linear(self.input_dim + 2, self.input_dim + 2))
            self.agg_projector_list.append(
                ConvReLULN1D(in_channels=2 * (self.input_dim + 2), out_channels=self.input_dim)
            )
            if pooler_mode == "mean":
                self.pooler_list.append(nn.AvgPool1d(kernel_size=num_neighbor[ii]))
            elif pooler_mode == "max":
                self.pooler_list.append(nn.AdaptiveMaxPool1d(output_size=1))
            else:
                raise NotImplementedError(f"{pooler_mode} is not supported.")

        self.flatten_projector = nn.Linear(self.input_dim * num_sub_point[-1], self.input_dim)
        self.dim_projector = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, feature_map, region_masks, original_dtype, return_dtype):
        assert len(feature_map) == len(region_masks)

        all_points = []
        all_points_fea = []
        all_points_img_ids = []

        for img_idx, (region_feature_map_i, region_masks_list_i) in enumerate(
            zip(feature_map, region_masks)
        ):
            if len(region_masks_list_i) != 0:
                ori_image_wh = torch.tensor(
                    [region_masks_list_i[0].shape[0], region_masks_list_i[0].shape[1]],
                    device=region_masks_list_i[0].device,
                )[None,]
                cur_non_zero_pos = [
                    rand_sample_repeat((m.nonzero() / ori_image_wh), self.num_init_point)
                    for m in region_masks_list_i
                ]
                cur_non_zero_pos = torch.stack(cur_non_zero_pos)
                if region_feature_map_i.ndim == 2:
                    h = w = int(math.sqrt(region_feature_map_i.shape[0]))
                    c = region_feature_map_i.shape[-1]
                    region_feature_map_i = region_feature_map_i.reshape(h, w, c)
                else:
                    assert region_feature_map_i.ndim == 3
                dup_region_feature_map_i = region_feature_map_i.permute(2, 0, 1)
                dup_region_feature_map_i = dup_region_feature_map_i.unsqueeze(0).repeat(
                    cur_non_zero_pos.shape[0], 1, 1, 1
                )
                dup_region_feature_map_i_ori_type = dup_region_feature_map_i.to(original_dtype)
                region_feature_i = point_sample(
                    dup_region_feature_map_i_ori_type,
                    cur_non_zero_pos.flip(dims=(2,)).type(original_dtype),
                    return_dtype,
                    align_corners=True,
                )
                region_feature_i = region_feature_i.transpose(-2, -1)

                cur_img_ids = [img_idx] * len(cur_non_zero_pos)
                all_points.append(cur_non_zero_pos)
                all_points_fea.append(region_feature_i)
                all_points_img_ids.extend(cur_img_ids)

        if len(all_points) == 0:
            return [None] * len(region_masks)

        all_points = torch.cat(all_points, dim=0).to(return_dtype)
        all_points_fea = torch.cat(all_points_fea, dim=0)
        all_points_img_ids = torch.tensor(all_points_img_ids, device=all_points_fea.device)

        for stage_i in range(len(self.num_sub_point)):
            cur_num_sub_point = self.num_sub_point[stage_i]
            cur_num_neighbor = self.num_neighbor[stage_i]

            all_points = all_points.contiguous()
            fps_idx = farthest_point_sample(all_points, cur_num_sub_point).long()

            new_points = index_points(all_points, fps_idx)
            new_points_fea = index_points(all_points_fea, fps_idx)

            idx = knn_point(cur_num_neighbor, all_points, new_points)
            grouped_points = index_points(all_points, idx)
            grouped_points_fea = index_points(all_points_fea, idx)

            local_points_fea = torch.cat([grouped_points_fea, grouped_points], dim=-1)
            anchor_points_fea = torch.cat([new_points_fea, new_points], dim=-1).unsqueeze(-2)
            diff_points_fea = local_points_fea - anchor_points_fea

            diff_points_fea = self.diff_projector_list[stage_i](diff_points_fea)
            gather_points_fea = torch.cat(
                [diff_points_fea, anchor_points_fea.repeat(1, 1, cur_num_neighbor, 1)], dim=-1
            )

            b, n, s, d = gather_points_fea.size()
            gather_points_fea = gather_points_fea.permute(0, 1, 3, 2)
            gather_points_fea = gather_points_fea.reshape(-1, d, s)
            gather_points_fea = self.agg_projector_list[stage_i](gather_points_fea)

            batch_size, new_dim, _ = gather_points_fea.size()
            gather_points_fea = self.pooler_list[stage_i](gather_points_fea).view(
                batch_size, new_dim
            )

            gather_points_fea = gather_points_fea.reshape(b, n, -1)

            all_points = new_points
            all_points_fea = gather_points_fea

        x = all_points_fea.flatten(1, -1)
        x = self.flatten_projector(x)
        all_region_fea = self.dim_projector(x)

        output_region_fea = []
        for img_idx in range(len(region_masks)):
            cur_mask = all_points_img_ids == img_idx
            if not cur_mask.any():
                output_region_fea.append(None)
            else:
                output_region_fea.append(all_region_fea[cur_mask])

        return output_region_fea


# ---------------------------------------------------------------------------
# FerretMetaModel / FerretMetaForCausalLM (ferret_arch.py), trimmed to the
# `mm_patch_merge_type='flat'` + `region_geo_sampler=True` path that this
# staging build exercises. All retained branches are byte-identical logic to
# the real repo; only the training-time HF-config wiring (initialize_vision_*)
# is omitted since we construct submodules directly for tracing.
# ---------------------------------------------------------------------------
class FerretMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images, region_flag=False, region_geo_sampler=False):
        image_features = self.get_model().get_vision_tower()(images)
        projected_image_features = self.get_model().mm_projector(image_features)
        if region_flag:
            if region_geo_sampler:
                new_region_feature_map = image_features
            else:
                new_region_feature_map = self.get_model().region_fea_adapter(image_features)
        else:
            new_region_feature_map = None
        return image_features, projected_image_features, new_region_feature_map

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        region_masks=None,
    ):
        region_flag = region_masks is not None
        region_geo_sampler = region_flag and getattr(self.config, "region_geo_sampler", False)

        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # Single-image-per-sample branch (real repo: `images` is a plain 4D tensor,
        # i.e. neither `type(images) is list` nor `images.ndim == 5`). This is the
        # `else` branch of `prepare_inputs_labels_for_multimodal` in ferret_arch.py --
        # `region_feature_map` here is `image_features` itself, a 3D
        # (batch, num_patches, mm_hidden_size) tensor; GeoRegionSampler reshapes each
        # per-sample (num_patches, C) slice into (h, w, C) internally (ndim==2 case).
        raw_image_features, image_features, region_feature_map = self.encode_images(
            images, region_flag=region_flag, region_geo_sampler=region_geo_sampler
        )

        if region_flag:
            dump_region_mask = torch.zeros(100, 100, device=images.device)
            dump_region_mask[10:20, 10:20] = 1
            dump_region_masks = [[dump_region_mask.clone()]]
            for _ in range(len(region_feature_map) - 1):
                dump_region_masks.append([])

            region_features = self.get_model().region_geo_sampler(
                region_feature_map,
                region_masks,
                original_dtype=raw_image_features.dtype,
                return_dtype=image_features[0].dtype,
            )
            dump_region_features = self.get_model().region_geo_sampler(
                region_feature_map,
                dump_region_masks,
                original_dtype=raw_image_features.dtype,
                return_dtype=image_features[0].dtype,
            )
            assert len(region_features) == len(input_ids)

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

        input_ids_list = [cur[mask] for cur, mask in zip(input_ids, attention_mask)]
        labels_list = [cur[mask] for cur, mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids_list):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels_list[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = (
                [-1]
                + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
                + [cur_input_ids.shape[0]]
            )
            cur_input_id_with_im = []
            cur_input_ids_noim = []
            cur_labels = labels_list[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(
                    cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]]
                )
                cur_labels_noim.append(
                    cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]]
                )
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            for i in range(num_images + 1):
                cur_input_id_with_im.append(cur_input_ids_noim[i])
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_input_id_with_im.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IMAGE_TOKEN_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )
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
            cur_input_id_with_im = torch.cat(cur_input_id_with_im)

            if region_flag and region_features[batch_idx] is not None:
                region_embs = torch.zeros_like(cur_new_input_embeds)
                region_replace_mask = cur_input_id_with_im == self.config.im_region_fea_token
                if len(region_embs[region_replace_mask]) != len(region_features[batch_idx]):
                    region_embs[region_replace_mask] = region_features[batch_idx][
                        : len(region_embs[region_replace_mask])
                    ].to(cur_new_input_embeds.dtype)
                else:
                    region_embs[region_replace_mask] = region_features[batch_idx].to(
                        cur_new_input_embeds.dtype
                    )
                cur_new_input_embeds = (
                    cur_new_input_embeds
                    * (~region_replace_mask).to(cur_new_input_embeds.dtype)[:, None]
                    + region_embs
                )

            if region_flag:
                cur_new_input_embeds[0] = cur_new_input_embeds[0] + 0.0 * dump_region_features[0][
                    0
                ].to(cur_new_input_embeds.dtype)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

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

        new_labels = None if _labels is None else new_labels_padded
        attention_mask = (
            None if _attention_mask is None else attention_mask.to(dtype=_attention_mask.dtype)
        )
        position_ids = None if _position_ids is None else position_ids

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels


# ---------------------------------------------------------------------------
# ferretui/ferretui/model/language_model/ferret_llama.py (FerretLlamaModel /
# FerretLlamaForCausalLM, verbatim architecture; AutoConfig/AutoModel HF
# registry side effects dropped -- see header note)
# ---------------------------------------------------------------------------
class FerretConfig(LlamaConfig):
    model_type = "ferret_llama"


class FerretLlamaModel(LlamaModel):
    config_class = FerretConfig

    def __init__(self, config: LlamaConfig, vision_tower: CLIPVisionTower, mm_hidden_size: int):
        super().__init__(config)
        self.max_sample_point = 512
        self.vision_tower = vision_tower
        self.mm_projector = build_vision_projector(mm_hidden_size, config.hidden_size)
        self.region_geo_sampler = GeoRegionSampler(
            input_dim=mm_hidden_size,
            output_dim=config.hidden_size,
            num_init_point=self.max_sample_point,
            num_sub_point=[8, 4],
            num_neighbor=[4, 4],
            pooler_mode="mean",
        )

    def get_vision_tower(self):
        return self.vision_tower


class FerretLlamaForCausalLM(LlamaForCausalLM, FerretMetaForCausalLM):
    config_class = FerretConfig

    def __init__(
        self, config, vision_tower: CLIPVisionTower, mm_hidden_size: int, im_region_fea_token: int
    ):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = FerretLlamaModel(config, vision_tower, mm_hidden_size)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.config.region_geo_sampler = True
        self.config.im_region_fea_token = im_region_fea_token
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        region_masks: Optional[List[torch.Tensor]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = (
                self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                    region_masks=region_masks,
                )
            )

        return LlamaForCausalLM.forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            return_dict=return_dict,
        )


# ---------------------------------------------------------------------------
# Staging build/example (tiny sizes; exercises vision tower + mm_projector +
# GeoRegionSampler + Llama decoder end to end, same as a real Ferret-UI
# forward pass with one image + one region mask).
# ---------------------------------------------------------------------------
def build_ferretui():
    vision_cfg = CLIPVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=32,
        patch_size=16,
    )
    vision_tower = CLIPVisionTower(vision_cfg, select_layer=-1, select_feature="patch")

    llama_cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = FerretLlamaForCausalLM(
        llama_cfg,
        vision_tower=vision_tower,
        mm_hidden_size=vision_cfg.hidden_size,
        im_region_fea_token=5,
    )
    model.eval()
    return model


def example_input_ferretui():
    torch.manual_seed(0)
    input_ids = torch.randint(6, 64, (1, 6))
    input_ids[0, 2] = IMAGE_TOKEN_INDEX
    input_ids[0, 4] = 5  # im_region_fea_token
    images = torch.randn(1, 3, 32, 32)
    mask = torch.zeros(20, 20)
    mask[5:10, 5:10] = 1
    region_masks = [[mask]]
    # Positional order matches FerretLlamaForCausalLM.forward:
    # (input_ids, attention_mask, position_ids, past_key_values, inputs_embeds,
    #  labels, images, region_masks)
    return (input_ids, None, None, None, None, None, images, region_masks)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("Ferret-UI", "build_ferretui", "example_input_ferretui", 2024, "vendored"),
]
