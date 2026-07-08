# SOURCE: vendored from https://github.com/woven-planet/l5kit @ main
# (l5kit/l5kit/planning/vectorized/{open_loop_model,common,global_graph,local_graph}.py)
"""UrbanDriver: vectorized closed-loop-trained imitation planner for autonomous driving
(Scheel et al., "Urban Driver: Learning to Drive from Real-world Demonstrations Using
Policy Gradients", CoRL 2021 / evaluated NeurIPS 2021, Woven Planet). Vendored verbatim
from the official l5kit repo's ``VectorizedModel`` (open-loop base architecture also used
as the closed-loop policy backbone): PointNet-like local subgraph over per-element
polylines, learned type embeddings, and a multi-head-attention global graph head.

Only two non-architectural fixes were applied to make the file self-contained outside the
full l5kit package:
  - ``VectorizedEmbedding`` originally imports ``PERCEPTION_LABEL_TO_INDEX`` from
    ``l5kit.data``; that dict is a plain data constant (perception-label name -> index),
    not architecture, so it is inlined verbatim from ``l5kit/l5kit/data/labels.py``.
  - The relative package imports (``.common``, ``.global_graph``, ``.local_graph``) are
    flattened into this single module since all three files are vendored together here.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


# ----------------------------- l5kit/data/labels.py (constant only) -----------------------------

PERCEPTION_LABELS = [
    "PERCEPTION_LABEL_NOT_SET",
    "PERCEPTION_LABEL_UNKNOWN",
    "PERCEPTION_LABEL_DONTCARE",
    "PERCEPTION_LABEL_CAR",
    "PERCEPTION_LABEL_VAN",
    "PERCEPTION_LABEL_TRAM",
    "PERCEPTION_LABEL_BUS",
    "PERCEPTION_LABEL_TRUCK",
    "PERCEPTION_LABEL_EMERGENCY_VEHICLE",
    "PERCEPTION_LABEL_OTHER_VEHICLE",
    "PERCEPTION_LABEL_BICYCLE",
    "PERCEPTION_LABEL_MOTORCYCLE",
    "PERCEPTION_LABEL_CYCLIST",
    "PERCEPTION_LABEL_MOTORCYCLIST",
    "PERCEPTION_LABEL_PEDESTRIAN",
    "PERCEPTION_LABEL_ANIMAL",
    "AVRESEARCH_LABEL_DONTCARE",
]
PERCEPTION_LABEL_TO_INDEX = {label: index for (index, label) in enumerate(PERCEPTION_LABELS)}


# ----------------------------- planning/vectorized/common.py -----------------------------


def pad_points(polylines: torch.Tensor, pad_to: int) -> torch.Tensor:
    """Pad vectors to `pad_to` size. Dimensions are:
    B: batch
    N: number of elements (polylines)
    P: number of points
    F: number of features

    :param polylines: polylines to be padded, should be (B,N,P,F) and we're padding P
    :param pad_to: nums of points we want
    :return: the padded polylines (B,N,pad_to,F)
    """
    batch, num_els, num_points, num_feats = polylines.shape
    pad_len = pad_to - num_points
    pad = torch.zeros(
        batch, num_els, pad_len, num_feats, dtype=polylines.dtype, device=polylines.device
    )
    return torch.cat([polylines, pad], dim=-2)


def pad_avail(avails: torch.Tensor, pad_to: int) -> torch.Tensor:
    """Pad avails to `pad_to` size

    :param avails: avails to be padded, should be (B,N,P) and we're padding P
    :param pad_to: nums of points we want
    :return: the padded avails (B,N,pad_to)
    """
    batch, num_els, num_points = avails.shape
    pad_len = pad_to - num_points
    pad = torch.zeros(batch, num_els, pad_len, dtype=avails.dtype, device=avails.device)
    return torch.cat([avails, pad], dim=-1)


def build_target_normalization(nsteps: int) -> torch.Tensor:
    """Normalization coefficients approximated with 3-rd degree polynomials
    to avoid storing them explicitly, and allow changing the length

    :param nsteps: number of steps to generate normalisation for
    :return: XY scaling for the steps
    """

    normalization_polynomials = np.asarray(
        [
            # x scaling
            [3.28e-05, -0.0017684, 1.8088969, 2.211737],
            # y scaling
            [-5.67e-05, 0.0052056, 0.0138343, 0.0588579],  # manually decreased by 5
        ]
    )
    # assuming we predict x, y and yaw
    coefs = np.stack([np.poly1d(p)(np.arange(nsteps)) for p in normalization_polynomials])
    coefs = coefs.astype(np.float32)
    return torch.from_numpy(coefs).T


# ----------------------------- planning/vectorized/global_graph.py -----------------------------


class VectorizedEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        """A module which associates learnable embeddings to types

        :param embedding_dim: features of the embedding
        """
        super(VectorizedEmbedding, self).__init__()
        # Torchscript did not like enums, so we are going more primitive.
        self.polyline_types = {
            "AGENT_OF_INTEREST": 0,
            "AGENT_NO": 1,
            "AGENT_CAR": 2,
            "AGENT_BIKE": 3,
            "AGENT_PEDESTRIAN": 4,
            "TL_UNKNOWN": 5,  # unknown TL state for lane
            "TL_RED": 6,
            "TL_YELLOW": 7,
            "TL_GREEN": 8,
            "TL_NONE": 9,  # no TL for lane
            "CROSSWALK": 10,
            "LANE_BDRY_LEFT": 11,
            "LANE_BDRY_RIGHT": 12,
        }

        self.embedding = nn.Embedding(len(self.polyline_types), embedding_dim)

        # Torch script did not like dicts as Tensor selectors, so we are going more primitive.
        self.PERCEPTION_LABEL_CAR: int = PERCEPTION_LABEL_TO_INDEX["PERCEPTION_LABEL_CAR"]
        self.PERCEPTION_LABEL_PEDESTRIAN: int = PERCEPTION_LABEL_TO_INDEX[
            "PERCEPTION_LABEL_PEDESTRIAN"
        ]
        self.PERCEPTION_LABEL_CYCLIST: int = PERCEPTION_LABEL_TO_INDEX["PERCEPTION_LABEL_CYCLIST"]

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Model forward: embed the given elements based on their type.

        Assumptions:
        - agent of interest is the first one in the batch
        - other agents follow
        - then we have polylines (lanes)
        """

        with torch.no_grad():
            polyline_types = data_batch["type"]
            other_agents_types = data_batch["all_other_agents_types"]

            other_agents_len = other_agents_types.shape[1]
            lanes_len = data_batch["lanes_mid"].shape[1]
            crosswalks_len = data_batch["crosswalks"].shape[1]
            lanes_bdry_len = data_batch["lanes"].shape[1]
            total_len = 1 + other_agents_len + lanes_len + crosswalks_len + lanes_bdry_len

            other_agents_start_idx = 1
            lanes_start_idx = other_agents_start_idx + other_agents_len
            crosswalks_start_idx = lanes_start_idx + lanes_len
            lanes_bdry_start_idx = crosswalks_start_idx + crosswalks_len

            indices = torch.full(
                (len(polyline_types), total_len),
                fill_value=self.polyline_types["AGENT_NO"],
                dtype=torch.long,
                device=polyline_types.device,
            )

            # set agent of interest
            indices[:, 0].fill_(self.polyline_types["AGENT_OF_INTEREST"])
            # set others
            indices[:, other_agents_start_idx:lanes_start_idx][
                other_agents_types == self.PERCEPTION_LABEL_CAR
            ].fill_(self.polyline_types["AGENT_CAR"])
            indices[:, other_agents_start_idx:lanes_start_idx][
                other_agents_types == self.PERCEPTION_LABEL_PEDESTRIAN
            ].fill_(self.polyline_types["AGENT_PEDESTRIAN"])
            indices[:, other_agents_start_idx:lanes_start_idx][
                other_agents_types == self.PERCEPTION_LABEL_CYCLIST
            ].fill_(self.polyline_types["AGENT_BIKE"])

            # set lanes given their TL state.
            indices[:, lanes_start_idx:crosswalks_start_idx].copy_(
                data_batch["lanes_mid"][:, :, 0, -1]
            ).add_(self.polyline_types["TL_UNKNOWN"])

            indices[:, crosswalks_start_idx:lanes_bdry_start_idx].fill_(
                self.polyline_types["CROSSWALK"]
            )
            indices[:, lanes_bdry_start_idx::2].fill_(self.polyline_types["LANE_BDRY_LEFT"])
            indices[:, lanes_bdry_start_idx + 1 :: 2].fill_(self.polyline_types["LANE_BDRY_RIGHT"])

        return self.embedding.forward(indices)


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.layers.children():
            nn.init.zeros_(layer.bias)
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MultiheadAttentionGlobalHead(nn.Module):
    """Global graph making use of multi-head attention."""

    def __init__(
        self,
        d_model: int,
        num_timesteps: int,
        num_outputs: int,
        nhead: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.num_outputs = num_outputs
        self.encoder = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.output_embed = MLP(d_model, d_model * 4, num_timesteps * num_outputs, num_layers=3)

    def forward(
        self, inputs: torch.Tensor, type_embedding: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Model forward:

        :param inputs: model inputs
        :param type_embedding: type embedding describing the different input types
        :param mask: availability mask

        :return tuple of outputs, attention
        """
        # dot-product attention:
        #   - query is ego's vector
        #   - key is inputs plus type embedding
        #   - value is inputs
        out, attns = self.encoder(inputs[[0]], inputs + type_embedding, inputs, mask)
        outputs = self.output_embed(out[0]).view(-1, self.num_timesteps, self.num_outputs)
        return outputs, attns


# ----------------------------- planning/vectorized/local_graph.py -----------------------------


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        """A positional embedding module.
        Useful to inject the position of sequence elements in local graphs

        :param d_model: feature size
        :param max_len: max length of the sequences, defaults to 5000
        """
        super().__init__()

        # Positional Encoder
        pe = torch.zeros(max_len, d_model)
        t = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        log_value = torch.log(torch.tensor([1e4])).item()
        omega = torch.exp((-log_value / d_model) * torch.arange(0, d_model, 2).float())
        pe[:, 0::2] = torch.sin(t * omega)
        pe[:, 1::2] = torch.cos(t * omega)
        self.register_buffer("static_embedding", pe.unsqueeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input tensor of shape batch_size x num_agents x sequence_length x d_model
        """
        return self.static_embedding[: x.shape[2], :]


class LocalMLP(nn.Module):
    def __init__(self, dim_in: int, use_norm: bool = True):
        """a Local 1 layer MLP

        :param dim_in: feat in size
        :param use_norm: if to apply layer norm, defaults to True
        """
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_in, bias=not use_norm)
        self.use_norm = use_norm
        if use_norm:
            self.norm = nn.LayerNorm(dim_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward of the module

        :param x: input tensor (..., dim_in)
        :return: output tensor (..., dim_in)
        """
        x = self.linear(x)
        if hasattr(self, "norm"):
            x = self.norm(x)
        x = F.relu(x, inplace=True)
        return x


class LocalSubGraphLayer(nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        """Local subgraph layer

        :param dim_in: input feat size
        :param dim_out: output feat size
        """
        super(LocalSubGraphLayer, self).__init__()
        self.mlp = LocalMLP(dim_in)
        self.linear_remap = nn.Linear(dim_in * 2, dim_out)

    def forward(self, x: torch.Tensor, invalid_mask: torch.Tensor) -> torch.Tensor:
        """Forward of the model

        :param x: input tensor (B,N,P,dim_in)
        :param invalid_mask: invalid mask for x (B,N,P)
        :return: output tensor (B,N,P,dim_out)
        """
        # x input -> polys * num_vectors * embedded_vector_length
        _, num_vectors, _ = x.shape
        # x mlp -> polys * num_vectors * dim_in
        x = self.mlp(x)
        # compute the masked max for each feature in the sequence

        masked_x = x.masked_fill(invalid_mask[..., None] > 0, float("-inf"))
        x_agg = masked_x.max(dim=1, keepdim=True).values
        # repeat it along the sequence length
        x_agg = x_agg.repeat(1, num_vectors, 1)
        x = torch.cat([x, x_agg], dim=-1)
        x = self.linear_remap(x)  # remap to a possibly different feature length
        return x


class LocalSubGraph(nn.Module):
    def __init__(self, num_layers: int, dim_in: int) -> None:
        """PointNet-like local subgraph - implemented as a collection of local graph layers

        :param num_layers: number of LocalSubGraphLayer
        :param dim_in: input, hidden, output dim for features
        """
        super(LocalSubGraph, self).__init__()
        assert num_layers > 0
        self.layers = nn.ModuleList()
        self.dim_in = dim_in
        for _ in range(num_layers):
            self.layers.append(LocalSubGraphLayer(dim_in, dim_in))

    def forward(
        self, x: torch.Tensor, invalid_mask: torch.Tensor, pos_enc: torch.Tensor
    ) -> torch.Tensor:
        """Forward of the module:
        - Add positional encoding
        - Forward to layers
        - Aggregates using max
        (calculates a feature descriptor per element - reduces over points)

        :param x: input tensor (B,N,P,dim_in)
        :param invalid_mask: invalid mask for x (B,N,P)
        :param pos_enc: positional_encoding for x
        :return: output tensor (B,N,P,dim_in)
        """
        batch_size, polys_num, seq_len, vector_size = x.shape

        x += pos_enc
        # exclude completely invalid sequences from local subgraph to avoid NaN in weights
        x_flat = x.view(-1, seq_len, vector_size)
        invalid_mask_flat = invalid_mask.view(-1, seq_len)
        # (batch_size x (1 + M),)
        valid_polys = ~invalid_mask.all(-1).flatten()
        # valid_seq x seq_len x vector_size
        x_to_process = x_flat[valid_polys]
        mask_to_process = invalid_mask_flat[valid_polys]
        for layer in self.layers:
            x_to_process = layer(x_to_process, mask_to_process)

        # aggregate sequence features
        x_to_process = x_to_process.masked_fill(mask_to_process[..., None] > 0, float("-inf"))
        # valid_seq x vector_size
        x_to_process = torch.max(x_to_process, dim=1).values

        # restore back the batch
        x = torch.zeros_like(x_flat[:, 0])
        x[valid_polys] = x_to_process
        x = x.view(batch_size, polys_num, self.dim_in)
        return x


# ----------------------------- planning/vectorized/open_loop_model.py -----------------------------


class VectorizedModel(nn.Module):
    """Vectorized planning model (UrbanDriver base architecture)."""

    def __init__(
        self,
        history_num_frames_ego: int,
        history_num_frames_agents: int,
        num_targets: int,
        weights_scaling: List[float],
        criterion: nn.Module,
        global_head_dropout: float,
        disable_other_agents: bool,
        disable_map: bool,
        disable_lane_boundaries: bool,
    ) -> None:
        """Initializes the model.

        :param history_num_frames_ego: number of history ego frames to include
        :param history_num_frames_agents: number of history agent frames to include
        :param num_targets: number of values to predict
        :param weights_scaling: target weights for loss calculation
        :param global_head_dropout: float in range [0,1] for the dropout in the MHA global head. Set to 0 to disable it
        :param criterion: loss function to use
        :param disable_other_agents: ignore agents
        :param disable_map: ignore map
        :param disable_lane_boundaries: ignore lane boundaries
        """
        super().__init__()
        self.disable_map = disable_map
        self.disable_other_agents = disable_other_agents
        self.disable_lane_boundaries = disable_lane_boundaries

        self._history_num_frames_ego = history_num_frames_ego
        self._history_num_frames_agents = history_num_frames_agents
        self._num_targets = num_targets

        self._global_head_dropout = global_head_dropout

        self._d_local = 256
        self._d_global = 256

        self._agent_features = ["start_x", "start_y", "yaw"]
        self._lane_features = ["start_x", "start_y", "tl_feature"]
        self._vector_agent_length = len(self._agent_features)
        self._vector_lane_length = len(self._lane_features)
        self._subgraph_layers = 3

        self.register_buffer("weights_scaling", torch.as_tensor(weights_scaling))
        self.criterion = criterion

        self.normalize_targets = True
        num_outputs = len(weights_scaling)
        num_timesteps = num_targets // num_outputs

        if self.normalize_targets:
            scale = build_target_normalization(num_timesteps)
            self.register_buffer("xy_scale", scale)

        # normalization buffers
        self.register_buffer("agent_std", torch.tensor([1.6919, 0.0365, 0.0218]))
        self.register_buffer("other_agent_std", torch.tensor([33.2631, 21.3976, 1.5490]))

        self.input_embed = nn.Linear(self._vector_agent_length, self._d_local)
        self.positional_embedding = SinusoidalPositionalEmbedding(self._d_local)
        self.type_embedding = VectorizedEmbedding(self._d_global)

        self.disable_pos_encode = False

        self.local_subgraph = LocalSubGraph(num_layers=self._subgraph_layers, dim_in=self._d_local)

        if self._d_global != self._d_local:
            self.global_from_local = nn.Linear(self._d_local, self._d_global)

        self.global_head = MultiheadAttentionGlobalHead(
            self._d_global, num_timesteps, num_outputs, dropout=self._global_head_dropout
        )

    def embed_polyline(
        self, features: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embeds the inputs, generates the positional embedding and calls the local subgraph.

        :param features: input features
        :tensor features: [batch_size, num_elements, max_num_points, max_num_features]
        :param mask: availability mask
        :tensor mask: [batch_size, num_elements, max_num_points]

        :return tuple of local subgraphout output, (in-)availability mask
        """
        # embed inputs
        # [batch_size, num_elements, max_num_points, embed_dim]
        polys = self.input_embed(features)
        # calculate positional embedding
        # [1, 1, max_num_points, embed_dim]
        pos_embedding = self.positional_embedding(features).unsqueeze(0).transpose(1, 2)
        # [batch_size, num_elements, max_num_points]
        invalid_mask = ~mask
        invalid_polys = invalid_mask.all(-1)
        # input features to local subgraoh and return result -
        # local subgraph reduces features over elements, i.e. creates one descriptor
        # per element
        # [batch_size, num_elements, embed_dim]
        polys = self.local_subgraph(polys, invalid_mask, pos_embedding)
        return polys, invalid_polys

    def model_call(
        self,
        agents_polys: torch.Tensor,
        static_polys: torch.Tensor,
        agents_avail: torch.Tensor,
        static_avail: torch.Tensor,
        type_embedding: torch.Tensor,
        lane_bdry_len: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encapsulates calling the global_head (TODO?) and preparing needed data.

        :param agents_polys: dynamic elements - i.e. vectors corresponding to agents
        :param static_polys: static elements - i.e. vectors corresponding to map elements
        :param agents_avail: availability of agents
        :param static_avail: availability of map elements
        :param type_embedding:
        :param lane_bdry_len:
        """
        # Standardize inputs
        agents_polys_feats = torch.cat(
            [agents_polys[:, :1] / self.agent_std, agents_polys[:, 1:] / self.other_agent_std],
            dim=1,
        )
        static_polys_feats = static_polys / self.other_agent_std

        all_polys = torch.cat([agents_polys_feats, static_polys_feats], dim=1)
        all_avail = torch.cat([agents_avail, static_avail], dim=1)

        # Embed inputs, calculate positional embedding, call local subgraph
        all_embs, invalid_polys = self.embed_polyline(all_polys, all_avail)
        if hasattr(self, "global_from_local"):
            all_embs = self.global_from_local(all_embs)

        all_embs = F.normalize(all_embs, dim=-1) * (self._d_global**0.5)
        all_embs = all_embs.transpose(0, 1)

        other_agents_len = agents_polys.shape[1] - 1

        # disable certain elements on demand
        if self.disable_other_agents:
            invalid_polys[:, 1 : (1 + other_agents_len)] = 1  # agents won't create attention

        if self.disable_map:  # lanes (mid), crosswalks, and lanes boundaries.
            invalid_polys[:, (1 + other_agents_len) :] = 1  # lanes won't create attention

        if self.disable_lane_boundaries:
            type_embedding = type_embedding[:-lane_bdry_len]

        invalid_polys[:, 0] = 0  # make AoI always available in global graph

        # call and return global graph
        outputs, attns = self.global_head(all_embs, type_embedding, invalid_polys)
        return outputs, attns

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Load and prepare vectors for the model call, split into map and agents

        # ==== LANES ====
        # batch size x num lanes x num vectors x num features
        polyline_keys = ["lanes_mid", "crosswalks"]
        if not self.disable_lane_boundaries:
            polyline_keys += ["lanes"]
        avail_keys = [f"{k}_availabilities" for k in polyline_keys]

        max_num_vectors = max([data_batch[key].shape[-2] for key in polyline_keys])

        map_polys = torch.cat(
            [pad_points(data_batch[key], max_num_vectors) for key in polyline_keys], dim=1
        )
        map_polys[..., -1].fill_(0)
        # batch size x num lanes x num vectors
        map_availabilities = torch.cat(
            [pad_avail(data_batch[key], max_num_vectors) for key in avail_keys], dim=1
        )

        # ==== AGENTS ====
        # batch_size x (1 + M) x seq len x self._vector_length
        agents_polys = torch.cat(
            [
                data_batch["agent_trajectory_polyline"].unsqueeze(1),
                data_batch["other_agents_polyline"],
            ],
            dim=1,
        )
        # batch_size x (1 + M) x num vectors x self._vector_length
        agents_polys = pad_points(agents_polys, max_num_vectors)

        # batch_size x (1 + M) x seq len
        agents_availabilities = torch.cat(
            [
                data_batch["agent_polyline_availability"].unsqueeze(1),
                data_batch["other_agents_polyline_availability"],
            ],
            dim=1,
        )
        # batch_size x (1 + M) x num vectors
        agents_availabilities = pad_avail(agents_availabilities, max_num_vectors)

        # batch_size x (1 + M) x num features
        type_embedding = self.type_embedding(data_batch).transpose(0, 1)
        lane_bdry_len = data_batch["lanes"].shape[1]

        # call the model with these features
        outputs, attns = self.model_call(
            agents_polys,
            map_polys,
            agents_availabilities,
            map_availabilities,
            type_embedding,
            lane_bdry_len,
        )

        # calculate loss or return predicted position for inference
        if self.training:
            if self.criterion is None:
                raise NotImplementedError("Loss function is undefined.")

            xy = data_batch["target_positions"]
            yaw = data_batch["target_yaws"]
            if self.normalize_targets:
                xy /= self.xy_scale
            targets = torch.cat((xy, yaw), dim=-1)
            target_weights = (
                data_batch["target_availabilities"].unsqueeze(-1) * self.weights_scaling
            )
            loss = torch.mean(self.criterion(outputs, targets) * target_weights)
            train_dict = {"loss": loss}
            return train_dict
        else:
            pred_positions, pred_yaws = outputs[..., :2], outputs[..., 2:3]
            if self.normalize_targets:
                pred_positions *= self.xy_scale

            eval_dict = {"positions": pred_positions, "yaws": pred_yaws}
            if attns is not None:
                eval_dict["attention_weights"] = attns
            return eval_dict


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo)
# ---------------------------------------------------------------------------


def build_urbandriver():
    # Real config from l5kit/l5kit/tests/artefacts/config_vectorized.yaml (model_params).
    weights_scaling = [1.0, 1.0, 1.0]
    future_num_frames = 12
    num_targets = len(weights_scaling) * future_num_frames
    model = VectorizedModel(
        history_num_frames_ego=0,
        history_num_frames_agents=3,
        num_targets=num_targets,
        weights_scaling=weights_scaling,
        criterion=nn.L1Loss(reduction="none"),
        global_head_dropout=0.0,
        disable_other_agents=False,
        disable_map=False,
        disable_lane_boundaries=True,
    )
    model.eval()
    return model


def example_input_urbandriver():
    # Mirrors l5kit/l5kit/tests/planning/common_test.py::mock_vectorizer_data, sized down
    # from the real test config (config_vectorized.yaml's data_generation_params).
    batch_size = 2
    num_steps = 12  # future_num_frames
    num_history = 3  # max(history_num_frames_ego=0, history_num_frames_agents=3)
    num_agents = 4  # other_agents_num (real config: 30)
    num_lanes = 3  # max_num_lanes (real config: 30)
    num_crosswalks = 2  # max_num_crosswalks (real config: 20)
    num_points_per_element = 4  # max_points_per_lane/crosswalk (real config: 20)
    type_max = 16  # PERCEPTION_LABELS has 17 entries (indices 0..16)

    data_batch = {
        "type": torch.randint(0, type_max, (batch_size,)),
        "target_positions": torch.rand(batch_size, num_steps, 2),
        "target_yaws": torch.rand(batch_size, num_steps, 1),
        "target_availabilities": torch.rand(batch_size, num_steps) > 0.5,
        "all_other_agents_types": torch.randint(0, type_max, (batch_size, num_agents)),
        "agent_trajectory_polyline": torch.rand(batch_size, num_history + 1, 3),
        "agent_polyline_availability": torch.rand(batch_size, num_history + 1) > 0.5,
        "other_agents_polyline": torch.rand(batch_size, num_agents, num_history + 1, 3),
        "other_agents_polyline_availability": torch.rand(batch_size, num_agents, num_history + 1)
        > 0.5,
        "lanes": torch.rand(batch_size, num_lanes, num_points_per_element, 3),
        "lanes_availabilities": torch.rand(batch_size, num_lanes, num_points_per_element) > 0.5,
        "lanes_mid": torch.rand(batch_size, num_lanes, num_points_per_element, 3),
        "lanes_mid_availabilities": torch.rand(batch_size, num_lanes, num_points_per_element) > 0.5,
        "crosswalks": torch.rand(batch_size, num_crosswalks, num_points_per_element, 3),
        "crosswalks_availabilities": torch.rand(batch_size, num_crosswalks, num_points_per_element)
        > 0.5,
    }
    return (data_batch,)


MENAGERIE_ENTRIES = [
    ("urbandriver", build_urbandriver, example_input_urbandriver, 2021, "SOURCE_AVAILABLE"),
]
