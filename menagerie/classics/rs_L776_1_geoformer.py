# SOURCE: vendored from microsoft/AI2BMD (Geoformer branch) @ Geoformer
# Files: geoformer/model/configuration_geoformer.py, geoformer/model/modeling_geoformer_layers.py,
#        geoformer/model/modeling_priors.py, geoformer/model/modeling_geoformer.py
# https://github.com/microsoft/AI2BMD/tree/Geoformer/geoformer
#
# Geoformer: an equivariant Transformer for molecular property prediction using an
# "interatomic positional encoding" (IPE) derived from Atomic Cluster Expansion (ACE)
# theory, built on a HuggingFace `PreTrainedModel` base (NeurIPS 2023).
#
# Import-fix only (per rung-2 rules, architecture code is untouched):
#   - `geoformer.datasets.__init__` eagerly imports both `QM9` and `Molecule3D`; the
#     latter needs `rdkit` (not installed here) and is never referenced by the model
#     itself (only used, optionally, to build a `prior_model`). We import `QM9`
#     directly from `geoformer.datasets.qm9` (needs only `torch_geometric`, which is
#     installed) to avoid pulling in the unused, rdkit-gated `Molecule3D` class.
#   - `geoformer.model.__init__.py` upstream is an empty file; nothing to preserve.
#   - The vendored files below are otherwise verbatim (only relative imports were
#     rewritten to import from this single staging module).

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Optional

import ase
import torch
from einops import rearrange, repeat
from torch import nn
from transformers import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

MENAGERIE_ZOO = "vendored-pytorch"


# --- geoformer/model/configuration_geoformer.py (verbatim) ---------------------------


class GeoformerConfig(PretrainedConfig):
    model_type = "geoformer"

    def __init__(
        self,
        max_z: int = 100,
        embedding_dim: int = 512,
        ffn_embedding_dim: int = 2048,
        num_layers: int = 9,
        num_attention_heads: int = 8,
        cutoff: int = 5.0,
        num_rbf: int = 64,
        rbf_trainable: bool = True,
        norm_type: str = "max_min",
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation_dropout: float = 0.0,
        activation_function: str = "silu",
        decoder_type: str = "scalar",
        aggr="sum",
        dataset_root=None,
        dataset_arg=None,
        mean=None,
        std=None,
        prior_model=None,
        num_classes: int = 1,
        pad_token_id: int = 0,
        **kwargs,
    ):
        self.max_z = max_z
        self.embedding_dim = embedding_dim
        self.ffn_embedding_dim = ffn_embedding_dim
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.rbf_trainable = rbf_trainable
        self.norm_type = norm_type
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.decoder_type = decoder_type
        self.aggr = aggr
        self.dataset_root = dataset_root
        self.dataset_arg = dataset_arg
        self.mean = mean
        self.std = std
        self.prior_model = prior_model
        self.num_classes = num_classes

        super(GeoformerConfig, self).__init__(pad_token_id=pad_token_id, **kwargs)


# --- geoformer/model/modeling_geoformer_layers.py (verbatim) -------------------------

import math  # noqa: E402

act_class_mapping = {"silu": nn.SiLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}


class CosineCutoff(nn.Module):
    def __init__(self, cutoff):
        super(CosineCutoff, self).__init__()

        self.cutoff = cutoff

    def forward(self, distances):
        cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff).float()
        return cutoffs


class ExpNormalSmearing(nn.Module):
    def __init__(self, cutoff=5.0, num_rbf=50, trainable=True):
        super(ExpNormalSmearing, self).__init__()

        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(cutoff)
        self.alpha = 5.0 / cutoff

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        start_value = torch.exp(torch.scalar_tensor(-self.cutoff))
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor([(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf)
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(
            -self.betas * (torch.exp(self.alpha * (-dist)) - self.means) ** 2
        )


class VecLayerNorm(nn.Module):
    def __init__(self, hidden_channels, trainable, norm_type="max_min"):
        super(VecLayerNorm, self).__init__()

        self.hidden_channels = hidden_channels
        self.eps = 1e-6

        weight = torch.ones(self.hidden_channels)
        if trainable:
            self.register_parameter("weight", nn.Parameter(weight))
        else:
            self.register_buffer("weight", weight)

        if norm_type == "max_min":
            self.norm = self.max_min_norm
        else:
            self.norm = self.none_norm

        self.reset_parameters()

    def reset_parameters(self):
        weight = torch.ones(self.hidden_channels)
        self.weight.data.copy_(weight)

    def none_norm(self, vec):
        return vec

    def max_min_norm(self, vec):
        # vec: (B, N, 3 or 5, hidden_channels)
        dist = torch.norm(vec, dim=-2, keepdim=True)

        if (dist == 0).all():
            return torch.zeros_like(vec)

        dist = dist.clamp(min=self.eps)
        direct = vec / dist

        max_val, _ = torch.max(dist, dim=-1)
        min_val, _ = torch.min(dist, dim=-1)
        # delta: (B, N, 1)
        delta = max_val - min_val
        delta = torch.where(delta == 0, torch.ones_like(delta), delta)
        dist = (dist - min_val.unsqueeze(-1)) / delta.unsqueeze(-1)

        return dist * direct

    def forward(self, vec):
        # vec: (num_atoms, 3 or 8, hidden_channels)
        if vec.shape[-2] == 3:
            vec = self.norm(vec)
            return vec * self.weight.view(1, 1, 1, -1)
        elif vec.shape[-2] == 8:
            vec1, vec2 = torch.split(vec, [3, 5], dim=-2)
            vec1 = self.norm(vec1)
            vec2 = self.norm(vec2)
            vec = torch.cat([vec1, vec2], dim=-2)
            return vec * self.weight.view(1, 1, 1, -1)
        else:
            raise ValueError("VecLayerNorm only support 3 or 8 channels")


# --- geoformer/model/modeling_priors.py (verbatim) -----------------------------------


class BasePrior(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super(BasePrior, self).__init__()

    @abstractmethod
    def get_init_args(self):
        return

    @abstractmethod
    def forward(self, x, z):
        return


class Atomref(BasePrior):
    """
    Atomref prior model.
    When using this in combination with some dataset, the dataset class must implement
    the function `get_atomref`, which returns the atomic reference values as a tensor.
    """

    def __init__(self, max_z=None, utils=None):
        super(Atomref, self).__init__()
        if max_z is None and utils is None:
            raise ValueError("Can't instantiate Atomref prior, all arguments are None.")
        if utils is None:
            atomref = torch.zeros(max_z, 1)
        else:
            atomref = utils.get_atomref()
            if atomref is None:
                print(
                    "The atomref returned by the dataset is None, defaulting to zeros with max. "
                    "atomic number 99. Maybe atomref is not defined for the current target."
                )
                atomref = torch.zeros(100, 1)

        if atomref.ndim == 1:
            atomref = atomref.view(-1, 1)
        self.register_buffer("initial_atomref", atomref)
        self.atomref = nn.Embedding(len(atomref), 1)
        self.atomref.weight.data.copy_(atomref)

    def reset_parameters(self):
        self.atomref.weight.data.copy_(self.initial_atomref)

    def get_init_args(self):
        return dict(max_z=self.initial_atomref.size(0))

    def forward(self, x, z):
        return x + self.atomref(z)


__all_priors__ = ["Atomref"]


class _ModelingPriorsModule:
    """Stand-in for the `geoformer.model.modeling_priors` module object.

    The real `GeoformerForEnergyRegression._register_prior_model` does
    `getattr(modeling_priors, self.config.prior_model)` and reads
    `modeling_priors.__all__`; we keep both attributes so that code path
    (unused when `prior_model=None`, our default) still resolves correctly.
    """

    Atomref = Atomref
    __all__ = __all_priors__


modeling_priors = _ModelingPriorsModule()


# --- geoformer/model/modeling_geoformer.py (verbatim, QM9 import narrowed) -----------


class GeoformerMultiHeadAttention(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super(GeoformerMultiHeadAttention, self).__init__(*args, **kwargs)

        self.embedding_dim = config.embedding_dim
        self.num_heads = config.num_attention_heads
        self.head_dim = config.embedding_dim // config.num_attention_heads
        if not (self.head_dim * config.num_attention_heads == self.embedding_dim):
            raise AssertionError("The embedding_dim must be divisible by num_heads.")

        self.act = act_class_mapping[config.activation_function]()
        self.cutoff = CosineCutoff(config.cutoff)

        self.dropout_module = nn.Dropout(p=config.attention_dropout, inplace=False)

        self.k_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.q_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.v_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.dk_proj = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.du_update_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.du_norm = VecLayerNorm(self.embedding_dim, trainable=False, norm_type=config.norm_type)
        self.dihedral_proj = nn.Linear(self.embedding_dim, 2 * self.embedding_dim, bias=False)
        self.edge_attr_update = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.dk_proj.weight)
        self.dk_proj.bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.du_update_proj.weight)
        self.du_update_proj.bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.dihedral_proj.weight)
        nn.init.xavier_uniform_(self.edge_attr_update.weight)
        self.edge_attr_update.bias.data.fill_(0.0)

    def forward(
        self,
        x: torch.Tensor,  # (B, N, F)
        vec: Optional[torch.Tensor],  # (B, N, N, 3)
        dist: Optional[torch.Tensor],  # (B, N, N)
        edge_attr: Optional[torch.Tensor],  # (B, N, N, F)
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, N)
        **kwargs,
    ):
        q = rearrange(self.q_proj(x), "b n (h d) -> (b h) n d", h=self.num_heads)  # (BH, N, D)
        k = rearrange(self.k_proj(x), "b n (h d) -> (b h) n d", h=self.num_heads)  # (BH, N, D)
        v = rearrange(self.v_proj(x), "b n (h d) -> (b h) n d", h=self.num_heads)  # (BH, N, D)
        dk = rearrange(
            self.act(self.dk_proj(edge_attr)),
            "b n m (h d) -> (b h) n m d",
            h=self.num_heads,
        )  # (BH, N, N, D)

        attn_weights = ((q.unsqueeze(-2) * k.unsqueeze(-3)) * dk).sum(dim=-1)  # (BH, N, N)

        if key_padding_mask is not None:
            attn_weights = rearrange(attn_weights, "(b h) n m -> b h n m", h=self.num_heads)
            attn_weights = attn_weights.masked_fill(
                rearrange(key_padding_mask, "b n m -> b () n m"),
                0.0,
            )
            attn_weights = rearrange(attn_weights, "b h n m -> (b h) n m")

        attn_scale = repeat(self.cutoff(dist), "b n m -> b h n m", h=self.num_heads)  # (BH, N, N)
        attn_scale = rearrange(attn_scale, "b h n m -> (b h) n m", h=self.num_heads)  # (BH, N, N)
        attn_probs = self.act(attn_weights) * attn_scale  # (BH, N, N)

        attn_per_nodes = attn_probs.unsqueeze(-1) * v.unsqueeze(-3)  # (BH, N, N, D)
        attn_per_nodes = rearrange(
            attn_per_nodes, "(b h) n m d -> b n m (h d)", h=self.num_heads
        )  # (B, N, N, F)
        attn = attn_per_nodes.sum(dim=2)  # (B, N, F)

        du = (
            self.du_update_proj(attn_per_nodes)
            .masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
            .unsqueeze(-2)
            * vec.unsqueeze(-1)
        ).sum(dim=-3)  # (B, N, 3, F)
        du = self.du_norm(du)  # (B, N, 3, F)
        ws, wt = torch.split(self.dihedral_proj(du), self.embedding_dim, dim=-1)  # (B, N, 3, F)
        ipe = (wt.unsqueeze(1) * ws.unsqueeze(2)).sum(dim=-2)  # (B, N, N, F)
        ipe = self.act(self.edge_attr_update(edge_attr)) * ipe  # (B, N, N, F)

        return attn, ipe


class GeoformerAttnBlock(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super(GeoformerAttnBlock, self).__init__(*args, **kwargs)

        self.embedding_dim = config.embedding_dim
        self.dropout_module = nn.Dropout(p=config.dropout, inplace=False)

        self.act = act_class_mapping[config.activation_function]()

        self.self_attn = GeoformerMultiHeadAttention(config)

        self.ffn = nn.Sequential(
            nn.Linear(self.embedding_dim, config.ffn_embedding_dim),
            self.act,
            nn.Dropout(p=config.activation_dropout, inplace=False),
            nn.Linear(config.ffn_embedding_dim, self.embedding_dim),
        )

        self.attn_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.self_attn.reset_parameters()
        nn.init.xavier_uniform_(self.ffn[0].weight)
        self.ffn[0].bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.ffn[3].weight)
        self.ffn[3].bias.data.fill_(0.0)
        self.attn_layer_norm.reset_parameters()
        self.final_layer_norm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,  # (B, N, F)
        vec: torch.Tensor,  # (B, N, N, 3)
        dist: torch.Tensor,  # (B, N, N)
        edge_attr: torch.Tensor,  # (B, N, N, ?)
        key_padding_mask: Optional[torch.Tensor],  # [padding, cutoff] (B, N, N)
        **kwargs,
    ):
        # attention
        dx, dedge_attr = x, edge_attr
        x, edge_attr = self.self_attn(
            x=x,
            vec=vec,
            dist=dist,
            edge_attr=edge_attr,
            key_padding_mask=key_padding_mask,
        )

        x = self.dropout_module(x)
        x = x + dx
        x = self.attn_layer_norm(x)

        # ipe update
        edge_attr = edge_attr + dedge_attr

        # ffn
        dx = x
        x = self.ffn(x)
        x = self.dropout_module(x)
        x = x + dx

        x = self.final_layer_norm(x)

        return x, edge_attr


class GeoformerEncoder(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super(GeoformerEncoder, self).__init__(*args, **kwargs)

        self.pad_token_id = config.pad_token_id
        self.embedding_dim = config.embedding_dim
        self.cutoff = config.cutoff

        self.embedding = nn.Embedding(
            config.max_z, self.embedding_dim, padding_idx=self.pad_token_id
        )
        self.distance_expansion = ExpNormalSmearing(
            cutoff=config.cutoff,
            num_rbf=config.num_rbf,
            trainable=config.rbf_trainable,
        )
        self.dist_proj = nn.Linear(config.num_rbf, self.embedding_dim)
        self.act = act_class_mapping[config.activation_function]()

        self.layers = nn.ModuleList([GeoformerAttnBlock(config) for _ in range(config.num_layers)])

        self.x_in_layernorm = nn.LayerNorm(self.embedding_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.distance_expansion.reset_parameters()
        nn.init.xavier_uniform_(self.dist_proj.weight)
        self.dist_proj.bias.data.fill_(0.0)
        for layer in self.layers:
            layer.reset_parameters()
        self.x_in_layernorm.reset_parameters()

    def forward(
        self,
        z: torch.Tensor,  # (B, N)
        pos: torch.Tensor,  # (B, N, 3)
        **kwargs,
    ):
        B, N, *_ = z.shape
        # generate mask
        padding_mask = z == self.pad_token_id  # (B, N)
        pos_mask = ~(padding_mask.unsqueeze(1) | padding_mask.unsqueeze(2))  # (B, N, N)
        dist = torch.norm(pos.unsqueeze(1) - pos.unsqueeze(2), dim=-1)  # (B, N, N)
        loop_mask = torch.eye(N, dtype=torch.bool, device=dist.device)
        loop_mask = repeat(loop_mask, "n m -> b n m", b=B)  # (B, N, N)
        dist = dist.masked_fill(loop_mask, 0.0)  # (B, N, N)
        adj_mask = (dist < self.cutoff) & pos_mask  # (B, N, N)
        loop_adj_mask = ~loop_mask & adj_mask  # (B, N, N)

        vec = (pos.unsqueeze(1) - pos.unsqueeze(2)) / (dist.unsqueeze(-1) + 1e-8)  # (B, N, N, 3)
        vec = vec.masked_fill(~loop_adj_mask.unsqueeze(-1), 0.0)  # (B, N, N, 3)

        key_padding_mask = (
            (~adj_mask)
            .masked_fill(padding_mask.unsqueeze(-1), False)
            .masked_fill(padding_mask.unsqueeze(-2), True)
        )

        x = self.embedding(z)  # (B, N, F)
        x = self.x_in_layernorm(x)
        edge_attr = self.distance_expansion(dist)  # (B, N, N, num_rbf)
        edge_attr = self.act(self.dist_proj(edge_attr))  # (B, N, N, F)
        edge_attr = edge_attr.masked_fill(~adj_mask.unsqueeze(-1), 0.0)  # (B, N, N, F)

        for layer in self.layers:
            x, edge_attr = layer(
                x=x,
                vec=vec,
                dist=dist,
                edge_attr=edge_attr,
                key_padding_mask=key_padding_mask,
            )

        return x, edge_attr


class GeoformerScalarDecoder(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super(GeoformerScalarDecoder, self).__init__(*args, **kwargs)

        self.embedding_dim = config.embedding_dim
        self.num_classes = config.num_classes
        self.act = act_class_mapping[config.activation_function]()
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            self.act,
            nn.Linear(self.embedding_dim // 2, self.num_classes),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.classifier[0].weight)
        self.classifier[0].bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.classifier[2].weight)
        self.classifier[2].bias.data.fill_(0.0)

    def forward(
        self,
        x: torch.Tensor,  # (B, N, F)
        edge_attr: torch.Tensor,  # (B, N, N, F)
        **kwargs,
    ):
        return self.classifier(x) + edge_attr.sum() * 0


class GeoformerDipoleMomentDecoder(GeoformerScalarDecoder):
    def __init__(self, config, *args, **kwargs):
        super(GeoformerDipoleMomentDecoder, self).__init__(config, *args, **kwargs)
        atomic_mass = torch.from_numpy(ase.data.atomic_masses).float()
        self.register_buffer("atomic_mass", atomic_mass)

    def forward(
        self,
        x: torch.Tensor,  # (B, N, F)
        edge_attr: torch.Tensor,  # (B, N, N, F)
        **kwargs,
    ):
        x = self.classifier(x) + edge_attr.sum() * 0  # (B, N, 1)

        # Get center of mass.
        z = kwargs["z"]  # (B, N)
        pos = kwargs["pos"]  # (B, N, 3)
        padding_mask = kwargs["padding_mask"]  # (B, N)
        mass = self.atomic_mass[z].masked_fill(padding_mask, 0.0).unsqueeze(-1)  # (B, N, 1)
        c = torch.sum(mass * pos, dim=-2) / torch.sum(mass, dim=-2)
        x = x * (pos - c.unsqueeze(-2))
        return x  # (B, N, 3)


class GeoformerElectronicSpatialExtentDecoder(GeoformerScalarDecoder):
    def __init__(self, config, *args, **kwargs):
        super(GeoformerElectronicSpatialExtentDecoder, self).__init__(config, *args, **kwargs)
        atomic_mass = torch.from_numpy(ase.data.atomic_masses).float()
        self.register_buffer("atomic_mass", atomic_mass)

    def forward(
        self,
        x: torch.Tensor,  # (B, N, F)
        edge_attr: torch.Tensor,  # (B, N, N, F)
        **kwargs,
    ):
        x = self.classifier(x) + edge_attr.sum() * 0  # (B, N, 1)

        # Get center of mass.
        z = kwargs["z"]  # (B, N)
        pos = kwargs["pos"]  # (B, N, 3)
        padding_mask = kwargs["padding_mask"]  # (B, N)
        mass = self.atomic_mass[z].masked_fill(padding_mask, 0.0).unsqueeze(-1)  # (B, N, 1)
        c = torch.sum(mass * pos, dim=-2) / torch.sum(mass, dim=-2)
        x = torch.norm(pos - c.unsqueeze(-2), dim=-1, keepdim=True) ** 2 * x
        return x  # (B, N, 1)


class GeoformerModel(PreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super(GeoformerModel, self).__init__(config, *inputs, **kwargs)

        self.geo_encoder = GeoformerEncoder(config)
        if config.decoder_type == "scalar":
            self.geo_decoder = GeoformerScalarDecoder(config)
        elif config.decoder_type == "dipole_moment":
            self.geo_decoder = GeoformerDipoleMomentDecoder(config)
        elif config.decoder_type == "electronic_spatial_extent":
            self.geo_decoder = GeoformerElectronicSpatialExtentDecoder(config)
        else:
            raise ValueError(f"Unknown decoder type: {config.decoder_type}")

        self.post_init()

    def init_weights(self):
        self.geo_encoder.reset_parameters()
        self.geo_decoder.reset_parameters()


class GeoformerForEnergyRegression(GeoformerModel):
    def __init__(self, config, *inputs, **kwargs):
        super(GeoformerForEnergyRegression, self).__init__(config, *inputs, **kwargs)

        self.config = config
        self.aggr = config.aggr
        self.pad_token_id = config.pad_token_id
        self.prior_model = self._register_prior_model()
        mean = torch.scalar_tensor(0) if config.mean is None else config.mean
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean).float()
        self.register_buffer("mean", mean)
        std = torch.scalar_tensor(1) if config.std is None else config.std
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std).float()
        self.register_buffer("std", std)

    def _register_prior_model(self):
        prior_model = None
        if self.config.prior_model is not None:
            assert hasattr(modeling_priors, self.config.prior_model), (
                f"Unknown prior model {self.config.prior_model}. "
                f"Available models are {', '.join(modeling_priors.__all__)}"
            )
            # `QM9` is imported lazily here (only when a prior_model is actually
            # requested) to keep the rdkit-gated `Molecule3D` class out of the
            # default import path -- see module-level note at the top of this file.
            from geoformer.datasets.qm9 import QM9

            # initialize the prior model
            prior_model = getattr(modeling_priors, self.config.prior_model)(
                utils=QM9(
                    root=self.config.dataset_root,
                    dataset_arg=self.config.dataset_arg,
                )
            )
        return prior_model

    def forward(
        self,
        z: torch.Tensor,  # (B, N)
        pos: torch.Tensor,  # (B, N, 3)
        **kwargs,
    ):
        x, edge_attr = self.geo_encoder(z=z, pos=pos)

        padding_mask = z == self.pad_token_id  # (B, N)

        # (B, N, 1) or (B, N, 3)
        x = self.geo_decoder(x=x, edge_attr=edge_attr, z=z, pos=pos, padding_mask=padding_mask)

        logits = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)  # (B, N, 1)

        if self.std is not None:
            logits = logits * self.std

        logits = self.prior_model(logits, z) if self.prior_model is not None else logits

        if self.aggr == "sum":
            logits = logits.sum(dim=1)  # (B, 1)
        elif self.aggr == "mean":
            logits = logits.sum(dim=1) / (~padding_mask).sum(dim=-1).unsqueeze(-1)  # (B, 1)
        else:
            NotImplementedError(f"Unknown aggregation method: {self.aggr}")

        if self.config.decoder_type == "dipole_moment":
            logits = torch.norm(logits, dim=-1, keepdim=True)

        if self.mean is not None:
            logits = logits + self.mean

        return logits


# --- staging entry points -------------------------------------------------------------


def build_geoformer():
    """Tiny random-init GeoformerForEnergyRegression (scalar decoder, no prior)."""
    config = GeoformerConfig(
        max_z=20,
        embedding_dim=32,
        ffn_embedding_dim=64,
        num_layers=2,
        num_attention_heads=4,
        cutoff=5.0,
        num_rbf=16,
        rbf_trainable=True,
        norm_type="max_min",
        decoder_type="scalar",
        aggr="sum",
        prior_model=None,
        num_classes=1,
        pad_token_id=0,
    )
    return GeoformerForEnergyRegression(config=config)


def example_input_geoformer():
    """(z, pos) for a tiny padded batch of B=2 molecules, N=6 atoms each."""
    torch.manual_seed(0)
    z = torch.randint(1, 10, (2, 6), dtype=torch.long)
    z[0, 4:] = 0  # pad_token_id
    pos = torch.randn(2, 6, 3)
    return (z, pos)


MENAGERIE_ENTRIES = [
    (
        "Geoformer",
        "build_geoformer",
        "example_input_geoformer",
        "2023",
        "SOURCE_AVAILABLE",
    ),
]
