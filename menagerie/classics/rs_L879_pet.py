# SOURCE: vendored from spozdn/pet @ ff66e8f3d33184bdf193855828921b003f8f6cdb (main)
#
# PET (Point Edge Transformer): a message-passing interatomic-potential architecture
# built from stacked local-frame Cartesian transformers over per-atom neighbor tokens
# (Pozdnyakov & Ceriotti, "Smooth, exact rotational symmetrization for deep learning
# on point clouds"). Every class below is copied verbatim (imports/paths fixed
# minimally so the module is self-contained; the `Molecule`/`MoleculeCPP` graph
# builders and their `matscipy`/`ase.neighborlist` dependencies are NOT vendored --
# they are pure preprocessing helpers, not part of the `PET` model architecture, and
# the tiny example graph below is built directly in the exact `Data`/`batch_dict`
# schema those helpers produce, per `molecule.py::Molecule.get_graph`/`batch_to_dict`).
# Files combined:
#   - src/utilities.py  -> NeverRun (torchscript-safe dummy submodule placeholder)
#   - src/hypers.py     -> Hypers (dict -> attribute-namespace config wrapper)
#   - src/transformer.py -> AttentionBlock, TransformerLayer, Transformer
#   - src/pet.py        -> CentralSplitter, CentralUniter, cutoff_func,
#                           get_activation, CartesianTransformer, CentralSpecificModel,
#                           FeedForward, Head, CentralTokensPredictor,
#                           MessagesPredictor, MessagesBondsPredictor, PET
#
# MENAGERIE_ZOO = "vendored-pytorch"

from __future__ import annotations

import copy
from typing import Dict, Optional

import numpy as np
import torch
import torch_geometric
from torch import nn


MENAGERIE_ZOO = "vendored-pytorch"


# ------------------------------------------------------------------
# src/utilities.py (verbatim, relevant piece)
# ------------------------------------------------------------------
class NeverRun(torch.nn.Module):
    """Dummy torch module to make torchscript happy.
    This model should never be run"""

    def __init__(self):
        super(NeverRun, self).__init__()

    def forward(self, x) -> torch.Tensor:
        raise RuntimeError("This model should never be run")


# ------------------------------------------------------------------
# src/hypers.py (verbatim, relevant piece)
# ------------------------------------------------------------------
class Hypers:
    def __init__(self, hypers_dict):
        for key, value in hypers_dict.items():
            if isinstance(value, dict):
                self.__dict__[key] = Hypers(value)
            else:
                self.__dict__[key] = value


# ------------------------------------------------------------------
# src/transformer.py (verbatim)
# ------------------------------------------------------------------
class AttentionBlock(nn.Module):
    def __init__(self, total_dim, num_heads, dropout=0.0, epsilon=1e-15):
        super(AttentionBlock, self).__init__()

        self.input_linear = nn.Linear(total_dim, 3 * total_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(total_dim, total_dim)

        nn.init.xavier_uniform_(self.input_linear.weight)
        nn.init.constant_(self.input_linear.bias, 0.0)
        nn.init.constant_(self.output_linear.bias, 0.0)

        self.num_heads = num_heads
        self.epsilon = epsilon

        if total_dim % num_heads != 0:
            raise ValueError("total dimension is not divisible by the number of heads")
        self.head_dim = total_dim // num_heads
        self.preconditioning = 1.0 / np.sqrt(self.head_dim)

    def forward(self, x, multipliers: Optional[torch.Tensor] = None):
        initial_shape = x.shape
        x = self.input_linear(x)
        x = x.reshape(initial_shape[0], initial_shape[1], 3, self.num_heads, self.head_dim)
        x = x.permute(2, 0, 3, 1, 4)

        queries, keys, values = x[0], x[1], x[2]
        alpha = torch.matmul(queries, keys.transpose(-2, -1)) * self.preconditioning
        alpha = torch.nn.functional.softmax(alpha, dim=-1)
        alpha = self.dropout(alpha)

        if multipliers is not None:
            alpha = alpha * multipliers[:, None, :, :]
            alpha = alpha / (alpha.sum(dim=-1)[..., None] + self.epsilon)

        x = torch.matmul(alpha, values).transpose(1, 2).reshape(initial_shape)
        x = self.output_linear(x)
        return x


class TransformerLayer(torch.nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        dim_feedforward=512,
        dropout=0.0,
        activation=torch.nn.functional.silu,
        transformer_type="PostLN",
    ):
        super(TransformerLayer, self).__init__()
        self.attention = AttentionBlock(d_model, n_heads, dropout=dropout)

        if transformer_type not in ["PostLN", "PreLN"]:
            raise ValueError("unknown transformer type")
        self.transformer_type = transformer_type
        self.d_model = d_model
        self.norm_attention = nn.LayerNorm(d_model)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = activation

        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, multipliers: Optional[torch.Tensor] = None):
        if self.transformer_type == "PostLN":
            x = self.norm_attention(x + self.dropout(self.attention(x, multipliers)))
            x = self.norm_mlp(x + self.mlp(x))
        if self.transformer_type == "PreLN":
            x = x + self.dropout(self.attention(self.norm_attention(x), multipliers))
            x = x + self.mlp(self.norm_mlp(x))
        return x


class Transformer(torch.nn.Module):
    def __init__(self, trans_layer, num_layers):
        super(Transformer, self).__init__()
        self.transformer_type = trans_layer.transformer_type

        self.final_norm = NeverRun()  # for torchscript
        if trans_layer.transformer_type == "PreLN":
            self.final_norm = nn.LayerNorm(trans_layer.d_model)
        self.layers = [copy.deepcopy(trans_layer) for _ in range(num_layers)]
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x: torch.Tensor, multipliers: Optional[torch.Tensor] = None):
        for layer in self.layers:
            x = layer(x, multipliers)
        if self.transformer_type == "PreLN":
            x = self.final_norm(x)
        return x


# ------------------------------------------------------------------
# src/pet.py (verbatim)
# ------------------------------------------------------------------
class CentralSplitter(torch.nn.Module):
    def __init__(self):
        super(CentralSplitter, self).__init__()

    def forward(self, features, central_species):
        all_species = np.unique(central_species)
        all_species = [str(specie) for specie in all_species]

        result = {}
        for specie in all_species:
            result[specie] = {}

        for key, value in features.items():
            for specie in all_species:
                mask_now = central_species == int(specie)
                result[specie][key] = value[mask_now]
        return result


class CentralUniter(torch.nn.Module):
    def __init__(self):
        super(CentralUniter, self).__init__()

    def forward(self, features, central_species):
        all_species = np.unique(central_species)
        all_species = [str(specie) for specie in all_species]
        specie = all_species[0]

        shapes = {}
        for key, value in features[specie].items():
            now = list(value.shape)
            now[0] = 0
            shapes[key] = now

        device = None
        for specie in all_species:
            for key, value in features[specie].items():
                num = features[specie][key].shape[0]
                device = features[specie][key].device
                shapes[key][0] += num

        result = {
            key: torch.empty(shape, dtype=torch.get_default_dtype()).to(device)
            for key, shape in shapes.items()
        }

        for specie in features.keys():
            for key, value in features[specie].items():
                mask = int(specie) == central_species
                result[key][mask] = features[specie][key]

        return result


def cutoff_func(grid: torch.Tensor, r_cut: float, delta: float):
    mask_bigger = grid >= r_cut
    mask_smaller = grid <= r_cut - delta
    grid = (grid - r_cut + delta) / delta
    f = 1 / 2.0 + torch.cos(np.pi * grid) / 2.0

    f[mask_bigger] = 0.0
    f[mask_smaller] = 1.0
    return f


def get_activation(hypers):
    if hypers.ACTIVATION == "mish":
        return nn.Mish()
    if hypers.ACTIVATION == "silu":
        return nn.SiLU()
    raise ValueError("unknown activation")


class CartesianTransformer(torch.nn.Module):
    def __init__(
        self,
        hypers,
        d_model,
        n_head,
        dim_feedforward,
        n_layers,
        dropout,
        n_atomic_species,
        add_central_token,
        is_first,
    ):
        super(CartesianTransformer, self).__init__()
        self.hypers = hypers
        self.is_first = is_first
        self.trans_layer = TransformerLayer(
            d_model=d_model,
            n_heads=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=get_activation(hypers),
            transformer_type=hypers.TRANSFORMER_TYPE,
        )
        self.trans = Transformer(self.trans_layer, num_layers=n_layers)

        if hypers.USE_ONLY_LENGTH:
            input_dim = 1
        else:
            input_dim = 3
            if hypers.USE_LENGTH:
                input_dim += 1

        if hypers.USE_ADDITIONAL_SCALAR_ATTRIBUTES:
            input_dim += hypers.SCALAR_ATTRIBUTES_SIZE

        if hypers.R_EMBEDDING_ACTIVATION:
            self.r_embedding = nn.Sequential(nn.Linear(input_dim, d_model), get_activation(hypers))
        else:
            self.r_embedding = nn.Linear(input_dim, d_model)

        if hypers.BLEND_NEIGHBOR_SPECIES and (not is_first):
            n_merge = 3
        else:
            n_merge = 2

        self.compress = None
        if hypers.COMPRESS_MODE == "linear":
            self.compress = nn.Linear(n_merge * d_model, d_model)
        if hypers.COMPRESS_MODE == "mlp":
            self.compress = nn.Sequential(
                nn.Linear(n_merge * d_model, d_model),
                get_activation(hypers),
                nn.Linear(d_model, d_model),
            )
        if self.compress is None:
            raise ValueError("unknown compress mode")

        self.neighbor_embedder = NeverRun()  # for torchscript
        if hypers.BLEND_NEIGHBOR_SPECIES and (not is_first):
            self.neighbor_embedder = nn.Embedding(n_atomic_species + 1, d_model)

        self.add_central_token = add_central_token

        self.central_embedder = NeverRun()  # for torchscript
        self.central_scalar_embedding = NeverRun()  # for torchscript
        self.central_compress = NeverRun()  # for torchscript

        if add_central_token:
            self.central_embedder = nn.Embedding(n_atomic_species + 1, d_model)
            if hypers.USE_ADDITIONAL_SCALAR_ATTRIBUTES:
                if hypers.R_EMBEDDING_ACTIVATION:
                    self.central_scalar_embedding = nn.Sequential(
                        nn.Linear(hypers.SCALAR_ATTRIBUTES_SIZE, d_model),
                        get_activation(hypers),
                    )
                else:
                    self.central_scalar_embedding = nn.Linear(
                        hypers.SCALAR_ATTRIBUTES_SIZE, d_model
                    )

                if hypers.COMPRESS_MODE == "linear":
                    self.central_compress = nn.Linear(2 * d_model, d_model)
                if hypers.COMPRESS_MODE == "mlp":
                    self.central_compress = nn.Sequential(
                        nn.Linear(2 * d_model, d_model),
                        get_activation(hypers),
                        nn.Linear(d_model, d_model),
                    )

        # assign hypers one by one for torch.script
        self.USE_LENGTH = hypers.USE_LENGTH
        self.BLEND_NEIGHBOR_SPECIES = hypers.BLEND_NEIGHBOR_SPECIES
        self.USE_ADDITIONAL_SCALAR_ATTRIBUTES = hypers.USE_ADDITIONAL_SCALAR_ATTRIBUTES
        self.USE_ONLY_LENGTH = hypers.USE_ONLY_LENGTH
        self.R_CUT = hypers.R_CUT
        self.CUTOFF_DELTA = hypers.CUTOFF_DELTA

    def forward(self, batch_dict: Dict[str, torch.Tensor]):
        x = batch_dict["x"]

        if self.USE_LENGTH:
            neighbor_lengths = torch.sqrt(torch.sum(x**2, dim=2) + 1e-15)[:, :, None]
        else:
            neighbor_lengths = torch.empty(0, device=x.device, dtype=x.dtype)  # for torch script

        central_species = batch_dict["central_species"]
        neighbor_species = batch_dict["neighbor_species"]
        input_messages = batch_dict["input_messages"]
        mask = batch_dict["mask"]
        nums = batch_dict["nums"]

        if self.BLEND_NEIGHBOR_SPECIES and (not self.is_first):
            neighbor_embedding = self.neighbor_embedder(neighbor_species)
        else:
            neighbor_embedding = torch.empty(0, device=x.device, dtype=x.dtype)  # for torch script

        if self.USE_ADDITIONAL_SCALAR_ATTRIBUTES:
            neighbor_scalar_attributes = batch_dict["neighbor_scalar_attributes"]
            central_scalar_attributes = batch_dict["central_scalar_attributes"]
        else:
            neighbor_scalar_attributes = torch.empty(
                0, device=x.device, dtype=x.dtype
            )  # for torch script
            central_scalar_attributes = torch.empty(
                0, device=x.device, dtype=x.dtype
            )  # for torch script

        initial_n_tokens = x.shape[1]
        max_number = int(torch.max(nums))

        if self.USE_ONLY_LENGTH:
            coordinates = [neighbor_lengths]
        else:
            coordinates = [x]
            if self.USE_LENGTH:
                coordinates.append(neighbor_lengths)

        if self.USE_ADDITIONAL_SCALAR_ATTRIBUTES:
            coordinates.append(neighbor_scalar_attributes)
        coordinates = torch.cat(coordinates, dim=2)
        coordinates = self.r_embedding(coordinates)

        if self.BLEND_NEIGHBOR_SPECIES and (not self.is_first):
            tokens = torch.cat([coordinates, neighbor_embedding, input_messages], dim=2)
        else:
            tokens = torch.cat([coordinates, input_messages], dim=2)

        tokens = self.compress(tokens)

        if self.add_central_token:
            central_specie_embedding = self.central_embedder(central_species)
            if self.USE_ADDITIONAL_SCALAR_ATTRIBUTES:
                central_scalar_embedding = self.central_scalar_embedding(central_scalar_attributes)
                central_token = torch.cat(
                    [central_specie_embedding, central_scalar_embedding], dim=1
                )
                central_token = self.central_compress(central_token)
            else:
                central_token = central_specie_embedding

            tokens = torch.cat([central_token[:, None, :], tokens], dim=1)

            submask = torch.zeros(mask.shape[0], dtype=torch.bool).to(mask.device)
            total_mask = torch.cat([submask[:, None], mask], dim=1)

            lengths = torch.sqrt(torch.sum(x * x, dim=2) + 1e-16)
            multipliers = cutoff_func(lengths, self.R_CUT, self.CUTOFF_DELTA)
            sub_multipliers = torch.ones(mask.shape[0], device=mask.device)
            multipliers = torch.cat([sub_multipliers[:, None], multipliers], dim=1)
            multipliers[total_mask] = 0.0

            multipliers = multipliers[:, None, :]
            multipliers = multipliers.repeat(1, multipliers.shape[2], 1)

            output_messages = self.trans(
                tokens[:, : (max_number + 1), :],
                multipliers=multipliers[:, : (max_number + 1), : (max_number + 1)],
            )
            if max_number < initial_n_tokens:
                padding = torch.zeros(
                    output_messages.shape[0],
                    initial_n_tokens - max_number,
                    output_messages.shape[2],
                    device=output_messages.device,
                )
                output_messages = torch.cat([output_messages, padding], dim=1)

            return {
                "output_messages": output_messages[:, 1:, :],
                "central_token": output_messages[:, 0, :],
            }
        else:
            lengths = torch.sqrt(torch.sum(x * x, dim=2) + 1e-16)

            multipliers = cutoff_func(lengths, self.R_CUT, self.CUTOFF_DELTA)
            multipliers[mask] = 0.0

            multipliers = multipliers[:, None, :]
            multipliers = multipliers.repeat(1, multipliers.shape[2], 1)

            output_messages = self.trans(
                tokens[:, :max_number, :],
                multipliers=multipliers[:, :max_number, :max_number],
            )
            if max_number < initial_n_tokens:
                padding = torch.zeros(
                    output_messages.shape[0],
                    initial_n_tokens - max_number,
                    output_messages.shape[2],
                    device=output_messages.device,
                )
                output_messages = torch.cat([output_messages, padding], dim=1)

            return {"output_messages": output_messages}


class CentralSpecificModel(torch.nn.Module):
    def __init__(self, models):
        super(CentralSpecificModel, self).__init__()
        self.models = torch.nn.ModuleDict(models)
        self.splitter = CentralSplitter()
        self.uniter = CentralUniter()

    def forward(self, batch_dict):
        central_indices = batch_dict["central_species"].data.cpu().numpy()
        splitted = self.splitter(batch_dict, central_indices)

        result = {}
        for key in splitted.keys():
            result[str(key)] = self.models[str(key)](splitted[key])

        result = self.uniter(result, central_indices)
        return result


class FeedForward(torch.nn.Module):
    def __init__(self, hypers, n_in, n_neurons):
        super(FeedForward, self).__init__()
        self.hypers = hypers
        self.nn = nn.Sequential(
            nn.Linear(n_in, n_neurons),
            get_activation(hypers),
            nn.Linear(n_neurons, n_neurons),
            get_activation(hypers),
            nn.Linear(n_neurons, hypers.D_OUTPUT),
        )

    def forward(self, x):
        return self.nn(x)


class Head(torch.nn.Module):
    def __init__(self, hypers, n_in, n_neurons):
        super(Head, self).__init__()
        self.n_targets = hypers.N_TARGETS
        self.d_output = hypers.D_OUTPUT
        self.hypers = hypers
        if self.n_targets == 1:
            self.model = FeedForward(hypers, n_in, n_neurons)
        else:
            self.models = nn.ModuleList(
                [FeedForward(hypers, n_in, n_neurons) for _ in range(self.n_targets)]
            )

    def forward(self, batch: Dict[str, torch.Tensor]):
        x = batch["pooled"]
        if self.n_targets == 1:
            return {"atomic_predictions": self.model(x)}

        target_indices = batch["target_indices"]
        if target_indices is None:
            raise ValueError("target indices should be provided for multitarget fitting")

        if torch.any(target_indices < 0) or torch.any(target_indices >= self.n_targets):
            raise ValueError(
                f"All target indices must be within 0 and {self.n_targets - 1} inclusive."
            )

        if x.size(0) != target_indices.size(0):
            raise ValueError("The first dimension of x and target_indices must match.")

        output_shape = list(x.shape)
        output_shape[-1] = self.d_output
        outputs = torch.zeros(output_shape, device=x.device)

        for target_idx in range(self.n_targets):
            mask = target_indices == target_idx
            if mask.sum().item() == 0:
                continue

            x_subtensor = x[mask]
            model_output = self.models[target_idx](x_subtensor)
            outputs[mask] = model_output

        return {"atomic_predictions": outputs}


class CentralTokensPredictor(torch.nn.Module):
    def __init__(self, hypers, head):
        super(CentralTokensPredictor, self).__init__()
        self.head = head
        self.hypers = hypers

    def forward(
        self,
        central_tokens: torch.Tensor,
        central_species: torch.Tensor,
        target_indices: torch.Tensor,
    ):
        predictions = self.head({"pooled": central_tokens, "target_indices": target_indices})[
            "atomic_predictions"
        ]
        return predictions


class MessagesPredictor(torch.nn.Module):
    def __init__(self, hypers, head):
        super(MessagesPredictor, self).__init__()
        self.head = head
        self.AVERAGE_POOLING = hypers.AVERAGE_POOLING

    def forward(
        self,
        messages: torch.Tensor,
        mask: torch.Tensor,
        nums: torch.Tensor,
        central_species: torch.Tensor,
        multipliers: torch.Tensor,
        target_indices: torch.Tensor,
    ):
        messages_proceed = messages * multipliers[:, :, None]
        messages_proceed[mask] = 0.0
        if self.AVERAGE_POOLING:
            total_weight = multipliers.sum(dim=1)[:, None]
            pooled = messages_proceed.sum(dim=1) / total_weight
        else:
            pooled = messages_proceed.sum(dim=1)

        predictions = self.head({"pooled": pooled, "target_indices": target_indices})[
            "atomic_predictions"
        ]
        return predictions


class MessagesBondsPredictor(torch.nn.Module):
    def __init__(self, hypers, head):
        super(MessagesBondsPredictor, self).__init__()
        self.head = head
        self.AVERAGE_BOND_ENERGIES = hypers.AVERAGE_BOND_ENERGIES

    def forward(
        self,
        messages: torch.Tensor,
        mask: torch.Tensor,
        nums: torch.Tensor,
        central_species: torch.Tensor,
        multipliers: torch.Tensor,
        target_indices: torch.Tensor,
    ):
        predictions = self.head({"pooled": messages, "target_indices": target_indices})[
            "atomic_predictions"
        ]

        mask_expanded = mask[..., None].repeat(1, 1, predictions.shape[2])
        predictions = torch.where(mask_expanded, 0.0, predictions)

        predictions = predictions * multipliers[:, :, None]
        if self.AVERAGE_BOND_ENERGIES:
            total_weight = multipliers.sum(dim=1)[:, None]
            result = predictions.sum(dim=1) / total_weight
        else:
            result = predictions.sum(dim=1)
        return result


class PET(torch.nn.Module):
    def __init__(self, hypers, transformer_dropout, n_atomic_species):
        super(PET, self).__init__()
        self.hypers = hypers
        transformer_d_model = hypers.TRANSFORMER_D_MODEL
        transformer_n_head = hypers.TRANSFORMER_N_HEAD
        transformer_dim_feedforward = hypers.TRANSFORMER_DIM_FEEDFORWARD
        transformer_n_layers = hypers.N_TRANS_LAYERS
        n_gnn_layers = hypers.N_GNN_LAYERS
        head_n_neurons = hypers.HEAD_N_NEURONS
        transformers_central_specific = hypers.TRANSFORMERS_CENTRAL_SPECIFIC
        heads_central_specific = hypers.HEADS_CENTRAL_SPECIFIC

        add_central_tokens = []
        for _ in range(hypers.N_GNN_LAYERS - 1):
            add_central_tokens.append(hypers.ADD_TOKEN_FIRST)
        add_central_tokens.append(hypers.ADD_TOKEN_SECOND)

        self.embedding = nn.Embedding(n_atomic_species + 1, transformer_d_model)
        gnn_layers = []
        if transformers_central_specific:
            raise NotImplementedError(
                "TRANSFORMERS_CENTRAL_SPECIFIC=True requires the real all_species global "
                "from train_model.py; not needed for the menagerie's tiny random-init trace."
            )
        else:
            for layer_index in range(n_gnn_layers):
                is_first = layer_index == 0
                model = CartesianTransformer(
                    hypers,
                    transformer_d_model,
                    transformer_n_head,
                    transformer_dim_feedforward,
                    transformer_n_layers,
                    transformer_dropout,
                    n_atomic_species,
                    add_central_tokens[layer_index],
                    is_first,
                )
                gnn_layers.append(model)

        self.gnn_layers = torch.nn.ModuleList(gnn_layers)

        heads = []
        if heads_central_specific:
            raise NotImplementedError(
                "HEADS_CENTRAL_SPECIFIC=True requires the real all_species global "
                "from train_model.py; not needed for the menagerie's tiny random-init trace."
            )
        else:
            for _ in range(n_gnn_layers):
                heads.append(Head(hypers, transformer_d_model, head_n_neurons))

        self.heads = torch.nn.ModuleList(heads)
        self.central_tokens_predictors = torch.nn.ModuleList(
            [CentralTokensPredictor(hypers, head) for head in heads]
        )
        self.messages_predictors = torch.nn.ModuleList(
            [MessagesPredictor(hypers, head) for head in heads]
        )

        if hypers.USE_BOND_ENERGIES:
            bond_heads = []
            if heads_central_specific:
                raise NotImplementedError("see above")
            else:
                for _ in range(n_gnn_layers):
                    bond_heads.append(Head(hypers, transformer_d_model, head_n_neurons))

            self.bond_heads = torch.nn.ModuleList(bond_heads)
            self.messages_bonds_predictors = torch.nn.ModuleList(
                [MessagesBondsPredictor(hypers, head) for head in bond_heads]
            )
        else:
            self.messages_bonds_predictors = torch.nn.ModuleList(
                [NeverRun() for _ in range(n_gnn_layers)]
            )

        self.R_CUT = hypers.R_CUT
        self.CUTOFF_DELTA = hypers.CUTOFF_DELTA
        self.USE_BOND_ENERGIES = hypers.USE_BOND_ENERGIES
        self.TARGET_TYPE = hypers.TARGET_TYPE
        self.TARGET_AGGREGATION = hypers.TARGET_AGGREGATION
        self.N_GNN_LAYERS = hypers.N_GNN_LAYERS
        self.RESIDUAL_FACTOR = hypers.RESIDUAL_FACTOR

    def get_predictions(self, batch_dict: Dict[str, torch.Tensor]):
        x = batch_dict["x"]
        central_species = batch_dict["central_species"]
        neighbor_species = batch_dict["neighbor_species"]
        batch = batch_dict["batch"]
        mask = batch_dict["mask"]
        nums = batch_dict["nums"]

        if "target_id" in batch_dict.keys():
            target_indices = batch_dict["target_id"]
            target_indices = target_indices[batch]
        else:
            target_indices = None

        lengths = torch.sqrt(torch.sum(x * x, dim=2) + 1e-16)
        multipliers = cutoff_func(lengths, self.R_CUT, self.CUTOFF_DELTA)
        multipliers[mask] = 0.0

        neighbors_index = batch_dict["neighbors_index"]
        neighbors_pos = batch_dict["neighbors_pos"]

        batch_dict["input_messages"] = self.embedding(neighbor_species)
        atomic_predictions = torch.zeros(1, dtype=x.dtype, device=x.device)

        for layer_index, (
            central_tokens_predictor,
            messages_predictor,
            gnn_layer,
            messages_bonds_predictor,
        ) in enumerate(
            zip(
                self.central_tokens_predictors,
                self.messages_predictors,
                self.gnn_layers,
                self.messages_bonds_predictors,
            )
        ):
            result = gnn_layer(batch_dict)
            output_messages = result["output_messages"]

            new_input_messages = output_messages[neighbors_index, neighbors_pos]
            batch_dict["input_messages"] = self.RESIDUAL_FACTOR * (
                batch_dict["input_messages"] + new_input_messages
            )

            if "central_token" in result.keys():
                atomic_predictions = atomic_predictions + central_tokens_predictor(
                    result["central_token"], central_species, target_indices
                )
            else:
                atomic_predictions = atomic_predictions + messages_predictor(
                    output_messages, mask, nums, central_species, multipliers, target_indices
                )

            if self.USE_BOND_ENERGIES:
                atomic_predictions = atomic_predictions + messages_bonds_predictor(
                    output_messages, mask, nums, central_species, multipliers, target_indices
                )

        if self.TARGET_TYPE == "structural":
            if self.TARGET_AGGREGATION == "sum":
                return torch_geometric.nn.global_add_pool(
                    atomic_predictions, batch=batch_dict["batch"]
                )
            if self.TARGET_AGGREGATION == "mean":
                return torch_geometric.nn.global_mean_pool(
                    atomic_predictions, batch=batch_dict["batch"]
                )
            raise ValueError("unknown target aggregation")
        if self.TARGET_TYPE == "atomic":
            return atomic_predictions
        raise ValueError("unknown target type")

    def forward(
        self,
        batch_dict: Dict[str, torch.Tensor],
        rotations: Optional[torch.Tensor] = None,
    ):
        if rotations is not None:
            x_initial = batch_dict["x"]
            batch_dict["x"] = torch.bmm(x_initial, rotations)
            predictions = self.get_predictions(batch_dict)
            batch_dict["x"] = x_initial
            return predictions
        else:
            return self.get_predictions(batch_dict)


# ------------------------------------------------------------------
# Menagerie staging entrypoints
# ------------------------------------------------------------------
# Real default hyperparameters from default_hypers/default_hypers.yaml
# (ARCHITECTURAL_HYPERS group), trimmed to a tiny model for a fast random-init
# trace. D_OUTPUT=1 matches train_model.py's MLIP-fitting override
# ("ARCHITECTURAL_HYPERS.D_OUTPUT = 1  # energy is a single scalar").
_ARCHITECTURAL_HYPERS = {
    "CUTOFF_DELTA": 0.2,
    "AVERAGE_POOLING": False,
    "TRANSFORMERS_CENTRAL_SPECIFIC": False,
    "HEADS_CENTRAL_SPECIFIC": False,
    "ADD_TOKEN_FIRST": True,
    "ADD_TOKEN_SECOND": True,
    "N_GNN_LAYERS": 2,
    "TRANSFORMER_D_MODEL": 16,
    "TRANSFORMER_N_HEAD": 2,
    "TRANSFORMER_DIM_FEEDFORWARD": 32,
    "HEAD_N_NEURONS": 16,
    "N_TRANS_LAYERS": 1,
    "ACTIVATION": "silu",
    "USE_LENGTH": True,
    "USE_ONLY_LENGTH": False,
    "R_CUT": 5.0,
    "R_EMBEDDING_ACTIVATION": False,
    "COMPRESS_MODE": "mlp",
    "BLEND_NEIGHBOR_SPECIES": False,
    "AVERAGE_BOND_ENERGIES": False,
    "USE_BOND_ENERGIES": True,
    "USE_ADDITIONAL_SCALAR_ATTRIBUTES": False,
    "SCALAR_ATTRIBUTES_SIZE": None,
    "TRANSFORMER_TYPE": "PostLN",
    "N_TARGETS": 1,
    "RESIDUAL_FACTOR": 0.5,
    "TARGET_TYPE": "structural",
    "TARGET_AGGREGATION": "sum",
    "D_OUTPUT": 1,
}


def build_pet():
    torch.manual_seed(0)
    hypers = Hypers(_ARCHITECTURAL_HYPERS)
    n_atomic_species = 4  # matches the 4 distinct species in example_input_pet
    return PET(hypers, transformer_dropout=0.0, n_atomic_species=n_atomic_species)


def example_input_pet():
    # A tiny 4-atom synthetic molecule built directly in the batch_dict schema
    # produced by molecule.py::Molecule.get_graph / batch_to_dict (species mapped
    # to local indices 0..n_atomic_species-1, a fixed max-neighbor padding of 3
    # with a boolean mask marking padded slots, and the "neighbor talks back"
    # position index used for the input_messages residual update).
    rng = np.random.default_rng(0)
    n_atoms = 4
    max_num = 3  # every atom is within cutoff of every other atom here

    central_species = torch.tensor([0, 1, 1, 2], dtype=torch.long)
    neighbor_species = torch.tensor(
        [
            [1, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 1],
        ],
        dtype=torch.long,
    )
    x = torch.tensor(
        rng.normal(scale=1.5, size=(n_atoms, max_num, 3)), dtype=torch.get_default_dtype()
    )
    nums = torch.full((n_atoms,), float(max_num), dtype=torch.get_default_dtype())
    mask = torch.zeros(
        n_atoms, max_num, dtype=torch.bool
    )  # no padding: every slot is a real neighbor

    # neighbors_index[i, k] = flat position of the k-th neighbor of atom i
    # (index into the "neighbor talks back" input_messages residual).
    neighbors_index = torch.tensor(
        [
            [1, 2, 3],
            [0, 2, 3],
            [0, 1, 3],
            [0, 1, 2],
        ],
        dtype=torch.long,
    )
    # neighbors_pos[i, k] = the slot position within neighbor j's own neighbor
    # list that points back at atom i (every atom sees every other exactly once
    # here, so the back-reference is always slot 0 among the 3 neighbors minus
    # self -- concretely: n_atoms=4 fully-connected means each atom's neighbor
    # list has length 3 = n_atoms-1, and every back-reference lands on a valid
    # slot in [0, max_num)).
    neighbors_pos = torch.tensor(
        [
            [0, 0, 0],
            [0, 1, 1],
            [1, 2, 2],
            [2, 2, 2],
        ],
        dtype=torch.long,
    )
    batch = torch.zeros(n_atoms, dtype=torch.long)  # single structure in the batch

    batch_dict = {
        "x": x,
        "central_species": central_species,
        "neighbor_species": neighbor_species,
        "mask": mask,
        "batch": batch,
        "nums": nums,
        "neighbors_index": neighbors_index,
        "neighbors_pos": neighbors_pos,
    }
    return (batch_dict,)


MENAGERIE_ENTRIES = [
    ("pet", build_pet, example_input_pet, 2023, MENAGERIE_ZOO),
]
