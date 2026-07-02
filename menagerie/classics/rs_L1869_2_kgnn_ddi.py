# FAITHFUL PORT of xzenglab/KGNN @ master (models/kgcn.py + layers/aggregator.py,
# original framework: Keras 2.3.0 / TensorFlow 1.13 -- graph-mode Keras `Lambda`/
# `K.gather`/`K.variable` ops that only run under TF1, incompatible with this
# repo's base env). KGNN (IJCAI 2020, "Knowledge Graph Neural Network for
# Drug-Drug Interaction Prediction") is the KGCN receptive-field aggregator
# architecture applied twice (once per drug) and combined with a dot-product
# score head. Faithfully transcribed mechanism-for-mechanism from the real
# repo code: (1) `get_receptive_field` iteratively expands each drug's
# receptive field over the fixed knowledge-graph adjacency table for
# `n_depth` hops (gather -> reshape, same `neighbor_sample_size` fan-out);
# (2) `get_neighbor_info` computes the drug-relation attention score per
# neighbor edge and reduces to a weighted-neighbor embedding
# (`drug_rel_score = sum(drug * rel)`, `weighted_ent = drug_rel_score * ent`,
# reshape + sum over the neighbor axis); (3) the depth-wise aggregation loop
# repeatedly folds neighbor embeddings into entity embeddings via
# `SumAggregator` (`activation(W @ (entity + neighbor) + b)`, tanh on the
# final depth, relu otherwise) -- the repo's default `aggregator_type='sum'`
# (config.py `ModelConfig.aggregator_type`); (4) final score is the sigmoid
# dot product of the two drugs' depth-0 aggregated embeddings. Default
# hyperparameters (`n_depth=2`, `embed_dim=32`, `neighbor_sample_size=4`) are
# copied verbatim from `config.py`. The adjacency table (`adj_entity`,
# `adj_relation`) is baked-in KG structure in the original (loaded from
# `raw_data/*/train2id.txt` at train time, then held fixed as a Keras
# `K.variable` inside the model); ported here as registered buffers sized for
# a small synthetic entity/relation vocabulary so the module is
# self-contained and traceable without the KEGG/DrugBank raw data files.
"""Faithful PyTorch port of KGNN (KGCN aggregator for drug-drug interaction)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


class SumAggregator(nn.Module):
    """Port of layers/aggregator.py::SumAggregator."""

    def __init__(self, embed_dim: int, activation: str = "relu"):
        super().__init__()
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.activation = torch.tanh if activation == "tanh" else F.relu

    def forward(self, entity: torch.Tensor, neighbor: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(entity + neighbor))


class KGCN(nn.Module):
    """Port of models/kgcn.py::KGCN.build(), applied to a drug-pair input.

    Ported forward pass (mirrors the Keras `build()` method structure,
    duplicated once per drug via a shared `_score_drug` helper -- the
    original literally duplicates the receptive-field + aggregation block
    for `input_drug_one` and `input_drug_two`).
    """

    def __init__(
        self,
        drug_vocab_size: int = 64,
        entity_vocab_size: int = 128,
        relation_vocab_size: int = 16,
        embed_dim: int = 32,
        n_depth: int = 2,
        neighbor_sample_size: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_depth = n_depth
        self.neighbor_sample_size = neighbor_sample_size

        self.drug_embedding = nn.Embedding(drug_vocab_size, embed_dim)
        self.entity_embedding = nn.Embedding(entity_vocab_size, embed_dim)
        self.relation_embedding = nn.Embedding(relation_vocab_size, embed_dim)

        # Fixed knowledge-graph adjacency table (in the original, a
        # `K.variable` built from `raw_data/*/train2id.txt` at train time).
        # Random-but-fixed here so the module is self-contained for tracing.
        generator = torch.Generator().manual_seed(0)
        self.register_buffer(
            "adj_entity",
            torch.randint(
                0, entity_vocab_size, (entity_vocab_size, neighbor_sample_size), generator=generator
            ),
        )
        self.register_buffer(
            "adj_relation",
            torch.randint(
                0,
                relation_vocab_size,
                (entity_vocab_size, neighbor_sample_size),
                generator=generator,
            ),
        )

        # aggregator_type='sum' (config.py default); tanh on the last depth
        # only, relu otherwise -- matches `activation='tanh' if depth ==
        # n_depth-1 else 'relu'` in kgcn.py.
        self.aggregators_drug_one = nn.ModuleList(
            [
                SumAggregator(embed_dim, activation="tanh" if depth == n_depth - 1 else "relu")
                for depth in range(n_depth)
            ]
        )
        self.aggregators_drug_two = nn.ModuleList(
            [
                SumAggregator(embed_dim, activation="tanh" if depth == n_depth - 1 else "relu")
                for depth in range(n_depth)
            ]
        )

    def _get_receptive_field(self, entity_ids: torch.Tensor):
        """Port of KGCN.get_receptive_field: multi-hop neighbor expansion."""
        neigh_ent_list = [entity_ids]
        neigh_rel_list = []
        for _ in range(self.n_depth):
            prev = neigh_ent_list[-1]
            new_neigh_ent = self.adj_entity[prev]
            new_neigh_rel = self.adj_relation[prev]
            batch_size = entity_ids.shape[0]
            neigh_ent_list.append(new_neigh_ent.reshape(batch_size, -1))
            neigh_rel_list.append(new_neigh_rel.reshape(batch_size, -1))
        return neigh_ent_list, neigh_rel_list

    def _get_neighbor_info(
        self, drug: torch.Tensor, rel: torch.Tensor, ent: torch.Tensor
    ) -> torch.Tensor:
        """Port of KGCN.get_neighbor_info: drug-relation-gated neighbor pooling."""
        drug_rel_score = (drug * rel).sum(dim=-1, keepdim=True)
        weighted_ent = drug_rel_score * ent
        batch_size = weighted_ent.shape[0]
        weighted_ent = weighted_ent.reshape(
            batch_size, -1, self.neighbor_sample_size, self.embed_dim
        )
        return weighted_ent.sum(dim=2)

    def _score_drug(
        self, drug_embed: torch.Tensor, entity_ids: torch.Tensor, aggregators: nn.ModuleList
    ) -> torch.Tensor:
        neigh_ent_list, neigh_rel_list = self._get_receptive_field(entity_ids)
        neigh_ent_embed_list = [self.entity_embedding(e) for e in neigh_ent_list]
        neigh_rel_embed_list = [self.relation_embedding(r) for r in neigh_rel_list]

        for depth in range(self.n_depth):
            aggregator = aggregators[depth]
            next_neigh_ent_embed_list = []
            for hop in range(self.n_depth - depth):
                neighbor_embed = self._get_neighbor_info(
                    drug_embed, neigh_rel_embed_list[hop], neigh_ent_embed_list[hop + 1]
                )
                next_entity_embed = aggregator(neigh_ent_embed_list[hop], neighbor_embed)
                next_neigh_ent_embed_list.append(next_entity_embed)
            neigh_ent_embed_list = next_neigh_ent_embed_list

        return neigh_ent_embed_list[0].squeeze(1)

    def forward(self, input_drug_one: torch.Tensor, input_drug_two: torch.Tensor) -> torch.Tensor:
        drug_embed = self.drug_embedding(input_drug_one)  # [batch, 1, embed_dim]

        drug1_embed = self._score_drug(drug_embed, input_drug_one, self.aggregators_drug_one)
        drug2_embed = self._score_drug(drug_embed, input_drug_two, self.aggregators_drug_two)

        drug_drug_score = torch.sigmoid((drug1_embed * drug2_embed).sum(dim=-1, keepdim=True))
        return drug_drug_score


# ---------------------------------------------------------------------------
# Staging build/example helpers. Inputs are drug-index tensors
# `input_drug_one`/`input_drug_two`, shape (batch, 1), matching the Keras
# `Input(shape=(1,), dtype='int64')` placeholders in kgcn.py::build().
# ---------------------------------------------------------------------------


def build_kgnn_ddi():
    return KGCN(
        drug_vocab_size=64,
        entity_vocab_size=128,
        relation_vocab_size=16,
        embed_dim=32,
        n_depth=2,
        neighbor_sample_size=4,
    )


def example_input_kgnn_ddi():
    drug_one = torch.randint(0, 64, (2, 1))
    drug_two = torch.randint(0, 64, (2, 1))
    return (drug_one, drug_two)


MENAGERIE_ENTRIES = [
    (
        "KGNN (KGCN for Drug-Drug Interaction)",
        "build_kgnn_ddi",
        "example_input_kgnn_ddi",
        "2020",
        "ported-pytorch",
    ),
]
