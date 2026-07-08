# SOURCE: vendored from higgsfield/interaction_network_pytorch @ master
# Files: Interaction Network.ipynb (RelationalModel, ObjectModel, InteractionNetwork cells)
# https://github.com/higgsfield/interaction_network_pytorch
#
# Interaction Network for learning physical dynamics (Battaglia, Pascanu, Lai, Rezende,
# Kavukcuoglu, NeurIPS 2016, https://arxiv.org/abs/1612.00222). The reference notebook's
# n-body-solar-system demo: a `RelationalModel` MLP scores every (sender, receiver, relation
# feature) triple with a shared network to produce per-edge "effects", the effects are
# summed onto each receiver via the one-hot `receiver_relations` matrix, and an
# `ObjectModel` MLP maps each object's own state plus its aggregated incoming effect to a
# predicted next-step (speedX, speedY). This is the canonical relation-centric /
# object-centric two-MLP message-passing design from the paper (predates and is
# architecturally distinct from later encode-process-decode GNN variants in this
# menagerie).
#
# Import-fix only (per rung-2 rules, architecture code is untouched): the notebook cells
# used no cross-file imports beyond stdlib/torch (a top cell imports `from Physics_Engine
# import gen`, the physical-dynamics data generator used only for training-data synthesis,
# not part of the model itself, so it is not vendored here). No other code was changed.

from __future__ import annotations

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# --- Interaction Network.ipynb, "Relation-centric Neural Network" cell (verbatim) --------


class RelationalModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(RelationalModel, self).__init__()

        self.output_size = output_size

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        Args:
            x: [batch_size, n_relations, input_size]
        Returns:
            [batch_size, n_relations, output_size]
        """
        batch_size, n_relations, input_size = x.size()
        x = x.view(-1, input_size)
        x = self.layers(x)
        x = x.view(batch_size, n_relations, self.output_size)
        return x


# --- "Object-centric Neural Network" cell (verbatim) --------------------------------------


class ObjectModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ObjectModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),  # speedX and speedY
        )

    def forward(self, x):
        """
        Args:
            x: [batch_size, n_objects, input_size]
        Returns:
            [batch_size * n_objects, 2] speedX and speedY
        """
        input_size = x.size(2)
        x = x.view(-1, input_size)
        return self.layers(x)


# --- "Interaction Network" cell (verbatim) -------------------------------------------------


class InteractionNetwork(nn.Module):
    def __init__(self, n_objects, object_dim, n_relations, relation_dim, effect_dim):
        super(InteractionNetwork, self).__init__()

        self.relational_model = RelationalModel(2 * object_dim + relation_dim, effect_dim, 150)
        self.object_model = ObjectModel(object_dim + effect_dim, 100)

    def forward(self, objects, sender_relations, receiver_relations, relation_info):
        senders = sender_relations.permute(0, 2, 1).bmm(objects)
        receivers = receiver_relations.permute(0, 2, 1).bmm(objects)
        effects = self.relational_model(torch.cat([senders, receivers, relation_info], 2))
        effect_receivers = receiver_relations.bmm(effects)
        predicted = self.object_model(torch.cat([objects, effect_receivers], 2))
        return predicted


# --- staging entry points ----------------------------------------------------------------


def build_interaction_network():
    """Tiny random-init InteractionNetwork, matching the notebook's n-body-solar-system
    config (n_objects=5, object_dim=5 -- mass/x/y/speedx/speedy, fully-connected relation
    graph so n_relations = n_objects*(n_objects-1), relation_dim=1)."""
    n_objects = 5
    object_dim = 5
    n_relations = n_objects * (n_objects - 1)
    relation_dim = 1
    effect_dim = 20
    return InteractionNetwork(n_objects, object_dim, n_relations, relation_dim, effect_dim)


def example_input_interaction_network():
    """Real multi-tensor input matching the notebook's get_batch(): (objects,
    sender_relations, receiver_relations, relation_info).

    objects: (batch, n_objects, object_dim) object state (mass, x, y, speedx, speedy).
    sender_relations / receiver_relations: (batch, n_objects, n_relations) one-hot
        incidence matrices over the fully-connected relation graph.
    relation_info: (batch, n_relations, relation_dim) per-edge relation features (the
        notebook fills this with zeros for the solar-system task, no external relation
        info is available there).
    """
    torch.manual_seed(0)
    batch = 2
    n_objects = 5
    n_relations = n_objects * (n_objects - 1)
    relation_dim = 1

    objects = torch.randn(batch, n_objects, 5)

    sender_relations = torch.zeros(batch, n_objects, n_relations)
    receiver_relations = torch.zeros(batch, n_objects, n_relations)
    cnt = 0
    for i in range(n_objects):
        for j in range(n_objects):
            if i != j:
                receiver_relations[:, i, cnt] = 1.0
                sender_relations[:, j, cnt] = 1.0
                cnt += 1

    relation_info = torch.zeros(batch, n_relations, relation_dim)

    return (objects, sender_relations, receiver_relations, relation_info)


MENAGERIE_ENTRIES = [
    (
        "InteractionNetwork",
        "build_interaction_network",
        "example_input_interaction_network",
        "2016",
        "SOURCE_AVAILABLE",
    ),
]
