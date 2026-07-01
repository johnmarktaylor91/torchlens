# SOURCE: vendored from THUDM/KBRD @ master (parlai/agents/kbrd/modules.py)
#
# KBRD (Chen, Liao, Zhang, Wang, Xie, Liu, "Towards Knowledge-Based
# Recommender Dialog System", EMNLP-IJCNLP 2019). Real architecture: a
# knowledge-graph-augmented recommender -- entity/relation embedding
# tables, an `RGCNConv` (relational graph convolution, `torch_geometric`)
# that propagates over the full entity knowledge graph (self-loops +
# frequency-filtered relations built by `_edge_list`) to produce
# graph-contextualized node features, a `SelfAttentionLayer` (learned
# additive attention over a user's "seed set" of mentioned entities) that
# pools the RGCN node features into a per-user representation, and a final
# linear-with-shared-weight scoring layer (`F.linear(u_emb,
# nodes_features, self.output.bias)`) that scores every entity in the KG
# as a recommendation candidate against the user representation. This is
# the paper's real KG-recommender architecture, so it is vendored (rung 2)
# rather than reimplemented, using only `torch` + `torch_geometric` (both
# base libs).
#
# Vendoring notes (imports/config fixes only, architecture untouched):
#   - `SelfAttentionLayer`, `_edge_list`, and `KBRD.__init__` /
#     `KBRD.user_representation` are copied verbatim (unchanged compute;
#     only whitespace/formatting cleanup).
#   - `KBRD.forward` is copied verbatim except it now returns the raw
#     `scores` tensor directly instead of the dict
#     `dict(scores=..., base_loss=..., loss=...)` -- the original computes
#     `base_loss`/`loss` via `nn.CrossEntropyLoss()` against a `labels`
#     tensor purely for training-time supervision (a scalar loss, not part
#     of the model's forward representation graph beyond `scores`); the
#     traced entry point below drops the loss computation (no `labels`
#     input) and returns `scores`, which is the model's real predicted
#     recommendation-score output.
#   - `self.criterion = nn.CrossEntropyLoss()` / `self.kge_criterion =
#     nn.Softplus()` (loss modules, unused by the traced score-only
#     forward path) are dropped as dead code for this entry point.
#   - `edge_list_tensor = torch.LongTensor(edge_list).cuda()` /
#     `torch.zeros(self.dim).cuda()` (unconditional CUDA placement in the
#     original, written for a GPU-only training environment) are changed
#     to CPU-portable equivalents (`torch.LongTensor(edge_list)` /
#     `torch.zeros(self.dim)`) so the module runs on the CPU tracing
#     environment; no compute is altered.
#   - `import networkx as nx` and `from sklearn.metrics import
#     roc_auc_score` (both unused in `modules.py` itself -- vestigial
#     imports from the original file) are dropped.
#   - The knowledge graph (`kg`, an `entity -> [(relation, tail), ...]`
#     adjacency dict consumed by `_edge_list`) is normally built from the
#     downloaded DBpedia dump (`train_kbrd.py` / `kbrd.py`
#     `_edge_list_from_kg`); a small synthetic KG with the same dict shape
#     is constructed here purely to exercise the real `_edge_list` +
#     `RGCNConv` graph-construction/propagation path without requiring the
#     dataset download.

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv.rgcn_conv import RGCNConv


class SelfAttentionLayer(nn.Module):
    def __init__(self, dim, da, alpha=0.2, dropout=0.5):
        super(SelfAttentionLayer, self).__init__()
        self.dim = dim
        self.da = da
        self.alpha = alpha
        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(self.dim, self.da)))
        self.b = nn.Parameter(torch.zeros(size=(self.da, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_uniform_(self.b.data, gain=1.414)

    def forward(self, h):
        e = torch.matmul(torch.tanh(torch.matmul(h, self.a)), self.b).squeeze(dim=1)
        attention = F.softmax(e, dim=0)
        return torch.matmul(attention, h)


def _edge_list(kg, n_entity):
    edge_list = []
    self_loop_id = None
    for entity in range(n_entity):
        if entity not in kg:
            continue
        for tail_and_relation in kg[entity]:
            if entity != tail_and_relation[1]:
                edge_list.append((entity, tail_and_relation[1], tail_and_relation[0]))
                edge_list.append((tail_and_relation[1], entity, tail_and_relation[0]))
            else:
                self_loop_id = tail_and_relation[0]
    assert self_loop_id
    for entity in range(n_entity):
        # add self loop
        edge_list.append((entity, entity, self_loop_id))

    relation_cnt = defaultdict(int)
    relation_idx = {}
    for h, t, r in edge_list:
        relation_cnt[r] += 1
    # Discard infrequent relations
    for h, t, r in edge_list:
        if relation_cnt[r] > 1000 and r not in relation_idx:
            relation_idx[r] = len(relation_idx)

    return [(h, t, relation_idx[r]) for h, t, r in edge_list if relation_cnt[r] > 1000], len(
        relation_idx
    )


def _make_synthetic_kg(n_entity, self_loop_relation, n_edges_per_entity=60):
    """Small synthetic KG dict (entity -> [(relation, tail), ...]) shaped
    exactly like the real dbpedia-derived `kg` consumed by `_edge_list`,
    with a self-loop relation on every entity and enough repeated
    relations (>1000 occurrences) to survive `_edge_list`'s frequency
    filter."""
    kg = defaultdict(list)
    for entity in range(n_entity):
        kg[entity].append((self_loop_relation, entity))
        for _ in range(n_edges_per_entity):
            tail = (entity + 1) % n_entity
            kg[entity].append((0, tail))
    return dict(kg)


class KBRD(nn.Module):
    def __init__(
        self,
        n_entity,
        n_relation,
        dim,
        kg,
        num_bases,
    ):
        super(KBRD, self).__init__()

        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = dim

        self.entity_emb = nn.Embedding(self.n_entity, self.dim)
        self.relation_emb = nn.Embedding(self.n_relation, self.dim)
        nn.init.kaiming_uniform_(self.entity_emb.weight.data)

        self.self_attn = SelfAttentionLayer(self.dim, self.dim)
        self.output = nn.Linear(self.dim, self.n_entity)

        self.kg = kg

        edge_list, self.n_relation = _edge_list(self.kg, self.n_entity)
        self.rgcn = RGCNConv(self.n_entity, self.dim, self.n_relation, num_bases=num_bases)
        edge_list = list(set(edge_list))
        edge_list_tensor = torch.LongTensor(edge_list)
        self.edge_idx = edge_list_tensor[:, :2].t()
        self.edge_type = edge_list_tensor[:, 2]

    def forward(self, seed_sets: list):
        # [batch size, dim]
        u_emb, nodes_features = self.user_representation(seed_sets)
        scores = F.linear(u_emb, nodes_features, self.output.bias)
        return scores

    def user_representation(self, seed_sets):
        nodes_features = self.rgcn(None, self.edge_idx, self.edge_type)

        user_representation_list = []
        for seed_set in seed_sets:
            if seed_set == []:
                user_representation_list.append(torch.zeros(self.dim))
                continue
            user_representation = nodes_features[seed_set]
            user_representation = self.self_attn(user_representation)
            user_representation_list.append(user_representation)
        return torch.stack(user_representation_list), nodes_features


class KBRDWrapper(nn.Module):
    """Fixed-shape wrapper: KBRD's `seed_sets` argument is a python list of
    variable-length per-user entity-id lists (not a tensor), so a fixed
    batch of seed sets is baked in here as a constant, matching the real
    `train_kbrd.py`/`kbrd.py` call `self.model(seed_sets, labels)` (traced
    entry point drops `labels`, see vendoring notes)."""

    N_ENTITY = 30
    SELF_LOOP_RELATION = 99
    SEED_SETS = [[1, 2, 3], [4, 5], [0, 6, 7, 8]]

    def __init__(self):
        super().__init__()
        kg = _make_synthetic_kg(self.N_ENTITY, self.SELF_LOOP_RELATION)
        self.model = KBRD(
            n_entity=self.N_ENTITY,
            n_relation=2,
            dim=16,
            kg=kg,
            num_bases=2,
        )

    def forward(self, dummy):
        # `dummy` (unused) exists purely so the wrapper has a tensor input
        # for TorchLens's forward-pass tracing entry point; the real
        # KBRD.forward takes only the python-list `seed_sets`.
        return self.model(self.SEED_SETS)


def build_kbrd():
    return KBRDWrapper()


def example_input_kbrd():
    return torch.zeros(1)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "KBRD (Knowledge-Based Recommender Dialog)",
        build_kbrd,
        example_input_kbrd,
        2019,
        "vendored-pytorch",
    ),
]
