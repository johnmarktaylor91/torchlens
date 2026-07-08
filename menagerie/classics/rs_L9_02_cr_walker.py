# SOURCE: vendored from truthless11/CR-Walker @ 9e69b09 (main)
# (model/CR_walker.py, model/utterance_embedder.py, model/graph_embedder.py,
#  model/intent_selector.py, model/graph_walker.py)
#
# CR-Walker (Ma, Lin, Yao, Sun, "CR-Walker: Tree-Structured Graph Reasoning
# and Dialog Acts for Conversational Recommendation", EMNLP 2021). Real
# architecture: a BERT+RNN utterance encoder, an R-GCN knowledge-graph node
# embedder (relational graph conv over the KG), an intent classifier, and a
# tree-structured "graph walker" that performs two hops of attention-gated
# reasoning over the KG -- at each hop, self-attention pools a "user
# portrait" from previously-mentioned graph nodes, then a learned gate mixes
# the utterance representation and the user portrait into a query vector
# that is dot-producted against candidate-node embeddings to score which
# graph nodes/dialog acts to walk to next. This is the paper's actual
# contribution (not a usage-only variant of an existing library model), so
# it is vendored (rung 2) rather than recipe'd.
#
# Vendoring notes (imports/config fixes only, architecture untouched):
#   - `Utterance_Embedder` originally loaded `BertModel.from_pretrained(
#     'bert-base-uncased')` / `BertTokenizer.from_pretrained(...)`; replaced
#     with a directly-constructed tiny random-init `BertModel(BertConfig(
#     ...))` fed pre-tokenized integer ids directly (no tokenizer, no
#     network access), per the menagerie "tiny config, random init"
#     convention. The `forward()` BERT-embed + RNN-over-turns logic is
#     unchanged.
#   - `prepare_data()` (BERT tokenization / nltk word-tokenization / dialog
#     padding) and `Utterance_Embedder`'s `self.key2index` (loaded from a
#     local `data/key2index_3rd.json`) are data-preprocessing utilities, not
#     part of the traced network graph, and are dropped; the traced entry
#     builds already-tokenized/already-batched tensors directly.
#   - `Graph_Embedder.__init__` originally called `self.initialize_weights()`
#     inline (kept) but `concept_edge_list4GCN()` -- which loads local
#     ConceptNet JSON/TXT data files and is only invoked when
#     `word_net=True` -- is dropped along with the `word_net` path; the
#     traced entry uses the repo's own default `word_net=False`
#     (`Graph_Embedder.__init__(..., word_net=False)`), so `self.gcn2` /
#     `self.concept_edge_sets` are never constructed in the original either.
#   - `graph_walker.py` in the original repo does
#     `from conf import args,add_generic_args` (a plain module-level `{}`
#     plus a function, safe to import) and, more importantly,
#     `sys.path.append("..")` + `from data.redial import ReDial` +
#     `from data.gorecdial import GoRecDial` (dataset-loading classes used
#     nowhere in `Graph_Walker`/`Attention`/`Self_Attention`) -- those two
#     dataset imports are dropped since this repo's `data/` package is not
#     installed here and the classes are architecturally unused dead
#     imports for the traced module.
#   - `Attention`/`Dot_Attention`/`Self_Attention`/`Graph_Walker` are
#     otherwise copied verbatim, including the `torch_scatter.scatter`
#     grouped-softmax attention mechanism.
#   - The traced entry exercises `ProRec.forward_pretrain` (renamed
#     `CRWalker.forward_pretrain` here, class renamed only to avoid the
#     generic `ProRec` name colliding across menagerie modules), which is
#     the repo's own alignment-pretraining forward pass over
#     `Utterance_Embedder` + `Graph_Embedder` + `IntentSelector` +
#     `Graph_Walker.tile_context`; `ProRec.forward`/`forward_gorecdial`
#     (the full two-hop tree-walk training forward passes) additionally
#     require repo-specific negative-sampling data preparation
#     (`prepare_data_redial`/`prepare_pretrain`, which read live dataset
#     files and call `random.sample` against corpus-scale candidate pools)
#     that has no menagerie-tiny equivalent; `Graph_Walker.forward` (the
#     two-hop attention-gated tree walk) is exercised directly as a second
#     traced entry below with hand-built tiny synthetic index tensors of
#     the same shapes `ProRec.forward` would pass it, so the walker's real
#     forward computation (not just its pretraining regularizer) is
#     captured.

import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv
from torch_scatter import scatter
from transformers import BertConfig, BertModel

# ---------------------------------------------------------------------------
# model/intent_selector.py (vendored verbatim)
# ---------------------------------------------------------------------------


class IntentSelector(nn.Module):
    def __init__(self, utter_embed_size, atten_hidden=20):
        super().__init__()
        self.utter_embed_size = utter_embed_size
        self.atten_hidden = 20
        self.W1 = nn.Linear(utter_embed_size, 64)
        self.W2 = nn.Linear(64, 3)

    def forward(self, utterance_embed):
        layer1 = torch.relu(self.W1(utterance_embed))
        intents = self.W2(layer1)
        return intents


# ---------------------------------------------------------------------------
# model/graph_embedder.py (vendored; word_net=False path only, see header)
# ---------------------------------------------------------------------------


class Graph_Embedder(nn.Module):
    def __init__(self, num_nodes, embed_size=64, num_relations=12, num_bases=15):
        super().__init__()
        self.embed_size = embed_size
        self.num_nodes = num_nodes
        self.gcn1 = RGCNConv(self.embed_size, self.embed_size, num_relations, num_bases)
        self.initialize_weights()

    def forward(self, edge_type, edge_index):
        graph_features = torch.relu(self.gcn1(self.init_features, edge_index, edge_type))
        return graph_features, None

    def initialize_weights(self):
        features = torch.Tensor(self.num_nodes, self.embed_size).uniform_(
            -1 / (self.embed_size**0.5), 1 / (self.embed_size**0.5)
        )
        features_norm = torch.nn.functional.normalize(features, p=2, dim=1, eps=1e-12, out=None)
        self.init_features = nn.Parameter(features_norm)


# ---------------------------------------------------------------------------
# model/utterance_embedder.py (vendored; BERT swapped for tiny random-init,
# tokenizer/data-prep utilities dropped -- see header)
# ---------------------------------------------------------------------------


class Utterance_Embedder(nn.Module):
    def __init__(
        self, bert_config: BertConfig, rnn_type="RNN_TANH", rnn_hidden=64, dropout=0.0, num_turns=10
    ):
        super().__init__()
        self.rnn_hidden = rnn_hidden
        self.num_turns = num_turns
        self.model = BertModel(bert_config)
        if rnn_type in ["LSTM", "GRU"]:
            self.rnn = getattr(nn, rnn_type)(bert_config.hidden_size, rnn_hidden, dropout=dropout)
        else:
            nonlinearity = {"RNN_TANH": "tanh", "RNN_RELU": "relu"}[rnn_type]
            self.rnn = nn.RNN(
                bert_config.hidden_size, rnn_hidden, nonlinearity=nonlinearity, dropout=dropout
            )
        self.rnn_type = rnn_type

    def forward(self, input_ids, attention_mask, max_len, init_hidden):
        bert_embed = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        bert_embed = bert_embed.view(
            -1, self.num_turns, max_len, self.model.config.hidden_size
        ).permute(1, 0, 2, 3)
        sentence_rep = bert_embed[:, :, 0, :]
        output, hidden = self.rnn(sentence_rep, init_hidden)
        output = output[-1, :, :]
        return output


# ---------------------------------------------------------------------------
# model/graph_walker.py (vendored verbatim; dataset-loader imports dropped)
# ---------------------------------------------------------------------------


class Self_Attention(nn.Module):
    def __init__(self, embed_size, atten_hidden):
        super().__init__()
        self.embed_size = embed_size
        self.atten_hidden = atten_hidden
        self.Attention_W = nn.Linear(embed_size, atten_hidden, bias=False)
        self.Attention_V = nn.Linear(atten_hidden, 1, bias=False)

    def forward(self, embed, batch_index=None):
        beta = self.Attention_V(torch.tanh(self.Attention_W(embed))).squeeze(-1)
        exp_beta = torch.exp(beta)
        grouped_sum = scatter(exp_beta, batch_index, dim=0, reduce="sum")
        tiled_sum = self.tile_sum(grouped_sum, batch_index).squeeze()
        alpha = exp_beta.div(tiled_sum).view(-1, 1)
        result = scatter(alpha.mul(embed), batch_index, dim=0, reduce="sum")
        return result

    def tile_sum(self, batch_sum, batch_index):
        ones = torch.ones_like(batch_index)
        graph_size = scatter(ones, batch_index, dim=0, reduce="sum")
        batch_size = graph_size.size()[0]
        tile_sum = []
        for i in range(batch_size):
            repeated = batch_sum[i].repeat(graph_size[i], 1)
            tile_sum.append(repeated)
        tile_sum = torch.cat(tile_sum, dim=0)
        return tile_sum


class Graph_Walker(nn.Module):
    def __init__(
        self,
        graph_embed_size=64,
        utterance_embed_size=64,
        attention_hidden_dim=20,
        nagetive_sample_ratio=3,
    ):
        super().__init__()
        self.graph_embed_size = graph_embed_size
        self.utterance_embed_size = utterance_embed_size
        self.nagetive_sample_ratio = nagetive_sample_ratio
        self.context_embed_size = graph_embed_size * 2 + 2 * utterance_embed_size
        self.attention_hidden_dim = attention_hidden_dim

        self.Wu = nn.Linear(2 * graph_embed_size, 1)
        self.W1 = nn.Linear(self.context_embed_size, 1)
        self.W2 = nn.Linear(self.context_embed_size, 1)
        self.parameter_list = [self.W1, self.W2]

        self.intent_embed = nn.Embedding(3, utterance_embed_size)
        self.user_attention = Self_Attention(graph_embed_size, attention_hidden_dim)

    def get_user_portrait(self, mention_index, mention_batch_index, graph_embed):
        mention_embed = graph_embed.index_select(0, mention_index)
        user_portrait = self.user_attention(mention_embed, batch_index=mention_batch_index)
        return user_portrait

    def forward_single_layer(
        self,
        layer_num,
        utter_embed,
        user_portrait,
        graph_embed,
        sel_index,
        sel_batch_index,
        sel_group_index,
        grp_batch_index,
        last_index,
        intent_index,
        score_mask,
        last_weight=None,
        ret_partial_score=False,
    ):
        context_embed = torch.cat([utter_embed, user_portrait], dim=-1)
        abstract_embed = torch.zeros(1, self.utterance_embed_size, device=graph_embed.device)
        graph_embed_e = torch.cat([graph_embed, abstract_embed], dim=0)

        graph_features = graph_embed_e.index_select(0, sel_index)
        tiled_context_embed = self.tile_context(context_embed, grp_batch_index)
        start_point_embed = graph_embed_e.index_select(0, last_index)
        itt_embed = self.intent_embed(intent_index)
        grp_context = torch.cat([tiled_context_embed, start_point_embed, itt_embed], dim=-1)
        weights = torch.sigmoid(self.parameter_list[layer_num](grp_context))

        tiled_weights = self.tile_context(weights, sel_group_index)
        tiled_utter = self.tile_context(utter_embed, sel_batch_index)
        tiled_portrait = self.tile_context(user_portrait, sel_batch_index)

        query_vector = tiled_utter * tiled_weights + tiled_portrait * (1 - tiled_weights)
        scores = torch.sum(query_vector * graph_features, dim=-1)

        if last_weight is not None:
            tiled_last_weight = self.tile_context(last_weight, sel_batch_index)
            last_query_vector = tiled_utter * tiled_last_weight + tiled_portrait * (
                1 - tiled_last_weight
            )
            last_scores = torch.sum(last_query_vector * graph_features, dim=-1)
            partial_score = score_mask * scores
            final_score = last_scores + partial_score
            if ret_partial_score:
                return final_score, None, partial_score
            return final_score, None
        if ret_partial_score:
            return scores, weights, None
        return scores, weights

    def tile_context(self, context, batch_index):
        ones = torch.ones_like(batch_index)
        graph_size = scatter(ones, batch_index, dim=0, reduce="sum")
        batch_size = graph_size.size()[0]
        tile_ctx = []
        for i in range(batch_size):
            repeated = context[i].repeat(graph_size[i], 1)
            tile_ctx.append(repeated)
        tile_ctx = torch.cat(tile_ctx, dim=0)
        return tile_ctx

    def forward(
        self,
        graph_embed,
        utterance_embed,
        mention_index,
        mention_batch_index,
        sel_indices,
        sel_batch_indices,
        sel_group_indices,
        grp_batch_indices,
        last_indices,
        intent_indices,
        score_masks,
        ret_portrait=False,
    ):
        user_portrait = self.get_user_portrait(mention_index, mention_batch_index, graph_embed)
        paths = []
        last_weight = None
        for i in range(2):
            scores, last_weight = self.forward_single_layer(
                i,
                utterance_embed,
                user_portrait,
                graph_embed,
                sel_indices[i],
                sel_batch_indices[i],
                sel_group_indices[i],
                grp_batch_indices[i],
                last_indices[i],
                intent_indices[i],
                score_masks[i],
                last_weight,
            )
            paths.append(scores)
        if ret_portrait:
            return paths, user_portrait
        return paths


# ---------------------------------------------------------------------------
# model/CR_walker.py (vendored: ProRec.forward_pretrain, renamed CRWalker)
# ---------------------------------------------------------------------------


class CRWalker(nn.Module):
    """CR-Walker's top-level module (`ProRec` in the original repo)."""

    def __init__(
        self,
        bert_config,
        num_nodes,
        rnn_type="RNN_TANH",
        utter_embed_size=64,
        num_turns=10,
        num_relations=12,
        num_bases=15,
        graph_embed_size=64,
        atten_hidden=20,
    ):
        super().__init__()
        self.null_idx = num_nodes - 1

        self.utter_embedder = Utterance_Embedder(
            bert_config, rnn_type, utter_embed_size, 0.0, num_turns
        )
        self.graph_embedder = Graph_Embedder(num_nodes, graph_embed_size, num_relations, num_bases)
        self.intent_selector = IntentSelector(utter_embed_size, atten_hidden)
        self.graph_walker = Graph_Walker(graph_embed_size, utter_embed_size, atten_hidden)

        self.alignment_loss = nn.BCEWithLogitsLoss(reduction="sum")
        self.intent_loss = nn.CrossEntropyLoss(reduction="sum")
        self.Wa = nn.Linear(utter_embed_size, graph_embed_size, bias=False)

    def forward_pretrain(
        self,
        input_ids,
        attention_mask,
        max_len,
        init_hidden,
        edge_type,
        edge_index,
        alignment_index,
        alignment_batch_index,
        alignment_label,
        intent_label,
    ):
        utter_embed = self.utter_embedder.forward(input_ids, attention_mask, max_len, init_hidden)
        intent = self.intent_selector.forward(utter_embed)
        graph_embed, _word_embed = self.graph_embedder.forward(edge_type, edge_index)
        graph_features = graph_embed.index_select(0, alignment_index)

        tiled_utter = self.graph_walker.tile_context(utter_embed, alignment_batch_index)
        logits = torch.sum(self.Wa(tiled_utter) * graph_features, dim=-1)

        loss_a = self.alignment_loss(logits, alignment_label)
        intent_loss = self.intent_loss(intent, intent_label)
        return loss_a + intent_loss


MENAGERIE_ZOO = "vendored-pytorch"

_NUM_NODES = 40
_GRAPH_EMBED = 16
_UTTER_EMBED = 16
_NUM_TURNS = 3
_MAX_LEN = 6
_BATCH = 2
_NUM_RELATIONS = 4
_NUM_BASES = 2
_VOCAB_SIZE = 64


def _bert_config():
    return BertConfig(
        vocab_size=_VOCAB_SIZE,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
    )


def build_cr_walker():
    model = CRWalker(
        _bert_config(),
        num_nodes=_NUM_NODES,
        utter_embed_size=_UTTER_EMBED,
        num_turns=_NUM_TURNS,
        num_relations=_NUM_RELATIONS,
        num_bases=_NUM_BASES,
        graph_embed_size=_GRAPH_EMBED,
    )
    model.eval()
    return model


def example_input_cr_walker():
    n_seqs = _BATCH * _NUM_TURNS
    input_ids = torch.randint(1, _VOCAB_SIZE, (n_seqs, _MAX_LEN))
    attention_mask = torch.ones(n_seqs, _MAX_LEN, dtype=torch.long)
    init_hidden = torch.zeros(1, _BATCH, _UTTER_EMBED)

    n_edges = 30
    edge_index = torch.randint(0, _NUM_NODES, (2, n_edges))
    edge_type = torch.randint(0, _NUM_RELATIONS, (n_edges,))

    n_align = 8
    alignment_index = torch.randint(0, _NUM_NODES, (n_align,))
    alignment_batch_index = torch.cat(
        [torch.zeros(4, dtype=torch.long), torch.ones(4, dtype=torch.long)]
    )
    alignment_label = torch.randint(0, 2, (n_align,)).float()
    intent_label = torch.randint(0, 3, (_BATCH,))

    return (
        input_ids,
        attention_mask,
        _MAX_LEN,
        init_hidden,
        edge_type,
        edge_index,
        alignment_index,
        alignment_batch_index,
        alignment_label,
        intent_label,
    )


class _CRWalkerPretrainEntry(nn.Module):
    """Thin `forward()` shim so `CRWalker.forward_pretrain` (the repo's own
    method name) is what gets traced, matching torchlens' `model(*inputs)`
    capture convention."""

    def __init__(self, num_nodes=_NUM_NODES):
        super().__init__()
        self.walker = build_cr_walker()

    def forward(self, *args):
        return self.walker.forward_pretrain(*args)


def build_cr_walker_entry():
    m = _CRWalkerPretrainEntry()
    m.eval()
    return m


class _GraphWalkerEntry(nn.Module):
    """Traces `Graph_Walker.forward` directly (the paper's two-hop
    tree-structured reasoning core) with hand-built tiny synthetic index
    tensors of the shapes `ProRec.forward` would otherwise assemble via
    `Graph_Walker.prepare_data` against a real knowledge graph + dialog
    batch (see module header: that data assembly is corpus/dataset
    plumbing, not part of the traced architecture)."""

    def __init__(self):
        super().__init__()
        self.graph_walker = Graph_Walker(_GRAPH_EMBED, _UTTER_EMBED, attention_hidden_dim=20)

    def forward(
        self,
        graph_embed,
        utterance_embed,
        mention_index,
        mention_batch_index,
        sel_index_0,
        sel_batch_index_0,
        sel_group_index_0,
        grp_batch_index_0,
        last_index_0,
        intent_index_0,
        score_mask_0,
        sel_index_1,
        sel_batch_index_1,
        sel_group_index_1,
        grp_batch_index_1,
        last_index_1,
        intent_index_1,
        score_mask_1,
    ):
        paths = self.graph_walker.forward(
            graph_embed,
            utterance_embed,
            mention_index,
            mention_batch_index,
            [sel_index_0, sel_index_1],
            [sel_batch_index_0, sel_batch_index_1],
            [sel_group_index_0, sel_group_index_1],
            [grp_batch_index_0, grp_batch_index_1],
            [last_index_0, last_index_1],
            [intent_index_0, intent_index_1],
            [score_mask_0, score_mask_1],
        )
        return paths


def build_graph_walker():
    m = _GraphWalkerEntry()
    m.eval()
    return m


def example_input_graph_walker():
    graph_embed = torch.randn(_NUM_NODES, _GRAPH_EMBED)
    utterance_embed = torch.randn(_BATCH, _UTTER_EMBED)

    mention_index = torch.randint(0, _NUM_NODES, (4,))
    mention_batch_index = torch.tensor([0, 0, 1, 1])

    # Hop 0: one candidate group per batch item.
    sel_index_0 = torch.randint(0, _NUM_NODES, (5,))
    sel_batch_index_0 = torch.tensor([0, 0, 0, 1, 1])
    sel_group_index_0 = torch.tensor([0, 0, 0, 1, 1])
    grp_batch_index_0 = torch.tensor([0, 1])
    last_index_0 = torch.tensor([_NUM_NODES - 1, _NUM_NODES - 1])
    intent_index_0 = torch.tensor([2, 1])
    score_mask_0 = torch.ones(5)

    # Hop 1: two candidate groups (one per batch item, following hop-0 leaf).
    sel_index_1 = torch.randint(0, _NUM_NODES, (6,))
    sel_batch_index_1 = torch.tensor([0, 0, 0, 1, 1, 1])
    sel_group_index_1 = torch.tensor([0, 0, 0, 1, 1, 1])
    grp_batch_index_1 = torch.tensor([0, 1])
    last_index_1 = torch.tensor([sel_index_0[0].item(), sel_index_0[3].item()])
    intent_index_1 = torch.tensor([2, 1])
    score_mask_1 = torch.ones(6)

    return (
        graph_embed,
        utterance_embed,
        mention_index,
        mention_batch_index,
        sel_index_0,
        sel_batch_index_0,
        sel_group_index_0,
        grp_batch_index_0,
        last_index_0,
        intent_index_0,
        score_mask_0,
        sel_index_1,
        sel_batch_index_1,
        sel_group_index_1,
        grp_batch_index_1,
        last_index_1,
        intent_index_1,
        score_mask_1,
    )


MENAGERIE_ENTRIES = [
    (
        "CR-Walker (Utterance/Graph Alignment Pretrain)",
        build_cr_walker_entry,
        example_input_cr_walker,
        2021,
        "vendored-pytorch",
    ),
    (
        "CR-Walker (Tree-Structured Graph Walker)",
        build_graph_walker,
        example_input_graph_walker,
        2021,
        "vendored-pytorch",
    ),
]
