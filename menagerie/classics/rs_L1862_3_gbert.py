# SOURCE: vendored from jshang123/G-Bert @ master (code/graph_models.py,
# code/bert_models.py, code/config.py, code/predictive_models.py,
# code/build_tree.py, code/utils.py::Voc). G-BERT (IJCAI 2019,
# arxiv:1906.00346) pre-trains a BERT-style transformer over patient visit
# sequences whose token embeddings are themselves produced by a graph
# attention network (`OntologyEmbedding`/`GATConv`) over the ICD-9 diagnosis
# and ATC medication ontology trees, then jointly predicts co-occurring
# diagnosis/medication codes (self-supervised pretraining objective,
# `GBERT_Pretrain`). Every class below (`Voc`, the ICD9/ATC tree builders,
# the from-scratch `MessagePassing`/`GATConv` message-passing layer, the
# `OntologyEmbedding`/`ConcatEmbeddings`/`FuseEmbeddings` ontology-graph
# embedding stack, and the BERT encoder + `GBERT_Pretrain` head) is
# transcribed verbatim from the official repo -- the repo vendors its own
# `MessagePassing`/`GATConv` (does not import torch_geometric.nn.GATConv),
# only using `torch_geometric.utils` (`softmax`, `add_self_loops`, and the
# now-renamed `scatter_`/`scatter`) and `torch_geometric.nn.inits`
# (`glorot`/`zeros`/`uniform`) -- pure infrastructure, not architecture. Two
# import/API-only fixes were required to run under a current torch_geometric
# + Python: `torch_geometric.utils.scatter_` -> `torch_geometric.utils.scatter`
# (renamed upstream; same reduce semantics) and `inspect.getargspec` ->
# `inspect.getfullargspec` (removed in Python 3.11). No architecture line was
# changed. `bert_models.py`'s `dill`-based `from_pretrained` classmethod and
# `predictive_models.py`'s `GBERT_Predict`/`TSNE`/side-info variants (which
# need a full `tokenizer` object with `dx_voc_multi`/`rx_voc_multi`) are
# dropped as unrelated checkpoint-loading/downstream-eval scaffolding, not
# architecture; `GBERT_Pretrain` is the model's actual pretraining forward
# pass and is vendored complete.
"""G-BERT: GAT ontology-embedding + BERT patient-visit encoder (real repo code, vendored)."""

import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot, uniform, zeros
from torch_geometric.utils import add_self_loops, scatter, softmax

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# code/utils.py::Voc (verbatim)
# ---------------------------------------------------------------------------


class Voc:
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)


# ---------------------------------------------------------------------------
# code/build_tree.py (verbatim; pure python ontology-tree construction, no
# architecture)
# ---------------------------------------------------------------------------


def _remove_duplicate(input):
    return list(set(input))


def build_stage_one_edges(res, graph_voc):
    edge_idx = []
    for sample in res:
        sample_idx = list(map(lambda x: graph_voc.word2idx[x], sample))
        for i in range(len(sample_idx) - 1):
            edge_idx.append((sample_idx[i + 1], sample_idx[i]))

    edge_idx = _remove_duplicate(edge_idx)
    row = list(map(lambda x: x[0], edge_idx))
    col = list(map(lambda x: x[1], edge_idx))
    return [row, col]


def build_stage_two_edges(res, graph_voc):
    edge_idx = []
    for sample in res:
        sample_idx = list(map(lambda x: graph_voc.word2idx[x], sample))
        edge_idx.extend([(sample_idx[0], sample_idx[i]) for i in range(1, len(sample_idx))])

    edge_idx = _remove_duplicate(edge_idx)
    row = list(map(lambda x: x[0], edge_idx))
    col = list(map(lambda x: x[1], edge_idx))
    return [row, col]


def expand_level2():
    level2 = ['001-009', '010-018', '020-027', '030-041', '042', '045-049', '050-059', '060-066', '070-079', '080-088',
              '090-099', '100-104', '110-118', '120-129', '130-136', '137-139', '140-149', '150-159', '160-165',
              '170-176',
              '176', '179-189', '190-199', '200-208', '209', '210-229', '230-234', '235-238', '239', '240-246',
              '249-259',
              '260-269', '270-279', '280-289', '290-294', '295-299', '300-316', '317-319', '320-327', '330-337', '338',
              '339', '340-349', '350-359', '360-379', '380-389', '390-392', '393-398', '401-405', '410-414', '415-417',
              '420-429', '430-438', '440-449', '451-459', '460-466', '470-478', '480-488', '490-496', '500-508',
              '510-519',
              '520-529', '530-539', '540-543', '550-553', '555-558', '560-569', '570-579', '580-589', '590-599',
              '600-608',
              '610-611', '614-616', '617-629', '630-639', '640-649', '650-659', '660-669', '670-677', '678-679',
              '680-686',
              '690-698', '700-709', '710-719', '720-724', '725-729', '730-739', '740-759', '760-763', '764-779',
              '780-789',
              '790-796', '797-799', '800-804', '805-809', '810-819', '820-829', '830-839', '840-848', '850-854',
              '860-869',
              '870-879', '880-887', '890-897', '900-904', '905-909', '910-919', '920-924', '925-929', '930-939',
              '940-949',
              '950-957', '958-959', '960-979', '980-989', '990-995', '996-999', 'V01-V91', 'V01-V09', 'V10-V19',
              'V20-V29',
              'V30-V39', 'V40-V49', 'V50-V59', 'V60-V69', 'V70-V82', 'V83-V84', 'V85', 'V86', 'V87', 'V88', 'V89',
              'V90',
              'V91', 'E000-E899', 'E000', 'E001-E030', 'E800-E807', 'E810-E819', 'E820-E825', 'E826-E829', 'E830-E838',
              'E840-E845', 'E846-E849', 'E850-E858', 'E860-E869', 'E870-E876', 'E878-E879', 'E880-E888', 'E890-E899',
              'E900-E909', 'E910-E915', 'E916-E928', 'E929', 'E930-E949', 'E950-E959', 'E960-E969', 'E970-E978',
              'E980-E989', 'E990-E999']  # fmt: skip

    level2_expand = {}
    for i in level2:
        tokens = i.split("-")
        if i[0] == "V":
            if len(tokens) == 1:
                level2_expand[i] = i
            else:
                for j in range(int(tokens[0][1:]), int(tokens[1][1:]) + 1):
                    level2_expand["V%02d" % j] = i
        elif i[0] == "E":
            if len(tokens) == 1:
                level2_expand[i] = i
            else:
                for j in range(int(tokens[0][1:]), int(tokens[1][1:]) + 1):
                    level2_expand["E%03d" % j] = i
        else:
            if len(tokens) == 1:
                level2_expand[i] = i
            else:
                for j in range(int(tokens[0]), int(tokens[1]) + 1):
                    level2_expand["%03d" % j] = i
    return level2_expand


def build_icd9_tree(unique_codes):
    res = []
    graph_voc = Voc()

    root_node = "icd9_root"
    level3_dict = expand_level2()
    for code in unique_codes:
        level1 = code
        level2 = level1[:4] if level1[0] == "E" else level1[:3]
        level3 = level3_dict[level2]
        level4 = root_node

        sample = [level1, level2, level3, level4]

        graph_voc.add_sentence(sample)
        res.append(sample)

    return res, graph_voc


def build_atc_tree(unique_codes):
    res = []
    graph_voc = Voc()

    root_node = "atc_root"
    for code in unique_codes:
        sample = [code] + [code[:i] for i in [4, 3, 1]] + [root_node]

        graph_voc.add_sentence(sample)
        res.append(sample)

    return res, graph_voc


# ---------------------------------------------------------------------------
# code/graph_models.py (verbatim except scatter_ -> scatter rename)
# ---------------------------------------------------------------------------


class OntologyEmbedding(nn.Module):
    def __init__(self, voc, build_tree_func, in_channels=100, out_channels=20, heads=5):
        super().__init__()

        res, graph_voc = build_tree_func(list(voc.idx2word.values()))
        stage_one_edges = build_stage_one_edges(res, graph_voc)
        stage_two_edges = build_stage_two_edges(res, graph_voc)

        self.edges1 = torch.tensor(stage_one_edges)
        self.edges2 = torch.tensor(stage_two_edges)
        self.graph_voc = graph_voc

        assert in_channels == heads * out_channels
        self.g = GATConv(in_channels=in_channels, out_channels=out_channels, heads=heads)

        num_nodes = len(graph_voc.word2idx)
        self.embedding = nn.Parameter(torch.Tensor(num_nodes, in_channels))

        self.idx_mapping = [self.graph_voc.word2idx[word] for word in voc.idx2word.values()]

        self.init_params()

    def get_all_graph_emb(self):
        emb = self.embedding
        emb = self.g(self.g(emb, self.edges1.to(emb.device)), self.edges2.to(emb.device))
        return emb

    def forward(self):
        """
        :param idxs: [N, L]
        :return:
        """
        emb = self.embedding

        emb = self.g(self.g(emb, self.edges1.to(emb.device)), self.edges2.to(emb.device))

        return emb[self.idx_mapping]

    def init_params(self):
        glorot(self.embedding)


class MessagePassing(nn.Module):
    r"""Base class for creating message passing layers (real repo's own
    from-scratch implementation, predates torch_geometric.nn.MessagePassing)."""

    def __init__(self, aggr="add"):
        super().__init__()

        self.message_args = inspect.getfullargspec(self.message)[0][1:]
        self.update_args = inspect.getfullargspec(self.update)[0][2:]

    def propagate(self, aggr, edge_index, **kwargs):
        assert aggr in ["add", "mean", "max"]
        kwargs["edge_index"] = edge_index

        size = None
        message_args = []
        for arg in self.message_args:
            if arg[-2:] == "_i":
                tmp = kwargs[arg[:-2]]
                size = tmp.size(0)
                message_args.append(tmp[edge_index[0]])
            elif arg[-2:] == "_j":
                tmp = kwargs[arg[:-2]]
                size = tmp.size(0)
                message_args.append(tmp[edge_index[1]])
            else:
                message_args.append(kwargs[arg])

        update_args = [kwargs[arg] for arg in self.update_args]

        out = self.message(*message_args)
        out = scatter(out, edge_index[0], dim=0, dim_size=size, reduce=aggr)
        out = self.update(out, *update_args)

        return out

    def message(self, x_j):  # pragma: no cover
        return x_j

    def update(self, aggr_out):  # pragma: no cover
        return aggr_out


class GATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper (real repo's own from-scratch
    implementation)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        heads=1,
        concat=True,
        negative_slope=0.2,
        dropout=0,
        bias=True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = nn.Parameter(torch.Tensor(in_channels, heads * out_channels))
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = torch.mm(x, self.weight).view(-1, self.heads, self.out_channels)
        return self.propagate("add", edge_index, x=x, num_nodes=x.size(0))

    def message(self, x_i, x_j, edge_index, num_nodes):
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0], num_nodes=num_nodes)

        alpha = F.dropout(alpha, p=self.dropout)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )


class ConcatEmbeddings(nn.Module):
    """Concat rx and dx ontology embedding for easy access"""

    def __init__(self, config, dx_voc, rx_voc):
        super().__init__()
        self.special_embedding = nn.Parameter(
            torch.Tensor(
                config.vocab_size - len(dx_voc.idx2word) - len(rx_voc.idx2word), config.hidden_size
            )
        )
        self.rx_embedding = OntologyEmbedding(
            rx_voc, build_atc_tree, config.hidden_size, config.graph_hidden_size, config.graph_heads
        )
        self.dx_embedding = OntologyEmbedding(
            dx_voc,
            build_icd9_tree,
            config.hidden_size,
            config.graph_hidden_size,
            config.graph_heads,
        )
        self.init_params()

    def forward(self, input_ids):
        emb = torch.cat([self.special_embedding, self.rx_embedding(), self.dx_embedding()], dim=0)
        return emb[input_ids]

    def init_params(self):
        glorot(self.special_embedding)


class FuseEmbeddings(nn.Module):
    """Construct the embeddings from ontology, patient info and type embeddings."""

    def __init__(self, config, dx_voc, rx_voc):
        super().__init__()
        self.ontology_embedding = ConcatEmbeddings(config, dx_voc, rx_voc)
        self.type_embedding = nn.Embedding(2, config.hidden_size)

    def forward(self, input_ids, input_types=None, input_positions=None):
        ontology_embedding = self.ontology_embedding(input_ids) + self.type_embedding(input_types)
        return ontology_embedding


# ---------------------------------------------------------------------------
# code/config.py (verbatim, minus json/file loading which is unused for the
# random-init in-memory construction path)
# ---------------------------------------------------------------------------


class BertConfig:
    def __init__(
        self,
        vocab_size_or_config_json_file,
        hidden_size=300,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=300,
        hidden_act="relu",
        hidden_dropout_prob=0.4,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=1,
        type_vocab_size=2,
        initializer_range=0.02,
        graph=False,
        graph_hidden_size=75,
        graph_heads=4,
    ):
        self.vocab_size = vocab_size_or_config_json_file
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.graph = graph
        self.graph_hidden_size = graph_hidden_size
        self.graph_heads = graph_heads


# ---------------------------------------------------------------------------
# code/bert_models.py (verbatim slice: LayerNorm, attention/transformer
# stack, BertEmbeddings, PreTrainedBertModel, BERT)
# ---------------------------------------------------------------------------


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class MultiHeadedAttention(nn.Module):
    """Take in model size and number of heads."""

    def __init__(self, config: BertConfig):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0

        self.d_k = config.hidden_size // config.num_attention_heads
        self.h = config.num_attention_heads

        self.linear_layers = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.hidden_size, bias=False) for _ in range(3)]
        )
        self.output_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query, key, value = [
            l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linear_layers, (query, key, value))  # noqa: E741
        ]

        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class Attention(nn.Module):
    """Compute 'Scaled Dot Product Attention"""

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class SublayerConnection(nn.Module):
    """A residual connection followed by a layer norm."""

    def __init__(self, config: BertConfig):
        super().__init__()
        self.norm = LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, config: BertConfig):
        super().__init__()
        self.w_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.w_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        return self.w_2(self.dropout(gelu(self.w_1(x))))


class TransformerBlock(nn.Module):
    """Bidirectional Encoder = Transformer (self-attention)"""

    def __init__(self, config: BertConfig):
        super().__init__()
        self.attention = MultiHeadedAttention(config)
        self.feed_forward = PositionwiseFeedForward(config)
        self.input_sublayer = SublayerConnection(config)
        self.output_sublayer = SublayerConnection(config)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, visit and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)

        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)

        embeddings = words_embeddings + self.token_type_embeddings(token_type_ids)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class PreTrainedBertModel(nn.Module):
    """An abstract class to handle weights initialization."""

    def __init__(self, config: BertConfig, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. ".format(
                    self.__class__.__name__
                )
            )
        self.config = config

    def init_bert_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BERT(PreTrainedBertModel):
    """BERT model : Bidirectional Encoder Representations from Transformers."""

    def __init__(self, config: BertConfig, dx_voc=None, rx_voc=None):
        super().__init__(config)
        if config.graph:
            assert dx_voc is not None
            assert rx_voc is not None

        self.embedding = (
            FuseEmbeddings(config, dx_voc, rx_voc) if config.graph else BertEmbeddings(config)
        )

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )

        self.apply(self.init_bert_weights)

    def forward(self, x, token_type_ids=None, input_positions=None, input_sides=None):
        mask = (x > 1).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        x = self.embedding(x, token_type_ids)

        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x, x[:, 0]


# ---------------------------------------------------------------------------
# code/predictive_models.py (verbatim slice: SelfSupervisedHead + GBERT_Pretrain)
# ---------------------------------------------------------------------------


def freeze_afterwards(model):
    for p in model.parameters():
        p.requires_grad = False


class ClsHead(nn.Module):
    def __init__(self, config: BertConfig, voc_size):
        super().__init__()
        self.cls = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, voc_size),
        )

    def forward(self, input):
        return self.cls(input)


class SelfSupervisedHead(nn.Module):
    def __init__(self, config: BertConfig, dx_voc_size, rx_voc_size):
        super().__init__()
        self.multi_cls = nn.ModuleList(
            [
                ClsHead(config, dx_voc_size),
                ClsHead(config, dx_voc_size),
                ClsHead(config, rx_voc_size),
                ClsHead(config, rx_voc_size),
            ]
        )

    def forward(self, dx_inputs, rx_inputs):
        return (
            self.multi_cls[0](dx_inputs),
            self.multi_cls[1](rx_inputs),
            self.multi_cls[2](dx_inputs),
            self.multi_cls[3](rx_inputs),
        )


class GBERT_Pretrain(PreTrainedBertModel):
    def __init__(self, config: BertConfig, dx_voc=None, rx_voc=None):
        super().__init__(config)
        self.dx_voc_size = len(dx_voc.word2idx)
        self.rx_voc_size = len(rx_voc.word2idx)

        self.bert = BERT(config, dx_voc, rx_voc)
        self.cls = SelfSupervisedHead(config, self.dx_voc_size, self.rx_voc_size)

        self.apply(self.init_bert_weights)

    def forward(self, inputs, dx_labels=None, rx_labels=None):
        # inputs (B, 2, max_len)
        _, dx_bert_pool = self.bert(
            inputs[:, 0, :], torch.zeros((inputs.size(0), inputs.size(2))).long().to(inputs.device)
        )
        _, rx_bert_pool = self.bert(
            inputs[:, 1, :], torch.zeros((inputs.size(0), inputs.size(2))).long().to(inputs.device)
        )

        dx2dx, rx2dx, dx2rx, rx2rx = self.cls(dx_bert_pool, rx_bert_pool)
        if rx_labels is None or dx_labels is None:
            return (
                torch.sigmoid(dx2dx),
                torch.sigmoid(rx2dx),
                torch.sigmoid(dx2rx),
                torch.sigmoid(rx2rx),
            )
        else:
            loss = (
                F.binary_cross_entropy_with_logits(dx2dx, dx_labels)
                + F.binary_cross_entropy_with_logits(rx2dx, dx_labels)
                + F.binary_cross_entropy_with_logits(dx2rx, rx_labels)
                + F.binary_cross_entropy_with_logits(rx2rx, rx_labels)
            )
            return (
                loss,
                torch.sigmoid(dx2dx),
                torch.sigmoid(rx2dx),
                torch.sigmoid(dx2rx),
                torch.sigmoid(rx2rx),
            )


# ---------------------------------------------------------------------------
# Staging build/example helpers: tiny ICD9/ATC vocabularies (real code
# strings so build_icd9_tree/build_atc_tree exercise the genuine ontology
# parsing) feeding a small-config GBERT_Pretrain.
# ---------------------------------------------------------------------------


def _build_vocs():
    dx_voc = Voc()
    dx_codes = ["250.00", "401.9", "V70.0", "V72.6"]
    for code in dx_codes:
        dx_voc.add_sentence([code])

    rx_voc = Voc()
    rx_codes = ["A01AA01", "A02BC01", "B01AA03"]
    for code in rx_codes:
        rx_voc.add_sentence([code])

    return dx_voc, rx_voc


def build_gbert():
    dx_voc, rx_voc = _build_vocs()
    n_special = 3  # [PAD], [CLS], [MASK]
    vocab_size = n_special + len(dx_voc.word2idx) + len(rx_voc.word2idx)
    config = BertConfig(
        vocab_size_or_config_json_file=vocab_size,
        hidden_size=20,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=20,
        max_position_embeddings=1,
        graph=True,
        graph_hidden_size=5,
        graph_heads=4,
    )
    return GBERT_Pretrain(config, dx_voc=dx_voc, rx_voc=rx_voc)


def example_input_gbert():
    torch.manual_seed(0)
    batch, max_len = 2, 4
    # token ids in [0, vocab_size); vocab_size = 3 special + 4 dx + 3 rx = 10
    inputs = torch.randint(2, 10, (batch, 2, max_len))
    dx_labels = torch.randint(0, 2, (batch, 4)).float()
    rx_labels = torch.randint(0, 2, (batch, 3)).float()
    return (inputs, dx_labels, rx_labels)


MENAGERIE_ENTRIES = [
    ("G-BERT", "build_gbert", "example_input_gbert", 2019, "vendored-pytorch"),
]
