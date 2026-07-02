# SOURCE: vendored from sjy1203/GAMENet @ master (code/models.py `GAMENet`
# class + code/layers.py `GraphConvolution`). The `GAMENet` class itself only
# uses torch/numpy (the file-level `from dnc import DNC` import is used only
# by the separate `DMNC` baseline class in the same file, which is dropped
# here -- GAMENet does not reference DNC anywhere in its own __init__/forward).
# Copied verbatim except for that unused import and the relative `from layers
# import GraphConvolution`, inlined as `GraphConvolution` in this file.
"""GAMENet: graph-augmented memory network for medication recommendation.

AAAI 2019, "GAMENet: Graph Augmented MEmory Networks for Recommending
Medication Combination" (Shang et al., arxiv:1809.01852). GRU-encoded
per-visit diagnosis/procedure history feeds an attention query over a dual
GCN "drug memory" (EHR co-prescription graph minus a DDI graph), combined
with a dynamic memory bank over the patient's own visit history.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

MENAGERIE_ZOO = "vendored-pytorch"


class GraphConvolution(nn.Module):
    """Simple GCN layer, similar to https://arxiv.org/abs/1609.02907."""

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, adj, device=torch.device("cpu:0")):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        adj = self.normalize(adj + np.eye(adj.shape[0]))

        self.adj = torch.FloatTensor(adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        node_embedding = self.gcn1(self.x, self.adj)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj)
        return node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx


class GAMENet(nn.Module):
    def __init__(
        self,
        vocab_size,
        ehr_adj,
        ddi_adj,
        emb_dim=64,
        device=torch.device("cpu:0"),
        ddi_in_memory=True,
    ):
        super(GAMENet, self).__init__()
        K = len(vocab_size)
        self.K = K
        self.vocab_size = vocab_size
        self.device = device
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.ddi_in_memory = ddi_in_memory
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(K - 1)]
        )
        self.dropout = nn.Dropout(p=0.4)

        self.encoders = nn.ModuleList(
            [nn.GRU(emb_dim, emb_dim * 2, batch_first=True) for _ in range(K - 1)]
        )

        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )

        self.ehr_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ehr_adj, device=device)
        self.ddi_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ddi_adj, device=device)
        self.inter = nn.Parameter(torch.FloatTensor(1))

        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, vocab_size[2]),
        )

        self.init_weights()

    def forward(self, input):
        # input (adm, 3, codes)

        # generate medical embeddings and queries
        i1_seq = []
        i2_seq = []

        def mean_embedding(embedding):
            return embedding.mean(dim=1).unsqueeze(dim=0)  # (1,1,dim)

        for adm in input:
            i1 = mean_embedding(
                self.dropout(
                    self.embeddings[0](torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device))
                )
            )  # (1,1,dim)
            i2 = mean_embedding(
                self.dropout(
                    self.embeddings[1](torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device))
                )
            )
            i1_seq.append(i1)
            i2_seq.append(i2)
        i1_seq = torch.cat(i1_seq, dim=1)  # (1,seq,dim)
        i2_seq = torch.cat(i2_seq, dim=1)  # (1,seq,dim)

        o1, h1 = self.encoders[0](i1_seq)  # o1:(1, seq, dim*2) hi:(1,1,dim*2)
        o2, h2 = self.encoders[1](i2_seq)
        patient_representations = torch.cat([o1, o2], dim=-1).squeeze(dim=0)  # (seq, dim*4)
        queries = self.query(patient_representations)  # (seq, dim)

        # graph memory module
        # I: generate current input
        query = queries[-1:]  # (1,dim)

        # G: generate graph memory bank and insert history information
        if self.ddi_in_memory:
            drug_memory = self.ehr_gcn() - self.ddi_gcn() * self.inter  # (size, dim)
        else:
            drug_memory = self.ehr_gcn()

        if len(input) > 1:
            history_keys = queries[: (queries.size(0) - 1)]  # (seq-1, dim)

            history_values = np.zeros((len(input) - 1, self.vocab_size[2]))
            for idx, adm in enumerate(input):
                if idx == len(input) - 1:
                    break
                history_values[idx, adm[2]] = 1
            history_values = torch.FloatTensor(history_values).to(self.device)  # (seq-1, size)

        # O: read from global memory bank and dynamic memory bank
        key_weights1 = F.softmax(torch.mm(query, drug_memory.t()), dim=-1)  # (1, size)
        fact1 = torch.mm(key_weights1, drug_memory)  # (1, dim)

        if len(input) > 1:
            visit_weight = F.softmax(torch.mm(query, history_keys.t()), dim=-1)  # (1, seq-1)
            weighted_values = visit_weight.mm(history_values)  # (1, size)
            fact2 = torch.mm(weighted_values, drug_memory)  # (1, dim)
        else:
            fact2 = fact1
        # R: convert O and predict
        output = self.output(torch.cat([query, fact1, fact2], dim=-1))  # (1, dim)

        if self.training:
            neg_pred_prob = torch.sigmoid(output)
            neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)
            batch_neg = neg_pred_prob.mul(self.tensor_ddi_adj).mean()

            return output, batch_neg
        else:
            return output

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)

        self.inter.data.uniform_(-initrange, initrange)


# ---------------------------------------------------------------------------
# Staging build/example helpers (tiny vocab/adjacency sizes, matches the
# repo's default emb_dim=64 config family, scaled down for fast tracing).
# ---------------------------------------------------------------------------


def build_gamenet():
    rng = np.random.default_rng(0)
    diag_size, proc_size, med_size = 12, 10, 8
    ehr_adj = (rng.random((med_size, med_size)) > 0.7).astype(np.float32)
    ddi_adj = (rng.random((med_size, med_size)) > 0.8).astype(np.float32)
    model = GAMENet(
        vocab_size=(diag_size, proc_size, med_size),
        ehr_adj=ehr_adj,
        ddi_adj=ddi_adj,
        emb_dim=8,
        device=torch.device("cpu"),
        ddi_in_memory=True,
    )
    model.eval()
    return model


def example_input_gamenet():
    rng = np.random.default_rng(1)
    diag_size, proc_size, med_size = 12, 10, 8
    # two visits, each a [diag_codes, proc_codes, med_codes] admission record
    admissions = [
        [
            rng.integers(0, diag_size, size=4).tolist(),
            rng.integers(0, proc_size, size=3).tolist(),
            rng.integers(0, med_size, size=2).tolist(),
        ],
        [
            rng.integers(0, diag_size, size=3).tolist(),
            rng.integers(0, proc_size, size=2).tolist(),
            rng.integers(0, med_size, size=3).tolist(),
        ],
    ]
    return (admissions,)


MENAGERIE_ENTRIES = [
    ("GAMENet", "build_gamenet", "example_input_gamenet", 2019, "vendored-pytorch"),
]
