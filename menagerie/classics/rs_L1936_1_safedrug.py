# SOURCE: vendored from ycq091044/SafeDrug @ main
# https://raw.githubusercontent.com/ycq091044/SafeDrug/main/src/models.py
# https://raw.githubusercontent.com/ycq091044/SafeDrug/main/src/layers.py
#
# Yang, Wu, Zhao, Xiao, Sun, 2021 (IJCAI) "SafeDrug: Dual Molecular Graph
# Encoders for Recommending Effective and Safe Drug Combinations". `SafeDrugModel`
# is the paper's flagship model: two per-visit GRU encoders (diagnosis/procedure
# codes) feed a query vector that is matched against (a) a molecular substructure
# "bipartite" embedding built from a DDI-safety mask (`MaskLinear` masked-weight
# linear layer over `ddi_mask_H`), and (b) a global drug embedding produced by
# running an actual Message-Passing Neural Network (`MolecularGraphNeuralNetwork`,
# the real molecular-fingerprint MPNN from the repo) once over the drug
# vocabulary's molecule set and projecting it with an `average_projection`
# matrix. This is the model's real, novel dual-encoder architecture -- vendored
# verbatim from `src/models.py` (`GCN`/`MaskLinear`/`MolecularGraphNeuralNetwork`/
# `SafeDrugModel`) and `src/layers.py` (`GraphConvolution`, used by `GCN`, which
# `SafeDrugModel.__init__` itself does not call but is kept for completeness of
# the vendored file). The sibling `GAMENet`/`DMNC`/`Leap`/`Retain` baselines in
# the same `models.py` are NOT the queue candidate and are omitted; `DMNC` is the
# only one requiring the external `dnc` package, which is why it is dropped here
# rather than vendored.
#
# `SafeDrugModel.__init__` takes precomputed structures (`ddi_adj`, `ddi_mask_H`,
# `MPNNSet`, `N_fingerprints`, `average_projection`) that a real training run
# derives from the MIMIC-III EHR + DrugBank molecule data. `build_safedrug()`
# below synthesizes tiny random versions of exactly those structures (small
# vocab, small fingerprint dict, a handful of molecules with 2-4 dummy atoms
# each) so the constructor runs its real code path -- including running the
# real `MolecularGraphNeuralNetwork` forward pass over the synthetic molecule
# set at construction time, exactly as upstream `SafeDrugModel.__init__` does.
# `SafeDrugModel.forward` takes one multi-visit patient record (a Python list of
# `[diag_codes, proc_codes]` per visit, each a list of vocab ints) rather than a
# tensor, so this ships as a MODULE (not a recipe): `example_input_safedrug()`
# returns that real nested-list structure, matching the repo's own input format.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

MENAGERIE_ZOO = "vendored-pytorch"

# ============================================================================
# layers.py (verbatim)
# ============================================================================


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

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


# ============================================================================
# models.py (verbatim: GCN, MaskLinear, MolecularGraphNeuralNetwork, SafeDrugModel)
# ============================================================================


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


class MaskLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskLinear, self).__init__()
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

    def forward(self, input, mask):
        weight = torch.mul(self.weight, mask)
        output = torch.mm(input, weight)

        if self.bias is not None:
            return output + self.bias
        else:
            return output


class MolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, N_fingerprint, dim, layer_hidden, device):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.device = device
        self.embed_fingerprint = nn.Embedding(N_fingerprint, dim).to(self.device)
        self.W_fingerprint = nn.ModuleList(
            [nn.Linear(dim, dim).to(self.device) for _ in range(layer_hidden)]
        )
        self.layer_hidden = layer_hidden

    def pad(self, matrices, pad_value):
        """Pad the list of matrices
        with a pad_value (e.g., 0) for batch proc essing.
        For example, given a list of matrices [A, B, C],
        we obtain a new matrix [A00, 0B0, 00C],
        where 0 is the zero (i.e., pad value) matrix.
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N))).to(self.device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i : i + m, j : j + n] = matrix
            i += m
            j += n
        return pad_matrices

    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors))
        return hidden_vectors + torch.mm(matrix, hidden_vectors)

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def mean(self, vectors, axis):
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(mean_vectors)

    def forward(self, inputs):
        """Cat or pad each input data for batch processing."""
        fingerprints, adjacencies, molecular_sizes = inputs
        fingerprints = torch.cat(fingerprints)
        adjacencies = self.pad(adjacencies, 0)

        """MPNN layer (update the fingerprint vectors)."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        for layer_idx in range(self.layer_hidden):
            hs = self.update(adjacencies, fingerprint_vectors, layer_idx)
            fingerprint_vectors = hs

        """Molecular vector by sum or mean of the fingerprint vectors."""
        molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)

        return molecular_vectors


class SafeDrugModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        ddi_adj,
        ddi_mask_H,
        MPNNSet,
        N_fingerprints,
        average_projection,
        emb_dim=256,
        device=torch.device("cpu:0"),
    ):
        super(SafeDrugModel, self).__init__()

        self.device = device

        # pre-embedding
        self.embeddings = nn.ModuleList([nn.Embedding(vocab_size[i], emb_dim) for i in range(2)])
        self.dropout = nn.Dropout(p=0.5)
        self.encoders = nn.ModuleList(
            [nn.GRU(emb_dim, emb_dim, batch_first=True) for _ in range(2)]
        )
        self.query = nn.Sequential(nn.ReLU(), nn.Linear(2 * emb_dim, emb_dim))

        # bipartite local embedding
        self.bipartite_transform = nn.Sequential(nn.Linear(emb_dim, ddi_mask_H.shape[1]))
        self.bipartite_output = MaskLinear(ddi_mask_H.shape[1], vocab_size[2], False)

        # MPNN global embedding
        self.MPNN_molecule_Set = list(zip(*MPNNSet))

        self.MPNN_emb = MolecularGraphNeuralNetwork(
            N_fingerprints, emb_dim, layer_hidden=2, device=device
        ).forward(self.MPNN_molecule_Set)
        self.MPNN_emb = torch.mm(
            average_projection.to(device=self.device),
            self.MPNN_emb.to(device=self.device),
        )
        self.MPNN_emb.to(device=self.device)
        self.MPNN_output = nn.Linear(vocab_size[2], vocab_size[2])
        self.MPNN_layernorm = nn.LayerNorm(vocab_size[2])

        # graphs, bipartite matrix
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.tensor_ddi_mask_H = torch.FloatTensor(ddi_mask_H).to(device)
        self.init_weights()

    def forward(self, input):
        # patient health representation
        i1_seq = []
        i2_seq = []

        def sum_embedding(embedding):
            return embedding.sum(dim=1).unsqueeze(dim=0)  # (1,1,dim)

        for adm in input:
            i1 = sum_embedding(
                self.dropout(
                    self.embeddings[0](torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device))
                )
            )  # (1,1,dim)
            i2 = sum_embedding(
                self.dropout(
                    self.embeddings[1](torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device))
                )
            )
            i1_seq.append(i1)
            i2_seq.append(i2)
        i1_seq = torch.cat(i1_seq, dim=1)  # (1,seq,dim)
        i2_seq = torch.cat(i2_seq, dim=1)  # (1,seq,dim)

        o1, h1 = self.encoders[0](i1_seq)
        o2, h2 = self.encoders[1](i2_seq)
        patient_representations = torch.cat([o1, o2], dim=-1).squeeze(dim=0)  # (seq, dim*2)
        query = self.query(patient_representations)[-1:, :]  # (seq, dim)

        # MPNN embedding
        MPNN_match = F.sigmoid(torch.mm(query, self.MPNN_emb.t()))
        MPNN_att = self.MPNN_layernorm(MPNN_match + self.MPNN_output(MPNN_match))

        # local embedding
        bipartite_emb = self.bipartite_output(
            F.sigmoid(self.bipartite_transform(query)), self.tensor_ddi_mask_H.t()
        )

        result = torch.mul(bipartite_emb, MPNN_att)

        neg_pred_prob = F.sigmoid(result)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)

        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        return result, batch_neg

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)


# ============================================================================
# build_/example_input_ harness
# ============================================================================


def _synthesize_safedrug_structures():
    """Build tiny random versions of the precomputed structures a real
    SafeDrug training run derives from MIMIC-III + DrugBank (diag/proc/drug
    vocab sizes, a DDI adjacency + DDI-safety mask, and a handful of toy
    molecules for the real MPNN forward pass)."""
    rng = np.random.RandomState(0)
    n_diag, n_proc, n_drug = 12, 10, 8
    vocab_size = (n_diag, n_proc, n_drug)

    ddi_adj = (rng.rand(n_drug, n_drug) > 0.7).astype(np.float64)
    ddi_adj = np.triu(ddi_adj, 1)
    ddi_adj = ddi_adj + ddi_adj.T

    n_substructures = 6
    ddi_mask_H = (rng.rand(n_drug, n_substructures) > 0.5).astype(np.float64)

    # Toy molecule set: one small molecule per drug, 2-4 dummy "atoms" each,
    # each atom a fingerprint id into a small fingerprint vocabulary.
    N_fingerprints = 16
    fingerprints_list = []
    adjacencies_list = []
    molecular_sizes = []
    for _ in range(n_drug):
        n_atoms = rng.randint(2, 5)
        fp = torch.LongTensor(rng.randint(0, N_fingerprints, size=n_atoms))
        adj = torch.FloatTensor((rng.rand(n_atoms, n_atoms) > 0.5).astype(np.float32))
        fingerprints_list.append(fp)
        adjacencies_list.append(adj)
        molecular_sizes.append(n_atoms)
    MPNNSet = list(zip(fingerprints_list, adjacencies_list, molecular_sizes))

    # average_projection maps each drug's fingerprint-vector rows (concatenated
    # across molecules) down to one row per drug (mean over its own atoms).
    average_projection = torch.zeros(n_drug, n_drug)
    for i in range(n_drug):
        average_projection[i, i] = 1.0

    return vocab_size, ddi_adj, ddi_mask_H, MPNNSet, N_fingerprints, average_projection


def build_safedrug():
    vocab_size, ddi_adj, ddi_mask_H, MPNNSet, N_fingerprints, average_projection = (
        _synthesize_safedrug_structures()
    )
    model = SafeDrugModel(
        vocab_size,
        ddi_adj,
        ddi_mask_H,
        MPNNSet,
        N_fingerprints,
        average_projection,
        emb_dim=32,
    )
    model.eval()
    return model


def example_input_safedrug():
    rng = np.random.RandomState(1)
    n_diag, n_proc = 12, 10
    n_visits = 3
    visits = []
    for _ in range(n_visits):
        diag_codes = rng.randint(0, n_diag, size=rng.randint(1, 4)).tolist()
        proc_codes = rng.randint(0, n_proc, size=rng.randint(1, 4)).tolist()
        visits.append([diag_codes, proc_codes])
    return (visits,)


MENAGERIE_ENTRIES = [
    ("SafeDrug", build_safedrug, example_input_safedrug, 2021, "vendored-pytorch"),
]
