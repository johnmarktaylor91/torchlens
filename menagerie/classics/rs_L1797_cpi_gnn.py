# SOURCE: vendored from masashitsubaki/CPI_prediction @ c82e0c1f1c78da54977880167361b5e93f111c43 (code/run_training.py)
"""CPI-GNN: compound-protein interaction prediction (NeurIPS 2018 MLCB / Bioinformatics
2019 masashitsubaki et al.). A graph-CNN (message passing on the molecular fingerprint
graph) encodes the compound; an attention-weighted 1D-CNN over amino-acid n-grams encodes
the protein; the two vectors are concatenated and passed through an MLP interaction head.

Vendored verbatim from ``code/run_training.py``'s
``CompoundProteinInteractionPrediction`` class -- only the training/eval scaffolding
(Trainer/Tester/data loading/``__main__``) is dropped, and the module-level hyperparameter
globals (``n_fingerprint``, ``n_word``, ``dim``, ``layer_gnn``, ``layer_cnn``,
``layer_output``) that the original script sets from ``sys.argv`` are pinned here as small
menagerie-sized constants instead. No architectural change.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"

# menagerie-sized hyperparameters (originally CLI args in run_training.py's __main__)
n_fingerprint = 100
n_word = 100
dim = 10
layer_gnn = 3
window = 11
layer_cnn = 3
layer_output = 3


class CompoundProteinInteractionPrediction(nn.Module):
    def __init__(self):
        super(CompoundProteinInteractionPrediction, self).__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.embed_word = nn.Embedding(n_word, dim)
        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_gnn)])
        self.W_cnn = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=1,
                    kernel_size=2 * window + 1,
                    stride=1,
                    padding=window,
                )
                for _ in range(layer_cnn)
            ]
        )
        self.W_attention = nn.Linear(dim, dim)
        self.W_out = nn.ModuleList([nn.Linear(2 * dim, 2 * dim) for _ in range(layer_output)])
        self.W_interaction = nn.Linear(2 * dim, 2)

    def gnn(self, xs, A, layer):
        for i in range(layer):
            hs = torch.relu(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
        # return torch.unsqueeze(torch.sum(xs, 0), 0)
        return torch.unsqueeze(torch.mean(xs, 0), 0)

    def attention_cnn(self, x, xs, layer):
        """The attention mechanism is applied to the last layer of CNN."""

        xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
        for i in range(layer):
            xs = torch.relu(self.W_cnn[i](xs))
        xs = torch.squeeze(torch.squeeze(xs, 0), 0)

        h = torch.relu(self.W_attention(x))
        hs = torch.relu(self.W_attention(xs))
        weights = torch.tanh(F.linear(h, hs))
        ys = torch.t(weights) * hs

        # return torch.unsqueeze(torch.sum(ys, 0), 0)
        return torch.unsqueeze(torch.mean(ys, 0), 0)

    def forward(self, inputs):
        fingerprints, adjacency, words = inputs

        """Compound vector with GNN."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        compound_vector = self.gnn(fingerprint_vectors, adjacency, layer_gnn)

        """Protein vector with attention-CNN."""
        word_vectors = self.embed_word(words)
        protein_vector = self.attention_cnn(compound_vector, word_vectors, layer_cnn)

        """Concatenate the above two vectors and output the interaction."""
        cat_vector = torch.cat((compound_vector, protein_vector), 1)
        for j in range(layer_output):
            cat_vector = torch.relu(self.W_out[j](cat_vector))
        interaction = self.W_interaction(cat_vector)

        return interaction


def build_cpi_gnn():
    return CompoundProteinInteractionPrediction()


def example_input_cpi_gnn():
    n_atoms = 8
    n_residues = 20
    fingerprints = torch.randint(0, n_fingerprint, (n_atoms,), dtype=torch.long)
    adjacency = torch.rand(n_atoms, n_atoms)
    words = torch.randint(0, n_word, (n_residues,), dtype=torch.long)
    return ((fingerprints, adjacency, words),)


MENAGERIE_ENTRIES = [
    ("CPI-GNN", "build_cpi_gnn", "example_input_cpi_gnn", 2018, "SOURCE_AVAILABLE"),
]
