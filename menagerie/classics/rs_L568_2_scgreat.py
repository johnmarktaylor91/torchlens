# SOURCE: vendored from WangyuchenCS/scGREAT @ 73f3f63d7f06762d1e4f5ac3182386cbef86954b
# https://github.com/WangyuchenCS/scGREAT/blob/main/model.py
#
# scGREAT (Wang et al. 2024, "scGREAT: Transformer-based deep-learning for
# gene regulatory network inference from single-cell transcriptomics") is a
# Transformer-encoder gene-pair classifier: two genes' expression profiles
# are projected to an embedding, summed with a pretrained BioBERT gene-name
# embedding and a 2-slot position embedding, run through a
# `nn.TransformerEncoder`, then decoded through an MLP head (with a pooled
# residual tap) to a sigmoid regulatory-link score. The real constructor
# loads pretrained BioBERT gene embeddings from a `.npy` file via
# `np.load(biobert_embedding_path)`; that file is a *data* artifact (not part
# of the architecture), so here it is synthesized as a small random array on
# disk and passed in exactly the way the real code expects it, with zero
# architectural changes to `scGREAT.__init__`/`forward` (copied verbatim from
# `model.py`, only reformatted with normal spacing).

import os
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# --- verbatim from model.py (class body unchanged) ---
class scGREAT(nn.Module):
    def __init__(
        self, expression_data_shape, embed_size, num_layers, num_head, biobert_embedding_path
    ):
        super(scGREAT, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size, nhead=num_head, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.biobert = np.load(biobert_embedding_path)[1:]
        self.biobert_embedding = nn.Embedding.from_pretrained(torch.from_numpy(self.biobert))
        self.position_embedding = nn.Embedding(2, embed_size)

        self.encoder512 = nn.Linear(expression_data_shape[1], 512)
        self.encoder768 = nn.Linear(512, embed_size)

        self.flatten = nn.Flatten()
        self.linear1024 = nn.Linear(1536, 1024)
        self.layernorm1024 = nn.LayerNorm(1024)
        self.batchnorm1024 = nn.BatchNorm1d(1024)

        self.linear512 = nn.Linear(1024, 512)
        self.layernorm512 = nn.LayerNorm(512)
        self.batchnorm512 = nn.BatchNorm1d(512)

        self.linear256 = nn.Linear(512, 256)
        self.layernorm256 = nn.LayerNorm(256)
        self.batchnorm256 = nn.BatchNorm1d(256)

        self.linear2 = nn.Linear(256, 1)
        self.actf = nn.PReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.softmax = nn.Softmax(dim=1)
        self.pool = nn.AvgPool1d(kernel_size=4, stride=4)

    def forward(self, gene_pair_index, expr_embedding):
        bs = expr_embedding.shape[0]
        position = torch.Tensor([0, 1] * bs).reshape(bs, -1).to(torch.int32)
        position = position.to(self.device)
        p_e = self.position_embedding(position)
        expr_embedding = expr_embedding.to(self.device)
        gene_pair_index = gene_pair_index.to(self.device)

        out_expr_e = self.encoder512(expr_embedding)
        out_expr_e = F.leaky_relu(self.encoder768(out_expr_e))
        b_e = self.biobert_embedding(gene_pair_index)
        input_ = torch.add(out_expr_e, torch.add(b_e, p_e))
        out = self.transformer_encoder(input_)
        out = self.flatten(out)

        out = self.linear1024(out)
        out = self.dropout(out)
        out = self.actf(out)

        r = out.unsqueeze(1)
        r = self.pool(r)
        r = r.squeeze(1)

        out = self.linear512(out)
        out = self.dropout(out)
        out = self.actf(out)

        out = self.linear256(out) + r
        out = self.dropout(out)
        out = self.actf(out)

        outs = self.linear2(out)
        outs = nn.Sigmoid()(outs)

        return outs


# --- menagerie staging glue ---
_N_GENES = 64  # stand-in vocabulary size for the BioBERT gene-embedding table
_EMBED_SIZE = 768  # scGREAT's real embed_size (BioBERT hidden dim)


def _synth_biobert_embedding_path():
    """Write a small random (_N_GENES+1, _EMBED_SIZE) array to a temp .npy
    file, standing in for the real pretrained BioBERT gene embeddings the
    real repo loads from disk (row 0 is dropped by `[1:]` in __init__, matching
    the real code's convention of a reserved/padding row)."""
    arr = np.random.randn(_N_GENES + 1, _EMBED_SIZE).astype(np.float32)
    fd, path = tempfile.mkstemp(suffix=".npy")
    os.close(fd)
    np.save(path, arr)
    return path


def build_scgreat():
    expr_dim = 32  # gene-pair expression profile length (real repo uses ~binned expr bins)
    m = scGREAT(
        expression_data_shape=(2, expr_dim),
        embed_size=_EMBED_SIZE,
        num_layers=2,
        num_head=8,
        biobert_embedding_path=_synth_biobert_embedding_path(),
    )
    # matches the real repo's main.py: `T = T.to(device)` right after
    # construction, since scGREAT.__init__ only sets self.device (used to
    # place *inputs*) but never moves its own parameters.
    m = m.to(m.device)
    m.eval()
    return m


def example_input_scgreat():
    bs = 4
    gene_pair_index = torch.randint(0, _N_GENES, (bs, 2), dtype=torch.long)
    expr_embedding = torch.randn(bs, 2, 32)
    return (gene_pair_index, expr_embedding)


MENAGERIE_ENTRIES = [
    ("scGREAT", "build_scgreat", "example_input_scgreat", 2024, "vendored-pytorch"),
]
