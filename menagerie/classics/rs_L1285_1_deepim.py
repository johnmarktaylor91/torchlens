# SOURCE: vendored from tanmoysr/DeepIM @ main
# (code/models.py classes Encoder/Decoder/VAEModel/DiffusionPropagate; code/gat.py classes
# SpGAT/SpGraphAttentionLayer/SpecialSpmm/SpecialSpmmFunction; unchanged)
"""DeepIM: Deep Graph Representation Learning and Optimization for Influence Maximization
(Ling, Jiang, Ju, Fan, Zhao, ICML 2023). Official repo: https://github.com/tanmoysr/DeepIM
(``code/models.py`` + ``code/gat.py`` @ main).

The real DeepIM training pipeline (``code/run_model.py``) pairs two vendored real-code
pieces, unmodified here:
  - a VAE (``Encoder``/``Decoder``/``VAEModel`` from ``models.py``) that reconstructs a
    per-node seed-vector (binary influence-seed indicator over the graph's nodes) through a
    low-dimensional latent bottleneck -- this is DeepIM's "seed vector representation
    learning" half.
  - a Sparse Graph Attention Network (``SpGAT``/``SpGraphAttentionLayer`` from ``gat.py``,
    the sparse-adjacency GAT of Velickovic et al. 2018 as vendored verbatim by the DeepIM
    authors) that propagates the reconstructed seed vector over the graph's normalized
    adjacency to predict final influence spread -- this is the "forward influence
    diffusion" half (``forward_model`` in ``run_model.py``).
``DiffusionPropagate`` (also in ``models.py``) is vendored too; it is the paper's optional
closed-form diffusion-propagation layer, not used on the ``run_model.py`` default path (which
uses ``SpGAT`` as the forward model) but kept for completeness since it is real DeepIM code.
No layer, activation, or forward-pass control-flow was changed from the source files.
"""

from __future__ import annotations

import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# code/models.py (VAE half: Encoder / Decoder / VAEModel)
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, latent_dim)

        self.bn = nn.BatchNorm1d(num_features=latent_dim)

    def forward(self, x):
        h_ = F.relu(self.FC_input(x))
        h_ = F.relu(self.FC_input2(h_))
        h_ = F.relu(self.FC_input2(h_))
        output = self.FC_output(h_)
        return output


class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_input = nn.Linear(input_dim, latent_dim)
        self.FC_hidden_1 = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden_2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = F.relu(self.FC_input(x))
        h = F.relu(self.FC_hidden_1(h))
        h = F.relu(self.FC_hidden_2(h))
        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat


class VAEModel(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(VAEModel, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder

    def reparameterization(self, mean, var):
        std = torch.exp(0.5 * var)
        epsilon = torch.randn_like(var)
        return mean + std * epsilon

    def forward(self, x, adj=None):
        if adj is not None:
            z = self.Encoder(x, adj)
        else:
            z = self.Encoder(x)
        x_hat = self.Decoder(z)
        return x_hat


class DiffusionPropagate(nn.Module):
    def __init__(self, prob_matrix, niter):
        super(DiffusionPropagate, self).__init__()

        self.niter = niter

        if sp.isspmatrix(prob_matrix):
            prob_matrix = prob_matrix.toarray()

        self.register_buffer("prob_matrix", torch.FloatTensor(prob_matrix))

    def forward(self, preds, seed_idx):
        device = preds.device

        for i in range(preds.shape[0]):
            prop_pred = preds[i]
            for j in range(self.niter):
                P2 = self.prob_matrix.T * prop_pred.view((1, -1)).expand(self.prob_matrix.shape)
                P3 = torch.ones(self.prob_matrix.shape).to(device) - P2
                prop_pred = torch.ones((self.prob_matrix.shape[0],)).to(device) - torch.prod(
                    P3, dim=1
                )
                prop_pred = prop_pred.unsqueeze(0)
            if i == 0:
                prop_preds = prop_pred
            else:
                prop_preds = torch.cat((prop_preds, prop_pred), 0)

        return prop_preds


# ---------------------------------------------------------------------------
# code/gat.py (forward diffusion half: SpGAT / SpGraphAttentionLayer)
# ---------------------------------------------------------------------------
class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert not indices.requires_grad
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = "cuda" if input.is_cuda else "cpu"

        N = input.size()[0]
        if adj.layout == torch.sparse_coo:
            edge = adj.indices()
        else:
            edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(
            edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv)
        )
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [
            SpGraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
            for _ in range(nheads)
        ]

        self.attentions1 = [
            SpGraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=True)
            for _ in range(nheads)
        ]
        for i, attention in enumerate(self.attentions):
            self.add_module("attention_{}".format(i), attention)

        for i, attention in enumerate(self.attentions1):
            self.add_module("attention1_{}".format(i), attention)

        self.out_att = SpGraphAttentionLayer(
            nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False
        )

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(torch.cat([att(x, adj) for att in self.attentions], dim=1))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x


# ---------------------------------------------------------------------------
# Menagerie staging harness
# ---------------------------------------------------------------------------
_N_NODES = 12  # tiny toy graph size (real run_model.py defaults use full datasets)
_SEED_DIM = _N_NODES  # inverse_pairs.shape[1] in the source == number of graph nodes


def _tiny_normalized_adj():
    """A tiny symmetric-normalized adjacency matching run_model.py's
    ``utils.normalize_adj(adj + sp.eye(adj.shape[0]))`` pipeline, without depending on the
    repo's data-loading utilities. Kept dense (rather than the source's ``.to_sparse()``)
    because ``SpGraphAttentionLayer.forward`` itself branches on
    ``adj.layout == torch.sparse_coo`` and falls back to ``adj.nonzero().t()`` for a dense
    adjacency -- exactly the source's own supported dense-tensor code path -- and TorchLens
    capture assumes dense strided tensor inputs."""
    torch.manual_seed(0)
    dense = (torch.rand(_N_NODES, _N_NODES) > 0.7).float()
    dense = dense + dense.T + torch.eye(_N_NODES)
    dense = (dense > 0).float()
    deg = dense.sum(dim=1)
    d_inv_sqrt = torch.pow(deg, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    norm_adj = d_inv_sqrt.unsqueeze(1) * dense * d_inv_sqrt.unsqueeze(0)
    return norm_adj


def build_deepim_vae():
    """Distillation VAE half (``vae_model`` in run_model.py), tiny hidden/latent dims."""
    hidden_dim = 8  # default 1024 (netscience) / 4096 (random5)
    latent_dim = 4  # default 512 (netscience) / 1024 (random5)
    encoder = Encoder(input_dim=_SEED_DIM, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = Decoder(
        input_dim=latent_dim, latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=_SEED_DIM
    )
    return VAEModel(Encoder=encoder, Decoder=decoder)


def example_input_deepim_vae():
    torch.manual_seed(0)
    return (torch.rand(1, _SEED_DIM) > 0.7).float()


def build_deepim_forward():
    """Forward diffusion GAT half (``forward_model`` in run_model.py)."""
    return SpGAT(nfeat=1, nhid=4, nclass=1, dropout=0.2, alpha=0.2, nheads=2)


def example_input_deepim_forward():
    return (torch.rand(_N_NODES, 1), _tiny_normalized_adj())


MENAGERIE_ENTRIES = [
    (
        "DeepIM Seed-Vector VAE",
        "build_deepim_vae",
        "example_input_deepim_vae",
        2023,
        "vendored-pytorch",
    ),
    (
        "DeepIM Sparse-GAT Forward Diffusion",
        "build_deepim_forward",
        "example_input_deepim_forward",
        2023,
        "vendored-pytorch",
    ),
]
