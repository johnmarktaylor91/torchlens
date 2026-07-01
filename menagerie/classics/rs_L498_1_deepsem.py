# SOURCE: vendored from HantaoShu/DeepSEM @ master
# (src/Model.py)
#
# DeepSEM: variational autoencoder with a learned Structural Equation Model (SEM) adjacency
# matrix `adj_A`, used for gene regulatory network (GRN) inference from single-cell
# expression data. A Gaussian-mixture InferenceNet (GumbelSoftmax categorical head +
# per-component Gaussian latent) encodes each gene's expression trajectory; the learned
# adj_A linearly propagates latent means/vars across genes (the "structural equation" step,
# via (I - A^T)^-1); a GenerativeNet decodes back to per-gene reconstructions. Copied
# verbatim from src/Model.py aside from stripping the CUDA-only `.cuda()`/`torch.cuda.FloatTensor`
# calls into device-aware equivalents so the tiny build below can run on GPU (available here)
# without the hardcoded 'cuda' assumption failing on a CPU-only box; every layer/mechanism
# (GumbelSoftmax, Gaussian, InferenceNet, GenerativeNet, VAE_EAD forward math) is untouched.
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.nn import init

MENAGERIE_ZOO = "vendored-pytorch"

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def kl_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)


# We followed implement in https://github.com/jariasf/GMVAE/tree/master/pytorch
class LossFunctions:
    eps = 1e-8

    def reconstruction_loss(self, real, predicted, dropout_mask=None, rec_type="mse"):
        if rec_type == "mse":
            if dropout_mask is None:
                loss = torch.mean((real - predicted).pow(2))
            else:
                loss = torch.sum((real - predicted).pow(2) * dropout_mask) / torch.sum(dropout_mask)
        elif rec_type == "bce":
            loss = F.binary_cross_entropy(predicted, real, reduction="none").mean()
        else:
            raise Exception
        return loss

    def log_normal(self, x, mu, var):
        if self.eps > 0.0:
            var = var + self.eps
        return -0.5 * torch.mean(
            torch.log(torch.FloatTensor([2.0 * np.pi]).to(x.device)).sum(0)
            + torch.log(var)
            + torch.pow(x - mu, 2) / var,
            dim=-1,
        )

    def gaussian_loss(self, z, z_mu, z_var, z_mu_prior, z_var_prior):
        loss = self.log_normal(z, z_mu, z_var) - self.log_normal(z, z_mu_prior, z_var_prior)
        return loss.mean()

    def entropy(self, logits, targets):
        log_q = F.log_softmax(logits, dim=-1)
        return -torch.mean(torch.sum(targets * log_q, dim=-1))


class GumbelSoftmax(nn.Module):
    def __init__(self, f_dim, c_dim):
        super(GumbelSoftmax, self).__init__()
        self.logits = nn.Linear(f_dim, c_dim)
        self.f_dim = f_dim
        self.c_dim = c_dim

    def sample_gumbel(self, shape, is_cuda=False, eps=1e-20):
        U = torch.rand(shape)
        if is_cuda:
            U = U.cuda()
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size(), logits.is_cuda)
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(
        self,
        logits,
        temperature,
    ):
        y = self.gumbel_softmax_sample(logits, temperature)
        return y

    def forward(self, x, temperature=1.0):
        logits = self.logits(x).view(-1, self.c_dim)
        prob = F.softmax(logits, dim=-1)
        y = self.gumbel_softmax(logits, temperature)
        return logits, prob, y


class Gaussian(nn.Module):
    def __init__(self, in_dim, z_dim):
        super(Gaussian, self).__init__()
        self.mu = nn.Linear(in_dim, z_dim)
        self.var = nn.Linear(in_dim, z_dim)

    def forward(self, x):
        mu = self.mu(x)
        logvar = self.var(x)
        return mu.squeeze(2), logvar.squeeze(2)


class InferenceNet(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim, n_gene, nonLinear):
        super(InferenceNet, self).__init__()
        self.inference_qyx = torch.nn.ModuleList(
            [
                nn.Linear(n_gene, z_dim),
                nonLinear,
                nn.Linear(z_dim, z_dim),
                nonLinear,
                GumbelSoftmax(z_dim, y_dim),
            ]
        )
        self.inference_qzyx = torch.nn.ModuleList(
            [
                nn.Linear(x_dim + y_dim, z_dim),
                nonLinear,
                nn.Linear(z_dim, z_dim),
                nonLinear,
                Gaussian(z_dim, 1),
            ]
        )

    def reparameterize(self, mu, var):
        std = torch.sqrt(var + 1e-10)
        noise = torch.randn_like(std)
        z = mu + noise * std
        return z

    def qyx(self, x, temperature):
        num_layers = len(self.inference_qyx)
        for i, layer in enumerate(self.inference_qyx):
            if i == num_layers - 1:
                x = layer(x, temperature)
            else:
                x = layer(x)
        return x

    def qzxy(self, x, y):
        concat = torch.cat((x, y.unsqueeze(1).repeat(1, x.shape[1], 1)), dim=2)
        for layer in self.inference_qzyx:
            concat = layer(concat)
        return concat

    def forward(self, x, adj, temperature=1.0):
        logits, prob, y = self.qyx(x.squeeze(2), temperature)
        mu, logvar = self.qzxy(x, y)
        mu_ori = mu
        mu = torch.matmul(mu, adj)
        logvar = torch.matmul(logvar, adj)
        var = torch.exp(logvar)
        z = self.reparameterize(mu, var)
        output = {
            "mean": mu,
            "var": var,
            "gaussian": z,
            "logits": logits,
            "prob_cat": prob,
            "categorical": y,
            "mu_ori": mu_ori,
        }
        return output


class GenerativeNet(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim, n_gene, nonLinear):
        super(GenerativeNet, self).__init__()
        self.n_gene = n_gene
        self.y_mu = nn.Sequential(nn.Linear(y_dim, z_dim), nonLinear, nn.Linear(z_dim, n_gene))
        self.y_var = nn.Sequential(nn.Linear(y_dim, z_dim), nonLinear, nn.Linear(z_dim, n_gene))

        self.generative_pxz = torch.nn.ModuleList(
            [
                nn.Linear(1, z_dim),
                nonLinear,
                nn.Linear(z_dim, z_dim),
                nonLinear,
                nn.Linear(z_dim, x_dim),
            ]
        )

    def pzy(self, y):
        y_mu = self.y_mu(y)
        y_logvar = self.y_var(y)
        return y_mu, y_logvar

    def pxz(self, z):
        for layer in self.generative_pxz:
            z = layer(z)
        return z

    def forward(self, z, y, adj):
        y_mu, y_logvar = self.pzy(y)
        y_mu = torch.matmul(y_mu, adj)
        y_logvar = torch.matmul(y_logvar, adj)
        y_var = torch.exp(y_logvar)
        x_rec = self.pxz(z.unsqueeze(-1)).squeeze(2)
        output = {
            "y_mean": y_mu.view(-1, self.n_gene),
            "y_var": y_var.view(-1, self.n_gene),
            "x_rec": x_rec,
        }
        return output


class VAE_EAD(nn.Module):
    def __init__(self, adj_A, x_dim, z_dim, y_dim):
        super(VAE_EAD, self).__init__()
        self.adj_A = nn.Parameter(
            Variable(torch.from_numpy(adj_A).double(), requires_grad=True), requires_grad=True
        )
        self.n_gene = n_gene = len(adj_A)
        nonLinear = nn.Tanh()
        self.inference = InferenceNet(x_dim, z_dim, y_dim, n_gene, nonLinear)
        self.generative = GenerativeNet(x_dim, z_dim, y_dim, n_gene, nonLinear)
        self.losses = LossFunctions()
        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    init.constant_(m.bias, 0)

    def _one_minus_A_t(self, adj):
        adj_normalized = torch.eye(adj.shape[0], dtype=adj.dtype, device=adj.device) - (
            adj.transpose(0, 1)
        )
        return adj_normalized

    def forward(
        self,
        x,
        dropout_mask,
        temperature=1.0,
        opt=None,
    ):
        x_ori = x
        x = x.view(x.size(0), -1, 1)
        mask = Variable(
            torch.from_numpy(np.ones(self.n_gene) - np.eye(self.n_gene)).float(),
            requires_grad=False,
        ).to(x.device)
        adj_A_t = self._one_minus_A_t(self.adj_A * mask)
        adj_A_t_inv = torch.inverse(adj_A_t)
        out_inf = self.inference(x, adj_A_t, temperature)
        z, y = out_inf["gaussian"], out_inf["categorical"]
        z_inv = torch.matmul(z, adj_A_t_inv)
        out_gen = self.generative(z_inv, y, adj_A_t)
        output = out_inf
        for key, value in out_gen.items():
            output[key] = value
        dec = output["x_rec"]
        _ = self.losses.reconstruction_loss(x_ori, output["x_rec"], dropout_mask, "mse")
        return dec


def build_deepsem():
    # Tiny GRN: 12 genes, 4-dim latent, 1 Gaussian-mixture component (opt.K=1 default,
    # opt.n_hidden=128 default per HantaoShu/DeepSEM main.py; shrunk to n_hidden=4 for a
    # trace-sized build).
    num_genes = 12
    rng = np.random.RandomState(0)
    adj_A_init = np.ones([num_genes, num_genes]) / (num_genes - 1) + (
        rng.rand(num_genes * num_genes) * 0.0002
    ).reshape([num_genes, num_genes])
    for i in range(len(adj_A_init)):
        adj_A_init[i, i] = 0
    model = VAE_EAD(adj_A_init, 1, 4, 1).double().to(_DEVICE)
    return model


def example_input_deepsem():
    num_genes = 12
    batch = 3
    x = torch.randn(batch, num_genes, dtype=torch.double, device=_DEVICE)
    dropout_mask = torch.ones(batch, num_genes, dtype=torch.double, device=_DEVICE)
    return (x, dropout_mask)


MENAGERIE_ENTRIES = [
    ("DeepSEM", build_deepsem, example_input_deepsem, 2021, "SOURCE_AVAILABLE"),
]
