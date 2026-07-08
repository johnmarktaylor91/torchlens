# SOURCE: vendored from hui2000ji/scETM @ 0a34c345d70b262ebc38e033bae683fa4929ed3e
#   src/scETM/models/scETM.py
#   src/scETM/models/BaseCellModel.py
#   src/scETM/models/model_utils.py
#
# scETM ("Single-cell Embedded Topic Model", Zhao et al. 2021) -- a VAE-style topic
# model for scRNA-seq: an encoder MLP predicts a Normal posterior over per-cell topic
# proportions (delta -> softmax -> theta), decoded against a (fixed + trainable) gene
# embedding matrix rho and a learned topic embedding alpha via beta = alpha @ rho.
# Real classes, minimal import fix only: `BaseCellModel` and `scETM.forward`/`decode`
# have zero functional dependency on `anndata`; the upstream files import `anndata`
# only for docstring type hints and for two training/eval convenience methods
# (`_apply_to`, `get_cell_embeddings_and_nll`) that route through `CellSampler`
# (itself anndata-dependent). Those two methods and the `anndata`/`CellSampler`
# imports are dropped here (anndata is not an installed base lib); every layer,
# parameter, and the full `__init__`/`forward`/`decode` architecture is unmodified.
from typing import Any, Mapping, Sequence, Union
import math
import torch
from torch import nn
from torch.distributions import Normal, Independent
import torch.nn.functional as F


# ---- src/scETM/models/model_utils.py (verbatim) ----------------------------------------
class InputPartlyTrainableLinear(nn.Module):
    """A linear layer with partially trainable input weights.

    The weights are divided into two parts, one of shape [I_trainable, O] is
    trainable, the other of shape [I_fixed, O] is fixed.
    If bias = True, the trainable part would have a trainbale bias, and if
    n_trainable_input is 0, a trainable bias is added to the layer.

    In the forward pass, the input x of shape [B, I] is split into x_fixed of
    shape [B, I_fixed] and x_trainable of shape [B, I_trainable]. The two parts
    are separately affinely transformed and results are summed.

    B: batch size; I: input dim; O: output dim.
    """

    def __init__(
        self, n_fixed_input: int, n_output: int, n_trainable_input: int = 0, bias: bool = True
    ) -> None:
        super().__init__()
        self.fixed: nn.Linear = nn.Linear(n_fixed_input, n_output, bias=False)
        self.fixed.requires_grad_(False)
        self.trainable_bias: Union[None, nn.Parameter] = None
        if n_trainable_input > 0:
            self.trainable: nn.Linear = nn.Linear(n_trainable_input, n_output, bias=bias)
        elif bias:
            self.trainable_bias = nn.Parameter(torch.Tensor(n_output))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fixed.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.trainable_bias, -bound, bound)
        self.n_fixed_input: int = n_fixed_input
        self.n_trainable_input: int = n_trainable_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_trainable_input > 0:
            x_fixed, x_trainable = x[:, : self.n_fixed_input], x[:, self.n_fixed_input :]
            with torch.no_grad():
                out = self.fixed(x_fixed)
            return out + self.trainable(x_trainable)
        elif self.trainable_bias is not None:
            return self.fixed(x) + self.trainable_bias
        else:
            return self.fixed(x)

    @property
    def weight(self) -> torch.Tensor:
        if self.n_trainable_input > 0:
            return torch.cat([self.fixed.weight, self.trainable.weight], dim=1)
        else:
            return self.fixed.weight

    @property
    def bias(self) -> Union[torch.Tensor, None]:
        if self.n_trainable_input > 0:
            return self.trainable.bias
        else:
            return self.trainable_bias


class PartlyTrainableParameter2D(nn.Module):
    """A partly trainable 2D parameter.

    The [H, W] parameter is split to two parts, the fixed [H, W_fixed] and the
    trainbale [H, W_trainable].
    """

    def __init__(self, height: int, n_fixed_width: int, n_trainable_width: int) -> None:
        super().__init__()
        self.height: int = height
        self.n_fixed_width: int = n_fixed_width
        self.n_trainable_width: int = n_trainable_width
        self.fixed: Union[None, torch.Tensor] = None
        self.trainable: Union[None, nn.Parameter] = None
        if n_fixed_width > 0:
            self.fixed = torch.randn(height, n_fixed_width)
        if n_trainable_width > 0:
            self.trainable = nn.Parameter(torch.randn(height, n_trainable_width))

    def get_param(self) -> Union[None, torch.Tensor]:
        params = [param for param in (self.fixed, self.trainable) if param is not None]
        if len(params) == 2:
            return torch.cat(params, dim=1)
        elif len(params) == 1:
            return params[0]
        else:
            return None

    def __repr__(self):
        return f"{self.__class__.__name__}(height={self.height}, fixed={self.n_fixed_width}, trainable={self.n_trainable_width})"


def get_fully_connected_layers(
    n_trainable_input: int,
    hidden_sizes: Union[int, Sequence[int]],
    n_trainable_output: Union[None, int] = None,
    bn: bool = True,
    bn_track_running_stats: bool = True,
    dropout_prob: float = 0.0,
    n_fixed_input: int = 0,
    n_fixed_output: int = 0,
) -> nn.Sequential:
    if isinstance(hidden_sizes, int):
        hidden_sizes = [hidden_sizes]
    layers = []
    for i, size in enumerate(hidden_sizes):
        if i == 0 and n_fixed_input > 0:
            layers.append(InputPartlyTrainableLinear(n_fixed_input, size, n_trainable_input))
        else:
            layers.append(nn.Linear(n_trainable_input, size))
        layers.append(nn.ReLU())
        if bn:
            layers.append(nn.BatchNorm1d(size, track_running_stats=bn_track_running_stats))
        if dropout_prob:
            layers.append(nn.Dropout(dropout_prob))
        n_trainable_input = size
    if n_trainable_output is not None:
        layers.append(nn.Linear(n_trainable_input, n_trainable_output))
    return nn.Sequential(*layers)


def get_kl(mu: torch.Tensor, logsigma: torch.Tensor):
    """Calculate KL(q||p) where q = Normal(mu, sigma) and p = Normal(0, I)."""
    logsigma = 2 * logsigma
    return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(-1)


# ---- src/scETM/models/BaseCellModel.py (anndata-free subset) ---------------------------
class BaseCellModel(nn.Module):
    """Base class for single cell models."""

    clustering_input: str
    emb_names: Sequence[str]

    def __init__(
        self,
        n_trainable_genes: int,
        n_batches: int,
        n_fixed_genes: int = 0,
        need_batch: bool = False,
        device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ) -> None:
        super().__init__()
        self.device: torch.device = device
        self.n_trainable_genes: int = n_trainable_genes
        self.n_fixed_genes: int = n_fixed_genes
        self.n_batches: int = n_batches
        self.need_batch: bool = need_batch

    @property
    def n_genes(self):
        return self.n_trainable_genes + self.n_fixed_genes

    def _get_batch_indices_oh(self, data_dict: Mapping[str, torch.Tensor]):
        if "batch_indices_oh" in data_dict:
            w_batch_id = data_dict["batch_indices_oh"]
        else:
            batch_indices = data_dict["batch_indices"].unsqueeze(1)
            w_batch_id = torch.zeros(
                (batch_indices.shape[0], self.n_batches), dtype=torch.float32, device=self.device
            )
            w_batch_id.scatter_(1, batch_indices, 1.0)
            w_batch_id = w_batch_id[:, : self.n_batches - 1]
            data_dict["batch_indices_oh"] = w_batch_id
        return w_batch_id


# ---- src/scETM/models/scETM.py (verbatim architecture) ----------------------------------
class scETM(BaseCellModel):
    """Single-cell Embedded Topic Model.

    From paper "Learning interpretable cellular and gene signature
    embeddings from single-cell transcriptomic data".
    Link: https://www.biorxiv.org/content/10.1101/2021.01.13.426593v1.full
    """

    clustering_input: str = "delta"
    emb_names: Sequence[str] = ["delta", "theta"]
    max_logsigma = 10
    min_logsigma = -10

    def __init__(
        self,
        n_trainable_genes: int,
        n_batches: int,
        n_fixed_genes: int = 0,
        n_topics: int = 50,
        trainable_gene_emb_dim: int = 400,
        hidden_sizes: Sequence[int] = (128,),
        bn: bool = True,
        dropout_prob: float = 0.1,
        normalize_beta: bool = False,
        normed_loss: bool = True,
        norm_cells: bool = True,
        input_batch_id: bool = False,
        enable_batch_bias: bool = True,
        enable_global_bias: bool = False,
        rho_fixed_emb=None,
        rho_fixed_gene=None,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__(
            n_trainable_genes,
            n_batches,
            n_fixed_genes,
            need_batch=n_batches > 1 and (input_batch_id or enable_batch_bias),
            device=device,
        )

        self.n_topics: int = n_topics
        self.trainable_gene_emb_dim: int = trainable_gene_emb_dim
        self.hidden_sizes: Sequence[int] = hidden_sizes
        self.bn: bool = bn
        self.dropout_prob: float = dropout_prob
        self.normalize_beta: bool = normalize_beta
        self.normed_loss: bool = normed_loss
        self.norm_cells: bool = norm_cells
        self.input_batch_id: bool = input_batch_id
        self.enable_batch_bias: bool = enable_batch_bias
        self.enable_global_bias: bool = enable_global_bias
        if self.n_batches <= 1:
            self.enable_batch_bias = False
            self.input_batch_id = False

        self.q_delta: nn.Sequential = get_fully_connected_layers(
            n_trainable_input=self.n_trainable_genes
            + ((self.n_batches - 1) if self.input_batch_id else 0),
            hidden_sizes=self.hidden_sizes,
            bn=self.bn,
            dropout_prob=self.dropout_prob,
            n_fixed_input=self.n_fixed_genes,
        )
        hidden_dim = self.hidden_sizes[-1]
        self.mu_q_delta: nn.Linear = nn.Linear(hidden_dim, self.n_topics, bias=True)
        self.logsigma_q_delta: nn.Linear = nn.Linear(hidden_dim, self.n_topics, bias=True)

        self.rho_fixed_emb: Union[None, torch.Tensor] = None
        self.rho_trainable_emb: Union[None, PartlyTrainableParameter2D] = None
        self._init_rho_trainable_emb()
        if (
            self.trainable_gene_emb_dim > 0
            and self.n_fixed_genes > 0
            and rho_fixed_gene is not None
        ):
            assert rho_fixed_gene.shape == (self.trainable_gene_emb_dim, self.n_fixed_genes)
            self.rho_trainable_emb.fixed = torch.FloatTensor(rho_fixed_gene)
        if rho_fixed_emb is not None:
            assert rho_fixed_emb.shape[1] == self.n_fixed_genes + self.n_trainable_genes
            self.rho_fixed_emb = torch.FloatTensor(rho_fixed_emb).to(device)

        self.alpha: nn.Parameter = nn.Parameter(
            torch.randn(
                self.n_topics,
                self.trainable_gene_emb_dim
                + (self.rho_fixed_emb.shape[0] if self.rho_fixed_emb is not None else 0),
            )
        )
        self._init_batch_and_global_biases()

        self.to(device)

    @property
    def rho(self) -> torch.Tensor:
        rho = [
            param
            for param in (self.rho_fixed_emb, self.rho_trainable_emb.get_param())
            if param is not None
        ]
        rho = torch.cat(rho, dim=0) if len(rho) > 1 else rho[0]
        return rho

    def _init_encoder_first_layer(self) -> None:
        trainable_dim = self.n_trainable_genes + (
            (self.n_batches - 1) if self.input_batch_id else 0
        )
        if self.n_fixed_genes > 0:
            self.q_delta[0] = InputPartlyTrainableLinear(
                self.n_fixed_genes, self.hidden_sizes[0], trainable_dim
            )
        else:
            self.q_delta[0] = nn.Linear(trainable_dim, self.hidden_sizes[0])

    def _init_rho_trainable_emb(self) -> None:
        if self.trainable_gene_emb_dim > 0:
            self.rho_trainable_emb = PartlyTrainableParameter2D(
                self.trainable_gene_emb_dim, self.n_fixed_genes, self.n_trainable_genes
            )

    def _init_batch_and_global_biases(self) -> None:
        if self.enable_batch_bias:
            self.batch_bias: nn.Parameter = nn.Parameter(
                torch.randn(self.n_batches, self.n_fixed_genes + self.n_trainable_genes)
            )

        self.global_bias: nn.Parameter = (
            nn.Parameter(torch.randn(1, self.n_fixed_genes + self.n_trainable_genes))
            if self.enable_global_bias
            else None
        )

    def decode(self, theta: torch.Tensor, batch_indices: Union[None, torch.Tensor]) -> torch.Tensor:
        beta = self.alpha @ self.rho

        if self.normalize_beta:
            recon = torch.mm(theta, F.softmax(beta, dim=-1))
            recon_log = (recon + 1e-30).log()
        else:
            recon_logit = torch.mm(theta, beta)  # [batch_size, n_genes]
            if self.enable_global_bias:
                recon_logit += self.global_bias
            if self.enable_batch_bias:
                recon_logit += self.batch_bias[batch_indices]
            recon_log = F.log_softmax(recon_logit, dim=-1)
        return recon_log

    def forward(
        self, data_dict: Mapping[str, torch.Tensor], hyper_param_dict: Mapping[str, Any] = dict()
    ) -> Mapping[str, Any]:
        """scETM forward computation.

        The cells are encoded into topic embeddings (delta), which is further
        normalized to the topic proportions (theta). Next, theta is decoded
        to form the reconstructions.
        """

        cells, library_size = data_dict["cells"], data_dict["library_size"]
        normed_cells = cells / library_size
        input_cells = normed_cells if self.norm_cells else cells
        if self.input_batch_id:
            input_cells = torch.cat((input_cells, self._get_batch_indices_oh(data_dict)), dim=1)

        q_delta = self.q_delta(input_cells)
        mu_q_delta = self.mu_q_delta(q_delta)
        logsigma_q_delta = self.logsigma_q_delta(q_delta).clamp(
            self.min_logsigma, self.max_logsigma
        )
        q_delta = Independent(Normal(loc=mu_q_delta, scale=logsigma_q_delta.exp()), 1)

        delta = q_delta.rsample()
        theta = F.softmax(delta, dim=-1)  # [batch_size, n_topics]

        if not self.training:
            theta = F.softmax(mu_q_delta, dim=-1)
            fwd_dict = dict(theta=theta, delta=mu_q_delta)
            if "decode" in hyper_param_dict and hyper_param_dict["decode"]:
                recon_log = self.decode(theta, data_dict.get("batch_indices", None))
                fwd_dict["recon_log"] = recon_log
                fwd_dict["nll"] = (-recon_log * (normed_cells if self.normed_loss else cells)).sum()
            return fwd_dict

        recon_log = self.decode(theta, data_dict.get("batch_indices", None))

        nll = (-recon_log * normed_cells if self.normed_loss else cells).sum(-1).mean()
        kl_delta = get_kl(mu_q_delta, logsigma_q_delta).mean()
        loss = nll + hyper_param_dict["kl_weight"] * kl_delta
        record = dict(loss=loss, nll=nll, kl_delta=kl_delta)

        record = {k: v.detach().item() for k, v in record.items()}

        fwd_dict = dict(theta=theta, delta=delta, recon_log=recon_log)

        return loss, fwd_dict, record


def build_scetm():
    n_genes = 60
    n_batches = 3
    return scETM(
        n_trainable_genes=n_genes,
        n_batches=n_batches,
        n_topics=8,
        trainable_gene_emb_dim=16,
        hidden_sizes=(24,),
        bn=True,
        dropout_prob=0.1,
        enable_batch_bias=True,
        enable_global_bias=True,
        device=torch.device("cpu"),
    )


def example_input_scetm():
    batch_size = 10
    n_genes = 60
    n_batches = 3
    cells = torch.rand(batch_size, n_genes) * 5
    library_size = cells.sum(dim=1, keepdim=True).clamp_min(1.0)
    batch_indices = torch.randint(0, n_batches, (batch_size,))
    data_dict = {
        "cells": cells,
        "library_size": library_size,
        "batch_indices": batch_indices,
    }
    hyper_param_dict = {"kl_weight": 1.0}
    return (data_dict, hyper_param_dict)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("scETM", "build_scetm", "example_input_scetm", 2021, MENAGERIE_ZOO),
]
