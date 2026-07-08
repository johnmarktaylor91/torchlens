# SOURCE: vendored from ziyewang/COMEBin @ master (COMEBin/models/mlp.py, COMEBin/models/mlp2.py)
"""COMEBin: contrastive multi-view encoder for metagenomic contig binning
(Nat Commun 2024). The real architecture is a k-mer branch ``EmbeddingNet``
(``models/mlp.py``) feeding into a fusion ``EmbeddingNet`` (``models/mlp2.py``)
that concatenates the (optionally pretrained/frozen) k-mer embedding with a
coverage-branch ``EmbeddingNet`` embedding before its own MLP tower -- exactly
as constructed in the official ``train_CLmodel.py`` (``EmbeddingNet2(... ,
cov_model=cov_model, pretrained_model=kmerMetric_model)``).

Code below is copied verbatim from the official repo's two ``EmbeddingNet``
classes (only unused imports dropped). Architecture logic is untouched.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


class EmbeddingNetKmer(nn.Module):
    """models/mlp.py::EmbeddingNet -- k-mer / coverage branch encoder."""

    # Useful code from fast.ai tabular model
    # https://github.com/fastai/fastai/blob/3b7c453cfa3845c6ffc496dd4043c07f3919270e/fastai/tabular/models.py#L6
    def __init__(self, in_sz, out_sz, emb_szs, ps, use_bn=True, actn=nn.ReLU()):
        super(EmbeddingNetKmer, self).__init__()
        self.in_sz = in_sz
        self.out_sz = out_sz
        self.n_embs = len(emb_szs) - 1
        if ps == 0:
            ps = np.zeros(self.n_embs)
        # input layer
        layers = [nn.Linear(self.in_sz, emb_szs[0]), actn]
        # hidden layers
        for i in range(self.n_embs):
            layers += self.bn_drop_lin(
                n_in=emb_szs[i], n_out=emb_szs[i + 1], bn=use_bn, p=ps[i], actn=actn
            )
        # output layer
        layers.append(nn.Linear(emb_szs[-1], self.out_sz))
        self.fc = nn.Sequential(*layers)

    def bn_drop_lin(
        self,
        n_in: int,
        n_out: int,
        bn: bool = True,
        p: float = 0.0,
        actn: nn.Module = None,
    ):
        # https://github.com/fastai/fastai/blob/3b7c453cfa3845c6ffc496dd4043c07f3919270e/fastai/layers.py#L44
        "Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
        layers = [nn.BatchNorm1d(n_in)] if bn else []
        if p != 0:
            layers.append(nn.Dropout(p))
        layers.append(nn.Linear(n_in, n_out))
        if actn is not None:
            layers.append(actn)
        return layers

    def forward(self, x):
        output = self.fc(x)
        return output


class EmbeddingNetFusion(nn.Module):
    """models/mlp2.py::EmbeddingNet -- fusion encoder combining an optional
    pretrained k-mer branch with a coverage branch."""

    # Useful code from fast.ai tabular model
    # https://github.com/fastai/fastai/blob/3b7c453cfa3845c6ffc496dd4043c07f3919270e/fastai/tabular/models.py#L6
    def __init__(
        self,
        in_sz,
        out_sz,
        emb_szs,
        ps,
        use_bn=True,
        actn=nn.ReLU(),
        pretrained_model=None,
        cov_model=None,
        covmodel_notl2normalize=False,
    ):
        super(EmbeddingNetFusion, self).__init__()
        self.pretrained_model = pretrained_model
        self.cov_model = cov_model
        self.in_sz = in_sz
        self.out_sz = out_sz
        self.n_embs = len(emb_szs) - 1
        self.covmodel_notl2normalize = covmodel_notl2normalize
        if ps == 0:
            ps = np.zeros(self.n_embs)
        # input layer
        layers = [nn.Linear(self.in_sz, emb_szs[0]), actn]
        # hidden layers
        for i in range(self.n_embs):
            layers += self.bn_drop_lin(
                n_in=emb_szs[i], n_out=emb_szs[i + 1], bn=use_bn, p=ps[i], actn=actn
            )
        # output layer
        layers.append(nn.Linear(emb_szs[-1], self.out_sz))
        self.fc = nn.Sequential(*layers)
        project_layer = [actn, nn.Linear(self.out_sz, self.out_sz)]
        self.fc2 = nn.Sequential(*project_layer)

    def bn_drop_lin(
        self,
        n_in: int,
        n_out: int,
        bn: bool = True,
        p: float = 0.0,
        actn: nn.Module = None,
    ):
        "Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
        layers = [nn.BatchNorm1d(n_in)] if bn else []
        if p != 0:
            layers.append(nn.Dropout(p))
        layers.append(nn.Linear(n_in, n_out))
        if actn is not None:
            layers.append(actn)
        return layers

    def forward(self, x, x2=None):
        if self.pretrained_model is not None and self.cov_model is None:
            kmeremb = self.pretrained_model(x)
            x = torch.cat([F.normalize(self.pretrained_model(x)), x2], dim=-1)
        if self.pretrained_model is not None and self.cov_model is not None:
            kmeremb = self.pretrained_model(x)
            if self.covmodel_notl2normalize:
                x = torch.cat([F.normalize(self.pretrained_model(x)), self.cov_model(x2)], dim=-1)
            else:
                x = torch.cat(
                    [F.normalize(self.pretrained_model(x)), F.normalize(self.cov_model(x2))], dim=-1
                )

        if self.pretrained_model is None and self.cov_model is not None:
            if self.covmodel_notl2normalize:
                x = torch.cat([x, self.cov_model(x2)], dim=-1)
            else:
                x = torch.cat([x, F.normalize(self.cov_model(x2))], dim=-1)

        output = self.fc(x)

        if self.cov_model is not None and self.pretrained_model is not None:
            return output, self.cov_model(x2), kmeremb
        elif self.cov_model is not None and self.pretrained_model is None:
            return output, self.cov_model(x2)
        else:
            return output


# --- staging harness -------------------------------------------------------
# Mirrors train_CLmodel.py's default (add_model_for_coverage=True,
# pretrain_kmer_model_path == 'no') construction path: a coverage-branch
# EmbeddingNetKmer feeding EmbeddingNetFusion via cov_model=, with no
# pretrained kmer branch (pretrained_model=None) -- the real code path
# actually exercised by COMEBin's default training pipeline.


def build_comebin():
    cov_dim = 20
    out_dim_forcov = 16
    kmer_dim = 32  # x already-embedded k-mer-derived features (raw x branch)
    input_size = out_dim_forcov + kmer_dim

    cov_model = EmbeddingNetKmer(
        in_sz=cov_dim,
        out_sz=out_dim_forcov,
        emb_szs=[32, 32],
        ps=[0.1],
        use_bn=True,
        actn=nn.LeakyReLU(),
    )

    model = EmbeddingNetFusion(
        in_sz=input_size,
        out_sz=24,
        emb_szs=[32, 32],
        ps=[0.1],
        use_bn=True,
        actn=nn.LeakyReLU(),
        cov_model=cov_model,
        covmodel_notl2normalize=False,
    )
    return model


def example_input_comebin():
    batch = 4
    x = torch.randn(batch, 32)
    x2 = torch.randn(batch, 20)
    return (x, x2)


MENAGERIE_ENTRIES = [
    ("COMEBin", "build_comebin", "example_input_comebin", 2024, "vendored"),
]
