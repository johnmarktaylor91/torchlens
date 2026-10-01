# SOURCE: vendored from samsledje/ConPLex @ ccd8c804a76dff41649e58f692293b55c8907129 (conplex_dti/model/architectures.py)
# https://github.com/samsledje/ConPLex/blob/ccd8c804a76dff41649e58f692293b55c8907129/conplex_dti/model/architectures.py
#
# ConPLex: contrastive coembedding model for drug-target interaction (DTI)
# prediction. `SimpleCoembedding` (the model class used as the paper's
# flagship architecture, aliased in the real repo as
# `SimpleCoembeddingNoSigmoid`) projects a precomputed drug (Morgan
# fingerprint) embedding and a precomputed protein language-model (ESM)
# embedding into a shared latent space via two independent linear+activation
# projector heads, then scores drug-target pairs by a configurable latent
# distance metric (Cosine/SquaredCosine/Euclidean/SquaredEuclidean).
#
# Only import fixes were made (dropped the unused `omegaconf`/`tqdm`/
# `torch.utils.data`/relative-package imports that the real file pulls in
# for training-script plumbing but that `SimpleCoembedding` itself never
# uses). The architecture (`LogisticActivation`, the `DISTANCE_METRICS`
# registry, and `SimpleCoembedding`) is transcribed unmodified from the
# real repo.

import torch
import torch.nn as nn

#################################
# Latent Space Distance Metrics #
#################################


class Cosine(nn.Module):
    def forward(self, x1, x2):
        return nn.CosineSimilarity()(x1, x2)


class SquaredCosine(nn.Module):
    def forward(self, x1, x2):
        return nn.CosineSimilarity()(x1, x2) ** 2


class Euclidean(nn.Module):
    def forward(self, x1, x2):
        return torch.cdist(x1, x2, p=2.0)


class SquaredEuclidean(nn.Module):
    def forward(self, x1, x2):
        return torch.cdist(x1, x2, p=2.0) ** 2


DISTANCE_METRICS = {
    "Cosine": Cosine,
    "SquaredCosine": SquaredCosine,
    "Euclidean": Euclidean,
    "SquaredEuclidean": SquaredEuclidean,
}

ACTIVATIONS = {"ReLU": nn.ReLU, "GELU": nn.GELU, "ELU": nn.ELU, "Sigmoid": nn.Sigmoid}

#######################
# Model Architectures #
#######################


class LogisticActivation(nn.Module):
    """
    Implementation of Generalized Sigmoid
    Applies the element-wise function:
    :math:`\\sigma(x) = \\frac{1}{1 + \\exp(-k(x-x_0))}`
    :param x0: The value of the sigmoid midpoint
    :type x0: float
    :param k: The slope of the sigmoid - trainable -  :math:`k \\geq 0`
    :type k: float
    :param train: Whether :math:`k` is a trainable parameter
    :type train: bool
    """

    def __init__(self, x0=0, k=1, train=False):
        super(LogisticActivation, self).__init__()
        self.x0 = x0
        self.k = nn.Parameter(torch.FloatTensor([float(k)]), requires_grad=False)
        self.k.requiresGrad = train

    def forward(self, x):
        o = torch.clamp(1 / (1 + torch.exp(-self.k * (x - self.x0))), min=0, max=1).squeeze()
        return o

    def clip(self):
        """
        Restricts sigmoid slope :math:`k` to be greater than or equal to 0, if :math:`k` is trained.
        :meta private:
        """
        self.k.data.clamp_(min=0)


class SimpleCoembedding(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        latent_dimension=1024,
        latent_activation="ReLU",
        latent_distance="Cosine",
        classify=True,
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.do_classify = classify
        self.latent_activation = ACTIVATIONS[latent_activation]

        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, latent_dimension), self.latent_activation()
        )
        nn.init.xavier_normal_(self.drug_projector[0].weight)

        self.target_projector = nn.Sequential(
            nn.Linear(self.target_shape, latent_dimension), self.latent_activation()
        )
        nn.init.xavier_normal_(self.target_projector[0].weight)

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target):
        if self.do_classify:
            return self.classify(drug, target)
        else:
            return self.regress(drug, target)

    def regress(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        inner_prod = torch.bmm(
            drug_projection.view(-1, 1, self.latent_dimension),
            target_projection.view(-1, self.latent_dimension, 1),
        ).squeeze()
        return inner_prod.squeeze()

    def classify(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        distance = self.activator(drug_projection, target_projection)
        return distance.squeeze()


SimpleCoembeddingNoSigmoid = SimpleCoembedding


def build_conplex():
    torch.manual_seed(0)
    return SimpleCoembedding(drug_shape=32, target_shape=24, latent_dimension=16)


def example_input_conplex():
    torch.manual_seed(0)
    drug = torch.randn(4, 32)
    target = torch.randn(4, 24)
    return (drug, target)


MENAGERIE_ZOO = "vendored-pytorch"

MENAGERIE_ENTRIES = [
    ("ConPLex", build_conplex, example_input_conplex, 2023, "vendored-pytorch"),
]
