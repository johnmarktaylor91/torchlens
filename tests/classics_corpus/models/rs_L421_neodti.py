# SOURCE: vendored from Dr-Patient/pytorch_NeoDTI @ main
# https://raw.githubusercontent.com/Dr-Patient/pytorch_NeoDTI/main/src/model.py
# https://raw.githubusercontent.com/Dr-Patient/pytorch_NeoDTI/main/src/pytorch_NeoDTI_cv.py
#
# Wan, Zeng, Liao, Zhang 2019 (Bioinformatics) "NeoDTI: Neural integration of neighbor
# information from a heterogeneous network for discovering new drug-target
# interactions" -- a heterogeneous-network neural matrix-completion model. Per-node-type
# embeddings (drug/protein/disease/sideeffect) are propagated one hop across every
# typed edge relation (drug-drug, drug-chemical, drug-disease, drug-sideeffect,
# drug-protein, protein-protein, protein-sequence, protein-disease, disease-drug,
# disease-protein, sideeffect-drug) via a shared relation-specific linear transform
# (`*_wr`) + row-normalized adjacency matmul, concatenated with the node's own base
# embedding and passed through a shared projection `W1` + ReLU + L2-normalize to
# produce the propagated node vectors. Each typed edge is then reconstructed from a
# low-rank bilinear form of the propagated vectors (`G`/`H` projection matrices) and
# the model is trained to minimize the sum of per-relation reconstruction losses (plus
# L2 weight regularization) -- the "neighbor information integration" the paper's name
# refers to. The queue's linked official `FangpingWan/NeoDTI` repo is TensorFlow 1.x
# (unmaintained/non-installable); this actively-maintained community PyTorch port
# reproduces the exact same architecture (same tensor algebra: relation-wise
# `matmul(adj_normalize, wr(embedding))` sums, same `W1` projection, same bilinear
# `G`/`H` reconstruction, same loss composition) and is the real, runnable source used
# here per rung-2 (real repo code, base-lib-only deps).
#
# `initialize_weights` and `NeoDTI` (the base model, without the drug-protein-affinity
# extension `NeoDTI_with_aff`, which is a distinct variant not requested here) are
# copied verbatim from the real `src/model.py`. No architectural changes were made.
# Real training-config hyperparameters (`src/pytorch_NeoDTI_cv.py` argparse defaults):
# `d=1024` (node embedding dim), `k=512` (bilinear reprojection dim), `l2_factor=0.1`.
# Real entity counts (`num_drug`/`num_protein`/`num_disease`/`num_sideeffect`) come
# from the paper's specific heterogeneous-network dataset (not shipped standalone);
# small placeholder counts are used here to keep the trace tiny while exercising every
# relation branch and every reconstruction head.

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import MSELoss


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")
        nn.init.constant_(m.bias.data, 0.1)


class NeoDTI(nn.Module):
    def __init__(self, args, num_drug, num_disease, num_protein, num_sideeffect):
        super(NeoDTI, self).__init__()
        self.args = args
        self.criterion = MSELoss(reduction="sum")
        # f0(v)
        self.drug_embedding = nn.Parameter(torch.Tensor(num_drug, args.d))
        self.protein_embedding = nn.Parameter(torch.Tensor(num_protein, args.d))
        self.disease_embedding = nn.Parameter(torch.Tensor(num_disease, args.d))
        self.sideeffect_embedding = nn.Parameter(torch.Tensor(num_sideeffect, args.d))

        # Wr
        self.drug_drug_wr = nn.Linear(args.d, args.d)
        self.drug_chemical_wr = nn.Linear(args.d, args.d)
        self.drug_disease_wr = nn.Linear(args.d, args.d)
        self.drug_sideeffect_wr = nn.Linear(args.d, args.d)
        self.drug_protein_wr = nn.Linear(args.d, args.d)

        self.protein_protein_wr = nn.Linear(args.d, args.d)
        self.protein_sequence_wr = nn.Linear(args.d, args.d)
        self.protein_disease_wr = nn.Linear(args.d, args.d)
        self.protein_drug_wr = nn.Linear(args.d, args.d)

        self.disease_drug_wr = nn.Linear(args.d, args.d)
        self.disease_protein_wr = nn.Linear(args.d, args.d)

        self.sideeffect_drug_wr = nn.Linear(args.d, args.d)

        # W1
        self.W1 = nn.Linear(2 * args.d, args.d)
        # projection G and H
        self.drug_disease_G = nn.Parameter(torch.Tensor(args.d, args.k))
        self.drug_disease_H = nn.Parameter(torch.Tensor(args.d, args.k))

        self.drug_drug_G = nn.Parameter(torch.Tensor(args.d, args.k))

        self.drug_chemical_G = nn.Parameter(torch.Tensor(args.d, args.k))

        self.drug_sideeffect_G = nn.Parameter(torch.Tensor(args.d, args.k))
        self.drug_sideeffect_H = nn.Parameter(torch.Tensor(args.d, args.k))

        self.drug_protein_G = nn.Parameter(torch.Tensor(args.d, args.k))
        self.drug_protein_H = nn.Parameter(torch.Tensor(args.d, args.k))

        self.protein_disease_G = nn.Parameter(torch.Tensor(args.d, args.k))
        self.protein_disease_H = nn.Parameter(torch.Tensor(args.d, args.k))

        self.protein_protein_G = nn.Parameter(torch.Tensor(args.d, args.k))

        self.protein_sequence_G = nn.Parameter(torch.Tensor(args.d, args.k))

        self.reset_parameters()
        self.apply(initialize_weights)

    def reset_parameters(self):
        nn.init.trunc_normal_(self.drug_disease_G, std=0.1, a=-0.2, b=+0.2)
        nn.init.trunc_normal_(self.drug_disease_H, std=0.1, a=-0.2, b=+0.2)
        nn.init.trunc_normal_(self.drug_drug_G, std=0.1, a=-0.2, b=+0.2)
        nn.init.trunc_normal_(self.drug_chemical_G, std=0.1, a=-0.2, b=+0.2)
        nn.init.trunc_normal_(self.drug_sideeffect_G, std=0.1, a=-0.2, b=+0.2)
        nn.init.trunc_normal_(self.drug_sideeffect_H, std=0.1, a=-0.2, b=+0.2)
        nn.init.trunc_normal_(self.drug_protein_G, std=0.1, a=-0.2, b=+0.2)
        nn.init.trunc_normal_(self.drug_protein_H, std=0.1, a=-0.2, b=+0.2)
        nn.init.trunc_normal_(self.protein_disease_G, std=0.1, a=-0.2, b=+0.2)
        nn.init.trunc_normal_(self.protein_disease_H, std=0.1, a=-0.2, b=+0.2)
        nn.init.trunc_normal_(self.protein_protein_G, std=0.1, a=-0.2, b=+0.2)
        nn.init.trunc_normal_(self.protein_sequence_G, std=0.1, a=-0.2, b=+0.2)
        nn.init.trunc_normal_(self.drug_embedding, std=0.1, a=-0.2, b=+0.2)
        nn.init.trunc_normal_(self.protein_embedding, std=0.1, a=-0.2, b=+0.2)
        nn.init.trunc_normal_(self.disease_embedding, std=0.1, a=-0.2, b=+0.2)
        nn.init.trunc_normal_(self.sideeffect_embedding, std=0.1, a=-0.2, b=+0.2)

    def forward(
        self,
        drug_drug_normalize,
        drug_chemical_normalize,
        drug_disease_normalize,
        drug_sideeffect_normalize,
        protein_protein_normalize,
        protein_sequence_normalize,
        protein_disease_normalize,
        disease_drug_normalize,
        disease_protein_normalize,
        sideeffect_drug_normalize,
        drug_protein_normalize,
        protein_drug_normalize,
        drug_drug,
        drug_chemical,
        drug_disease,
        drug_sideeffect,
        protein_protein,
        protein_sequence,
        protein_disease,
        drug_protein,
        drug_protein_mask,
    ):
        # passing 1 times (can be easily extended to multiple passes)
        drug_vector1 = F.normalize(
            F.relu(
                self.W1(
                    torch.cat(
                        [
                            torch.matmul(
                                drug_drug_normalize, self.drug_drug_wr(self.drug_embedding)
                            )
                            + torch.matmul(
                                drug_chemical_normalize, self.drug_chemical_wr(self.drug_embedding)
                            )
                            + torch.matmul(
                                drug_disease_normalize, self.drug_disease_wr(self.disease_embedding)
                            )
                            + torch.matmul(
                                drug_sideeffect_normalize,
                                self.drug_sideeffect_wr(self.sideeffect_embedding),
                            )
                            + torch.matmul(
                                drug_protein_normalize, self.drug_protein_wr(self.protein_embedding)
                            ),
                            self.drug_embedding,
                        ],
                        dim=1,
                    )
                )
            ),
            p=2.0,
            dim=1,
        )

        protein_vector1 = F.normalize(
            F.relu(
                self.W1(
                    torch.cat(
                        [
                            torch.matmul(
                                protein_protein_normalize,
                                self.protein_protein_wr(self.protein_embedding),
                            )
                            + torch.matmul(
                                protein_sequence_normalize,
                                self.protein_sequence_wr(self.protein_embedding),
                            )
                            + torch.matmul(
                                protein_disease_normalize,
                                self.protein_disease_wr(self.disease_embedding),
                            )
                            + torch.matmul(
                                protein_drug_normalize, self.protein_drug_wr(self.drug_embedding)
                            ),
                            self.protein_embedding,
                        ],
                        dim=1,
                    )
                )
            ),
            p=2.0,
            dim=1,
        )

        disease_vector1 = F.normalize(
            F.relu(
                self.W1(
                    torch.cat(
                        [
                            torch.matmul(
                                disease_drug_normalize, self.disease_drug_wr(self.drug_embedding)
                            )
                            + torch.matmul(
                                disease_protein_normalize,
                                self.disease_protein_wr(self.protein_embedding),
                            ),
                            self.disease_embedding,
                        ],
                        dim=1,
                    )
                )
            ),
            p=2.0,
            dim=1,
        )

        sideeffect_vector1 = F.normalize(
            F.relu(
                self.W1(
                    torch.cat(
                        [
                            torch.matmul(
                                sideeffect_drug_normalize,
                                self.sideeffect_drug_wr(self.drug_embedding),
                            ),
                            self.sideeffect_embedding,
                        ],
                        dim=1,
                    )
                )
            ),
            p=2.0,
            dim=1,
        )
        # reconstructing networks
        drug_drug_reconstruct = torch.linalg.multi_dot(
            [drug_vector1, self.drug_drug_G, self.drug_drug_G.T, drug_vector1.T]
        )
        drug_drug_reconstruct_loss = self.criterion(drug_drug_reconstruct, drug_drug)

        drug_chemical_reconstruct = torch.linalg.multi_dot(
            [drug_vector1, self.drug_chemical_G, self.drug_chemical_G.T, drug_vector1.T]
        )
        drug_chemical_reconstruct_loss = self.criterion(drug_chemical_reconstruct, drug_chemical)

        drug_disease_reconstruct = torch.linalg.multi_dot(
            [drug_vector1, self.drug_disease_G, self.drug_disease_H.T, disease_vector1.T]
        )
        drug_disease_reconstruct_loss = self.criterion(drug_disease_reconstruct, drug_disease)

        drug_sideeffect_reconstruct = torch.linalg.multi_dot(
            [drug_vector1, self.drug_sideeffect_G, self.drug_sideeffect_H.T, sideeffect_vector1.T]
        )
        drug_sideeffect_reconstruct_loss = self.criterion(
            drug_sideeffect_reconstruct, drug_sideeffect
        )

        protein_protein_reconstruct = torch.linalg.multi_dot(
            [protein_vector1, self.protein_protein_G, self.protein_protein_G.T, protein_vector1.T]
        )
        protein_protein_reconstruct_loss = self.criterion(
            protein_protein_reconstruct, protein_protein
        )

        protein_sequence_reconstruct = torch.linalg.multi_dot(
            [protein_vector1, self.protein_sequence_G, self.protein_sequence_G.T, protein_vector1.T]
        )
        protein_sequence_reconstruct_loss = self.criterion(
            protein_sequence_reconstruct, protein_sequence
        )

        protein_disease_reconstruct = torch.linalg.multi_dot(
            [protein_vector1, self.protein_disease_G, self.protein_disease_H.T, disease_vector1.T]
        )
        protein_disease_reconstruct_loss = self.criterion(
            protein_disease_reconstruct, protein_disease
        )

        drug_protein_reconstruct = torch.linalg.multi_dot(
            [drug_vector1, self.drug_protein_G, self.drug_protein_H.T, protein_vector1.T]
        )
        tmp = torch.mul(drug_protein_mask, (drug_protein_reconstruct - drug_protein))
        drug_protein_reconstruct_loss = torch.sum(tmp.pow(2))

        # l2-regularization
        l2_loss = (
            torch.linalg.matrix_norm(self.drug_embedding) ** 2
            + torch.linalg.matrix_norm(self.protein_embedding) ** 2
            + torch.linalg.matrix_norm(self.disease_embedding) ** 2
            + torch.linalg.matrix_norm(self.sideeffect_embedding) ** 2
            + torch.linalg.matrix_norm(self.drug_drug_wr.weight) ** 2
            + torch.linalg.matrix_norm(self.drug_chemical_wr.weight) ** 2
            + torch.linalg.matrix_norm(self.drug_disease_wr.weight) ** 2
            + torch.linalg.matrix_norm(self.drug_sideeffect_wr.weight) ** 2
            + torch.linalg.matrix_norm(self.drug_protein_wr.weight) ** 2
            + torch.linalg.matrix_norm(self.protein_protein_wr.weight) ** 2
            + torch.linalg.matrix_norm(self.protein_sequence_wr.weight) ** 2
            + torch.linalg.matrix_norm(self.protein_disease_wr.weight) ** 2
            + torch.linalg.matrix_norm(self.protein_drug_wr.weight) ** 2
            + torch.linalg.matrix_norm(self.disease_drug_wr.weight) ** 2
            + torch.linalg.matrix_norm(self.disease_protein_wr.weight) ** 2
            + torch.linalg.matrix_norm(self.sideeffect_drug_wr.weight) ** 2
            + torch.linalg.matrix_norm(self.W1.weight) ** 2
            + torch.linalg.matrix_norm(self.drug_disease_G) ** 2
            + torch.linalg.matrix_norm(self.drug_disease_H) ** 2
            + torch.linalg.matrix_norm(self.drug_drug_G) ** 2
            + torch.linalg.matrix_norm(self.drug_chemical_G) ** 2
            + torch.linalg.matrix_norm(self.drug_sideeffect_G) ** 2
            + torch.linalg.matrix_norm(self.drug_sideeffect_H) ** 2
            + torch.linalg.matrix_norm(self.drug_protein_G) ** 2
            + torch.linalg.matrix_norm(self.drug_protein_H) ** 2
            + torch.linalg.matrix_norm(self.protein_disease_G) ** 2
            + torch.linalg.matrix_norm(self.protein_disease_H) ** 2
            + torch.linalg.matrix_norm(self.protein_protein_G) ** 2
            + torch.linalg.matrix_norm(self.protein_sequence_G) ** 2
        )

        total_loss = (
            drug_protein_reconstruct_loss
            + 1.0
            * (
                drug_drug_reconstruct_loss
                + drug_chemical_reconstruct_loss
                + drug_disease_reconstruct_loss
                + drug_sideeffect_reconstruct_loss
                + protein_protein_reconstruct_loss
                + protein_sequence_reconstruct_loss
                + protein_disease_reconstruct_loss
            )
            + l2_loss * self.args.l2_factor
        )

        return (
            total_loss,
            drug_protein_reconstruct_loss,
            drug_protein_reconstruct.detach().cpu().numpy(),
        )


class _Args:
    """Plain namespace mirroring the real repo's argparse defaults
    (`src/pytorch_NeoDTI_cv.py`): d=1024, k=512, l2-factor=0.1. d/k are shrunk here
    (tiny embedding/projection dims) to keep the trace lightweight; l2_factor keeps
    its real default value."""

    def __init__(self, d, k):
        self.d = d
        self.k = k
        self.l2_factor = 0.1


def build_neodti():
    # Small placeholder entity counts (the real heterogeneous-network dataset's counts
    # are not shipped standalone); d/k shrunk from the real 1024/512 defaults to keep
    # every relation branch and reconstruction head cheap to trace.
    args = _Args(d=16, k=8)
    num_drug, num_protein, num_disease, num_sideeffect = 12, 10, 6, 5
    return NeoDTI(args, num_drug, num_disease, num_protein, num_sideeffect)


def example_input_neodti():
    num_drug, num_protein, num_disease, num_sideeffect = 12, 10, 6, 5

    def row_normalize(mat):
        row_sums = mat.sum(dim=1, keepdim=True) + 1e-12
        return mat / row_sums

    drug_drug = torch.rand(num_drug, num_drug)
    drug_chemical = torch.rand(num_drug, num_drug)
    drug_disease = torch.rand(num_drug, num_disease)
    drug_sideeffect = torch.rand(num_drug, num_sideeffect)
    protein_protein = torch.rand(num_protein, num_protein)
    protein_sequence = torch.rand(num_protein, num_protein)
    protein_disease = torch.rand(num_protein, num_disease)
    drug_protein = torch.rand(num_drug, num_protein)
    drug_protein_mask = torch.ones(num_drug, num_protein)

    return (
        row_normalize(drug_drug),
        row_normalize(drug_chemical),
        row_normalize(drug_disease),
        row_normalize(drug_sideeffect),
        row_normalize(protein_protein),
        row_normalize(protein_sequence),
        row_normalize(protein_disease),
        row_normalize(drug_disease.T),
        row_normalize(protein_disease.T),
        row_normalize(drug_sideeffect.T),
        row_normalize(drug_protein),
        row_normalize(drug_protein.T),
        drug_drug,
        drug_chemical,
        drug_disease,
        drug_sideeffect,
        protein_protein,
        protein_sequence,
        protein_disease,
        drug_protein,
        drug_protein_mask,
    )


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("NeoDTI", "build_neodti", "example_input_neodti", 2019, "vendored"),
]
