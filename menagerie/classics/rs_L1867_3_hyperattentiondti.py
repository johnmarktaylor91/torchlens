# SOURCE: vendored from zhaoqichang/HpyerAttentionDTI @ ab121088fa95
# (model.py::AttentionDTI, hyperparameter.py::hyperparameter). HyperAttentionDTI
# (Bioinformatics 2022) is a drug-target interaction predictor: parallel 1D-CNN
# towers over the drug SMILES character sequence and the protein amino-acid
# sequence, fused via a learned pairwise hyper-attention map (an outer-sum of
# per-position linear projections passed through a shared attention layer,
# reduced with mean-pooling to per-position gates) that reweights each CNN
# feature map before max-pooling and a 3-layer MLP classification head. Class
# bodies below are copied verbatim from the real repo files; only the module
# docstring/header comment was added.
"""Vendored HyperAttentionDTI (zhaoqichang/HpyerAttentionDTI)."""

import torch
import torch.nn as nn
from datetime import datetime

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# hyperparameter.py (real repo class, verbatim)
# ---------------------------------------------------------------------------


class hyperparameter:
    def __init__(self):
        self.current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        self.Learning_rate = 5e-5
        self.Epoch = 200
        self.Batch_size = 32
        self.Resume = False
        self.Patience = 50
        self.FC_Dropout = 0.5
        self.test_split = 0.2
        self.validation_split = 0.2
        self.decay_interval = 10
        self.lr_decay = 0.5
        self.weight_decay = 1e-4
        self.embed_dim = 64

        self.protein_kernel = [4, 8, 12]
        self.drug_kernel = [4, 6, 8]
        self.conv = 40
        self.char_dim = 64


# ---------------------------------------------------------------------------
# model.py (real repo class, verbatim)
# ---------------------------------------------------------------------------


class AttentionDTI(nn.Module):
    def __init__(self, hp, protein_MAX_LENGH=1000, drug_MAX_LENGH=100):
        super(AttentionDTI, self).__init__()
        self.dim = hp.char_dim
        self.conv = hp.conv
        self.drug_MAX_LENGH = drug_MAX_LENGH
        self.drug_kernel = hp.drug_kernel
        self.protein_MAX_LENGH = protein_MAX_LENGH
        self.protein_kernel = hp.protein_kernel

        self.protein_embed = nn.Embedding(26, self.dim, padding_idx=0)
        self.drug_embed = nn.Embedding(65, self.dim, padding_idx=0)
        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(
                in_channels=self.dim, out_channels=self.conv, kernel_size=self.drug_kernel[0]
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=self.conv, out_channels=self.conv * 2, kernel_size=self.drug_kernel[1]
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=self.conv * 2,
                out_channels=self.conv * 4,
                kernel_size=self.drug_kernel[2],
            ),
            nn.ReLU(),
        )
        self.Drug_max_pool = nn.MaxPool1d(
            self.drug_MAX_LENGH
            - self.drug_kernel[0]
            - self.drug_kernel[1]
            - self.drug_kernel[2]
            + 3
        )
        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(
                in_channels=self.dim, out_channels=self.conv, kernel_size=self.protein_kernel[0]
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=self.conv,
                out_channels=self.conv * 2,
                kernel_size=self.protein_kernel[1],
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=self.conv * 2,
                out_channels=self.conv * 4,
                kernel_size=self.protein_kernel[2],
            ),
            nn.ReLU(),
        )
        self.Protein_max_pool = nn.MaxPool1d(
            self.protein_MAX_LENGH
            - self.protein_kernel[0]
            - self.protein_kernel[1]
            - self.protein_kernel[2]
            + 3
        )
        self.attention_layer = nn.Linear(self.conv * 4, self.conv * 4)
        self.protein_attention_layer = nn.Linear(self.conv * 4, self.conv * 4)
        self.drug_attention_layer = nn.Linear(self.conv * 4, self.conv * 4)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(self.conv * 8, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 2)

    def forward(self, drug, protein):
        drugembed = self.drug_embed(drug)
        proteinembed = self.protein_embed(protein)
        drugembed = drugembed.permute(0, 2, 1)
        proteinembed = proteinembed.permute(0, 2, 1)

        drugConv = self.Drug_CNNs(drugembed)
        proteinConv = self.Protein_CNNs(proteinembed)

        drug_att = self.drug_attention_layer(drugConv.permute(0, 2, 1))
        protein_att = self.protein_attention_layer(proteinConv.permute(0, 2, 1))

        d_att_layers = torch.unsqueeze(drug_att, 2).repeat(
            1, 1, proteinConv.shape[-1], 1
        )  # repeat along protein size
        p_att_layers = torch.unsqueeze(protein_att, 1).repeat(
            1, drugConv.shape[-1], 1, 1
        )  # repeat along drug size
        Atten_matrix = self.attention_layer(self.relu(d_att_layers + p_att_layers))
        Compound_atte = torch.mean(Atten_matrix, 2)
        Protein_atte = torch.mean(Atten_matrix, 1)
        Compound_atte = self.sigmoid(Compound_atte.permute(0, 2, 1))
        Protein_atte = self.sigmoid(Protein_atte.permute(0, 2, 1))

        drugConv = drugConv * 0.5 + drugConv * Compound_atte
        proteinConv = proteinConv * 0.5 + proteinConv * Protein_atte

        drugConv = self.Drug_max_pool(drugConv).squeeze(2)
        proteinConv = self.Protein_max_pool(proteinConv).squeeze(2)

        pair = torch.cat([drugConv, proteinConv], dim=1)
        pair = self.dropout1(pair)
        fully1 = self.leaky_relu(self.fc1(pair))
        fully1 = self.dropout2(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout3(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3)
        return predict


# ---------------------------------------------------------------------------
# Staging build/example helpers. Real constructor with default kwargs
# (protein_MAX_LENGH=1000, drug_MAX_LENGH=100 matching the repo's own
# CHARISOSMILEN/CHARPROTLEN maximum sequence lengths used at training time);
# drug/protein token ids are integer-coded sequences (drug vocab size 65,
# protein vocab size 26, both padding_idx=0), matching the repo's
# label_smiles/label_sequence tokenizers, so this is a MODULE
# (two-tensor-input contract) rather than a single-tensor recipe.
# ---------------------------------------------------------------------------


def build_hyperattentiondti():
    hp = hyperparameter()
    return AttentionDTI(hp, protein_MAX_LENGH=1000, drug_MAX_LENGH=100)


def example_input_hyperattentiondti():
    torch.manual_seed(0)
    drug = torch.randint(1, 65, (2, 100))
    protein = torch.randint(1, 26, (2, 1000))
    return (drug, protein)


MENAGERIE_ENTRIES = [
    (
        "HyperAttentionDTI",
        "build_hyperattentiondti",
        "example_input_hyperattentiondti",
        2022,
        "vendored-pytorch",
    ),
]
