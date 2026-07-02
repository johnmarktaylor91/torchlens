# SOURCE: vendored from https://github.com/Liruiqing-ustc/HFBSurv @ main
# (HFBSurv/model/HFB_fusion.py)
# HFBSurv: hierarchical factorized bilinear fusion for genomic+histology multimodal
# survival prediction (Li, Liu et al., Bioinformatics 2022). The `SubNet` and
# `HFBSurv` classes below are copied verbatim from the official repo's
# model/HFB_fusion.py (modality-specific SubNet encoders, intra-modal + inter-modal
# hierarchical factorized bilinear pooling (`mfb`), gated attention, and the
# survival (Cox) prediction head). Only the module-level `Parameter` import and
# construction below were adjusted for staging; no architectural line was changed.
"""Vendored HFBSurv model definition (SubNet, HFBSurv)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

MENAGERIE_ZOO = "vendored-pytorch"


class SubNet(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(SubNet, self).__init__()
        encoder1 = nn.Sequential(nn.Linear(in_size, hidden_size), nn.Tanh())
        encoder2 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.encoder = nn.Sequential(encoder1, encoder2)

    def forward(self, x):
        y = self.encoder(x)
        return y


class HFBSurv(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, dropouts, rank, fac_drop):
        super(HFBSurv, self).__init__()

        self.gene_in = input_dims[0]
        self.path_in = input_dims[1]
        self.cona_in = input_dims[2]

        self.gene_hidden = hidden_dims[0]
        self.path_hidden = hidden_dims[1]
        self.cona_hidden = hidden_dims[2]
        self.cox_hidden = hidden_dims[3]

        self.output_intra = output_dims[0]
        self.output_inter = output_dims[1]
        self.label_dim = output_dims[2]
        self.rank = rank
        self.factor_drop = fac_drop

        self.gene_prob = dropouts[0]
        self.path_prob = dropouts[1]
        self.cona_prob = dropouts[2]
        self.cox_prob = dropouts[3]

        self.joint_output_intra = self.rank * self.output_intra
        self.joint_output_inter = self.rank * self.output_inter
        self.in_size = self.gene_hidden + self.output_intra + self.output_inter
        self.hid_size = self.gene_hidden

        self.norm = nn.BatchNorm1d(self.in_size)
        self.factor_drop = nn.Dropout(self.factor_drop)
        self.attention = nn.Sequential(
            nn.Linear((self.hid_size + self.output_intra), 1), nn.Sigmoid()
        )

        self.encoder_gene = SubNet(self.gene_in, self.gene_hidden)
        self.encoder_path = SubNet(self.path_in, self.path_hidden)
        self.encoder_cona = SubNet(self.cona_in, self.cona_hidden)

        self.Linear_gene = nn.Linear(self.gene_hidden, self.joint_output_intra)
        self.Linear_path = nn.Linear(self.path_hidden, self.joint_output_intra)
        self.Linear_cona = nn.Linear(self.cona_hidden, self.joint_output_intra)

        self.Linear_gene_a = nn.Linear(
            self.gene_hidden + self.output_intra, self.joint_output_inter
        )
        self.Linear_path_a = nn.Linear(
            self.path_hidden + self.output_intra, self.joint_output_inter
        )
        self.Linear_cona_a = nn.Linear(
            self.cona_hidden + self.output_intra, self.joint_output_inter
        )

        #########################the layers of survival prediction#####################################
        encoder1 = nn.Sequential(
            nn.Linear(self.in_size, self.cox_hidden), nn.Tanh(), nn.Dropout(p=self.cox_prob)
        )
        encoder2 = nn.Sequential(
            nn.Linear(self.cox_hidden, 64), nn.Tanh(), nn.Dropout(p=self.cox_prob)
        )
        self.encoder = nn.Sequential(encoder1, encoder2)
        self.classifier = nn.Sequential(nn.Linear(64, self.label_dim), nn.Sigmoid())

        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def mfb(self, x1, x2, output_dim):
        self.output_dim = output_dim
        fusion = torch.mul(x1, x2)
        fusion = self.factor_drop(fusion)
        fusion = fusion.view(-1, 1, self.output_dim, self.rank)
        fusion = torch.squeeze(torch.sum(fusion, 3))
        fusion = torch.sqrt(F.relu(fusion)) - torch.sqrt(F.relu(-fusion))
        fusion = F.normalize(fusion)
        return fusion

    def forward(self, x1, x2, x3):
        gene_feature = self.encoder_gene(x1.squeeze(1))
        path_feature = self.encoder_path(x2.squeeze(1))
        cona_feature = self.encoder_cona(x3.squeeze(1))

        ######################### modelity-specific###############################
        gene_h = self.Linear_gene(gene_feature)
        path_h = self.Linear_path(path_feature)
        cona_h = self.Linear_cona(cona_feature)

        # intra_interaction#
        intra_gene = self.mfb(gene_h, gene_h, self.output_intra)
        intra_path = self.mfb(path_h, path_h, self.output_intra)
        intra_cona = self.mfb(cona_h, cona_h, self.output_intra)

        gene_x = torch.cat((gene_feature, intra_gene), 1)
        path_x = torch.cat((path_feature, intra_path), 1)
        cona_x = torch.cat((cona_feature, intra_cona), 1)

        sg = self.attention(gene_x)
        sp = self.attention(path_x)
        sc = self.attention(cona_x)

        sg_a = sg.expand(gene_feature.size(0), (self.gene_hidden + self.output_intra))
        sp_a = sp.expand(path_feature.size(0), (self.path_hidden + self.output_intra))
        sc_a = sc.expand(cona_feature.size(0), (self.cona_hidden + self.output_intra))

        gene_x_a = sg_a * gene_x
        path_x_a = sp_a * path_x
        cona_x_a = sc_a * gene_x

        unimodal = gene_x_a + path_x_a + cona_x_a

        ######################### cross-modelity######################################
        g = F.softmax(gene_x_a, 1)
        p = F.softmax(path_x_a, 1)
        c = F.softmax(cona_x_a, 1)

        sg = sg.squeeze()
        sp = sp.squeeze()
        sc = sc.squeeze()

        sgp = 1 / (torch.matmul(g.unsqueeze(1), p.unsqueeze(2)).squeeze() + 0.5) * (sg + sp)
        sgc = 1 / (torch.matmul(g.unsqueeze(1), c.unsqueeze(2)).squeeze() + 0.5) * (sg + sc)
        spc = 1 / (torch.matmul(p.unsqueeze(1), c.unsqueeze(2)).squeeze() + 0.5) * (sp + sc)
        normalize = torch.cat((sgp.unsqueeze(1), sgc.unsqueeze(1), spc.unsqueeze(1)), 1)
        normalize = F.softmax(normalize, 1)
        sgp_a = normalize[:, 0].unsqueeze(1).expand(gene_feature.size(0), self.output_inter)
        sgc_a = normalize[:, 1].unsqueeze(1).expand(path_feature.size(0), self.output_inter)
        spc_a = normalize[:, 2].unsqueeze(1).expand(cona_feature.size(0), self.output_inter)

        # inter_interaction#
        gene_l = self.Linear_gene_a(gene_x_a)
        path_l = self.Linear_gene_a(path_x_a)
        cona_l = self.Linear_gene_a(cona_x_a)

        inter_gene_path = self.mfb(gene_l, path_l, self.output_inter)
        inter_gene_cona = self.mfb(gene_l, cona_l, self.output_inter)
        inter_path_cona = self.mfb(path_l, cona_l, self.output_inter)

        bimodal = sgp_a * inter_gene_path + sgc_a * inter_gene_cona + spc_a * inter_path_cona
        ############################################### fusion layer ###################################################

        fusion = torch.cat((unimodal, bimodal), 1)
        fusion = self.norm(fusion)
        code = self.encoder(fusion)
        out = self.classifier(code)
        out = out * self.output_range + self.output_shift
        return out, code


def build_hfbsurv():
    # Real hyperparameters from HFBSurv/model/train_test.py's `HFBSurv((80, 80, 80),
    # (50, 50, 50, 256), (20, 20, 1), (0.1, 0.1, 0.1, 0.3), 20, 0.1)` construction call.
    model = HFBSurv(
        input_dims=(80, 80, 80),
        hidden_dims=(50, 50, 50, 256),
        output_dims=(20, 20, 1),
        dropouts=(0.1, 0.1, 0.1, 0.3),
        rank=20,
        fac_drop=0.1,
    )
    model.eval()
    return model


def example_input_hfbsurv():
    # Real DataLoader path feeds (batch, 1, feature_dim) tensors per modality
    # (gene/pathway/CNA); use batch=4 so BatchNorm1d in eval mode is well-defined.
    x_gene = torch.randn(4, 1, 80)
    x_path = torch.randn(4, 1, 80)
    x_cna = torch.randn(4, 1, 80)
    return (x_gene, x_path, x_cna)


MENAGERIE_ENTRIES = [
    ("HFBSurv", build_hfbsurv, example_input_hfbsurv, 2022, "vendored-pytorch"),
]
