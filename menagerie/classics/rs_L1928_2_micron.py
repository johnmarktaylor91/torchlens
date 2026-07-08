# SOURCE: vendored from ycq091044/MICRON @ main
# https://raw.githubusercontent.com/ycq091044/MICRON/main/src/models.py
#
# "Change Matters: Medication Change Prediction with Recurrent Residual Networks"
# (Yang et al., IJCAI 2021). `MICRON` is copied verbatim from src/models.py (the
# `GCN`/`GraphConvolution` sibling classes used by the repo's other baselines are
# dropped -- MICRON's forward pass never calls them; only the unused `# from dnc
# import DNC` import is dropped). MICRON encodes a patient's current + previous
# clinical visit (diagnosis codes + procedure codes) into a "health representation",
# then predicts the current medication set both directly and as a residual update
# on top of the previous visit's prediction (the paper's central "recurrent residual"
# mechanism) plus a DDI-graph regularization term and a reconstruction-consistency term.
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


class MICRON(nn.Module):
    def __init__(self, vocab_size, ddi_adj, emb_dim=256, device=torch.device("cpu:0")):
        super(MICRON, self).__init__()

        self.device = device

        # pre-embedding
        self.embeddings = nn.ModuleList([nn.Embedding(vocab_size[i], emb_dim) for i in range(2)])
        self.dropout = nn.Dropout(p=0.5)

        self.health_net = nn.Sequential(nn.Linear(2 * emb_dim, emb_dim))

        #
        self.prescription_net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4), nn.ReLU(), nn.Linear(emb_dim * 4, vocab_size[2])
        )

        # graphs, bipartite matrix
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.init_weights()

    def forward(self, input):
        # patient health representation
        def sum_embedding(embedding):
            return embedding.sum(dim=1).unsqueeze(dim=0)  # (1,1,dim)

        diag_emb = sum_embedding(
            self.dropout(
                self.embeddings[0](torch.LongTensor(input[-1][0]).unsqueeze(dim=0).to(self.device))
            )
        )  # (1,1,dim)
        prod_emb = sum_embedding(
            self.dropout(
                self.embeddings[1](torch.LongTensor(input[-1][1]).unsqueeze(dim=0).to(self.device))
            )
        )

        if len(input) < 2:
            diag_emb_last = diag_emb * torch.tensor(0.0)
            prod_emb_last = diag_emb * torch.tensor(0.0)
        else:
            diag_emb_last = sum_embedding(
                self.dropout(
                    self.embeddings[0](
                        torch.LongTensor(input[-2][0]).unsqueeze(dim=0).to(self.device)
                    )
                )
            )  # (1,1,dim)
            prod_emb_last = sum_embedding(
                self.dropout(
                    self.embeddings[1](
                        torch.LongTensor(input[-2][1]).unsqueeze(dim=0).to(self.device)
                    )
                )
            )

        health_representation = torch.cat([diag_emb, prod_emb], dim=-1).squeeze(
            dim=0
        )  # (seq, dim*2)
        health_representation_last = torch.cat([diag_emb_last, prod_emb_last], dim=-1).squeeze(
            dim=0
        )  # (seq, dim*2)

        health_rep = self.health_net(health_representation)[-1:, :]  # (seq, dim)
        health_rep_last = self.health_net(health_representation_last)[-1:, :]  # (seq, dim)
        health_residual_rep = health_rep - health_rep_last

        # drug representation
        drug_rep = self.prescription_net(health_rep)
        drug_rep_last = self.prescription_net(health_rep_last)
        drug_residual_rep = self.prescription_net(health_residual_rep)

        # reconstructon loss
        rec_loss = (
            1
            / self.tensor_ddi_adj.shape[0]
            * torch.sum(
                torch.pow((F.sigmoid(drug_rep) - F.sigmoid(drug_rep_last + drug_residual_rep)), 2)
            )
        )

        # ddi_loss
        neg_pred_prob = F.sigmoid(drug_rep)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)

        batch_neg = 1 / self.tensor_ddi_adj.shape[0] * neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        return drug_rep, drug_rep_last, drug_residual_rep, batch_neg, rec_loss

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)


def build_micron():
    torch.manual_seed(0)
    import numpy as np

    n_diag, n_proc, n_med = 30, 20, 15
    ddi_adj = np.zeros((n_med, n_med), dtype="float32")
    vocab_size = (n_diag, n_proc, n_med)
    return MICRON(vocab_size, ddi_adj, emb_dim=8, device=torch.device("cpu"))


def example_input_micron():
    # A patient history: list of visits, each visit = [diag_code_list, proc_code_list, med_code_list].
    # MICRON's forward() only reads input[-1] (current visit) and input[-2] (previous visit)
    # diag/proc codes -- med codes at those indices are unused by forward (only present in
    # real data for downstream loss targets computed outside the model).
    visit_prev = [[1, 4, 7], [2, 5], [0, 1, 2]]
    visit_curr = [[1, 4, 9, 12], [2, 6], [0, 1, 3]]
    return ([visit_prev, visit_curr],)


MENAGERIE_ENTRIES = [
    ("MICRON", "build_micron", "example_input_micron", 2021, "vendored"),
]
