# SOURCE: vendored from pykt-team/pykt-toolkit @ main
# (pykt/models/atdkt.py + the ut_mask helper it imports from pykt/models/utils.py)
#
# AT-DKT (Auxiliary Task Deep Knowledge Tracing): an LSTM-based knowledge-tracing model
# (interaction embedding -> LSTM -> sigmoid output head) augmented, in its "predcurc"/
# "predhis" emb_type variants, with auxiliary-task branches (a question-classifier head
# built from either a TransformerEncoder or a second LSTM, and a history-correctness
# regression head) that are jointly trained alongside the main knowledge-tracing objective.
# ATDKT.forward/predcurc and the ut_mask helper below are the REAL model code from
# atdkt.py/utils.py, copied verbatim (only base-lib deps: torch). build_atdkt() below
# traces the "qid_predcurc_trans" emb_type path (the auxiliary-task + Transformer variant)
# at tiny size -- the architecture is unmodified. `dcur` is the real multi-tensor input
# dict the model's forward() consumes (qseqs/cseqs/rseqs/smasks), so this ships as a module
# (dict input), not a single-tensor recipe row.

import torch
from torch import nn
from torch.nn import (
    CrossEntropyLoss,
    Dropout,
    Embedding,
    LayerNorm,
    Linear,
    LSTM,
    Module,
    TransformerEncoder,
    TransformerEncoderLayer,
)

MENAGERIE_ZOO = "vendored-pytorch"

device = "cpu"


def ut_mask(seq_len):
    """Upper Triangular Mask"""
    return torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(dtype=torch.bool).to(device)


class ATDKT(Module):
    def __init__(
        self,
        num_q,
        num_c,
        seq_len,
        emb_size,
        dropout=0.1,
        emb_type="qid",
        num_layers=1,
        num_attn_heads=5,
        l1=0.5,
        l2=0.5,
        l3=0.5,
        start=50,
        emb_path="",
        pretrain_dim=768,
    ):
        super().__init__()
        self.model_name = "atdkt"
        print(f"qnum: {num_q}, cnum: {num_c}")
        print(f"emb_type: {emb_type}")
        self.num_q = num_q
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type

        self.interaction_emb = Embedding(self.num_c * 2, self.emb_size)

        self.lstm_layer = LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(self.hidden_size // 2, self.num_c),
        )

        if self.emb_type.endswith("predhis"):
            self.l1 = l1
            self.l2 = l2
            if self.emb_type.find("cemb") != -1:
                self.concept_emb = Embedding(self.num_c, self.emb_size)  # add concept emb
            if self.emb_type.find("qemb") != -1:
                self.question_emb = Embedding(self.num_q, self.emb_size)

            self.start = start
            self.hisclasifier = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_size // 2, 1),
            )
            self.hisloss = nn.MSELoss()

        if self.emb_type.endswith("predcurc"):  # predict cur question' cur concept
            self.l1 = l1
            self.l2 = l2
            self.l3 = l3
            if self.num_q > 0:
                self.question_emb = Embedding(self.num_q, self.emb_size)  # 1.2
            if self.emb_type.find("trans") != -1:
                self.nhead = num_attn_heads
                d_model = self.hidden_size  # * 2
                encoder_layer = TransformerEncoderLayer(d_model, nhead=self.nhead)
                encoder_norm = LayerNorm(d_model)
                self.trans = TransformerEncoder(
                    encoder_layer, num_layers=num_layers, norm=encoder_norm
                )
            else:
                self.qlstm = LSTM(self.emb_size, self.hidden_size, batch_first=True)
            self.qclasifier = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                Linear(self.hidden_size // 2, self.num_c),
            )
            if self.emb_type.find("cemb") != -1:
                self.concept_emb = Embedding(self.num_c, self.emb_size)  # add concept emb

            self.closs = CrossEntropyLoss()
            if self.emb_type.find("his") != -1:
                self.start = start
                self.hisclasifier = nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.hidden_size // 2, 1),
                )
                self.hisloss = nn.MSELoss()

    def predcurc(self, dcur, q, c, r, xemb, train):
        emb_type = self.emb_type
        y2, y3 = 0, 0
        if emb_type.find("delxemb") != -1:
            qemb = self.question_emb(q)
            cemb = self.concept_emb(c)
            catemb = qemb + cemb
        else:
            catemb = xemb
            if self.num_q > 0:
                qemb = self.question_emb(q)
                catemb = qemb + xemb

            if emb_type.find("cemb") != -1:
                cemb = self.concept_emb(c)
                catemb += cemb

        if emb_type.find("trans") != -1:
            mask = ut_mask(seq_len=catemb.shape[1])
            qh = self.trans(catemb.transpose(0, 1), mask).transpose(0, 1)
        else:
            qh, _ = self.qlstm(catemb)
        if train:
            sm = dcur["smasks"].long()
            start = 0
            cpreds = self.qclasifier(qh[:, start:, :])
            flag = sm[:, start:] == 1
            y2 = self.closs(cpreds[flag], c[:, start:][flag])

        # predict response
        xemb = xemb + qh + cemb
        if emb_type.find("qemb") != -1:
            xemb = xemb + qemb
        h, _ = self.lstm_layer(xemb)

        # predict history correctness rates
        rpreds = None
        if train and emb_type.find("his") != -1:
            sm = dcur["smasks"].long()
            start = self.start
            rpreds = torch.sigmoid(self.hisclasifier(h)).squeeze(-1)
            rsm = sm[:, start:]
            rflag = rsm == 1
            rtrues = dcur["historycorrs"][:, start:]
            y3 = self.hisloss(rpreds[:, start:][rflag], rtrues[rflag])

        # predict response
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        y = torch.sigmoid(y)
        return y, y2, y3

    def forward(self, dcur, train=False):  # F * xemb
        q, c, r = dcur["qseqs"].long(), dcur["cseqs"].long(), dcur["rseqs"].long()

        y2, y3 = 0, 0

        emb_type = self.emb_type
        if emb_type.startswith("qid"):
            x = c + self.num_c * r
            xemb = self.interaction_emb(x)
        rpreds, qh = None, None  # noqa: F841 -- qh unused on this branch (verbatim from source)
        if emb_type == "qid":
            h, _ = self.lstm_layer(xemb)
            h = self.dropout_layer(h)
            y = torch.sigmoid(self.out_layer(h))
        elif emb_type.endswith("predhis"):  # only predict history correct ratios
            if self.emb_type.find("cemb") != -1:
                cemb = self.concept_emb(c)
                xemb = xemb + cemb
            if emb_type.find("qemb") != -1:
                qemb = self.question_emb(q)
                xemb = xemb + qemb
            h, _ = self.lstm_layer(xemb)
            if train:
                sm = dcur["smasks"].long()
                start = self.start
                rpreds = torch.sigmoid(self.hisclasifier(h)[:, start:, :]).squeeze(-1)
                rsm = sm[:, start:]
                rflag = rsm == 1
                rtrues = dcur["historycorrs"][:, start:]
                y2 = self.hisloss(rpreds[rflag], rtrues[rflag])

            h = self.dropout_layer(h)
            y = self.out_layer(h)
            y = torch.sigmoid(y)
        elif emb_type.endswith("predcurc"):  # predict current question' current concept
            y, y2, y3 = self.predcurc(dcur, q, c, r, xemb, train)

        if train:
            return y, y2, y3
        else:
            return y


def build_atdkt():
    """Tiny AT-DKT for tracing, using the exact emb_type from the repo's own tuned config
    (examples/seedwandb/atdkt.yaml): "qiddelxembhistranscembpredcurc" -- the full
    auxiliary-task variant (question-classifier head via Transformer + concept embedding +
    history-correctness regression head), all real branches of ATDKT.forward/predcurc.
    Architecture is unmodified from the real repo."""
    model = ATDKT(
        num_q=20,
        num_c=10,
        seq_len=8,
        emb_size=16,
        dropout=0.1,
        emb_type="qiddelxembhistranscembpredcurc",
        num_layers=1,
        num_attn_heads=4,
        start=2,
    )
    model.eval()
    return model


def example_input_atdkt():
    batch, seq_len, num_c = 2, 8, 10
    return {
        "qseqs": torch.randint(0, 20, (batch, seq_len)),
        "cseqs": torch.randint(0, num_c, (batch, seq_len)),
        "rseqs": torch.randint(0, 2, (batch, seq_len)),
        "smasks": torch.ones(batch, seq_len, dtype=torch.long),
        "historycorrs": torch.rand(batch, seq_len),
    }


MENAGERIE_ENTRIES = [
    ("ATDKT", build_atdkt, example_input_atdkt, 2023, "vendored-pytorch"),
]
