# SOURCE: vendored from https://github.com/bigdata-ustc/EduCDM @ main
#   Vendored file:
#     - EduCDM/NCDM/NCDM.py -> `PosLinear` and `Net`, the NCDM prediction sub-net
#       (the `NCDM` wrapper class's train/eval/save/load driver methods are training
#       harness, not architecture, and are not vendored; only the traced `Net`
#       nn.Module is reproduced here)
#
# NCDM ("Neural Cognitive Diagnosis for Intelligent Education Systems", AAAI 2020;
# also mirrored as bigdata-ustc/Neural_Cognitive_Diagnosis-NeuralCD) is a cognitive
# diagnosis model: student proficiency, per-exercise knowledge-concept difficulty,
# and per-exercise overall discrimination are each learned embeddings; a Rasch-style
# interaction term `e_difficulty * (student_proficiency - k_difficulty) *
# knowledge_relevancy_mask` is fed through a 3-layer MLP whose Linear layers use a
# monotonicity constraint (`PosLinear`: weights are clamped non-negative via
# `2*relu(-w)+w`, enforcing "higher proficiency never lowers predicted correctness")
# to produce a single correctness-probability logit per (student, exercise) pair.

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)


class Net(nn.Module):
    def __init__(self, knowledge_n, exer_n, student_n):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        super(Net, self).__init__()

        # prediction sub-net
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_difficulty = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        # initialize
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, input_exercise, input_knowledge_point):
        # before prednet
        stu_emb = self.student_emb(stu_id)
        stat_emb = torch.sigmoid(stu_emb)
        k_difficulty = torch.sigmoid(self.k_difficulty(input_exercise))
        e_difficulty = torch.sigmoid(self.e_difficulty(input_exercise))  # * 10
        # prednet
        input_x = e_difficulty * (stat_emb - k_difficulty) * input_knowledge_point
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        return output_1.view(-1)


# ---- tiny build/example (architecture unmodified from the real repo) ----


def build_ncdm():
    """Tiny NCDM Net for tracing. Architecture is unmodified from the real repo."""
    model = Net(knowledge_n=8, exer_n=20, student_n=15)
    model.eval()
    return model


def example_input_ncdm():
    batch = 4
    stu_id = torch.randint(0, 15, (batch,))
    input_exercise = torch.randint(0, 20, (batch,))
    # multi-hot knowledge-concept relevancy mask per exercise, shape (batch, knowledge_n)
    input_knowledge_point = (torch.rand(batch, 8) > 0.5).float()
    return (stu_id, input_exercise, input_knowledge_point)


MENAGERIE_ENTRIES = [
    ("NCDM", build_ncdm, example_input_ncdm, 2020, "vendored-pytorch"),
]
