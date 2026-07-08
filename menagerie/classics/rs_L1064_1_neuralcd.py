# SOURCE: vendored from bigdata-ustc/Neural_Cognitive_Diagnosis-NeuralCD @ master
# https://raw.githubusercontent.com/bigdata-ustc/Neural_Cognitive_Diagnosis-NeuralCD/master/model.py
#
# Wang et al. 2020 (AAAI) "Neural Cognitive Diagnosis for Intelligent Education Systems" --
# NeuralCDM: a cognitive-diagnosis model that embeds student/exercise/knowledge-difficulty
# and exercise-discrimination as `nn.Embedding` tables, combines them via the item-response
# style interaction `e_discrimination * (student_ability - exercise_difficulty) * knowledge_relevancy`,
# then passes the result through a 3-layer monotonic ("non-negative-clipped") prediction network
# to output a per-(student, exercise) probability of a correct response. `NoneNegClipper` is the
# monotonicity constraint applied to the prediction-net weights after each optimizer step (called
# from the original repo's `train.py`; not required for a forward pass, kept here verbatim since it
# is part of the model's defined behavior/API surface, `Net.apply_clipper`).
#
# `Net` is the model exactly as defined upstream (constructor signature
# `(student_n, exer_n, knowledge_n)`, unchanged from the source). No architectural changes were
# made; only the top-of-file `import torch` / `import torch.nn as nn` are kept as in the original
# (already base-env-only) and a `build_neuralcd()` / `example_input_neuralcd()` pair was added for
# tiny-size random-init staging.

import torch
import torch.nn as nn


class Net(nn.Module):
    """
    NeuralCDM
    """

    def __init__(self, student_n, exer_n, knowledge_n):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        super(Net, self).__init__()

        # network structure
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_discrimination = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)

        # initialization
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id, kn_emb):
        """
        :param stu_id: LongTensor
        :param exer_id: LongTensor
        :param kn_emb: FloatTensor, the knowledge relevancy vectors
        :return: FloatTensor, the probabilities of answering correctly
        """
        # before prednet
        stu_emb = torch.sigmoid(self.student_emb(stu_id))
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        # prednet
        input_x = e_discrimination * (stu_emb - k_difficulty) * kn_emb
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output = torch.sigmoid(self.prednet_full3(input_x))

        return output

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

    def get_knowledge_status(self, stu_id):
        stat_emb = torch.sigmoid(self.student_emb(stu_id))
        return stat_emb.data

    def get_exer_params(self, exer_id):
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        return k_difficulty.data, e_discrimination.data


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, "weight"):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)


def build_neuralcd():
    student_n = 20
    exer_n = 15
    knowledge_n = 8
    return Net(student_n, exer_n, knowledge_n)


def example_input_neuralcd():
    batch = 4
    exer_n = 15
    knowledge_n = 8
    stu_id = torch.randint(0, 20, (batch,))
    exer_id = torch.randint(0, exer_n, (batch,))
    kn_emb = torch.rand(batch, knowledge_n)
    return (stu_id, exer_id, kn_emb)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("NeuralCD", "build_neuralcd", "example_input_neuralcd", 2020, "vendored"),
]
