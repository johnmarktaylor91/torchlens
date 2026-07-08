# SOURCE: vendored from https://github.com/ECNU-ILOG/InsCD @ master
#   Vendored files:
#     - inscd/extractor/default.py   (Default extractor: student/exercise/knowledge
#       embeddings)
#     - inscd/interfunc/_util.py     (none_neg_clipper monotonicity utility)
#     - inscd/interfunc/kscd.py      (KSCD_IF interaction function)
#     - inscd/models/neural/kscd.py  (KSCD wrapper model, minimal _base.py shims below)
#   `_base._Extractor` / `_base._InteractionFunction` / `_base._CognitiveDiagnosisModel`
#   are reproduced as minimal ABCs/nn.Module shims (their real bodies are just
#   `@abstractmethod` stubs plus an nn.Module __init__ in the actual repo; nothing is
#   invented -- see inscd/_base.py).
#
# KSCD ("Knowledge-Sensed Cognitive Diagnosis for Intelligent Education Platforms",
# Ma et al., CIKM 2022) is included in the InsCD toolkit (ECNU-ILOG/InsCD). Its
# architecture: student/exercise/knowledge-concept embeddings (Default extractor) feed a
# knowledge-sensed interaction function (KSCD_IF) that computes per-concept student
# "preference" and exercise "difficulty" vectors via two shared linear layers over
# concatenated (ability|knowledge) and (difficulty|knowledge) representations, then
# predicts a response probability per concept and aggregates over the exercise's
# knowledge-concept mask. We trace `KSCD.diagnose()`, the model's public forward-style
# entry point (mirrors how inscd_run.py invokes trained models), fed by the extractor's
# per-student/-exercise embedding lookups computed in `extract()`.

from abc import abstractmethod

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# inscd/_base.py (minimal reproduction: ABC/nn.Module base classes only)
# ---------------------------------------------------------------------------
class _Extractor:
    @abstractmethod
    def extract(self, **kwargs): ...

    @abstractmethod
    def __getitem__(self, item): ...


class _InteractionFunction:
    @abstractmethod
    def compute(self, **kwargs): ...

    @abstractmethod
    def transform(self, mastery, knowledge): ...

    def monotonicity(self): ...


class _CognitiveDiagnosisModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.inter_func: _InteractionFunction = ...
        self.extractor: _Extractor = ...

    @abstractmethod
    def diagnose(self): ...


# ---------------------------------------------------------------------------
# inscd/interfunc/_util.py
# ---------------------------------------------------------------------------
class _NoneNegClipper(object):
    def __init__(self):
        super(_NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, "weight"):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)


none_neg_clipper = _NoneNegClipper()


# ---------------------------------------------------------------------------
# inscd/extractor/default.py
# ---------------------------------------------------------------------------
class Default(_Extractor, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.student_num = config["student_num"]
        self.exercise_num = config["exercise_num"]
        self.knowledge_num = config["knowledge_num"]

        if config.get("latent_dim", None) is None:
            self.latent_dim = config["knowledge_num"]
        else:
            self.latent_dim = config["latent_dim"]

        self.__student_emb = nn.Embedding(self.student_num, self.latent_dim)
        self.__knowledge_emb = nn.Embedding(self.knowledge_num, self.latent_dim)
        self.__diff_emb = nn.Embedding(self.exercise_num, self.latent_dim)
        self.__disc_emb = nn.Embedding(self.exercise_num, 1)

        self.__emb_map = {
            "mastery": self.__student_emb.weight,
            "diff": self.__diff_emb.weight,
            "disc": self.__disc_emb.weight,
            "knowledge": self.__knowledge_emb.weight,
        }
        self.apply(self.initialize_weights)

    @staticmethod
    def initialize_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_normal_(module.weight)

    def extract(self, student_id, exercise_id, q_mask):
        student_ts = self.__student_emb(student_id)
        diff_ts = self.__diff_emb(exercise_id)
        disc_ts = self.__disc_emb(exercise_id)
        knowledge_ts = self.__knowledge_emb.weight
        return student_ts, diff_ts, disc_ts, knowledge_ts, {}

    def __getitem__(self, item):
        if item not in self.__emb_map.keys():
            raise ValueError("We can only detach {} from embeddings.".format(self.__emb_map.keys()))
        self.__emb_map["mastery"] = self.__student_emb.weight
        self.__emb_map["diff"] = self.__diff_emb.weight
        self.__emb_map["disc"] = self.__disc_emb.weight
        self.__emb_map["knowledge"] = self.__knowledge_emb.weight
        return self.__emb_map[item]


# ---------------------------------------------------------------------------
# inscd/interfunc/kscd.py
# ---------------------------------------------------------------------------
class KSCD_IF(_InteractionFunction, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.knowledge_num = config["knowledge_num"]
        self.latent_dim = config["latent_dim"]

        self.prednet_full1 = nn.Linear(
            self.knowledge_num + self.latent_dim, self.knowledge_num, bias=False
        )
        self.drop_1 = nn.Dropout(p=config["dropout"])
        self.prednet_full2 = nn.Linear(
            self.knowledge_num + self.latent_dim, self.knowledge_num, bias=False
        )
        self.drop_2 = nn.Dropout(p=config["dropout"])
        self.prednet_full3 = nn.Linear(1 * self.knowledge_num, 1)

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)

    def compute(self, **kwargs):
        student_ts = kwargs["student_ts"]
        diff_ts = kwargs["diff_ts"]
        q_mask = kwargs["q_mask"]
        knowledge_ts = kwargs["knowledge_ts"]

        stu_ability = torch.mm(student_ts, knowledge_ts.T).sigmoid()
        exer_diff = torch.mm(diff_ts, knowledge_ts.T).sigmoid()
        batch_stu_vector = stu_ability.repeat(1, self.knowledge_num).reshape(
            stu_ability.shape[0], self.knowledge_num, stu_ability.shape[1]
        )
        batch_exer_vector = exer_diff.repeat(1, self.knowledge_num).reshape(
            exer_diff.shape[0], self.knowledge_num, exer_diff.shape[1]
        )

        kn_vector = knowledge_ts.repeat(stu_ability.shape[0], 1).reshape(
            stu_ability.shape[0], self.knowledge_num, self.latent_dim
        )

        # CD
        preference = torch.tanh(self.prednet_full1(torch.cat((batch_stu_vector, kn_vector), dim=2)))
        diff = torch.tanh(self.prednet_full2(torch.cat((batch_exer_vector, kn_vector), dim=2)))
        o = torch.sigmoid(self.prednet_full3(preference - diff))

        sum_out = torch.sum(o * q_mask.unsqueeze(2), dim=1)
        count_of_concept = torch.sum(q_mask, dim=1).unsqueeze(1)
        y_pd = sum_out / count_of_concept
        return y_pd.view(-1)

    def transform(self, mastery, knowledge):
        stu_mastery = torch.mm(mastery, knowledge.T).sigmoid()
        stu_vector = stu_mastery.repeat(1, self.knowledge_num).reshape(
            stu_mastery.shape[0], self.knowledge_num, stu_mastery.shape[1]
        )
        kn_vector = knowledge.repeat(stu_mastery.shape[0], 1).reshape(
            stu_mastery.shape[0], self.knowledge_num, self.latent_dim
        )
        preference = torch.tanh(self.prednet_full1(torch.cat((stu_vector, kn_vector), dim=2)))
        o = torch.sigmoid(self.prednet_full3(preference))
        return o.squeeze(-1)

    def monotonicity(self):
        self.prednet_full1.apply(none_neg_clipper)
        self.prednet_full2.apply(none_neg_clipper)
        self.prednet_full3.apply(none_neg_clipper)


# ---------------------------------------------------------------------------
# inscd/models/neural/kscd.py
# ---------------------------------------------------------------------------
class KSCD(_CognitiveDiagnosisModel):
    def __init__(self, config):
        """
        Description:
        Knowledge-sensed Cognitive Diagnosis Model (KSCD)
        Haiping Ma et al. Knowledge-Sensed Cognitive Diagnosis for Intelligent Education Platforms. CIKM'22.
        """
        super().__init__(config=config)
        self.extractor = Default(config)
        self.inter_func = KSCD_IF(config)

    def diagnose(self):
        if self.inter_func is Ellipsis or self.extractor is Ellipsis:
            raise RuntimeError(
                'Call "build" method to build interaction function before calling this method.'
            )
        return self.inter_func.transform(self.extractor["mastery"], self.extractor["knowledge"])


class _KSCDForward(nn.Module):
    """Thin traceable wrapper: calls extract() to populate embedding lookups (for the
    given batch of student/exercise ids), then calls the real diagnose() entry point --
    matching how inscd_run.py exercises a trained KSCD model."""

    def __init__(self, kscd_model):
        super().__init__()
        self.kscd_model = kscd_model

    def forward(self, sample):
        student_id, exercise_id, q_mask = sample
        self.kscd_model.extractor.extract(student_id, exercise_id, q_mask)
        return self.kscd_model.diagnose()


_STUDENT_NUM = 12
_EXERCISE_NUM = 10
_KNOWLEDGE_NUM = 8
_LATENT_DIM = 8


def build_kscd():
    config = {
        "student_num": _STUDENT_NUM,
        "exercise_num": _EXERCISE_NUM,
        "knowledge_num": _KNOWLEDGE_NUM,
        "latent_dim": _LATENT_DIM,
        "dropout": 0.5,
    }
    model = KSCD(config)
    model.eval()
    return _KSCDForward(model)


def example_input_kscd():
    batch = 4
    student_id = torch.randint(0, _STUDENT_NUM, (batch,))
    exercise_id = torch.randint(0, _EXERCISE_NUM, (batch,))
    q_mask = torch.randint(0, 2, (batch, _KNOWLEDGE_NUM)).float()
    return (student_id, exercise_id, q_mask)


MENAGERIE_ENTRIES = [
    ("KSCD", build_kscd, example_input_kscd, 2022, "MENAGERIE_ZOO"),
]
