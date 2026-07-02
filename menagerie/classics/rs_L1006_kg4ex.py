# SOURCE: vendored from https://github.com/chanllon/KG4EX.Exercise-Recommendation @ main
#   Vendored file: codes/model.py (KGEModel class only; training/eval helper methods
#   dropped since they are not part of the nn.Module forward graph).
#
# KG4EX (CIKM 2023) is a knowledge-graph-based explainable exercise recommendation
# system. Its trainable architecture is a standard knowledge-graph embedding model
# (TransE / RotatE, following Sun et al. 2019's RotatE codebase) applied to a KG of
# (student/exercise/knowledge-concept) entities and relations; KG4EX's contribution is
# the KG construction (mastery level, exercise recommendation score, exercise
# discrimination degree relations) and downstream recommendation logic built on top of
# the learned embeddings, not a change to the KGE architecture itself. We trace the
# 'single' mode of KGEModel.forward, which scores a batch of (head, relation, tail)
# triples via the RotatE scoring function (TransE is also supported by the same class
# via `model_name`).

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


class KGEModel(nn.Module):
    def __init__(
        self,
        model_name,
        nentity,
        nrelation,
        hidden_dim,
        gamma,
        triplere_u,
        double_entity_embedding=False,
        double_relation_embedding=False,
    ):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.u = triplere_u

        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), requires_grad=False
        )

        self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim * 2 if double_relation_embedding else hidden_dim

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item(),
        )

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item(),
        )

        # Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ["TransE", "RotatE"]:
            raise ValueError("model %s not supported" % model_name)

        if model_name == "RotatE" and (not double_entity_embedding or double_relation_embedding):
            raise ValueError("RotatE should use --double_entity_embedding")

    def forward(self, sample, mode="single"):
        """
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        """

        if mode == "single":
            batch_size, negative_sample_size = sample.size(0), 1

            head = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 0]).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding, dim=0, index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 2]).unsqueeze(1)

        elif mode == "head-batch":
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(self.entity_embedding, dim=0, index=head_part.view(-1)).view(
                batch_size, negative_sample_size, -1
            )

            relation = torch.index_select(
                self.relation_embedding, dim=0, index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding, dim=0, index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode == "tail-batch":
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding, dim=0, index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding, dim=0, index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(
                batch_size, negative_sample_size, -1
            )

        else:
            raise ValueError("mode %s not supported" % mode)

        model_func = {
            "TransE": self.TransE,
            "RotatE": self.RotatE,
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError("model %s not supported" % self.model_name)

        return score

    def TransE(self, head, relation, tail, mode):
        if mode == "head-batch":
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == "head-batch":
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=2)
        return score


_NENTITY = 24
_NRELATION = 6
_HIDDEN_DIM = 16


def build_kg4ex():
    model = KGEModel(
        model_name="RotatE",
        nentity=_NENTITY,
        nrelation=_NRELATION,
        hidden_dim=_HIDDEN_DIM,
        gamma=12.0,
        triplere_u=1.0,
        double_entity_embedding=True,
    )
    model.eval()
    return model


def example_input_kg4ex():
    batch = 4
    heads = torch.randint(0, _NENTITY, (batch,))
    relations = torch.randint(0, _NRELATION, (batch,))
    tails = torch.randint(0, _NENTITY, (batch,))
    return torch.stack([heads, relations, tails], dim=1)


MENAGERIE_ENTRIES = [
    ("KG4Ex", build_kg4ex, example_input_kg4ex, 2023, "MENAGERIE_ZOO"),
]
