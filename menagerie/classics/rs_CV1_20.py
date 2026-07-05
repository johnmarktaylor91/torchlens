# SOURCE: vendored from xiaopengguo/ATKT @ main (model.py)
# coding: utf-8
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class KT_backbone(nn.Module):
    def __init__(self, skill_dim: int, answer_dim: int, hidden_dim: int, output_dim: int) -> None:
        """Initialize the ATKT knowledge-tracing backbone."""
        super(KT_backbone, self).__init__()
        self.skill_dim = skill_dim
        self.answer_dim = answer_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rnn = nn.LSTM(self.skill_dim + self.answer_dim, self.hidden_dim, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim * 2, self.output_dim)
        self.sig = nn.Sigmoid()

        self.skill_emb = nn.Embedding(self.output_dim + 1, self.skill_dim)
        self.skill_emb.weight.data[-1] = 0

        self.answer_emb = nn.Embedding(2 + 1, self.answer_dim)
        self.answer_emb.weight.data[-1] = 0

        self.attention_dim = 80
        self.mlp = nn.Linear(self.hidden_dim, self.attention_dim)
        self.similarity = nn.Linear(self.attention_dim, 1, bias=False)

    def _get_next_pred(self, res: torch.Tensor, skill: torch.Tensor) -> torch.Tensor:
        """Select the next-skill prediction from all skill logits."""

        one_hot = torch.eye(self.output_dim, device=res.device)
        one_hot = torch.cat((one_hot, torch.zeros(1, self.output_dim, device=res.device)), dim=0)
        next_skill = skill[:, 1:]
        one_hot_skill = F.embedding(next_skill, one_hot)

        pred = (res * one_hot_skill).sum(dim=-1)
        return pred

    def attention_module(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """Apply the cumulative attention module over LSTM outputs."""

        att_w = self.mlp(lstm_output)
        att_w = torch.tanh(att_w)
        att_w = self.similarity(att_w)

        alphas = nn.Softmax(dim=1)(att_w)

        attn_ouput = alphas * lstm_output
        attn_output_cum = torch.cumsum(attn_ouput, dim=1)
        attn_output_cum_1 = attn_output_cum - attn_ouput

        final_output = torch.cat((attn_output_cum_1, lstm_output), 2)

        return final_output

    def forward(
        self,
        skill: torch.Tensor,
        answer: torch.Tensor,
        perturbation: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the ATKT backbone on skill and answer sequences."""

        skill_embedding = self.skill_emb(skill)
        answer_embedding = self.answer_emb(answer)

        skill_answer = torch.cat((skill_embedding, answer_embedding), 2)
        answer_skill = torch.cat((answer_embedding, skill_embedding), 2)

        answer = answer.unsqueeze(2).expand_as(skill_answer)

        skill_answer_embedding = torch.where(answer == 1, skill_answer, answer_skill)

        skill_answer_embedding1 = skill_answer_embedding

        if perturbation is not None:
            skill_answer_embedding += perturbation

        out, _ = self.rnn(skill_answer_embedding)
        out = self.attention_module(out)
        res = self.sig(self.fc(out))

        res = res[:, :-1, :]
        pred_res = self._get_next_pred(res, skill)

        return pred_res, skill_answer_embedding1


def build_atkt() -> KT_backbone:
    """Build a tiny ATKT model for TorchLens tracing."""
    model = KT_backbone(skill_dim=8, answer_dim=8, hidden_dim=16, output_dim=12)
    model.eval()
    return model


def example_input_atkt() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a small skill/answer sequence pair."""
    skill = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    answer = torch.tensor([[1, 0, 1, 1, 0, 1]], dtype=torch.long)
    return skill, answer


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "Adversarial Training Knowledge Tracing (ATKT)",
        "build_atkt",
        "example_input_atkt",
        2020,
        "CV1",
    )
]
