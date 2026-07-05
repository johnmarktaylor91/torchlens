# SOURCE: vendored from Boyiliee/AOD-Net @ HEAD (AOD-Net with PONO/model.py)
# SOURCE: vendored from xiaopengguo/ATKT @ HEAD (model.py)
# SOURCE: vendored from vivinousi/gw-detection-deep-learning @ HEAD (modules/resnet.py)
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

MENAGERIE_ZOO = "vendored-pytorch"


class AODnet(nn.Module):
    """All-in-One Dehazing Network from the official AOD-Net PyTorch source."""

    def __init__(self) -> None:
        """Initialize the official AOD-Net convolution stack."""
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=7, stride=1, padding=3)
        self.conv5 = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.b = 1

    def forward(self, x: Tensor) -> Tensor:
        """Run the official AOD-Net forward pass.

        Parameters
        ----------
        x
            Hazy RGB image tensor.

        Returns
        -------
        Tensor
            Dehazed RGB image tensor.
        """
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        cat1 = torch.cat((x1, x2), 1)
        x3 = F.relu(self.conv3(cat1))
        cat2 = torch.cat((x2, x3), 1)
        x4 = F.relu(self.conv4(cat2))
        cat3 = torch.cat((x1, x2, x3, x4), 1)
        k = F.relu(self.conv5(cat3))

        if k.size() != x.size():
            raise ValueError("k and haze image are different sizes")

        output = k * x - k + self.b
        return F.relu(output)


class KTBackbone(nn.Module):
    """ATKT knowledge-tracing backbone from the official ATKT source."""

    def __init__(
        self,
        skill_dim: int,
        answer_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        """Initialize ATKT embeddings, recurrent layer, and attention head."""
        super().__init__()
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

    def _get_next_pred(self, res: Tensor, skill: Tensor) -> Tensor:
        """Select predictions for the next skill in the input sequence."""
        one_hot = torch.eye(self.output_dim, device=res.device)
        one_hot = torch.cat((one_hot, torch.zeros(1, self.output_dim, device=res.device)), dim=0)
        next_skill = skill[:, 1:]
        one_hot_skill = F.embedding(next_skill, one_hot)

        pred = (res * one_hot_skill).sum(dim=-1)
        return pred

    def attention_module(self, lstm_output: Tensor) -> Tensor:
        """Run ATKT's cumulative attention over LSTM outputs."""
        att_w = self.mlp(lstm_output)
        att_w = torch.tanh(att_w)
        att_w = self.similarity(att_w)

        alphas = nn.Softmax(dim=1)(att_w)

        attn_output = alphas * lstm_output
        attn_output_cum = torch.cumsum(attn_output, dim=1)
        attn_output_cum_1 = attn_output_cum - attn_output

        final_output = torch.cat((attn_output_cum_1, lstm_output), 2)

        return final_output

    def forward(
        self,
        skill: Tensor,
        answer: Tensor,
        perturbation: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run ATKT over skill and answer token sequences."""
        skill_embedding = self.skill_emb(skill)
        answer_embedding = self.answer_emb(answer)

        skill_answer = torch.cat((skill_embedding, answer_embedding), 2)
        answer_skill = torch.cat((answer_embedding, skill_embedding), 2)

        answer_mask = answer.unsqueeze(2).expand_as(skill_answer)

        skill_answer_embedding = torch.where(answer_mask == 1, skill_answer, answer_skill)

        skill_answer_embedding1 = skill_answer_embedding

        if perturbation is not None:
            skill_answer_embedding += perturbation

        out, _ = self.rnn(skill_answer_embedding)
        out = self.attention_module(out)
        res = self.sig(self.fc(out))

        res = res[:, :-1, :]
        pred_res = self._get_next_pred(res, skill)

        return pred_res, skill_answer_embedding1


class ResBlock(nn.Module):
    """Residual block from the official AresGW ResNet module."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """Initialize a one-dimensional residual block."""
        super().__init__()
        if out_channels != in_channels or stride > 1:
            self.x_transform: nn.Module = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
            )
        else:
            self.x_transform = nn.Identity()

        self.body = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run the residual block."""
        x = F.relu(self.body(x) + self.x_transform(x))
        return x


class ResNet54(nn.Module):
    """AresGW 54-layer one-dimensional ResNet classifier."""

    def __init__(self) -> None:
        """Initialize the official ResNet54 feature extractor and head."""
        super().__init__()
        self.feature_extractor = nn.Sequential(
            ResBlock(2, 8),
            ResBlock(8, 8),
            ResBlock(8, 8),
            ResBlock(8, 8),
            ResBlock(8, 16, stride=2),
            ResBlock(16, 16),
            ResBlock(16, 16),
            ResBlock(16, 32, stride=2),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 64, stride=2),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64, stride=2),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64, stride=2),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 16),
            ResBlock(16, 16),
            ResBlock(16, 16),
        )
        self.cls_head = nn.Sequential(
            nn.Conv1d(16, 32, 64),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 2, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run the AresGW ResNet54 classifier."""
        x = self.feature_extractor(x)
        return self.cls_head(x).squeeze(2)


def build_aod_net() -> AODnet:
    """Build a traceable AOD-Net instance."""
    return AODnet()


def example_input_aod_net() -> Tensor:
    """Return an RGB image example for AOD-Net."""
    return torch.randn(1, 3, 16, 16)


def build_atkt() -> KTBackbone:
    """Build a traceable ATKT backbone instance."""
    return KTBackbone(skill_dim=4, answer_dim=4, hidden_dim=8, output_dim=6)


def example_input_atkt() -> tuple[Tensor, Tensor]:
    """Return skill and answer token sequences for ATKT."""
    skill = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)
    answer = torch.tensor([[0, 1, 0, 1, 0, 1]], dtype=torch.long)
    return skill, answer


def build_aresgw_resnet54() -> ResNet54:
    """Build a traceable AresGW ResNet54 instance."""
    model = ResNet54()
    model.eval()
    return model


def example_input_aresgw_resnet54() -> Tensor:
    """Return a two-channel strain series example for AresGW ResNet54."""
    return torch.randn(1, 2, 2048)


MENAGERIE_ENTRIES = [
    ("AOD-Net", build_aod_net, example_input_aod_net, 2017, "CV2a-aod-net"),
    ("ATKT", build_atkt, example_input_atkt, 2020, "CV2a-atkt"),
    ("AresGW ResNet54", build_aresgw_resnet54, example_input_aresgw_resnet54, 2021, "CV2a-aresgw"),
]
