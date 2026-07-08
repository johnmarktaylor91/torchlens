# SOURCE: vendored from zhaoqichang/AttentionDTA_TCBB @ main, model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class mutil_head_attention(nn.Module):
    def __init__(self, head=8, conv=32):
        super(mutil_head_attention, self).__init__()
        self.conv = conv
        self.head = head
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.d_a = nn.Linear(self.conv * 3, self.conv * 3 * head)
        self.p_a = nn.Linear(self.conv * 3, self.conv * 3 * head)
        self.scale = torch.sqrt(torch.FloatTensor([self.conv * 3]))

    def forward(self, drug, protein):
        bsz, d_ef, d_il = drug.shape
        bsz, p_ef, p_il = protein.shape
        drug_att = self.relu(self.d_a(drug.permute(0, 2, 1))).view(bsz, self.head, d_il, d_ef)
        protein_att = self.relu(self.p_a(protein.permute(0, 2, 1))).view(bsz, self.head, p_il, p_ef)
        interaction_map = torch.mean(
            self.tanh(
                torch.matmul(drug_att, protein_att.permute(0, 1, 3, 2)) / self.scale.to(drug.device)
            ),
            1,
        )
        Compound_atte = self.tanh(torch.sum(interaction_map, 2)).unsqueeze(1)
        Protein_atte = self.tanh(torch.sum(interaction_map, 1)).unsqueeze(1)
        drug = drug * Compound_atte
        protein = protein * Protein_atte
        return drug, protein


class AttentionDTA(nn.Module):
    def __init__(
        self,
        protein_MAX_LENGH=1200,
        protein_kernel=[4, 8, 12],
        drug_MAX_LENGH=100,
        drug_kernel=[4, 6, 8],
        conv=32,
        char_dim=128,
        head_num=8,
        dropout_rate=0.1,
    ):
        super(AttentionDTA, self).__init__()
        self.dim = char_dim
        self.conv = conv
        self.dropout_rate = dropout_rate
        self.head_num = head_num
        self.drug_MAX_LENGH = drug_MAX_LENGH
        self.drug_kernel = drug_kernel
        self.protein_MAX_LENGH = protein_MAX_LENGH
        self.protein_kernel = protein_kernel

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
                out_channels=self.conv * 3,
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
                out_channels=self.conv * 3,
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
        self.attention = mutil_head_attention(head=self.head_num, conv=self.conv)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(192, 1024)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(1024, 1024)
        self.dropout2 = nn.Dropout(self.dropout_rate)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1)
        torch.nn.init.constant_(self.out.bias, 5)

    def forward(self, drug, protein):
        drugembed = self.drug_embed(drug)
        proteinembed = self.protein_embed(protein)
        drugembed = drugembed.permute(0, 2, 1)
        proteinembed = proteinembed.permute(0, 2, 1)

        drugConv = self.Drug_CNNs(drugembed)
        proteinConv = self.Protein_CNNs(proteinembed)

        drugConv, proteinConv = self.attention(drugConv, proteinConv)
        drugConv = self.Drug_max_pool(drugConv).squeeze(2)
        proteinConv = self.Protein_max_pool(proteinConv).squeeze(2)
        pair = torch.cat([drugConv, proteinConv], dim=1)
        fully1 = self.leaky_relu(self.fc1(pair))
        fully1 = self.dropout1(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout2(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3)
        return predict


class AttentionDTAWrapper(nn.Module):
    """TorchLens wrapper packing drug and protein token ids into one tensor."""

    def __init__(self):
        """Initialize the wrapped tiny AttentionDTA model."""
        super().__init__()
        self.model = AttentionDTA(
            protein_MAX_LENGH=32,
            protein_kernel=[3, 3, 3],
            drug_MAX_LENGH=24,
            drug_kernel=[3, 3, 3],
            conv=32,
            char_dim=16,
            head_num=2,
            dropout_rate=0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run AttentionDTA from a packed integer token tensor.

        Parameters
        ----------
        x:
            Integer tensor with drug ids in ``x[:, 0, :24]`` and protein ids in
            ``x[:, 1, :32]``.

        Returns
        -------
        torch.Tensor
            Predicted affinity score.
        """

        drug = x[:, 0, :24].long().remainder(65)
        protein = x[:, 1, :32].long().remainder(26)
        return self.model(drug, protein)


def build_attentiondta() -> AttentionDTAWrapper:
    """Build a tiny vendored AttentionDTA model.

    Returns
    -------
    AttentionDTAWrapper
        Traceable AttentionDTA wrapper.
    """

    return AttentionDTAWrapper()


def example_input_attentiondta() -> torch.Tensor:
    """Create an example packed token tensor.

    Returns
    -------
    torch.Tensor
        Packed token tensor.
    """

    return torch.zeros((1, 2, 32), dtype=torch.long)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("AttentionDTA", build_attentiondta, example_input_attentiondta, 2021, "CV2b"),
]
