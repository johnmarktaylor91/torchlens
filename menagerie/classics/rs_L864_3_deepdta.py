# FAITHFUL PORT of https://github.com/hkmztrk/DeepDTA @ a546a8433a6822e958f36171c4356ad6f414d623
# (original framework: Keras/TensorFlow 1.x)
#
# Transcribed from `source/run_experiments.py::build_combined_categorical`, the primary
# DeepDTA model used for the KIBA/Davis drug-target binding-affinity experiments (the
# `deepmethod` selected in `__main__`). Two independent branches -- SMILES string and
# protein-sequence string, each embedded then passed through 3 stacked 1D convolutions
# with increasing filter counts (NUM_FILTERS, 2x, 3x) and global max-pooled -- are
# concatenated and passed through a 3-layer MLP regression head (1024 -> 1024 -> 512 ->
# 1), with dropout(0.1) after the first two FC layers, exactly mirroring the Keras graph:
#
#   encode_smiles  = Embedding -> Conv1D(NF,fl1) -> Conv1D(2NF,fl1) -> Conv1D(3NF,fl1) -> GlobalMaxPool1D
#   encode_protein = Embedding -> Conv1D(NF,fl2) -> Conv1D(2NF,fl2) -> Conv1D(3NF,fl2) -> GlobalMaxPool1D
#   concat -> Dense(1024,relu) -> Dropout(0.1) -> Dense(1024,relu) -> Dropout(0.1)
#          -> Dense(512,relu) -> Dense(1)
#
# Default hyperparameters (source/go.sh / source/arguments.py):
#   NUM_FILTERS=32, FILTER_LENGTH1=4 (SMILES), FILTER_LENGTH2=8 (protein),
#   max_smi_len=100, max_seq_len=1000, charsmiset_size=64 (isomeric SMILES charset,
#   source/datahelper.py CHARISOSMILEN), charseqset_size=25 (source/datahelper.py
#   CHARPROTLEN). Embedding output_dim=128 for both branches (hardcoded in the Keras
#   source). This staging module keeps those semantics but uses small tensor sizes
#   (short sequences, small charset alphabets, few filters) for a fast trace.
import torch
import torch.nn as nn


class DeepDTA(nn.Module):
    """PyTorch port of DeepDTA's `build_combined_categorical` (Keras) architecture."""

    def __init__(
        self,
        charsmiset_size=64,
        charseqset_size=25,
        max_smi_len=100,
        max_seq_len=1000,
        num_filters=32,
        filter_length1=4,
        filter_length2=8,
        embed_dim=128,
    ):
        super().__init__()
        self.max_smi_len = max_smi_len
        self.max_seq_len = max_seq_len

        # +1 to match Keras `input_dim=charsmiset_size+1` (reserves index 0 for padding)
        self.smiles_embedding = nn.Embedding(charsmiset_size + 1, embed_dim)
        self.smiles_conv1 = nn.Conv1d(embed_dim, num_filters, kernel_size=filter_length1)
        self.smiles_conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size=filter_length1)
        self.smiles_conv3 = nn.Conv1d(num_filters * 2, num_filters * 3, kernel_size=filter_length1)

        self.protein_embedding = nn.Embedding(charseqset_size + 1, embed_dim)
        self.protein_conv1 = nn.Conv1d(embed_dim, num_filters, kernel_size=filter_length2)
        self.protein_conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size=filter_length2)
        self.protein_conv3 = nn.Conv1d(num_filters * 2, num_filters * 3, kernel_size=filter_length2)

        self.relu = nn.ReLU()

        combined_dim = (num_filters * 3) * 2
        self.fc1 = nn.Linear(combined_dim, 1024)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(1024, 1024)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(1024, 512)
        self.predictions = nn.Linear(512, 1)

    def forward(self, smiles_idx, protein_idx):
        # smiles_idx: (B, max_smi_len) long; protein_idx: (B, max_seq_len) long
        smi = self.smiles_embedding(smiles_idx)  # (B, L, E)
        smi = smi.transpose(1, 2)  # (B, E, L) for Conv1d
        smi = self.relu(self.smiles_conv1(smi))
        smi = self.relu(self.smiles_conv2(smi))
        smi = self.relu(self.smiles_conv3(smi))
        smi = torch.amax(smi, dim=2)  # GlobalMaxPooling1D

        prot = self.protein_embedding(protein_idx)
        prot = prot.transpose(1, 2)
        prot = self.relu(self.protein_conv1(prot))
        prot = self.relu(self.protein_conv2(prot))
        prot = self.relu(self.protein_conv3(prot))
        prot = torch.amax(prot, dim=2)

        combined = torch.cat([smi, prot], dim=1)
        x = self.relu(self.fc1(combined))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        return self.predictions(x)


def build_deepdta():
    return DeepDTA(
        charsmiset_size=64,
        charseqset_size=25,
        max_smi_len=24,
        max_seq_len=32,
        num_filters=8,
        filter_length1=4,
        filter_length2=8,
        embed_dim=16,
    ).eval()


def example_input_deepdta():
    torch.manual_seed(0)
    smiles_idx = torch.randint(1, 65, (1, 24))
    protein_idx = torch.randint(1, 26, (1, 32))
    return (smiles_idx, protein_idx)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("DeepDTA", build_deepdta, example_input_deepdta, 2018, "SOURCE_AVAILABLE"),
]
