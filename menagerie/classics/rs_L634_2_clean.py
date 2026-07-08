# SOURCE: vendored from tttianhao/CLEAN @ f2bf2a4f497fa2cc87dac2a1bb314fee587c0a15 (app/src/CLEAN/model.py)
# https://github.com/tttianhao/CLEAN/blob/f2bf2a4f497fa2cc87dac2a1bb314fee587c0a15/app/src/CLEAN/model.py
#
# CLEAN (Contrastive Learning enabled Enzyme ANnotation): predicts EC numbers
# from ESM-1b protein-language-model embeddings via a supervised-contrastive
# projection head. `LayerNormNet` is the architecture used for CLEAN's
# reported training/inference (per the paper and app scripts, e.g.
# `app/CLEAN_infer_fasta.py` / `app/train-supconH.py`): a 3-layer MLP with
# LayerNorm + Dropout + ReLU between a 1280-dim ESM-1b input and a
# configurable output embedding dimension.
#
# No changes to the architecture were made; only the unused `device`/`dtype`
# module-level plumbing was kept as real constructor args (defaulted here to
# CPU/float32 for tracing) since the real class threads them through every
# submodule at construction time.

import torch
import torch.nn as nn


class LayerNormNet(nn.Module):
    def __init__(self, hidden_dim, out_dim, device, dtype, drop_out=0.1):
        super(LayerNormNet, self).__init__()
        self.hidden_dim1 = hidden_dim
        self.out_dim = out_dim
        self.drop_out = drop_out
        self.device = device
        self.dtype = dtype

        self.fc1 = nn.Linear(1280, hidden_dim, dtype=dtype, device=device)
        self.ln1 = nn.LayerNorm(hidden_dim, dtype=dtype, device=device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, dtype=dtype, device=device)
        self.ln2 = nn.LayerNorm(hidden_dim, dtype=dtype, device=device)
        self.fc3 = nn.Linear(hidden_dim, out_dim, dtype=dtype, device=device)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        x = self.dropout(self.ln1(self.fc1(x)))
        x = torch.relu(x)
        x = self.dropout(self.ln2(self.fc2(x)))
        x = torch.relu(x)
        x = self.fc3(x)
        return x


def build_clean():
    torch.manual_seed(0)
    return LayerNormNet(hidden_dim=64, out_dim=32, device="cpu", dtype=torch.float32)


def example_input_clean():
    torch.manual_seed(0)
    return (torch.randn(4, 1280),)


MENAGERIE_ZOO = "vendored-pytorch"

MENAGERIE_ENTRIES = [
    ("CLEAN", build_clean, example_input_clean, 2023, "vendored-pytorch"),
]
