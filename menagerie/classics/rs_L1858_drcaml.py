# SOURCE: vendored from jamesmullenbach/caml-mimic @ master
#   learn/models.py :: BaseModel, ConvAttnPool
#
# ConvAttnPool IS the CAML model (Convolutional Attention for Multi-Label
# classification); DR-CAML (Description-Regularized CAML) is the SAME class run
# with `lmbda > 0` to enable the extra description-embedding regularization loss
# term (`embed_descriptions` / `_compare_label_embeddings`), confirmed by the
# repo's own `learn/training.py` argparse (`--lmbda`, `desc_embed = args.lmbda > 0`)
# and README ("We provide our pre-trained models for CAML and DR-CAML ... They are
# saved as model.pth"). Mullenbach et al., "Explainable Prediction of Medical Codes
# from Clinical Text" (NAACL 2018).
#
# Only import/API fixes applied (no architectural changes): the original targets
# PyTorch ~0.3 (`Variable`, `torch.nn.init.xavier_uniform` (non-underscore, in-place
# variant), `F.sigmoid`/`F.tanh` (removed in modern torch)). These are replaced with
# their modern equivalents (`torch.xavier_uniform_`, `torch.sigmoid`, `torch.tanh`,
# plain tensors instead of `Variable`). The `gensim`-backed `_code_emb_init` path
# (loading external pretrained code embeddings) and the `desc_data`/lambda
# description-regularization branch are real code paths in the class but are not
# exercised by construction/forward here (no `code_emb`, `desc_data=None`) -- both
# require external non-base resources (gensim KeyedVectors file, label description
# corpus) that are orthogonal to the CNN+attention architecture itself.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

MENAGERIE_ZOO = "vendored-pytorch"


class BaseModel(nn.Module):
    def __init__(self, Y, dicts, lmbda=0, dropout=0.5, embed_size=100):
        super(BaseModel, self).__init__()
        torch.manual_seed(1337)
        self.Y = Y
        self.embed_size = embed_size
        self.embed_drop = nn.Dropout(p=dropout)
        self.lmbda = lmbda

        # make embedding layer (embed_file=None path: randomly initialized, +2 for UNK/PAD)
        vocab_size = len(dicts["ind2w"])
        self.embed = nn.Embedding(vocab_size + 2, embed_size, padding_idx=0)

    def _get_loss(self, yhat, target, diffs=None):
        loss = F.binary_cross_entropy_with_logits(yhat, target)
        if self.lmbda > 0 and diffs is not None:
            diff = torch.stack(diffs).mean()
            loss = loss + diff
        return loss

    def embed_descriptions(self, desc_data, device):
        b_batch = []
        for inst in desc_data:
            if len(inst) > 0:
                lt = torch.LongTensor(inst).to(device)
                d = self.desc_embedding(lt)
                d = d.transpose(1, 2)
                d = self.label_conv(d)
                d = F.max_pool1d(torch.tanh(d), kernel_size=d.size()[2])
                d = d.squeeze(2)
                b_inst = self.label_fc1(d)
                b_batch.append(b_inst)
            else:
                b_batch.append([])
        return b_batch

    def _compare_label_embeddings(self, target, b_batch, desc_data):
        diffs = []
        for i, bi in enumerate(b_batch):
            ti = target[i]
            inds = torch.nonzero(ti.data).squeeze().cpu().numpy()
            zi = self.final.weight[inds, :]
            diff = (zi - bi).mul(zi - bi).mean()
            diffs.append(self.lmbda * diff * bi.size()[0])
        return diffs


class ConvAttnPool(BaseModel):
    """CAML / DR-CAML: per-label convolutional attention pooling over a document,
    per section 2.1-2.3 (and 2.5 for the description-regularization extension) of
    Mullenbach et al. 2018."""

    def __init__(self, Y, kernel_size, num_filter_maps, lmbda, dicts, embed_size=100, dropout=0.5):
        super(ConvAttnPool, self).__init__(Y, dicts, lmbda, dropout=dropout, embed_size=embed_size)

        # convolution over word embeddings, as in 2.1
        self.conv = nn.Conv1d(
            self.embed_size, num_filter_maps, kernel_size=kernel_size, padding=int(kernel_size // 2)
        )
        xavier_uniform_(self.conv.weight)

        # per-label attention context vectors, as in 2.2
        self.U = nn.Linear(num_filter_maps, Y)
        xavier_uniform_(self.U.weight)

        # per-label linear classifiers, as in 2.3
        self.final = nn.Linear(num_filter_maps, Y)
        xavier_uniform_(self.final.weight)

        # description-regularization module (2.5), only wired when lmbda > 0
        if lmbda > 0:
            W = self.embed.weight.data
            self.desc_embedding = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
            self.desc_embedding.weight.data = W.clone()

            self.label_conv = nn.Conv1d(
                self.embed_size,
                num_filter_maps,
                kernel_size=kernel_size,
                padding=int(kernel_size // 2),
            )
            xavier_uniform_(self.label_conv.weight)

            self.label_fc1 = nn.Linear(num_filter_maps, num_filter_maps)
            xavier_uniform_(self.label_fc1.weight)

    def forward(self, x, target, desc_data=None):
        # get embeddings and apply dropout
        x = self.embed(x)
        x = self.embed_drop(x)
        x = x.transpose(1, 2)

        # convolution + nonlinearity (tanh)
        x = torch.tanh(self.conv(x).transpose(1, 2))
        # per-label attention over the sequence
        alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)
        # document representation per label: weighted sum via attention
        m = alpha.matmul(x)
        # final per-label linear classification
        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)

        if desc_data is not None:
            b_batch = self.embed_descriptions(desc_data, x.device)
            diffs = self._compare_label_embeddings(target, b_batch, desc_data)
        else:
            diffs = None

        yhat = y
        loss = self._get_loss(yhat, target, diffs)
        return yhat, loss, alpha


def build_drcaml():
    torch.manual_seed(0)
    vocab_size = 200
    dicts = {"ind2w": {i: f"w{i}" for i in range(vocab_size)}}
    Y = 20  # number of ICD codes
    return ConvAttnPool(
        Y=Y, kernel_size=5, num_filter_maps=32, lmbda=0.1, dicts=dicts, embed_size=64, dropout=0.5
    ).eval()


def example_input_drcaml():
    torch.manual_seed(0)
    batch, seq_len, vocab_size, Y = 2, 60, 200, 20
    x = torch.randint(1, vocab_size + 1, (batch, seq_len))
    target = torch.zeros(batch, Y)
    for b in range(batch):
        idx = torch.randint(0, Y, (3,))
        target[b, idx] = 1.0
    return (x, target)


MENAGERIE_ENTRIES = [
    ("DR-CAML", build_drcaml, example_input_drcaml, 2018, "vendored-pytorch"),
]
