# SOURCE: vendored from aaberdam/AdaLISTA @ master (models.py)
#
# Ada-LISTA ("Ada-LISTA: Learned Solvers Adaptive to Varying Models", Aberdam, Golts &
# Elad, ICLR 2021). The `Adaptive_ISTA` class below is copied verbatim from the official
# repo's models.py (only the `scipy.linalg.eigvalsh` import path and stray unused
# `reinit`/`.cuda()` device dispatch are kept as-is -- no architectural change). It unrolls
# T ISTA iterations with per-iteration learned step sizes (`etas`) and momentum-like scale
# factors (`gammas`) applied on top of an input-adaptive dictionary `D`, using two learned
# weight matrices `W1`/`W2` that condition the linear operators `_A`/`_B` on the dictionary
# passed at call time (the "adaptive" part of Ada-LISTA, distinguishing it from plain LISTA).
#
# MENAGERIE_ZOO = "vendored-pytorch"

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


##################################################

# # #            Adaptive-LISTA              # # #

##################################################


class Adaptive_ISTA(nn.Module):
    def __init__(self, n, m, D=None, T=6, lambd=1.0):
        super(Adaptive_ISTA, self).__init__()
        self.n, self.m = n, m
        self.D = D
        self.T = T  # ISTA Iterations
        self.lambd = lambd  # Lagrangian Multiplier
        self.W1 = nn.Linear(n, n, bias=False)  # Weight Matrix
        self.W2 = nn.Linear(n, n, bias=False)  # Weight Matrix
        self.W1.weight.data = torch.eye(n)
        self.W2.weight.data = torch.eye(n)
        # ISTA Stepsizes
        self.etas = nn.Parameter(torch.ones(T + 1, 1, 1, 1), requires_grad=True)
        self.gammas = nn.Parameter(torch.ones(T + 1, 1, 1, 1), requires_grad=True)
        # Initialization
        if D is not None:
            L = 5  # float(eigvalsh(D.T @ D, eigvals=(m - 1, m - 1)))
            self.etas.data *= 1 / L
            self.gammas.data *= 1 / L
        self.reinit_num = 0  # Number of re-initializations

    def _A(self, D, i):
        A_tmp = self.W1.weight @ D
        return self.gammas[i, :, :, :] * A_tmp.transpose(1, 2)

    def _B(self, D, i):
        B_tmp = self.W2.weight @ D
        return self.gammas[i, :, :, :] * B_tmp.transpose(1, 2) @ B_tmp

    def _shrink(self, x, eta):
        return eta * F.softshrink(x / eta, lambd=self.lambd)

    def forward(self, y, D):
        y = y.unsqueeze(2)
        x = torch.zeros(y.shape[0], self.m, y.shape[2])
        if y.is_cuda:
            x = x.cuda()
        for i in range(0, self.T + 1):
            x = self._shrink(x - self._B(D, i) @ x + self._A(D, i) @ y, self.etas[i, :, :, :])
        return x.squeeze()

    def reinit(self):
        reinit_num = self.reinit_num + 1
        self.__init__(n=self.n, m=self.m, D=self.D, T=self.T, lambd=self.lambd)
        self.reinit_num = reinit_num


def build_adalista():
    # Tiny signal/atom dims (n=8 measurements, m=12 dictionary atoms) and T=3 unfoldings for
    # fast tracing; same constructor knobs the original repo exposes (n_dict/m_dict/T in
    # params.py just pick larger scenario-specific values).
    model = Adaptive_ISTA(n=8, m=12, D=None, T=3, lambd=1.0)
    model.eval()
    return model


def example_input_adalista():
    # y: batch of measurement vectors (batch, n). D: batch of per-sample dictionaries
    # (batch, n, m), matching the real repo's `model(y, D)` call convention in eval.py/main.py.
    return (torch.randn(2, 8), torch.randn(2, 8, 12))


MENAGERIE_ENTRIES = [
    ("Ada-LISTA", "build_adalista", "example_input_adalista", 2021, "CODE"),
]
