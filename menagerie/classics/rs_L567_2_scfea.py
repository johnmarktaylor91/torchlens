# SOURCE: vendored from changwn/scFEA @ 4c1fb76d52f07bafad84ce7686ad7c3acfcf0126
#   src/ClassFlux.py
#
# scFEA ("single-cell Flux Estimation Analysis", Alghamdi et al. 2021) predicts, per
# metabolic module, a flux value from that module's input gene expression via a small
# per-module MLP (Linear -> Tanhshrink -> Linear -> Tanhshrink); module fluxes are then
# combined through a fixed stoichiometric matrix (`cmMat`) to reconstruct per-compound
# balance. Real class, verbatim (only the module-level `import sys` used by the
# original CLI entrypoint, not the model, is dropped).
import torch
import torch.nn as nn


class FLUX(nn.Module):
    def __init__(self, matrix, n_modules, f_in=50, f_out=1):
        super(FLUX, self).__init__()
        # gene to flux
        self.inSize = f_in

        self.m_encoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.inSize, 8, bias=False),
                    nn.Tanhshrink(),
                    nn.Linear(8, f_out),
                    nn.Tanhshrink(),
                )
                for i in range(n_modules)
            ]
        )

    def updateC(self, m, n_comps, cmMat):  # stoichiometric matrix
        c = torch.zeros((m.shape[0], n_comps))
        for i in range(c.shape[1]):
            tmp = m * cmMat[i, :]
            c[:, i] = torch.sum(tmp, dim=1)

        return c

    def forward(self, x, n_modules, n_genes, n_comps, cmMat):
        for i in range(n_modules):
            x_block = x[
                :,
                i * n_genes : (i + 1) * n_genes,
            ]
            subnet = self.m_encoder[i]
            if i == 0:
                m = subnet(x_block)
            else:
                m = torch.cat((m, subnet(x_block)), 1)

        c = self.updateC(m, n_comps, cmMat)

        return m, c


def build_scfea():
    n_modules = 6
    n_genes = 5
    return FLUX(matrix=None, n_modules=n_modules, f_in=n_genes, f_out=1)


def example_input_scfea():
    batch_size = 4
    n_modules = 6
    n_genes = 5
    n_comps = 5
    x = torch.rand(batch_size, n_modules * n_genes)
    cmMat = (torch.rand(n_comps, n_modules) > 0.5).float()
    return (x, n_modules, n_genes, n_comps, cmMat)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("scFEA", "build_scfea", "example_input_scfea", 2021, MENAGERIE_ZOO),
]
