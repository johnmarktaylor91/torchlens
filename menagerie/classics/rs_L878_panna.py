# ruff: noqa: E741  (verbatim upstream uses single-letter `l` for MLP layer loop
# variables -- standard shorthand in the original file, kept as-is)
# SOURCE: vendored from PANNAdevs/panna (GitLab) @ 132a2df662598ecbbb2c9b8d37b992bb6ada6348
# (master), file src/panna/jax/model_torch.py.
#
# PANNA (Properties from Artificial Neural Network Architectures): a per-atom
# feedforward neural-network interatomic potential built on the LATTE local
# many-body descriptor (a generalization of Behler-Parrinello symmetry
# functions with learnable/parametric radial-basis + tensorial angular terms),
# followed by species-specific (or contiguous-species-sliced) MLP energy heads,
# with forces obtained via `torch.autograd.grad` of the total energy w.r.t. the
# pairwise displacement vectors (Lubbers, Smith & Barros lineage of NN
# potentials; PANNA: Lot et al. 2020, "PANNA: Properties from Artificial Neural
# Network Architectures"). The PANNA project's PRIMARY implementation is
# TensorFlow (`src/panna/neuralnet/panna_model.py`); this module vendors the
# real, separate, self-contained PyTorch reimplementation the project itself
# ships (`src/panna/jax/model_torch.py`, "a torch MLP model, basically just for
# evaluation") verbatim -- classes copied as-is, only the `@torch.jit.script`
# decoration is left in place (it is never invoked; the module is used and
# traced in plain eager mode) and the unused `get_scripted_model_torch`
# TorchScript helper is dropped since it is orthogonal to the eager
# architecture and is exactly the kind of scripted-artifact path TorchLens
# does not capture.
#
# The real model is constructed from a `parameters` dict (descriptor/MLP
# hyperparameters) and a `weights` dict (a flat namespace of the real
# checkpoint's parameter tensors, keyed exactly as the real Haiku-style
# checkpoint tree the loader expects -- e.g. `weights['panna_mlp/~/h_kpre']
# ['DV_spw_-']`, `weights['panna_mlp/~/atom__linear']['w']`). No pretrained
# checkpoint ships in the repo (checkpoints are user-trained artifacts); the
# staging build below supplies a tiny 2-species, single pure-radial-term
# ("4,-" in the real `descriptor_shape` DSL from `gvect/gvect_latte.py`)
# configuration with random tensors in exactly that real schema -- the same
# role tiny random-init hyperparameters play for any other from-scratch build
# in this menagerie, not an architectural guess.
#
# MENAGERIE_ZOO = "vendored-pytorch"

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

MENAGERIE_ZOO = "vendored-pytorch"


@torch.jit.script
def RBF(x, xc, sig):
    y = (x - xc) / sig
    t = torch.nn.functional.relu(1 - y**2)
    return t * t * t / (xc * xc)


class LATTE_descr(torch.nn.Module):
    # Annotating here for scripting
    indices: List[List[List[List[int]]]]
    varsig: torch.jit.Final[bool]
    Nterms: torch.jit.Final[int]
    sign: torch.jit.Final[List[str]]
    sizes: torch.jit.Final[List[int]]
    bounds: torch.jit.Final[List[int]]
    intsizes: torch.jit.Final[List[int]]
    ms: torch.jit.Final[List[int]]
    cs: torch.jit.Final[List[int]]
    spec_weights: torch.jit.Final[bool]

    def __init__(self, parameters, weights):
        super().__init__()
        dw = weights["panna_mlp/~/h_kpre"]
        self.varsig = parameters["learnsig"]
        self.spw = []
        self.cent = []
        self.sig = []
        # Total number of terms
        self.Nterms = 0
        # Signature of each term
        self.sign = []
        # Number of elements for each term
        self.sizes = []
        # Cumulant of sizes
        self.bounds = [0]
        # Number of elements in tensor
        self.intsizes = []
        # Body order(-1) for each term
        self.ms = []
        # Max number of indices for each term
        self.cs = []
        # List of indices for each tensor of each term
        self.indices = []
        self.spec_weights = parameters["spec_weights"] == "specific"
        for part in parameters["descriptor_shape"].split(":"):
            self.Nterms += 1
            elem = [e.strip() for e in part.split(",")]
            els = ",".join(elem[1:])
            self.sign.append(els)
            self.spw.append(torch.nn.Parameter(torch.tensor(np.asarray(dw["DV_spw_" + els]))))
            self.cent.append(torch.nn.Parameter(torch.tensor(np.asarray(dw["DV_cent_" + els]))))
            if self.varsig:
                self.sig.append(torch.nn.Parameter(torch.tensor(np.asarray(dw["DV_sig_" + els]))))
            else:
                self.sig.append(torch.nn.Parameter(torch.tensor(np.asarray(parameters["sigma"]))))
            self.sizes.append(int(elem[0]))
            self.bounds.append(self.bounds[-1] + int(elem[0]))
            self.ms.append(len(elem) - 1)
            inds = list(set([i for ii in elem[1:] for i in ii]))
            self.cs.append(len(inds))
            if els == "-":
                self.intsizes.append(self.sizes[-1])
            else:
                self.intsizes.append(self.sizes[-1] * self.ms[-1] * (3 ** self.cs[-1]))
            eeinds = []
            # Loop over bodies
            for j, ee in enumerate(elem[1:]):
                einds = []
                # Loop over indices of that body
                for e in ee:
                    ti = inds.index(e)
                    ai = [1] * self.cs[-1]
                    ai[ti] = 3
                    # List up to max c, with 3 only for the index of this term
                    einds.append(ai)
                eeinds.append(einds)
            self.indices.append(eeinds)

    def forward(self, vij: torch.Tensor, si: torch.Tensor, sj: torch.Tensor) -> List[torch.Tensor]:
        npairs = vij.shape[0]
        rij = torch.sqrt(torch.sum(vij**2, dim=1))  # [Npairs]
        vers_ij = vij / (rij[:, None] + 1e-20)  # [Npairs,3]
        dterms = []
        for i in range(self.Nterms):
            if self.spec_weights:
                s_spw = self.spw[i][si, sj]  # [Npairs,n*m]
                s_cent = self.cent[i][si]  # [Npairs,n*m]
            else:
                s_spw = self.spw[i][sj]  # [Npairs,n*m]
                s_cent = torch.unsqueeze(self.cent[i], 0)  # [1,n*m]
            if self.varsig:
                if self.spec_weights:
                    s_sig = self.sig[i][si]  # [Npairs,n*m]
                else:
                    s_sig = torch.unsqueeze(self.sig[i], 0)  # [1,n*m]
            else:
                s_sig = self.sig[i]  # [1]
            rbf_term = s_spw * RBF(torch.unsqueeze(rij, 1), s_cent, s_sig)  # [Npairs,n]
            # Special case for '-'
            if self.sign[i] == "-":
                dterms.append(rbf_term)  # [Npairs,n]
            else:
                # Computing the tensorial part for each term
                t = []
                for ee in self.indices[i]:
                    # Expanding explicitly to avoid broadcasting errors
                    v = (
                        vers_ij.view([npairs] + ee[0]).expand([npairs] + [3] * self.cs[i]).clone()
                    )  # [Npairs,3^c]
                    if len(ee) > 1:
                        for e in ee[1:]:
                            v *= vers_ij.view([npairs] + e)  # [Npairs,3^c]
                    t.append(v)
                tens = torch.unsqueeze(torch.stack(t, dim=1), 2)  # [Npairs,m,1,3^c]
                dterms.append(
                    (rbf_term.view([-1, self.ms[i], self.sizes[i]] + [1] * self.cs[i]) * tens).view(
                        npairs, self.intsizes[i]
                    )
                )  # [Npairs,m*n*3^c]
        return dterms

    @torch.jit.export
    def postp(self, terms: List[torch.Tensor]) -> torch.Tensor:
        nats = terms[0].shape[0]
        descrL = []
        for i in range(self.Nterms):
            descrL.append(
                torch.sum(
                    torch.prod(terms[i].view(nats, self.ms[i], self.sizes[i], -1), dim=1), dim=-1
                )
            )  # [Nats,n]
        descr = torch.cat(descrL, dim=1)
        return descr


class sp_lay(torch.nn.Module):
    act: torch.jit.Final[str]
    sp_weights: torch.jit.Final[str]

    def __init__(self, w, b, act, sp_weights):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(np.asarray(w)))
        self.b = torch.nn.Parameter(torch.tensor(np.asarray(b)))
        self.act = act
        self.sp_weights = sp_weights

    def forward(self, x: torch.Tensor, si: torch.Tensor) -> torch.Tensor:
        if self.sp_weights == "s":
            out = torch.matmul(torch.index_select(self.w, 0, si), x) + torch.unsqueeze(
                self.b[si], -1
            )
        else:
            out = torch.matmul(self.w, x) + torch.unsqueeze(self.b, -1)
        if self.act == "exp":
            out = torch.exp(-out * out)
        return out


class ssp_lay(torch.nn.Module):
    act: torch.jit.Final[str]

    def __init__(self, w, b, act, s):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(np.asarray(w[s]).transpose().copy()))
        self.b = torch.nn.Parameter(torch.tensor(np.asarray(b[s])[None, :]))
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.matmul(x, self.w) + self.b
        if self.act == "exp":
            out = torch.exp(-out * out)
        return out


class snet(torch.nn.Module):
    def __init__(self, parameters, weights, sp_weights, s):
        super().__init__()
        self.Nlayers = len(parameters["mlp_arch"])
        layerlist = []
        for l in range(self.Nlayers):
            if sp_weights[l] == "s":
                act = parameters["mlp_act"] if l < self.Nlayers - 1 else "lin"
                lname = "" if l == 0 else "_" + str(l)
                w = weights["panna_mlp/~/atom__linear" + lname]["w"]
                b = weights["panna_mlp/~/atom__linear" + lname]["b"]
                if l == self.Nlayers - 1:
                    b = b + np.asarray(parameters["offsets"])[:, np.newaxis]
                layerlist.append(ssp_lay(w, b, act, s))
        self.layers = torch.nn.ModuleList(layerlist)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for l in self.layers:
            x = l(x)
        return x


class MLP_torch(torch.nn.Module):
    """A torch MLP model, basically just for evaluation (real upstream class)."""

    Nspecies: torch.jit.Final[int]
    Nlayers: torch.jit.Final[int]
    cont_sp: torch.jit.Final[bool]
    sp_weights: torch.jit.Final[List[str]]

    def __init__(self, parameters, weights):
        super().__init__()
        self.descr = LATTE_descr(parameters, weights)
        self.dsize = self.descr.bounds[-1]
        self.arch = [self.dsize] + parameters["mlp_arch"]
        self.Nlayers = len(parameters["mlp_arch"])
        self.Nspecies = len(parameters["species"])
        self.sp_weights = parameters["mlp_sp_weights"]
        if self.sp_weights:
            self.sp_weights = [s.strip() for s in self.sp_weights.split(":")]
        else:
            self.sp_weights = ["s"] * len(self.arch)
        # Flag to expect atoms of the same species as contiguous
        self.cont_sp = parameters["extf"] == "cont_sp" or parameters["extf"] == "std"
        # Handle common layers or specific without cont_sp
        layerlist = []
        for l in range(self.Nlayers):
            if self.sp_weights[l] == "c" or not self.cont_sp:
                act = parameters["mlp_act"] if l < self.Nlayers - 1 else "lin"
                lname = "" if l == 0 else "_" + str(l)
                w = weights["panna_mlp/~/atom__linear" + lname]["w"]
                b = weights["panna_mlp/~/atom__linear" + lname]["b"]
                if l == self.Nlayers - 1:
                    b = b + np.asarray(parameters["offsets"])[:, np.newaxis]
                layerlist.append(sp_lay(w, b, act, self.sp_weights[l]))
        self.layers = torch.nn.ModuleList(layerlist)
        # Handle species specific layers in the cont_sp case
        if self.cont_sp and ("s" in self.sp_weights):
            self.slayers = torch.nn.ModuleList(
                [snet(parameters, weights, self.sp_weights, s) for s in range(self.Nspecies)]
            )
        else:
            self.slayers = torch.nn.ModuleList([])

    def forward(self, batch: Dict[str, torch.Tensor]):
        nats = batch["species"].shape[0]
        device = batch["vec_ij"].device
        vij = batch["vec_ij"]
        vij.requires_grad_()
        d_pairs = self.descr(vij, batch["type_i"], batch["type_j"])

        x = self.descr.postp(
            [
                torch.jit.wait(
                    torch.jit.fork(
                        torch.zeros((nats, self.descr.intsizes[i]), device=device).scatter_add,
                        0,
                        batch["ind_i"][:, None].expand(-1, self.descr.intsizes[i]),
                        dt,
                    )
                )
                for i, dt in enumerate(d_pairs)
            ]
        )

        # Normal layers
        if len(self.layers) > 0:
            x = torch.unsqueeze(x, -1)
            spT = batch["spsh"] if self.cont_sp else batch["species"]
            for l in self.layers:
                x = l(x, spT)
            x = x[:, :, 0]
        # Layers handling a slice of contiguous atoms of one species
        if len(self.slayers) > 0:
            spsh = batch["spsh"].cpu()
            xl = []
            for s, l in enumerate(self.slayers):
                if spsh[s + 1] > spsh[s]:
                    xl.append(l(x[spsh[s] : spsh[s + 1]]))
            x = torch.cat(xl, dim=0)
        F = torch.autograd.grad(
            [
                torch.sum(x),
            ],
            [
                vij,
            ],
            allow_unused=True,
        )[0]
        return {"E": x.flatten(), "F": F}


def make_model_torch(parameters, weights):
    if parameters["model_type"] == "MLP":
        model = MLP_torch(parameters, weights)
    return model


# ------------------------------------------------------------------
# Menagerie staging entrypoints
# ------------------------------------------------------------------
# Tiny 2-species, single pure-radial-term ("4,-") configuration. The
# `descriptor_shape` DSL (`N,idx1,idx2,...:...`) and the `weights` dict's flat
# checkpoint-tensor namespace both come straight from the real loader code in
# `LATTE_descr.__init__`/`gvect/gvect_latte.py`; only the numeric VALUES
# (sizes, random tensor contents) are synthesized -- exactly analogous to the
# tiny random-init hyperparameters every other from-scratch build in this
# menagerie uses, not an architectural change.
_N_SPECIES = 2
_DESCRIPTOR_SIZE = 4
_PARAMETERS = {
    "learnsig": False,
    "spec_weights": "common",
    "descriptor_shape": f"{_DESCRIPTOR_SIZE},-",
    "sigma": np.float32(0.5),
    "mlp_arch": [8, 1],
    "mlp_act": "exp",
    "mlp_sp_weights": "s:s",
    "extf": "std",
    "offsets": [0.0, 0.0],
    "species": ["H", "O"],
    "model_type": "MLP",
}


def _build_weights():
    rng = np.random.default_rng(0)
    return {
        "panna_mlp/~/h_kpre": {
            "DV_spw_-": rng.normal(size=(_N_SPECIES, _DESCRIPTOR_SIZE)).astype(np.float32),
            "DV_cent_-": rng.normal(size=(_DESCRIPTOR_SIZE,)).astype(np.float32),
        },
        "panna_mlp/~/atom__linear": {
            "w": rng.normal(size=(_N_SPECIES, 8, _DESCRIPTOR_SIZE)).astype(np.float32),
            "b": rng.normal(size=(_N_SPECIES, 8)).astype(np.float32),
        },
        "panna_mlp/~/atom__linear_1": {
            "w": rng.normal(size=(_N_SPECIES, 1, 8)).astype(np.float32),
            "b": rng.normal(size=(_N_SPECIES, 1)).astype(np.float32),
        },
    }


def build_panna():
    torch.manual_seed(0)
    return make_model_torch(_PARAMETERS, _build_weights())


def example_input_panna():
    # 3-atom synthetic molecule (H, O, H), fully-connected pair list (6 directed
    # i->j pairs), matching the `batch` dict schema `MLP_torch.forward` expects.
    rng = np.random.default_rng(0)
    species = torch.tensor([0, 1, 0], dtype=torch.long)
    ind_i = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    ind_j = torch.tensor([1, 2, 0, 2, 0, 1], dtype=torch.long)
    type_i = species[ind_i]
    type_j = species[ind_j]
    vec_ij = torch.tensor(rng.normal(scale=1.5, size=(6, 3)), dtype=torch.float32)
    # species-contiguous shape boundaries: 2 atoms of species 0 then 1 of species 1
    spsh = torch.tensor([0, 2, 3], dtype=torch.long)

    batch = {
        "species": species,
        "vec_ij": vec_ij,
        "type_i": type_i,
        "type_j": type_j,
        "ind_i": ind_i,
        "spsh": spsh,
    }
    return (batch,)


MENAGERIE_ENTRIES = [
    ("panna", build_panna, example_input_panna, 2020, MENAGERIE_ZOO),
]
