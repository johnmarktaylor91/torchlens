# FAITHFUL PORT of yuedongyang/SPIDER2 @ master (original framework: raw NumPy /
# hand-written matrix feedforward, weights loaded from .mat/.npz files -- Python 2,
# print-statement syntax; not runnable as-is)
#
# SPIDER2/SPIDER3 predicts protein secondary structure (SS), solvent accessible surface
# area (ASA), and backbone torsion angles (theta/tau/phi/psi) from a windowed PSSM +
# physicochemical-property feature vector, refined over 3 ITERATIONS (the "iterative
# deep learning" of Heffernan et al. 2015, Scientific Reports 5:11476 -- the paper cited
# directly in the SPIDER2 README). Each iteration is itself a plain sigmoid MLP
# ("deep" only in the sense of the original paper's terminology for a handful of hidden
# layers) run three times per iteration (once each for SS / ASA / TTPP-torsion-angle
# heads); the output of iteration k (predicted SS probabilities, predicted ASA, and
# predicted torsion angles re-encoded as sin/cos pairs) is concatenated with the
# original windowed PSSM+phys7 features and fed into iteration k+1.
#
# `misc/pred_pssm.py`'s `nn_feedforward` is the actual real-code inference routine:
# a sigmoid feedforward network,
#     x <- sigmoid([1, x] @ W_i)   for each hidden layer i
#     out <- sigmoid([1, x] @ W_last)
# with weights originally stored as MATLAB-style nested-cell arrays inside a .mat/.npz
# checkpoint (three separate per-iteration weight sets: one each for the SS head, ASA
# head, and torsion ("TTPP") head, run in sequence exactly as `run_iter` does). This
# module transcribes that exact feedforward computation and the exact 3-head /
# 3-iteration control flow faithfully into torch `nn.Module`s (`SpiderMLPHead` for one
# sigmoid feedforward head, `SpiderIterativeNet` for the full 3-iteration / 3-head
# network with the theta/tau/phi/psi sin-cos re-encoding feeding the next iteration,
# matching `pred1()`/`run_iter()` in misc/pred_pssm.py). Random torch init replaces the
# original .mat/.npz-loaded weights (this repo has no trained checkpoint bundled, only a
# tiny toy `dat/pp{1,2,3}.npz`), and the windowing/PSSM/BLOSUM feature construction
# (`window_data`, `read_pssm`, `get_phys7`, `build_pssm`) is out of scope for a traced
# nn.Module (it is pure numpy preprocessing, not part of the network) -- the module
# below takes an already-windowed feature tensor as input, exactly as `nn_feedforward`
# does.

import torch
import torch.nn as nn


class SpiderMLPHead(nn.Module):
    """One iteration's sigmoid feedforward head, matching `nn_feedforward` in
    misc/pred_pssm.py: a stack of `x <- sigmoid([1, x] @ W_i)` layers (bias folded in
    via prepended ones-column, exactly as the real code does with
    `x = numpy.concatenate((b, x), axis=1)`), sigmoid nonlinearity on every layer
    including the last (the real code applies `sigmoid(output)` unconditionally)."""

    def __init__(self, in_dim, hidden_dims, out_dim):
        super().__init__()
        dims = [in_dim] + list(hidden_dims) + [out_dim]
        self.linears = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1], bias=True) for i in range(len(dims) - 1)]
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for linear in self.linears:
            x = self.sigmoid(linear(x))
        return x


class SpiderIterativeNet(nn.Module):
    """Full 3-iteration iterative-refinement network, matching `pred1()` /
    `run_iter()` in misc/pred_pssm.py.

    Each of the 3 iterations runs 3 heads (SS: 3-way secondary-structure softmax-like
    sigmoid output; ASA: 1-way solvent-accessibility; TTPP: 8-way sin/cos-encoded
    torsion angles theta/tau/phi/psi). After iterations 1 and 2, the SS probabilities,
    ASA prediction, and a re-derived sin/cos torsion-angle encoding are concatenated
    onto the base windowed PSSM+phys7 features to form the next iteration's input,
    exactly as `pred1()` builds `input_feature = window_data(pssm, phys, pred_ss_1,
    pred_asa_1, ttpp_input)`. Iteration 3's outputs are the final prediction (the real
    code breaks out of the loop after iteration 3 without building a 4th input).
    """

    N_SS = 3  # C / E / H secondary-structure classes
    N_ASA = 1  # solvent accessible surface area (fraction of max)
    N_TTPP = 8  # sin(theta),sin(tau),sin(phi),sin(psi),cos(theta),cos(tau),cos(phi),cos(psi)

    def __init__(self, base_feature_dim, hidden_dims=(30,)):
        super().__init__()
        self.base_feature_dim = base_feature_dim
        refined_feature_dim = base_feature_dim + self.N_SS + self.N_ASA + self.N_TTPP

        def make_iter_heads(in_dim):
            return nn.ModuleDict(
                {
                    "SS": SpiderMLPHead(in_dim, hidden_dims, self.N_SS),
                    "ASA": SpiderMLPHead(in_dim, hidden_dims, self.N_ASA),
                    "TTPP": SpiderMLPHead(in_dim, hidden_dims, self.N_TTPP),
                }
            )

        self.iter1_heads = make_iter_heads(base_feature_dim)
        self.iter2_heads = make_iter_heads(refined_feature_dim)
        self.iter3_heads = make_iter_heads(refined_feature_dim)

    @staticmethod
    def _build_ttpp_input(pred_ttpp):
        # matches pred_pssm.py's ttpp_input re-encoding: denormalize to [-1, 1], take
        # atan2(sin, cos) pairs for theta/tau/phi/psi, then re-encode as sin/cos halves
        # scaled to [0, 1] (the SS/ASA/TTPP nn.Linear heads output raw sigmoid
        # activations in [0,1], so this directly mirrors run_iter's angle round-trip
        # without materializing degrees, which is immaterial to the traced graph).
        denorm = (pred_ttpp - 0.5) * 2  # -> [-1, 1]
        sin_part, cos_part = denorm[:, :4], denorm[:, 4:]
        theta = torch.atan2(sin_part[:, 0:1], cos_part[:, 0:1])
        tau = torch.atan2(sin_part[:, 1:2], cos_part[:, 1:2])
        phi = torch.atan2(sin_part[:, 2:3], cos_part[:, 2:3])
        psi = torch.atan2(sin_part[:, 3:4], cos_part[:, 3:4])
        angles = torch.cat([theta, tau, phi, psi], dim=1)
        tt_input = torch.sin(angles) / 2 + 0.5
        pp_input = torch.cos(angles) / 2 + 0.5
        return torch.cat([tt_input, pp_input], dim=1)

    def _run_iteration(self, heads, x):
        pred_ss = heads["SS"](x)
        pred_asa = heads["ASA"](x)
        pred_ttpp = heads["TTPP"](x)
        return pred_ss, pred_asa, pred_ttpp

    def forward(self, base_features):
        # Iteration 1: base windowed PSSM+phys7 features only.
        pred_ss_1, pred_asa_1, pred_ttpp_1 = self._run_iteration(self.iter1_heads, base_features)
        ttpp_input_1 = self._build_ttpp_input(pred_ttpp_1)
        refined_2 = torch.cat([base_features, pred_ss_1, pred_asa_1, ttpp_input_1], dim=1)

        # Iteration 2: refined with iteration-1 outputs.
        pred_ss_2, pred_asa_2, pred_ttpp_2 = self._run_iteration(self.iter2_heads, refined_2)
        ttpp_input_2 = self._build_ttpp_input(pred_ttpp_2)
        refined_3 = torch.cat([base_features, pred_ss_2, pred_asa_2, ttpp_input_2], dim=1)

        # Iteration 3: final prediction, no further refinement (matches `if it1 == 3: break`).
        pred_ss_3, pred_asa_3, pred_ttpp_3 = self._run_iteration(self.iter3_heads, refined_3)

        return pred_ss_3, pred_asa_3, pred_ttpp_3


# --- staging harness (tiny sizes; not part of the real repo) ---


def build_spider_iterative_net():
    # Real usage: base features are windowed (winsize=8 -> 17 positions) PSSM (20-dim)
    # + phys7 (7-dim) = 27-dim per position * 17 = 459-dim input vector per residue
    # (see misc/pred_pssm.py `window`/`window_data`, winsize default 8). Kept smaller
    # here (window=2 -> 5 positions * 27 = 135) to keep the trace small/fast while
    # preserving the exact feature-composition math.
    return SpiderIterativeNet(base_feature_dim=135, hidden_dims=(30,))


def example_input_spider_iterative_net():
    n_residues = 20
    return (torch.rand(n_residues, 135),)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    (
        "SPIDER2-IterativeNet",
        "build_spider_iterative_net",
        "example_input_spider_iterative_net",
        2015,
        "ported",
    ),
]
