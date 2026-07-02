# FAITHFUL PORT of zachglick/AP-Net @ master (original framework: TensorFlow 2 / Keras
# functional API, `tf.keras.backend.set_floatx('float64')`)
#
# File transcribed: util.py's `make_model(component, nZ=6, ACSF_nmu=43, APSF_nmu=21)`
# (the repo's only model-definition function; `features.py` is pure numpy feature
# engineering, not part of the network).
#
# AP-Net (Glick, Metcalf, Koutsoukas, Spronk, Cheney, Sherrill, "AP-Net: An atomic-pairwise
# neural network for smooth and transferable interaction potentials", J. Chem. Phys. 2020,
# arXiv:2003.03906) predicts intermolecular (dimer) interaction-energy components (SAPT0
# Elst/Exch/Ind/Disp/Total) from atomic-pairwise symmetry-function descriptors. Despite the
# queue's "PyTorch" framework tag, the real repo (`train_sapt_component.py`,
# `import tensorflow as tf`; `util.py`, `import tensorflow as tf`) is TensorFlow/Keras, not
# PyTorch, and TF is not in the installed base-lib set -- so this is a faithful port, not a
# vendor. Every structural choice in `make_model` is preserved verbatim: the two shared
# Dense encoders (`dense_r` for ACSF-derived per-atom features, `dense_i` for
# APSF-derived per-atom-pair features, both `ACSF_nodes=100`/`APSF_nodes=50` ReLU),
# concatenation order (`[ZA, flatten(GA)]` then `[ZA, flatten(GA)] -> dense_r`, etc.), the
# fully-shared 3-hidden-layer (`dense_nodes=128`, ReLU) + linear-readout feed-forward stack
# applied identically to the `AB` and `BA` orderings (same `nn.Linear` weights reused, per
# the real code's layer-object reuse across both directions), A/B-symmetrization via
# addition, and the final `* (1/r)` distance renormalization
# (`output_layer = multiply([add([AB_, BA_]), input_layerR[:, 1]])`). Only the TF/Keras
# functional-graph/session machinery is replaced with an eager `torch.nn.Module`; the
# hand-engineered ACSF/APSF descriptor pipeline (`features.py`) is not needed for a random
# -init structural trace, so the example input below directly synthesizes tensors matching
# the real code's documented per-atom-pair feature shapes
# (`ZA[i]: NA*NB x (NZ+1)`, `GA[i]: NA x NMU1 x NZ`, `IA[i]: NA*NB x NMU2 x NZ`,
# `RAB[i]: NA*NB x 2`) instead of running the numpy ACSF/APSF feature calculator.
#
# MENAGERIE_ZOO = "ported-pytorch"

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class APNet(nn.Module):
    """
    Faithful port of `util.make_model(component, nZ=6, ACSF_nmu=43, APSF_nmu=21)`.

    Real Keras architecture (transcribed verbatim):
      - flatten GA/GB (ACSF, atom-centered symmetry functions) and IA/IB (APSF,
        atom-pair symmetry functions)
      - encode [onehot(Z), flatten(ACSF)] through a SHARED Dense(ACSF_nodes, relu)
        (dense_r reused for both A and B sides)
      - encode [onehot(Z), flatten(APSF)] through a SHARED Dense(APSF_nodes, relu)
        (dense_i reused for both A and B sides)
      - concatenate [onehot(Z), ACSF-encoding, APSF-encoding] per side -> G/I combined
      - build the two directed feature vectors AB_ = [ZA, ZB, R, GA_comb, GB_comb] and
        BA_ = [ZB, ZA, R, GB_comb, GA_comb]
      - push both AB_ and BA_ through the SAME 3-hidden-layer (dense_nodes=128, relu)
        + linear(1) feed-forward stack (weight-shared across the two directions)
      - symmetrize by addition: output = AB_ + BA_
      - renormalize by 1/r: output = output * R[:, 1]
    """

    def __init__(
        self,
        nZ: int = 6,
        ACSF_nmu: int = 43,
        APSF_nmu: int = 21,
        APSF_nodes: int = 50,
        ACSF_nodes: int = 100,
        dense_nodes: int = 128,
    ):
        super().__init__()
        self.nZ = nZ
        self.ACSF_nmu = ACSF_nmu
        self.APSF_nmu = APSF_nmu

        # shared per-side encoders (dense_r / dense_i in the real code)
        self.dense_r = nn.Sequential(nn.Linear(nZ + 1 + ACSF_nmu * nZ, ACSF_nodes), nn.ReLU())
        self.dense_i = nn.Sequential(nn.Linear(nZ + 1 + APSF_nmu * nZ, APSF_nodes), nn.ReLU())

        # shared 3-hidden-layer feed-forward stack + linear readout, reused for AB_ and BA_
        ff_in = (nZ + 1) * 2 + 2 + (ACSF_nodes + APSF_nodes) * 2
        self.dense_1 = nn.Sequential(nn.Linear(ff_in, dense_nodes), nn.ReLU())
        self.dense_2 = nn.Sequential(nn.Linear(dense_nodes, dense_nodes), nn.ReLU())
        self.dense_3 = nn.Sequential(nn.Linear(dense_nodes, dense_nodes), nn.ReLU())
        self.linear = nn.Linear(dense_nodes, 1)

    def _encode_side(self, Z, G, apsf):
        # G: [N, ACSF_nmu, nZ] -> flatten -> concat with Z -> dense_r
        G_flat = torch.flatten(G, start_dim=1)
        G_enc = self.dense_r(torch.cat([Z, G_flat], dim=-1))

        # apsf: [N, APSF_nmu, nZ] -> flatten -> concat with Z -> dense_i
        apsf_flat = torch.flatten(apsf, start_dim=1)
        apsf_enc = self.dense_i(torch.cat([Z, apsf_flat], dim=-1))

        return torch.cat([G_enc, apsf_enc], dim=-1)

    def _feedforward(self, x):
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.linear(x)
        return x

    def forward(self, ZA, ZB, R, GA, GB, IA, IB):
        # side encodings (weight-shared dense_r/dense_i across A and B, as in the real code)
        GA_comb = self._encode_side(ZA, GA, IA)
        GB_comb = self._encode_side(ZB, GB, IB)

        AB_ = torch.cat([ZA, ZB, R, GA_comb, GB_comb], dim=-1)
        BA_ = torch.cat([ZB, ZA, R, GB_comb, GA_comb], dim=-1)

        AB_out = self._feedforward(AB_)
        BA_out = self._feedforward(BA_)

        out = AB_out + BA_out
        out = out * R[:, 1:2]
        return out


def build_apnet():
    torch.manual_seed(0)
    # real repo defaults: nZ=6, ACSF_nmu=43, APSF_nmu=21, ACSF_nodes=100, APSF_nodes=50,
    # dense_nodes=128; shrunk to menagerie-recipe scale
    model = APNet(nZ=6, ACSF_nmu=8, APSF_nmu=4, APSF_nodes=12, ACSF_nodes=16, dense_nodes=24)
    model.eval()
    return model


def example_input_apnet():
    torch.manual_seed(0)
    # NA*NB atom-pair rows (a toy 3-atom x 3-atom dimer pairing = 9 rows), matching the real
    # code's documented per-atom-pair feature shapes:
    #   ZA[i], ZB[i]: NA*NB x (nZ+1)      (one-hot(Z) tiled across the pair axis)
    #   GA[i], GB[i]: NA*NB x ACSF_nmu x nZ (ACSF is atomic, would normally be NA x ..., but
    #       the real code tiles per pair at runtime before flatten -- see get_dataset's
    #       "We won't tile ACSFs ... into atom pairs b/c memory, do it at runtime instead"
    #       comment; this recipe pre-tiles for a single self-contained forward call)
    #   IA[i], IB[i]: NA*NB x APSF_nmu x nZ (APSF is already atom-pair-shaped)
    #   RAB[i]: NA*NB x 2                 (distance, 1/distance)
    n_pairs = 9
    nZ = 6
    ZA = torch.rand(n_pairs, nZ + 1)
    ZB = torch.rand(n_pairs, nZ + 1)
    R = torch.rand(n_pairs, 2) + 0.5
    GA = torch.randn(n_pairs, 8, nZ)
    GB = torch.randn(n_pairs, 8, nZ)
    IA = torch.randn(n_pairs, 4, nZ)
    IB = torch.randn(n_pairs, 4, nZ)
    return (ZA, ZB, R, GA, GB, IA, IB)


MENAGERIE_ENTRIES = [
    (
        "AP-Net (atomic-pairwise interaction potential)",
        "build_apnet",
        "example_input_apnet",
        2020,
        MENAGERIE_ZOO,
    ),
]
