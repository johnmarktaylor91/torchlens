# SOURCE: vendored from brennanaba/ImmuneBuilder @ main
#   ImmuneBuilder/models.py     (InvariantPointAttention, BackboneUpdate, TorsionAngles,
#                                 StructureUpdate, StructureModule)
#   ImmuneBuilder/rigids.py     (Vector, Rot, Rigid + frame/torsion geometry helpers)
#   ImmuneBuilder/constants.py  (residue geometry constant tables)
#   ImmuneBuilder/util.py       (get_one_hot, get_encoding -- input featurization)
"""ABodyBuilder2: end-to-end antibody Fv structure prediction (Abanades,
Wong, Boyles, Georges, Bujotzek & Deane, Bioinformatics 2024; part of the
``ImmuneBuilder`` suite alongside NanoBodyBuilder2/TCRBuilder2, which share
this exact ``StructureModule``).

The real network is an AlphaFold-Multimer-style Invariant Point Attention
(IPA) structure module: per-residue one-hot + chain-region node features are
embedded, then refined over ``n_layers`` of IPA + backbone-update + torsion
prediction blocks operating on rigid reference frames (no MSA/pairformer --
ImmuneBuilder skips co-evolutionary features entirely and predicts structure
directly from sequence + per-model learned parameters), finally expanded into
full-atom coordinates via the chi-angle rotamer geometry tables.

This is the real, unmodified ``ImmuneBuilder.models.StructureModule`` (and
its ``rigids.py``/``constants.py`` support code) vendored verbatim -- the
published ``ABodyBuilder2`` class only constructs
``StructureModule(rel_pos_dim=64, embed_dim=...)`` and loads Zenodo-hosted
weights into it (real repo: ``model = StructureModule(rel_pos_dim=64,
embed_dim=embed_dim[model_file])``); the featurization (``get_encoding``) and
rigid-body/torsion-angle geometry (``rigids.py``, keyed on
``constants.py`` atom-position tables) are vendored unchanged as well, since
``StructureModule.forward`` calls directly into them.
"""

from __future__ import annotations

import numpy as np
import torch
from einops import rearrange

# ---------------------------------------------------------------------------
# ImmuneBuilder/constants.py (verbatim: residue geometry constant tables)
# ---------------------------------------------------------------------------

restypes = "ARNDCQEGHILKMFPSTWYV"

residue_atoms = {
    "A": ["CA", "N", "C", "CB", "O"],
    "C": ["CA", "N", "C", "CB", "O", "SG"],
    "D": ["CA", "N", "C", "CB", "O", "CG", "OD1", "OD2"],
    "E": ["CA", "N", "C", "CB", "O", "CG", "CD", "OE1", "OE2"],
    "F": ["CA", "N", "C", "CB", "O", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "G": [
        "CA",
        "N",
        "C",
        "CA",
        "O",
    ],  # G has no CB so I am padding it with CA so the Os are aligned
    "H": ["CA", "N", "C", "CB", "O", "CG", "CD2", "CE1", "ND1", "NE2"],
    "I": ["CA", "N", "C", "CB", "O", "CG1", "CG2", "CD1"],
    "K": ["CA", "N", "C", "CB", "O", "CG", "CD", "CE", "NZ"],
    "L": ["CA", "N", "C", "CB", "O", "CG", "CD1", "CD2"],
    "M": ["CA", "N", "C", "CB", "O", "CG", "CE", "SD"],
    "N": ["CA", "N", "C", "CB", "O", "CG", "ND2", "OD1"],
    "P": ["CA", "N", "C", "CB", "O", "CG", "CD"],
    "Q": ["CA", "N", "C", "CB", "O", "CG", "CD", "NE2", "OE1"],
    "R": ["CA", "N", "C", "CB", "O", "CG", "CD", "CZ", "NE", "NH1", "NH2"],
    "S": ["CA", "N", "C", "CB", "O", "OG"],
    "T": ["CA", "N", "C", "CB", "O", "CG2", "OG1"],
    "V": ["CA", "N", "C", "CB", "O", "CG1", "CG2"],
    "W": ["CA", "N", "C", "CB", "O", "CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2", "NE1"],
    "Y": ["CA", "N", "C", "CB", "O", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
}

residue_atoms_mask = {
    res: len(residue_atoms[res]) * [True] + (14 - len(residue_atoms[res])) * [False]
    for res in residue_atoms
}

# Position of atoms in each ref frame
rigid_group_atom_positions2 = {
    "A": {
        "C": [0, (1.526, -0.0, -0.0)],
        "CA": [0, (0.0, 0.0, 0.0)],
        "CB": [0, (-0.529, -0.774, -1.205)],
        "N": [0, (-0.525, 1.363, 0.0)],
        "O": [3, (-0.627, 1.062, 0.0)],
    },
    "C": {
        "C": [0, (1.524, 0.0, 0.0)],
        "CA": [0, (0.0, 0.0, 0.0)],
        "CB": [0, (-0.519, -0.773, -1.212)],
        "N": [0, (-0.522, 1.362, -0.0)],
        "O": [3, (-0.625, 1.062, -0.0)],
        "SG": [4, (-0.728, 1.653, 0.0)],
    },
    "D": {
        "C": [0, (1.527, 0.0, -0.0)],
        "CA": [0, (0.0, 0.0, 0.0)],
        "CB": [0, (-0.526, -0.778, -1.208)],
        "CG": [4, (-0.593, 1.398, -0.0)],
        "N": [0, (-0.525, 1.362, -0.0)],
        "O": [3, (-0.626, 1.062, -0.0)],
        "OD1": [5, (-0.61, 1.091, 0.0)],
        "OD2": [5, (-0.592, -1.101, 0.003)],
    },
    "E": {
        "C": [0, (1.526, -0.0, -0.0)],
        "CA": [0, (0.0, 0.0, 0.0)],
        "CB": [0, (-0.526, -0.781, -1.207)],
        "CD": [5, (-0.6, 1.397, 0.0)],
        "CG": [4, (-0.615, 1.392, 0.0)],
        "N": [0, (-0.528, 1.361, 0.0)],
        "O": [3, (-0.626, 1.062, 0.0)],
        "OE1": [6, (-0.607, 1.095, -0.0)],
        "OE2": [6, (-0.589, -1.104, 0.001)],
    },
    "F": {
        "C": [0, (1.524, 0.0, -0.0)],
        "CA": [0, (0.0, 0.0, 0.0)],
        "CB": [0, (-0.525, -0.776, -1.212)],
        "CD1": [5, (-0.709, 1.195, -0.0)],
        "CD2": [5, (-0.706, -1.196, 0.0)],
        "CE1": [5, (-2.102, 1.198, -0.0)],
        "CE2": [5, (-2.098, -1.201, -0.0)],
        "CG": [4, (-0.607, 1.377, 0.0)],
        "CZ": [5, (-2.794, -0.003, 0.001)],
        "N": [0, (-0.518, 1.363, 0.0)],
        "O": [3, (-0.626, 1.062, -0.0)],
    },
    "G": {
        "C": [0, (1.517, -0.0, -0.0)],
        "CA": [0, (0.0, 0.0, 0.0)],
        "N": [0, (-0.572, 1.337, 0.0)],
        "O": [3, (-0.626, 1.062, -0.0)],
    },
    "H": {
        "C": [0, (1.525, 0.0, 0.0)],
        "CA": [0, (0.0, 0.0, 0.0)],
        "CB": [0, (-0.525, -0.778, -1.208)],
        "CD2": [5, (-0.889, -1.021, -0.003)],
        "CE1": [5, (-2.03, 0.851, -0.002)],
        "CG": [4, (-0.6, 1.37, -0.0)],
        "N": [0, (-0.527, 1.36, 0.0)],
        "ND1": [5, (-0.744, 1.16, -0.0)],
        "NE2": [5, (-2.145, -0.466, -0.004)],
        "O": [3, (-0.625, 1.063, 0.0)],
    },
    "I": {
        "C": [0, (1.527, -0.0, -0.0)],
        "CA": [0, (0.0, 0.0, 0.0)],
        "CB": [0, (-0.536, -0.793, -1.213)],
        "CD1": [5, (-0.619, 1.391, 0.0)],
        "CG1": [4, (-0.534, 1.437, -0.0)],
        "CG2": [4, (-0.54, -0.785, 1.199)],
        "N": [0, (-0.493, 1.373, -0.0)],
        "O": [3, (-0.627, 1.062, -0.0)],
    },
    "K": {
        "C": [0, (1.526, 0.0, 0.0)],
        "CA": [0, (0.0, 0.0, 0.0)],
        "CB": [0, (-0.524, -0.778, -1.208)],
        "CD": [5, (-0.559, 1.417, 0.0)],
        "CE": [6, (-0.56, 1.416, 0.0)],
        "CG": [4, (-0.619, 1.39, 0.0)],
        "N": [0, (-0.526, 1.362, -0.0)],
        "NZ": [7, (-0.554, 1.387, 0.0)],
        "O": [3, (-0.626, 1.062, -0.0)],
    },
    "L": {
        "C": [0, (1.525, -0.0, -0.0)],
        "CA": [0, (0.0, 0.0, 0.0)],
        "CB": [0, (-0.522, -0.773, -1.214)],
        "CD1": [5, (-0.53, 1.43, -0.0)],
        "CD2": [5, (-0.535, -0.774, -1.2)],
        "CG": [4, (-0.678, 1.371, 0.0)],
        "N": [0, (-0.52, 1.363, 0.0)],
        "O": [3, (-0.625, 1.063, -0.0)],
    },
    "M": {
        "C": [0, (1.525, 0.0, 0.0)],
        "CA": [0, (0.0, 0.0, 0.0)],
        "CB": [0, (-0.523, -0.776, -1.21)],
        "CE": [6, (-0.32, 1.786, -0.0)],
        "CG": [4, (-0.613, 1.391, -0.0)],
        "N": [0, (-0.521, 1.364, -0.0)],
        "O": [3, (-0.625, 1.062, -0.0)],
        "SD": [5, (-0.703, 1.695, 0.0)],
    },
    "N": {
        "C": [0, (1.526, -0.0, -0.0)],
        "CA": [0, (0.0, 0.0, 0.0)],
        "CB": [0, (-0.531, -0.787, -1.2)],
        "CG": [4, (-0.584, 1.399, 0.0)],
        "N": [0, (-0.536, 1.357, 0.0)],
        "ND2": [5, (-0.593, -1.188, -0.001)],
        "O": [3, (-0.625, 1.062, 0.0)],
        "OD1": [5, (-0.633, 1.059, 0.0)],
    },
    "P": {
        "C": [0, (1.527, -0.0, 0.0)],
        "CA": [0, (0.0, 0.0, 0.0)],
        "CB": [0, (-0.546, -0.611, -1.293)],
        "CD": [5, (-0.477, 1.424, 0.0)],
        "CG": [4, (-0.382, 1.445, 0.0)],
        "N": [0, (-0.566, 1.351, -0.0)],
        "O": [3, (-0.621, 1.066, 0.0)],
    },
    "Q": {
        "C": [0, (1.526, 0.0, 0.0)],
        "CA": [0, (0.0, 0.0, 0.0)],
        "CB": [0, (-0.525, -0.779, -1.207)],
        "CD": [5, (-0.587, 1.399, -0.0)],
        "CG": [4, (-0.615, 1.393, 0.0)],
        "N": [0, (-0.526, 1.361, -0.0)],
        "NE2": [6, (-0.593, -1.189, 0.001)],
        "O": [3, (-0.626, 1.062, -0.0)],
        "OE1": [6, (-0.634, 1.06, 0.0)],
    },
    "R": {
        "C": [0, (1.525, -0.0, -0.0)],
        "CA": [0, (0.0, 0.0, 0.0)],
        "CB": [0, (-0.524, -0.778, -1.209)],
        "CD": [5, (-0.564, 1.414, 0.0)],
        "CG": [4, (-0.616, 1.39, -0.0)],
        "CZ": [7, (-0.758, 1.093, -0.0)],
        "N": [0, (-0.524, 1.362, -0.0)],
        "NE": [6, (-0.539, 1.357, -0.0)],
        "NH1": [7, (-0.206, 2.301, 0.0)],
        "NH2": [7, (-2.078, 0.978, -0.0)],
        "O": [3, (-0.626, 1.062, 0.0)],
    },
    "S": {
        "C": [0, (1.525, -0.0, -0.0)],
        "CA": [0, (0.0, 0.0, 0.0)],
        "CB": [0, (-0.518, -0.777, -1.211)],
        "N": [0, (-0.529, 1.36, -0.0)],
        "O": [3, (-0.626, 1.062, -0.0)],
        "OG": [4, (-0.503, 1.325, 0.0)],
    },
    "T": {
        "C": [0, (1.526, 0.0, -0.0)],
        "CA": [0, (0.0, 0.0, 0.0)],
        "CB": [0, (-0.516, -0.793, -1.215)],
        "CG2": [4, (-0.55, -0.718, 1.228)],
        "N": [0, (-0.517, 1.364, 0.0)],
        "O": [3, (-0.626, 1.062, 0.0)],
        "OG1": [4, (-0.472, 1.353, 0.0)],
    },
    "V": {
        "C": [0, (1.527, -0.0, -0.0)],
        "CA": [0, (0.0, 0.0, 0.0)],
        "CB": [0, (-0.533, -0.795, -1.213)],
        "CG1": [4, (-0.54, 1.429, -0.0)],
        "CG2": [4, (-0.533, -0.776, -1.203)],
        "N": [0, (-0.494, 1.373, -0.0)],
        "O": [3, (-0.627, 1.062, -0.0)],
    },
    "W": {
        "C": [0, (1.525, -0.0, 0.0)],
        "CA": [0, (0.0, 0.0, 0.0)],
        "CB": [0, (-0.523, -0.776, -1.212)],
        "CD1": [5, (-0.824, 1.091, 0.0)],
        "CD2": [5, (-0.854, -1.148, -0.005)],
        "CE2": [5, (-2.186, -0.678, -0.007)],
        "CE3": [5, (-0.622, -2.53, -0.007)],
        "CG": [4, (-0.609, 1.37, -0.0)],
        "CH2": [5, (-3.028, -2.89, -0.013)],
        "CZ2": [5, (-3.283, -1.543, -0.011)],
        "CZ3": [5, (-1.715, -3.389, -0.011)],
        "N": [0, (-0.521, 1.363, 0.0)],
        "NE1": [5, (-2.14, 0.69, -0.004)],
        "O": [3, (-0.627, 1.062, 0.0)],
    },
    "Y": {
        "C": [0, (1.524, -0.0, -0.0)],
        "CA": [0, (0.0, 0.0, 0.0)],
        "CB": [0, (-0.522, -0.776, -1.213)],
        "CD1": [5, (-0.716, 1.195, -0.0)],
        "CD2": [5, (-0.713, -1.194, -0.001)],
        "CE1": [5, (-2.107, 1.2, -0.002)],
        "CE2": [5, (-2.104, -1.201, -0.003)],
        "CG": [4, (-0.607, 1.382, -0.0)],
        "CZ": [5, (-2.791, -0.001, -0.003)],
        "N": [0, (-0.522, 1.362, 0.0)],
        "O": [3, (-0.627, 1.062, -0.0)],
        "OH": [5, (-4.168, -0.002, -0.005)],
    },
}

chi_angles_atoms = {
    "A": [],
    "C": [["N", "CA", "CB", "SG"]],
    "D": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "E": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"], ["CB", "CG", "CD", "OE1"]],
    "F": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "G": [],
    "H": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]],
    "I": [["N", "CA", "CB", "CG1"], ["CA", "CB", "CG1", "CD1"]],
    "K": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "CE"],
        ["CG", "CD", "CE", "NZ"],
    ],
    "L": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "M": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "SD"], ["CB", "CG", "SD", "CE"]],
    "N": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "P": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"]],
    "Q": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"], ["CB", "CG", "CD", "OE1"]],
    "R": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "NE"],
        ["CG", "CD", "NE", "CZ"],
    ],
    "S": [["N", "CA", "CB", "OG"]],
    "T": [["N", "CA", "CB", "OG1"]],
    "V": [["N", "CA", "CB", "CG1"]],
    "W": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "Y": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
}

chi2_centers = {
    x: chi_angles_atoms[x][1][-2] if len(chi_angles_atoms[x]) > 1 else "CA"
    for x in chi_angles_atoms
}
chi3_centers = {
    x: chi_angles_atoms[x][2][-2] if len(chi_angles_atoms[x]) > 2 else "CA"
    for x in chi_angles_atoms
}
chi4_centers = {
    x: chi_angles_atoms[x][3][-2] if len(chi_angles_atoms[x]) > 3 else "CA"
    for x in chi_angles_atoms
}

rel_pos = {
    x: [
        rigid_group_atom_positions2[x][residue_atoms[x][atom_id]]
        if len(residue_atoms[x]) > atom_id
        else [0, (0, 0, 0)]
        for atom_id in range(14)
    ]
    for x in rigid_group_atom_positions2
}

r2n = {x: i for i, x in enumerate(restypes)}


def res_to_num(x):
    return r2n[x] if x in r2n else len(r2n)


# ---------------------------------------------------------------------------
# ImmuneBuilder/util.py (verbatim: input featurization helpers)
# ---------------------------------------------------------------------------


def get_one_hot(targets, nb_classes=21):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])


def get_encoding(sequence_dict, chain_ids="HL"):
    encodings = []

    for j, chain in enumerate(chain_ids):
        seq = sequence_dict[chain]
        one_hot_amino = get_one_hot(np.array([res_to_num(x) for x in seq]))
        one_hot_region = get_one_hot(j * np.ones(len(seq), dtype=int), 2)
        encoding = np.concatenate([one_hot_amino, one_hot_region], axis=-1)
        encodings.append(encoding)

    return np.concatenate(encodings, axis=0)


# ---------------------------------------------------------------------------
# ImmuneBuilder/rigids.py (verbatim: Vector/Rot/Rigid + frame geometry)
# ---------------------------------------------------------------------------


class Vector:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.shape = x.shape
        assert (x.shape == y.shape) and (y.shape == z.shape), "x y and z should have the same shape"

    def __add__(self, vec):
        return Vector(vec.x + self.x, vec.y + self.y, vec.z + self.z)

    def __sub__(self, vec):
        return Vector(-vec.x + self.x, -vec.y + self.y, -vec.z + self.z)

    def __mul__(self, param):
        return Vector(param * self.x, param * self.y, param * self.z)

    def __matmul__(self, vec):
        return vec.x * self.x + vec.y * self.y + vec.z * self.z

    def norm(self):
        return (self.x**2 + self.y**2 + self.z**2 + 1e-8) ** (1 / 2)

    def cross(self, other):
        a = self.y * other.z - self.z * other.y
        b = self.z * other.x - self.x * other.z
        c = self.x * other.y - self.y * other.x
        return Vector(a, b, c)

    def dist(self, other):
        return (
            (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2 + 1e-8
        ) ** (1 / 2)

    def unsqueeze(self, dim):
        return Vector(self.x.unsqueeze(dim), self.y.unsqueeze(dim), self.z.unsqueeze(dim))

    def squeeze(self, dim):
        return Vector(self.x.squeeze(dim), self.y.squeeze(dim), self.z.squeeze(dim))

    def map(self, func):
        return Vector(func(self.x), func(self.y), func(self.z))

    def to(self, device):
        return Vector(self.x.to(device), self.y.to(device), self.z.to(device))

    def __getitem__(self, key):
        return Vector(self.x[key], self.y[key], self.z[key])


class Rot:
    def __init__(self, xx, xy, xz, yx, yy, yz, zx, zy, zz):
        self.xx = xx
        self.xy = xy
        self.xz = xz
        self.yx = yx
        self.yy = yy
        self.yz = yz
        self.zx = zx
        self.zy = zy
        self.zz = zz
        self.shape = xx.shape

    def __matmul__(self, other):
        if isinstance(other, Vector):
            return Vector(
                other.x * self.xx + other.y * self.xy + other.z * self.xz,
                other.x * self.yx + other.y * self.yy + other.z * self.yz,
                other.x * self.zx + other.y * self.zy + other.z * self.zz,
            )

        if isinstance(other, Rot):
            return Rot(
                xx=self.xx * other.xx + self.xy * other.yx + self.xz * other.zx,
                xy=self.xx * other.xy + self.xy * other.yy + self.xz * other.zy,
                xz=self.xx * other.xz + self.xy * other.yz + self.xz * other.zz,
                yx=self.yx * other.xx + self.yy * other.yx + self.yz * other.zx,
                yy=self.yx * other.xy + self.yy * other.yy + self.yz * other.zy,
                yz=self.yx * other.xz + self.yy * other.yz + self.yz * other.zz,
                zx=self.zx * other.xx + self.zy * other.yx + self.zz * other.zx,
                zy=self.zx * other.xy + self.zy * other.yy + self.zz * other.zy,
                zz=self.zx * other.xz + self.zy * other.yz + self.zz * other.zz,
            )

        else:
            raise ValueError("Matmul against {}".format(type(other)))

    def inv(self):
        return Rot(
            xx=self.xx,
            xy=self.yx,
            xz=self.zx,
            yx=self.xy,
            yy=self.yy,
            yz=self.zy,
            zx=self.xz,
            zy=self.yz,
            zz=self.zz,
        )

    def unsqueeze(self, dim):
        return Rot(
            self.xx.unsqueeze(dim=dim),
            self.xy.unsqueeze(dim=dim),
            self.xz.unsqueeze(dim=dim),
            self.yx.unsqueeze(dim=dim),
            self.yy.unsqueeze(dim=dim),
            self.yz.unsqueeze(dim=dim),
            self.zx.unsqueeze(dim=dim),
            self.zy.unsqueeze(dim=dim),
            self.zz.unsqueeze(dim=dim),
        )

    def squeeze(self, dim):
        return Rot(
            self.xx.squeeze(dim=dim),
            self.xy.squeeze(dim=dim),
            self.xz.squeeze(dim=dim),
            self.yx.squeeze(dim=dim),
            self.yy.squeeze(dim=dim),
            self.yz.squeeze(dim=dim),
            self.zx.squeeze(dim=dim),
            self.zy.squeeze(dim=dim),
            self.zz.squeeze(dim=dim),
        )

    def to(self, device):
        return Rot(
            self.xx.to(device),
            self.xy.to(device),
            self.xz.to(device),
            self.yx.to(device),
            self.yy.to(device),
            self.yz.to(device),
            self.zx.to(device),
            self.zy.to(device),
            self.zz.to(device),
        )

    def __getitem__(self, key):
        return Rot(
            self.xx[key],
            self.xy[key],
            self.xz[key],
            self.yx[key],
            self.yy[key],
            self.yz[key],
            self.zx[key],
            self.zy[key],
            self.zz[key],
        )


class Rigid:
    def __init__(self, origin, rot):
        self.origin = origin
        self.rot = rot
        self.shape = self.origin.shape

    def __matmul__(self, other):
        if isinstance(other, Vector):
            return self.rot @ other + self.origin
        elif isinstance(other, Rigid):
            return Rigid(self.rot @ other.origin + self.origin, self.rot @ other.rot)
        else:
            raise TypeError(f"can't multiply rigid by object of type {type(other)}")

    def inv(self):
        inv_rot = self.rot.inv()
        t = inv_rot @ self.origin
        return Rigid(Vector(-t.x, -t.y, -t.z), inv_rot)

    def unsqueeze(self, dim=None):
        return Rigid(self.origin.unsqueeze(dim=dim), self.rot.unsqueeze(dim=dim))

    def squeeze(self, dim=None):
        return Rigid(self.origin.squeeze(dim=dim), self.rot.squeeze(dim=dim))

    def to(self, device):
        return Rigid(self.origin.to(device), self.rot.to(device))

    def __getitem__(self, key):
        return Rigid(self.origin[key], self.rot[key])


def rigid_body_identity(shape):
    return Rigid(
        Vector(*3 * [torch.zeros(shape)]),
        Rot(
            torch.ones(shape),
            *3 * [torch.zeros(shape)],
            torch.ones(shape),
            *3 * [torch.zeros(shape)],
            torch.ones(shape),
        ),
    )


def vec_from_tensor(tens):
    assert tens.shape[-1] == 3, "What dimension you in?"
    return Vector(tens[..., 0], tens[..., 1], tens[..., 2])


def rigid_from_three_points(origin, y_x_plane, x_axis):
    v1 = x_axis - origin
    v2 = y_x_plane - origin

    v1 *= 1 / v1.norm()
    v2 = v2 - v1 * (v1 @ v2)
    v2 *= 1 / v2.norm()
    v3 = v1.cross(v2)
    rot = Rot(v1.x, v2.x, v3.x, v1.y, v2.y, v3.y, v1.z, v2.z, v3.z)
    return Rigid(origin, rot)


def stack_rigids(rigids, **kwargs):
    stacked_origin = Vector(
        torch.stack([rig.origin.x for rig in rigids], **kwargs),
        torch.stack([rig.origin.y for rig in rigids], **kwargs),
        torch.stack([rig.origin.z for rig in rigids], **kwargs),
    )
    stacked_rot = Rot(
        torch.stack([rig.rot.xx for rig in rigids], **kwargs),
        torch.stack([rig.rot.xy for rig in rigids], **kwargs),
        torch.stack([rig.rot.xz for rig in rigids], **kwargs),
        torch.stack([rig.rot.yx for rig in rigids], **kwargs),
        torch.stack([rig.rot.yy for rig in rigids], **kwargs),
        torch.stack([rig.rot.yz for rig in rigids], **kwargs),
        torch.stack([rig.rot.zx for rig in rigids], **kwargs),
        torch.stack([rig.rot.zy for rig in rigids], **kwargs),
        torch.stack([rig.rot.zz for rig in rigids], **kwargs),
    )
    return Rigid(stacked_origin, stacked_rot)


def rotate_x_axis_to_new_vector(new_vector):
    c, b, a = new_vector[..., 0], new_vector[..., 1], new_vector[..., 2]

    n = (c**2 + a**2 + b**2 + 1e-16) ** (1 / 2)
    a, b, c = a / n, b / n, -c / n

    new_origin = vec_from_tensor(torch.zeros_like(new_vector))

    k = (1 - c) / (a**2 + b**2 + 1e-8)
    new_rot = Rot(-c, b, -a, b, 1 - k * b**2, a * b * k, a, -a * b * k, k * a**2 - 1)

    return Rigid(new_origin, new_rot)


def rigid_transformation_from_torsion_angles(torsion_angles, distance_to_new_origin):
    dev = torsion_angles.device

    zero = torch.zeros(torsion_angles.shape[:-1]).to(dev)
    one = torch.ones(torsion_angles.shape[:-1]).to(dev)
    new_rot = Rot(
        -one,
        zero,
        zero,
        zero,
        torsion_angles[..., 0],
        torsion_angles[..., 1],
        zero,
        torsion_angles[..., 1],
        -torsion_angles[..., 0],
    )
    new_origin = Vector(distance_to_new_origin, zero, zero)

    return Rigid(new_origin, new_rot)


def global_frames_from_bb_frame_and_torsion_angles(bb_frame, torsion_angles, seq):
    dev = bb_frame.origin.x.device

    # We start with psi
    psi_local_frame_origin = (
        torch.tensor([rel_pos[x][2][1] for x in seq]).to(dev).pow(2).sum(-1).pow(1 / 2)
    )
    psi_local_frame = rigid_transformation_from_torsion_angles(
        torsion_angles[:, 0], psi_local_frame_origin
    )
    psi_global_frame = bb_frame @ psi_local_frame

    # Now all the chis
    chi1_local_frame_origin = torch.tensor([rel_pos[x][3][1] for x in seq]).to(dev)
    chi1_local_frame = rotate_x_axis_to_new_vector(
        chi1_local_frame_origin
    ) @ rigid_transformation_from_torsion_angles(
        torsion_angles[:, 1], chi1_local_frame_origin.pow(2).sum(-1).pow(1 / 2)
    )
    chi1_global_frame = bb_frame @ chi1_local_frame

    chi2_local_frame_origin = torch.tensor(
        [rigid_group_atom_positions2[x][chi2_centers[x]][1] for x in seq]
    ).to(dev)
    chi2_local_frame = rotate_x_axis_to_new_vector(
        chi2_local_frame_origin
    ) @ rigid_transformation_from_torsion_angles(
        torsion_angles[:, 2], chi2_local_frame_origin.pow(2).sum(-1).pow(1 / 2)
    )
    chi2_global_frame = chi1_global_frame @ chi2_local_frame

    chi3_local_frame_origin = torch.tensor(
        [rigid_group_atom_positions2[x][chi3_centers[x]][1] for x in seq]
    ).to(dev)
    chi3_local_frame = rotate_x_axis_to_new_vector(
        chi3_local_frame_origin
    ) @ rigid_transformation_from_torsion_angles(
        torsion_angles[:, 3], chi3_local_frame_origin.pow(2).sum(-1).pow(1 / 2)
    )
    chi3_global_frame = chi2_global_frame @ chi3_local_frame

    chi4_local_frame_origin = torch.tensor(
        [rigid_group_atom_positions2[x][chi4_centers[x]][1] for x in seq]
    ).to(dev)
    chi4_local_frame = rotate_x_axis_to_new_vector(
        chi4_local_frame_origin
    ) @ rigid_transformation_from_torsion_angles(
        torsion_angles[:, 4], chi4_local_frame_origin.pow(2).sum(-1).pow(1 / 2)
    )
    chi4_global_frame = chi3_global_frame @ chi4_local_frame

    return stack_rigids(
        [
            bb_frame,
            psi_global_frame,
            chi1_global_frame,
            chi2_global_frame,
            chi3_global_frame,
            chi4_global_frame,
        ],
        dim=-1,
    )


def all_atoms_from_global_reference_frames(global_reference_frames, seq):
    dev = global_reference_frames.origin.x.device

    all_atoms = torch.zeros((len(seq), 14, 3)).to(dev)
    for atom_pos in range(14):
        relative_positions = [rel_pos[x][atom_pos][1] for x in seq]
        local_reference_frame = [max(rel_pos[x][atom_pos][0] - 2, 0) for x in seq]
        local_reference_frame_mask = torch.tensor(
            [[y == x for y in range(6)] for x in local_reference_frame]
        ).to(dev)
        global_atom_vector = global_reference_frames[local_reference_frame_mask] @ vec_from_tensor(
            torch.tensor(relative_positions).to(dev)
        )
        all_atoms[:, atom_pos] = torch.stack(
            [global_atom_vector.x, global_atom_vector.y, global_atom_vector.z], dim=-1
        )

    all_atom_mask = torch.tensor([residue_atoms_mask[x] for x in seq]).to(dev)
    all_atoms[~all_atom_mask] = float("Nan")
    return all_atoms


# ---------------------------------------------------------------------------
# ImmuneBuilder/models.py (verbatim: IPA structure module)
# ---------------------------------------------------------------------------


class InvariantPointAttention(torch.nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        heads=12,
        head_dim=16,
        n_query_points=4,
        n_value_points=8,
        **kwargs,
    ):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.n_query_points = n_query_points

        node_scalar_attention_inner_dim = heads * head_dim
        node_vector_attention_inner_dim = 3 * n_query_points * heads
        node_vector_attention_value_dim = 3 * n_value_points * heads
        after_final_cat_dim = heads * edge_dim + heads * head_dim + heads * n_value_points * 4

        point_weight_init_value = torch.log(torch.exp(torch.full((heads,), 1.0)) - 1.0)
        self.point_weight = torch.nn.Parameter(point_weight_init_value)

        self.to_scalar_qkv = torch.nn.Linear(
            node_dim, 3 * node_scalar_attention_inner_dim, bias=False
        )
        self.to_vector_qk = torch.nn.Linear(
            node_dim, 2 * node_vector_attention_inner_dim, bias=False
        )
        self.to_vector_v = torch.nn.Linear(node_dim, node_vector_attention_value_dim, bias=False)
        self.to_scalar_edge_attention_bias = torch.nn.Linear(edge_dim, heads, bias=False)
        self.final_linear = torch.nn.Linear(after_final_cat_dim, node_dim)

        with torch.no_grad():
            self.final_linear.weight.fill_(0.0)
            self.final_linear.bias.fill_(0.0)

    def forward(self, node_features, edge_features, rigid):
        # Classic attention on nodes
        scalar_qkv = self.to_scalar_qkv(node_features).chunk(3, dim=-1)
        scalar_q, scalar_k, scalar_v = map(
            lambda t: rearrange(t, "n (h d) -> h n d", h=self.heads), scalar_qkv
        )
        node_scalar = torch.einsum("h i d, h j d -> h i j", scalar_q, scalar_k) * self.head_dim ** (
            -1 / 2
        )

        # Linear bias on edges
        edge_bias = rearrange(self.to_scalar_edge_attention_bias(edge_features), "i j h -> h i j")

        # Reference frame attention
        wc = (2 / self.n_query_points) ** (1 / 2) / 6
        vector_qk = self.to_vector_qk(node_features).chunk(2, dim=-1)
        vector_q, vector_k = map(
            lambda x: vec_from_tensor(rearrange(x, "n (h p d) -> h n p d", h=self.heads, d=3)),
            vector_qk,
        )
        rigid_ = rigid.unsqueeze(0).unsqueeze(-1)  # add head and point dimension to rigids

        global_vector_k = rigid_ @ vector_k
        global_vector_q = rigid_ @ vector_q
        global_frame_distance = (
            wc
            * global_vector_q.unsqueeze(-2).dist(global_vector_k.unsqueeze(-3)).sum(-1)
            * rearrange(self.point_weight, "h -> h () ()")
        )

        # Combining attentions
        attention_matrix = (
            3 ** (-1 / 2) * (node_scalar + edge_bias - global_frame_distance)
        ).softmax(-1)

        # Obtaining outputs
        edge_output = (
            rearrange(attention_matrix, "h i j -> i h () j")
            * rearrange(edge_features, "i j d -> i () d j")
        ).sum(-1)
        scalar_node_output = torch.einsum("h i j, h j d -> i h d", attention_matrix, scalar_v)

        vector_v = vec_from_tensor(
            rearrange(self.to_vector_v(node_features), "n (h p d) -> h n p d", h=self.heads, d=3)
        )
        global_vector_v = rigid_ @ vector_v
        attended_global_vector_v = global_vector_v.map(
            lambda x: torch.einsum("h i j, h j p -> h i p", attention_matrix, x)
        )
        vector_node_output = rigid_.inv() @ attended_global_vector_v
        vector_node_output = torch.stack(
            [
                vector_node_output.norm(),
                vector_node_output.x,
                vector_node_output.y,
                vector_node_output.z,
            ],
            dim=-1,
        )

        # Concatenate along heads and points
        edge_output = rearrange(edge_output, "n h d -> n (h d)")
        scalar_node_output = rearrange(scalar_node_output, "n h d -> n (h d)")
        vector_node_output = rearrange(vector_node_output, "h n p d -> n (h p d)")

        combined = torch.cat([edge_output, scalar_node_output, vector_node_output], dim=-1)

        return node_features + self.final_linear(combined)


class BackboneUpdate(torch.nn.Module):
    def __init__(self, node_dim):
        super().__init__()

        self.to_correction = torch.nn.Linear(node_dim, 6)

    def forward(self, node_features, update_mask=None):
        # Predict quaternions and translation vector
        rot, t = self.to_correction(node_features).chunk(2, dim=-1)

        # I may not want to update all residues
        if update_mask is not None:
            rot = update_mask[:, None] * rot
            t = update_mask[:, None] * t

        # Normalize quaternions
        norm = (1 + rot.pow(2).sum(-1, keepdim=True)).pow(1 / 2)
        b, c, d = (rot / norm).chunk(3, dim=-1)
        a = 1 / norm
        a, b, c, d = a.squeeze(-1), b.squeeze(-1), c.squeeze(-1), d.squeeze(-1)

        # Make rotation matrix from quaternions
        R = Rot(
            (a**2 + b**2 - c**2 - d**2),
            (2 * b * c - 2 * a * d),
            (2 * b * d + 2 * a * c),
            (2 * b * c + 2 * a * d),
            (a**2 - b**2 + c**2 - d**2),
            (2 * c * d - 2 * a * b),
            (2 * b * d - 2 * a * c),
            (2 * c * d + 2 * a * b),
            (a**2 - b**2 - c**2 + d**2),
        )

        return Rigid(vec_from_tensor(t), R)


class TorsionAngles(torch.nn.Module):
    def __init__(self, node_dim):
        super().__init__()
        self.residual1 = torch.nn.Sequential(
            torch.nn.Linear(2 * node_dim, 2 * node_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * node_dim, 2 * node_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * node_dim, 2 * node_dim),
        )

        self.residual2 = torch.nn.Sequential(
            torch.nn.Linear(2 * node_dim, 2 * node_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * node_dim, 2 * node_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * node_dim, 2 * node_dim),
        )

        self.final_pred = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(2 * node_dim, 10))

        with torch.no_grad():
            self.residual1[-1].weight.fill_(0.0)
            self.residual2[-1].weight.fill_(0.0)
            self.residual1[-1].bias.fill_(0.0)
            self.residual2[-1].bias.fill_(0.0)

    def forward(self, node_features, s_i):
        full_feat = torch.cat([node_features, s_i], axis=-1)

        full_feat = full_feat + self.residual1(full_feat)
        full_feat = full_feat + self.residual2(full_feat)
        torsions = rearrange(self.final_pred(full_feat), "i (t d) -> i t d", d=2)
        norm = torch.norm(torsions, dim=-1, keepdim=True)

        return torsions / norm, norm


class StructureUpdate(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, dropout=0.0, **kwargs):
        super().__init__()
        self.IPA = InvariantPointAttention(node_dim, edge_dim, **kwargs)
        self.norm1 = torch.nn.Sequential(torch.nn.Dropout(dropout), torch.nn.LayerNorm(node_dim))
        self.norm2 = torch.nn.Sequential(torch.nn.Dropout(dropout), torch.nn.LayerNorm(node_dim))
        self.residual = torch.nn.Sequential(
            torch.nn.Linear(node_dim, 2 * node_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * node_dim, 2 * node_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * node_dim, node_dim),
        )

        self.torsion_angles = TorsionAngles(node_dim)
        self.backbone_update = BackboneUpdate(node_dim)

        with torch.no_grad():
            self.residual[-1].weight.fill_(0.0)
            self.residual[-1].bias.fill_(0.0)

    def forward(self, node_features, edge_features, rigid_pred, update_mask=None):
        s_i = self.IPA(node_features, edge_features, rigid_pred)
        s_i = self.norm1(s_i)
        s_i = s_i + self.residual(s_i)
        s_i = self.norm2(s_i)
        rigid_new = rigid_pred @ self.backbone_update(s_i, update_mask)

        return s_i, rigid_new


class StructureModule(torch.nn.Module):
    def __init__(self, node_dim=23, n_layers=8, rel_pos_dim=64, embed_dim=128, **kwargs):
        super().__init__()
        self.n_layers = n_layers
        self.rel_pos_dim = rel_pos_dim
        self.node_embed = torch.nn.Linear(node_dim, embed_dim)
        self.edge_embed = torch.nn.Linear(2 * rel_pos_dim + 1, embed_dim - 1)

        self.layers = torch.nn.ModuleList(
            [
                StructureUpdate(
                    node_dim=embed_dim,
                    edge_dim=embed_dim,
                    propagate_rotation_gradient=(i == n_layers - 1),
                    **kwargs,
                )
                for i in range(n_layers)
            ]
        )

    def forward(self, node_features, sequence):
        rigid_in = rigid_body_identity(len(sequence)).to(node_features.device)
        relative_positions = (
            torch.arange(node_features.shape[-2])[None]
            - torch.arange(node_features.shape[-2])[:, None]
        )
        relative_positions = (
            relative_positions.clamp(min=-self.rel_pos_dim, max=self.rel_pos_dim) + self.rel_pos_dim
        )

        rel_pos_embeddings = torch.nn.functional.one_hot(
            relative_positions, num_classes=2 * self.rel_pos_dim + 1
        )
        rel_pos_embeddings = rel_pos_embeddings.to(
            dtype=node_features.dtype, device=node_features.device
        )
        rel_pos_embeddings = self.edge_embed(rel_pos_embeddings)

        new_node_features = self.node_embed(node_features)

        for layer in self.layers:
            edge_features = torch.cat(
                [
                    rigid_in.origin.unsqueeze(-1).dist(rigid_in.origin).unsqueeze(-1),
                    rel_pos_embeddings,
                ],
                dim=-1,
            )
            new_node_features, rigid_in = layer(new_node_features, edge_features, rigid_in)

        torsions, _ = self.layers[-1].torsion_angles(
            self.node_embed(node_features), new_node_features
        )

        all_reference_frames = global_frames_from_bb_frame_and_torsion_angles(
            rigid_in, torsions, sequence
        )
        all_atoms = all_atoms_from_global_reference_frames(all_reference_frames, sequence)

        # Remove atoms of side chains with outrageous clashes
        ds = torch.linalg.norm(all_atoms[None, :, None] - all_atoms[:, None, :, None], axis=-1)
        ds[torch.isnan(ds) | (ds == 0.0)] = 10
        min_ds = ds.min(dim=-1)[0].min(dim=-1)[0].min(dim=-1)[0]
        all_atoms[min_ds < 0.2, 5:, :] = float("Nan")

        return all_atoms, new_node_features


# ---------------------------------------------------------------------------
# Tiny-scale staging wrapper (torchlens capture needs a single forward()
# call with a concrete example input; the real forward() takes the node
# feature tensor plus the amino-acid sequence string, since the geometry
# helpers key rigid-group/torsion constants by residue letter).
# ---------------------------------------------------------------------------

_HEAVY_SEQ = "EVQLVESGGGLVQPGGSLRLSCAASGFTFS"  # short IMGT-style heavy-chain stub
_LIGHT_SEQ = "DIQMTQSPSSLSASVGDRVTITCRASQSIS"  # short IMGT-style light-chain stub


class ABodyBuilder2TraceWrapper(torch.nn.Module):
    """Wraps StructureModule.forward with fixed example inputs so
    TorchLens can capture a plain single-tensor-in forward pass."""

    def __init__(self, model: StructureModule, sequence: str):
        super().__init__()
        self.model = model
        self.sequence = sequence

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        all_atoms, _new_node_features = self.model(node_features, self.sequence)
        return all_atoms


def build_abodybuilder2() -> ABodyBuilder2TraceWrapper:
    sequence_dict = {"H": _HEAVY_SEQ, "L": _LIGHT_SEQ}
    full_seq = sequence_dict["H"] + sequence_dict["L"]
    # Real default is embed_dim[model_file] in {128, 256}; rel_pos_dim=64 is
    # the real released constant. n_layers shrunk from the real default of 8
    # for a tiny trace instance.
    model = StructureModule(rel_pos_dim=8, embed_dim=16, n_layers=2, heads=2, head_dim=4)
    model.eval()
    return ABodyBuilder2TraceWrapper(model, full_seq)


def example_input_abodybuilder2() -> torch.Tensor:
    sequence_dict = {"H": _HEAVY_SEQ, "L": _LIGHT_SEQ}
    encoding = get_encoding(sequence_dict)
    return torch.tensor(encoding, dtype=torch.get_default_dtype())


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "ABodyBuilder2",
        "build_abodybuilder2",
        "example_input_abodybuilder2",
        2023,
        "vendored-pytorch",
    ),
]
