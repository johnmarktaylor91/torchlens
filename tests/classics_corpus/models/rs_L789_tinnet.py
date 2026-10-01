# SOURCE: vendored from https://github.com/hlxin/tinnet @ master (c80f5d12d304)
# (tinnet/band_center.py's ConvLayer + CrystalGraphConvNet)
#
# TinNet (Lin, Wang, Cao, Xin, "Automated Discovery of Adsorbate-Metal
# Interaction Descriptors from Machine Learning with Physics-Informed
# Priors", Nature Communications 2024, arXiv). Theory-infused neural network
# for catalysis: a CGCNN-style (crystal graph convolutional network) graph
# encoder over an adsorbate/metal-surface crystal graph, whose atom-level
# convolved features are read out through TWO heads -- a per-atom head
# (`fc_out`, used elsewhere in the repo's d-band physics module to regress
# tabulated d-band-center descriptors) and a crystal-level head (`fc_out_crys`,
# 3-way output used to regress the d-band physics-model parameters: filling,
# center, width). Only `ConvLayer`/`CrystalGraphConvNet` (the actual GNN) are
# vendored; the repo's `shap`/`ase` explainability and atomistic-geometry
# preprocessing helpers in the surrounding `BandCenter`/`Features` classes are
# not part of the model architecture and are not needed to construct/trace it.
# No architecture altered.

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """

    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2 * self.atom_fea_len + self.nbr_fea_len, 2 * self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx, tabulated_padding_fillter):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution

        """
        N, M = nbr_fea_idx.shape
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len), atom_nbr_fea, nbr_fea], dim=2
        )
        total_gated_fea = self.fc_full(total_nbr_fea)
        tabulated_padding_fillter_flatten = tabulated_padding_fillter.view(-1)
        total_gated_fea = total_gated_fea.view(-1, self.atom_fea_len * 2)
        total_gated_fea_bn1 = self.bn1(
            total_gated_fea[torch.where(tabulated_padding_fillter_flatten == 1)[0]]
        )
        total_gated_fea[torch.where(tabulated_padding_fillter_flatten == 1)[0]] = (
            total_gated_fea_bn1
        )
        total_gated_fea = total_gated_fea.view(N, M, self.atom_fea_len * 2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core * tabulated_padding_fillter[:, :, None], dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """

    def __init__(
        self,
        orig_atom_fea_len,
        nbr_fea_len,
        atom_fea_len=64,
        n_conv=3,
        h_fea_len=128,
        n_h=1,
        model_num_input=1,
    ):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------

        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        """
        super(CrystalGraphConvNet, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList(
            [ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len) for _ in range(n_conv)]
        )
        self.conv_to_fc = nn.Linear(atom_fea_len + nbr_fea_len, h_fea_len)
        self.conv_to_fc_crys = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)])
            self.softpluses = nn.ModuleList([nn.Softplus() for _ in range(n_h - 1)])

        if n_h > 1:
            self.fcs_crys = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)])
            self.softpluses_crys = nn.ModuleList([nn.Softplus() for _ in range(n_h - 1)])

        self.fc_out = nn.Linear(h_fea_len, model_num_input)
        self.fc_out_crys = nn.Linear(h_fea_len, 3)

    def forward(
        self, atom_fea, nbr_fea, nbr_fea_idx, tabulated_padding_fillter, crystal_atom_idx, atom_inx
    ):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx

        Returns
        -------

        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution

        """
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx, tabulated_padding_fillter)

        N, M = nbr_fea_idx.shape
        # convolution
        atom_nbr_fea = atom_fea[nbr_fea_idx, :]

        avg_fea = (atom_fea.unsqueeze(1).expand(N, M, self.atom_fea_len) + atom_nbr_fea) / 2.0

        total_nbr_fea = torch.cat([avg_fea, nbr_fea], dim=2)

        total_nbr_fea = self.conv_to_fc_softplus(total_nbr_fea)
        total_nbr_fea = self.conv_to_fc(total_nbr_fea)
        total_nbr_fea = self.conv_to_fc_softplus(total_nbr_fea)
        if hasattr(self, "fcs") and hasattr(self, "softpluses"):
            for fc, softplus in zip(self.fcs, self.softpluses):
                total_nbr_fea = softplus(fc(total_nbr_fea))

        out = self.fc_out(total_nbr_fea) * tabulated_padding_fillter[:, :, None]

        crys_fea = torch.atleast_2d(atom_fea[atom_inx])
        crys_fea = self.conv_to_fc_crys(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)

        if hasattr(self, "fcs") and hasattr(self, "softpluses"):
            for fc_crys, softplus_crys in zip(self.fcs_crys, self.softpluses_crys):
                crys_fea = softplus_crys(fc_crys(crys_fea))

        out_crys = self.fc_out_crys(crys_fea)

        return out, out_crys


def build_tinnet_cgcnn():
    torch.manual_seed(0)
    # Real repo defaults: atom_fea_len=64, n_conv=3, h_fea_len=128. Shrunk to
    # menagerie-recipe scale; orig_atom_fea_len/nbr_fea_len match the CGCNN
    # featurization dims used across the repo's `Features` preprocessor.
    model = CrystalGraphConvNet(
        orig_atom_fea_len=8,
        nbr_fea_len=4,
        atom_fea_len=16,
        n_conv=2,
        h_fea_len=16,
        n_h=1,
        model_num_input=1,
    )
    model.eval()
    return model


def example_input_tinnet_cgcnn():
    torch.manual_seed(0)
    n_atoms = 12
    max_nbrs = 6
    orig_atom_fea_len = 8
    nbr_fea_len = 4

    atom_fea = torch.randn(n_atoms, orig_atom_fea_len)
    nbr_fea = torch.randn(n_atoms, max_nbrs, nbr_fea_len)
    nbr_fea_idx = torch.randint(0, n_atoms, (n_atoms, max_nbrs), dtype=torch.long)
    tabulated_padding_fillter = torch.ones(n_atoms, max_nbrs)
    crystal_atom_idx = [torch.arange(n_atoms, dtype=torch.long)]
    atom_inx = torch.tensor([0], dtype=torch.long)

    return (atom_fea, nbr_fea, nbr_fea_idx, tabulated_padding_fillter, crystal_atom_idx, atom_inx)


MENAGERIE_ENTRIES = [
    (
        "TinNet (theory-infused CGCNN + d-band physics readout)",
        "build_tinnet_cgcnn",
        "example_input_tinnet_cgcnn",
        2024,
        MENAGERIE_ZOO,
    ),
]
