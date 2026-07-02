# SOURCE: vendored from https://github.com/AliceDeLorenci/CayleyNets @ main
#   Vendored file:
#     - src/CayleyNet.py -> `ComplexLinear`, `jacobi_method`, `jacobi_method_sparse`,
#       `CayleyConv`, `CayleyNet` (verbatim). This is the student re-implementation
#       (AliceDeLorenci) of the CayleyNets graph-spectral-filter architecture
#       (Levie, Monti, Bresson & Bronstein, ICLR 2019), built directly on
#       PyTorch Geometric's Laplacian/self-loop utilities. `jacobi_method` (dense
#       Jacobi solver, unused by `CayleyConv.forward`, which calls the sparse
#       variant) is kept since it is real file content.
#
#   Import fix only (no architectural changes): the module-level `device` global
#   picked `cuda` whenever a GPU was present (`torch.device('cuda' if
#   torch.cuda.is_available() else 'cpu')`), which stranded `CayleyConv.h` (a
#   `Parameter` built with that global at construction time) on the GPU while
#   plain-CPU trace inputs stayed on the CPU. Hardcoded to `torch.device('cpu')` here
#   so the module runs on whatever device the caller's tensors are on by default.
#
# CayleyNet (Levie et al., ICLR 2019) is a spectral graph convolutional network that
# replaces ChebNet's real polynomial filters with complex rational Cayley polynomials
# of the graph Laplacian, giving filters with a tunable "zoom" parameter that can
# localize to any frequency band (not just low frequencies) without a full eigen-
# decomposition. Each `CayleyConv` layer applies `r` iterations of a learned Cayley
# filter (approximated via Jacobi iterations for the required inverse) with complex-
# valued linear coefficients, sums their real parts, and adds a real "self" term;
# `CayleyNet` stacks these layers with ReLU/dropout for node classification.
# Architecture is reproduced verbatim.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import get_laplacian, add_self_loops, to_dense_adj

MENAGERIE_ZOO = "vendored-pytorch"

device = torch.device("cpu")


class ComplexLinear(torch.nn.Module):
    """
    Linear layer for complex valued weights and inputs

    For a complex valued input $z = a + ib $ and a complex valued weight $M=M_R+iM_b, the output is
    $Mz = M_R a - M_I b + i ( M_I a + M_R b)$

    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    bias : bool
        If True, adds a learnable complex bias to the output
    """

    def __init__(self, in_channels, out_channels, bias=False, weight_initializer="glorot"):
        super(ComplexLinear, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias

        self.linear_re = Linear(
            in_channels, out_channels, bias=bias, weight_initializer=weight_initializer
        )
        self.linear_im = Linear(
            in_channels, out_channels, bias=bias, weight_initializer=weight_initializer
        )

    def reset_parameters(self):
        self.linear_re.reset_parameters()
        self.linear_im.reset_parameters()

    def forward(self, x):
        # (a + ib) * (c + id) = (ac - bd) + i(ad + bc)
        x_re = self.linear_re(x.real) - self.linear_im(x.imag)
        x_im = self.linear_re(x.real) + self.linear_im(x.imag)

        return torch.complex(x_re, x_im)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, bias={self.bias})"
        )


def jacobi_method(A, b, K):
    """
    Computes approximate solution to Ax = b using Jacobi's Method with K iterations.

    Jacobi Method
    -------------
    Let A = D + (L+U), where D is the diagonal matrix and (L+U) is the off diagonal matrix,
    then, x is the solution to the following fixed point equation:
    D@x = -(L+U)@x + b
    Jacobi's iterations are of the form:
    x(k) = D^{-1}@( b - (L+U)@x(k-1) )
    denoting J = - D^{-1}@(L+U) and d = D^{-1}@b:
    x(k) = d + J@x(k-1)


    Parameters
    ----------
    A : torch.Tensor
        Symmetric matrix of shape (n, n)
    b : torch.Tensor
        Vector of shape (n, p)
    K : int
        Number of iterations

    Returns
    -------
    x : torch.Tensor
        Solution of the system of shape (n, p)
    """

    # Iterative method notation: u(k+1) = d + J@u(k)

    assert A.layout == torch.strided, "Sparse matrix: use sparse Jacobi instead"

    diag_inv = torch.diag(A)
    diag_inv = torch.where(diag_inv == 0, diag_inv, diag_inv**-1)

    off_diag = A - torch.diag(torch.diag(A))

    # CSR format affords efficient matrix-vector multiplication
    # too bad that '_to_sparse_csr does not support automatic differentiation for outputs with complex dtype'
    # off_diag = off_diag.to_sparse_csr()

    diag_inv = torch.reshape(diag_inv, (-1, 1))  # column vector of inverse degrees
    d = diag_inv.mul(
        b
    )  # elementwise multiplication of diag_inv by each column of b, equivalent to D^{-1}@b

    J = (-1.0) * diag_inv.mul(off_diag)  # J = - D^{-1}@(L+U)

    # initialize x
    x = b.clone()
    # Jacobi iteration
    for k in range(K):
        x = d + J.matmul(x)
    return x


def jacobi_method_sparse(A_idx, A_weight, b, jacobi_iterations, num_nodes):
    """
    Computes approximate solution to Ax = b using Jacobi's Method with K iterations,
    A should be in sparse COO format, where A_idx holds the data indices and A_weight the data values.

    Jacobi Method
    -------------
    Let A = D + (L+U), where D is the diagonal matrix and (L+U) is the off diagonal matrix,
    then, x is the solution to the following fixed point equation:
    D@x = -(L+U)@x + b
    Jacobi's iterations are of the form:
    x(k) = D^{-1}@( b - (L+U)@x(k-1) )
    denoting J = - D^{-1}@(L+U) and d = D^{-1}@b:
    x(k) = d + J@x(k-1)

    Parameters
    ----------


    Returns
    -------
    x : torch.Tensor
        Solution of the system of shape (n, 1)
    """

    # Diagonal matrix
    diag_nz_mask = A_idx[0, :] == A_idx[1, :]
    diag_nz_weight = A_weight[diag_nz_mask]

    diag_weight = torch.zeros(
        (num_nodes), dtype=A_weight.dtype, device=A_weight.device
    )  # initialize diagonal weights to zero
    diag_weight.scatter_add_(
        0, A_idx[1, diag_nz_mask], diag_nz_weight
    )  # sum all diagonal entries for each node

    diag_idx = torch.vstack([torch.arange(num_nodes), torch.arange(num_nodes)])
    diag_idx = diag_idx.to(device)

    off_diag_nz_mask = A_idx[0] != A_idx[1]
    off_diag_nz_idx = A_idx[:, off_diag_nz_mask]
    off_diag_nz_weight = A_weight[off_diag_nz_mask]

    inv_diag = diag_weight**-1
    inv_diag[inv_diag == torch.inf] = 0

    J = -torch.sparse.mm(
        torch.sparse_coo_tensor(diag_idx, inv_diag, torch.Size([num_nodes, num_nodes])),
        torch.sparse_coo_tensor(
            off_diag_nz_idx, off_diag_nz_weight, torch.Size([num_nodes, num_nodes])
        ),
    )
    d = torch.sparse.mm(
        torch.sparse_coo_tensor(diag_idx, inv_diag, torch.Size([num_nodes, num_nodes])), b
    )

    x = d.clone()
    for k in range(jacobi_iterations):
        x = torch.sparse.mm(J, x) + d

    return x


class CayleyConv(nn.Module):
    def __init__(self, in_channels, out_channels, r, normalization="sym", jacobi_iterations=10):
        """
        Implementation of a graph convolutional layer based on Cayley filters.
        Adapted from https://github.com/anon767/CayleyNet
        """
        super(CayleyConv, self).__init__()

        assert r > 0, "Invalid polynomial degree"
        assert normalization in [None, "sym", "rw"], "Invalid normalization"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization

        self.jacobi_iterations = jacobi_iterations

        self.r = r

        self.c = torch.nn.ModuleList(
            [Linear(in_channels, out_channels, bias=False, weight_initializer="glorot")]
            + [
                ComplexLinear(in_channels, out_channels, bias=False, weight_initializer="glorot")
                for _ in range(r)
            ]
        )

        self.h = Parameter(torch.ones(1, device=device))  # zoom parameter

        self.reset_parameters()

    def reset_parameters(self):
        self.h = Parameter(torch.ones(1, device=device))
        for c in self.c:
            c.reset_parameters()

    def forward(self, x: torch.tensor, edge_index: torch.tensor, edge_weight: torch.tensor = None):
        """
        Source: https://github.com/WhiteNoyse/SiGCN
        """
        num_nodes = x.shape[0]

        L_edge_index, L_edge_weight = get_laplacian(
            edge_index,
            edge_weight,
            normalization=self.normalization,
            dtype=torch.complex64,
            num_nodes=num_nodes,
        )

        L_edge_weight_zoomed = self.h * L_edge_weight

        # A = (hL + iI),  b = (hL - iI)x
        A_idx, A_weight = add_self_loops(
            edge_index=L_edge_index, edge_attr=L_edge_weight_zoomed, fill_value=torch.tensor(-1j)
        )  # h*Delta - i*Id
        B_idx, B_weight = add_self_loops(
            edge_index=L_edge_index, edge_attr=L_edge_weight_zoomed, fill_value=torch.tensor(1j)
        )  # h*Delta + i*Id
        B = torch.sparse_coo_tensor(
            B_idx, B_weight, torch.Size([num_nodes, num_nodes]), device=device
        )

        cumsum = 0 + 0j
        y_i = x.to(torch.complex64)
        for i in range(1, self.r + 1):
            b = torch.sparse.mm(B, y_i)
            y_i = jacobi_method_sparse(A_idx, A_weight, b, self.jacobi_iterations, num_nodes)
            cumsum += self.c[i](y_i)

        return self.c[0](x) + 2 * torch.real(cumsum)

    def __repr__(self):
        return "{}({}, {}, r={}, normalization={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.r, self.normalization
        )


class CayleyNet(torch.nn.Module):
    def __init__(
        self,
        in_feats,
        n_classes,
        n_hidden,
        n_layers,
        r=5,
        p_dropout=0.5,
        normalization="sym",
        seed=None,
    ):
        super(CayleyNet, self).__init__()
        if seed:
            torch.manual_seed(seed)

        self.layers = nn.ModuleList()
        self.layers.append(CayleyConv(in_feats, n_hidden, r, normalization=normalization))
        for _ in range(n_layers - 1):
            self.layers.append(CayleyConv(n_hidden, n_hidden, r, normalization=normalization))

        self.layers.append(CayleyConv(n_hidden, n_classes, r, normalization=normalization))
        self.p = p_dropout  # dropout probability

        self.p_dropout = p_dropout
        self.normalization = normalization

    def forward(self, x, edge_index):
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.p_dropout, training=self.training)

        x = self.layers[-1](x, edge_index)

        return F.log_softmax(x.float(), dim=-1)


# ---- tiny build/example (architecture unmodified from the real repo) ----


def build_cayleynet():
    """CayleyNet at tiny size for tracing (small in_feats/n_hidden/n_classes/r).
    Architecture is unmodified from the real repo."""
    model = CayleyNet(
        in_feats=8,
        n_classes=4,
        n_hidden=6,
        n_layers=1,
        r=2,
        p_dropout=0.0,
        normalization="sym",
        seed=0,
    )
    model.eval()
    return model


def example_input_cayleynet():
    """Matches CayleyNet.forward(x, edge_index): x is (num_nodes, in_feats) node
    features, edge_index is (2, num_edges) COO graph connectivity (int64)."""
    torch.manual_seed(0)
    x = torch.randn(10, 8)
    edge_index = torch.randint(0, 10, (2, 20))
    return (x, edge_index)


MENAGERIE_ENTRIES = [
    ("CayleyNet", "build_cayleynet", "example_input_cayleynet", 2019, MENAGERIE_ZOO),
]
