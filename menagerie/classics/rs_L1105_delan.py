# SOURCE: vendored from milutter/deep_lagrangian_networks @ master
# (deep_lagrangian_networks/DeLaN_model.py)
#
# Deep Lagrangian Networks (DeLaN), Lutter, Ritter & Peters, ICLR 2019
# ("Deep Lagrangian Networks: Using Physics as Model Prior for Deep Learning").
# The official repo is JAX-primary but ships a full, actively-maintained
# PyTorch reference implementation (`deep_lagrangian_networks/DeLaN_model.py`,
# used by `example_DeLaN.py`). This file is that PyTorch implementation,
# copied verbatim (imports untouched -- it only imports torch/numpy, both
# base libs) with the module docstring below replacing the file's original
# top-of-file comments. `DeepLagrangianNetwork` learns a physically-structured
# forward pass: two small "LagrangianLayer" MLP towers (with hand-derived
# per-layer Jacobians propagated alongside the activations) produce the
# Cholesky factor of a positive-definite mass matrix H(q) and a potential
# energy V(q); the analytic Euler-Lagrange equation is then assembled from
# H, its time/configuration derivatives, and grad(V) to predict the joint
# torque tau for a given (q, qd, qdd) trajectory point -- i.e. it is a
# learned rigid-body inverse-dynamics model with the Lagrangian mechanics
# structure baked into the forward computation graph (not just the loss).

# ruff: noqa: E701, E741 -- vendored verbatim from the upstream repo (see header);
# single-letter `l` matches the paper's Cholesky-factor notation and the dense
# one-line ternaries are the original init-gain selection code. Same exemption
# already applies to menagerie/classics/*.py for vendored/ported model source.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

MENAGERIE_ZOO = "vendored-pytorch"


class LowTri:
    def __init__(self, m):
        # Calculate lower triangular matrix indices using numpy
        self._m = m
        self._idx = np.tril_indices(self._m)

    def __call__(self, l):
        batch_size = l.shape[0]
        self._L = torch.zeros(batch_size, self._m, self._m).type_as(l)

        # Assign values to matrix:
        self._L[:batch_size, self._idx[0], self._idx[1]] = l[:]
        return self._L[:batch_size]


class SoftplusDer(nn.Module):
    def __init__(self, beta=1.0):
        super(SoftplusDer, self).__init__()
        self._beta = beta

    def forward(self, x):
        cx = torch.clamp(x, -20.0, 20.0)
        exp_x = torch.exp(self._beta * cx)
        out = exp_x / (exp_x + 1.0)

        if torch.isnan(out).any():
            print("SoftPlus Forward output is NaN.")
        return out


class ReLUDer(nn.Module):
    def __init__(self):
        super(ReLUDer, self).__init__()

    def forward(self, x):
        return torch.ceil(torch.clamp(x, 0, 1))


class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()

    def forward(self, x):
        return x


class LinearDer(nn.Module):
    def __init__(self):
        super(LinearDer, self).__init__()

    def forward(self, x):
        return torch.clamp(x, 1, 1)


class Cos(nn.Module):
    def __init__(self):
        super(Cos, self).__init__()

    def forward(self, x):
        return torch.cos(x)


class CosDer(nn.Module):
    def __init__(self):
        super(CosDer, self).__init__()

    def forward(self, x):
        return -torch.sin(x)


class LagrangianLayer(nn.Module):
    def __init__(self, input_size, n_dof, activation="ReLu"):
        super(LagrangianLayer, self).__init__()

        # Create layer weights and biases:
        self.n_dof = n_dof
        self.weight = nn.Parameter(torch.Tensor(n_dof, input_size))
        self.bias = nn.Parameter(torch.Tensor(n_dof))

        # Initialize activation function and its derivative:
        if activation == "ReLu":
            self.g = nn.ReLU()
            self.g_prime = ReLUDer()

        elif activation == "SoftPlus":
            self.softplus_beta = 1.0
            self.g = nn.Softplus(beta=self.softplus_beta)
            self.g_prime = SoftplusDer(beta=self.softplus_beta)

        elif activation == "Cos":
            self.g = Cos()
            self.g_prime = CosDer()

        elif activation == "Linear":
            self.g = Linear()
            self.g_prime = LinearDer()

        else:
            raise ValueError(
                "Activation Type must be in ['Linear', 'ReLu', 'SoftPlus', 'Cos'] but is {0}".format(
                    activation
                )
            )

    def forward(self, q, der_prev):
        # Apply Affine Transformation:
        a = F.linear(q, self.weight, self.bias)
        out = self.g(a)
        der = torch.matmul(self.g_prime(a).view(-1, self.n_dof, 1) * self.weight, der_prev)
        return out, der


class DeepLagrangianNetwork(nn.Module):
    def __init__(self, n_dof, **kwargs):
        super(DeepLagrangianNetwork, self).__init__()

        # Read optional arguments:
        self.n_width = kwargs.get("n_width", 128)
        self.n_hidden = kwargs.get("n_depth", 1)
        self._b0 = kwargs.get("b_init", 0.1)
        self._b0_diag = kwargs.get("b_diag_init", 0.1)

        self._w_init = kwargs.get("w_init", "xavier_normal")
        self._g_hidden = kwargs.get("g_hidden", np.sqrt(2.0))
        self._g_output = kwargs.get("g_hidden", 0.125)
        self._p_sparse = kwargs.get("p_sparse", 0.2)
        self._epsilon = kwargs.get("diagonal_epsilon", 1.0e-5)

        # Construct Weight Initialization:
        if self._w_init == "xavier_normal":
            # Construct initialization function:
            def init_hidden(layer):
                # Set the Hidden Gain:
                if self._g_hidden <= 0.0:
                    hidden_gain = torch.nn.init.calculate_gain("relu")
                else:
                    hidden_gain = self._g_hidden

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.xavier_normal_(layer.weight, hidden_gain)

            def init_output(layer):
                # Set Output Gain:
                if self._g_output <= 0.0:
                    output_gain = torch.nn.init.calculate_gain("linear")
                else:
                    output_gain = self._g_output

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.xavier_normal_(layer.weight, output_gain)

        elif self._w_init == "orthogonal":
            # Construct initialization function:
            def init_hidden(layer):
                # Set the Hidden Gain:
                if self._g_hidden <= 0.0:
                    hidden_gain = torch.nn.init.calculate_gain("relu")
                else:
                    hidden_gain = self._g_hidden

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.orthogonal_(layer.weight, hidden_gain)

            def init_output(layer):
                # Set Output Gain:
                if self._g_output <= 0.0:
                    output_gain = torch.nn.init.calculate_gain("linear")
                else:
                    output_gain = self._g_output

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.orthogonal_(layer.weight, output_gain)

        elif self._w_init == "sparse":
            assert self._p_sparse < 1.0 and self._p_sparse >= 0.0

            # Construct initialization function:
            def init_hidden(layer):
                p_non_zero = self._p_sparse
                hidden_std = self._g_hidden

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.sparse_(layer.weight, p_non_zero, hidden_std)

            def init_output(layer):
                p_non_zero = self._p_sparse
                output_std = self._g_output

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.sparse_(layer.weight, p_non_zero, output_std)

        else:
            raise ValueError(
                "Weight Initialization Type must be in ['xavier_normal', 'orthogonal', 'sparse'] but is {0}".format(
                    self._w_init
                )
            )

        # Compute In- / Output Sizes:
        self.n_dof = n_dof
        self.m = int((n_dof**2 + n_dof) / 2)

        # Compute non-zero elements of L:
        l_output_size = int((self.n_dof**2 + self.n_dof) / 2)
        l_lower_size = l_output_size - self.n_dof

        # Calculate the indices of the diagonal elements of L:
        idx_diag = np.arange(self.n_dof) + 1
        idx_diag = idx_diag * (idx_diag + 1) / 2 - 1

        # Calculate the indices of the off-diagonal elements of L:
        idx_tril = np.extract(
            [x not in idx_diag for x in np.arange(l_output_size)], np.arange(l_output_size)
        )

        # Indexing for concatenation of l_o  and l_d
        cat_idx = np.hstack((idx_diag, idx_tril))
        order = np.argsort(cat_idx)
        self._idx = np.arange(cat_idx.size)[order]

        # create it once and only apply repeat, this may decrease memory allocation
        self._eye = torch.eye(self.n_dof).view(1, self.n_dof, self.n_dof)
        self.low_tri = LowTri(self.n_dof)

        # Create Network:
        self.layers = nn.ModuleList()
        non_linearity = kwargs.get("activation", "ReLu")

        # Create Input Layer:
        self.layers.append(LagrangianLayer(self.n_dof, self.n_width, activation=non_linearity))
        init_hidden(self.layers[-1])

        # Create Hidden Layer:
        for _ in range(1, self.n_hidden):
            self.layers.append(
                LagrangianLayer(self.n_width, self.n_width, activation=non_linearity)
            )
            init_hidden(self.layers[-1])

        # Create output Layer:
        self.net_g = LagrangianLayer(self.n_width, 1, activation="Linear")
        init_output(self.net_g)

        self.net_lo = LagrangianLayer(self.n_width, l_lower_size, activation="Linear")
        init_hidden(self.net_lo)

        # The diagonal must be non-negative. Therefore, the non-linearity is set to ReLu.
        self.net_ld = LagrangianLayer(self.n_width, self.n_dof, activation="ReLu")
        init_hidden(self.net_ld)
        torch.nn.init.constant_(self.net_ld.bias, self._b0_diag)

    def forward(self, q, qd, qdd):
        out = self._dyn_model(q, qd, qdd)
        tau_pred = out[0]
        dEdt = out[6] + out[7]

        return tau_pred, dEdt

    def _dyn_model(self, q, qd, qdd):
        qd_3d = qd.view(-1, self.n_dof, 1)
        qd_4d = qd.view(-1, 1, self.n_dof, 1)

        # Create initial derivative of dq/dq.
        der = self._eye.repeat(q.shape[0], 1, 1).type_as(q)

        # Compute shared network between l & g:
        y, der = self.layers[0](q, der)

        for i in range(1, len(self.layers)):
            y, der = self.layers[i](y, der)

        # Compute the network heads including the corresponding derivative:
        l_lower, der_l_lower = self.net_lo(y, der)
        l_diag, der_l_diag = self.net_ld(y, der)

        # Compute the potential energy and the gravitational force:
        V, der_V = self.net_g(y, der)
        V = V.squeeze()
        g = der_V.squeeze()

        # Assemble l and der_l
        l_diag = l_diag
        l = torch.cat((l_diag, l_lower), 1)[:, self._idx]
        der_l = torch.cat((der_l_diag, der_l_lower), 1)[:, self._idx, :]

        # Compute H:
        L = self.low_tri(l)
        LT = L.transpose(dim0=1, dim1=2)
        H = torch.matmul(L, LT) + self._epsilon * torch.eye(self.n_dof).type_as(L)

        # Calculate dH/dt
        Ldt = self.low_tri(torch.matmul(der_l, qd_3d).view(-1, self.m))
        Hdt = torch.matmul(L, Ldt.transpose(dim0=1, dim1=2)) + torch.matmul(Ldt, LT)

        # Calculate dH/dq:
        Ldq = self.low_tri(der_l.transpose(2, 1).reshape(-1, self.m)).reshape(
            -1, self.n_dof, self.n_dof, self.n_dof
        )
        Hdq = torch.matmul(Ldq, LT.view(-1, 1, self.n_dof, self.n_dof)) + torch.matmul(
            L.view(-1, 1, self.n_dof, self.n_dof), Ldq.transpose(2, 3)
        )

        # Compute the Coriolis & Centrifugal forces:
        Hdt_qd = torch.matmul(Hdt, qd_3d).view(-1, self.n_dof)
        quad_dq = torch.matmul(qd_4d.transpose(dim0=2, dim1=3), torch.matmul(Hdq, qd_4d)).view(
            -1, self.n_dof
        )
        c = Hdt_qd - 1.0 / 2.0 * quad_dq

        # Compute the Torque using the inverse model:
        H_qdd = torch.matmul(H, qdd.view(-1, self.n_dof, 1)).view(-1, self.n_dof)
        tau_pred = H_qdd + c + g

        # Compute kinetic energy T
        H_qd = torch.matmul(H, qd_3d).view(-1, self.n_dof)
        T = (
            1.0
            / 2.0
            * torch.matmul(qd_4d.transpose(dim0=2, dim1=3), H_qd.view(-1, 1, self.n_dof, 1)).view(
                -1
            )
        )

        # Compute dT/dt:
        qd_H_qdd = torch.matmul(
            qd_4d.transpose(dim0=2, dim1=3), H_qdd.view(-1, 1, self.n_dof, 1)
        ).view(-1)
        qd_Hdt_qd = torch.matmul(
            qd_4d.transpose(dim0=2, dim1=3), Hdt_qd.view(-1, 1, self.n_dof, 1)
        ).view(-1)
        dTdt = qd_H_qdd + 0.5 * qd_Hdt_qd

        # Compute dV/dt
        dVdt = torch.matmul(qd_4d.transpose(dim0=2, dim1=3), g.view(-1, 1, self.n_dof, 1)).view(-1)
        return tau_pred, H, c, g, T, V, dTdt, dVdt

    def inv_dyn(self, q, qd, qdd):
        out = self._dyn_model(q, qd, qdd)
        tau_pred = out[0]
        return tau_pred

    def for_dyn(self, q, qd, tau):
        out = self._dyn_model(q, qd, torch.zeros_like(q))
        H, c, g = out[1], out[2], out[3]

        # Compute Acceleration, e.g., forward model:
        invH = torch.inverse(H)
        qdd_pred = torch.matmul(invH, (tau - c - g).view(-1, self.n_dof, 1)).view(-1, self.n_dof)
        return qdd_pred

    def energy(self, q, qd):
        out = self._dyn_model(q, qd, torch.zeros_like(q))
        E = out[4] + out[5]
        return E

    def energy_dot(self, q, qd, qdd):
        out = self._dyn_model(q, qd, qdd)
        dEdt = out[6] + out[7]
        return dEdt


# ---------------------------------------------------------------------------
# Menagerie staging wrapper: DeLaN's forward() needs three co-shaped tensors
# (q, qd, qdd) for an n_dof-joint system, so it is staged as a module (not a
# single-input recipe). n_dof=2 mirrors the repo's own 2-DOF example
# character task; n_width/n_depth are shrunk from the repo defaults
# (n_width=128->16, n_depth kept at the repo default of 2) for a fast trace.
# ---------------------------------------------------------------------------
def build_delan():
    torch.manual_seed(0)
    model = DeepLagrangianNetwork(
        n_dof=2,
        n_width=16,
        n_depth=2,
        diagonal_epsilon=0.01,
        activation="SoftPlus",
        b_init=1.0e-4,
        b_diag_init=0.001,
        w_init="xavier_normal",
    )
    model.eval()
    return model


def example_input_delan():
    torch.manual_seed(0)
    batch, n_dof = 2, 2
    q = torch.randn(batch, n_dof)
    qd = torch.randn(batch, n_dof)
    qdd = torch.randn(batch, n_dof)
    return (q, qd, qdd)


MENAGERIE_ENTRIES = [
    ("DeLaN", "build_delan", "example_input_delan", 2019, MENAGERIE_ZOO),
]
