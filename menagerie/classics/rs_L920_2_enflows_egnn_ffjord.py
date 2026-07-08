# SOURCE: vendored from vgsatorras/en_flows @ main
#   (repo: https://github.com/vgsatorras/en_flows)
#   (paper: "E(n) Equivariant Normalizing Flows", Satorras et al., NeurIPS 2021,
#    https://arxiv.org/abs/2105.09016)
#
# E-NF is a continuous-time normalizing flow whose ODE vector field is an
# E(n)-equivariant graph neural network (EGNN). This file vendors the
# ACTUAL classes the repo instantiates together for the 'egnn_dynamics'
# model variant (see dw4_experiment/models.py get_model(), 'egnn_dynamics'
# branch): `flow = FFJORD(EGNN_dynamics(...), trace_method='hutch', ...)`.
#
#   - egnn/gcl.py: E_GCL (equivariant graph conv layer) verbatim.
#   - egnn/models.py: EGNN, EGNN_dynamics verbatim (the GNN_dynamics /
#     EGNN_dynamics_QM9 branches and helper GNN/GCL classes not used by the
#     'egnn_dynamics' path are omitted for a focused staged module; EGNN and
#     EGNN_dynamics themselves are transcribed verbatim, unmodified).
#   - flows/ffjord.py: FFJORD + ODEfunc verbatim (the continuous-time
#     free-form Jacobian flow that calls the EGNN dynamics as f(t,x) at
#     every ODE step and estimates the log-det-Jacobian via Hutchinson's
#     trace estimator).
#   - flows/utils.py: remove_mean (helper used by EGNN_dynamics.forward)
#     verbatim.
#
# Only import paths were adjusted (inlined into a single file) and
# `flows.ffjord`'s `from torchdiffeq import odeint_adjoint as odeint` was
# kept unchanged -- torchdiffeq is a real installed lightweight ODE-solver
# dependency (already used elsewhere in this menagerie, e.g.
# menagerie/classics/rs_L862_3_confvae.py, rs_L205_2.py, rs_L426_scnode.py).
# No architecture code was rewritten, approximated, or simplified.

import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# Verbatim from flows/utils.py (only the helper EGNN_dynamics.forward needs)
# ---------------------------------------------------------------------------


def remove_mean(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    x = x - mean
    return x


# ---------------------------------------------------------------------------
# Verbatim from egnn/gcl.py
# ---------------------------------------------------------------------------


def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


class E_GCL(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.

    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        edges_in_d=0,
        nodes_att_dim=0,
        act_fn=nn.SiLU(),
        recurrent=True,
        attention=False,
        clamp=False,
        norm_diff=True,
        tanh=False,
        coords_range=1,
        agg="sum",
    ):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.agg_type = agg
        self.tanh = tanh
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = coords_range

        self.coord_mlp = nn.Sequential(*coord_mlp)
        self.clamp = clamp

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr, edge_mask):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val

        if edge_mask is not None:
            out = out * edge_mask
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.recurrent:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, radial, edge_feat, node_mask, edge_mask):
        row, col = edge_index
        if self.tanh:
            trans = coord_diff * self.coord_mlp(edge_feat) * self.coords_range
        else:
            trans = coord_diff * self.coord_mlp(edge_feat)
        if edge_mask is not None:
            trans = trans * edge_mask

        if self.agg_type == "sum":
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.agg_type == "mean":
            if node_mask is not None:
                agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
                M = unsorted_segment_sum(node_mask[col], row, num_segments=coord.size(0))
                agg = agg / (M - 1)
            else:
                agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception("Wrong coordinates aggregation type")
        coord = coord + agg
        return coord

    def forward(
        self, h, edge_index, coord, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None
    ):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr, edge_mask)
        coord = self.coord_model(
            coord, edge_index, coord_diff, radial, edge_feat, node_mask, edge_mask
        )

        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        if node_mask is not None:
            h = h * node_mask
            coord = coord * node_mask
        return h, coord, edge_attr

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)

        norm = torch.sqrt(radial + 1e-8)
        coord_diff = coord_diff / (norm + 1)

        return radial, coord_diff


# ---------------------------------------------------------------------------
# Verbatim from egnn/models.py (EGNN + EGNN_dynamics only; the GNN/GCL
# fallback path and the QM9-conditioning variant are omitted -- not part of
# the 'egnn_dynamics' model this file stages)
# ---------------------------------------------------------------------------


class EGNN(nn.Module):
    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        device="cpu",
        act_fn=nn.SiLU(),
        n_layers=4,
        recurrent=True,
        attention=False,
        norm_diff=True,
        out_node_nf=None,
        tanh=False,
        coords_range=15,
        agg="sum",
    ):
        super(EGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range) / self.n_layers
        if agg == "mean":
            self.coords_range_layer = self.coords_range_layer * 19
        ### Encoder
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module(
                "gcl_%d" % i,
                E_GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    edges_in_d=in_edge_nf,
                    act_fn=act_fn,
                    recurrent=recurrent,
                    attention=attention,
                    norm_diff=norm_diff,
                    tanh=tanh,
                    coords_range=self.coords_range_layer,
                    agg=agg,
                ),
            )

        self.to(self.device)

    def forward(self, h, x, edges, edge_attr=None, node_mask=None, edge_mask=None):
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](
                h, edges, x, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask
            )
        h = self.embedding_out(h)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h, x


class EGNN_dynamics(nn.Module):
    def __init__(
        self,
        n_particles,
        n_dimension,
        hidden_nf=64,
        device="cpu",
        act_fn=torch.nn.SiLU(),
        n_layers=4,
        recurrent=True,
        attention=False,
        condition_time=True,
        tanh=False,
        mode="egnn_dynamics",
        agg="sum",
    ):
        super().__init__()
        self.mode = mode
        if mode == "egnn_dynamics":
            self.egnn = EGNN(
                in_node_nf=1,
                in_edge_nf=1,
                hidden_nf=hidden_nf,
                device=device,
                act_fn=act_fn,
                n_layers=n_layers,
                recurrent=recurrent,
                attention=attention,
                tanh=tanh,
                agg=agg,
            )

        self.device = device
        self._n_particles = n_particles
        self._n_dimension = n_dimension
        self.edges = self._create_edges()
        self._edges_dict = {}
        self.condition_time = condition_time

    def forward(self, t, xs):
        n_batch = xs.shape[0]
        edges = self._cast_edges2batch(self.edges, n_batch, self._n_particles)
        edges = [edges[0], edges[1]]
        x = xs.view(n_batch * self._n_particles, self._n_dimension).clone()
        h = torch.ones(n_batch * self._n_particles, 1).to(self.device)
        if self.condition_time:
            h = h * t

        edge_attr = torch.sum((x[edges[0]] - x[edges[1]]) ** 2, dim=1, keepdim=True)
        _, x_final = self.egnn(h, x, edges, edge_attr=edge_attr)
        vel = x_final - x

        vel = vel.view(n_batch, self._n_particles, self._n_dimension)
        vel = remove_mean(vel)
        return vel

    def _create_edges(self):
        rows, cols = [], []
        for i in range(self._n_particles):
            for j in range(i + 1, self._n_particles):
                rows.append(i)
                cols.append(j)
                rows.append(j)
                cols.append(i)
        return [torch.LongTensor(rows), torch.LongTensor(cols)]

    def _cast_edges2batch(self, edges, n_batch, n_nodes):
        if n_batch not in self._edges_dict:
            self._edges_dict = {}
            rows, cols = edges
            rows_total, cols_total = [], []
            for i in range(n_batch):
                rows_total.append(rows + i * n_nodes)
                cols_total.append(cols + i * n_nodes)
            rows_total = torch.cat(rows_total).to(self.device)
            cols_total = torch.cat(cols_total).to(self.device)

            self._edges_dict[n_batch] = [rows_total, cols_total]
        return self._edges_dict[n_batch]


# ---------------------------------------------------------------------------
# Verbatim from flows/ffjord.py
# ---------------------------------------------------------------------------


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(-1)


class FFJORD(torch.nn.Module):
    """
    Continuous-time flow FFJORD [1].

    Args:
        dynamics (nn.Module): The ODE dynamics function f(t,x).
        trace_method (str): The trace estimation method. One of {'exact', 'hutch'}.

    References:
        [1] FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models,
            Grathwohl et al., 2019, https://arxiv.org/abs/1810.01367
    """

    def __init__(
        self, dynamics, trace_method="hutch", ode_regularization=0, hutch_noise="gaussian"
    ):
        super(FFJORD, self).__init__()

        self.odefunc = ODEfunc(
            dynamics,
            method=trace_method,
            ode_regularization=ode_regularization,
            hutch_noise=hutch_noise,
        )

        self.set_integration_time()
        self.set_odeint()

    def set_integration_time(self, times=[0.0, 1.0]):
        device = next(iter(self.odefunc.parameters())).device
        self.register_buffer("int_time", torch.tensor(times, dtype=torch.float, device=device))
        self.register_buffer(
            "inv_int_time", torch.tensor(list(reversed(times)), dtype=torch.float, device=device)
        )

    def set_odeint(self, method="dopri5", rtol=1e-4, atol=1e-4):
        self.method = method
        self._atol = atol
        self._rtol = rtol
        self._atol_test = 1e-7
        self._rtol_test = 1e-7

    def set_trace(self, trace):
        assert trace == "exact" or trace == "hutch"
        self.odefunc.method = trace

    @property
    def atol(self):
        return self._atol if self.training else self._atol_test

    @property
    def rtol(self):
        return self._rtol if self.training else self._rtol_test

    def forward(self, x, node_mask=None, edge_mask=None, context=None):
        ldj = x.new_zeros(x.shape[0])
        reg_term = x.new_zeros(x.shape[0])

        state = (x, ldj, reg_term)

        self.odefunc.before_odeint(x)

        if node_mask is not None or edge_mask is not None or context is not None:
            self.odefunc.dynamics.forward = self.odefunc.dynamics.wrap_forward(
                node_mask, edge_mask, context
            )

        statet = odeint(
            self.odefunc, state, self.int_time, method=self.method, rtol=self.rtol, atol=self.atol
        )

        zt, ldjt, reg_termt = statet
        z, ldj, reg_term = zt[-1], ldjt[-1], reg_termt[-1]
        return z, ldj, reg_term


class ODEfunc(torch.nn.Module):
    def __init__(self, dynamics, method="hutch", ode_regularization=0, hutch_noise="gaussian"):
        assert method in {"exact", "hutch"}
        super(ODEfunc, self).__init__()
        self.dynamics = dynamics
        self.hutch_noise = hutch_noise
        self.method = method
        self.ode_regularization = ode_regularization

    def set_trace_exact(self):
        self.method = "exact"

    def set_trace_hutch(self):
        self.method = "hutch"

    @staticmethod
    def hutch_trace(f, y, e=None):
        """Hutchinson's estimator for the Jacobian trace"""
        e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
        e_dzdx_e = e_dzdx * e
        approx_tr_dzdx = sum_except_batch(e_dzdx_e)
        return approx_tr_dzdx

    @staticmethod
    def only_frobenius(f, y, e=None):
        """Hutchinson's estimator for the Jacobian trace"""
        e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
        frobenius = sum_except_batch(e_dzdx.pow(2))
        return frobenius

    @staticmethod
    def hutch_trace_and_frobenius(f, y, e=None):
        """Hutchinson's estimator for the Jacobian trace"""
        e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
        frobenius = sum_except_batch(e_dzdx.pow(2))
        e_dzdx_e = e_dzdx * e
        approx_tr_dzdx = sum_except_batch(e_dzdx_e)
        return approx_tr_dzdx, frobenius

    @staticmethod
    def exact_trace(f, y):
        """Exact Jacobian trace"""
        import itertools

        dims = y.size()[1:]
        tr_dzdx = 0.0
        dim_ranges = [range(d) for d in dims]
        for idcs in itertools.product(*dim_ranges):
            batch_idcs = (slice(None),) + idcs
            tr_dzdx += torch.autograd.grad(f[batch_idcs].sum(), y, create_graph=True)[0][batch_idcs]
        return tr_dzdx

    def before_odeint(self, tensor):
        self.num_evals = 0
        if self.method == "hutch":
            if self.hutch_noise == "gaussian":
                # With _eps ~ Normal(0, 1).
                self._eps = torch.randn_like(tensor)
            elif self.hutch_noise == "bernoulli":
                # With _eps ~ Rademacher (== Bernoulli on -1 +1 with 50/50 chance).
                self._eps = torch.randint(low=0, high=2, size=tensor.size()).to(tensor) * 2 - 1
            else:
                raise Exception("Wrong hutchinson noise type")

    def forward(self, t, state):
        x, ldj, reg_term = state

        self.num_evals += 1
        with torch.set_grad_enabled(True):
            x.requires_grad_(True)
            t.requires_grad_(True)

            # We always need the dynamics :).
            dx = self.dynamics(t, x)

            if self.ode_regularization > 0:
                # L2-squared norm of (dx)
                dx2 = sum_except_batch(dx.pow(2))

                # If trace is computed exact, frobenius norm is still estimated.
                if self.method == "exact":
                    ldj = self.exact_trace(dx, x)
                    frobenius = self.only_frobenius(dx, x, e=self._eps)

                # Combined computation for trace and frobenius estimators.
                elif self.method == "hutch":
                    ldj, frobenius = self.hutch_trace_and_frobenius(dx, x, e=self._eps)

                reg_term = frobenius + dx2

            else:
                if self.method == "exact":
                    ldj = self.exact_trace(dx, x)

                elif self.method == "hutch":
                    ldj = self.hutch_trace(dx, x, e=self._eps)

                # No regularization terms, set to zero.
                reg_term = torch.zeros_like(ldj)

        return dx, ldj, reg_term


# ---------------------------------------------------------------------------
# Menagerie staging entry points
# ---------------------------------------------------------------------------


def build_enflows_egnn_ffjord():
    """Tiny-size real E-NF flow: FFJORD continuous-time flow wrapping the
    real EGNN_dynamics E(n)-equivariant vector field, exactly as constructed
    in the repo's dw4_experiment/models.py get_model() 'egnn_dynamics'
    branch: flow = FFJORD(EGNN_dynamics(...), trace_method='hutch', ...)."""
    dynamics = EGNN_dynamics(
        n_particles=4,
        n_dimension=2,
        hidden_nf=8,
        device="cpu",
        act_fn=torch.nn.SiLU(),
        n_layers=2,
        recurrent=True,
        attention=False,
        condition_time=True,
        tanh=False,
        mode="egnn_dynamics",
        agg="sum",
    )
    flow = FFJORD(dynamics, trace_method="hutch", ode_regularization=0, hutch_noise="gaussian")
    # Fast fixed-step Euler solve (single step over int_time=[0.0, 1.0]),
    # for a quick random-init trace. FFJORD.forward() always calls the
    # module-level `odeint_adjoint as odeint` import (no use_adjoint
    # toggle in this repo); 'euler' with the default single-interval grid
    # keeps the ODE solve to one dynamics evaluation.
    flow.set_odeint(method="euler", rtol=1e-2, atol=1e-2)
    return flow


def example_input_enflows_egnn_ffjord():
    """A tiny batch of 4-particle, 2D point clouds (mirrors the repo's
    dw4/lj13 toy-system inputs to the flow: main_dw4_lj13.py reshapes each
    batch to (batch_size, n_particles, n_dims) via `batch.view(...)` before
    calling `flow(batch)`; EGNN_dynamics.forward flattens/unflattens this
    same shape internally)."""
    torch.manual_seed(0)
    batch, n_particles, n_dimension = 2, 4, 2
    x = torch.randn(batch, n_particles, n_dimension)
    return (x,)


MENAGERIE_ENTRIES = [
    (
        "E-NF-EGNN-FFJORD",
        build_enflows_egnn_ffjord,
        example_input_enflows_egnn_ffjord,
        2021,
        "CODE",
    ),
]
