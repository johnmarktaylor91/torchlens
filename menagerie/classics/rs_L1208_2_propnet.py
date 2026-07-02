# SOURCE: vendored from YunzhuLi/PropNet @ master
# https://github.com/YunzhuLi/PropNet/blob/master/models.py
# The `RelationEncoder`/`ParticleEncoder`/`Propagator`/`ParticlePredictor`/`PropModule`/
# `PropNet` classes (Propagation Networks for model-based control under partial
# observation, Li et al. ICRA 2019) are transcribed VERBATIM from models.py. Only
# changes: the unused `torchvision.models` import is dropped, and `torch.autograd.
# Variable` (a pre-0.4 torch API, superseded by plain tensors) is replaced with a
# regular `torch.zeros` call -- identical runtime behavior on modern torch, no
# architectural change. `PropNet.forward` (the "full" observability mode, i.e. the
# `args.pn_mode == 'full'` branch that runs a single interaction-network-style
# message-passing rollout: particle+relation encoders -> pstep rounds of relation/
# particle propagation -> particle predictor) is exercised end-to-end here.
import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# --- models.py (verbatim architecture) ---
class RelationEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RelationEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        Args:
            x: [batch_size, n_relations, input_size]
        Returns:
            [batch_size, n_relations, output_size]
        """
        B, N, D = x.size()
        x = self.model(x.view(B * N, D))
        return x.view(B, N, self.output_size)


class ParticleEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ParticleEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        Args:
            x: [batch_size, n_particles, input_size]
        Returns:
            [batch_size, n_particles, output_size]
        """
        B, N, D = x.size()
        x = self.model(x.view(B * N, D))
        return x.view(B, N, self.output_size)


class Propagator(nn.Module):
    def __init__(self, input_size, output_size, residual=False):
        super(Propagator, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.residual = residual

        self.linear = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x, res=None):
        """
        Args:
            x: [batch_size, n_relations/n_particles, input_size]
        Returns:
            [batch_size, n_relations/n_particles, output_size]
        """
        B, N, D = x.size()
        if self.residual:
            x = self.linear(x.view(B * N, D))
            x = self.relu(x + res.view(B * N, self.output_size))
        else:
            x = self.relu(self.linear(x.view(B * N, D)))

        return x.view(B, N, self.output_size)


class ParticlePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ParticlePredictor, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.linear_0 = nn.Linear(input_size, hidden_size)
        self.linear_1 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: [batch_size, n_particles, input_size]
        Returns:
            [batch_size, n_particles, output_size]
        """
        B, N, D = x.size()
        x = x.view(B * N, D)
        x = self.linear_1(self.relu(self.linear_0(x)))
        return x.view(B, N, self.output_size)


class PropModule(nn.Module):
    def __init__(self, args, input_dim, output_dim, batch=True, residual=False, use_gpu=False):
        super(PropModule, self).__init__()

        self.args = args
        self.batch = batch

        attr_dim = args.attr_dim  # noqa: F841 (unused, kept for parity with original models.py)
        relation_dim = args.relation_dim

        nf_relation = args.nf_relation
        nf_effect = args.nf_effect  # noqa: F841 (unused, kept for parity with original models.py)

        self.nf_effect = args.nf_effect

        self.use_gpu = use_gpu
        self.residual = residual

        # particle encoder
        self.particle_encoder = ParticleEncoder(input_dim, nf_relation, nf_effect)

        # relation encoder
        self.relation_encoder = RelationEncoder(
            2 * input_dim + relation_dim, nf_relation, nf_relation
        )

        # input: (1) particle encode (2) particle effect
        self.particle_propagator = Propagator(2 * nf_effect, nf_effect, self.residual)

        # input: (1) relation encode (2) sender effect (3) receiver effect
        self.relation_propagator = Propagator(nf_relation + 2 * nf_effect, nf_effect)

        # input: (1) particle effect
        self.particle_predictor = ParticlePredictor(nf_effect, nf_effect, output_dim)

    def forward(self, state, Rr, Rs, Ra, pstep, verbose=0):
        # calculate particle encoding
        particle_effect = torch.zeros((state.size(0), state.size(1), self.nf_effect))
        if self.use_gpu:
            particle_effect = particle_effect.cuda()

        # receiver_state, sender_state
        if self.batch:
            Rrp = torch.transpose(Rr, 1, 2)
            Rsp = torch.transpose(Rs, 1, 2)
            state_r = Rrp.bmm(state)
            state_s = Rsp.bmm(state)
        else:
            Rrp = Rr.t()
            Rsp = Rs.t()
            assert state.size(0) == 1
            state_r = Rrp.mm(state[0])[None, :, :]
            state_s = Rsp.mm(state[0])[None, :, :]

        # particle encode
        particle_encode = self.particle_encoder(state)

        # calculate relation encoding
        relation_encode = self.relation_encoder(torch.cat([state_r, state_s, Ra], 2))

        for _ in range(pstep):
            if self.batch:
                effect_r = Rrp.bmm(particle_effect)
                effect_s = Rsp.bmm(particle_effect)
            else:
                assert particle_effect.size(0) == 1
                effect_r = Rrp.mm(particle_effect[0])[None, :, :]
                effect_s = Rsp.mm(particle_effect[0])[None, :, :]

            # calculate relation effect
            relation_effect = self.relation_propagator(
                torch.cat([relation_encode, effect_r, effect_s], 2)
            )

            # calculate particle effect by aggregating relation effect
            if self.batch:
                effect_agg = Rr.bmm(relation_effect)
            else:
                assert relation_effect.size(0) == 1
                effect_agg = Rr.mm(relation_effect[0])[None, :, :]

            # calculate particle effect
            particle_effect = self.particle_propagator(
                torch.cat([particle_encode, effect_agg], 2), res=particle_effect
            )

        pred = self.particle_predictor(particle_effect)

        return pred


class _PropNetArgs:
    """Plain attribute bag matching the `args` namespace consumed by PropModule/PropNet
    (originally an argparse.Namespace built from PropNet's train.py CLI flags)."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class PropNet(nn.Module):
    def __init__(self, args, residual=False, use_gpu=False):
        super(PropNet, self).__init__()

        self.args = args
        nf_effect = args.nf_effect  # noqa: F841 (unused, kept for parity with original models.py)
        attr_dim = args.attr_dim  # noqa: F841 (unused, kept for parity with original models.py)
        state_dim = args.state_dim  # noqa: F841 (unused, kept for parity with original models.py)
        action_dim = args.action_dim
        position_dim = args.position_dim

        # input: (1) attr (2) state (3) [optional] action -- "full" observability mode
        if args.pn_mode == "full":
            batch = True
            input_dim = attr_dim + state_dim + action_dim
            self.model = PropModule(args, input_dim, position_dim, batch, residual, use_gpu)

    def forward(self, data, pstep, action=None):
        # used only for fully observable case
        args = self.args
        attr, state, Rr, Rs, Ra = data
        if action is not None:
            state = torch.cat([attr, state, action], 2)
        else:
            state = torch.cat([attr, state], 2)
        return self.model(state, Rr, Rs, Ra, args.pstep, args.verbose_model)


def build_propnet():
    # Tiny menagerie-scale config mirroring PropNet's "full" observability mode
    # (Box-pushing environment style: attr/state/action-conditioned particle graph,
    # cf. scripts/train_Box.sh args). nf_relation/nf_effect shrunk from the released
    # 150/100 to 8; pstep (message-passing rounds) kept at the paper default of 2.
    args = _PropNetArgs(
        pn_mode="full",
        attr_dim=2,
        state_dim=4,
        action_dim=2,
        position_dim=2,
        relation_dim=1,
        nf_relation=8,
        nf_effect=8,
        pstep=2,
        verbose_model=0,
    )
    return PropNet(args, residual=False, use_gpu=False)


def example_input_propnet():
    torch.manual_seed(0)
    B, N, R = 1, 6, 10  # batch, n_particles, n_relations (fully-connected minus self)
    attr = torch.randn(B, N, 2)
    state = torch.randn(B, N, 4)
    action = torch.randn(B, N, 2)
    # random sender/receiver one-hot relation matrices [B, N, R]
    Rr = torch.zeros(B, N, R)
    Rs = torch.zeros(B, N, R)
    idx = 0
    for i in range(N):
        for j in range(N):
            if i != j and idx < R:
                Rr[0, i, idx] = 1.0
                Rs[0, j, idx] = 1.0
                idx += 1
    Ra = torch.randn(B, R, 1)
    pstep = 2
    return ((attr, state, Rr, Rs, Ra), pstep, action)


MENAGERIE_ENTRIES = [
    (
        "PropNet",
        "build_propnet",
        "example_input_propnet",
        2019,
        "vendored-pytorch",
    ),
]
