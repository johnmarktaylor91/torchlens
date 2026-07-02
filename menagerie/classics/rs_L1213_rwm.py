# SOURCE: vendored from leggedrobotics/rsl_rl_rwm @ main
# https://github.com/leggedrobotics/rsl_rl_rwm/blob/main/rsl_rl/modules/system_dynamics.py
# https://github.com/leggedrobotics/rsl_rl_rwm/blob/main/rsl_rl/modules/architectures/rnn.py
# https://github.com/leggedrobotics/rsl_rl_rwm/blob/main/rsl_rl/modules/architectures/mlp.py
# rsl_rl_rwm is the model package that leggedrobotics/robotic_world_model_lite installs
# via `rsl-rl-lib @ git+https://github.com/leggedrobotics/rsl_rl_rwm.git@main` (see
# robotic_world_model_lite/setup.py). `SystemDynamicsEnsemble` (with the GRU-based
# `RNNBase`) is the actual Robotic World Model (RWM) architecture used by RWM/RWM-U:
# a dual-head (state + auxiliary) recurrent ensemble dynamics model that consumes a
# history of (state, action) pairs and forecasts the next robot state along with
# aleatoric/epistemic uncertainty and optional contact/termination/extension signals.
# Transcribed verbatim; only change is dropping the unused MLPBase/rssm architecture
# branches' surrounding config machinery is kept but instantiation below pins
# architecture_config["type"]="rnn" (the GRU dual-autoregressive variant referenced in
# the queue notes) at menagerie scale.
import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# --- rsl_rl/modules/architectures/rnn.py (verbatim) ---
class RNNBase(nn.Module):
    def __init__(
        self,
        input_dim: int,
        device: str,
        architecture_config: dict = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.device = device

        rnn_type = architecture_config["rnn_type"]
        rnn_num_layers = architecture_config["rnn_num_layers"]
        rnn_hidden_size = architecture_config["rnn_hidden_size"]
        self.memory = Memory(
            input_dim, device, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size
        )

    def forward(self, x_state_batch, x_action_batch):
        x = torch.cat([x_state_batch, x_action_batch], dim=-1)
        x = self.memory(x)
        return x

    def reset(self):
        self.memory.reset()

    def reset_partial(self, batch_indices):
        self.memory.reset_partial(batch_indices)


class Memory(nn.Module):
    def __init__(self, input_dim: int, device: str, type: str, num_layers: int, hidden_size: int):
        super().__init__()
        self.input_dim = input_dim
        self.device = device
        rnn_cls = nn.GRU if type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=self.input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            device=self.device,
            batch_first=True,
        )
        self.hidden_states = None

    def forward(self, x):
        x, self.hidden_states = self.rnn(x, self.hidden_states)
        return x[:, -1]

    def reset(self):
        self.hidden_states = None

    def reset_partial(self, batch_indices):
        if self.hidden_states is not None:
            self.hidden_states[:, batch_indices] = 0.0


# --- rsl_rl/modules/architectures/mlp.py (verbatim, heads only used by every arch type) ---
class MLPStateHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        device: str,
        architecture_config: dict = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.device = device
        self.state_mean_shape = architecture_config["state_mean_shape"]
        self.state_logstd_shape = architecture_config["state_logstd_shape"]

        state_mean_layers = []
        curr_in_dim = self.input_dim
        for hidden_dim in self.state_mean_shape:
            state_mean_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            state_mean_layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        state_mean_layers.append(nn.Linear(self.state_mean_shape[-1], state_dim))
        self.state_mean_layers = nn.Sequential(*state_mean_layers).to(self.device)
        self.state_mean_layers.train()

        if self.state_logstd_shape is not None:
            self.output_std = True
            state_logstd_layers = []
            curr_in_dim = self.input_dim
            for hidden_dim in self.state_logstd_shape:
                state_logstd_layers.append(nn.Linear(curr_in_dim, hidden_dim))
                state_logstd_layers.append(nn.ReLU())
                curr_in_dim = hidden_dim
            state_logstd_layers.append(nn.Linear(self.state_logstd_shape[-1], state_dim))
            self.state_logstd_layers = nn.Sequential(*state_logstd_layers).to(self.device)
            self.state_logstd_layers.train()
        else:
            self.output_std = False

        if self.output_std:
            self.state_min_logstd = nn.Parameter(
                torch.ones(1, state_dim, device=self.device) * -5.0
            )
            self.state_log_delta_logstd = nn.Parameter(
                torch.ones(1, state_dim, device=self.device) * 0.0
            )

    def forward(self, x, x_state_batch):
        if x.dim() == 3:
            sequence_len = x.shape[1]
            x = x.flatten(0, 1)
            x_state_batch = x_state_batch.flatten(0, 1).unsqueeze(1)
        else:
            sequence_len = 0
        state_mean = self.state_mean_layers(x) + x_state_batch[:, -1]
        state_logstd = (
            self.state_logstd_layers(x)
            if self.output_std
            else -torch.inf * torch.ones(x.shape[0], self.state_dim, device=self.device)
        )
        if self.output_std:
            self.state_max_logstd = self.state_min_logstd + torch.exp(self.state_log_delta_logstd)
            state_logstd = self.state_max_logstd - nn.functional.softplus(
                self.state_max_logstd - state_logstd
            )
            state_logstd = self.state_min_logstd + nn.functional.softplus(
                state_logstd - self.state_min_logstd
            )
        if sequence_len > 0:
            state_mean = state_mean.view(-1, sequence_len, self.state_dim)
            state_logstd = state_logstd.view(-1, sequence_len, self.state_dim)
        return state_mean, torch.exp(state_logstd)

    def reset(self):
        pass

    def reset_partial(self, batch_indices):
        pass


class MLPAuxiliaryHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        extension_dim: int,
        contact_dim: int,
        termination_dim: int,
        device: str,
        architecture_config: dict = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.extension_dim = extension_dim
        self.contact_dim = contact_dim
        self.termination_dim = termination_dim
        self.device = device

        if extension_dim > 0:
            extension_shape = architecture_config["extension_shape"]
            extension_layers = []
            curr_in_dim = self.input_dim
            for hidden_dim in extension_shape:
                extension_layers.append(nn.Linear(curr_in_dim, hidden_dim))
                extension_layers.append(nn.ReLU())
                curr_in_dim = hidden_dim
            extension_layers.append(nn.Linear(extension_shape[-1], extension_dim))
            self.extension_layers = nn.Sequential(*extension_layers).to(self.device)
            self.extension_layers.train()

        if contact_dim > 0:
            contact_shape = architecture_config["contact_shape"]
            contact_layers = []
            curr_in_dim = self.input_dim
            for hidden_dim in contact_shape:
                contact_layers.append(nn.Linear(curr_in_dim, hidden_dim))
                contact_layers.append(nn.ReLU())
                curr_in_dim = hidden_dim
            contact_layers.append(nn.Linear(contact_shape[-1], contact_dim))
            self.contact_layers = nn.Sequential(*contact_layers).to(self.device)
            self.contact_layers.train()

        if termination_dim > 0:
            termination_shape = architecture_config["termination_shape"]
            termination_layers = []
            curr_in_dim = self.input_dim
            for hidden_dim in termination_shape:
                termination_layers.append(nn.Linear(curr_in_dim, hidden_dim))
                termination_layers.append(nn.ReLU())
                curr_in_dim = hidden_dim
            termination_layers.append(nn.Linear(termination_shape[-1], termination_dim))
            self.termination_layers = nn.Sequential(*termination_layers).to(self.device)
            self.termination_layers.train()

    def forward(self, x, x_state_batch):
        if x.dim() == 3:
            sequence_len = x.shape[1]
            x = x.flatten(0, 1)
        else:
            sequence_len = 0

        extension_pred = self.extension_layers(x) if self.extension_dim > 0 else None
        contact_logits = self.contact_layers(x) if self.contact_dim > 0 else None
        termination_logits = self.termination_layers(x) if self.termination_dim > 0 else None

        if sequence_len > 0:
            extension_pred = (
                extension_pred.view(-1, sequence_len, self.extension_dim)
                if self.extension_dim > 0
                else None
            )
            contact_logits = (
                contact_logits.view(-1, sequence_len, self.contact_dim)
                if self.contact_dim > 0
                else None
            )
            termination_logits = (
                termination_logits.view(-1, sequence_len, self.termination_dim)
                if self.termination_dim > 0
                else None
            )

        return extension_pred, contact_logits, termination_logits

    def reset(self):
        pass

    def reset_partial(self, batch_indices):
        pass


# --- rsl_rl/modules/system_dynamics.py (verbatim) ---
class SystemDynamicsEnsemble(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        extension_dim: int,
        contact_dim: int,
        termination_dim: int,
        device: str,
        ensemble_size: int = 1,
        history_horizon: int = 1,
        architecture_config: dict = None,
        freeze_auxiliary: bool = False,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.extension_dim = extension_dim
        self.contact_dim = contact_dim
        self.termination_dim = termination_dim
        self.device = device
        self.ensemble_size = ensemble_size
        self.history_horizon = history_horizon
        self.architecture_config = architecture_config
        self.freeze_auxiliary = freeze_auxiliary

        self._init_networks()

    def _init_networks(self):
        self.state_base = self._create_base()
        self.state_heads = nn.ModuleList(
            [
                MLPStateHead(
                    self.base_output_dim, self.state_dim, self.device, self.architecture_config
                ).to(self.device)
                for _ in range(self.ensemble_size)
            ]
        )

        self.auxiliary_base = self._create_base()
        self.auxiliary_heads = nn.ModuleList(
            [
                MLPAuxiliaryHead(
                    self.base_output_dim,
                    self.extension_dim,
                    self.contact_dim,
                    self.termination_dim,
                    self.device,
                    self.architecture_config,
                ).to(self.device)
                for _ in range(self.ensemble_size)
            ]
        )

        if self.freeze_auxiliary:
            for param in self.auxiliary_base.parameters():
                param.requires_grad = False
            for head in self.auxiliary_heads:
                for param in head.parameters():
                    param.requires_grad = False

    def _create_base(self):
        # menagerie build only wires the "rnn" (GRU) branch of the real
        # `_create_base`; the "mlp" branch is dropped unused per model-scope rules.
        if self.architecture_config["type"] == "rnn":
            input_dim = self.state_dim + self.action_dim
            self.base_output_dim = self.architecture_config["rnn_hidden_size"]
            self.prediction_type = "single"
            return RNNBase(
                input_dim=input_dim,
                device=self.device,
                architecture_config=self.architecture_config,
            )
        else:
            raise ValueError("Invalid architecture type.")

    def forward(self, x_state_batch, x_action_batch, model_ids=None):
        state_means, state_stds, extensions, contacts, terminations = [], [], [], [], []
        state_base_output = self.state_base(x_state_batch, x_action_batch)

        for head in self.state_heads:
            state_mean, state_std = head(state_base_output, x_state_batch)
            if self.prediction_type == "sequence":
                state_mean = state_mean[:, -1]
                state_std = state_std[:, -1]
            state_means.append(state_mean.unsqueeze(0))
            state_stds.append(state_std.unsqueeze(0))

        auxiliary_base_output = self.auxiliary_base(x_state_batch, x_action_batch)
        for head in self.auxiliary_heads:
            extension, contact, termination = head(auxiliary_base_output, x_state_batch)
            if self.prediction_type == "sequence":
                extension = extension[:, -1] if extension is not None else None
                contact = contact[:, -1] if contact is not None else None
                termination = termination[:, -1] if termination is not None else None
            extensions.append(extension.unsqueeze(0) if extension is not None else None)
            contacts.append(contact.unsqueeze(0) if contact is not None else None)
            terminations.append(termination.unsqueeze(0) if termination is not None else None)

        state_means = torch.cat(state_means, dim=0)
        state_stds = torch.cat(state_stds, dim=0)
        extensions = torch.cat(extensions, dim=0) if self.extension_dim > 0 else None
        contacts = torch.cat(contacts, dim=0) if self.contact_dim > 0 else None
        terminations = torch.cat(terminations, dim=0) if self.termination_dim > 0 else None

        if model_ids is None:
            output_state_means = state_means.mean(dim=0)
            output_extensions = extensions.mean(dim=0) if extensions is not None else None
            output_contacts = contacts.mean(dim=0) if contacts is not None else None
            output_terminations = terminations.mean(dim=0) if terminations is not None else None
        else:
            output_state_means = torch.gather(
                state_means, 0, model_ids.repeat(1, 1, self.state_dim)
            ).squeeze(0)
            output_extensions = (
                torch.gather(extensions, 0, model_ids.repeat(1, 1, self.extension_dim)).squeeze(0)
                if extensions is not None
                else None
            )
            output_contacts = (
                torch.gather(contacts, 0, model_ids.repeat(1, 1, self.contact_dim)).squeeze(0)
                if contacts is not None
                else None
            )
            output_terminations = (
                torch.gather(terminations, 0, model_ids.repeat(1, 1, self.termination_dim)).squeeze(
                    0
                )
                if terminations is not None
                else None
            )

        aleatoric_uncertainty = state_stds.mean(dim=0).sum(dim=1)
        epistemic_uncertainty = (
            state_means.std(dim=0).sum(dim=1)
            if self.ensemble_size > 1
            else torch.zeros(output_state_means.shape[0], device=self.device)
        )
        return (
            output_state_means,
            aleatoric_uncertainty,
            epistemic_uncertainty,
            output_extensions,
            output_contacts,
            output_terminations,
        )

    def reset(self):
        self.state_base.reset()
        for head in self.state_heads:
            head.reset()
        if self.auxiliary_base is not None:
            self.auxiliary_base.reset()
            for head in self.auxiliary_heads:
                head.reset()


def build_rwm_system_dynamics():
    # Menagerie-scale config mirroring robotic_world_model_lite's ANYmal-D flat-terrain
    # RNN (GRU) dynamics config: state_dim/action_dim shrunk to the legged-robot
    # observation cardinality used by that task (state=48, action=12), ensemble_size=1
    # (RWM; >1 gives the uncertainty-aware RWM-U ensemble from the paper), single GRU
    # layer at menagerie scale, and small MLP heads.
    architecture_config = {
        "type": "rnn",
        "rnn_type": "gru",
        "rnn_num_layers": 1,
        "rnn_hidden_size": 16,
        "state_mean_shape": [16],
        "state_logstd_shape": [16],
        "extension_shape": [16],
        "contact_shape": [16],
        "termination_shape": [16],
    }
    return SystemDynamicsEnsemble(
        state_dim=8,
        action_dim=4,
        extension_dim=2,
        contact_dim=4,
        termination_dim=1,
        device="cpu",
        ensemble_size=1,
        history_horizon=4,
        architecture_config=architecture_config,
        freeze_auxiliary=False,
    )


def example_input_rwm_system_dynamics():
    torch.manual_seed(0)
    batch_size = 2
    history_horizon = 4
    state_dim = 8
    action_dim = 4
    x_state_batch = torch.randn(batch_size, history_horizon, state_dim)
    x_action_batch = torch.randn(batch_size, history_horizon, action_dim)
    return (x_state_batch, x_action_batch)


MENAGERIE_ENTRIES = [
    (
        "RWM-SystemDynamicsEnsemble",
        "build_rwm_system_dynamics",
        "example_input_rwm_system_dynamics",
        2026,
        "vendored-pytorch",
    ),
]
