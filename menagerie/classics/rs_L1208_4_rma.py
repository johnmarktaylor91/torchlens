# SOURCE: vendored from antonilo/rl_locomotion @ master
# https://github.com/antonilo/rl_locomotion/blob/master/raisimGymTorch/algo/ppo/module.py
# https://github.com/antonilo/rl_locomotion/blob/master/raisimGymTorch/env/envs/dagger_a1/dagger.py
# Official code release for Kumar et al., "RMA: Rapid Motor Adaptation for Legged
# Robots" (RSS 2021) / its rl_locomotion follow-up. The `StateHistoryEncoder` (temporal
# 1D-conv encoder that compresses a length-T window of proprioceptive base
# observations into the RMA "phase-2" adaptation latent) and `MLP` (the RMA locomotion
# action policy consuming [base_obs, adaptation_latent]) classes are transcribed
# VERBATIM from raisimGymTorch/algo/ppo/module.py. `RMAPolicy` wires them together
# exactly as dagger_a1/dagger.py does at train time (student_mlp =
# ppo_module.MLP(...); prop_latent_encoder = ppo_module.StateHistoryEncoder(...);
# actor observation -> history encoder -> latent -> concat with current base obs ->
# action_mlp), which is the real RMA architecture: base policy MLP + small
# adaptation-module CNN over observation history, no architectural change.
import numpy as np
import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# --- raisimGymTorch/algo/ppo/module.py (verbatim architecture) ---
class StateHistoryEncoder(nn.Module):
    def __init__(self, activation_fn, input_size, tsteps, output_size):
        super(StateHistoryEncoder, self).__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps
        self.input_shape = input_size * tsteps
        self.output_shape = output_size

        if tsteps == 50:
            self.encoder = nn.Sequential(nn.Linear(input_size, 32), self.activation_fn())
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels=32, out_channels=32, kernel_size=8, stride=4),
                nn.LeakyReLU(),
                nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1),
                nn.LeakyReLU(),
                nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1),
                nn.LeakyReLU(),
                nn.Flatten(),
            )
            self.linear_output = nn.Sequential(nn.Linear(32 * 3, output_size), self.activation_fn())
        elif tsteps == 10:
            self.encoder = nn.Sequential(nn.Linear(input_size, 32), self.activation_fn())
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels=32, out_channels=32, kernel_size=4, stride=2),
                nn.LeakyReLU(),
                nn.Conv1d(in_channels=32, out_channels=32, kernel_size=2, stride=1),
                nn.LeakyReLU(),
                nn.Flatten(),
            )
            self.linear_output = nn.Sequential(nn.Linear(32 * 3, output_size), self.activation_fn())
        elif tsteps == 20:
            self.encoder = nn.Sequential(nn.Linear(input_size, 32), self.activation_fn())
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels=32, out_channels=32, kernel_size=6, stride=2),
                nn.LeakyReLU(),
                nn.Conv1d(in_channels=32, out_channels=32, kernel_size=4, stride=2),
                nn.LeakyReLU(),
                nn.Flatten(),
            )
            self.linear_output = nn.Sequential(nn.Linear(32 * 3, output_size), self.activation_fn())
        else:
            raise NotImplementedError()

    def forward(self, obs):
        bs = obs.shape[0]
        T = self.tsteps
        projection = self.encoder(obs.reshape([bs * T, -1]))
        output = self.conv_layers(projection.reshape([bs, -1, T]))
        output = self.linear_output(output)
        return output


class MLP(nn.Module):
    def __init__(
        self,
        shape,
        actionvation_fn,
        input_size,
        output_size,
        output_activation_fn=None,
        small_init=False,
        base_obdim=None,
    ):
        super(MLP, self).__init__()
        self.activation_fn = actionvation_fn
        self.output_activation_fn = output_activation_fn

        modules = [nn.Linear(input_size, shape[0]), self.activation_fn()]
        scale = [np.sqrt(2)]

        for idx in range(len(shape) - 1):
            modules.append(nn.Linear(shape[idx], shape[idx + 1]))
            modules.append(self.activation_fn())
            scale.append(np.sqrt(2))

        modules.append(nn.Linear(shape[-1], output_size))
        action_output_layer = modules[-1]
        if self.output_activation_fn is not None:
            modules.append(self.output_activation_fn())
        self.architecture = nn.Sequential(*modules)
        scale.append(np.sqrt(2))

        self.init_weights(self.architecture, scale)
        if small_init:
            action_output_layer.weight.data *= 1e-6

        self.input_shape = [input_size]
        self.output_shape = [output_size]

    @staticmethod
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]


# --- dagger_a1/dagger.py wiring (verbatim composition, adapted into one nn.Module) ---
class RMAPolicy(nn.Module):
    """RMA phase-2 locomotion policy: StateHistoryEncoder (temporal adaptation module,
    the RMA contribution) compresses a T-step history of base observations into a
    latent, which is concatenated with the current base observation and fed to the
    MLP action policy -- exactly the `prop_latent_encoder` + `student_mlp` composition
    built in dagger_a1/dagger.py's training script."""

    def __init__(self, base_obdim, tsteps, latent_dim, policy_shape, act_dim):
        super().__init__()
        self.base_obdim = base_obdim
        self.tsteps = tsteps
        self.history_encoder = StateHistoryEncoder(
            nn.LeakyReLU, input_size=base_obdim, tsteps=tsteps, output_size=latent_dim
        )
        self.action_mlp = MLP(
            policy_shape,
            nn.LeakyReLU,
            input_size=base_obdim + latent_dim,
            output_size=act_dim,
            output_activation_fn=None,
            small_init=False,
        )

    def forward(self, obs_history, current_obs):
        """
        Args:
            obs_history: [B, base_obdim * tsteps] flattened history window
            current_obs: [B, base_obdim] current-step base observation
        Returns:
            [B, act_dim] action
        """
        latent = self.history_encoder(obs_history)
        x = torch.cat([current_obs, latent], dim=1)
        return self.action_mlp.architecture(x)


def build_rma():
    # Tiny menagerie-scale config: base_obdim=8 (shrunk from the released A1-quadruped
    # 45-48 dim proprioceptive observation), tsteps=10 (shortest supported history
    # window, matching the tsteps==10 branch of StateHistoryEncoder), latent_dim=8
    # (shrunk from the released 8-24 dim adaptation latent), policy_shape=[32, 32]
    # (shrunk from the released [128, 64] action MLP), act_dim=12 (real A1 joint count).
    return RMAPolicy(base_obdim=8, tsteps=10, latent_dim=8, policy_shape=[32, 32], act_dim=12)


def example_input_rma():
    torch.manual_seed(0)
    batch = 2
    base_obdim = 8
    tsteps = 10
    obs_history = torch.randn(batch, base_obdim * tsteps)
    current_obs = torch.randn(batch, base_obdim)
    return (obs_history, current_obs)


MENAGERIE_ENTRIES = [
    (
        "RMA (Rapid Motor Adaptation)",
        "build_rma",
        "example_input_rma",
        2021,
        "vendored-pytorch",
    ),
]
