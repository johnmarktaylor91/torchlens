# FAITHFUL PORT of brendanator/predictron @ master (original framework: TensorFlow 0.x/1.x, functional API)
#
# Source model file ported: predictron/predictron.py, functions state_representation(),
# model_network(), rollout_states(), output_network(), value_network(), reward_network(),
# discount_network(), lambda_network(), preturn_network(), lambda_preturn_network(), and the
# top-level predictron() orchestration. The repo predates the current TensorFlow entirely (uses
# `tf.pack`, `tf.batch_matmul`, `tf.mul`, `tf.app.flags` -- TF 0.x/early-1.x APIs long removed from
# any installable TensorFlow release, with no viable base-env TF1-compat shim), so it cannot be
# vendored/run as-is. The functional graph-construction style is fully self-contained (no external
# deps beyond raw TF ops) and every layer/mechanism below is transcribed faithfully from that code:
#   - state_representation: 2x (conv3x3 + bias + ReLU) encoder
#   - model_network (the shared "core"): 3x (conv3x3 + bias + BatchNorm(affine=False) + ReLU),
#     rolled out `predictron_depth` times with a SHARED core (shared_core=True branch of
#     rollout_states / model_network), returning (hidden_layer_1, next_state) each step
#   - output_network (shared_core branch): flatten -> Linear -> Linear per quantity (value /
#     reward / discount / lambda), applied at every unrolled depth
#   - preturn_network: cumulative-product/cumulative-sum combination of rewards/discounts/values
#     into k-step returns (the Predictron's core "returns" mechanism, eq. 3-4 of Silver et al. 2017)
#   - lambda_preturn_network: TD(lambda)-style weighted combination of preturns via lambdas
#     (eq. 5-6 of Silver et al. 2017)
# The reward/discount/lambda "insert first/last value" slicing-and-concat pattern in the original
# (rewards[0]:=0, discounts[0]:=1, lambdas[-1]:=0) is preserved exactly. Training-only code
# (loss(), train(), TF summaries, weight-decay bookkeeping) is intentionally not ported; only the
# forward network architecture is in scope for TorchLens capture.
#
# Reference (paper): "The Predictron: End-To-End Learning and Planning" (Silver et al., ICML 2017).

import torch
import torch.nn as nn
import torch.nn.functional as F


class StateRepresentation(nn.Module):
    """Faithful port of state_representation() in predictron.py."""

    def __init__(self, input_channels: int, state_kernels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, state_kernels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(state_kernels, state_kernels, kernel_size=3, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden_1 = F.relu(self.conv1(inputs))
        state = F.relu(self.conv2(hidden_1))
        return state


class ModelCore(nn.Module):
    """Faithful port of model_network() in predictron.py (the shared predictron "core")."""

    def __init__(self, state_kernels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(state_kernels, state_kernels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(state_kernels, affine=False)
        self.conv2 = nn.Conv2d(state_kernels, state_kernels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(state_kernels, affine=False)
        self.conv3 = nn.Conv2d(state_kernels, state_kernels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(state_kernels, affine=False)

    def forward(self, state: torch.Tensor):
        hidden_layer_1 = F.relu(self.bn1(self.conv1(state)))
        hidden_layer_2 = F.relu(self.bn2(self.conv2(hidden_layer_1)))
        next_state = F.relu(self.bn3(self.conv3(hidden_layer_2)))
        return hidden_layer_1, next_state


class OutputNetwork(nn.Module):
    """Faithful port of output_network() (shared_core=True branch) in predictron.py.

    Applied identically at every unrolled depth step (shared weights, as in the original
    shared_core output_network branch)."""

    def __init__(self, state_size: int, output_hidden_size: int, reward_size: int):
        super().__init__()
        self.fc1 = nn.Linear(state_size, output_hidden_size)
        self.fc2 = nn.Linear(output_hidden_size, reward_size)

    def forward(self, inputs_flat: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(inputs_flat)
        logits = self.fc2(hidden)
        return logits


class Predictron(nn.Module):
    """Faithful port of the predictron() top-level orchestration in predictron.py, with
    shared_core=True (the repo's default FLAGS.shared_core)."""

    def __init__(
        self,
        input_channels: int = 3,
        state_kernels: int = 32,
        predictron_depth: int = 8,
        output_hidden_size: int = 32,
        reward_size: int = 1,
        spatial_size: int = 8,
    ):
        super().__init__()
        self.predictron_depth = predictron_depth
        self.reward_size = reward_size
        self.state_kernels = state_kernels
        self.spatial_size = spatial_size
        self.state_size = spatial_size * spatial_size * state_kernels

        self.state_representation = StateRepresentation(input_channels, state_kernels)
        # shared_core=True: a single ModelCore reused at every depth step
        self.core = ModelCore(state_kernels)

        self.value_net = OutputNetwork(self.state_size, output_hidden_size, reward_size)
        self.reward_net = OutputNetwork(self.state_size, output_hidden_size, reward_size)
        self.discount_net = OutputNetwork(self.state_size, output_hidden_size, reward_size)
        self.lambda_net = OutputNetwork(self.state_size, output_hidden_size, reward_size)

    def _rollout_states(self, state: torch.Tensor):
        # Faithful port of rollout_states(): unroll the shared core `predictron_depth` times,
        # collecting the pre-core "state" at each step and the intermediate hidden_layer_1.
        states = [state]
        hidden_states = []
        for _ in range(self.predictron_depth):
            hidden_state, state = self.core(state)
            states.append(state)
            hidden_states.append(hidden_state)
        # states[:-1] mirrors the original states[:-1] slice before stacking
        states_stack = torch.stack(states[:-1], dim=1)  # (B, depth, C, H, W)
        hidden_states_stack = torch.stack(hidden_states, dim=1)  # (B, depth, C, H, W)
        bsz = state.shape[0]
        states_flat = states_stack.reshape(bsz, self.predictron_depth, self.state_size)
        hidden_states_flat = hidden_states_stack.reshape(
            bsz, self.predictron_depth, self.state_size
        )
        return states_flat, hidden_states_flat

    def _output_network(self, net: OutputNetwork, inputs: torch.Tensor) -> torch.Tensor:
        # Faithful port of output_network() (shared_core branch): flatten (B, depth, state) ->
        # (B*depth, state), apply the shared 2-layer MLP, reshape back.
        bsz, depth, state_size = inputs.shape
        flat = inputs.reshape(bsz * depth, state_size)
        logits = net(flat)
        return logits.reshape(bsz, depth, self.reward_size)

    def _preturn_network(
        self, rewards: torch.Tensor, discounts: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        # Faithful port of preturn_network(): cumulative-product/cumulative-sum combination.
        accum_value_discounts = torch.cumprod(discounts, dim=1)  # inclusive cumprod
        accum_reward_discounts = torch.cat(
            [torch.ones_like(discounts[:, :1]), torch.cumprod(discounts, dim=1)[:, :-1]], dim=1
        )  # exclusive cumprod
        discounted_values = values * accum_value_discounts
        discounted_rewards = rewards * accum_reward_discounts
        cumulative_rewards = torch.cumsum(discounted_rewards, dim=1)
        preturns = cumulative_rewards + discounted_values
        return preturns

    def _lambda_preturn_network(
        self, preturns: torch.Tensor, lambdas: torch.Tensor
    ) -> torch.Tensor:
        # Faithful port of lambda_preturn_network(): exclusive cumprod of lambdas, weighted sum.
        accum_lambda = torch.cat(
            [torch.ones_like(lambdas[:, :1]), torch.cumprod(lambdas, dim=1)[:, :-1]], dim=1
        )  # exclusive cumprod
        lambda_bar = (1 - lambdas) * accum_lambda
        lambda_preturn = torch.sum(lambda_bar * preturns, dim=1)
        return lambda_preturn

    def forward(self, inputs: torch.Tensor):
        state = self.state_representation(inputs)
        states, hidden_states = self._rollout_states(state)

        values = self._output_network(self.value_net, states)
        rewards = self._output_network(self.reward_net, hidden_states)
        discounts_logits = self._output_network(self.discount_net, hidden_states)
        discounts = torch.sigmoid(discounts_logits)
        lambdas_logits = self._output_network(self.lambda_net, hidden_states)
        lambdas = torch.sigmoid(lambdas_logits)

        # rewards[0] := 0 (first reward), discounts[0] := 1 (first discount), lambdas[-1] := 0
        depth = self.predictron_depth
        rewards = torch.cat([torch.zeros_like(rewards[:, :1]), rewards[:, : depth - 1]], dim=1)
        discounts = torch.cat([torch.ones_like(discounts[:, :1]), discounts[:, : depth - 1]], dim=1)
        lambdas = torch.cat([lambdas[:, : depth - 1], torch.zeros_like(lambdas[:, :1])], dim=1)

        preturns = self._preturn_network(rewards, discounts, values)
        lambda_preturn = self._lambda_preturn_network(preturns, lambdas)

        return preturns, lambda_preturn


MENAGERIE_ZOO = "ported-pytorch"


def build_predictron():
    # Tiny config: small spatial size + shallow depth to keep the trace fast.
    return Predictron(
        input_channels=3,
        state_kernels=8,
        predictron_depth=4,
        output_hidden_size=16,
        reward_size=1,
        spatial_size=8,
    )


def example_input_predictron():
    return torch.randn(2, 3, 8, 8)


MENAGERIE_ENTRIES = [
    ("predictron", build_predictron, example_input_predictron, 2017, MENAGERIE_ZOO),
]
