# SOURCE: vendored from https://github.com/opendilab/PPOxFamily @ 78f781115681ebb245b7c56675d64db7cd732323
# (chapter4_reward/popart.py: PopArt, MLP, lines 1-170)
#
# PopArt -- "Preserving Outputs Precisely while Adapting Rescaling Targets"
# (van Hasselt et al. 2018, arXiv:1809.04474), adaptive value-normalization
# layer (ART: rescale targets; POP: preserve outputs under rescaling) used to
# stabilize multi-magnitude-reward RL (this exact `PopArt` layer is also
# embedded in IMPALA-PopArt/torchbeastpopart-style agents). OpenDILab's
# PPOxFamily ships an official, actively-maintained, from-scratch PyTorch
# implementation (`PopArt(nn.Module)`: a Linear layer whose weight/bias are
# updated by `update_parameters` alongside running `mu`/`sigma`/`v` buffers to
# preserve unnormalized outputs -- the paper's defining art+pop mechanism) and
# an `MLP` network that uses it as the final layer. Both classes have no
# dependency beyond torch EXCEPT the original `forward`'s return value is
# packaged via `treetensor.torch.as_tensor(...)` (a third-party tree-tensor
# container for training-loop ergonomics; not installed in the base env and
# not architectural -- no tensor computation depends on it, it only repackages
# the two already-computed output tensors). That one output-wrapper call is
# swapped for a plain `dict` here (a minimal, non-architectural import fix);
# every weight/bias linear-layer op and every `mu`/`sigma`/`v` buffer
# initialization/update path is vendored verbatim.

import math
import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---- chapter4_reward/popart.py (vendored; treetensor output -> plain dict) ----
class PopArt(nn.Module):
    """
    The definition of Pop-Art layer, i.e., a linear layer with popart
    normalization, which should be used as the last layer of a network.
    <link https://arxiv.org/abs/1809.04474 link>
    """

    def __init__(self, input_features: int, output_features: int, beta: float = 0.5) -> None:
        super(PopArt, self).__init__()

        self.beta = beta
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        self.bias = nn.Parameter(torch.Tensor(output_features))
        self.register_buffer("mu", torch.zeros(output_features, requires_grad=False))
        self.register_buffer("sigma", torch.ones(output_features, requires_grad=False))
        self.register_buffer("v", torch.ones(output_features, requires_grad=False))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor):
        # Execute the linear layer computation $$y=Wx+b$$
        normalized_output = x.mm(self.weight.t())
        normalized_output += self.bias.unsqueeze(0).expand_as(normalized_output)
        # Unnormalize the output for more convenient usage.
        with torch.no_grad():
            output = normalized_output * self.sigma + self.mu

        return {"output": output, "normalized_output": normalized_output}

    def update_parameters(self, value: torch.Tensor):
        self.mu = self.mu.to(value.device)
        self.sigma = self.sigma.to(value.device)
        self.v = self.v.to(value.device)

        old_mu = self.mu
        old_std = self.sigma
        batch_mean = torch.mean(value, 0)
        batch_v = torch.mean(torch.pow(value, 2), 0)
        batch_mean[torch.isnan(batch_mean)] = self.mu[torch.isnan(batch_mean)]
        batch_v[torch.isnan(batch_v)] = self.v[torch.isnan(batch_v)]
        batch_mean = (1 - self.beta) * self.mu + self.beta * batch_mean
        batch_v = (1 - self.beta) * self.v + self.beta * batch_v
        batch_std = torch.sqrt(batch_v - (batch_mean**2))
        batch_std = torch.clamp(batch_std, min=1e-4, max=1e6)
        batch_std[torch.isnan(batch_std)] = self.sigma[torch.isnan(batch_std)]

        self.mu = batch_mean
        self.v = batch_v
        self.sigma = batch_std
        self.weight.data = (self.weight.t() * old_std / self.sigma).t()
        self.bias.data = (old_std * self.bias + old_mu - self.mu) / self.sigma

        return {"new_mean": batch_mean, "new_std": batch_std}


class MLP(nn.Module):
    def __init__(self, obs_shape: int, action_shape: int) -> None:
        """
        A MLP network with popart as the final layer.
        Input: observations and actions
        Output: Estimated Q value
        ``cat(obs,actions) -> encoder -> popart`` .
        """
        super(MLP, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_shape + action_shape, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
        )
        self.popart = PopArt(32, 1)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor):
        x = torch.cat((obs, actions), 1)
        x = self.encoder(x)
        x = self.popart(x)
        return x


# ---- end vendored popart.py ----


def build_popart_mlp():
    return MLP(obs_shape=8, action_shape=1)


def example_input_popart_mlp():
    return (torch.randn(4, 8), torch.randn(4, 1))


MENAGERIE_ENTRIES = [
    ("PopArt (MLP)", build_popart_mlp, example_input_popart_mlp, 2018, "vendored-pytorch"),
]
