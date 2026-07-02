# SOURCE: vendored from https://github.com/alex-petrenko/sample-factory @ master
# (sample_factory/model/{model_utils,encoder,core,decoder,action_parameterization,
#  actor_critic}.py, sample_factory/algo/utils/{action_distributions,running_mean_std,
#  torch_utils}.py, sample_factory/utils/{normalize,attr_dict}.py,
#  sample_factory/cfg/configurable.py -- vendored near-verbatim, combined into a single
#  file for staging; only cosmetic import-path adjustments were made.)
#
# Sample Factory (Petrenko et al. 2020, "Sample Factory: Egocentric 3D Control from
# Pixels at 100000 FPS with Asynchronous Reinforcement Learning", ICLR 2020). This is
# the real `ActorCriticSharedWeights` policy network: a `MultiInputEncoder` (per-key
# CNN/MLP sub-encoders concatenated), a recurrent-or-identity `ModelCore`, an
# `MlpDecoder`, a critic value head, and a `CategoricalActionDistribution` action head
# via `ActionParameterizationDefault` -- exactly the architecture
# `default_make_actor_critic_func` builds for a discrete-action, image-observation
# Atari/VizDoom-style environment (the library's most common configuration). Training
# infrastructure (samplers, learners, multiprocessing) is intentionally omitted; only
# the `nn.Module` architecture is vendored.

import math
from typing import Dict, Final, List, Optional, Union

import gymnasium as gym
import torch
import torch.nn as nn
from gymnasium import spaces
from torch import Tensor
from torch.jit import RecursiveScriptModule, ScriptModule
from torch.nn import functional
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

MENAGERIE_ZOO = "vendored-pytorch"


# ---- sample_factory/utils/attr_dict.py (vendored verbatim) ----
class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattribute__(self, item):
        if item in self:
            return self[item]
        else:
            return super().__getattribute__(item)


# ---- sample_factory/cfg/configurable.py (vendored verbatim) ----
class Configurable:
    def __init__(self, cfg: AttrDict):
        self.cfg: AttrDict = cfg


# ---- sample_factory/algo/utils/tensor_dict.py (minimal vendor: only what forward() needs) ----
class TensorDict(dict):
    dict_key_type = str

    def __getitem__(self, key):
        if isinstance(key, self.dict_key_type):
            return dict.__getitem__(self, key)
        else:
            return self._index_func(self, key)

    def _index_func(self, x, indices):
        if isinstance(x, (dict, TensorDict)):
            res = TensorDict()
            for key, value in x.items():
                res[key] = self._index_func(value, indices)
            return res
        else:
            return x[indices]

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)


# ---- sample_factory/algo/utils/torch_utils.py (calc_num_elements, vendored verbatim) ----
def calc_num_elements(module, module_input_shape):
    shape_with_batch_dim = (1,) + module_input_shape
    some_input = torch.rand(shape_with_batch_dim)
    num_elements = module(some_input).numel()
    return num_elements


# ---- sample_factory/model/model_utils.py (vendored verbatim) ----
def nonlinearity(cfg, inplace: bool = False) -> nn.Module:
    if cfg.nonlinearity == "elu":
        return nn.ELU(inplace=inplace)
    elif cfg.nonlinearity == "relu":
        return nn.ReLU(inplace=inplace)
    elif cfg.nonlinearity == "tanh":
        return nn.Tanh()
    else:
        raise Exception(f"Unknown {cfg.nonlinearity=}")


def fc_layer(in_features: int, out_features: int, bias=True, spec_norm=False) -> nn.Module:
    layer = nn.Linear(in_features, out_features, bias)
    if spec_norm:
        layer = torch.nn.utils.spectral_norm(layer)
    return layer


def create_mlp(layer_sizes: List[int], input_size: int, activation: nn.Module) -> nn.Module:
    """Sequential fully connected layers."""
    layers = []
    for i, size in enumerate(layer_sizes):
        layers.extend([fc_layer(input_size, size), activation])
        input_size = size

    if len(layers) > 0:
        return nn.Sequential(*layers)
    else:
        return nn.Identity()


class ModelModule(nn.Module, Configurable):
    def __init__(self, cfg):
        nn.Module.__init__(self)
        Configurable.__init__(self, cfg)

    def get_out_size(self):
        raise NotImplementedError()


def model_device(model: nn.Module) -> Optional[torch.device]:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return None


# ---- sample_factory/model/encoder.py (vendored verbatim) ----
class Encoder(ModelModule):
    def __init__(self, cfg):
        super().__init__(cfg)

    def get_out_size(self) -> int:
        raise NotImplementedError()

    def model_to_device(self, device):
        self.to(device)

    def device_for_input_tensor(self, input_tensor_name: str) -> Optional[torch.device]:
        return model_device(self)

    def type_for_input_tensor(self, input_tensor_name: str) -> torch.dtype:
        return torch.float32


class MultiInputEncoder(Encoder):
    def __init__(self, cfg, obs_space):
        super().__init__(cfg)
        self.obs_keys = list(sorted(obs_space.keys()))
        self.encoders = nn.ModuleDict()

        out_size = 0
        for obs_key in self.obs_keys:
            shape = obs_space[obs_key].shape

            if len(shape) == 1:
                encoder_fn = MlpEncoder
            elif len(shape) > 1:
                encoder_fn = make_img_encoder
            else:
                raise NotImplementedError(f"Unsupported observation space {obs_space}")

            self.encoders[obs_key] = encoder_fn(cfg, obs_space[obs_key])
            out_size += self.encoders[obs_key].get_out_size()

        self.encoder_out_size = out_size

    def forward(self, obs_dict):
        if len(self.obs_keys) == 1:
            key = self.obs_keys[0]
            return self.encoders[key](obs_dict[key])

        encodings = []
        for key in self.obs_keys:
            x = self.encoders[key](obs_dict[key])
            encodings.append(x)

        return torch.cat(encodings, 1)

    def get_out_size(self) -> int:
        return self.encoder_out_size


class MlpEncoder(Encoder):
    def __init__(self, cfg, obs_space):
        super().__init__(cfg)

        mlp_layers: List[int] = cfg.encoder_mlp_layers
        self.mlp_head = create_mlp(mlp_layers, obs_space.shape[0], nonlinearity(cfg))
        if len(mlp_layers) > 0:
            self.mlp_head = torch.jit.script(self.mlp_head)
        self.encoder_out_size = calc_num_elements(self.mlp_head, obs_space.shape)

    def forward(self, obs: Tensor):
        x = self.mlp_head(obs)
        return x

    def get_out_size(self) -> int:
        return self.encoder_out_size


class ConvEncoderImpl(nn.Module):
    """
    After we parse all the configuration and figure out the exact architecture of the model,
    we devote a separate module to it to be able to use torch.jit.script (hopefully benefit from some layer
    fusion).
    """

    def __init__(
        self, obs_shape, conv_filters: List, extra_mlp_layers: List[int], activation: nn.Module
    ):
        super().__init__()

        conv_layers = []
        for layer in conv_filters:
            if layer == "maxpool_2x2":
                conv_layers.append(nn.MaxPool2d((2, 2)))
            elif isinstance(layer, (list, tuple)):
                inp_ch, out_ch, filter_size, stride = layer
                conv_layers.append(nn.Conv2d(inp_ch, out_ch, filter_size, stride=stride))
                conv_layers.append(activation)
            else:
                raise NotImplementedError(f"Layer {layer} not supported!")

        self.conv_head = nn.Sequential(*conv_layers)
        self.conv_head_out_size = calc_num_elements(self.conv_head, obs_shape)
        self.mlp_layers = create_mlp(extra_mlp_layers, self.conv_head_out_size, activation)

    def forward(self, obs: Tensor) -> Tensor:
        x = self.conv_head(obs)
        x = x.contiguous().view(-1, self.conv_head_out_size)
        x = self.mlp_layers(x)
        return x


class ConvEncoder(Encoder):
    def __init__(self, cfg, obs_space):
        super().__init__(cfg)

        input_channels = obs_space.shape[0]

        if cfg.encoder_conv_architecture == "convnet_simple":
            conv_filters = [[input_channels, 32, 8, 4], [32, 64, 4, 2], [64, 128, 3, 2]]
        elif cfg.encoder_conv_architecture == "convnet_impala":
            conv_filters = [[input_channels, 16, 8, 4], [16, 32, 4, 2]]
        elif cfg.encoder_conv_architecture == "convnet_atari":
            conv_filters = [[input_channels, 32, 8, 4], [32, 64, 4, 2], [64, 64, 3, 1]]
        else:
            raise NotImplementedError(
                f"Unknown encoder architecture {cfg.encoder_conv_architecture}"
            )

        activation = nonlinearity(self.cfg)
        extra_mlp_layers: List[int] = cfg.encoder_conv_mlp_layers
        enc = ConvEncoderImpl(obs_space.shape, conv_filters, extra_mlp_layers, activation)
        self.enc = torch.jit.script(enc)

        self.encoder_out_size = calc_num_elements(self.enc, obs_space.shape)

    def get_out_size(self) -> int:
        return self.encoder_out_size

    def forward(self, obs: Tensor):
        return self.enc(obs)


def make_img_encoder(cfg, obs_space) -> Encoder:
    """Make (most likely convolutional) encoder for image-based observations."""
    if cfg.encoder_conv_architecture.startswith("convnet"):
        return ConvEncoder(cfg, obs_space)
    else:
        raise NotImplementedError(
            f"Unknown convolutional architecture {cfg.encoder_conv_architecture}"
        )


def default_make_encoder_func(cfg, obs_space) -> Encoder:
    return MultiInputEncoder(cfg, obs_space)


# ---- sample_factory/model/core.py (vendored verbatim) ----
class ModelCore(ModelModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.core_output_size = -1

    def get_out_size(self) -> int:
        return self.core_output_size


class ModelCoreRNN(ModelCore):
    def __init__(self, cfg, input_size):
        super().__init__(cfg)

        self.cfg = cfg
        self.is_gru = False

        if cfg.rnn_type == "gru":
            self.core = nn.GRU(input_size, cfg.rnn_size, cfg.rnn_num_layers)
            self.is_gru = True
        elif cfg.rnn_type == "lstm":
            self.core = nn.LSTM(input_size, cfg.rnn_size, cfg.rnn_num_layers)
        else:
            raise RuntimeError(f"Unknown RNN type {cfg.rnn_type}")

        self.core_output_size = cfg.rnn_size
        self.rnn_num_layers = cfg.rnn_num_layers

    def forward(self, head_output, rnn_states):
        is_seq = not torch.is_tensor(head_output)
        if not is_seq:
            head_output = head_output.unsqueeze(0)

        if self.rnn_num_layers > 1:
            rnn_states = rnn_states.view(rnn_states.size(0), self.cfg.rnn_num_layers, -1)
            rnn_states = rnn_states.permute(1, 0, 2)
        else:
            rnn_states = rnn_states.unsqueeze(0)

        if self.is_gru:
            x, new_rnn_states = self.core(head_output, rnn_states.contiguous())
        else:
            h, c = torch.split(rnn_states, self.cfg.rnn_size, dim=2)
            x, (h, c) = self.core(head_output, (h.contiguous(), c.contiguous()))
            new_rnn_states = torch.cat((h, c), dim=2)

        if not is_seq:
            x = x.squeeze(0)

        if self.rnn_num_layers > 1:
            new_rnn_states = new_rnn_states.permute(1, 0, 2)
            new_rnn_states = new_rnn_states.reshape(new_rnn_states.size(0), -1)
        else:
            new_rnn_states = new_rnn_states.squeeze(0)

        return x, new_rnn_states


class ModelCoreIdentity(ModelCore):
    """A noop core (no recurrency)."""

    def __init__(self, cfg, input_size):
        super().__init__(cfg)
        self.cfg = cfg
        self.core_output_size = input_size

    def forward(self, head_output, fake_rnn_states):
        return head_output, fake_rnn_states


def default_make_core_func(cfg, core_input_size: int) -> ModelCore:
    if cfg.use_rnn:
        core = ModelCoreRNN(cfg, core_input_size)
    else:
        core = ModelCoreIdentity(cfg, core_input_size)
    return core


# ---- sample_factory/model/decoder.py (vendored verbatim) ----
class Decoder(ModelModule):
    pass


class MlpDecoder(Decoder):
    def __init__(self, cfg, decoder_input_size: int):
        super().__init__(cfg)
        self.core_input_size = decoder_input_size
        decoder_layers: List[int] = cfg.decoder_mlp_layers
        activation = nonlinearity(cfg)
        self.mlp = create_mlp(decoder_layers, decoder_input_size, activation)
        if len(decoder_layers) > 0:
            self.mlp = torch.jit.script(self.mlp)

        self.decoder_out_size = calc_num_elements(self.mlp, (decoder_input_size,))

    def forward(self, core_output):
        return self.mlp(core_output)

    def get_out_size(self):
        return self.decoder_out_size


def default_make_decoder_func(cfg, core_input_size: int) -> Decoder:
    return MlpDecoder(cfg, core_input_size)


# ---- sample_factory/algo/utils/action_distributions.py (vendored: calc_num_action_parameters,
#      is_continuous_action_space, get_action_distribution, sample_actions_log_probs,
#      masked_softmax, masked_log_softmax, CategoricalActionDistribution) ----
def calc_num_action_parameters(action_space) -> int:
    if isinstance(action_space, gym.spaces.Discrete):
        return action_space.n
    elif isinstance(action_space, gym.spaces.Tuple):
        return sum([calc_num_action_parameters(a) for a in action_space])
    elif isinstance(action_space, gym.spaces.Box):
        import numpy as np

        return int(np.prod(action_space.shape) * 2)
    else:
        raise NotImplementedError(f"Action space type {type(action_space)} not supported!")


def is_continuous_action_space(action_space) -> bool:
    return isinstance(action_space, gym.spaces.Box)


def masked_softmax(logits, mask):
    logits = logits + (mask == 0) * -1e9
    result = functional.softmax(logits, dim=-1)
    result = result * mask
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
    return result


def masked_log_softmax(logits, mask):
    logits = logits + (mask == 0) * -1e9
    return functional.log_softmax(logits, dim=-1)


class CategoricalActionDistribution:
    def __init__(self, raw_logits, action_mask=None):
        self.raw_logits = raw_logits
        self.action_mask = action_mask
        self.log_p = self.p = None

    @property
    def probs(self):
        if self.p is None:
            if self.action_mask is not None:
                self.p = masked_softmax(self.raw_logits, self.action_mask)
            else:
                self.p = functional.softmax(self.raw_logits, dim=-1)
        return self.p

    @property
    def log_probs(self):
        if self.log_p is None:
            if self.action_mask is not None:
                self.log_p = masked_log_softmax(self.raw_logits, self.action_mask)
            else:
                self.log_p = functional.log_softmax(self.raw_logits, dim=-1)
        return self.log_p

    def sample(self):
        probs = self.probs
        if self.action_mask is not None:
            all_zero = (probs.sum(dim=-1) == 0).unsqueeze(-1)
            epsilons = torch.full_like(probs, 1e-6)
            probs = torch.where(all_zero, epsilons, probs)

        samples = torch.multinomial(probs, 1, True)
        return samples

    def log_prob(self, value):
        value = value.long()
        log_probs = torch.gather(self.log_probs, -1, value).view(-1)
        return log_probs

    def entropy(self):
        p_log_p = self.log_probs * self.probs
        return -p_log_p.sum(-1)


def get_action_distribution(action_space, raw_logits, action_mask=None):
    assert calc_num_action_parameters(action_space) == raw_logits.shape[-1]

    if isinstance(action_space, gym.spaces.Discrete):
        return CategoricalActionDistribution(raw_logits, action_mask)
    else:
        raise NotImplementedError(
            f"Action space type {type(action_space)} not supported in this vendor slice!"
        )


def sample_actions_log_probs(distribution):
    actions = distribution.sample()
    log_prob_actions = distribution.log_prob(actions)
    return actions, log_prob_actions


# ---- sample_factory/model/action_parameterization.py (vendored verbatim, discrete-space
#      classes only) ----
class ActionsParameterization(nn.Module):
    def __init__(self, cfg, action_space):
        super().__init__()
        self.cfg = cfg
        self.action_space = action_space


class ActionParameterizationDefault(ActionsParameterization):
    """
    A single fully-connected layer to output all parameters of the action distribution. Suitable for
    categorical action distributions, as well as continuous actions with learned state-dependent stddev.
    """

    def __init__(self, cfg, core_out_size, action_space):
        super().__init__(cfg, action_space)

        num_action_outputs = calc_num_action_parameters(action_space)
        self.distribution_linear = nn.Linear(core_out_size, num_action_outputs)

    def forward(self, actor_core_output, action_mask=None):
        action_distribution_params = self.distribution_linear(actor_core_output)
        action_distribution = get_action_distribution(
            self.action_space, raw_logits=action_distribution_params, action_mask=action_mask
        )
        return action_distribution_params, action_distribution


# ---- sample_factory/algo/utils/running_mean_std.py (vendored verbatim) ----
_NORM_EPS = 1e-5
_DEFAULT_CLIP = 5.0


class RunningMeanStdInPlace(nn.Module):
    def __init__(
        self, input_shape, epsilon=_NORM_EPS, clip=_DEFAULT_CLIP, per_channel=False, norm_only=False
    ):
        super().__init__()
        self.input_shape: Final = input_shape
        self.eps: Final[float] = epsilon
        self.clip: Final[float] = clip

        self.norm_only: Final[bool] = norm_only
        self.per_channel: Final[bool] = per_channel

        if per_channel:
            if len(self.input_shape) == 3:
                self.axis = [0, 2, 3]
            if len(self.input_shape) == 2:
                self.axis = [0, 2]
            if len(self.input_shape) == 1:
                self.axis = [0]
            shape = self.input_shape[0]
        else:
            self.axis = [0]
            shape = input_shape

        self.register_buffer("running_mean", torch.zeros(shape, dtype=torch.float64))
        self.register_buffer("running_var", torch.ones(shape, dtype=torch.float64))
        self.register_buffer("count", torch.ones([1], dtype=torch.float64))

    @staticmethod
    @torch.jit.script
    def _update_mean_var_count_from_moments(
        mean: Tensor,
        var: Tensor,
        count: Tensor,
        batch_mean: Tensor,
        batch_var: Tensor,
        batch_count: int,
    ):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta**2) * count * batch_count / tot_count
        new_var = M2 / tot_count
        return new_mean, new_var, tot_count

    def forward(self, x: Tensor, denormalize: bool = False) -> None:
        """Normalizes in-place! This function modifies the input tensor and returns nothing."""
        if self.training and not denormalize:
            assert x.shape[1:] == self.input_shape or (
                x.shape[1:] == () and self.input_shape == (1,)
            ), f"RMS expected input shape {self.input_shape}, got {x.shape[1:]}"

            batch_count = x.size()[0]
            mu = x.mean(self.axis)
            sigma2 = x.var(self.axis)
            self.running_mean[:], self.running_var[:], self.count[:] = (
                self._update_mean_var_count_from_moments(
                    self.running_mean, self.running_var, self.count, mu, sigma2, batch_count
                )
            )

        if self.per_channel:
            if len(self.input_shape) == 3:
                current_mean = self.running_mean.view([1, self.input_shape[0], 1, 1]).expand_as(x)
                current_var = self.running_var.view([1, self.input_shape[0], 1, 1]).expand_as(x)
            elif len(self.input_shape) == 2:
                current_mean = self.running_mean.view([1, self.input_shape[0], 1]).expand_as(x)
                current_var = self.running_var.view([1, self.input_shape[0], 1]).expand_as(x)
            elif len(self.input_shape) == 1:
                current_mean = self.running_mean.view([1, self.input_shape[0]]).expand_as(x)
                current_var = self.running_var.view([1, self.input_shape[0]]).expand_as(x)
            else:
                raise RuntimeError(f"RunningMeanStd input shape {self.input_shape} not supported")
        else:
            current_mean = self.running_mean
            current_var = self.running_var

        mu = current_mean.float()
        sigma2 = current_var.float()
        sigma = torch.sqrt(sigma2 + self.eps)
        clip = self.clip

        if self.norm_only:
            if denormalize:
                x.mul_(sigma)
            else:
                x.mul_(1 / sigma)
        else:
            if denormalize:
                x.clamp_(-clip, clip).mul_(sigma).add_(mu)
            else:
                x.sub_(mu).mul_(1 / sigma).clamp_(-clip, clip)


class RunningMeanStdDictInPlace(nn.Module):
    def __init__(
        self,
        obs_space,
        keys_to_normalize: Optional[List[str]] = None,
        epsilon=_NORM_EPS,
        clip=_DEFAULT_CLIP,
        per_channel=False,
        norm_only=False,
    ):
        super(RunningMeanStdDictInPlace, self).__init__()
        self.obs_space: Final = obs_space
        self.running_mean_std = nn.ModuleDict(
            {
                k: RunningMeanStdInPlace(space.shape, epsilon, clip, per_channel, norm_only)
                for k, space in obs_space.spaces.items()
                if keys_to_normalize is None or k in keys_to_normalize
            }
        )

    def forward(self, x: Dict[str, Tensor]) -> None:
        for k, module in self.running_mean_std.items():
            module(x[k])


def running_mean_std_summaries(
    running_mean_std_module: Union[nn.Module, ScriptModule, RecursiveScriptModule],
):
    m = running_mean_std_module
    res = dict()

    for name, buf in m.named_buffers():
        name = "_".join(name.split(".")[-2:])
        if name.endswith("running_mean"):
            res[name] = buf.float().mean()
        elif name.endswith("running_var"):
            res[name.replace("_var", "_std")] = torch.sqrt(buf.float() + _NORM_EPS).mean()

    return res


# ---- sample_factory/utils/normalize.py (vendored verbatim, minus the dict-structure helper
#      import which is inlined) ----
EPS = 1e-8


def _copy_dict_structure(d):
    res = {}
    for k, v in d.items():
        res[k] = _copy_dict_structure(v) if isinstance(v, dict) else None
    return res


class ObservationNormalizer(nn.Module):
    def __init__(self, obs_space, cfg):
        super().__init__()

        self.sub_mean = cfg.obs_subtract_mean
        self.scale = cfg.obs_scale

        self.running_mean_std = None
        if cfg.normalize_input:
            self.running_mean_std = RunningMeanStdDictInPlace(obs_space, cfg.normalize_input_keys)

        self.should_sub_mean = abs(self.sub_mean) > EPS
        self.should_scale = abs(self.scale - 1.0) > EPS
        self.should_normalize = (
            self.should_sub_mean or self.should_scale or self.running_mean_std is not None
        )

    @staticmethod
    def _clone_tensordict(obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        obs_clone = {}
        for k, x in obs_dict.items():
            if x.dtype != torch.float:
                obs_clone[k] = x.float()
            else:
                obs_clone[k] = x.clone()
        return obs_clone

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not self.should_normalize:
            return obs_dict

        with torch.no_grad():
            obs_clone = self._clone_tensordict(obs_dict)

            if self.should_sub_mean:
                obs_clone["obs"].sub_(self.sub_mean)

            if self.should_scale:
                obs_clone["obs"].mul_(1.0 / self.scale)

            if self.running_mean_std:
                self.running_mean_std(obs_clone)

        return obs_clone

    def summaries(self) -> Dict:
        res = dict()
        if self.running_mean_std:
            res.update(running_mean_std_summaries(self.running_mean_std))
        return res


# ---- sample_factory/model/actor_critic.py (vendored: ActorCritic + ActorCriticSharedWeights;
#      ActorCriticSeparateWeights and the continuous-action-space branches are omitted from
#      this vendor slice, but the shared-weights path -- the library's default -- is verbatim) ----
class ActorCritic(nn.Module, Configurable):
    def __init__(self, obs_space, action_space, cfg):
        nn.Module.__init__(self)
        Configurable.__init__(self, cfg)
        self.action_space = action_space
        self.encoders = []

        self.obs_normalizer: ObservationNormalizer = ObservationNormalizer(obs_space, cfg)

        self.returns_normalizer = None
        if cfg.normalize_returns:
            returns_shape = (1,)
            self.returns_normalizer = RunningMeanStdInPlace(returns_shape)
            self.returns_normalizer = torch.jit.script(self.returns_normalizer)

        self.last_action_distribution = None

    def get_action_parameterization(self, decoder_output_size: int):
        return ActionParameterizationDefault(self.cfg, decoder_output_size, self.action_space)

    def model_to_device(self, device):
        for module in self.children():
            if hasattr(module, "model_to_device"):
                module.model_to_device(device)
            else:
                module.to(device)

    def device_for_input_tensor(self, input_tensor_name: str) -> torch.device:
        device = self.encoders[0].device_for_input_tensor(input_tensor_name)
        if device is None:
            device = model_device(self)
        return device

    def type_for_input_tensor(self, input_tensor_name: str) -> torch.dtype:
        return self.encoders[0].type_for_input_tensor(input_tensor_name)

    def initialize_weights(self, layer):
        gain = self.cfg.policy_init_gain

        if hasattr(layer, "bias") and isinstance(layer.bias, torch.nn.parameter.Parameter):
            layer.bias.data.fill_(0)

        if self.cfg.policy_initialization == "orthogonal":
            if type(layer) is nn.Conv2d or type(layer) is nn.Linear:
                nn.init.orthogonal_(layer.weight.data, gain=gain)
        elif self.cfg.policy_initialization == "xavier_uniform":
            if type(layer) is nn.Conv2d or type(layer) is nn.Linear:
                nn.init.xavier_uniform_(layer.weight.data, gain=gain)
        elif self.cfg.policy_initialization == "torch_default":
            pass

    def normalize_obs(self, obs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self.obs_normalizer(obs)

    def action_distribution(self):
        return self.last_action_distribution

    def _maybe_sample_actions(self, sample_actions: bool, result: TensorDict) -> None:
        if sample_actions:
            actions, result["log_prob_actions"] = sample_actions_log_probs(
                self.last_action_distribution
            )
            assert actions.dim() == 2
            result["actions"] = actions.squeeze(dim=1)

    def forward_head(self, normalized_obs_dict: Dict[str, Tensor]) -> Tensor:
        raise NotImplementedError()

    def forward_core(self, head_output, rnn_states):
        raise NotImplementedError()

    def forward_tail(
        self, core_output, values_only: bool, sample_actions: bool, action_mask=None
    ) -> TensorDict:
        raise NotImplementedError()

    def forward(
        self, normalized_obs_dict, rnn_states, values_only: bool = False, action_mask=None
    ) -> TensorDict:
        raise NotImplementedError()


class ActorCriticSharedWeights(ActorCritic):
    def __init__(self, obs_space, action_space, cfg):
        super().__init__(obs_space, action_space, cfg)

        # in case of shared weights we're using only a single encoder and a single core
        self.encoder = default_make_encoder_func(cfg, obs_space)
        self.encoders = [self.encoder]

        self.core = default_make_core_func(cfg, self.encoder.get_out_size())

        self.decoder = default_make_decoder_func(cfg, self.core.get_out_size())
        decoder_out_size: int = self.decoder.get_out_size()

        self.critic_linear = nn.Linear(decoder_out_size, 1)
        self.action_parameterization = self.get_action_parameterization(decoder_out_size)

        self.apply(self.initialize_weights)

    def forward_head(self, normalized_obs_dict: Dict[str, Tensor]) -> Tensor:
        x = self.encoder(normalized_obs_dict)
        return x

    def forward_core(self, head_output: Tensor, rnn_states):
        x, new_rnn_states = self.core(head_output, rnn_states)
        return x, new_rnn_states

    def forward_tail(
        self, core_output, values_only: bool, sample_actions: bool, action_mask=None
    ) -> TensorDict:
        decoder_output = self.decoder(core_output)
        values = self.critic_linear(decoder_output).squeeze()

        result = TensorDict(values=values)
        if values_only:
            return result

        action_distribution_params, self.last_action_distribution = self.action_parameterization(
            decoder_output, action_mask
        )

        result["action_logits"] = action_distribution_params

        self._maybe_sample_actions(sample_actions, result)
        return result

    def forward(
        self, normalized_obs_dict, rnn_states, values_only=False, action_mask=None
    ) -> TensorDict:
        x = self.forward_head(normalized_obs_dict)
        x, new_rnn_states = self.forward_core(x, rnn_states)
        result = self.forward_tail(x, values_only, sample_actions=True, action_mask=action_mask)
        result["new_rnn_states"] = new_rnn_states
        return result


# ---- staging wrapper ----
def _make_cfg() -> AttrDict:
    """A minimal cfg matching sample_factory's default_cfg for a discrete-action,
    single-image-observation, non-recurrent Atari/VizDoom-style policy (the library's
    most common configuration -- convnet_atari encoder, identity core, small MLP
    decoder)."""
    return AttrDict(
        nonlinearity="relu",
        encoder_mlp_layers=[64],
        encoder_conv_architecture="convnet_atari",
        encoder_conv_mlp_layers=[128],
        decoder_mlp_layers=[64],
        use_rnn=False,
        rnn_type="gru",
        rnn_size=64,
        rnn_num_layers=1,
        actor_critic_share_weights=True,
        policy_init_gain=1.0,
        policy_initialization="orthogonal",
        normalize_returns=False,
        normalize_input=True,
        normalize_input_keys=None,
        obs_subtract_mean=0.0,
        obs_scale=255.0,
        adaptive_stddev=False,
        continuous_tanh_scale=0.0,
        initial_stddev=1.0,
    )


def build_sample_factory_actor_critic():
    cfg = _make_cfg()
    obs_space = spaces.Dict(
        {"obs": spaces.Box(low=0, high=255, shape=(4, 72, 72), dtype="float32")}
    )
    action_space = spaces.Discrete(6)
    model = ActorCriticSharedWeights(obs_space, action_space, cfg)
    model.eval()
    return model


def example_input_sample_factory_actor_critic():
    torch.manual_seed(0)
    batch = 2
    obs_dict = {"obs": torch.rand(batch, 4, 72, 72)}
    rnn_states = torch.zeros(batch, 1)  # unused by ModelCoreIdentity, kept for signature parity
    return (obs_dict, rnn_states)


MENAGERIE_ENTRIES = [
    (
        "SampleFactory_ActorCritic",
        "build_sample_factory_actor_critic",
        "example_input_sample_factory_actor_critic",
        2020,
        "vendored-pytorch",
    ),
]
