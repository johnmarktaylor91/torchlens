# SOURCE: vendored from jiupinjia/stylized-neural-painting @ master (networks.py)
# SOURCE: vendored from BrunoKM/deep-pilco-torch @ master (torchpilco/dynamics_models.py, policy_models.py)
# SOURCE: vendored from Bogacz-Group/PredictiveCoding @ main (predictive_coding/pc_layer.py)
# SOURCE: vendored from facebookresearch/deepmeg-recurrent-encoder @ master (neural/model.py)
# SOURCE: vendored from baccuslab/torch-deep-retina @ master (torchdeepretina/models.py, custom_modules.py)
# SOURCE: vendored from xdfeng7370/Deep-Ritz-Method @ master (deep_ritz_ls.py)
# SOURCE: vendored from faisaljayousi/LISTA-SC @ master (src/architecture.py)
# SOURCE: vendored from thw1021/BNNRANS @ main (bnn_training/nn/turbnn.py)
# SOURCE: vendored from lukasruff/Deep-SVDD-PyTorch @ master (src/networks/mnist_LeNet.py)
# SOURCE: vendored from Sherry-Xu/Deep-Switching-State-Space-Model @ main (src/DSSSMCode.py)
# SOURCE: vendored from lululxvi/deepxde @ master (deepxde/nn/pytorch/deeponet.py, fnn.py)
from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

MENAGERIE_ZOO = "vendored-pytorch"


class _RendererSpec:
    """Small stand-in for stylized-neural-painting renderer metadata."""

    def __init__(self, d: int = 8, d_shape: int = 4, renderer: str = "markerpen") -> None:
        """Initialize renderer metadata."""
        self.d = d
        self.d_shape = d_shape
        self.renderer = renderer


class PixelShuffleNet32(nn.Module):
    """PixelShuffleNet_32 from Stylized Neural Painting."""

    def __init__(self, input_nc: int) -> None:
        """Initialize the 32-pixel stroke renderer."""
        super().__init__()
        self.fc1 = nn.Linear(input_nc, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.conv1 = nn.Conv2d(8, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 4 * 3, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x: Tensor) -> Tensor:
        """Render a compact stroke parameter tensor."""
        x = x.squeeze()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.view(-1, 8, 16, 16)
        x = F.relu(self.conv1(x))
        x = self.pixel_shuffle(self.conv2(x))
        return x.view(-1, 3, 32, 32)


class DCGAN32(nn.Module):
    """DCGAN_32 renderer from Stylized Neural Painting."""

    def __init__(self, rdrr: _RendererSpec, ngf: int = 8) -> None:
        """Initialize the 32-pixel deconvolution renderer."""
        super().__init__()
        input_nc = rdrr.d
        self.out_size = 32
        self.main = nn.Sequential(
            nn.ConvTranspose2d(input_nc, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, 6, 4, 2, 1, bias=False),
        )

    def forward(self, input: Tensor) -> tuple[Tensor, Tensor]:
        """Render foreground and alpha tensors."""
        output_tensor = self.main(input)
        return output_tensor[:, 0:3, :, :], output_tensor[:, 3:6, :, :]


class ZouFCNFusionLight(nn.Module):
    """Light fusion renderer from Stylized Neural Painting."""

    def __init__(self, rdrr: _RendererSpec) -> None:
        """Initialize the fusion renderer."""
        super().__init__()
        self.rdrr = rdrr
        self.out_size = 32
        self.huangnet = PixelShuffleNet32(rdrr.d_shape)
        self.dcgan = DCGAN32(rdrr)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Fuse shape mask and color renderer outputs."""
        x_shape = x[:, 0 : self.rdrr.d_shape, :, :]
        x_alpha = x[:, [-1], :, :]
        if self.rdrr.renderer in ["oilpaintbrush", "airbrush"]:
            x_alpha = torch.tensor(1.0, device=x.device)
        mask = self.huangnet(x_shape)
        color, _ = self.dcgan(x)
        return color * mask, x_alpha * mask


class MCDropoutDynamicsNN(nn.Module):
    """MC-Dropout dynamics model from deep-pilco-torch."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int = 200,
        drop_prob: float = 0.1,
        batch_size: int = 1,
        drop_input: bool = True,
    ) -> None:
        """Initialize the dynamics model."""
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.drop_prob = drop_prob
        self.drop_input = drop_input
        self.input_mask: Tensor | None = None
        self.hidden1_mask: Tensor | None = None
        self.hidden2_mask: Tensor | None = None
        self.batch_size = batch_size
        self.fc1 = nn.Linear(self.input_dim, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_dim)
        self.sample_new_mask(batch_size)

    def forward(self, x: Tensor) -> Tensor:
        """Predict transition dynamics."""
        if x.shape[0] != self.batch_size:
            msg = (
                f"The input batch dimension is {x.shape[0]}, but the size of "
                f"the sampled dropout mask is {self.batch_size}"
            )
            raise ValueError(msg)
        x_in = x * self.input_mask if self.drop_input else x
        hidden1 = torch.sigmoid(self.fc1(x_in))
        hidden2 = torch.sigmoid(self.fc2(hidden1 * self.hidden1_mask))
        return self.out(hidden2 * self.hidden2_mask)

    def sample_new_mask(self, batch_size: int | None = None) -> None:
        """Sample fixed MC-dropout masks."""
        device = self.get_param_device()
        if batch_size:
            self.batch_size = batch_size
        self.input_mask = torch.bernoulli(
            torch.ones(self.batch_size, self.input_dim) * (1 - self.drop_prob)
        ).to(device)
        self.hidden1_mask = torch.bernoulli(
            torch.ones(self.batch_size, self.hidden_size) * (1 - self.drop_prob)
        ).to(device)
        self.hidden2_mask = torch.bernoulli(
            torch.ones(self.batch_size, self.hidden_size) * (1 - self.drop_prob)
        ).to(device)

    def get_param_device(self) -> torch.device:
        """Return the current parameter device."""
        return next(self.parameters()).device


class PCLayer(nn.Module):
    """Predictive-coding layer from Bogacz-Group/PredictiveCoding."""

    def __init__(
        self,
        energy_fn: Callable[[dict[str, Tensor]], Tensor] | None = None,
        energy_fn_kwargs: dict[str, Any] | None = None,
        sample_x_fn: Callable[[dict[str, Tensor]], Tensor] | None = None,
        is_holding_error: bool = False,
    ) -> None:
        """Initialize the predictive-coding layer."""
        super().__init__()
        self._energy_fn = energy_fn or (lambda inputs: 0.5 * (inputs["mu"] - inputs["x"]) ** 2)
        self._energy_fn_kwargs = energy_fn_kwargs or {}
        self._sample_x_fn = sample_x_fn or (lambda inputs: inputs["mu"].detach().clone())
        self.is_holding_error = is_holding_error
        self._energy: Tensor | None = None
        self._error: Tensor | None = None
        self._is_sample_x = False
        self._x: nn.Parameter | None = None
        self.eval()

    def set_is_sample_x(self, is_sample_x: bool) -> None:
        """Set whether the value node should be resampled."""
        self._is_sample_x = is_sample_x

    def energy(self) -> Tensor | None:
        """Return the currently held energy."""
        return self._energy

    def clear_energy(self) -> None:
        """Clear held energy."""
        self._energy = None

    def forward(
        self,
        mu: Tensor,
        energy_fn_additional_inputs: dict[str, Tensor] | None = None,
    ) -> Tensor:
        """Run the predictive-coding layer."""
        additional = energy_fn_additional_inputs or {}
        if self.training:
            if (
                self._x is None
                or self._is_sample_x
                or mu.device != self._x.device
                or mu.size() != self._x.size()
            ):
                sampled = self._sample_x_fn({"mu": mu})
                self._x = nn.Parameter(sampled)
                self._is_sample_x = False
            inputs = {"mu": mu, "x": self._x, **additional}
            energy = self._energy_fn(inputs, **self._energy_fn_kwargs)
            self._energy = energy.sum()
            if self.is_holding_error:
                self._error = (mu - self._x).detach().clone()
            return self._x
        return mu


def center_trim(tensor: Tensor, reference: int | Tensor) -> Tensor:
    """Trim a sequence tensor to the target length."""
    ref_size = reference if isinstance(reference, int) else reference.size(-1)
    delta = tensor.size(-1) - ref_size
    if delta < 0:
        raise ValueError("tensor must be longer than reference")
    if delta:
        tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
    return tensor


class MegPredictor(nn.Module):
    """Deep MEG recurrent encoder model."""

    def __init__(
        self,
        meg_dim: int,
        forcing_dims: dict[str, int],
        meg_init: int = 40,
        n_subjects: int = 100,
        subject_dim: int = 16,
        conv_layers: int = 2,
        kernel: int = 4,
        stride: int = 2,
        conv_channels: int = 256,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
    ) -> None:
        """Initialize the MEG predictor."""
        super().__init__()
        self.forcing_dims = dict(forcing_dims)
        self.meg_init = meg_init
        in_channels = meg_dim + 1 + subject_dim + sum(forcing_dims.values())
        self.subject_embedding = nn.Embedding(n_subjects, subject_dim) if subject_dim else None
        channels = conv_channels
        encoder: list[nn.Module] = []
        for _ in range(conv_layers):
            encoder += [
                nn.Conv1d(in_channels, channels, kernel, stride, padding=kernel // 2),
                nn.ReLU(),
            ]
            in_channels = channels
        self.encoder = nn.Sequential(*encoder)
        self.lstm = nn.LSTM(in_channels, lstm_hidden, lstm_layers) if lstm_layers else None
        if self.lstm is not None:
            in_channels = lstm_hidden
        self.conv_layers = conv_layers
        self.stride = stride
        self.kernel = kernel
        if conv_layers == 0:
            self.decoder = nn.Conv1d(in_channels, meg_dim, 1)
        else:
            decoder: list[nn.Module] = []
            for index in range(conv_layers):
                if index == conv_layers - 1:
                    channels = meg_dim
                decoder.append(
                    nn.ConvTranspose1d(in_channels, channels, kernel, stride, padding=kernel // 2)
                )
                if index < conv_layers - 1:
                    decoder.append(nn.ReLU())
                in_channels = channels
            self.decoder = nn.Sequential(*decoder)

    def get_meg_mask(self, meg: Tensor, forcings: dict[str, Tensor]) -> Tensor:
        """Build the observed-prefix mask."""
        batch, _, time = meg.size()
        mask = torch.zeros(batch, 1, time, device=meg.device)
        mask[:, :, : self.meg_init] = 1.0
        return mask

    def valid_length(self, length: int) -> int:
        """Return the valid padded length."""
        for _ in range(self.conv_layers):
            length = math.ceil(length / self.stride) + 1
        for _ in range(self.conv_layers):
            length = (length - 1) * self.stride
        return int(length)

    def pad(self, x: Tensor) -> Tensor:
        """Pad a temporal tensor to a valid length."""
        length = x.size(-1)
        valid_length = self.valid_length(length)
        delta = valid_length - length
        return F.pad(x, (delta // 2, delta - delta // 2))

    def forward(self, meg: Tensor, forcings: dict[str, Tensor], subject_id: Tensor) -> Tensor:
        """Predict MEG responses."""
        forcings = dict(forcings)
        batch, _, length = meg.size()
        inputs = []
        mask = self.get_meg_mask(meg, forcings)
        meg = meg * mask
        inputs += [meg, mask]
        if self.subject_embedding is not None:
            subject = self.subject_embedding(subject_id)
            inputs.append(subject.view(batch, -1, 1).expand(-1, -1, length))
        if self.forcing_dims:
            _, sorted_forcings = zip(
                *sorted((k, v) for k, v in forcings.items() if k in self.forcing_dims)
            )
        else:
            sorted_forcings = ()
        inputs.extend(sorted_forcings)
        x = torch.cat(inputs, dim=1)
        x = self.pad(x)
        x = self.encoder(x)
        if self.lstm is not None:
            x = x.permute(2, 0, 1)
            x, _ = self.lstm(x)
            x = x.permute(1, 2, 0)
        return center_trim(self.decoder(x), length)


class Flatten(nn.Module):
    """Flatten helper from torch-deep-retina."""

    def forward(self, x: Tensor) -> Tensor:
        """Flatten all non-batch dimensions."""
        return x.view(x.size(0), -1)


class Reshape(nn.Module):
    """Reshape helper from torch-deep-retina."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize the target shape."""
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        """Reshape the tensor."""
        return x.view(*self.shape)


class GaussianNoise(nn.Module):
    """GaussianNoise helper from torch-deep-retina."""

    def __init__(self, std: float = 0.05) -> None:
        """Initialize the noise layer."""
        super().__init__()
        self.std = std

    def forward(self, x: Tensor) -> Tensor:
        """Add Gaussian noise while training."""
        if not self.training or self.std == 0:
            return x
        return x + torch.empty_like(x).normal_(0, self.std)


class Exponential(nn.Module):
    """Exponential activation helper from torch-deep-retina."""

    def __init__(self, train_off: bool = False) -> None:
        """Initialize the activation."""
        super().__init__()
        self.train_off = train_off

    def forward(self, x: Tensor) -> Tensor:
        """Apply exp unless disabled during training."""
        if self.train_off and self.training:
            return x
        return torch.exp(x)


def update_shape(
    shape: tuple[int, int], kernel: int, padding: int = 0, stride: int = 1
) -> tuple[int, int]:
    """Return the spatial shape after a valid convolution."""
    return tuple(int((dim + 2 * padding - kernel) / stride + 1) for dim in shape)


class TDRModel(nn.Module):
    """Base model from torch-deep-retina."""

    def __init__(
        self,
        n_units: int = 5,
        noise: float = 0.05,
        bias: bool = True,
        gc_bias: bool | None = None,
        chans: list[int] | None = None,
        bn_moment: float = 0.01,
        softplus: bool = True,
        inference_exp: bool = False,
        img_shape: tuple[int, int, int] = (40, 50, 50),
        ksizes: tuple[int, int, int] = (15, 11, 11),
        groups: int = 1,
        bnorm_d: int = 1,
        activ_fxn: str = "ReLU",
        bnaftrelu: bool = False,
    ) -> None:
        """Initialize shared retinal model settings."""
        super().__init__()
        self.n_units = n_units
        self.chans = chans or [8, 8]
        self.softplus = softplus
        self.infr_exp = inference_exp
        self.bias = bias
        self.img_shape = img_shape
        self.ksizes = ksizes
        self.groups = groups
        self.gc_bias = gc_bias
        self.noise = noise
        self.bn_moment = bn_moment
        self.bnorm_d = bnorm_d
        self.activ_fxn = activ_fxn
        self.bnaftrelu = bnaftrelu


class BNCNN(TDRModel):
    """Batch-normalized CNN from torch-deep-retina."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the retinal CNN."""
        super().__init__(**kwargs)
        modules: list[nn.Module] = []
        shape = self.img_shape[1:]
        modules.append(
            nn.Conv2d(self.img_shape[0], self.chans[0], self.ksizes[0], bias=self.bias, groups=1)
        )
        shape = update_shape(shape, self.ksizes[0])
        if self.bnaftrelu:
            modules += [GaussianNoise(std=self.noise), getattr(nn, self.activ_fxn)()]
        if self.bnorm_d == 1:
            modules.append(Flatten())
            size = self.chans[0] * shape[0] * shape[1]
            modules += [
                nn.BatchNorm1d(size, eps=1e-3, momentum=self.bn_moment),
                Reshape((-1, self.chans[0], *shape)),
            ]
        else:
            modules.append(nn.BatchNorm2d(self.chans[0], eps=1e-3, momentum=self.bn_moment))
        if not self.bnaftrelu:
            modules += [GaussianNoise(std=self.noise), getattr(nn, self.activ_fxn)()]
        modules.append(
            nn.Conv2d(
                self.chans[0], self.chans[1], self.ksizes[1], bias=self.bias, groups=self.groups
            )
        )
        shape = update_shape(shape, self.ksizes[1])
        if self.bnaftrelu:
            modules += [GaussianNoise(std=self.noise), getattr(nn, self.activ_fxn)()]
        if self.bnorm_d == 1:
            modules.append(Flatten())
            size = self.chans[1] * shape[0] * shape[1]
            modules += [
                nn.BatchNorm1d(size, eps=1e-3, momentum=self.bn_moment),
                Reshape((-1, self.chans[1], *shape)),
            ]
        else:
            modules.append(nn.BatchNorm2d(self.chans[1], eps=1e-3, momentum=self.bn_moment))
        if not self.bnaftrelu:
            modules += [GaussianNoise(std=self.noise), getattr(nn, self.activ_fxn)()]
        modules += [
            Flatten(),
            nn.Linear(self.chans[1] * shape[0] * shape[1], self.n_units, bias=self.gc_bias),
        ]
        modules.append(nn.BatchNorm1d(self.n_units, eps=1e-3, momentum=self.bn_moment))
        modules.append(nn.Softplus() if self.softplus else Exponential(train_off=True))
        self.sequential = nn.Sequential(*modules)

    def forward(self, x: Tensor) -> Tensor:
        """Run the retinal CNN."""
        if not self.training and self.infr_exp:
            return torch.exp(self.sequential(x))
        return self.sequential(x)


class Block(nn.Module):
    """Residual block from Deep Ritz Method."""

    def __init__(self, in_n: int, width: int, out_n: int) -> None:
        """Initialize the Deep Ritz residual block."""
        super().__init__()
        self.L1 = nn.Linear(in_n, width)
        self.L2 = nn.Linear(width, out_n)
        self.phi = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        """Run the residual block."""
        return self.phi(self.L2(self.phi(self.L1(x)))) + x


class DRRNN(nn.Module):
    """Deep Ritz residual neural network."""

    def __init__(self, in_n: int, m: int, out_n: int, depth: int = 5) -> None:
        """Initialize the Deep Ritz network."""
        super().__init__()
        self.stack = nn.ModuleList([nn.Linear(in_n, m)])
        for _ in range(depth):
            self.stack.append(Block(m, m, m))
        self.stack.append(nn.Linear(m, out_n))

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate the Deep Ritz network."""
        for layer in self.stack:
            x = layer(x)
        return x


class LISTA(nn.Module):
    """LISTA sparse-coding network."""

    def __init__(
        self, n: int, m: int, w_d: Tensor, max_iterations: int, lipschitz_const: float, theta: float
    ) -> None:
        """Initialize the LISTA network."""
        super().__init__()
        self.W_d = w_d
        self.max_iterations = max_iterations
        self.lipschitz_const = lipschitz_const
        self._W = nn.Linear(n, m, bias=False)
        self._S = nn.Linear(m, m, bias=False)
        self.shrinkage = nn.Softshrink(theta)
        self.weights_init()

    def weights_init(self) -> None:
        """Initialize LISTA weights from the dictionary."""
        a = self.W_d.cpu()
        s = torch.eye(a.shape[1]) - (1 / self.lipschitz_const) * torch.matmul(a.T, a)
        w = (1 / self.lipschitz_const) * a.T
        self._S.weight = nn.Parameter(s.float(), requires_grad=True)
        self._W.weight = nn.Parameter(w.float(), requires_grad=True)

    def forward(self, y: Tensor) -> Tensor:
        """Estimate the sparse code."""
        x = self.shrinkage(self._W(y))
        if self.max_iterations == 1:
            return x
        for _ in range(self.max_iterations):
            x = self.shrinkage(self._W(y) + self._S(x))
        return x


class TurbNN(nn.Module):
    """Turbulence neural network from BNNRANS."""

    def __init__(self, d_in: int, h: int, d_out: int) -> None:
        """Initialize the turbulence network."""
        super().__init__()
        self.linear1 = nn.Linear(d_in, h)
        self.f1 = nn.LeakyReLU()
        self.linear2 = nn.Linear(h, h)
        self.f2 = nn.LeakyReLU()
        self.linear3 = nn.Linear(h, h)
        self.f3 = nn.LeakyReLU()
        self.linear4 = nn.Linear(h, h)
        self.f4 = nn.LeakyReLU()
        self.linear5 = nn.Linear(h, int(h / 5))
        self.f5 = nn.LeakyReLU()
        self.linear6 = nn.Linear(int(h / 5), int(h / 10))
        self.f6 = nn.LeakyReLU()
        self.linear7 = nn.Linear(int(h / 10), d_out)

    def forward(self, x: Tensor) -> Tensor:
        """Run the turbulence network."""
        lin1 = self.f1(self.linear1(x))
        lin2 = self.f2(self.linear2(lin1))
        lin3 = self.f3(self.linear3(lin2))
        lin4 = self.f4(self.linear4(lin3))
        lin5 = self.f5(self.linear5(lin4))
        lin6 = self.f6(self.linear6(lin5))
        return self.linear7(lin6)


class MNISTLeNet(nn.Module):
    """Deep SVDD MNIST LeNet."""

    def __init__(self) -> None:
        """Initialize the Deep SVDD LeNet."""
        super().__init__()
        self.rep_dim = 32
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-4, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-4, affine=False)
        self.fc1 = nn.Linear(4 * 7 * 7, self.rep_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Return the SVDD representation."""
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        return self.fc1(x)


class DSSSM(nn.Module):
    """Deep Switching State-Space Model."""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        h_dim: int,
        z_dim: int,
        d_dim: int,
        n_layers: int,
        device: torch.device,
        bidirection: bool = False,
        dataname: str | None = None,
    ) -> None:
        """Initialize DSSSM."""
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.d_dim = d_dim
        self.n_layers = n_layers
        self.device = device
        self.temperature = 0.5
        self.bidirection = bidirection
        self.dataname = dataname
        self.Transition_initial = (
            torch.eye(self.d_dim, device=self.device) * (1 - 0.05 * self.d_dim)
            + torch.ones((self.d_dim, self.d_dim), device=self.device) * 0.05
        )
        self.dprior = nn.Sequential(nn.Linear(d_dim, d_dim), nn.Softmax(dim=1))
        self.ztrainsition_list = nn.ModuleList()
        self.ztrainsition_mean_list = nn.ModuleList()
        self.ztrainsition_std_list = nn.ModuleList()
        self.dposterior_list = nn.ModuleList()
        self.zposterior_list = nn.ModuleList()
        self.zposterior_mean_list = nn.ModuleList()
        self.zposterior_std_list = nn.ModuleList()
        self.yemission_list = nn.ModuleList()
        self.yemission_mean_list = nn.ModuleList()
        self.yemission_std_list = nn.ModuleList()
        for _ in range(self.d_dim):
            self.dposterior_list.append(nn.Sequential(nn.Linear(h_dim, d_dim), nn.Softmax(dim=1)))
            self.zposterior_list.append(
                nn.Sequential(
                    nn.Linear(z_dim + h_dim, z_dim), nn.ReLU(), nn.Linear(z_dim, z_dim), nn.ReLU()
                )
            )
            self.zposterior_mean_list.append(nn.Linear(z_dim, z_dim))
            self.zposterior_std_list.append(nn.Sequential(nn.Linear(z_dim, z_dim), nn.Softplus()))
            self.ztrainsition_list.append(
                nn.Sequential(
                    nn.Linear(z_dim + h_dim, z_dim), nn.ReLU(), nn.Linear(z_dim, z_dim), nn.ReLU()
                )
            )
            self.ztrainsition_mean_list.append(nn.Linear(z_dim, z_dim))
            self.ztrainsition_std_list.append(nn.Sequential(nn.Linear(z_dim, z_dim), nn.Softplus()))
            self.yemission_list.append(
                nn.Sequential(
                    nn.Linear(z_dim + h_dim, y_dim), nn.ReLU(), nn.Linear(y_dim, y_dim), nn.ReLU()
                )
            )
            self.yemission_mean_list.append(nn.Linear(y_dim, y_dim))
            self.yemission_std_list.append(nn.Sequential(nn.Linear(y_dim, y_dim), nn.Softplus()))
        self.rnn_forward = nn.GRU(x_dim, h_dim, n_layers, bidirectional=False)
        hidden = int(h_dim / 2) if self.bidirection else h_dim
        self.rnn_backward = nn.GRU(y_dim + h_dim, hidden, n_layers, bidirectional=self.bidirection)

    def TransitionMatrix(self) -> Tensor:
        """Return the learned transition matrix."""
        if self.dataname == "Sleep":
            return (
                self.dprior(self.Transition_initial) * 0.2
                + torch.eye(self.d_dim, device=self.device) * 0.8
            )
        return (
            self.dprior(self.Transition_initial) / 2 + torch.eye(self.d_dim, device=self.device) / 2
        )

    def forward(self, x: Tensor, y: Tensor) -> tuple[Any, ...]:
        """Run DSSSM inference and loss terms."""
        transition = self.TransitionMatrix()
        all_d_posterior = [torch.ones((x.size(1), self.d_dim), device=self.device) / self.d_dim]
        samples = torch.distributions.Categorical(
            torch.ones((self.d_dim), device=self.device) / self.d_dim
        ).sample((x.size(1),))
        all_d_t_sampled = [self._one_hot_encode(samples, self.d_dim)]
        all_z_posterior_mean = [torch.zeros((x.size(1), self.z_dim), device=self.device)]
        all_z_posterior_std = [torch.zeros((x.size(1), self.z_dim), device=self.device)]
        all_z_t_sampled = [torch.zeros((x.size(1), self.z_dim), device=self.device)]
        all_y_emission_mean: list[Tensor] = []
        all_y_emission_std: list[Tensor] = []
        kld_gaussian_loss = torch.tensor(0.0, device=self.device)
        kld_category_loss = torch.tensor(0.0, device=self.device)
        nll_loss = torch.tensor(0.0, device=self.device)
        h0 = torch.zeros((self.n_layers, x.size(1), self.h_dim), device=self.device)
        a0_layers = self.n_layers * 2 if self.bidirection else self.n_layers
        a0_hidden = int(self.h_dim / 2) if self.bidirection else self.h_dim
        a0 = torch.zeros((a0_layers, x.size(1), a0_hidden), device=self.device)
        output_forward, _ = self.rnn_forward(x, h0)
        output_backward, _ = self.rnn_backward(
            torch.flip(torch.cat([y, output_forward], 2), [0]), a0
        )
        for t in range(x.size(0)):
            d_prior = torch.mm(all_d_t_sampled[t], transition)
            d_posterior_list = []
            d_posterior = torch.zeros_like(d_prior)
            for i in range(self.d_dim):
                d_posterior_list.append(self.dposterior_list[i](output_backward[x.size(0) - t - 1]))
                d_posterior = d_posterior + d_posterior_list[i] * all_d_t_sampled[t][:, i : (i + 1)]
            all_d_posterior.append(d_posterior)
            d_t_samples = torch.distributions.Categorical(d_posterior).sample().to(self.device)
            all_d_t_sampled.append(self._one_hot_encode(d_t_samples, self.d_dim))
            z_prior_mean_list, z_prior_std_list = [], []
            z_posterior_mean_list, z_posterior_std_list = [], []
            z_prior_mean = torch.zeros((x.size(1), self.z_dim), device=self.device)
            z_prior_std = torch.zeros_like(z_prior_mean)
            z_posterior_mean = torch.zeros_like(z_prior_mean)
            z_posterior_std = torch.zeros_like(z_prior_mean)
            for i in range(self.d_dim):
                z_prior = self.ztrainsition_list[i](
                    torch.cat([output_forward[t], all_z_t_sampled[t]], 1)
                )
                z_prior_mean_list.append(self.ztrainsition_mean_list[i](z_prior))
                z_prior_std_list.append(self.ztrainsition_std_list[i](z_prior))
                z_prior_mean = (
                    z_prior_mean + z_prior_mean_list[i] * all_d_t_sampled[t + 1][:, i : (i + 1)]
                )
                z_prior_std = (
                    z_prior_std + z_prior_std_list[i] * all_d_t_sampled[t + 1][:, i : (i + 1)]
                )
                z_post = self.zposterior_list[i](
                    torch.cat([output_backward[x.size(0) - t - 1], all_z_t_sampled[t]], 1)
                )
                z_posterior_mean_list.append(self.zposterior_mean_list[i](z_post))
                z_posterior_std_list.append(self.zposterior_std_list[i](z_post))
                z_posterior_mean = (
                    z_posterior_mean
                    + z_posterior_mean_list[i] * all_d_t_sampled[t + 1][:, i : (i + 1)]
                )
                z_posterior_std = (
                    z_posterior_std
                    + z_posterior_std_list[i] * all_d_t_sampled[t + 1][:, i : (i + 1)]
                )
            all_z_posterior_mean.append(z_posterior_mean)
            all_z_posterior_std.append(z_posterior_std)
            z_t = self._reparameterized_normal_sample(z_posterior_mean, z_posterior_std)
            all_z_t_sampled.append(z_t)
            y_emission_mean_list, y_emission_std_list = [], []
            y_emission_mean = torch.zeros((x.size(1), self.y_dim), device=self.device)
            y_emission_std = torch.zeros_like(y_emission_mean)
            for i in range(self.d_dim):
                y_emission = self.yemission_list[i](
                    torch.cat([output_forward[t], all_z_t_sampled[t + 1]], 1)
                )
                y_emission_mean_list.append(self.yemission_mean_list[i](y_emission))
                y_emission_std_list.append(self.yemission_std_list[i](y_emission))
                y_emission_mean = (
                    y_emission_mean
                    + y_emission_mean_list[i] * all_d_t_sampled[t + 1][:, i : (i + 1)]
                )
                y_emission_std = (
                    y_emission_std + y_emission_std_list[i] * all_d_t_sampled[t + 1][:, i : (i + 1)]
                )
            all_y_emission_mean.append(y_emission_mean)
            all_y_emission_std.append(y_emission_std)
            for i in range(self.d_dim):
                kld_gaussian_loss = kld_gaussian_loss + torch.sum(
                    self._kld_gauss(
                        z_posterior_mean_list[i],
                        z_posterior_std_list[i],
                        z_prior_mean_list[i],
                        z_prior_std_list[i],
                    )
                    * d_posterior[:, i : (i + 1)]
                )
                kld_category_loss = kld_category_loss + torch.sum(
                    self._kld_category(d_posterior_list[i], transition[i : (i + 1), :])
                    * all_d_posterior[-2][:, i]
                )
                nll_loss = nll_loss + torch.sum(
                    self._nll_gauss(y_emission_mean_list[i], y_emission_std_list[i], y[t])
                    * d_posterior[:, i : (i + 1)]
                )
        return (
            kld_gaussian_loss,
            kld_category_loss,
            nll_loss,
            (all_z_posterior_mean, all_z_posterior_std),
            (all_y_emission_mean, all_y_emission_std),
            all_z_t_sampled,
            all_d_posterior,
            all_d_t_sampled,
        )

    def _reparameterized_normal_sample(self, mean: Tensor, std: Tensor) -> Tensor:
        """Sample a normal latent with reparameterization."""
        eps = torch.empty(std.size(), device=self.device).normal_()
        return eps.mul(std).add_(mean)

    def _kld_gauss(self, mean_1: Tensor, std_1: Tensor, mean_2: Tensor, std_2: Tensor) -> Tensor:
        """Compute Gaussian KLD."""
        return 0.5 * (
            2 * torch.log(std_2)
            - 2 * torch.log(std_1)
            + (std_1.pow(2) + (mean_1 - mean_2).pow(2)) / std_2.pow(2)
            - 1
        )

    def _kld_category(self, d_posterior: Tensor, d_prior: Tensor) -> Tensor:
        """Compute categorical KLD."""
        return torch.sum(torch.mul(torch.log(torch.div(d_posterior, d_prior)), d_posterior), axis=1)

    def _nll_gauss(self, mean: Tensor, std: Tensor, x: Tensor) -> Tensor:
        """Compute Gaussian negative log likelihood."""
        return (
            0.5 * torch.log(torch.tensor(2 * math.pi, device=self.device))
            + torch.log(std)
            + (x - mean).pow(2) / (2 * std.pow(2))
        )

    def _one_hot_encode(self, x: Tensor, n_classes: int) -> Tensor:
        """One-hot encode sample labels."""
        return torch.eye(n_classes, device=self.device)[x]


def get_activation(name: str) -> Callable[[Tensor], Tensor]:
    """Return a DeepXDE-style activation function."""
    if name == "tanh":
        return torch.tanh
    if name == "relu":
        return F.relu
    raise ValueError(f"unsupported activation: {name}")


def apply_initializer(name: str, tensor: Tensor) -> None:
    """Apply a DeepXDE-style initializer."""
    if name == "Glorot normal":
        nn.init.xavier_normal_(tensor)
    elif name == "Glorot uniform":
        nn.init.xavier_uniform_(tensor)
    elif name == "zeros":
        nn.init.zeros_(tensor)
    else:
        raise ValueError(f"unsupported initializer: {name}")


class FNN(nn.Module):
    """DeepXDE PyTorch fully connected network."""

    def __init__(
        self,
        layer_sizes: list[int],
        activation: str,
        kernel_initializer: str,
        dropout_rate: float = 0,
    ) -> None:
        """Initialize the FNN."""
        super().__init__()
        self.activation = get_activation(activation)
        self.dropout_rate = [dropout_rate] * (len(layer_sizes) - 1)
        self.linears = nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
            apply_initializer(kernel_initializer, self.linears[-1].weight)
            apply_initializer("zeros", self.linears[-1].bias)

    def forward(self, inputs: Tensor) -> Tensor:
        """Run the FNN."""
        x = inputs
        for j, linear in enumerate(self.linears[:-1]):
            x = self.activation(linear(x))
            if self.dropout_rate[j] > 0:
                x = F.dropout(x, p=self.dropout_rate[j], training=self.training)
        return self.linears[-1](x)


class PODDeepONet(nn.Module):
    """DeepXDE PyTorch PODDeepONet."""

    def __init__(
        self,
        pod_basis: Tensor,
        layer_sizes_branch: list[int],
        activation: str,
        kernel_initializer: str,
        layer_sizes_trunk: list[int] | None = None,
        dropout_rate: float = 0,
    ) -> None:
        """Initialize PODDeepONet."""
        super().__init__()
        self.pod_basis = torch.as_tensor(pod_basis, dtype=torch.float32)
        self.activation_trunk = get_activation(activation)
        self.branch = FNN(
            layer_sizes_branch, activation, kernel_initializer, dropout_rate=dropout_rate
        )
        self.trunk = None
        if layer_sizes_trunk is not None:
            self.trunk = FNN(
                layer_sizes_trunk, activation, kernel_initializer, dropout_rate=dropout_rate
            )
            self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, inputs: tuple[Tensor, Tensor]) -> Tensor:
        """Run PODDeepONet."""
        x_func = inputs[0]
        x_loc = inputs[1]
        x_func = self.branch(x_func)
        if self.trunk is None:
            return torch.einsum("bi,ni->bn", x_func, self.pod_basis)
        x_loc = self.activation_trunk(self.trunk(x_loc))
        x = torch.cat((self.pod_basis, x_loc), 1)
        return torch.einsum("bi,ni->bn", x_func, x) + self.b


def build_deep_painterly_rendering() -> nn.Module:
    """Build the Stylized Neural Painting renderer."""
    return ZouFCNFusionLight(_RendererSpec())


def example_input_deep_painterly_rendering() -> Tensor:
    """Return an example stroke tensor."""
    return torch.randn(2, 8, 1, 1)


def build_deep_pilco() -> nn.Module:
    """Build the Deep PILCO dynamics model."""
    return MCDropoutDynamicsNN(5, 4, hidden_size=16, batch_size=2)


def example_input_deep_pilco() -> Tensor:
    """Return an example state-action tensor."""
    return torch.randn(2, 5)


def build_predictive_coding() -> nn.Module:
    """Build a predictive-coding layer."""
    return PCLayer()


def example_input_predictive_coding() -> Tensor:
    """Return an example predictive-coding tensor."""
    return torch.randn(2, 6)


def build_deep_recurrent_encoder() -> nn.Module:
    """Build the MEG recurrent encoder."""
    return MegPredictor(
        3,
        {"word": 2},
        meg_init=4,
        n_subjects=4,
        subject_dim=2,
        conv_channels=4,
        lstm_hidden=4,
        lstm_layers=1,
    )


def example_input_deep_recurrent_encoder() -> tuple[Tensor, dict[str, Tensor], Tensor]:
    """Return example MEG inputs."""
    return (
        torch.randn(2, 3, 16),
        {"word": torch.randn(2, 2, 16)},
        torch.tensor([0, 1], dtype=torch.long),
    )


def build_deep_retina_feedback() -> nn.Module:
    """Build the torch-deep-retina BNCNN."""
    return BNCNN(n_units=3, noise=0.0, chans=[4, 4], img_shape=(4, 16, 16), ksizes=(3, 3, 3))


def example_input_deep_retina_feedback() -> Tensor:
    """Return an example retinal stimulus."""
    return torch.randn(2, 4, 16, 16)


def build_deep_ritz() -> nn.Module:
    """Build the Deep Ritz network."""
    return DRRNN(2, 8, 1, depth=2)


def example_input_deep_ritz() -> Tensor:
    """Return example PDE coordinates."""
    return torch.randn(3, 2)


def build_lista_sparse_coding() -> nn.Module:
    """Build the LISTA sparse-coding network."""
    dictionary = torch.randn(5, 4)
    return LISTA(5, 4, dictionary, max_iterations=2, lipschitz_const=10.0, theta=0.1)


def example_input_lista_sparse_coding() -> Tensor:
    """Return an example measurement vector."""
    return torch.randn(2, 5)


def build_bnnrans() -> nn.Module:
    """Build the BNNRANS turbulence network."""
    return TurbNN(6, 20, 3)


def example_input_bnnrans() -> Tensor:
    """Return example turbulence features."""
    return torch.randn(2, 6)


def build_deep_svdd() -> nn.Module:
    """Build the Deep SVDD LeNet."""
    return MNISTLeNet()


def example_input_deep_svdd() -> Tensor:
    """Return example MNIST images."""
    return torch.randn(2, 1, 28, 28)


def build_dsssm() -> nn.Module:
    """Build the Deep Switching State-Space Model."""
    return DSSSM(2, 2, 4, 3, 2, 1, torch.device("cpu"))


def example_input_dsssm() -> tuple[Tensor, Tensor]:
    """Return example sequence inputs."""
    return torch.randn(3, 2, 2), torch.randn(3, 2, 2)


def build_pod_deeponet() -> nn.Module:
    """Build the DeepXDE PODDeepONet."""
    return PODDeepONet(torch.randn(5, 4), [3, 8, 4], "tanh", "Glorot uniform")


def example_input_pod_deeponet() -> tuple[Tensor, Tensor]:
    """Return example PODDeepONet inputs."""
    return torch.randn(2, 3), torch.randn(5, 2)


MENAGERIE_ENTRIES = [
    (
        "Deep Painterly Rendering",
        "build_deep_painterly_rendering",
        "example_input_deep_painterly_rendering",
        2021,
        "CV13_004",
    ),
    ("Deep PILCO", "build_deep_pilco", "example_input_deep_pilco", 2018, "CV13_008"),
    (
        "Deep Predictive Coding Network",
        "build_predictive_coding",
        "example_input_predictive_coding",
        2020,
        "CV13_014",
    ),
    (
        "Deep Recurrent Encoder for MEG",
        "build_deep_recurrent_encoder",
        "example_input_deep_recurrent_encoder",
        2020,
        "CV13_016",
    ),
    (
        "Deep Retina with recurrent feedback",
        "build_deep_retina_feedback",
        "example_input_deep_retina_feedback",
        2020,
        "CV13_019",
    ),
    ("Deep Ritz Method network", "build_deep_ritz", "example_input_deep_ritz", 2018, "CV13_020"),
    (
        "Deep Sparse Coding network",
        "build_lista_sparse_coding",
        "example_input_lista_sparse_coding",
        2023,
        "CV13_024",
    ),
    (
        "Deep Structured Neural Network Turbulence",
        "build_bnnrans",
        "example_input_bnnrans",
        2020,
        "CV13_026",
    ),
    ("Deep SVDD", "build_deep_svdd", "example_input_deep_svdd", 2018, "CV13_028"),
    ("Deep Switching State-Space Model", "build_dsssm", "example_input_dsssm", 2022, "CV13_029"),
    ("Deep-O-Net with POD", "build_pod_deeponet", "example_input_pod_deeponet", 2021, "CV13_034"),
]

warnings.filterwarnings("ignore", category=RuntimeWarning)
