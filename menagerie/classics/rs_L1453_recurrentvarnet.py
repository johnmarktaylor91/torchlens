# SOURCE: vendored from NKI-AI/direct @ f83b7bbbbf6de13f16d141a8731d1cd52a7fb834
# https://raw.githubusercontent.com/NKI-AI/direct/main/direct/nn/recurrentvarnet/recurrentvarnet.py
# https://raw.githubusercontent.com/NKI-AI/direct/main/direct/nn/recurrent/recurrent.py
# https://raw.githubusercontent.com/NKI-AI/direct/main/direct/data/transforms.py (fft2/ifft2 + helpers)
#
# Recurrent Variational Network (RecurrentVarNet), "Recurrent Variational Network:
# A Deep Learning Inverse Problem Solver Applied to the Task of Accelerated MRI
# Reconstruction" (Yiasemis et al., CVPR 2022). An unrolled variational k-space
# solver: at every one of `num_steps` iterations a `RecurrentVarNetBlock` computes
# the k-space data-consistency error, transforms the coil-combined image-domain
# residual through a convolutional-GRU regularizer (`Conv2dGRU`), and re-expands +
# re-transforms it back into a k-space update. `RecurrentInit`, `RecurrentVarNet`,
# `RecurrentVarNetBlock` (from `recurrentvarnet.py`) and `Conv2dGRU`/`NormConv2dGRU`
# (from `recurrent.py`) are transcribed verbatim. The package-internal helper
# functions `complex_multiplication`/`conjugate`/`expand_operator`/`reduce_operator`
# and the centered `fft2`/`ifft2` (+ `roll`/`fftshift`/`ifftshift`/
# `view_as_complex`/`view_as_real`/`assert_complex`/`is_complex_data`) are vendored
# unmodified from `direct/data/transforms.py` and `direct/utils/`; only the tiny
# `DirectEnum`/`InitType` string-enum and `COMPLEX_SIZE` constant from
# `direct/types.py`/`direct/constants.py` are inlined minimally (dropping the
# `omegaconf`-dependent `direct.types` module import, which is unrelated
# configuration plumbing, not part of the traced architecture) so the model can
# run standalone without the full `direct` package + its `omegaconf` dependency.

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

COMPLEX_SIZE = 2
COMPLEX_DIM = 2


class DirectEnum(str):
    """Minimal stand-in for direct.types.DirectEnum (str-valued enum comparison).

    Only used here so `InitType` string constants compare equal to plain strings,
    matching the original class's case-insensitive string-equality semantics.
    """

    def __eq__(self, other):
        _other = str(other.value) if hasattr(other, "value") else str(other)
        return self.lower() == _other.lower()

    def __hash__(self):
        return hash(self.lower())


class InitType:
    INPUT_IMAGE = "input_image"
    SENSE = "sense"
    ZERO_FILLED = "zero_filled"
    ZEROS = "zeros"


# --------------------------------------------------------------------------
# direct/utils/__init__.py + direct/utils/asserts.py (vendored subset)
# --------------------------------------------------------------------------


def is_complex_data(data: torch.Tensor, complex_axis: int = -1) -> bool:
    return data.size(complex_axis) == COMPLEX_DIM


def is_power_of_two(number: int) -> bool:
    return number != 0 and ((number & (number - 1)) == 0)


def assert_complex(
    data: torch.Tensor, complex_axis: int = -1, complex_last: Optional[bool] = None
) -> None:
    if complex_last:
        complex_axis = -1
    assert is_complex_data(data, complex_axis), (
        f"Complex dimension assumed to be 2 (complex valued), but not found in shape {data.shape}."
    )


# --------------------------------------------------------------------------
# direct/data/transforms.py (vendored subset)
# --------------------------------------------------------------------------


def view_as_complex(data):
    return torch.view_as_complex(data)


def view_as_real(data):
    return torch.view_as_real(data)


def roll_one_dim(data: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    shift = shift % data.size(dim)
    if shift == 0:
        return data

    left = data.narrow(dim, 0, data.size(dim) - shift)
    right = data.narrow(dim, data.size(dim) - shift, shift)

    return torch.cat((right, left), dim=dim)


def roll(data, shift, dim):
    if len(shift) != len(dim):
        raise ValueError("len(shift) must match len(dim)")

    for s, d in zip(shift, dim):
        data = roll_one_dim(data, s, d)

    return data


def fftshift(data, dim=None):
    if dim is None:
        dim = [0] * (data.dim())
        for idx in range(1, data.dim()):
            dim[idx] = idx

    shift = [0] * len(dim)
    for idx, dim_num in enumerate(dim):
        shift[idx] = data.shape[dim_num] // 2

    return roll(data, shift, dim)


def ifftshift(data, dim=None):
    if dim is None:
        dim = [0] * (data.dim())
        for i in range(1, data.dim()):
            dim[i] = i

    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = (data.shape[dim_num] + 1) // 2

    return roll(data, shift, dim)


def verify_fft_dtype_possible(data: torch.Tensor, dims) -> bool:
    is_complex64 = data.dtype == torch.complex64
    is_complex32_and_power_of_two = (data.dtype == torch.float32) and all(
        is_power_of_two(_) for _ in [data.size(idx) for idx in dims]
    )

    return is_complex64 or is_complex32_and_power_of_two


def fft2(
    data: torch.Tensor,
    dim: Tuple[int, int] = (1, 2),
    centered: bool = True,
    normalized: bool = True,
    complex_input: bool = True,
) -> torch.Tensor:
    if not all((_ >= 0 and isinstance(_, int)) for _ in dim):
        raise TypeError(
            f"Currently fft2 does not support negative indexing. Dim should contain only positive integers. Got {dim}."
        )
    if complex_input:
        assert_complex(data, complex_last=True)
        data = view_as_complex(data)

    if centered:
        data = ifftshift(data, dim=dim)
    if verify_fft_dtype_possible(data, dim):
        data = torch.fft.fftn(
            data,
            dim=dim,
            norm="ortho" if normalized else None,
        )
    else:
        raise ValueError("Currently half precision FFT is not supported.")

    if centered:
        data = fftshift(data, dim=dim)

    if complex_input:
        data = view_as_real(data)
    return data


def ifft2(
    data: torch.Tensor,
    dim: Tuple[int, int] = (1, 2),
    centered: bool = True,
    normalized: bool = True,
    complex_input: bool = True,
) -> torch.Tensor:
    if not all((_ >= 0 and isinstance(_, int)) for _ in dim):
        raise TypeError(
            f"Currently ifft2 does not support negative indexing. Dim should contain only positive integers. Got {dim}."
        )

    if complex_input:
        assert_complex(data, complex_last=True)
        data = view_as_complex(data)
    if centered:
        data = ifftshift(data, dim=dim)
    if verify_fft_dtype_possible(data, dim):
        data = torch.fft.ifftn(
            data,
            dim=dim,
            norm="ortho" if normalized else None,
        )
    else:
        raise ValueError("Currently half precision FFT is not supported.")

    if centered:
        data = fftshift(data, dim=dim)
    if complex_input:
        data = view_as_real(data)
    return data


def complex_multiplication(input_tensor: torch.Tensor, other_tensor: torch.Tensor) -> torch.Tensor:
    assert_complex(input_tensor, complex_last=True)
    assert_complex(other_tensor, complex_last=True)

    complex_index = -1

    real_part = (
        input_tensor[..., 0] * other_tensor[..., 0] - input_tensor[..., 1] * other_tensor[..., 1]
    )
    imaginary_part = (
        input_tensor[..., 0] * other_tensor[..., 1] + input_tensor[..., 1] * other_tensor[..., 0]
    )

    multiplication = torch.cat(
        [
            real_part.unsqueeze(dim=complex_index),
            imaginary_part.unsqueeze(dim=complex_index),
        ],
        dim=complex_index,
    )
    return multiplication


def conjugate(data: torch.Tensor) -> torch.Tensor:
    assert_complex(data, complex_last=True)
    data = data.clone()  # Clone is required as the data in the next line is changed in-place.
    data[..., 1] = data[..., 1] * -1.0

    return data


def reduce_operator(
    coil_data: torch.Tensor,
    sensitivity_map: torch.Tensor,
    dim: int = 0,
) -> torch.Tensor:
    assert_complex(coil_data, complex_last=True)
    assert_complex(sensitivity_map, complex_last=True)

    return complex_multiplication(conjugate(sensitivity_map), coil_data).sum(dim)


def expand_operator(
    data: torch.Tensor,
    sensitivity_map: torch.Tensor,
    dim: int = 0,
) -> torch.Tensor:
    assert_complex(data, complex_last=True)
    assert_complex(sensitivity_map, complex_last=True)

    return complex_multiplication(sensitivity_map, data.unsqueeze(dim))


# --------------------------------------------------------------------------
# direct/nn/recurrent/recurrent.py (vendored verbatim)
# --------------------------------------------------------------------------


class Conv2dGRU(nn.Module):
    """2D Convolutional GRU Network."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: Optional[int] = None,
        num_layers: int = 2,
        gru_kernel_size=1,
        orthogonal_initialization: bool = True,
        instance_norm: bool = False,
        dense_connect: int = 0,
        replication_padding: bool = True,
    ):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.dense_connect = dense_connect

        self.reset_gates = nn.ModuleList([])
        self.update_gates = nn.ModuleList([])
        self.out_gates = nn.ModuleList([])
        self.conv_blocks = nn.ModuleList([])

        # Create convolutional blocks
        for idx in range(num_layers + 1):
            in_ch = in_channels if idx == 0 else (1 + min(idx, dense_connect)) * hidden_channels
            out_ch = hidden_channels if idx < num_layers else out_channels
            padding = 0 if replication_padding else (2 if idx == 0 else 1)
            block = []
            if replication_padding:
                if idx == 1:
                    block.append(nn.ReplicationPad2d(2))
                else:
                    block.append(nn.ReplicationPad2d(2 if idx == 0 else 1))
            block.append(
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=5 if idx == 0 else 3,
                    dilation=(2 if idx == 1 else 1),
                    padding=padding,
                )
            )
            self.conv_blocks.append(nn.Sequential(*block))

        # Create GRU blocks
        for idx in range(num_layers):
            for gru_part in [self.reset_gates, self.update_gates, self.out_gates]:
                gru_block = []
                if instance_norm:
                    gru_block.append(nn.InstanceNorm2d(2 * hidden_channels))
                gru_block.append(
                    nn.Conv2d(
                        in_channels=2 * hidden_channels,
                        out_channels=hidden_channels,
                        kernel_size=gru_kernel_size,
                        padding=gru_kernel_size // 2,
                    )
                )
                gru_part.append(nn.Sequential(*gru_block))

        if orthogonal_initialization:
            for reset_gate, update_gate, out_gate in zip(
                self.reset_gates, self.update_gates, self.out_gates
            ):
                nn.init.orthogonal_(reset_gate[-1].weight)
                nn.init.orthogonal_(update_gate[-1].weight)
                nn.init.orthogonal_(out_gate[-1].weight)
                nn.init.constant_(reset_gate[-1].bias, -1.0)
                nn.init.constant_(update_gate[-1].bias, 0.0)
                nn.init.constant_(out_gate[-1].bias, 0.0)

    def forward(
        self,
        cell_input: torch.Tensor,
        previous_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        new_states = []
        conv_skip = []

        if previous_state is None:
            batch_size, spatial_size = cell_input.size(0), (cell_input.size(2), cell_input.size(3))
            state_size = [batch_size, self.hidden_channels] + list(spatial_size) + [self.num_layers]
            previous_state = torch.zeros(*state_size, dtype=cell_input.dtype).to(cell_input.device)

        for idx in range(self.num_layers):
            if len(conv_skip) > 0:
                cell_input = F.relu(
                    self.conv_blocks[idx](
                        torch.cat([*conv_skip[-self.dense_connect :], cell_input], dim=1)
                    ),
                    inplace=True,
                )
            else:
                cell_input = F.relu(self.conv_blocks[idx](cell_input), inplace=True)
            if self.dense_connect > 0:
                conv_skip.append(cell_input)

            stacked_inputs = torch.cat([cell_input, previous_state[:, :, :, :, idx]], dim=1)

            update = torch.sigmoid(self.update_gates[idx](stacked_inputs))
            reset = torch.sigmoid(self.reset_gates[idx](stacked_inputs))
            delta = torch.tanh(
                self.out_gates[idx](
                    torch.cat([cell_input, previous_state[:, :, :, :, idx] * reset], dim=1)
                )
            )
            cell_input = previous_state[:, :, :, :, idx] * (1 - update) + delta * update
            new_states.append(cell_input)
            cell_input = F.relu(cell_input, inplace=False)
        if len(conv_skip) > 0:
            out = self.conv_blocks[self.num_layers](
                torch.cat([*conv_skip[-self.dense_connect :], cell_input], dim=1)
            )
        else:
            out = self.conv_blocks[self.num_layers](cell_input)

        return out, torch.stack(new_states, dim=-1)


class NormConv2dGRU(nn.Module):
    """Normalized 2D Convolutional GRU Network.

    Normalization methods adapted from NormUnet of [1]_.

    References
    ----------

    .. [1] https://github.com/facebookresearch/fastMRI/blob/
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: Optional[int] = None,
        num_layers: int = 2,
        gru_kernel_size=1,
        orthogonal_initialization: bool = True,
        instance_norm: bool = False,
        dense_connect: int = 0,
        replication_padding: bool = True,
        norm_groups: int = 2,
    ):
        super().__init__()
        self.convgru = Conv2dGRU(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            gru_kernel_size=gru_kernel_size,
            orthogonal_initialization=orthogonal_initialization,
            instance_norm=instance_norm,
            dense_connect=dense_connect,
            replication_padding=replication_padding,
        )
        self.norm_groups = norm_groups

    @staticmethod
    def norm(
        input_data: torch.Tensor, num_groups: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = input_data.shape
        input_data = input_data.reshape(b, num_groups, -1)

        mean = input_data.mean(-1, keepdim=True)
        std = input_data.std(-1, keepdim=True)

        output = (input_data - mean) / std
        output = output.reshape(b, c, h, w)

        return output, mean, std

    @staticmethod
    def unnorm(
        input_data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, num_groups: int
    ) -> torch.Tensor:
        b, c, h, w = input_data.shape
        input_data = input_data.reshape(b, num_groups, -1)
        return (input_data * std + mean).reshape(b, c, h, w)

    def forward(
        self,
        cell_input: torch.Tensor,
        previous_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Normalize
        cell_input, mean, std = self.norm(cell_input, self.norm_groups)
        # Pass normalized input
        cell_input, previous_state = self.convgru(cell_input, previous_state)
        # Unnormalize output
        cell_input = self.unnorm(cell_input, mean, std, self.norm_groups)

        return cell_input, previous_state


# --------------------------------------------------------------------------
# direct/nn/recurrentvarnet/recurrentvarnet.py (vendored verbatim)
# --------------------------------------------------------------------------


class RecurrentInit(nn.Module):
    """Recurrent State Initializer (RSI) module of Recurrent Variational Network as presented in [1]_.

    The RSI module learns to initialize the recurrent hidden state :math:`h_0`, input of the first RecurrentVarNetBlock
    of the RecurrentVarNet.

    References
    ----------

    .. [1] Yiasemis, George, et al. "Recurrent Variational Network: A Deep Learning Inverse Problem Solver Applied to
        the Task of Accelerated MRI Reconstruction." ArXiv:2111.09639 [Physics], Nov. 2021. arXiv.org,
        http://arxiv.org/abs/2111.09639.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: Tuple[int, ...],
        dilations: Tuple[int, ...],
        depth: int = 2,
        multiscale_depth: int = 1,
    ):
        super().__init__()

        self.conv_blocks = nn.ModuleList()
        self.out_blocks = nn.ModuleList()
        self.depth = depth
        self.multiscale_depth = multiscale_depth
        tch = in_channels
        for curr_channels, curr_dilations in zip(channels, dilations):
            block = [
                nn.ReplicationPad2d(curr_dilations),
                nn.Conv2d(tch, curr_channels, 3, padding=0, dilation=curr_dilations),
            ]
            tch = curr_channels
            self.conv_blocks.append(nn.Sequential(*block))
        tch = np.sum(channels[-multiscale_depth:])
        for _ in range(depth):
            block = [nn.Conv2d(tch, out_channels, 1, padding=0)]
            self.out_blocks.append(nn.Sequential(*block))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = []
        for block in self.conv_blocks:
            x = F.relu(block(x), inplace=True)
            if self.multiscale_depth > 1:
                features.append(x)
        if self.multiscale_depth > 1:
            x = torch.cat(features[-self.multiscale_depth :], dim=1)
        output_list = []
        for block in self.out_blocks:
            y = F.relu(block(x), inplace=True)
            output_list.append(y)
        out = torch.stack(output_list, dim=-1)
        return out


class RecurrentVarNet(nn.Module):
    """Recurrent Variational Network implementation as presented in [1]_.

    References
    ----------

    .. [1] Yiasemis, George, et al. "Recurrent Variational Network: A Deep Learning Inverse Problem Solver Applied to
        the Task of Accelerated MRI Reconstruction." ArXiv:2111.09639 [Physics], Nov. 2021. arXiv.org,
        http://arxiv.org/abs/2111.09639.
    """

    def __init__(
        self,
        forward_operator,
        backward_operator,
        in_channels: int = COMPLEX_SIZE,
        num_steps: int = 15,
        recurrent_hidden_channels: int = 64,
        recurrent_num_layers: int = 4,
        no_parameter_sharing: bool = True,
        learned_initializer: bool = False,
        initializer_initialization: Optional[str] = None,
        initializer_channels: Optional[Tuple[int, ...]] = (32, 32, 64, 64),
        initializer_dilations: Optional[Tuple[int, ...]] = (1, 1, 2, 4),
        initializer_multiscale: int = 1,
        normalized: bool = False,
        **kwargs,
    ):
        super().__init__()

        extra_keys = kwargs.keys()
        for extra_key in extra_keys:
            if extra_key not in [
                "model_name",
            ]:
                raise ValueError(
                    f"{type(self).__name__} got key `{extra_key}` which is not supported."
                )

        self.initializer: Optional[nn.Module] = None
        if (
            learned_initializer
            and initializer_initialization is not None
            and initializer_channels is not None
            and initializer_dilations is not None
        ):
            if initializer_initialization not in [
                "sense",
                "input_image",
                "zero_filled",
            ]:
                raise ValueError(
                    f"Unknown initializer_initialization. Expected `sense`, `'input_image` or `zero_filled`."
                    f"Got {initializer_initialization}."
                )
            self.initializer_initialization = initializer_initialization
            self.initializer = RecurrentInit(
                in_channels,
                recurrent_hidden_channels,
                channels=initializer_channels,
                dilations=initializer_dilations,
                depth=recurrent_num_layers,
                multiscale_depth=initializer_multiscale,
            )
        self.num_steps = num_steps
        self.no_parameter_sharing = no_parameter_sharing
        self.block_list = nn.ModuleList()
        for _ in range(self.num_steps if self.no_parameter_sharing else 1):
            self.block_list.append(
                RecurrentVarNetBlock(
                    forward_operator=forward_operator,
                    backward_operator=backward_operator,
                    in_channels=in_channels,
                    hidden_channels=recurrent_hidden_channels,
                    num_layers=recurrent_num_layers,
                    normalized=normalized,
                )
            )
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self._coil_dim = 1
        self._spatial_dims = (2, 3)

    def compute_sense_init(
        self, kspace: torch.Tensor, sensitivity_map: torch.Tensor
    ) -> torch.Tensor:
        input_image = complex_multiplication(
            conjugate(sensitivity_map),
            self.backward_operator(kspace, dim=self._spatial_dims),
        )
        input_image = input_image.sum(self._coil_dim)

        return input_image

    def forward(
        self,
        masked_kspace: torch.Tensor,
        sampling_mask: torch.Tensor,
        sensitivity_map: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        previous_state: Optional[torch.Tensor] = None

        if self.initializer is not None:
            if self.initializer_initialization == "sense":
                initializer_input_image = self.compute_sense_init(
                    kspace=masked_kspace,
                    sensitivity_map=sensitivity_map,
                ).unsqueeze(self._coil_dim)
            elif self.initializer_initialization == "input_image":
                if "initial_image" not in kwargs:
                    raise ValueError(
                        f"`'initial_image` is required as input if initializer_initialization "
                        f"is {self.initializer_initialization}."
                    )
                initializer_input_image = kwargs["initial_image"].unsqueeze(self._coil_dim)
            elif self.initializer_initialization == "zero_filled":
                initializer_input_image = self.backward_operator(
                    masked_kspace, dim=self._spatial_dims
                )

            previous_state = self.initializer(
                self.forward_operator(initializer_input_image, dim=self._spatial_dims)
                .sum(self._coil_dim)
                .permute(0, 3, 1, 2)
            )

        kspace_prediction = masked_kspace.clone()

        for step in range(self.num_steps):
            block = self.block_list[step] if self.no_parameter_sharing else self.block_list[0]
            kspace_prediction, previous_state = block(
                kspace_prediction,
                masked_kspace,
                sampling_mask,
                sensitivity_map,
                previous_state,
                self._coil_dim,
                self._spatial_dims,
            )

        return kspace_prediction


class RecurrentVarNetBlock(nn.Module):
    r"""Recurrent Variational Network Block :math:`\mathcal{H}_{\theta_{t}}` as presented in [1]_.

    References
    ----------

    .. [1] Yiasemis, George, et al. "Recurrent Variational Network: A Deep Learning Inverse Problem Solver Applied to
        the Task of Accelerated MRI Reconstruction." ArXiv:2111.09639 [Physics], Nov. 2021. arXiv.org,
        http://arxiv.org/abs/2111.09639.

    """

    def __init__(
        self,
        forward_operator,
        backward_operator,
        in_channels: int = 2,
        hidden_channels: int = 64,
        num_layers: int = 4,
        normalized: bool = False,
    ):
        super().__init__()
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator

        self.learning_rate = nn.Parameter(torch.tensor([1.0]))  # :math:`\alpha_t`
        regularizer_params = {
            "in_channels": in_channels,
            "hidden_channels": hidden_channels,
            "num_layers": num_layers,
            "replication_padding": True,
        }
        # Recurrent Unit of RecurrentVarNet Block :math:`\mathcal{H}_{\theta_t}`
        self.regularizer = (
            NormConv2dGRU(**regularizer_params) if normalized else Conv2dGRU(**regularizer_params)
        )

    def forward(
        self,
        current_kspace: torch.Tensor,
        masked_kspace: torch.Tensor,
        sampling_mask: torch.Tensor,
        sensitivity_map: torch.Tensor,
        hidden_state: Union[None, torch.Tensor],
        coil_dim: int = 1,
        spatial_dims: Tuple[int, int] = (2, 3),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        kspace_error = torch.where(
            sampling_mask == 0,
            torch.tensor([0.0], dtype=masked_kspace.dtype).to(masked_kspace.device),
            current_kspace - masked_kspace,
        )

        recurrent_term = reduce_operator(
            self.backward_operator(current_kspace, dim=spatial_dims),
            sensitivity_map,
            dim=coil_dim,
        ).permute(0, 3, 1, 2)

        recurrent_term, hidden_state = self.regularizer(
            recurrent_term, hidden_state
        )  # :math:`w_t`, :math:`h_{t+1}`
        recurrent_term = recurrent_term.permute(0, 2, 3, 1)

        recurrent_term = self.forward_operator(
            expand_operator(recurrent_term, sensitivity_map, dim=coil_dim),
            dim=spatial_dims,
        )

        new_kspace = current_kspace - self.learning_rate * kspace_error + recurrent_term

        return new_kspace, hidden_state


def build_recurrentvarnet():
    torch.manual_seed(0)
    # Tiny config: 2 unrolled steps, 8 hidden channels, 2 GRU layers keeps the
    # trace small while exercising the full recurrent data-consistency loop.
    model = RecurrentVarNet(
        forward_operator=fft2,
        backward_operator=ifft2,
        in_channels=COMPLEX_SIZE,
        num_steps=2,
        recurrent_hidden_channels=8,
        recurrent_num_layers=2,
        no_parameter_sharing=True,
        learned_initializer=False,
        normalized=False,
    )
    model.eval()
    return model


def example_input_recurrentvarnet():
    torch.manual_seed(0)
    # (masked_kspace, sampling_mask, sensitivity_map): shape (N, coil, H, W, complex=2).
    # 1 batch, 2 coils, 16x16 spatial (power-of-two, keeps the ortho-FFT path exact).
    masked_kspace = torch.randn(1, 2, 16, 16, 2)
    sampling_mask = (torch.rand(1, 1, 16, 16, 1) > 0.5).float()
    sensitivity_map = torch.randn(1, 2, 16, 16, 2)
    return (masked_kspace, sampling_mask, sensitivity_map)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "RecurrentVarNet",
        "build_recurrentvarnet",
        "example_input_recurrentvarnet",
        2022,
        MENAGERIE_ZOO,
    ),
]
