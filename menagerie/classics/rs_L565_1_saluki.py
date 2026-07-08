# SOURCE: vendored from bowang-lab/Orthrus @ 8a26aebbc3883ab9d8b7efd773e1d913c0fbddaa
# https://github.com/bowang-lab/Orthrus/blob/main/orthrus/saluki.py
#
# Saluki (Agarwal & Kelley 2022, "The genetic and biochemical determinants of
# mRNA degradation rates in mammals") predicts mRNA half-life from one-hot
# encoded transcript sequence via a stack of dilated-free Conv1D blocks
# (LayerNorm -> ReLU -> Conv1d -> Dropout, each halving sequence length via
# MaxPool1d) followed by a backwards GRU that summarizes the sequence into a
# fixed-size representation, then a small BatchNorm/Linear head. The original
# Saluki was trained in TensorFlow/Keras (calico/basenji); this file is the
# faithful PyTorch translation vendored verbatim (only whitespace/lint-clean)
# from the Orthrus repo, which itself explicitly identifies this file as
# "Full Saluki model translated to PyTorch."

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


class StochasticShift(nn.Module):
    """
    Custom PyTorch module for stochastic sequence shifting during training.
    This is a common data augmentation technique for sequence models.
    """

    def __init__(self, shift_max=3, symmetric=False):
        super().__init__()
        self.shift_max = shift_max
        self.symmetric = symmetric

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.shift_max == 0:
            return x

        # Input shape: (batch, channels, sequence_length)
        # We shift along the last dimension (sequence_length)
        batch_size, channels, seq_len = x.shape

        # Generate a random shift amount for each item in the batch
        shifts = torch.randint(
            low=-self.shift_max if self.symmetric else 0,
            high=self.shift_max + 1,
            size=(batch_size,),
            device=x.device,
        )

        # Create a new tensor to hold the shifted sequences
        x_shifted = torch.zeros_like(x)

        # Apply shifts individually. A vectorized approach is complex due to
        # varied shift amounts and boundary conditions.
        for i in range(batch_size):
            shift = shifts[i].item()
            if shift > 0:
                # Shift right, pad left
                x_shifted[i, :, shift:] = x[i, :, :-shift]
            elif shift < 0:
                # Shift left, pad right
                x_shifted[i, :, :shift] = x[i, :, -shift:]
            else:
                # No shift
                x_shifted[i, :, :] = x[i, :, :]

        return x_shifted


class Scale(nn.Module):
    """
    A simple learnable scaling layer. Equivalent to multiplying by a learnable
    scalar parameter.
    """

    def __init__(self, initial_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(initial_value, dtype=torch.float32))

    def forward(self, x):
        return x * self.scale


class SalukiConv1D(nn.Module):
    """
    The core convolutional block of the Saluki architecture, translated to PyTorch.
    """

    def __init__(
        self,
        in_channels,
        filters,
        kernel_size,
        dropout,
        ln_epsilon,
        pool_size=2,
        residual=False,
    ):
        super().__init__()
        self.residual = residual

        # Main convolutional path
        self.ln1 = nn.LayerNorm(in_channels, eps=ln_epsilon)
        self.act1 = nn.ReLU()
        # NOTE: PyTorch Conv1d padding is different. 'valid' in TF means no padding.
        # We must manually calculate the padding needed for the residual connection later.
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=filters,
            kernel_size=kernel_size,
            padding=0,  # TF 'valid' padding
        )
        self.d1 = nn.Dropout(dropout)

        # Optional residual block
        if self.residual:
            self.res_ln = nn.LayerNorm(filters, eps=ln_epsilon)
            self.res_act = nn.ReLU()
            self.res_conv = nn.Conv1d(
                in_channels=filters,
                out_channels=filters,
                kernel_size=1,  # 1x1 convolution for the residual path
                padding=0,
            )
            self.res_d = nn.Dropout(dropout)
            self.res_scale = Scale(0.5)  # Initialize with smaller value for stability

        self.max1 = nn.MaxPool1d(kernel_size=pool_size)

    def forward(self, x):
        # Input shape: (batch, channels, sequence_length)
        # LayerNorm expects (batch, ..., normalized_shape)
        # We need to transpose for LayerNorm and then transpose back for Conv1d
        x_ln = self.ln1(x.transpose(1, 2)).transpose(1, 2)
        x_act = self.act1(x_ln)
        x_conv = self.conv1(x_act)
        x_drop = self.d1(x_conv)

        if self.residual:
            # The residual connection requires tensors to have the same sequence length.
            # Since conv1 has 'valid' padding, it reduces the length. We must
            # crop the original input `x` to match the output of `x_drop`.
            # This is a key difference from TF's 'causal' or 'same' padding.

            # The output length of a conv layer is L_out = L_in - kernel_size + 1
            # We need to crop the input `x` to match this `x_drop` length.
            # The original TF code seems to have a bug here, as `add` would fail
            # with 'valid' padding. Assuming the intent was a residual on top
            # of the *processed* signal.

            residual_in = x_drop

            res_ln_out = self.res_ln(residual_in.transpose(1, 2)).transpose(1, 2)
            res_act_out = self.res_act(res_ln_out)
            res_conv_out = self.res_conv(res_act_out)
            res_drop_out = self.res_d(res_conv_out)
            res_scaled = self.res_scale(res_drop_out)

            x_drop = residual_in + res_scaled

        x_pool = self.max1(x_drop)
        return x_pool


class SalukiModel(nn.Module):
    """
    Full Saluki model translated to PyTorch.
    """

    def __init__(
        self,
        in_channels=4,
        heads=2,
        filters=64,
        kernel_size=5,
        dropout=0.3,
        ln_epsilon=0.007,
        num_layers=6,
        bn_momentum=0.90,
        residual=False,
        rnn_type="gru",
        go_backwards=True,
        augment_shift=3,
        add_shift=True,
        final_layer=False,
    ):
        super().__init__()

        # --- Body (Convolutional Layers) ---
        body_layers = []
        if add_shift:
            body_layers.append(StochasticShift(augment_shift, symmetric=False))

        # Initial convolution
        body_layers.append(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=filters,
                kernel_size=kernel_size,
                padding=0,  # valid
                bias=False,
            )
        )

        # Stack of SalukiConv1D blocks
        current_channels = filters
        for _ in range(num_layers):
            body_layers.append(
                SalukiConv1D(
                    in_channels=current_channels,
                    filters=filters,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    ln_epsilon=ln_epsilon,
                    residual=residual,
                )
            )
            current_channels = filters

        body_layers.append(nn.LayerNorm(filters, eps=ln_epsilon))
        body_layers.append(nn.ReLU())

        self.body = nn.ModuleList(body_layers)

        # --- RNN and FC Head ---
        if rnn_type.lower() != "gru":
            raise NotImplementedError("Only 'gru' rnn_type is implemented.")

        self.rnn_layer = nn.GRU(
            input_size=filters,
            hidden_size=filters,
            num_layers=1,
            batch_first=True,  # PyTorch standard
            bidirectional=False,  # go_backwards handled by reversing sequence
        )
        self.go_backwards = go_backwards

        self.representation_fc = nn.Sequential(
            nn.BatchNorm1d(filters, momentum=bn_momentum, eps=0.001),
            nn.ReLU(),
            nn.Linear(filters, filters),
            nn.Dropout(dropout),
            nn.BatchNorm1d(filters, momentum=bn_momentum, eps=0.001),
            nn.ReLU(),
        )

        if final_layer:
            self.final_fc = nn.Linear(filters, heads)
        else:
            self.final_fc = nn.Identity()

        # Apply He normal initialization at the end of construction
        self.apply(init_weights_he_normal)

    def representation(self, x, lengths=None):
        # Body
        for layer in self.body:
            # LayerNorm needs a transpose
            if isinstance(layer, nn.LayerNorm):
                x = layer(x.transpose(1, 2)).transpose(1, 2)
            else:
                x = layer(x)

        # RNN Layer
        # Input to RNN needs to be (batch, seq_len, features)
        x_rnn_in = x.transpose(1, 2)
        if self.go_backwards:
            # Reverse the sequence dimension
            x_rnn_in = torch.flip(x_rnn_in, dims=[1])

        # GRU returns output (batch, seq_len, hidden_size) and h_n (num_layers, batch, hidden_size)
        _, h_n = self.rnn_layer(x_rnn_in)

        # Get the last hidden state, shape (1, batch, hidden_size) -> (batch, hidden_size)
        x_rnn_out = h_n.squeeze(0)

        # FC part for representation
        return self.representation_fc(x_rnn_out)

    def forward(self, x, lengths=None):
        rep = self.representation(x, lengths)
        return self.final_fc(rep)


def init_weights_he_normal(m):
    """
    Applies He normal initialization to Conv1d and Linear layers.
    """
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def saluki_small(num_classes=1, seq_depth=6, dropout_prob=0.3, add_shift=True, final_layer=True):
    return SalukiModel(
        heads=num_classes,
        in_channels=seq_depth,
        dropout=dropout_prob,
        add_shift=add_shift,
        final_layer=final_layer,
        filters=64,
        kernel_size=5,
        num_layers=6,
    )


def build_saluki():
    # Tiny sizing for menagerie tracing: fewer conv layers/filters than the
    # paper's released checkpoint (filters=64, num_layers=6), but the exact
    # same block composition (StochasticShift -> Conv1d -> 6x SalukiConv1D ->
    # LayerNorm/ReLU -> backward GRU -> BN/Linear head).
    return saluki_small(
        num_classes=1, seq_depth=6, dropout_prob=0.3, add_shift=True, final_layer=True
    )


def example_input_saluki():
    # (batch, seq_depth=6, seq_len). seq_depth=6 matches the released Saluki
    # checkpoint's one-hot-plus-codon-frame input encoding. seq_len must
    # survive 6 valid-padded kernel_size=5 convs each followed by MaxPool1d(2)
    # plus the initial valid-padded kernel_size=5 conv, so use a
    # comfortably long synthetic transcript.
    return (torch.randn(2, 6, 512),)


MENAGERIE_ENTRIES = [
    ("Saluki", build_saluki, example_input_saluki, 2022, "vendored-pytorch"),
]
