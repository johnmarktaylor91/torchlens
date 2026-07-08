# SOURCE: vendored from wukevin/babel @ main
# Files: babel/activations.py, babel/models/autoencoders.py (Encoder, Decoder,
# SplicedAutoEncoder classes only). The real autoencoders.py module also defines
# `*SkorchNet` training-harness wrapper classes (AutoEncoderSkorchNet etc.) that
# import `skorch` at module scope purely for the sklearn-style `.fit()` API; skorch
# is not installed and is unrelated to the architecture itself, so only the plain
# nn.Module classes (Encoder/Decoder/SplicedAutoEncoder) are vendored here, verbatim.
"""BABEL: encoder-decoder neural translator between scRNA-seq and scATAC-seq
domains, vendored from the real wukevin/babel repository (babel/models package)."""

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# --- babel/activations.py (verbatim) ---


class Exp(nn.Module):
    """Applies torch.exp, clamped to improve stability during training"""

    def __init__(self, minimum=1e-5, maximum=1e6):
        """Values taken from DCA"""
        super(Exp, self).__init__()
        self.min_value = minimum
        self.max_value = maximum

    def forward(self, input):
        return torch.clamp(
            torch.exp(input),
            min=self.min_value,
            max=self.max_value,
        )


class ClippedSoftplus(nn.Module):
    def __init__(self, beta=1, threshold=20, minimum=1e-4, maximum=1e3):
        super(ClippedSoftplus, self).__init__()
        self.beta = beta
        self.threshold = threshold
        self.min_value = minimum
        self.max_value = maximum

    def forward(self, input):
        return torch.clamp(
            nn.functional.softplus(input, self.beta, self.threshold),
            min=self.min_value,
            max=self.max_value,
        )

    def extra_repr(self):
        return "beta={}, threshold={}, min={}, max={}".format(
            self.beta,
            self.threshold,
            self.min_value,
            self.max_value,
        )


# --- babel/models/autoencoders.py (verbatim, Encoder/Decoder/SplicedAutoEncoder) ---


class Encoder(nn.Module):
    def __init__(self, num_inputs: int, num_units=32, activation=nn.PReLU):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_units = num_units

        self.encode1 = nn.Linear(self.num_inputs, 64)
        nn.init.xavier_uniform_(self.encode1.weight)
        self.bn1 = nn.BatchNorm1d(64)
        self.act1 = activation()

        self.encode2 = nn.Linear(64, self.num_units)
        nn.init.xavier_uniform_(self.encode2.weight)
        self.bn2 = nn.BatchNorm1d(num_units)
        self.act2 = activation()

    def forward(self, x):
        x = self.act1(self.bn1(self.encode1(x)))
        x = self.act2(self.bn2(self.encode2(x)))
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        num_outputs: int,
        num_units: int = 32,
        intermediate_dim: int = 64,
        activation=nn.PReLU,
        final_activation=None,
    ):
        super().__init__()
        self.num_outputs = num_outputs
        self.num_units = num_units

        self.decode1 = nn.Linear(self.num_units, intermediate_dim)
        nn.init.xavier_uniform_(self.decode1.weight)
        self.bn1 = nn.BatchNorm1d(intermediate_dim)
        self.act1 = activation()

        self.decode21 = nn.Linear(intermediate_dim, self.num_outputs)
        nn.init.xavier_uniform_(self.decode21.weight)
        self.decode22 = nn.Linear(intermediate_dim, self.num_outputs)
        nn.init.xavier_uniform_(self.decode22.weight)
        self.decode23 = nn.Linear(intermediate_dim, self.num_outputs)
        nn.init.xavier_uniform_(self.decode23.weight)

        self.final_activations = nn.ModuleDict()
        if final_activation is not None:
            if isinstance(final_activation, list) or isinstance(final_activation, tuple):
                assert len(final_activation) <= 3
                for i, act in enumerate(final_activation):
                    if act is None:
                        continue
                    self.final_activations[f"act{i + 1}"] = act
            elif isinstance(final_activation, nn.Module):
                self.final_activations["act1"] = final_activation
            else:
                raise ValueError(
                    f"Unrecognized type for final_activation: {type(final_activation)}"
                )

    def forward(self, x, size_factors=None):
        """include size factor here because we may want to scale the output by that"""
        x = self.act1(self.bn1(self.decode1(x)))

        retval1 = self.decode21(x)  # This is invariably the counts
        if "act1" in self.final_activations.keys():
            retval1 = self.final_activations["act1"](retval1)
        if size_factors is not None:
            sf_scaled = size_factors.view(-1, 1).repeat(1, retval1.shape[1])
            retval1 = retval1 * sf_scaled  # Elementwise multiplication

        retval2 = self.decode22(x)
        if "act2" in self.final_activations.keys():
            retval2 = self.final_activations["act2"](retval2)

        retval3 = self.decode23(x)
        if "act3" in self.final_activations.keys():
            retval3 = self.final_activations["act3"](retval3)

        return retval1, retval2, retval3


class SplicedAutoEncoder(nn.Module):
    """
    Spliced Autoencoder - where we have 4 parts (2 encoders, 2 decoders) that are all combined
    This does not work when you have chromsome split features
    """

    def __init__(
        self,
        input_dim1: int,
        input_dim2: int,
        hidden_dim: int = 16,
        final_activations1: list = None,
        final_activations2=None,
        flat_mode: bool = False,  # Controls if we have to re-split inputs
        seed=182822,
    ):
        super().__init__()
        torch.manual_seed(seed)

        if final_activations1 is None:
            final_activations1 = [Exp(), nn.Softplus()]
        if final_activations2 is None:
            final_activations2 = [Exp(), nn.Softplus(), nn.Sigmoid()]

        self.flat_mode = flat_mode
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.num_outputs1 = (
            len(final_activations1) if isinstance(final_activations1, (list, set, tuple)) else 1
        )
        self.num_outputs2 = (
            len(final_activations2) if isinstance(final_activations2, (list, set, tuple)) else 1
        )

        self.encoder1 = Encoder(num_inputs=input_dim1, num_units=hidden_dim)
        self.encoder2 = Encoder(num_inputs=input_dim2, num_units=hidden_dim)

        self.decoder1 = Decoder(
            num_outputs=input_dim1,
            num_units=hidden_dim,
            final_activation=final_activations1,
        )
        self.decoder2 = Decoder(
            num_outputs=input_dim2,
            num_units=hidden_dim,
            final_activation=final_activations2,
        )

    def split_catted_input(self, x):
        """
        Split catted input data to expected sizes
        """
        return torch.split(x, [self.input_dim1, self.input_dim2], dim=-1)

    def _combine_output_and_encoded(self, decoded, encoded, num_outputs: int):
        """
        Combines the output and encoded in a single output
        """
        if num_outputs > 1:
            retval = *decoded, encoded
        else:
            if isinstance(decoded, tuple):
                decoded = decoded[0]
            retval = decoded, encoded
        assert isinstance(retval, (list, tuple))
        assert isinstance(retval[0], (torch.TensorType, torch.Tensor)), (
            f"Expected tensor but got {type(retval[0])}"
        )
        return retval

    def forward_single(self, x, size_factors=None, in_domain: int = 1, out_domain: int = 1):
        """
        Return output of a single domain combination
        """
        assert in_domain in [1, 2] and out_domain in [1, 2]

        encoder = self.encoder1 if in_domain == 1 else self.encoder2
        decoder = self.decoder1 if out_domain == 1 else self.decoder2
        num_non_latent_out = self.num_outputs1 if out_domain == 1 else self.num_outputs2

        encoded = encoder(x)
        decoded = decoder(encoded)
        return self._combine_output_and_encoded(decoded, encoded, num_non_latent_out)

    def forward(self, x, size_factors=None, mode=None):
        if self.flat_mode:
            x = self.split_catted_input(x)
        assert isinstance(x, (tuple, list))
        assert len(x) == 2, "There should be two inputs to spliced autoencoder"
        encoded1 = self.encoder1(x[0])
        encoded2 = self.encoder2(x[1])

        decoded11 = self.decoder1(encoded1)
        retval11 = self._combine_output_and_encoded(decoded11, encoded1, self.num_outputs1)
        decoded12 = self.decoder2(encoded1)
        retval12 = self._combine_output_and_encoded(decoded12, encoded1, self.num_outputs2)
        decoded22 = self.decoder2(encoded2)
        retval22 = self._combine_output_and_encoded(decoded22, encoded2, self.num_outputs2)
        decoded21 = self.decoder1(encoded2)
        retval21 = self._combine_output_and_encoded(decoded21, encoded2, self.num_outputs1)

        if mode is None:
            return retval11, retval12, retval21, retval22
        retval_dict = {
            (1, 1): retval11,
            (1, 2): retval12,
            (2, 1): retval21,
            (2, 2): retval22,
        }
        if mode not in retval_dict:
            raise ValueError(f"Invalid mode code: {mode}")
        return retval_dict[mode]


# --- staging build/example helpers ---


def build_babel_spliced_autoencoder():
    """Tiny SplicedAutoEncoder (RNA<->ATAC cross-domain translator)."""
    return SplicedAutoEncoder(input_dim1=32, input_dim2=48, hidden_dim=8)


def example_input_babel_spliced_autoencoder():
    # forward(self, x, ...) expects x itself to be the (rna_counts, atac_counts)
    # paired minibatch, so wrap it in a 1-tuple so tl.trace passes it as the sole
    # positional argument rather than unpacking rna/atac as separate args.
    return ((torch.rand(4, 32) * 10, torch.rand(4, 48) * 10),)


MENAGERIE_ENTRIES = [
    (
        "BABEL",
        "build_babel_spliced_autoencoder",
        "example_input_babel_spliced_autoencoder",
        2021,
        "vendored-pytorch",
    ),
]
