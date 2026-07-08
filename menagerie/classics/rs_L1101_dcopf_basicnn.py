# SOURCE: vendored from https://github.com/AI4OPT/ML4OPF @ main
# (ml4opf/models/basic_nn/lightning_basic_nn.py class BasicNN.make_network/forward,
#  ml4opf/models/basic_nn/dcopf_basic_nn.py class DCBasicNN.add_boundrepair,
#  ml4opf/layers/bound_repair.py class BoundRepair)
#
# "DCOPF-Net" -- the DC Optimal Power Flow feed-forward solver from the ML4OPF
# library (AI4OPT / Georgia Tech + PNNL). ML4OPF's basic-NN family is a plain
# feed-forward MLP (Linear + activation, repeated) that maps DC-OPF problem
# inputs (bus active power demand `pd`) to primal outputs (generator power
# `pg`, bus voltage angle `va`), followed by a REAL differentiable
# constraint-repair output layer, `BoundRepair`, that clips the generator
# output channels to their [pg_min, pg_max] box using one of several smooth
# ReLU/sigmoid/softplus/tanh clipping mechanisms -- this repair layer is the
# architectural contribution distinguishing DCOPF-Net from a bare MLP, and is
# vendored verbatim (`BoundRepair.__init__`/`forward`/`relu`/`double_relu`/
# `lower_relu`/`upper_relu`/`preprocess_bounds`) below.
#
# The real `BasicNN.__init__`/`make_network`/`forward` build exactly
# `nn.Sequential(Linear, activation, ..., Linear)`; `DCBasicNN.add_boundrepair`
# (real code, copied verbatim) appends a `BoundRepair` module sized to the
# output, with bounds on the pg (generator power) output slice only (va bus
# angles are left unbounded, matching the real class -- `lower`/`upper`
# default to +-inf outside the pg slice, i.e. "none" bound-repair method for
# those channels). We drop only the `pytorch_lightning.LightningModule` base
# class, hyperparameter-logging, dataset loaders, optimizer wiring, and the
# `OPFModel`/`OPFViolation`/`slices` real-grid dataset objects the true
# constructor reads (those come from a matpower case file loaded through the
# `ml4opf` package's dataset pipeline, not from the neural network itself) --
# none of that participates in `forward()`. `_data` here only supplies the
# `pgmin`/`pgmax` bound tensors and the input/output/pg slice info the real
# `__init__`/`add_boundrepair`/`forward` read, matching a tiny toy DC-OPF case.

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


class BoundRepair(nn.Module):
    """Real ML4OPF `ml4opf/layers/bound_repair.py` BoundRepair, verbatim
    (only the "relu"-method code paths this module exercises are retained;
    sigmoid/softplus/tanh/clamp/none are omitted here for brevity but the
    "relu" path -- the one DCBasicNN.add_boundrepair actually installs when
    boundrepair != "none" -- is copied unmodified, including the
    torch.jit.script-decorated double_relu/lower_relu/upper_relu helpers)."""

    SUPPORTED_METHODS = ["relu", "sigmoid", "clamp", "softplus", "tanh", "none"]

    def __init__(
        self, xmin, xmax, method: str = "relu", sanity_check: bool = True, memory_efficient: int = 0
    ):
        super().__init__()

        if sanity_check:
            assert (xmin is None and xmax is None) or (xmin is not None and xmax is not None)
            if xmin is not None and xmax is not None:
                assert xmin.shape == xmax.shape
                assert method.lower() in self.SUPPORTED_METHODS
                for lo, hi in zip(xmin, xmax):
                    assert lo <= hi
                    assert lo != torch.inf
                    assert hi != -torch.inf
            self.sanity_check = True

        self.register_buffer(
            "method_idx", torch.as_tensor(self.SUPPORTED_METHODS.index(method.lower()))
        )

        if xmin is None and xmax is None:
            memory_efficient = 2
        else:
            self.register_buffer("xmin", xmin)
            self.register_buffer("xmax", xmax)

        self.register_buffer("memory_efficient", torch.as_tensor(memory_efficient))
        self.preprocess_bounds(self.memory_efficient)

        self._forward = getattr(self, self.SUPPORTED_METHODS[self.method_idx])

    def forward(self, x, xmin=None, xmax=None):
        if xmin is not None and xmax is not None:
            self.xmin = xmin
            self.xmax = xmax
            self.preprocess_bounds(2)
        return self._forward(x)

    def none(self, x):
        return x

    @staticmethod
    @torch.jit.script
    def double_relu(x: torch.Tensor, xmin: torch.Tensor, xmax: torch.Tensor):
        return torch.relu(x - xmin) - torch.relu(x - xmax) + xmin

    @staticmethod
    @torch.jit.script
    def lower_relu(x: torch.Tensor, xmin: torch.Tensor):
        return torch.relu(x - xmin) + xmin

    @staticmethod
    @torch.jit.script
    def upper_relu(x: torch.Tensor, xmax: torch.Tensor):
        return -torch.relu(xmax - x) + xmax

    def relu(self, x):
        y = torch.clone(x)
        y[..., self.lower_mask] = self.lower_relu(x[..., self.lower_mask], self.xmin_lower)
        y[..., self.upper_mask] = self.upper_relu(x[..., self.upper_mask], self.xmax_upper)
        y[..., self.double_mask] = self.double_relu(
            x[..., self.double_mask], self.xmin_double, self.xmax_double
        )
        return y

    def preprocess_bounds(self, memory_efficient: int):
        if hasattr(self, "_memory_mode") and ((self._memory_mode == 2) and (memory_efficient == 2)):
            return

        self._properties = {}

        for k, v in {
            "lower_mask": lambda self: self.xmin.isfinite() & ~self.xmax.isfinite(),
            "upper_mask": lambda self: ~self.xmin.isfinite() & self.xmax.isfinite(),
            "double_mask": lambda self: self.xmin.isfinite() & self.xmax.isfinite(),
            "none_mask": lambda self: ~self.xmin.isfinite() & ~self.xmax.isfinite(),
        }.items():
            if memory_efficient in (0, 1):
                self.register_buffer(k, v(self), persistent=False)
            else:
                self._properties[k] = v

        for k, v in {
            "xmin_lower": lambda self: self.xmin[self.lower_mask],
            "xmax_upper": lambda self: self.xmax[self.upper_mask],
            "xmin_double": lambda self: self.xmin[self.double_mask],
            "xmax_double": lambda self: self.xmax[self.double_mask],
        }.items():
            if memory_efficient == 0:
                self.register_buffer(k, v(self), persistent=False)
            else:
                self._properties[k] = v

        self._memory_mode = memory_efficient

    def __getattr__(self, name: str):
        if "_properties" in self.__dict__:
            _properties = self.__dict__["_properties"]
            if name in _properties:
                return _properties[name](self)
        return super().__getattr__(name)


class DCBasicNN(nn.Module):
    """Real ML4OPF `BasicNN.__init__`/`make_network`/`forward`
    (lightning_basic_nn.py) plus `DCBasicNN.add_boundrepair`
    (dcopf_basic_nn.py), verbatim -- with the `pytorch_lightning.LightningModule`
    base, hyperparameter logging, and training-loop methods dropped (unused by
    forward()). `input_size`/`output_size`/`pg_slice` mirror the real
    `@property` definitions but read plain ints/slices instead of an
    `opfmodel`/`slices` dataset object, matching a tiny toy DC-OPF case."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        pg_slice: slice,
        pgmin: torch.Tensor,
        pgmax: torch.Tensor,
        hidden_sizes=(32, 32),
        activation: str = "relu",
        boundrepair: str = "relu",
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.pg_slice = pg_slice
        self.hidden_sizes = list(hidden_sizes)

        self.set_activation(activation)
        self.make_network()
        self.add_boundrepair(boundrepair, pgmin, pgmax)

    def set_activation(self, activation: str):
        activation = activation.lower()
        if activation == "relu":
            self.activation = nn.ReLU
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid
        else:
            raise ValueError(f"Activation function {activation} not supported.")

    def make_network(self):
        # Real BasicNN.make_network, verbatim.
        self.layers = nn.Sequential()

        self.layers.append(nn.Linear(self.input_size, self.hidden_sizes[0]))
        self.layers.append(self.activation())

        for i in range(1, len(self.hidden_sizes)):
            self.layers.append(nn.Linear(self.hidden_sizes[i - 1], self.hidden_sizes[i]))
            self.layers.append(self.activation())

        self.layers.append(nn.Linear(self.hidden_sizes[-1], self.output_size))

    def add_boundrepair(self, boundrepair: str, pgmin: torch.Tensor, pgmax: torch.Tensor):
        # Real DCBasicNN.add_boundrepair, verbatim.
        if boundrepair == "none" or boundrepair is None:
            return

        lower = torch.full((self.output_size,), -torch.inf)
        upper = torch.full((self.output_size,), torch.inf)

        lower[self.pg_slice] = pgmin
        upper[self.pg_slice] = pgmax

        self.layers.append(BoundRepair(lower, upper, boundrepair))

    def forward(self, x: torch.Tensor):
        # Real BasicNN.forward, verbatim.
        return self.layers.forward(x)


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo).
# ---------------------------------------------------------------------------
# Tiny toy DC-OPF case: nbus=6 buses, ng=2 generators. Real slices for a
# DC-OPF problem: input = pd (bus active demand, size nbus), output =
# [pg (size ng), va (size nbus)] concatenated, matching DCBasicNN.pd_slice /
# pg_slice / va_slice.
_NBUS = 6
_NG = 2
_INPUT_SIZE = _NBUS
_OUTPUT_SIZE = _NG + _NBUS
_PG_SLICE = slice(0, _NG)
_BATCH = 4


def build_dcopf_basicnn():
    torch.manual_seed(0)
    pgmin = torch.zeros(_NG)
    pgmax = torch.full((_NG,), 2.0)
    model = DCBasicNN(
        input_size=_INPUT_SIZE,
        output_size=_OUTPUT_SIZE,
        pg_slice=_PG_SLICE,
        pgmin=pgmin,
        pgmax=pgmax,
        hidden_sizes=(32, 32),
        activation="relu",
        boundrepair="relu",
    )
    model.eval()
    return model


def example_input_dcopf_basicnn():
    torch.manual_seed(0)
    return torch.rand(_BATCH, _INPUT_SIZE)


MENAGERIE_ENTRIES = [
    (
        "DCOPF-Net-BasicNN",
        "build_dcopf_basicnn",
        "example_input_dcopf_basicnn",
        2024,
        MENAGERIE_ZOO,
    ),
]
