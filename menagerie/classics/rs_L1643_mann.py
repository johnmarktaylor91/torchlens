# SOURCE: vendored from https://github.com/ami-iit/mann-pytorch @ 7386952b61440c19459bbe680b818faf7f461ede
#
# MANN (Mode-Adaptive Neural Networks for Quadruped Motion Control, Zhang/
# Starke/Komura/Saito, SIGGRAPH 2018) -- the official PyTorch architecture is
# TensorFlow-1.x only (AI4Animation/SIGGRAPH_2018/TensorFlow/MANN); this repo
# is the maintained PyTorch reimplementation cited by the queue notes
# (ami-iit/mann-pytorch). Vendored verbatim from src/mann_pytorch/
# GatingNetwork.py and src/mann_pytorch/MotionPredictionNetwork.py -- the
# real two-subnetwork MANN architecture: a 3-layer Gating Network producing
# softmax blending coefficients over `num_experts` weight banks, and a
# 3-layer Motion Prediction Network whose per-layer weights/biases are
# einsum-blended from `num_experts` expert weight tensors using those
# coefficients (the paper's mixture-of-experts gating mechanism). Layer code
# is copied unmodified; only src/mann_pytorch/MANN.py's forward() composition
# (`gn(x.T)` -> `mpn(x, blending_coefficients)`) is reproduced directly here
# (as build_/example_input_ functions) instead of vendoring MANN.__init__,
# because the real MANN.__init__ derives input/output sizes from a
# DataLoader (training-only plumbing, not architecture).
import numpy as np
import torch
from torch import nn


"""
===============================================================================
Gating Network (src/mann_pytorch/GatingNetwork.py, vendored verbatim)
===============================================================================
"""


class GatingNetwork(nn.Module):
    """Class for the Gating Network included in the MANN architecture."""

    def __init__(
        self, input_size: int, output_size: int, hidden_size: int, dropout_probability: float
    ):
        """Gating Network constructor.

        Args:
            input_size (int): The dimension of the input vector for the network
            output_size (int): The dimension of the output vector for the network
            hidden_size (int): The dimension of the 3 hidden layers of the network
            dropout_probability (float): The probability of an element to be zeroed in the network training
        """

        # Superclass constructor
        super(GatingNetwork, self).__init__()

        # Dimensions
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # Parameters
        w0 = torch.zeros(self.hidden_size, self.input_size)
        w1 = torch.zeros(self.hidden_size, self.hidden_size)
        w2 = torch.zeros(self.output_size, self.hidden_size)
        b0 = torch.zeros(self.hidden_size, 1)
        b1 = torch.zeros(self.hidden_size, 1)
        b2 = torch.zeros(self.output_size, 1)

        # Initialization
        w0 = self.initialize_gn_weights(w0)
        w1 = self.initialize_gn_weights(w1)
        w2 = self.initialize_gn_weights(w2)

        # Make explicit that the parameters must be optimized in the learning process
        self.w0 = nn.Parameter(w0)
        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.b0 = nn.Parameter(b0)
        self.b1 = nn.Parameter(b1)
        self.b2 = nn.Parameter(b2)

        # Activation functions and layers to be exploited in the forward call
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=dropout_probability)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gating Network 3-layers architecture.

        Args:
            x (torch.Tensor): The input vector for the network

        Returns:
            y (torch.Tensor): The normalized blending coefficients representing the output of the network
        """

        # Input processing
        x = self.dropout(x)

        # Layer 1
        H0 = torch.matmul(self.w0, x) + self.b0
        H0 = self.elu(H0)
        H0 = self.dropout(H0)

        # Layer 2
        H1 = torch.matmul(self.w1, H0) + self.b1
        H1 = self.elu(H1)
        H1 = self.dropout(H1)

        # Layer 3
        H2 = torch.matmul(self.w2, H1) + self.b2

        # Softmax to output normalized blending coefficients
        y = self.softmax(H2)

        return y

    @staticmethod
    def initialize_gn_weights(w: torch.Tensor) -> torch.Tensor:
        """Initialize the Gating Network weights using uniform distribution with
        bounds defined on the basis of the dimensions of the network layers.

        Args:
            w (torch.Tensor): The empty network layer weights, to be initialized

        Returns:
            w_init (torch.Tensor): The initialized network layer weights
        """

        bound = np.sqrt(6.0 / np.sum(w.shape[-2:]))
        w_init = w.uniform_(-bound, bound)

        return w_init


"""
===============================================================================
Motion Prediction Network (src/mann_pytorch/MotionPredictionNetwork.py,
vendored verbatim)
===============================================================================
"""


class MotionPredictionNetwork(nn.Module):
    """Class for the Motion Prediction Network included in the MANN architecture."""

    def __init__(
        self,
        num_experts: int,
        input_size: int,
        output_size: int,
        hidden_size: int,
        dropout_probability: float,
    ):
        """Motion Prediction Network constructor.

        Args:
            num_experts (int): The number of expert weights constituting the network
            input_size (int): The dimension of the input vector for the network
            output_size (int): The dimension of the output vector for the network
            hidden_size (int): The dimension of the 3 hidden layers of the network
            dropout_probability (float): The probability of an element to be zeroed in the network training
        """

        # Superclass constructor
        super(MotionPredictionNetwork, self).__init__()

        # Dimensions
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_experts = num_experts

        # Parameters
        w0 = torch.zeros(self.num_experts, self.hidden_size, self.input_size)
        w1 = torch.zeros(self.num_experts, self.hidden_size, self.hidden_size)
        w2 = torch.zeros(self.num_experts, self.output_size, self.hidden_size)
        b0 = torch.zeros(self.num_experts, self.hidden_size, 1)
        b1 = torch.zeros(self.num_experts, self.hidden_size, 1)
        b2 = torch.zeros(self.num_experts, self.output_size, 1)

        # Initialization
        w0 = self.initialize_mpn_weights(w0)
        w1 = self.initialize_mpn_weights(w1)
        w2 = self.initialize_mpn_weights(w2)

        # Make explicit that the parameters must be optimized in the learning process
        self.w0 = nn.Parameter(w0)
        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.b0 = nn.Parameter(b0)
        self.b1 = nn.Parameter(b1)
        self.b2 = nn.Parameter(b2)

        # Activation functions and layers to be exploited in the forward call
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, x: torch.Tensor, blending_coefficients: torch.Tensor) -> torch.Tensor:
        """Motion Prediction Network 3-layers architecture.

        Args:
            x (torch.Tensor): The input vector for the network
            blending_coefficients (torch.Tensor): The coefficients, outputed by the Gating Network, used to blend the expert weights

        Returns:
            y (torch.Tensor): The vector outputed by the network
        """

        # Input processing
        x = torch.unsqueeze(x, -1)
        x = self.dropout(x)

        # Layer 1
        blended_w0 = torch.einsum("ij,ikl->jkl", blending_coefficients, self.w0)
        blended_b0 = torch.einsum("ij,ikl->jkl", blending_coefficients, self.b0)
        H0 = torch.matmul(blended_w0, x) + blended_b0
        H0 = self.elu(H0)
        H0 = self.dropout(H0)

        # Layer 2
        blended_w1 = torch.einsum("ij,ikl->jkl", blending_coefficients, self.w1)
        blended_b1 = torch.einsum("ij,ikl->jkl", blending_coefficients, self.b1)
        H1 = torch.matmul(blended_w1, H0) + blended_b1
        H1 = self.elu(H1)
        H1 = self.dropout(H1)

        # Layer 3
        blended_w2 = torch.einsum("ij,ikl->jkl", blending_coefficients, self.w2)
        blended_b2 = torch.einsum("ij,ikl->jkl", blending_coefficients, self.b2)
        H2 = torch.matmul(blended_w2, H1) + blended_b2
        y = torch.squeeze(H2, -1)

        return y

    @staticmethod
    def initialize_mpn_weights(w: torch.Tensor) -> torch.Tensor:
        """Initialize the Motion Prediction Network weights using uniform distribution
        with bounds defined on the basis of the dimensions of the network layers.

        Args:
            w (torch.Tensor): The empty network layer weights, to be initialized

        Returns:
            w_init (torch.Tensor): The initialized network layer weights
        """

        bound = np.sqrt(6.0 / np.prod(w.shape[-2:]))
        w_init = w.uniform_(-bound, bound)

        return w_init


"""
===============================================================================
MANN composition (reproduces src/mann_pytorch/MANN.py forward(): the real
MANN.__init__ derives input_size/output_size from a DataLoader batch, which
is training plumbing, not architecture -- we compose the two REAL vendored
subnetworks the same way MANN.forward() does.
===============================================================================
"""


class MANN(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_experts: int,
        gn_hidden_size: int,
        mpn_hidden_size: int,
        dropout_probability: float,
    ):
        super().__init__()
        self.gn = GatingNetwork(
            input_size=input_size,
            output_size=num_experts,
            hidden_size=gn_hidden_size,
            dropout_probability=dropout_probability,
        )
        self.mpn = MotionPredictionNetwork(
            num_experts=num_experts,
            input_size=input_size,
            output_size=output_size,
            hidden_size=mpn_hidden_size,
            dropout_probability=dropout_probability,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blending_coefficients = self.gn(x.T)
        y = self.mpn(x, blending_coefficients=blending_coefficients)
        return y


# --- staging harness: build + example input ---------------------------------


def build_mann():
    # tiny sizes: 8 experts (matches the SIGGRAPH-2018 paper's canonical
    # expert count), small input/hidden dims for a fast trace.
    return MANN(
        input_size=20,
        output_size=12,
        num_experts=8,
        gn_hidden_size=16,
        mpn_hidden_size=16,
        dropout_probability=0.0,
    )


def example_input_mann():
    batch_size = 4
    return torch.randn(batch_size, 20)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("MANN", build_mann, example_input_mann, 2018, MENAGERIE_ZOO),
]
