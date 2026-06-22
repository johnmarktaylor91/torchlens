"""Exact-name FFJORD continuous normalizing flow entry.

Paper: Grathwohl et al. 2019, "FFJORD: Free-form Continuous Dynamics for
Scalable Reversible Generative Models." The compact implementation keeps a
time-conditioned ODE dynamics MLP and fixed-step Euler CNF integration.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from menagerie.classics.neural_ode_flows import build_ffjord_cnf as _build
from menagerie.classics.neural_ode_flows import example_input_ffjord as _example_input


def build() -> nn.Module:
    """Build compact FFJORD CNF.

    Returns
    -------
    nn.Module
        Random-initialized FFJORD-style CNF.
    """

    return _build()


def example_input() -> torch.Tensor:
    """Create a simple latent tensor.

    Returns
    -------
    torch.Tensor
        Latent tensor of shape ``(2, 8)``.
    """

    return _example_input()


MENAGERIE_ENTRIES = [("ffjord_cnf", "build", "example_input", "2019", "DC")]
