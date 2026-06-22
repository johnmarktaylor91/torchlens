"""Exact-name aliases for already implemented classics."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from menagerie.classics.spinn import build as _build_spinn
from menagerie.classics.spinn import example_input as _example_spinn_input


def build_spinn_alias() -> nn.Module:
    """Build the existing faithful SPINN reconstruction.

    Returns
    -------
    nn.Module
        SPINN model.
    """

    return _build_spinn()


def example_spinn_alias() -> Tensor:
    """Return the existing SPINN example input.

    Returns
    -------
    Tensor
        Token id input.
    """

    return _example_spinn_input()


MENAGERIE_ENTRIES = [("sPINN", "build_spinn_alias", "example_spinn_alias", "2016", "parser/rnn")]
