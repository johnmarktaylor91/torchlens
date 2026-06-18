"""Evolving Neuro-Fuzzy Inference Systems.

Three online self-constructing neuro-fuzzy networks from the 2001-2002 Kasabov
group and related contemporaries.  All three grow their rule base dynamically
during learning; only the forward inference pass is implemented here (no online
learning loop), which makes each module trace-clean.

DENFIS, 2002, Kasabov and Song:
    "DENFIS: Dynamic Evolving Neural-Fuzzy Inference System and Its Application
    for Time-Series Prediction."
    Gaussian rules organized by an ECM cluster bank; m most-active rules
    contribute a Takagi-Sugeno output via normalized firing.

EFuNN, 2001, Kasabov:
    "Evolving Fuzzy Neural Networks -- Algorithms, Applications and Biological
    Motivation."
    Five-layer evolving fuzzy NN: input -> fuzzy MF layer -> rule-node layer
    (W1 weights) -> fuzzy output layer (W2 weights) -> defuzzification.
    Growth of rule nodes is governed by error/novelty thresholds (omitted here).

SONFIN, 1998, Juang and Lin:
    "A Self-Constructing Fuzzy Neural Network."
    Self-constructing TSK: input -> Gaussian MF layer -> rule firing ->
    TSK consequent (linear) -> normalized output.
    Structure-learning part omitted; forward inference is differentiable.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# DENFIS
# ---------------------------------------------------------------------------


class DENFIS(nn.Module):
    """Dynamic Evolving Neuro-Fuzzy Inference System -- forward inference only.

    Rules are Gaussian clusters with Takagi-Sugeno (affine) consequents.
    The m nearest rules (measured by Gaussian activation) are used, normalized,
    and combined.  ECM online clustering is omitted; clusters are initialized
    randomly.  ``n_rules`` plays the role of the maximum rule bank size.
    """

    def __init__(
        self,
        n_inputs: int = 4,
        n_rules: int = 8,
        n_outputs: int = 1,
        sigma: float = 0.5,
        m_active: int = 4,
    ) -> None:
        """Initialize Gaussian rule clusters and TSK consequents.

        Parameters
        ----------
        n_inputs:
            Number of input features.
        n_rules:
            Maximum number of rules in the bank.
        n_outputs:
            Number of output values.
        sigma:
            Fixed Gaussian width for rule activation.
        m_active:
            Number of most-active rules used per forward pass.
        """
        super().__init__()
        self.sigma = sigma
        self.m_active = min(m_active, n_rules)
        # Rule centres
        self.centres = nn.Parameter(torch.randn(n_rules, n_inputs) * 0.5)
        # TSK consequent: affine output per rule -- [n_rules, n_inputs + 1]
        self.consequents = nn.Parameter(torch.randn(n_rules, n_inputs + 1) * 0.1)
        self.n_outputs = n_outputs
        if n_outputs != 1:
            # For multi-output, one consequent set per output
            self.consequents = nn.Parameter(torch.randn(n_rules, n_outputs * (n_inputs + 1)) * 0.1)

    def forward(self, x: Tensor) -> Tensor:
        """Run DENFIS forward inference over m most-active rules.

        Parameters
        ----------
        x:
            Input tensor with shape ``(batch, n_inputs)``.

        Returns
        -------
        Tensor
            Output tensor with shape ``(batch, n_outputs)``.
        """
        n_inputs = x.shape[-1]
        # Gaussian activation for each rule: (B, n_rules)
        diff = x[:, None, :] - self.centres[None, :, :]  # (B, R, D)
        activation = torch.exp(-diff.pow(2).sum(dim=-1) / (2.0 * self.sigma**2))

        # Select m_active most-active rules (differentiable via soft selection)
        # We use top-k activations as a mask, then softmax-normalize
        topk_vals, _ = activation.topk(self.m_active, dim=-1)  # (B, m)
        threshold = topk_vals[:, -1:].detach()  # minimum of top-k
        mask = (activation >= threshold).float()  # (B, R)
        masked = activation * mask  # zero out non-top rules
        norm = masked / masked.sum(dim=-1, keepdim=True).clamp_min(1e-6)  # (B, R)

        # TSK consequent: y_r = [x | 1] @ consequent_r
        x_aug = torch.cat([x, torch.ones_like(x[:, :1])], dim=-1)  # (B, D+1)
        # (B, R) via dot(x_aug, consequents[r])
        rule_out = x_aug @ self.consequents[:, : n_inputs + 1].T  # (B, R)
        return (norm * rule_out).sum(dim=-1, keepdim=True)  # (B, 1)


# ---------------------------------------------------------------------------
# EFuNN
# ---------------------------------------------------------------------------


class EFuNN(nn.Module):
    """Evolving Fuzzy Neural Network -- five-layer forward inference.

    Layer 1: input
    Layer 2: fuzzy membership (Gaussian) -- n_mfs per input
    Layer 3: rule nodes (W1: membership -> rule activation)
    Layer 4: fuzzy output (W2: rule -> output membership)
    Layer 5: defuzzification (centroid of output memberships)

    Rule growth (ECOS algorithm) is omitted; the rule bank is pre-allocated
    randomly.
    """

    def __init__(
        self,
        n_inputs: int = 4,
        n_mfs: int = 3,
        n_rules: int = 8,
        n_outputs: int = 2,
    ) -> None:
        """Initialize membership functions and W1/W2 weight matrices.

        Parameters
        ----------
        n_inputs:
            Number of input variables.
        n_mfs:
            Gaussian membership functions per input.
        n_rules:
            Number of rule nodes (max rule bank size).
        n_outputs:
            Number of output classes / values.
        """
        super().__init__()
        self.n_mfs = n_mfs
        # Gaussian MF parameters: one centre + log_sigma per (input, mf)
        self.centres = nn.Parameter(torch.linspace(-1.0, 1.0, n_mfs).repeat(n_inputs, 1))
        self.log_sigma = nn.Parameter(torch.zeros(n_inputs, n_mfs))
        # W1: (n_mfs * n_inputs) -> n_rules
        self.W1 = nn.Parameter(torch.randn(n_rules, n_inputs * n_mfs) * 0.1)
        # W2: n_rules -> n_outputs
        self.W2 = nn.Parameter(torch.randn(n_outputs, n_rules) * 0.1)

    def forward(self, x: Tensor) -> Tensor:
        """Propagate through the five EFuNN layers.

        Parameters
        ----------
        x:
            Input tensor with shape ``(batch, n_inputs)``.

        Returns
        -------
        Tensor
            Output class confidences with shape ``(batch, n_outputs)``.
        """
        # L2: Gaussian memberships (B, n_inputs, n_mfs)
        sigma = torch.nn.functional.softplus(self.log_sigma) + 0.1
        diff = x[:, :, None] - self.centres[None, :, :]
        mu = torch.exp(-0.5 * (diff / sigma[None]).pow(2))  # (B, n_in, n_mfs)
        # Flatten to (B, n_in * n_mfs)
        mu_flat = mu.flatten(start_dim=1)
        # L3: rule activation via W1 (B, n_rules)
        rule_act = torch.sigmoid(mu_flat @ self.W1.T)
        # L4: output membership via W2 (B, n_out)
        out = rule_act @ self.W2.T
        # L5: sigmoid for [0,1] output membership (defuzzification surrogate)
        return torch.sigmoid(out)


# ---------------------------------------------------------------------------
# SONFIN
# ---------------------------------------------------------------------------


class SONFIN(nn.Module):
    """Self-cOnstructing Neuro-Fuzzy Inference Net -- forward TSK inference.

    Architecture:
        Input -> Gaussian MF layer -> rule firing (product T-norm) ->
        normalization -> affine TSK consequents -> weighted sum output.

    Structure-learning (partition-growing via aligned clustering and rule
    pruning) is omitted.  The inference graph is identical to a fixed ANFIS
    with product firing strengths, which traces cleanly.
    """

    def __init__(
        self,
        n_inputs: int = 4,
        n_rules: int = 6,
    ) -> None:
        """Initialize Gaussian MFs (one per rule per input) and TSK consequents.

        Parameters
        ----------
        n_inputs:
            Number of input variables.
        n_rules:
            Number of fuzzy rules (each rule has one MF per input).
        """
        super().__init__()
        self.n_inputs = n_inputs
        self.n_rules = n_rules
        # Gaussian MF: centre + log_sigma, shape (n_rules, n_inputs)
        centres = torch.linspace(-1.0, 1.0, n_rules).unsqueeze(1).expand(n_rules, n_inputs)
        self.centres = nn.Parameter(centres.clone())
        self.log_sigma = nn.Parameter(torch.zeros(n_rules, n_inputs))
        # TSK consequents: affine output per rule, shape (n_rules, n_inputs + 1)
        self.consequents = nn.Parameter(torch.randn(n_rules, n_inputs + 1) * 0.1)

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate SONFIN Takagi-Sugeno output.

        Parameters
        ----------
        x:
            Input tensor with shape ``(batch, n_inputs)``.

        Returns
        -------
        Tensor
            Scalar TSK output with shape ``(batch, 1)``.
        """
        sigma = torch.nn.functional.softplus(self.log_sigma) + 0.1
        # Gaussian MF per (rule, input): (B, n_rules, n_inputs)
        diff = x[:, None, :] - self.centres[None, :, :]
        mu = torch.exp(-0.5 * (diff / sigma[None]).pow(2))
        # Product T-norm firing strength: (B, n_rules)
        firing = mu.prod(dim=-1)
        # Normalize
        norm = firing / firing.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        # TSK consequents: (B, n_rules) via [x | 1] @ consequents.T
        x_aug = torch.cat([x, torch.ones_like(x[:, :1])], dim=-1)
        rule_out = x_aug @ self.consequents.T  # (B, n_rules)
        return (norm * rule_out).sum(dim=-1, keepdim=True)  # (B, 1)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def build_denfis() -> nn.Module:
    """Build a small DENFIS module.

    Returns
    -------
    nn.Module
        Configured ``DENFIS`` instance.
    """
    return DENFIS(n_inputs=4, n_rules=8, m_active=4)


def example_input_denfis() -> Tensor:
    """Create an input example for DENFIS.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 4)``.
    """
    return torch.randn(1, 4)


def build_efunn() -> nn.Module:
    """Build a small EFuNN module.

    Returns
    -------
    nn.Module
        Configured ``EFuNN`` instance.
    """
    return EFuNN(n_inputs=4, n_mfs=3, n_rules=8, n_outputs=2)


def example_input_efunn() -> Tensor:
    """Create an input example for EFuNN.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 4)``.
    """
    return torch.randn(1, 4)


def build_sonfin() -> nn.Module:
    """Build a small SONFIN module.

    Returns
    -------
    nn.Module
        Configured ``SONFIN`` instance.
    """
    return SONFIN(n_inputs=4, n_rules=6)


def example_input_sonfin() -> Tensor:
    """Create an input example for SONFIN.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 4)``.
    """
    return torch.randn(1, 4)


MENAGERIE_ENTRIES = [
    (
        "DENFIS (Dynamic Evolving Neuro-Fuzzy Inference)",
        "build_denfis",
        "example_input_denfis",
        "2002",
        "RT",
    ),
    (
        "EFuNN (Evolving Fuzzy Neural Network)",
        "build_efunn",
        "example_input_efunn",
        "2001",
        "RT",
    ),
    (
        "SONFIN (Self-cOnstructing Neuro-Fuzzy Inference Net)",
        "build_sonfin",
        "example_input_sonfin",
        "1998",
        "RT",
    ),
]
