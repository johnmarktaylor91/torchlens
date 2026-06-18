"""Interval Type-2 Fuzzy Neural Network (IT2-FNN), 2000, Liang and Mendel.

A seven-layer network that explicitly models second-order uncertainty in fuzzy
membership degrees.  Each input variable is covered by interval type-2 Gaussian
membership functions (IT2 GMFs) that produce upper and lower membership bounds
(mu_upper, mu_lower) rather than a single crisp degree.  The Karnik-Mendel (KM)
type-reduction layer collapses the interval rule firings to a pair of scalar
bounds (y_l, y_r); the final defuzzified output is (y_l + y_r) / 2.

The KM algorithm is iterative in principle; here a differentiable closed-form
approximation is used (mean of weighted lower/upper bounds) so the module traces
cleanly without data-dependent Python loops on tensor values.

Paper: Liang and Mendel 2000, "Interval Type-2 Fuzzy Logic Systems: Theory and
       Design," IEEE Transactions on Fuzzy Systems.
"""

import itertools

import torch
from torch import Tensor, nn


class IT2FNN(nn.Module):
    """Interval Type-2 Fuzzy Neural Network with differentiable type-reduction.

    Layers:
    1. Input layer (pass-through)
    2. IT2 Gaussian MF layer: (mu_upper, mu_lower) per (input, rule) pair
    3. Rule firing interval: w_upper = prod(mu_upper_i), w_lower = prod(mu_lower_i)
    4. Consequents: affine TSK output per rule (crisp)
    5. Normalized firing (upper/lower bounds separately)
    6. KM type-reduction approximation: y_l = weighted sum by lower, y_r by upper
    7. Defuzzification: (y_l + y_r) / 2

    Only two inputs and two MFs per input are used by default to keep the rule
    base small and the forward pass fast.
    """

    def __init__(
        self,
        n_inputs: int = 2,
        n_mfs: int = 3,
    ) -> None:
        """Initialize IT2 Gaussian MF parameters and TSK consequents.

        Parameters
        ----------
        n_inputs:
            Number of crisp input variables.
        n_mfs:
            Number of IT2 GMFs per input.
        """
        super().__init__()
        n_rules = n_mfs**n_inputs

        # IT2 GMF parameters: centres are shared; widths form an interval.
        # centres: (n_inputs, n_mfs)
        centres = torch.linspace(-1.0, 1.0, n_mfs).repeat(n_inputs, 1)
        self.centres = nn.Parameter(centres)
        # sigma_upper >= sigma_lower: parameterize as (sigma_l, delta >= 0)
        self.log_sigma_lower = nn.Parameter(torch.zeros(n_inputs, n_mfs))
        self.log_delta = nn.Parameter(torch.full((n_inputs, n_mfs), -2.0))

        # Rule index combinations: (n_rules, n_inputs) with MF indices
        rule_idx = torch.tensor(
            list(itertools.product(range(n_mfs), repeat=n_inputs)), dtype=torch.long
        )
        self.register_buffer("rule_idx", rule_idx)  # (n_rules, n_inputs)
        # TSK consequents: (n_rules, n_inputs + 1)
        self.consequents = nn.Parameter(torch.randn(n_rules, n_inputs + 1) * 0.1)

    def _it2_mf(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Compute upper and lower IT2 Gaussian membership degrees.

        Parameters
        ----------
        x:
            Input tensor with shape ``(batch, n_inputs)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(mu_upper, mu_lower)`` each with shape
            ``(batch, n_inputs, n_mfs)``.
        """
        sigma_lower = torch.nn.functional.softplus(self.log_sigma_lower) + 0.1
        delta = torch.nn.functional.softplus(self.log_delta)
        sigma_upper = sigma_lower + delta  # wider interval = more uncertainty

        diff = x[:, :, None] - self.centres[None, :, :]  # (B, n_in, n_mfs)
        mu_upper = torch.exp(-0.5 * (diff / sigma_upper[None]).pow(2))
        mu_lower = torch.exp(-0.5 * (diff / sigma_lower[None]).pow(2))
        return mu_upper, mu_lower

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate IT2-FNN from input to defuzzified output.

        Parameters
        ----------
        x:
            Input tensor with shape ``(batch, n_inputs)``.

        Returns
        -------
        Tensor
            Crisp output with shape ``(batch, 1)``.
        """
        mu_u, mu_l = self._it2_mf(x)  # (B, n_in, n_mfs)

        # Gather MF activations per rule: (B, n_rules, n_inputs)
        # rule_idx[r, i] = which MF index for input i in rule r
        ru = mu_u[:, torch.arange(x.shape[1])[None, :], self.rule_idx]  # (B, R, D)
        rl = mu_l[:, torch.arange(x.shape[1])[None, :], self.rule_idx]

        # Product T-norm firing strength intervals
        w_upper = ru.prod(dim=-1)  # (B, n_rules)
        w_lower = rl.prod(dim=-1)  # (B, n_rules)

        # TSK consequent output per rule: (B, n_rules)
        x_aug = torch.cat([x, torch.ones_like(x[:, :1])], dim=-1)
        f = x_aug @ self.consequents.T  # (B, n_rules)

        # KM type-reduction approximation:
        # y_l: normalize by lower firing weights; y_r: normalize by upper
        def _normalized_output(w: Tensor) -> Tensor:
            denom = w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            return (w / denom * f).sum(dim=-1, keepdim=True)  # (B, 1)

        y_l = _normalized_output(w_lower)
        y_r = _normalized_output(w_upper)
        # Defuzzification: centroid of the reduced type-1 interval
        return (y_l + y_r) * 0.5


def build() -> nn.Module:
    """Build a small IT2-FNN module.

    Returns
    -------
    nn.Module
        Configured ``IT2FNN`` instance.
    """
    return IT2FNN(n_inputs=2, n_mfs=3)


def example_input() -> Tensor:
    """Create an example input for the IT2-FNN.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 2)``.
    """
    return torch.tensor([[0.3, -0.4]], dtype=torch.float32)


MENAGERIE_ENTRIES = [
    (
        "IT2-FNN (Interval Type-2 Fuzzy Neural Network)",
        "build",
        "example_input",
        "2000",
        "RT",
    )
]
