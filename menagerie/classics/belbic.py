"""BELBIC (Brain Emotional Learning Based Intelligent Controller), 2004, Lucas et al.

A controller inspired by the mammalian limbic system.  Two competing pathways --
the amygdala (fear-conditioning / fast approach) and the orbitofrontal cortex
(inhibitory / refinement) -- each compute a weighted sum of sensory inputs.
The emotional output is E = A - O where A is amygdala activation and O is OFC
inhibition; this signal drives a controller action.

Learning rules (Moren and Balkenius, 2000):
    dV_i = alpha * s_i * max(0, REW - A)   (amygdala, monotone-increasing)
    dW_i = beta  * s_i * (E - REW)         (OFC)

Only the differentiable forward inference graph is implemented here; the
weight-update learning rules are included as a ``learning_step`` method but are
NOT called inside ``forward``, keeping tracing clean.

Paper: Lucas, Shahmirzadi, Sheikholeslami 2004, "Introducing BELBIC: Brain
       Emotional Learning Based Intelligent Controller."
"""

import torch
from torch import Tensor, nn


class BELBIC(nn.Module):
    """Brain Emotional Learning Based Intelligent Controller.

    Sensory inputs are multiplied by amygdala weights V and OFC weights W.
    The emotional output E = A - O is the control signal fed back to the plant.

    In a control loop the reward signal REW is external; here it is passed
    as an optional argument so the module can also run in open-loop mode.
    """

    def __init__(self, n_sensory: int = 4) -> None:
        """Initialize amygdala and OFC weight vectors.

        Parameters
        ----------
        n_sensory:
            Number of sensory input features.
        """
        super().__init__()
        # Amygdala weights V -- constrained non-negative by construction via
        # softplus; initialized near zero as in the BELBIC literature.
        self.log_V = nn.Parameter(torch.zeros(n_sensory))
        # OFC inhibitory weights W -- unconstrained
        self.W = nn.Parameter(torch.zeros(n_sensory))

    def forward(self, s: Tensor, rew: Tensor | None = None) -> Tensor:
        """Compute emotional output E = A - O.

        Parameters
        ----------
        s:
            Sensory input with shape ``(batch, n_sensory)``.
        rew:
            Reward signal with shape ``(batch, 1)`` or ``(batch,)``.
            Unused in the forward inference pass (included for API completeness).

        Returns
        -------
        Tensor
            Emotional control signal with shape ``(batch, 1)``.
        """
        # Non-negative amygdala weights via softplus
        V = torch.nn.functional.softplus(self.log_V)
        # Amygdala activation: weighted sum of sensory inputs
        A = (s * V).sum(dim=-1, keepdim=True)
        # OFC inhibition: unconstrained weighted sum
        ofc = (s * self.W).sum(dim=-1, keepdim=True)
        return A - ofc

    @torch.no_grad()
    def learning_step(
        self,
        s: Tensor,
        rew: Tensor,
        alpha: float = 0.1,
        beta: float = 0.05,
    ) -> None:
        """Apply one step of the BELBIC Hebbian update rules (in-place).

        This is NOT called during forward(); it is a utility for training loops
        that want to use the biologically-motivated learning signal.

        Parameters
        ----------
        s:
            Sensory input used in the most recent forward pass.
        rew:
            Reward signal matching the most recent forward output.
        alpha:
            Amygdala learning rate.
        beta:
            OFC learning rate.
        """
        V = torch.nn.functional.softplus(self.log_V)
        A = (s * V).sum(dim=-1, keepdim=True)
        E = self.forward(s)
        # Amygdala: monotone-increasing, only strengthen
        dV = alpha * (s * (rew - A).clamp_min(0.0)).mean(dim=0)
        dW = beta * (s * (E - rew)).mean(dim=0)
        # Update raw parameters
        # log_V: softplus is monotone so shift to preserve positivity
        self.log_V.data += dV
        self.W.data -= dW


def build() -> nn.Module:
    """Build a small BELBIC controller module.

    Returns
    -------
    nn.Module
        Configured ``BELBIC`` instance.
    """
    return BELBIC(n_sensory=4)


def example_input() -> Tensor:
    """Create a sensory input example for BELBIC.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 4)``.
    """
    return torch.randn(1, 4)


MENAGERIE_ENTRIES = [
    (
        "BELBIC (Brain Emotional Learning Controller)",
        "build",
        "example_input",
        "2004",
        "RT",
    )
]
