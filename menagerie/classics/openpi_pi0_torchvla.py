"""OpenPI π0 and π0.5 compact vision-language-action models.

Paper: Black et al. 2024, "π0: A Vision-Language-Action Flow Model for
General Robot Control"; OpenPI also exposes π0-FAST, an autoregressive VLA
using FAST action tokenization.
"""

from __future__ import annotations

import torch
import torch.nn as nn

FLOW_IMAGE_VALUES = 3 * 32 * 32
FLOW_TEXT_VALUES = 5
FLOW_STATE_VALUES = 7
FLOW_ACTION_STEPS = 4
FLOW_ACTION_VALUES = FLOW_ACTION_STEPS * 7
FLOW_INPUT_VALUES = (
    FLOW_IMAGE_VALUES + FLOW_TEXT_VALUES + FLOW_STATE_VALUES + FLOW_ACTION_VALUES + 1
)
FAST_ACTION_VALUES = 8
FAST_INPUT_VALUES = FLOW_IMAGE_VALUES + FLOW_TEXT_VALUES + FAST_ACTION_VALUES


class BlockwiseCausalSelfAttention(nn.Module):
    """Transformer block with π0-style blockwise causal masking."""

    def __init__(self, dim: int) -> None:
        """Initialize attention block.

        Parameters
        ----------
        dim:
            Token width.
        """

        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 3), nn.GELU(), nn.Linear(dim * 3, dim)
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply masked self-attention and feed-forward update.

        Parameters
        ----------
        tokens:
            Token sequence.
        mask:
            Attention mask.

        Returns
        -------
        torch.Tensor
            Updated tokens.
        """

        y, _ = self.attn(self.norm(tokens), self.norm(tokens), self.norm(tokens), attn_mask=mask)
        tokens = tokens + y
        return tokens + self.ff(tokens)


class Pi0FlowPolicy(nn.Module):
    """Compact π0 flow-matching VLA with VLM and action expert tokens."""

    def __init__(self, dim: int = 48, action_dim: int = 7) -> None:
        """Initialize π0 model.

        Parameters
        ----------
        dim:
            Token width.
        action_dim:
            Continuous action dimension.
        """

        super().__init__()
        self.image = nn.Conv2d(3, dim, 8, stride=8)
        self.text = nn.Embedding(128, dim)
        self.state = nn.Linear(action_dim, dim)
        self.action = nn.Linear(action_dim + 1, dim)
        self.blocks = nn.ModuleList([BlockwiseCausalSelfAttention(dim) for _ in range(2)])
        self.out = nn.Linear(dim, action_dim)

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        state: torch.Tensor,
        noisy_action: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        """Predict flow vectors for continuous action chunks.

        Parameters
        ----------
        image:
            Observation image.
        text:
            Prompt token ids.
        state:
            Proprioceptive state.
        noisy_action:
            Noisy action chunk.
        tau:
            Flow-matching time.

        Returns
        -------
        torch.Tensor
            Action vector field.
        """

        vlm = torch.cat([self.image(image).flatten(2).transpose(1, 2), self.text(text)], dim=1)
        st = self.state(state).unsqueeze(1)
        time = tau.view(tau.shape[0], 1, 1).expand(-1, noisy_action.shape[1], 1)
        act = self.action(torch.cat([noisy_action, time], dim=-1))
        tokens = torch.cat([vlm, st, act], dim=1)
        mask = torch.zeros(tokens.shape[1], tokens.shape[1], device=tokens.device)
        mask[: vlm.shape[1], vlm.shape[1] :] = float("-inf")
        for block in self.blocks:
            tokens = block(tokens, mask)
        return self.out(tokens[:, -noisy_action.shape[1] :])


class Pi0FASTPolicy(nn.Module):
    """Compact π0-FAST autoregressive action-token policy."""

    def __init__(self, dim: int = 48, vocab: int = 256) -> None:
        """Initialize FAST policy.

        Parameters
        ----------
        dim:
            Token width.
        vocab:
            FAST action-token vocabulary.
        """

        super().__init__()
        self.image = nn.Conv2d(3, dim, 8, stride=8)
        self.text = nn.Embedding(128, dim)
        self.action_tokens = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([BlockwiseCausalSelfAttention(dim) for _ in range(2)])
        self.head = nn.Linear(dim, vocab)

    def forward(
        self, image: torch.Tensor, text: torch.Tensor, action_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Predict next FAST action-token logits.

        Parameters
        ----------
        image:
            Observation image.
        text:
            Prompt tokens.
        action_tokens:
            Previous FAST action tokens.

        Returns
        -------
        torch.Tensor
            Token logits.
        """

        tokens = torch.cat(
            [
                self.image(image).flatten(2).transpose(1, 2),
                self.text(text),
                self.action_tokens(action_tokens),
            ],
            dim=1,
        )
        mask = torch.triu(
            torch.full((tokens.shape[1], tokens.shape[1]), float("-inf"), device=tokens.device), 1
        )
        for block in self.blocks:
            tokens = block(tokens, mask)
        return self.head(tokens[:, -action_tokens.shape[1] :])


class Pi0ConcatInputWrapper(nn.Module):
    """Single-tensor input wrapper for the structured π0 VLA policy."""

    def __init__(self) -> None:
        """Initialize the wrapped flow-matching policy."""

        super().__init__()
        self.policy = Pi0FlowPolicy()

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        """Unpack image, text, state, action, and time from one tensor.

        Parameters
        ----------
        packed:
            Concatenated tensor with image, token, state, noisy action, and
            flow-time fields.

        Returns
        -------
        torch.Tensor
            Flow vectors for action chunks.
        """

        batch = packed.shape[0]
        cursor = 0
        image = packed[:, cursor : cursor + FLOW_IMAGE_VALUES].reshape(batch, 3, 32, 32)
        cursor += FLOW_IMAGE_VALUES
        text_raw = packed[:, cursor : cursor + FLOW_TEXT_VALUES]
        text = torch.remainder(torch.round(torch.abs(text_raw)), 128).long()
        cursor += FLOW_TEXT_VALUES
        state = packed[:, cursor : cursor + FLOW_STATE_VALUES]
        cursor += FLOW_STATE_VALUES
        noisy_action = packed[:, cursor : cursor + FLOW_ACTION_VALUES].reshape(
            batch, FLOW_ACTION_STEPS, 7
        )
        cursor += FLOW_ACTION_VALUES
        tau = torch.sigmoid(packed[:, cursor])
        return self.policy(image, text, state, noisy_action, tau)


class Pi0FASTConcatInputWrapper(nn.Module):
    """Single-tensor input wrapper for the π0-FAST action-token policy."""

    def __init__(self) -> None:
        """Initialize the wrapped autoregressive FAST policy."""

        super().__init__()
        self.policy = Pi0FASTPolicy()

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        """Unpack image, text tokens, and previous action tokens.

        Parameters
        ----------
        packed:
            Concatenated tensor with image, prompt-token, and FAST action-token
            fields.

        Returns
        -------
        torch.Tensor
            Next-token logits for FAST action tokens.
        """

        batch = packed.shape[0]
        cursor = 0
        image = packed[:, cursor : cursor + FLOW_IMAGE_VALUES].reshape(batch, 3, 32, 32)
        cursor += FLOW_IMAGE_VALUES
        text_raw = packed[:, cursor : cursor + FLOW_TEXT_VALUES]
        text = torch.remainder(torch.round(torch.abs(text_raw)), 128).long()
        cursor += FLOW_TEXT_VALUES
        action_raw = packed[:, cursor : cursor + FAST_ACTION_VALUES]
        action_tokens = torch.remainder(torch.round(torch.abs(action_raw)), 256).long()
        return self.policy(image, text, action_tokens)


def build() -> nn.Module:
    """Build π0 flow policy.

    Returns
    -------
    nn.Module
        π0 flow model.
    """

    return Pi0ConcatInputWrapper()


def build_fast() -> nn.Module:
    """Build π0-FAST autoregressive policy.

    Returns
    -------
    nn.Module
        π0-FAST model.
    """

    return Pi0FASTConcatInputWrapper()


def example_input() -> torch.Tensor:
    """Create π0 flow inputs.

    Returns
    -------
    torch.Tensor
        Single concatenated flow-policy input.
    """

    return torch.randn(1, FLOW_INPUT_VALUES)


def example_fast_input() -> torch.Tensor:
    """Create π0-FAST inputs.

    Returns
    -------
    torch.Tensor
        Single concatenated autoregressive VLA input.
    """

    return torch.randn(1, FAST_INPUT_VALUES)


MENAGERIE_ENTRIES = [
    ("openpi_pi0_jax", "build", "example_input", "2024", "E7"),
    ("Pi0_openpi", "build", "example_input", "2024", "E7"),
    ("Pi05_openpi", "build", "example_input", "2025", "E7"),
    ("openpi_pi0_fast_jax", "build_fast", "example_fast_input", "2025", "E7"),
]
