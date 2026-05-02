"""Shared helpers for TorchLens Total Audit notebooks."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import nn


class TinyCNN(nn.Module):
    """Small deterministic convolutional model for image-shaped audits."""

    def __init__(self) -> None:
        """Initialize two convolution blocks and a classifier head."""

        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the CNN over one image batch.

        Parameters
        ----------
        x:
            Input tensor with shape ``(batch, 1, height, width)``.

        Returns
        -------
        torch.Tensor
            Logits with shape ``(batch, 2)``.
        """

        hidden = self.features(x).flatten(1)
        return self.classifier(hidden)


class TinyRecurrent(nn.Module):
    """Small deterministic recurrent model for sequence audits."""

    def __init__(self) -> None:
        """Initialize a one-layer RNN and classifier head."""

        super().__init__()
        self.rnn = nn.RNN(input_size=3, hidden_size=5, batch_first=True)
        self.head = nn.Linear(5, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the RNN and classify the final timestep.

        Parameters
        ----------
        x:
            Sequence tensor with shape ``(batch, steps, 3)``.

        Returns
        -------
        torch.Tensor
            Logits with shape ``(batch, 2)``.
        """

        sequence, _hidden = self.rnn(x)
        return self.head(sequence[:, -1])


class TinyDynamicModel(nn.Module):
    """Tiny model with data-dependent if/for/while control flow."""

    def __init__(self) -> None:
        """Initialize the dynamic-control-flow layers."""

        super().__init__()
        self.up = nn.Linear(4, 4)
        self.down = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run dynamic branches and loops over a small tensor.

        Parameters
        ----------
        x:
            Input tensor with shape ``(batch, 4)``.

        Returns
        -------
        torch.Tensor
            Logits with shape ``(batch, 2)``.
        """

        hidden = self.up(x) if x.mean() >= 0 else self.down(x)
        for _index in range(2):
            hidden = torch.relu(hidden)
        steps = 0
        while steps < 1:
            hidden = hidden + 0.01
            steps += 1
        return self.head(hidden)


class TinyBranchedModel(nn.Module):
    """Tiny model with two branches merged before the output head."""

    def __init__(self) -> None:
        """Initialize branch layers and merge head."""

        super().__init__()
        self.left = nn.Linear(4, 4)
        self.right = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run two branches and merge them additively.

        Parameters
        ----------
        x:
            Input tensor with shape ``(batch, 4)``.

        Returns
        -------
        torch.Tensor
            Logits with shape ``(batch, 2)``.
        """

        return self.head(torch.relu(self.left(x)) + torch.sigmoid(self.right(x)))


class TinyTransformer(nn.Module):
    """Small deterministic transformer-like model for recipe notebooks."""

    def __init__(
        self,
        *,
        vocab_size: int = 11,
        d_model: int = 8,
        n_heads: int = 2,
        max_len: int = 6,
    ) -> None:
        """Initialize embeddings, attention, MLP, and language-model head.

        Parameters
        ----------
        vocab_size:
            Number of token IDs accepted by the embedding table.
        d_model:
            Hidden width used by the block.
        n_heads:
            Attention head count.
        max_len:
            Maximum sequence length supported by positional embeddings.
        """

        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(max_len, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run one transformer-style block over token IDs.

        Parameters
        ----------
        tokens:
            Integer token IDs with shape ``(batch, seq_len)``.

        Returns
        -------
        torch.Tensor
            Per-position logits with shape ``(batch, seq_len, vocab_size)``.
        """

        positions = self.pos_embed[: tokens.shape[1]].unsqueeze(0)
        hidden = self.token_embed(tokens) + positions
        attended, _weights = self.attn(hidden, hidden, hidden, need_weights=False)
        hidden = self.ln_1(hidden + attended)
        hidden = self.ln_2(hidden + self.mlp(hidden))
        return self.head(hidden)


def tiny_model(seed: int = 0) -> nn.Module:
    """Return a deterministic three-layer MLP for audit notebooks.

    Parameters
    ----------
    seed:
        Torch RNG seed used before constructing the model.

    Returns
    -------
    nn.Module
        Three-layer MLP with deterministic initial weights.
    """

    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )


def tiny_cnn(seed: int = 0) -> TinyCNN:
    """Return a deterministic two-convolution CNN.

    Parameters
    ----------
    seed:
        Torch RNG seed used before constructing the model.

    Returns
    -------
    TinyCNN
        Evaluation-mode CNN with deterministic weights.
    """

    torch.manual_seed(seed)
    return TinyCNN().eval()


def tiny_recurrent(seed: int = 0) -> TinyRecurrent:
    """Return a deterministic small RNN.

    Parameters
    ----------
    seed:
        Torch RNG seed used before constructing the model.

    Returns
    -------
    TinyRecurrent
        Evaluation-mode recurrent model with deterministic weights.
    """

    torch.manual_seed(seed)
    return TinyRecurrent().eval()


def tiny_dynamic_model(seed: int = 0) -> TinyDynamicModel:
    """Return a deterministic model with dynamic control flow.

    Parameters
    ----------
    seed:
        Torch RNG seed used before constructing the model.

    Returns
    -------
    TinyDynamicModel
        Evaluation-mode dynamic model with deterministic weights.
    """

    torch.manual_seed(seed)
    return TinyDynamicModel().eval()


def tiny_branched_model(seed: int = 0) -> TinyBranchedModel:
    """Return a deterministic branched model.

    Parameters
    ----------
    seed:
        Torch RNG seed used before constructing the model.

    Returns
    -------
    TinyBranchedModel
        Evaluation-mode branched model with deterministic weights.
    """

    torch.manual_seed(seed)
    return TinyBranchedModel().eval()


def tiny_transformer(seed: int = 0) -> TinyTransformer:
    """Return a deterministic tiny transformer-like model.

    Parameters
    ----------
    seed:
        Torch RNG seed used before constructing the model.

    Returns
    -------
    TinyTransformer
        Evaluation-mode transformer-like module with deterministic weights.
    """

    torch.manual_seed(seed)
    return TinyTransformer().eval()


def make_clean_corrupt_pair(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a deterministic clean/corrupt activation pair.

    Parameters
    ----------
    seed:
        Torch RNG seed used to create the tensors.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Clean and corrupt inputs with shape ``(1, 4)``.
    """

    torch.manual_seed(seed)
    clean = torch.randn(1, 4)
    corrupt = clean.flip(dims=(1,))
    return clean, corrupt


def pretty_print_fields(obj: object, field_names: list[str] | tuple[str, ...]) -> None:
    """Print selected object fields in a compact deterministic form.

    Parameters
    ----------
    obj:
        Object whose fields should be displayed.
    field_names:
        Attribute names to read from ``obj``.
    """

    for field_name in field_names:
        value = getattr(obj, field_name, "<missing>")
        print(f"{field_name}: {value!r}")


def inline_show(label: str, value: object) -> None:
    """Display a small inline value without requiring notebook display extras.

    Parameters
    ----------
    label:
        Label printed before the value.
    value:
        Value to display.
    """

    print(f"{label}: {value}")


def summarize_value(value: object) -> str:
    """Return a compact, deterministic summary for notebook audit output.

    Parameters
    ----------
    value:
        Object to summarize.

    Returns
    -------
    str
        Short human-readable summary.
    """

    if isinstance(value, torch.Tensor):
        return (
            f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device.type})"
        )
    if isinstance(value, Mapping):
        return f"{type(value).__name__}(len={len(value)})"
    if isinstance(value, (list, tuple, set, frozenset)):
        return f"{type(value).__name__}(len={len(value)})"
    if inspect.ismodule(value):
        return f"module {value.__name__}"
    if inspect.isclass(value):
        return f"class {value.__name__}"
    if callable(value):
        return f"callable {getattr(value, '__name__', type(value).__name__)}"
    if (
        isinstance(value, str)
        and len(value) >= 32
        and all(character in "0123456789abcdefABCDEF" for character in value)
    ):
        return f"<hex-string len={len(value)}>"
    text = repr(value)
    if len(text) > 120:
        text = text[:117] + "..."
    return text


def audit_touch_items(
    label: str,
    items: list[str] | tuple[str, ...],
    namespace: Mapping[str, object],
    *,
    examples: Mapping[str, Callable[[], object]] | None = None,
    max_samples: int = 8,
) -> None:
    """Resolve and touch coverage items against live TorchLens objects.

    Parameters
    ----------
    label:
        Label printed for the demonstrated cluster.
    items:
        Coverage item names such as ``"tl.ModelLog.layer_list"``.
    namespace:
        Mapping containing ``tl`` and representative instances keyed by class name.
    examples:
        Optional exact-item call examples for methods that need arguments.
    max_samples:
        Maximum number of compact item summaries to print.
    """

    examples = examples or {}
    touched: list[str] = []
    skipped: list[str] = []
    samples: list[str] = []

    for item in items:
        try:
            if item in examples:
                value = examples[item]()
            else:
                value = _resolve_audit_item(item, namespace)
            touched.append(item)
            if len(samples) < max_samples:
                samples.append(f"{item} -> {summarize_value(value)}")
        except Exception as exc:  # pragma: no cover - notebooks print skipped items.
            skipped.append(f"{item} ({type(exc).__name__}: {str(exc).splitlines()[0][:80]})")

    print(f"{label}: touched {len(touched)}/{len(items)}")
    for sample in samples:
        print(f"  {sample}")
    for item in skipped[:max_samples]:
        print(f"  # skipped: {item}")


def _resolve_audit_item(item: str, namespace: Mapping[str, object]) -> object:
    """Resolve one coverage item against a TorchLens namespace.

    Parameters
    ----------
    item:
        Coverage item name.
    namespace:
        Mapping containing ``tl`` and representative instances.

    Returns
    -------
    object
        Resolved value.
    """

    if not item.startswith("tl."):
        raise ValueError(f"Unsupported audit item {item!r}")

    tl_module = namespace["tl"]
    parts = item.split(".")[1:]
    if len(parts) == 1:
        return getattr(tl_module, parts[0])

    owner_name, member_name = parts[0], parts[1]
    try:
        owner = getattr(tl_module, owner_name)
    except AttributeError:
        owner = getattr(tl_module.types, owner_name)
    instance = namespace.get(owner_name)
    if instance is not None:
        try:
            return getattr(instance, member_name)
        except AttributeError:
            pass
    return getattr(owner, member_name)
