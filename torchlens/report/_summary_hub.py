"""Hub-name input form for ``tl.summary`` (spelling DOCUMENTED-UNSTABLE).

``tl.summary("gpt2", input_size=(1, 8))`` fetches the model's CONFIG only,
builds the architecture on the meta device (no weights are downloaded or
allocated), and hands the all-meta model to the existing weights-free
summary path, which needs input evidence and renders under the hypothesis
banner. ``trust_remote_code`` is never set: only architectures that ship
inside ``transformers`` build.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def build_hub_meta_model(name: str) -> nn.Module:
    """Build a Hugging Face Hub model from its config on the meta device.

    Parameters
    ----------
    name:
        A Hub model id (``"gpt2"``) or a local directory holding a
        ``config.json``.

    Returns
    -------
    nn.Module
        The architecture with every parameter and buffer on ``meta``, in
        eval mode.

    Raises
    ------
    ImportError
        If ``transformers`` is not installed.
    """

    try:
        import transformers
    except ImportError as exc:
        raise ImportError(
            "tl.summary(<hub model name>) requires the `transformers` package. "
            "Install with `pip install transformers` (or `pip install torchlens[hf]`), "
            "or build the model yourself on the meta device and pass the module."
        ) from exc

    config = transformers.AutoConfig.from_pretrained(name)
    model_class = _architecture_class(transformers, config)
    with torch.device("meta"):
        model = model_class(config)
    return model.eval()


def _architecture_class(transformers: Any, config: Any) -> Any:
    """Return the config's declared architecture class, else ``AutoModel``.

    The declared class (``GPT2LMHeadModel`` for ``"gpt2"``) keeps the task
    head the checkpoint ships; a name ``transformers`` does not export falls
    back to the base ``AutoModel`` (constructed through ``from_config``).
    """

    for architecture in getattr(config, "architectures", None) or ():
        candidate = getattr(transformers, str(architecture), None)
        if isinstance(candidate, type) and issubclass(candidate, nn.Module):
            return candidate
    return _AutoModelFromConfig(transformers.AutoModel)


class _AutoModelFromConfig:
    """Adapter so the base ``AutoModel`` builds through ``from_config``."""

    def __init__(self, auto_model: Any) -> None:
        self._auto_model = auto_model

    def __call__(self, config: Any) -> nn.Module:
        return self._auto_model.from_config(config)
