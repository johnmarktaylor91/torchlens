"""U32.3: the post-capture recompile burst of a compiled real model is bounded, then steady.

The coexistence contract for ``torch.compile`` (torch >= 2.6) is not "at most one
recompile". Measured on real architectures (torch 2.13.0 CPU, ``aot_eager``):

* ZERO compiles happen during a capture (the ``force_eager`` stance);
* the first compiled call after a capture can recompile a burst of frames when
  the model has Dynamo graph breaks around TorchLens's wrapper install/uninstall
  (tiny GPT-2: 6 frames; ``resnet18``, a single fully-captured graph: 0);
* the burst lands entirely on that first call, stays within Dynamo's recompile
  limit, and every later call recompiles ZERO frames (steady state);
* the post-capture outputs are bitwise-equal to the warm compiled outputs;
* a second capture is followed by its own burst of the same size, so the honest
  statement is "a bounded burst after each capture, then steady state".

The bound is Dynamo's own ceiling (``recompile_limit``; ``cache_size_limit`` on
older torch), found by feature check, never by parsing the torch version.

Backend: ``aot_eager``. Dynamo guards and recompiles are what is measured, and
that backend reproduces the same frame counts as the default inductor backend
on CPU (6 frames after the second capture; 6-7 after the first) in about a
tenth of the time, with no host C++ toolchain requirement.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import pytest
import torch
import torchvision
from torch import nn

import torchlens as tl
from torchlens.utils import _torch_compat

pytestmark = [pytest.mark.heavy, pytest.mark.real_model, pytest.mark.real_class]

# Module-level skip: set_stance is the capability the zero-compile leg needs, and
# the skip must run before any import-time torch.compile (see
# tests/test_compile_set_stance.py).
if not _torch_compat.HAS_SET_STANCE:
    pytest.skip(
        "torch.compiler.set_stance requires torch >= 2.6",
        allow_module_level=True,
    )

_BACKEND = "aot_eager"
_CALLS_AFTER_CAPTURE = 6
_CAPTURES = 2


def _frames_compiled() -> int:
    """Return Dynamo's cumulative frame-compilation count (recompiles included)."""

    counters = _torch_compat.get_dynamo_compile_counters(force_probe=True)
    assert counters is not None, "torch >= 2.6 is expected to expose Dynamo counters"
    return int(counters["frames"]["total"])


def _recompile_limit() -> int:
    """Return Dynamo's per-code-object recompile ceiling by feature check."""

    import torch._dynamo as dynamo

    for name in ("recompile_limit", "cache_size_limit"):
        if hasattr(dynamo.config, name):
            return int(getattr(dynamo.config, name))
    pytest.skip("this torch exposes no Dynamo recompile limit to bound against")


def _build_resnet18() -> tuple[nn.Module, torch.Tensor]:
    """Build a real torchvision resnet18 (random weights, no download)."""

    torch.manual_seed(0)
    model = torchvision.models.resnet18(weights=None, num_classes=10).eval()
    return model, torch.randn(2, 3, 32, 32)


def _build_tiny_gpt2() -> tuple[nn.Module, torch.Tensor]:
    """Build a tiny-config Hugging Face GPT-2 (random weights, no download)."""

    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.GPT2Config(n_layer=2, n_embd=64, n_head=2, vocab_size=128, n_positions=32)
    model = transformers.GPT2LMHeadModel(config).eval()
    model.config.use_cache = False
    return model, torch.randint(0, 128, (2, 16))


def _forward(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run a no-grad forward and return the logits tensor."""

    with torch.no_grad():
        out = model(x)
    return out.logits if hasattr(out, "logits") else out


@pytest.mark.parametrize("build", [_build_resnet18, _build_tiny_gpt2], ids=["resnet18", "gpt2"])
def test_post_capture_recompile_burst_is_bounded_then_steady(
    build: Callable[[], tuple[nn.Module, torch.Tensor]],
) -> None:
    """Zero compiles in capture; a bounded burst on the first call after; then none."""

    model, x = build()
    torch.compiler.reset()
    compiled = torch.compile(model, backend=_BACKEND)
    warm = _forward(compiled, x)
    assert torch.equal(warm, _forward(compiled, x)), "the warm artifact must be deterministic"
    limit = _recompile_limit()

    bursts: list[int] = []
    for _ in range(_CAPTURES):
        before = _frames_compiled()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            trace = tl.trace(compiled, x)
        assert _frames_compiled() - before == 0, "zero compiles during the stance capture"
        assert len(trace.layer_labels) > 0

        per_call: list[int] = []
        for _call in range(_CALLS_AFTER_CAPTURE):
            call_start = _frames_compiled()
            post = _forward(compiled, x)
            per_call.append(_frames_compiled() - call_start)
            assert torch.equal(post, warm), "post-capture output must equal the warm output"

        burst = per_call[0]
        assert burst <= limit, f"burst {burst} exceeds Dynamo's recompile limit {limit}"
        assert per_call[1:] == [0] * (_CALLS_AFTER_CAPTURE - 1), (
            f"steady state expected after the burst, got per-call counts {per_call}"
        )
        bursts.append(burst)

    # Measured, not assumed: every capture is followed by its own burst of the
    # same size (resnet18 0 and 0; gpt2 6 then 6, the first can be 7 on inductor).
    print(f"post-capture bursts per capture: {bursts}")
