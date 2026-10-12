"""PROBE (to be replaced by the final assertions): per-call recompile counts."""

from __future__ import annotations

import warnings

import pytest
import torch
import torchvision
from torch import nn

import torchlens as tl
from torchlens.utils import _torch_compat

pytestmark = [pytest.mark.real_model, pytest.mark.real_class]

if not _torch_compat.HAS_SET_STANCE:
    pytest.skip("torch.compiler.set_stance requires torch >= 2.6", allow_module_level=True)


def _frames() -> int:
    counters = _torch_compat.get_dynamo_compile_counters(force_probe=True)
    assert counters is not None
    return int(counters["frames"]["total"])


def _resnet() -> tuple[nn.Module, torch.Tensor]:
    torch.manual_seed(0)
    return torchvision.models.resnet18(weights=None, num_classes=10).eval(), torch.randn(
        2, 3, 32, 32
    )


def _gpt2() -> tuple[nn.Module, torch.Tensor]:
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.GPT2Config(n_layer=2, n_embd=64, n_head=2, vocab_size=128, n_positions=32)
    model = transformers.GPT2LMHeadModel(cfg).eval()
    model.config.use_cache = False
    return model, torch.randint(0, 128, (2, 16))


def _call(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        out = m(x)
    return out.logits if hasattr(out, "logits") else out


@pytest.mark.parametrize("backend", ["inductor", "aot_eager", "eager"])
@pytest.mark.parametrize("name", ["resnet18", "gpt2"])
@pytest.mark.parametrize("target", ["compiled", "plain"])
def test_probe(name: str, backend: str, target: str) -> None:
    import time

    import torch._dynamo as dynamo

    t0 = time.time()
    model, x = _resnet() if name == "resnet18" else _gpt2()
    torch.compiler.reset()
    limit = getattr(dynamo.config, "recompile_limit", None)
    legacy = getattr(dynamo.config, "cache_size_limit", None)
    compiled = torch.compile(model, backend=backend)
    try:
        w1 = _call(compiled, x)
        w2 = _call(compiled, x)
    except Exception as exc:  # noqa: BLE001
        print(f"PROBE {name} {backend} {target} WARMFAIL {type(exc).__name__}: {str(exc)[:200]}")
        return
    warm_equal = torch.equal(w1, w2)
    tgt = compiled if target == "compiled" else model
    results = {}
    for cap in (1, 2):
        before = _frames()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            if name == "gpt2":
                tl.trace(tgt, x)
            else:
                tl.trace(tgt, x)
        during = _frames() - before
        per_call = []
        outs = []
        for _ in range(6):
            b = _frames()
            outs.append(_call(compiled, x))
            per_call.append(_frames() - b)
        results[cap] = (during, per_call, [bool(torch.equal(o, w1)) for o in outs])
    print(
        f"PROBE torch={torch.__version__} {name} {backend} {target} limit={limit} legacy={legacy} "
        f"warm_equal={warm_equal} results={results} secs={time.time() - t0:.1f}"
    )
