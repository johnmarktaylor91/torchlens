"""Anti-recurrence test for cert10 ``tl.swap_with`` string-label honesty fix.

``tl.swap_with(<string label>)`` was publicly documented as working (README,
docs/for-ai-agents.md, docs/intervention_api.md) but was dead code: the only
resolution path depended on ``hook.run_ctx["swap_sources"]``, a dict that no
execution path (live, replay, or rerun) ever populated. Every string-label
call therefore raised a confusing ``HookValueError`` deep inside hook
execution, only if/when the helper actually fired.

Implementing real string-label resolution (Route A) requires populating a
fire-time label -> tensor lookup as the forward pass progresses, which is a
runtime/replay concern (``torchlens/intervention/runtime.py``,
``torchlens/intervention/replay.py``), not something reachable from
``torchlens/intervention/helpers.py`` alone. So this fix takes the honesty
route (Route B): ``swap_with`` now raises a clear, early ``HookValueError``
the moment a string label is passed -- at spec-construction time, before any
trace/hook ever fires -- and the docs were corrected to stop claiming the
string form works.

This test locks in:
1. ``tl.swap_with("<label>")`` raises immediately, with a message that names
   the unsupported form and points at the supported alternatives.
2. The tensor and Op-like forms still work end-to-end in a real
   intervention (the only forms that were ever functional).
3. None of the shipped docs contain a live ``swap_with("...")`` /
   ``swap_with('...')`` invocation example -- the exact documented-but-dead
   pattern from the incident.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.errors import HookValueError

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DOC_FILES = (
    _REPO_ROOT / "README.md",
    _REPO_ROOT / "docs" / "for-ai-agents.md",
    _REPO_ROOT / "docs" / "intervention_api.md",
)


class _TwoLinear(nn.Module):
    """Tiny two-layer model with two distinguishable linear outputs."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(3, 3)
        self.b = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run both linears in sequence and return the second's out."""

        x = self.a(x)
        return self.b(x)


@pytest.mark.smoke
def test_swap_with_string_label_raises_immediately() -> None:
    """A bare string label must raise HookValueError at construction time.

    No trace, hook attachment, or forward pass should be required to
    surface the failure -- the label can never resolve, so the error must
    not wait for fire time.
    """

    with pytest.raises(HookValueError, match="not implemented"):
        tl.swap_with("linear_1_1")


def test_swap_with_string_label_error_names_alternatives() -> None:
    """The error message must point at the supported tensor / Op-like forms."""

    with pytest.raises(HookValueError) as excinfo:
        tl.swap_with("some_layer")

    message = str(excinfo.value)
    assert "torch.Tensor" in message
    assert "out" in message


def test_swap_with_tensor_form_still_works_in_real_intervention() -> None:
    """The only ever-functional forms (tensor / Op-like) must still work.

    Regression guard: the honesty fix must not collaterally break the
    tensor-argument path while gating the string-label path.
    """

    model = _TwoLinear()
    x = torch.randn(2, 3)

    # Select only the second linear's site by module address (`b`), since
    # live-time predicates match against raw/in-progress capture -- final
    # labels like "linear_2_2" are only resolved after the pass completes.
    replacement = torch.full((2, 3), 7.0)
    swapped = tl.trace(
        model,
        x,
        save=tl.func("linear"),
        intervene=tl.when(tl.in_module("b"), tl.swap_with(replacement)),
    )

    result = swapped["linear_2_2"].out
    assert torch.allclose(result, replacement)
    # The first (unswapped) linear must be untouched.
    assert not torch.allclose(swapped["linear_1_1"].out, replacement)

    # Op-like form: swap the second linear's out with the first linear's
    # already-captured out from a separate, prior trace.
    baseline = tl.trace(model, x, save=tl.func("linear"))
    first_layer_out = baseline["linear_1_1"].out

    op_swapped = tl.trace(
        model,
        x,
        save=tl.func("linear"),
        intervene=tl.when(tl.in_module("b"), tl.swap_with(baseline["linear_1_1"])),
    )
    assert torch.allclose(op_swapped["linear_2_2"].out, first_layer_out)


@pytest.mark.parametrize("doc_path", _DOC_FILES, ids=lambda p: p.name)
def test_docs_contain_no_live_string_label_swap_with_example(doc_path: Path) -> None:
    """No shipped doc may show a live ``swap_with("...")`` call example.

    This is the exact documented-but-dead pattern from the cert10 incident:
    a string-label ``swap_with`` example presented as working code. Table
    mentions of the bare function name/signature are fine; an actual quoted
    string argument to a ``swap_with(`` call is not.
    """

    assert doc_path.exists(), f"expected doc file missing: {doc_path}"
    text = doc_path.read_text(encoding="utf-8")

    forbidden_patterns = ('swap_with("', "swap_with('")
    for pattern in forbidden_patterns:
        assert pattern not in text, (
            f"{doc_path} still shows a live string-label swap_with(...) example "
            f"({pattern!r}); the string-label form is unimplemented and must not "
            "be documented as working."
        )
