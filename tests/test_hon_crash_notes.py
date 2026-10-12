"""U30.2: every crash-note site goes through the guarded helper.

``BaseException.add_note`` exists from Python 3.11; on 3.10 the six crash-note
sites used a bare ``exc.add_note(...)`` inside a blanket suppress, so the note
silently never appeared. Each site now calls
``torchlens.utils._exception_notes.annotate_guarded``: a note on 3.11+, a
``RuntimeWarning`` with the same text on 3.10. These tests read either channel
and run in both, so the pinned 3.10 environment exercises the fallback.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.partial as tl_partial
from torchlens.errors import ScalarEscapeWarning
from torchlens.utils._exception_notes import annotate_guarded


class _Boom(nn.Module):
    """Runs one op, then raises from the forward."""

    def __init__(self, error: type[Exception] = RuntimeError) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.error = error

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.lin(x)
        raise self.error("boom")


class _ScalarThenBoom(nn.Module):
    """Reads a captured tensor as a Python scalar, then raises."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.lin(x)
        _ = float(y.sum())
        raise RuntimeError("boom")


class _NaNModel(nn.Module):
    """Produces a NaN so ``raise_on_nan=True`` stops the capture."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(self.lin(x) - 1e6)


class _FrozenError(RuntimeError):
    """An exception that rejects attribute assignment (partial_log included)."""

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"{name} is read-only on this exception")


def _crash_texts(
    run: Callable[[], Any], expected: type[BaseException], *filters: tuple[str, Any]
) -> tuple[BaseException, list[str]]:
    """Run a failing capture; return the exception and its note/warning texts."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for message, category in filters:
            warnings.filterwarnings("error", message=message, category=category)
        with pytest.raises(expected) as info:
            run()
    texts = list(getattr(info.value, "__notes__", []))
    texts += [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    return info.value, texts


def test_helper_notes_or_warns_by_python_version() -> None:
    """The shared helper adds a note where notes exist, else warns the same text."""

    exc = RuntimeError("x")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        annotate_guarded(exc, "the note text")
    if hasattr(exc, "add_note"):
        assert exc.__notes__ == ["the note text"]
        assert not caught
    else:
        assert [str(w.message) for w in caught] == ["the note text"]
        assert caught[0].category is RuntimeWarning


def test_success_note_reaches_the_user_and_its_remedy_runs() -> None:
    """Site backend.py partial attached: the note names the recovery that works."""

    exc, texts = _crash_texts(lambda: tl.trace(_Boom(), torch.randn(2, 4)), RuntimeError)
    assert any("exc.partial_log" in text for text in texts), texts
    recovered = tl_partial.from_failed_capture(exc)
    assert recovered is not None


def test_construction_failure_note_appears(monkeypatch: pytest.MonkeyPatch) -> None:
    """Site backend.py construction failed: the note survives on 3.10 too."""

    from torchlens.partial import PartialTrace

    def _broken(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("planted construction failure")

    monkeypatch.setattr(PartialTrace, "from_trace", _broken)
    _, texts = _crash_texts(lambda: tl.trace(_Boom(), torch.randn(2, 4)), RuntimeError)
    assert any("partial-trace construction also failed" in text for text in texts), texts


def test_suppressed_advisory_note_appears(monkeypatch: pytest.MonkeyPatch) -> None:
    """Site backend.py advisory suppressed: a filter raising the advisory leaves a note."""

    from torchlens.partial import PartialTrace

    def _broken(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("planted construction failure")

    monkeypatch.setattr(PartialTrace, "from_trace", _broken)
    _, texts = _crash_texts(
        lambda: tl.trace(_Boom(), torch.randn(2, 4)),
        RuntimeError,
        ("TorchLens could not construct partial-trace recovery", RuntimeWarning),
    )
    assert any("TorchLens advisory suppressed" in text for text in texts), texts


def test_attachment_failure_note_appears_and_its_remedy_runs() -> None:
    """Site backend.py attachment refused: the registry note and its recovery work."""

    exc, texts = _crash_texts(
        lambda: tl.trace(_Boom(_FrozenError), torch.randn(2, 4)), _FrozenError
    )
    assert any("exception-identity registry" in text for text in texts), texts
    assert tl_partial.from_failed_capture(exc) is not None


def test_scalar_escape_advisory_note_appears() -> None:
    """Site _completeness_storage: a raising escape filter leaves a note instead."""

    _, texts = _crash_texts(
        lambda: tl.trace(_ScalarThenBoom(), torch.randn(2, 4)),
        RuntimeError,
        (".*", ScalarEscapeWarning),
    )
    assert any("scalar-escape advisory suppressed" in text for text in texts), texts


def test_nonfinite_prefix_failure_note_appears(monkeypatch: pytest.MonkeyPatch) -> None:
    """Site capture/_nonfinite_prefix: a failed prefix finalization leaves a note."""

    import torchlens.capture.trace as capture_trace

    def _broken(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("planted prefix failure")

    monkeypatch.setattr(capture_trace, "_extract_and_mark_outputs", _broken)
    _, texts = _crash_texts(
        lambda: tl.trace(
            _NaNModel(),
            torch.randn(2, 4),
            capture=tl.options.CaptureOptions(raise_on_nan=True),
        ),
        Exception,
    )
    assert any("prefix finalization for public NaN labels failed" in text for text in texts), texts
