"""The tensor search's log mute (``_utils/_quiet_property.py``, ``_utils/_log_mute.py``).

A property the search reads on an argument object may log; the search mutes
records at WARNING and below while it reads (as ``logging.disable(WARNING)``
did), without clearing any logger's level cache. These tests pin that a
property logging during a crawl stays silent, that the model's own logging in
forward is still emitted, that the user's logging state is exactly as found
after every crawl (a crawl that raises included, and a getter that calls
``logging.disable`` itself), and that a capture clears no level cache.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
import torch
from torch import Tensor, nn

import torchlens as tl
from torchlens._utils._quiet_property import _QuietPropertyLogs, read_attr_quietly

_CRAWL_LOGGER = "torchlens_test.crawl_property"
_FORWARD_LOGGER = "torchlens_test.model_forward"

#: Property reads and inner forward runs, counted outside the model (a capture may
#: rewind attributes a forward writes on its modules).
_COUNTS = {"reads": 0, "forwards": 0}


class _Records(logging.Handler):
    """A handler that keeps every record it is given."""

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def records() -> Any:
    """Handlers on the crawl and forward loggers, removed afterwards."""

    _COUNTS.update(reads=0, forwards=0)
    handlers = {}
    for name in (_CRAWL_LOGGER, _FORWARD_LOGGER):
        logger = logging.getLogger(name)
        handler = _Records()
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        handlers[name] = handler
    yield handlers
    for name, handler in handlers.items():
        logger = logging.getLogger(name)
        logger.removeHandler(handler)
        logger.setLevel(logging.NOTSET)


class _Chatty:
    """An argument object whose property logs when read (a deprecated-property notice)."""

    def __init__(self, x: Tensor) -> None:
        self.x = x

    @property
    def size_notice(self) -> int:
        _COUNTS["reads"] += 1
        logger = logging.getLogger(_CRAWL_LOGGER)
        logger.warning("size_notice is deprecated")
        logger.info("size_notice read")
        return 0


class _Inner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: Tensor, cache: _Chatty | None = None) -> Tensor:
        _COUNTS["forwards"] += 1
        logging.getLogger(_FORWARD_LOGGER).warning("inner forward ran")
        return self.lin(x) if cache is None else self.lin(x + cache.x)


class _Outer(nn.Module):
    def __init__(self, fail: bool = False) -> None:
        super().__init__()
        self.inner = _Inner()
        self.fail = fail

    def forward(self, x: Tensor) -> Tensor:
        y = self.inner(x, cache=_Chatty(x))
        if self.fail:
            raise RuntimeError("forward failed after the crawl")
        return self.inner(y)


def _logging_state() -> tuple[Any, ...]:
    """Everything of the user's logging state a crawl must leave as found."""

    manager = logging.root.manager
    loggers = [logging.root] + [
        logger for logger in manager.loggerDict.values() if isinstance(logger, logging.Logger)
    ]
    per_logger = tuple(
        (
            logger.name,
            logger.level,
            tuple(logger.handlers),
            tuple(logger.filters),
            logger.disabled,
            logger.propagate,
            tuple(logger.isEnabledFor(level) for level in (10, 20, 30, 40, 50)),
        )
        for logger in loggers
    )
    return manager.disable, logging.Logger.__dict__["isEnabledFor"], per_logger


def test_a_property_read_by_the_crawl_logs_nothing(records: Any) -> None:
    model = _Outer()
    tl.trace(model, torch.randn(2, 4))
    assert _COUNTS["reads"] > 0, "the module-entry search reads the property"
    assert records[_CRAWL_LOGGER].records == []


def test_the_models_own_logging_in_forward_is_still_emitted(records: Any) -> None:
    model = _Outer()
    tl.trace(model, torch.randn(2, 4))
    messages = [record.getMessage() for record in records[_FORWARD_LOGGER].records]
    assert _COUNTS["reads"] > 0
    assert _COUNTS["forwards"] > 0
    assert messages == ["inner forward ran"] * _COUNTS["forwards"]


def test_logging_state_is_as_found_after_a_trace(records: Any) -> None:
    before = _logging_state()
    tl.trace(_Outer(), torch.randn(2, 4))
    assert _logging_state() == before


def test_a_user_disable_below_warning_is_kept(records: Any) -> None:
    logging.disable(logging.INFO)
    try:
        before = _logging_state()
        model = _Outer()
        tl.trace(model, torch.randn(2, 4))
        assert _logging_state() == before
        assert records[_CRAWL_LOGGER].records == []
        assert len(records[_FORWARD_LOGGER].records) == _COUNTS["forwards"]
    finally:
        logging.disable(logging.NOTSET)


def test_a_trace_whose_forward_raises_after_a_crawl_restores_logging_state(records: Any) -> None:
    before = _logging_state()
    model = _Outer(fail=True)
    with pytest.raises(Exception) as raised:  # noqa: PT011 (capture may wrap the error)
        tl.trace(model, torch.randn(2, 4))
    chain: list[BaseException] = []
    error: BaseException | None = raised.value
    while error is not None and error not in chain:
        chain.append(error)
        error = error.__cause__ or error.__context__
    assert any("forward failed after the crawl" in str(e) for e in chain)
    assert _COUNTS["reads"] > 0
    assert records[_CRAWL_LOGGER].records == []
    assert _logging_state() == before


def test_a_crawl_that_raises_restores_logging_state(records: Any) -> None:
    holder = _Chatty(torch.zeros(1))
    crawl_logger = logging.getLogger(_CRAWL_LOGGER)
    before = _logging_state()
    with pytest.raises(RuntimeError, match="crawl failed"):
        with _QuietPropertyLogs():
            read_attr_quietly(holder, "size_notice")
            assert not crawl_logger.isEnabledFor(logging.WARNING)
            assert crawl_logger.isEnabledFor(logging.ERROR)
            raise RuntimeError("crawl failed")
    assert _logging_state() == before
    assert records[_CRAWL_LOGGER].records == []


class _Raising:
    @property
    def broken(self) -> int:
        logging.getLogger(_CRAWL_LOGGER).warning("about to fail")
        raise KeyError("broken")


def test_a_raising_property_outside_a_scope_restores_logging_state(records: Any) -> None:
    before = _logging_state()
    with pytest.raises(KeyError):
        read_attr_quietly(_Raising(), "broken")
    assert _logging_state() == before
    assert records[_CRAWL_LOGGER].records == []


class _Disabling:
    @property
    def disables(self) -> int:
        logging.disable(logging.CRITICAL)
        return 0


def test_a_getter_that_calls_logging_disable_is_undone(records: Any) -> None:
    error_logger = logging.getLogger(_FORWARD_LOGGER)
    assert error_logger.isEnabledFor(logging.ERROR)
    before = _logging_state()
    with _QuietPropertyLogs():
        read_attr_quietly(_Disabling(), "disables")
        assert not error_logger.isEnabledFor(logging.ERROR)  # cached False inside
    assert _logging_state() == before
    assert error_logger.isEnabledFor(logging.ERROR)


def test_a_capture_clears_no_level_cache(records: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    clears = []
    original = logging.Manager._clear_cache

    def counted(self: logging.Manager) -> None:
        clears.append(1)
        original(self)

    model = _Outer()
    x = torch.randn(2, 4)
    tl.trace(model, x)  # warm: first-capture setup may configure loggers
    _COUNTS["reads"] = 0
    monkeypatch.setattr(logging.Manager, "_clear_cache", counted)
    tl.trace(model, x)
    assert _COUNTS["reads"] > 0
    assert clears == []
