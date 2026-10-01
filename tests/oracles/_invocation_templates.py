"""Invocation-template schema + seed templates (purity/state skeleton).

"Templates, not compute, are the cost" (the panel's fact 11: a feared
30-180 s purity sweep ran in 0.7 s, but 10 of 12 doors REFUSED for want of
invocation templates). This module is the ONE template asset with three
consumers: the purity/state harness (build item 7, Wave 1), the option
witness registry (item 11), and the call-time deprecation gate (item 5).

Wave 0 ships the schema plus seed templates for the cheap top-level doors;
item 7 grows the set to the 12-18 the memo funds. Every template carries a
POSITIVE CONTROL (D7): a named check proving the measurement channel was
alive before any verdict is trusted -- the panel's own pickle probe used a
function-local fixture class, reported "False -> False", and would have
exonerated a real defect through a dead channel.

State contracts stay ``UNSET`` pending FORK-A (the default state contract of
``tl.trace`` on a stateful model is JMT's call, 2-1 lean PURE_OBSERVER);
the harness, registry, and templates are identical under both branches --
only the declared cell differs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class InvocationTemplate:
    """One door's real-invocation recipe.

    Parameters
    ----------
    template_id:
        Stable id (``TPL-...``).
    door:
        Dotted path of the public door.
    invoke:
        Zero-arg callable performing ONE real invocation and returning the
        door's product.
    positive_control:
        Zero-arg callable that raises ``AssertionError`` unless the
        invocation's measurement channel is alive (D7).
    state_contract:
        Declared purity contract; ``UNSET`` until FORK-A rules.
    """

    template_id: str
    door: str
    invoke: Callable[[], Any]
    positive_control: Callable[[], Any]
    state_contract: str


def _fixture_model() -> torch.nn.Module:
    """Build the shared MODULE-LEVEL fixture model.

    Module-level by design (CF-022): a function-local fixture class cannot
    be pickled, which silently kills the pickle postcondition channel.

    Returns
    -------
    torch.nn.Module
        A tiny two-layer eval-mode model with deterministic weights.
    """

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
    return model.eval()


def _fixture_input() -> torch.Tensor:
    """Return the deterministic fixture input.

    Returns
    -------
    torch.Tensor
        A fixed ``(2, 4)`` batch.
    """

    torch.manual_seed(1)
    return torch.randn(2, 4)


def _invoke_trace() -> Any:
    """Template body: one plain ``tl.trace`` capture."""

    import torchlens

    return torchlens.trace(_fixture_model(), _fixture_input())


def _invoke_record() -> Any:
    """Template body: one sparse ``tl.record`` capture."""

    import torchlens

    return torchlens.record(_fixture_model(), _fixture_input(), save=torchlens.func("relu"))


def _invoke_pluck() -> Any:
    """Template body: one ``tl.pluck`` read."""

    import torchlens

    return torchlens.pluck(_fixture_model(), _fixture_input(), "relu_1_2")


def _invoke_extract() -> Any:
    """Template body: one ``tl.extract`` read."""

    import torchlens

    return torchlens.extract(_fixture_model(), _fixture_input(), ["relu_1_2"])


def _invoke_aggregate() -> Any:
    """Template body: one ``tl.aggregate`` streaming-stats run.

    This is the call-time probe that settles the panel's fact-2 residual:
    the door either warns (deprecated) or does not (undeclared convenience).
    """

    import torchlens
    from torchlens import stats

    return torchlens.aggregate(
        _fixture_model(),
        [_fixture_input()],
        metrics={"relu_1_2": stats.Mean()},
    )


def _control_trace_product() -> None:
    """Positive control: the capture channel yields a nonempty layer set."""

    product = _invoke_trace()
    assert product.layer_labels, "trace fixture channel dead: no layers captured"


def _control_record_product() -> None:
    """Positive control: the recording channel matched the predicate."""

    product = _invoke_record()
    assert product is not None and type(product).__name__ == "Recording"


def _control_pluck_product() -> None:
    """Positive control: the plucked activation has the fixture shape."""

    out = _invoke_pluck()
    assert tuple(out.shape) == (2, 8), f"pluck channel dead or misaimed: {tuple(out.shape)}"


def _control_extract_product() -> None:
    """Positive control: extraction returns exactly the requested layer."""

    outs = _invoke_extract()
    assert set(outs) == {"relu_1_2"}, f"extract channel dead or misaimed: {set(outs)}"


def _control_aggregate_product() -> None:
    """Positive control: aggregation returns the requested metric."""

    result = _invoke_aggregate()
    assert result, "aggregate channel dead: empty result"


SEED_TEMPLATES: tuple[InvocationTemplate, ...] = (
    InvocationTemplate(
        "TPL-001", "torchlens.trace", _invoke_trace, _control_trace_product, "UNSET"
    ),
    InvocationTemplate(
        "TPL-002", "torchlens.record", _invoke_record, _control_record_product, "UNSET"
    ),
    InvocationTemplate(
        "TPL-003", "torchlens.pluck", _invoke_pluck, _control_pluck_product, "UNSET"
    ),
    InvocationTemplate(
        "TPL-004", "torchlens.extract", _invoke_extract, _control_extract_product, "UNSET"
    ),
    InvocationTemplate(
        "TPL-005", "torchlens.aggregate", _invoke_aggregate, _control_aggregate_product, "UNSET"
    ),
)
