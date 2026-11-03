"""A binding refuses label and graph selectors at bind time, naming a spelling that runs.

A binding runs the model capture-free, so its ops carry only the raw labels
minted during the forward. Final, short and pass-qualified labels exist only
after a capture, and ``followed_by``/``preceded_by``/``input_at`` read a
recorded graph. Such a rule used to fail at the first op of the first call
(``bind_rule_runtime_error``) or never fire; it now refuses when bound, before
any forward runs, and the remedy it prints runs on the same model.
"""

from __future__ import annotations

import re

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.errors import BindingPreflightError


def _double(out: torch.Tensor, *, ctx: object) -> torch.Tensor:
    """Double the edited op's output."""

    return out * 2.0


class _Counted(nn.Sequential):
    """A small CNN that counts its forwards."""

    def __init__(self) -> None:
        """Build conv, relu, conv, tanh."""

        super().__init__(nn.Conv2d(3, 4, 3), nn.ReLU(), nn.Conv2d(4, 4, 3), nn.Tanh())
        self.forwards = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Count the call, then run the layers."""

        self.forwards += 1
        return super().forward(x)


@pytest.fixture()
def cnn() -> tuple[_Counted, torch.Tensor]:
    """A seeded eval-mode CNN plus one fixed input."""

    torch.manual_seed(0)
    return _Counted().eval(), torch.randn(1, 3, 8, 8)


REFUSED = {
    "label_final": lambda: tl.label("relu_1_2"),
    "short_label": lambda: tl.label("relu_1"),
    "label_pass": lambda: tl.label("conv2d_2_3:1"),
    "label_short_pass": lambda: tl.label("relu_1:1"),
    "label_unknown": lambda: tl.label("nonexistent"),
    "contains_final": lambda: tl.label_contains("relu_1_2"),
    "regex_final": lambda: tl.label_matches("relu_1_2"),
    "negated_label": lambda: ~tl.label("relu_1_2"),
    "label_and_func": lambda: tl.func("relu") & tl.label("relu_1_2"),
    "label_or_func": lambda: tl.label("tanh_1_4") | tl.func("relu"),
    "followed_by": lambda: tl.followed_by(tl.func("relu")),
    "preceded_by": lambda: tl.preceded_by(tl.func("relu")),
    "input_at": lambda: tl.model_input(0),
}


@pytest.mark.parametrize("case", sorted(REFUSED))
def test_bind_refuses_post_capture_selector_before_any_forward(cnn, case: str) -> None:
    """Every post-capture term refuses at bind with the user's selector and rule in the message."""

    model, _x = cnn
    selector = REFUSED[case]()
    spec = tl.when(selector, _double)
    with pytest.raises(BindingPreflightError) as excinfo:
        spec.bind(model)
    err = excinfo.value
    assert err.fields["code"] == "bind_rule_unsupported"
    message = str(err)
    assert spec.rules[0].rule_id in message
    assert "can never match" in message
    assert "tl.func(" in err.fields["remedy"]
    assert "known-gotchas.md" in err.fields["remedy"]
    assert model.forwards == 0


@pytest.mark.parametrize("label", ["tanh_1"])
def test_printed_remedy_runs_and_edits_the_labelled_op(cnn, label: str) -> None:
    """The selector copied out of the remedy binds, fires, and edits the op the label names."""

    model, x = cnn
    with pytest.raises(BindingPreflightError) as excinfo:
        tl.when(tl.label(label), _double).bind(model)
    printed = re.search(r"tl\.func\('[A-Za-z0-9_]+'\)", excinfo.value.fields["remedy"])
    assert printed is not None
    selector = eval(printed.group(0), {"tl": tl})  # noqa: S307 - the refusal's own text
    bound = tl.when(selector, _double).bind(model)
    bound(x)
    op_type = printed.group(0)[len("tl.func('") : -2]
    trace = tl.trace(model, x)
    assert trace[label.split(":")[0]].func_name == op_type
    same_type = [layer for layer in trace._layer_list if layer.func_name == op_type]
    assert bound.last_report.num_fires == len(same_type) >= 1


@pytest.mark.parametrize(
    "selector",
    [
        pytest.param(lambda: tl.label_matches("^relu"), id="regex_plain"),
        pytest.param(lambda: tl.func("relu") & tl.in_module("1"), id="func_in_module"),
    ],
)
def test_live_matchable_label_terms_still_bind_and_fire(cnn, selector) -> None:
    """Raw labels and plain substrings match live contexts and keep working."""

    model, x = cnn
    bound = tl.when(selector(), _double).bind(model)
    bound(x)
    assert bound.last_report.num_fires == 1
