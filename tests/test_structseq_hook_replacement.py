"""A live hook on a ``torch.return_types`` structseq output must land.

``torch.max(x, dim=...)`` returns a ``torch.return_types.max`` structseq: a
tuple subclass with named ``values`` / ``indices`` members but no ``_fields``.
The container walker records its children under ``NamedField`` paths, so the
hook results are keyed by ``NamedField("values")``. The output rebuild must
match those keys instead of treating the structseq as a positional tuple,
otherwise the scaled tensor and its fire record are dropped silently: the
recorded value is unscaled and the op carries no ``edits``.
"""

from __future__ import annotations

import torch

import torchlens as tl
from torchlens._ir.container import NamedField
from torchlens.backends.torch._ops_predicates import _replace_output_tensors_by_path


class _MaxModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = torch.max(x, dim=2)[0]
        return values.sum(-1)


def test_replace_output_tensors_by_path_matches_structseq_named_fields() -> None:
    out = torch.max(torch.arange(6.0).reshape(2, 3), dim=1)
    replacement = torch.full((2,), 7.0)
    rebuilt = _replace_output_tensors_by_path(out, {(NamedField("values"),): replacement})
    assert type(rebuilt) is type(out)
    assert torch.equal(rebuilt.values, replacement)
    assert torch.equal(rebuilt.indices, out.indices)


def test_live_scale_on_structseq_values_lands_with_an_edit_record() -> None:
    model = _MaxModel().eval()
    x = torch.arange(24.0).reshape(2, 4, 3)
    plain = tl.trace(model, x)
    (plain_max,) = [op for op in plain.ops if op.func_name == "max" and op.out.is_floating_point()]
    (plain_idx,) = [
        op for op in plain.ops if op.func_name == "max" and not op.out.is_floating_point()
    ]

    selector = tl.func("max") & ~tl.label(f"{plain_idx.raw_label}_raw")
    edited = tl.trace(
        model,
        x,
        intervene=tl.when(selector, tl.scale(0.5)),
        on_zero_fire="error",
        capture=tl.options.CaptureOptions(replayable=True),
    )
    edited_max = edited[plain_max.label]
    assert edited_max.is_edited
    assert edited_max.edits
    assert all(record.replaced for record in edited_max.edits)
    torch.testing.assert_close(edited_max.out, plain_max.out * 0.5)
    torch.testing.assert_close(edited.ops[-1].out, plain.ops[-1].out * 0.5)
    assert torch.equal(edited[plain_idx.label].out, plain_idx.out)
