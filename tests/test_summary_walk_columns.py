"""Walk batch 1 summary additions: optional columns, tuple shapes, footer extras.

Rulings (round-2 walk, 2026-10-11): every member shape of a tuple-returning
module (U02.1); optional input / kernel_size / groups / params_trainable
columns and a params-size line (U01.4, U02.3, U06.7); a ``mode`` column over
``module.training`` (U03.5); ``flops_pct`` beside ``params_pct`` (U04.3);
the full-square attention disclosure on every attention FLOP line (U04.2);
raw and effective nonzero counts on pruned models (U01.2); the optimizer
state footer (U06.2). The attention note also rides the profile footer, the
explain FLOPs line and graph-label FLOP rows.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import prune

import torchlens as tl
from torchlens.capture.flops import ATTENTION_FULL_SQUARE_NOTE
from torchlens.report._summary_render import ATTENTION_MARK


class _Recurrent(nn.Module):
    """An LSTM whose row must show output, h_n and c_n."""

    def __init__(self) -> None:
        """One LSTM and a head."""

        super().__init__()
        self.lstm = nn.LSTM(4, 6, batch_first=True)
        self.head = nn.Linear(6, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """lstm -> last step -> head."""

        out, (_h, _c) = self.lstm(x)
        return self.head(out[:, -1])


class _Convs(nn.Module):
    """A dense and a depthwise conv, a frozen eval-mode head."""

    def __init__(self) -> None:
        """Two convs and a Linear head."""

        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3)
        self.dw = nn.Conv2d(8, 8, 3, groups=8)
        self.head = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """conv -> depthwise -> pool -> head."""

        return self.head(self.dw(self.conv(x)).mean(dim=(2, 3)))


class _Attention(nn.Module):
    """A projection feeding causal SDPA."""

    def __init__(self) -> None:
        """One projection."""

        super().__init__()
        self.proj = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """proj -> causal scaled dot-product attention."""

        q = self.proj(x).unsqueeze(1)
        return nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)


def _frozen_eval_head() -> _Convs:
    """The conv model with its head frozen and in eval mode."""

    torch.manual_seed(0)
    model = _Convs()
    model.head.eval()
    for param in model.head.parameters():
        param.requires_grad_(False)
    return model


def _body_line(text: str, prefix: str) -> str:
    """The first table line starting with ``prefix``."""

    return next(line for line in text.splitlines() if line.startswith(prefix))


def test_tuple_output_row_prints_every_member_shape() -> None:
    """U02.1: an LSTM row shows output, h_n and c_n in return order."""

    torch.manual_seed(0)
    report = tl.summary(_Recurrent(), torch.randn(2, 5, 4))
    row = next(row for row in report._rebuilt.visible_rows if row.name == "lstm")
    assert row.output_shapes == ((2, 5, 6), (1, 2, 6), (1, 2, 6))
    assert row.output_shape == (2, 5, 6)
    assert "(2, 5, 6), (1, 2, 6), (1, 2, 6)" in _body_line(str(report), "lstm (LSTM)")


def test_optional_columns_render_their_facts() -> None:
    """U01.4/U02.3/U03.5/U04.3: input, kernel, groups, trainable, mode, flops_pct."""

    model = _frozen_eval_head()
    trace = tl.trace(model, torch.randn(1, 3, 8, 8))
    try:
        report = trace.summary(
            columns=[
                "+input",
                "+kernel_size",
                "+groups",
                "+params_trainable",
                "+flops_pct",
                "+mode",
            ]
        )
    finally:
        trace.cleanup()
    header = str(report).splitlines()[2].split()
    assert header[:6] == ["name", "(type)", "input", "output", "kernel", "groups"]
    rows = {row.name: row for row in report._rebuilt.visible_rows}
    assert rows["conv"].input_shapes == ((1, 3, 8, 8),)
    assert (rows["conv"].kernel_size, rows["conv"].groups) == ((3, 3), 1)
    assert (rows["dw"].kernel_size, rows["dw"].groups) == ((3, 3), 8)
    assert rows["head"].kernel_size is None and rows["head"].groups is None
    assert rows["conv"].params_trainable == 224
    assert rows["head"].params_trainable == 0
    assert (rows["conv"].mode, rows["head"].mode) == ("train", "eval")
    dw_line = _body_line(str(report), "dw (Conv2d)")
    flops_total = report.totals.flops_forward_fma2
    assert f"{100.0 * rows['dw'].flops_display / flops_total:.1f}%" in dw_line
    assert " train " in dw_line  # mode, before the auto-shown requires_grad column


def test_mode_mixed_on_a_row_whose_modules_disagree() -> None:
    """A depth-1 container row reads ``mixed`` when its children disagree."""

    torch.manual_seed(0)
    outer = nn.Sequential(nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4)))
    outer[0][1].eval()
    report = tl.trace(outer, torch.randn(1, 4)).summary(depth=1, columns=["+mode"])
    assert report._rebuilt.visible_rows[0].mode == "mixed"


def test_params_size_and_raw_totals() -> None:
    """U01.4/U06.7: the Params footer names the unique parameters' storage."""

    model = _Convs().to(torch.float64)
    report = tl.summary(model, torch.randn(1, 3, 8, 8, dtype=torch.float64))
    expected = sum(param.numel() * 8 for param in model.parameters())
    assert report.totals.params_bytes == expected
    assert " | size " in _body_line(str(report), "Params")


def test_pruned_model_shows_raw_and_effective_nonzero_counts() -> None:
    """U01.2: prune's weight_orig + weight_mask pair is named with both counts."""

    model = _Convs()
    prune.l1_unstructured(model.conv, "weight", amount=0.5)
    masked = int((model.conv.weight_mask == 0).sum())
    report = tl.summary(model, torch.randn(1, 3, 8, 8), columns=["+kernel_size"])
    raw = sum(param.numel() for param in model.parameters())
    assert report.totals.params_total == raw
    assert report.totals.params_effective == raw - masked
    params_line = _body_line(str(report), "Params")
    assert (
        f"pruned: {raw - masked:,} effective nonzero of {raw:,} raw "
        f"({masked:,} masked in conv.weight)" in params_line
    )
    conv_row = next(row for row in report._rebuilt.visible_rows if row.name == "conv")
    assert conv_row.pruned and conv_row.kernel_size == (3, 3)
    assert "conv (Conv2d) [pruned]" in str(report)


def test_optimizer_state_footer_before_and_after_a_step() -> None:
    """U06.2: 'optimizer state N params, X' from the live optimizer state."""

    torch.manual_seed(0)
    model = _Convs()
    opt = torch.optim.Adam(model.parameters())
    x = torch.randn(1, 3, 8, 8)
    options = tl.options.CaptureOptions(optimizer=opt)
    before = tl.trace(model, x, capture=options).summary()
    assert "optimizer state 0 params (allocated at the first step())" in str(before)
    model(x).sum().backward()
    opt.step()
    after = tl.trace(model, x, capture=options).summary()
    state = [value for entry in opt.state.values() for value in entry.values()]
    numel = sum(value.numel() for value in state)
    assert after.totals.optimizer_state_params == numel
    assert after.totals.optimizer_state_bytes == sum(
        value.numel() * value.element_size() for value in state
    )
    assert f"optimizer state {numel:,} params, " in _body_line(str(after), "Params")
    plain = tl.trace(model, x).summary()
    assert "optimizer state" not in str(plain)
    assert plain.totals.optimizer_state_params is None


def test_attention_flop_lines_disclose_the_full_square() -> None:
    """U04.2 (b): every attention FLOP cell is marked and the footer explains."""

    torch.manual_seed(0)
    model = _Attention()
    x = torch.randn(2, 5, 8)
    report = tl.summary(model, x, columns=["+flops_pct"])
    sdpa = next(row for row in report._rebuilt.visible_rows if row.attention_full_square)
    line = _body_line(str(report), sdpa.name)
    assert line.count(ATTENTION_MARK) == 2  # the flops and flops_pct cells
    assert "attention masks and is_causal are not discounted" in str(report)
    proj = _body_line(str(report), "proj (Linear)")
    assert ATTENTION_MARK not in proj
    flops = tl.report.flops(model, x)
    assert flops.attention_full_square_ops == 1
    assert "full query x key square" in str(flops)


def test_no_attention_legend_without_a_flop_column() -> None:
    """The legend rides only beside a printed FLOP cell."""

    report = tl.summary(_Attention(), torch.randn(2, 5, 8), columns=["name", "params"])
    assert "full query x key square" not in str(report)


def _attention_trace() -> tl.Trace:
    """A captured causal-SDPA model."""

    torch.manual_seed(0)
    return tl.trace(_Attention(), torch.randn(2, 5, 8))


def test_profile_footer_discloses_the_full_square() -> None:
    """U04.2 (b): the profile footer carries the attention note."""

    profile = _attention_trace().profile()
    assert profile.attention_full_square_ops == 1
    assert ATTENTION_FULL_SQUARE_NOTE in repr(profile)


def test_profile_footer_has_no_note_without_attention() -> None:
    """A model with no attention op prints no attention note."""

    trace = tl.trace(_frozen_eval_head(), torch.randn(1, 3, 8, 8))
    profile = trace.profile()
    assert profile.attention_full_square_ops == 0
    assert ATTENTION_FULL_SQUARE_NOTE not in repr(profile)


def test_explain_flops_line_discloses_the_full_square() -> None:
    """U04.2 (b): the explain FLOPs line carries the attention note."""

    text = _attention_trace().explain()
    flops_line = next(line for line in text.splitlines() if "Forward FLOPs" in line)
    assert ATTENTION_FULL_SQUARE_NOTE in flops_line


def test_graph_label_flops_rows_carry_the_note_on_attention_nodes_only() -> None:
    """U04.2 (b): an attention node's flops label field and flops overlay add the note."""

    from torchlens.visualization._render_nodes import compute_default_node_lines

    trace = _attention_trace()
    layers = list(trace._layer_logs.values())
    sdpa = next(layer for layer in layers if layer.func_name == "scaled_dot_product_attention")
    proj = next(layer for layer in layers if layer.func_name == "linear")

    field_lines = compute_default_node_lines(sdpa, node_label_fields=["label", "flops"])
    assert field_lines[-1] == ATTENTION_FULL_SQUARE_NOTE
    assert field_lines[-2] == str(sdpa.flops_forward)
    overlay_lines = compute_default_node_lines(sdpa, node_overlay="flops")
    assert overlay_lines[-2].startswith("flops: ")
    assert overlay_lines[-1] == ATTENTION_FULL_SQUARE_NOTE
    both = compute_default_node_lines(sdpa, node_label_fields=["flops"], node_overlay="flops")
    assert both.count(ATTENTION_FULL_SQUARE_NOTE) == 1
    assert ATTENTION_FULL_SQUARE_NOTE not in compute_default_node_lines(
        proj, node_label_fields=["flops"], node_overlay="flops"
    )
    assert ATTENTION_FULL_SQUARE_NOTE not in compute_default_node_lines(sdpa)
