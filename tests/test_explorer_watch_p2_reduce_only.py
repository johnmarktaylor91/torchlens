"""P2: the reduce-only save path skips the raw clone (explorer memo item 2).

Proof strategy: the summary reducer records the storage pointer of the
tensor it is handed; a module forward hook records the live output's
storage pointer. When they match, the reducer saw the LIVE storage -- no
transient copy was made for it. The alias guard then proves the RETAINED
payload never aliases that live storage. Budget behavior: with the raw
admission skipped, a budget too small for the raw tensor but large enough
for the summary passes; a budget too small even for the summary refuses.
"""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
from torchlens.fastlog import CaptureSpec
from torchlens.ir.summary_role import summary

pytestmark = pytest.mark.smoke


def _model() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.ReLU())


class _PtrProbe:
    """Summary reducer that records the storage pointers it observes."""

    _tl_transform_role = "summary"

    def __init__(self) -> None:
        self.seen_ptrs: list[int] = []

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        self.seen_ptrs.append(t.untyped_storage().data_ptr())
        return t.detach().float().sum().unsqueeze(0)


class TestCloneSkipTrace:
    """Exhaustive trace path: the reducer sees live storage."""

    def test_reducer_sees_live_storage(self) -> None:
        model = _model()
        x = torch.randn(2, 64)
        live_ptrs: list[int] = []

        def hook(module: torch.nn.Module, args: tuple, output: torch.Tensor) -> None:
            live_ptrs.append(output.untyped_storage().data_ptr())

        handles = [m.register_forward_hook(hook) for m in model]
        probe = _PtrProbe()
        try:
            log = tl.trace(
                model,
                x,
                save=tl.options.SaveOptions(activation_transform=probe, save_raw_activations=False),
            )
        finally:
            for handle in handles:
                handle.remove()
        assert probe.seen_ptrs, "reducer never ran"
        assert set(live_ptrs) & set(probe.seen_ptrs), (
            "reducer never observed a live module-output storage: the "
            "reduce-only path cloned before reducing"
        )
        # Nothing raw retained; summaries landed.
        op = log["linear_1_1"]
        assert op.out is None
        assert op.transformed_out is not None

    def test_retained_summary_never_aliases_live_storage(self) -> None:
        model = _model()
        x = torch.randn(2, 64)
        probe = _PtrProbe()
        log = tl.trace(
            model,
            x,
            save=tl.options.SaveOptions(activation_transform=probe, save_raw_activations=False),
        )
        retained_ptrs = {
            op.transformed_out.untyped_storage().data_ptr()
            for op in log.ops
            if getattr(op, "transformed_out", None) is not None
        }
        assert not retained_ptrs & set(probe.seen_ptrs)

    def test_raw_retention_still_clones(self) -> None:
        # With save_raw_activations=True the raw copy path is unchanged: the
        # retained out never aliases the live output storage.
        model = _model()
        x = torch.randn(2, 64)
        live_ptrs: list[int] = []

        def hook(module: torch.nn.Module, args: tuple, output: torch.Tensor) -> None:
            live_ptrs.append(output.untyped_storage().data_ptr())

        handles = [m.register_forward_hook(hook) for m in model]
        try:
            log = tl.trace(
                model,
                x,
                save=tl.options.SaveOptions(activation_transform=summary(lambda t: t.sum())),
            )
        finally:
            for handle in handles:
                handle.remove()
        op = log["linear_1_1"]
        assert op.out is not None
        assert op.out.untyped_storage().data_ptr() not in live_ptrs


class TestCloneSkipFastlog:
    """Fastlog path: same live-storage proof through _resolve_storage."""

    def test_reducer_sees_live_storage(self) -> None:
        model = _model()
        x = torch.randn(2, 64)
        live_ptrs: list[int] = []

        def hook(module: torch.nn.Module, args: tuple, output: torch.Tensor) -> None:
            live_ptrs.append(output.untyped_storage().data_ptr())

        handles = [m.register_forward_hook(hook) for m in model]
        probe = _PtrProbe()
        try:
            rec = tl.record(
                model,
                x,
                default_op=CaptureSpec(save_out=True),
                activation_transform=probe,
                save_raw_activations=False,
            )
        finally:
            for handle in handles:
                handle.remove()
        assert set(live_ptrs) & set(probe.seen_ptrs)
        for record in rec.records:
            if record.ctx.kind == "op":
                assert record.ram_payload is None

    def test_dtype_respelling_keeps_copy_path(self) -> None:
        # spec.dtype forces the copy path (the conversion allocates anyway);
        # the reducer then sees the converted COPY, not live storage.
        model = _model()
        x = torch.randn(2, 64)
        live_ptrs: list[int] = []

        def hook(module: torch.nn.Module, args: tuple, output: torch.Tensor) -> None:
            live_ptrs.append(output.untyped_storage().data_ptr())

        handles = [m.register_forward_hook(hook) for m in model]
        probe = _PtrProbe()
        try:
            tl.record(
                model,
                x,
                default_op=CaptureSpec(save_out=True, dtype=torch.float64),
                activation_transform=probe,
                save_raw_activations=False,
            )
        finally:
            for handle in handles:
                handle.remove()
        assert probe.seen_ptrs
        assert not set(live_ptrs) & set(probe.seen_ptrs)


class TestReduceOnlyBudget:
    """The save budget sees the summary retention, not the skipped raw."""

    def test_budget_below_raw_passes_reduce_only(self) -> None:
        model = _model()
        x = torch.randn(64, 64)  # raw op outputs: 64*64*4 = 16 KiB each
        log = tl.trace(
            model,
            x,
            save=tl.options.SaveOptions(
                activation_transform=summary(lambda t: t.float().sum().unsqueeze(0)),
                save_raw_activations=False,
            ),
            capture=tl.options.CaptureOptions(save_budget=8192),
        )
        assert log["linear_1_1"].transformed_out is not None

    def test_budget_below_summary_refuses(self) -> None:
        from torchlens.errors import SaveBudgetExceededError

        model = _model()
        x = torch.randn(64, 64)
        with pytest.raises(SaveBudgetExceededError):
            tl.trace(
                model,
                x,
                save=tl.options.SaveOptions(
                    # A summary that is itself large: a full-size detached copy.
                    activation_transform=summary(lambda t: t.float().flatten().clone()),
                    save_raw_activations=False,
                ),
                capture=tl.options.CaptureOptions(save_budget=8192),
            )
