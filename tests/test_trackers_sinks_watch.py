"""F26: the tier-P/M watch engine -- step law, cadence, loud failures.

Realism rule: the tier-M rows run against a REAL config-built GPT-2
(transformers), zero network; tier-P rows use a real MLP + AdamW loop.
"""

from __future__ import annotations

import pytest
import torch

import torchlens.errors
import torchlens.trackers as trk
from torchlens.errors import HistorySchemaError
from torchlens.trackers._errors import WatchConfigError, WatchRuntimeError


def _mlp() -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    """A small real training target: MLP + AdamW."""

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2))
    return model, torch.optim.AdamW(model.parameters(), lr=1e-3)


def _train_steps(session: trk.WatchSession, model, opt, steps) -> None:
    """Drive real forward/backward/step iterations through the step scope."""

    for step in steps:
        with session.step(step):
            opt.zero_grad(set_to_none=True)
            model(torch.randn(4, 8)).sum().backward()
            opt.step()


class TestTierP:
    """The default tier: parameters/gradients/updates, no forward wrap."""

    # Kept by the shrink-r4 coverage proof: sole branch coverage of _observability/_collector.py.
    def test_updates_are_true_deltas(self) -> None:
        """The updates family reflects the accepted optimizer delta."""

        model, opt = _mlp()
        sink = trk.MemorySink()
        with trk.watch(model, to=sink, signals=("updates",), optimizer=opt, every=1) as session:
            _train_steps(session, model, opt, [0])
        update_norms = [p for p in sink.scalars if p.tag.startswith("updates/norm/")]
        assert update_norms and all(p.value > 0 for p in update_norms)

    # Kept by the shrink-r4 coverage proof: sole branch coverage of _observability/_collector.py, _observability/_schema.py and more.
    def test_duplicate_step_refuses_and_resume_declares_segment(self) -> None:
        """Step-axis law: duplicates refuse; resume is a declared new segment."""

        model, opt = _mlp()
        session = trk.watch(model, to=trk.MemorySink(), optimizer=opt)
        _train_steps(session, model, opt, [50])
        with pytest.raises(HistorySchemaError) as info, session.step(50):
            pass
        assert info.value.fields["code"] == "history_step_regression"
        # Resume at a lower step is legal ONLY as a declared new segment.
        with session.step(10, new_segment=True):
            opt.zero_grad(set_to_none=True)
            model(torch.randn(4, 8)).sum().backward()
            opt.step()
        session.close()

    # Kept by the shrink-r4 coverage proof: sole branch coverage of trackers/_watch.py.
    def test_phase_missing_refuses_then_demotes(self) -> None:
        """A step scope without its optimizer boundary refuses (demotable)."""

        model, opt = _mlp()
        session = trk.watch(model, to=trk.MemorySink(), optimizer=opt)
        with pytest.raises(WatchRuntimeError) as info, session.step(0):
            model(torch.randn(4, 8)).sum().backward()  # no opt.step()
        assert info.value.fields["code"] == "watch_phase_missing"
        model2, opt2 = _mlp()
        demoted = trk.watch(
            model2,
            to=trk.MemorySink(),
            optimizer=opt2,
            allow_missing_phases=True,
        )
        with demoted.step(0):
            pass
        report = demoted.close()
        assert any("phase_missing" in skip for skip in report.named_skips)

    # Kept by the shrink-r4 coverage proof: sole branch coverage of trackers/_watch.py.
    def test_no_step_source_refuses_typed(self) -> None:
        """An optimizer boundary with no step source names all three ways."""

        model, opt = _mlp()
        trk.watch(model, to=trk.MemorySink(), optimizer=opt)
        model(torch.randn(4, 8)).sum().backward()
        with pytest.raises(WatchRuntimeError) as info:
            opt.step()
        assert info.value.fields["code"] == "watch_step_source_missing"
        assert "watch.step" in info.value.fields["remedy"]


class TestLoudFailures:
    """Memo 3.13: nothing becomes an empty panel without an explanation."""

    # Kept by the shrink-r4 coverage proof: sole branch coverage of trackers/_watch.py.
    def test_signals_vocabulary_refuses_unknown(self) -> None:
        model, opt = _mlp()
        with pytest.raises(WatchConfigError) as info:
            trk.watch(model, to=trk.MemorySink(), signals=("grads",), optimizer=opt)
        assert info.value.fields["code"] == "watch_signals_invalid"

    # Kept by the shrink-r4 coverage proof: sole branch coverage of trackers/_watch.py.
    def test_activations_without_selector_refuse(self) -> None:
        model, _opt = _mlp()
        with pytest.raises(WatchConfigError) as info:
            trk.watch(model, to=trk.MemorySink(), signals=("activations",))
        assert info.value.fields["code"] == "watch_signals_invalid"

    # Kept by the shrink-r4 coverage proof: sole branch coverage of trackers/_op_grain.py, trackers/_watch.py.
    def test_tier_o_ships_and_activation_grads_defer_typed(self) -> None:
        """Removal test (U32.1): op grain is no longer a deferred tier.

        ``grain="op"`` used to refuse ``watch_tier_unavailable`` ("not shipped
        in this build"); it now ships behind consent, so the same call refuses
        only for the missing consent. ``activation_grads`` stays deferred.
        """

        model, _opt = _mlp()
        with pytest.raises(WatchConfigError) as info:
            trk.watch(model, to=trk.MemorySink(), grain="op")
        assert info.value.fields["code"] == "watch_forward_wrap_consent_required"
        assert "not shipped" not in str(info.value)
        with pytest.raises(WatchConfigError) as info:
            trk.watch(model, to=trk.MemorySink(), signals=("activation_grads",))
        assert info.value.fields["code"] == "watch_tier_unavailable"

    # Kept by the shrink-r4 coverage proof: sole branch coverage of trackers/_watch.py.
    def test_budget_exceeded_is_a_preflight_fact(self) -> None:
        model, opt = _mlp()
        with pytest.raises(WatchConfigError) as info:
            trk.watch(
                model,
                to=trk.MemorySink(),
                optimizer=opt,
                budgets={"max_sites": 1},
            )
        assert info.value.fields["code"] == "watch_budget_exceeded"
        assert info.value.fields["actual"] > 1

    # Kept by the shrink-r4 coverage proof: sole branch coverage of trackers/_watch.py.
    def test_unknown_budget_key_refuses(self) -> None:
        """The budget vocabulary is closed: a typo'd cap never silently noops."""

        model, opt = _mlp()
        with pytest.raises(WatchConfigError) as info:
            trk.watch(
                model,
                to=trk.MemorySink(),
                optimizer=opt,
                budgets={"max_stuff": 1},
            )
        assert info.value.fields["code"] == "watch_budget_invalid"

    # Kept by the shrink-r4 coverage proof: sole branch coverage of trackers/_watch.py.
    def test_negative_step_refuses(self) -> None:
        """The step axis is the caller's nonnegative training coordinate."""

        model, opt = _mlp()
        session = trk.watch(model, to=trk.MemorySink(), optimizer=opt)
        with pytest.raises(WatchRuntimeError) as info, session.step(-1):
            pass
        assert info.value.fields["code"] == "watch_step_invalid"
        session.close(unwinding=True)

    # Kept by the shrink-r4 coverage proof: sole branch coverage of trackers/_watch.py.
    def test_check_outcome_vocabulary_closed(self) -> None:
        """emit_check accepts exactly the 0/1/2 verdict vocabulary."""

        model, opt = _mlp()
        session = trk.watch(model, to=trk.MemorySink(), optimizer=opt)
        session.emit_check("dead_layer", 1, step=0)
        with pytest.raises(WatchRuntimeError) as info:
            session.emit_check("dead_layer", 5, step=1)
        assert info.value.fields["code"] == "watch_check_outcome_invalid"
        session.close(unwinding=True)

    # Kept by the shrink-r4 coverage proof: sole branch coverage of trackers/_protocol.py, trackers/_watch.py.
    def test_sink_capability_refuses_before_emission(self) -> None:
        """A histogram request against a scalar-only sink refuses at attach."""

        model, opt = _mlp()
        narrow = trk.MemorySink(capabilities=frozenset({"scalar", "text_manifest"}))
        with pytest.raises(torchlens.errors.SinkProtocolError) as info:
            trk.watch(model, to=narrow, optimizer=opt, hist_every=10)
        assert info.value.fields["code"] == "tracker_sink_capability_missing"
        assert narrow.scalars == []

    # Kept by the shrink-r4 coverage proof: sole branch coverage of trackers/_watch.py.
    def test_sink_failure_latches_and_close_reports(self) -> None:
        """A throwing sink latches failed; close reports, never corrupts."""

        class _Exploding(trk.MemorySink):
            def emit_scalar(self, point):  # noqa: ANN001
                raise OSError("disk full")

        model, opt = _mlp()
        bad, good = _Exploding(), trk.MemorySink()
        session = trk.watch(model, to=(bad, good), signals=("gradients",), optimizer=opt, every=1)
        with session:
            _train_steps(session, model, opt, [0, 1])
        report = session.close()
        rows = {row["sink"]: row for row in report.sink_rows}
        assert rows["_Exploding"]["failed"]
        assert rows["MemorySink"]["emitted_scalars"] > 0

    # Kept by the shrink-r4 coverage proof: sole branch coverage of trackers/_records.py, trackers/_watch.py.
    def test_foreign_watcher_refuses_with_remedy(self) -> None:
        """G7: a wandb-owned hook on the model refuses; namespace= remedies."""

        def _fake_wandb_hook(module, inputs, output):  # noqa: ANN001
            return None

        _fake_wandb_hook.__module__ = "wandb.wandb_torch"
        model, opt = _mlp()
        handle = model[0].register_forward_hook(_fake_wandb_hook)
        try:
            with pytest.raises(WatchConfigError) as info:
                trk.watch(model, to=trk.MemorySink(), optimizer=opt)
            assert info.value.fields["code"] == "tracker_namespace_collision"
            # The named remedy: re-root and both watchers coexist.
            session = trk.watch(
                model,
                to=trk.MemorySink(),
                optimizer=opt,
                namespace="torchlens",
            )
            session.close(unwinding=True)
        finally:
            handle.remove()

    # Kept by the shrink-r4 coverage proof: sole branch coverage of trackers/_watch.py.
    def test_kill_switch_writes_disabled_row(self, monkeypatch) -> None:  # noqa: ANN001
        """TORCHLENS_WATCH_DISABLE=1: off-only, still discloses itself."""

        monkeypatch.setenv("TORCHLENS_WATCH_DISABLE", "1")
        model, opt = _mlp()
        sink = trk.MemorySink()
        session = trk.watch(model, to=sink, optimizer=opt)
        report = session.close()
        assert report.is_disabled
        assert [p.tag for p in sink.scalars] == ["torchlens/run/disabled"]


class TestTierM:
    """Module-grain activations on a REAL config-built GPT-2 (zero network)."""

    @pytest.fixture()
    def gpt2(self):
        """A tiny real GPT-2 architecture from config (no checkpoint)."""

        transformers = pytest.importorskip("transformers")
        config = transformers.GPT2Config(
            n_layer=2, n_head=2, n_embd=32, vocab_size=128, n_positions=32
        )
        model = transformers.GPT2LMHeadModel(config)
        example = torch.randint(0, 128, (1, 8))
        return model, example

    # Kept by the shrink-r4 coverage proof: sole branch coverage of _observability/_collector.py, trackers/_watch.py.
    def test_activations_collect_at_module_grain(self, gpt2) -> None:  # noqa: ANN001
        """Selected transformer.h.0 module outputs stream as activations/*."""

        model, example = gpt2
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        sink = trk.MemorySink()
        session = trk.watch(
            model,
            to=sink,
            signals=("activations", "gradients"),
            select=("transformer.h.0",),
            optimizer=opt,
            inputs=example,
            every=1,
        )
        with session, session.step(100):
            opt.zero_grad(set_to_none=True)
            out = model(example)
            out.logits.sum().backward()
            opt.step()
        activation_tags = {p.tag for p in sink.scalars if p.tag.startswith("activations/")}
        assert activation_tags, "module-grain activations must stream"
        assert any("transformer.h.0" in tag for tag in activation_tags)
        # The manifest disclosed both grains for the split selection.
        manifest = next(p for p in sink.texts if p.tag == "torchlens/meta/manifest")
        assert '"grain": "module"' in manifest.text
        assert '"grain": "param"' in manifest.text

    # Kept by the shrink-r4 coverage proof: sole branch coverage of _observability/_collector.py, trackers/_watch.py.
    def test_zero_match_refuses_at_attach(self, gpt2) -> None:  # noqa: ANN001
        """Zero-match selections refuse at plan time, never step 0 silence."""

        model, example = gpt2
        with pytest.raises(Exception) as info:
            trk.watch(
                model,
                to=trk.MemorySink(),
                signals=("activations",),
                select=("no.such.module",),
                inputs=example,
            )
        code = getattr(info.value, "fields", {}).get("code", "")
        assert code in ("watch_plan_empty", "watch_selector_matched_no_sites")
