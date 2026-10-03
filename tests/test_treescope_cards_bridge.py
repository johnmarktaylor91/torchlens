"""F16 B4: the treescope bridge -- registration, probe, sniff, ledger.

Test-discipline law (treescope memo s12), each rule earned by a measured
trap:

1. bridge renders are UNCOMPRESSED (compression hides the duplicate-box
   marker from grep);
2. ``ignore_exceptions=False`` plus a POSITIVE our-handler marker (a dead
   bridge is indistinguishable from no bridge by any size assertion);
3. byte assertions ride ``render_to_html``, never
   ``display_formatter.format`` (the kernel path streams);
4. content assertions, never non-emptiness.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.bridge import treescope as bridge

treescope = pytest.importorskip("treescope")


@pytest.fixture(autouse=True)
def _clean_bridge_state() -> object:
    """Every test starts and ends unregistered with a clear one-shot slot."""

    bridge.unregister()
    bridge._STATE.last_render_id = None
    yield
    bridge.unregister()
    bridge._STATE.last_render_id = None


@pytest.fixture(scope="module")
def small_log() -> object:
    """One small capture shared by the module's render tests."""

    log = tl.trace(nn.Sequential(nn.Linear(4, 4), nn.ReLU()), torch.randn(2, 4))
    try:
        yield log
    finally:
        log.cleanup()


def _render(obj: object) -> str:
    """Law-conformant render: uncompressed, exceptions surfaced."""

    return treescope.render_to_html(obj, ignore_exceptions=False, compressed=False)


def test_probe_is_behavioral_and_cached() -> None:
    """F-A: capability is probed on a real tensor, never version-pinned."""

    verdict, reason = bridge.probe_tensor_support()
    assert isinstance(verdict, bool)
    if not verdict:
        assert reason  # a broken adapter always names its reason
    assert bridge.probe_tensor_support() == (verdict, reason)  # cached


def test_register_status_unregister_roundtrip() -> None:
    """register() is idempotent; unregister removes exactly our handlers."""

    bridge.register()
    bridge.register()  # idempotent
    st = bridge.status()
    assert st.registered and st.treescope_importable
    assert any("Layer" in name for name in st.handlers)
    assert any("Op" in name for name in st.handlers)
    registry = treescope.type_registries.TREESCOPE_HANDLER_REGISTRY
    from torchlens.data_classes.layer import Layer

    assert Layer in registry
    bridge.unregister()
    assert Layer not in registry
    assert not bridge.status().registered


def test_foreign_slot_conflict_refuses_typed() -> None:
    """An occupied slot raises treescope_slot_occupied, never a silent swap."""

    from torchlens.data_classes.op import Op

    registry = treescope.type_registries.TREESCOPE_HANDLER_REGISTRY

    def foreign_handler(node, path, subtree_renderer):  # pragma: no cover - never called
        """Stand-in for a user's own handler."""
        return NotImplemented

    registry[Op] = foreign_handler
    try:
        with pytest.raises(Exception) as excinfo:
            bridge.register()
        assert excinfo.value.fields["code"] == "treescope_slot_occupied"
        # The foreign handler survives untouched.
        assert registry[Op] is foreign_handler
    finally:
        del registry[Op]


def test_registered_scope_restores(small_log) -> None:
    """The registered() scope undoes only its own installation."""

    with bridge.registered():
        assert bridge.status().registered
    assert not bridge.status().registered


def test_explicit_render_bounded_with_positive_marker(small_log) -> None:
    """Explicit path: our marker present, uncompressed render bounded."""

    bridge.register()
    html = _render(small_log["relu_1_2"])
    assert "TorchLens" in html  # POSITIVE our-handler marker (rule 2)
    assert len(html) < 400_000  # the memo's gpt2 guardrail band, tiny model
    if not bridge.probe_tensor_support()[0]:
        # Strict degradation names the reason and the native door (memo 3.5).
        assert "tensor view unavailable via treescope" in html


def test_trace_method_route_and_container_rule(small_log) -> None:
    """Trace renders via __treescope_repr__: bounded, keys as strings."""

    bridge.register()
    html = _render(small_log)
    assert "TorchLens Trace" in html
    assert "lookup keys" in html
    assert len(html) < 400_000
    # Container rule: the trace render embeds no child full cards.
    assert "tl-grid-table" not in html


def test_nested_container_stays_bounded(small_log) -> None:
    """A Trace nested in a rendered dict stays bounded (the F-B habitat)."""

    bridge.register()
    html = treescope.render_to_html({"before": small_log}, ignore_exceptions=True, compressed=False)
    assert "TorchLens Trace" in html
    assert len(html) < 500_000


def test_sniff_matrix(small_log) -> None:
    """Memo 3.6 pinned matrix: sentinel exactly where it belongs.

    Explicit path = sentinel present exactly once, full payload absent;
    kernel path (no postprocessor frame) = full card, never the sentinel;
    a half-armed predicate (slot without frame, frame without slot) keeps
    the full card.
    """

    bridge.register()
    layer = small_log["relu_1_2"]
    html = _render(layer)
    assert html.count("card omitted: rendered by the treescope bridge above") == 1
    assert "tl-grid-table" not in html  # duplicate payload suppressed

    # Kernel path: plain _repr_html_ with no treescope postprocessor frame.
    direct = layer._repr_html_()
    assert "card omitted" not in direct and "TorchLens" in direct

    # Slot armed but NO postprocessor frame: full card (broken-handler guard).
    bridge._STATE.last_render_id = id(layer)
    assert "card omitted" not in layer._repr_html_()
    bridge._STATE.last_render_id = None


def test_disabled_scope_falls_through(small_log) -> None:
    """disabled(): handlers return NotImplemented; treescope defaults apply."""

    bridge.register()
    with bridge.disabled():
        html = treescope.render_to_html(
            small_log["relu_1_2"], ignore_exceptions=True, compressed=False
        )
        assert "card omitted" not in html
    assert not bridge.status().disabled


def test_degraded_warning_fires_once(small_log) -> None:
    """Strict degradation warns treescope_bridge_degraded once per process."""

    if bridge.probe_tensor_support()[0]:
        pytest.skip("installed treescope renders tensors; degradation leg inactive")
    bridge.register()
    bridge._STATE.degraded_warned = False
    with pytest.warns(UserWarning) as caught:
        _render(small_log["relu_1_2"])
    codes = [getattr(w.message, "fields", {}).get("code") for w in caught]
    assert "treescope_bridge_degraded" in codes


# ---------------------------------------------------------------------------
# The three-way ledger audit (memo 3.3)


def _exported_public_classes() -> dict[str, type]:
    """Exported types from the export ledger, not dir(tl).

    The root SURFACE names (``tl.__all__``) plus the named non-root
    exports the memo calls out (``PartialTrace`` is absent from
    ``dir(tl)``).
    """

    def _resolve(name: str) -> object | None:
        """Resolve one root name; facade gates may refuse offline."""
        try:
            return getattr(tl, name)
        except Exception:  # noqa: BLE001 - absent extras are not exports
            return None

    resolved: dict[str, type] = {}
    for name in tl.__all__:
        value = _resolve(name)
        if isinstance(value, type):
            resolved[f"{value.__module__}.{value.__qualname__}"] = value
    from torchlens.partial import PartialTrace

    resolved[f"{PartialTrace.__module__}.{PartialTrace.__qualname__}"] = PartialTrace
    return resolved


def test_ledger_classifies_every_exported_dataclass() -> None:
    """F-C: the leak class is dataclasses; every exported one is a decision.

    Every exported DATACLASS type must be explicitly bucketed rich/one_line
    in the ledger, or provably bounded under treescope reflection (LEAVE):
    < 32 KB uncompressed with zero ``object at 0x`` markers on a default
    instance when one is constructible.
    """

    def _default_instance(cls: type) -> object | None:
        """Construct a default instance, or None (classification-only row)."""
        try:
            return cls()
        except Exception:  # noqa: BLE001 - unconstructible is a valid answer
            return None

    ledger_names = {dotted.rsplit(".", 1)[-1] for dotted in bridge.BRIDGE_LEDGER}
    for dotted, cls in _exported_public_classes().items():
        if not dataclasses.is_dataclass(cls) or cls.__qualname__ in ledger_names:
            continue
        # LEAVE bucket: pin boundedness when a default instance exists.
        instance = _default_instance(cls)
        if instance is None:
            continue
        html = treescope.render_to_html(instance, ignore_exceptions=True, compressed=False)
        assert len(html) < 32_000, f"{dotted} unbounded under treescope ({len(html)} B)"
        assert " at 0x" not in html, f"{dotted} leaks object ids"


def test_rich_types_are_the_memo_four() -> None:
    """The RICH bucket is exactly Trace/PartialTrace/Layer/Op in wave 1."""

    rich = sorted(
        dotted.rsplit(".", 1)[-1]
        for dotted, bucket in bridge.BRIDGE_LEDGER.items()
        if bucket == "rich"
    )
    assert rich == ["Layer", "Op", "PartialTrace", "Trace"]


def test_bare_layer_op_stay_in_the_bounded_band(small_log) -> None:
    """UNBRIDGED Layer/Op renders stay bounded with no internals markers.

    The memo pins the ~8 KB band so a future dataclass conversion cannot
    silently open a megabyte leak; the ceiling here is 3x the band.
    """

    import dataclasses as dataclasses_module

    assert not bridge.status().registered
    for obj in (small_log["relu_1_2"], small_log["linear_1_1"]):
        # The band's guard premise: Layer/Op are NOT dataclasses, so
        # treescope falls back to bounded repr text instead of reflecting
        # fields (a silent dataclass conversion is what opens the leak).
        assert not dataclasses_module.is_dataclass(type(obj))
        html = treescope.render_to_html(obj, ignore_exceptions=True, compressed=False)
        assert len(html) < 24_000, f"bare render left the band: {len(html)} B"
        # No reflected internals: the private store handle must not appear.
        assert "_core" not in html and "OpRowStore" not in html


def test_missing_treescope_door_refuses_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without treescope the door raises treescope_bridge_unavailable."""

    import builtins

    real_import = builtins.__import__

    def blocked(name: str, *args: object, **kwargs: object) -> object:
        """Simulate an environment without treescope."""
        if name == "treescope" or name.startswith("treescope."):
            raise ImportError("treescope blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(Exception) as excinfo:
        bridge._import_treescope()
    assert excinfo.value.fields["code"] == "treescope_bridge_unavailable"
