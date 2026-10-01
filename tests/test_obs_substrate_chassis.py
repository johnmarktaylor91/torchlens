"""Observer/check slot tests: event schema + subscriber isolation (checks 4.7)."""

from __future__ import annotations

import pytest

from torchlens.errors._base import TorchLensWarning
from torchlens.observability import EventStream, ObserverEvent, ObserverEventError
from torchlens.observability._chassis import SUBSCRIBER_FAILURE_LIMIT

pytestmark = pytest.mark.smoke


class TestEventSchema:
    """The load-bearing event contract: kinds, reasons, gradient stamps."""

    def test_scalar_event(self) -> None:
        event = ObserverEvent(key="grad_norm", kind="scalar", value=1.5, global_step=3)
        assert event.value == 1.5

    def test_scalar_without_value_refuses(self) -> None:
        with pytest.raises(ObserverEventError) as excinfo:
            ObserverEvent(key="grad_norm", kind="scalar")
        assert excinfo.value.fields["code"] == "observer_event_invalid"
        assert "unavailable" in str(excinfo.value)  # the remedy teaches the alternative

    def test_unavailable_is_mandatory_vocabulary_with_reason(self) -> None:
        event = ObserverEvent(
            key="update_ratio", kind="unavailable", reason="optimizer not attached"
        )
        assert event.reason
        with pytest.raises(ObserverEventError):
            ObserverEvent(key="update_ratio", kind="unavailable")

    def test_histogram_counts_geometry(self) -> None:
        event = ObserverEvent(
            key="w.hist",
            kind="histogram_counts",
            bucket_limits=(0.0, 1.0, 2.0),
            bucket_counts=(3, 4),
        )
        assert event.bucket_counts == (3, 4)
        with pytest.raises(ObserverEventError):
            ObserverEvent(
                key="w.hist",
                kind="histogram_counts",
                bucket_limits=(0.0, 1.0),
                bucket_counts=(3, 4),
            )

    def test_verdict_requires_token(self) -> None:
        assert ObserverEvent(key="nan_check", kind="verdict", verdict="pass").verdict
        with pytest.raises(ObserverEventError):
            ObserverEvent(key="nan_check", kind="verdict")

    def test_gradient_scalar_requires_stage_and_scale_provenance(self) -> None:
        with pytest.raises(ObserverEventError) as excinfo:
            ObserverEvent(key="grad_norm", kind="scalar", value=1.0, gradient=True)
        assert "scale_provenance" in str(excinfo.value)
        stamped = ObserverEvent(
            key="grad_norm",
            kind="scalar",
            value=1.0,
            gradient=True,
            stage="pre_clip",
            scale_provenance="unknown",
        )
        assert stamped.scale_provenance == "unknown"

    def test_closed_kind_and_axis_vocabularies(self) -> None:
        with pytest.raises(ObserverEventError):
            ObserverEvent(key="x", kind="metric", value=1.0)
        with pytest.raises(ObserverEventError):
            ObserverEvent(key="x", kind="scalar", value=1.0, axis_provenance="psychic")

    def test_empty_key_refuses(self) -> None:
        with pytest.raises(ObserverEventError):
            ObserverEvent(key="", kind="scalar", value=1.0)


class TestEventStream:
    """Bounded stream; throwing subscribers cannot corrupt collection."""

    def test_publish_snapshot_and_bound(self) -> None:
        stream = EventStream(capacity=3)
        for index in range(5):
            stream.publish(ObserverEvent(key=f"k{index}", kind="scalar", value=float(index)))
        snapshot = stream.snapshot()
        assert [event.key for event in snapshot] == ["k2", "k3", "k4"]
        assert stream.published_total == 5

    def test_non_event_publish_refuses(self) -> None:
        with pytest.raises(ObserverEventError):
            EventStream().publish({"key": "x"})  # type: ignore[arg-type]

    def test_throwing_subscriber_cannot_corrupt_collection(self) -> None:
        stream = EventStream()
        seen: list[str] = []

        def bad(_event: ObserverEvent) -> None:
            raise RuntimeError("subscriber bug")

        def good(event: ObserverEvent) -> None:
            seen.append(event.key)

        stream.subscribe(bad)
        stream.subscribe(good)
        with pytest.warns(TorchLensWarning, match="raised") as caught:
            stream.publish(ObserverEvent(key="a", kind="scalar", value=1.0))
        assert caught[0].message.fields["code"] == "observer_subscriber_failed"
        stream.publish(ObserverEvent(key="b", kind="scalar", value=2.0))
        # Collection and the healthy subscriber both survived.
        assert [event.key for event in stream.snapshot()] == ["a", "b"]
        assert seen == ["a", "b"]

    def test_persistently_failing_subscriber_is_disabled_loudly(self) -> None:
        stream = EventStream()
        calls = {"n": 0}

        def bad(_event: ObserverEvent) -> None:
            calls["n"] += 1
            raise RuntimeError("always broken")

        stream.subscribe(bad)
        with pytest.warns(TorchLensWarning) as caught:
            for index in range(SUBSCRIBER_FAILURE_LIMIT + 3):
                stream.publish(ObserverEvent(key=f"k{index}", kind="scalar", value=1.0))
        codes = [warning.message.fields["code"] for warning in caught]
        assert codes == ["observer_subscriber_failed", "observer_subscriber_disabled"]
        # Disabled after the limit; collection never stopped.
        assert calls["n"] == SUBSCRIBER_FAILURE_LIMIT
        assert stream.published_total == SUBSCRIBER_FAILURE_LIMIT + 3

    def test_unsubscribe(self) -> None:
        stream = EventStream()
        seen: list[str] = []
        token = stream.subscribe(lambda event: seen.append(event.key))
        stream.publish(ObserverEvent(key="a", kind="scalar", value=1.0))
        stream.unsubscribe(token)
        stream.publish(ObserverEvent(key="b", kind="scalar", value=1.0))
        assert seen == ["a"]
        stream.unsubscribe(token)  # idempotent

    def test_recovering_subscriber_failure_count_resets(self) -> None:
        stream = EventStream()
        state = {"fail": True}

        def flaky(_event: ObserverEvent) -> None:
            if state["fail"]:
                raise RuntimeError("transient")

        token = stream.subscribe(flaky)
        with pytest.warns(TorchLensWarning):
            stream.publish(ObserverEvent(key="a", kind="scalar", value=1.0))
        state["fail"] = False
        stream.publish(ObserverEvent(key="b", kind="scalar", value=1.0))
        assert stream.failure_counts()[token] == 0
