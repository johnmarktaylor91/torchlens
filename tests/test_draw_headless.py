"""Tests for TorchLens draw behavior in non-interactive shells."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.smoke
def test_view_rendered_file_skips_open_in_headless_context(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Headless Linux draw paths should skip viewer launch with one stderr note."""

    from torchlens.visualization import _render_utils
    from torchlens.visualization.rendering import _view_rendered_file

    rendered_path = str(tmp_path / "modelgraph.pdf")
    monkeypatch.setattr(_render_utils.sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setenv("SSH_CONNECTION", "192.0.2.1 12345 192.0.2.2 22")

    def fail_if_opened(*_: object, **__: object) -> None:
        """Fail the test if the platform viewer would have been launched."""

        raise AssertionError("viewer launch should be skipped")

    monkeypatch.setattr(_render_utils.subprocess, "Popen", fail_if_opened)

    _view_rendered_file(rendered_path)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "torchlens.draw: headless context detected; "
        f"rendered file at {rendered_path}, skipping auto-open.\n"
    )


@pytest.mark.smoke
def test_view_rendered_file_silent_in_notebook(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """In a notebook the figure shows inline: no viewer launch, no headless note."""

    from torchlens.visualization import _render_utils
    from torchlens.visualization.rendering import _view_rendered_file

    rendered_path = str(tmp_path / "modelgraph.pdf")
    # Headless Linux remote kernel (no DISPLAY) but inside a notebook: the
    # inline ``display()`` already showed the graph, so auto-open must be a
    # complete no-op -- no viewer launch and nothing printed.
    monkeypatch.setattr(_render_utils.sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr("torchlens.utils.display.in_notebook", lambda: True)

    def fail_if_opened(*_: object, **__: object) -> None:
        """Fail the test if the platform viewer would have been launched."""

        raise AssertionError("viewer launch should be skipped")

    monkeypatch.setattr(_render_utils.subprocess, "Popen", fail_if_opened)

    _view_rendered_file(rendered_path)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
