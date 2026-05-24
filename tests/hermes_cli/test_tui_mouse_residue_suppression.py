"""Tests for the TUI-hot-path mouse-residue suppression.

The Python launcher (`hermes --tui …`) has a ~100–300ms cold-start window
where stdin is still in cooked + echo mode. If a previous Hermes session
left DEC mouse-tracking asserted, any mouse motion during that window
echoes literal ``^[[<…M`` text into the user's scrollback.

`_suppress_mouse_residue_early()` writes the disable sequence to stdout
before the heavy imports so the terminal stops emitting events ASAP.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch


def _reload_main():
    """Reimport `hermes_cli.main` from scratch so module-level code re-runs."""
    sys.modules.pop("hermes_cli.main", None)
    import hermes_cli.main  # noqa: F401


class TestEarlyMouseDisable:
    def _expected(self) -> bytes:
        return (
            b"\x1b[?1003l\x1b[?1002l\x1b[?1001l\x1b[?1000l\x1b[?9l"
            b"\x1b[?1006l\x1b[?1005l\x1b[?1015l\x1b[?1016l\x1b[?2029l"
        )

    def test_writes_disable_sequence_when_tui_flag_in_argv(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["hermes", "--tui", "-c", "abc"])
        monkeypatch.delenv("HERMES_TUI", raising=False)
        monkeypatch.delenv("HERMES_TUI_NO_EARLY_DISABLE", raising=False)

        with patch("os.write") as mock_write:
            from hermes_cli.main import _suppress_mouse_residue_early

            _suppress_mouse_residue_early()

        mock_write.assert_called_once_with(1, self._expected())

    def test_writes_disable_sequence_when_hermes_tui_env_set(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["hermes"])
        monkeypatch.setenv("HERMES_TUI", "1")
        monkeypatch.delenv("HERMES_TUI_NO_EARLY_DISABLE", raising=False)

        with patch("os.write") as mock_write:
            from hermes_cli.main import _suppress_mouse_residue_early

            _suppress_mouse_residue_early()

        mock_write.assert_called_once_with(1, self._expected())

    def test_no_op_on_non_tui_invocation(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["hermes", "--version"])
        monkeypatch.delenv("HERMES_TUI", raising=False)
        monkeypatch.delenv("HERMES_TUI_NO_EARLY_DISABLE", raising=False)

        with patch("os.write") as mock_write:
            from hermes_cli.main import _suppress_mouse_residue_early

            _suppress_mouse_residue_early()

        mock_write.assert_not_called()

    def test_respects_diagnostic_escape_hatch(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["hermes", "--tui"])
        monkeypatch.delenv("HERMES_TUI", raising=False)
        monkeypatch.setenv("HERMES_TUI_NO_EARLY_DISABLE", "1")

        with patch("os.write") as mock_write:
            from hermes_cli.main import _suppress_mouse_residue_early

            _suppress_mouse_residue_early()

        mock_write.assert_not_called()

    def test_oserror_is_swallowed(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["hermes", "--tui"])
        monkeypatch.delenv("HERMES_TUI", raising=False)
        monkeypatch.delenv("HERMES_TUI_NO_EARLY_DISABLE", raising=False)

        def boom(*_a, **_k):
            raise OSError("stdout closed")

        with patch("os.write", side_effect=boom):
            from hermes_cli.main import _suppress_mouse_residue_early

            # Must not propagate — startup hot path can never break.
            _suppress_mouse_residue_early()
