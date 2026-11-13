"""Unit tests for DM Lab wrapper graceful degradation (T046).

These tests verify that:
1. The module imports cleanly even when deepmind_lab is not installed.
2. Constructing DMLabWrapper raises DMLabUnavailable (not ImportError) with a
   helpful message when the deepmind_lab binding is absent.
"""

from __future__ import annotations

import pytest


def test_module_imports_without_deepmind_lab() -> None:
    """Importing dmlab_wrapper must succeed even without deepmind_lab installed."""
    import importlib
    # Re-import to be sure (module may already be cached, which is also fine).
    mod = importlib.import_module("environments.layer2_memory.dmlab_wrapper")
    assert hasattr(mod, "DMLabWrapper")
    assert hasattr(mod, "DMLabUnavailable")


def test_dmlab_wrapper_raises_unavailable_not_import_error() -> None:
    """If deepmind_lab is absent, DMLabWrapper.__init__ must raise DMLabUnavailable."""
    from environments.layer2_memory.dmlab_wrapper import DMLabWrapper, DMLabUnavailable, _HAVE_DMLAB

    if _HAVE_DMLAB:
        pytest.skip("deepmind_lab is installed — graceful-degradation path not reachable")

    with pytest.raises(DMLabUnavailable) as exc_info:
        DMLabWrapper(level="nav_maze_static_01")

    msg = str(exc_info.value).lower()
    # The error message should mention the install path / how to fix it.
    assert "deepmind_lab" in msg or "install" in msg, (
        f"DMLabUnavailable message is not helpful: {exc_info.value!r}"
    )


def test_dmlab_unavailable_is_runtime_error() -> None:
    """DMLabUnavailable must be a subclass of RuntimeError (not ImportError)."""
    from environments.layer2_memory.dmlab_wrapper import DMLabUnavailable

    assert issubclass(DMLabUnavailable, RuntimeError)
    assert not issubclass(DMLabUnavailable, ImportError)
