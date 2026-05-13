"""
conftest.py — pytest configuration for lerobot-isaac-synthetic tests.

Two responsibilities:

1. In monorepo mode: add sibling package src directories to sys.path so tests
   can import lerobot_isaac_adapters without requiring pixi install.

2. In standalone (post-spinout) mode: detect that we are NOT inside the
   monorepo and auto-skip tests marked `requires_workspace_root`. The same
   tests still execute and pass when run from the monorepo.

A tree is considered "monorepo" when BOTH of these are true:
  - workspace root (3 levels up from this file) contains a sibling
    `packages/` directory that holds at least one OTHER lerobot-isaac-*
    package alongside synthetic itself
  - workspace root contains a top-level workspace marker file declaring
    a uv or pixi workspace (`[tool.uv.workspace]` in pyproject.toml OR
    `[workspace]` in pixi.toml)

A spun-out subtree under /tmp lacks the sibling packages, so it is
considered standalone and `requires_workspace_root` tests auto-skip.

The synthetic test_quality_hook.py tests use unittest.mock.patch against
`lerobot_isaac_adapters.quality.apply_quality_filter`. mock.patch resolves
the target string at patch-time, which requires lerobot_isaac_adapters to
be importable. In a spun-out tree that sibling is not installed, so the
4 happy/error-path tests fail with ModuleNotFoundError. The negative test
(test_import_error_when_adapters_missing) explicitly hides the module and
remains valid in both layouts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Workspace root is 3 levels up from this file (when in monorepo).
_WORKSPACE_ROOT = Path(__file__).parent.parent.parent.parent
_PACKAGES = _WORKSPACE_ROOT / "packages"

# Sibling packages that must exist alongside synthetic for "in monorepo" to be true.
_REQUIRED_SIBLINGS = (
    "lerobot-isaac-adapters",
    "lerobot-isaac-configs",
    "lerobot-isaac-env",
)


def _has_workspace_marker(root: Path) -> bool:
    """True iff `root` declares a uv-workspace or pixi-workspace table."""
    py = root / "pyproject.toml"
    if py.is_file():
        try:
            if "[tool.uv.workspace]" in py.read_text():
                return True
        except OSError:
            pass
    pixi = root / "pixi.toml"
    if pixi.is_file():
        try:
            text = pixi.read_text()
            if "[workspace]" in text or "[tool.pixi.workspace]" in text:
                return True
        except OSError:
            pass
    return False


def _in_monorepo() -> bool:
    """Return True iff this tree looks like the monorepo workspace layout."""
    if not _PACKAGES.is_dir():
        return False
    # At least one required sibling package directory must exist.
    if not any((_PACKAGES / s).is_dir() for s in _REQUIRED_SIBLINGS):
        return False
    return _has_workspace_marker(_WORKSPACE_ROOT)


_MONOREPO = _in_monorepo()


# Add sibling src directories to path (editable install simulation) — monorepo only.
if _MONOREPO:
    for pkg_src in [
        _PACKAGES / "lerobot-isaac-adapters" / "src",
        _PACKAGES / "lerobot-isaac-configs" / "src",
    ]:
        pkg_src_str = str(pkg_src)
        if pkg_src_str not in sys.path and pkg_src.is_dir():
            sys.path.insert(0, pkg_src_str)


@pytest.fixture(autouse=True)
def _skip_requires_workspace_root_when_standalone(request):
    """Auto-skip tests marked `requires_workspace_root` when not in monorepo.

    In monorepo mode this fixture is a no-op — the marked tests still run.
    """
    if request.node.get_closest_marker("requires_workspace_root") and not _MONOREPO:
        pytest.skip(
            "requires monorepo workspace layout (sibling packages); "
            "skipped in standalone/spun-out tree"
        )
