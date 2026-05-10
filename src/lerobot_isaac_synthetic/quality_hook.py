"""
quality_hook.py — Post-replay quality filter for synthetic datasets.

Calls apply_quality_filter() from lerobot_isaac_adapters on the output of
isaac_dr/replay_runner to remove low-quality synthetic episodes before
they are merged with real data.

This module is intentionally thin: it just calls the adapters quality module
so that the skill path-bridging logic lives in one place (quality.py).

Plan reference: §13.1 Bundle A, deliverable A2
Last-updated: 2026-05-07
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def filter_after_replay(
    replayed_dataset_path: str | Path,
    sal_threshold: float = 0.2,
    ted_threshold: float = 2.0,
    min_episode_length: int = 50,
    output_path: str | Path | None = None,
    dry_run: bool = False,
) -> Path:
    """Apply quality filtering to a replayed (synthetic) LeRobotDataset.

    Intended to be called immediately after replay_runner produces a synthetic
    dataset, before merge_utilities merges real + synthetic data.

    Parameters
    ----------
    replayed_dataset_path:
        Path to the synthetic LeRobotDataset produced by
        ``lerobot_isaac_synthetic.isaac_dr.replay_runner``.
    sal_threshold:
        SAL-based filter percentile as a fraction (0.2 → remove bottom 20%).
        Passed through to apply_quality_filter().
    ted_threshold:
        Absolute TED upper bound; episodes above this are additionally removed.
    min_episode_length:
        Unconditionally remove episodes shorter than this many timesteps.
    output_path:
        Where to write the filtered dataset.  Defaults to
        ``<replayed_dataset_path>_filtered``.
    dry_run:
        If True, report what would be removed without writing files.

    Returns
    -------
    Path
        Path to the filtered output dataset (same as output_path, resolved).

    Raises
    ------
    RuntimeError
        If apply_quality_filter reports success=False.

    Quality skill path-bridge note
    --------------------------------
    This function delegates to lerobot_isaac_adapters.quality.apply_quality_filter,
    which bridges to:
        ${CLAUDE_CODE_ROOT}/skills/lerobot_dataset_quality/
    Override the skill root via: export LEROBOT_CLAUDE_CODE_ROOT=/path/to/claude_code
    """
    # Soft-import to avoid circular/hard dependency;
    # lerobot_isaac_adapters is a sibling workspace package.
    try:
        from lerobot_isaac_adapters.quality import apply_quality_filter
    except ImportError as exc:
        raise ImportError(
            "lerobot_isaac_adapters is required for quality filtering.  "
            "Install it with: pip install -e packages/lerobot-isaac-adapters "
            "or run: pixi shell"
        ) from exc

    replayed_dataset_path = Path(replayed_dataset_path)

    if output_path is None:
        output_path = Path(str(replayed_dataset_path) + "_filtered")
    output_path = Path(output_path)

    logger.info(
        "filter_after_replay: input=%s output=%s sal=%.2f ted=%.2f min_len=%d dry_run=%s",
        replayed_dataset_path,
        output_path,
        sal_threshold,
        ted_threshold,
        min_episode_length,
        dry_run,
    )

    result = apply_quality_filter(
        dataset_path=replayed_dataset_path,
        sal_threshold=sal_threshold,
        ted_threshold=ted_threshold,
        min_episode_length=min_episode_length,
        output_path=output_path,
        dry_run=dry_run,
    )

    if not result.success:
        raise RuntimeError(
            f"Quality filtering failed: {result.error}. "
            f"Suggestions: {result.suggestions}"
        )

    if result.data:
        logger.info(
            "filter_after_replay complete: kept=%s removed=%s output=%s",
            result.data.get("kept", "?"),
            result.data.get("removed", "?"),
            output_path,
        )

    return output_path.resolve()
