# lerobot-isaac-synthetic — Package Orientation

**Role:** Synthetic data generation — Isaac Lab DR replay (implemented) and MimicGen bridge (deferred).
**Phase:** 4 — DR replay implemented; MimicGen deferred to Phase 4b.
**Status:** DR replay + parquet writer + merge utilities implemented. MimicGen stub only.

---

## What This Package Does

Three responsibilities:
1. **Isaac Lab DR replay** (`isaac_dr/`) — replays real episodes through an Isaac Lab
   environment with domain randomization re-sampled on each reset, yielding `Episode`
   objects written to `LeRobotDataset` Parquet format tagged `source="sim_dr"`.
2. **MimicGen bridge** (`mimicgen/`) — deferred stub. Raises `NotImplementedError` unless
   `LEROBOT_MIMICGEN_ENABLED=1` is set AND the stub is implemented.
3. **Merge utilities** (`merge_utilities.py`) — merges real + DR + MimicGen datasets with
   balanced sampling, deduplication, and re-indexing.

---

## Public API Surface

- `replay_with_randomization(source_dataset_path, n_variants_per_episode, dr_config, task, ...)` — generator
- `Episode` — dataclass: `episode_index`, `source_episode_index`, `dr_seed`, `observations`, `actions`, `success`, `metadata`
- `write_episodes_to_lerobot_dataset(episodes, output_path, source_tag, task_name, fps, features, ...)` — returns `Path`
- `merge_datasets(real_path, sim_paths, output_path, sim_weight, dedup, ...)` — returns `Path`
- `run_mimicgen(...)` — **stub**, `NotImplementedError` (deferred path)
- `convert_real_to_mimicgen_hdf5(...)` — **stub**, `NotImplementedError`
- `convert_mimicgen_hdf5_to_lerobot(...)` — **stub**, `NotImplementedError`

---

## File Map

```
src/lerobot_isaac_synthetic/
├── __init__.py                  — package docstring, __version__
├── isaac_dr/
│   ├── __init__.py              — sub-package docstring
│   ├── replay_runner.py         — replay_with_randomization() + Episode + CLI + _apply_dr_config()
│   └── parquet_writer.py        — write_episodes_to_lerobot_dataset() + _tag_source_column()
├── mimicgen/
│   ├── __init__.py              — deferred-path docstring
│   └── bridge_invocation.py     — run_mimicgen() stub + _check_enabled()
└── merge_utilities.py           — merge_datasets() + _load_episodes_df() + _dedup_against_real() + _copy_and_reindex_episodes()

tests/
├── test_imports.py              — smoke-test: package imports without lerobot/Isaac
├── test_replay_signature.py     — function signatures, Episode fields, CLI --help
└── test_dry_run.py              — replay_runner CLI --dry_run output check
```

---

## Key Design Constraints

- **No lerobot import at module load.** All `lerobot.*` imports inside function bodies.
- **No Isaac Lab import at module load.** Same rule.
- **MimicGen never executes without explicit opt-in.** `_check_enabled()` checks env var.
- **Dataset schema invariant.** Output always: `observation.state (12,)`, `action (6,)`,
  `observation.images.wrist/overhead` as video binary, `next.done (1,)`.

---

## Coupling (plan §11.6)

- Depends on `lerobot_isaac_env` at runtime (for gym environment registration).
- Does NOT import `lerobot_isaac_env` at module load — deferred to `replay_with_randomization()`.
- No hard dep on `lerobot-isaac-configs` or `lerobot-isaac-meta`.
- After spinout, `lerobot-isaac-env` and `lerobot-isaac-configs` become external pip deps.

---

## Heavy Dependencies

| Dependency | Import location | Import style |
|------------|----------------|--------------|
| `lerobot.common.datasets.lerobot_dataset.LeRobotDataset` | `replay_runner.py`, `parquet_writer.py`, `merge_utilities.py` | deferred inside function body |
| `gymnasium` | `replay_runner.py` | deferred inside `replay_with_randomization()` |
| `lerobot_isaac_env` | `replay_runner.py` | deferred inside `replay_with_randomization()` |
| `pandas` | `merge_utilities.py`, `parquet_writer.py` | stdlib-style lazy import (also in `pyproject.toml`) |
| `pyarrow` | `merge_utilities.py` | lazy |

`pyarrow`, `pandas`, `numpy` are declared in `pyproject.toml` as hard deps (lightweight enough).
`lerobot`, Isaac Lab, MimicGen are soft imports only.

---

## How to Extend

### Add a new DR parameter

1. Add the param key to `_PARAM_MAP` in `replay_runner._apply_dr_config()`.
2. Map it to the attribute path on `env.cfg.events.*`.
3. Add a test in `test_replay_signature.py` checking `_apply_dr_config` handles the key.

### Implement the MimicGen bridge

1. Set `LEROBOT_MIMICGEN_ENABLED=1`.
2. Install MimicGen + robosuite.
3. Implement `run_mimicgen()` body in `bridge_invocation.py` following the docstring steps.
4. Or: invoke `lerobot-sim-augmentation-agent` for the full pipeline.

---

## Testing Notes

Three test files, all run without external deps:

- `test_imports.py` — `import lerobot_isaac_synthetic` + sub-modules succeed
- `test_replay_signature.py` — function signature inspection, `Episode` field names, CLI `--help` parses
- `test_dry_run.py` — `replay_runner.main()` with `--dry_run` flag produces expected output

---

## Spinout Note

```bash
git subtree split -P packages/lerobot-isaac-synthetic -b spinout-synthetic
```

After spinout: `lerobot-isaac-env` and `lerobot-isaac-configs` become PyPI deps.
The `lerobot_mimicgen_bridge` and `lerobot_dataset_quality` skills remain in the
`claude_code` repo — reference by absolute path in standalone docs.

See `../../docs/ARCHITECTURE.md` (spinout section).

---

## Source-of-Truth Pointers

- Build plan: `/home/koen/tools/claude_code/plans/2026-05-06-lerobot-isaac-workspace-plan.md` — Phase 4
- MimicGen skill: `/home/koen/tools/claude_code/skills/lerobot_mimicgen_bridge/SKILL.md`
- Augmentation agent: `/home/koen/tools/claude_code/agents/workers/lerobot-sim-augmentation-agent.md`
- Dataset quality skill: `/home/koen/tools/claude_code/skills/lerobot_dataset_quality/SKILL.md`
