# lerobot-isaac-synthetic — Package Orientation

**Role:** Synthetic data generation — Isaac Lab DR replay (priority) and MimicGen bridge (deferred).

**Package path:** `~/workspaces/lerobot-isaac-training/packages/lerobot-isaac-synthetic/`

---

## What this package does

Two paths to expand the training corpus beyond real SO-101 teleoperation:

1. **Isaac Lab DR replay** (`src/lerobot_isaac_synthetic/isaac_dr/`) — replays
   real episodes through an Isaac Lab environment with domain randomization,
   producing synthetic `LeRobotDataset` Parquet data tagged `source="sim_dr"`.

2. **MimicGen bridge** (`src/lerobot_isaac_synthetic/mimicgen/`) — deferred stub
   delegating to the `lerobot_mimicgen_bridge` skill.  Raises `NotImplementedError`
   by default.

3. **Merge utilities** (`src/lerobot_isaac_synthetic/merge_utilities.py`) — combines
   real + DR + MimicGen datasets with balanced sampling and source tagging.

---

## File map

```
src/lerobot_isaac_synthetic/
├── __init__.py                  — package docstring, __version__
├── isaac_dr/
│   ├── __init__.py              — sub-package docstring
│   ├── replay_runner.py         — replay_with_randomization() + Episode dataclass + CLI
│   └── parquet_writer.py        — write_episodes_to_lerobot_dataset()
├── mimicgen/
│   ├── __init__.py              — deferred-path docstring
│   └── bridge_invocation.py    — run_mimicgen() stub + LEROBOT_MIMICGEN_ENABLED toggle
└── merge_utilities.py           — merge_datasets()

tests/
├── test_imports.py              — smoke-test: package imports without lerobot/Isaac
└── test_replay_signature.py     — function signatures, dataclass fields, CLI --help
```

---

## Key design constraints

- **No lerobot import at module load.** All `lerobot.*` imports are deferred to
  function bodies.  Tests run without lerobot installed.
- **No Isaac Lab import at module load.** Same rule.
- **No MimicGen execution.** `bridge_invocation.py` always raises `NotImplementedError`
  unless `LEROBOT_MIMICGEN_ENABLED=1` is set AND a real implementation exists.
- **Dataset schema invariant.** All Parquet output must match the real SO-101
  column schema: `observation.state (12,)`, `action (6,)`, images as JPEG binary.

---

## Dependencies and references

| Dependency | Location | Role |
|------------|----------|------|
| `lerobot_isaac_env` | `../lerobot-isaac-env/` | gym env for DR replay |
| `lerobot_mimicgen_bridge` skill | `/home/koen/tools/claude_code/skills/lerobot_mimicgen_bridge/SKILL.md` | Parquet ↔ MimicGen HDF5 |
| `lerobot-sim-augmentation-agent` | `/home/koen/tools/claude_code/agents/workers/lerobot-sim-augmentation-agent.md` | Full MimicGen pipeline |
| `lerobot_dataset_quality` skill | `/home/koen/tools/claude_code/skills/lerobot_dataset_quality/SKILL.md` | SAL+TED quality filter |

---

## How to implement the DR replay stub

1. Open `src/lerobot_isaac_synthetic/isaac_dr/replay_runner.py`.
2. Follow the docstring steps in `replay_with_randomization()`.
3. Ensure `lerobot_isaac_env` is importable: `cd ../lerobot-isaac-env && pip install -e .`
4. Test with: `python -m lerobot_isaac_synthetic.isaac_dr.replay_runner --source_dataset_path /data/real --output_path /tmp/test --dry_run`

## How to enable MimicGen bridge

1. Install MimicGen + robosuite.
2. Set `LEROBOT_MIMICGEN_ENABLED=1`.
3. Implement `bridge_invocation.run_mimicgen()` following its docstring.
4. Or invoke the agent directly: `Task(lerobot-sim-augmentation-agent, {...})`.

---

## Build plan reference

Phase 4 spec: `/home/koen/tools/claude_code/plans/2026-05-06-lerobot-isaac-workspace-plan.md` — sections 4 (Phase 4a/4b), 11.3, 11.5.

---

## Spinout

```bash
git subtree split -P packages/lerobot-isaac-synthetic -b spinout-synthetic
```

After spinout, `lerobot-isaac-env` and `lerobot-isaac-configs` become external pip deps.
The `lerobot_mimicgen_bridge` and `lerobot_dataset_quality` skills remain in `claude_code` repo — reference them by absolute path in the standalone repo docs.
