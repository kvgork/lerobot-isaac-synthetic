# lerobot-isaac-synthetic

Synthetic data generation for the LeRobot + Isaac Lab training workspace.

Provides two paths to expand the training corpus beyond real SO-101 teleoperation data:

---

## Synthetic Data Paths

### Path 1 — Isaac Lab Domain Randomization (priority)

**Status: scaffolded — implement `isaac_dr/replay_runner.py` to activate.**

Replay recorded teleoperation episodes through an Isaac Lab environment whose
domain-randomization (DR) parameters are re-sampled on each reset.  Each real
episode produces N synthetic variants with randomized:

- Object pose (±5 cm, stage-dependent)
- Table surface friction
- Lighting intensity and direction
- Wrist/overhead camera field-of-view jitter
- Joint friction

The result is a `LeRobotDataset`-compatible Parquet directory with the same
column schema as real data, tagged `source="sim_dr"`.

**Entry-points:**

```python
from lerobot_isaac_synthetic.isaac_dr.replay_runner import replay_with_randomization
from lerobot_isaac_synthetic.isaac_dr.parquet_writer import write_episodes_to_lerobot_dataset

episodes = replay_with_randomization(
    source_dataset_path="/data/real",
    n_variants_per_episode=5,
    dr_config={"object_pose_noise_m": 0.03},
    env_id="Isaac-SO101-PickPlace-v0",
)
write_episodes_to_lerobot_dataset(episodes, output_path="/data/synthetic_dr")
```

**CLI:**

```bash
python -m lerobot_isaac_synthetic.isaac_dr.replay_runner \
    --source_dataset_path /data/real \
    --output_path /data/synthetic_dr \
    --n_variants 5 \
    --env_id Isaac-SO101-PickPlace-v0 \
    --dry_run
```

**Prerequisites:**
- Isaac Lab installed system-wide (not via pip)
- `lerobot-isaac-env` package (`Isaac-SO101-PickPlace-v0` gym registration)
- `lerobot` installed: `pip install lerobot`

---

### Path 2 — MimicGen Bridge (deferred)

**Status: stub only — raises `NotImplementedError` by default.**

MimicGen runs inside MuJoCo/robosuite (its native simulator) and is decoupled
from the Isaac Lab stack.  The pipeline is:

```
Real Parquet
  └─[lerobot_mimicgen_bridge skill]─► MimicGen HDF5
                                           │
                                    MimicGen runs in MuJoCo
                                           │
                                   MimicGen HDF5 output
  └─[lerobot_mimicgen_bridge skill]─► LeRobot Parquet (source="mimicgen")
                                           │
                                    merge_datasets()
```

The recommended invocation is via the `lerobot-sim-augmentation-agent` which
handles the full pipeline with error recovery.  This package only provides a
thin stub that delegates to the skill:

```python
from lerobot_isaac_synthetic.mimicgen.bridge_invocation import run_mimicgen
# Raises NotImplementedError — use the agent or enable the path explicitly
```

To enable: set `LEROBOT_MIMICGEN_ENABLED=1` in your environment AND install
MimicGen + robosuite, then implement the stub body.

Skill reference:
`/home/koen/tools/claude_code/skills/lerobot_mimicgen_bridge/SKILL.md`

Agent reference:
`/home/koen/tools/claude_code/agents/workers/lerobot-sim-augmentation-agent.md`

---

## Merge Utilities

Combine real, DR, and MimicGen datasets into one unified `LeRobotDataset`:

```python
from lerobot_isaac_synthetic.merge_utilities import merge_datasets

merged = merge_datasets(
    real_path="/data/real",
    sim_paths=["/data/synthetic_dr", "/data/synthetic_mimicgen"],
    output_path="/data/merged",
    sim_weight=0.5,   # 50% real, 50% synthetic
)
```

The merged dataset:
- Re-assigns contiguous `episode_index` values
- Preserves `source` column (`"real"` / `"sim_dr"` / `"mimicgen"`) in metadata
- Deduplicates near-identical episodes (real episodes are never dropped)
- Updates `meta/stats.json` and `meta/info.json`

---

## Dataset Schema Invariant

All Parquet output matches the SO-101 real-teleoperation column schema so that
policies trained on real data can consume synthetic data without conversion:

| Column | dtype | shape |
|--------|-------|-------|
| `observation.state` | float32 | (12,) — joint pos + vel |
| `observation.images.wrist` | binary (JPEG) | 480×640×3 |
| `observation.images.overhead` | binary (JPEG) | 480×640×3 |
| `action` | float32 | (6,) — joint position targets (radians) |
| `next.done` | bool | (1,) |

---

## Installation

```bash
# From workspace root
pip install -e packages/lerobot-isaac-synthetic

# Or with dev extras
pip install -e "packages/lerobot-isaac-synthetic[dev]"
```

---

## Tests

```bash
pytest packages/lerobot-isaac-synthetic/tests/ -v
```

Tests require no external dependencies (Isaac Lab, lerobot, MimicGen are all
soft imports).

---

## Related packages in this workspace

| Package | Role |
|---------|------|
| `lerobot-isaac-env` | Isaac Lab SO-101 gym environment |
| `lerobot-isaac-adapters` | Training entrypoint + world-model adapters |
| `lerobot-isaac-configs` | Shared YAML configs |
| `lerobot-isaac-meta` | Umbrella CLI + workspace docs |

Build plan: `/home/koen/tools/claude_code/plans/2026-05-06-lerobot-isaac-workspace-plan.md` (Phase 4)
